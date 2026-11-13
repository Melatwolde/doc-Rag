"""docurag/generation/llm.py

Grounded LLM answer-generation pipeline for Documentation Q&A.

Architecture
------------
1. Build a grounding prompt from assembled context passages.
2. Send to the LLM backend (Flan-T5 locally, or OpenAI if key is set).
3. Post-process: extract citation indices, check abstention, return a
   structured ``GenerationResult``.

Abstention
----------
The generator says "I don't know" when:
  * No retrieved passages pass the cross-encoder score threshold, OR
  * The model output contains known abstention phrases.

This is the primary hallucination-reduction mechanism: the prompt
explicitly prohibits the model from answering outside the evidence.

LLM Backends
------------
* **Flan-T5** (default) — ``google/flan-t5-base`` loaded locally via
  HuggingFace Transformers.  No API key required, runs on CPU.
* **OpenAI** — activated when ``OPENAI_API_KEY`` is in the environment.
  Set ``backend="openai"`` or let the class auto-detect.

Usage::

    from docurag.generation.llm import RAGGenerator
    gen = RAGGenerator()
    result = gen.generate(query="What is the JWT expiry?", passages=reranked)
    print(result.answer)
    for c in result.citations:
        print(f"  [{c.index}] {c.section_title}")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..reranking.context_builder import CitationMeta, ContextBuilder
from ..retrieval.result import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ABSTENTION_PHRASES = (
    "i don't know",
    "i do not know",
    "cannot be determined",
    "not mentioned",
    "not found in",
    "no information",
    "insufficient information",
    "i'm not sure",
    "not available in",
    "not covered",
)

_GROUNDING_PROMPT_TEMPLATE = """\
You are a precise documentation assistant. Your ONLY job is to answer the user's \
question using the evidence passages provided below.

STRICT RULES:
1. Answer ONLY using the evidence passages. Do NOT use any prior knowledge.
2. After each claim, add a citation like [1] or [2] referencing the passage number.
3. If the passages do not contain enough information to answer, respond EXACTLY with:
   "I don't know based on the available documentation."
4. Keep the answer concise (2–5 sentences unless more detail is needed).
5. Do NOT fabricate URLs, function names, or configuration values.

--- EVIDENCE ---
{context}
--- END EVIDENCE ---

Question: {query}

Answer:"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Structured result returned by :class:`RAGGenerator`.

    Attributes
    ----------
    answer : str
        The generated answer text, with inline ``[1]``, ``[2]`` citations.
    citations : List[CitationMeta]
        Metadata for each cited passage (section title, URL, snippet …).
    is_abstention : bool
        ``True`` when the model could not find evidence to answer.
    grounded_passages : List[SearchResult]
        The passages that were passed to the LLM as context.
    raw_prompt : str
        Full prompt sent to the LLM (useful for debugging).
    model_used : str
        Name of the LLM backend used.
    """

    answer: str
    citations: List[CitationMeta] = field(default_factory=list)
    is_abstention: bool = False
    grounded_passages: List[SearchResult] = field(default_factory=list)
    raw_prompt: str = ""
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "is_abstention": self.is_abstention,
            "model_used": self.model_used,
            "citations": [
                {
                    "index": c.index,
                    "section_title": c.section_title,
                    "source_url": c.source_url,
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                    "snippet": c.snippet,
                }
                for c in self.citations
            ],
        }


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _is_abstention(text: str) -> bool:
    lowered = text.lower().strip()
    return any(phrase in lowered for phrase in _ABSTENTION_PHRASES)


def _generate_flan_t5(prompt: str, model_name: str, max_new_tokens: int) -> str:
    """Run inference with a local Flan-T5 model."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the local backend. "
            "Install it with: pip install transformers"
        ) from exc

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _generate_openai(prompt: str, model_name: str, max_tokens: int) -> str:
    """Run inference via OpenAI Chat API."""
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# RAGGenerator
# ---------------------------------------------------------------------------


class RAGGenerator:
    """Grounded LLM answer generator for the RAG pipeline.

    Parameters
    ----------
    backend : str
        ``"local"`` (Flan-T5, default) or ``"openai"``.
        When set to ``"auto"``, uses OpenAI if ``OPENAI_API_KEY`` is
        present in the environment, otherwise falls back to local.
    local_model_name : str
        HuggingFace model for the local backend.
    openai_model_name : str
        OpenAI model identifier (default ``"gpt-3.5-turbo"``).
    max_new_tokens : int
        Maximum tokens the LLM may generate.
    max_context_tokens : int
        Token budget for the evidence context window.
    """

    def __init__(
        self,
        backend: str = "auto",
        local_model_name: str = "google/flan-t5-base",
        openai_model_name: str = "gpt-3.5-turbo",
        max_new_tokens: int = 256,
        max_context_tokens: int = 1800,
    ) -> None:
        # Resolve backend
        if backend == "auto":
            self.backend = (
                "openai" if os.environ.get("OPENAI_API_KEY") else "local"
            )
        else:
            self.backend = backend

        self.local_model_name = local_model_name
        self.openai_model_name = openai_model_name
        self.max_new_tokens = max_new_tokens
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens)

        logger.info("RAGGenerator initialised with backend=%s", self.backend)

    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, context: str) -> str:
        return _GROUNDING_PROMPT_TEMPLATE.format(
            context=context, query=query
        )

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "openai":
            return _generate_openai(
                prompt, self.openai_model_name, self.max_new_tokens
            )
        # Default: local Flan-T5
        return _generate_flan_t5(
            prompt, self.local_model_name, self.max_new_tokens
        )

    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        passages: List[SearchResult],
    ) -> GenerationResult:
        """Generate a grounded answer from reranked evidence passages.

        Parameters
        ----------
        query : str
            The user's question.
        passages : List[SearchResult]
            Reranked passages from :class:`CrossEncoderReranker`.
            Pass an empty list to force abstention.

        Returns
        -------
        GenerationResult
            Structured result with answer, citations, and provenance.
        """
        model_id = (
            self.openai_model_name
            if self.backend == "openai"
            else self.local_model_name
        )

        # --- No evidence → abstain immediately ---
        if not passages:
            logger.info("No passages provided — generating abstention response.")
            return GenerationResult(
                answer="I don't know based on the available documentation.",
                citations=[],
                is_abstention=True,
                grounded_passages=[],
                raw_prompt="",
                model_used=model_id,
            )

        # --- Assemble context ---
        context_str, all_citations = self.context_builder.build(query, passages)

        if not context_str:
            return GenerationResult(
                answer="I don't know based on the available documentation.",
                citations=[],
                is_abstention=True,
                grounded_passages=passages,
                raw_prompt="",
                model_used=model_id,
            )

        prompt = self._build_prompt(query, context_str)

        # --- Call LLM ---
        try:
            raw_answer = self._call_llm(prompt)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise

        # --- Post-process ---
        answer = raw_answer.strip()
        abstention = _is_abstention(answer)

        # Filter citations to only those actually used in the answer
        used_citations = self.context_builder.filter_citations(
            answer, all_citations
        )
        # If model didn't cite anything but isn't abstaining, attach all
        if not used_citations and not abstention:
            used_citations = all_citations

        return GenerationResult(
            answer=answer,
            citations=used_citations,
            is_abstention=abstention,
            grounded_passages=passages,
            raw_prompt=prompt,
            model_used=model_id,
        )

    # ------------------------------------------------------------------

    def check_grounding(
        self,
        answer: str,
        passages: List[SearchResult],
    ) -> dict:
        """Test whether an answer is supported by its cited passages.

        Uses simple lexical overlap (Jaccard) as a lightweight grounding
        check that works without a second LLM call.

        Returns
        -------
        dict with keys:
          * ``"grounded"`` – bool, True if any passage has overlap > threshold
          * ``"max_overlap"`` – float, highest passage-level Jaccard score
          * ``"supporting_chunk_ids"`` – list of chunk_ids with overlap > 0.1
        """
        answer_tokens = set(re.findall(r"\w+", answer.lower()))
        if not answer_tokens:
            return {
                "grounded": False,
                "max_overlap": 0.0,
                "supporting_chunk_ids": [],
            }

        supporting = []
        max_overlap = 0.0

        for passage in passages:
            passage_tokens = set(re.findall(r"\w+", passage.content.lower()))
            if not passage_tokens:
                continue
            intersection = answer_tokens & passage_tokens
            union = answer_tokens | passage_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard > 0.1:
                supporting.append(passage.chunk_id)
            max_overlap = max(max_overlap, jaccard)

        return {
            "grounded": bool(supporting),
            "max_overlap": round(max_overlap, 4),
            "supporting_chunk_ids": supporting,
        }
