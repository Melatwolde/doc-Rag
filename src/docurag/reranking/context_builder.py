"""docurag/reranking/context_builder.py

Context assembly for the LLM generation stage.

Takes reranked SearchResult objects and formats them into a numbered
evidence block that the generator LLM can reference with inline citations.

Usage::

    builder = ContextBuilder(max_tokens=1800)
    context_str, citations = builder.build(query, reranked_results)
    # context_str → passed to the LLM prompt
    # citations   → list of CitationMeta for the final answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..retrieval.result import SearchResult


@dataclass
class CitationMeta:
    """Metadata for a single cited evidence passage."""

    index: int             # 1-based citation index used in the answer
    chunk_id: str
    section_title: Optional[str]
    source_url: Optional[str]
    score: float           # cross-encoder relevance score
    snippet: str           # first 200 chars of passage for display


class ContextBuilder:
    """Assembles the evidence context string sent to the LLM.

    Parameters
    ----------
    max_tokens : int
        Approximate token budget for the context window.
        Each token is estimated as ~4 characters (English prose).
    snippet_length : int
        Characters shown per passage in the assembled context block.
        Set to a large value (e.g. 2000) to pass the full passage.
    """

    CHARS_PER_TOKEN = 4  # rough estimate for English text

    def __init__(
        self,
        max_tokens: int = 1800,
        snippet_length: int = 2000,
    ) -> None:
        self.max_tokens = max_tokens
        self.snippet_length = snippet_length
        self._char_budget = max_tokens * self.CHARS_PER_TOKEN

    # ------------------------------------------------------------------

    def build(
        self,
        query: str,
        passages: List[SearchResult],
    ) -> Tuple[str, List[CitationMeta]]:
        """Format reranked passages into a numbered context string.

        Parameters
        ----------
        query : str
            Original user question (included in context header for context).
        passages : List[SearchResult]
            Reranked passages from :class:`CrossEncoderReranker`.

        Returns
        -------
        context_str : str
            Formatted evidence block ready for prompt injection.
        citations : List[CitationMeta]
            Ordered list of citation metadata matching ``[1]``, ``[2]`` …
            references in the context string.
        """
        citations: List[CitationMeta] = []
        blocks: List[str] = []
        chars_used = 0

        for i, result in enumerate(passages, 1):
            section = result.metadata.get("section_title") or "Documentation"
            url = result.metadata.get("source_url") or ""
            ce_score = result.metadata.get("cross_encoder_score", result.score)

            # Truncate passage to budget
            passage_text = result.content.strip()
            remaining_budget = self._char_budget - chars_used
            if remaining_budget <= 0:
                break
            truncated = passage_text[:min(self.snippet_length, remaining_budget)]

            # Build evidence block
            header = f"[{i}] Section: {section}"
            if url:
                header += f" | Source: {url}"
            block = f"{header}\n{truncated}"
            blocks.append(block)
            chars_used += len(block)

            citations.append(
                CitationMeta(
                    index=i,
                    chunk_id=result.chunk_id,
                    section_title=section,
                    source_url=url if url else None,
                    score=float(ce_score),
                    snippet=passage_text[:200],
                )
            )

        if not blocks:
            return "", []

        context_str = (
            f"The following are relevant documentation passages for the question:\n"
            f'"{query}"\n\n'
            + "\n\n---\n\n".join(blocks)
        )
        return context_str, citations

    # ------------------------------------------------------------------

    @staticmethod
    def extract_citation_indices(answer: str) -> List[int]:
        """Parse ``[1]``, ``[2]`` … citation references from an answer string.

        Returns
        -------
        List[int]
            Sorted, deduplicated list of cited indices.
        """
        raw = re.findall(r"\[(\d+)\]", answer)
        return sorted(set(int(x) for x in raw))

    def filter_citations(
        self,
        answer: str,
        citations: List[CitationMeta],
    ) -> List[CitationMeta]:
        """Return only citations that were actually referenced in the answer."""
        used_indices = self.extract_citation_indices(answer)
        return [c for c in citations if c.index in used_indices]
