"""docurag/pipeline.py

End-to-end RAG Pipeline coordinator.

Wires together:
  HybridRetriever → CrossEncoderReranker → RAGGenerator → GenerationResult

Usage::

    from docurag.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_chunks_json("data/processed/chunks.json")
    result = pipeline.answer("How do I refresh a JWT token?")
    print(result.answer)
    for c in result.citations:
        print(f"  [{c.index}] {c.section_title} — score {c.score:.3f}")
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

from .generation.llm import GenerationResult, RAGGenerator
from .reranking.context_builder import ContextBuilder
from .reranking.cross_encoder import CrossEncoderReranker
from .retrieval.hybrid_search import HybridRetriever
from .retrieval.result import SearchResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Full Retrieve → Rerank → Generate pipeline.

    Parameters
    ----------
    retriever : HybridRetriever
        Hybrid BM25 + dense retriever (Member 3's component).
    reranker : CrossEncoderReranker
        Cross-encoder reranker for passage refinement.
    generator : RAGGenerator
        LLM-backed answer generator.
    retrieve_top_k : int
        Number of candidates to fetch from the hybrid retriever before
        reranking (default 10).
    rerank_top_k : int
        Number of passages to pass to the generator after reranking
        (default 3).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: RAGGenerator,
        retrieve_top_k: int = 10,
        rerank_top_k: int = 3,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_chunks_json(
        cls,
        chunks_path: Union[str, pathlib.Path],
        reranker_model: str = CrossEncoderReranker.DEFAULT_MODEL,
        llm_backend: str = "auto",
        retrieve_top_k: int = 10,
        rerank_top_k: int = 3,
        score_threshold: float = -5.0,
    ) -> "RAGPipeline":
        """Build a complete RAG pipeline from a chunks JSON file.

        Parameters
        ----------
        chunks_path : str | Path
            Path to ``data/processed/chunks.json``.
        reranker_model : str
            HuggingFace cross-encoder model name.
        llm_backend : str
            ``"auto"``, ``"local"``, or ``"openai"``.
        retrieve_top_k : int
            Candidates fetched from retriever before reranking.
        rerank_top_k : int
            Top passages forwarded to the LLM.
        score_threshold : float
            Cross-encoder score below which passages are excluded.

        Returns
        -------
        RAGPipeline
        """
        logger.info("Building RAGPipeline from %s", chunks_path)
        retriever = HybridRetriever.from_json(chunks_path)
        reranker = CrossEncoderReranker(
            model_name=reranker_model,
            score_threshold=score_threshold,
        )
        generator = RAGGenerator(backend=llm_backend)
        return cls(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
        )

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        verbose: bool = False,
    ) -> GenerationResult:
        """Answer a documentation question end-to-end.

        Pipeline stages
        ---------------
        1. **Retrieve** – hybrid BM25 + dense search for ``retrieve_top_k``
           candidates.
        2. **Rerank** – cross-encoder selects and scores the top
           ``rerank_top_k`` passages.
        3. **Generate** – LLM produces a grounded answer with citations.

        Parameters
        ----------
        question : str
            Natural-language user question.
        verbose : bool
            If True, log intermediate retrieval and reranking steps.

        Returns
        -------
        GenerationResult
            Structured answer with inline citations.
        """
        # Stage 1: Retrieve
        candidates: List[SearchResult] = self.retriever.retrieve(
            question, top_k=self.retrieve_top_k
        )
        if verbose:
            logger.info(
                "[Retrieve] %d candidates for: %r", len(candidates), question
            )

        # Stage 2: Rerank (with threshold filtering for abstention)
        reranked: List[SearchResult] = self.reranker.select_context(
            question, candidates, top_k=self.rerank_top_k
        )
        if verbose:
            logger.info(
                "[Rerank] %d high-confidence passages selected", len(reranked)
            )
            for r in reranked:
                logger.info(
                    "  chunk=%s score=%.4f section=%s",
                    r.chunk_id[:8],
                    r.score,
                    r.metadata.get("section_title"),
                )

        # Stage 3: Generate
        result = self.generator.generate(question, reranked)
        if verbose:
            logger.info(
                "[Generate] abstention=%s citations=%d",
                result.is_abstention,
                len(result.citations),
            )

        return result

    def answer_batch(
        self,
        questions: List[str],
    ) -> List[GenerationResult]:
        """Answer multiple questions sequentially."""
        return [self.answer(q) for q in questions]

    def inspect(self, question: str) -> Dict[str, Any]:
        """Return a full diagnostic breakdown of the pipeline stages.

        Returns
        -------
        dict with keys: ``"question"``, ``"retrieved"``, ``"reranked"``,
        ``"result"``.
        """
        candidates = self.retriever.retrieve(question, top_k=self.retrieve_top_k)
        reranked_all = self.reranker.rerank(question, candidates, top_k=self.retrieve_top_k)
        filtered = [r for r in reranked_all if r.score >= self.reranker.score_threshold][:self.rerank_top_k]
        result = self.generator.generate(question, filtered)

        return {
            "question": question,
            "retrieved": [
                {
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                    "section": c.metadata.get("section_title"),
                    "snippet": c.content[:120],
                }
                for c in candidates
            ],
            "reranked": [
                {
                    "chunk_id": r.chunk_id,
                    "ce_score": r.metadata.get("cross_encoder_score", r.score),
                    "pre_rank": r.metadata.get("pre_rerank_rank"),
                    "section": r.metadata.get("section_title"),
                }
                for r in reranked_all[: self.rerank_top_k]
            ],
            "result": result.to_dict(),
        }
