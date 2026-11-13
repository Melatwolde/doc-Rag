"""docurag/reranking/cross_encoder.py

Cross-Encoder Reranker — refines top-k retrieved chunks from HybridRetriever.

The CrossEncoder jointly encodes the full (query, passage) pair so it can
compute fine-grained relevance scores that bi-encoder models cannot.

Model default: cross-encoder/ms-marco-MiniLM-L-6-v2
  — fast, <100 MB, strong MS-MARCO re-ranking performance.

Usage::

    reranker = CrossEncoderReranker()
    top_passages = reranker.rerank(query, hybrid_results, top_k=3)
    context     = reranker.select_context(query, hybrid_results)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

from ..retrieval.result import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker using a Cross-Encoder to compute query-chunk relevance scores.

    Parameters
    ----------
    model_name : str
        HuggingFace cross-encoder model identifier.
        Default: ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
    device : Optional[str]
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        ``None`` lets sentence-transformers auto-detect.
    batch_size : int
        Number of (query, passage) pairs scored per forward pass.
        Reduce if you run out of memory.
    score_threshold : float
        Minimum cross-encoder score for a passage to be considered
        relevant.  Passages below this threshold are excluded from
        ``select_context()`` output.  Has no effect on ``rerank()``.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        score_threshold: float = -5.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self._model: Optional[CrossEncoder] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the CrossEncoder (downloads once, then cached)."""
        if self._model is None:
            logger.info("Loading CrossEncoder model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _score_pairs(
        self, query: str, candidates: List[SearchResult]
    ) -> List[float]:
        """Score all (query, passage) pairs and return raw floats."""
        pairs = [[query, c.content] for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in scores]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 3,
    ) -> List[SearchResult]:
        """Rerank candidate passages and return the top-k by cross-encoder score.

        Parameters
        ----------
        query : str
            User question or search string.
        candidates : List[SearchResult]
            Retrieved passages from :class:`~docurag.retrieval.hybrid_search.HybridRetriever`.
        top_k : int
            Maximum number of passages to return (default 3).

        Returns
        -------
        List[SearchResult]
            Reranked results.  Each result carries extra metadata keys:

            * ``cross_encoder_score`` – raw CE logit
            * ``reranked_rank`` – 1-based rank after reranking
            * ``pre_rerank_rank`` – 1-based rank before reranking
            * ``pre_rerank_score`` – original RRF/retrieval score
        """
        if not candidates:
            return []

        scores = self._score_pairs(query, candidates)

        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda i: scores[i],
            reverse=True,
        )

        reranked: List[SearchResult] = []
        for new_rank, idx in enumerate(ranked_indices[:top_k], 1):
            original = candidates[idx]
            meta = original.metadata.copy()
            meta["cross_encoder_score"] = scores[idx]
            meta["reranked_rank"] = new_rank
            meta["pre_rerank_rank"] = idx + 1
            meta["pre_rerank_score"] = original.score

            reranked.append(
                SearchResult(
                    chunk_id=original.chunk_id,
                    content=original.content,
                    score=scores[idx],
                    metadata=meta,
                )
            )

        return reranked

    def rerank_with_scores(
        self,
        query: str,
        candidates: List[SearchResult],
    ) -> List[Tuple[SearchResult, float]]:
        """Return *all* candidates paired with their cross-encoder scores.

        Useful for threshold analysis and evaluation diagnostics.

        Returns
        -------
        List[Tuple[SearchResult, float]]
            All candidates sorted descending by score.
        """
        if not candidates:
            return []

        scores = self._score_pairs(query, candidates)
        pairs = sorted(
            zip(candidates, scores),
            key=lambda t: t[1],
            reverse=True,
        )
        return list(pairs)

    def select_context(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 3,
    ) -> List[SearchResult]:
        """Rerank and filter to only passages above ``score_threshold``.

        Unlike :meth:`rerank`, this method can return *fewer* than ``top_k``
        passages when confidence is low — signalling to the generator that
        there is insufficient evidence to answer.

        Parameters
        ----------
        query : str
            User question.
        candidates : List[SearchResult]
            Retrieved passages.
        top_k : int
            Hard upper bound on passages returned.

        Returns
        -------
        List[SearchResult]
            High-confidence passages (may be empty → triggers abstention).
        """
        reranked = self.rerank(query, candidates, top_k=top_k)
        return [r for r in reranked if r.score >= self.score_threshold]

    def score_passage_support(
        self,
        query: str,
        passage: str,
        answer: str,
    ) -> Dict[str, float]:
        """Check whether a *specific* passage supports a generated answer.

        Uses two cross-encoder calls:
        1. (query, passage)  → retrieval relevance
        2. (answer, passage) → answer-passage entailment proxy

        Returns
        -------
        Dict with keys ``"relevance"`` and ``"support"``.
        """
        rel_score = float(
            self.model.predict([[query, passage]], batch_size=1)[0]
        )
        sup_score = float(
            self.model.predict([[answer, passage]], batch_size=1)[0]
        )
        return {"relevance": rel_score, "support": sup_score}
