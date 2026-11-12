"""docurag/reranking/cross_encoder.py

Cross-Encoder Reranker interface for refining top-k retrieved chunks.
Directly consumes SearchResult objects emitted by HybridRetriever.
"""

from __future__ import annotations

from typing import List, Optional, Union

from sentence_transformers import CrossEncoder

from ..retrieval.result import SearchResult


class CrossEncoderReranker:
    """Reranker using a Cross-Encoder to compute query-chunk cross-attention scores.

    Parameters
    ----------
    model_name : str
        Cross-encoder model name (default: "cross-encoder/ms-marco-MiniLM-L-6-v2").
    device : Optional[str]
        Device to run inference on.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 3,
    ) -> List[SearchResult]:
        """Rerank a list of SearchResult candidates for a query.

        Parameters
        ----------
        query : str
            Search query string.
        candidates : List[SearchResult]
            Candidates produced by HybridRetriever.retrieve().
        top_k : int
            Number of top reranked chunks to return.

        Returns
        -------
        List[SearchResult]
            Reranked results with cross-encoder relevance scores.
        """
        if not candidates:
            return []

        # Build query-chunk text pairs for cross-attention
        pairs = [[query, c.content] for c in candidates]
        scores = self.model.predict(pairs)

        # Sort indices by cross-encoder score descending
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )

        reranked: List[SearchResult] = []
        for new_rank, idx in enumerate(ranked_indices[:top_k], 1):
            original = candidates[idx]
            metadata = original.metadata.copy()
            metadata["cross_encoder_score"] = float(scores[idx])
            metadata["reranked_rank"] = new_rank
            metadata["pre_rerank_rank"] = idx + 1
            metadata["pre_rerank_score"] = original.score

            reranked.append(
                SearchResult(
                    chunk_id=original.chunk_id,
                    content=original.content,
                    score=float(scores[idx]),
                    metadata=metadata,
                )
            )

        return reranked
