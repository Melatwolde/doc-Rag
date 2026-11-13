"""docurag/retrieval/hybrid_search.py

Hybrid Retrieval Engine combining BM25 keyword search and MiniLM dense semantic search
via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Union

from ..embeddings.encoder import MiniLMEncoder
from .bm25_search import BM25Retriever
from .dense_search import DenseRetriever
from .rank_fusion import reciprocal_rank_fusion
from .result import SearchResult


class HybridRetriever:
    """Hybrid retrieval engine combining lexical (BM25) and semantic (MiniLM) search.

    Parameters
    ----------
    bm25_retriever : BM25Retriever
        Initialized BM25 lexical retriever.
    dense_retriever : DenseRetriever
        Initialized dense vector retriever.
    rrf_k : int
        Smoothing constant for Reciprocal Rank Fusion (default 60).
    bm25_weight : float
        Relative weight for BM25 rankings in fusion (default 1.0).
    dense_weight : float
        Relative weight for Dense rankings in fusion (default 1.0).
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    @classmethod
    def from_chunks(
        cls,
        chunks: List[Union[Dict[str, Any], Any]],
        encoder: Optional[MiniLMEncoder] = None,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> HybridRetriever:
        """Create a HybridRetriever by indexing a list of chunks."""
        bm25 = BM25Retriever(chunks=chunks)
        dense = DenseRetriever(chunks=chunks, encoder=encoder, model_name=model_name)
        return cls(
            bm25_retriever=bm25,
            dense_retriever=dense,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    @classmethod
    def from_json(
        cls,
        json_path: Union[str, pathlib.Path],
        encoder: Optional[MiniLMEncoder] = None,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> HybridRetriever:
        """Create a HybridRetriever directly from chunks.json."""
        path = pathlib.Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        return cls.from_chunks(
            chunks=chunks,
            encoder=encoder,
            model_name=model_name,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Execute hybrid search using Reciprocal Rank Fusion.

        Parameters
        ----------
        query : str
            Query string to search for.
        top_k : int
            Number of top fused results to return for downstream reranker / generation.
        candidate_k : Optional[int]
            Number of candidates to fetch from each retriever before fusion.
            Defaults to max(top_k * 4, 30).

        Returns
        -------
        List[SearchResult]
            Fused, deduplicated search results sorted by descending RRF score.
        """
        if candidate_k is None:
            candidate_k = max(top_k * 4, 30)

        # Retrieve candidate pools from both retrieval modalities
        bm25_candidates = self.bm25.retrieve(query, top_k=candidate_k)
        dense_candidates = self.dense.retrieve(query, top_k=candidate_k)

        # Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[bm25_candidates, dense_candidates],
            weights=[self.bm25_weight, self.dense_weight],
            k=self.rrf_k,
            top_k=top_k,
        )

        return fused_results

    def retrieve_with_breakdown(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute search and return individual retriever results alongside fused results.

        Useful for evaluation, inspection, and rank shift analysis.
        """
        if candidate_k is None:
            candidate_k = max(top_k * 4, 30)

        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
        dense_results = self.dense.retrieve(query, top_k=candidate_k)
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[bm25_results, dense_results],
            weights=[self.bm25_weight, self.dense_weight],
            k=self.rrf_k,
            top_k=top_k,
        )

        return {
            "query": query,
            "bm25_results": bm25_results[:top_k],
            "dense_results": dense_results[:top_k],
            "hybrid_results": fused_results,
        }
