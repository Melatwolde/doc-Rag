"""docurag/retrieval/rank_fusion.py

Reciprocal Rank Fusion (RRF) algorithm for combining multiple retriever rankings.

Formula:
    RRF_score(d) = sum_i [ w_i / (k + rank_i(d)) ]

Where:
    - d is a document / chunk
    - rank_i(d) is the 1-based rank position of document d in ranking list i
    - k is a smoothing constant (default: 60)
    - w_i is the weight allocated to retriever i (default: 1.0)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .result import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: List[List[SearchResult]],
    weights: Optional[List[float]] = None,
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """Fuse multiple ranked lists of SearchResult objects using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : List[List[SearchResult]]
        Lists of ranked search results from different retrievers (e.g. [bm25_results, dense_results]).
    weights : Optional[List[float]]
        Importance weight for each retriever list. Defaults to 1.0 for all lists.
    k : int
        RRF smoothing constant (standard default is 60).
    top_k : Optional[int]
        Number of fused results to return. If None, returns all deduplicated candidates.

    Returns
    -------
    List[SearchResult]
        Deduplicated, merged results sorted in descending order of fused RRF score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match length of ranked_lists ({len(ranked_lists)})"
        )

    # Accumulator: chunk_id -> { "score": float, "content": str, "metadata": dict, "ranks": dict }
    fused_candidates: Dict[str, Dict[str, Any]] = {}

    for list_idx, (result_list, weight) in enumerate(zip(ranked_lists, weights)):
        retriever_name = f"retriever_{list_idx}"

        for rank_zero_based, item in enumerate(result_list):
            rank = rank_zero_based + 1  # 1-based ranking
            chunk_id = item.chunk_id
            rrf_contribution = weight * (1.0 / (k + rank))

            item_retriever = item.metadata.get("retriever", retriever_name)

            if chunk_id not in fused_candidates:
                fused_candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": item.content,
                    "rrf_score": 0.0,
                    "metadata": item.metadata.copy(),
                    "sources": {},
                }

            entry = fused_candidates[chunk_id]
            entry["rrf_score"] += rrf_contribution
            entry["sources"][item_retriever] = {
                "rank": rank,
                "raw_score": item.score,
                "rrf_contribution": rrf_contribution,
            }

            # Merge any non-empty metadata fields from overlapping chunks
            for meta_k, meta_v in item.metadata.items():
                if meta_v is not None and meta_k not in entry["metadata"]:
                    entry["metadata"][meta_k] = meta_v

    # Sort candidates by combined RRF score in descending order
    sorted_candidates = sorted(
        fused_candidates.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    if top_k is not None:
        sorted_candidates = sorted_candidates[:top_k]

    fused_results: List[SearchResult] = []
    for final_rank, item in enumerate(sorted_candidates, 1):
        meta = item["metadata"]
        meta["fused_rank"] = final_rank
        meta["retriever"] = "hybrid_rrf"
        meta["rrf_k"] = k
        meta["retrieval_sources"] = item["sources"]

        fused_results.append(
            SearchResult(
                chunk_id=item["chunk_id"],
                content=item["content"],
                score=item["rrf_score"],
                metadata=meta,
            )
        )

    return fused_results
