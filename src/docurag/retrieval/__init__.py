"""docurag/retrieval package."""

from .result import SearchResult
from .dense_search import DenseRetriever
from .vector_index import FaissVectorIndex
from .rank_fusion import reciprocal_rank_fusion

__all__ = [
    "SearchResult",
    "BM25Retriever",
    "tokenize_technical_text",
    "DenseRetriever",
    "FaissVectorIndex",
    "reciprocal_rank_fusion",
    "HybridRetriever",
]


def __getattr__(name: str):
    """Load optional lexical/hybrid dependencies only when they are requested.

    Dense-only indexing should not fail merely because the optional ``rank-bm25``
    package has not been installed yet.
    """
    if name in {"BM25Retriever", "tokenize_technical_text"}:
        from .bm25_search import BM25Retriever, tokenize_technical_text

        return {"BM25Retriever": BM25Retriever, "tokenize_technical_text": tokenize_technical_text}[name]
    if name == "HybridRetriever":
        from .hybrid_search import HybridRetriever

        return HybridRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
