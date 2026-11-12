"""hybrid_retriever.py

Root module alias for Hybrid Retrieval Engine.
Combines BM25 keyword search and MiniLM dense semantic search via Reciprocal Rank Fusion (RRF).
Exposes a clean callable interface retrieve(query, top_k=5) directly consumable by Lyd's Cross-Encoder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path if not present
src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from docurag.retrieval.bm25_search import BM25Retriever
from docurag.retrieval.dense_search import DenseRetriever
from docurag.retrieval.hybrid_search import HybridRetriever
from docurag.retrieval.rank_fusion import reciprocal_rank_fusion
from docurag.retrieval.result import SearchResult

__all__ = [
    "HybridRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "reciprocal_rank_fusion",
    "SearchResult",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Hybrid RRF Retriever")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--chunks-path",
        type=str,
        default="data/processed/chunks.json",
        help="Path to processed chunks JSON file",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant k")
    parser.add_argument("--bm25-weight", type=float, default=1.0, help="Weight for BM25")
    parser.add_argument("--dense-weight", type=float, default=1.0, help="Weight for Dense")
    args = parser.parse_args()

    chunks_file = Path(args.chunks_path)
    if not chunks_file.exists():
        print(f"Error: Chunks file not found at {chunks_file}")
        sys.exit(1)

    print("Initializing Hybrid Retriever (BM25 + MiniLM Dense)...")
    hybrid = HybridRetriever.from_json(
        chunks_file,
        rrf_k=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )

    breakdown = hybrid.retrieve_with_breakdown(args.query, top_k=args.top_k)
    fused_results = breakdown["hybrid_results"]

    print(f"\n=================================================================")
    print(f"HYBRID RETRIEVAL RESULTS: '{args.query}' (Top {len(fused_results)})")
    print(f"=================================================================")
    for rank, res in enumerate(fused_results, 1):
        sources = res.metadata.get("retrieval_sources", {})
        source_str = ", ".join(
            f"{name} (rank {info['rank']}, raw={info['raw_score']:.3f})"
            for name, info in sources.items()
        )
        print(f"\n[{rank}] Chunk ID: {res.chunk_id}")
        print(f"    RRF Score : {res.score:.5f}")
        print(f"    Sources   : {source_str}")
        print(f"    Section   : {res.metadata.get('section_title')}")
        snippet = res.content.replace('\n', ' ')[:160]
        print(f"    Content   : {snippet}...")


if __name__ == "__main__":
    main()
