"""bm25_retriever.py

Root module alias for BM25 Keyword-Search Baseline.
Provides direct access to BM25Retriever, tokenize_technical_text, and SearchResult.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path if not present
src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from docurag.retrieval.bm25_search import BM25Retriever, tokenize_technical_text
from docurag.retrieval.result import SearchResult

__all__ = ["BM25Retriever", "tokenize_technical_text", "SearchResult"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the BM25 Lexical Retriever")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--chunks-path",
        type=str,
        default="data/processed/chunks.json",
        help="Path to processed chunks JSON file",
    )
    args = parser.parse_args()

    chunks_file = Path(args.chunks_path)
    if not chunks_file.exists():
        print(f"Error: Chunks file not found at {chunks_file}")
        sys.exit(1)

    retriever = BM25Retriever.from_json(chunks_file)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\n--- BM25 Search Results for: '{args.query}' (Top {len(results)}) ---")
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] Chunk ID: {res.chunk_id} | Score: {res.score:.4f}")
        print(f"    Section: {res.metadata.get('section_title')}")
        snippet = res.content.replace('\n', ' ')[:140]
        print(f"    Content: {snippet}...")


if __name__ == "__main__":
    main()
