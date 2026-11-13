"""scripts/interactive_search.py
Interactive query runner for recording live RAG demonstrations.
"""

from __future__ import annotations

import pathlib
import sys

# Add src to sys.path
root_dir = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "src"))

from docurag.retrieval.bm25_search import BM25Retriever
from docurag.retrieval.dense_search import DenseRetriever
from docurag.retrieval.hybrid_search import HybridRetriever


def main() -> None:
    chunks_path = root_dir / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Run the pipeline first.")
        sys.exit(1)

    print("Initializing Hybrid Search Engine (BM25 + MiniLM Dense via RRF)...")
    bm25 = BM25Retriever.from_json(chunks_path)
    dense = DenseRetriever.from_json(chunks_path)
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_k=60)

    print("\n=======================================================")
    print("🐍 PYTHON DOC-RAG INTERACTIVE DEMO READY")
    print("=======================================================")
    print("Type any Python question to test retrieval (or type 'quit' to exit).\n")

    while True:
        try:
            query = input("Enter query: ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit"]:
                print("Exiting demo. Good luck with your video!")
                break

            results = hybrid.retrieve(query, top_k=3)
            print(f"\n🔍 Top 3 Hybrid Results for: '{query}'\n" + "-" * 50)
            
            for i, res in enumerate(results, 1):
                section = res.metadata.get("section_title", "General Section")
                # Clean up newlines so text prints as a normal sentence
                clean_preview = " ".join(res.content.split())
                print(f"[{i}] Section: {section}")
                print(f"    Preview: {clean_preview[:250]}...\n")
            print("-" * 50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break


if __name__ == "__main__":
    main()