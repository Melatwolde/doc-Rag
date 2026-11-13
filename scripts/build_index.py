"""Build and persist the MiniLM dense-vector index for processed chunks.

Usage: python scripts/build_index.py [chunks.json] [output-directory]
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docurag.retrieval.dense_search import DenseRetriever


def main() -> None:
    chunks_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/chunks.json"
    output_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/processed/dense_index"
    retriever = DenseRetriever.from_json(chunks_path)
    retriever.save_index(output_path)
    print(
        f"Indexed {len(retriever.chunks)} chunks ({retriever.vector_index.dimension} dimensions) "
        f"with {retriever.vector_index.backend}; saved to {output_path}"
    )


if __name__ == "__main__":
    main()
