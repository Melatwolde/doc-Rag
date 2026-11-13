"""data_pipeline/run_pipeline.py

Command‑line entry point for the Hybrid RAG ingestion pipeline.

Typical usage::

    python -m data_pipeline.run_pipeline

The script performs three main steps:

1. Load raw documents from ``data/raw`` using :func:`data_pipeline.ingestion.load_documents`.
2. Run a quick benchmark (printing a markdown table) to give the user an idea of
   chunking behaviour.
3. Chunk the documents with the default configuration (512 token limit, 10 % overlap)
   and write the resulting chunks to ``data/processed/chunks.json``.

The output format is a JSON array where each element corresponds to a chunk and
contains the fields defined in :class:`data_pipeline.models.Chunk`.
"""

import argparse
import json
import pathlib
from typing import List

from .ingestion import load_documents
from .chunker import chunk_document
from .experiment import benchmark, save_experiment
from .models import Chunk


def _chunks_to_dicts(chunks: List[Chunk]) -> List[dict]:
    """Serialize a list of :class:`Chunk` objects to plain dictionaries.
    ``pydantic``'s ``dict()`` method does the heavy lifting.
    """
    return [c.dict() for c in chunks]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hybrid RAG data pipeline")
    parser.add_argument(
        "--token-limit",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)",
    )
    parser.add_argument(
        "--experiment-output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "chunking_experiment.json",
        help="Path for saved chunking experiment results.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.1,
        help="Overlap ratio between consecutive chunks (default: 0.1)",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip the benchmark step",
    )
    args = parser.parse_args()

    # Step 1: load documents
    docs = load_documents()
    if not docs:
        print("No documents found in data/raw.")
        return

    # Step 2: optional benchmark
    if not args.no_benchmark:
        print("Running chunking benchmark…")
        records = benchmark(docs, token_limits=[256, 512, 1024], overlap_ratios=[0.0, 0.1, 0.2])
        save_experiment(records, args.experiment_output)
        print(f"Wrote chunking experiment results to {args.experiment_output}")

    # Step 3: chunk the documents
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, token_limit=args.token_limit, overlap_ratio=args.overlap))

    # Write out JSON
    output_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_chunks_to_dicts(all_chunks), f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()

