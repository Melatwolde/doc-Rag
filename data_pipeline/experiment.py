"""data_pipeline/experiment.py

Benchmarking utilities for the Hybrid RAG chunker.

The script iterates over a set of token limits and overlap ratios, chunks a
sample of documents, and prints a summary table showing:

* ``token_limit`` – target maximum tokens per chunk
* ``overlap`` – overlap ratio used
* ``chunk_count`` – total number of chunks generated
* ``avg_tokens`` – average token count per chunk
* ``max_tokens`` – largest token count observed

The functions are deliberately lightweight so they can be imported from the
CLI entry point without pulling in heavy optional dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

from .chunker import chunk_document
from .models import Document, Chunk


def _stats(chunks: List[Chunk]) -> Tuple[int, float, int]:
    """Return ``(count, avg, max)`` token statistics for a list of chunks."""
    if not chunks:
        return 0, 0.0, 0
    counts = [c.token_count for c in chunks]
    total = sum(counts)
    return len(chunks), total / len(chunks), max(counts)


def benchmark_records(documents: List[Document],
              token_limits: List[int] = [256, 512, 1024],
              overlap_ratios: List[float] = [0.0, 0.1, 0.2]) -> List[dict[str, Any]]:
    """Run the benchmark and return reproducible chunking statistics."""
    records = []
    for limit in token_limits:
        for overlap in overlap_ratios:
            all_chunks: List[Chunk] = []
            for doc in documents:
                all_chunks.extend(chunk_document(doc, token_limit=limit, overlap_ratio=overlap))
            count, avg, max_tok = _stats(all_chunks)
            records.append({"token_limit": limit, "overlap_ratio": overlap, "chunk_count": count, "average_tokens": round(avg, 2), "max_tokens": max_tok})
    return records


def benchmark(documents: List[Document], token_limits: List[int] = [256, 512, 1024], overlap_ratios: List[float] = [0.0, 0.1, 0.2]) -> List[dict[str, Any]]:
    """Print and return chunking experiment records."""
    # Header for markdown table
    print("| Token Limit | Overlap | Chunk Count | Avg Tokens | Max Tokens |")
    print("|------------|---------|------------|------------|------------|")
    records = benchmark_records(documents, token_limits, overlap_ratios)
    for record in records:
        print(f"| {record['token_limit']:<11} | {record['overlap_ratio']:<7.2f} | {record['chunk_count']:<10} | {record['average_tokens']:<10.2f} | {record['max_tokens']:<10} |")
    return records


def save_experiment(records: List[dict[str, Any]], output_path: Path) -> None:
    """Persist machine-readable experiment results; never overwrite source docs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # Simple demo when executed directly – load docs from the ingestion module.
    from .ingestion import load_documents

    docs = load_documents()
    benchmark(docs)

