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

from collections import defaultdict
from typing import List, Tuple

from .chunker import chunk_document
from .models import Document, Chunk


def _stats(chunks: List[Chunk]) -> Tuple[int, float, int]:
    """Return ``(count, avg, max)`` token statistics for a list of chunks."""
    if not chunks:
        return 0, 0.0, 0
    counts = [c.token_count for c in chunks]
    total = sum(counts)
    return len(chunks), total / len(chunks), max(counts)


def benchmark(documents: List[Document],
              token_limits: List[int] = [256, 512, 1024],
              overlap_ratios: List[float] = [0.0, 0.1, 0.2]) -> None:
    """Run the benchmark and print a markdown table.

    Parameters
    ----------
    documents:
        List of :class:`Document` objects to process.
    token_limits:
        Target token limits to test.
    overlap_ratios:
        Overlap ratios (e.g. 0.1 = 10 %).
    """
    # Header for markdown table
    print("| Token Limit | Overlap | Chunk Count | Avg Tokens | Max Tokens |")
    print("|------------|---------|------------|------------|------------|")
    for limit in token_limits:
        for overlap in overlap_ratios:
            all_chunks: List[Chunk] = []
            for doc in documents:
                all_chunks.extend(chunk_document(doc, token_limit=limit, overlap_ratio=overlap))
            count, avg, max_tok = _stats(all_chunks)
            print(f"| {limit:<11} | {overlap:<7.2f} | {count:<10} | {avg:<10.2f} | {max_tok:<10} |")


if __name__ == "__main__":
    # Simple demo when executed directly – load docs from the ingestion module.
    from .ingestion import load_documents

    docs = load_documents()
    benchmark(docs)

