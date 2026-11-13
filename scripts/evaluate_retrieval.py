"""scripts/evaluate_retrieval.py

Comparative Retrieval Evaluation Benchmark.
Compares BM25-only, Dense-only (MiniLM), and Hybrid (RRF) search across:
- Exact technical queries (error codes, CLI flags, snake_case function names, config keys)
- Conceptual / semantic queries (natural language questions, explanations, how-tos)

Computes Recall@1, Recall@3, Recall@5, and MRR@5.
Generates a comprehensive performance report: hybrid_search_report.md.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add src to sys.path
root_dir = pathlib.Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from docurag.retrieval.dense_search import DenseRetriever
from docurag.retrieval.result import SearchResult

# Test Query Benchmark Definition
# Each query defines its text, category, and an identifier predicate for relevant chunks.
BENCHMARK_QUERIES = [
    # -------------------------------------------------------------
    # Category A: Exact Technical Queries
    # -------------------------------------------------------------
    {
        "query": "if statements",
        "category": "Exact Technical",
        "description": "Control flow conditional branching syntax",
        "match_fn": lambda c: "if" in (c.get("content") or "").lower() and "statement" in (c.get("section_title") or "").lower(),
    },
    {
        "query": "for statements",
        "category": "Exact Technical",
        "description": "Iterating over sequences using for loops",
        "match_fn": lambda c: "for" in (c.get("content") or "").lower() or "for statements" in (c.get("section_title") or "").lower(),
    },
    {
        "query": "defining functions",
        "category": "Exact Technical",
        "description": "Creating functions using the def keyword",
        "match_fn": lambda c: "def " in (c.get("content") or "") or "defining functions" in (c.get("section_title") or "").lower(),
    },
    {
        "query": "dictionaries",
        "category": "Exact Technical",
        "description": "Mapping data structures with key-value pairs",
        "match_fn": lambda c: "dictionaries" in (c.get("section_title") or "").lower() or "dict" in (c.get("content") or "").lower(),
    },
    {
        "query": "Reading and Writing Files",
        "category": "Exact Technical",
        "description": "File I/O operations and methods",
        "match_fn": lambda c: "reading and writing files" in (c.get("section_title") or "").lower(),
    },
    {
        "query": "Handling Exceptions",
        "category": "Exact Technical",
        "description": "Try and except error handling blocks",
        "match_fn": lambda c: "handling exceptions" in (c.get("section_title") or "").lower() or "try" in (c.get("content") or "").lower(),
    },
    {
        "query": "Inheritance",
        "category": "Exact Technical",
        "description": "Object-oriented class inheritance in Python",
        "match_fn": lambda c: "inheritance" in (c.get("section_title") or "").lower(),
    },
    # -------------------------------------------------------------
    # Category B: Conceptual / Semantic Queries
    # -------------------------------------------------------------
    {
        "query": "How do you make decisions and branch your code based on conditions?",
        "category": "Conceptual / Semantic",
        "description": "Conceptual branching and conditional execution",
        "match_fn": lambda c: "if statements" in (c.get("section_title") or "").lower() or "condition" in (c.get("content") or "").lower(),
    },
    {
        "query": "How can I loop over items in a list or a sequence?",
        "category": "Conceptual / Semantic",
        "description": "Conceptual iteration and looping structures",
        "match_fn": lambda c: "for statements" in (c.get("section_title") or "").lower() or "iterate" in (c.get("content") or "").lower(),
    },
    {
        "query": "What is the best way to organize code into reusable blocks and modules?",
        "category": "Conceptual / Semantic",
        "description": "Code organization, modules, and packages",
        "match_fn": lambda c: "modules" in (c.get("section_title") or "").lower() or "package" in (c.get("content") or "").lower(),
    },
    {
        "query": "How do I catch errors so my script doesn't crash when something goes wrong?",
        "category": "Conceptual / Semantic",
        "description": "Error handling and exception management",
        "match_fn": lambda c: "handling exceptions" in (c.get("section_title") or "").lower() or "exception" in (c.get("content") or "").lower(),
    },
    {
        "query": "How do classes and object-oriented programming work in Python?",
        "category": "Conceptual / Semantic",
        "description": "Classes, objects, and object-oriented basics",
        "match_fn": lambda c: "classes" in (c.get("section_title") or "").lower() or "class" in (c.get("content") or "").lower(),
    },
    {
        "query": "How do you store collections of key-value data efficiently?",
        "category": "Conceptual / Semantic",
        "description": "Dictionaries and mapping structures",
        "match_fn": lambda c: "dictionaries" in (c.get("section_title") or "").lower() or "key" in (c.get("content") or "").lower(),
    },
]


def evaluate_query(
    query_item: Dict[str, Any],
    bm25: BM25Retriever,
    dense: DenseRetriever,
    hybrid: HybridRetriever,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate a single query across BM25, Dense, and Hybrid retrieval modes."""
    query = query_item["query"]
    match_fn = query_item["match_fn"]

    bm25_res = bm25.retrieve(query, top_k=top_k)
    dense_res = dense.retrieve(query, top_k=top_k)
    hybrid_res = hybrid.retrieve(query, top_k=top_k)

    def find_rank(results: List[SearchResult]) -> Optional[int]:
        for rank, res in enumerate(results, 1):
            chunk_dict = {"content": res.content, "section_title": res.metadata.get("section_title", "")}
            if match_fn(chunk_dict):
                return rank
        return None

    bm25_rank = find_rank(bm25_res)
    dense_rank = find_rank(dense_res)
    hybrid_rank = find_rank(hybrid_res)

    return {
        "query": query,
        "category": query_item["category"],
        "description": query_item["description"],
        "bm25_rank": bm25_rank,
        "dense_rank": dense_rank,
        "hybrid_rank": hybrid_rank,
        "bm25_top1": bm25_res[0].metadata.get("section_title") if bm25_res else "None",
        "dense_top1": dense_res[0].metadata.get("section_title") if dense_res else "None",
        "hybrid_top1": hybrid_res[0].metadata.get("section_title") if hybrid_res else "None",
    }


def compute_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute Recall@1, Recall@3, Recall@5, and MRR@5 for each retrieval mode."""
    modes = ["bm25", "dense", "hybrid"]
    metrics: Dict[str, Dict[str, float]] = {m: {} for m in modes}
    n = len(eval_records)
    if n == 0:
        return metrics

    for mode in modes:
        rank_key = f"{mode}_rank"
        ranks = [r[rank_key] for r in eval_records]

        r1 = sum(1 for r in ranks if r is not None and r == 1) / n
        r3 = sum(1 for r in ranks if r is not None and r <= 3) / n
        r5 = sum(1 for r in ranks if r is not None and r <= 5) / n
        mrr = sum((1.0 / r) if (r is not None and r <= 5) else 0.0 for r in ranks) / n

        metrics[mode] = {
            "Recall@1": round(r1 * 100, 1),
            "Recall@3": round(r3 * 100, 1),
            "Recall@5": round(r5 * 100, 1),
            "MRR@5": round(mrr, 4),
        }

    return metrics


def generate_markdown_report(
    all_records: List[Dict[str, Any]],
    overall_metrics: Dict[str, Dict[str, float]],
    tech_metrics: Dict[str, Dict[str, float]],
    sem_metrics: Dict[str, Dict[str, float]],
    output_path: pathlib.Path,
) -> str:
    """Generate the comprehensive Markdown report detailing findings and rank shifts."""
    lines: List[str] = []

    lines.append("# Comparative Retrieval Performance Report")
    lines.append("## BM25 Lexical vs. MiniLM Dense vs. Hybrid Reciprocal Rank Fusion (RRF)")
    lines.append("")
    lines.append(
        "This evaluation benchmarks the three retrieval mechanisms on the processed document chunks "
        "produced by the `DocuRAG` data pipeline. The test set comprises two distinct query profiles: "
        "**Exact Technical Queries** (targeting specific concepts, syntax statements, and language features) "
        "and **Conceptual / Semantic Queries** (targeting natural language explanations and how-to questions)."
    )
    lines.append("")
    lines.append("---")
    lines.append("## 1. Executive Summary & Aggregate Metrics")
    lines.append("")
    lines.append(
        "These metrics describe this labelled benchmark and should not be read as a general guarantee that one "
        "retrieval mode is always best. Compare the aggregate metrics with the per-query ranks and category breakdown."
    )
    lines.append("")
    lines.append("### Overall Benchmark Performance (All 13 Queries)")
    lines.append("")
    lines.append("| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for mode, label in [("bm25", "BM25 (Lexical)"), ("dense", "Dense (MiniLM-L6-v2)"), ("hybrid", "**Hybrid (RRF k=60)**")]:
        m = overall_metrics[mode]
        lines.append(f"| {label} | {m['Recall@1']}% | {m['Recall@3']}% | {m['Recall@5']}% | {m['MRR@5']:.4f} |")

    lines.append("")
    lines.append("### Category Breakdown")
    lines.append("")
    lines.append("#### Exact Technical Queries (7 Queries)")
    lines.append("")
    lines.append("| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for mode, label in [("bm25", "BM25 (Lexical)"), ("dense", "Dense (MiniLM-L6-v2)"), ("hybrid", "**Hybrid (RRF k=60)**")]:
        m = tech_metrics[mode]
        lines.append(f"| {label} | {m['Recall@1']}% | {m['Recall@3']}% | {m['Recall@5']}% | {m['MRR@5']:.4f} |")

    lines.append("")
    lines.append("#### Conceptual / Semantic Queries (6 Queries)")
    lines.append("")
    lines.append("| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for mode, label in [("bm25", "BM25 (Lexical)"), ("dense", "Dense (MiniLM-L6-v2)"), ("hybrid", "**Hybrid (RRF k=60)**")]:
        m = sem_metrics[mode]
        lines.append(f"| {label} | {m['Recall@1']}% | {m['Recall@3']}% | {m['Recall@5']}% | {m['MRR@5']:.4f} |")

    lines.append("")
    lines.append("---")
    lines.append("## 2. Detailed Per-Query Retrieval & Rank Shift Log")
    lines.append("")
    lines.append(
        "The table below details the 1-based rank at which the ground-truth relevant chunk was retrieved "
        "by each mode (`-` indicates not retrieved in Top 5)."
    )
    lines.append("")
    lines.append("| Query | Category | BM25 Rank | Dense Rank | Hybrid Rank | Hybrid Top-1 Section |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :--- |")

    for rec in all_records:
        b_rank = str(rec["bm25_rank"]) if rec["bm25_rank"] is not None else "-"
        d_rank = str(rec["dense_rank"]) if rec["dense_rank"] is not None else "-"
        h_rank = f"**{rec['hybrid_rank']}**" if rec["hybrid_rank"] is not None else "-"
        q_display = f"{rec['query'][:42]}..." if len(rec['query']) > 45 else rec['query']
        lines.append(
            f"| `{q_display}` | {rec['category']} | {b_rank} | {d_rank} | {h_rank} | {rec['hybrid_top1']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("## 3. Interpreting the Comparison")
    lines.append("")
    lines.append("- BM25 can help with exact terms and identifiers in the indexed documentation.")
    lines.append("- Dense retrieval can help when a query paraphrases the source wording.")
    lines.append("- RRF combines ranks; use the benchmark rows above to judge its effect on this corpus.")
    lines.append("")
    lines.append("---")
    lines.append("## 4. Downstream Handoff to the Cross-Encoder Reranker")
    lines.append(
        "1. **Contract Stability**: `HybridRetriever.retrieve(query, top_k=5)` returns `List[SearchResult]` containing "
        "`chunk_id`, `content`, `score`, and `metadata`. This fulfills the input contract for "
        "`CrossEncoderReranker.rerank(query, candidates, top_k=3)`."
    )
    lines.append(
        "2. **Candidate Pool**: Candidate count is a recall/latency trade-off. Tune it on labelled queries for the target corpus; "
        "a fixed candidate count does not guarantee a particular recall."
    )
    lines.append(
        "3. **Provenance**: Retrieved chunks retain `source_url`, `doc_id`, section, and retrieval metadata when supplied. "
        "Source URLs require manifest entries. The answer stage is extractive, not LLM-generated."
    )

    report_content = "\n".join(lines)
    output_path.write_text(report_content, encoding="utf-8")
    return report_content


def main() -> None:
    from docurag.retrieval.bm25_search import BM25Retriever
    from docurag.retrieval.hybrid_search import HybridRetriever

    chunks_path = root_dir / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Run 'python -m data_pipeline.run_pipeline' first.")
        sys.exit(1)

    print("Initializing Retrievers for Evaluation Benchmark...")
    bm25 = BM25Retriever.from_json(chunks_path)
    dense = DenseRetriever.from_json(chunks_path)
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_k=60)

    print(f"\nRunning Benchmark on {len(BENCHMARK_QUERIES)} queries...")
    eval_records: List[Dict[str, Any]] = []

    for item in BENCHMARK_QUERIES:
        record = evaluate_query(item, bm25, dense, hybrid, top_k=5)
        eval_records.append(record)

    tech_records = [r for r in eval_records if r["category"] == "Exact Technical"]
    sem_records = [r for r in eval_records if r["category"] == "Conceptual / Semantic"]

    overall_metrics = compute_metrics(eval_records)
    tech_metrics = compute_metrics(tech_records)
    sem_metrics = compute_metrics(sem_records)

    print("\n=======================================================")
    print("EVALUATION BENCHMARK RESULTS")
    print("=======================================================")
    print(f"Overall Metrics (N={len(eval_records)}):")
    for mode in ["bm25", "dense", "hybrid"]:
        print(f"  {mode.upper():<7}: {overall_metrics[mode]}")

    print(f"\nExact Technical Queries (N={len(tech_records)}):")
    for mode in ["bm25", "dense", "hybrid"]:
        print(f"  {mode.upper():<7}: {tech_metrics[mode]}")

    print(f"\nConceptual / Semantic Queries (N={len(sem_records)}):")
    for mode in ["bm25", "dense", "hybrid"]:
        print(f"  {mode.upper():<7}: {sem_metrics[mode]}")

    report_path = root_dir / "hybrid_search_report.md"
    generate_markdown_report(eval_records, overall_metrics, tech_metrics, sem_metrics, report_path)
    print(f"\nGenerated comprehensive report at: {report_path}")


if __name__ == "__main__":
    main()