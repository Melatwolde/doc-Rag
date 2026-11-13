"""Evaluate whether a relevant paragraph is present at top-3, top-5, and top-10.

Uses the labelled benchmark maintained for the project and writes a focused dense
retrieval report. Run after `scripts/build_index.py`, or it will embed chunks once.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docurag.retrieval.dense_search import DenseRetriever
from evaluate_retrieval import BENCHMARK_QUERIES


def relevant_rank(query_item: dict[str, Any], results: list[Any]) -> int | None:
    for rank, result in enumerate(results, start=1):
        candidate = {"content": result.content, "section_title": result.metadata.get("section_title", "")}
        if query_item["match_fn"](candidate):
            return rank
    return None


def main() -> None:
    index_path = ROOT / "data/processed/dense_index"
    retriever = DenseRetriever.load_index(index_path) if index_path.exists() else DenseRetriever.from_json(ROOT / "data/processed/chunks.json")
    cutoffs = (3, 5, 10)
    records = []
    for item in BENCHMARK_QUERIES:
        results = retriever.retrieve(item["query"], top_k=max(cutoffs))
        rank = relevant_rank(item, results)
        records.append({
            "query": item["query"], "category": item["category"], "relevant_rank": rank,
            **{f"hit_at_{k}": rank is not None and rank <= k for k in cutoffs},
        })

    summary = {f"Recall@{k}": round(100 * sum(r[f"hit_at_{k}"] for r in records) / len(records), 1) for k in cutoffs}
    report = ["# Dense Retrieval Performance Report", "", "Model: `all-MiniLM-L6-v2`; similarity: cosine (normalized inner product).", "", "| Cutoff | Relevant paragraph retrieved |", "| :--- | :---: |"]
    report.extend(f"| Top-{k} | {summary[f'Recall@{k}']}% |" for k in cutoffs)
    report.extend(["", "## Per-query results", "", "| Query | Category | First relevant rank |", "| :--- | :--- | :---: |"])
    report.extend(f"| `{r['query'][:70]}` | {r['category']} | {r['relevant_rank'] or '-'} |" for r in records)
    (ROOT / "dense_retrieval_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (ROOT / "data/processed/dense_retrieval_metrics.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote dense_retrieval_report.md and data/processed/dense_retrieval_metrics.json")


if __name__ == "__main__":
    main()
