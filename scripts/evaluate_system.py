"""Evaluate retrieval grounding, abstention, citation support, and latency.

The script measures an extractive pipeline, not free-form LLM answer quality.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docurag.evaluation.metrics import abstention_rates, answer_accuracy, citation_accuracy, latency_summary
from docurag.pipeline import HybridRAGPipeline
from evaluate_retrieval import BENCHMARK_QUERIES

UNANSWERABLE_QUERIES = [
    "How do I configure a Kubernetes service mesh in this project?",
    "What is the company holiday policy?",
    "How can I train a computer vision model with these documents?",
]


def is_relevant(item: dict[str, Any], match_fn: Any) -> bool:
    return bool(match_fn({"content": item.get("content", ""), "section_title": item.get("section_title") or item.get("metadata", {}).get("section_title", "")}))


def evaluate_threshold(threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pipeline = HybridRAGPipeline(
        chunks_path=ROOT / "data" / "processed" / "chunks.json",
        confidence_threshold=threshold,
        top_k_retrieve=10,
        top_k_rerank=3,
        use_cross_encoder=False,
    )
    records: list[dict[str, Any]] = []
    for item in BENCHMARK_QUERIES:
        response = pipeline.run(item["query"])
        relevant_ids = {chunk["chunk_id"] for chunk in response["reranked_chunks"] if is_relevant(chunk, item["match_fn"])}
        cited_ids = {citation["chunk_id"] for citation in response["citations"]}
        records.append({
            "query": item["query"], "answerable": True, "is_abstaining": response["is_abstaining"],
            "has_relevant_evidence": bool(relevant_ids),
            "has_relevant_citation": bool(relevant_ids.intersection(cited_ids)),
            "latency_ms": response["latency_ms"],
        })
    for query in UNANSWERABLE_QUERIES:
        response = pipeline.run(query)
        records.append({
            "query": query, "answerable": False, "is_abstaining": response["is_abstaining"],
            "has_relevant_evidence": False, "has_relevant_citation": False,
            "latency_ms": response["latency_ms"],
        })
    metrics = {
        "threshold": threshold,
        "grounded_answer_accuracy": round(answer_accuracy(records) * 100, 1),
        "citation_accuracy": round(citation_accuracy(records) * 100, 1),
        **{key: round(value * 100, 1) for key, value in abstention_rates(records).items()},
        **latency_summary(record["latency_ms"] for record in records),
    }
    return metrics, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocuRAG reliability metrics")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.45, 0.7])
    args = parser.parse_args()
    all_results = [evaluate_threshold(threshold) for threshold in args.thresholds]
    metrics = [result[0] for result in all_results]
    payload = {"metrics": metrics, "records": {str(metric["threshold"]): records for metric, records in all_results}}
    (ROOT / "data" / "processed" / "system_evaluation_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# System Evaluation and Reliability Report", "", "Answer accuracy is **grounded answer accuracy**: an answer is counted only when it does not abstain and its retrieved evidence contains the labelled relevant passage. Citation accuracy requires a citation to such a passage.", "", "| Threshold | Grounded answer accuracy | Citation accuracy | Correct abstention | False abstention | Mean latency (ms) | P95 latency (ms) |", "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for metric in metrics:
        lines.append(f"| {metric['threshold']:.2f} | {metric['grounded_answer_accuracy']}% | {metric['citation_accuracy']}% | {metric['correct_abstention_rate']}% | {metric['false_abstention_rate']}% | {metric['mean_latency_ms']} | {metric['p95_latency_ms']} |")
    lines.extend(["", "The evaluation includes 13 labelled answerable retrieval queries and 3 explicitly unanswerable queries. It reports measured values only when this script is run."])
    (ROOT / "system_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Wrote system_evaluation_report.md and data/processed/system_evaluation_metrics.json")


if __name__ == "__main__":
    main()
