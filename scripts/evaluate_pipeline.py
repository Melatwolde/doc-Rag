"""Evaluate retrieval, grounded answers, citations, abstention, and latency."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List

root_dir = pathlib.Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from docurag.pipeline import HybridRAGPipeline


SYSTEM_EVAL_TEST_CASES = [
    {
        "query": "How can I implement JWT authentication and refresh an expired token?",
        "relevant_chunk_ids": [
            "485bb80f-7058-4431-ab02-536397eaa619",
            "32c53daa-aaa0-47ec-b4ef-f4e4d900ec45",
        ],
        "is_answerable": True,
        "expected_keywords": ["JWT", "token", "refresh", "POST /api/v1/auth/refresh"],
    },
    {
        "query": "What error codes are raised when the database connection pool is exhausted?",
        "relevant_chunk_ids": ["cec75777-c4f8-44fd-9e1a-eae46c133e26"],
        "is_answerable": True,
        "expected_keywords": ["ERR_CONN_TIMEOUT", "ERR_POOL_EXHAUSTED", "database.pool_size"],
    },
    {
        "query": "What is the quantum encryption protocol supported by the database connector?",
        "relevant_chunk_ids": [],
        "is_answerable": False,
        "expected_keywords": [],
    },
]


def run_system_evaluation(
    test_cases: List[Dict[str, Any]],
    output_path: pathlib.Path | None = None,
    *,
    confidence_threshold: float = 0.45,
    top_k_retrieve: int = 10,
    top_k_rerank: int = 3,
    hybrid_retriever: Any | None = None,
    cross_encoder_reranker: Any | None = None,
    use_cross_encoder: bool = True,
) -> Dict[str, Any]:
    pipeline = HybridRAGPipeline(
        chunks_path=root_dir / "data" / "processed" / "chunks.json",
        confidence_threshold=confidence_threshold,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        hybrid_retriever=hybrid_retriever,
        cross_encoder_reranker=cross_encoder_reranker,
        use_cross_encoder=use_cross_encoder,
    )

    recall_scores: List[float] = []
    mrr_scores: List[float] = []
    answer_accuracies: List[float] = []
    citation_accuracies: List[float] = []
    latencies: List[float] = []
    correct_abstentions = 0
    missed_abstentions = 0
    false_abstentions = 0

    for case in test_cases:
        start_time = time.perf_counter()
        response = pipeline.run(case["query"])
        latencies.append(time.perf_counter() - start_time)

        relevant_ids = set(case["relevant_chunk_ids"])
        retrieved_ids = [
            chunk.get("chunk_id")
            for chunk in response.get("retrieved_chunks", [])
        ]
        abstaining = response["is_abstaining"]

        if relevant_ids:
            recall_scores.append(
                len(set(retrieved_ids[:3]).intersection(relevant_ids)) / len(relevant_ids)
            )
            reciprocal_rank = next(
                (
                    1.0 / rank
                    for rank, chunk_id in enumerate(retrieved_ids, start=1)
                    if chunk_id in relevant_ids
                ),
                0.0,
            )
            mrr_scores.append(reciprocal_rank)

        if case["is_answerable"]:
            false_abstentions += int(abstaining)
            expected_keywords = case["expected_keywords"]
            answer = response.get("answer", "").lower()
            answer_accuracies.append(
                sum(keyword.lower() in answer for keyword in expected_keywords)
                / len(expected_keywords)
                if expected_keywords and not abstaining
                else (0.0 if expected_keywords else float(not abstaining))
            )
            citations = response.get("citations", [])
            citation_accuracies.append(
                sum(citation in relevant_ids for citation in citations) / len(citations)
                if citations and not abstaining
                else 0.0
            )
        elif abstaining:
            correct_abstentions += 1
        else:
            missed_abstentions += 1

    answerable_count = sum(bool(case["is_answerable"]) for case in test_cases)
    unanswerable_count = len(test_cases) - answerable_count
    report = {
        "confidence_threshold": confidence_threshold,
        "top_k_retrieve": top_k_retrieve,
        "top_k_rerank": top_k_rerank,
        "reranker": "cross_encoder" if use_cross_encoder else "hybrid_confidence",
        "total_test_cases": len(test_cases),
        "mean_latency_sec": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "recall_at_3": round(sum(recall_scores) / len(recall_scores) * 100, 2) if recall_scores else 0.0,
        "mean_mrr": round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else 0.0,
        "correct_abstention_rate": round(
            correct_abstentions / max(1, unanswerable_count) * 100, 2
        ),
        "missed_abstention_rate": round(
            missed_abstentions / max(1, unanswerable_count) * 100, 2
        ),
        "false_abstention_rate": round(
            false_abstentions / max(1, answerable_count) * 100, 2
        ),
        "mean_answer_accuracy": round(
            sum(answer_accuracies) / len(answer_accuracies) * 100, 2
        ) if answer_accuracies else 0.0,
        "mean_citation_accuracy": round(
            sum(citation_accuracies) / len(citation_accuracies) * 100, 2
        ) if citation_accuracies else 0.0,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
        print(f"Evaluation report saved to: {output_path}")
    return report


def run_threshold_comparison(
    test_cases: List[Dict[str, Any]],
    output_path: pathlib.Path,
    *,
    use_cross_encoder: bool = True,
) -> List[Dict[str, Any]]:
    from docurag.retrieval.hybrid_search import HybridRetriever

    retriever = HybridRetriever.from_json(root_dir / "data" / "processed" / "chunks.json")
    reranker = None
    if use_cross_encoder:
        from docurag.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker()

    comparisons = []
    for threshold in (0.35, 0.45, 0.55):
        for retrieve_k, rerank_k in ((5, 1), (10, 3)):
            report = run_system_evaluation(
                test_cases,
                confidence_threshold=threshold,
                top_k_retrieve=retrieve_k,
                top_k_rerank=rerank_k,
                hybrid_retriever=retriever,
                cross_encoder_reranker=reranker,
                use_cross_encoder=use_cross_encoder,
            )
            comparisons.append(report)
            print(
                f"threshold={threshold:.2f} retrieve={retrieve_k} rerank={rerank_k} "
                f"Recall@3={report['recall_at_3']:.2f}% MRR={report['mean_mrr']:.4f} "
                f"answer={report['mean_answer_accuracy']:.2f}% "
                f"citation={report['mean_citation_accuracy']:.2f}% "
                f"abstain={report['correct_abstention_rate']:.2f}% "
                f"false-abstain={report['false_abstention_rate']:.2f}% "
                f"latency={report['mean_latency_sec']:.4f}s"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparisons, indent=4), encoding="utf-8")
    print(f"Threshold comparison saved to: {output_path}")
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--without-cross-encoder",
        action="store_true",
        help="Skip cross-encoder reranking and use hybrid confidence ordering.",
    )
    args = parser.parse_args()
    report_path = root_dir / "data" / "evaluation" / "system_evaluation_report.json"
    run_threshold_comparison(
        SYSTEM_EVAL_TEST_CASES,
        report_path,
        use_cross_encoder=not args.without_cross_encoder,
    )


if __name__ == "__main__":
    main()
