"""scripts/evaluate_generation.py

Evaluation script for the full RAG generation pipeline.

Computes three required metrics:
  1. Retrieval Recall@3  — fraction of answerable questions where the
                           correct chunk appears in the top-3 retrieved results.
  2. % Answers Correct   — fraction of answerable answers with
                           significant token overlap against ground truth.
  3. % I-don't-know      — abstention rate on unanswerable questions.

Additionally reports:
  * Grounding rate       — fraction of answers lexically grounded in cited passages.
  * Average CE score     — mean cross-encoder score across reranked passages.

Usage::

    python scripts/evaluate_generation.py \\
        --benchmark data/benchmark/qa_benchmark.json \\
        --chunks    data/processed/chunks.json \\
        --top-k     3 \\
        --out       data/benchmark/eval_report.json

"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
import time
from typing import Any, Dict, List

# Allow running from repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _token_overlap(ref: str, hyp: str) -> float:
    """F1-style token overlap between reference and hypothesis."""
    ref_toks = _tokenize(ref)
    hyp_toks = _tokenize(hyp)
    if not ref_toks or not hyp_toks:
        return 0.0
    common = ref_toks & hyp_toks
    precision = len(common) / len(hyp_toks)
    recall = len(common) / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_chunk_ids: List[str], relevant_ids: List[str]) -> bool:
    """Return True if any relevant chunk ID appears in the retrieved list."""
    return bool(set(relevant_ids) & set(retrieved_chunk_ids))


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate(
    benchmark_path: str,
    chunks_path: str,
    top_k: int = 3,
    correct_threshold: float = 0.25,
    llm_backend: str = "auto",
    out_path: str | None = None,
) -> Dict[str, Any]:
    from docurag.pipeline import RAGPipeline
    from docurag.reranking.cross_encoder import CrossEncoderReranker

    logger.info("Loading benchmark: %s", benchmark_path)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    logger.info("Building RAG pipeline from: %s", chunks_path)
    pipeline = RAGPipeline.from_chunks_json(
        chunks_path,
        llm_backend=llm_backend,
        retrieve_top_k=top_k * 3,   # over-retrieve before reranking
        rerank_top_k=top_k,
    )

    answerable = [q for q in benchmark if q["answerable"]]
    unanswerable = [q for q in benchmark if not q["answerable"]]

    logger.info(
        "Benchmark: %d total (%d answerable, %d unanswerable)",
        len(benchmark), len(answerable), len(unanswerable),
    )

    # --- Metric accumulators ---
    recall_hits = 0
    correct_hits = 0
    abstention_on_unanswerable = 0
    grounded_count = 0
    ce_scores: List[float] = []
    per_question_results: List[Dict] = []

    # --- Evaluate answerable questions ---
    for item in answerable:
        qid = item["id"]
        question = item["question"]
        ref_answer = item["answer"]
        relevant_ids = item["relevant_chunk_ids"]

        logger.info("[%s] %s", qid, question[:70])
        t0 = time.time()

        # Run pipeline
        result = pipeline.answer(question, verbose=False)
        elapsed = time.time() - t0

        # Retrieved chunk IDs (from grounded passages)
        retrieved_ids = [p.chunk_id for p in result.grounded_passages]

        # Metric 1: Recall@K
        hit = recall_at_k(retrieved_ids, relevant_ids)
        recall_hits += int(hit)

        # Metric 2: Answer correctness (token overlap F1)
        overlap = _token_overlap(ref_answer, result.answer)
        correct = overlap >= correct_threshold
        correct_hits += int(correct)

        # Grounding check
        grounding = pipeline.generator.check_grounding(
            result.answer, result.grounded_passages
        )
        grounded_count += int(grounding["grounded"])

        # Collect CE scores
        for p in result.grounded_passages:
            score = p.metadata.get("cross_encoder_score")
            if score is not None:
                ce_scores.append(score)

        per_question_results.append({
            "id": qid,
            "question": question,
            "reference_answer": ref_answer,
            "generated_answer": result.answer,
            "is_abstention": result.is_abstention,
            "recall_hit": hit,
            "token_f1": round(overlap, 4),
            "correct": correct,
            "grounded": grounding["grounded"],
            "max_overlap": grounding["max_overlap"],
            "citations": [c.index for c in result.citations],
            "retrieved_chunk_ids": retrieved_ids,
            "elapsed_s": round(elapsed, 2),
        })

    # --- Evaluate unanswerable questions ---
    for item in unanswerable:
        qid = item["id"]
        question = item["question"]

        logger.info("[%s] (unanswerable) %s", qid, question[:70])
        t0 = time.time()
        result = pipeline.answer(question, verbose=False)
        elapsed = time.time() - t0

        abstained = result.is_abstention
        abstention_on_unanswerable += int(abstained)

        per_question_results.append({
            "id": qid,
            "question": question,
            "answerable": False,
            "generated_answer": result.answer,
            "is_abstention": abstained,
            "elapsed_s": round(elapsed, 2),
        })

    # --- Aggregate metrics ---
    n_answerable = len(answerable)
    n_unanswerable = len(unanswerable)

    recall_at_k_pct = recall_hits / n_answerable if n_answerable else 0.0
    pct_correct = correct_hits / n_answerable if n_answerable else 0.0
    pct_abstention = (
        abstention_on_unanswerable / n_unanswerable if n_unanswerable else 0.0
    )
    pct_grounded = grounded_count / n_answerable if n_answerable else 0.0
    avg_ce_score = sum(ce_scores) / len(ce_scores) if ce_scores else 0.0

    report = {
        "summary": {
            "total_questions": len(benchmark),
            "answerable": n_answerable,
            "unanswerable": n_unanswerable,
            "top_k": top_k,
            "correct_threshold_f1": correct_threshold,
            "llm_backend": pipeline.generator.backend,
            "reranker_model": pipeline.reranker.model_name,
        },
        "metrics": {
            "retrieval_recall_at_k": round(recall_at_k_pct * 100, 2),
            "pct_answers_correct": round(pct_correct * 100, 2),
            "pct_idontknow_on_unanswerable": round(pct_abstention * 100, 2),
            "pct_answers_grounded": round(pct_grounded * 100, 2),
            "avg_cross_encoder_score": round(avg_ce_score, 4),
            "recall_hits": recall_hits,
            "correct_hits": correct_hits,
            "abstention_hits": abstention_on_unanswerable,
        },
        "per_question": per_question_results,
    }

    # --- Print summary ---
    print("\n" + "=" * 65)
    print("  DocuRAG Generation Evaluation Report")
    print("=" * 65)
    print(f"  Questions evaluated : {len(benchmark)} total")
    print(f"  Answerable          : {n_answerable}")
    print(f"  Unanswerable        : {n_unanswerable}")
    print(f"  LLM backend         : {pipeline.generator.backend}")
    print(f"  Reranker model      : {pipeline.reranker.model_name}")
    print("-" * 65)
    print(f"  Retrieval Recall@{top_k}  : {recall_at_k_pct*100:.1f}%  ({recall_hits}/{n_answerable})")
    print(f"  % Answers Correct   : {pct_correct*100:.1f}%  ({correct_hits}/{n_answerable})")
    print(f"  % I-don't-know      : {pct_abstention*100:.1f}%  ({abstention_on_unanswerable}/{n_unanswerable})")
    print(f"  % Answers Grounded  : {pct_grounded*100:.1f}%  ({grounded_count}/{n_answerable})")
    print(f"  Avg CE Score        : {avg_ce_score:.4f}")
    print("=" * 65 + "\n")

    # --- Write output ---
    if out_path:
        out = pathlib.Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved to %s", out_path)

        # Also write markdown summary
        md_path = out.with_suffix(".md")
        _write_markdown_report(report, md_path)
        logger.info("Markdown report saved to %s", md_path)

    return report


def _write_markdown_report(report: Dict, path: pathlib.Path) -> None:
    s = report["summary"]
    m = report["metrics"]

    lines = [
        "# DocuRAG Generation Evaluation Report\n",
        "## Summary\n",
        f"| Setting | Value |",
        f"|---------|-------|",
        f"| Total questions | {s['total_questions']} |",
        f"| Answerable | {s['answerable']} |",
        f"| Unanswerable | {s['unanswerable']} |",
        f"| Retrieval top-k | {s['top_k']} |",
        f"| LLM backend | {s['llm_backend']} |",
        f"| Reranker model | {s['reranker_model']} |",
        "",
        "## Metrics\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Retrieval Recall@{s['top_k']}** | **{m['retrieval_recall_at_k']}%** |",
        f"| **% Answers Correct** | **{m['pct_answers_correct']}%** |",
        f"| **% \"I don't know\" on Unanswerable** | **{m['pct_idontknow_on_unanswerable']}%** |",
        f"| % Answers Grounded | {m['pct_answers_grounded']}% |",
        f"| Avg Cross-Encoder Score | {m['avg_cross_encoder_score']} |",
        "",
        "## Per-Question Results (Answerable)\n",
        "| ID | Question | Recall Hit | F1 | Correct | Grounded |",
        "|----|----------|------------|----|---------|----------|",
    ]

    for q in report["per_question"]:
        if not q.get("answerable", True):
            continue
        lines.append(
            f"| {q['id']} | {q['question'][:50]}… | "
            f"{'✅' if q['recall_hit'] else '❌'} | "
            f"{q['token_f1']:.2f} | "
            f"{'✅' if q['correct'] else '❌'} | "
            f"{'✅' if q['grounded'] else '❌'} |"
        )

    lines.extend([
        "",
        "## Abstention on Unanswerable Questions\n",
        "| ID | Question | Abstained |",
        "|----|----------|-----------|",
    ])

    for q in report["per_question"]:
        if q.get("answerable", True):
            continue
        lines.append(
            f"| {q['id']} | {q['question'][:60]}… | "
            f"{'✅' if q['is_abstention'] else '❌'} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DocuRAG generation pipeline on the Q&A benchmark."
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/qa_benchmark.json",
        help="Path to Q&A benchmark JSON file.",
    )
    parser.add_argument(
        "--chunks",
        default="data/processed/chunks.json",
        help="Path to processed chunks JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of passages to rerank to for each question (default 3).",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.25,
        help="Minimum token F1 score to count an answer as correct (default 0.25).",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "local", "openai"],
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--out",
        default="data/benchmark/eval_report.json",
        help="Output path for the JSON report.",
    )
    args = parser.parse_args()

    evaluate(
        benchmark_path=args.benchmark,
        chunks_path=args.chunks,
        top_k=args.top_k,
        correct_threshold=args.correct_threshold,
        llm_backend=args.backend,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
