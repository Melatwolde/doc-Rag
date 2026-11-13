"""Deterministic integration checks for the RAG pipeline and abstention guard."""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
import importlib.util
from typing import Any

root_dir = pathlib.Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest

from docurag.pipeline import HybridRAGPipeline


@dataclass
class StubResult:
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any]


class StubRetriever:
    def __init__(self, results: list[StubResult]) -> None:
        self.results = results
        self.query: str | None = None

    def retrieve(self, query: str, top_k: int) -> list[StubResult]:
        self.query = query
        return self.results[:top_k]


class StubReranker:
    def __init__(self) -> None:
        self.called = False

    def rerank(self, query: str, candidates: list[StubResult], top_k: int) -> list[StubResult]:
        self.called = True
        return list(reversed(candidates))[:top_k]


def _result(
    chunk_id: str,
    *,
    dense_score: float,
    lexical_overlap: int,
    bm25_score: float,
) -> StubResult:
    return StubResult(
        chunk_id=chunk_id,
        content="Refresh tokens are exchanged at POST /api/v1/auth/refresh.",
        score=2 / 61,
        metadata={
            "doc_id": "doc-1",
            "section_title": "Refresh Token Exchange",
            "rrf_k": 60,
            "term_overlap_count": lexical_overlap,
            "retrieval_sources": {
                "bm25": {"rank": 1, "raw_score": bm25_score},
                "dense": {"rank": 1, "raw_score": dense_score},
            },
        },
    )


def test_pipeline_generates_grounded_answer_and_citations() -> None:
    retriever = StubRetriever(
        [_result("chunk-1", dense_score=0.8, lexical_overlap=2, bm25_score=1.2)]
    )
    pipeline = HybridRAGPipeline(hybrid_retriever=retriever, use_cross_encoder=False)

    response = pipeline.run("  How do I refresh a token?  ")

    assert retriever.query == "How do I refresh a token?"
    assert response["is_abstaining"] is False
    assert response["citations"] == ["chunk-1"]
    assert "POST /api/v1/auth/refresh" in response["answer"]
    assert response["reranked_chunks"][0]["doc_id"] == "doc-1"


def test_pipeline_abstains_when_retrieval_has_no_evidence() -> None:
    retriever = StubRetriever(
        [_result("chunk-1", dense_score=0.2, lexical_overlap=0, bm25_score=0.0)]
    )
    pipeline = HybridRAGPipeline(hybrid_retriever=retriever, use_cross_encoder=False)

    response = pipeline.run("quantum encryption protocol")

    assert response["is_abstaining"] is True
    assert response["citations"] == []
    assert "ERR_INSUFFICIENT_EVIDENCE" in response["abstention_reason"]


def test_pipeline_rejects_empty_query() -> None:
    pipeline = HybridRAGPipeline(
        hybrid_retriever=StubRetriever([]),
        use_cross_encoder=False,
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        pipeline.run("   ")


def test_pipeline_uses_injected_reranker() -> None:
    results = [
        _result("chunk-1", dense_score=0.8, lexical_overlap=2, bm25_score=1.2),
        _result("chunk-2", dense_score=0.7, lexical_overlap=2, bm25_score=0.9),
    ]
    reranker = StubReranker()
    pipeline = HybridRAGPipeline(
        hybrid_retriever=StubRetriever(results),
        cross_encoder_reranker=reranker,
        top_k_rerank=2,
    )

    response = pipeline.run("refresh token")

    assert reranker.called
    assert [chunk["chunk_id"] for chunk in response["reranked_chunks"]] == [
        "chunk-2",
        "chunk-1",
    ]


def test_abstention_threshold_and_top_k_are_validated() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        HybridRAGPipeline(
            confidence_threshold=1.1,
            hybrid_retriever=StubRetriever([]),
            use_cross_encoder=False,
        )

    with pytest.raises(ValueError, match="top_k_retrieve"):
        HybridRAGPipeline(
            top_k_retrieve=0,
            hybrid_retriever=StubRetriever([]),
            use_cross_encoder=False,
        )


def test_evaluation_reports_metrics(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "pipeline_evaluation",
        root_dir / "scripts" / "evaluate_pipeline.py",
    )
    assert spec and spec.loader
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    class FakePipeline:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, query: str) -> dict[str, Any]:
            if query == "answerable":
                return {
                    "retrieved_chunks": [{"chunk_id": "chunk-a"}],
                    "is_abstaining": False,
                    "answer": "The JWT token is refreshed.",
                    "citations": ["chunk-a"],
                }
            return {
                "retrieved_chunks": [],
                "is_abstaining": True,
                "answer": "",
                "citations": [],
            }

    monkeypatch.setattr(evaluator, "HybridRAGPipeline", FakePipeline)
    report = evaluator.run_system_evaluation(
        [
            {
                "query": "answerable",
                "relevant_chunk_ids": ["chunk-a"],
                "is_answerable": True,
                "expected_keywords": ["JWT", "refreshed"],
            },
            {
                "query": "unanswerable",
                "relevant_chunk_ids": [],
                "is_answerable": False,
                "expected_keywords": [],
            },
        ],
        tmp_path / "report.json",
    )

    assert report["recall_at_3"] == 100.0
    assert report["mean_mrr"] == 1.0
    assert report["mean_answer_accuracy"] == 100.0
    assert report["mean_citation_accuracy"] == 100.0
    assert report["correct_abstention_rate"] == 100.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
