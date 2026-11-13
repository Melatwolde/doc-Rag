"""tests/integration/test_pipeline.py

Integration tests for the full RAG pipeline.

The HybridRetriever uses real chunks from data/processed/chunks.json.
The CrossEncoder and LLM are mocked to avoid downloads and API calls.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from docurag.generation.llm import GenerationResult, RAGGenerator
from docurag.pipeline import RAGPipeline
from docurag.reranking.cross_encoder import CrossEncoderReranker
from docurag.retrieval.hybrid_search import HybridRetriever
from docurag.retrieval.result import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNKS_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed" / "chunks.json"
BENCHMARK_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "benchmark" / "qa_benchmark.json"


def _chunks_available() -> bool:
    return CHUNKS_PATH.exists() and CHUNKS_PATH.stat().st_size > 100


def _mock_reranker_scores(candidates, scores=None):
    """Return reranker that assigns mock scores."""
    mock_scores = scores or [float(5 - i) for i in range(len(candidates))]
    return mock_scores


# ---------------------------------------------------------------------------
# Pipeline construction tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _chunks_available(), reason="chunks.json not found")
class TestRAGPipelineConstruction:
    def test_from_chunks_json(self):
        """RAGPipeline.from_chunks_json must build without error."""
        pipeline = RAGPipeline.from_chunks_json(
            CHUNKS_PATH,
            llm_backend="local",
        )
        assert pipeline.retriever is not None
        assert pipeline.reranker is not None
        assert pipeline.generator is not None

    def test_retrieve_top_k_default(self):
        pipeline = RAGPipeline.from_chunks_json(CHUNKS_PATH, llm_backend="local")
        assert pipeline.retrieve_top_k == 10
        assert pipeline.rerank_top_k == 3


# ---------------------------------------------------------------------------
# End-to-end pipeline tests (retriever real, reranker+LLM mocked)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _chunks_available(), reason="chunks.json not found")
class TestRAGPipelineAnswer:
    @pytest.fixture()
    def pipeline(self):
        retriever = HybridRetriever.from_json(CHUNKS_PATH)
        reranker = CrossEncoderReranker()
        generator = RAGGenerator(backend="local")
        return RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            retrieve_top_k=10,
            rerank_top_k=3,
        )

    def test_answer_returns_generation_result(self, pipeline):
        """pipeline.answer() must return a GenerationResult."""
        with (
            patch.object(pipeline.reranker, "_score_pairs", return_value=[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0, -0.1]),
            patch.object(pipeline.generator, "_call_llm", return_value="JWT expires in 3600 seconds [1]."),
        ):
            result = pipeline.answer("What is the JWT expiry?")
        assert isinstance(result, GenerationResult)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_answer_on_known_question(self, pipeline):
        """Answer for JWT question should not be an abstention."""
        with (
            patch.object(pipeline.reranker, "_score_pairs", return_value=[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0, -0.1]),
            patch.object(pipeline.generator, "_call_llm", return_value="JWT tokens expire after 3600 seconds [1]."),
        ):
            result = pipeline.answer("What is the JWT expiry time?")
        assert result.is_abstention is False

    def test_answer_abstains_on_unanswerable(self, pipeline):
        """Pipeline must abstain when no passages pass the CE threshold."""
        # Set impossibly high threshold → select_context returns empty
        pipeline.reranker.score_threshold = 999.0
        with patch.object(pipeline.reranker, "_score_pairs", return_value=[1.0] * 10):
            result = pipeline.answer("What is the billing plan for the cloud service?")
        assert result.is_abstention is True

    def test_answer_grounded_passages_match_top_k(self, pipeline):
        """grounded_passages should be at most rerank_top_k."""
        with (
            patch.object(pipeline.reranker, "_score_pairs", return_value=[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0, -0.1]),
            patch.object(pipeline.generator, "_call_llm", return_value="Some answer [1]."),
        ):
            result = pipeline.answer("How does connection pooling work?")
        assert len(result.grounded_passages) <= pipeline.rerank_top_k

    def test_answer_batch(self, pipeline):
        """answer_batch must process all questions."""
        questions = [
            "What is the JWT expiry?",
            "What is the pool size?",
            "How does BM25 work in DocuRAG?",
        ]
        scores = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0, -0.1]
        with (
            patch.object(pipeline.reranker, "_score_pairs", return_value=scores),
            patch.object(pipeline.generator, "_call_llm", return_value="Mocked answer [1]."),
        ):
            results = pipeline.answer_batch(questions)
        assert len(results) == 3
        assert all(isinstance(r, GenerationResult) for r in results)

    def test_inspect_returns_all_stages(self, pipeline):
        """inspect() must return retrieved, reranked, and result keys."""
        with (
            patch.object(pipeline.reranker, "_score_pairs", return_value=[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0, -0.1]),
            patch.object(pipeline.generator, "_call_llm", return_value="Token expires [1]."),
        ):
            breakdown = pipeline.inspect("What is the token expiry?")
        assert "question" in breakdown
        assert "retrieved" in breakdown
        assert "reranked" in breakdown
        assert "result" in breakdown

    def test_result_to_dict_serializable(self, pipeline):
        """GenerationResult.to_dict() must produce JSON-serializable dict."""
        with (
            patch.object(pipeline.reranker, "_score_pairs", return_value=[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0, -0.1]),
            patch.object(pipeline.generator, "_call_llm", return_value="Token expires in 60 mins [1]."),
        ):
            result = pipeline.answer("JWT token expiry?")
        d = result.to_dict()
        # Should serialize without raising
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# Citation support test (grounding verification)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _chunks_available(), reason="chunks.json not found")
class TestCitationSupport:
    def test_grounded_answer_passes_check(self):
        """An answer drawn directly from a passage should pass grounding check."""
        generator = RAGGenerator(backend="local")
        passage = SearchResult(
            chunk_id="test-chunk",
            content="JWT access tokens expire after 3600 seconds by default.",
            score=3.0,
            metadata={},
        )
        answer = "JWT access tokens expire after 3600 seconds."
        grounding = generator.check_grounding(answer, [passage])
        assert grounding["grounded"] is True

    def test_hallucinated_answer_fails_check(self):
        """An answer not based on the passage should not pass grounding check."""
        generator = RAGGenerator(backend="local")
        passage = SearchResult(
            chunk_id="test-chunk",
            content="JWT access tokens expire after 3600 seconds by default.",
            score=3.0,
            metadata={},
        )
        hallucinated_answer = "The database uses PostgreSQL 14 with TLS 1.3 encryption."
        grounding = generator.check_grounding(hallucinated_answer, [passage])
        # Very different tokens — overlap should be low
        assert grounding["max_overlap"] < 0.3


# ---------------------------------------------------------------------------
# Benchmark dataset format test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BENCHMARK_PATH.exists(), reason="benchmark not found")
class TestBenchmarkDataset:
    @pytest.fixture()
    def benchmark(self):
        with open(BENCHMARK_PATH) as f:
            return json.load(f)

    def test_benchmark_has_60_questions(self, benchmark):
        assert len(benchmark) == 60

    def test_benchmark_has_50_answerable(self, benchmark):
        answerable = [q for q in benchmark if q["answerable"]]
        assert len(answerable) == 50

    def test_benchmark_has_10_unanswerable(self, benchmark):
        unanswerable = [q for q in benchmark if not q["answerable"]]
        assert len(unanswerable) == 10

    def test_benchmark_schema(self, benchmark):
        """Every question must have required fields."""
        for q in benchmark:
            assert "id" in q
            assert "question" in q
            assert "answer" in q
            assert "answerable" in q
            assert "relevant_chunk_ids" in q
            assert "category" in q

    def test_answerable_have_chunk_ids(self, benchmark):
        """All answerable questions must reference at least one chunk ID."""
        for q in benchmark:
            if q["answerable"]:
                assert len(q["relevant_chunk_ids"]) >= 1

    def test_unanswerable_have_empty_chunk_ids(self, benchmark):
        """Unanswerable questions should have no relevant chunk IDs."""
        for q in benchmark:
            if not q["answerable"]:
                assert q["relevant_chunk_ids"] == []

    def test_question_ids_unique(self, benchmark):
        ids = [q["id"] for q in benchmark]
        assert len(ids) == len(set(ids))
