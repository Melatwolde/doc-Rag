"""tests/unit/test_reranking.py

Unit tests for CrossEncoderReranker and ContextBuilder.

The CrossEncoder model is fully mocked — no network access or GPU required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from docurag.reranking.context_builder import CitationMeta, ContextBuilder
from docurag.reranking.cross_encoder import CrossEncoderReranker
from docurag.retrieval.result import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(chunk_id: str, content: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        metadata={"section_title": f"Section {chunk_id[:4]}"},
    )


@pytest.fixture()
def candidates():
    return [
        _make_result("aaa", "JWT tokens expire after 3600 seconds.", 0.9),
        _make_result("bbb", "Database pool size defaults to 20 connections.", 0.7),
        _make_result("ccc", "Refresh tokens are stored in HTTP-only cookies.", 0.6),
        _make_result("ddd", "BM25 handles exact keyword matching.", 0.4),
        _make_result("eee", "Chunk overlap ratio is 10%.", 0.3),
    ]


@pytest.fixture()
def reranker():
    r = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return r


# ---------------------------------------------------------------------------
# CrossEncoderReranker tests
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    def test_rerank_returns_top_k(self, candidates, reranker):
        """rerank() must return exactly top_k results."""
        mock_scores = [2.0, -1.0, 3.5, 0.5, 1.2]
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            results = reranker.rerank("What is JWT expiry?", candidates, top_k=3)
        assert len(results) == 3

    def test_rerank_ordering(self, candidates, reranker):
        """Results must be sorted descending by cross-encoder score."""
        mock_scores = [2.0, -1.0, 3.5, 0.5, 1.2]
        # Expected order by score: ccc(3.5) > aaa(2.0) > eee(1.2)
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            results = reranker.rerank("query", candidates, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_metadata_added(self, candidates, reranker):
        """Each reranked result must have cross_encoder_score and reranked_rank."""
        mock_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            results = reranker.rerank("query", candidates, top_k=3)
        for r in results:
            assert "cross_encoder_score" in r.metadata
            assert "reranked_rank" in r.metadata
            assert "pre_rerank_rank" in r.metadata
            assert "pre_rerank_score" in r.metadata

    def test_rerank_rank_numbering(self, candidates, reranker):
        """reranked_rank must be 1-based and sequential."""
        mock_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            results = reranker.rerank("query", candidates, top_k=3)
        ranks = [r.metadata["reranked_rank"] for r in results]
        assert ranks == [1, 2, 3]

    def test_rerank_empty_candidates(self, reranker):
        """Empty candidate list must return empty list without error."""
        results = reranker.rerank("query", [], top_k=3)
        assert results == []

    def test_rerank_fewer_than_top_k(self, reranker):
        """If fewer candidates than top_k, return all of them."""
        two_candidates = [
            _make_result("x", "text x"),
            _make_result("y", "text y"),
        ]
        mock_scores = [1.0, 2.0]
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            results = reranker.rerank("query", two_candidates, top_k=5)
        assert len(results) == 2

    def test_select_context_filters_below_threshold(self, candidates, reranker):
        """select_context() must exclude passages below score_threshold."""
        reranker.score_threshold = 1.0  # Only scores >= 1.0 pass
        mock_scores = [2.0, 0.5, 3.0, -0.5, 0.8]  # 2 pass: idx0, idx2
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            results = reranker.select_context("query", candidates, top_k=5)
        for r in results:
            assert r.score >= reranker.score_threshold

    def test_select_context_empty_when_all_below_threshold(self, candidates, reranker):
        """select_context() must return empty list if nothing passes threshold."""
        reranker.score_threshold = 100.0  # Impossible threshold
        mock_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            results = reranker.select_context("query", candidates, top_k=3)
        assert results == []

    def test_rerank_with_scores_returns_all(self, candidates, reranker):
        """rerank_with_scores must return ALL candidates with their scores."""
        mock_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            pairs = reranker.rerank_with_scores("query", candidates)
        assert len(pairs) == len(candidates)
        # Verify each pair is (SearchResult, float)
        for result, score in pairs:
            assert isinstance(result, SearchResult)
            assert isinstance(score, float)

    def test_rerank_with_scores_sorted_descending(self, candidates, reranker):
        """rerank_with_scores result must be sorted descending by score."""
        mock_scores = [3.0, 1.0, 5.0, 2.0, 4.0]
        with patch.object(reranker, "_score_pairs", return_value=mock_scores):
            pairs = reranker.rerank_with_scores("query", candidates)
        scores_only = [score for _, score in pairs]
        assert scores_only == sorted(scores_only, reverse=True)

    def test_model_lazy_loaded(self, reranker):
        """CrossEncoder model should not load until first use."""
        # _model is None at construction
        assert reranker._model is None

    def test_score_passage_support(self, reranker):
        """score_passage_support must return relevance and support keys."""
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [2.0]
        result = reranker.score_passage_support(
            query="JWT expiry?",
            passage="JWT tokens expire after 3600 seconds.",
            answer="JWT expires in 3600 seconds.",
        )
        assert "relevance" in result
        assert "support" in result
        assert isinstance(result["relevance"], float)
        assert isinstance(result["support"], float)


# ---------------------------------------------------------------------------
# ContextBuilder tests
# ---------------------------------------------------------------------------


class TestContextBuilder:
    def test_build_returns_context_and_citations(self, candidates):
        builder = ContextBuilder(max_tokens=1800)
        context_str, citations = builder.build("test query", candidates)
        assert isinstance(context_str, str)
        assert isinstance(citations, list)

    def test_build_numbering(self, candidates):
        builder = ContextBuilder()
        _, citations = builder.build("query", candidates[:3])
        assert [c.index for c in citations] == [1, 2, 3]

    def test_build_includes_section_title(self):
        passage = SearchResult(
            chunk_id="abc",
            content="Some content.",
            score=1.0,
            metadata={"section_title": "JWT Lifecycle"},
        )
        builder = ContextBuilder()
        context_str, citations = builder.build("query", [passage])
        assert "JWT Lifecycle" in context_str
        assert citations[0].section_title == "JWT Lifecycle"

    def test_build_empty_passages(self):
        builder = ContextBuilder()
        context_str, citations = builder.build("query", [])
        assert context_str == ""
        assert citations == []

    def test_build_respects_token_budget(self):
        """Very small token budget should truncate passage content."""
        builder = ContextBuilder(max_tokens=10)  # ~40 chars
        long_passage = SearchResult(
            chunk_id="x",
            content="A" * 1000,
            score=1.0,
            metadata={},
        )
        context_str, _ = builder.build("q", [long_passage])
        # Context should not be huge
        assert len(context_str) < 500

    def test_citation_snippet_length(self, candidates):
        builder = ContextBuilder()
        _, citations = builder.build("query", candidates)
        for c in citations:
            assert len(c.snippet) <= 200

    def test_extract_citation_indices(self):
        builder = ContextBuilder()
        answer = "According to [1] and [3], tokens expire in 3600 seconds [2]."
        indices = builder.extract_citation_indices(answer)
        assert indices == [1, 2, 3]

    def test_extract_citation_indices_empty(self):
        builder = ContextBuilder()
        assert builder.extract_citation_indices("No citations here.") == []

    def test_filter_citations(self, candidates):
        builder = ContextBuilder()
        _, citations = builder.build("query", candidates)
        answer = "The answer uses [1] and [3]."
        filtered = builder.filter_citations(answer, citations)
        assert [c.index for c in filtered] == [1, 3]

    def test_citation_meta_fields(self, candidates):
        builder = ContextBuilder()
        _, citations = builder.build("query", candidates[:1])
        c = citations[0]
        assert isinstance(c.index, int)
        assert isinstance(c.chunk_id, str)
        assert isinstance(c.score, float)
        assert isinstance(c.snippet, str)
