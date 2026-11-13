"""tests/unit/test_retrieval.py

Unit tests for BM25 lexical retriever, MiniLM dense retriever,
Reciprocal Rank Fusion (RRF), and Cross-Encoder reranker handoff.
"""

from __future__ import annotations

import pytest
import numpy as np

from docurag.retrieval.bm25_search import BM25Retriever, tokenize_technical_text
from docurag.retrieval.dense_search import DenseRetriever
from docurag.retrieval.rank_fusion import reciprocal_rank_fusion
from docurag.retrieval.hybrid_search import HybridRetriever
from docurag.retrieval.result import SearchResult
from docurag.reranking.cross_encoder import CrossEncoderReranker


@pytest.fixture
def sample_chunks():
    """Sample chunks representing diverse technical and conceptual content."""
    return [
        {
            "chunk_id": "chunk-auth-01",
            "doc_id": "doc-auth",
            "content": (
                "The authentication system handles JWT access tokens. "
                "Tokens expire after 3600 seconds. If an expired token is presented, "
                "the gateway returns error ERR_AUTH_TOKEN_EXPIRED. Use the CLI flag "
                "--token-expiry-seconds to configure."
            ),
            "source_url": "https://docs.example.com/auth",
            "section_title": "JWT Architecture",
            "char_count": 210,
            "token_count": 32,
            "metadata": {"author": "Security Team"},
        },
        {
            "chunk_id": "chunk-auth-02",
            "doc_id": "doc-auth",
            "content": (
                "To renew session credentials, call POST /api/v1/auth/refresh. "
                "The function def refresh_access_token(token: str) -> dict validates "
                "the signature. In config.yaml, set auth.jwt.secret_key."
            ),
            "source_url": "https://docs.example.com/auth",
            "section_title": "Refresh Tokens",
            "char_count": 195,
            "token_count": 28,
            "metadata": {"author": "Security Team"},
        },
        {
            "chunk_id": "chunk-db-01",
            "doc_id": "doc-db",
            "content": (
                "Database connection management uses a connection pool. Configure "
                "database.pool_size = 20. If exhausted under heavy concurrent load, "
                "the system raises ERR_CONN_TIMEOUT. Use --db-replica-url for read replicas."
            ),
            "source_url": "https://docs.example.com/db",
            "section_title": "Database Connection Pool",
            "char_count": 215,
            "token_count": 30,
            "metadata": {"author": "Database Team"},
        },
    ]


class TestTechnicalTokenizer:
    """Tests for the technical-aware tokenizer."""

    def test_preserves_snake_case_and_subparts(self):
        tokens = tokenize_technical_text("def refresh_access_token():")
        assert "refresh_access_token" in tokens
        assert "refresh" in tokens
        assert "access" in tokens
        assert "token" in tokens

    def test_preserves_cli_flags(self):
        tokens = tokenize_technical_text("run with --token-limit and --db-replica-url")
        assert "--token-limit" in tokens
        assert "--db-replica-url" in tokens
        assert "token" in tokens
        assert "limit" in tokens

    def test_preserves_error_codes(self):
        tokens = tokenize_technical_text("Received ERR_AUTH_TOKEN_EXPIRED on gateway")
        assert "err_auth_token_expired" in tokens
        assert "auth" in tokens
        assert "expired" in tokens

    def test_preserves_dotted_configs(self):
        tokens = tokenize_technical_text("Set database.pool_size in configs/config.yaml")
        assert "database.pool_size" in tokens
        assert "pool_size" in tokens or "pool" in tokens

    def test_empty_string(self):
        assert tokenize_technical_text("") == []
        assert tokenize_technical_text("   ") == []


class TestSearchResultDataStructure:
    """Tests for unified SearchResult data structure."""

    def test_attribute_and_dict_access(self):
        result = SearchResult(
            chunk_id="test-123",
            content="sample text",
            score=0.95,
            metadata={"doc_id": "doc-1", "section_title": "Overview"},
        )
        assert result.chunk_id == "test-123"
        assert result.content == "sample text"
        assert result.score == 0.95
        assert result["chunk_id"] == "test-123"
        assert result["doc_id"] == "doc-1"
        assert result.get("section_title") == "Overview"
        assert result.get("missing", "default") == "default"

    def test_to_dict_conversion(self):
        result = SearchResult(
            chunk_id="test-456",
            content="sample text 2",
            score=1.23,
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["chunk_id"] == "test-456"
        assert d["score"] == 1.23
        assert d["metadata"]["key"] == "value"


class TestBM25Retriever:
    """Tests for BM25 lexical retriever."""

    def test_exact_technical_retrieval(self, sample_chunks):
        retriever = BM25Retriever(sample_chunks)
        results = retriever.retrieve("ERR_AUTH_TOKEN_EXPIRED", top_k=2)
        assert len(results) > 0
        assert results[0].chunk_id == "chunk-auth-01"
        assert results[0].score > 0
        assert results[0].metadata["doc_id"] == "doc-auth"
        assert results[0].metadata["section_title"] == "JWT Architecture"

    def test_dotted_config_retrieval(self, sample_chunks):
        retriever = BM25Retriever(sample_chunks)
        results = retriever.retrieve("database.pool_size", top_k=2)
        assert len(results) > 0
        assert results[0].chunk_id == "chunk-db-01"

    def test_cli_flag_retrieval(self, sample_chunks):
        retriever = BM25Retriever(sample_chunks)
        results = retriever.retrieve("--token-expiry-seconds", top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-auth-01"


class TestRankFusion:
    """Tests for Reciprocal Rank Fusion algorithm."""

    def test_rrf_scoring_formula(self):
        res1 = SearchResult("chunk-A", "text A", 10.0, {"retriever": "bm25"})
        res2 = SearchResult("chunk-B", "text B", 5.0, {"retriever": "bm25"})
        bm25_list = [res1, res2]

        res3 = SearchResult("chunk-B", "text B", 0.9, {"retriever": "dense"})
        res4 = SearchResult("chunk-C", "text C", 0.8, {"retriever": "dense"})
        dense_list = [res3, res4]

        # RRF with k=60
        # chunk-A: bm25 rank 1 -> 1 / (60 + 1) = 1/61 ~= 0.016393
        # chunk-B: bm25 rank 2 -> 1 / (60 + 2) = 1/62 ~= 0.016129
        #          dense rank 1 -> 1 / (60 + 1) = 1/61 ~= 0.016393
        #          sum(B) = 1/62 + 1/61 ~= 0.032522
        # chunk-C: dense rank 2 -> 1 / (60 + 2) = 1/62 ~= 0.016129
        fused = reciprocal_rank_fusion([bm25_list, dense_list], k=60)

        assert len(fused) == 3
        # chunk-B appears in both lists, so it must be top ranked!
        assert fused[0].chunk_id == "chunk-B"
        expected_b_score = (1.0 / 62) + (1.0 / 61)
        assert abs(fused[0].score - expected_b_score) < 1e-5

        # chunk-A has rank 1 in one list vs chunk-C rank 2 in one list
        assert fused[1].chunk_id == "chunk-A"
        assert fused[2].chunk_id == "chunk-C"

    def test_rrf_deduplication(self):
        res1 = SearchResult("chunk-X", "text X", 5.0, {"tag": "first"})
        res2 = SearchResult("chunk-X", "text X", 0.8, {"tag": "second", "category": "auth"})
        fused = reciprocal_rank_fusion([[res1], [res2]], k=60)

        assert len(fused) == 1
        assert fused[0].chunk_id == "chunk-X"
        assert fused[0].metadata["category"] == "auth"


class TestHybridRetrieverAndHandoff:
    """Tests for end-to-end HybridRetriever and Cross-Encoder handoff."""

    def test_hybrid_retrieve_returns_valid_structure(self, sample_chunks):
        hybrid = HybridRetriever.from_chunks(sample_chunks)
        results = hybrid.retrieve("refresh token session renewal", top_k=2)

        assert len(results) <= 2
        for r in results:
            assert isinstance(r, SearchResult)
            assert r.chunk_id in {"chunk-auth-01", "chunk-auth-02", "chunk-db-01"}
            assert r.score > 0
            assert "retrieval_sources" in r.metadata

    def test_cross_encoder_reranker_handoff(self, sample_chunks):
        hybrid = HybridRetriever.from_chunks(sample_chunks)
        candidates = hybrid.retrieve("renewing user sessions", top_k=3)

        # Pass retrieved candidates directly to CrossEncoderReranker
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("renewing user sessions", candidates, top_k=2)

        assert len(reranked) <= 2
        for r in reranked:
            assert isinstance(r, SearchResult)
            assert "cross_encoder_score" in r.metadata
            assert "pre_rerank_rank" in r.metadata
