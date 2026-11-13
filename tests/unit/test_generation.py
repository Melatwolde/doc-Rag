"""tests/unit/test_generation.py

Unit tests for RAGGenerator and GenerationResult.

The LLM backend is mocked — no model download or API key required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from docurag.generation.llm import (
    GenerationResult,
    RAGGenerator,
    _is_abstention,
    _ABSTENTION_PHRASES,
)
from docurag.retrieval.result import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(chunk_id: str, content: str, score: float = 2.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        metadata={
            "section_title": "Test Section",
            "cross_encoder_score": score,
        },
    )


@pytest.fixture()
def passages():
    return [
        _make_result(
            "aaa",
            "JWT access tokens expire after 3600 seconds by default.",
            score=3.5,
        ),
        _make_result(
            "bbb",
            "Clients must call POST /api/v1/auth/refresh to renew credentials.",
            score=2.1,
        ),
        _make_result(
            "ccc",
            "The JWT secret key is defined in auth.jwt.secret_key in config.yaml.",
            score=1.8,
        ),
    ]


@pytest.fixture()
def generator():
    return RAGGenerator(backend="local")


# ---------------------------------------------------------------------------
# _is_abstention helper
# ---------------------------------------------------------------------------


class TestIsAbstention:
    @pytest.mark.parametrize("phrase", _ABSTENTION_PHRASES)
    def test_known_phrases_detected(self, phrase):
        assert _is_abstention(f"This is a response: {phrase}.")

    def test_non_abstention_text(self):
        assert not _is_abstention("JWT tokens expire after 3600 seconds.")

    def test_case_insensitive(self):
        assert _is_abstention("I DON'T KNOW what you mean.")

    def test_empty_string(self):
        assert not _is_abstention("")


# ---------------------------------------------------------------------------
# RAGGenerator tests
# ---------------------------------------------------------------------------


class TestRAGGenerator:
    def test_generate_returns_generation_result(self, generator, passages):
        """generate() must return a GenerationResult object."""
        with patch.object(generator, "_call_llm", return_value="JWT expires in 3600 seconds [1]."):
            result = generator.generate("What is JWT expiry?", passages)
        assert isinstance(result, GenerationResult)

    def test_generate_answer_in_result(self, generator, passages):
        """The answer field must match LLM output."""
        llm_output = "JWT access tokens expire after 3600 seconds [1]."
        with patch.object(generator, "_call_llm", return_value=llm_output):
            result = generator.generate("What is JWT expiry?", passages)
        assert result.answer == llm_output

    def test_generate_not_abstention(self, generator, passages):
        """Normal answer must have is_abstention=False."""
        with patch.object(generator, "_call_llm", return_value="JWT expires after 3600s [1]."):
            result = generator.generate("What is JWT expiry?", passages)
        assert result.is_abstention is False

    def test_generate_abstention_on_empty_passages(self, generator):
        """Empty passages list must trigger abstention without calling LLM."""
        with patch.object(generator, "_call_llm") as mock_llm:
            result = generator.generate("Any question?", [])
        mock_llm.assert_not_called()
        assert result.is_abstention is True
        assert "don't know" in result.answer.lower()

    def test_generate_abstention_when_llm_says_idontknow(self, generator, passages):
        """LLM response containing abstention phrase must set is_abstention=True."""
        with patch.object(
            generator, "_call_llm",
            return_value="I don't know based on the available documentation."
        ):
            result = generator.generate("Unknowable question?", passages)
        assert result.is_abstention is True

    def test_generate_includes_grounded_passages(self, generator, passages):
        """grounded_passages must be the passages passed in."""
        with patch.object(generator, "_call_llm", return_value="Some answer [1]."):
            result = generator.generate("query", passages)
        assert len(result.grounded_passages) == len(passages)

    def test_generate_raw_prompt_contains_query(self, generator, passages):
        """The raw prompt must include the user query."""
        query = "What is the JWT expiry time?"
        with patch.object(generator, "_call_llm", return_value="3600 seconds [1]."):
            result = generator.generate(query, passages)
        assert query in result.raw_prompt

    def test_generate_raw_prompt_contains_evidence(self, generator, passages):
        """The raw prompt must include passage content."""
        with patch.object(generator, "_call_llm", return_value="3600 seconds [1]."):
            result = generator.generate("query", passages)
        # At least one passage should appear in the prompt
        assert any(p.content[:30] in result.raw_prompt for p in passages)

    def test_generate_model_used_field(self, generator, passages):
        """model_used field must reflect the configured model."""
        with patch.object(generator, "_call_llm", return_value="answer [1]."):
            result = generator.generate("query", passages)
        assert result.model_used == generator.local_model_name

    def test_generate_citations_filtered_by_answer(self, generator, passages):
        """Citations must only include indices referenced in the answer."""
        # Answer only references [1], not [2] or [3]
        with patch.object(generator, "_call_llm", return_value="JWT expires in 3600s [1]."):
            result = generator.generate("What is JWT expiry?", passages)
        cited_indices = [c.index for c in result.citations]
        assert 1 in cited_indices
        # [2] and [3] should not be cited
        assert 2 not in cited_indices
        assert 3 not in cited_indices

    def test_generate_to_dict(self, generator, passages):
        """GenerationResult.to_dict() must include all required keys."""
        with patch.object(generator, "_call_llm", return_value="answer [1]."):
            result = generator.generate("query", passages)
        d = result.to_dict()
        for key in ("answer", "is_abstention", "model_used", "citations"):
            assert key in d

    def test_check_grounding_grounded(self, generator, passages):
        """check_grounding should detect token overlap between answer and passage."""
        answer = "JWT access tokens expire after 3600 seconds."
        grounding = generator.check_grounding(answer, passages[:1])
        assert grounding["grounded"] is True
        assert grounding["max_overlap"] > 0

    def test_check_grounding_not_grounded(self, generator):
        """Completely unrelated answer should return grounded=False."""
        passage = _make_result("x", "Completely unrelated content about databases.")
        answer = "The moon is made of cheese."
        grounding = generator.check_grounding(answer, [passage])
        assert grounding["max_overlap"] < 0.5

    def test_check_grounding_empty_answer(self, generator, passages):
        """Empty answer string should return grounded=False."""
        grounding = generator.check_grounding("", passages)
        assert grounding["grounded"] is False

    def test_backend_auto_local_without_key(self, monkeypatch):
        """Without OPENAI_API_KEY, 'auto' backend should resolve to 'local'."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        gen = RAGGenerator(backend="auto")
        assert gen.backend == "local"

    def test_backend_auto_openai_with_key(self, monkeypatch):
        """With OPENAI_API_KEY set, 'auto' backend should resolve to 'openai'."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        gen = RAGGenerator(backend="auto")
        assert gen.backend == "openai"

    def test_generation_result_dataclass(self):
        """GenerationResult should be constructible with defaults."""
        result = GenerationResult(answer="Test answer")
        assert result.answer == "Test answer"
        assert result.is_abstention is False
        assert result.citations == []
        assert result.grounded_passages == []
