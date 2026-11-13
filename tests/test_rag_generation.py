from docurag.reranking.cross_encoder import RerankedPassage
from docurag.generation.llm import RAGGenerator, validate_citations


def test_generation_and_citation_reference_validation():
    passages = [
        RerankedPassage(
            passage=None,
            text="Python is a programming language.",
            score=0.95,
            source_id="chunk_1",
        )
    ]

    def fake_llm(prompt):
        assert "Python is a programming language." in prompt
        return "Python is a programming language [S1]."

    result = RAGGenerator(fake_llm).generate("What is Python?", passages)

    assert result.citations == ["1"]
    assert result.is_grounded is True
    assert validate_citations(result, passages) is True


def test_empty_context_returns_safe_answer():
    result = RAGGenerator(lambda prompt: "Should not be called").generate(
        "Question?", []
    )

    assert result.citations == []
    assert result.is_grounded is False
