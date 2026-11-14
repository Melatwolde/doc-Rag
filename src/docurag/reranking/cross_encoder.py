from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder


@dataclass
class RerankedPassage:
    passage: Any
    text: str
    score: float
    source_id: str


class CrossEncoderReranker:
    """Score retrieved passages against a query and return the top results."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    @staticmethod
    def _get_text(passage: Any) -> str:
        if isinstance(passage, dict):
            value = (
                passage.get("text")
                or passage.get("content")
                or passage.get("page_content")
                or ""
            )
        else:
            value = (
                getattr(passage, "text", None)
                or getattr(passage, "content", None)
                or getattr(passage, "page_content", None)
                or ""
            )
        return str(value)

    @staticmethod
    def _get_source_id(passage: Any, fallback: int) -> str:
        if isinstance(passage, dict):
            value = passage.get("id", fallback)
        else:
            value = getattr(passage, "id", fallback)
        return str(value if value is not None else fallback)

    def rerank(self, query: str, passages: list[Any]) -> list[RerankedPassage]:
        """Return the highest-scoring passages, preserving original passage data."""
        if not query.strip() or not passages:
            return []

        valid_items = []
        pairs = []

        for passage in passages:
            passage_text = self._get_text(passage)
            if passage_text.strip():
                valid_items.append(passage)
                pairs.append((query, passage_text))

        if not pairs:
            return []

        scores = self.model.predict(pairs)

        ranked = [
            RerankedPassage(
                passage=passage,
                text=self._get_text(passage),
                score=float(score),
                source_id=self._get_source_id(passage, index),
            )
            for index, (passage, score) in enumerate(zip(valid_items, scores))
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: self.top_k]
