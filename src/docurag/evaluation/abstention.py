"""src/docurag/evaluation/abstention.py

Confidence calibration and abstention guardrail mechanism for DocuRAG.
Evaluates reranked candidate confidence scores to prevent hallucinations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


class AbstentionEngine:
    def __init__(self, confidence_threshold: float = 0.45, min_relevance_chunks: int = 1) -> None:
        """Initialize the abstention engine with calibrated score thresholds."""
        self.confidence_threshold = float(confidence_threshold)
        self.min_relevance_chunks = int(min_relevance_chunks)
        if not math.isfinite(self.confidence_threshold) or not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be a finite value between 0.0 and 1.0")
        if self.min_relevance_chunks < 1:
            raise ValueError("min_relevance_chunks must be at least 1")

    @staticmethod
    def _value(result: Any, key: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(key, default)
        getter = getattr(result, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(result, key, default)

    def confidence_score(self, result: Any) -> float:
        raw_score = self._value(result, "confidence_score", None)
        metadata = self._value(result, "metadata", {}) or {}
        rrf_k = metadata.get("rrf_k") if isinstance(metadata, dict) else None
        sources = metadata.get("retrieval_sources") if isinstance(metadata, dict) else None
        if raw_score is not None:
            value = self._validated_score(raw_score)
            if not 0.0 <= value <= 1.0:
                raise ValueError("confidence_score must be between 0.0 and 1.0")
            return value

        if rrf_k is None or not isinstance(sources, dict) or not sources:
            return self._validated_score(self._value(result, "score", 0.0))

        rrf_k = int(rrf_k)
        if rrf_k < 1:
            raise ValueError("retrieval metadata rrf_k must be at least 1")

        dense_score: float | None = None
        for name, source in sources.items():
            if not isinstance(source, dict):
                continue
            source_score = float(source.get("raw_score", 0.0))
            if "dense" in str(name).lower():
                dense_score = source_score if dense_score is None else max(dense_score, source_score)

        if "query_token_count" in metadata:
            lexical_score = float(metadata.get("lexical_confidence", 0.0))
            semantic_score = max(0.0, dense_score or 0.0)
            return min(max(semantic_score, lexical_score), 1.0)

        rrf_score = self._validated_score(self._value(result, "score", 0.0))
        return min(max(rrf_score / (2.0 / (rrf_k + 1)), 0.0), 1.0)

    @staticmethod
    def _validated_score(score: Any) -> float:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("retrieved passage score must be finite")
        return value

    def evaluate_answerability(self, reranked_results: List[Dict[str, Any] | Any]) -> Tuple[bool, str]:
        """Evaluate whether retrieved/reranked passages contain sufficient evidence.

        Returns:
            Tuple[bool, str]: (is_abstaining, abstention_reason)
        """
        if not reranked_results or len(reranked_results) < self.min_relevance_chunks:
            return True, "ERR_INSUFFICIENT_EVIDENCE: No relevant document chunks retrieved."

        top_result = reranked_results[0]
        top_score = self.confidence_score(top_result)

        if top_score < self.confidence_threshold:
            return True, (
                f"ERR_INSUFFICIENT_EVIDENCE: Top passage confidence score ({top_score:.4f}) "
                f"falls below minimum threshold ({self.confidence_threshold})."
            )

        return False, "Sufficient evidence verified for generation."