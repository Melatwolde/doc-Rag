"""Transparent metrics for retrieval, grounding, abstention, and latency."""

from __future__ import annotations

from statistics import mean
from typing import Any, Callable, Iterable, Sequence


def recall_at_k(results: Sequence[Any], is_relevant: Callable[[Any], bool], k: int) -> float:
    """Return 1 when any relevant item appears in the first ``k`` results."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return float(any(is_relevant(item) for item in results[:k]))


def reciprocal_rank(results: Sequence[Any], is_relevant: Callable[[Any], bool]) -> float:
    """Return the reciprocal rank of the first relevant result, else zero."""
    for rank, item in enumerate(results, start=1):
        if is_relevant(item):
            return 1.0 / rank
    return 0.0


def answer_accuracy(records: Iterable[dict[str, Any]]) -> float:
    """Grounded answer accuracy over answerable records.

    A prediction is correct when it does not abstain and its cited/retrieved
    evidence includes the labelled relevant passage. This does not judge prose
    quality, which would require human-labelled reference answers.
    """
    answerable = [record for record in records if record["answerable"]]
    if not answerable:
        return 0.0
    return sum(not item["is_abstaining"] and item["has_relevant_evidence"] for item in answerable) / len(answerable)


def citation_accuracy(records: Iterable[dict[str, Any]]) -> float:
    """Fraction of non-abstained answers that cite at least one relevant source."""
    answered = [record for record in records if not record["is_abstaining"]]
    if not answered:
        return 0.0
    return sum(item["has_relevant_citation"] for item in answered) / len(answered)


def abstention_rates(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Measure correct and false abstention rates against explicit labels."""
    rows = list(records)
    answerable = [item for item in rows if item["answerable"]]
    unanswerable = [item for item in rows if not item["answerable"]]
    return {
        "correct_abstention_rate": (sum(item["is_abstaining"] for item in unanswerable) / len(unanswerable) if unanswerable else 0.0),
        "false_abstention_rate": (sum(item["is_abstaining"] for item in answerable) / len(answerable) if answerable else 0.0),
    }


def latency_summary(latencies_ms: Iterable[float]) -> dict[str, float]:
    """Return mean and p95 latency in milliseconds."""
    values = sorted(float(value) for value in latencies_ms)
    if not values:
        return {"mean_latency_ms": 0.0, "p95_latency_ms": 0.0}
    p95_index = max(0, min(len(values) - 1, int((len(values) - 1) * 0.95)))
    return {"mean_latency_ms": round(mean(values), 2), "p95_latency_ms": round(values[p95_index], 2)}
