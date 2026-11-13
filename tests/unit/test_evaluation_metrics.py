from docurag.evaluation.metrics import abstention_rates, answer_accuracy, citation_accuracy, latency_summary, recall_at_k, reciprocal_rank


def test_ranking_metrics():
    items = ["wrong", "right", "other"]
    assert recall_at_k(items, lambda item: item == "right", 1) == 0.0
    assert recall_at_k(items, lambda item: item == "right", 2) == 1.0
    assert reciprocal_rank(items, lambda item: item == "right") == 0.5


def test_reliability_metrics():
    records = [
        {"answerable": True, "is_abstaining": False, "has_relevant_evidence": True, "has_relevant_citation": True},
        {"answerable": True, "is_abstaining": True, "has_relevant_evidence": False, "has_relevant_citation": False},
        {"answerable": False, "is_abstaining": True, "has_relevant_evidence": False, "has_relevant_citation": False},
    ]
    assert answer_accuracy(records) == 0.5
    assert citation_accuracy(records) == 1.0
    assert abstention_rates(records) == {"correct_abstention_rate": 1.0, "false_abstention_rate": 0.5}
    assert latency_summary([1.0, 4.0, 2.0])["mean_latency_ms"] == 2.33
