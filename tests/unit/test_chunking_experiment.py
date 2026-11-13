from __future__ import annotations

import json

from data_pipeline.experiment import benchmark_records, save_experiment
from data_pipeline.models import Document


def test_chunking_experiment_is_saved_as_reproducible_json(tmp_path):
    docs = [Document(doc_id="one", content="# Title\n\nOne two three", metadata={})]
    records = benchmark_records(docs, token_limits=[2], overlap_ratios=[0.0, 0.5])
    output = tmp_path / "experiment.json"
    save_experiment(records, output)

    assert len(records) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == records
