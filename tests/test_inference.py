"""Opt-in integration test against the real model, not a stub."""

import os
import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 after installing the ML runtime and fetching Git LFS.",
)
def test_real_model_scores_and_input_quality():
    from inference import SentimentEngine

    engine = SentimentEngine()
    rows = engine.analyze(
        [
            "The camera is excellent and I love this phone.",
            "The battery is terrible and the phone is broken.",
            "qzxvplm",
            "good " * 400,
        ]
    )
    assert len(rows) == 4
    for row in rows:
        assert len(row["scores"]) == 5
        assert abs(sum(row["scores"].values()) - 1) < 0.001
        assert 0 <= row["confidence"] <= 1
        assert row["label"] in row["scores"]
        assert row["model_version"] == "dc474c132178"
    assert rows[2]["needs_review"]
    assert (
        rows[3]["needs_review"]
        and "Input truncated to 350 tokens" in rows[3]["warnings"]
    )
