from __future__ import annotations

import numpy as np

from bigan.v8.polymarket.moe_terminal_diagnostic import (
    _plugin_sample_count,
    _probability_metrics,
    verify_frozen_terminal_diagnostic,
)


def test_frozen_terminal_report_reproduces_from_hash_bound_scored_rows() -> None:
    result = verify_frozen_terminal_diagnostic()

    assert result["verification_passed"] is True
    assert result["scored_row_count"] == 3200
    assert result["terminal_failed"] is True
    assert result["scored_rows_recomputed_from_raw"] is False


def test_probability_metrics_use_frozen_clipping_and_tie_aware_auc() -> None:
    rows = [
        {"probability": 1.0, "target": 1.0},
        {"probability": 0.5, "target": 1.0},
        {"probability": 0.5, "target": 0.0},
        {"probability": 0.0, "target": 0.0},
    ]

    metrics = _probability_metrics(
        rows,
        probability_field="probability",
        epsilon=1e-15,
    )

    assert metrics["auc"] == 0.875
    assert np.isfinite(metrics["log_loss"])
    assert metrics["probability_clipping_rule"].endswith("_for_log_loss_only")


def test_plugin_counts_distinguish_half_crossing_from_eighty_percent() -> None:
    half = _plugin_sample_count(
        mean=0.00442906250000002,
        standard_deviation=0.43359054947668135,
        z_confidence=1.9599639845400534,
        z_target=0.0,
    )
    eighty = _plugin_sample_count(
        mean=0.00442906250000002,
        standard_deviation=0.43359054947668135,
        z_confidence=1.9599639845400534,
        z_target=0.8416212335729143,
    )

    assert half == 36816
    assert eighty == 75222
    assert eighty > half
