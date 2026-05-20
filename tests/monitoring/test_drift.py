"""Feature drift detection tests for issue #47."""

from __future__ import annotations

import json

import pytest

from bigan.mlops import connect_mlops_db, initialize_mlops_db
from bigan.monitoring import (
    compute_feature_drift,
    drift_metrics_json,
    drift_report_from_rows,
    kolmogorov_smirnov_statistic,
    population_stability_index,
    wasserstein_distance,
    write_drift_metrics,
)


def test_drift_metrics_are_zero_for_identical_distributions() -> None:
    reference = [0.1, 0.2, 0.3, 0.4, 0.5]
    current = [0.1, 0.2, 0.3, 0.4, 0.5]

    row = compute_feature_drift(
        reference,
        current,
        feature_name="prob_up_15m",
        model_version="xgb-v1",
        date="2026-05-20",
        bins=5,
    )

    assert row.psi == pytest.approx(0.0)
    assert row.ks_statistic == pytest.approx(0.0)
    assert row.wasserstein_distance == pytest.approx(0.0)
    assert row.severity == "ok"


def test_shifted_distribution_triggers_critical_drift() -> None:
    row = compute_feature_drift(
        [0.10, 0.12, 0.14, 0.16, 0.18],
        [0.70, 0.72, 0.74, 0.76, 0.78],
        feature_name="imbalance",
        model_version="xgb-v1",
        date="2026-05-20",
        bins=5,
        wasserstein_critical=0.10,
    )

    assert row.psi > 0.25
    assert row.ks_statistic == pytest.approx(1.0)
    assert row.wasserstein_distance > 0.10
    assert row.severity == "critical"


def test_drift_metrics_are_persisted_and_serialized_for_daily_monitoring() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)
    rows = [
        compute_feature_drift(
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            feature_name="best_bid",
            model_version="xgb-v1",
            date="2026-05-20",
        ),
        compute_feature_drift(
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            feature_name="best_ask",
            model_version="xgb-v1",
            date="2026-05-20",
        ),
    ]

    write_drift_metrics(conn, rows)

    stored = conn.execute(
        """
        SELECT feature_name, severity
        FROM drift_metrics
        WHERE "date" = CAST(? AS DATE)
        ORDER BY feature_name
        """,
        ["2026-05-20"],
    ).fetchall()
    assert stored == [("best_ask", "critical"), ("best_bid", "ok")]
    payload = json.loads(drift_metrics_json(rows))
    assert payload["best_bid"]["psi"] == pytest.approx(0.0)
    assert payload["best_ask"]["severity"] == "critical"


def test_drift_report_from_feature_rows_handles_key_feature_names() -> None:
    reference_rows = [
        {"best_bid": 0.49, "best_ask": 0.51, "imbalance": 0.1, "prob_up_15m": 0.55},
        {"best_bid": 0.48, "best_ask": 0.52, "imbalance": 0.0, "prob_up_15m": 0.50},
    ]
    current_rows = [
        {"best_bid": 0.45, "best_ask": 0.55, "imbalance": -0.4, "prob_up_15m": 0.70},
        {"best_bid": 0.44, "best_ask": 0.56, "imbalance": -0.5, "prob_up_15m": 0.72},
    ]

    rows = drift_report_from_rows(
        reference_rows,
        current_rows,
        feature_names=["best_bid", "best_ask", "imbalance", "prob_up_15m"],
        model_version="xgb-v1",
        date="2026-05-20",
    )

    assert [row.feature_name for row in rows] == [
        "best_bid",
        "best_ask",
        "imbalance",
        "prob_up_15m",
    ]
    assert any(row.severity == "critical" for row in rows)


def test_metric_helpers_validate_empty_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index([], [0.1])
    with pytest.raises(ValueError, match="non-empty"):
        kolmogorov_smirnov_statistic([0.1], [])
    with pytest.raises(ValueError, match="non-empty"):
        wasserstein_distance([], [0.1])
