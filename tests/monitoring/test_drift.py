"""Feature drift detection tests for issue #47."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.mlops import connect_mlops_db, initialize_mlops_db
from bigan.monitoring import (
    ChampionDriftThresholds,
    PredictionEvent,
    PredictionOutcome,
    build_champion_drift_baseline,
    champion_baseline_distribution,
    compute_brier_component,
    compute_feature_drift,
    drift_metrics_json,
    drift_report_from_rows,
    evaluate_live_champion_drift,
    kolmogorov_smirnov_statistic,
    open_data_quality_incidents,
    population_stability_index,
    record_champion_drift_incidents,
    record_prediction_event,
    record_prediction_outcome,
    run_live_champion_monitoring,
    wasserstein_distance,
    write_drift_metrics,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_xgboost_v4_champion_baseline_is_registered() -> None:
    baseline = champion_baseline_distribution("xgboost-v4")

    assert baseline["count"] == 2602
    assert baseline["mean"] == pytest.approx(0.5633657718039641)
    assert baseline["std"] == pytest.approx(0.43144887503556245)


def test_build_champion_drift_baseline_writes_reference_artifact(tmp_path: Path) -> None:
    offline_reference = _write_json(
        tmp_path / "candidate-eval" / "offline_reference.json",
        {
            "model_version": "xgboost-v4",
            "model_path": "runs/xgboost-v4/model.json",
            "dataset_dir": "runs/training-dataset",
            "dataset_version": "dataset-v1",
            "split": "val",
            "probability_distribution": {
                "count": 100,
                "mean": 0.55,
                "std": 0.10,
                "p50": 0.56,
            },
            "edge_distribution": {
                "count": 100,
                "mean": 0.04,
                "std": 0.08,
            },
            "edge_trigger_rate_at_0_30": 0.12,
        },
    )
    output_path = tmp_path / "cutover" / "drift-baseline.json"

    baseline = build_champion_drift_baseline(
        str(offline_reference),
        str(output_path),
        thresholds=ChampionDriftThresholds(probability_mean_shift_abs=0.07),
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == baseline
    assert baseline["source_offline_reference_path"] == str(offline_reference)
    assert baseline["model_version"] == "xgboost-v4"
    assert baseline["dataset_version"] == "dataset-v1"
    assert baseline["split"] == "val"
    assert baseline["probability_distribution"]["count"] == 100
    assert baseline["thresholds"]["probability_mean_shift_abs"] == 0.07


def test_build_champion_drift_baseline_rejects_non_validation_reference(
    tmp_path: Path,
) -> None:
    offline_reference = _write_json(
        tmp_path / "candidate-eval" / "offline_reference.json",
        {
            "model_version": "xgboost-v4",
            "dataset_dir": "runs/training-dataset",
            "dataset_version": "dataset-v1",
            "split": "test",
            "probability_distribution": {"count": 100, "mean": 0.55, "std": 0.10},
        },
    )

    with pytest.raises(ValueError, match="split must be val"):
        build_champion_drift_baseline(str(offline_reference))


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


def test_live_champion_drift_records_prediction_and_label_alerts() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)
    now_ms = 1_800_000
    thresholds = ChampionDriftThresholds(label_consecutive_samples=5)
    for idx in range(6):
        event_ts = now_ms - 5 * 60_000 + idx * 60_000
        probability = 0.75 + idx * 0.01
        event = PredictionEvent(
            event_id=f"evt-{idx}",
            ts=event_ts,
            model_version="xgboost-v3",
            feature_version="bigan-mvp-v1.0.0",
            prob_up_15m=probability,
            confidence_bucket="high_up",
            top_features_json="[]",
            feature_hash=f"hash-{idx}",
            feature_snapshot_json=json.dumps({"market_implied_prob": 0.60}),
            serving_latency_ms=4.0,
        )
        record_prediction_event(conn, event)
        record_prediction_outcome(
            conn,
            PredictionOutcome(
                event_id=event.event_id,
                target_ts=event_ts + 900_000,
                realized_label=False,
                realized_return=-0.60,
                brier_component=compute_brier_component(probability, False),
                outcome_ts=event_ts + 900_000,
            ),
        )

    report = evaluate_live_champion_drift(
        conn,
        model_version="xgboost-v3",
        reference_distribution={"mean": 0.55, "std": 0.10},
        now_ms=now_ms,
        windows_ms=(60 * 60 * 1000, 6 * 60 * 60 * 1000),
        thresholds=thresholds,
    )
    incident_ids = record_champion_drift_incidents(conn, report)

    assert report["passed"] is False
    assert {"probability_mean_shift", "edge_trigger_zero", "label_hit_rate_low"} <= {
        alert["alert_type"] for alert in report["alerts"]
    }
    assert incident_ids
    incidents = open_data_quality_incidents(conn)
    assert {"prediction_drift", "label_shift"} <= {row["incident_type"] for row in incidents}


def test_live_champion_monitoring_uses_registered_baseline_and_records_incidents() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)
    now_ms = 7_200_000
    thresholds = ChampionDriftThresholds(label_consecutive_samples=5)
    for idx in range(5):
        event_ts = now_ms - 4 * 60_000 + idx * 60_000
        event = PredictionEvent(
            event_id=f"live-bad-{idx}",
            ts=event_ts,
            model_version="xgboost-v3",
            feature_version="bigan-mvp-v1.0.0",
            prob_up_15m=0.10,
            confidence_bucket="neutral",
            top_features_json="[]",
            feature_hash=f"hash-bad-{idx}",
            feature_snapshot_json=json.dumps({"market_implied_prob": 0.60}),
            serving_latency_ms=3.0,
        )
        record_prediction_event(conn, event)
        record_prediction_outcome(
            conn,
            PredictionOutcome(
                event_id=event.event_id,
                target_ts=event_ts + 900_000,
                realized_label=False,
                realized_return=-0.60,
                brier_component=compute_brier_component(event.prob_up_15m, False),
                outcome_ts=event_ts + 900_000,
            ),
        )

    report = run_live_champion_monitoring(
        conn,
        model_version="xgboost-v3",
        now_ms=now_ms,
        windows_ms=(60 * 60 * 1000,),
        thresholds=thresholds,
    )

    assert report["passed"] is False
    assert report["reference_distribution"]["mean"] == pytest.approx(0.5784535391099995)
    assert {
        "probability_mean_shift",
        "probability_std_shift",
        "edge_trigger_zero",
        "label_hit_rate_low",
    } <= {
        alert["alert_type"] for alert in report["alerts"]
    }
    assert len(report["incident_ids"]) == 4


def test_live_champion_monitoring_passes_when_prediction_stream_matches_baseline() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)
    now_ms = 7_200_000
    probabilities = [0.0, 0.10, 0.60, 0.99, 0.99]
    for idx, probability in enumerate(probabilities):
        event_ts = now_ms - 4 * 60_000 + idx * 60_000
        record_prediction_event(
            conn,
            PredictionEvent(
                event_id=f"live-ok-{idx}",
                ts=event_ts,
                model_version="xgboost-v3",
                feature_version="bigan-mvp-v1.0.0",
                prob_up_15m=probability,
                confidence_bucket="medium_up",
                top_features_json="[]",
                feature_hash=f"hash-ok-{idx}",
                feature_snapshot_json=json.dumps({"market_implied_prob": 0.0}),
                serving_latency_ms=3.0,
            ),
        )

    report = run_live_champion_monitoring(
        conn,
        model_version="xgboost-v3",
        now_ms=now_ms,
        windows_ms=(60 * 60 * 1000,),
    )

    assert report["passed"] is True
    assert report["alerts"] == []
    assert report["incident_ids"] == []


def test_live_edge_trigger_monitoring_is_outcome_side_aware() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)
    now_ms = 7_200_000
    record_prediction_event(
        conn,
        PredictionEvent(
            event_id="down-token-edge",
            ts=now_ms - 60_000,
            model_version="xgboost-v3",
            feature_version="bigan-mvp-v1.0.0",
            prob_up_15m=0.10,
            confidence_bucket="medium_down",
            top_features_json="[]",
            feature_hash="hash-down-token-edge",
            feature_snapshot_json=json.dumps(
                {
                    "canonical_symbol": "BTC-15M:btc-updown-15m-1:DOWN",
                    "features": {"market_implied_prob": 0.60},
                }
            ),
            serving_latency_ms=3.0,
        ),
    )

    report = evaluate_live_champion_drift(
        conn,
        model_version="xgboost-v3",
        reference_distribution={"mean": 0.10, "std": 0.0},
        now_ms=now_ms,
        windows_ms=(60 * 60 * 1000,),
    )

    assert report["edge_zero_window"]["trigger_rate"] == pytest.approx(1.0)
    assert "edge_trigger_zero" not in {alert["alert_type"] for alert in report["alerts"]}
