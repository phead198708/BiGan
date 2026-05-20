"""Posterior evaluation and challenger trigger tests for issue #48."""

from __future__ import annotations

import json

from bigan.mlops import (
    RetrainingTriggerRules,
    connect_mlops_db,
    evaluate_challenger_trigger,
    initialize_mlops_db,
    record_challenger_trigger,
    register_training_result_as_challenger,
)
from bigan.monitoring import compute_feature_drift, write_drift_metrics


def _insert_daily(
    conn,
    *,
    hit_rate: float,
    brier_score: float,
    ece: float,
) -> None:
    conn.execute(
        """
        INSERT INTO model_monitoring_daily (
            "date", model_version, prediction_count, avg_prob,
            hit_rate, brier_score, ece, drift_metrics_json, created_at
        ) VALUES (CAST('2026-05-20' AS DATE), 'xgb-v1', 100, 0.52, ?, ?, ?, '{}', 1)
        """,
        [hit_rate, brier_score, ece],
    )


def test_challenger_trigger_stays_false_for_healthy_daily_metrics() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)
    _insert_daily(conn, hit_rate=0.58, brier_score=0.20, ece=0.04)

    decision = evaluate_challenger_trigger(
        conn,
        date="2026-05-20",
        model_version="xgb-v1",
    )

    assert decision.triggered is False
    assert decision.reasons == ()
    trigger_id = record_challenger_trigger(conn, decision)
    row = conn.execute(
        "SELECT trigger_id, triggered, status FROM challenger_triggers"
    ).fetchone()
    assert row == (trigger_id, False, "skipped")


def test_challenger_trigger_uses_hit_rate_brier_ece_and_critical_drift() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)
    _insert_daily(conn, hit_rate=0.42, brier_score=0.31, ece=0.18)
    drift = compute_feature_drift(
        [0.1, 0.2, 0.3],
        [0.8, 0.9, 1.0],
        feature_name="prob_up_15m",
        model_version="xgb-v1",
        date="2026-05-20",
    )
    write_drift_metrics(conn, [drift])

    decision = evaluate_challenger_trigger(
        conn,
        date="2026-05-20",
        model_version="xgb-v1",
        rules=RetrainingTriggerRules(min_hit_rate=0.50, max_brier_score=0.25, max_ece=0.10),
    )

    assert decision.triggered is True
    assert set(decision.reasons) == {
        "hit_rate_below_threshold",
        "brier_score_above_threshold",
        "ece_above_threshold",
        "critical_feature_drift",
    }
    trigger_id = record_challenger_trigger(
        conn,
        decision,
        retraining_job_uri="local://jobs/retrain-1",
        challenger_model_version="xgb-v2",
    )
    row = conn.execute(
        """
        SELECT trigger_id, triggered, retraining_job_uri, challenger_model_version, status
        FROM challenger_triggers
        """
    ).fetchone()
    assert row == (trigger_id, True, "local://jobs/retrain-1", "xgb-v2", "pending")


def test_missing_daily_monitoring_does_not_trigger_but_records_reason() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)

    decision = evaluate_challenger_trigger(
        conn,
        date="2026-05-20",
        model_version="xgb-v1",
    )

    assert decision.triggered is False
    assert decision.reasons == ("monitoring_daily_missing",)


def test_training_result_can_be_registered_as_active_challenger() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)

    register_training_result_as_challenger(
        conn,
        model_version="xgb-v2",
        model_family="btc-updown-15m",
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="bigan-training-15m-v1.0.0",
        train_config_hash="hash-v2",
        artifact_uri="models/btc-updown-15m/xgb-v2/model.json",
        train_started_at=1_000,
        train_finished_at=2_000,
        metrics_json=json.dumps({"test": {"brier_score": 0.20}}),
        notes="auto challenger from posterior trigger",
    )

    row = conn.execute(
        "SELECT model_version, status FROM model_registry WHERE model_version = 'xgb-v2'"
    ).fetchone()
    assert row == ("xgb-v2", "challenger")
