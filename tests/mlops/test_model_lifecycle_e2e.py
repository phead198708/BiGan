"""End-to-end model lifecycle drill covering issues #39-#48."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from bigan.mlops import (
    ModelDeploymentRecord,
    ModelRegistryRecord,
    complete_deployment,
    connect_mlops_db,
    current_champion,
    current_online_model,
    evaluate_challenger_trigger,
    initialize_mlops_db,
    model_artifact_uri,
    promote_model,
    record_challenger_trigger,
    record_deployment,
    register_model,
    register_training_result_as_challenger,
    rollback_deployment,
    run_shadow_comparison,
)
from bigan.monitoring import (
    PredictionEvent,
    PredictionOutcome,
    compute_brier_component,
    compute_feature_drift,
    drift_metrics_json,
    open_data_quality_incidents,
    record_prediction_event,
    record_prediction_outcome,
    summarize_model_monitoring_daily,
    write_drift_metrics,
)
from bigan.serving import (
    FeatureSchemaMismatch,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    build_feature_schema_artifact,
    validate_features_fail_closed,
)


class FixedOffsetModel:
    """Tiny fake probability model for lifecycle orchestration tests."""

    def __init__(self, *, model_version: str, offset: float = 0.0) -> None:
        self.model_version = model_version
        self.offset = offset

    def predict_proba(self, row: dict) -> float:
        return max(0.0, min(1.0, float(row["base_prob"]) + self.offset))


def test_model_lifecycle_e2e_drill_covers_registry_serving_monitoring_and_rollback() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)

    # #39: Register and promote the initial champion model.
    register_model(conn, _registry_record("xgb-v1", train_config_hash="hash-v1"))
    promote_model(conn, "xgb-v1", promoted_at=_ts("2026-05-20T00:00:00"))
    champion = current_champion(conn, "btc-updown-15m")
    assert champion is not None
    assert champion["model_version"] == "xgb-v1"

    # #40: Deploy the champion and locate the online production model.
    record_deployment(
        conn,
        ModelDeploymentRecord(
            deployment_id="deploy-xgb-v1",
            model_version="xgb-v1",
            environment="prod",
            rollout_strategy="full",
            traffic_percent=100.0,
            deployment_status="running",
            started_at=_ts("2026-05-20T00:01:00"),
            operator="ml-oncall",
            reason="initial champion deployment",
        ),
    )
    complete_deployment(conn, "deploy-xgb-v1", completed_at=_ts("2026-05-20T00:02:00"))
    online = current_online_model(conn, "prod")
    assert online is not None
    assert online["model_version"] == "xgb-v1"

    # #43 and #44: Fixed serving contract plus fail-closed schema validation.
    schema = build_feature_schema_artifact(
        ["spread", "mid_price", "ret_15m"],
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="bigan-training-15m-v1.0.0",
        model_version="xgb-v1",
    )
    request = PredictRequest(
        source_symbol="token-up",
        feature_version="bigan-mvp-v1.0.0",
        features={"spread": 0.02, "mid_price": 0.51, "ret_15m": 0.01},
        request_id="req-ok",
    )
    validated_features = validate_features_fail_closed(
        request.features,
        schema,
        incident_conn=conn,
        affected_symbol=request.source_symbol,
        request_id=request.request_id,
    )
    assert validated_features == {"spread": 0.02, "mid_price": 0.51, "ret_15m": 0.01}

    health = HealthResponse(
        status="ok",
        model_version="xgb-v1",
        checks={"model_loaded": True, "schema_loaded": True},
    )
    model_info = ModelInfoResponse(
        model_version="xgb-v1",
        model_family="btc-updown-15m",
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="bigan-training-15m-v1.0.0",
        calibration_method="platt",
        status="champion",
        artifact_uri=model_artifact_uri(
            "models",
            model_family="btc-updown-15m",
            model_version="xgb-v1",
        ),
        loaded_at=_ts("2026-05-20T00:02:00"),
    )
    response = PredictResponse(
        prob_up_15m=0.80,
        model_version=model_info.model_version,
        feature_version=request.feature_version,
        confidence_bucket="high_up",
        top_features_json=json.dumps([{"feature": "ret_15m", "contribution": 0.12}]),
        inference_ts=_ts("2026-05-20T00:03:00"),
        serving_latency_ms=4.5,
        request_id=request.request_id,
        event_id="evt-1",
    )
    assert health.status == "ok"
    assert response.prob_up_15m == pytest.approx(0.80)

    with pytest.raises(FeatureSchemaMismatch):
        validate_features_fail_closed(
            {"spread": 0.02, "mid_price": 0.51},
            schema,
            incident_conn=conn,
            affected_symbol="token-up",
            request_id="req-schema-bad",
        )
    critical_incidents = open_data_quality_incidents(conn, severity="critical")
    assert critical_incidents
    assert critical_incidents[0]["incident_type"] == "schema_change"

    # #41: Log prediction events, then backfill outcomes and daily monitoring.
    probs_and_labels = [
        ("evt-1", 0.90, False),
        ("evt-2", 0.80, False),
        ("evt-3", 0.70, False),
        ("evt-4", 0.60, False),
    ]
    for idx, (event_id, probability, label) in enumerate(probs_and_labels):
        event_ts = _ts(f"2026-05-20T00:0{idx}:00")
        record_prediction_event(
            conn,
            PredictionEvent(
                event_id=event_id,
                ts=event_ts,
                model_version="xgb-v1",
                feature_version="bigan-mvp-v1.0.0",
                prob_up_15m=probability,
                confidence_bucket="high_up",
                top_features_json=response.top_features_json,
                feature_hash=f"feature-hash-{idx}",
                feature_snapshot_json=json.dumps(validated_features, sort_keys=True),
                serving_latency_ms=4.0 + idx,
            ),
        )
        record_prediction_outcome(
            conn,
            PredictionOutcome(
                event_id=event_id,
                target_ts=event_ts + 900_000,
                realized_label=label,
                realized_return=-0.02,
                brier_component=compute_brier_component(probability, label),
                outcome_ts=event_ts + 900_000,
            ),
        )

    daily = summarize_model_monitoring_daily(
        conn,
        date="2026-05-20",
        model_version="xgb-v1",
    )
    assert daily.prediction_count == 4
    assert daily.hit_rate == pytest.approx(0.0)
    assert daily.brier_score is not None
    assert daily.brier_score > 0.25

    # #47: Compute feature drift and attach drift JSON to daily monitoring.
    drift_rows = [
        compute_feature_drift(
            [0.10, 0.12, 0.14, 0.16],
            [0.70, 0.72, 0.74, 0.76],
            feature_name="prob_up_15m",
            model_version="xgb-v1",
            date="2026-05-20",
            wasserstein_critical=0.10,
        )
    ]
    write_drift_metrics(conn, drift_rows)
    drift_payload = drift_metrics_json(drift_rows)
    daily_with_drift = summarize_model_monitoring_daily(
        conn,
        date="2026-05-20",
        model_version="xgb-v1",
        drift_metrics_json=drift_payload,
    )
    assert json.loads(daily_with_drift.drift_metrics_json or "{}")["prob_up_15m"][
        "severity"
    ] == "critical"

    # #48: Posterior metrics and critical drift trigger challenger training.
    decision = evaluate_challenger_trigger(
        conn,
        date="2026-05-20",
        model_version="xgb-v1",
    )
    assert decision.triggered is True
    assert {"brier_score_above_threshold", "critical_feature_drift"} <= set(decision.reasons)
    trigger_id = record_challenger_trigger(
        conn,
        decision,
        retraining_job_uri="local://jobs/retrain-20260520",
        challenger_model_version="xgb-v2",
    )
    assert trigger_id == "2026-05-20:xgb-v1"

    register_training_result_as_challenger(
        conn,
        model_version="xgb-v2",
        model_family="btc-updown-15m",
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="bigan-training-15m-v1.0.0",
        train_config_hash="hash-v2",
        artifact_uri=model_artifact_uri(
            "models",
            model_family="btc-updown-15m",
            model_version="xgb-v2",
        ),
        train_started_at=_ts("2026-05-20T01:00:00"),
        train_finished_at=_ts("2026-05-20T01:30:00"),
        metrics_json=json.dumps({"test": {"brier_score": 0.20, "roc_auc": 0.60}}),
        notes="triggered by e2e posterior drill",
    )
    challenger = conn.execute(
        "SELECT status FROM model_registry WHERE model_version = 'xgb-v2'"
    ).fetchone()
    assert challenger == ("challenger",)

    # #45: Run challenger in shadow mode without changing champion output.
    shadow = run_shadow_comparison(
        champion_model=FixedOffsetModel(model_version="xgb-v1"),
        challenger_model=FixedOffsetModel(model_version="xgb-v2", offset=0.05),
        feature_rows=[
            {"feature_ts": _ts("2026-05-20T02:00:00"), "source_symbol": "token-up", "base_prob": 0.45},
            {"feature_ts": _ts("2026-05-20T02:01:00"), "source_symbol": "token-up", "base_prob": 0.55},
        ],
        champion_model_version="xgb-v1",
        challenger_model_version="xgb-v2",
    )
    assert shadow.scored_count == 2
    assert shadow.challenger_error_count == 0
    assert shadow.mean_abs_probability_delta == pytest.approx(0.05)

    # #40 and #46: Deploy challenger, simulate regression, and roll back.
    record_deployment(
        conn,
        ModelDeploymentRecord(
            deployment_id="deploy-xgb-v2",
            model_version="xgb-v2",
            environment="prod",
            rollout_strategy="canary",
            traffic_percent=25.0,
            deployment_status="running",
            started_at=_ts("2026-05-20T03:00:00"),
            operator="ml-oncall",
            reason="challenger canary",
        ),
    )
    complete_deployment(conn, "deploy-xgb-v2", completed_at=_ts("2026-05-20T03:05:00"))
    assert current_online_model(conn, "prod")["model_version"] == "xgb-v2"

    rollback_deployment(
        conn,
        "deploy-xgb-v2",
        rollback_to_version="xgb-v1",
        rolled_back_at=_ts("2026-05-20T03:10:00"),
        operator="ml-oncall",
        reason="rollback drill: latency regression",
    )
    rolled_back = conn.execute(
        """
        SELECT deployment_status, rollback_to_version, reason
        FROM model_deployments
        WHERE deployment_id = 'deploy-xgb-v2'
        """
    ).fetchone()
    assert rolled_back == ("rolled_back", "xgb-v1", "rollback drill: latency regression")
    assert current_online_model(conn, "prod")["model_version"] == "xgb-v1"


def _registry_record(model_version: str, *, train_config_hash: str) -> ModelRegistryRecord:
    return ModelRegistryRecord(
        model_version=model_version,
        model_family="btc-updown-15m",
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="bigan-training-15m-v1.0.0",
        train_config_hash=train_config_hash,
        artifact_uri=model_artifact_uri(
            "models",
            model_family="btc-updown-15m",
            model_version=model_version,
        ),
        calibration_artifact_uri=None,
        status="candidate",
        train_started_at=_ts("2026-05-19T23:00:00"),
        train_finished_at=_ts("2026-05-19T23:30:00"),
        metrics_json=json.dumps({"test": {"brier_score": 0.22, "roc_auc": 0.58}}),
        backtest_json=json.dumps({"net_pnl": 1.0}),
        notes="e2e drill champion candidate",
    )


def _ts(iso_without_zone: str) -> int:
    return int(datetime.fromisoformat(iso_without_zone).replace(tzinfo=UTC).timestamp() * 1000)
