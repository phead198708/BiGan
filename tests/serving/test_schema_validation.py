"""Fail-closed feature schema validation tests for issue #44."""

from __future__ import annotations

import json

import pytest

from bigan.mlops import connect_mlops_db
from bigan.serving import (
    FeatureSchemaMismatch,
    build_feature_schema_artifact,
    load_feature_schema_artifact,
    validate_features_fail_closed,
    write_feature_schema_artifact,
)


def test_feature_schema_artifact_round_trips_with_hash(tmp_path) -> None:
    path = tmp_path / "feature_schema.json"
    artifact = write_feature_schema_artifact(
        path,
        ["spread", "mid_price", "ret_15m"],
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="bigan-training-15m-v1.0.0",
        model_version="xgboost-v1",
    )

    loaded = load_feature_schema_artifact(path)
    assert loaded == artifact
    assert json.loads(path.read_text(encoding="utf-8"))["schema_hash"] == artifact.schema_hash


def test_validate_features_accepts_exact_schema_and_order() -> None:
    schema = build_feature_schema_artifact(["spread", "mid_price", "ret_15m"])
    values = validate_features_fail_closed(
        {"spread": 0.02, "mid_price": 0.51, "ret_15m": 0.01},
        schema,
    )
    assert values == {"spread": 0.02, "mid_price": 0.51, "ret_15m": 0.01}


def test_validate_features_rejects_missing_extra_type_and_order_mismatches() -> None:
    schema = build_feature_schema_artifact(["spread", "mid_price", "ret_15m"])

    with pytest.raises(FeatureSchemaMismatch) as missing:
        validate_features_fail_closed({"spread": 0.02, "mid_price": 0.51}, schema)
    assert missing.value.details["missing"] == ["ret_15m"]

    with pytest.raises(FeatureSchemaMismatch) as extra:
        validate_features_fail_closed(
            {"spread": 0.02, "mid_price": 0.51, "ret_15m": 0.01, "surprise": 1.0},
            schema,
        )
    assert extra.value.details["extra"] == ["surprise"]

    with pytest.raises(FeatureSchemaMismatch) as bad_type:
        validate_features_fail_closed(
            {"spread": 0.02, "mid_price": "0.51", "ret_15m": 0.01},
            schema,
        )
    assert bad_type.value.details["type_errors"] == {"mid_price": "str"}

    with pytest.raises(FeatureSchemaMismatch) as wrong_order:
        validate_features_fail_closed(
            {"mid_price": 0.51, "spread": 0.02, "ret_15m": 0.01},
            schema,
        )
    assert wrong_order.value.details["expected_order"] == ["spread", "mid_price", "ret_15m"]


def test_schema_mismatch_logs_data_quality_incident() -> None:
    conn = connect_mlops_db()
    schema = build_feature_schema_artifact(
        ["spread", "mid_price", "ret_15m"],
        feature_version="bigan-mvp-v1.0.0",
        model_version="xgboost-v1",
    )

    with pytest.raises(FeatureSchemaMismatch):
        validate_features_fail_closed(
            {"spread": 0.02, "mid_price": 0.51},
            schema,
            incident_conn=conn,
            affected_symbol="token-1",
            request_id="req-1",
        )

    row = conn.execute(
        """
        SELECT incident_type, severity, affected_symbol, alert_id, details_json
        FROM data_quality_incidents
        """
    ).fetchone()
    assert row[:4] == ("schema_change", "critical", "token-1", "req-1")
    details = json.loads(row[4])
    assert details["missing"] == ["ret_15m"]
    assert details["request_id"] == "req-1"
