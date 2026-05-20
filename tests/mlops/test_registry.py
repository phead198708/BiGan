"""Model registry DDL and lifecycle tests for issue #39."""

from __future__ import annotations

import json

import duckdb
import pytest

from bigan.mlops import (
    MODEL_REGISTRY_STATUSES,
    ModelRegistryRecord,
    connect_mlops_db,
    current_champion,
    initialize_mlops_db,
    model_artifact_uri,
    promote_model,
    register_model,
    retire_model,
)


def _record(version: str, *, status: str = "candidate") -> ModelRegistryRecord:
    return ModelRegistryRecord(
        model_version=version,
        model_family="btc-updown-15m",
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="bigan-training-15m-v1.0.0",
        train_config_hash=f"hash-{version}",
        artifact_uri=model_artifact_uri(
            "models",
            model_family="btc-updown-15m",
            model_version=version,
        ),
        calibration_artifact_uri=None,
        status=status,
        train_started_at=1_000,
        train_finished_at=2_000,
        metrics_json=json.dumps({"test": {"roc_auc": 0.55}}),
        backtest_json=json.dumps({"net_pnl": 1.25}),
        notes="smoke",
    )


def test_model_registry_ddl_creates_required_columns_and_status_constraint() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info('model_registry')").fetchall()}
    assert {
        "model_version",
        "model_family",
        "feature_version",
        "dataset_version",
        "train_config_hash",
        "artifact_uri",
        "calibration_artifact_uri",
        "status",
        "train_started_at",
        "train_finished_at",
        "promoted_at",
        "retired_at",
        "metrics_json",
        "backtest_json",
        "notes",
    } <= columns

    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            """
            INSERT INTO model_registry (
                model_version, model_family, feature_version, dataset_version,
                train_config_hash, artifact_uri, status, train_started_at,
                train_finished_at, metrics_json, created_at, updated_at
            ) VALUES ('bad', 'family', 'feat', 'data', 'hash', 'uri',
                'production', 1, 2, '{}', 3, 3)
            """
        )


def test_register_promote_and_query_current_champion() -> None:
    conn = connect_mlops_db()
    register_model(conn, _record("xgb-v1"))
    register_model(conn, _record("xgb-v2", status="challenger"))

    promote_model(conn, "xgb-v1", promoted_at=3_000)
    champion = current_champion(conn, "btc-updown-15m")
    assert champion is not None
    assert champion["model_version"] == "xgb-v1"
    assert champion["status"] == "champion"

    promote_model(conn, "xgb-v2", promoted_at=4_000)
    champion = current_champion(conn, "btc-updown-15m")
    assert champion is not None
    assert champion["model_version"] == "xgb-v2"

    retired = conn.execute(
        "SELECT status, retired_at FROM model_registry WHERE model_version = 'xgb-v1'"
    ).fetchone()
    assert retired == ("retired", 4_000)


def test_registry_enforces_one_active_challenger_per_family() -> None:
    conn = connect_mlops_db()
    register_model(conn, _record("xgb-v2", status="challenger"))

    with pytest.raises(ValueError, match="active challenger"):
        register_model(conn, _record("xgb-v3", status="challenger"))

    retire_model(conn, "xgb-v2", retired_at=5_000)
    register_model(conn, _record("xgb-v3", status="challenger"))
    rows = conn.execute(
        "SELECT model_version FROM model_registry WHERE status = 'challenger'"
    ).fetchall()
    assert rows == [("xgb-v3",)]


def test_model_artifact_uri_uses_family_version_filename_layout() -> None:
    assert (
        model_artifact_uri("s3://bigan-models", model_family="xgb", model_version="v1")
        == "s3://bigan-models/xgb/v1/model.json"
    )
    assert set(MODEL_REGISTRY_STATUSES) == {"candidate", "challenger", "champion", "retired"}
