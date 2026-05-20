"""Deployment audit catalog tests for issue #40."""

from __future__ import annotations

import duckdb
import pytest

from bigan.mlops import (
    DEPLOYMENT_STATUSES,
    ModelDeploymentRecord,
    complete_deployment,
    connect_mlops_db,
    current_online_model,
    initialize_mlops_db,
    record_deployment,
    rollback_deployment,
)


def _deployment(
    deployment_id: str,
    *,
    model_version: str = "xgb-v1",
    status: str = "running",
    traffic_percent: float = 10.0,
    started_at: int = 1_000,
) -> ModelDeploymentRecord:
    return ModelDeploymentRecord(
        deployment_id=deployment_id,
        model_version=model_version,
        environment="prod",
        rollout_strategy="canary",
        traffic_percent=traffic_percent,
        deployment_status=status,
        started_at=started_at,
        operator="codex",
        reason="mvp rollout",
    )


def test_model_deployments_ddl_creates_audit_columns_and_constraints() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info('model_deployments')").fetchall()}
    assert {
        "deployment_id",
        "model_version",
        "environment",
        "rollout_strategy",
        "traffic_percent",
        "deployment_status",
        "started_at",
        "completed_at",
        "rolled_back_at",
        "rollback_to_version",
        "operator",
        "reason",
    } <= columns

    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            """
            INSERT INTO model_deployments (
                deployment_id, model_version, environment, rollout_strategy,
                traffic_percent, deployment_status, started_at, created_at, updated_at
            ) VALUES ('bad', 'xgb-v1', 'prod', 'canary', 110,
                'succeeded', 1, 1, 1)
            """
        )


def test_current_online_model_tracks_latest_succeeded_deployment() -> None:
    conn = connect_mlops_db()
    record_deployment(conn, _deployment("deploy-1", model_version="xgb-v1", started_at=1_000))
    complete_deployment(conn, "deploy-1", completed_at=2_000)
    record_deployment(conn, _deployment("deploy-2", model_version="xgb-v2", started_at=3_000))
    complete_deployment(conn, "deploy-2", completed_at=4_000)

    current = current_online_model(conn, "prod")
    assert current is not None
    assert current["deployment_id"] == "deploy-2"
    assert current["model_version"] == "xgb-v2"
    assert current["deployment_status"] == "succeeded"


def test_rollback_event_records_target_version_and_removes_from_current_view() -> None:
    conn = connect_mlops_db()
    record_deployment(conn, _deployment("deploy-1", model_version="xgb-v1", started_at=1_000))
    complete_deployment(conn, "deploy-1", completed_at=2_000)
    record_deployment(conn, _deployment("deploy-2", model_version="xgb-v2", started_at=3_000))
    complete_deployment(conn, "deploy-2", completed_at=4_000)

    rollback_deployment(
        conn,
        "deploy-2",
        rollback_to_version="xgb-v1",
        rolled_back_at=5_000,
        operator="ops",
        reason="latency regression",
    )

    rolled_back = conn.execute(
        """
        SELECT deployment_status, rolled_back_at, rollback_to_version, operator, reason
        FROM model_deployments
        WHERE deployment_id = 'deploy-2'
        """
    ).fetchone()
    assert rolled_back == ("rolled_back", 5_000, "xgb-v1", "ops", "latency regression")
    assert current_online_model(conn, "prod")["model_version"] == "xgb-v1"


def test_deployment_status_contract_is_fixed() -> None:
    assert set(DEPLOYMENT_STATUSES) == {
        "planned",
        "running",
        "succeeded",
        "failed",
        "rolled_back",
        "offline",
    }

    conn = connect_mlops_db()
    with pytest.raises(ValueError, match="invalid deployment status"):
        record_deployment(conn, _deployment("bad", status="unknown"))
