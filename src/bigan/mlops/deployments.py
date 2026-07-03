"""Model deployment audit catalog (issue #40)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import duckdb

ACTIVE_CHAMPION_MODEL_VERSION = "xgboost-v4"
ACTIVE_CHAMPION_ENVIRONMENT = "prod"
ACTIVE_CHAMPION_DEPLOYMENT_ID = "cutover-xgboost-v4-20260523T105710Z"

DEPLOYMENT_STATUSES: tuple[str, ...] = (
    "planned",
    "running",
    "succeeded",
    "failed",
    "rolled_back",
    "offline",
)

MODEL_DEPLOYMENTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS model_deployments (
    deployment_id VARCHAR PRIMARY KEY,
    model_version VARCHAR NOT NULL,
    environment VARCHAR NOT NULL,
    rollout_strategy VARCHAR NOT NULL,
    traffic_percent DOUBLE NOT NULL CHECK (traffic_percent >= 0 AND traffic_percent <= 100),
    deployment_status VARCHAR NOT NULL CHECK (
        deployment_status IN ('planned', 'running', 'succeeded', 'failed', 'rolled_back', 'offline')
    ),
    started_at BIGINT NOT NULL,
    completed_at BIGINT,
    rolled_back_at BIGINT,
    rollback_to_version VARCHAR,
    operator VARCHAR,
    reason VARCHAR,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
)
"""

MODEL_DEPLOYMENTS_VIEWS_DDL: tuple[str, ...] = (
    """
    CREATE OR REPLACE VIEW current_online_models AS
    SELECT *
    FROM model_deployments
    WHERE deployment_status = 'succeeded'
      AND traffic_percent > 0
      AND rolled_back_at IS NULL
    QUALIFY row_number() OVER (
        PARTITION BY environment
        ORDER BY completed_at DESC NULLS LAST, started_at DESC, deployment_id DESC
    ) = 1
    """,
)


@dataclass(frozen=True, slots=True)
class ModelDeploymentRecord:
    """One deployment/cutover/rollback audit event."""

    deployment_id: str
    model_version: str
    environment: str
    rollout_strategy: str
    traffic_percent: float
    deployment_status: str
    started_at: int
    completed_at: int | None = None
    rolled_back_at: int | None = None
    rollback_to_version: str | None = None
    operator: str | None = None
    reason: str | None = None
    created_at: int | None = None
    updated_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        now_ms = _now_ms()
        row["created_at"] = self.created_at or now_ms
        row["updated_at"] = self.updated_at or now_ms
        return row


def record_deployment(
    conn: duckdb.DuckDBPyConnection,
    record: ModelDeploymentRecord,
    *,
    replace: bool = False,
) -> None:
    """Insert one deployment audit event."""

    _initialize_deployment_tables(conn)
    _validate_record(record)
    row = record.to_row()
    columns = tuple(row)
    placeholders = ", ".join("?" for _ in columns)
    statement = (
        f"INSERT {'OR REPLACE ' if replace else ''}INTO model_deployments "
        f"({', '.join(columns)}) VALUES ({placeholders})"
    )
    conn.execute(statement, [row[column] for column in columns])


def complete_deployment(
    conn: duckdb.DuckDBPyConnection,
    deployment_id: str,
    *,
    completed_at: int | None = None,
    status: str = "succeeded",
) -> None:
    """Mark a deployment complete."""

    if status not in {"succeeded", "failed", "offline"}:
        raise ValueError("completion status must be succeeded, failed, or offline")
    _initialize_deployment_tables(conn)
    _require_known_deployment(conn, deployment_id)
    ts = completed_at or _now_ms()
    conn.execute(
        """
        UPDATE model_deployments
        SET deployment_status = ?, completed_at = ?, updated_at = ?
        WHERE deployment_id = ?
        """,
        [status, ts, ts, deployment_id],
    )


def rollback_deployment(
    conn: duckdb.DuckDBPyConnection,
    deployment_id: str,
    *,
    rollback_to_version: str,
    rolled_back_at: int | None = None,
    operator: str | None = None,
    reason: str | None = None,
) -> None:
    """Record rollback metadata for a deployment event."""

    _require_non_empty("rollback_to_version", rollback_to_version)
    _initialize_deployment_tables(conn)
    _require_known_deployment(conn, deployment_id)
    ts = rolled_back_at or _now_ms()
    conn.execute(
        """
        UPDATE model_deployments
        SET deployment_status = 'rolled_back',
            rolled_back_at = ?,
            rollback_to_version = ?,
            operator = coalesce(?, operator),
            reason = coalesce(?, reason),
            updated_at = ?
        WHERE deployment_id = ?
        """,
        [ts, rollback_to_version, operator, reason, ts, deployment_id],
    )


def current_online_model(
    conn: duckdb.DuckDBPyConnection,
    environment: str,
) -> dict[str, Any] | None:
    """Return the latest succeeded online deployment for one environment."""

    _initialize_deployment_tables(conn)
    row = conn.execute(
        """
        SELECT *
        FROM current_online_models
        WHERE environment = ?
        """,
        [environment],
    ).fetchone()
    if row is None:
        return None
    columns = [column[0] for column in conn.description]
    return dict(zip(columns, row, strict=True))


def _initialize_deployment_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(MODEL_DEPLOYMENTS_TABLE_DDL)
    for ddl in MODEL_DEPLOYMENTS_VIEWS_DDL:
        conn.execute(ddl)


def _validate_record(record: ModelDeploymentRecord) -> None:
    for name in ("deployment_id", "model_version", "environment", "rollout_strategy"):
        _require_non_empty(name, str(getattr(record, name)))
    if record.deployment_status not in DEPLOYMENT_STATUSES:
        raise ValueError(f"invalid deployment status: {record.deployment_status!r}")
    if record.traffic_percent < 0 or record.traffic_percent > 100:
        raise ValueError("traffic_percent must be in [0, 100]")
    if record.completed_at is not None and record.completed_at < record.started_at:
        raise ValueError("completed_at must be >= started_at")
    if record.deployment_status == "rolled_back" and not record.rollback_to_version:
        raise ValueError("rolled_back deployments require rollback_to_version")


def _require_known_deployment(conn: duckdb.DuckDBPyConnection, deployment_id: str) -> None:
    row = conn.execute(
        "SELECT deployment_id FROM model_deployments WHERE deployment_id = ?",
        [deployment_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown deployment_id: {deployment_id}")


def _require_non_empty(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} is required")


def _now_ms() -> int:
    return int(time.time() * 1000)
