"""Model registry catalog and lifecycle operations (issue #39)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from bigan.monitoring.events import (
    MONITORING_TABLES_DDL,
)

from .deployments import MODEL_DEPLOYMENTS_TABLE_DDL, MODEL_DEPLOYMENTS_VIEWS_DDL

DEFAULT_MLOPS_DB_PATH = Path("data/mlops/champion_catalog.duckdb")
ACTIVE_MODEL_FAMILY = "btc-updown-15m"
DEFAULT_MLOPS_CONNECT_RETRIES = 12
DEFAULT_MLOPS_CONNECT_RETRY_DELAY_SECONDS = 0.5

MODEL_REGISTRY_STATUSES: tuple[str, ...] = (
    "candidate",
    "challenger",
    "champion",
    "retired",
)

MODEL_REGISTRY_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS model_registry (
    model_version VARCHAR PRIMARY KEY,
    model_family VARCHAR NOT NULL,
    feature_version VARCHAR NOT NULL,
    dataset_version VARCHAR NOT NULL,
    train_config_hash VARCHAR NOT NULL,
    artifact_uri VARCHAR NOT NULL,
    calibration_artifact_uri VARCHAR,
    status VARCHAR NOT NULL CHECK (status IN ('candidate', 'challenger', 'champion', 'retired')),
    train_started_at BIGINT NOT NULL,
    train_finished_at BIGINT NOT NULL,
    promoted_at BIGINT,
    retired_at BIGINT,
    metrics_json VARCHAR NOT NULL,
    backtest_json VARCHAR,
    notes VARCHAR,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    UNIQUE (model_family, train_config_hash)
)
"""

MODEL_REGISTRY_VIEWS_DDL: tuple[str, ...] = (
    """
    CREATE OR REPLACE VIEW current_champion_models AS
    SELECT *
    FROM model_registry
    WHERE status = 'champion'
      AND retired_at IS NULL
    QUALIFY row_number() OVER (
        PARTITION BY model_family
        ORDER BY promoted_at DESC NULLS LAST, updated_at DESC, model_version DESC
    ) = 1
    """,
)


@dataclass(frozen=True, slots=True)
class ModelRegistryRecord:
    """Serializable model registry row."""

    model_version: str
    model_family: str
    feature_version: str
    dataset_version: str
    train_config_hash: str
    artifact_uri: str
    calibration_artifact_uri: str | None
    status: str
    train_started_at: int
    train_finished_at: int
    metrics_json: str
    backtest_json: str | None = None
    notes: str | None = None
    promoted_at: int | None = None
    retired_at: int | None = None
    created_at: int | None = None
    updated_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        now_ms = _now_ms()
        row["created_at"] = self.created_at or now_ms
        row["updated_at"] = self.updated_at or now_ms
        return row


def connect_mlops_db(
    path: Path | str = ":memory:",
    *,
    retry_attempts: int | None = None,
    retry_delay_seconds: float | None = None,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection for the MLOps catalog with lock retry."""

    attempts = _connect_retry_attempts(retry_attempts)
    delay = _connect_retry_delay(retry_delay_seconds)
    last_exc: duckdb.Error | None = None
    for attempt in range(attempts):
        try:
            return _connect_duckdb(path, read_only=read_only)
        except duckdb.Error as exc:
            if not _is_duckdb_lock_error(exc):
                raise
            last_exc = exc
            if attempt == attempts - 1:
                break
            if delay > 0:
                time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    return _connect_duckdb(path, read_only=read_only)


def _connect_duckdb(path: Path | str, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    if read_only:
        return duckdb.connect(str(path), read_only=True)
    return duckdb.connect(str(path))


def initialize_mlops_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create MLOps catalog tables and query views."""

    conn.execute(MODEL_REGISTRY_TABLE_DDL)
    conn.execute(MODEL_DEPLOYMENTS_TABLE_DDL)
    for ddl in MONITORING_TABLES_DDL:
        conn.execute(ddl)
    for ddl in MODEL_REGISTRY_VIEWS_DDL:
        conn.execute(ddl)
    for ddl in MODEL_DEPLOYMENTS_VIEWS_DDL:
        conn.execute(ddl)


def model_artifact_uri(
    root: Path | str,
    *,
    model_family: str,
    model_version: str,
    filename: str = "model.json",
) -> str:
    """Return the canonical relative artifact path for a registered model."""

    _require_non_empty("model_family", model_family)
    _require_non_empty("model_version", model_version)
    _require_non_empty("filename", filename)
    root_text = str(root).rstrip("/")
    if "://" in root_text:
        return f"{root_text}/{model_family}/{model_version}/{filename}"
    return str(Path(root) / model_family / model_version / filename)


def register_model(
    conn: duckdb.DuckDBPyConnection,
    record: ModelRegistryRecord,
    *,
    replace: bool = False,
) -> None:
    """Insert a model registry row after lifecycle validation."""

    _validate_record(record)
    initialize_mlops_db(conn)
    row = record.to_row()
    _validate_singleton_status(
        conn,
        model_family=record.model_family,
        status=record.status,
        model_version=record.model_version,
    )
    columns = tuple(row)
    placeholders = ", ".join("?" for _ in columns)
    statement = (
        f"INSERT {'OR REPLACE ' if replace else ''}INTO model_registry "
        f"({', '.join(columns)}) VALUES ({placeholders})"
    )
    conn.execute(statement, [row[column] for column in columns])


def promote_model(
    conn: duckdb.DuckDBPyConnection,
    model_version: str,
    *,
    promoted_at: int | None = None,
) -> None:
    """Promote one model to champion and retire the previous champion."""

    initialize_mlops_db(conn)
    model = _fetch_model(conn, model_version)
    if model is None:
        raise ValueError(f"unknown model_version: {model_version}")
    family = str(model["model_family"])
    ts = promoted_at or _now_ms()
    conn.execute(
        """
        UPDATE model_registry
        SET status = 'retired', retired_at = ?, updated_at = ?
        WHERE model_family = ?
          AND status = 'champion'
          AND model_version <> ?
          AND retired_at IS NULL
        """,
        [ts, ts, family, model_version],
    )
    conn.execute(
        """
        UPDATE model_registry
        SET status = 'champion', promoted_at = ?, retired_at = NULL, updated_at = ?
        WHERE model_version = ?
        """,
        [ts, ts, model_version],
    )


def retire_model(
    conn: duckdb.DuckDBPyConnection,
    model_version: str,
    *,
    retired_at: int | None = None,
) -> None:
    """Mark a model as retired."""

    initialize_mlops_db(conn)
    if _fetch_model(conn, model_version) is None:
        raise ValueError(f"unknown model_version: {model_version}")
    ts = retired_at or _now_ms()
    conn.execute(
        """
        UPDATE model_registry
        SET status = 'retired', retired_at = ?, updated_at = ?
        WHERE model_version = ?
        """,
        [ts, ts, model_version],
    )


def current_champion(
    conn: duckdb.DuckDBPyConnection,
    model_family: str,
) -> dict[str, Any] | None:
    """Return the active champion row for one model family."""

    initialize_mlops_db(conn)
    rows = conn.execute(
        """
        SELECT *
        FROM current_champion_models
        WHERE model_family = ?
        """,
        [model_family],
    ).fetchall()
    if not rows:
        return None
    columns = [column[0] for column in conn.description]
    return dict(zip(columns, rows[0], strict=True))


def model_by_version(
    conn: duckdb.DuckDBPyConnection,
    model_version: str,
) -> dict[str, Any] | None:
    """Return one registry row by model version."""

    initialize_mlops_db(conn)
    return _fetch_model(conn, model_version)


def _validate_record(record: ModelRegistryRecord) -> None:
    for name in (
        "model_version",
        "model_family",
        "feature_version",
        "dataset_version",
        "train_config_hash",
        "artifact_uri",
    ):
        _require_non_empty(name, str(getattr(record, name)))
    if record.status not in MODEL_REGISTRY_STATUSES:
        raise ValueError(f"invalid model status: {record.status!r}")
    if record.train_finished_at < record.train_started_at:
        raise ValueError("train_finished_at must be >= train_started_at")
    _validate_json("metrics_json", record.metrics_json)
    if record.backtest_json is not None:
        _validate_json("backtest_json", record.backtest_json)


def _validate_singleton_status(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_family: str,
    status: str,
    model_version: str,
) -> None:
    if status not in {"champion", "challenger"}:
        return
    existing = conn.execute(
        """
        SELECT model_version
        FROM model_registry
        WHERE model_family = ?
          AND status = ?
          AND retired_at IS NULL
          AND model_version <> ?
        """,
        [model_family, status, model_version],
    ).fetchone()
    if existing:
        raise ValueError(
            f"model_family {model_family!r} already has active {status}: {existing[0]}"
        )


def _fetch_model(
    conn: duckdb.DuckDBPyConnection,
    model_version: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM model_registry WHERE model_version = ?",
        [model_version],
    ).fetchone()
    if row is None:
        return None
    columns = [column[0] for column in conn.description]
    return dict(zip(columns, row, strict=True))


def _validate_json(field_name: str, value: str) -> None:
    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc


def _connect_retry_attempts(value: int | None) -> int:
    if value is None:
        value = _int_env("BIGAN_MLOPS_CONNECT_RETRIES", DEFAULT_MLOPS_CONNECT_RETRIES)
    return max(1, int(value))


def _connect_retry_delay(value: float | None) -> float:
    if value is None:
        value = _float_env(
            "BIGAN_MLOPS_CONNECT_RETRY_DELAY_SECONDS",
            DEFAULT_MLOPS_CONNECT_RETRY_DELAY_SECONDS,
        )
    return max(0.0, float(value))


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _is_duckdb_lock_error(exc: duckdb.Error) -> bool:
    text = str(exc)
    return "Could not set lock" in text or "Conflicting lock" in text


def _require_non_empty(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} is required")


def _now_ms() -> int:
    return int(time.time() * 1000)
