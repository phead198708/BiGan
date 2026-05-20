"""Prediction event, outcome, and daily monitoring tables (issue #41)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb

from .incidents import DATA_QUALITY_INCIDENTS_DDL

PREDICTION_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS prediction_events (
    event_id VARCHAR PRIMARY KEY,
    ts BIGINT NOT NULL,
    model_version VARCHAR NOT NULL,
    feature_version VARCHAR NOT NULL,
    prob_up_15m DOUBLE NOT NULL CHECK (prob_up_15m >= 0 AND prob_up_15m <= 1),
    confidence_bucket VARCHAR NOT NULL,
    top_features_json VARCHAR NOT NULL,
    feature_hash VARCHAR NOT NULL,
    feature_snapshot_json VARCHAR NOT NULL,
    serving_latency_ms DOUBLE NOT NULL CHECK (serving_latency_ms >= 0),
    created_at BIGINT NOT NULL
)
"""

PREDICTION_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    event_id VARCHAR PRIMARY KEY,
    target_ts BIGINT NOT NULL,
    realized_label BOOLEAN NOT NULL,
    realized_return DOUBLE,
    brier_component DOUBLE NOT NULL CHECK (brier_component >= 0),
    outcome_ts BIGINT NOT NULL,
    created_at BIGINT NOT NULL
)
"""

MODEL_MONITORING_DAILY_DDL = """
CREATE TABLE IF NOT EXISTS model_monitoring_daily (
    "date" DATE NOT NULL,
    model_version VARCHAR NOT NULL,
    prediction_count BIGINT NOT NULL,
    avg_prob DOUBLE,
    hit_rate DOUBLE,
    brier_score DOUBLE,
    ece DOUBLE,
    drift_metrics_json VARCHAR,
    created_at BIGINT NOT NULL,
    PRIMARY KEY ("date", model_version)
)
"""

MONITORING_TABLES_DDL: tuple[str, ...] = (
    PREDICTION_EVENTS_DDL,
    PREDICTION_OUTCOMES_DDL,
    MODEL_MONITORING_DAILY_DDL,
    DATA_QUALITY_INCIDENTS_DDL,
)


@dataclass(frozen=True, slots=True)
class PredictionEvent:
    """One online prediction request/response event."""

    event_id: str
    ts: int
    model_version: str
    feature_version: str
    prob_up_15m: float
    confidence_bucket: str
    top_features_json: str
    feature_hash: str
    feature_snapshot_json: str
    serving_latency_ms: float
    created_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["created_at"] = self.created_at or _now_ms()
        return row


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """Observed label/return for a prediction event after the target horizon."""

    event_id: str
    target_ts: int
    realized_label: bool
    realized_return: float | None
    brier_component: float
    outcome_ts: int
    created_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["created_at"] = self.created_at or _now_ms()
        return row


@dataclass(frozen=True, slots=True)
class MonitoringDailyRow:
    """Daily model monitoring aggregate."""

    date: str
    model_version: str
    prediction_count: int
    avg_prob: float | None
    hit_rate: float | None
    brier_score: float | None
    ece: float | None
    drift_metrics_json: str | None
    created_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initialize_monitoring_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create monitoring tables."""

    for ddl in MONITORING_TABLES_DDL:
        conn.execute(ddl)


def record_prediction_event(
    conn: duckdb.DuckDBPyConnection,
    event: PredictionEvent,
    *,
    replace: bool = False,
) -> None:
    """Insert one prediction event."""

    initialize_monitoring_tables(conn)
    _validate_prediction_event(event)
    row = event.to_row()
    _insert_row(conn, "prediction_events", row, replace=replace)


def record_prediction_outcome(
    conn: duckdb.DuckDBPyConnection,
    outcome: PredictionOutcome,
    *,
    replace: bool = False,
) -> None:
    """Insert one prediction outcome row."""

    initialize_monitoring_tables(conn)
    _validate_prediction_outcome(outcome)
    row = outcome.to_row()
    _insert_row(conn, "prediction_outcomes", row, replace=replace)


def summarize_model_monitoring_daily(
    conn: duckdb.DuckDBPyConnection,
    *,
    date: str,
    model_version: str,
    drift_metrics_json: str | None = None,
    bins: int = 10,
) -> MonitoringDailyRow:
    """Compute and upsert one model/day monitoring row."""

    initialize_monitoring_tables(conn)
    if drift_metrics_json is not None:
        _validate_json("drift_metrics_json", drift_metrics_json)
    start_ms = _date_to_epoch_ms(date)
    end_ms = start_ms + 24 * 60 * 60 * 1000
    rows = conn.execute(
        """
        SELECT e.prob_up_15m, o.realized_label, o.brier_component
        FROM prediction_events e
        LEFT JOIN prediction_outcomes o USING (event_id)
        WHERE e.model_version = ?
          AND e.ts >= ?
          AND e.ts < ?
        ORDER BY e.ts, e.event_id
        """,
        [model_version, start_ms, end_ms],
    ).fetchall()
    prediction_count = len(rows)
    probabilities = [float(row[0]) for row in rows]
    labeled = [
        (float(row[0]), bool(row[1]), float(row[2]))
        for row in rows
        if row[1] is not None and row[2] is not None
    ]
    avg_prob = _mean(probabilities)
    hit_rate = (
        None
        if not labeled
        else sum(1 for prob, label, _ in labeled if (prob >= 0.5) == label) / len(labeled)
    )
    brier_score = None if not labeled else sum(item[2] for item in labeled) / len(labeled)
    ece = (
        None
        if not labeled
        else _expected_calibration_error(
            [item[0] for item in labeled],
            [item[1] for item in labeled],
            bins=bins,
        )
    )
    row = MonitoringDailyRow(
        date=date,
        model_version=model_version,
        prediction_count=prediction_count,
        avg_prob=avg_prob,
        hit_rate=hit_rate,
        brier_score=brier_score,
        ece=ece,
        drift_metrics_json=drift_metrics_json,
        created_at=_now_ms(),
    )
    conn.execute(
        """
        DELETE FROM model_monitoring_daily
        WHERE "date" = CAST(? AS DATE)
          AND model_version = ?
        """,
        [date, model_version],
    )
    conn.execute(
        """
        INSERT INTO model_monitoring_daily (
            "date", model_version, prediction_count, avg_prob, hit_rate,
            brier_score, ece, drift_metrics_json, created_at
        ) VALUES (CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            row.date,
            row.model_version,
            row.prediction_count,
            row.avg_prob,
            row.hit_rate,
            row.brier_score,
            row.ece,
            row.drift_metrics_json,
            row.created_at,
        ],
    )
    return row


def compute_brier_component(prob_up_15m: float, realized_label: bool) -> float:
    """Return the per-event Brier contribution."""

    if prob_up_15m < 0.0 or prob_up_15m > 1.0:
        raise ValueError("prob_up_15m must be in [0, 1]")
    target = 1.0 if realized_label else 0.0
    return (prob_up_15m - target) ** 2


def _validate_prediction_event(event: PredictionEvent) -> None:
    for name in (
        "event_id",
        "model_version",
        "feature_version",
        "confidence_bucket",
        "feature_hash",
    ):
        _require_non_empty(name, str(getattr(event, name)))
    if event.ts < 0:
        raise ValueError("ts must be non-negative")
    if event.prob_up_15m < 0.0 or event.prob_up_15m > 1.0:
        raise ValueError("prob_up_15m must be in [0, 1]")
    if event.serving_latency_ms < 0:
        raise ValueError("serving_latency_ms must be non-negative")
    _validate_json("top_features_json", event.top_features_json)
    _validate_json("feature_snapshot_json", event.feature_snapshot_json)


def _validate_prediction_outcome(outcome: PredictionOutcome) -> None:
    _require_non_empty("event_id", outcome.event_id)
    if outcome.target_ts < 0 or outcome.outcome_ts < 0:
        raise ValueError("outcome timestamps must be non-negative")
    if outcome.brier_component < 0 or not math.isfinite(outcome.brier_component):
        raise ValueError("brier_component must be finite and non-negative")


def _insert_row(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    row: dict[str, Any],
    *,
    replace: bool,
) -> None:
    columns = tuple(row)
    placeholders = ", ".join("?" for _ in columns)
    statement = (
        f"INSERT {'OR REPLACE ' if replace else ''}INTO {table_name} "
        f"({', '.join(columns)}) VALUES ({placeholders})"
    )
    conn.execute(statement, [row[column] for column in columns])


def _expected_calibration_error(
    probabilities: list[float],
    labels: list[bool],
    *,
    bins: int,
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    total = len(probabilities)
    if total == 0:
        return 0.0
    ece = 0.0
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        bucket = [
            (prob, label)
            for prob, label in zip(probabilities, labels, strict=True)
            if (prob >= lower and (prob < upper or (idx == bins - 1 and prob <= upper)))
        ]
        if not bucket:
            continue
        avg_prob = sum(prob for prob, _ in bucket) / len(bucket)
        avg_label = sum(1.0 if label else 0.0 for _, label in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_prob - avg_label)
    return ece


def _date_to_epoch_ms(value: str) -> int:
    dt = datetime.fromisoformat(value).replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _validate_json(field_name: str, value: str) -> None:
    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc


def _require_non_empty(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} is required")


def _now_ms() -> int:
    return int(time.time() * 1000)
