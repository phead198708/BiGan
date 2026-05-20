"""Posterior evaluation and challenger trigger workflow (issue #48)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import duckdb

from .registry import ModelRegistryRecord, register_model

CHALLENGER_TRIGGER_STATUSES: tuple[str, ...] = (
    "pending",
    "started",
    "completed",
    "skipped",
)

CHALLENGER_TRIGGERS_DDL = """
CREATE TABLE IF NOT EXISTS challenger_triggers (
    trigger_id VARCHAR PRIMARY KEY,
    "date" DATE NOT NULL,
    model_version VARCHAR NOT NULL,
    triggered BOOLEAN NOT NULL,
    reasons_json VARCHAR NOT NULL,
    hit_rate DOUBLE,
    brier_score DOUBLE,
    ece DOUBLE,
    drift_severity VARCHAR,
    retraining_job_uri VARCHAR,
    challenger_model_version VARCHAR,
    status VARCHAR NOT NULL CHECK (status IN ('pending', 'started', 'completed', 'skipped')),
    created_at BIGINT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class RetrainingTriggerRules:
    """Thresholds used to trigger challenger training."""

    min_hit_rate: float = 0.50
    max_brier_score: float = 0.25
    max_ece: float = 0.10
    trigger_on_critical_drift: bool = True

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChallengerTriggerDecision:
    """Decision result for one model/day posterior evaluation."""

    date: str
    model_version: str
    triggered: bool
    reasons: tuple[str, ...]
    hit_rate: float | None
    brier_score: float | None
    ece: float | None
    drift_severity: str | None
    rules: RetrainingTriggerRules

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "model_version": self.model_version,
            "triggered": self.triggered,
            "reasons": list(self.reasons),
            "hit_rate": self.hit_rate,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "drift_severity": self.drift_severity,
            "rules": self.rules.to_dict(),
        }


def evaluate_challenger_trigger(
    conn: duckdb.DuckDBPyConnection,
    *,
    date: str,
    model_version: str,
    rules: RetrainingTriggerRules | None = None,
) -> ChallengerTriggerDecision:
    """Evaluate daily monitoring and drift rows for challenger trigger need."""

    active_rules = rules or RetrainingTriggerRules()
    _initialize_trigger_tables(conn)
    monitoring = conn.execute(
        """
        SELECT hit_rate, brier_score, ece
        FROM model_monitoring_daily
        WHERE "date" = CAST(? AS DATE)
          AND model_version = ?
        """,
        [date, model_version],
    ).fetchone()
    if monitoring is None:
        return ChallengerTriggerDecision(
            date=date,
            model_version=model_version,
            triggered=False,
            reasons=("monitoring_daily_missing",),
            hit_rate=None,
            brier_score=None,
            ece=None,
            drift_severity=None,
            rules=active_rules,
        )
    hit_rate, brier_score, ece = (
        _optional_float(monitoring[0]),
        _optional_float(monitoring[1]),
        _optional_float(monitoring[2]),
    )
    drift_severity = _max_drift_severity(conn, date=date, model_version=model_version)
    reasons: list[str] = []
    if hit_rate is not None and hit_rate < active_rules.min_hit_rate:
        reasons.append("hit_rate_below_threshold")
    if brier_score is not None and brier_score > active_rules.max_brier_score:
        reasons.append("brier_score_above_threshold")
    if ece is not None and ece > active_rules.max_ece:
        reasons.append("ece_above_threshold")
    if active_rules.trigger_on_critical_drift and drift_severity == "critical":
        reasons.append("critical_feature_drift")
    return ChallengerTriggerDecision(
        date=date,
        model_version=model_version,
        triggered=bool(reasons),
        reasons=tuple(reasons),
        hit_rate=hit_rate,
        brier_score=brier_score,
        ece=ece,
        drift_severity=drift_severity,
        rules=active_rules,
    )


def record_challenger_trigger(
    conn: duckdb.DuckDBPyConnection,
    decision: ChallengerTriggerDecision,
    *,
    retraining_job_uri: str | None = None,
    challenger_model_version: str | None = None,
    status: str | None = None,
) -> str:
    """Persist one challenger trigger decision and return trigger_id."""

    _initialize_trigger_tables(conn)
    trigger_status = status or ("pending" if decision.triggered else "skipped")
    if trigger_status not in CHALLENGER_TRIGGER_STATUSES:
        raise ValueError(f"invalid challenger trigger status: {trigger_status!r}")
    trigger_id = f"{decision.date}:{decision.model_version}"
    conn.execute("DELETE FROM challenger_triggers WHERE trigger_id = ?", [trigger_id])
    conn.execute(
        """
        INSERT INTO challenger_triggers (
            trigger_id, "date", model_version, triggered, reasons_json,
            hit_rate, brier_score, ece, drift_severity, retraining_job_uri,
            challenger_model_version, status, created_at
        ) VALUES (?, CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            trigger_id,
            decision.date,
            decision.model_version,
            decision.triggered,
            json.dumps(list(decision.reasons), sort_keys=True),
            decision.hit_rate,
            decision.brier_score,
            decision.ece,
            decision.drift_severity,
            retraining_job_uri,
            challenger_model_version,
            trigger_status,
            _now_ms(),
        ],
    )
    return trigger_id


def register_training_result_as_challenger(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    model_family: str,
    feature_version: str,
    dataset_version: str,
    train_config_hash: str,
    artifact_uri: str,
    train_started_at: int,
    train_finished_at: int,
    metrics_json: str,
    calibration_artifact_uri: str | None = None,
    backtest_json: str | None = None,
    notes: str | None = None,
) -> None:
    """Register a completed retraining result as the active challenger."""

    register_model(
        conn,
        ModelRegistryRecord(
            model_version=model_version,
            model_family=model_family,
            feature_version=feature_version,
            dataset_version=dataset_version,
            train_config_hash=train_config_hash,
            artifact_uri=artifact_uri,
            calibration_artifact_uri=calibration_artifact_uri,
            status="challenger",
            train_started_at=train_started_at,
            train_finished_at=train_finished_at,
            metrics_json=metrics_json,
            backtest_json=backtest_json,
            notes=notes,
        ),
    )


def _initialize_trigger_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(CHALLENGER_TRIGGERS_DDL)


def _max_drift_severity(
    conn: duckdb.DuckDBPyConnection,
    *,
    date: str,
    model_version: str,
) -> str | None:
    try:
        rows = conn.execute(
            """
            SELECT severity
            FROM drift_metrics
            WHERE "date" = CAST(? AS DATE)
              AND model_version = ?
            """,
            [date, model_version],
        ).fetchall()
    except duckdb.CatalogException:
        return None
    severities = {str(row[0]) for row in rows}
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    if "ok" in severities:
        return "ok"
    return None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _now_ms() -> int:
    return int(time.time() * 1000)
