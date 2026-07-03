"""Data quality and schema incident table helpers (issue #42)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import duckdb

INCIDENT_TYPES: tuple[str, ...] = (
    "data_missing",
    "schema_change",
    "latency_anomaly",
    "stream_gap",
    "decode_error",
    "quality_rule_failure",
    "prediction_drift",
    "label_shift",
)

INCIDENT_SEVERITIES: tuple[str, ...] = ("info", "warning", "critical")

DATA_QUALITY_INCIDENTS_DDL = """
CREATE TABLE IF NOT EXISTS data_quality_incidents (
    incident_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    incident_type VARCHAR NOT NULL CHECK (
        incident_type IN (
            'data_missing', 'schema_change', 'latency_anomaly',
            'stream_gap', 'decode_error', 'quality_rule_failure',
            'prediction_drift', 'label_shift'
        )
    ),
    severity VARCHAR NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    started_at BIGINT NOT NULL,
    resolved_at BIGINT,
    affected_symbol VARCHAR,
    details_json VARCHAR NOT NULL,
    alert_id VARCHAR,
    owner VARCHAR,
    resolution_note VARCHAR,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class DataQualityIncident:
    """One data quality, latency, stream, or schema incident."""

    incident_id: str
    source: str
    incident_type: str
    severity: str
    started_at: int
    affected_symbol: str | None
    details_json: str
    alert_id: str | None = None
    owner: str | None = None
    resolved_at: int | None = None
    resolution_note: str | None = None
    created_at: int | None = None
    updated_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        now_ms = _now_ms()
        row["created_at"] = self.created_at or now_ms
        row["updated_at"] = self.updated_at or now_ms
        return row


def record_data_quality_incident(
    conn: duckdb.DuckDBPyConnection,
    incident: DataQualityIncident,
    *,
    replace: bool = False,
) -> None:
    """Insert one data/schema incident event."""

    initialize_incident_tables(conn)
    _validate_incident(incident)
    row = incident.to_row()
    columns = tuple(row)
    placeholders = ", ".join("?" for _ in columns)
    statement = (
        f"INSERT {'OR REPLACE ' if replace else ''}INTO data_quality_incidents "
        f"({', '.join(columns)}) VALUES ({placeholders})"
    )
    conn.execute(statement, [row[column] for column in columns])


def resolve_data_quality_incident(
    conn: duckdb.DuckDBPyConnection,
    incident_id: str,
    *,
    resolved_at: int | None = None,
    owner: str | None = None,
    resolution_note: str | None = None,
) -> None:
    """Resolve an open data/schema incident."""

    initialize_incident_tables(conn)
    row = conn.execute(
        "SELECT started_at FROM data_quality_incidents WHERE incident_id = ?",
        [incident_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown incident_id: {incident_id}")
    ts = resolved_at or _now_ms()
    if ts < int(row[0]):
        raise ValueError("resolved_at must be >= started_at")
    conn.execute(
        """
        UPDATE data_quality_incidents
        SET resolved_at = ?,
            owner = coalesce(?, owner),
            resolution_note = coalesce(?, resolution_note),
            updated_at = ?
        WHERE incident_id = ?
        """,
        [ts, owner, resolution_note, ts, incident_id],
    )


def open_data_quality_incidents(
    conn: duckdb.DuckDBPyConnection,
    *,
    severity: str | None = None,
) -> list[dict[str, Any]]:
    """Return unresolved incidents, optionally filtered by severity."""

    initialize_incident_tables(conn)
    params: list[str] = []
    predicate = "resolved_at IS NULL"
    if severity is not None:
        if severity not in INCIDENT_SEVERITIES:
            raise ValueError(f"invalid incident severity: {severity!r}")
        predicate += " AND severity = ?"
        params.append(severity)
    rows = conn.execute(
        f"""
        SELECT *
        FROM data_quality_incidents
        WHERE {predicate}
        ORDER BY started_at DESC, incident_id
        """,
        params,
    ).fetchall()
    columns = [column[0] for column in conn.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def initialize_incident_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create data quality incident table."""

    conn.execute(DATA_QUALITY_INCIDENTS_DDL)


def _validate_incident(incident: DataQualityIncident) -> None:
    for name in ("incident_id", "source"):
        _require_non_empty(name, str(getattr(incident, name)))
    if incident.incident_type not in INCIDENT_TYPES:
        raise ValueError(f"invalid incident type: {incident.incident_type!r}")
    if incident.severity not in INCIDENT_SEVERITIES:
        raise ValueError(f"invalid incident severity: {incident.severity!r}")
    if incident.started_at < 0:
        raise ValueError("started_at must be non-negative")
    if incident.resolved_at is not None and incident.resolved_at < incident.started_at:
        raise ValueError("resolved_at must be >= started_at")
    _validate_json("details_json", incident.details_json)


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
