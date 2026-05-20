"""Data quality incident table tests for issue #42."""

from __future__ import annotations

import json

import duckdb
import pytest

from bigan.mlops import connect_mlops_db, initialize_mlops_db
from bigan.monitoring import (
    DataQualityIncident,
    INCIDENT_SEVERITIES,
    INCIDENT_TYPES,
    open_data_quality_incidents,
    record_data_quality_incident,
    resolve_data_quality_incident,
)


def _incident(incident_id: str, *, severity: str = "warning") -> DataQualityIncident:
    return DataQualityIncident(
        incident_id=incident_id,
        source="polymarket",
        incident_type="schema_change",
        severity=severity,
        started_at=1_000,
        affected_symbol="token-1",
        details_json=json.dumps({"missing_columns": ["bid_price"]}),
        alert_id="pager-123",
        owner="data-oncall",
    )


def test_data_quality_incidents_table_is_created_with_expected_columns() -> None:
    conn = connect_mlops_db()
    initialize_mlops_db(conn)

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info('data_quality_incidents')").fetchall()
    }
    assert {
        "incident_id",
        "source",
        "incident_type",
        "severity",
        "started_at",
        "resolved_at",
        "affected_symbol",
        "details_json",
        "alert_id",
        "owner",
        "resolution_note",
    } <= columns

    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            """
            INSERT INTO data_quality_incidents (
                incident_id, source, incident_type, severity, started_at,
                details_json, created_at, updated_at
            ) VALUES ('bad', 'polymarket', 'schema_change', 'page', 1, '{}', 1, 1)
            """
        )


def test_incident_open_and_resolve_lifecycle() -> None:
    conn = connect_mlops_db()
    record_data_quality_incident(conn, _incident("inc-1"))
    record_data_quality_incident(conn, _incident("inc-2", severity="critical"))

    critical = open_data_quality_incidents(conn, severity="critical")
    assert [row["incident_id"] for row in critical] == ["inc-2"]

    resolve_data_quality_incident(
        conn,
        "inc-2",
        resolved_at=2_000,
        owner="ml-oncall",
        resolution_note="schema parser updated",
    )
    assert [row["incident_id"] for row in open_data_quality_incidents(conn)] == ["inc-1"]
    resolved = conn.execute(
        """
        SELECT resolved_at, owner, resolution_note
        FROM data_quality_incidents
        WHERE incident_id = 'inc-2'
        """
    ).fetchone()
    assert resolved == (2_000, "ml-oncall", "schema parser updated")


def test_incident_validation_rejects_unknown_type_and_bad_json() -> None:
    conn = connect_mlops_db()
    with pytest.raises(ValueError, match="invalid incident type"):
        record_data_quality_incident(
            conn,
            DataQualityIncident(
                incident_id="bad-type",
                source="polymarket",
                incident_type="surprise",
                severity="warning",
                started_at=1,
                affected_symbol=None,
                details_json="{}",
            ),
        )

    bad_json = _incident("bad-json")
    bad_json = DataQualityIncident(
        **{**bad_json.to_row(), "created_at": None, "updated_at": None, "details_json": "{"}
    )
    with pytest.raises(ValueError, match="details_json"):
        record_data_quality_incident(conn, bad_json)


def test_incident_contract_lists_supported_types_and_severities() -> None:
    assert {"data_missing", "schema_change", "latency_anomaly", "stream_gap"} <= set(
        INCIDENT_TYPES
    )
    assert set(INCIDENT_SEVERITIES) == {"info", "warning", "critical"}
