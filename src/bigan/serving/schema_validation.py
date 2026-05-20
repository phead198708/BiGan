"""Fail-closed online feature schema validation (issue #44)."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb

from bigan.monitoring import DataQualityIncident, record_data_quality_incident


class FeatureSchemaMismatch(ValueError):
    """Raised when online features do not match the training schema."""

    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__("input feature schema does not match training schema")
        self.details = details


@dataclass(frozen=True, slots=True)
class FeatureSchemaArtifact:
    """Feature order and type contract saved with a trained model."""

    feature_columns: tuple[str, ...]
    feature_types: dict[str, str]
    schema_hash: str
    feature_version: str | None = None
    dataset_version: str | None = None
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "feature_types": dict(self.feature_types),
            "schema_hash": self.schema_hash,
            "feature_version": self.feature_version,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
        }


def build_feature_schema_artifact(
    feature_columns: Sequence[str],
    *,
    feature_types: Mapping[str, str] | None = None,
    feature_version: str | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
) -> FeatureSchemaArtifact:
    """Build a deterministic training feature schema artifact."""

    columns = tuple(str(column) for column in feature_columns)
    if not columns:
        raise ValueError("feature_columns must not be empty")
    if len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be unique")
    types = {column: "float64" for column in columns}
    if feature_types is not None:
        types.update({str(key): str(value) for key, value in feature_types.items()})
    unknown_types = sorted(set(types) - set(columns))
    if unknown_types:
        raise ValueError(f"feature_types contain unknown columns: {unknown_types}")
    payload = {
        "feature_columns": list(columns),
        "feature_types": {column: types[column] for column in columns},
        "feature_version": feature_version,
        "dataset_version": dataset_version,
        "model_version": model_version,
    }
    schema_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FeatureSchemaArtifact(
        feature_columns=columns,
        feature_types={column: types[column] for column in columns},
        schema_hash=schema_hash,
        feature_version=feature_version,
        dataset_version=dataset_version,
        model_version=model_version,
    )


def write_feature_schema_artifact(
    path: Path | str,
    feature_columns: Sequence[str],
    *,
    feature_types: Mapping[str, str] | None = None,
    feature_version: str | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
) -> FeatureSchemaArtifact:
    """Write ``feature_schema.json`` for serving-time validation."""

    artifact = build_feature_schema_artifact(
        feature_columns,
        feature_types=feature_types,
        feature_version=feature_version,
        dataset_version=dataset_version,
        model_version=model_version,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def load_feature_schema_artifact(path: Path | str) -> FeatureSchemaArtifact:
    """Load a saved feature schema artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = tuple(str(column) for column in data["feature_columns"])
    types = {str(key): str(value) for key, value in data["feature_types"].items()}
    expected = build_feature_schema_artifact(
        columns,
        feature_types=types,
        feature_version=data.get("feature_version"),
        dataset_version=data.get("dataset_version"),
        model_version=data.get("model_version"),
    )
    if data.get("schema_hash") != expected.schema_hash:
        raise ValueError("feature schema artifact hash mismatch")
    return expected


def validate_features_fail_closed(
    features: Mapping[str, Any],
    schema: FeatureSchemaArtifact,
    *,
    strict_order: bool = True,
    incident_conn: duckdb.DuckDBPyConnection | None = None,
    source: str = "serving",
    affected_symbol: str | None = None,
    request_id: str | None = None,
) -> dict[str, float]:
    """Validate online features against training schema or raise and log incident."""

    details = _schema_mismatch_details(features, schema, strict_order=strict_order)
    if details:
        if incident_conn is not None:
            _record_schema_incident(
                incident_conn,
                details=details,
                source=source,
                affected_symbol=affected_symbol,
                request_id=request_id,
            )
        raise FeatureSchemaMismatch(details)
    return {column: float(features[column]) for column in schema.feature_columns}


def _schema_mismatch_details(
    features: Mapping[str, Any],
    schema: FeatureSchemaArtifact,
    *,
    strict_order: bool,
) -> dict[str, Any]:
    expected = list(schema.feature_columns)
    actual = [str(column) for column in features.keys()]
    missing = [column for column in expected if column not in features]
    extra = [column for column in actual if column not in schema.feature_columns]
    wrong_order = strict_order and not missing and not extra and actual != expected
    type_errors = {
        column: type(features[column]).__name__
        for column in expected
        if column in features and not _is_numeric(features[column])
    }
    details: dict[str, Any] = {}
    if missing:
        details["missing"] = missing
    if extra:
        details["extra"] = extra
    if wrong_order:
        details["expected_order"] = expected
        details["actual_order"] = actual
    if type_errors:
        details["type_errors"] = type_errors
    if details:
        details["schema_hash"] = schema.schema_hash
        details["model_version"] = schema.model_version
        details["feature_version"] = schema.feature_version
    return details


def _record_schema_incident(
    conn: duckdb.DuckDBPyConnection,
    *,
    details: dict[str, Any],
    source: str,
    affected_symbol: str | None,
    request_id: str | None,
) -> None:
    incident = DataQualityIncident(
        incident_id=f"schema-mismatch-{int(time.time() * 1000)}-{uuid4().hex[:8]}",
        source=source,
        incident_type="schema_change",
        severity="critical",
        started_at=int(time.time() * 1000),
        affected_symbol=affected_symbol,
        details_json=json.dumps(
            {**details, "request_id": request_id},
            sort_keys=True,
        ),
        alert_id=request_id,
        owner="ml-serving",
    )
    record_data_quality_incident(conn, incident)


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))
