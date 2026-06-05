"""Local serving readiness checks for model promotion evidence."""

from __future__ import annotations

import json
import math
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .schema_validation import (
    FeatureSchemaArtifact,
    FeatureSchemaMismatch,
    load_feature_schema_artifact,
    validate_features_fail_closed,
)


def run_xgboost_serving_readiness(
    *,
    model_path: Path | str,
    feature_schema_path: Path | str,
    dataset_dir: Path | str,
    output_path: Path | str | None = None,
    split: str = "test",
    sample_size: int = 1_000,
    batch_sizes: Sequence[int] = (10_000, 100_000),
    latency_sla_ms: float = 50.0,
    max_error_rate: float = 0.0,
    fallback_model_path: Path | str | None = None,
    rollback_runbook_path: Path | str | None = None,
) -> dict[str, Any]:
    """Benchmark local XGBoost inference and schema-validation behavior.

    This is intentionally framework-neutral. It verifies the model artifact,
    training feature schema, fail-closed invalid-input path, and local inference
    latency/throughput before bootstrap promotion review.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not batch_sizes:
        raise ValueError("batch_sizes must not be empty")
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes values must be positive")

    model = _load_serving_model(model_path)
    schema = load_feature_schema_artifact(feature_schema_path)
    rows = _load_feature_rows(Path(dataset_dir), split=split, schema=schema)
    single_rows = _repeat_rows(rows, sample_size)

    schema_result = _check_schema_validation(schema, rows[0])
    latency = _benchmark_single_latency(model, schema, single_rows)
    throughput = [
        _benchmark_batch(model, schema, rows, batch_size=batch_size)
        for batch_size in batch_sizes
    ]
    fallback = _fallback_readiness(fallback_model_path, rollback_runbook_path)

    latency_ok = latency["p95_latency_ms"] <= latency_sla_ms
    error_ok = latency["error_rate"] <= max_error_rate
    schema_ok = (
        schema_result["valid_input_accepted"]
        and schema_result["invalid_input_rejected"]
        and not schema_result["silent_failure"]
    )
    fallback_ok = fallback["fallback_model_available"] and fallback["rollback_runbook_available"]
    ready = bool(latency_ok and error_ok and schema_ok and fallback_ok)
    report = {
        "schema_version": "serving_readiness_v1",
        "model_version": model.model_version,
        "status": "ok" if ready else "failed",
        "ready": ready,
        "serving_ready": ready,
        "generated_at_ms": int(time.time() * 1_000),
        "model_path": str(model_path),
        "feature_schema_path": str(feature_schema_path),
        "dataset_dir": str(dataset_dir),
        "split": split,
        "valid_feature_row_count": len(rows),
        "single_inference_sample_count": sample_size,
        "p50_latency_ms": latency["p50_latency_ms"],
        "p95_latency_ms": latency["p95_latency_ms"],
        "latency_p95_ms": latency["p95_latency_ms"],
        "p99_latency_ms": latency["p99_latency_ms"],
        "latency_sla_ms": latency_sla_ms,
        "error_count": latency["error_count"],
        "error_rate": latency["error_rate"],
        "max_error_rate": max_error_rate,
        "schema_validation": schema_result,
        "fallback": fallback,
        "batch_throughput": throughput,
        "environment": _environment(),
        "conclusion": (
            "local serving path passed latency, error, schema, and fallback checks"
            if ready
            else "local serving path is missing required readiness evidence"
        ),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_feature_rows(
    dataset_dir: Path,
    *,
    split: str,
    schema: FeatureSchemaArtifact,
) -> list[dict[str, float]]:
    path = dataset_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"dataset split not found: {path}")
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"dataset split has no rows: {path}")
    feature_rows = [
        feature_row
        for row in rows
        if (feature_row := _finite_feature_row(row, schema)) is not None
    ]
    if not feature_rows:
        raise ValueError(f"dataset split has no finite feature rows for schema: {path}")
    return feature_rows


def _finite_feature_row(
    row: dict[str, Any],
    schema: FeatureSchemaArtifact,
) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for column in schema.feature_columns:
        value = row.get(column)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        as_float = float(value)
        if not math.isfinite(as_float):
            return None
        values[column] = as_float
    return values


def _check_schema_validation(
    schema: FeatureSchemaArtifact,
    row: dict[str, Any],
) -> dict[str, Any]:
    valid_input_accepted = False
    invalid_input_rejected = False
    error_details: dict[str, Any] = {}
    try:
        validate_features_fail_closed(row, schema)
        valid_input_accepted = True
    except FeatureSchemaMismatch as exc:
        error_details["valid_input_error"] = exc.details

    invalid = dict(row)
    invalid.pop(schema.feature_columns[0])
    try:
        validate_features_fail_closed(invalid, schema)
    except FeatureSchemaMismatch as exc:
        invalid_input_rejected = True
        error_details["invalid_input_error"] = exc.details

    return {
        "valid_input_accepted": valid_input_accepted,
        "invalid_input_rejected": invalid_input_rejected,
        "silent_failure": valid_input_accepted and not invalid_input_rejected,
        "error_details": error_details,
    }


def _benchmark_single_latency(
    model: Any,
    schema: FeatureSchemaArtifact,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    error_count = 0
    for row in rows:
        start = time.perf_counter_ns()
        try:
            validated = validate_features_fail_closed(row, schema)
            _validate_prediction_output(model.predict_one(validated))
        except Exception:
            error_count += 1
        finally:
            latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "p50_latency_ms": _percentile(latencies_ms, 50),
        "p95_latency_ms": _percentile(latencies_ms, 95),
        "p99_latency_ms": _percentile(latencies_ms, 99),
        "error_count": error_count,
        "error_rate": error_count / len(rows),
    }


def _benchmark_batch(
    model: Any,
    schema: FeatureSchemaArtifact,
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    batch = _repeat_rows(rows, batch_size)
    start = time.perf_counter_ns()
    validated = [validate_features_fail_closed(row, schema) for row in batch]
    probabilities = model.predict_many(validated)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    if len(probabilities) != batch_size:
        raise ValueError("batch inference returned unexpected row count")
    for probability in probabilities:
        _validate_prediction_output(probability)
    return {
        "batch_size": batch_size,
        "elapsed_ms": elapsed_ms,
        "rows_per_second": batch_size / (elapsed_ms / 1_000.0),
    }


def _repeat_rows(rows: Sequence[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    return [dict(rows[idx % len(rows)]) for idx in range(size)]


@dataclass(frozen=True, slots=True)
class _ServingModel:
    raw_model: Any
    model_version: str
    payload_mode: str

    def predict_one(self, row: dict[str, Any]) -> Any:
        if self.payload_mode == "v6_payload":
            return self.raw_model.predict_payload(row)
        return self.raw_model.predict_proba(row)

    def predict_many(self, rows: list[dict[str, Any]]) -> list[Any]:
        if self.payload_mode == "v6_payload":
            return list(self.raw_model.predict_payload_many(rows))
        return list(self.raw_model.predict_proba_many(rows))


def _load_serving_model(model_path: Path | str) -> _ServingModel:
    path = Path(model_path)
    artifact: dict[str, Any] = {}
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        artifact = {}
    if artifact.get("schema_version") == "xgboost_v6_multihead_v1":
        from bigan.modeling import load_xgboost_v6_model

        model = load_xgboost_v6_model(path)
        return _ServingModel(
            raw_model=model,
            model_version=model.model_version,
            payload_mode="v6_payload",
        )

    from bigan.modeling import load_xgboost_v1_model

    model = load_xgboost_v1_model(path)
    return _ServingModel(
        raw_model=model,
        model_version=model.model_version,
        payload_mode="probability",
    )


def _validate_prediction_output(output: Any) -> None:
    if isinstance(output, int | float) and not isinstance(output, bool):
        _validate_probability(float(output), name="probability")
        return
    if not isinstance(output, dict):
        raise ValueError("model returned unsupported prediction payload")

    required = ("p_up", "p_down", "p_neutral", "p_vol_up", "p_vol_down")
    missing = [name for name in required if name not in output]
    if missing:
        raise ValueError(f"model payload missing probability fields: {missing}")
    for name in required:
        _validate_probability(float(output[name]), name=name)
    settlement_sum = (
        float(output["p_up"]) + float(output["p_down"]) + float(output["p_neutral"])
    )
    if not math.isclose(settlement_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("settlement probabilities do not sum to 1")


def _validate_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"model returned {name} outside [0, 1]")


def _fallback_readiness(
    fallback_model_path: Path | str | None,
    rollback_runbook_path: Path | str | None,
) -> dict[str, Any]:
    return {
        "fallback_model_path": None if fallback_model_path is None else str(fallback_model_path),
        "fallback_model_available": (
            fallback_model_path is not None and Path(fallback_model_path).exists()
        ),
        "rollback_runbook_path": None if rollback_runbook_path is None else str(rollback_runbook_path),
        "rollback_runbook_available": (
            rollback_runbook_path is not None and Path(rollback_runbook_path).exists()
        ),
    }


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile for empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
