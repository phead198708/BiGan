"""Evaluate saved probability models on a fixed training dataset split."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .calibration import load_probability_calibrator
from .logistic import (
    _labels,
    _load_dataset,
    _metrics,
    _metrics_by_market_family,
    _validate_feature_columns,
    load_logistic_baseline,
)
from .xgboost_v1 import SPLITS, load_xgboost_v1_model


class ProbabilityModel(Protocol):
    model_version: str
    feature_columns: tuple[str, ...]

    def predict_proba_many(self, rows: list[dict[str, Any]]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class ModelDatasetEvaluationReport:
    """Metrics for one saved model scored on one train/val/test dataset."""

    model_version: str
    model_path: str
    dataset_dir: str
    dataset_version: str | None
    metrics: dict[str, dict[str, float | int | None]]
    family_metrics: dict[str, dict[str, dict[str, float | int | None]]]
    probability_distributions: dict[str, dict[str, Any]]
    output_dir: str
    calibration_path: str | None = None
    calibration_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_probability_model_on_dataset(
    model_path: Path | str,
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    calibration_path: Path | str | None = None,
) -> ModelDatasetEvaluationReport:
    """Score a saved probability model on all dataset splits and write metrics."""

    model = load_probability_model(model_path)
    calibrator = None if calibration_path is None else load_probability_calibrator(calibration_path)
    dataset = _load_dataset(dataset_dir)
    _validate_feature_columns(dataset["tables"], tuple(model.feature_columns))

    metrics: dict[str, dict[str, float | int | None]] = {}
    family_metrics: dict[str, dict[str, dict[str, float | int | None]]] = {}
    probability_distributions: dict[str, dict[str, Any]] = {}
    dataset_version = _optional_str(dataset["manifest"].get("dataset_version"))
    for split in SPLITS:
        rows = dataset["tables"][split].to_pylist()
        probabilities = model.predict_proba_many(rows)
        if calibrator is not None:
            probabilities = calibrator.transform_many(probabilities)
        metrics[split] = _metrics(_labels(rows), probabilities)
        family_metrics[split] = _metrics_by_market_family(rows, probabilities)
        probability_distributions[split] = _offline_reference(
            rows,
            probabilities,
            model_version=str(model.model_version),
            model_path=str(model_path),
            dataset_dir=str(dataset_dir),
            dataset_version=dataset_version,
            split=split,
            calibration_path=None if calibration_path is None else str(calibration_path),
            calibration_method=None if calibrator is None else str(calibrator.method),
        )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report = ModelDatasetEvaluationReport(
        model_version=str(model.model_version),
        model_path=str(model_path),
        dataset_dir=str(dataset_dir),
        dataset_version=dataset_version,
        metrics=metrics,
        family_metrics=family_metrics,
        probability_distributions=probability_distributions,
        output_dir=str(target),
        calibration_path=None if calibration_path is None else str(calibration_path),
        calibration_method=None if calibrator is None else str(calibrator.method),
    )
    (target / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "family_metrics.json").write_text(
        json.dumps(family_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "probability_distributions.json").write_text(
        json.dumps(probability_distributions, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "offline_reference.json").write_text(
        json.dumps(probability_distributions["val"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "manifest.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def load_probability_model(model_path: Path | str) -> ProbabilityModel:
    """Load a supported saved probability model artifact."""

    path = Path(model_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        payload = None
    if isinstance(payload, dict) and {"coefficients", "intercept", "means", "scales"}.issubset(payload):
        return load_logistic_baseline(path)
    return load_xgboost_v1_model(path)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _offline_reference(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    *,
    model_version: str,
    model_path: str,
    dataset_dir: str,
    dataset_version: str | None,
    split: str,
    calibration_path: str | None,
    calibration_method: str | None,
) -> dict[str, Any]:
    edges = _edge_values(rows, probabilities)
    return {
        "model_version": model_version,
        "model_path": model_path,
        "dataset_dir": dataset_dir,
        "dataset_version": dataset_version,
        "split": split,
        "row_count": len(rows),
        "window_start_ts": _window_bound(rows, min),
        "window_end_ts": _window_bound(rows, max),
        "calibration_path": calibration_path,
        "calibration_method": calibration_method,
        "generated_at_ms": int(time.time() * 1_000),
        "probability_distribution": _distribution_summary(probabilities),
        "edge_distribution": None if not edges else _distribution_summary(edges),
        "edge_trigger_rate_at_0_30": None if not edges else _trigger_rate(edges, 0.30),
    }


def _distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / len(cleaned)
    return {
        "count": len(cleaned),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": cleaned[0],
        "p05": _quantile(cleaned, 0.05),
        "p25": _quantile(cleaned, 0.25),
        "p50": _quantile(cleaned, 0.50),
        "p75": _quantile(cleaned, 0.75),
        "p95": _quantile(cleaned, 0.95),
        "max": cleaned[-1],
    }


def _quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _edge_values(rows: list[dict[str, Any]], probabilities: list[float]) -> list[float]:
    edges: list[float] = []
    for row, probability in zip(rows, probabilities, strict=True):
        market = _market_implied_probability(row)
        if market is not None:
            edges.append(float(probability) - market)
    return edges


def _market_implied_probability(row: dict[str, Any]) -> float | None:
    for key in ("market_implied_prob", "best_ask", "entry_ask_price"):
        value = row.get(key)
        if value is None:
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _trigger_rate(values: list[float], threshold: float) -> float:
    return sum(1 for value in values if value >= threshold) / len(values)


def _window_bound(rows: list[dict[str, Any]], reducer: Any) -> int | None:
    values = [
        int(row["feature_ts"])
        for row in rows
        if row.get("feature_ts") is not None and math.isfinite(float(row["feature_ts"]))
    ]
    return None if not values else reducer(values)
