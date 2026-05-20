"""Deterministic logistic regression baseline training (issue #16)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from bigan.backtest import evaluate_predictions

MODEL_VERSION = "logreg-baseline-v1"
SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class LogisticBaselineConfig:
    """Hyperparameters for the pure-Python logistic regression baseline."""

    epochs: int = 500
    learning_rate: float = 0.10
    l2_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.l2_penalty < 0.0:
            raise ValueError("l2_penalty must be non-negative")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LogisticBaselineModel:
    """Saved logistic model with train-set preprocessing statistics."""

    model_version: str
    feature_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    means: dict[str, float]
    scales: dict[str, float]

    def predict_proba(self, row: dict[str, Any]) -> float:
        z = self.intercept
        for column, coefficient in zip(self.feature_columns, self.coefficients, strict=True):
            value = _as_float(row.get(column))
            if value is None:
                value = self.means[column]
            z += coefficient * ((value - self.means[column]) / self.scales[column])
        return _sigmoid(z)

    def predict_proba_many(self, rows: list[dict[str, Any]]) -> list[float]:
        return [self.predict_proba(row) for row in rows]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "feature_columns": list(self.feature_columns),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "means": {key: self.means[key] for key in sorted(self.means)},
            "scales": {key: self.scales[key] for key in sorted(self.scales)},
        }


@dataclass(frozen=True, slots=True)
class LogisticBaselineReport:
    """Training report for a saved logistic baseline run."""

    model_version: str
    dataset_version: str | None
    feature_columns: tuple[str, ...]
    metrics: dict[str, dict[str, float | int | None]]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_columns": list(self.feature_columns),
            "metrics": self.metrics,
            "output_dir": self.output_dir,
        }


def train_logistic_baseline(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: LogisticBaselineConfig | None = None,
) -> LogisticBaselineReport:
    """Train, evaluate, and save the logistic regression baseline."""

    cfg = config or LogisticBaselineConfig()
    dataset = _load_dataset(dataset_dir)
    feature_columns = tuple(dataset["manifest"]["feature_columns"])
    _validate_feature_columns(dataset["tables"], feature_columns)

    train_rows = dataset["tables"]["train"].to_pylist()
    if not train_rows:
        raise ValueError("train split must contain at least one row")

    means, scales = _fit_preprocessor(train_rows, feature_columns)
    x_train = _matrix(train_rows, feature_columns, means, scales)
    y_train = _labels(train_rows)
    coefficients, intercept = _fit_logistic(x_train, y_train, cfg)
    model = LogisticBaselineModel(
        model_version=MODEL_VERSION,
        feature_columns=feature_columns,
        coefficients=tuple(coefficients),
        intercept=intercept,
        means=means,
        scales=scales,
    )

    metrics: dict[str, dict[str, float | int | None]] = {}
    for split in SPLITS:
        rows = dataset["tables"][split].to_pylist()
        probabilities = model.predict_proba_many(rows)
        metrics[split] = _metrics(_labels(rows), probabilities)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report = LogisticBaselineReport(
        model_version=MODEL_VERSION,
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        feature_columns=feature_columns,
        metrics=metrics,
        output_dir=str(target),
    )
    (target / "model.json").write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "baseline_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "manifest.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def load_logistic_baseline(path: Path | str) -> LogisticBaselineModel:
    """Load a saved logistic baseline artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LogisticBaselineModel(
        model_version=str(data["model_version"]),
        feature_columns=tuple(str(column) for column in data["feature_columns"]),
        coefficients=tuple(float(value) for value in data["coefficients"]),
        intercept=float(data["intercept"]),
        means={str(key): float(value) for key, value in data["means"].items()},
        scales={str(key): float(value) for key, value in data["scales"].items()},
    )


def _load_dataset(dataset_dir: Path | str) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_columns = manifest.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("dataset manifest must include non-empty feature_columns")
    tables: dict[str, pa.Table] = {}
    for split in SPLITS:
        path = root / f"{split}.parquet"
        if not path.exists():
            raise ValueError(f"dataset split not found: {path}")
        tables[split] = pq.read_table(path)
    return {"manifest": manifest, "tables": tables}


def _validate_feature_columns(tables: dict[str, pa.Table], feature_columns: tuple[str, ...]) -> None:
    for split, table in tables.items():
        missing = sorted(set(feature_columns) - set(table.schema.names))
        if missing:
            raise ValueError(f"{split} split missing feature columns: {', '.join(missing)}")
        if "label_profit_up_15m" not in table.schema.names and "label_up_15m" not in table.schema.names:
            raise ValueError(f"{split} split missing label_profit_up_15m or label_up_15m")


def _fit_preprocessor(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in feature_columns:
        values = [_as_float(row.get(column)) for row in rows]
        present = [value for value in values if value is not None]
        mean = sum(present) / len(present) if present else 0.0
        variance = (
            sum((value - mean) ** 2 for value in present) / len(present)
            if present
            else 0.0
        )
        scale = math.sqrt(variance)
        means[column] = mean
        scales[column] = scale if scale > 0.0 else 1.0
    return means, scales


def _matrix(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[list[float]]:
    out: list[list[float]] = []
    for row in rows:
        features: list[float] = []
        for column in feature_columns:
            value = _as_float(row.get(column))
            if value is None:
                value = means[column]
            features.append((value - means[column]) / scales[column])
        out.append(features)
    return out


def _labels(rows: list[dict[str, Any]]) -> list[int]:
    return [1 if _label_value(row) else 0 for row in rows]


def _label_value(row: dict[str, Any]) -> bool:
    value = row.get("label_profit_up_15m")
    if value is None:
        value = row["label_up_15m"]
    return bool(value)


def _fit_logistic(
    x_rows: list[list[float]],
    labels: list[int],
    config: LogisticBaselineConfig,
) -> tuple[list[float], float]:
    feature_count = len(x_rows[0]) if x_rows else 0
    coefficients = [0.0] * feature_count
    intercept = _initial_intercept(labels)
    sample_count = len(labels)

    for _ in range(config.epochs):
        intercept_gradient = 0.0
        coefficient_gradients = [0.0] * feature_count
        for features, label in zip(x_rows, labels, strict=True):
            probability = _sigmoid(intercept + _dot(coefficients, features))
            error = probability - label
            intercept_gradient += error
            for idx, value in enumerate(features):
                coefficient_gradients[idx] += error * value

        intercept -= config.learning_rate * (intercept_gradient / sample_count)
        for idx, coefficient in enumerate(coefficients):
            gradient = coefficient_gradients[idx] / sample_count
            gradient += config.l2_penalty * coefficient
            coefficients[idx] -= config.learning_rate * gradient

    return coefficients, intercept


def _initial_intercept(labels: list[int]) -> float:
    positive_rate = sum(labels) / len(labels)
    clipped = min(1.0 - 1e-6, max(1e-6, positive_rate))
    return math.log(clipped / (1.0 - clipped))


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, float | int | None]:
    if not labels:
        return {
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
            "brier_score": None,
        }
    report = evaluate_predictions(y_true=labels, y_prob=probabilities, thresholds=[0.5])
    threshold = report.thresholds[0]
    return {
        "sample_count": report.sample_count,
        "positive_count": report.positive_count,
        "negative_count": report.negative_count,
        "accuracy": threshold.accuracy,
        "precision": threshold.precision,
        "recall": threshold.recall,
        "f1": threshold.f1,
        "roc_auc": report.roc_auc,
        "pr_auc": report.pr_auc,
        "brier_score": report.brier_score,
    }


def _dot(coefficients: list[float], features: list[float]) -> float:
    return sum(coefficient * value for coefficient, value in zip(coefficients, features, strict=True))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-min(value, 500.0))
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(max(value, -500.0))
    return exp_pos / (1.0 + exp_pos)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
