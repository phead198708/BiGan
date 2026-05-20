"""Probability calibration for model outputs (issue #18)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.backtest import evaluate_predictions

from .logistic import _labels, _load_dataset, _sigmoid
from .xgboost_v1 import load_xgboost_v1_model

CalibrationMethod = Literal["platt", "isotonic"]
SUPPORTED_METHODS: frozenset[str] = frozenset({"platt", "isotonic"})


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Calibration search settings."""

    methods: tuple[CalibrationMethod, ...] = ("platt", "isotonic")
    ece_bins: int = 10
    platt_epochs: int = 1_000
    platt_learning_rate: float = 0.10

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("methods must not be empty")
        invalid = sorted(set(self.methods) - SUPPORTED_METHODS)
        if invalid:
            raise ValueError(f"unsupported calibration methods: {', '.join(invalid)}")
        if self.ece_bins <= 0:
            raise ValueError("ece_bins must be positive")
        if self.platt_epochs <= 0:
            raise ValueError("platt_epochs must be positive")
        if self.platt_learning_rate <= 0.0:
            raise ValueError("platt_learning_rate must be positive")

    def to_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "methods": list(self.methods),
            "ece_bins": self.ece_bins,
            "platt_epochs": self.platt_epochs,
            "platt_learning_rate": self.platt_learning_rate,
        }


@dataclass(frozen=True, slots=True)
class ProbabilityCalibrator:
    """Loadable calibration artifact for online inference."""

    method: CalibrationMethod
    model_version: str
    params: dict[str, Any]

    def transform(self, probability: float) -> float:
        checked = _check_probability(probability)
        if self.method == "platt":
            logit = _logit(checked)
            return _sigmoid(float(self.params["a"]) * logit + float(self.params["b"]))
        blocks = self.params["blocks"]
        for block in blocks:
            if checked <= float(block["max_probability"]):
                return float(block["value"])
        return float(blocks[-1]["value"])

    def transform_many(self, probabilities: list[float]) -> list[float]:
        return [self.transform(probability) for probability in probabilities]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model_version": self.model_version,
            "params": self.params,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Calibration comparison report."""

    model_version: str
    method: CalibrationMethod
    raw_metrics: dict[str, float | int | None]
    calibrated_metrics: dict[str, float | int | None]
    candidates: dict[str, dict[str, float | int | None]]
    improved: bool
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_probability_calibration(
    model_path: Path | str,
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: CalibrationConfig | None = None,
) -> CalibrationReport:
    """Fit calibration from a saved XGBoost-v1 model's validation probabilities."""

    model = load_xgboost_v1_model(model_path)
    dataset = _load_dataset(dataset_dir)
    rows = dataset["tables"]["val"].to_pylist()
    if not rows:
        raise ValueError("val split must contain at least one row for calibration")
    return fit_calibration_from_predictions(
        y_true=_labels(rows),
        y_prob=model.predict_proba_many(rows),
        output_dir=output_dir,
        model_version=model.model_version,
        config=config,
    )


def fit_calibration_from_predictions(
    *,
    y_true: list[bool | int],
    y_prob: list[float],
    output_dir: Path | str,
    model_version: str,
    config: CalibrationConfig | None = None,
) -> CalibrationReport:
    """Fit candidate calibration methods from raw probabilities and labels."""

    cfg = config or CalibrationConfig()
    labels = _check_labels(y_true)
    probabilities = [_check_probability(probability) for probability in y_prob]
    if len(labels) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same length")
    if len(set(labels)) < 2:
        raise ValueError("calibration requires both positive and negative labels")

    raw_metrics = _metrics(labels, probabilities, cfg.ece_bins)
    candidate_rows: dict[str, tuple[ProbabilityCalibrator, dict[str, float | int | None]]] = {}
    for method in cfg.methods:
        calibrator = (
            _fit_platt(labels, probabilities, model_version, cfg)
            if method == "platt"
            else _fit_isotonic(labels, probabilities, model_version)
        )
        calibrated = calibrator.transform_many(probabilities)
        candidate_rows[method] = (calibrator, _metrics(labels, calibrated, cfg.ece_bins))

    method, (calibrator, calibrated_metrics) = min(
        candidate_rows.items(),
        key=lambda item: (
            _metric_or_inf(item[1][1]["brier_score"]),
            _metric_or_inf(item[1][1]["ece"]),
            item[0],
        ),
    )
    improved = (
        _metric_or_inf(calibrated_metrics["brier_score"]) <= _metric_or_inf(raw_metrics["brier_score"])
        or _metric_or_inf(calibrated_metrics["ece"]) <= _metric_or_inf(raw_metrics["ece"])
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report = CalibrationReport(
        model_version=model_version,
        method=method,  # type: ignore[arg-type]
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        candidates={name: metrics for name, (_, metrics) in candidate_rows.items()},
        improved=improved,
        output_dir=str(target),
    )
    (target / "calibration.json").write_text(
        json.dumps(calibrator.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "calibration_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def load_probability_calibrator(path: Path | str) -> ProbabilityCalibrator:
    """Load a saved calibration artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProbabilityCalibrator(
        method=data["method"],
        model_version=str(data["model_version"]),
        params=data["params"],
    )


def _fit_platt(
    labels: list[int],
    probabilities: list[float],
    model_version: str,
    config: CalibrationConfig,
) -> ProbabilityCalibrator:
    x_values = [_logit(probability) for probability in probabilities]
    a = 1.0
    b = 0.0
    sample_count = len(labels)
    for _ in range(config.platt_epochs):
        grad_a = grad_b = 0.0
        for x_value, label in zip(x_values, labels, strict=True):
            pred = _sigmoid(a * x_value + b)
            error = pred - label
            grad_a += error * x_value
            grad_b += error
        a -= config.platt_learning_rate * grad_a / sample_count
        b -= config.platt_learning_rate * grad_b / sample_count
    return ProbabilityCalibrator(
        method="platt",
        model_version=model_version,
        params={"a": a, "b": b},
    )


@dataclass(slots=True)
class _IsotonicBlock:
    min_probability: float
    max_probability: float
    positive_sum: float
    count: int

    @property
    def value(self) -> float:
        return self.positive_sum / self.count


def _fit_isotonic(
    labels: list[int],
    probabilities: list[float],
    model_version: str,
) -> ProbabilityCalibrator:
    blocks: list[_IsotonicBlock] = []
    for probability, label in sorted(zip(probabilities, labels, strict=True)):
        blocks.append(
            _IsotonicBlock(
                min_probability=probability,
                max_probability=probability,
                positive_sum=float(label),
                count=1,
            )
        )
        while len(blocks) >= 2 and blocks[-2].value > blocks[-1].value:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                _IsotonicBlock(
                    min_probability=left.min_probability,
                    max_probability=right.max_probability,
                    positive_sum=left.positive_sum + right.positive_sum,
                    count=left.count + right.count,
                )
            )
    return ProbabilityCalibrator(
        method="isotonic",
        model_version=model_version,
        params={
            "blocks": [
                {
                    "min_probability": block.min_probability,
                    "max_probability": block.max_probability,
                    "value": block.value,
                    "count": block.count,
                }
                for block in blocks
            ]
        },
    )


def _metrics(
    labels: list[int],
    probabilities: list[float],
    ece_bins: int,
) -> dict[str, float | int | None]:
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
        "ece": _ece(labels, probabilities, ece_bins),
    }


def _ece(labels: list[int], probabilities: list[float], bin_count: int) -> float:
    total = len(labels)
    error = 0.0
    for idx in range(bin_count):
        start = idx / bin_count
        end = (idx + 1) / bin_count
        members = [
            (label, probability)
            for label, probability in zip(labels, probabilities, strict=True)
            if (start <= probability < end) or (idx == bin_count - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probability for _, probability in members) / len(members)
        accuracy = sum(label for label, _ in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def _check_labels(values: list[bool | int]) -> list[int]:
    labels: list[int] = []
    for value in values:
        if value in (True, 1):
            labels.append(1)
        elif value in (False, 0):
            labels.append(0)
        else:
            raise ValueError(f"labels must be bool/0/1, got {value!r}")
    if not labels:
        raise ValueError("at least one calibration sample is required")
    return labels


def _check_probability(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"probability must be in [0, 1], got {value!r}")
    return probability


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-12, max(1e-12, probability))
    return math.log(clipped / (1.0 - clipped))


def _metric_or_inf(value: float | int | None) -> float:
    return float("inf") if value is None else float(value)
