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

CalibrationMethod = Literal["platt", "isotonic", "temperature", "beta"]
CalibrationSelectionMetric = Literal["brier_score", "ece"]
SUPPORTED_METHODS: frozenset[str] = frozenset(
    {"platt", "isotonic", "temperature", "beta"}
)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Calibration search settings."""

    methods: tuple[CalibrationMethod, ...] = ("platt", "isotonic")
    ece_bins: int = 10
    platt_epochs: int = 1_000
    platt_learning_rate: float = 0.10
    temperature_grid: tuple[float, ...] = (0.50, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0, 5.0)
    beta_epochs: int = 1_000
    beta_learning_rate: float = 0.05
    clip_bounds: tuple[float, float] | None = None

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
        if not self.temperature_grid:
            raise ValueError("temperature_grid must not be empty")
        if any(value <= 0.0 for value in self.temperature_grid):
            raise ValueError("temperature_grid values must be positive")
        if self.beta_epochs <= 0:
            raise ValueError("beta_epochs must be positive")
        if self.beta_learning_rate <= 0.0:
            raise ValueError("beta_learning_rate must be positive")
        if self.clip_bounds is not None:
            _check_clip_bounds(self.clip_bounds)

    def to_dict(self) -> dict[str, float | int | list[float] | list[str] | None]:
        return {
            "methods": list(self.methods),
            "ece_bins": self.ece_bins,
            "platt_epochs": self.platt_epochs,
            "platt_learning_rate": self.platt_learning_rate,
            "temperature_grid": list(self.temperature_grid),
            "beta_epochs": self.beta_epochs,
            "beta_learning_rate": self.beta_learning_rate,
            "clip_bounds": None if self.clip_bounds is None else list(self.clip_bounds),
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
            calibrated = _sigmoid(float(self.params["a"]) * logit + float(self.params["b"]))
        elif self.method == "temperature":
            calibrated = _sigmoid(_logit(checked) / float(self.params["temperature"]))
        elif self.method == "beta":
            clipped = min(1.0 - 1e-12, max(1e-12, checked))
            calibrated = _sigmoid(
                float(self.params["a"]) * math.log(clipped)
                + float(self.params["b"]) * math.log1p(-clipped)
                + float(self.params["c"])
            )
        else:
            blocks = self.params["blocks"]
            for block in blocks:
                if checked <= float(block["max_probability"]):
                    calibrated = float(block["value"])
                    break
            else:
                calibrated = float(blocks[-1]["value"])
        return _apply_clip_bounds(calibrated, self.params.get("clip_bounds"))

    def transform_many(self, probabilities: list[float]) -> list[float]:
        return [self.transform(probability) for probability in probabilities]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model_version": self.model_version,
            "params": self.params,
        }


@dataclass(frozen=True, slots=True)
class FamilyAwareProbabilityCalibrator:
    """Calibration artifact keyed by market family/horizon."""

    model_version: str
    family_calibrators: dict[str, ProbabilityCalibrator]
    global_calibrator: ProbabilityCalibrator | None = None
    default_family_key: str | None = None

    @property
    def method(self) -> str:
        return "family_aware"

    def transform(
        self,
        probability: float,
        *,
        family_key: str | None = None,
        feature: dict[str, Any] | None = None,
    ) -> float:
        key = family_key or family_key_from_feature(feature or {}) or self.default_family_key
        calibrator = self.family_calibrators.get(str(key)) if key is not None else None
        if calibrator is None:
            calibrator = self.global_calibrator
        if calibrator is None:
            return _check_probability(probability)
        return calibrator.transform(probability)

    def transform_many(
        self,
        probabilities: list[float],
        *,
        family_keys: list[str | None] | None = None,
    ) -> list[float]:
        if family_keys is None:
            return [self.transform(probability) for probability in probabilities]
        if len(probabilities) != len(family_keys):
            raise ValueError("probabilities and family_keys must have the same length")
        return [
            self.transform(probability, family_key=family_key)
            for probability, family_key in zip(probabilities, family_keys, strict=True)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "family_aware",
            "model_version": self.model_version,
            "default_family_key": self.default_family_key,
            "global_calibrator": None
            if self.global_calibrator is None
            else self.global_calibrator.to_dict(),
            "family_calibrators": {
                key: calibrator.to_dict()
                for key, calibrator in sorted(self.family_calibrators.items())
            },
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
    family_metrics: dict[str, dict[str, Any]] | None = None
    selection_metric: str = "brier_score"

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
    calibrator, calibrated_metrics, candidate_rows = _fit_best_calibrator(
        labels=labels,
        probabilities=probabilities,
        model_version=model_version,
        config=cfg,
        selection_metric="brier_score",
    )
    method = calibrator.method
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
        selection_metric="brier_score",
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


def fit_family_aware_calibration_from_predictions(
    *,
    y_true: list[bool | int],
    y_prob: list[float],
    family_keys: list[str],
    output_dir: Path | str,
    model_version: str,
    config: CalibrationConfig | None = None,
) -> CalibrationReport:
    """Fit independent calibrators by market family/horizon."""

    cfg = config or CalibrationConfig(
        methods=("platt", "isotonic", "temperature", "beta"),
        clip_bounds=(0.03, 0.97),
    )
    labels = _check_labels(y_true)
    probabilities = [_check_probability(probability) for probability in y_prob]
    if len(labels) != len(probabilities) or len(labels) != len(family_keys):
        raise ValueError("y_true, y_prob, and family_keys must have the same length")
    if len(set(labels)) < 2:
        raise ValueError("calibration requires both positive and negative labels")

    raw_metrics = _metrics(labels, probabilities, cfg.ece_bins)
    global_calibrator, global_metrics, global_candidates = _fit_best_calibrator(
        labels=labels,
        probabilities=probabilities,
        model_version=model_version,
        config=cfg,
        selection_metric="ece",
    )
    family_calibrators: dict[str, ProbabilityCalibrator] = {}
    family_metrics: dict[str, dict[str, Any]] = {}
    for family_key in sorted(set(family_keys)):
        indices = [idx for idx, key in enumerate(family_keys) if key == family_key]
        family_labels = [labels[idx] for idx in indices]
        family_probabilities = [probabilities[idx] for idx in indices]
        family_raw = _metrics(family_labels, family_probabilities, cfg.ece_bins)
        if len(set(family_labels)) < 2:
            calibrated = global_calibrator.transform_many(family_probabilities)
            family_metrics[family_key] = {
                "method": "global_fallback",
                "raw_metrics": family_raw,
                "calibrated_metrics": _metrics(family_labels, calibrated, cfg.ece_bins),
                "sample_count": len(family_labels),
                "fallback_reason": "single_class_family",
            }
            continue
        calibrator, calibrated_metrics, candidates = _fit_best_calibrator(
            labels=family_labels,
            probabilities=family_probabilities,
            model_version=model_version,
            config=cfg,
            selection_metric="ece",
        )
        family_calibrators[family_key] = calibrator
        family_metrics[family_key] = {
            "method": calibrator.method,
            "raw_metrics": family_raw,
            "calibrated_metrics": calibrated_metrics,
            "candidates": {name: metrics for name, (_, metrics) in candidates.items()},
            "sample_count": len(family_labels),
        }

    calibrated_all = [
        family_calibrators.get(family_key, global_calibrator).transform(probability)
        for probability, family_key in zip(probabilities, family_keys, strict=True)
    ]
    calibrated_metrics = _metrics(labels, calibrated_all, cfg.ece_bins)
    improved = (
        _metric_or_inf(calibrated_metrics["brier_score"]) <= _metric_or_inf(raw_metrics["brier_score"])
        or _metric_or_inf(calibrated_metrics["ece"]) <= _metric_or_inf(raw_metrics["ece"])
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifact = FamilyAwareProbabilityCalibrator(
        model_version=model_version,
        family_calibrators=family_calibrators,
        global_calibrator=global_calibrator,
    )
    report = CalibrationReport(
        model_version=model_version,
        method="isotonic" if global_calibrator.method == "isotonic" else global_calibrator.method,
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        candidates={name: metrics for name, (_, metrics) in global_candidates.items()},
        improved=improved,
        output_dir=str(target),
        family_metrics=family_metrics,
        selection_metric="ece",
    )
    (target / "calibration.json").write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "calibration_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def load_probability_calibrator(
    path: Path | str,
) -> ProbabilityCalibrator | FamilyAwareProbabilityCalibrator:
    """Load a saved calibration artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") == "family_aware":
        global_data = data.get("global_calibrator")
        return FamilyAwareProbabilityCalibrator(
            model_version=str(data["model_version"]),
            family_calibrators={
                str(key): _calibrator_from_dict(value)
                for key, value in data.get("family_calibrators", {}).items()
            },
            global_calibrator=None if global_data is None else _calibrator_from_dict(global_data),
            default_family_key=data.get("default_family_key"),
        )
    return _calibrator_from_dict(data)


def family_key_from_feature(feature: dict[str, Any]) -> str | None:
    """Return a stable family key such as ``BTC-15M`` from a feature row."""

    canonical = feature.get("canonical_symbol") or feature.get("symbol")
    if canonical:
        family = str(canonical).split(":", 1)[0].upper()
        parts = family.split("-")
        if len(parts) >= 2:
            if parts[1] in {"UP", "DOWN"} and len(parts) >= 3:
                return f"{parts[0]}-{parts[2]}"
            return f"{parts[0]}-{parts[1]}"
    underlying = _underlying_name(feature.get("underlying_id"))
    horizon = _horizon_name(feature.get("horizon_minutes"))
    if underlying is None or horizon is None:
        return None
    return f"{underlying}-{horizon}"


def transform_probability(
    calibrator: ProbabilityCalibrator | FamilyAwareProbabilityCalibrator | None,
    probability: float,
    *,
    feature: dict[str, Any] | None = None,
) -> float:
    """Apply either a global or family-aware calibration artifact."""

    if calibrator is None:
        return _check_probability(probability)
    if isinstance(calibrator, FamilyAwareProbabilityCalibrator):
        return calibrator.transform(probability, feature=feature)
    return calibrator.transform(probability)


def _calibrator_from_dict(data: dict[str, Any]) -> ProbabilityCalibrator:
    return ProbabilityCalibrator(
        method=data["method"],
        model_version=str(data["model_version"]),
        params=data["params"],
    )


def _fit_best_calibrator(
    *,
    labels: list[int],
    probabilities: list[float],
    model_version: str,
    config: CalibrationConfig,
    selection_metric: CalibrationSelectionMetric,
) -> tuple[
    ProbabilityCalibrator,
    dict[str, float | int | None],
    dict[str, tuple[ProbabilityCalibrator, dict[str, float | int | None]]],
]:
    candidate_rows: dict[str, tuple[ProbabilityCalibrator, dict[str, float | int | None]]] = {}
    for method in config.methods:
        calibrator = _fit_method(method, labels, probabilities, model_version, config)
        calibrated = calibrator.transform_many(probabilities)
        candidate_rows[method] = (calibrator, _metrics(labels, calibrated, config.ece_bins))

    method, (calibrator, calibrated_metrics) = min(
        candidate_rows.items(),
        key=lambda item: (
            _metric_or_inf(item[1][1][selection_metric]),
            _metric_or_inf(item[1][1]["brier_score" if selection_metric == "ece" else "ece"]),
            item[0],
        ),
    )
    return calibrator, calibrated_metrics, candidate_rows


def _fit_method(
    method: CalibrationMethod,
    labels: list[int],
    probabilities: list[float],
    model_version: str,
    config: CalibrationConfig,
) -> ProbabilityCalibrator:
    if method == "platt":
        calibrator = _fit_platt(labels, probabilities, model_version, config)
    elif method == "isotonic":
        calibrator = _fit_isotonic(labels, probabilities, model_version)
    elif method == "temperature":
        calibrator = _fit_temperature(labels, probabilities, model_version, config)
    else:
        calibrator = _fit_beta(labels, probabilities, model_version, config)
    if config.clip_bounds is None:
        return calibrator
    return ProbabilityCalibrator(
        method=calibrator.method,
        model_version=calibrator.model_version,
        params={**calibrator.params, "clip_bounds": list(config.clip_bounds)},
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


def _fit_temperature(
    labels: list[int],
    probabilities: list[float],
    model_version: str,
    config: CalibrationConfig,
) -> ProbabilityCalibrator:
    best_temperature = min(
        config.temperature_grid,
        key=lambda temperature: _metric_or_inf(
            _metrics(
                labels,
                [_sigmoid(_logit(probability) / temperature) for probability in probabilities],
                config.ece_bins,
            )["brier_score"]
        ),
    )
    return ProbabilityCalibrator(
        method="temperature",
        model_version=model_version,
        params={"temperature": float(best_temperature)},
    )


def _fit_beta(
    labels: list[int],
    probabilities: list[float],
    model_version: str,
    config: CalibrationConfig,
) -> ProbabilityCalibrator:
    features = [
        (
            math.log(min(1.0 - 1e-12, max(1e-12, probability))),
            math.log1p(-min(1.0 - 1e-12, max(1e-12, probability))),
        )
        for probability in probabilities
    ]
    a = b = c = 0.0
    sample_count = len(labels)
    for _ in range(config.beta_epochs):
        grad_a = grad_b = grad_c = 0.0
        for (x1, x2), label in zip(features, labels, strict=True):
            pred = _sigmoid(a * x1 + b * x2 + c)
            error = pred - label
            grad_a += error * x1
            grad_b += error * x2
            grad_c += error
        a -= config.beta_learning_rate * grad_a / sample_count
        b -= config.beta_learning_rate * grad_b / sample_count
        c -= config.beta_learning_rate * grad_c / sample_count
    return ProbabilityCalibrator(
        method="beta",
        model_version=model_version,
        params={"a": a, "b": b, "c": c},
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


def _check_clip_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower < 0.0 or upper > 1.0 or lower >= upper:
        raise ValueError("clip_bounds must satisfy 0 <= lower < upper <= 1")
    return lower, upper


def _apply_clip_bounds(value: float, bounds: Any) -> float:
    if bounds is None:
        return _check_probability(value)
    lower, upper = _check_clip_bounds((float(bounds[0]), float(bounds[1])))
    return min(upper, max(lower, _check_probability(value)))


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-12, max(1e-12, probability))
    return math.log(clipped / (1.0 - clipped))


def _metric_or_inf(value: float | int | None) -> float:
    return float("inf") if value is None else float(value)


def _underlying_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).upper()
        return text or None
    known = {1.0: "BTC", 2.0: "ETH", 3.0: "SOL"}
    return known.get(numeric)


def _horizon_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).upper()
        return text or None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    if numeric.is_integer():
        return f"{int(numeric)}M"
    return f"{numeric:g}M"
