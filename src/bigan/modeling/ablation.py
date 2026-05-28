"""Feature ablation reports for saved probability models."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .calibration import load_probability_calibrator
from .evaluation import load_probability_model
from .logistic import _as_float, _labels, _load_dataset, _metrics, _validate_feature_columns
from .xgboost_v1 import SPLITS

DEFAULT_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "tick_microstructure": ("tick_",),
}

DEFAULT_GROUP_FEATURES: dict[str, tuple[str, ...]] = {
    "time": ("minute_of_day", "day_of_week"),
    "long_window": ("ret_30m", "rv_30m"),
    "trade_structure": ("aggressor_buy_ratio_1m", "avg_trade_size_1m"),
}


@dataclass(frozen=True, slots=True)
class FeatureAblationRow:
    """One feature or feature-group ablation result."""

    name: str
    ablation_type: str
    features: list[str]
    metrics: dict[str, float | int | None]
    deltas: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureAblationReport:
    """Ablation report for a fixed model and dataset split."""

    model_version: str
    model_path: str
    dataset_dir: str
    dataset_version: str | None
    split: str
    calibration_path: str | None
    calibration_method: str | None
    baseline_metrics: dict[str, float | int | None]
    replacement_strategy: str
    replacement_values: dict[str, float]
    ablations: list[FeatureAblationRow]
    output_dir: str
    generated_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ablations"] = [row.to_dict() for row in self.ablations]
        return payload


def generate_feature_ablation_report(
    model_path: Path | str,
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    calibration_path: Path | str | None = None,
    split: str = "test",
) -> FeatureAblationReport:
    """Generate per-feature and grouped mean-replacement ablation artifacts."""

    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    model = load_probability_model(model_path)
    calibrator = None if calibration_path is None else load_probability_calibrator(calibration_path)
    dataset = _load_dataset(dataset_dir)
    feature_columns = tuple(model.feature_columns)
    _validate_feature_columns(dataset["tables"], feature_columns)
    rows = dataset["tables"][split].to_pylist()
    if not rows:
        raise ValueError(f"{split} split must contain at least one row")

    baseline_probabilities = model.predict_proba_many(rows)
    if calibrator is not None:
        baseline_probabilities = calibrator.transform_many(baseline_probabilities)
    baseline_metrics = _metrics(_labels(rows), baseline_probabilities)
    replacement_values = _mean_replacement_values(
        dataset["tables"]["train"].to_pylist(),
        feature_columns,
    )
    ablations = [
        _ablation_row(
            rows,
            model=model,
            calibrator=calibrator,
            baseline_metrics=baseline_metrics,
            name=feature,
            ablation_type="feature",
            features=[feature],
            replacement_values=replacement_values,
        )
        for feature in feature_columns
    ]
    ablations.extend(
        _ablation_row(
            rows,
            model=model,
            calibrator=calibrator,
            baseline_metrics=baseline_metrics,
            name=name,
            ablation_type="group",
            features=features,
            replacement_values=replacement_values,
        )
        for name, features in _feature_groups(feature_columns).items()
    )
    ablations.sort(
        key=lambda row: (
            -_metric_delta(row.deltas.get("brier_score_increase")),
            -_metric_delta(row.deltas.get("roc_auc_drop")),
            row.ablation_type,
            row.name,
        )
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report = FeatureAblationReport(
        model_version=str(model.model_version),
        model_path=str(model_path),
        dataset_dir=str(dataset_dir),
        dataset_version=None if dataset["manifest"].get("dataset_version") is None else str(
            dataset["manifest"].get("dataset_version")
        ),
        split=split,
        calibration_path=None if calibration_path is None else str(calibration_path),
        calibration_method=None if calibrator is None else str(calibrator.method),
        baseline_metrics=baseline_metrics,
        replacement_strategy="train_split_feature_mean",
        replacement_values=replacement_values,
        ablations=ablations,
        output_dir=str(target),
        generated_at_ms=int(time.time() * 1_000),
    )
    (target / "feature_ablation.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "feature_ablation.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _mean_replacement_values(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for feature in feature_columns:
        cleaned = [
            value
            for row in rows
            if (value := _as_float(row.get(feature))) is not None and math.isfinite(value)
        ]
        values[feature] = 0.0 if not cleaned else sum(cleaned) / len(cleaned)
    return values


def _feature_groups(feature_columns: tuple[str, ...]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name, expected in DEFAULT_GROUP_FEATURES.items():
        matched = [feature for feature in expected if feature in feature_columns]
        if matched:
            groups[name] = matched
    for name, prefixes in DEFAULT_GROUP_PREFIXES.items():
        matched = [
            feature
            for feature in feature_columns
            if any(feature.startswith(prefix) for prefix in prefixes)
        ]
        if matched:
            groups[name] = matched
    return groups


def _ablation_row(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    calibrator: Any,
    baseline_metrics: dict[str, float | int | None],
    name: str,
    ablation_type: str,
    features: list[str],
    replacement_values: dict[str, float],
) -> FeatureAblationRow:
    ablated_rows = [_replace_features(row, features, replacement_values) for row in rows]
    probabilities = model.predict_proba_many(ablated_rows)
    if calibrator is not None:
        probabilities = calibrator.transform_many(probabilities)
    metrics = _metrics(_labels(ablated_rows), probabilities)
    return FeatureAblationRow(
        name=name,
        ablation_type=ablation_type,
        features=features,
        metrics=metrics,
        deltas=_metric_deltas(baseline_metrics, metrics),
    )


def _replace_features(
    row: dict[str, Any],
    features: list[str],
    replacement_values: dict[str, float],
) -> dict[str, Any]:
    updated = dict(row)
    for feature in features:
        updated[feature] = replacement_values[feature]
    return updated


def _metric_deltas(
    baseline: dict[str, float | int | None],
    ablated: dict[str, float | int | None],
) -> dict[str, float | None]:
    baseline_brier = _optional_float(baseline.get("brier_score"))
    ablated_brier = _optional_float(ablated.get("brier_score"))
    baseline_auc = _optional_float(baseline.get("roc_auc"))
    ablated_auc = _optional_float(ablated.get("roc_auc"))
    return {
        "brier_score_increase": (
            None if baseline_brier is None or ablated_brier is None else ablated_brier - baseline_brier
        ),
        "roc_auc_drop": None if baseline_auc is None or ablated_auc is None else baseline_auc - ablated_auc,
    }


def _optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _metric_delta(value: float | None) -> float:
    return float("-inf") if value is None else float(value)


def _markdown_report(report: FeatureAblationReport) -> str:
    lines = [
        "# Feature Ablation Report",
        "",
        f"- model_version: `{report.model_version}`",
        f"- split: `{report.split}`",
        f"- replacement_strategy: `{report.replacement_strategy}`",
        "",
        "| Rank | Type | Name | Features | Brier Increase | ROC AUC Drop |",
        "|---:|---|---|---|---:|---:|",
    ]
    for idx, row in enumerate(report.ablations, start=1):
        lines.append(
            "| "
            f"{idx} | {row.ablation_type} | {row.name} | {', '.join(row.features)} | "
            f"{_format_delta(row.deltas.get('brier_score_increase'))} | "
            f"{_format_delta(row.deltas.get('roc_auc_drop'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
