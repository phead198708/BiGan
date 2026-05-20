"""Dependency-free XGBoost v1 candidate trainer (issue #17)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .logistic import (
    _as_float,
    _labels,
    _load_dataset,
    _metrics,
    _sigmoid,
    _validate_feature_columns,
)

XGBOOST_MODEL_VERSION = "xgboost-v1"
SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class XGBoostV1Config:
    """Small parameter search space for the v1 boosted-stump candidate."""

    rounds_grid: tuple[int, ...] = (20,)
    learning_rate_grid: tuple[float, ...] = (0.10,)
    l2_penalty_grid: tuple[float, ...] = (1.0,)
    max_depth_grid: tuple[int, ...] = (1,)
    min_split_gain: float = 0.0

    def __post_init__(self) -> None:
        if not self.rounds_grid:
            raise ValueError("rounds_grid must not be empty")
        if not self.learning_rate_grid:
            raise ValueError("learning_rate_grid must not be empty")
        if not self.l2_penalty_grid:
            raise ValueError("l2_penalty_grid must not be empty")
        if not self.max_depth_grid:
            raise ValueError("max_depth_grid must not be empty")
        if any(rounds <= 0 for rounds in self.rounds_grid):
            raise ValueError("rounds_grid values must be positive")
        if any(rate <= 0.0 for rate in self.learning_rate_grid):
            raise ValueError("learning_rate_grid values must be positive")
        if any(penalty < 0.0 for penalty in self.l2_penalty_grid):
            raise ValueError("l2_penalty_grid values must be non-negative")
        if any(depth != 1 for depth in self.max_depth_grid):
            raise ValueError("xgboost-v1 currently supports max_depth=1 boosted stumps")
        if self.min_split_gain < 0.0:
            raise ValueError("min_split_gain must be non-negative")

    def to_dict(self) -> dict[str, float | list[float] | list[int]]:
        return {
            "rounds_grid": list(self.rounds_grid),
            "learning_rate_grid": list(self.learning_rate_grid),
            "l2_penalty_grid": list(self.l2_penalty_grid),
            "max_depth_grid": list(self.max_depth_grid),
            "min_split_gain": self.min_split_gain,
        }


@dataclass(frozen=True, slots=True)
class XGBoostV1Stump:
    """One depth-1 tree in the v1 boosted candidate."""

    feature: str
    threshold: float
    left_value: float
    right_value: float
    gain: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class XGBoostV1Model:
    """Saved boosted-stump model artifact."""

    model_version: str
    feature_columns: tuple[str, ...]
    base_score: float
    stumps: tuple[XGBoostV1Stump, ...]
    feature_means: dict[str, float]
    params: dict[str, float | int]

    def predict_margin(self, row: dict[str, Any]) -> float:
        margin = self.base_score
        for stump in self.stumps:
            value = _feature_value(row, stump.feature, self.feature_means)
            margin += stump.left_value if value <= stump.threshold else stump.right_value
        return margin

    def predict_proba(self, row: dict[str, Any]) -> float:
        return _sigmoid(self.predict_margin(row))

    def predict_proba_many(self, rows: list[dict[str, Any]]) -> list[float]:
        return [self.predict_proba(row) for row in rows]

    def top_feature_contributions(
        self,
        row: dict[str, Any],
        *,
        limit: int = 5,
    ) -> list[dict[str, float | int | str]]:
        contributions: dict[str, float] = {}
        counts: dict[str, int] = {}
        for stump in self.stumps:
            value = _feature_value(row, stump.feature, self.feature_means)
            contribution = stump.left_value if value <= stump.threshold else stump.right_value
            contributions[stump.feature] = contributions.get(stump.feature, 0.0) + contribution
            counts[stump.feature] = counts.get(stump.feature, 0) + 1
        rows = [
            {
                "feature": feature,
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "split_count": counts[feature],
            }
            for feature, contribution in contributions.items()
        ]
        rows.sort(key=lambda item: (-float(item["abs_contribution"]), str(item["feature"])))
        return rows[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "algorithm": "logistic-loss-gradient-boosted-stumps",
            "feature_columns": list(self.feature_columns),
            "base_score": self.base_score,
            "stumps": [stump.to_dict() for stump in self.stumps],
            "feature_means": {key: self.feature_means[key] for key in sorted(self.feature_means)},
            "params": dict(sorted(self.params.items())),
        }


@dataclass(frozen=True, slots=True)
class XGBoostV1Report:
    """Training report for the XGBoost v1 candidate."""

    model_version: str
    dataset_version: str | None
    best_params: dict[str, float | int]
    metrics: dict[str, dict[str, float | int | None]]
    feature_importance: list[dict[str, float | int | str]]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "best_params": self.best_params,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "output_dir": self.output_dir,
        }


def train_xgboost_v1(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: XGBoostV1Config | None = None,
) -> XGBoostV1Report:
    """Train, tune on validation Brier score, and save XGBoost-v1 artifacts."""

    cfg = config or XGBoostV1Config()
    dataset = _load_dataset(dataset_dir)
    feature_columns = tuple(dataset["manifest"]["feature_columns"])
    _validate_feature_columns(dataset["tables"], feature_columns)

    train_rows = dataset["tables"]["train"].to_pylist()
    val_rows = dataset["tables"]["val"].to_pylist()
    if not train_rows:
        raise ValueError("train split must contain at least one row")

    candidates: list[tuple[tuple[float, float, int], XGBoostV1Model]] = []
    eval_rows = val_rows or train_rows
    eval_labels = _labels(eval_rows)
    for idx, params in enumerate(_parameter_space(cfg)):
        model = _fit_boosted_stumps(train_rows, feature_columns, params, cfg.min_split_gain)
        probabilities = model.predict_proba_many(eval_rows)
        metrics = _metrics(eval_labels, probabilities)
        brier = metrics["brier_score"]
        auc = metrics["roc_auc"]
        score = (
            float("inf") if brier is None else float(brier),
            -(-1.0 if auc is None else float(auc)),
            idx,
        )
        candidates.append((score, model))

    _, best_model = min(candidates, key=lambda pair: pair[0])
    metrics_by_split: dict[str, dict[str, float | int | None]] = {}
    for split in SPLITS:
        rows = dataset["tables"][split].to_pylist()
        metrics_by_split[split] = _metrics(_labels(rows), best_model.predict_proba_many(rows))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    feature_importance = _feature_importance(best_model)
    report = XGBoostV1Report(
        model_version=XGBOOST_MODEL_VERSION,
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        best_params=best_model.params,
        metrics=metrics_by_split,
        feature_importance=feature_importance,
        output_dir=str(target),
    )
    (target / "model.json").write_text(
        json.dumps(best_model.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "xgboost_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "metrics.json").write_text(
        json.dumps(metrics_by_split, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "feature_importance.json").write_text(
        json.dumps(feature_importance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "manifest.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def load_xgboost_v1_model(path: Path | str) -> XGBoostV1Model:
    """Load a saved XGBoost-v1 model artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return XGBoostV1Model(
        model_version=str(data["model_version"]),
        feature_columns=tuple(str(column) for column in data["feature_columns"]),
        base_score=float(data["base_score"]),
        stumps=tuple(
            XGBoostV1Stump(
                feature=str(stump["feature"]),
                threshold=float(stump["threshold"]),
                left_value=float(stump["left_value"]),
                right_value=float(stump["right_value"]),
                gain=float(stump["gain"]),
            )
            for stump in data["stumps"]
        ),
        feature_means={str(key): float(value) for key, value in data["feature_means"].items()},
        params={str(key): value for key, value in data["params"].items()},
    )


def _parameter_space(config: XGBoostV1Config) -> list[dict[str, float | int]]:
    return [
        {
            "rounds": rounds,
            "learning_rate": learning_rate,
            "l2_penalty": l2_penalty,
            "max_depth": max_depth,
        }
        for rounds in config.rounds_grid
        for learning_rate in config.learning_rate_grid
        for l2_penalty in config.l2_penalty_grid
        for max_depth in config.max_depth_grid
    ]


def _fit_boosted_stumps(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    params: dict[str, float | int],
    min_split_gain: float,
) -> XGBoostV1Model:
    feature_means = _feature_means(rows, feature_columns)
    x_rows = _matrix(rows, feature_columns, feature_means)
    labels = _labels(rows)
    base_score = _initial_score(labels)
    margins = [base_score for _ in labels]
    stumps: list[XGBoostV1Stump] = []
    learning_rate = float(params["learning_rate"])
    l2_penalty = float(params["l2_penalty"])

    for _ in range(int(params["rounds"])):
        gradients: list[float] = []
        hessians: list[float] = []
        for margin, label in zip(margins, labels, strict=True):
            probability = _sigmoid(margin)
            gradients.append(probability - label)
            hessians.append(max(probability * (1.0 - probability), 1e-12))

        stump = _best_stump(
            x_rows,
            gradients,
            hessians,
            feature_columns,
            l2_penalty=l2_penalty,
            learning_rate=learning_rate,
        )
        if stump is None or stump.gain <= min_split_gain:
            break
        stumps.append(stump)
        for idx, features in enumerate(x_rows):
            value = features[feature_columns.index(stump.feature)]
            margins[idx] += stump.left_value if value <= stump.threshold else stump.right_value

    return XGBoostV1Model(
        model_version=XGBOOST_MODEL_VERSION,
        feature_columns=feature_columns,
        base_score=base_score,
        stumps=tuple(stumps),
        feature_means=feature_means,
        params=params,
    )


def _best_stump(
    x_rows: list[list[float]],
    gradients: list[float],
    hessians: list[float],
    feature_columns: tuple[str, ...],
    *,
    l2_penalty: float,
    learning_rate: float,
) -> XGBoostV1Stump | None:
    total_gradient = sum(gradients)
    total_hessian = sum(hessians)
    parent_score = _node_score(total_gradient, total_hessian, l2_penalty)
    best: XGBoostV1Stump | None = None

    for feature_idx, feature in enumerate(feature_columns):
        values = sorted({row[feature_idx] for row in x_rows})
        thresholds = [
            (left + right) / 2.0
            for left, right in zip(values, values[1:], strict=False)
            if left != right
        ]
        for threshold in thresholds:
            left_gradient = left_hessian = right_gradient = right_hessian = 0.0
            left_count = right_count = 0
            for row, gradient, hessian in zip(x_rows, gradients, hessians, strict=True):
                if row[feature_idx] <= threshold:
                    left_gradient += gradient
                    left_hessian += hessian
                    left_count += 1
                else:
                    right_gradient += gradient
                    right_hessian += hessian
                    right_count += 1
            if left_count == 0 or right_count == 0:
                continue
            gain = (
                _node_score(left_gradient, left_hessian, l2_penalty)
                + _node_score(right_gradient, right_hessian, l2_penalty)
                - parent_score
            )
            if best is None or gain > best.gain + 1e-15:
                best = XGBoostV1Stump(
                    feature=feature,
                    threshold=threshold,
                    left_value=learning_rate * _leaf_weight(left_gradient, left_hessian, l2_penalty),
                    right_value=learning_rate
                    * _leaf_weight(right_gradient, right_hessian, l2_penalty),
                    gain=gain,
                )
    return best


def _node_score(gradient_sum: float, hessian_sum: float, l2_penalty: float) -> float:
    return 0.5 * gradient_sum * gradient_sum / (hessian_sum + l2_penalty)


def _leaf_weight(gradient_sum: float, hessian_sum: float, l2_penalty: float) -> float:
    return -gradient_sum / (hessian_sum + l2_penalty)


def _feature_importance(model: XGBoostV1Model) -> list[dict[str, float | int | str]]:
    rows: dict[str, dict[str, float | int | str]] = {
        feature: {"feature": feature, "gain": 0.0, "split_count": 0}
        for feature in model.feature_columns
    }
    for stump in model.stumps:
        row = rows[stump.feature]
        row["gain"] = float(row["gain"]) + stump.gain
        row["split_count"] = int(row["split_count"]) + 1
    out = [row for row in rows.values() if int(row["split_count"]) > 0]
    out.sort(key=lambda item: (-float(item["gain"]), str(item["feature"])))
    return out


def _feature_means(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for column in feature_columns:
        values = [_as_float(row.get(column)) for row in rows]
        present = [value for value in values if value is not None]
        out[column] = sum(present) / len(present) if present else 0.0
    return out


def _matrix(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    feature_means: dict[str, float],
) -> list[list[float]]:
    return [
        [_feature_value(row, column, feature_means) for column in feature_columns]
        for row in rows
    ]


def _feature_value(
    row: dict[str, Any],
    column: str,
    feature_means: dict[str, float],
) -> float:
    value = _as_float(row.get(column))
    return feature_means[column] if value is None else value


def _initial_score(labels: list[int]) -> float:
    positive_rate = sum(labels) / len(labels)
    clipped = min(1.0 - 1e-6, max(1e-6, positive_rate))
    return math.log(clipped / (1.0 - clipped))


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
