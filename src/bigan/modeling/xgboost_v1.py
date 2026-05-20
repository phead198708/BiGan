"""Native XGBoost v1 candidate trainer (issue #17)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.serving import write_feature_schema_artifact

from .logistic import (
    _as_float,
    _labels,
    _load_dataset,
    _metrics,
    _validate_feature_columns,
)

XGBOOST_MODEL_VERSION = "xgboost-v1"
XGBOOST_V2_MODEL_VERSION = "xgboost-v2"
SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class XGBoostV1Config:
    """Parameter search space for native XGBoost v1 training."""

    model_version: str = XGBOOST_MODEL_VERSION
    rounds_grid: tuple[int, ...] = (100, 200, 300)
    learning_rate_grid: tuple[float, ...] = (0.01, 0.05, 0.10)
    l2_penalty_grid: tuple[float, ...] = (0.10, 1.0, 5.0)
    max_depth_grid: tuple[int, ...] = (3, 4, 5)
    min_split_loss_grid: tuple[float, ...] = (0.0,)
    subsample_grid: tuple[float, ...] = (0.70, 0.80, 1.0)
    colsample_bytree_grid: tuple[float, ...] = (0.70, 0.80, 1.0)
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.model_version:
            raise ValueError("model_version must not be empty")
        if not self.rounds_grid:
            raise ValueError("rounds_grid must not be empty")
        if not self.learning_rate_grid:
            raise ValueError("learning_rate_grid must not be empty")
        if not self.l2_penalty_grid:
            raise ValueError("l2_penalty_grid must not be empty")
        if not self.max_depth_grid:
            raise ValueError("max_depth_grid must not be empty")
        if not self.min_split_loss_grid:
            raise ValueError("min_split_loss_grid must not be empty")
        if not self.subsample_grid:
            raise ValueError("subsample_grid must not be empty")
        if not self.colsample_bytree_grid:
            raise ValueError("colsample_bytree_grid must not be empty")
        if any(rounds <= 0 for rounds in self.rounds_grid):
            raise ValueError("rounds_grid values must be positive")
        if any(rate <= 0.0 for rate in self.learning_rate_grid):
            raise ValueError("learning_rate_grid values must be positive")
        if any(penalty < 0.0 for penalty in self.l2_penalty_grid):
            raise ValueError("l2_penalty_grid values must be non-negative")
        if any(depth <= 0 for depth in self.max_depth_grid):
            raise ValueError("max_depth_grid values must be positive")
        if any(loss < 0.0 for loss in self.min_split_loss_grid):
            raise ValueError("min_split_loss_grid values must be non-negative")
        if any(value <= 0.0 or value > 1.0 for value in self.subsample_grid):
            raise ValueError("subsample_grid values must be in (0, 1]")
        if any(value <= 0.0 or value > 1.0 for value in self.colsample_bytree_grid):
            raise ValueError("colsample_bytree_grid values must be in (0, 1]")

    def to_dict(self) -> dict[str, float | int | str | list[float] | list[int]]:
        return {
            "model_version": self.model_version,
            "rounds_grid": list(self.rounds_grid),
            "learning_rate_grid": list(self.learning_rate_grid),
            "l2_penalty_grid": list(self.l2_penalty_grid),
            "max_depth_grid": list(self.max_depth_grid),
            "min_split_loss_grid": list(self.min_split_loss_grid),
            "subsample_grid": list(self.subsample_grid),
            "colsample_bytree_grid": list(self.colsample_bytree_grid),
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class XGBoostV1Model:
    """Loadable native XGBoost model wrapper."""

    model_version: str
    feature_columns: tuple[str, ...]
    booster: xgb.Booster
    params: dict[str, float | int | str]

    def predict_proba(self, row: dict[str, Any]) -> float:
        return float(self.booster.predict(_dmatrix([row], self.feature_columns))[0])

    def predict_proba_many(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        return [float(value) for value in self.booster.predict(_dmatrix(rows, self.feature_columns))]

    def top_feature_contributions(
        self,
        row: dict[str, Any],
        *,
        limit: int = 5,
    ) -> list[dict[str, float | int | str]]:
        contributions = self.booster.predict(
            _dmatrix([row], self.feature_columns),
            pred_contribs=True,
        )[0]
        rows = [
            {
                "feature": feature,
                "contribution": float(contributions[idx]),
                "abs_contribution": abs(float(contributions[idx])),
                "split_count": int(self.booster.get_score(importance_type="weight").get(feature, 0)),
            }
            for idx, feature in enumerate(self.feature_columns)
            if float(contributions[idx]) != 0.0
            or self.booster.get_score(importance_type="weight").get(feature, 0)
        ]
        rows.sort(key=lambda item: (-float(item["abs_contribution"]), str(item["feature"])))
        return rows[:limit]


@dataclass(frozen=True, slots=True)
class XGBoostV1Report:
    """Training report for the native XGBoost v1 candidate."""

    model_version: str
    dataset_version: str | None
    best_params: dict[str, float | int | str]
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
    """Train native XGBoost models, tune on validation Brier, and save artifacts."""

    cfg = config or XGBoostV1Config()
    dataset = _load_dataset(dataset_dir)
    feature_columns = tuple(dataset["manifest"]["feature_columns"])
    _validate_feature_columns(dataset["tables"], feature_columns)

    train_rows = dataset["tables"]["train"].to_pylist()
    val_rows = dataset["tables"]["val"].to_pylist()
    if not train_rows:
        raise ValueError("train split must contain at least one row")

    eval_rows = val_rows or train_rows
    eval_labels = _labels(eval_rows)
    train_dmatrix = _dmatrix(train_rows, feature_columns, labels=_labels(train_rows))
    eval_dmatrix = _dmatrix(eval_rows, feature_columns)

    candidates: list[tuple[tuple[float, float, int], XGBoostV1Model]] = []
    for idx, params in enumerate(_parameter_space(cfg)):
        rounds = int(params.pop("rounds"))
        booster = xgb.train(
            params=params,
            dtrain=train_dmatrix,
            num_boost_round=rounds,
            verbose_eval=False,
        )
        params["rounds"] = rounds
        _attach_model_attrs(booster, feature_columns, params, model_version=cfg.model_version)
        model = XGBoostV1Model(
            model_version=cfg.model_version,
            feature_columns=feature_columns,
            booster=booster,
            params=params,
        )
        probabilities = [float(value) for value in booster.predict(eval_dmatrix)]
        metrics = _metrics(eval_labels, probabilities)
        brier = metrics["brier_score"]
        auc = metrics["roc_auc"]
        candidates.append(
            (
                (
                    float("inf") if brier is None else float(brier),
                    -(-1.0 if auc is None else float(auc)),
                    idx,
                ),
                model,
            )
        )

    _, best_model = min(candidates, key=lambda pair: pair[0])
    metrics_by_split: dict[str, dict[str, float | int | None]] = {}
    for split in SPLITS:
        rows = dataset["tables"][split].to_pylist()
        metrics_by_split[split] = _metrics(_labels(rows), best_model.predict_proba_many(rows))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    feature_importance = _feature_importance(best_model.booster)
    report = XGBoostV1Report(
        model_version=cfg.model_version,
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        best_params=best_model.params,
        metrics=metrics_by_split,
        feature_importance=feature_importance,
        output_dir=str(target),
    )
    best_model.booster.save_model(str(target / "model.json"))
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
    write_feature_schema_artifact(
        target / "feature_schema.json",
        feature_columns,
        feature_version=_first_or_none(dataset["manifest"].get("feature_versions")),
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        model_version=cfg.model_version,
    )
    (target / "manifest.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def train_xgboost_v2(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: XGBoostV1Config | None = None,
) -> XGBoostV1Report:
    """Train an expanded XGBoost-v2 challenger search."""

    cfg = config or XGBoostV1Config(
        model_version=XGBOOST_V2_MODEL_VERSION,
    )
    if cfg.model_version != XGBOOST_V2_MODEL_VERSION:
        raise ValueError(f"xgboost-v2 config must use model_version={XGBOOST_V2_MODEL_VERSION!r}")
    return train_xgboost_v1(dataset_dir, output_dir, config=cfg)


def load_xgboost_v1_model(path: Path | str) -> XGBoostV1Model:
    """Load a saved native XGBoost-v1 booster artifact."""

    booster = xgb.Booster()
    booster.load_model(str(path))
    model_version = booster.attr("model_version") or XGBOOST_MODEL_VERSION
    feature_columns_raw = booster.attr("feature_columns")
    if not feature_columns_raw:
        raise ValueError("XGBoost artifact missing feature_columns attribute")
    params_raw = booster.attr("params")
    return XGBoostV1Model(
        model_version=model_version,
        feature_columns=tuple(str(column) for column in json.loads(feature_columns_raw)),
        booster=booster,
        params={} if params_raw is None else json.loads(params_raw),
    )


def _parameter_space(config: XGBoostV1Config) -> list[dict[str, float | int | str]]:
    return [
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "seed": config.seed,
            "nthread": 1,
            "eta": learning_rate,
            "lambda": l2_penalty,
            "max_depth": max_depth,
            "gamma": min_split_loss,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "rounds": rounds,
        }
        for rounds in config.rounds_grid
        for learning_rate in config.learning_rate_grid
        for l2_penalty in config.l2_penalty_grid
        for max_depth in config.max_depth_grid
        for min_split_loss in config.min_split_loss_grid
        for subsample in config.subsample_grid
        for colsample_bytree in config.colsample_bytree_grid
    ]


def _dmatrix(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    *,
    labels: list[int] | None = None,
) -> xgb.DMatrix:
    values = [
        [_feature_value(row, column) for column in feature_columns]
        for row in rows
    ]
    matrix = np.asarray(values, dtype=float)
    return xgb.DMatrix(
        matrix,
        label=None if labels is None else np.asarray(labels, dtype=float),
        feature_names=list(feature_columns),
        missing=np.nan,
    )


def _feature_value(row: dict[str, Any], column: str) -> float:
    value = _as_float(row.get(column))
    return float("nan") if value is None else value


def _attach_model_attrs(
    booster: xgb.Booster,
    feature_columns: tuple[str, ...],
    params: dict[str, float | int | str],
    *,
    model_version: str,
) -> None:
    booster.set_attr(
        model_version=model_version,
        feature_columns=json.dumps(list(feature_columns)),
        params=json.dumps(params, sort_keys=True),
    )


def _feature_importance(booster: xgb.Booster) -> list[dict[str, float | int | str]]:
    gains = booster.get_score(importance_type="gain")
    weights = booster.get_score(importance_type="weight")
    rows = [
        {
            "feature": feature,
            "gain": float(gain),
            "split_count": int(weights.get(feature, 0)),
        }
        for feature, gain in gains.items()
    ]
    rows.sort(key=lambda item: (-float(item["gain"]), str(item["feature"])))
    return rows


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _first_or_none(value: Any) -> str | None:
    if isinstance(value, list | tuple) and value:
        return _optional_str(value[0])
    return _optional_str(value)
