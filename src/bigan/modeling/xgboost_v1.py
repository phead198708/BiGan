"""Native XGBoost v1 candidate trainer (issue #17)."""

from __future__ import annotations

import json
import math
import time
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
    _metrics_by_market_family,
    _validate_feature_columns,
)

XGBOOST_MODEL_VERSION = "xgboost-v1"
XGBOOST_V2_MODEL_VERSION = "xgboost-v2"
XGBOOST_V3_MODEL_VERSION = "xgboost-v3"
XGBOOST_V4_MODEL_VERSION = "xgboost-v4"
XGBOOST_V5_MODEL_VERSION = "xgboost-v5"
XGBOOST_V4_REQUIRED_MARKET_FEATURES: tuple[str, ...] = (
    "underlying_id",
    "horizon_minutes",
    "liquidity_bucket",
)
XGBOOST_V4_REQUIRED_ADDED_FEATURES: tuple[str, ...] = (
    "minute_of_day",
    "day_of_week",
    "ret_30m",
    "rv_30m",
    "aggressor_buy_ratio_1m",
    "avg_trade_size_1m",
)
XGBOOST_V4_REQUIRED_TICK_FEATURES: tuple[str, ...] = (
    "tick_spread",
    "tick_obi_l1",
    "tick_obi_l3",
    "tick_mid_price",
    "tick_price_velocity",
    "tick_trade_arrival_rate",
)
XGBOOST_V4_REQUIRED_FEATURES: tuple[str, ...] = (
    *XGBOOST_V4_REQUIRED_MARKET_FEATURES,
    *XGBOOST_V4_REQUIRED_ADDED_FEATURES,
    *XGBOOST_V4_REQUIRED_TICK_FEATURES,
)
SPLITS: tuple[str, ...] = ("train", "val", "test")
XGBOOST_ENSEMBLE_SCHEMA_VERSION = "xgboost_ensemble_v1"


@dataclass(frozen=True, slots=True)
class XGBoostV1Config:
    """Parameter search space for native XGBoost v1 training."""

    model_version: str = XGBOOST_MODEL_VERSION
    rounds_grid: tuple[int, ...] = (100, 200, 300)
    learning_rate_grid: tuple[float, ...] = (0.01, 0.05, 0.10)
    l2_penalty_grid: tuple[float, ...] = (0.10, 1.0, 5.0)
    max_depth_grid: tuple[int, ...] = (3, 4, 5)
    min_split_loss_grid: tuple[float, ...] = (0.0,)
    min_child_weight_grid: tuple[float, ...] = (1.0,)
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
        if not self.min_child_weight_grid:
            raise ValueError("min_child_weight_grid must not be empty")
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
        if any(weight <= 0.0 for weight in self.min_child_weight_grid):
            raise ValueError("min_child_weight_grid values must be positive")
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
            "min_child_weight_grid": list(self.min_child_weight_grid),
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
    ensemble_boosters: tuple[xgb.Booster, ...] = ()

    def predict_proba(self, row: dict[str, Any]) -> float:
        return self.predict_proba_many([row])[0]

    def predict_proba_many(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        matrix = _dmatrix(rows, self.feature_columns)
        predictions = [
            [float(value) for value in booster.predict(matrix)]
            for booster in self._boosters()
        ]
        return [
            sum(member_values) / len(member_values)
            for member_values in zip(*predictions, strict=True)
        ]

    def _boosters(self) -> tuple[xgb.Booster, ...]:
        return self.ensemble_boosters or (self.booster,)

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
    family_metrics: dict[str, dict[str, dict[str, float | int | None]]]
    feature_importance: list[dict[str, float | int | str]]
    output_dir: str
    cv_summary: dict[str, Any] | None = None
    ensemble: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "best_params": self.best_params,
            "metrics": self.metrics,
            "family_metrics": self.family_metrics,
            "feature_importance": self.feature_importance,
            "output_dir": self.output_dir,
        }
        if self.cv_summary is not None:
            payload["cv_summary"] = self.cv_summary
        if self.ensemble is not None:
            payload["ensemble"] = self.ensemble
        return payload


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
    family_metrics_by_split: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for split in SPLITS:
        rows = dataset["tables"][split].to_pylist()
        probabilities = best_model.predict_proba_many(rows)
        metrics_by_split[split] = _metrics(_labels(rows), probabilities)
        family_metrics_by_split[split] = _metrics_by_market_family(rows, probabilities)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    feature_importance = _feature_importance(best_model.booster)
    report = XGBoostV1Report(
        model_version=cfg.model_version,
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        best_params=best_model.params,
        metrics=metrics_by_split,
        family_metrics=family_metrics_by_split,
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
    (target / "family_metrics.json").write_text(
        json.dumps(family_metrics_by_split, indent=2, sort_keys=True),
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


def train_xgboost_v3(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: XGBoostV1Config | None = None,
) -> XGBoostV1Report:
    """Train a conservative XGBoost-v3 challenger focused on validation Brier."""

    cfg = config or XGBoostV1Config(
        model_version=XGBOOST_V3_MODEL_VERSION,
        rounds_grid=(100, 150, 200),
        learning_rate_grid=(0.01, 0.03, 0.05),
        l2_penalty_grid=(5.0, 10.0, 20.0),
        max_depth_grid=(3, 4),
        min_child_weight_grid=(2.0, 5.0, 10.0),
        subsample_grid=(0.80, 1.0),
        colsample_bytree_grid=(0.80, 1.0),
    )
    if cfg.model_version != XGBOOST_V3_MODEL_VERSION:
        raise ValueError(f"xgboost-v3 config must use model_version={XGBOOST_V3_MODEL_VERSION!r}")
    return train_xgboost_v1(dataset_dir, output_dir, config=cfg)


def train_xgboost_v4(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: XGBoostV1Config | None = None,
    ensemble_seeds: tuple[int, ...] = (0, 17, 42),
) -> XGBoostV1Report:
    """Train xgboost-v4 with issue #58 time-series CV and a light seed ensemble."""

    if not ensemble_seeds:
        raise ValueError("ensemble_seeds must not be empty")
    cfg = config or XGBoostV1Config(
        model_version=XGBOOST_V4_MODEL_VERSION,
        rounds_grid=(150, 200),
        learning_rate_grid=(0.03, 0.05),
        l2_penalty_grid=(5.0, 10.0),
        max_depth_grid=(3, 4),
        min_child_weight_grid=(2.0, 5.0),
        subsample_grid=(0.80, 1.0),
        colsample_bytree_grid=(0.80, 1.0),
    )
    if cfg.model_version != XGBOOST_V4_MODEL_VERSION:
        raise ValueError(f"xgboost-v4 config must use model_version={XGBOOST_V4_MODEL_VERSION!r}")

    return _train_xgboost_v4_like_ensemble(
        dataset_dir,
        output_dir,
        cfg=cfg,
        ensemble_seeds=ensemble_seeds,
    )


def train_xgboost_v5(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: XGBoostV1Config | None = None,
    ensemble_seeds: tuple[int, ...] = (0, 17, 42, 101, 257),
) -> XGBoostV1Report:
    """Train xgboost-v5 on the same v4 feature schema.

    v5 keeps the v4 32-feature contract and time-series CV but widens the
    regularised search and uses a larger seed ensemble for more stable
    probabilities. The v5 promotion differentiator is family-aware calibration
    (fitted as a separate step on the validation split), not the tree schema.
    """

    if not ensemble_seeds:
        raise ValueError("ensemble_seeds must not be empty")
    cfg = config or XGBoostV1Config(
        model_version=XGBOOST_V5_MODEL_VERSION,
        rounds_grid=(200, 300),
        learning_rate_grid=(0.03, 0.05),
        l2_penalty_grid=(5.0, 10.0, 20.0),
        max_depth_grid=(3, 4),
        min_child_weight_grid=(2.0, 5.0),
        subsample_grid=(0.80, 1.0),
        colsample_bytree_grid=(0.80, 1.0),
    )
    if cfg.model_version != XGBOOST_V5_MODEL_VERSION:
        raise ValueError(f"xgboost-v5 config must use model_version={XGBOOST_V5_MODEL_VERSION!r}")

    return _train_xgboost_v4_like_ensemble(
        dataset_dir,
        output_dir,
        cfg=cfg,
        ensemble_seeds=ensemble_seeds,
    )


def _train_xgboost_v4_like_ensemble(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    cfg: XGBoostV1Config,
    ensemble_seeds: tuple[int, ...],
) -> XGBoostV1Report:
    """Shared time-series CV + seed-ensemble pipeline for v4/v5 on the v4 schema."""

    dataset = _load_dataset(dataset_dir)
    feature_columns = tuple(dataset["manifest"]["feature_columns"])
    _require_xgboost_v4_feature_columns(feature_columns)
    _validate_feature_columns(dataset["tables"], feature_columns)
    _require_xgboost_v4_feature_values(dataset["tables"])
    train_rows = dataset["tables"]["train"].to_pylist()
    val_rows = dataset["tables"]["val"].to_pylist()
    if not train_rows:
        raise ValueError("train split must contain at least one row")

    eval_rows = val_rows or train_rows
    eval_labels = _labels(eval_rows)
    train_dmatrix = _dmatrix(train_rows, feature_columns, labels=_labels(train_rows))
    eval_dmatrix = _dmatrix(eval_rows, feature_columns)
    candidates: list[tuple[tuple[float, float, int], dict[str, float | int | str]]] = []
    for idx, params in enumerate(_parameter_space(cfg)):
        candidate_params = dict(params)
        rounds = int(candidate_params.pop("rounds"))
        booster = xgb.train(
            params=candidate_params,
            dtrain=train_dmatrix,
            num_boost_round=rounds,
            verbose_eval=False,
        )
        probabilities = [float(value) for value in booster.predict(eval_dmatrix)]
        metrics = _metrics(eval_labels, probabilities)
        brier = metrics["brier_score"]
        auc = metrics["roc_auc"]
        candidate_params["rounds"] = rounds
        candidates.append(
            (
                (
                    float("inf") if brier is None else float(brier),
                    -(-1.0 if auc is None else float(auc)),
                    idx,
                ),
                candidate_params,
            )
        )

    _, best_params = min(candidates, key=lambda pair: pair[0])
    cv_rows = sorted(train_rows + val_rows, key=lambda row: int(row["feature_ts"]))
    cv_summary = _time_series_cv_summary(
        cv_rows,
        feature_columns,
        best_params,
        model_version=cfg.model_version,
    )
    boosters: list[xgb.Booster] = []
    started = time.perf_counter()
    for seed in ensemble_seeds:
        member_params = {**best_params, "seed": int(seed)}
        booster = _train_booster(train_rows, feature_columns, member_params)
        _attach_model_attrs(
            booster,
            feature_columns,
            member_params,
            model_version=cfg.model_version,
        )
        boosters.append(booster)
    training_elapsed_seconds = time.perf_counter() - started
    model = XGBoostV1Model(
        model_version=cfg.model_version,
        feature_columns=feature_columns,
        booster=boosters[0],
        params=best_params,
        ensemble_boosters=tuple(boosters),
    )
    single_model = XGBoostV1Model(
        model_version=cfg.model_version,
        feature_columns=feature_columns,
        booster=boosters[0],
        params=best_params,
    )
    metrics_by_split: dict[str, dict[str, float | int | None]] = {}
    single_model_metrics_by_split: dict[str, dict[str, float | int | None]] = {}
    family_metrics_by_split: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for split in SPLITS:
        rows = dataset["tables"][split].to_pylist()
        probabilities = model.predict_proba_many(rows)
        metrics_by_split[split] = _metrics_with_pnl(rows, probabilities)
        single_probabilities = single_model.predict_proba_many(rows)
        single_model_metrics_by_split[split] = _metrics_with_pnl(rows, single_probabilities)
        family_metrics_by_split[split] = _metrics_by_market_family(rows, probabilities)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    member_files = []
    for seed, booster in zip(ensemble_seeds, boosters, strict=True):
        filename = f"model_seed_{int(seed)}.json"
        booster.save_model(str(target / filename))
        member_files.append({"seed": int(seed), "path": filename})
    _write_ensemble_model(
        target / "model.json",
        model_version=cfg.model_version,
        feature_columns=feature_columns,
        params=best_params,
        members=member_files,
    )
    feature_importance = _ensemble_feature_importance(boosters)
    ensemble_summary = {
        "schema_version": XGBOOST_ENSEMBLE_SCHEMA_VERSION,
        "model_version": cfg.model_version,
        "member_count": len(boosters),
        "seeds": list(ensemble_seeds),
        "training_elapsed_seconds": training_elapsed_seconds,
        "train_time_multiplier_estimate": len(boosters),
        "inference_eval_multiplier": len(boosters),
        "single_model_metrics": single_model_metrics_by_split,
        "ensemble_metrics": metrics_by_split,
        "ensemble_vs_single": _ensemble_vs_single_summary(
            single_model_metrics_by_split,
            metrics_by_split,
        ),
    }
    report = XGBoostV1Report(
        model_version=cfg.model_version,
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        best_params=best_params,
        metrics=metrics_by_split,
        family_metrics=family_metrics_by_split,
        feature_importance=feature_importance,
        output_dir=str(target),
        cv_summary=cv_summary,
        ensemble=ensemble_summary,
    )
    (target / "xgboost_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "metrics.json").write_text(
        json.dumps(metrics_by_split, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "family_metrics.json").write_text(
        json.dumps(family_metrics_by_split, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "feature_importance.json").write_text(
        json.dumps(feature_importance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "cv_summary.json").write_text(
        json.dumps(cv_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "ensemble_summary.json").write_text(
        json.dumps(ensemble_summary, indent=2, sort_keys=True),
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


def load_xgboost_v1_model(path: Path | str) -> XGBoostV1Model:
    """Load a saved native XGBoost-v1 booster artifact."""

    wrapper = _read_ensemble_model(path)
    if wrapper is not None:
        root = Path(path).parent
        boosters = []
        for member in wrapper["members"]:
            booster = xgb.Booster()
            booster.load_model(str(root / str(member["path"])))
            boosters.append(booster)
        if not boosters:
            raise ValueError("XGBoost ensemble artifact has no members")
        return XGBoostV1Model(
            model_version=str(wrapper["model_version"]),
            feature_columns=tuple(str(column) for column in wrapper["feature_columns"]),
            booster=boosters[0],
            params={str(key): value for key, value in wrapper.get("params", {}).items()},
            ensemble_boosters=tuple(boosters),
        )
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
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "rounds": rounds,
        }
        for rounds in config.rounds_grid
        for learning_rate in config.learning_rate_grid
        for l2_penalty in config.l2_penalty_grid
        for max_depth in config.max_depth_grid
        for min_split_loss in config.min_split_loss_grid
        for min_child_weight in config.min_child_weight_grid
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


def _train_booster(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    params: dict[str, float | int | str],
) -> xgb.Booster:
    booster_params = dict(params)
    rounds = int(booster_params.pop("rounds"))
    return xgb.train(
        params=booster_params,
        dtrain=_dmatrix(rows, feature_columns, labels=_labels(rows)),
        num_boost_round=rounds,
        verbose_eval=False,
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


def _require_xgboost_v4_feature_columns(feature_columns: tuple[str, ...]) -> None:
    missing_by_group = {
        "market structure": [
            feature for feature in XGBOOST_V4_REQUIRED_MARKET_FEATURES if feature not in feature_columns
        ],
        "issue #57 added": [
            feature for feature in XGBOOST_V4_REQUIRED_ADDED_FEATURES if feature not in feature_columns
        ],
        "tick": [
            feature for feature in XGBOOST_V4_REQUIRED_TICK_FEATURES if feature not in feature_columns
        ],
    }
    missing_parts = [
        f"{group}: {', '.join(missing)}"
        for group, missing in missing_by_group.items()
        if missing
    ]
    if missing_parts:
        raise ValueError(
            "xgboost-v4 dataset manifest missing required feature_columns: "
            + "; ".join(missing_parts)
        )


def _require_xgboost_v4_feature_values(tables: dict[str, Any]) -> None:
    missing_parts = []
    for split in SPLITS:
        rows = tables[split].to_pylist()
        missing = [
            feature
            for feature in XGBOOST_V4_REQUIRED_FEATURES
            if not any(_as_float(row.get(feature)) is not None for row in rows)
        ]
        if missing:
            missing_parts.append(f"{split}: {', '.join(missing)}")
    if missing_parts:
        raise ValueError(
            "xgboost-v4 dataset splits missing finite required feature values: "
            + "; ".join(missing_parts)
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


def _ensemble_feature_importance(boosters: list[xgb.Booster]) -> list[dict[str, float | int | str]]:
    gains_by_feature: dict[str, list[float]] = {}
    splits_by_feature: dict[str, int] = {}
    for booster in boosters:
        gains = booster.get_score(importance_type="gain")
        weights = booster.get_score(importance_type="weight")
        for feature, gain in gains.items():
            gains_by_feature.setdefault(feature, []).append(float(gain))
        for feature, split_count in weights.items():
            splits_by_feature[feature] = splits_by_feature.get(feature, 0) + int(split_count)
    rows = [
        {
            "feature": feature,
            "gain": sum(gains) / len(gains),
            "split_count": splits_by_feature.get(feature, 0),
        }
        for feature, gains in gains_by_feature.items()
    ]
    rows.sort(key=lambda item: (-float(item["gain"]), str(item["feature"])))
    return rows


def _metrics_with_pnl(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    metrics = dict(_metrics(_labels(rows), probabilities))
    selected_returns = [
        realized_return
        for row, probability in zip(rows, probabilities, strict=True)
        if probability >= threshold
        and (realized_return := _as_float(row.get("realized_return"))) is not None
    ]
    pnl = sum(selected_returns)
    metrics.update(
        {
            "pnl_threshold": threshold,
            "trade_count": len(selected_returns),
            "pnl": pnl,
            "avg_realized_return": (
                pnl / len(selected_returns) if selected_returns else None
            ),
        }
    )
    return metrics


def _ensemble_vs_single_summary(
    single_model_metrics_by_split: dict[str, dict[str, float | int | None]],
    ensemble_metrics_by_split: dict[str, dict[str, float | int | None]],
) -> dict[str, Any]:
    single_test = single_model_metrics_by_split.get("test", {})
    ensemble_test = ensemble_metrics_by_split.get("test", {})
    single_brier = _as_float(single_test.get("brier_score"))
    ensemble_brier = _as_float(ensemble_test.get("brier_score"))
    single_auc = _as_float(single_test.get("roc_auc"))
    ensemble_auc = _as_float(ensemble_test.get("roc_auc"))
    single_pnl = _as_float(single_test.get("pnl"))
    ensemble_pnl = _as_float(ensemble_test.get("pnl"))
    brier_delta = (
        None if single_brier is None or ensemble_brier is None else ensemble_brier - single_brier
    )
    roc_auc_delta = None if single_auc is None or ensemble_auc is None else ensemble_auc - single_auc
    pnl_delta = None if single_pnl is None or ensemble_pnl is None else ensemble_pnl - single_pnl
    acceptable = (
        (brier_delta is not None and brier_delta <= 0.0)
        or (roc_auc_delta is not None and roc_auc_delta >= 0.0)
        or (pnl_delta is not None and pnl_delta >= 0.0)
    )
    return {
        "split": "test",
        "acceptable": acceptable,
        "brier_delta": brier_delta,
        "roc_auc_delta": roc_auc_delta,
        "pnl_delta": pnl_delta,
        "rule": (
            "pass if ensemble test Brier is no worse, ROC AUC is no worse, "
            "or PnL is no worse than the first single model member"
        ),
    }


def _time_series_cv_summary(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    params: dict[str, float | int | str],
    *,
    model_version: str,
    max_folds: int = 3,
) -> dict[str, Any]:
    if len(rows) < 6:
        return {
            "model_version": model_version,
            "folds": [],
            "summary": {"fold_count": 0},
            "reason": "not enough rows for time-series CV",
        }
    fold_count = min(max_folds, max(1, len(rows) // 4))
    time_groups = _rows_by_feature_ts(rows)
    if len(time_groups) < 2:
        return {
            "model_version": model_version,
            "folds": [],
            "summary": {"fold_count": 0},
            "reason": "not enough timestamp groups for time-series CV",
        }
    fold_count = min(fold_count, len(time_groups) - 1)
    fold_rows = []
    for fold_idx in range(fold_count):
        train_end = max(1, int(len(time_groups) * (fold_idx + 1) / (fold_count + 2)))
        val_end = max(train_end + 1, int(len(time_groups) * (fold_idx + 2) / (fold_count + 2)))
        train_fold = _flatten_time_groups(time_groups[:train_end])
        val_fold = _flatten_time_groups(time_groups[train_end:val_end])
        if not val_fold:
            continue
        booster = _train_booster(train_fold, feature_columns, params)
        probabilities = [
            float(value)
            for value in booster.predict(_dmatrix(val_fold, feature_columns))
        ]
        metrics = _metrics_with_pnl(val_fold, probabilities)
        fold_rows.append(
            {
                "fold": len(fold_rows) + 1,
                "train_start_ts": int(train_fold[0]["feature_ts"]),
                "train_end_ts": int(train_fold[-1]["feature_ts"]),
                "val_start_ts": int(val_fold[0]["feature_ts"]),
                "val_end_ts": int(val_fold[-1]["feature_ts"]),
                "train_count": len(train_fold),
                "val_count": len(val_fold),
                "metrics": metrics,
            }
        )
    briers = [
        float(row["metrics"]["brier_score"])
        for row in fold_rows
        if row["metrics"].get("brier_score") is not None
    ]
    aucs = [
        float(row["metrics"]["roc_auc"])
        for row in fold_rows
        if row["metrics"].get("roc_auc") is not None
    ]
    pnls = [
        float(row["metrics"]["pnl"])
        for row in fold_rows
        if row["metrics"].get("pnl") is not None
    ]
    return {
        "model_version": model_version,
        "folds": fold_rows,
        "summary": {
            "fold_count": len(fold_rows),
            "brier_mean": _mean(briers),
            "brier_std": _std(briers),
            "roc_auc_mean": _mean(aucs),
            "roc_auc_std": _std(aucs),
            "pnl_mean": _mean(pnls),
            "pnl_std": _std(pnls),
        },
    }


def _rows_by_feature_ts(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    previous_ts: int | None = None
    for row in sorted(rows, key=lambda item: int(item["feature_ts"])):
        feature_ts = int(row["feature_ts"])
        if previous_ts is None or feature_ts != previous_ts:
            groups.append([])
            previous_ts = feature_ts
        groups[-1].append(row)
    return groups


def _flatten_time_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for group in groups for row in group]


def _write_ensemble_model(
    path: Path,
    *,
    model_version: str,
    feature_columns: tuple[str, ...],
    params: dict[str, float | int | str],
    members: list[dict[str, int | str]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": XGBOOST_ENSEMBLE_SCHEMA_VERSION,
                "model_version": model_version,
                "feature_columns": list(feature_columns),
                "params": params,
                "members": members,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _read_ensemble_model(path: Path | str) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != XGBOOST_ENSEMBLE_SCHEMA_VERSION:
        return None
    members = data.get("members")
    feature_columns = data.get("feature_columns")
    if not isinstance(members, list) or not isinstance(feature_columns, list):
        raise ValueError("malformed XGBoost ensemble artifact")
    return data


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _first_or_none(value: Any) -> str | None:
    if isinstance(value, list | tuple) and value:
        return _optional_str(value[0])
    return _optional_str(value)
