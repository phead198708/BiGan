"""XGBoost v6 multi-head trainer for settlement and volatility labels."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.modeling.families import market_family_from_symbol
from bigan.serving import write_feature_schema_artifact

from .logistic import _as_float, _load_dataset, _metrics, _optional_str
from .xgboost_v1 import (
    SPLITS,
    XGBOOST_V4_REQUIRED_FEATURES,
    _attach_model_attrs,
    _dmatrix,
    _feature_importance,
    _first_or_none,
    _mean,
    _require_xgboost_v4_feature_columns,
    _require_xgboost_v4_feature_values,
)

XGBOOST_V6_MODEL_VERSION = "xgboost-v6"
XGBOOST_V6_ARTIFACT_SCHEMA_VERSION = "xgboost_v6_multihead_v1"
SETTLEMENT_CLASSES: tuple[str, ...] = ("UP", "DOWN", "NEUTRAL")
SETTLEMENT_CLASS_TO_ID: dict[str, int] = {
    label: idx for idx, label in enumerate(SETTLEMENT_CLASSES)
}
VOLATILITY_UP_LABEL = "label_volatility_up"
VOLATILITY_DOWN_LABEL = "label_volatility_down"


@dataclass(frozen=True, slots=True)
class XGBoostV6Config:
    """Training and gate-search configuration for xgboost-v6."""

    model_version: str = XGBOOST_V6_MODEL_VERSION
    rounds_grid: tuple[int, ...] = (150, 250)
    learning_rate_grid: tuple[float, ...] = (0.03, 0.05)
    l2_penalty_grid: tuple[float, ...] = (5.0, 10.0, 20.0)
    max_depth_grid: tuple[int, ...] = (3, 4)
    min_split_loss_grid: tuple[float, ...] = (0.0,)
    min_child_weight_grid: tuple[float, ...] = (2.0, 5.0)
    subsample_grid: tuple[float, ...] = (0.80, 1.0)
    colsample_bytree_grid: tuple[float, ...] = (0.80, 1.0)
    temperature_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)
    threshold_up_grid: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65)
    neutral_cap_grid: tuple[float, ...] = (0.25, 0.35, 0.45)
    volatility_threshold_grid: tuple[float, ...] = (0.50, 0.60, 0.70)
    round_trip_cost: float = 0.072
    ev_margin: float = 0.01
    family_temperature_min_samples: int = 25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.model_version != XGBOOST_V6_MODEL_VERSION:
            raise ValueError(
                f"xgboost-v6 config must use model_version={XGBOOST_V6_MODEL_VERSION!r}"
            )
        if not self.rounds_grid:
            raise ValueError("rounds_grid must not be empty")
        if not self.learning_rate_grid:
            raise ValueError("learning_rate_grid must not be empty")
        if not self.l2_penalty_grid:
            raise ValueError("l2_penalty_grid must not be empty")
        if not self.max_depth_grid:
            raise ValueError("max_depth_grid must not be empty")
        if not self.temperature_grid:
            raise ValueError("temperature_grid must not be empty")
        if not self.threshold_up_grid:
            raise ValueError("threshold_up_grid must not be empty")
        if not self.neutral_cap_grid:
            raise ValueError("neutral_cap_grid must not be empty")
        if not self.volatility_threshold_grid:
            raise ValueError("volatility_threshold_grid must not be empty")
        if any(rounds <= 0 for rounds in self.rounds_grid):
            raise ValueError("rounds_grid values must be positive")
        if any(rate <= 0.0 for rate in self.learning_rate_grid):
            raise ValueError("learning_rate_grid values must be positive")
        if any(penalty < 0.0 for penalty in self.l2_penalty_grid):
            raise ValueError("l2_penalty_grid values must be non-negative")
        if any(depth <= 0 for depth in self.max_depth_grid):
            raise ValueError("max_depth_grid values must be positive")
        if any(value <= 0.0 for value in self.temperature_grid):
            raise ValueError("temperature_grid values must be positive")
        if any(value < 0.0 or value > 1.0 for value in self.threshold_up_grid):
            raise ValueError("threshold_up_grid values must be in [0, 1]")
        if any(value < 0.0 or value > 1.0 for value in self.neutral_cap_grid):
            raise ValueError("neutral_cap_grid values must be in [0, 1]")
        if any(value < 0.0 or value > 1.0 for value in self.volatility_threshold_grid):
            raise ValueError("volatility_threshold_grid values must be in [0, 1]")
        if self.round_trip_cost < 0.0:
            raise ValueError("round_trip_cost must be non-negative")
        if self.ev_margin < 0.0:
            raise ValueError("ev_margin must be non-negative")
        if self.family_temperature_min_samples <= 0:
            raise ValueError("family_temperature_min_samples must be positive")

    def to_dict(self) -> dict[str, Any]:
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
            "temperature_grid": list(self.temperature_grid),
            "threshold_up_grid": list(self.threshold_up_grid),
            "neutral_cap_grid": list(self.neutral_cap_grid),
            "volatility_threshold_grid": list(self.volatility_threshold_grid),
            "round_trip_cost": self.round_trip_cost,
            "ev_margin": self.ev_margin,
            "family_temperature_min_samples": self.family_temperature_min_samples,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class XGBoostV6VolatilityHead:
    """Loadable independent volatility head."""

    name: str
    label_column: str
    params: dict[str, float | int | str]
    sample_count: int
    positive_rate: float | None
    booster: xgb.Booster | None = None
    constant_probability: float | None = None

    def predict_many(
        self,
        rows: list[dict[str, Any]],
        feature_columns: tuple[str, ...],
    ) -> list[float]:
        if not rows:
            return []
        if self.booster is None:
            probability = 0.0 if self.constant_probability is None else self.constant_probability
            return [float(probability)] * len(rows)
        values = self.booster.predict(_dmatrix(rows, feature_columns))
        return [_clip_probability(float(value)) for value in values]

    def to_artifact(self, path: str | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "label_column": self.label_column,
            "path": path,
            "params": self.params,
            "sample_count": self.sample_count,
            "positive_rate": self.positive_rate,
            "constant_probability": self.constant_probability,
            "head_type": "xgboost" if self.booster is not None else "constant",
        }


@dataclass(frozen=True, slots=True)
class XGBoostV6Model:
    """Loadable v6 model with explicit settlement and volatility probabilities."""

    model_version: str
    feature_columns: tuple[str, ...]
    settlement_booster: xgb.Booster
    settlement_params: dict[str, float | int | str]
    settlement_temperature: float
    family_temperatures: dict[str, float]
    volatility_up_head: XGBoostV6VolatilityHead
    volatility_down_head: XGBoostV6VolatilityHead
    volatility_gain_priors: dict[str, float]

    def predict_payload(self, row: dict[str, Any]) -> dict[str, float | str]:
        return self.predict_payload_many([row])[0]

    def predict_payload_many(self, rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
        if not rows:
            return []
        settlement_probabilities = self.predict_settlement_proba_many(rows)
        p_vol_up = self.volatility_up_head.predict_many(rows, self.feature_columns)
        p_vol_down = self.volatility_down_head.predict_many(rows, self.feature_columns)
        payloads: list[dict[str, float | str]] = []
        for probs, vol_up, vol_down in zip(
            settlement_probabilities,
            p_vol_up,
            p_vol_down,
            strict=True,
        ):
            class_idx = int(np.argmax(np.asarray(probs, dtype=float)))
            payloads.append(
                {
                    "model_version": self.model_version,
                    "p_up": float(probs[SETTLEMENT_CLASS_TO_ID["UP"]]),
                    "p_down": float(probs[SETTLEMENT_CLASS_TO_ID["DOWN"]]),
                    "p_neutral": float(probs[SETTLEMENT_CLASS_TO_ID["NEUTRAL"]]),
                    "p_vol_up": float(vol_up),
                    "p_vol_down": float(vol_down),
                    "settlement_class": SETTLEMENT_CLASSES[class_idx],
                }
            )
        return payloads

    def predict_settlement_proba_many(
        self,
        rows: list[dict[str, Any]],
    ) -> list[tuple[float, float, float]]:
        raw = np.asarray(
            self.settlement_booster.predict(_dmatrix(rows, self.feature_columns)),
            dtype=float,
        )
        if raw.ndim == 1:
            raw = raw.reshape(1, len(SETTLEMENT_CLASSES))
        calibrated: list[tuple[float, float, float]] = []
        for row, probs in zip(rows, raw, strict=True):
            temperature = self.family_temperatures.get(
                _family(row),
                self.settlement_temperature,
            )
            scaled = _temperature_scale(probs, temperature)
            calibrated.append(tuple(float(value) for value in scaled))
        return calibrated

    def top_feature_contributions(
        self,
        row: dict[str, Any],
        *,
        limit: int = 5,
    ) -> list[dict[str, float | int | str]]:
        contributions = self.settlement_booster.predict(
            _dmatrix([row], self.feature_columns),
            pred_contribs=True,
        )[0]
        if np.asarray(contributions).ndim == 2:
            contributions = np.asarray(contributions, dtype=float).mean(axis=0)
        weights = self.settlement_booster.get_score(importance_type="weight")
        rows = [
            {
                "feature": feature,
                "contribution": float(contributions[idx]),
                "abs_contribution": abs(float(contributions[idx])),
                "split_count": int(weights.get(feature, 0)),
            }
            for idx, feature in enumerate(self.feature_columns)
            if float(contributions[idx]) != 0.0 or weights.get(feature, 0)
        ]
        rows.sort(key=lambda item: (-float(item["abs_contribution"]), str(item["feature"])))
        return rows[:limit]


@dataclass(frozen=True, slots=True)
class XGBoostV6Report:
    """Training report for the v6 settlement + volatility heads."""

    model_version: str
    dataset_version: str | None
    feature_columns: tuple[str, ...]
    best_params: dict[str, float | int | str]
    calibration: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    family_metrics: dict[str, dict[str, Any]]
    volatility_metrics: dict[str, dict[str, Any]]
    joint_rule: dict[str, Any]
    cost_adjusted_backtest: dict[str, dict[str, Any]]
    v5_comparison: dict[str, Any]
    feature_parity: dict[str, Any]
    coverage: dict[str, Any]
    output_dir: str
    feature_importance: list[dict[str, float | int | str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_columns": list(self.feature_columns),
            "best_params": self.best_params,
            "calibration": self.calibration,
            "metrics": self.metrics,
            "family_metrics": self.family_metrics,
            "volatility_metrics": self.volatility_metrics,
            "joint_rule": self.joint_rule,
            "cost_adjusted_backtest": self.cost_adjusted_backtest,
            "v5_comparison": self.v5_comparison,
            "feature_parity": self.feature_parity,
            "coverage": self.coverage,
            "output_dir": self.output_dir,
            "feature_importance": self.feature_importance,
        }


def train_xgboost_v6(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: XGBoostV6Config | None = None,
) -> XGBoostV6Report:
    """Train xgboost-v6 settlement and volatility heads and save artifacts."""

    cfg = config or XGBoostV6Config()
    dataset = _load_dataset(dataset_dir)
    feature_columns = tuple(str(column) for column in dataset["manifest"]["feature_columns"])
    _validate_v6_dataset(dataset["tables"], feature_columns)
    train_rows = dataset["tables"]["train"].to_pylist()
    val_rows = dataset["tables"]["val"].to_pylist()
    if not train_rows:
        raise ValueError("train split must contain at least one row")

    eval_rows = val_rows or train_rows
    train_labels = [_settlement_label_id(row) for row in train_rows]
    eval_labels = [_settlement_label_id(row) for row in eval_rows]
    train_matrix = _dmatrix(train_rows, feature_columns, labels=train_labels)
    eval_matrix = _dmatrix(eval_rows, feature_columns)

    candidates: list[tuple[tuple[float, int], xgb.Booster, dict[str, float | int | str], float]] = []
    for idx, params in enumerate(_parameter_space(cfg, objective="settlement")):
        booster_params = dict(params)
        rounds = int(booster_params.pop("rounds"))
        booster = xgb.train(
            params=booster_params,
            dtrain=train_matrix,
            num_boost_round=rounds,
            verbose_eval=False,
        )
        raw_probs = _raw_settlement_probabilities(booster, eval_matrix)
        temperature = _fit_temperature(eval_labels, raw_probs, cfg.temperature_grid)
        calibrated = [_temperature_scale(probs, temperature) for probs in raw_probs]
        metrics = _settlement_metrics(eval_rows, calibrated)
        params["rounds"] = rounds
        candidates.append(((_metric_sort_value(metrics["log_loss"]), idx), booster, params, temperature))

    _, settlement_booster, best_params, global_temperature = min(
        candidates,
        key=lambda item: item[0],
    )
    _attach_model_attrs(
        settlement_booster,
        feature_columns,
        best_params,
        model_version=cfg.model_version,
    )
    val_raw_probs = _raw_settlement_probabilities(settlement_booster, eval_matrix)
    family_temperatures = _fit_family_temperatures(
        eval_rows,
        eval_labels,
        val_raw_probs,
        cfg,
    )
    volatility_gain_priors = _volatility_gain_priors(train_rows)
    volatility_up_head = _train_volatility_head(
        train_rows,
        feature_columns,
        VOLATILITY_UP_LABEL,
        name="volatility_up",
        cfg=cfg,
    )
    volatility_down_head = _train_volatility_head(
        train_rows,
        feature_columns,
        VOLATILITY_DOWN_LABEL,
        name="volatility_down",
        cfg=cfg,
    )
    model = XGBoostV6Model(
        model_version=cfg.model_version,
        feature_columns=feature_columns,
        settlement_booster=settlement_booster,
        settlement_params=best_params,
        settlement_temperature=global_temperature,
        family_temperatures=family_temperatures,
        volatility_up_head=volatility_up_head,
        volatility_down_head=volatility_down_head,
        volatility_gain_priors=volatility_gain_priors,
    )

    payloads_by_split: dict[str, list[dict[str, float | str]]] = {}
    metrics_by_split: dict[str, dict[str, Any]] = {}
    family_metrics_by_split: dict[str, dict[str, Any]] = {}
    volatility_metrics_by_split: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        rows = dataset["tables"][split].to_pylist()
        payloads = model.predict_payload_many(rows)
        settlement_probabilities = [_settlement_tuple(payload) for payload in payloads]
        payloads_by_split[split] = payloads
        metrics_by_split[split] = _settlement_metrics(rows, settlement_probabilities)
        family_metrics_by_split[split] = _family_settlement_metrics(rows, settlement_probabilities)
        volatility_metrics_by_split[split] = {
            "up": _volatility_metrics(rows, payloads, side="up"),
            "down": _volatility_metrics(rows, payloads, side="down"),
        }

    joint_rule = _select_joint_rule(
        val_rows or train_rows,
        payloads_by_split["val"] if val_rows else payloads_by_split["train"],
        cfg,
        volatility_gain_priors,
    )
    cost_adjusted_backtest = {
        split: _cost_adjusted_backtest(
            dataset["tables"][split].to_pylist(),
            payloads_by_split[split],
            joint_rule=joint_rule,
            round_trip_cost=cfg.round_trip_cost,
            ev_margin=cfg.ev_margin,
            gain_priors=volatility_gain_priors,
        )
        for split in SPLITS
    }
    v5_comparison = _v5_comparison_report(
        {split: dataset["tables"][split].to_pylist() for split in SPLITS},
        payloads_by_split,
        joint_rule=joint_rule,
        round_trip_cost=cfg.round_trip_cost,
        ev_margin=cfg.ev_margin,
        gain_priors=volatility_gain_priors,
    )
    feature_parity = _feature_parity_report(dataset["manifest"], feature_columns)
    coverage = _coverage_report(dataset["manifest"], {split: dataset["tables"][split].to_pylist() for split in SPLITS})

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    feature_importance = _feature_importance(settlement_booster)
    report = XGBoostV6Report(
        model_version=cfg.model_version,
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        feature_columns=feature_columns,
        best_params=best_params,
        calibration={
            "method": "family-aware temperature scaling with global fallback",
            "global_temperature": global_temperature,
            "family_temperatures": family_temperatures,
        },
        metrics=metrics_by_split,
        family_metrics=family_metrics_by_split,
        volatility_metrics=volatility_metrics_by_split,
        joint_rule=joint_rule,
        cost_adjusted_backtest=cost_adjusted_backtest,
        v5_comparison=v5_comparison,
        feature_parity=feature_parity,
        coverage=coverage,
        output_dir=str(target),
        feature_importance=feature_importance,
    )

    settlement_booster.save_model(str(target / "settlement_model.json"))
    if volatility_up_head.booster is not None:
        volatility_up_head.booster.save_model(str(target / "volatility_up_model.json"))
    if volatility_down_head.booster is not None:
        volatility_down_head.booster.save_model(str(target / "volatility_down_model.json"))
    _write_v6_model_artifact(
        target / "model.json",
        model=model,
        up_model_path="volatility_up_model.json" if volatility_up_head.booster is not None else None,
        down_model_path=(
            "volatility_down_model.json" if volatility_down_head.booster is not None else None
        ),
    )
    (target / "xgboost_v6_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for filename, payload in {
        "metrics.json": metrics_by_split,
        "family_metrics.json": family_metrics_by_split,
        "volatility_metrics.json": volatility_metrics_by_split,
        "cost_adjusted_backtest.json": cost_adjusted_backtest,
        "v5_comparison.json": v5_comparison,
        "feature_importance.json": feature_importance,
        "manifest.json": report.to_dict(),
    }.items():
        (target / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    write_feature_schema_artifact(
        target / "feature_schema.json",
        feature_columns,
        feature_version=_first_or_none(dataset["manifest"].get("feature_versions")),
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        model_version=cfg.model_version,
    )
    _write_executor_integration_doc(target / "executor_integration.md", report)
    return report


def load_xgboost_v6_model(path: Path | str) -> XGBoostV6Model:
    """Load a saved xgboost-v6 multi-head artifact."""

    root = Path(path).parent
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != XGBOOST_V6_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("not an xgboost-v6 multihead artifact")
    settlement_booster = xgb.Booster()
    settlement_booster.load_model(str(root / str(artifact["settlement"]["path"])))
    up_head = _load_volatility_head(root, artifact["volatility_up"])
    down_head = _load_volatility_head(root, artifact["volatility_down"])
    return XGBoostV6Model(
        model_version=str(artifact["model_version"]),
        feature_columns=tuple(str(column) for column in artifact["feature_columns"]),
        settlement_booster=settlement_booster,
        settlement_params={str(key): value for key, value in artifact["settlement"]["params"].items()},
        settlement_temperature=float(artifact["calibration"]["global_temperature"]),
        family_temperatures={
            str(key): float(value)
            for key, value in artifact["calibration"].get("family_temperatures", {}).items()
        },
        volatility_up_head=up_head,
        volatility_down_head=down_head,
        volatility_gain_priors={
            str(key): float(value)
            for key, value in artifact.get("volatility_gain_priors", {}).items()
        },
    )


def _validate_v6_dataset(tables: dict[str, Any], feature_columns: tuple[str, ...]) -> None:
    _require_xgboost_v4_feature_columns(feature_columns)
    _validate_v6_feature_columns(tables, feature_columns)
    _require_xgboost_v4_feature_values(tables)
    for split, table in tables.items():
        names = set(table.schema.names)
        required_labels = {
            "label_settlement_3way",
            VOLATILITY_UP_LABEL,
            VOLATILITY_DOWN_LABEL,
        }
        missing = sorted(required_labels - names)
        if missing:
            raise ValueError(f"{split} split missing v6 label columns: {', '.join(missing)}")
    train_labels = [
        _settlement_label(row)
        for row in tables["train"].to_pylist()
    ]
    missing_classes = sorted(set(SETTLEMENT_CLASSES) - set(train_labels))
    if missing_classes:
        raise ValueError(
            "train split must include all settlement classes for multiclass v6: "
            + ", ".join(missing_classes)
        )


def _validate_v6_feature_columns(tables: dict[str, Any], feature_columns: tuple[str, ...]) -> None:
    for split, table in tables.items():
        missing = sorted(set(feature_columns) - set(table.schema.names))
        if missing:
            raise ValueError(f"{split} split missing feature columns: {', '.join(missing)}")


def _parameter_space(
    config: XGBoostV6Config,
    *,
    objective: str,
) -> list[dict[str, float | int | str]]:
    base = {
        "tree_method": "hist",
        "seed": config.seed,
        "nthread": 1,
    }
    if objective == "settlement":
        base.update(
            {
                "objective": "multi:softprob",
                "num_class": len(SETTLEMENT_CLASSES),
                "eval_metric": "mlogloss",
            }
        )
    elif objective == "volatility":
        base.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    else:
        raise ValueError(f"unknown xgboost-v6 objective: {objective}")
    return [
        {
            **base,
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


def _train_volatility_head(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    label_column: str,
    *,
    name: str,
    cfg: XGBoostV6Config,
) -> XGBoostV6VolatilityHead:
    labeled_rows = [
        row
        for row in rows
        if _binary_label(row.get(label_column)) is not None
    ]
    labels = [
        int(_binary_label(row.get(label_column)) or 0)
        for row in labeled_rows
    ]
    if not labeled_rows:
        return XGBoostV6VolatilityHead(
            name=name,
            label_column=label_column,
            params={},
            sample_count=0,
            positive_rate=None,
            constant_probability=0.0,
        )
    positive_rate = sum(labels) / len(labels)
    if len(set(labels)) < 2:
        return XGBoostV6VolatilityHead(
            name=name,
            label_column=label_column,
            params={},
            sample_count=len(labels),
            positive_rate=positive_rate,
            constant_probability=positive_rate,
        )
    params = _parameter_space(cfg, objective="volatility")[0]
    booster_params = dict(params)
    rounds = int(booster_params.pop("rounds"))
    booster = xgb.train(
        params=booster_params,
        dtrain=_dmatrix(labeled_rows, feature_columns, labels=labels),
        num_boost_round=rounds,
        verbose_eval=False,
    )
    params["rounds"] = rounds
    _attach_model_attrs(
        booster,
        feature_columns,
        params,
        model_version=f"{cfg.model_version}:{name}",
    )
    return XGBoostV6VolatilityHead(
        name=name,
        label_column=label_column,
        params=params,
        sample_count=len(labels),
        positive_rate=positive_rate,
        booster=booster,
    )


def _settlement_label(row: dict[str, Any]) -> str:
    value = row.get("label_settlement_3way")
    if value is None:
        raise ValueError("row missing label_settlement_3way")
    if isinstance(value, str):
        label = value.upper()
    else:
        try:
            label = SETTLEMENT_CLASSES[int(value)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"unsupported label_settlement_3way value: {value!r}") from exc
    if label not in SETTLEMENT_CLASS_TO_ID:
        raise ValueError(f"unsupported label_settlement_3way value: {value!r}")
    return label


def _settlement_label_id(row: dict[str, Any]) -> int:
    return SETTLEMENT_CLASS_TO_ID[_settlement_label(row)]


def _binary_label(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        return None
    return bool(value)


def _raw_settlement_probabilities(booster: xgb.Booster, matrix: xgb.DMatrix) -> list[np.ndarray]:
    values = np.asarray(booster.predict(matrix), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, len(SETTLEMENT_CLASSES))
    return [np.asarray(row, dtype=float) for row in values]


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    logits = np.log(clipped) / max(float(temperature), 1e-12)
    logits -= float(np.max(logits))
    exp_values = np.exp(logits)
    total = float(np.sum(exp_values))
    if total <= 0.0 or not math.isfinite(total):
        return np.asarray([1.0 / len(SETTLEMENT_CLASSES)] * len(SETTLEMENT_CLASSES))
    return exp_values / total


def _fit_temperature(
    labels: list[int],
    raw_probabilities: list[np.ndarray],
    grid: tuple[float, ...],
) -> float:
    best_temperature = float(grid[0])
    best_loss = float("inf")
    for temperature in grid:
        probabilities = [
            _temperature_scale(raw, float(temperature))
            for raw in raw_probabilities
        ]
        loss = _multiclass_log_loss(labels, probabilities)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def _fit_family_temperatures(
    rows: list[dict[str, Any]],
    labels: list[int],
    raw_probabilities: list[np.ndarray],
    cfg: XGBoostV6Config,
) -> dict[str, float]:
    grouped: dict[str, tuple[list[int], list[np.ndarray]]] = {}
    for row, label, raw in zip(rows, labels, raw_probabilities, strict=True):
        family_labels, family_raw = grouped.setdefault(_family(row), ([], []))
        family_labels.append(label)
        family_raw.append(raw)
    temperatures: dict[str, float] = {}
    for family, (family_labels, family_raw) in sorted(grouped.items()):
        if len(family_labels) < cfg.family_temperature_min_samples:
            continue
        temperatures[family] = _fit_temperature(family_labels, family_raw, cfg.temperature_grid)
    return temperatures


def _settlement_metrics(
    rows: list[dict[str, Any]],
    probabilities: list[tuple[float, float, float] | np.ndarray],
) -> dict[str, Any]:
    labels = [_settlement_label_id(row) for row in rows]
    if not labels:
        return {
            "sample_count": 0,
            "class_counts": dict.fromkeys(SETTLEMENT_CLASSES, 0),
            "accuracy": None,
            "log_loss": None,
            "multiclass_brier": None,
            "per_class": {},
            "per_class_ece": {},
            "reliability_curves": {},
        }
    predicted = [int(np.argmax(np.asarray(probs, dtype=float))) for probs in probabilities]
    per_class: dict[str, Any] = {}
    reliability: dict[str, Any] = {}
    per_class_ece: dict[str, float | None] = {}
    for class_name, class_idx in SETTLEMENT_CLASS_TO_ID.items():
        class_labels = [1 if label == class_idx else 0 for label in labels]
        class_probs = [float(np.asarray(probs, dtype=float)[class_idx]) for probs in probabilities]
        class_metrics = _metrics(class_labels, class_probs)
        per_class[class_name] = class_metrics
        per_class_ece[class_name] = class_metrics["ece"]
        reliability[class_name] = _reliability_curve(class_labels, class_probs)
    return {
        "sample_count": len(labels),
        "class_counts": {
            class_name: sum(1 for label in labels if label == class_idx)
            for class_name, class_idx in SETTLEMENT_CLASS_TO_ID.items()
        },
        "accuracy": sum(1 for truth, guess in zip(labels, predicted, strict=True) if truth == guess)
        / len(labels),
        "log_loss": _multiclass_log_loss(labels, probabilities),
        "multiclass_brier": _multiclass_brier(labels, probabilities),
        "per_class": per_class,
        "per_class_ece": per_class_ece,
        "reliability_curves": reliability,
        "high_confidence_neutral_low": _high_confidence_neutral_low(rows, probabilities),
    }


def _family_settlement_metrics(
    rows: list[dict[str, Any]],
    probabilities: list[tuple[float, float, float] | np.ndarray],
) -> dict[str, Any]:
    grouped: dict[str, tuple[list[dict[str, Any]], list[tuple[float, float, float] | np.ndarray]]] = {}
    for row, probs in zip(rows, probabilities, strict=True):
        family_rows, family_probs = grouped.setdefault(_family(row), ([], []))
        family_rows.append(row)
        family_probs.append(probs)
    return {
        family: _settlement_metrics(family_rows, family_probs)
        for family, (family_rows, family_probs) in sorted(grouped.items())
    }


def _volatility_metrics(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    *,
    side: str,
) -> dict[str, Any]:
    label_column = VOLATILITY_UP_LABEL if side == "up" else VOLATILITY_DOWN_LABEL
    probability_key = "p_vol_up" if side == "up" else "p_vol_down"
    gain_column = "max_exit_gain_up" if side == "up" else "max_exit_gain_down"
    labeled: list[tuple[dict[str, Any], dict[str, float | str], int]] = []
    for row, payload in zip(rows, payloads, strict=True):
        label = _binary_label(row.get(label_column))
        if label is not None:
            labeled.append((row, payload, int(label)))
    labels = [label for _, _, label in labeled]
    probabilities = [float(payload[probability_key]) for _, payload, _ in labeled]
    baseline = _trivial_volatility_baseline([row for row, _, _ in labeled])
    gains = [
        _as_float(row.get(gain_column))
        for row, _, _ in labeled
        if _as_float(row.get(gain_column)) is not None
    ]
    high_probability_gains = [
        _as_float(row.get(gain_column))
        for row, payload, _ in labeled
        if float(payload[probability_key]) >= 0.70
        and _as_float(row.get(gain_column)) is not None
    ]
    return {
        "label_column": label_column,
        "sample_count": len(labels),
        "positive_count": sum(labels),
        "base_rate": (sum(labels) / len(labels) if labels else None),
        "learned": _metrics(labels, probabilities),
        "trivial_baseline": _metrics(labels, baseline),
        "baseline_name": "recent realized vol + spread + OBI/velocity momentum",
        "bucket_hit_rate": _bucket_hit_rate(labels, probabilities),
        "trivial_bucket_hit_rate": _bucket_hit_rate(labels, baseline),
        "max_exit_gain": _distribution(gains),
        "high_probability_max_exit_gain": _distribution(high_probability_gains),
        "beats_trivial_baseline": _beats_baseline(
            _metrics(labels, probabilities),
            _metrics(labels, baseline),
        ),
        "family": _family_volatility_metrics([row for row, _, _ in labeled], labels, probabilities),
    }


def _family_volatility_metrics(
    rows: list[dict[str, Any]],
    labels: list[int],
    probabilities: list[float],
) -> dict[str, Any]:
    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for row, label, probability in zip(rows, labels, probabilities, strict=True):
        family_labels, family_probs = grouped.setdefault(_family(row), ([], []))
        family_labels.append(label)
        family_probs.append(probability)
    return {
        family: _metrics(family_labels, family_probs)
        for family, (family_labels, family_probs) in sorted(grouped.items())
    }


def _select_joint_rule(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    cfg: XGBoostV6Config,
    gain_priors: dict[str, float],
) -> dict[str, Any]:
    best_rule: dict[str, Any] | None = None
    for settlement_threshold in cfg.threshold_up_grid:
        for neutral_cap in cfg.neutral_cap_grid:
            for volatility_threshold in cfg.volatility_threshold_grid:
                rule = {
                    "settlement_threshold": settlement_threshold,
                    "neutral_cap": neutral_cap,
                    "volatility_threshold": volatility_threshold,
                    "round_trip_cost": cfg.round_trip_cost,
                    "ev_margin": cfg.ev_margin,
                    "rule": (
                        "UP if p_up is side max, p_up >= settlement_threshold, "
                        "p_neutral <= neutral_cap, p_vol_up >= volatility_threshold, "
                        "and p_vol_up * avg_train_max_exit_gain_up clears cost+margin; "
                        "DOWN is symmetric with explicit p_down and p_vol_down"
                    ),
                    "p_vol_gate_vs_sizing": (
                        "p_vol_* is an entry gate only; issue #90 running-bankroll "
                        "and min_size sizing remain independent"
                    ),
                }
                summary = _cost_adjusted_backtest(
                    rows,
                    payloads,
                    joint_rule=rule,
                    round_trip_cost=cfg.round_trip_cost,
                    ev_margin=cfg.ev_margin,
                    gain_priors=gain_priors,
                )
                candidate = {**rule, "validation": summary}
                if best_rule is None or (
                    float(summary["pnl"]) > float(best_rule["validation"]["pnl"])
                    or (
                        float(summary["pnl"]) == float(best_rule["validation"]["pnl"])
                        and int(summary["trade_count"]) > int(best_rule["validation"]["trade_count"])
                    )
                ):
                    best_rule = candidate
    assert best_rule is not None
    return best_rule


def _cost_adjusted_backtest(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    *,
    joint_rule: dict[str, Any],
    round_trip_cost: float,
    ev_margin: float,
    gain_priors: dict[str, float],
) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row, payload in zip(rows, payloads, strict=True):
        side = _joint_decision(
            payload,
            joint_rule=joint_rule,
            round_trip_cost=round_trip_cost,
            ev_margin=ev_margin,
            gain_priors=gain_priors,
        )
        if side is None:
            continue
        entry_cost = _entry_cost(row, side)
        true_label = _settlement_label(row)
        pnl = 1.0 - entry_cost - round_trip_cost if true_label == side else -entry_cost - round_trip_cost
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
        trades.append(
            {
                "side": side,
                "true_label": true_label,
                "entry_cost": entry_cost,
                "pnl": pnl,
                "family": _family(row),
            }
        )
    family: dict[str, dict[str, Any]] = {}
    for trade in trades:
        family_row = family.setdefault(
            str(trade["family"]),
            {"trade_count": 0, "pnl": 0.0, "wins": 0},
        )
        family_row["trade_count"] += 1
        family_row["pnl"] += float(trade["pnl"])
        family_row["wins"] += int(float(trade["pnl"]) > 0.0)
    for row in family.values():
        row["hit_rate"] = row["wins"] / row["trade_count"] if row["trade_count"] else None
    return {
        "metric_of_record": "cost_adjusted_account_cashflow_proxy_pnl",
        "trade_count": len(trades),
        "pnl": cumulative,
        "avg_pnl": cumulative / len(trades) if trades else None,
        "hit_rate": (
            sum(1 for trade in trades if float(trade["pnl"]) > 0.0) / len(trades)
            if trades
            else None
        ),
        "max_drawdown": max_drawdown,
        "family": family,
    }


def _joint_decision(
    payload: dict[str, float | str],
    *,
    joint_rule: dict[str, Any],
    round_trip_cost: float,
    ev_margin: float,
    gain_priors: dict[str, float],
) -> str | None:
    threshold = float(joint_rule["settlement_threshold"])
    neutral_cap = float(joint_rule["neutral_cap"])
    vol_threshold = float(joint_rule["volatility_threshold"])
    p_up = float(payload["p_up"])
    p_down = float(payload["p_down"])
    p_neutral = float(payload["p_neutral"])
    if p_neutral > neutral_cap:
        return None
    up_expected_gain = float(payload["p_vol_up"]) * gain_priors.get("up", 0.0)
    down_expected_gain = float(payload["p_vol_down"]) * gain_priors.get("down", 0.0)
    minimum_gain = round_trip_cost + ev_margin
    if (
        p_up >= p_down
        and p_up >= threshold
        and float(payload["p_vol_up"]) >= vol_threshold
        and up_expected_gain > minimum_gain
    ):
        return "UP"
    if (
        p_down > p_up
        and p_down >= threshold
        and float(payload["p_vol_down"]) >= vol_threshold
        and down_expected_gain > minimum_gain
    ):
        return "DOWN"
    return None


def joint_decision_from_payload(
    payload: dict[str, float | str],
    *,
    joint_rule: dict[str, Any],
    round_trip_cost: float,
    ev_margin: float,
    gain_priors: dict[str, float],
) -> str | None:
    """Return UP/DOWN when the v6 joint gate admits a trade, else None."""

    return _joint_decision(
        payload,
        joint_rule=joint_rule,
        round_trip_cost=round_trip_cost,
        ev_margin=ev_margin,
        gain_priors=gain_priors,
    )


def _v5_comparison_report(
    rows_by_split: dict[str, list[dict[str, Any]]],
    payloads_by_split: dict[str, list[dict[str, float | str]]],
    *,
    joint_rule: dict[str, Any],
    round_trip_cost: float,
    ev_margin: float,
    gain_priors: dict[str, float],
) -> dict[str, Any]:
    has_v5_reference = any(
        _as_float(row.get("v5_prob_up_15m")) is not None
        for rows in rows_by_split.values()
        for row in rows
    )
    report: dict[str, Any] = {
        "required_for_promotion": True,
        "metric_of_record": "cost_adjusted/account-cashflow PnL",
        "feature_set_diff_required_to_be_empty": True,
        "reference_column": "v5_prob_up_15m",
        "available": has_v5_reference,
        "note": (
            "Same-dataset v5_prob_up_15m reference is present; per-family "
            "cost-adjusted comparison is populated below."
            if has_v5_reference
            else "Populate v5_prob_up_15m on the same dataset to produce mandatory "
            "per-family v5-vs-v6 cost-adjusted comparison."
        ),
    }
    if not has_v5_reference:
        return report
    splits: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        v6_summary = _cost_adjusted_backtest(
            rows,
            payloads_by_split[split],
            joint_rule=joint_rule,
            round_trip_cost=round_trip_cost,
            ev_margin=ev_margin,
            gain_priors=gain_priors,
        )
        v5_payloads = [_legacy_v5_payload(row) for row in rows]
        v5_rule = {
            "settlement_threshold": joint_rule["settlement_threshold"],
            "neutral_cap": 1.0,
            "volatility_threshold": 0.0,
        }
        v5_summary = _cost_adjusted_backtest(
            rows,
            v5_payloads,
            joint_rule=v5_rule,
            round_trip_cost=round_trip_cost,
            ev_margin=0.0,
            gain_priors={"up": 1.0, "down": 1.0},
        )
        splits[split] = {
            "xgboost_v6": v6_summary,
            "xgboost_v5_reference": v5_summary,
            "pnl_delta_v6_minus_v5": float(v6_summary["pnl"]) - float(v5_summary["pnl"]),
            "legacy_v5_down_probability_source": "1 - v5_prob_up_15m",
        }
    report["splits"] = splits
    return report


def _legacy_v5_payload(row: dict[str, Any]) -> dict[str, float | str]:
    p_up = _clip_probability(_as_float(row.get("v5_prob_up_15m")) or 0.5)
    p_down = 1.0 - p_up
    return {
        "model_version": "xgboost-v5-reference",
        "p_up": p_up,
        "p_down": p_down,
        "p_neutral": 0.0,
        "p_vol_up": 1.0,
        "p_vol_down": 1.0,
        "settlement_class": "UP" if p_up >= p_down else "DOWN",
    }


def _feature_parity_report(manifest: dict[str, Any], feature_columns: tuple[str, ...]) -> dict[str, Any]:
    reference_columns = tuple(str(column) for column in manifest.get("v5_feature_columns", feature_columns))
    added = sorted(set(feature_columns) - set(reference_columns))
    removed = sorted(set(reference_columns) - set(feature_columns))
    return {
        "reference": (
            "manifest.v5_feature_columns"
            if "v5_feature_columns" in manifest
            else "dataset manifest feature_columns (same input used for v5/v6)"
        ),
        "added": added,
        "removed": removed,
        "empty": not added and not removed,
        "v5_required_missing": sorted(set(XGBOOST_V4_REQUIRED_FEATURES) - set(feature_columns)),
    }


def _coverage_report(
    manifest: dict[str, Any],
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    family_counts: dict[str, dict[str, int]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            family_row = family_counts.setdefault(_family(row), {"total": 0})
            family_row["total"] += 1
            family_row[split] = family_row.get(split, 0) + 1
    label_coverage = {
        split: {
            "settlement_3way": len(rows),
            "volatility_up": sum(1 for row in rows if _binary_label(row.get(VOLATILITY_UP_LABEL)) is not None),
            "volatility_down": sum(1 for row in rows if _binary_label(row.get(VOLATILITY_DOWN_LABEL)) is not None),
        }
        for split, rows in rows_by_split.items()
    }
    return {
        "depends_on_issue_91_v6_label_coverage": True,
        "expected_sample_count_per_family": manifest.get("expected_sample_count_per_family"),
        "v6_label_diagnostics": manifest.get("v6_label_diagnostics"),
        "family_counts": family_counts,
        "label_coverage": label_coverage,
    }


def _write_v6_model_artifact(
    path: Path,
    *,
    model: XGBoostV6Model,
    up_model_path: str | None,
    down_model_path: str | None,
) -> None:
    artifact = {
        "schema_version": XGBOOST_V6_ARTIFACT_SCHEMA_VERSION,
        "model_version": model.model_version,
        "feature_columns": list(model.feature_columns),
        "settlement": {
            "path": "settlement_model.json",
            "params": model.settlement_params,
            "classes": list(SETTLEMENT_CLASSES),
        },
        "calibration": {
            "method": "family-aware temperature scaling with global fallback",
            "global_temperature": model.settlement_temperature,
            "family_temperatures": model.family_temperatures,
        },
        "volatility_up": model.volatility_up_head.to_artifact(path=up_model_path),
        "volatility_down": model.volatility_down_head.to_artifact(path=down_model_path),
        "volatility_gain_priors": model.volatility_gain_priors,
        "serving_payload": [
            "p_up",
            "p_down",
            "p_neutral",
            "p_vol_up",
            "p_vol_down",
            "model_version",
        ],
        "compatibility": {
            "prob_up_15m": "legacy alias for p_up only",
            "down_probability": "must be read from p_down; never derive from 1 - p_up",
        },
    }
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")


def _load_volatility_head(root: Path, artifact: dict[str, Any]) -> XGBoostV6VolatilityHead:
    booster = None
    if artifact.get("path"):
        booster = xgb.Booster()
        booster.load_model(str(root / str(artifact["path"])))
    return XGBoostV6VolatilityHead(
        name=str(artifact["name"]),
        label_column=str(artifact["label_column"]),
        params={str(key): value for key, value in artifact.get("params", {}).items()},
        sample_count=int(artifact.get("sample_count") or 0),
        positive_rate=(
            None if artifact.get("positive_rate") is None else float(artifact["positive_rate"])
        ),
        booster=booster,
        constant_probability=(
            None
            if artifact.get("constant_probability") is None
            else float(artifact["constant_probability"])
        ),
    )


def _write_executor_integration_doc(path: Path, report: XGBoostV6Report) -> None:
    rule = report.joint_rule
    path.write_text(
        "\n".join(
            [
                "# xgboost-v6 executor interface",
                "",
                "Serving payload fields: `p_up`, `p_down`, `p_neutral`, `p_vol_up`, "
                "`p_vol_down`, `model_version`.",
                "",
                "`p_down` is a first-class model output. Executors must not derive DOWN "
                "confidence from `1 - p_up`.",
                "",
                "Entry rule:",
                "",
                "- UP: `p_up` is the settlement max, "
                f"`p_up >= {rule['settlement_threshold']}`, "
                f"`p_neutral <= {rule['neutral_cap']}`, "
                f"`p_vol_up >= {rule['volatility_threshold']}`, and expected max exit gain "
                "clears round-trip cost plus margin.",
                "- DOWN: same rule with explicit `p_down` and `p_vol_down`.",
                "",
                "Issue #90 sleeve linkage:",
                "",
                "- `p_vol_*` is only the entry gate.",
                "- The volatility sleeve still enforces expected max exit gain greater than "
                "round-trip cost plus margin.",
                "- Running-bankroll sizing and the `min_size` floor remain separate controls.",
                "",
                "Promotion go/no-go uses cost-adjusted/account-cashflow PnL, paper first.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _multiclass_log_loss(
    labels: list[int],
    probabilities: list[tuple[float, float, float] | np.ndarray],
) -> float | None:
    if not labels:
        return None
    losses = [
        -math.log(max(1e-12, float(np.asarray(probs, dtype=float)[label])))
        for label, probs in zip(labels, probabilities, strict=True)
    ]
    return sum(losses) / len(losses)


def _multiclass_brier(
    labels: list[int],
    probabilities: list[tuple[float, float, float] | np.ndarray],
) -> float | None:
    if not labels:
        return None
    total = 0.0
    for label, probs in zip(labels, probabilities, strict=True):
        values = np.asarray(probs, dtype=float)
        total += sum(
            (float(probability) - (1.0 if idx == label else 0.0)) ** 2
            for idx, probability in enumerate(values)
        )
    return total / len(labels)


def _high_confidence_neutral_low(
    rows: list[dict[str, Any]],
    probabilities: list[tuple[float, float, float] | np.ndarray],
) -> dict[str, Any]:
    members = [
        (row, np.asarray(probs, dtype=float))
        for row, probs in zip(rows, probabilities, strict=True)
        if max(float(probs[0]), float(probs[1])) >= 0.60
        and float(probs[SETTLEMENT_CLASS_TO_ID["NEUTRAL"]]) <= 0.35
    ]
    if not members:
        return {"sample_count": 0, "accuracy": None}
    correct = 0
    for row, probs in members:
        correct += int(_settlement_label_id(row) == int(np.argmax(probs)))
    return {"sample_count": len(members), "accuracy": correct / len(members)}


def _reliability_curve(
    labels: list[int],
    probabilities: list[float],
    *,
    bin_count: int = 10,
) -> list[dict[str, float | int | None]]:
    rows = []
    for idx in range(bin_count):
        start = idx / bin_count
        end = (idx + 1) / bin_count
        members = [
            (label, probability)
            for label, probability in zip(labels, probabilities, strict=True)
            if (start <= probability < end) or (idx == bin_count - 1 and probability == 1.0)
        ]
        rows.append(
            {
                "bin_lower": start,
                "bin_upper": end,
                "sample_count": len(members),
                "mean_confidence": (
                    sum(probability for _, probability in members) / len(members)
                    if members
                    else None
                ),
                "hit_rate": (
                    sum(label for label, _ in members) / len(members)
                    if members
                    else None
                ),
            }
        )
    return rows


def _bucket_hit_rate(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    buckets = {
        "0.00-0.50": (0.0, 0.5),
        "0.50-0.60": (0.5, 0.6),
        "0.60-0.70": (0.6, 0.7),
        "0.70-1.00": (0.7, 1.0),
    }
    out: dict[str, Any] = {}
    for name, (lower, upper) in buckets.items():
        members = [
            label
            for label, probability in zip(labels, probabilities, strict=True)
            if (lower <= probability < upper)
            or (upper == 1.0 and lower <= probability <= upper)
        ]
        out[name] = {
            "sample_count": len(members),
            "hit_rate": sum(members) / len(members) if members else None,
        }
    return out


def _trivial_volatility_baseline(rows: list[dict[str, Any]]) -> list[float]:
    raw_scores = []
    for row in rows:
        raw_scores.append(
            sum(
                value
                for value in (
                    abs(_as_float(row.get("rv_30m")) or 0.0),
                    abs(_as_float(row.get("tick_price_velocity")) or 0.0),
                    abs(_as_float(row.get("tick_obi_l1")) or 0.0),
                    abs(_as_float(row.get("tick_spread")) or _as_float(row.get("spread")) or 0.0),
                )
            )
        )
    if not raw_scores:
        return []
    lower = min(raw_scores)
    upper = max(raw_scores)
    if upper == lower:
        return [0.5] * len(raw_scores)
    return [(score - lower) / (upper - lower) for score in raw_scores]


def _beats_baseline(learned: dict[str, Any], baseline: dict[str, Any]) -> bool | None:
    learned_pr = _as_float(learned.get("pr_auc"))
    baseline_pr = _as_float(baseline.get("pr_auc"))
    if learned_pr is not None and baseline_pr is not None:
        return learned_pr >= baseline_pr
    learned_auc = _as_float(learned.get("roc_auc"))
    baseline_auc = _as_float(baseline.get("roc_auc"))
    if learned_auc is not None and baseline_auc is not None:
        return learned_auc >= baseline_auc
    return None


def _distribution(values: list[float | None]) -> dict[str, float | int | None]:
    present = sorted(float(value) for value in values if value is not None)
    if not present:
        return {
            "sample_count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    return {
        "sample_count": len(present),
        "mean": _mean(present),
        "p50": _quantile(present, 0.5),
        "p90": _quantile(present, 0.9),
        "max": max(present),
    }


def _quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _volatility_gain_priors(rows: list[dict[str, Any]]) -> dict[str, float]:
    up_values = [
        value
        for row in rows
        if _binary_label(row.get(VOLATILITY_UP_LABEL))
        and (value := _as_float(row.get("max_exit_gain_up"))) is not None
    ]
    down_values = [
        value
        for row in rows
        if _binary_label(row.get(VOLATILITY_DOWN_LABEL))
        and (value := _as_float(row.get("max_exit_gain_down"))) is not None
    ]
    return {
        "up": _mean(up_values) or 0.0,
        "down": _mean(down_values) or 0.0,
    }


def _entry_cost(row: dict[str, Any], side: str) -> float:
    side_lower = side.lower()
    side_columns = (
        f"entry_cost_{side_lower}",
        f"{side_lower}_entry_price",
        f"{side_lower}_ask",
        f"ask_{side_lower}",
    )
    for column in (*side_columns, "entry_price"):
        value = _as_float(row.get(column))
        if value is not None:
            return _clip_probability(value, lower=0.0, upper=1.0)
    mid_price = _as_float(row.get("mid_price"))
    if mid_price is not None:
        return _clip_probability(mid_price if side == "UP" else 1.0 - mid_price, lower=0.0, upper=1.0)
    implied = _as_float(row.get("market_implied_prob"))
    if implied is not None:
        return _clip_probability(implied if side == "UP" else 1.0 - implied, lower=0.0, upper=1.0)
    return 0.5


def _settlement_tuple(payload: dict[str, float | str]) -> tuple[float, float, float]:
    return (float(payload["p_up"]), float(payload["p_down"]), float(payload["p_neutral"]))


def _family(row: dict[str, Any]) -> str:
    return market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))


def _clip_probability(
    value: float,
    *,
    lower: float = 1e-6,
    upper: float = 1.0 - 1e-6,
) -> float:
    return min(upper, max(lower, float(value)))


def _metric_sort_value(value: Any) -> float:
    as_float = _as_float(value)
    return float("inf") if as_float is None else as_float
