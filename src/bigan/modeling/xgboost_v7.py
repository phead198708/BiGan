"""XGBoost v7 settlement model with tradable convergence reporting.

v7 keeps a calibrated settlement probability head for diagnostics, then adds
side-specific convergence heads that estimate future executable exit value for
UP and DOWN. Executors should use the emitted convergence mispricing fields for
entries; raw settlement probabilities stay as diagnostics.
"""

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

from .logistic import _as_float, _load_dataset, _optional_str
from .xgboost_v1 import (
    SPLITS,
    _attach_model_attrs,
    _dmatrix,
    _feature_importance,
    _feature_value,
    _first_or_none,
    _require_xgboost_v4_feature_columns,
    _require_xgboost_v4_feature_values,
)
from .xgboost_v6 import (
    SETTLEMENT_CLASS_TO_ID,
    SETTLEMENT_CLASSES,
    VOLATILITY_DOWN_LABEL,
    VOLATILITY_UP_LABEL,
    _family_settlement_metrics,
    _fit_family_temperatures,
    _fit_temperature,
    _metric_sort_value,
    _raw_settlement_probabilities,
    _settlement_label,
    _settlement_label_id,
    _settlement_metrics,
    _temperature_scale,
    _validate_v6_feature_columns,
)

XGBOOST_V7_MODEL_VERSION = "xgboost-v7"
XGBOOST_V7_ARTIFACT_SCHEMA_VERSION = "xgboost_v7_convergence_v1"
ROUND_AGE_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("0-180s", 0.0, 180.0),
    ("180-360s", 180.0, 360.0),
    ("360-540s", 360.0, 540.0),
    ("540s+", 540.0, None),
)
ENTRY_PRICE_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("<0.50", 0.0, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.85", 0.70, 0.85),
    ("0.85+", 0.85, None),
)
CONFIDENCE_TARGETS: tuple[str, ...] = (
    "hit_5c_before_loss_10c",
    "hit_10c_before_loss_10c",
    "loss_10c_before_hit_5c",
)


@dataclass(frozen=True, slots=True)
class XGBoostV7Config:
    """Training and evaluation settings for xgboost-v7."""

    model_version: str = XGBOOST_V7_MODEL_VERSION
    rounds_grid: tuple[int, ...] = (150, 250)
    learning_rate_grid: tuple[float, ...] = (0.03, 0.05)
    l2_penalty_grid: tuple[float, ...] = (5.0, 10.0)
    max_depth_grid: tuple[int, ...] = (3, 4)
    min_child_weight_grid: tuple[float, ...] = (2.0, 5.0)
    subsample_grid: tuple[float, ...] = (0.80, 1.0)
    colsample_bytree_grid: tuple[float, ...] = (0.80, 1.0)
    confidence_rounds_grid: tuple[int, ...] = (80,)
    confidence_learning_rate_grid: tuple[float, ...] = (0.05,)
    confidence_l2_penalty_grid: tuple[float, ...] = (5.0,)
    confidence_max_depth_grid: tuple[int, ...] = (2,)
    confidence_min_child_weight_grid: tuple[float, ...] = (5.0,)
    confidence_subsample_grid: tuple[float, ...] = (0.85,)
    confidence_colsample_bytree_grid: tuple[float, ...] = (0.85,)
    temperature_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5)
    settlement_threshold_grid: tuple[float, ...] = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
    edge_threshold_grid: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.082, 0.10, 0.12)
    gate_selection_min_trades_per_split: int = 5
    gate_selection_min_avg_pnl: float = 0.08
    buy_slippage: float = 0.02
    fee_bps: float = 0.0
    ev_margin: float = 0.01
    min_seconds_to_expiry: float = 300.0
    max_seconds_to_expiry: float = 1200.0
    no_new_entry_before_expiry_seconds: float = 300.0
    family_temperature_min_samples: int = 25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.model_version != XGBOOST_V7_MODEL_VERSION:
            raise ValueError(f"model_version must be {XGBOOST_V7_MODEL_VERSION!r}")
        for name in (
            "rounds_grid",
            "learning_rate_grid",
            "l2_penalty_grid",
            "max_depth_grid",
            "min_child_weight_grid",
            "subsample_grid",
            "colsample_bytree_grid",
            "confidence_rounds_grid",
            "confidence_learning_rate_grid",
            "confidence_l2_penalty_grid",
            "confidence_max_depth_grid",
            "confidence_min_child_weight_grid",
            "confidence_subsample_grid",
            "confidence_colsample_bytree_grid",
            "temperature_grid",
            "settlement_threshold_grid",
            "edge_threshold_grid",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.buy_slippage < 0.0:
            raise ValueError("buy_slippage must be non-negative")
        if self.fee_bps < 0.0:
            raise ValueError("fee_bps must be non-negative")
        if self.gate_selection_min_trades_per_split < 1:
            raise ValueError("gate_selection_min_trades_per_split must be positive")
        if self.gate_selection_min_avg_pnl < 0.0:
            raise ValueError("gate_selection_min_avg_pnl must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "rounds_grid": list(self.rounds_grid),
            "learning_rate_grid": list(self.learning_rate_grid),
            "l2_penalty_grid": list(self.l2_penalty_grid),
            "max_depth_grid": list(self.max_depth_grid),
            "min_child_weight_grid": list(self.min_child_weight_grid),
            "subsample_grid": list(self.subsample_grid),
            "colsample_bytree_grid": list(self.colsample_bytree_grid),
            "confidence_rounds_grid": list(self.confidence_rounds_grid),
            "confidence_learning_rate_grid": list(self.confidence_learning_rate_grid),
            "confidence_l2_penalty_grid": list(self.confidence_l2_penalty_grid),
            "confidence_max_depth_grid": list(self.confidence_max_depth_grid),
            "confidence_min_child_weight_grid": list(self.confidence_min_child_weight_grid),
            "confidence_subsample_grid": list(self.confidence_subsample_grid),
            "confidence_colsample_bytree_grid": list(self.confidence_colsample_bytree_grid),
            "temperature_grid": list(self.temperature_grid),
            "settlement_threshold_grid": list(self.settlement_threshold_grid),
            "edge_threshold_grid": list(self.edge_threshold_grid),
            "gate_selection_min_trades_per_split": self.gate_selection_min_trades_per_split,
            "gate_selection_min_avg_pnl": self.gate_selection_min_avg_pnl,
            "buy_slippage": self.buy_slippage,
            "fee_bps": self.fee_bps,
            "ev_margin": self.ev_margin,
            "min_seconds_to_expiry": self.min_seconds_to_expiry,
            "max_seconds_to_expiry": self.max_seconds_to_expiry,
            "no_new_entry_before_expiry_seconds": self.no_new_entry_before_expiry_seconds,
            "family_temperature_min_samples": self.family_temperature_min_samples,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class XGBoostV7Model:
    model_version: str
    feature_columns: tuple[str, ...]
    settlement_booster: xgb.Booster
    settlement_params: dict[str, float | int | str]
    settlement_temperature: float
    family_temperatures: dict[str, float]
    convergence_up_booster: xgb.Booster
    convergence_down_booster: xgb.Booster
    convergence_params: dict[str, float | int | str]
    buy_slippage: float
    fee_bps: float
    ev_margin: float
    confidence_up_boosters: dict[str, xgb.Booster] | None = None
    confidence_down_boosters: dict[str, xgb.Booster] | None = None
    confidence_params: dict[str, float | int | str] | None = None

    def predict_payload(self, row: dict[str, Any]) -> dict[str, float | str | bool | None]:
        return self.predict_payload_many([row])[0]

    def predict_payload_many(self, rows: list[dict[str, Any]]) -> list[dict[str, float | str | bool | None]]:
        if not rows:
            return []
        settlement_probabilities = self.predict_settlement_proba_many(rows)
        convergence_up = self.predict_convergence_many(rows, side="UP")
        convergence_down = self.predict_convergence_many(rows, side="DOWN")
        confidence_up = self.predict_confidence_many(rows, side="UP")
        confidence_down = self.predict_confidence_many(rows, side="DOWN")
        payloads: list[dict[str, float | str | bool | None]] = []
        for row, probs, up_value, down_value, up_confidence, down_confidence in zip(
            rows,
            settlement_probabilities,
            convergence_up,
            convergence_down,
            confidence_up,
            confidence_down,
            strict=True,
        ):
            p_up = float(probs[SETTLEMENT_CLASS_TO_ID["UP"]])
            p_down = float(probs[SETTLEMENT_CLASS_TO_ID["DOWN"]])
            p_neutral = float(probs[SETTLEMENT_CLASS_TO_ID["NEUTRAL"]])
            market = _market_implied_prob(row)
            token_side = _token_side(row)
            p_up_residual = _clip01(up_value)
            p_down_residual = _clip01(down_value)
            up_worst = _entry_worst_price(_entry_ask(row, "UP"), self.buy_slippage, self.fee_bps)
            down_worst = _entry_worst_price(_entry_ask(row, "DOWN"), self.buy_slippage, self.fee_bps)
            up_price = up_worst if up_worst is not None else _side_market_price(market, token_side, "UP")
            down_price = (
                down_worst
                if down_worst is not None
                else _side_market_price(market, token_side, "DOWN")
            )
            up_edge = None if up_price is None else p_up - up_price
            down_edge = None if down_price is None else p_down - down_price
            residual_up_edge = None if up_price is None else p_up_residual - up_price
            residual_down_edge = None if down_price is None else p_down_residual - down_price
            selected_side = _select_edge_side(up_edge=residual_up_edge, down_edge=residual_down_edge)
            selected_edge = residual_up_edge if selected_side == "UP" else residual_down_edge
            model_probability = p_up_residual if selected_side == "UP" else p_down_residual
            polymarket_price = up_price if selected_side == "UP" else down_price
            selected_confidence = up_confidence if selected_side == "UP" else down_confidence
            if polymarket_price is None:
                polymarket_price = market
            mispricing_edge = None if selected_edge is None else float(selected_edge)
            class_idx = int(np.argmax(np.asarray(probs, dtype=float)))
            payloads.append(
                {
                    "model_version": self.model_version,
                    "p_up": p_up,
                    "p_down": p_down,
                    "p_neutral": p_neutral,
                    "settlement_class": SETTLEMENT_CLASSES[class_idx],
                    "settlement_residual": mispricing_edge,
                    "market_implied_prob": market,
                    "token_side": token_side,
                    "model_probability": model_probability,
                    "polymarket_price": float(polymarket_price),
                    "mispricing_edge": mispricing_edge,
                    "token_expected_win_probability": model_probability,
                    "p_up_residual_adjusted": p_up_residual,
                    "p_down_residual_adjusted": p_down_residual,
                    "entry_worst_price_up": up_worst,
                    "entry_worst_price_down": down_worst,
                    "expected_edge_up": up_edge,
                    "expected_edge_down": down_edge,
                    "residual_expected_edge_up": residual_up_edge,
                    "residual_expected_edge_down": residual_down_edge,
                    "p_up_hit_5c_before_loss_10c": up_confidence.get(
                        "hit_5c_before_loss_10c"
                    ),
                    "p_up_hit_10c_before_loss_10c": up_confidence.get(
                        "hit_10c_before_loss_10c"
                    ),
                    "p_up_loss_10c_before_hit_5c": up_confidence.get(
                        "loss_10c_before_hit_5c"
                    ),
                    "p_down_hit_5c_before_loss_10c": down_confidence.get(
                        "hit_5c_before_loss_10c"
                    ),
                    "p_down_hit_10c_before_loss_10c": down_confidence.get(
                        "hit_10c_before_loss_10c"
                    ),
                    "p_down_loss_10c_before_hit_5c": down_confidence.get(
                        "loss_10c_before_hit_5c"
                    ),
                    "selected_hit_5c_before_loss_10c": selected_confidence.get(
                        "hit_5c_before_loss_10c"
                    ),
                    "selected_hit_10c_before_loss_10c": selected_confidence.get(
                        "hit_10c_before_loss_10c"
                    ),
                    "selected_loss_10c_before_hit_5c": selected_confidence.get(
                        "loss_10c_before_hit_5c"
                    ),
                    "selected_confidence_score": _confidence_score(selected_confidence),
                    "selected_side": selected_side,
                    "selected_expected_edge": selected_edge,
                    "should_enter_settlement": (
                        None if selected_edge is None else bool(selected_edge >= self.ev_margin)
                    ),
                }
            )
        return payloads

    def predict_settlement_proba_many(self, rows: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
        raw = _raw_settlement_probabilities(self.settlement_booster, _dmatrix(rows, self.feature_columns))
        calibrated: list[tuple[float, float, float]] = []
        for row, probs in zip(rows, raw, strict=True):
            temperature = self.family_temperatures.get(
                _family(row),
                self.settlement_temperature,
            )
            scaled = _temperature_scale(probs, temperature)
            calibrated.append(tuple(float(value) for value in scaled))
        return calibrated

    def predict_convergence_many(self, rows: list[dict[str, Any]], *, side: str) -> list[float]:
        booster = self.convergence_up_booster if side == "UP" else self.convergence_down_booster
        values = np.asarray(
            booster.predict(_dmatrix(rows, self.feature_columns)),
            dtype=float,
        )
        if values.ndim != 1:
            values = values.reshape(-1)
        return [_clip01(float(value)) for value in values]

    def predict_confidence_many(
        self,
        rows: list[dict[str, Any]],
        *,
        side: str,
    ) -> list[dict[str, float | None]]:
        boosters = self.confidence_up_boosters if side == "UP" else self.confidence_down_boosters
        if not boosters:
            return [
                {target: None for target in CONFIDENCE_TARGETS}
                for _ in rows
            ]
        matrix = _dmatrix(rows, self.feature_columns)
        predictions: dict[str, list[float | None]] = {}
        for target in CONFIDENCE_TARGETS:
            booster = boosters.get(target)
            if booster is None:
                predictions[target] = [None for _ in rows]
                continue
            values = np.asarray(booster.predict(matrix), dtype=float)
            if values.ndim != 1:
                values = values.reshape(-1)
            predictions[target] = [_clip01(float(value)) for value in values]
        return [
            {target: predictions[target][idx] for target in CONFIDENCE_TARGETS}
            for idx in range(len(rows))
        ]


@dataclass(frozen=True, slots=True)
class XGBoostV7Report:
    model_version: str
    dataset_version: str | None
    feature_columns: tuple[str, ...]
    settlement_params: dict[str, float | int | str]
    convergence_params: dict[str, float | int | str]
    calibration: dict[str, Any]
    outcome_metrics: dict[str, dict[str, Any]]
    family_outcome_metrics: dict[str, dict[str, Any]]
    residual_metrics: dict[str, dict[str, Any]]
    confidence_metrics: dict[str, dict[str, Any]]
    tradable_ev_metrics: dict[str, dict[str, Any]]
    selected_tradable_ev_rule: dict[str, Any]
    executor_contract: dict[str, Any]
    feature_importance: list[dict[str, float | int | str]]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_columns": list(self.feature_columns),
            "settlement_params": self.settlement_params,
            "convergence_params": self.convergence_params,
            "calibration": self.calibration,
            "outcome_metrics": self.outcome_metrics,
            "family_outcome_metrics": self.family_outcome_metrics,
            "residual_metrics": self.residual_metrics,
            "confidence_metrics": self.confidence_metrics,
            "tradable_ev_metrics": self.tradable_ev_metrics,
            "selected_tradable_ev_rule": self.selected_tradable_ev_rule,
            "executor_contract": self.executor_contract,
            "feature_importance": self.feature_importance,
            "output_dir": self.output_dir,
        }


def train_xgboost_v7(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    config: XGBoostV7Config | None = None,
) -> XGBoostV7Report:
    """Train xgboost-v7 settlement outcome and residual heads."""

    cfg = config or XGBoostV7Config()
    dataset = _load_dataset(dataset_dir)
    feature_columns = tuple(str(column) for column in dataset["manifest"]["feature_columns"])
    _validate_v7_dataset(dataset["tables"], feature_columns)
    train_rows = dataset["tables"]["train"].to_pylist()
    val_rows = dataset["tables"]["val"].to_pylist()
    if not train_rows:
        raise ValueError("train split must contain at least one row")
    eval_rows = val_rows or train_rows
    train_labels = [_settlement_label_id(row) for row in train_rows]
    eval_labels = [_settlement_label_id(row) for row in eval_rows]
    train_matrix = _dmatrix(train_rows, feature_columns, labels=train_labels)
    eval_matrix = _dmatrix(eval_rows, feature_columns)

    settlement_candidates: list[tuple[tuple[float, int], xgb.Booster, dict[str, Any], float]] = []
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
        metrics = _settlement_metrics(
            eval_rows,
            [_temperature_scale(raw, temperature) for raw in raw_probs],
        )
        params["rounds"] = rounds
        settlement_candidates.append(((_metric_sort_value(metrics["log_loss"]), idx), booster, params, temperature))
    _, settlement_booster, settlement_params, global_temperature = min(
        settlement_candidates,
        key=lambda item: item[0],
    )
    _attach_model_attrs(
        settlement_booster,
        feature_columns,
        settlement_params,
        model_version=cfg.model_version,
    )
    family_temperatures = _fit_family_temperatures(
        eval_rows,
        eval_labels,
        _raw_settlement_probabilities(settlement_booster, eval_matrix),
        _v6_like_config(cfg),
    )

    convergence_up_booster, convergence_params = _train_convergence_head(
        train_rows,
        eval_rows,
        feature_columns,
        cfg,
        side="UP",
    )
    convergence_down_booster, _ = _train_convergence_head(
        train_rows,
        eval_rows,
        feature_columns,
        cfg,
        side="DOWN",
    )
    confidence_up_boosters, confidence_params = _train_confidence_heads(
        train_rows,
        eval_rows,
        feature_columns,
        cfg,
        side="UP",
    )
    confidence_down_boosters, _ = _train_confidence_heads(
        train_rows,
        eval_rows,
        feature_columns,
        cfg,
        side="DOWN",
    )
    model = XGBoostV7Model(
        model_version=cfg.model_version,
        feature_columns=feature_columns,
        settlement_booster=settlement_booster,
        settlement_params=settlement_params,
        settlement_temperature=global_temperature,
        family_temperatures=family_temperatures,
        convergence_up_booster=convergence_up_booster,
        convergence_down_booster=convergence_down_booster,
        convergence_params=convergence_params,
        buy_slippage=cfg.buy_slippage,
        fee_bps=cfg.fee_bps,
        ev_margin=cfg.ev_margin,
        confidence_up_boosters=confidence_up_boosters,
        confidence_down_boosters=confidence_down_boosters,
        confidence_params=confidence_params,
    )

    rows_by_split = {split: dataset["tables"][split].to_pylist() for split in SPLITS}
    payloads_by_split = {split: model.predict_payload_many(rows) for split, rows in rows_by_split.items()}
    outcome_metrics = {
        split: _settlement_metrics(rows_by_split[split], [_settlement_tuple(payload) for payload in payloads_by_split[split]])
        for split in SPLITS
    }
    family_metrics = {
        split: _family_settlement_metrics(rows_by_split[split], [_settlement_tuple(payload) for payload in payloads_by_split[split]])
        for split in SPLITS
    }
    residual_metrics = {
        split: _residual_metrics(rows_by_split[split], payloads_by_split[split], cfg)
        for split in SPLITS
    }
    confidence_metrics = {
        split: _confidence_metrics(rows_by_split[split], payloads_by_split[split], cfg)
        for split in SPLITS
    }
    selected_rule = _select_tradable_ev_rule(rows_by_split, payloads_by_split, cfg)
    probability_reference_rule = {"settlement_threshold": 0.80, "edge_threshold": 0.082}
    tradable_ev_metrics = {
        split: {
            "v7_probability_ev_gate": _tradable_ev_backtest(
                rows_by_split[split],
                payloads_by_split[split],
                probability_reference_rule,
                cfg=cfg,
                probability_prefix="",
            ),
            "v7_convergence_edge_gate": _tradable_ev_backtest(
                rows_by_split[split],
                payloads_by_split[split],
                selected_rule,
                cfg=cfg,
                probability_prefix="residual_",
            ),
            "v6_current_gate_reference": _tradable_ev_backtest(
                rows_by_split[split],
                payloads_by_split[split],
                {"settlement_threshold": 0.80, "edge_threshold": 0.082},
                cfg=cfg,
                probability_prefix="",
            ),
            "zero_skill_market_baseline": _zero_skill_baseline(rows_by_split[split], cfg),
        }
        for split in SPLITS
    }

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    settlement_booster.save_model(str(target / "settlement_model.json"))
    convergence_up_booster.save_model(str(target / "convergence_up_model.json"))
    convergence_down_booster.save_model(str(target / "convergence_down_model.json"))
    _save_confidence_boosters(target, confidence_up_boosters, side="UP")
    _save_confidence_boosters(target, confidence_down_boosters, side="DOWN")
    feature_importance = _feature_importance(settlement_booster)
    report = XGBoostV7Report(
        model_version=cfg.model_version,
        dataset_version=_optional_str(dataset["manifest"].get("dataset_version")),
        feature_columns=feature_columns,
        settlement_params=settlement_params,
        convergence_params=convergence_params,
        calibration={
            "method": "family-aware temperature scaling with global fallback",
            "global_temperature": global_temperature,
            "family_temperatures": family_temperatures,
        },
        outcome_metrics=outcome_metrics,
        family_outcome_metrics=family_metrics,
        residual_metrics=residual_metrics,
        confidence_metrics=confidence_metrics,
        tradable_ev_metrics=tradable_ev_metrics,
        selected_tradable_ev_rule=selected_rule,
        executor_contract=_executor_contract(cfg),
        feature_importance=feature_importance,
        output_dir=str(target),
    )

    _write_v7_model_artifact(target / "model.json", model=model)
    (target / "xgboost_v7_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for filename, payload in {
        "metrics.json": report.to_dict(),
        "outcome_metrics.json": outcome_metrics,
        "family_outcome_metrics.json": family_metrics,
        "residual_metrics.json": residual_metrics,
        "confidence_metrics.json": confidence_metrics,
        "tradable_ev_metrics.json": tradable_ev_metrics,
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


def _validate_v7_dataset(tables: dict[str, Any], feature_columns: tuple[str, ...]) -> None:
    """Validate v7 convergence corpora without requiring all settlement classes.

    The settlement head is retained for diagnostics, but event-driven v7 corpora
    can be built from Polymarket top-of-book history where the underlying
    neutral-margin abstention class is not recoverable. The convergence heads
    only need complete v4-style features plus side-specific future exit labels.
    """

    _require_xgboost_v4_feature_columns(feature_columns)
    _validate_v6_feature_columns(tables, feature_columns)
    _require_xgboost_v4_feature_values(tables)
    required_labels = {
        "label_settlement_3way",
        VOLATILITY_UP_LABEL,
        VOLATILITY_DOWN_LABEL,
    }
    for split, table in tables.items():
        missing = sorted(required_labels - set(table.schema.names))
        if missing:
            raise ValueError(f"{split} split missing v7 label columns: {', '.join(missing)}")
    train_labels = [_settlement_label(row) for row in tables["train"].to_pylist()]
    if not train_labels:
        raise ValueError("train split must contain settlement labels")


def load_xgboost_v7_model(path: Path | str) -> XGBoostV7Model:
    root = Path(path).parent
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != XGBOOST_V7_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("not an xgboost-v7 convergence artifact")
    settlement_booster = xgb.Booster()
    settlement_booster.load_model(str(root / str(artifact["settlement"]["path"])))
    convergence_up_booster = xgb.Booster()
    convergence_down_booster = xgb.Booster()
    convergence_up_booster.load_model(str(root / str(artifact["convergence"]["up_path"])))
    convergence_down_booster.load_model(str(root / str(artifact["convergence"]["down_path"])))
    confidence = artifact.get("confidence") or {}
    confidence_up_boosters = _load_confidence_boosters(
        root,
        confidence.get("up_paths") if isinstance(confidence, dict) else None,
    )
    confidence_down_boosters = _load_confidence_boosters(
        root,
        confidence.get("down_paths") if isinstance(confidence, dict) else None,
    )
    return XGBoostV7Model(
        model_version=str(artifact["model_version"]),
        feature_columns=tuple(str(column) for column in artifact["feature_columns"]),
        settlement_booster=settlement_booster,
        settlement_params={str(key): value for key, value in artifact["settlement"]["params"].items()},
        settlement_temperature=float(artifact["calibration"]["global_temperature"]),
        family_temperatures={
            str(key): float(value)
            for key, value in artifact["calibration"].get("family_temperatures", {}).items()
        },
        convergence_up_booster=convergence_up_booster,
        convergence_down_booster=convergence_down_booster,
        convergence_params={
            str(key): value for key, value in artifact["convergence"]["params"].items()
        },
        buy_slippage=float(artifact["serving_config"]["buy_slippage"]),
        fee_bps=float(artifact["serving_config"]["fee_bps"]),
        ev_margin=float(artifact["serving_config"]["ev_margin"]),
        confidence_up_boosters=confidence_up_boosters,
        confidence_down_boosters=confidence_down_boosters,
        confidence_params={
            str(key): value
            for key, value in (confidence.get("params") or {}).items()
        },
    )


def _load_confidence_boosters(root: Path, paths: Any) -> dict[str, xgb.Booster]:
    if not isinstance(paths, dict):
        return {}
    boosters: dict[str, xgb.Booster] = {}
    for target in CONFIDENCE_TARGETS:
        model_path = paths.get(target)
        if not model_path:
            continue
        booster = xgb.Booster()
        booster.load_model(str(root / str(model_path)))
        boosters[target] = booster
    return boosters


def _parameter_space(config: XGBoostV7Config, *, objective: str) -> list[dict[str, float | int | str]]:
    base: dict[str, float | int | str] = {
        "tree_method": "hist",
        "seed": config.seed,
        "nthread": 1,
    }
    if objective == "settlement":
        base.update({"objective": "multi:softprob", "num_class": len(SETTLEMENT_CLASSES), "eval_metric": "mlogloss"})
    elif objective == "residual":
        base.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
    elif objective == "confidence":
        base.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    else:
        raise ValueError(f"unknown xgboost-v7 objective: {objective}")
    if objective == "confidence":
        return [
            {
                **base,
                "eta": learning_rate,
                "lambda": l2_penalty,
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "colsample_bytree": colsample,
                "rounds": rounds,
            }
            for rounds in config.confidence_rounds_grid
            for learning_rate in config.confidence_learning_rate_grid
            for l2_penalty in config.confidence_l2_penalty_grid
            for max_depth in config.confidence_max_depth_grid
            for min_child_weight in config.confidence_min_child_weight_grid
            for subsample in config.confidence_subsample_grid
            for colsample in config.confidence_colsample_bytree_grid
        ]
    return [
        {
            **base,
            "eta": learning_rate,
            "lambda": l2_penalty,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "rounds": rounds,
        }
        for rounds in config.rounds_grid
        for learning_rate in config.learning_rate_grid
        for l2_penalty in config.l2_penalty_grid
        for max_depth in config.max_depth_grid
        for min_child_weight in config.min_child_weight_grid
        for subsample in config.subsample_grid
        for colsample in config.colsample_bytree_grid
    ]


def _train_convergence_head(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    cfg: XGBoostV7Config,
    *,
    side: str,
) -> tuple[xgb.Booster, dict[str, float | int | str]]:
    train_labels = [_convergence_exit_value_label(row, side=side, cfg=cfg) for row in train_rows]
    train_weights = _convergence_weights(train_rows, side=side, cfg=cfg)
    eval_labels = [_convergence_exit_value_label(row, side=side, cfg=cfg) for row in eval_rows]
    eval_weights = _convergence_weights(eval_rows, side=side, cfg=cfg)
    candidates: list[tuple[tuple[float, int], xgb.Booster, dict[str, Any]]] = []
    for idx, params in enumerate(_parameter_space(cfg, objective="residual")):
        booster_params = dict(params)
        rounds = int(booster_params.pop("rounds"))
        booster = xgb.train(
            params=booster_params,
            dtrain=_weighted_dmatrix(
                train_rows,
                feature_columns,
                labels=train_labels,
                weights=train_weights,
            ),
            num_boost_round=rounds,
            verbose_eval=False,
        )
        predictions = [float(value) for value in booster.predict(_dmatrix(eval_rows, feature_columns))]
        rmse = _weighted_rmse(eval_labels, predictions, eval_weights)
        params["rounds"] = rounds
        candidates.append(((rmse, idx), booster, params))
    _, booster, params = min(candidates, key=lambda item: item[0])
    _attach_model_attrs(
        booster,
        feature_columns,
        params,
        model_version=f"{cfg.model_version}:convergence-{side.lower()}",
    )
    return booster, params


def _train_confidence_heads(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    cfg: XGBoostV7Config,
    *,
    side: str,
) -> tuple[dict[str, xgb.Booster], dict[str, float | int | str]]:
    if not _confidence_labels_available(train_rows, side=side):
        return {}, {}
    boosters: dict[str, xgb.Booster] = {}
    selected_params: dict[str, float | int | str] = {}
    for target in CONFIDENCE_TARGETS:
        booster, params = _train_confidence_head(
            train_rows,
            eval_rows,
            feature_columns,
            cfg,
            side=side,
            target=target,
        )
        if booster is not None:
            boosters[target] = booster
            selected_params = params
    return boosters, selected_params


def _train_confidence_head(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    cfg: XGBoostV7Config,
    *,
    side: str,
    target: str,
) -> tuple[xgb.Booster | None, dict[str, float | int | str]]:
    train_labels = [_confidence_label(row, side=side, target=target) for row in train_rows]
    train_weights = _confidence_weights(train_rows, side=side, target=target, cfg=cfg)
    if sum(train_weights) <= 0.0:
        return None, {}
    eval_labels = [_confidence_label(row, side=side, target=target) for row in eval_rows]
    eval_weights = _confidence_weights(eval_rows, side=side, target=target, cfg=cfg)
    candidates: list[tuple[tuple[float, int], xgb.Booster, dict[str, Any]]] = []
    for idx, params in enumerate(_parameter_space(cfg, objective="confidence")):
        booster_params = dict(params)
        rounds = int(booster_params.pop("rounds"))
        booster = xgb.train(
            params=booster_params,
            dtrain=_weighted_dmatrix(
                train_rows,
                feature_columns,
                labels=[0.0 if item is None else float(item) for item in train_labels],
                weights=train_weights,
            ),
            num_boost_round=rounds,
            verbose_eval=False,
        )
        predictions = [
            _clip01(float(value))
            for value in booster.predict(_dmatrix(eval_rows, feature_columns))
        ]
        loss = _weighted_binary_logloss(
            [0.0 if item is None else float(item) for item in eval_labels],
            predictions,
            eval_weights,
        )
        params["rounds"] = rounds
        candidates.append(((loss, idx), booster, params))
    _, booster, params = min(candidates, key=lambda item: item[0])
    _attach_model_attrs(
        booster,
        feature_columns,
        params,
        model_version=f"{cfg.model_version}:confidence-{side.lower()}-{target}",
    )
    return booster, params


def _weighted_dmatrix(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    *,
    labels: list[float],
    weights: list[float],
) -> xgb.DMatrix:
    matrix = np.asarray(
        [[_feature_value(row, column) for column in feature_columns] for row in rows],
        dtype=float,
    )
    return xgb.DMatrix(
        matrix,
        label=np.asarray(labels, dtype=float),
        weight=np.asarray(weights, dtype=float),
        feature_names=list(feature_columns),
        missing=np.nan,
    )


def _convergence_weights(rows: list[dict[str, Any]], *, side: str, cfg: XGBoostV7Config) -> list[float]:
    weights = [
        max(0.01, abs(_convergence_exit_gain(row, side=side) or 0.0))
        if _execution_eligible(row, cfg) and _convergence_path_valid(row, side=side)
        else 0.0
        for row in rows
    ]
    if sum(weights) <= 0.0:
        return [1.0] * len(rows)
    return weights


def _confidence_weights(
    rows: list[dict[str, Any]],
    *,
    side: str,
    target: str,
    cfg: XGBoostV7Config,
) -> list[float]:
    weights = [
        1.0
        if _execution_eligible(row, cfg)
        and _convergence_path_valid(row, side=side)
        and _confidence_label(row, side=side, target=target) is not None
        else 0.0
        for row in rows
    ]
    if sum(weights) <= 0.0:
        return [0.0] * len(rows)
    return weights


def _convergence_exit_value_label(row: dict[str, Any], *, side: str, cfg: XGBoostV7Config) -> float:
    value = _as_float(row.get("best_exit_price_up" if side == "UP" else "best_exit_price_down"))
    if value is not None:
        return _clip01(value)
    entry = _entry_worst_price(_entry_ask(row, side), cfg.buy_slippage, cfg.fee_bps)
    gain = _convergence_exit_gain(row, side=side)
    if entry is not None and gain is not None:
        return _clip01(entry + gain)
    fallback = _side_market_price(_market_implied_prob(row), _token_side(row), side)
    return _clip01(0.5 if fallback is None else fallback)


def _convergence_exit_gain(row: dict[str, Any], *, side: str) -> float | None:
    return _as_float(row.get("max_exit_gain_up" if side == "UP" else "max_exit_gain_down"))


def _convergence_path_valid(row: dict[str, Any], *, side: str) -> bool:
    return _as_float(row.get("best_exit_price_up" if side == "UP" else "best_exit_price_down")) is not None


def _confidence_labels_available(rows: list[dict[str, Any]], *, side: str) -> bool:
    suffix = "up" if side == "UP" else "down"
    return any(
        f"{target}_{suffix}" in row
        for row in rows
        for target in CONFIDENCE_TARGETS
    )


def _confidence_label(row: dict[str, Any], *, side: str, target: str) -> float | None:
    suffix = "up" if side == "UP" else "down"
    value = row.get(f"{target}_{suffix}")
    if value is None:
        return None
    return 1.0 if bool(value) else 0.0


def _settlement_residual_label(row: dict[str, Any]) -> float:
    return _settlement_win(row) - _market_implied_prob(row)


def _settlement_realized_pnl(row: dict[str, Any], cfg: XGBoostV7Config) -> float | None:
    worst = _entry_worst_price(_entry_ask(row, _token_side(row)), cfg.buy_slippage, cfg.fee_bps)
    if worst is None:
        return None
    return _settlement_win(row) - worst


def _settlement_should_enter(row: dict[str, Any], cfg: XGBoostV7Config) -> bool | None:
    pnl = _settlement_realized_pnl(row, cfg)
    return None if pnl is None else pnl > cfg.ev_margin


def _settlement_win(row: dict[str, Any]) -> float:
    return 1.0 if _settlement_label(row) == _token_side(row) else 0.0


def _market_implied_prob(row: dict[str, Any]) -> float:
    value = _as_float(row.get("market_implied_prob"))
    if value is None:
        value = _as_float(row.get("entry_ask_price"))
    return _clip01(0.5 if value is None else value)


def _entry_ask(row: dict[str, Any], side: str) -> float | None:
    side_specific = _as_float(
        row.get("entry_ask_price_up" if side == "UP" else "entry_ask_price_down")
    )
    if side_specific is not None:
        return _clip01(side_specific)
    ask = _as_float(row.get("entry_ask_price"))
    if ask is None:
        return None
    ask = _clip01(ask)
    token_side = _token_side(row)
    if side == token_side:
        return ask
    return _clip01(1.0 - ask)


def _entry_worst_price(ask: float | None, buy_slippage: float, fee_bps: float) -> float | None:
    if ask is None:
        return None
    fee = float(ask) * fee_bps / 10_000.0
    return max(0.0, min(0.99, float(ask) + buy_slippage + fee))


def _token_side(row: dict[str, Any]) -> str:
    symbol = str(row.get("canonical_symbol") or "")
    side = symbol.rsplit(":", 1)[-1].upper() if ":" in symbol else "UP"
    return side if side in {"UP", "DOWN"} else "UP"


def _family(row: dict[str, Any]) -> str:
    return market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))


def _round_slug(row: dict[str, Any]) -> str:
    symbol = str(row.get("canonical_symbol") or row.get("symbol") or "")
    if ":" in symbol:
        return symbol.split(":", 2)[1]
    return str(row.get("source_market") or symbol)


def _seconds_to_expiry(row: dict[str, Any]) -> float | None:
    feature_ts = row.get("feature_ts")
    round_end_ts = row.get("round_end_ts")
    if feature_ts is None or round_end_ts is None:
        return None
    return max(0.0, (int(round_end_ts) - int(feature_ts)) / 1000.0)


def _seconds_since_round_start(row: dict[str, Any]) -> float | None:
    feature_ts = row.get("feature_ts")
    round_start_ts = row.get("round_start_ts")
    if feature_ts is None or round_start_ts is None:
        return None
    return (int(feature_ts) - int(round_start_ts)) / 1000.0


def _execution_eligible(row: dict[str, Any], cfg: XGBoostV7Config) -> bool:
    if _entry_ask(row, _token_side(row)) is None:
        return False
    seconds_since_start = _seconds_since_round_start(row)
    seconds_to_expiry = _seconds_to_expiry(row)
    if seconds_since_start is None or seconds_to_expiry is None:
        return False
    return (
        seconds_since_start >= 0.0
        and seconds_to_expiry >= cfg.no_new_entry_before_expiry_seconds
        and seconds_to_expiry >= cfg.min_seconds_to_expiry
        and seconds_to_expiry <= cfg.max_seconds_to_expiry
    )


def _residual_metrics(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str | bool | None]],
    cfg: XGBoostV7Config,
) -> dict[str, Any]:
    eligible = [
        idx for idx, row in enumerate(rows)
        if _execution_eligible(row, cfg)
    ]
    valid_up = [idx for idx, row in enumerate(rows) if _convergence_path_valid(row, side="UP")]
    valid_down = [idx for idx, row in enumerate(rows) if _convergence_path_valid(row, side="DOWN")]
    up_labels = [_convergence_exit_value_label(row, side="UP", cfg=cfg) for row in rows]
    down_labels = [_convergence_exit_value_label(row, side="DOWN", cfg=cfg) for row in rows]
    up_predictions = [float(payload["p_up_residual_adjusted"] or 0.0) for payload in payloads]
    down_predictions = [float(payload["p_down_residual_adjusted"] or 0.0) for payload in payloads]
    return {
        "label_formula": "side_convergence_exit_value = future best executable exit price by side",
        "sample_count": len(rows),
        "execution_eligible_count": len(eligible),
        "path_valid_count_up": len(valid_up),
        "path_valid_count_down": len(valid_down),
        "up_exit_value_rmse": _rmse([up_labels[idx] for idx in valid_up], [up_predictions[idx] for idx in valid_up]),
        "up_exit_value_mae": _mae([up_labels[idx] for idx in valid_up], [up_predictions[idx] for idx in valid_up]),
        "down_exit_value_rmse": _rmse(
            [down_labels[idx] for idx in valid_down],
            [down_predictions[idx] for idx in valid_down],
        ),
        "down_exit_value_mae": _mae(
            [down_labels[idx] for idx in valid_down],
            [down_predictions[idx] for idx in valid_down],
        ),
        "up_take_profit_02_count": sum(
            1 for row in rows if (_convergence_exit_gain(row, side="UP") or 0.0) >= 0.02
        ),
        "down_take_profit_02_count": sum(
            1 for row in rows if (_convergence_exit_gain(row, side="DOWN") or 0.0) >= 0.02
        ),
    }


def _confidence_metrics(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str | bool | None]],
    cfg: XGBoostV7Config,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "label_formula": (
            "side barrier confidence: first executable +5c/+10c hit before "
            "first executable -10c adverse loss"
        ),
        "sample_count": len(rows),
    }
    for side in ("UP", "DOWN"):
        side_key = side.lower()
        valid = [
            idx for idx, row in enumerate(rows)
            if _execution_eligible(row, cfg)
            and _convergence_path_valid(row, side=side)
        ]
        metrics[f"{side_key}_valid_count"] = len(valid)
        for target in CONFIDENCE_TARGETS:
            labels = [_confidence_label(rows[idx], side=side, target=target) for idx in valid]
            predictions = [
                _confidence_payload_probability(payloads[idx], side=side, target=target)
                for idx in valid
            ]
            paired = [
                (float(label), float(prediction))
                for label, prediction in zip(labels, predictions, strict=True)
                if label is not None and prediction is not None
            ]
            metrics[f"{side_key}_{target}"] = _binary_metrics(paired)
    return metrics


def _select_tradable_ev_rule(
    rows_by_split: dict[str, list[dict[str, Any]]],
    payloads_by_split: dict[str, list[dict[str, float | str | bool | None]]],
    cfg: XGBoostV7Config,
) -> dict[str, Any]:
    selection_splits = [
        split
        for split in ("train", "val")
        if rows_by_split.get(split) and payloads_by_split.get(split)
    ]
    if not selection_splits:
        raise ValueError("at least one train/val split is required for v7 gate selection")
    validation_split = "val" if "val" in selection_splits else selection_splits[-1]
    candidates: list[dict[str, Any]] = []
    for edge in cfg.edge_threshold_grid:
        rule = {
            "settlement_threshold": 0.0,
            "edge_threshold": edge,
            "gate": "convergence_edge",
        }
        summaries = {
            split: _tradable_ev_backtest(
                rows_by_split[split],
                payloads_by_split[split],
                rule,
                cfg=cfg,
                probability_prefix="residual_",
            )
            for split in selection_splits
        }
        diagnostics = _gate_selection_diagnostics(
            summaries,
            min_trades_per_split=cfg.gate_selection_min_trades_per_split,
            min_avg_pnl=cfg.gate_selection_min_avg_pnl,
        )
        candidates.append(
            {
                **rule,
                "selection_method": "train_val_stability_min_avg_path_pnl",
                "selection_splits": selection_splits,
                "selection_min_trades_per_split": cfg.gate_selection_min_trades_per_split,
                "selection_min_avg_pnl": cfg.gate_selection_min_avg_pnl,
                "selection_score": _gate_selection_score(diagnostics, summaries, validation_split),
                "selection_diagnostics": diagnostics,
                "selection_metrics": summaries,
                "validation": summaries[validation_split],
            }
        )
    best = max(
        candidates,
        key=lambda candidate: tuple(candidate["selection_score"]),
    )
    assert best is not None
    return {
        **best,
        "candidate_count": len(candidates),
        "top_candidates": _compact_gate_candidates(candidates),
    }


def _gate_selection_diagnostics(
    summaries: dict[str, dict[str, Any]],
    *,
    min_trades_per_split: int,
    min_avg_pnl: float,
) -> dict[str, Any]:
    split_names = list(summaries)
    trade_counts = {split: int(summary["trade_count"]) for split, summary in summaries.items()}
    pnls = {split: float(summary["pnl"]) for split, summary in summaries.items()}
    avg_pnls = {
        split: summary.get("avg_pnl")
        for split, summary in summaries.items()
    }
    positive_all_splits = all(pnls[split] > 0.0 for split in split_names)
    enough_trades_all_splits = all(
        trade_counts[split] >= min_trades_per_split
        for split in split_names
    )
    min_split_avg_pnl = (
        min(float(value) for value in avg_pnls.values() if value is not None)
        if avg_pnls and all(value is not None for value in avg_pnls.values())
        else None
    )
    strong_average_all_splits = (
        min_split_avg_pnl is not None
        and min_split_avg_pnl >= min_avg_pnl
    )
    return {
        "positive_all_splits": positive_all_splits,
        "enough_trades_all_splits": enough_trades_all_splits,
        "strong_average_all_splits": strong_average_all_splits,
        "preferred": positive_all_splits and enough_trades_all_splits and strong_average_all_splits,
        "min_trade_count": min(trade_counts.values()) if trade_counts else 0,
        "min_pnl": min(pnls.values()) if pnls else 0.0,
        "min_avg_pnl": min_split_avg_pnl,
        "trade_counts": trade_counts,
        "pnls": pnls,
        "avg_pnls": avg_pnls,
    }


def _gate_selection_score(
    diagnostics: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    validation_split: str,
) -> tuple[float, ...]:
    validation = summaries[validation_split]
    return (
        float(bool(diagnostics["preferred"])),
        _score_float(diagnostics.get("min_pnl")),
        _score_float(diagnostics.get("min_avg_pnl")),
        _score_float(validation.get("avg_pnl")),
        float(validation["pnl"]),
        float(diagnostics["min_trade_count"]),
        float(validation["trade_count"]),
    )


def _compact_gate_candidates(candidates: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    top = sorted(candidates, key=lambda candidate: tuple(candidate["selection_score"]), reverse=True)[:limit]
    compact: list[dict[str, Any]] = []
    for candidate in top:
        diagnostics = candidate["selection_diagnostics"]
        compact.append(
            {
                "settlement_threshold": candidate["settlement_threshold"],
                "edge_threshold": candidate["edge_threshold"],
                "selection_score": candidate["selection_score"],
                "preferred": diagnostics["preferred"],
                "strong_average_all_splits": diagnostics["strong_average_all_splits"],
                "min_avg_pnl": diagnostics["min_avg_pnl"],
                "min_pnl": diagnostics["min_pnl"],
                "min_trade_count": diagnostics["min_trade_count"],
                "trade_counts": diagnostics["trade_counts"],
                "pnls": diagnostics["pnls"],
                "avg_pnls": diagnostics["avg_pnls"],
            }
        )
    return compact


def _score_float(value: Any) -> float:
    if value is None:
        return -1_000_000_000.0
    return float(value)


def _tradable_ev_backtest(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str | bool | None]],
    rule: dict[str, Any],
    *,
    cfg: XGBoostV7Config,
    probability_prefix: str,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    seen_rounds: set[str] = set()
    candidate_rounds: set[str] = set()
    for row, payload in zip(rows, payloads, strict=True):
        if not _execution_eligible(row, cfg):
            continue
        if probability_prefix == "residual_":
            side = _select_edge_side(
                up_edge=_edge(payload, "UP", probability_prefix),
                down_edge=_edge(payload, "DOWN", probability_prefix),
            )
        else:
            side = _select_side(
                p_up=_probability(payload, "UP", probability_prefix),
                p_down=_probability(payload, "DOWN", probability_prefix),
                threshold=float(rule["settlement_threshold"]),
            )
        if side is None:
            continue
        round_slug = _round_slug(row)
        candidate_rounds.add(round_slug)
        if round_slug in seen_rounds:
            continue
        edge = _edge(payload, side, probability_prefix)
        if edge is None or edge < float(rule["edge_threshold"]):
            continue
        worst = payload["entry_worst_price_up"] if side == "UP" else payload["entry_worst_price_down"]
        if worst is None:
            continue
        if probability_prefix == "residual_":
            pnl_value = _convergence_exit_gain(row, side=side)
            if pnl_value is None:
                continue
            pnl = float(pnl_value)
        else:
            pnl = (1.0 - float(worst)) if _settlement_label(row) == side else -float(worst)
        selected.append(
            {
                "round_slug": round_slug,
                "side": side,
                "true_label": _settlement_label(row),
                "p_side": _probability(payload, side, probability_prefix),
                "entry_worst_price": float(worst),
                "expected_edge": float(edge),
                "pnl": pnl,
                "round_age_seconds": _round_age_seconds(row),
                "seconds_to_expiry": _seconds_to_expiry(row),
            }
        )
        seen_rounds.add(round_slug)
    metric_of_record = (
        "best_exit_path_pnl"
        if probability_prefix == "residual_"
        else "executable_one_way_settlement_pnl"
    )
    return _summarize_trades(
        selected,
        candidate_round_count=len(candidate_rounds),
        metric_of_record=metric_of_record,
    )


def _probability(payload: dict[str, Any], side: str, prefix: str) -> float:
    if prefix == "residual_":
        key = "p_up_residual_adjusted" if side == "UP" else "p_down_residual_adjusted"
    else:
        key = "p_up" if side == "UP" else "p_down"
    return float(payload[key])


def _edge(payload: dict[str, Any], side: str, prefix: str) -> float | None:
    if prefix == "residual_":
        key = "residual_expected_edge_up" if side == "UP" else "residual_expected_edge_down"
    else:
        key = "expected_edge_up" if side == "UP" else "expected_edge_down"
    value = payload.get(key)
    return None if value is None else float(value)


def _confidence_payload_probability(
    payload: dict[str, Any],
    *,
    side: str,
    target: str,
) -> float | None:
    prefix = "p_up" if side == "UP" else "p_down"
    key = f"{prefix}_{target}"
    value = payload.get(key)
    return None if value is None else float(value)


def _select_side(*, p_up: float, p_down: float, threshold: float) -> str | None:
    if p_up >= p_down and p_up >= threshold:
        return "UP"
    if p_down > p_up and p_down >= threshold:
        return "DOWN"
    return None


def _select_edge_side(*, up_edge: float | None, down_edge: float | None) -> str | None:
    if up_edge is None and down_edge is None:
        return None
    if down_edge is None or (up_edge is not None and up_edge >= down_edge):
        return "UP"
    return "DOWN"


def _side_market_price(market: float | None, token_side: str, side: str) -> float | None:
    if market is None:
        return None
    if side == token_side:
        return market
    return _clip01(1.0 - market)


def _summarize_trades(
    trades: list[dict[str, Any]],
    *,
    candidate_round_count: int,
    metric_of_record: str = "executable_one_way_settlement_pnl",
) -> dict[str, Any]:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    wins = 0
    for trade in trades:
        pnl = float(trade["pnl"])
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
        wins += int(pnl > 0.0)
    return {
        "metric_of_record": metric_of_record,
        "trade_count": len(trades),
        "candidate_round_count": candidate_round_count,
        "coverage": len(trades) / candidate_round_count if candidate_round_count else None,
        "pnl": cumulative,
        "avg_pnl": cumulative / len(trades) if trades else None,
        "hit_rate": wins / len(trades) if trades else None,
        "max_drawdown": max_drawdown,
        "mean_expected_edge": _mean([float(trade["expected_edge"]) for trade in trades]),
        "mean_entry_worst_price": _mean([float(trade["entry_worst_price"]) for trade in trades]),
        "round_age_buckets": _bucketed_pnl(trades, ROUND_AGE_BUCKETS, key="round_age_seconds"),
        "entry_price_buckets": _bucketed_pnl(trades, ENTRY_PRICE_BUCKETS, key="entry_worst_price"),
        "sample_trades": trades[:12],
    }


def _zero_skill_baseline(rows: list[dict[str, Any]], cfg: XGBoostV7Config) -> dict[str, Any]:
    trades = []
    seen_rounds: set[str] = set()
    for row in rows:
        if not _execution_eligible(row, cfg):
            continue
        round_slug = _round_slug(row)
        if round_slug in seen_rounds:
            continue
        side = _token_side(row)
        worst = _entry_worst_price(_entry_ask(row, side), cfg.buy_slippage, cfg.fee_bps)
        if worst is None:
            continue
        pnl = (1.0 - worst) if _settlement_label(row) == side else -worst
        trades.append(
            {
                "round_slug": round_slug,
                "side": side,
                "true_label": _settlement_label(row),
                "entry_worst_price": worst,
                "expected_edge": _market_implied_prob(row) - worst,
                "pnl": pnl,
                "round_age_seconds": _round_age_seconds(row),
            }
        )
        seen_rounds.add(round_slug)
    return _summarize_trades(trades, candidate_round_count=len(seen_rounds))


def _round_age_seconds(row: dict[str, Any]) -> float | None:
    value = _seconds_since_round_start(row)
    return None if value is None else max(0.0, value)


def _bucketed_pnl(
    trades: list[dict[str, Any]],
    buckets: tuple[tuple[str, float, float | None], ...],
    *,
    key: str,
) -> list[dict[str, Any]]:
    out = []
    for label, lower, upper in buckets:
        members = [
            trade
            for trade in trades
            if trade.get(key) is not None
            and float(trade[key]) >= lower
            and (upper is None or float(trade[key]) < upper)
        ]
        pnl = sum(float(trade["pnl"]) for trade in members)
        out.append(
            {
                "bucket": label,
                "trade_count": len(members),
                "pnl": pnl,
                "hit_rate": (
                    sum(1 for trade in members if float(trade["pnl"]) > 0.0) / len(members)
                    if members
                    else None
                ),
            }
        )
    return out


def _settlement_tuple(payload: dict[str, Any]) -> tuple[float, float, float]:
    return (float(payload["p_up"]), float(payload["p_down"]), float(payload["p_neutral"]))


def _v6_like_config(cfg: XGBoostV7Config) -> Any:
    class _Config:
        temperature_grid = cfg.temperature_grid
        family_temperature_min_samples = cfg.family_temperature_min_samples

    return _Config()


def _rmse(labels: list[float], predictions: list[float]) -> float | None:
    if not labels:
        return None
    return math.sqrt(sum((label - pred) ** 2 for label, pred in zip(labels, predictions, strict=True)) / len(labels))


def _weighted_rmse(labels: list[float], predictions: list[float], weights: list[float]) -> float:
    weighted_error = sum(
        float(weight) * (float(label) - float(pred)) ** 2
        for label, pred, weight in zip(labels, predictions, weights, strict=True)
    )
    total_weight = sum(float(weight) for weight in weights)
    if total_weight <= 0.0:
        fallback = _rmse(labels, predictions)
        return float("inf") if fallback is None else fallback
    return math.sqrt(weighted_error / total_weight)


def _mae(labels: list[float], predictions: list[float]) -> float | None:
    if not labels:
        return None
    return sum(abs(label - pred) for label, pred in zip(labels, predictions, strict=True)) / len(labels)


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _confidence_score(confidence: dict[str, float | None]) -> float | None:
    hit = confidence.get("hit_5c_before_loss_10c")
    loss = confidence.get("loss_10c_before_hit_5c")
    if hit is None or loss is None:
        return None
    return _clip01(hit) - _clip01(loss)


def _binary_metrics(paired: list[tuple[float, float]]) -> dict[str, Any]:
    if not paired:
        return {
            "sample_count": 0,
            "positive_rate": None,
            "avg_prediction": None,
            "brier": None,
            "logloss": None,
            "auc": None,
        }
    labels = [label for label, _prediction in paired]
    predictions = [_clip01(prediction) for _label, prediction in paired]
    return {
        "sample_count": len(paired),
        "positive_rate": sum(labels) / len(labels),
        "avg_prediction": sum(predictions) / len(predictions),
        "brier": sum((prediction - label) ** 2 for label, prediction in paired) / len(paired),
        "logloss": _weighted_binary_logloss(labels, predictions, [1.0] * len(paired)),
        "auc": _binary_auc(labels, predictions),
    }


def _weighted_binary_logloss(
    labels: list[float],
    predictions: list[float],
    weights: list[float],
) -> float:
    total_weight = sum(float(weight) for weight in weights)
    if total_weight <= 0.0:
        return float("inf")
    loss = 0.0
    for label, prediction, weight in zip(labels, predictions, weights, strict=True):
        if weight <= 0.0:
            continue
        clipped = min(1.0 - 1e-6, max(1e-6, float(prediction)))
        loss += float(weight) * (
            -float(label) * math.log(clipped)
            - (1.0 - float(label)) * math.log(1.0 - clipped)
        )
    return loss / total_weight


def _binary_auc(labels: list[float], predictions: list[float]) -> float | None:
    positives = [
        prediction for label, prediction in zip(labels, predictions, strict=True)
        if label >= 0.5
    ]
    negatives = [
        prediction for label, prediction in zip(labels, predictions, strict=True)
        if label < 0.5
    ]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _save_confidence_boosters(target: Path, boosters: dict[str, xgb.Booster], *, side: str) -> None:
    side_lower = side.lower()
    for target_name, booster in boosters.items():
        booster.save_model(str(target / f"confidence_{side_lower}_{target_name}_model.json"))


def _confidence_model_paths(boosters: dict[str, xgb.Booster], *, side: str) -> dict[str, str]:
    side_lower = side.lower()
    return {
        target: f"confidence_{side_lower}_{target}_model.json"
        for target in CONFIDENCE_TARGETS
        if target in boosters
    }


def _write_v7_model_artifact(path: Path, *, model: XGBoostV7Model) -> None:
    confidence_up_boosters = model.confidence_up_boosters or {}
    confidence_down_boosters = model.confidence_down_boosters or {}
    artifact = {
        "schema_version": XGBOOST_V7_ARTIFACT_SCHEMA_VERSION,
        "model_version": model.model_version,
        "feature_columns": list(model.feature_columns),
        "settlement": {
            "path": "settlement_model.json",
            "params": model.settlement_params,
            "classes": list(SETTLEMENT_CLASSES),
        },
        "convergence": {
            "up_path": "convergence_up_model.json",
            "down_path": "convergence_down_model.json",
            "params": model.convergence_params,
            "target": "future best executable exit price by side",
        },
        "confidence": {
            "available": bool(confidence_up_boosters and confidence_down_boosters),
            "up_paths": _confidence_model_paths(confidence_up_boosters, side="UP"),
            "down_paths": _confidence_model_paths(confidence_down_boosters, side="DOWN"),
            "params": model.confidence_params or {},
            "targets": {
                "hit_5c_before_loss_10c": (
                    "probability executable +5c is reached before executable -10c"
                ),
                "hit_10c_before_loss_10c": (
                    "probability executable +10c is reached before executable -10c"
                ),
                "loss_10c_before_hit_5c": (
                    "probability executable -10c is reached before executable +5c"
                ),
            },
        },
        "calibration": {
            "method": "family-aware temperature scaling with global fallback",
            "global_temperature": model.settlement_temperature,
            "family_temperatures": model.family_temperatures,
        },
        "serving_config": {
            "buy_slippage": model.buy_slippage,
            "fee_bps": model.fee_bps,
            "ev_margin": model.ev_margin,
        },
        "serving_payload": [
            "p_up",
            "p_down",
            "p_neutral",
            "settlement_residual",
            "market_implied_prob",
            "model_probability",
            "polymarket_price",
            "mispricing_edge",
            "token_expected_win_probability",
            "p_up_residual_adjusted",
            "p_down_residual_adjusted",
            "entry_worst_price_up",
            "entry_worst_price_down",
            "expected_edge_up",
            "expected_edge_down",
            "residual_expected_edge_up",
            "residual_expected_edge_down",
            "p_up_hit_5c_before_loss_10c",
            "p_up_hit_10c_before_loss_10c",
            "p_up_loss_10c_before_hit_5c",
            "p_down_hit_5c_before_loss_10c",
            "p_down_hit_10c_before_loss_10c",
            "p_down_loss_10c_before_hit_5c",
            "selected_hit_5c_before_loss_10c",
            "selected_hit_10c_before_loss_10c",
            "selected_loss_10c_before_hit_5c",
            "selected_confidence_score",
            "selected_side",
            "selected_expected_edge",
            "should_enter_settlement",
            "model_version",
        ],
        "compatibility": {
            "executor": (
                "use model_version dispatch; v7 trading decisions consume "
                "model_probability - polymarket_price / mispricing_edge plus "
                "executor-only safety gates; p_up/p_down/p_neutral are diagnostics"
            ),
            "volatility": "not implemented in v7 convergence artifact",
        },
    }
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")


def _executor_contract(cfg: XGBoostV7Config) -> dict[str, Any]:
    return {
        "model_version": XGBOOST_V7_MODEL_VERSION,
        "entry_gate": "selected side only when convergence_edge_side >= configured min edge",
        "model_consumed_fields": [
            "model_probability",
            "polymarket_price",
            "mispricing_edge",
            "p_up_residual_adjusted",
            "p_down_residual_adjusted",
            "residual_expected_edge_up",
            "residual_expected_edge_down",
            "p_up_hit_5c_before_loss_10c",
            "p_up_hit_10c_before_loss_10c",
            "p_up_loss_10c_before_hit_5c",
            "p_down_hit_5c_before_loss_10c",
            "p_down_hit_10c_before_loss_10c",
            "p_down_loss_10c_before_hit_5c",
            "selected_hit_5c_before_loss_10c",
            "selected_hit_10c_before_loss_10c",
            "selected_loss_10c_before_hit_5c",
            "selected_confidence_score",
            "entry_worst_price_up",
            "entry_worst_price_down",
        ],
        "executor_only_gates": [
            "signal_age",
            "price freshness",
            "account/funder balance",
            "max concurrent positions",
            "one settlement bet per round",
            "daily loss limit",
            "expiry/no-new-entry windows",
            "cashflow reconciliation",
        ],
        "issue_101_guardrail": (
            "Do not use live run PnL as promotion evidence while pending settlement or "
            "account reconciliation is unresolved."
        ),
        "default_buy_slippage": cfg.buy_slippage,
        "default_fee_bps": cfg.fee_bps,
    }


def _write_executor_integration_doc(path: Path, report: XGBoostV7Report) -> None:
    path.write_text(
        "\n".join(
            [
                "# xgboost-v7 executor integration",
                "",
                "v7 emits settlement-token UP/DOWN convergence signals. Volatility entries stay disabled.",
                "",
                "The model emits side-specific predicted future executable exit values "
                "and executable mispricing fields. The executor should not infer EV "
                "from raw settlement hit rate or p_up/p_down; it should consume "
                "`model_probability`, `polymarket_price`, and `mispricing_edge`, "
                "then apply executor-only safety gates.",
                "",
                "Model-side fields:",
                "",
                "- `p_up`, `p_down`, `p_neutral` for settlement diagnostics only",
                "- `p_up_residual_adjusted = predicted future executable exit value for buying UP`",
                "- `p_down_residual_adjusted = predicted future executable exit value for buying DOWN`",
                "- `model_probability = selected side predicted future executable exit value`",
                "- `polymarket_price = selected side executable entry price, or current quote ask in the event-driven overlay`",
                "- `mispricing_edge = selected convergence exit value - polymarket_price`",
                "- `entry_worst_price_up/down`",
                "- `expected_edge_up/down = diagnostic settlement p_side - entry_worst_price_side`",
                "- `residual_expected_edge_up/down = convergence exit value side - entry_worst_price_side`",
                "- `p_up/down_hit_5c_before_loss_10c = model-side reliability probability for +5c before -10c`",
                "- `p_up/down_hit_10c_before_loss_10c = model-side reliability probability for +10c before -10c`",
                "- `p_up/down_loss_10c_before_hit_5c = model-side adverse-first probability`",
                "- `selected_hit_5c_before_loss_10c`, `selected_loss_10c_before_hit_5c`, `selected_confidence_score`",
                "- `selected_side`, `selected_expected_edge`, `should_enter_settlement`",
                "",
                "Executor-only gates:",
                "",
                *[f"- {gate}" for gate in report.executor_contract["executor_only_gates"]],
                "",
                "Issue #101 guardrail:",
                "",
                report.executor_contract["issue_101_guardrail"],
                "",
            ]
        ),
        encoding="utf-8",
    )
