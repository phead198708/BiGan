#!/usr/bin/env python3
"""Train and replay a v7 meta-confidence gate from historical signal logs.

The v7 value model estimates a token value. This script asks a separate
question: given the current signal context, how reliable is that value estimate
as an entry? It trains small meta heads on prior paper-run signals and replays
candidate entry gates using predicted convergence and tail-loss probabilities.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


def _load_calibration_module() -> Any:
    script = Path(__file__).with_name("replay_v7_convergence_calibration.py")
    spec = importlib.util.spec_from_file_location("replay_v7_convergence_calibration", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_cal = _load_calibration_module()


FEATURE_NAMES = [
    "side_up",
    "price",
    "execution_price",
    "model_value",
    "edge",
    "execution_edge",
    "raw_p_side",
    "raw_p_opposite",
    "raw_margin",
    "seconds_to_expiry",
    "price_bucket_id",
    "raw_bucket_id",
    "edge_bucket_id",
    "model_bucket_id",
    "cal_sample_log1p",
    "cal_hit_5c_rate",
    "cal_hit_10c_rate",
    "cal_close_rate",
    "cal_median_best_move",
    "cal_median_close_move",
    "cal_median_value_error",
    "cal_over_error_p80",
    "cal_adjusted_median_edge",
    "cal_adjusted_p80_edge",
]


@dataclass(frozen=True, slots=True)
class ConfidenceExample:
    labeled: Any
    future_min_price: float
    future_min_ts_ms: int
    min_move: float
    loss_10c: bool
    close_loss_10c: bool
    first_hit_5c_ts_ms: int | None
    first_hit_10c_ts_ms: int | None
    first_loss_10c_ts_ms: int | None
    hit_5c_before_loss_10c: bool
    hit_10c_before_loss_10c: bool
    loss_10c_before_hit_5c: bool


@dataclass(frozen=True, slots=True)
class ConfidencePrediction:
    p_hit_5c: float
    p_hit_10c: float
    p_loss_10c: float

    @property
    def score(self) -> float:
        return self.p_hit_5c - self.p_loss_10c


@dataclass(frozen=True, slots=True)
class MetaReplayConfig:
    min_price: float
    max_price: float
    min_execution_edge: float
    min_p_hit_5c: float
    min_p_hit_10c: float
    max_p_loss_10c: float
    min_confidence_score: float


@dataclass(frozen=True, slots=True)
class ConstantBinaryModel:
    probability: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.full((matrix.shape[0],), self.probability, dtype=float)

    def importance(self) -> dict[str, float]:
        return {}


@dataclass(frozen=True, slots=True)
class XgbBinaryModel:
    booster: xgb.Booster

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        dmatrix = xgb.DMatrix(matrix, feature_names=FEATURE_NAMES)
        return self.booster.predict(dmatrix)

    def importance(self) -> dict[str, float]:
        return self.booster.get_score(importance_type="gain")


def main() -> int:
    args = _parse_args()
    train_signals = _cal._load_signals([Path(item) for item in args.train_jsonl])
    replay_signals = _cal._load_signals([Path(item) for item in args.replay_jsonl])
    train_examples = _build_examples(train_signals)
    replay_examples = _build_examples(replay_signals)
    calibrator = _cal._BucketCalibrator(
        [item.labeled for item in train_examples],
        min_bucket_size=args.min_bucket_size,
    )

    train_matrix = _feature_matrix(train_examples, calibrator)
    labels = {
        "hit_5c": np.array([1.0 if item.hit_5c_before_loss_10c else 0.0 for item in train_examples]),
        "hit_10c": np.array([1.0 if item.hit_10c_before_loss_10c else 0.0 for item in train_examples]),
        "loss_10c": np.array([1.0 if item.loss_10c_before_hit_5c else 0.0 for item in train_examples]),
    }
    models = {
        target: _fit_binary_model(train_matrix, target_labels, seed=args.seed)
        for target, target_labels in labels.items()
    }
    replay_matrix = _feature_matrix(replay_examples, calibrator)
    replay_predictions = _predict_examples(models, replay_matrix)

    replay_results = [
        _replay_config(
            replay_examples,
            replay_predictions,
            config=config,
            take_profit_delta=args.take_profit_delta,
            respect_existing_entry_skips=args.respect_existing_entry_skips,
            ignored_existing_entry_skip_reasons=args.ignored_existing_entry_skip_reasons,
        )
        for config in _config_grid(args)
    ]
    replay_results.sort(
        key=lambda item: (
            item["pnl_proxy_usdc"],
            item["trade_count"],
            item["hit_10c_rate"] or -1.0,
            item["hit_5c_rate"] or -1.0,
        ),
        reverse=True,
    )

    report = {
        "inputs": {
            "train_jsonl": args.train_jsonl,
            "replay_jsonl": args.replay_jsonl,
        },
        "config": {
            "min_bucket_size": args.min_bucket_size,
            "take_profit_delta": args.take_profit_delta,
            "price_range": [args.min_price, args.max_price],
            "min_execution_edge_grid": args.min_execution_edge_grid,
            "min_p_hit_5c_grid": args.min_p_hit_5c_grid,
            "min_p_hit_10c_grid": args.min_p_hit_10c_grid,
            "max_p_loss_10c_grid": args.max_p_loss_10c_grid,
            "min_confidence_score_grid": args.min_confidence_score_grid,
            "respect_existing_entry_skips": args.respect_existing_entry_skips,
            "ignored_existing_entry_skip_reasons": sorted(args.ignored_existing_entry_skip_reasons),
        },
        "train_summary": _summary(train_examples),
        "replay_baseline": _summary(replay_examples),
        "model_metrics": {
            target: _metrics(labels[target], models[target].predict(train_matrix))
            for target in labels
        },
        "replay_model_metrics": _replay_model_metrics(replay_examples, replay_predictions),
        "replay_prediction_summary": _prediction_summary(replay_predictions),
        "feature_importance": {
            target: _top_importance(model.importance(), limit=args.importance_limit)
            for target, model in models.items()
        },
        "top_replay_configs": replay_results[: args.top_limit],
        "recommended": _recommended(replay_results, min_trades=args.min_recommended_trades),
    }
    if args.output_json_path:
        output = Path(args.output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        output = Path(args.report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_markdown_report(report), encoding="utf-8")
    if args.examples_output_jsonl:
        _write_examples(Path(args.examples_output_jsonl), train_examples, replay_examples, replay_predictions, calibrator)
    print(json.dumps({"recommended": report["recommended"], "replay_baseline": report["replay_baseline"]}, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", action="append", required=True)
    parser.add_argument("--replay-jsonl", action="append", required=True)
    parser.add_argument("--min-bucket-size", type=int, default=20)
    parser.add_argument("--min-price", type=float, default=0.40)
    parser.add_argument("--max-price", type=float, default=0.70)
    parser.add_argument("--min-execution-edge-grid", default="0.15,0.20,0.25,0.30")
    parser.add_argument("--min-p-hit-5c-grid", default="0,0.45,0.55,0.65")
    parser.add_argument("--min-p-hit-10c-grid", default="0,0.20,0.30,0.40")
    parser.add_argument("--max-p-loss-10c-grid", default="1.0,0.60,0.45,0.35,0.25")
    parser.add_argument("--min-confidence-score-grid", default="-1.0,0.0,0.10,0.20,0.30")
    parser.add_argument("--take-profit-delta", type=float, default=0.10)
    parser.add_argument(
        "--respect-existing-entry-skips",
        action="store_true",
        help="Keep existing non-target engineering skips in replay.",
    )
    parser.add_argument(
        "--ignored-existing-entry-skip-reasons",
        default="",
        help="Comma-separated existing skip reasons to ignore when respecting skips.",
    )
    parser.add_argument("--min-recommended-trades", type=int, default=1)
    parser.add_argument("--top-limit", type=int, default=20)
    parser.add_argument("--importance-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--examples-output-jsonl", default="")
    args = parser.parse_args()
    args.min_execution_edge_grid = _float_grid(args.min_execution_edge_grid)
    args.min_p_hit_5c_grid = _float_grid(args.min_p_hit_5c_grid)
    args.min_p_hit_10c_grid = _float_grid(args.min_p_hit_10c_grid)
    args.max_p_loss_10c_grid = _float_grid(args.max_p_loss_10c_grid)
    args.min_confidence_score_grid = _float_grid(args.min_confidence_score_grid)
    args.ignored_existing_entry_skip_reasons = _cal._str_set(
        args.ignored_existing_entry_skip_reasons
    )
    if args.min_bucket_size < 1:
        raise ValueError("--min-bucket-size must be positive")
    if args.min_price >= args.max_price:
        raise ValueError("--min-price must be below --max-price")
    if args.take_profit_delta < 0:
        raise ValueError("--take-profit-delta must be non-negative")
    return args


def _float_grid(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            value = float(item)
            if not math.isfinite(value):
                raise ValueError(f"non-finite grid value: {item}")
            values.append(value)
    if not values:
        raise ValueError("grid must contain at least one value")
    return sorted(set(values))


def _build_examples(signals: list[Any]) -> list[ConfidenceExample]:
    labeled = _cal._label_signals(signals)
    by_key: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for signal in signals:
        by_key[(signal.run_id, signal.round_slug, signal.side)].append(signal)
    for rows in by_key.values():
        rows.sort(key=lambda item: (item.created_at_ms, item.event_id))

    examples: list[ConfidenceExample] = []
    for row in labeled:
        signal = row.signal
        future = [
            item
            for item in by_key[(signal.run_id, signal.round_slug, signal.side)]
            if item.created_at_ms > signal.created_at_ms
            and (signal.round_end_ts_ms <= 0 or item.created_at_ms <= signal.round_end_ts_ms)
        ]
        if not future:
            continue
        min_row = min(future, key=lambda item: item.price)
        min_move = min_row.price - signal.execution_price
        close_loss_10c = row.close_move <= -0.10 + 1e-12
        first_hit_5c_ts_ms = _first_barrier_ts(
            future,
            lambda item: item.price >= signal.execution_price + 0.05 - 1e-12,
        )
        first_hit_10c_ts_ms = _first_barrier_ts(
            future,
            lambda item: item.price >= signal.execution_price + 0.10 - 1e-12,
        )
        first_loss_10c_ts_ms = _first_barrier_ts(
            future,
            lambda item: item.price <= signal.execution_price - 0.10 + 1e-12,
        )
        hit_5c_before_loss_10c = first_hit_5c_ts_ms is not None and (
            first_loss_10c_ts_ms is None or first_hit_5c_ts_ms < first_loss_10c_ts_ms
        )
        hit_10c_before_loss_10c = first_hit_10c_ts_ms is not None and (
            first_loss_10c_ts_ms is None or first_hit_10c_ts_ms < first_loss_10c_ts_ms
        )
        loss_10c_before_hit_5c = first_loss_10c_ts_ms is not None and (
            first_hit_5c_ts_ms is None or first_loss_10c_ts_ms < first_hit_5c_ts_ms
        )
        examples.append(
            ConfidenceExample(
                labeled=row,
                future_min_price=min_row.price,
                future_min_ts_ms=min_row.created_at_ms,
                min_move=min_move,
                loss_10c=min_move <= -0.10 + 1e-12,
                close_loss_10c=close_loss_10c,
                first_hit_5c_ts_ms=first_hit_5c_ts_ms,
                first_hit_10c_ts_ms=first_hit_10c_ts_ms,
                first_loss_10c_ts_ms=first_loss_10c_ts_ms,
                hit_5c_before_loss_10c=hit_5c_before_loss_10c,
                hit_10c_before_loss_10c=hit_10c_before_loss_10c,
                loss_10c_before_hit_5c=loss_10c_before_hit_5c,
            )
        )
    return examples


def _first_barrier_ts(rows: list[Any], predicate: Any) -> int | None:
    for row in rows:
        if predicate(row):
            return row.created_at_ms
    return None


def _feature_matrix(examples: list[ConfidenceExample], calibrator: Any) -> np.ndarray:
    matrix = [_features(item, calibrator) for item in examples]
    if not matrix:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return np.asarray(matrix, dtype=float)


def _features(example: ConfidenceExample, calibrator: Any) -> list[float]:
    signal = example.labeled.signal
    stats = calibrator.lookup(signal)
    raw_p_side = _num(signal.raw_p_side, 0.5)
    raw_p_opposite = _num(signal.raw_p_opposite, 0.5)
    seconds_to_expiry = _num(signal.seconds_to_expiry, 0.0)
    return [
        1.0 if signal.side == "UP" else 0.0,
        signal.price,
        signal.execution_price,
        signal.model_value,
        signal.edge,
        _cal._execution_edge(signal),
        raw_p_side,
        raw_p_opposite,
        raw_p_side - raw_p_opposite,
        seconds_to_expiry,
        _bucket_id(_cal._price_bucket(signal.execution_price), PRICE_BUCKET_IDS),
        _bucket_id(_cal._raw_bucket(signal.raw_p_side), RAW_BUCKET_IDS),
        _bucket_id(_cal._edge_bucket(_cal._execution_edge(signal)), EDGE_BUCKET_IDS),
        _bucket_id(_cal._model_bucket(signal.model_value), MODEL_BUCKET_IDS),
        math.log1p(max(0, int(stats.sample_count))),
        stats.hit_5c_rate,
        stats.hit_10c_rate,
        stats.close_rate,
        stats.median_best_move,
        stats.median_close_move,
        stats.median_value_error,
        stats.model_over_error_p80,
        _cal._adjusted_median_edge(signal, stats),
        _cal._adjusted_p80_edge(signal, stats),
    ]


PRICE_BUCKET_IDS = {"<0.30": 0, "0.30-0.40": 1, "0.40-0.50": 2, "0.50-0.70": 3, ">=0.70": 4}
RAW_BUCKET_IDS = {"missing": 0, "<0.55": 1, "0.55-0.60": 2, "0.60-0.65": 3, ">=0.65": 4}
EDGE_BUCKET_IDS = {"<0.30": 0, "0.30-0.40": 1, "0.40-0.50": 2, ">=0.50": 3}
MODEL_BUCKET_IDS = {"<0.70": 0, "0.70-0.80": 1, ">=0.80": 2}


def _bucket_id(value: str, mapping: dict[str, int]) -> float:
    return float(mapping.get(value, -1))


def _num(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        return default if math.isnan(number) else number
    except (TypeError, ValueError):
        return default


def _fit_binary_model(matrix: np.ndarray, labels: np.ndarray, *, seed: int) -> Any:
    if matrix.shape[0] == 0:
        return ConstantBinaryModel(0.0)
    positive_rate = float(np.mean(labels)) if labels.size else 0.0
    unique = set(float(item) for item in labels.tolist())
    if len(unique) < 2 or matrix.shape[0] < 12:
        return ConstantBinaryModel(positive_rate)
    dtrain = xgb.DMatrix(matrix, label=labels, feature_names=FEATURE_NAMES)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 2,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 5.0,
        "alpha": 1.0,
        "seed": seed,
        "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=80, verbose_eval=False)
    return XgbBinaryModel(booster)


def _predict_examples(models: dict[str, Any], matrix: np.ndarray) -> list[ConfidencePrediction]:
    if matrix.shape[0] == 0:
        return []
    hit_5c = models["hit_5c"].predict(matrix)
    hit_10c = models["hit_10c"].predict(matrix)
    loss_10c = models["loss_10c"].predict(matrix)
    return [
        ConfidencePrediction(
            p_hit_5c=float(hit_5c[idx]),
            p_hit_10c=float(hit_10c[idx]),
            p_loss_10c=float(loss_10c[idx]),
        )
        for idx in range(matrix.shape[0])
    ]


def _config_grid(args: argparse.Namespace) -> list[MetaReplayConfig]:
    return [
        MetaReplayConfig(
            min_price=args.min_price,
            max_price=args.max_price,
            min_execution_edge=min_edge,
            min_p_hit_5c=min_hit_5c,
            min_p_hit_10c=min_hit_10c,
            max_p_loss_10c=max_loss_10c,
            min_confidence_score=min_score,
        )
        for min_edge, min_hit_5c, min_hit_10c, max_loss_10c, min_score in itertools.product(
            args.min_execution_edge_grid,
            args.min_p_hit_5c_grid,
            args.min_p_hit_10c_grid,
            args.max_p_loss_10c_grid,
            args.min_confidence_score_grid,
        )
    ]


def _replay_config(
    examples: list[ConfidenceExample],
    predictions: list[ConfidencePrediction],
    *,
    config: MetaReplayConfig,
    take_profit_delta: float,
    respect_existing_entry_skips: bool,
    ignored_existing_entry_skip_reasons: set[str],
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    skipped = Counter()
    open_until_ms = -1
    for example, prediction in sorted(
        zip(examples, predictions, strict=True),
        key=lambda item: (item[0].labeled.signal.created_at_ms, item[0].labeled.signal.event_id),
    ):
        signal = example.labeled.signal
        if respect_existing_entry_skips:
            blocking_reasons = [
                reason
                for reason in signal.skip_reasons
                if reason not in ignored_existing_entry_skip_reasons
            ]
            if blocking_reasons:
                skipped[f"existing_skip_{blocking_reasons[0]}"] += 1
                continue
        if signal.created_at_ms < open_until_ms:
            skipped["sim_max_combined_concurrent_positions"] += 1
            continue
        reason = _skip_reason(signal, prediction, config)
        if reason is not None:
            skipped[reason] += 1
            continue
        if example.labeled.future_max_price >= signal.execution_price + take_profit_delta - 1e-12:
            exit_price = signal.execution_price + take_profit_delta
            exit_ts_ms = example.labeled.future_max_ts_ms
            exit_reason = f"tp_{take_profit_delta:.2f}"
        else:
            exit_price = example.labeled.future_last_price
            exit_ts_ms = max(example.labeled.future_last_ts_ms, signal.round_end_ts_ms)
            exit_reason = "last_observed"
        pnl_proxy = (
            (exit_price - signal.execution_price) / signal.execution_price
            if signal.execution_price > 0
            else 0.0
        )
        selected.append(_trade_dict(example, prediction, exit_ts_ms, exit_price, exit_reason, pnl_proxy))
        open_until_ms = exit_ts_ms
    return {
        "config": asdict(config),
        "trade_count": len(selected),
        "pnl_proxy_usdc": sum(item["pnl_proxy_usdc"] for item in selected),
        "hit_5c_rate": _rate(item["exit_price"] >= item["execution_price"] + 0.05 - 1e-12 for item in selected),
        "hit_10c_rate": _rate(item["exit_price"] >= item["execution_price"] + 0.10 - 1e-12 for item in selected),
        "any_loss_10c_rate": _rate(item["min_move"] <= -0.10 + 1e-12 for item in selected),
        "loss_10c_rate": _rate(item["actual_loss_10c_before_hit_5c"] for item in selected),
        "loss_10c_before_hit_5c_rate": _rate(
            item["actual_loss_10c_before_hit_5c"] for item in selected
        ),
        "tp_exit_count": sum(1 for item in selected if item["exit_reason"].startswith("tp_")),
        "entry_filled_overlap": sum(1 for item in selected if item["entry_filled"]),
        "entry_gate_passed_overlap": sum(1 for item in selected if item["entry_gate_passed"]),
        "skipped": dict(sorted(skipped.items())),
        "trades": selected,
    }


def _skip_reason(signal: Any, prediction: ConfidencePrediction, config: MetaReplayConfig) -> str | None:
    if signal.execution_price < config.min_price:
        return "entry_price_below_min"
    if signal.execution_price > config.max_price:
        return "entry_price_above_max"
    if _cal._execution_edge(signal) < config.min_execution_edge:
        return "execution_edge_below_min"
    if prediction.p_hit_5c < config.min_p_hit_5c:
        return "meta_p_hit_5c_below_min"
    if prediction.p_hit_10c < config.min_p_hit_10c:
        return "meta_p_hit_10c_below_min"
    if prediction.p_loss_10c > config.max_p_loss_10c:
        return "meta_p_loss_10c_above_max"
    if prediction.score < config.min_confidence_score:
        return "meta_confidence_score_below_min"
    return None


def _trade_dict(
    example: ConfidenceExample,
    prediction: ConfidencePrediction,
    exit_ts_ms: int,
    exit_price: float,
    exit_reason: str,
    pnl_proxy: float,
) -> dict[str, Any]:
    signal = example.labeled.signal
    return {
        "run_id": signal.run_id,
        "round_slug": signal.round_slug,
        "side": signal.side,
        "event_id": signal.event_id,
        "created_at_ms": signal.created_at_ms,
        "price": signal.price,
        "execution_price": signal.execution_price,
        "model_value": signal.model_value,
        "edge": signal.edge,
        "execution_edge": _cal._execution_edge(signal),
        "raw_p_side": signal.raw_p_side,
        "raw_p_opposite": signal.raw_p_opposite,
        "entry_filled": signal.entry_filled,
        "entry_gate_passed": signal.entry_gate_passed,
        "skip_reasons": list(signal.skip_reasons),
        "future_max_price": example.labeled.future_max_price,
        "future_last_price": example.labeled.future_last_price,
        "future_min_price": example.future_min_price,
        "best_move": example.labeled.best_move,
        "close_move": example.labeled.close_move,
        "min_move": example.min_move,
        "actual_hit_5c": example.labeled.hit_5c,
        "actual_hit_10c": example.labeled.hit_10c,
        "actual_loss_10c": example.loss_10c,
        "actual_hit_5c_before_loss_10c": example.hit_5c_before_loss_10c,
        "actual_hit_10c_before_loss_10c": example.hit_10c_before_loss_10c,
        "actual_loss_10c_before_hit_5c": example.loss_10c_before_hit_5c,
        "first_hit_5c_ts_ms": example.first_hit_5c_ts_ms,
        "first_hit_10c_ts_ms": example.first_hit_10c_ts_ms,
        "first_loss_10c_ts_ms": example.first_loss_10c_ts_ms,
        "p_hit_5c": prediction.p_hit_5c,
        "p_hit_10c": prediction.p_hit_10c,
        "p_loss_10c": prediction.p_loss_10c,
        "confidence_score": prediction.score,
        "exit_ts_ms": exit_ts_ms,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl_proxy_usdc": pnl_proxy,
    }


def _summary(examples: list[ConfidenceExample]) -> dict[str, Any]:
    labeled = [item.labeled for item in examples]
    return {
        "sample_count": len(examples),
        "entry_filled_count": sum(1 for item in examples if item.labeled.signal.entry_filled),
        "avg_execution_price": _mean(item.labeled.signal.execution_price for item in examples),
        "avg_model_value": _mean(item.labeled.signal.model_value for item in examples),
        "avg_execution_edge": _mean(_cal._execution_edge(item.labeled.signal) for item in examples),
        "hit_5c_rate": _rate(item.labeled.hit_5c for item in examples),
        "hit_10c_rate": _rate(item.labeled.hit_10c for item in examples),
        "loss_10c_rate": _rate(item.loss_10c for item in examples),
        "hit_5c_before_loss_10c_rate": _rate(item.hit_5c_before_loss_10c for item in examples),
        "hit_10c_before_loss_10c_rate": _rate(item.hit_10c_before_loss_10c for item in examples),
        "loss_10c_before_hit_5c_rate": _rate(item.loss_10c_before_hit_5c for item in examples),
        "close_loss_10c_rate": _rate(item.close_loss_10c for item in examples),
        "close_rate": _rate(item.close_converged for item in labeled),
        "median_best_move": _median(item.labeled.best_move for item in examples),
        "median_close_move": _median(item.labeled.close_move for item in examples),
        "median_min_move": _median(item.min_move for item in examples),
    }


def _prediction_summary(predictions: list[ConfidencePrediction]) -> dict[str, Any]:
    return {
        "sample_count": len(predictions),
        "avg_p_hit_5c": _mean(item.p_hit_5c for item in predictions),
        "avg_p_hit_10c": _mean(item.p_hit_10c for item in predictions),
        "avg_p_loss_10c": _mean(item.p_loss_10c for item in predictions),
        "avg_confidence_score": _mean(item.score for item in predictions),
        "p80_p_loss_10c": _quantile([item.p_loss_10c for item in predictions], 0.80),
    }


def _replay_model_metrics(
    examples: list[ConfidenceExample],
    predictions: list[ConfidencePrediction],
) -> dict[str, Any]:
    labels = {
        "hit_5c": np.array([1.0 if item.hit_5c_before_loss_10c else 0.0 for item in examples]),
        "hit_10c": np.array([1.0 if item.hit_10c_before_loss_10c else 0.0 for item in examples]),
        "loss_10c": np.array([1.0 if item.loss_10c_before_hit_5c else 0.0 for item in examples]),
    }
    scores = {
        "hit_5c": np.array([item.p_hit_5c for item in predictions]),
        "hit_10c": np.array([item.p_hit_10c for item in predictions]),
        "loss_10c": np.array([item.p_loss_10c for item in predictions]),
    }
    return {target: _metrics(labels[target], scores[target]) for target in labels}


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "sample_count": int(labels.size),
        "positive_rate": float(np.mean(labels)) if labels.size else None,
        "avg_prediction": float(np.mean(predictions)) if predictions.size else None,
        "brier": _brier(labels, predictions),
        "logloss": _logloss(labels, predictions),
        "auc": _auc(labels.tolist(), predictions.tolist()),
    }


def _brier(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    if labels.size == 0:
        return None
    return float(np.mean((predictions - labels) ** 2))


def _logloss(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    if labels.size == 0:
        return None
    clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def _auc(labels: list[float], predictions: list[float]) -> float | None:
    pos = [score for label, score in zip(labels, predictions, strict=True) if label >= 0.5]
    neg = [score for label, score in zip(labels, predictions, strict=True) if label < 0.5]
    if not pos or not neg:
        return None
    wins = 0.0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _top_importance(importance: dict[str, float], *, limit: int) -> list[dict[str, Any]]:
    items = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": key, "gain": value} for key, value in items[:limit]]


def _recommended(results: list[dict[str, Any]], *, min_trades: int) -> dict[str, Any] | None:
    for item in results:
        if int(item["trade_count"]) >= min_trades:
            return item
    return results[0] if results else None


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# v7 meta-confidence replay",
        "",
        "## Inputs",
        "",
        f"- Train logs: `{', '.join(report['inputs']['train_jsonl'])}`",
        f"- Replay logs: `{', '.join(report['inputs']['replay_jsonl'])}`",
        "",
        "## Signal Quality",
        "",
        _summary_line("Train", report["train_summary"]),
        _summary_line("Replay baseline", report["replay_baseline"]),
        "",
        "## Meta Heads",
        "",
        "|Target|Positive rate|Avg pred|Brier|Logloss|AUC|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, metrics in report["model_metrics"].items():
        lines.append(
            "|{target}|{pos}|{avg}|{brier}|{logloss}|{auc}|".format(
                target=target,
                pos=_pct(metrics["positive_rate"]),
                avg=_pct(metrics["avg_prediction"]),
                brier=_fmt(metrics["brier"]),
                logloss=_fmt(metrics["logloss"]),
                auc=_fmt(metrics["auc"]),
            )
        )
    lines.extend(["", "## Replay Head Check", ""])
    lines.append("|Target|Positive rate|Avg pred|Brier|Logloss|AUC|")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for target, metrics in report["replay_model_metrics"].items():
        lines.append(
            "|{target}|{pos}|{avg}|{brier}|{logloss}|{auc}|".format(
                target=target,
                pos=_pct(metrics["positive_rate"]),
                avg=_pct(metrics["avg_prediction"]),
                brier=_fmt(metrics["brier"]),
                logloss=_fmt(metrics["logloss"]),
                auc=_fmt(metrics["auc"]),
            )
        )
    lines.extend(["", "## Recommended Replay", ""])
    recommended = report.get("recommended")
    if recommended is None:
        lines.append("No replay configuration was produced.")
    else:
        lines.extend(_replay_block(recommended))
    lines.extend(["", "## Top Replay Configs", ""])
    lines.append("|Rank|Trades|PnL proxy|Hit 5c|Hit 10c|Loss 10c|TP exits|Config|")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for idx, item in enumerate(report["top_replay_configs"], start=1):
        lines.append(
            "|{idx}|{trades}|{pnl}|{hit5}|{hit10}|{loss10}|{tp}|`{config}`|".format(
                idx=idx,
                trades=item["trade_count"],
                pnl=_fmt(item["pnl_proxy_usdc"]),
                hit5=_pct(item["hit_5c_rate"]),
                hit10=_pct(item["hit_10c_rate"]),
                loss10=_pct(item["loss_10c_rate"]),
                tp=item["tp_exit_count"],
                config=json.dumps(item["config"], sort_keys=True),
            )
        )
    lines.extend(["", "## Recommended Trades", ""])
    if recommended:
        lines.append("|Run|Round|Side|Exec price|Model value|Exec edge|p_hit5|p_hit10|p_loss10|Score|Exit|PnL|")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for trade in recommended["trades"]:
            lines.append(
                "|{run}|{round}|{side}|{exec_price}|{model}|{edge}|{hit5}|{hit10}|{loss10}|{score}|{exit}|{pnl}|".format(
                    run=trade["run_id"],
                    round=trade["round_slug"],
                    side=trade["side"],
                    exec_price=_fmt(trade["execution_price"]),
                    model=_fmt(trade["model_value"]),
                    edge=_fmt(trade["execution_edge"]),
                    hit5=_pct(trade["p_hit_5c"]),
                    hit10=_pct(trade["p_hit_10c"]),
                    loss10=_pct(trade["p_loss_10c"]),
                    score=_fmt(trade["confidence_score"]),
                    exit=trade["exit_reason"],
                    pnl=_fmt(trade["pnl_proxy_usdc"]),
                )
            )
    return "\n".join(lines) + "\n"


def _summary_line(name: str, summary: dict[str, Any]) -> str:
    return (
        f"**{name}**: n `{summary['sample_count']}`, filled `{summary['entry_filled_count']}`, "
        f"avg exec price `{_fmt(summary['avg_execution_price'])}`, "
        f"avg model value `{_fmt(summary['avg_model_value'])}`, "
        f"avg exec edge `{_fmt(summary['avg_execution_edge'])}`, "
        f"hit 5c `{_pct(summary['hit_5c_rate'])}`, hit 10c `{_pct(summary['hit_10c_rate'])}`, "
        f"hit5-before-loss `{_pct(summary['hit_5c_before_loss_10c_rate'])}`, "
        f"loss-before-hit5 `{_pct(summary['loss_10c_before_hit_5c_rate'])}`, "
        f"loss 10c `{_pct(summary['loss_10c_rate'])}`, median best `{_fmt(summary['median_best_move'])}`, "
        f"median min `{_fmt(summary['median_min_move'])}`."
    )


def _replay_block(item: dict[str, Any]) -> list[str]:
    return [
        f"- Trades: `{item['trade_count']}`",
        f"- PnL proxy: `{_fmt(item['pnl_proxy_usdc'])}`",
        f"- Hit 5c / 10c: `{_pct(item['hit_5c_rate'])}` / `{_pct(item['hit_10c_rate'])}`",
        f"- Loss 10c: `{_pct(item['loss_10c_rate'])}`",
        f"- TP exits: `{item['tp_exit_count']}`",
        f"- Filled overlap: `{item['entry_filled_overlap']}`",
        f"- Config: `{json.dumps(item['config'], sort_keys=True)}`",
    ]


def _write_examples(
    path: Path,
    train_examples: list[ConfidenceExample],
    replay_examples: list[ConfidenceExample],
    replay_predictions: list[ConfidencePrediction],
    calibrator: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for split, examples, predictions in (
            ("train", train_examples, [None] * len(train_examples)),
            ("replay", replay_examples, replay_predictions),
        ):
            for idx, example in enumerate(examples):
                signal = example.labeled.signal
                stats = calibrator.lookup(signal)
                row = {
                    "split": split,
                    "run_id": signal.run_id,
                    "round_slug": signal.round_slug,
                    "side": signal.side,
                    "event_id": signal.event_id,
                    "execution_price": signal.execution_price,
                    "model_value": signal.model_value,
                    "execution_edge": _cal._execution_edge(signal),
                    "raw_p_side": signal.raw_p_side,
                    "best_move": example.labeled.best_move,
                    "close_move": example.labeled.close_move,
                    "min_move": example.min_move,
                    "hit_5c": example.labeled.hit_5c,
                    "hit_10c": example.labeled.hit_10c,
                    "loss_10c": example.loss_10c,
                    "hit_5c_before_loss_10c": example.hit_5c_before_loss_10c,
                    "hit_10c_before_loss_10c": example.hit_10c_before_loss_10c,
                    "loss_10c_before_hit_5c": example.loss_10c_before_hit_5c,
                    "calibration": _cal._stats_dict(stats),
                }
                prediction = predictions[idx]
                if prediction is not None:
                    row.update(
                        {
                            "p_hit_5c": prediction.p_hit_5c,
                            "p_hit_10c": prediction.p_hit_10c,
                            "p_loss_10c": prediction.p_loss_10c,
                            "confidence_score": prediction.score,
                        }
                    )
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def _mean(values: Any) -> float | None:
    cleaned = [float(item) for item in values if item is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def _median(values: Any) -> float | None:
    cleaned = sorted(float(item) for item in values if item is not None)
    if not cleaned:
        return None
    mid = len(cleaned) // 2
    if len(cleaned) % 2:
        return cleaned[mid]
    return (cleaned[mid - 1] + cleaned[mid]) / 2.0


def _rate(values: Any) -> float | None:
    cleaned = [bool(item) for item in values if item is not None]
    return sum(1 for item in cleaned if item) / len(cleaned) if cleaned else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.4f}"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
