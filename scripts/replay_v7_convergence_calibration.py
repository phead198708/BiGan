#!/usr/bin/env python3
"""Calibrate v7 convergence signal quality and replay calibrated entry gates.

This is a log-derived audit. It treats each full executor signal payload as a
hypothetical same-side entry and measures whether later same-round same-side
signals showed an executable convergence window. The calibration side is a
bucketed historical lookup; the replay side applies that lookup sequentially so
blocked entries can release later opportunities.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class SignalRow:
    run_id: str
    source_path: str
    source_line: int
    event_id: str
    round_slug: str
    side: str
    ts_ms: int
    created_at_ms: int
    round_end_ts_ms: int
    price: float
    execution_price: float
    model_value: float
    edge: float
    raw_p_side: float | None
    raw_p_opposite: float | None
    seconds_to_expiry: float | None
    entry_filled: bool
    entry_gate_evaluated: bool
    entry_gate_passed: bool
    skip_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabeledSignal:
    signal: SignalRow
    future_count: int
    future_max_price: float
    future_max_ts_ms: int
    future_last_price: float
    future_last_ts_ms: int
    best_move: float
    close_move: float
    hit_5c: bool
    hit_10c: bool
    close_converged: bool
    hit_model_value: bool
    value_error: float
    overprediction_error: float


@dataclass(frozen=True, slots=True)
class CalibrationStats:
    key: tuple[str, ...]
    sample_count: int
    hit_5c_rate: float
    hit_10c_rate: float
    close_rate: float
    median_best_move: float
    median_close_move: float
    median_value_error: float
    model_over_error_p80: float


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    min_price: float
    max_price: float
    min_edge: float
    min_raw_p_side: float
    min_hit_5c_probability: float
    min_hit_10c_probability: float
    max_model_over_error_p80: float
    min_adjusted_median_edge: float = -1.0
    min_adjusted_p80_edge: float = -1.0
    raw_gate_mode: str = "fixed"
    dynamic_raw_base: float = 0.0
    dynamic_raw_price_reference: float = 0.45
    dynamic_raw_price_slope: float = 0.0
    dynamic_raw_edge_reference: float = 0.40
    dynamic_raw_edge_slope: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplayTrade:
    signal: SignalRow
    calibration: CalibrationStats
    raw_p_side_required: float
    exit_ts_ms: int
    exit_price: float
    pnl_proxy_usdc: float
    exit_reason: str


def main() -> int:
    args = _parse_args()
    calibration_rows = _label_signals(_load_signals([Path(item) for item in args.calibration_jsonl]))
    replay_rows = _label_signals(_load_signals([Path(item) for item in args.replay_jsonl]))
    calibrator = _BucketCalibrator(
        calibration_rows,
        min_bucket_size=args.min_bucket_size,
    )
    configs = list(_config_grid(args))
    replay_results = [
        _replay_config(
            replay_rows,
            calibrator=calibrator,
            config=config,
            take_profit_delta=args.take_profit_delta,
            require_entry_gate_pass=args.require_entry_gate_pass,
            respect_existing_entry_skips=args.respect_existing_entry_skips,
            ignored_existing_entry_skip_reasons=args.ignored_existing_entry_skip_reasons,
        )
        for config in configs
    ]
    replay_results.sort(
        key=lambda item: (
            item["pnl_proxy_usdc"],
            item["hit_10c_rate"] or -1.0,
            item["hit_5c_rate"] or -1.0,
            item["trade_count"],
        ),
        reverse=True,
    )
    report = {
        "inputs": {
            "calibration_jsonl": args.calibration_jsonl,
            "replay_jsonl": args.replay_jsonl,
        },
        "config": {
            "min_bucket_size": args.min_bucket_size,
            "take_profit_delta": args.take_profit_delta,
            "price_range": [args.min_price, args.max_price],
            "min_edge_grid": args.min_edge_grid,
            "min_raw_p_side_grid": args.min_raw_p_side_grid,
            "raw_gate_mode_grid": args.raw_gate_mode_grid,
            "dynamic_raw_base_grid": args.dynamic_raw_base_grid,
            "dynamic_raw_price_reference": args.dynamic_raw_price_reference,
            "dynamic_raw_price_slope_grid": args.dynamic_raw_price_slope_grid,
            "dynamic_raw_edge_reference": args.dynamic_raw_edge_reference,
            "dynamic_raw_edge_slope_grid": args.dynamic_raw_edge_slope_grid,
            "min_hit_5c_probability_grid": args.min_hit_5c_probability_grid,
            "min_hit_10c_probability_grid": args.min_hit_10c_probability_grid,
            "max_model_over_error_p80_grid": args.max_model_over_error_p80_grid,
            "min_adjusted_median_edge_grid": args.min_adjusted_median_edge_grid,
            "min_adjusted_p80_edge_grid": args.min_adjusted_p80_edge_grid,
            "require_entry_gate_pass": args.require_entry_gate_pass,
            "respect_existing_entry_skips": args.respect_existing_entry_skips,
            "ignored_existing_entry_skip_reasons": sorted(
                args.ignored_existing_entry_skip_reasons
            ),
        },
        "calibration_summary": _summary(calibration_rows),
        "replay_baseline": _summary(replay_rows),
        "calibration_tables": calibrator.table_report(limit=args.table_limit),
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
    print(json.dumps({"recommended": report["recommended"], "replay_baseline": report["replay_baseline"]}, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-jsonl", action="append", required=True)
    parser.add_argument("--replay-jsonl", action="append", required=True)
    parser.add_argument("--min-bucket-size", type=int, default=20)
    parser.add_argument("--min-price", type=float, default=0.40)
    parser.add_argument("--max-price", type=float, default=0.70)
    parser.add_argument("--min-edge-grid", default="0.30,0.35,0.40")
    parser.add_argument("--min-raw-p-side-grid", default="0,0.55,0.60")
    parser.add_argument(
        "--raw-gate-mode-grid",
        default="fixed",
        help="Comma-separated raw gate modes: fixed, dynamic.",
    )
    parser.add_argument("--dynamic-raw-base-grid", default="0.55")
    parser.add_argument("--dynamic-raw-price-reference", type=float, default=0.45)
    parser.add_argument("--dynamic-raw-price-slope-grid", default="0.20")
    parser.add_argument("--dynamic-raw-edge-reference", type=float, default=0.40)
    parser.add_argument("--dynamic-raw-edge-slope-grid", default="0.50")
    parser.add_argument("--min-hit-5c-probability-grid", default="0,0.40,0.50,0.60")
    parser.add_argument("--min-hit-10c-probability-grid", default="0,0.20,0.30,0.40")
    parser.add_argument("--max-model-over-error-p80-grid", default="1.0,0.45,0.35,0.25")
    parser.add_argument(
        "--min-adjusted-median-edge-grid",
        default="-1.0",
        help=(
            "Comma-separated adjusted-edge thresholds using model_value + "
            "median_value_error - execution_price. Use -1.0 to disable."
        ),
    )
    parser.add_argument(
        "--min-adjusted-p80-edge-grid",
        default="-1.0",
        help=(
            "Comma-separated adjusted-edge thresholds using model_value - "
            "model_over_error_p80 - execution_price. Use -1.0 to disable."
        ),
    )
    parser.add_argument("--take-profit-delta", type=float, default=0.10)
    parser.add_argument(
        "--require-entry-gate-pass",
        action="store_true",
        help="Replay only signals that the current executor entry gate admitted.",
    )
    parser.add_argument(
        "--respect-existing-entry-skips",
        action="store_true",
        help=(
            "Skip signals already rejected by the executor, except reasons "
            "listed in --ignored-existing-entry-skip-reasons."
        ),
    )
    parser.add_argument(
        "--ignored-existing-entry-skip-reasons",
        default="",
        help=(
            "Comma-separated existing skip reasons to ignore when "
            "--respect-existing-entry-skips is enabled."
        ),
    )
    parser.add_argument("--min-recommended-trades", type=int, default=1)
    parser.add_argument("--top-limit", type=int, default=20)
    parser.add_argument("--table-limit", type=int, default=40)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()
    args.min_edge_grid = _float_grid(args.min_edge_grid)
    args.min_raw_p_side_grid = _float_grid(args.min_raw_p_side_grid)
    args.raw_gate_mode_grid = _str_grid(args.raw_gate_mode_grid)
    args.dynamic_raw_base_grid = _float_grid(args.dynamic_raw_base_grid)
    args.dynamic_raw_price_slope_grid = _float_grid(args.dynamic_raw_price_slope_grid)
    args.dynamic_raw_edge_slope_grid = _float_grid(args.dynamic_raw_edge_slope_grid)
    args.min_hit_5c_probability_grid = _float_grid(args.min_hit_5c_probability_grid)
    args.min_hit_10c_probability_grid = _float_grid(args.min_hit_10c_probability_grid)
    args.max_model_over_error_p80_grid = _float_grid(args.max_model_over_error_p80_grid)
    args.min_adjusted_median_edge_grid = _float_grid(args.min_adjusted_median_edge_grid)
    args.min_adjusted_p80_edge_grid = _float_grid(args.min_adjusted_p80_edge_grid)
    args.ignored_existing_entry_skip_reasons = _str_set(
        args.ignored_existing_entry_skip_reasons
    )
    if args.min_bucket_size < 1:
        raise ValueError("--min-bucket-size must be positive")
    if args.min_price >= args.max_price:
        raise ValueError("--min-price must be below --max-price")
    if args.take_profit_delta < 0:
        raise ValueError("--take-profit-delta must be non-negative")
    allowed_modes = {"fixed", "dynamic"}
    unknown_modes = sorted(set(args.raw_gate_mode_grid) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"unknown raw gate mode(s): {', '.join(unknown_modes)}")
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


def _str_grid(text: str) -> list[str]:
    values = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return sorted(set(values))


def _str_set(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def _config_grid(args: argparse.Namespace) -> list[ReplayConfig]:
    configs: list[ReplayConfig] = []
    common_items = list(
        itertools.product(
            args.min_edge_grid,
            args.min_hit_5c_probability_grid,
            args.min_hit_10c_probability_grid,
            args.max_model_over_error_p80_grid,
            args.min_adjusted_median_edge_grid,
            args.min_adjusted_p80_edge_grid,
        )
    )
    if "fixed" in args.raw_gate_mode_grid:
        for (
            min_edge,
            min_hit_5c,
            min_hit_10c,
            max_over_error,
            min_adjusted_median_edge,
            min_adjusted_p80_edge,
        ) in common_items:
            for min_raw_p_side in args.min_raw_p_side_grid:
                configs.append(
                    ReplayConfig(
                        args.min_price,
                        args.max_price,
                        min_edge,
                        min_raw_p_side,
                        min_hit_5c,
                        min_hit_10c,
                        max_over_error,
                        min_adjusted_median_edge=min_adjusted_median_edge,
                        min_adjusted_p80_edge=min_adjusted_p80_edge,
                        raw_gate_mode="fixed",
                    )
                )
    if "dynamic" in args.raw_gate_mode_grid:
        for (
            min_edge,
            min_hit_5c,
            min_hit_10c,
            max_over_error,
            min_adjusted_median_edge,
            min_adjusted_p80_edge,
        ) in common_items:
            for dynamic_base, price_slope, edge_slope in itertools.product(
                args.dynamic_raw_base_grid,
                args.dynamic_raw_price_slope_grid,
                args.dynamic_raw_edge_slope_grid,
            ):
                configs.append(
                    ReplayConfig(
                        args.min_price,
                        args.max_price,
                        min_edge,
                        0.0,
                        min_hit_5c,
                        min_hit_10c,
                        max_over_error,
                        min_adjusted_median_edge=min_adjusted_median_edge,
                        min_adjusted_p80_edge=min_adjusted_p80_edge,
                        raw_gate_mode="dynamic",
                        dynamic_raw_base=dynamic_base,
                        dynamic_raw_price_reference=args.dynamic_raw_price_reference,
                        dynamic_raw_price_slope=price_slope,
                        dynamic_raw_edge_reference=args.dynamic_raw_edge_reference,
                        dynamic_raw_edge_slope=edge_slope,
                    )
                )
    return configs


def _load_signals(paths: list[Path]) -> list[SignalRow]:
    by_event_id: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        run_id = _run_id(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                signal = payload.get("signal")
                if not isinstance(signal, dict):
                    continue
                event_id = str(signal.get("event_id") or "")
                if not event_id:
                    continue
                row = _signal_from_payload(
                    payload,
                    signal,
                    path=path,
                    run_id=run_id,
                    line_number=line_number,
                )
                if row is None:
                    continue
                key = (run_id, event_id)
                current = by_event_id.get(key)
                if current is None:
                    by_event_id[key] = {
                        "row": row,
                        "execution_price": row.execution_price,
                        "entry_filled": payload.get("event") == "paper_entry_filled",
                        "entry_gate_evaluated": payload.get("event") == "entry_gate_evaluated",
                        "entry_gate_passed": _entry_gate_passed(payload),
                        "skip_reasons": set(),
                    }
                elif row.created_at_ms < current["row"].created_at_ms:
                    current["row"] = row
                if payload.get("event") in {"entry_gate_evaluated", "paper_entry_filled"}:
                    by_event_id[key]["execution_price"] = row.execution_price
                if payload.get("event") == "paper_entry_filled":
                    by_event_id[key]["entry_filled"] = True
                    by_event_id[key]["entry_gate_passed"] = True
                if payload.get("event") == "entry_gate_evaluated":
                    by_event_id[key]["entry_gate_evaluated"] = True
                    by_event_id[key]["entry_gate_passed"] = (
                        bool(by_event_id[key]["entry_gate_passed"])
                        or _entry_gate_passed(payload)
                    )
                if payload.get("event") == "entry_skipped":
                    reason = str(payload.get("reason") or "")
                    if reason:
                        by_event_id[key]["skip_reasons"].add(reason)
    rows: list[SignalRow] = []
    for item in by_event_id.values():
        base: SignalRow = item["row"]
        rows.append(
            SignalRow(
                **{
                    **asdict(base),
                    "execution_price": float(item["execution_price"]),
                    "entry_filled": bool(item["entry_filled"]),
                    "entry_gate_evaluated": bool(item["entry_gate_evaluated"]),
                    "entry_gate_passed": bool(item["entry_gate_passed"])
                    and not bool(item["skip_reasons"]),
                    "skip_reasons": tuple(sorted(item["skip_reasons"])),
                }
            )
        )
    return sorted(rows, key=lambda item: (item.created_at_ms, item.event_id))


def _signal_from_payload(
    payload: dict[str, Any],
    signal: dict[str, Any],
    *,
    path: Path,
    run_id: str,
    line_number: int,
) -> SignalRow | None:
    side = str(signal.get("selected_side") or signal.get("outcome_side") or "").upper()
    if side not in {"UP", "DOWN"}:
        return None
    price = _float(signal.get("polymarket_price"))
    model_value = _float(signal.get("model_probability") or signal.get("token_expected_win_probability"))
    if price is None or model_value is None:
        return None
    execution_price = _float(payload.get("worst_price"))
    if execution_price is None:
        execution_price = price
    ts_ms = _millis(signal.get("ts") or signal.get("created_at"))
    created_at_ms = _millis(signal.get("created_at") or signal.get("ts"))
    round_end_ts_ms = _millis(signal.get("round_end_ts"))
    if round_end_ts_ms and round_end_ts_ms < 10_000_000_000:
        round_end_ts_ms *= 1000
    if ts_ms <= 0 or created_at_ms <= 0:
        return None
    if round_end_ts_ms <= 0:
        round_end_ts_ms = _round_end_ms(str(signal.get("round_slug") or ""))
    raw_p_side, raw_p_opposite = _raw_probabilities(signal, side)
    seconds_to_expiry = _float(payload.get("seconds_to_expiry"))
    if seconds_to_expiry is None and round_end_ts_ms > 0:
        seconds_to_expiry = max(0.0, (round_end_ts_ms - created_at_ms) / 1000.0)
    return SignalRow(
        run_id=run_id,
        source_path=str(path),
        source_line=line_number,
        event_id=str(signal.get("event_id")),
        round_slug=str(signal.get("round_slug") or ""),
        side=side,
        ts_ms=ts_ms,
        created_at_ms=created_at_ms,
        round_end_ts_ms=round_end_ts_ms,
        price=price,
        execution_price=execution_price,
        model_value=model_value,
        edge=model_value - price,
        raw_p_side=raw_p_side,
        raw_p_opposite=raw_p_opposite,
        seconds_to_expiry=seconds_to_expiry,
        entry_filled=False,
        entry_gate_evaluated=False,
        entry_gate_passed=False,
        skip_reasons=(),
    )


def _entry_gate_passed(payload: dict[str, Any]) -> bool:
    event = payload.get("event")
    if event == "paper_entry_filled":
        return True
    if event != "entry_gate_evaluated":
        return False
    gate = payload.get("gate_evaluation") or {}
    if gate.get("settlement_gate_passed") is False:
        return False
    raw_gate = gate.get("v7_raw_side_agreement") or {}
    if raw_gate.get("skip_reason"):
        return False
    calibration_gate = gate.get("v7_convergence_calibration") or {}
    if calibration_gate.get("skip_reason"):
        return False
    return True


def _raw_probabilities(signal: dict[str, Any], side: str) -> tuple[float | None, float | None]:
    p_up = _float(signal.get("p_up"))
    p_down = _float(signal.get("p_down"))
    if side == "UP":
        return p_up, p_down
    return p_down, p_up


def _label_signals(signals: list[SignalRow]) -> list[LabeledSignal]:
    by_path: dict[tuple[str, str, str], list[SignalRow]] = defaultdict(list)
    for signal in signals:
        by_path[(signal.run_id, signal.round_slug, signal.side)].append(signal)
    for rows in by_path.values():
        rows.sort(key=lambda item: (item.created_at_ms, item.event_id))
    labeled: list[LabeledSignal] = []
    for signal in signals:
        future = [
            item
            for item in by_path[(signal.run_id, signal.round_slug, signal.side)]
            if item.created_at_ms > signal.created_at_ms
            and (signal.round_end_ts_ms <= 0 or item.created_at_ms <= signal.round_end_ts_ms)
        ]
        if not future:
            continue
        max_row = max(future, key=lambda item: item.price)
        last_row = future[-1]
        best_move = max_row.price - signal.execution_price
        close_move = last_row.price - signal.execution_price
        value_error = max_row.price - signal.model_value
        labeled.append(
            LabeledSignal(
                signal=signal,
                future_count=len(future),
                future_max_price=max_row.price,
                future_max_ts_ms=max_row.created_at_ms,
                future_last_price=last_row.price,
                future_last_ts_ms=last_row.created_at_ms,
                best_move=best_move,
                close_move=close_move,
                hit_5c=best_move >= 0.05 - 1e-12,
                hit_10c=best_move >= 0.10 - 1e-12,
                close_converged=close_move > 0.0,
                hit_model_value=max_row.price >= signal.model_value - 1e-12,
                value_error=value_error,
                overprediction_error=max(0.0, -value_error),
            )
        )
    return labeled


class _BucketCalibrator:
    def __init__(self, rows: list[LabeledSignal], *, min_bucket_size: int) -> None:
        self._min_bucket_size = min_bucket_size
        self._tables: dict[str, dict[tuple[str, ...], CalibrationStats]] = {}
        for name, key_fn in _hierarchy():
            buckets: dict[tuple[str, ...], list[LabeledSignal]] = defaultdict(list)
            for row in rows:
                buckets[key_fn(row.signal)].append(row)
            self._tables[name] = {
                key: _calibration_stats(key, items)
                for key, items in buckets.items()
                if len(items) >= min_bucket_size
            }
        self._global = _calibration_stats(("GLOBAL",), rows)

    def lookup(self, signal: SignalRow) -> CalibrationStats:
        for name, key_fn in _hierarchy():
            table = self._tables[name]
            key = key_fn(signal)
            stats = table.get(key)
            if stats is not None:
                return stats
        return self._global

    def table_report(self, *, limit: int) -> dict[str, list[dict[str, Any]]]:
        report: dict[str, list[dict[str, Any]]] = {}
        for name, table in self._tables.items():
            rows = sorted(
                table.values(),
                key=lambda item: (item.sample_count, item.hit_10c_rate, item.hit_5c_rate),
                reverse=True,
            )
            report[name] = [_stats_dict(row) for row in rows[:limit]]
        return report


def _hierarchy() -> list[tuple[str, Any]]:
    return [
        (
            "price_raw_edge_model",
            lambda signal: (
                _price_bucket(signal.price),
                _raw_bucket(signal.raw_p_side),
                _edge_bucket(signal.edge),
                _model_bucket(signal.model_value),
            ),
        ),
        (
            "price_raw_edge",
            lambda signal: (
                _price_bucket(signal.price),
                _raw_bucket(signal.raw_p_side),
                _edge_bucket(signal.edge),
            ),
        ),
        ("price_raw", lambda signal: (_price_bucket(signal.price), _raw_bucket(signal.raw_p_side))),
        ("price_edge", lambda signal: (_price_bucket(signal.price), _edge_bucket(signal.edge))),
        ("price", lambda signal: (_price_bucket(signal.price),)),
    ]


def _calibration_stats(key: tuple[str, ...], rows: list[LabeledSignal]) -> CalibrationStats:
    if not rows:
        return CalibrationStats(key, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return CalibrationStats(
        key=key,
        sample_count=len(rows),
        hit_5c_rate=_rate(row.hit_5c for row in rows),
        hit_10c_rate=_rate(row.hit_10c for row in rows),
        close_rate=_rate(row.close_converged for row in rows),
        median_best_move=median(row.best_move for row in rows),
        median_close_move=median(row.close_move for row in rows),
        median_value_error=median(row.value_error for row in rows),
        model_over_error_p80=_quantile([row.overprediction_error for row in rows], 0.80),
    )


def _replay_config(
    rows: list[LabeledSignal],
    *,
    calibrator: _BucketCalibrator,
    config: ReplayConfig,
    take_profit_delta: float,
    require_entry_gate_pass: bool,
    respect_existing_entry_skips: bool,
    ignored_existing_entry_skip_reasons: set[str],
) -> dict[str, Any]:
    selected: list[ReplayTrade] = []
    skipped = Counter()
    open_until_ms = -1
    for row in sorted(rows, key=lambda item: (item.signal.created_at_ms, item.signal.event_id)):
        signal = row.signal
        if require_entry_gate_pass and not signal.entry_gate_passed:
            skipped["entry_gate_not_passed"] += 1
            continue
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
        stats = calibrator.lookup(signal)
        raw_p_side_required = _raw_p_side_required(signal, config)
        reason = _skip_reason(signal, stats, config)
        if reason is not None:
            skipped[reason] += 1
            continue
        if row.future_max_price >= signal.execution_price + take_profit_delta - 1e-12:
            exit_price = signal.execution_price + take_profit_delta
            exit_ts_ms = row.future_max_ts_ms
            exit_reason = f"tp_{take_profit_delta:.2f}"
        else:
            exit_price = row.future_last_price
            exit_ts_ms = max(row.future_last_ts_ms, signal.round_end_ts_ms)
            exit_reason = "last_observed"
        pnl_proxy = (
            (exit_price - signal.execution_price) / signal.execution_price
            if signal.execution_price > 0
            else 0.0
        )
        selected.append(
            ReplayTrade(
                signal=signal,
                calibration=stats,
                raw_p_side_required=raw_p_side_required,
                exit_ts_ms=exit_ts_ms,
                exit_price=exit_price,
                pnl_proxy_usdc=pnl_proxy,
                exit_reason=exit_reason,
            )
        )
        open_until_ms = exit_ts_ms
    return {
        "config": asdict(config),
        "trade_count": len(selected),
        "pnl_proxy_usdc": sum(item.pnl_proxy_usdc for item in selected),
        "hit_5c_rate": _rate(
            item.exit_price >= item.signal.execution_price + 0.05 - 1e-12
            for item in selected
        ),
        "hit_10c_rate": _rate(
            item.exit_price >= item.signal.execution_price + 0.10 - 1e-12
            for item in selected
        ),
        "tp_exit_count": sum(1 for item in selected if item.exit_reason.startswith("tp_")),
        "last_exit_count": sum(1 for item in selected if item.exit_reason == "last_observed"),
        "entry_filled_overlap": sum(1 for item in selected if item.signal.entry_filled),
        "entry_gate_passed_overlap": sum(1 for item in selected if item.signal.entry_gate_passed),
        "skipped": dict(sorted(skipped.items())),
        "trades": [_trade_dict(item) for item in selected],
    }


def _skip_reason(signal: SignalRow, stats: CalibrationStats, config: ReplayConfig) -> str | None:
    if signal.execution_price < config.min_price:
        return "entry_price_below_min"
    if signal.execution_price > config.max_price:
        return "entry_price_above_max"
    if _execution_edge(signal) < config.min_edge:
        return "edge_below_min"
    raw_p_side_required = _raw_p_side_required(signal, config)
    if raw_p_side_required > 0 and (
        signal.raw_p_side is None or signal.raw_p_side < raw_p_side_required
    ):
        return "raw_p_side_below_min"
    if stats.hit_5c_rate < config.min_hit_5c_probability:
        return "calibrated_hit_5c_below_min"
    if stats.hit_10c_rate < config.min_hit_10c_probability:
        return "calibrated_hit_10c_below_min"
    if stats.model_over_error_p80 > config.max_model_over_error_p80:
        return "model_over_error_p80_above_max"
    if _adjusted_median_edge(signal, stats) < config.min_adjusted_median_edge:
        return "calibrated_median_edge_below_min"
    if _adjusted_p80_edge(signal, stats) < config.min_adjusted_p80_edge:
        return "calibrated_p80_edge_below_min"
    return None


def _raw_p_side_required(signal: SignalRow, config: ReplayConfig) -> float:
    if config.raw_gate_mode == "dynamic":
        price_penalty = config.dynamic_raw_price_slope * max(
            0.0,
            signal.execution_price - config.dynamic_raw_price_reference,
        )
        edge_penalty = config.dynamic_raw_edge_slope * max(
            0.0,
            config.dynamic_raw_edge_reference - _execution_edge(signal),
        )
        return config.dynamic_raw_base + price_penalty + edge_penalty
    return config.min_raw_p_side


def _execution_edge(signal: SignalRow) -> float:
    return signal.model_value - signal.execution_price


def _adjusted_median_value(signal: SignalRow, stats: CalibrationStats) -> float:
    return _clip_probability(signal.model_value + stats.median_value_error)


def _adjusted_median_edge(signal: SignalRow, stats: CalibrationStats) -> float:
    return _adjusted_median_value(signal, stats) - signal.execution_price


def _adjusted_p80_value(signal: SignalRow, stats: CalibrationStats) -> float:
    return _clip_probability(signal.model_value - stats.model_over_error_p80)


def _adjusted_p80_edge(signal: SignalRow, stats: CalibrationStats) -> float:
    return _adjusted_p80_value(signal, stats) - signal.execution_price


def _clip_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _recommended(results: list[dict[str, Any]], *, min_trades: int) -> dict[str, Any] | None:
    for item in results:
        if int(item["trade_count"]) >= min_trades:
            return item
    return results[0] if results else None


def _summary(rows: list[LabeledSignal], *, include_buckets: bool = True) -> dict[str, Any]:
    result = {
        "sample_count": len(rows),
        "entry_filled_count": sum(1 for row in rows if row.signal.entry_filled),
        "avg_price": _mean(row.signal.price for row in rows),
        "avg_execution_price": _mean(row.signal.execution_price for row in rows),
        "avg_model_value": _mean(row.signal.model_value for row in rows),
        "avg_edge": _mean(row.signal.edge for row in rows),
        "avg_execution_edge": _mean(_execution_edge(row.signal) for row in rows),
        "hit_5c_rate": _rate(row.hit_5c for row in rows),
        "hit_10c_rate": _rate(row.hit_10c for row in rows),
        "close_rate": _rate(row.close_converged for row in rows),
        "median_best_move": _median(row.best_move for row in rows),
        "median_close_move": _median(row.close_move for row in rows),
        "median_value_error": _median(row.value_error for row in rows),
        "model_over_error_p80": _quantile([row.overprediction_error for row in rows], 0.80) if rows else None,
    }
    if include_buckets:
        result["by_price_bucket"] = {
            key: _summary(items, include_buckets=False)
            for key, items in sorted(_group_labeled(rows, lambda row: _price_bucket(row.signal.price)).items())
        }
    return result


def _group_labeled(rows: list[LabeledSignal], key_fn: Any) -> dict[str, list[LabeledSignal]]:
    groups: dict[str, list[LabeledSignal]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return groups


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# v7 convergence calibration replay",
        "",
        "## Inputs",
        "",
        f"- Calibration logs: `{', '.join(report['inputs']['calibration_jsonl'])}`",
        f"- Replay logs: `{', '.join(report['inputs']['replay_jsonl'])}`",
        "",
        "## Summary",
        "",
        _summary_table("Calibration", report["calibration_summary"]),
        "",
        _summary_table("Replay baseline", report["replay_baseline"]),
        "",
        "## Recommended Replay",
        "",
    ]
    recommended = report.get("recommended")
    if recommended is None:
        lines.append("No replay configuration was produced.")
    else:
        lines.extend(_replay_block(recommended))
    lines.extend(["", "## Top Replay Configs", ""])
    lines.append("|Rank|Trades|PnL proxy|Hit 5c|Hit 10c|TP exits|Filled overlap|Config|")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for idx, item in enumerate(report["top_replay_configs"], start=1):
        lines.append(
            "|{idx}|{trades}|{pnl}|{hit5}|{hit10}|{tp}|{overlap}|`{config}`|".format(
                idx=idx,
                trades=item["trade_count"],
                pnl=_fmt(item["pnl_proxy_usdc"]),
                hit5=_pct(item["hit_5c_rate"]),
                hit10=_pct(item["hit_10c_rate"]),
                tp=item["tp_exit_count"],
                overlap=item["entry_filled_overlap"],
                config=json.dumps(item["config"], sort_keys=True),
            )
        )
    lines.extend(["", "## Replay Trades", ""])
    if recommended:
        lines.append("|Run|Round|Side|Price|Exec price|Model value|Exec edge|Adj med edge|Adj p80 edge|Raw p_side|Cal hit 5c|Cal hit 10c|Exit|PnL proxy|")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for trade in recommended["trades"]:
            lines.append(
                "|{run}|{round}|{side}|{price}|{exec_price}|{model}|{exec_edge}|{adj_med}|{adj_p80}|{raw}|{hit5}|{hit10}|{exit}|{pnl}|".format(
                    run=trade["run_id"],
                    round=trade["round_slug"],
                    side=trade["side"],
                    price=_fmt(trade["price"]),
                    exec_price=_fmt(trade["execution_price"]),
                    model=_fmt(trade["model_value"]),
                    exec_edge=_fmt(trade["execution_edge"]),
                    adj_med=_fmt(trade["adjusted_median_edge"]),
                    adj_p80=_fmt(trade["adjusted_p80_edge"]),
                    raw=_fmt(trade["raw_p_side"]),
                    hit5=_pct(trade["calibration"]["hit_5c_rate"]),
                    hit10=_pct(trade["calibration"]["hit_10c_rate"]),
                    exit=trade["exit_reason"],
                    pnl=_fmt(trade["pnl_proxy_usdc"]),
                )
            )
    return "\n".join(lines) + "\n"


def _summary_table(name: str, summary: dict[str, Any]) -> str:
    return (
        f"**{name}**: n `{summary['sample_count']}`, filled `{summary['entry_filled_count']}`, "
        f"avg price `{_fmt(summary['avg_price'])}`, avg exec price `{_fmt(summary['avg_execution_price'])}`, "
        f"avg model value `{_fmt(summary['avg_model_value'])}`, avg edge `{_fmt(summary['avg_edge'])}`, "
        f"avg exec edge `{_fmt(summary['avg_execution_edge'])}`, hit 5c `{_pct(summary['hit_5c_rate'])}`, "
        f"hit 10c `{_pct(summary['hit_10c_rate'])}`, close `{_pct(summary['close_rate'])}`, "
        f"median best move `{_fmt(summary['median_best_move'])}`, "
        f"p80 over-error `{_fmt(summary['model_over_error_p80'])}`."
    )


def _replay_block(item: dict[str, Any]) -> list[str]:
    return [
        f"- Trades: `{item['trade_count']}`",
        f"- PnL proxy: `{_fmt(item['pnl_proxy_usdc'])}`",
        f"- Hit 5c / 10c: `{_pct(item['hit_5c_rate'])}` / `{_pct(item['hit_10c_rate'])}`",
        f"- TP exits: `{item['tp_exit_count']}`",
        f"- Filled overlap: `{item['entry_filled_overlap']}`",
        f"- Config: `{json.dumps(item['config'], sort_keys=True)}`",
    ]


def _trade_dict(trade: ReplayTrade) -> dict[str, Any]:
    return {
        "run_id": trade.signal.run_id,
        "round_slug": trade.signal.round_slug,
        "side": trade.signal.side,
        "event_id": trade.signal.event_id,
        "created_at_ms": trade.signal.created_at_ms,
        "price": trade.signal.price,
        "execution_price": trade.signal.execution_price,
        "model_value": trade.signal.model_value,
        "edge": trade.signal.edge,
        "execution_edge": _execution_edge(trade.signal),
        "adjusted_median_value": _adjusted_median_value(trade.signal, trade.calibration),
        "adjusted_median_edge": _adjusted_median_edge(trade.signal, trade.calibration),
        "adjusted_p80_value": _adjusted_p80_value(trade.signal, trade.calibration),
        "adjusted_p80_edge": _adjusted_p80_edge(trade.signal, trade.calibration),
        "raw_p_side": trade.signal.raw_p_side,
        "raw_p_side_required": trade.raw_p_side_required,
        "entry_filled": trade.signal.entry_filled,
        "entry_gate_evaluated": trade.signal.entry_gate_evaluated,
        "entry_gate_passed": trade.signal.entry_gate_passed,
        "skip_reasons": list(trade.signal.skip_reasons),
        "calibration": _stats_dict(trade.calibration),
        "exit_ts_ms": trade.exit_ts_ms,
        "exit_price": trade.exit_price,
        "exit_reason": trade.exit_reason,
        "pnl_proxy_usdc": trade.pnl_proxy_usdc,
    }


def _stats_dict(stats: CalibrationStats) -> dict[str, Any]:
    return {
        "key": list(stats.key),
        "sample_count": stats.sample_count,
        "hit_5c_rate": stats.hit_5c_rate,
        "hit_10c_rate": stats.hit_10c_rate,
        "close_rate": stats.close_rate,
        "median_best_move": stats.median_best_move,
        "median_close_move": stats.median_close_move,
        "median_value_error": stats.median_value_error,
        "model_over_error_p80": stats.model_over_error_p80,
    }


def _price_bucket(price: float) -> str:
    if price < 0.30:
        return "<0.30"
    if price < 0.40:
        return "0.30-0.40"
    if price < 0.50:
        return "0.40-0.50"
    if price < 0.70:
        return "0.50-0.70"
    return ">=0.70"


def _raw_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.55:
        return "<0.55"
    if value < 0.60:
        return "0.55-0.60"
    if value < 0.65:
        return "0.60-0.65"
    return ">=0.65"


def _edge_bucket(edge: float) -> str:
    if edge < 0.30:
        return "<0.30"
    if edge < 0.40:
        return "0.30-0.40"
    if edge < 0.50:
        return "0.40-0.50"
    return ">=0.50"


def _model_bucket(value: float) -> str:
    if value < 0.70:
        return "<0.70"
    if value < 0.80:
        return "0.70-0.80"
    return ">=0.80"


def _run_id(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("xgboost-v7-paper-shadow-"):
            return part
    match = re.search(r"phase4-(.+?)\\.jsonl$", path.name)
    return match.group(1) if match else path.stem


def _round_end_ms(round_slug: str) -> int:
    match = re.search(r"-(\\d+)$", round_slug)
    return (int(match.group(1)) + 900) * 1000 if match else 0


def _millis(value: Any) -> int:
    try:
        if value is None:
            return 0
        number = float(value)
        if math.isnan(number):
            return 0
        if number < 10_000_000_000:
            number *= 1000
        return int(number)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return None if math.isnan(result) else result
    except (TypeError, ValueError):
        return None


def _mean(values: Any) -> float | None:
    cleaned = [float(item) for item in values if item is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def _median(values: Any) -> float | None:
    cleaned = [float(item) for item in values if item is not None]
    return median(cleaned) if cleaned else None


def _rate(values: Any) -> float | None:
    cleaned = [bool(item) for item in values if item is not None]
    return sum(1 for item in cleaned if item) / len(cleaned) if cleaned else None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
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
