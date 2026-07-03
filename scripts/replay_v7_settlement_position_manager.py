#!/usr/bin/env python3
"""Replay an EV-based v7 settlement position manager on paper-shadow logs.

The v7 entry target is no longer "which side is more likely" alone; it is
"does the model probability beat the executable market price."  This replay
therefore starts from actual paper settlement fills and evaluates post-entry
signals with hold/add/reduce/exit decisions based on current EV.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
import urllib.error as urlerror
import urllib.parse as parse
import urllib.request as request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Quote:
    ts_ms: int
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True, slots=True)
class SignalRow:
    line_number: int
    event_id: str
    round_slug: str
    canonical_symbol: str
    side: str
    ts_ms: int
    created_at_ms: int
    bridged_at_ms: int
    round_end_ts_ms: int
    token_probability: float
    p_up: float
    p_down: float
    p_neutral: float
    market_implied_prob: float | None
    model_probability: float | None
    polymarket_price: float | None
    mispricing_edge: float | None
    selected_expected_edge: float | None


@dataclass(frozen=True, slots=True)
class ActualEntry:
    round_slug: str
    side: str
    canonical_symbol: str
    event_id: str
    entry_price: float
    shares: float
    cost_basis_usdc: float
    entry_signal_event_id: str
    entry_signal_ts_ms: int
    entry_signal_created_at_ms: int
    opened_at_ms: int
    p_up: float
    p_down: float
    p_neutral: float
    entry_model_probability: float | None
    entry_polymarket_price: float | None
    entry_mispricing_edge: float | None


V7_LATE_FORCE_EXIT_REASON = "convergence_force_exit_before_expiry"
V7_PROFIT_LOCK_BEFORE_EXPIRY_REASON = "convergence_profit_lock_before_expiry"
V7_LOSS_SALVAGE_BEFORE_EXPIRY_REASON = "convergence_loss_salvage_before_expiry"
V7_SLOT_RELEASE_BEFORE_EXPIRY_REASON = "convergence_slot_release_before_expiry"


@dataclass(slots=True)
class SimPosition:
    round_slug: str
    side: str
    canonical_symbol: str
    shares: float
    cost_basis_usdc: float
    realized_pnl: float = 0.0
    reversal_count: int = 0
    weak_hold_count: int = 0
    divergence_count: int = 0
    take_profit_count: int = 0
    take_profit_reason: str = ""
    adverse_confidence_count: int = 0
    adverse_confidence_reduce_count: int = 0
    last_divergence_reduce_at_ms: int = 0
    last_adverse_confidence_reduce_at_ms: int = 0
    exited: bool = False

    @property
    def open(self) -> bool:
        return (not self.exited) and self.shares > 1e-12 and self.cost_basis_usdc > 1e-12

    @property
    def avg_price(self) -> float:
        if self.shares <= 0.0:
            return 0.0
        return self.cost_basis_usdc / self.shares


@dataclass(frozen=True, slots=True)
class Decision:
    round_slug: str
    feature_ts_ms: int
    created_at_ms: int
    action: str
    reason: str
    side: str
    p_side: float | None
    p_opposite: float | None
    hold_bid: float | None
    hold_ask: float | None
    opposite_ask: float | None
    hold_edge: float | None
    add_edge: float | None
    reversal_edge: float | None
    target_cost_basis_usdc: float
    prior_cost_basis_usdc: float
    shares_delta: float
    cash_delta_usdc: float
    realized_pnl_delta: float
    position_shares: float
    position_cost_basis_usdc: float
    reversal_count: int
    weak_hold_count: int
    divergence_count: int
    take_profit_count: int
    take_profit_reason: str
    adverse_confidence_count: int
    adverse_confidence_reduce_count: int
    adverse_confidence_reduce_allowed: bool
    adverse_confidence_add_blocked: bool
    divergence_reduce_allowed: bool
    add_cooldown_remaining_seconds: float
    adverse_confidence: dict[str, Any]
    convergence: dict[str, Any]


def main() -> int:
    args = _parse_args()
    events = _load_jsonl(args.executor_jsonl)
    entries = _actual_entries(events)
    outcomes = _executor_outcomes(events)
    if args.resolve_gamma_outcomes:
        missing = sorted({entry.round_slug for entry in entries} - set(outcomes))
        outcomes.update(
            _resolve_gamma_outcomes(
                missing,
                gamma_api_base=args.gamma_api_base,
                timeout_seconds=args.gamma_timeout_seconds,
            )
        )
    signals = _load_signals(
        Path(args.signal_jsonl),
        min_line_number=_signal_start_line(events),
    )
    signals = _merge_signals(signals, _signals_from_executor_events(events))
    quote_bounds = _quote_load_bounds(entries, signals, args=args)
    quotes = _load_quotes(
        Path(args.raw_jsonl),
        allowed_symbols=quote_bounds["symbols"],
        min_ts_ms=quote_bounds["min_ts_ms"],
        max_ts_ms=quote_bounds["max_ts_ms"],
    )
    decisions, position_summaries = _replay_positions(
        entries=entries,
        signals=signals,
        quotes=quotes,
        outcomes=outcomes,
        args=args,
    )
    report = {
        "inputs": {
            "executor_jsonl": args.executor_jsonl,
            "signal_jsonl": args.signal_jsonl,
            "raw_jsonl": args.raw_jsonl,
        },
        "config": {
            "round_cap_usdc": args.round_cap_usdc,
            "add_edge_min": args.add_edge_min,
            "full_add_edge": args.full_add_edge,
            "weak_hold_edge": args.weak_hold_edge,
            "reduce_fraction": args.reduce_fraction,
            "divergence_reduce_max_hold_edge": args.divergence_reduce_max_hold_edge,
            "exit_hold_edge": args.exit_hold_edge,
            "exit_hysteresis_bars": args.exit_hysteresis_bars,
            "reversal_min_confidence": args.reversal_min_confidence,
            "reversal_min_edge": args.reversal_min_edge,
            "reversal_hysteresis_bars": args.reversal_hysteresis_bars,
            "min_rebalance_usdc": args.min_rebalance_usdc,
            "convergence_price_tolerance": args.convergence_price_tolerance,
            "convergence_model_decay_tolerance": args.convergence_model_decay_tolerance,
            "divergence_hysteresis_bars": args.divergence_hysteresis_bars,
            "add_cooldown_after_divergence_reduce_seconds": (
                args.add_cooldown_after_divergence_reduce_seconds
            ),
            "take_profit_enabled": args.take_profit_enabled,
            "take_profit_hold_edge": args.take_profit_hold_edge,
            "take_profit_residual_ratio": args.take_profit_residual_ratio,
            "take_profit_price_convergence_move": args.take_profit_price_convergence_move,
            "take_profit_price_convergence_hold_edge_ratio": (
                args.take_profit_price_convergence_hold_edge_ratio
            ),
            "take_profit_force_exit_seconds": args.take_profit_force_exit_seconds,
            "take_profit_hysteresis_bars": args.take_profit_hysteresis_bars,
            "take_profit_up_hold_edge_tighten": args.take_profit_up_hold_edge_tighten,
            "take_profit_min_profit_delta": args.take_profit_min_profit_delta,
            "take_profit_min_profit_return": args.take_profit_min_profit_return,
            "adverse_confidence_decay_enabled": args.adverse_confidence_decay_enabled,
            "adverse_confidence_price_delta_start": args.adverse_confidence_price_delta_start,
            "adverse_confidence_base_allowed_decay": args.adverse_confidence_base_allowed_decay,
            "adverse_confidence_price_decay_slope": args.adverse_confidence_price_decay_slope,
            "adverse_confidence_min_allowed_decay": args.adverse_confidence_min_allowed_decay,
            "adverse_confidence_max_required_probability": (
                args.adverse_confidence_max_required_probability
            ),
            "adverse_confidence_exit_probability_buffer": (
                args.adverse_confidence_exit_probability_buffer
            ),
            "adverse_confidence_full_exit_min_model_decay": (
                args.adverse_confidence_full_exit_min_model_decay
            ),
            "adverse_confidence_full_exit_max_hold_edge": (
                args.adverse_confidence_full_exit_max_hold_edge
            ),
            "adverse_confidence_reduce_min_model_decay": (
                args.adverse_confidence_reduce_min_model_decay
            ),
            "adverse_confidence_dust_exit_max_cost": (
                args.adverse_confidence_dust_exit_max_cost
            ),
            "adverse_confidence_dust_exit_min_candidate_count": (
                args.adverse_confidence_dust_exit_min_candidate_count
            ),
            "adverse_confidence_hysteresis_bars": args.adverse_confidence_hysteresis_bars,
            "adverse_confidence_max_reduces": args.adverse_confidence_max_reduces,
            "adverse_confidence_post_reduce_full_exit_enabled": (
                args.adverse_confidence_post_reduce_full_exit_enabled
            ),
            "adverse_confidence_post_reduce_full_exit_bars": (
                args.adverse_confidence_post_reduce_full_exit_bars
            ),
            "adverse_confidence_post_reduce_full_exit_min_model_decay": (
                args.adverse_confidence_post_reduce_full_exit_min_model_decay
            ),
            "adverse_confidence_post_reduce_full_exit_max_hold_edge": (
                args.adverse_confidence_post_reduce_full_exit_max_hold_edge
            ),
            "block_add_after_adverse_confidence_reduce": (
                args.block_add_after_adverse_confidence_reduce
            ),
        },
        "counts": {
            "executor_events": len(events),
            "actual_entries": len(entries),
            "signal_rows": len(signals),
            "quote_symbols": len(quotes),
            "decisions": len(decisions),
            "quote_load_symbols": len(quote_bounds["symbols"]),
        },
        "summary": _summary(position_summaries, decisions),
        "positions": position_summaries,
        "decisions": [asdict(item) for item in decisions],
    }
    if args.output_json_path:
        path = Path(args.output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--signal-jsonl", required=True)
    parser.add_argument("--raw-jsonl", required=True)
    parser.add_argument("--round-cap-usdc", type=float, default=1.0)
    parser.add_argument("--add-edge-min", type=float, default=0.08)
    parser.add_argument("--full-add-edge", type=float, default=0.20)
    parser.add_argument("--weak-hold-edge", type=float, default=0.02)
    parser.add_argument("--reduce-fraction", type=float, default=0.50)
    parser.add_argument("--divergence-reduce-max-hold-edge", type=float, default=0.08)
    parser.add_argument("--exit-hold-edge", type=float, default=-0.02)
    parser.add_argument("--exit-hysteresis-bars", type=int, default=2)
    parser.add_argument("--reversal-min-confidence", type=float, default=0.75)
    parser.add_argument("--reversal-min-edge", type=float, default=0.04)
    parser.add_argument("--reversal-hysteresis-bars", type=int, default=2)
    parser.add_argument("--min-rebalance-usdc", type=float, default=0.05)
    parser.add_argument("--convergence-price-tolerance", type=float, default=0.02)
    parser.add_argument("--convergence-model-decay-tolerance", type=float, default=0.10)
    parser.add_argument("--divergence-hysteresis-bars", type=int, default=2)
    parser.add_argument("--add-cooldown-after-divergence-reduce-seconds", type=float, default=120.0)
    parser.add_argument("--take-profit-enabled", action="store_true")
    parser.add_argument("--take-profit-hold-edge", type=float, default=0.03)
    parser.add_argument("--take-profit-residual-ratio", type=float, default=0.40)
    parser.add_argument("--take-profit-price-convergence-move", type=float, default=0.10)
    parser.add_argument("--take-profit-price-convergence-hold-edge-ratio", type=float, default=0.50)
    parser.add_argument("--take-profit-force-exit-seconds", type=float, default=180.0)
    parser.add_argument("--take-profit-hysteresis-bars", type=int, default=2)
    parser.add_argument("--take-profit-up-hold-edge-tighten", type=float, default=0.01)
    parser.add_argument("--take-profit-min-profit-delta", type=float, default=0.10)
    parser.add_argument("--take-profit-min-profit-return", type=float, default=0.35)
    parser.add_argument("--adverse-confidence-decay-enabled", action="store_true")
    parser.add_argument("--adverse-confidence-price-delta-start", type=float, default=0.10)
    parser.add_argument("--adverse-confidence-base-allowed-decay", type=float, default=0.08)
    parser.add_argument("--adverse-confidence-price-decay-slope", type=float, default=0.30)
    parser.add_argument("--adverse-confidence-min-allowed-decay", type=float, default=0.015)
    parser.add_argument("--adverse-confidence-max-required-probability", type=float, default=0.97)
    parser.add_argument("--adverse-confidence-exit-probability-buffer", type=float, default=0.03)
    parser.add_argument("--adverse-confidence-full-exit-min-model-decay", type=float, default=0.06)
    parser.add_argument("--adverse-confidence-full-exit-max-hold-edge", type=float, default=0.25)
    parser.add_argument("--adverse-confidence-reduce-min-model-decay", type=float, default=0.06)
    parser.add_argument("--adverse-confidence-dust-exit-max-cost", type=float, default=0.15)
    parser.add_argument("--adverse-confidence-dust-exit-min-candidate-count", type=int, default=3)
    parser.add_argument("--adverse-confidence-hysteresis-bars", type=int, default=2)
    parser.add_argument("--adverse-confidence-max-reduces", type=int, default=0)
    parser.add_argument("--adverse-confidence-post-reduce-full-exit-enabled", action="store_true")
    parser.add_argument("--adverse-confidence-post-reduce-full-exit-bars", type=int, default=1)
    parser.add_argument(
        "--adverse-confidence-post-reduce-full-exit-min-model-decay",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--adverse-confidence-post-reduce-full-exit-max-hold-edge",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--block-add-after-adverse-confidence-reduce", action="store_true")
    parser.add_argument(
        "--quote-lookback-seconds",
        type=float,
        default=900.0,
        help="Load raw quotes from this many seconds before the first replay signal.",
    )
    parser.add_argument("--resolve-gamma-outcomes", action="store_true")
    parser.add_argument("--gamma-api-base", default="https://gamma-api.polymarket.com")
    parser.add_argument("--gamma-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()
    if args.round_cap_usdc <= 0:
        raise ValueError("--round-cap-usdc must be positive")
    if args.full_add_edge <= args.add_edge_min:
        raise ValueError("--full-add-edge must be greater than --add-edge-min")
    if not 0 < args.reduce_fraction <= 1:
        raise ValueError("--reduce-fraction must be in (0, 1]")
    if args.exit_hysteresis_bars <= 0 or args.reversal_hysteresis_bars <= 0:
        raise ValueError("hysteresis bars must be positive")
    if args.min_rebalance_usdc < 0:
        raise ValueError("--min-rebalance-usdc must be non-negative")
    if args.convergence_price_tolerance < 0:
        raise ValueError("--convergence-price-tolerance must be non-negative")
    if args.convergence_model_decay_tolerance < 0:
        raise ValueError("--convergence-model-decay-tolerance must be non-negative")
    if args.divergence_hysteresis_bars <= 0:
        raise ValueError("--divergence-hysteresis-bars must be positive")
    if args.add_cooldown_after_divergence_reduce_seconds < 0:
        raise ValueError("--add-cooldown-after-divergence-reduce-seconds must be non-negative")
    if not 0.0 <= args.take_profit_residual_ratio <= 1.0:
        raise ValueError("--take-profit-residual-ratio must be between 0 and 1")
    if args.take_profit_price_convergence_move < 0:
        raise ValueError("--take-profit-price-convergence-move must be non-negative")
    if not 0.0 <= args.take_profit_price_convergence_hold_edge_ratio <= 1.0:
        raise ValueError(
            "--take-profit-price-convergence-hold-edge-ratio must be between 0 and 1"
        )
    if args.take_profit_force_exit_seconds < 0:
        raise ValueError("--take-profit-force-exit-seconds must be non-negative")
    if args.take_profit_hysteresis_bars <= 0:
        raise ValueError("--take-profit-hysteresis-bars must be positive")
    if args.take_profit_up_hold_edge_tighten < 0:
        raise ValueError("--take-profit-up-hold-edge-tighten must be non-negative")
    if args.take_profit_min_profit_delta < 0:
        raise ValueError("--take-profit-min-profit-delta must be non-negative")
    if args.take_profit_min_profit_return < 0:
        raise ValueError("--take-profit-min-profit-return must be non-negative")
    if args.adverse_confidence_price_delta_start < 0:
        raise ValueError("--adverse-confidence-price-delta-start must be non-negative")
    if not 0.0 <= args.adverse_confidence_base_allowed_decay <= 1.0:
        raise ValueError("--adverse-confidence-base-allowed-decay must be between 0 and 1")
    if args.adverse_confidence_price_decay_slope < 0:
        raise ValueError("--adverse-confidence-price-decay-slope must be non-negative")
    if not 0.0 <= args.adverse_confidence_min_allowed_decay <= 1.0:
        raise ValueError("--adverse-confidence-min-allowed-decay must be between 0 and 1")
    if args.adverse_confidence_min_allowed_decay > args.adverse_confidence_base_allowed_decay:
        raise ValueError(
            "--adverse-confidence-min-allowed-decay must be less than or equal to "
            "--adverse-confidence-base-allowed-decay"
        )
    if not 0.0 <= args.adverse_confidence_max_required_probability <= 1.0:
        raise ValueError("--adverse-confidence-max-required-probability must be between 0 and 1")
    if args.adverse_confidence_exit_probability_buffer < 0:
        raise ValueError("--adverse-confidence-exit-probability-buffer must be non-negative")
    if args.adverse_confidence_full_exit_min_model_decay < 0:
        raise ValueError("--adverse-confidence-full-exit-min-model-decay must be non-negative")
    if args.adverse_confidence_reduce_min_model_decay < 0:
        raise ValueError("--adverse-confidence-reduce-min-model-decay must be non-negative")
    if args.adverse_confidence_dust_exit_max_cost < 0:
        raise ValueError("--adverse-confidence-dust-exit-max-cost must be non-negative")
    if args.adverse_confidence_dust_exit_min_candidate_count <= 0:
        raise ValueError("--adverse-confidence-dust-exit-min-candidate-count must be positive")
    if args.adverse_confidence_hysteresis_bars <= 0:
        raise ValueError("--adverse-confidence-hysteresis-bars must be positive")
    if args.adverse_confidence_max_reduces < 0:
        raise ValueError("--adverse-confidence-max-reduces must be non-negative")
    if args.adverse_confidence_post_reduce_full_exit_bars <= 0:
        raise ValueError("--adverse-confidence-post-reduce-full-exit-bars must be positive")
    if args.adverse_confidence_post_reduce_full_exit_min_model_decay < 0:
        raise ValueError(
            "--adverse-confidence-post-reduce-full-exit-min-model-decay must be non-negative"
        )
    if args.quote_lookback_seconds < 0:
        raise ValueError("--quote-lookback-seconds must be non-negative")
    return args


def _load_jsonl(path_text: str | Path) -> list[dict[str, Any]]:
    path = Path(path_text)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _actual_entries(events: list[dict[str, Any]]) -> list[ActualEntry]:
    entries: list[ActualEntry] = []
    seen: set[str] = set()
    for event in events:
        if event.get("event") != "paper_entry_filled":
            continue
        position = event.get("position") or {}
        signal = event.get("signal") or {}
        event_id = str(position.get("event_id") or "")
        if not event_id or event_id in seen:
            continue
        side = str(position.get("side") or signal.get("outcome_side") or "").upper()
        if side not in {"UP", "DOWN"}:
            continue
        entry_price = _float(position.get("entry_price") or position.get("fill_price"))
        shares = _float(position.get("size"))
        if entry_price is None or shares is None or entry_price <= 0 or shares <= 0:
            continue
        entries.append(
            ActualEntry(
                round_slug=str(position.get("round_slug") or signal.get("round_slug") or ""),
                side=side,
                canonical_symbol=str(signal.get("canonical_symbol") or position.get("symbol") or ""),
                event_id=event_id,
                entry_price=entry_price,
                shares=shares,
                cost_basis_usdc=entry_price * shares,
                entry_signal_event_id=str(signal.get("event_id") or position.get("entry_signal_event_id") or ""),
                entry_signal_ts_ms=_millis(signal.get("ts") or position.get("entry_signal_ts")),
                entry_signal_created_at_ms=_millis(
                    signal.get("created_at") or position.get("entry_signal_created_at")
                ),
                opened_at_ms=_millis(position.get("opened_at") or event.get("ts")),
                p_up=_float(signal.get("p_up") or position.get("entry_p_up")) or 0.0,
                p_down=_float(signal.get("p_down") or position.get("entry_p_down")) or 0.0,
                p_neutral=_float(signal.get("p_neutral") or position.get("entry_p_neutral")) or 0.0,
                entry_model_probability=(
                    _float(position.get("entry_model_probability"))
                    or _float(signal.get("model_probability"))
                    or _float(signal.get("token_expected_win_probability"))
                ),
                entry_polymarket_price=(
                    _float(position.get("entry_polymarket_price"))
                    or _float(signal.get("polymarket_price"))
                ),
                entry_mispricing_edge=(
                    _float(position.get("entry_mispricing_edge"))
                    or _float(signal.get("mispricing_edge"))
                    or _float(signal.get("edge"))
                ),
            )
        )
        seen.add(event_id)
    return entries


def _executor_outcomes(events: list[dict[str, Any]]) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for event in events:
        if event.get("event") != "paper_settlement_resolved":
            continue
        position = event.get("position") or {}
        round_slug = str(position.get("round_slug") or "")
        outcome = str(event.get("settlement_result") or position.get("settlement_result") or "").upper()
        if round_slug and outcome in {"UP", "DOWN"}:
            outcomes[round_slug] = outcome
    return outcomes


def _signal_start_line(events: list[dict[str, Any]]) -> int:
    for event in events:
        if event.get("event") != "phase4_started":
            continue
        cursor = event.get("cursor") or {}
        line_number = cursor.get("line_number")
        if isinstance(line_number, int):
            return line_number
    return 0


def _load_signals(path: Path, *, min_line_number: int = 0) -> list[SignalRow]:
    rows: list[SignalRow] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number <= min_line_number or not line.strip():
                continue
            payload = json.loads(line)
            row = _signal_from_payload(payload, line_number=line_number)
            if row is None:
                continue
            dedupe_key = (row.event_id, row.canonical_symbol)
            if row.event_id and dedupe_key in seen:
                continue
            rows.append(row)
            if row.event_id:
                seen.add(dedupe_key)
    rows.sort(key=lambda item: (item.round_slug, item.ts_ms, item.created_at_ms, item.side))
    return rows


def _signals_from_executor_events(events: list[dict[str, Any]]) -> list[SignalRow]:
    rows: list[SignalRow] = []
    for event in events:
        line_number = -int(event.get("_line_number") or 0)
        signal = event.get("signal")
        if isinstance(signal, dict):
            row = _signal_from_payload(signal, line_number=line_number)
            if row is not None:
                rows.append(row)
        signals = event.get("signals")
        if isinstance(signals, list):
            for item in signals:
                if not isinstance(item, dict):
                    continue
                row = _signal_from_payload(item, line_number=line_number)
                if row is not None:
                    rows.append(row)
    return rows


def _merge_signals(*groups: list[SignalRow]) -> list[SignalRow]:
    merged: list[SignalRow] = []
    seen: set[tuple[str, str, int, str]] = set()
    for group in groups:
        for signal in group:
            key = (
                signal.event_id,
                signal.canonical_symbol,
                signal.created_at_ms,
                signal.side,
            )
            if signal.event_id and key in seen:
                continue
            merged.append(signal)
            if signal.event_id:
                seen.add(key)
    merged.sort(key=lambda item: (item.round_slug, item.ts_ms, item.created_at_ms, item.side))
    return merged


def _signal_from_payload(payload: dict[str, Any], *, line_number: int) -> SignalRow | None:
    side = str(payload.get("outcome_side") or payload.get("selected_side") or "").upper()
    if side not in {"UP", "DOWN"}:
        return None
    p_up = _float(payload.get("p_up"))
    p_down = _float(payload.get("p_down"))
    if p_up is None or p_down is None:
        return None
    model_probability = _float(payload.get("model_probability"))
    token_probability = model_probability
    if token_probability is None:
        token_probability = _float(payload.get("token_probability"))
    if token_probability is None:
        token_probability = _float(payload.get("token_expected_win_probability"))
    if token_probability is None:
        token_probability = p_up if side == "UP" else p_down
    polymarket_price = _float(payload.get("polymarket_price"))
    if polymarket_price is None:
        polymarket_price = _float(payload.get("market_implied_prob"))
    mispricing_edge = _float(payload.get("mispricing_edge"))
    if mispricing_edge is None:
        mispricing_edge = _float(payload.get("selected_expected_edge") or payload.get("edge"))
    row = SignalRow(
        line_number=line_number,
        event_id=str(payload.get("event_id") or ""),
        round_slug=str(payload.get("round_slug") or ""),
        canonical_symbol=str(payload.get("canonical_symbol") or ""),
        side=side,
        ts_ms=_millis(payload.get("ts")),
        created_at_ms=_millis(payload.get("created_at")),
        bridged_at_ms=_millis(payload.get("bridged_at")),
        round_end_ts_ms=_millis(payload.get("round_end_ts")),
        token_probability=token_probability,
        p_up=p_up,
        p_down=p_down,
        p_neutral=_float(payload.get("p_neutral")) or 0.0,
        market_implied_prob=_float(payload.get("market_implied_prob")),
        model_probability=model_probability,
        polymarket_price=polymarket_price,
        mispricing_edge=mispricing_edge,
        selected_expected_edge=_float(payload.get("selected_expected_edge") or payload.get("edge")),
    )
    if not row.round_slug or not row.canonical_symbol or row.created_at_ms <= 0:
        return None
    return row


def _quote_load_bounds(
    entries: list[ActualEntry],
    signals: list[SignalRow],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    entry_rounds = {entry.round_slug for entry in entries}
    symbols = {entry.canonical_symbol for entry in entries if entry.canonical_symbol}
    relevant_signal_times: list[int] = []
    for signal in signals:
        if signal.round_slug not in entry_rounds:
            continue
        symbols.add(signal.canonical_symbol)
        relevant_signal_times.append(signal.created_at_ms)
    entry_times = [
        timestamp
        for entry in entries
        for timestamp in (entry.entry_signal_created_at_ms, entry.opened_at_ms)
        if timestamp > 0
    ]
    all_times = entry_times + relevant_signal_times
    lookback_ms = int(args.quote_lookback_seconds * 1000)
    return {
        "symbols": symbols,
        "min_ts_ms": max(0, min(all_times) - lookback_ms) if all_times else 0,
        "max_ts_ms": max(all_times) + 60_000 if all_times else 0,
    }


def _load_quotes(
    path: Path,
    *,
    allowed_symbols: set[str] | None = None,
    min_ts_ms: int = 0,
    max_ts_ms: int = 0,
) -> dict[str, list[Quote]]:
    quotes: dict[str, list[Quote]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or '"raw_top_of_book"' not in line:
                continue
            payload = json.loads(line)
            if payload.get("table") != "raw_top_of_book":
                continue
            row = payload.get("row") or {}
            symbol = str(row.get("canonical_symbol") or "")
            if allowed_symbols and symbol not in allowed_symbols:
                continue
            bid = _float(row.get("bid_price"))
            ask = _float(row.get("ask_price"))
            ts = _millis(
                row.get("ts")
                or row.get("message_ts")
                or row.get("capture_timestamp_ms")
                or row.get("ingest_ts")
                or payload.get("published_at_ms")
            )
            if not symbol or bid is None or ask is None or ts <= 0:
                continue
            if min_ts_ms and ts < min_ts_ms:
                continue
            if max_ts_ms and ts > max_ts_ms:
                continue
            if bid < 0 or ask <= 0 or ask < bid:
                continue
            quotes[symbol].append(Quote(ts_ms=ts, bid=bid, ask=ask))
    for symbol, items in quotes.items():
        items.sort(key=lambda item: item.ts_ms)
        compacted: list[Quote] = []
        last_ts: int | None = None
        for item in items:
            if last_ts == item.ts_ms and compacted:
                compacted[-1] = item
            else:
                compacted.append(item)
                last_ts = item.ts_ms
        quotes[symbol] = compacted
    return dict(quotes)


def _replay_positions(
    *,
    entries: list[ActualEntry],
    signals: list[SignalRow],
    quotes: dict[str, list[Quote]],
    outcomes: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[Decision], list[dict[str, Any]]]:
    signals_by_round: dict[str, list[SignalRow]] = defaultdict(list)
    for signal in signals:
        signals_by_round[signal.round_slug].append(signal)

    decisions: list[Decision] = []
    summaries: list[dict[str, Any]] = []
    for entry in entries:
        position = SimPosition(
            round_slug=entry.round_slug,
            side=entry.side,
            canonical_symbol=entry.canonical_symbol,
            shares=entry.shares,
            cost_basis_usdc=entry.cost_basis_usdc,
        )
        groups = _signal_groups_after_entry(
            signals_by_round.get(entry.round_slug, []),
            entry=entry,
        )
        for feature_ts_ms, group in groups:
            if not position.open:
                break
            decision = _evaluate_group(entry, position, feature_ts_ms, group, quotes, args)
            if decision is None:
                continue
            _apply_decision(position, decision)
            decisions.append(decision)
        outcome = outcomes.get(entry.round_slug)
        settlement_pnl = None
        total_pnl = position.realized_pnl
        if position.open and outcome in {"UP", "DOWN"}:
            settlement_pnl = position.shares * (
                (1.0 if outcome == position.side else 0.0) - position.avg_price
            )
            total_pnl += settlement_pnl
        summaries.append(
            {
                "round_slug": entry.round_slug,
                "entry_event_id": entry.event_id,
                "side": entry.side,
                "entry_price": entry.entry_price,
                "entry_shares": entry.shares,
                "entry_cost_basis_usdc": entry.cost_basis_usdc,
                "entry_p_up": entry.p_up,
                "entry_p_down": entry.p_down,
                "entry_p_neutral": entry.p_neutral,
                "entry_model_probability": entry.entry_model_probability,
                "entry_polymarket_price": entry.entry_polymarket_price,
                "entry_mispricing_edge": entry.entry_mispricing_edge,
                "outcome": outcome,
                "open": position.open,
                "exited": position.exited,
                "remaining_shares": position.shares,
                "remaining_cost_basis_usdc": position.cost_basis_usdc,
                "avg_price": position.avg_price,
                "realized_pnl": position.realized_pnl,
                "settlement_pnl": settlement_pnl,
                "total_pnl_if_known": total_pnl if outcome in {"UP", "DOWN"} else None,
                "adverse_confidence_reduce_count": position.adverse_confidence_reduce_count,
                "decision_count": sum(1 for item in decisions if item.round_slug == entry.round_slug),
            }
        )
    return decisions, summaries


def _signal_groups_after_entry(
    signals: list[SignalRow],
    *,
    entry: ActualEntry,
) -> list[tuple[int, dict[str, SignalRow]]]:
    grouped: dict[int, dict[str, SignalRow]] = defaultdict(dict)
    for signal in signals:
        if signal.created_at_ms <= entry.entry_signal_created_at_ms:
            continue
        if signal.ts_ms < entry.entry_signal_ts_ms:
            continue
        current = grouped[signal.ts_ms].get(signal.side)
        if current is None or signal.created_at_ms > current.created_at_ms:
            grouped[signal.ts_ms][signal.side] = signal
    return sorted(grouped.items(), key=lambda item: item[0])


def _evaluate_group(
    entry: ActualEntry,
    position: SimPosition,
    feature_ts_ms: int,
    group: dict[str, SignalRow],
    quotes: dict[str, list[Quote]],
    args: argparse.Namespace,
) -> Decision | None:
    side_signal = group.get(position.side)
    opposite = _opposite(position.side)
    opposite_signal = group.get(opposite)
    if side_signal is None and opposite_signal is None:
        return None
    evidence_signal = side_signal or opposite_signal
    assert evidence_signal is not None
    eval_ts_ms = max(item.created_at_ms for item in group.values())
    hold_quote = _quote_at(quotes.get(position.canonical_symbol, []), eval_ts_ms)
    p_side = _probability_for_side(side_signal, opposite_signal, position.side)
    p_opposite = _probability_for_side(side_signal, opposite_signal, opposite)
    hold_bid = hold_quote.bid if hold_quote is not None else None
    hold_ask = hold_quote.ask if hold_quote is not None else None
    hold_edge = None if p_side is None or hold_bid is None else p_side - hold_bid
    add_edge = None if p_side is None or hold_ask is None else p_side - hold_ask
    opposite_quote = (
        _quote_at(quotes.get(opposite_signal.canonical_symbol, []), eval_ts_ms)
        if opposite_signal is not None
        else None
    )
    opposite_ask = opposite_quote.ask if opposite_quote is not None else None
    reversal_edge = (
        None if p_opposite is None or opposite_ask is None else p_opposite - opposite_ask
    )
    signal_price_for_side = _signal_price_for_side(side_signal, opposite_signal, position.side)
    convergence = _convergence_evaluation(
        entry=entry,
        position=position,
        p_side=p_side,
        signal_price_for_side=signal_price_for_side,
        args=args,
    )

    reversal_confirmed = (
        p_opposite is not None
        and p_opposite >= args.reversal_min_confidence
        and reversal_edge is not None
        and reversal_edge >= args.reversal_min_edge
    )
    if reversal_confirmed:
        position.reversal_count += 1
    else:
        position.reversal_count = 0

    weak_hold = hold_edge is not None and hold_edge < args.weak_hold_edge
    if weak_hold:
        position.weak_hold_count += 1
    else:
        position.weak_hold_count = 0
    if convergence.get("diverged") is True:
        position.divergence_count += 1
    elif convergence.get("diverged") is False:
        position.divergence_count = 0
    seconds_to_expiry = None
    if evidence_signal.round_end_ts_ms:
        seconds_to_expiry = (evidence_signal.round_end_ts_ms - eval_ts_ms) / 1000.0
    take_profit_candidate, take_profit_reason = _take_profit_candidate(
        side=position.side,
        hold_edge=hold_edge,
        hold_bid=hold_bid,
        avg_price=position.avg_price,
        convergence=convergence,
        seconds_to_expiry=seconds_to_expiry,
        args=args,
    )
    if take_profit_candidate:
        position.take_profit_count += 1
        position.take_profit_reason = take_profit_reason
    else:
        position.take_profit_count = 0
        position.take_profit_reason = ""
    adverse_confidence = _adverse_confidence_decay_evaluation(
        entry=entry,
        position=position,
        p_side=p_side,
        hold_bid=hold_bid,
        args=args,
    )
    if adverse_confidence.get("triggered") is True:
        position.adverse_confidence_count += 1
    else:
        position.adverse_confidence_count = 0
    adverse_confidence["candidate_count"] = position.adverse_confidence_count

    prior_cost = position.cost_basis_usdc
    divergence_reduce_allowed = _divergence_reduce_allowed(
        hold_edge=hold_edge,
        max_hold_edge=args.divergence_reduce_max_hold_edge,
    )
    add_cooldown_remaining_seconds = _add_cooldown_remaining_seconds(
        position=position,
        created_at_ms=evidence_signal.created_at_ms,
        cooldown_seconds=args.add_cooldown_after_divergence_reduce_seconds,
    )
    adverse_confidence_reduce_allowed = _adverse_confidence_reduce_allowed(
        position=position,
        model_decay=float(adverse_confidence.get("model_decay") or 0.0),
        min_model_decay=args.adverse_confidence_reduce_min_model_decay,
        max_reduces=args.adverse_confidence_max_reduces,
    )
    adverse_confidence_add_blocked = (
        args.block_add_after_adverse_confidence_reduce
        and position.adverse_confidence_reduce_count > 0
    )
    action = "HOLD"
    reason = "ev_hold"
    target_cost = prior_cost
    if position.reversal_count >= args.reversal_hysteresis_bars:
        action = "EXIT"
        reason = "confirmed_opposite_ev_reversal"
        target_cost = 0.0
    elif args.take_profit_enabled and position.take_profit_count >= args.take_profit_hysteresis_bars:
        action = "EXIT"
        reason = position.take_profit_reason or "convergence_take_profit"
        target_cost = 0.0
    elif args.take_profit_enabled and position.take_profit_count > 0:
        reason = f"{position.take_profit_reason or 'take_profit'}_hysteresis_wait"
    elif (
        adverse_confidence.get("triggered") is True
        and position.adverse_confidence_count >= args.adverse_confidence_hysteresis_bars
    ):
        shortfall = float(adverse_confidence.get("threshold_shortfall") or 0.0)
        model_decay = float(adverse_confidence.get("model_decay") or 0.0)
        full_exit_allowed = _adverse_confidence_full_exit_allowed(
            shortfall=shortfall,
            model_decay=model_decay,
            hold_edge=hold_edge,
            args=args,
        )
        post_reduce_full_exit_allowed = (
            _adverse_confidence_post_reduce_full_exit_allowed(
                position=position,
                candidate_count=position.adverse_confidence_count,
                shortfall=shortfall,
                model_decay=model_decay,
                hold_edge=hold_edge,
                args=args,
            )
        )
        adverse_confidence["full_exit_allowed"] = full_exit_allowed
        adverse_confidence["post_reduce_full_exit_allowed"] = (
            post_reduce_full_exit_allowed
        )
        adverse_confidence["full_exit_min_model_decay"] = (
            args.adverse_confidence_full_exit_min_model_decay
        )
        adverse_confidence["full_exit_max_hold_edge"] = (
            args.adverse_confidence_full_exit_max_hold_edge
        )
        adverse_confidence["post_reduce_full_exit_bars"] = (
            args.adverse_confidence_post_reduce_full_exit_bars
        )
        adverse_confidence["post_reduce_full_exit_min_model_decay"] = (
            args.adverse_confidence_post_reduce_full_exit_min_model_decay
        )
        adverse_confidence["post_reduce_full_exit_max_hold_edge"] = (
            args.adverse_confidence_post_reduce_full_exit_max_hold_edge
        )
        dust_exit_allowed = _adverse_confidence_dust_exit_allowed(
            prior_cost=prior_cost,
            projected_reduce_cost=max(0.0, prior_cost * (1.0 - args.reduce_fraction)),
            candidate_count=position.adverse_confidence_count,
            args=args,
        )
        adverse_confidence["dust_exit_allowed"] = dust_exit_allowed
        adverse_confidence["dust_exit_max_cost"] = args.adverse_confidence_dust_exit_max_cost
        adverse_confidence["dust_exit_min_candidate_count"] = (
            args.adverse_confidence_dust_exit_min_candidate_count
        )
        reduce_model_decay_allowed = (
            model_decay >= args.adverse_confidence_reduce_min_model_decay
        )
        adverse_confidence["reduce_min_model_decay"] = (
            args.adverse_confidence_reduce_min_model_decay
        )
        adverse_confidence["reduce_model_decay_allowed"] = reduce_model_decay_allowed
        if full_exit_allowed:
            action = "EXIT"
            reason = "adverse_confidence_decay_exit"
            target_cost = 0.0
        elif post_reduce_full_exit_allowed:
            action = "EXIT"
            reason = "adverse_confidence_post_reduce_full_exit"
            target_cost = 0.0
        elif dust_exit_allowed:
            action = "EXIT"
            reason = "adverse_confidence_dust_exit"
            target_cost = 0.0
        elif adverse_confidence_reduce_allowed:
            action = "REDUCE"
            reason = "adverse_confidence_decay_reduce"
            target_cost = max(0.0, prior_cost * (1.0 - args.reduce_fraction))
        elif not reduce_model_decay_allowed:
            reason = "adverse_confidence_reduce_blocked_by_model_decay"
        else:
            reason = "adverse_confidence_reduce_blocked_by_max_reduces"
    elif adverse_confidence.get("triggered") is True:
        reason = "adverse_confidence_decay_hysteresis_wait"
    elif (
        hold_edge is not None
        and hold_edge <= args.exit_hold_edge
        and position.weak_hold_count >= args.exit_hysteresis_bars
    ):
        action = "EXIT"
        reason = "confirmed_negative_hold_edge"
        target_cost = 0.0
    elif (
        hold_edge is not None
        and hold_edge < args.weak_hold_edge
        and position.weak_hold_count >= args.exit_hysteresis_bars
    ):
        action = "REDUCE"
        reason = "weak_hold_edge_reduce"
        target_cost = max(0.0, prior_cost * (1.0 - args.reduce_fraction))
    elif (
        convergence.get("diverged") is True
        and position.divergence_count >= args.divergence_hysteresis_bars
    ):
        if divergence_reduce_allowed:
            action = "REDUCE"
            reason = "residual_divergence_reduce"
            target_cost = max(0.0, prior_cost * (1.0 - args.reduce_fraction))
        else:
            reason = "residual_divergence_reduce_blocked_by_hold_edge"
    elif add_edge is not None and add_edge >= args.add_edge_min:
        if adverse_confidence_add_blocked:
            reason = "positive_add_edge_blocked_by_adverse_confidence_reduce"
        elif add_cooldown_remaining_seconds > 0:
            reason = "positive_add_edge_blocked_by_divergence_reduce_cooldown"
        elif convergence.get("diverged") is True:
            reason = "positive_add_edge_blocked_by_residual_divergence"
        else:
            target_cost = _target_from_edge(
                add_edge=add_edge,
                min_edge=args.add_edge_min,
                full_edge=args.full_add_edge,
                cap_usdc=args.round_cap_usdc,
            )
            if target_cost >= prior_cost + args.min_rebalance_usdc:
                action = "ADD"
                reason = "positive_add_edge"
            else:
                target_cost = prior_cost

    shares_delta = 0.0
    cash_delta = 0.0
    realized_delta = 0.0
    if action in {"EXIT", "REDUCE"}:
        if hold_bid is None:
            action = "HOLD"
            reason = "missing_hold_bid"
            target_cost = prior_cost
        else:
            target_cost = min(target_cost, prior_cost)
            cost_to_sell = max(0.0, prior_cost - target_cost)
            shares_to_sell = min(position.shares, cost_to_sell / max(position.avg_price, 1e-12))
            shares_delta = -shares_to_sell
            cash_delta = shares_to_sell * hold_bid
            realized_delta = shares_to_sell * (hold_bid - position.avg_price)
            if cost_to_sell < args.min_rebalance_usdc and action == "REDUCE":
                action = "HOLD"
                reason = "below_min_reduce"
                target_cost = prior_cost
                shares_delta = cash_delta = realized_delta = 0.0
    elif action == "ADD":
        if hold_ask is None:
            action = "HOLD"
            reason = "missing_hold_ask"
            target_cost = prior_cost
        else:
            add_usdc = max(0.0, target_cost - prior_cost)
            shares_delta = add_usdc / hold_ask
            cash_delta = -add_usdc
            if add_usdc < args.min_rebalance_usdc:
                action = "HOLD"
                reason = "below_min_add"
                target_cost = prior_cost
                shares_delta = cash_delta = 0.0

    return Decision(
        round_slug=entry.round_slug,
        feature_ts_ms=feature_ts_ms,
        created_at_ms=evidence_signal.created_at_ms,
        action=action,
        reason=reason,
        side=position.side,
        p_side=p_side,
        p_opposite=p_opposite,
        hold_bid=hold_bid,
        hold_ask=hold_ask,
        opposite_ask=opposite_ask,
        hold_edge=hold_edge,
        add_edge=add_edge,
        reversal_edge=reversal_edge,
        target_cost_basis_usdc=target_cost,
        prior_cost_basis_usdc=prior_cost,
        shares_delta=shares_delta,
        cash_delta_usdc=cash_delta,
        realized_pnl_delta=realized_delta,
        position_shares=position.shares + shares_delta,
        position_cost_basis_usdc=target_cost,
        reversal_count=position.reversal_count,
        weak_hold_count=position.weak_hold_count,
        divergence_count=position.divergence_count,
        take_profit_count=position.take_profit_count,
        take_profit_reason=position.take_profit_reason,
        adverse_confidence_count=position.adverse_confidence_count,
        adverse_confidence_reduce_count=position.adverse_confidence_reduce_count,
        adverse_confidence_reduce_allowed=adverse_confidence_reduce_allowed,
        adverse_confidence_add_blocked=adverse_confidence_add_blocked,
        divergence_reduce_allowed=divergence_reduce_allowed,
        add_cooldown_remaining_seconds=add_cooldown_remaining_seconds,
        adverse_confidence=adverse_confidence,
        convergence=convergence,
    )


def _probability_for_side(
    side_signal: SignalRow | None,
    opposite_signal: SignalRow | None,
    side: str,
) -> float | None:
    if side_signal is not None and side_signal.side == side:
        return side_signal.token_probability
    if opposite_signal is not None and opposite_signal.side == side:
        return opposite_signal.token_probability
    if side_signal is not None and side_signal.side == _opposite(side):
        return 1.0 - side_signal.token_probability
    if opposite_signal is not None and opposite_signal.side == _opposite(side):
        return 1.0 - opposite_signal.token_probability
    return None


def _signal_price_for_side(
    side_signal: SignalRow | None,
    opposite_signal: SignalRow | None,
    side: str,
) -> float | None:
    if side_signal is not None and side_signal.side == side:
        return _price_from_signal(side_signal)
    if opposite_signal is not None and opposite_signal.side == side:
        return _price_from_signal(opposite_signal)
    if side_signal is not None and side_signal.side == _opposite(side):
        price = _price_from_signal(side_signal)
        return None if price is None else _clamp_probability(1.0 - price)
    if opposite_signal is not None and opposite_signal.side == _opposite(side):
        price = _price_from_signal(opposite_signal)
        return None if price is None else _clamp_probability(1.0 - price)
    return None


def _price_from_signal(signal: SignalRow) -> float | None:
    if signal.polymarket_price is not None:
        return _clamp_probability(signal.polymarket_price)
    if signal.market_implied_prob is not None:
        return _clamp_probability(signal.market_implied_prob)
    return None


def _convergence_evaluation(
    *,
    entry: ActualEntry,
    position: SimPosition,
    p_side: float | None,
    signal_price_for_side: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    entry_probability = entry.entry_model_probability
    if entry_probability is None:
        return {"available": False, "reason": "missing_entry_model_probability"}
    if p_side is None:
        return {
            "available": False,
            "reason": "missing_current_model_probability",
            "entry_model_probability": entry_probability,
        }
    if signal_price_for_side is None:
        return {
            "available": False,
            "reason": "missing_current_signal_price",
            "entry_model_probability": entry_probability,
        }
    entry_price = entry.entry_price
    if entry_price <= 0 and entry.entry_polymarket_price is not None:
        entry_price = entry.entry_polymarket_price
    entry_residual = entry_probability - entry_price
    current_residual = p_side - signal_price_for_side
    if abs(entry_residual) <= 1e-12:
        return {
            "available": False,
            "reason": "entry_residual_too_small",
            "entry_model_probability": entry_probability,
            "entry_price": entry_price,
            "current_model_probability": p_side,
            "current_price": signal_price_for_side,
        }
    direction = 1.0 if entry_residual > 0 else -1.0
    price_move_toward_model = (signal_price_for_side - entry_price) * direction
    model_move_toward_market = (p_side - entry_probability) * direction
    price_diverged = price_move_toward_model < -args.convergence_price_tolerance
    price_converged = price_move_toward_model > args.convergence_price_tolerance
    model_degraded = model_move_toward_market < -args.convergence_model_decay_tolerance
    return {
        "available": True,
        "entry_model_probability": entry_probability,
        "entry_price": entry_price,
        "entry_residual": entry_residual,
        "current_model_probability": p_side,
        "current_price": signal_price_for_side,
        "current_residual": current_residual,
        "price_move_toward_model": price_move_toward_model,
        "model_move_toward_market": model_move_toward_market,
        "residual_abs_ratio": abs(current_residual) / max(abs(entry_residual), 1e-12),
        "price_converged": price_converged,
        "price_diverged": price_diverged,
        "model_degraded": model_degraded,
        "diverged": bool(price_diverged or (model_degraded and not price_converged)),
        "prior_divergence_count": position.divergence_count,
    }


def _adverse_confidence_decay_evaluation(
    *,
    entry: ActualEntry,
    position: SimPosition,
    p_side: float | None,
    hold_bid: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.adverse_confidence_decay_enabled:
        return {"available": False, "reason": "disabled"}
    entry_probability = entry.entry_model_probability
    if entry_probability is None:
        return {"available": False, "reason": "missing_entry_model_probability"}
    if p_side is None:
        return {
            "available": False,
            "reason": "missing_current_model_probability",
            "entry_model_probability": entry_probability,
        }
    if hold_bid is None:
        return {
            "available": False,
            "reason": "missing_hold_bid",
            "entry_model_probability": entry_probability,
            "current_model_probability": p_side,
        }
    if position.avg_price <= 0:
        return {
            "available": False,
            "reason": "missing_average_price",
            "entry_model_probability": entry_probability,
            "current_model_probability": p_side,
            "hold_bid": hold_bid,
        }
    adverse_price_delta = max(0.0, position.avg_price - hold_bid)
    model_decay = max(0.0, entry_probability - p_side)
    raw_allowed_decay = (
        args.adverse_confidence_base_allowed_decay
        - adverse_price_delta * args.adverse_confidence_price_decay_slope
    )
    allowed_decay = max(args.adverse_confidence_min_allowed_decay, raw_allowed_decay)
    required_p_side = min(
        args.adverse_confidence_max_required_probability,
        max(0.0, entry_probability - allowed_decay),
    )
    threshold_shortfall = max(0.0, required_p_side - p_side)
    triggered = (
        adverse_price_delta + 1e-12 >= args.adverse_confidence_price_delta_start
        and threshold_shortfall > 0.0
    )
    return {
        "available": True,
        "triggered": triggered,
        "entry_model_probability": entry_probability,
        "current_model_probability": p_side,
        "avg_price": position.avg_price,
        "hold_bid": hold_bid,
        "adverse_price_delta": adverse_price_delta,
        "price_delta_start": args.adverse_confidence_price_delta_start,
        "model_decay": model_decay,
        "raw_allowed_decay": raw_allowed_decay,
        "allowed_decay": allowed_decay,
        "required_p_side": required_p_side,
        "threshold_shortfall": threshold_shortfall,
    }


def _apply_decision(position: SimPosition, decision: Decision) -> None:
    if decision.action in {"EXIT", "REDUCE"}:
        position.shares = max(0.0, position.shares + decision.shares_delta)
        position.cost_basis_usdc = max(0.0, decision.target_cost_basis_usdc)
        position.realized_pnl += decision.realized_pnl_delta
        if decision.reason == "residual_divergence_reduce":
            position.last_divergence_reduce_at_ms = decision.created_at_ms
        if decision.reason == "adverse_confidence_decay_reduce":
            position.adverse_confidence_reduce_count += 1
            position.last_adverse_confidence_reduce_at_ms = decision.created_at_ms
        if decision.action == "EXIT" or position.shares <= 1e-12:
            position.exited = True
            position.shares = 0.0
            position.cost_basis_usdc = 0.0
    elif decision.action == "ADD":
        position.shares += decision.shares_delta
        position.cost_basis_usdc = decision.target_cost_basis_usdc


def _quote_at(items: list[Quote], ts_ms: int) -> Quote | None:
    if not items:
        return None
    timestamps = [item.ts_ms for item in items]
    index = bisect.bisect_right(timestamps, ts_ms) - 1
    if index < 0:
        return None
    return items[index]


def _divergence_reduce_allowed(*, hold_edge: float | None, max_hold_edge: float) -> bool:
    if max_hold_edge < 0:
        return True
    return hold_edge is not None and hold_edge < max_hold_edge


def _adverse_confidence_reduce_allowed(
    *,
    position: SimPosition,
    model_decay: float,
    min_model_decay: float,
    max_reduces: int,
) -> bool:
    if model_decay < min_model_decay:
        return False
    return max_reduces <= 0 or position.adverse_confidence_reduce_count < max_reduces


def _add_cooldown_remaining_seconds(
    *,
    position: SimPosition,
    created_at_ms: int,
    cooldown_seconds: float,
) -> float:
    if cooldown_seconds <= 0 or position.last_divergence_reduce_at_ms <= 0 or created_at_ms <= 0:
        return 0.0
    elapsed_seconds = (created_at_ms - position.last_divergence_reduce_at_ms) / 1000.0
    return max(0.0, cooldown_seconds - elapsed_seconds)


def _take_profit_candidate(
    *,
    side: str,
    hold_edge: float | None,
    hold_bid: float | None,
    avg_price: float | None,
    convergence: dict[str, Any],
    seconds_to_expiry: float | None,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if not args.take_profit_enabled:
        return False, ""
    tau = args.take_profit_hold_edge
    if side.upper() == "UP":
        tau = max(0.0, tau - args.take_profit_up_hold_edge_tighten)
    if (
        seconds_to_expiry is not None
        and seconds_to_expiry <= args.take_profit_force_exit_seconds
    ):
        return True, _late_force_exit_reason(hold_bid=hold_bid, avg_price=avg_price)
    if _profit_protect_take_profit_candidate(
        hold_bid=hold_bid,
        avg_price=avg_price,
        min_profit_delta=args.take_profit_min_profit_delta,
        min_profit_return=args.take_profit_min_profit_return,
    ):
        return True, "profit_protect_take_profit"
    if hold_edge is not None and hold_edge <= tau:
        return True, "convergence_edge_captured_take_profit"
    if not convergence.get("available"):
        return False, ""
    if convergence.get("price_converged") and convergence.get("model_degraded"):
        return True, "convergence_fake_convergence_model_decay"
    residual_ratio = float(convergence.get("residual_abs_ratio") or 1.0)
    if convergence.get("price_converged") and residual_ratio <= args.take_profit_residual_ratio:
        return True, "convergence_gap_filled_take_profit"
    entry_residual = abs(float(convergence.get("entry_residual") or 0.0))
    price_move_toward_model = float(convergence.get("price_move_toward_model") or 0.0)
    if (
        hold_edge is not None
        and entry_residual > 1e-12
        and price_move_toward_model >= args.take_profit_price_convergence_move
        and hold_edge <= entry_residual * args.take_profit_price_convergence_hold_edge_ratio
    ):
        return True, "convergence_price_move_take_profit"
    return False, ""


def _profit_protect_take_profit_candidate(
    *,
    hold_bid: float | None,
    avg_price: float | None,
    min_profit_delta: float,
    min_profit_return: float,
) -> bool:
    if hold_bid is None or avg_price is None or avg_price <= 0:
        return False
    profit_delta = hold_bid - avg_price
    if min_profit_delta > 0.0 and profit_delta >= min_profit_delta:
        return True
    profit_return = profit_delta / avg_price
    return min_profit_return > 0.0 and profit_return >= min_profit_return


def _adverse_confidence_full_exit_allowed(
    *,
    shortfall: float,
    model_decay: float,
    hold_edge: float | None,
    args: argparse.Namespace,
) -> bool:
    if shortfall < args.adverse_confidence_exit_probability_buffer:
        return False
    if model_decay < args.adverse_confidence_full_exit_min_model_decay:
        return False
    max_hold_edge = args.adverse_confidence_full_exit_max_hold_edge
    if max_hold_edge < 0:
        return True
    return hold_edge is not None and hold_edge <= max_hold_edge


def _adverse_confidence_dust_exit_allowed(
    *,
    prior_cost: float,
    projected_reduce_cost: float,
    candidate_count: int,
    args: argparse.Namespace,
) -> bool:
    max_cost = args.adverse_confidence_dust_exit_max_cost
    if max_cost <= 0.0:
        return False
    return (
        (prior_cost <= max_cost or projected_reduce_cost <= max_cost)
        and candidate_count >= args.adverse_confidence_dust_exit_min_candidate_count
    )


def _adverse_confidence_post_reduce_full_exit_allowed(
    *,
    position: SimPosition,
    candidate_count: int,
    shortfall: float,
    model_decay: float,
    hold_edge: float | None,
    args: argparse.Namespace,
) -> bool:
    if not args.adverse_confidence_post_reduce_full_exit_enabled:
        return False
    if position.adverse_confidence_reduce_count <= 0:
        return False
    max_reduces = args.adverse_confidence_max_reduces
    if max_reduces <= 0 or position.adverse_confidence_reduce_count < max_reduces:
        return False
    required_count = (
        args.adverse_confidence_hysteresis_bars
        + args.adverse_confidence_post_reduce_full_exit_bars
    )
    if candidate_count < required_count:
        return False
    if shortfall < args.adverse_confidence_exit_probability_buffer:
        return False
    if model_decay < args.adverse_confidence_post_reduce_full_exit_min_model_decay:
        return False
    max_hold_edge = args.adverse_confidence_post_reduce_full_exit_max_hold_edge
    if max_hold_edge < 0:
        return True
    return hold_edge is not None and hold_edge <= max_hold_edge


def _late_force_exit_reason(
    *,
    hold_bid: float | None,
    avg_price: float | None,
) -> str:
    if hold_bid is None or avg_price is None or avg_price <= 0:
        return V7_SLOT_RELEASE_BEFORE_EXPIRY_REASON
    if hold_bid >= avg_price:
        return V7_PROFIT_LOCK_BEFORE_EXPIRY_REASON
    return V7_LOSS_SALVAGE_BEFORE_EXPIRY_REASON


def _target_from_edge(*, add_edge: float, min_edge: float, full_edge: float, cap_usdc: float) -> float:
    if add_edge < min_edge:
        return 0.0
    fraction = (add_edge - min_edge) / (full_edge - min_edge)
    return min(cap_usdc, max(0.0, fraction) * cap_usdc)


def _summary(positions: list[dict[str, Any]], decisions: list[Decision]) -> dict[str, Any]:
    action_counts = Counter(item.action for item in decisions)
    reason_counts = Counter(item.reason for item in decisions)
    known_pnls = [
        float(item["total_pnl_if_known"])
        for item in positions
        if item.get("total_pnl_if_known") is not None
    ]
    return {
        "position_count": len(positions),
        "open_position_count": sum(1 for item in positions if item.get("open")),
        "known_outcome_position_count": len(known_pnls),
        "known_total_pnl": sum(known_pnls),
        "decision_count": len(decisions),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# xgboost-v7 Settlement EV Position Replay",
        "",
        "## Summary",
        "",
        f"- Positions: {summary['position_count']}",
        f"- Open positions: {summary['open_position_count']}",
        f"- Decisions: {summary['decision_count']}",
        f"- Action counts: `{json.dumps(summary['action_counts'], sort_keys=True)}`",
        f"- Reason counts: `{json.dumps(summary['reason_counts'], sort_keys=True)}`",
        f"- Known-outcome PnL: {summary['known_total_pnl']:.6f}",
        "",
        "## Positions",
        "",
        "|Round|Side|Entry|Open|Outcome|Realized|Settlement|Total if known|Decisions|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["positions"]:
        lines.append(
            "|{round_slug}|{side}|{entry_price:.4f}|{open}|{outcome}|{realized_pnl:.6f}|{settlement}|{total}|{decision_count}|".format(
                round_slug=item["round_slug"],
                side=item["side"],
                entry_price=float(item["entry_price"]),
                open=str(bool(item["open"])),
                outcome=item.get("outcome") or "",
                realized_pnl=float(item["realized_pnl"]),
                settlement=_fmt_optional(item.get("settlement_pnl")),
                total=_fmt_optional(item.get("total_pnl_if_known")),
                decision_count=int(item["decision_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "|Time UTC|Round|Action|Reason|p_side|p_opp|hold_bid|hold_edge|add_edge|rev_edge|target|realized_delta|",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["decisions"][:200]:
        lines.append(
            "|{time}|{round_slug}|{action}|{reason}|{p_side}|{p_opposite}|{hold_bid}|{hold_edge}|{add_edge}|{reversal_edge}|{target:.4f}|{pnl:.6f}|".format(
                time=_format_utc(int(item["created_at_ms"])),
                round_slug=item["round_slug"],
                action=item["action"],
                reason=item["reason"],
                p_side=_fmt_optional(item.get("p_side")),
                p_opposite=_fmt_optional(item.get("p_opposite")),
                hold_bid=_fmt_optional(item.get("hold_bid")),
                hold_edge=_fmt_optional(item.get("hold_edge")),
                add_edge=_fmt_optional(item.get("add_edge")),
                reversal_edge=_fmt_optional(item.get("reversal_edge")),
                target=float(item["target_cost_basis_usdc"]),
                pnl=float(item["realized_pnl_delta"]),
            )
        )
    if len(report["decisions"]) > 200:
        lines.append(f"\n_Only first 200 decisions shown of {len(report['decisions'])}._")
    lines.append("")
    return "\n".join(lines)


def _resolve_gamma_outcomes(
    round_slugs: list[str],
    *,
    gamma_api_base: str,
    timeout_seconds: float,
) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for round_slug in round_slugs:
        result = _fetch_gamma_outcome(
            round_slug,
            gamma_api_base=gamma_api_base,
            timeout_seconds=timeout_seconds,
        )
        if result in {"UP", "DOWN"}:
            outcomes[round_slug] = result
    return outcomes


def _fetch_gamma_outcome(
    round_slug: str,
    *,
    gamma_api_base: str,
    timeout_seconds: float,
) -> str | None:
    base = str(gamma_api_base).rstrip("/")
    url = f"{base}/markets?{parse.urlencode({'slug': round_slug, 'limit': '1'})}"
    request_obj = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "BiGan-v7-position-replay/1.0",
        },
    )
    try:
        with request.urlopen(request_obj, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urlerror.URLError, json.JSONDecodeError):
        return None
    market = None
    if isinstance(payload, list) and payload:
        market = payload[0]
    elif isinstance(payload, dict):
        markets = payload.get("markets") or payload.get("data") or []
        if isinstance(markets, list) and markets:
            market = markets[0]
    if not isinstance(market, dict):
        return None
    outcomes_raw = market.get("outcomes")
    prices_raw = market.get("outcomePrices")
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    except json.JSONDecodeError:
        return None
    if not isinstance(outcomes, list) or not isinstance(prices, list) or len(outcomes) != len(prices):
        return None
    parsed = [_float(price) for price in prices]
    if any(price is None for price in parsed):
        return None
    best_index = max(range(len(parsed)), key=lambda index: float(parsed[index] or 0.0))
    if float(parsed[best_index] or 0.0) < 0.95:
        return None
    outcome = str(outcomes[best_index]).upper()
    return outcome if outcome in {"UP", "DOWN"} else None


def _millis(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number * 1000) if 0 < number < 10_000_000_000 else int(number)
    text = str(value).strip()
    if not text:
        return 0
    parsed = _float(text)
    if parsed is not None:
        return int(parsed * 1000) if 0 < parsed < 10_000_000_000 else int(parsed)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return int(dt.datetime.fromisoformat(text).timestamp() * 1000)
    except ValueError:
        return 0


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _opposite(side: str) -> str:
    return "DOWN" if side == "UP" else "UP"


def _clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _format_utc(ms: int) -> str:
    if ms <= 0:
        return ""
    return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.UTC).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_optional(value: Any) -> str:
    parsed = _float(value)
    if parsed is None:
        return ""
    return f"{parsed:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
