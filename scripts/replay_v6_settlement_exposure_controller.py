#!/usr/bin/env python3
"""Replay a v6 settlement target-exposure controller on BTC-15M rows.

The existing v6 settlement replay answers "should we take one fixed-size bet?"
This diagnostic answers a different execution question: given fresh settlement
probabilities and executable prices, what target USDC exposure should one round
enter with, and whether a risk-only exit policy improves over a fixed one-shot
settlement entry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
import urllib.error as urlerror
import urllib.parse as parse
import urllib.request as request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = REPO_ROOT / "scripts" / "replay_v6_btc15m_execution_restricted.py"


@dataclass(frozen=True, slots=True)
class Quote:
    ask: float
    bid: float
    buy_price: float
    sell_price: float


@dataclass(slots=True)
class SimPosition:
    side: str = ""
    shares: float = 0.0
    cost_basis_usdc: float = 0.0
    exit_opposite_candidate_side: str = ""
    exit_opposite_candidate_count: int = 0

    @property
    def open(self) -> bool:
        return self.side in {"UP", "DOWN"} and self.shares > 0.0 and self.cost_basis_usdc > 0.0

    @property
    def avg_price(self) -> float:
        if self.shares <= 0.0:
            return 0.0
        return self.cost_basis_usdc / self.shares


@dataclass(frozen=True, slots=True)
class SimFill:
    split: str
    strategy: str
    round_slug: str
    feature_ts: int
    action: str
    side: str
    price: float
    shares: float
    notional_usdc: float
    realized_pnl: float
    p_up: float
    p_down: float
    p_side: float
    edge_buy: float
    edge_hold: float | None
    target_usdc: float
    reason: str


@dataclass(frozen=True, slots=True)
class RoundResult:
    split: str
    strategy: str
    round_slug: str
    side: str
    true_label: str
    pnl: float
    buy_notional_usdc: float
    sell_notional_usdc: float
    fill_count: int
    entry_count: int
    exit_count: int
    final_shares: float


@dataclass(frozen=True, slots=True)
class ShadowSignal:
    split: str
    round_slug: str
    event_ts_ms: int
    signal_ts_ms: int
    side: str
    quote_side: str
    can_enter: bool
    p_up: float
    p_down: float
    p_neutral: float
    ask: float
    bid: float
    worst_price: float
    seconds_to_expiry: float
    settlement_edge: float
    signal_age_seconds: float | None


def main() -> int:
    args = _parse_args()
    if args.exit_opposite_hysteresis_bars <= 0:
        raise ValueError("--exit-opposite-hysteresis-bars must be positive")
    if args.shadow_log_jsonl:
        return _main_shadow_log(args)

    replay = _load_replay_module()
    model = replay.load_xgboost_v6_model(Path(args.model_json_path))
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    report: dict[str, Any] = {
        "model_json_path": args.model_json_path,
        "dataset_dir": args.dataset_dir,
        "splits": splits,
        "family_filter": "BTC-15M",
        "controller_config": {
            "round_cap_usdc": args.round_cap_usdc,
            "entry_edge_min": args.entry_edge_min,
            "hold_edge_min": args.hold_edge_min,
            "exit_negative_edge_threshold": args.exit_negative_edge_threshold,
            "exit_confidence_floor": args.exit_confidence_floor,
            "exit_opposite_min_confidence": args.exit_opposite_min_confidence,
            "exit_opposite_hysteresis_bars": args.exit_opposite_hysteresis_bars,
            "exit_opposite_gap": args.exit_opposite_gap,
            "exit_price_stop_delta": args.exit_price_stop_delta,
            "exit_stop_loss_usdc": args.exit_stop_loss_usdc,
            "full_size_edge": args.full_size_edge,
            "flip_edge_gap": args.flip_edge_gap,
            "min_rebalance_usdc": args.min_rebalance_usdc,
            "settlement_threshold": args.settlement_threshold,
            "settlement_min_confidence": args.settlement_min_confidence,
            "probability_shrinkage": args.probability_shrinkage,
            "buy_slippage": args.buy_slippage,
            "sell_slippage": args.sell_slippage,
            "fallback_spread": args.fallback_spread,
            "min_seconds_to_expiry": args.min_seconds_to_expiry,
            "max_seconds_to_expiry": args.max_seconds_to_expiry,
            "no_new_entry_before_expiry_seconds": args.no_new_entry_before_expiry_seconds,
            "allow_reduce_until_expiry": args.allow_reduce_until_expiry,
        },
        "strategies": {},
        "splits_detail": {},
    }

    combined: dict[str, list[RoundResult]] = defaultdict(list)
    combined_fills: dict[str, list[SimFill]] = defaultdict(list)
    for split in splits:
        rows = replay._load_btc15_rows(Path(args.dataset_dir) / f"{split}.parquet")
        payloads = model.predict_payload_many(rows)
        fixed_results, fixed_fills = _simulate_fixed_one_shot(
            replay=replay,
            split=split,
            rows=rows,
            payloads=payloads,
            args=args,
        )
        dynamic_results, dynamic_fills = _simulate_dynamic_exposure(
            replay=replay,
            split=split,
            rows=rows,
            payloads=payloads,
            args=args,
        )
        split_payload = {
            "fixed_one_shot": _summarize_results(fixed_results, fixed_fills),
            "dynamic_exposure": _summarize_results(dynamic_results, dynamic_fills),
        }
        report["splits_detail"][split] = split_payload
        combined["fixed_one_shot"].extend(fixed_results)
        combined["dynamic_exposure"].extend(dynamic_results)
        combined_fills["fixed_one_shot"].extend(fixed_fills)
        combined_fills["dynamic_exposure"].extend(dynamic_fills)

    for strategy in ("fixed_one_shot", "dynamic_exposure"):
        report["strategies"][strategy] = _summarize_results(
            combined[strategy],
            combined_fills[strategy],
        )

    if args.output_json_path:
        path = Path(args.output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_report(report, args=args), encoding="utf-8")

    print(json.dumps(report["strategies"], indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    default_model = (
        "data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/"
        "model-single-grid/model.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json-path", default=default_model)
    parser.add_argument(
        "--dataset-dir",
        default=(
            "data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/dataset"
        ),
    )
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--settlement-threshold", type=float, default=0.50)
    parser.add_argument("--settlement-min-confidence", type=float, default=0.80)
    parser.add_argument("--entry-edge-min", type=float, default=0.082)
    parser.add_argument("--hold-edge-min", type=float, default=-0.02)
    parser.add_argument("--exit-negative-edge-threshold", type=float, default=-0.02)
    parser.add_argument("--exit-confidence-floor", type=float, default=0.55)
    parser.add_argument("--exit-opposite-min-confidence", type=float, default=0.75)
    parser.add_argument("--exit-opposite-hysteresis-bars", type=int, default=2)
    parser.add_argument("--exit-opposite-gap", type=float, default=0.0)
    parser.add_argument("--exit-price-stop-delta", type=float, default=0.15)
    parser.add_argument("--exit-stop-loss-usdc", type=float, default=0.50)
    parser.add_argument("--full-size-edge", type=float, default=0.10)
    parser.add_argument("--flip-edge-gap", type=float, default=0.12)
    parser.add_argument("--round-cap-usdc", type=float, default=1.0)
    parser.add_argument("--min-rebalance-usdc", type=float, default=0.05)
    parser.add_argument("--probability-shrinkage", type=float, default=0.0)
    parser.add_argument("--buy-slippage", type=float, default=0.02)
    parser.add_argument("--sell-slippage", type=float, default=0.02)
    parser.add_argument("--fallback-spread", type=float, default=0.02)
    parser.add_argument("--min-seconds-to-expiry", type=float, default=300.0)
    parser.add_argument("--max-seconds-to-expiry", type=float, default=1200.0)
    parser.add_argument("--no-new-entry-before-expiry-seconds", type=float, default=300.0)
    parser.add_argument(
        "--allow-reduce-until-expiry",
        dest="allow_reduce_until_expiry",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-reduce-until-expiry",
        dest="allow_reduce_until_expiry",
        action="store_false",
    )
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    parser.add_argument(
        "--shadow-log-jsonl",
        default="",
        help="Replay using settlement entry_gate_evaluated events from a paper-shadow executor JSONL.",
    )
    parser.add_argument("--shadow-log-label", default="shadow")
    parser.add_argument("--resolve-missing-shadow-outcomes", action="store_true")
    parser.add_argument("--gamma-api-base", default="https://gamma-api.polymarket.com")
    parser.add_argument("--gamma-timeout-seconds", type=float, default=10.0)
    return parser.parse_args()


def _simulate_fixed_one_shot(
    *,
    replay: Any,
    split: str,
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    args: argparse.Namespace,
) -> tuple[list[RoundResult], list[SimFill]]:
    filled_rounds: set[str] = set()
    results: list[RoundResult] = []
    fills: list[SimFill] = []
    for row, payload in zip(rows, payloads, strict=True):
        round_slug = replay._round_slug(row)
        if round_slug in filled_rounds:
            continue
        timing_skip = _entry_timing_skip(replay, row, args)
        if timing_skip is not None:
            continue
        market_payload = replay.market_v6_payload_from_token_payload(
            payload,
            token_side=replay._outcome_side(row),
        )
        side, p_side, p_up, p_down = _settlement_side(market_payload, args=args)
        if side is None:
            continue
        if p_side < args.settlement_min_confidence:
            continue
        quote = _quote_for_side(row, side, args=args)
        if quote is None:
            continue
        edge_buy = p_side - quote.buy_price
        if edge_buy < args.entry_edge_min:
            continue
        shares = args.round_cap_usdc / quote.buy_price
        true_label = replay._settlement_label(row)
        pnl = _settlement_pnl(side=side, true_label=true_label, shares=shares, avg_price=quote.buy_price)
        feature_ts = int(row.get("feature_ts") or 0)
        fills.append(
            SimFill(
                split=split,
                strategy="fixed_one_shot",
                round_slug=round_slug,
                feature_ts=feature_ts,
                action="BUY",
                side=side,
                price=quote.buy_price,
                shares=shares,
                notional_usdc=args.round_cap_usdc,
                realized_pnl=0.0,
                p_up=p_up,
                p_down=p_down,
                p_side=p_side,
                edge_buy=edge_buy,
                edge_hold=None,
                target_usdc=args.round_cap_usdc,
                reason="first_eligible_full_size",
            )
        )
        results.append(
            RoundResult(
                split=split,
                strategy="fixed_one_shot",
                round_slug=round_slug,
                side=side,
                true_label=true_label,
                pnl=pnl,
                buy_notional_usdc=args.round_cap_usdc,
                sell_notional_usdc=0.0,
                fill_count=1,
                entry_count=1,
                exit_count=0,
                final_shares=shares,
            )
        )
        filled_rounds.add(round_slug)
    return results, fills


def _simulate_dynamic_exposure(
    *,
    replay: Any,
    split: str,
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    args: argparse.Namespace,
) -> tuple[list[RoundResult], list[SimFill]]:
    rows_by_round: dict[str, list[tuple[dict[str, Any], dict[str, float | str]]]] = defaultdict(list)
    for row, payload in zip(rows, payloads, strict=True):
        rows_by_round[replay._round_slug(row)].append((row, payload))

    results: list[RoundResult] = []
    fills: list[SimFill] = []
    for round_slug, pairs in sorted(rows_by_round.items()):
        pairs.sort(key=lambda item: int(item[0].get("feature_ts") or 0))
        position = SimPosition()
        realized_pnl = 0.0
        buy_notional = 0.0
        sell_notional = 0.0
        entry_count = 0
        exit_count = 0
        final_label = "NEUTRAL"
        final_side = ""
        last_row: dict[str, Any] | None = None
        for row, payload in pairs:
            last_row = row
            final_label = replay._settlement_label(row)
            feature_ts = int(row.get("feature_ts") or 0)
            market_payload = replay.market_v6_payload_from_token_payload(
                payload,
                token_side=replay._outcome_side(row),
            )
            selected_side, selected_p_side, p_up, p_down = _settlement_side(
                market_payload,
                args=args,
            )
            current_side = position.side if position.open else ""
            current_p_side = (
                _side_probability(market_payload, current_side, args=args)
                if current_side
                else None
            )

            if position.open and current_p_side is not None:
                seconds_to_expiry = replay._seconds_to_expiry(row)
                if seconds_to_expiry is not None and seconds_to_expiry <= 0.0:
                    continue
                if (
                    not args.allow_reduce_until_expiry
                    and seconds_to_expiry is not None
                    and seconds_to_expiry < args.no_new_entry_before_expiry_seconds
                ):
                    continue
                current_quote = _quote_for_side(row, current_side, args=args)
                opposite_p_side = _side_probability(
                    market_payload,
                    _opposite_side(current_side),
                    args=args,
                )
                if current_quote is not None and opposite_p_side is not None:
                    target_hold, exit_reason, edge_hold = _risk_exit_target(
                        position=position,
                        current_p_side=current_p_side,
                        opposite_p_side=opposite_p_side,
                        quote=current_quote,
                        args=args,
                    )
                    if target_hold + args.min_rebalance_usdc <= position.cost_basis_usdc:
                        pnl, sold_notional, sold_shares = _sell_to_cost_basis(
                            position,
                            target_cost_basis=target_hold,
                            sell_price=current_quote.sell_price,
                        )
                        realized_pnl += pnl
                        sell_notional += sold_notional
                        exit_count += 1
                        fills.append(
                            SimFill(
                                split=split,
                                strategy="dynamic_exposure",
                                round_slug=round_slug,
                                feature_ts=feature_ts,
                                action="SELL",
                                side=current_side,
                                price=current_quote.sell_price,
                                shares=sold_shares,
                                notional_usdc=sold_notional,
                                realized_pnl=pnl,
                                p_up=p_up,
                                p_down=p_down,
                                p_side=current_p_side,
                                edge_buy=math.nan,
                                edge_hold=edge_hold,
                                target_usdc=target_hold,
                                reason=exit_reason,
                            )
                        )

            if selected_side is None:
                continue
            timing_skip = _entry_timing_skip(replay, row, args)
            if timing_skip is not None:
                continue
            selected_quote = _quote_for_side(row, selected_side, args=args)
            if selected_quote is None:
                continue
            edge_buy = selected_p_side - selected_quote.buy_price
            target_buy = _target_from_edge(
                edge_buy,
                min_edge=args.entry_edge_min,
                full_edge=args.full_size_edge,
                cap_usdc=args.round_cap_usdc,
            )
            if selected_p_side < args.settlement_min_confidence:
                target_buy = 0.0
            if target_buy <= 0:
                continue
            if position.open and position.side != selected_side:
                current_quote = _quote_for_side(row, position.side, args=args)
                current_p = _side_probability(market_payload, position.side, args=args)
                current_edge_hold = (
                    current_p - current_quote.sell_price
                    if current_quote is not None and current_p is not None
                    else -1.0
                )
                if position.cost_basis_usdc > 0:
                    continue
                if edge_buy - current_edge_hold < args.flip_edge_gap:
                    continue
            if position.open and position.side == selected_side:
                current_cost_basis = position.cost_basis_usdc
            elif not position.open:
                position = SimPosition(side=selected_side)
                current_cost_basis = 0.0
            else:
                continue
            delta_usdc = min(args.round_cap_usdc, target_buy) - current_cost_basis
            if delta_usdc + 1e-9 < args.min_rebalance_usdc:
                continue
            shares = delta_usdc / selected_quote.buy_price
            position.shares += shares
            position.cost_basis_usdc += delta_usdc
            buy_notional += delta_usdc
            entry_count += 1
            final_side = position.side
            fills.append(
                SimFill(
                    split=split,
                    strategy="dynamic_exposure",
                    round_slug=round_slug,
                    feature_ts=feature_ts,
                    action="BUY",
                    side=selected_side,
                    price=selected_quote.buy_price,
                    shares=shares,
                    notional_usdc=delta_usdc,
                    realized_pnl=0.0,
                    p_up=p_up,
                    p_down=p_down,
                    p_side=selected_p_side,
                    edge_buy=edge_buy,
                    edge_hold=None,
                    target_usdc=target_buy,
                    reason="increase_to_buy_edge_target",
                )
            )

        if position.open and last_row is not None:
            final_side = position.side
            settlement = _settlement_pnl(
                side=position.side,
                true_label=final_label,
                shares=position.shares,
                avg_price=position.avg_price,
            )
            total_pnl = realized_pnl + settlement
            results.append(
                RoundResult(
                    split=split,
                    strategy="dynamic_exposure",
                    round_slug=round_slug,
                    side=final_side,
                    true_label=final_label,
                    pnl=total_pnl,
                    buy_notional_usdc=buy_notional,
                    sell_notional_usdc=sell_notional,
                    fill_count=entry_count + exit_count,
                    entry_count=entry_count,
                    exit_count=exit_count,
                    final_shares=position.shares,
                )
            )
        elif realized_pnl != 0.0:
            results.append(
                RoundResult(
                    split=split,
                    strategy="dynamic_exposure",
                    round_slug=round_slug,
                    side=final_side,
                    true_label=final_label,
                    pnl=realized_pnl,
                    buy_notional_usdc=buy_notional,
                    sell_notional_usdc=sell_notional,
                    fill_count=entry_count + exit_count,
                    entry_count=entry_count,
                    exit_count=exit_count,
                    final_shares=0.0,
                )
            )
    return results, fills


def _main_shadow_log(args: argparse.Namespace) -> int:
    log_path = Path(args.shadow_log_jsonl)
    events = _load_jsonl_events(log_path)
    signals = _shadow_settlement_signals(events, split=args.shadow_log_label)
    outcomes = _shadow_settlement_outcomes(events)
    gamma_outcomes: dict[str, str] = {}
    if args.resolve_missing_shadow_outcomes:
        missing_rounds = sorted({signal.round_slug for signal in signals} - set(outcomes))
        gamma_outcomes = _resolve_gamma_outcomes(
            missing_rounds,
            gamma_api_base=args.gamma_api_base,
            timeout_seconds=args.gamma_timeout_seconds,
        )
        outcomes.update(gamma_outcomes)
    actual_results, actual_fills = _shadow_actual_executor_results(
        events,
        split=args.shadow_log_label,
    )
    dynamic_results, dynamic_fills, unresolved_dynamic = _simulate_dynamic_shadow_log(
        signals,
        outcomes=outcomes,
        args=args,
    )

    report: dict[str, Any] = {
        "shadow_log_jsonl": str(log_path),
        "shadow_log_label": args.shadow_log_label,
        "controller_config": {
            "round_cap_usdc": args.round_cap_usdc,
            "entry_edge_min": args.entry_edge_min,
            "hold_edge_min": args.hold_edge_min,
            "exit_negative_edge_threshold": args.exit_negative_edge_threshold,
            "exit_confidence_floor": args.exit_confidence_floor,
            "exit_opposite_min_confidence": args.exit_opposite_min_confidence,
            "exit_opposite_hysteresis_bars": args.exit_opposite_hysteresis_bars,
            "exit_opposite_gap": args.exit_opposite_gap,
            "exit_price_stop_delta": args.exit_price_stop_delta,
            "exit_stop_loss_usdc": args.exit_stop_loss_usdc,
            "full_size_edge": args.full_size_edge,
            "flip_edge_gap": args.flip_edge_gap,
            "min_rebalance_usdc": args.min_rebalance_usdc,
            "settlement_threshold": args.settlement_threshold,
            "settlement_min_confidence": args.settlement_min_confidence,
            "probability_shrinkage": args.probability_shrinkage,
            "sell_slippage": args.sell_slippage,
            "allow_reduce_until_expiry": args.allow_reduce_until_expiry,
        },
        "shadow_log_counts": _shadow_event_counts(events),
        "settlement_signal_count": len(signals),
        "settlement_entry_signal_count": sum(1 for signal in signals if signal.can_enter),
        "settlement_hold_observation_count": sum(1 for signal in signals if not signal.can_enter),
        "known_settlement_outcome_count": len(outcomes),
        "log_settlement_outcome_count": len(outcomes) - len(gamma_outcomes),
        "gamma_resolved_outcome_count": len(gamma_outcomes),
        "gamma_resolved_outcomes": gamma_outcomes,
        "strategies": {
            "shadow_actual_executor": _summarize_results(actual_results, actual_fills),
            "dynamic_exposure_log_replay": _summarize_results(dynamic_results, dynamic_fills),
        },
        "unresolved_dynamic_open_rounds": unresolved_dynamic,
    }

    if args.output_json_path:
        path = Path(args.output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_shadow_report(report), encoding="utf-8")

    print(json.dumps(report["strategies"], indent=2, sort_keys=True))
    return 0


def _load_jsonl_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            event["_line_number"] = line_number
            events.append(event)
    return events


def _shadow_event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        counts[str(event.get("event") or "")] += 1
    return dict(sorted(counts.items()))


def _shadow_settlement_signals(events: list[dict[str, Any]], *, split: str) -> list[ShadowSignal]:
    signals: list[ShadowSignal] = []
    stop_bid_events = _shadow_stop_bid_events(events)
    for event in events:
        if event.get("event") != "entry_gate_evaluated":
            continue
        gate = event.get("gate_evaluation") or {}
        if gate.get("settlement_confidence") is None:
            continue
        signal = event.get("signal") or {}
        side = str(signal.get("outcome_side") or signal.get("v6_joint_side") or "").upper()
        if side not in {"UP", "DOWN"}:
            continue
        ask = _optional_float(event.get("ask"))
        bid = _optional_float(event.get("bid"))
        worst_price = _optional_float(event.get("worst_price"))
        seconds_to_expiry = _optional_float(event.get("seconds_to_expiry"))
        if ask is None or bid is None or worst_price is None or seconds_to_expiry is None:
            continue
        signal_ts_raw = signal.get("ts")
        signal_ts_ms = _epoch_millis(signal_ts_raw)
        event_ts_ms = _iso_to_epoch_millis(event.get("ts")) or signal_ts_ms
        p_up = _optional_float(signal.get("p_up"))
        p_down = _optional_float(signal.get("p_down"))
        p_neutral = _optional_float(signal.get("p_neutral")) or 0.0
        round_slug = str(signal.get("round_slug") or "")
        if not round_slug or signal_ts_ms <= 0 or p_up is None or p_down is None:
            continue
        signals.append(
            ShadowSignal(
                split=split,
                round_slug=round_slug,
                event_ts_ms=event_ts_ms,
                signal_ts_ms=signal_ts_ms,
                side=side,
                quote_side=side,
                can_enter=True,
                p_up=p_up,
                p_down=p_down,
                p_neutral=p_neutral,
                ask=_clamp_price(ask),
                bid=_clamp_price(bid),
                worst_price=_clamp_price(worst_price),
                seconds_to_expiry=seconds_to_expiry,
                settlement_edge=float(gate.get("settlement_edge") or event.get("fresh_edge_at_worst") or 0.0),
                signal_age_seconds=_optional_float(gate.get("signal_age_seconds")),
            )
        )
    for event in events:
        if event.get("event") != "settlement_confidence_decay_exit_evaluated":
            continue
        signal = event.get("signal") or {}
        position = event.get("position") or {}
        round_slug = str(signal.get("round_slug") or position.get("round_slug") or "")
        quote_side = str(position.get("side") or "").upper()
        side = str(signal.get("outcome_side") or signal.get("v6_joint_side") or "").upper()
        if not round_slug or quote_side not in {"UP", "DOWN"} or side not in {"UP", "DOWN"}:
            continue
        p_up = _optional_float(signal.get("p_up"))
        p_down = _optional_float(signal.get("p_down"))
        if p_up is None or p_down is None:
            continue
        event_ts_ms = _iso_to_epoch_millis(event.get("ts")) or _epoch_millis(signal.get("ts"))
        stop_bid = _nearest_stop_bid(stop_bid_events.get(round_slug, []), event_ts_ms)
        if stop_bid is None:
            continue
        signal_ts_ms = _epoch_millis(signal.get("ts"))
        signals.append(
            ShadowSignal(
                split=split,
                round_slug=round_slug,
                event_ts_ms=event_ts_ms,
                signal_ts_ms=signal_ts_ms,
                side=side,
                quote_side=quote_side,
                can_enter=False,
                p_up=p_up,
                p_down=p_down,
                p_neutral=_optional_float(signal.get("p_neutral")) or 0.0,
                ask=stop_bid,
                bid=stop_bid,
                worst_price=stop_bid,
                seconds_to_expiry=_optional_float(event.get("seconds_to_expiry")) or 0.0,
                settlement_edge=float(signal.get("edge") or 0.0),
                signal_age_seconds=None,
            )
        )
    for event in events:
        if event.get("event") != "settlement_stop_exit_evaluated":
            continue
        position = event.get("position") or {}
        round_slug = str(position.get("round_slug") or "")
        quote_side = str(position.get("side") or "").upper()
        if not round_slug or quote_side not in {"UP", "DOWN"}:
            continue
        bid = _optional_float(event.get("bid"))
        if bid is None:
            continue
        p_up = _optional_float(position.get("entry_p_up"))
        p_down = _optional_float(position.get("entry_p_down"))
        if p_up is None or p_down is None:
            continue
        event_ts_ms = _iso_to_epoch_millis(event.get("ts")) or _epoch_millis(position.get("entry_signal_ts"))
        signal_ts_ms = _epoch_millis(position.get("entry_signal_ts"))
        signals.append(
            ShadowSignal(
                split=split,
                round_slug=round_slug,
                event_ts_ms=event_ts_ms,
                signal_ts_ms=signal_ts_ms,
                side=quote_side,
                quote_side=quote_side,
                can_enter=False,
                p_up=p_up,
                p_down=p_down,
                p_neutral=_optional_float(position.get("entry_p_neutral")) or 0.0,
                ask=_clamp_price(bid),
                bid=_clamp_price(bid),
                worst_price=_clamp_price(bid),
                seconds_to_expiry=_optional_float(event.get("seconds_to_expiry")) or 0.0,
                settlement_edge=0.0,
                signal_age_seconds=None,
            )
        )
    signals.sort(key=lambda item: (item.round_slug, item.event_ts_ms, item.signal_ts_ms))
    return signals


def _shadow_stop_bid_events(events: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    by_round: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for event in events:
        if event.get("event") != "settlement_stop_exit_evaluated":
            continue
        position = event.get("position") or {}
        round_slug = str(position.get("round_slug") or "")
        event_ts_ms = _iso_to_epoch_millis(event.get("ts"))
        bid = _optional_float(event.get("bid"))
        if round_slug and event_ts_ms is not None and bid is not None:
            by_round[round_slug].append((event_ts_ms, _clamp_price(bid)))
    for entries in by_round.values():
        entries.sort(key=lambda item: item[0])
    return by_round


def _nearest_stop_bid(entries: list[tuple[int, float]], event_ts_ms: int) -> float | None:
    best: tuple[int, float] | None = None
    best_delta = 2_000
    for ts_ms, bid in entries:
        delta = abs(ts_ms - event_ts_ms)
        if delta <= best_delta:
            best = (ts_ms, bid)
            best_delta = delta
    return None if best is None else best[1]


def _shadow_settlement_outcomes(events: list[dict[str, Any]]) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for event in events:
        if event.get("event") != "paper_settlement_resolved":
            continue
        position = event.get("position") or {}
        round_slug = str(position.get("round_slug") or "")
        result = str(event.get("settlement_result") or "").upper()
        if round_slug and result in {"UP", "DOWN"}:
            outcomes[round_slug] = result
    return outcomes


def _resolve_gamma_outcomes(
    round_slugs: list[str],
    *,
    gamma_api_base: str,
    timeout_seconds: float,
) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for round_slug in round_slugs:
        result = _fetch_gamma_outcome(round_slug, gamma_api_base=gamma_api_base, timeout_seconds=timeout_seconds)
        if result in {"UP", "DOWN"}:
            outcomes[round_slug] = result
    return outcomes


def _fetch_gamma_outcome(
    round_slug: str,
    *,
    gamma_api_base: str,
    timeout_seconds: float,
) -> str | None:
    base = str(gamma_api_base or "https://gamma-api.polymarket.com").rstrip("/")
    market: dict[str, Any] | None = None
    param_sets = (
        {"slug": round_slug, "closed": "true", "limit": "1"},
        {"slug": round_slug, "active": "true", "closed": "false", "limit": "1"},
        {"slug": round_slug, "limit": "1"},
    )
    for params in param_sets:
        url = f"{base}/markets?{parse.urlencode(params)}"
        req = request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "BiGan-v6-shadow-log-replay/1.0",
            },
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urlerror.URLError, json.JSONDecodeError):
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            market = payload[0]
            break
        if isinstance(payload, dict):
            markets = payload.get("markets") or payload.get("data") or []
            if isinstance(markets, list) and markets and isinstance(markets[0], dict):
                market = markets[0]
                break
    if market is None:
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
    parsed_prices = [_optional_float(price) for price in prices]
    if any(price is None for price in parsed_prices):
        return None
    best_index = max(range(len(parsed_prices)), key=lambda index: float(parsed_prices[index] or 0.0))
    best_price = float(parsed_prices[best_index] or 0.0)
    if best_price < 0.95:
        return None
    outcome = str(outcomes[best_index]).upper()
    if outcome == "UP":
        return "UP"
    if outcome == "DOWN":
        return "DOWN"
    return None


def _shadow_actual_executor_results(
    events: list[dict[str, Any]],
    *,
    split: str,
) -> tuple[list[RoundResult], list[SimFill]]:
    entries: dict[str, dict[str, Any]] = {}
    results: list[RoundResult] = []
    fills: list[SimFill] = []
    completed_rounds: set[str] = set()

    for event in events:
        name = event.get("event")
        if name == "paper_entry_filled":
            signal = event.get("signal") or {}
            position = event.get("position") or {}
            round_slug = str(position.get("round_slug") or signal.get("round_slug") or "")
            if not round_slug:
                continue
            side = str(position.get("side") or signal.get("outcome_side") or "").upper()
            price = float(position.get("fill_price") or position.get("entry_price") or 0.0)
            shares = float(position.get("size") or 0.0)
            size_usdc = float(event.get("size_usdc") or price * shares)
            p_up = float(signal.get("p_up") or position.get("entry_p_up") or 0.0)
            p_down = float(signal.get("p_down") or position.get("entry_p_down") or 0.0)
            p_side = p_up if side == "UP" else p_down
            entries[round_slug] = {
                "side": side,
                "price": price,
                "shares": shares,
                "size_usdc": size_usdc,
                "p_up": p_up,
                "p_down": p_down,
            }
            fills.append(
                SimFill(
                    split=split,
                    strategy="shadow_actual_executor",
                    round_slug=round_slug,
                    feature_ts=_epoch_millis(signal.get("ts")),
                    action="BUY",
                    side=side,
                    price=price,
                    shares=shares,
                    notional_usdc=size_usdc,
                    realized_pnl=0.0,
                    p_up=p_up,
                    p_down=p_down,
                    p_side=p_side,
                    edge_buy=float((event.get("gate_evaluation") or {}).get("settlement_edge") or math.nan),
                    edge_hold=None,
                    target_usdc=size_usdc,
                    reason="shadow_paper_entry_filled",
                )
            )
        elif name == "paper_exit_filled":
            position = event.get("position") or {}
            round_slug = str(position.get("round_slug") or "")
            entry = entries.get(round_slug)
            if not entry or round_slug in completed_rounds:
                continue
            side = str(position.get("side") or entry["side"]).upper()
            shares = float(position.get("size") or entry["shares"])
            exit_price = float(event.get("exit_price") or 0.0)
            pnl = float(event.get("realized_pnl") or 0.0)
            fills.append(
                SimFill(
                    split=split,
                    strategy="shadow_actual_executor",
                    round_slug=round_slug,
                    feature_ts=_iso_to_epoch_millis(event.get("ts")) or 0,
                    action="SELL",
                    side=side,
                    price=exit_price,
                    shares=shares,
                    notional_usdc=exit_price * shares,
                    realized_pnl=pnl,
                    p_up=float(entry["p_up"]),
                    p_down=float(entry["p_down"]),
                    p_side=float(entry["p_up"] if side == "UP" else entry["p_down"]),
                    edge_buy=math.nan,
                    edge_hold=None,
                    target_usdc=0.0,
                    reason=str(event.get("reason") or "shadow_paper_exit_filled"),
                )
            )
            results.append(
                RoundResult(
                    split=split,
                    strategy="shadow_actual_executor",
                    round_slug=round_slug,
                    side=side,
                    true_label="EXITED",
                    pnl=pnl,
                    buy_notional_usdc=float(entry["size_usdc"]),
                    sell_notional_usdc=exit_price * shares,
                    fill_count=2,
                    entry_count=1,
                    exit_count=1,
                    final_shares=0.0,
                )
            )
            completed_rounds.add(round_slug)
        elif name == "paper_settlement_resolved":
            position = event.get("position") or {}
            round_slug = str(position.get("round_slug") or "")
            entry = entries.get(round_slug)
            if not entry or round_slug in completed_rounds:
                continue
            side = str(position.get("side") or entry["side"]).upper()
            shares = float(position.get("size") or entry["shares"])
            pnl = float(event.get("realized_pnl") or 0.0)
            results.append(
                RoundResult(
                    split=split,
                    strategy="shadow_actual_executor",
                    round_slug=round_slug,
                    side=side,
                    true_label=str(event.get("settlement_result") or "UNKNOWN").upper(),
                    pnl=pnl,
                    buy_notional_usdc=float(entry["size_usdc"]),
                    sell_notional_usdc=0.0,
                    fill_count=1,
                    entry_count=1,
                    exit_count=0,
                    final_shares=0.0,
                )
            )
            completed_rounds.add(round_slug)
    return results, fills


def _simulate_dynamic_shadow_log(
    signals: list[ShadowSignal],
    *,
    outcomes: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[RoundResult], list[SimFill], list[dict[str, Any]]]:
    by_round: dict[str, list[ShadowSignal]] = defaultdict(list)
    for signal in signals:
        by_round[signal.round_slug].append(signal)

    results: list[RoundResult] = []
    fills: list[SimFill] = []
    unresolved: list[dict[str, Any]] = []
    for round_slug, round_signals in sorted(by_round.items()):
        round_signals.sort(key=lambda item: (item.event_ts_ms, item.signal_ts_ms))
        position = SimPosition()
        realized_pnl = 0.0
        buy_notional = 0.0
        sell_notional = 0.0
        entry_count = 0
        exit_count = 0
        final_side = ""

        for signal in round_signals:
            selected_side, selected_p_side, p_up, p_down = _settlement_side(
                {"p_up": signal.p_up, "p_down": signal.p_down},
                args=args,
            )

            if position.open:
                current_side = position.side
                current_p_side = _shadow_side_probability(signal, current_side, args=args)
                opposite_p_side = _shadow_side_probability(
                    signal,
                    _opposite_side(current_side),
                    args=args,
                )
                current_quote = _shadow_quote_for_side(signal, current_side, args=args)
                if current_p_side is not None and opposite_p_side is not None and current_quote is not None:
                    target_hold, exit_reason, edge_hold = _risk_exit_target(
                        position=position,
                        current_p_side=current_p_side,
                        opposite_p_side=opposite_p_side,
                        quote=current_quote,
                        args=args,
                    )
                    if target_hold + args.min_rebalance_usdc <= position.cost_basis_usdc:
                        pnl, sold_notional, sold_shares = _sell_to_cost_basis(
                            position,
                            target_cost_basis=target_hold,
                            sell_price=current_quote.sell_price,
                        )
                        realized_pnl += pnl
                        sell_notional += sold_notional
                        exit_count += 1
                        fills.append(
                            SimFill(
                                split=signal.split,
                                strategy="dynamic_exposure_log_replay",
                                round_slug=round_slug,
                                feature_ts=signal.event_ts_ms,
                                action="SELL",
                                side=current_side,
                                price=current_quote.sell_price,
                                shares=sold_shares,
                                notional_usdc=sold_notional,
                                realized_pnl=pnl,
                                p_up=p_up,
                                p_down=p_down,
                                p_side=current_p_side,
                                edge_buy=math.nan,
                                edge_hold=edge_hold,
                                target_usdc=target_hold,
                                reason=exit_reason,
                            )
                        )

            if not signal.can_enter:
                continue
            if selected_side is None or selected_side != signal.side:
                continue
            if signal.seconds_to_expiry < args.no_new_entry_before_expiry_seconds:
                continue
            selected_quote = _shadow_quote_for_side(signal, selected_side, args=args)
            if selected_quote is None:
                continue
            edge_buy = selected_p_side - selected_quote.buy_price
            target_buy = _target_from_edge(
                edge_buy,
                min_edge=args.entry_edge_min,
                full_edge=args.full_size_edge,
                cap_usdc=args.round_cap_usdc,
            )
            if selected_p_side < args.settlement_min_confidence:
                target_buy = 0.0
            if target_buy <= 0.0:
                continue
            if position.open and position.side != selected_side:
                current_quote = _shadow_quote_for_side(signal, position.side, args=args)
                current_p = _shadow_side_probability(signal, position.side, args=args)
                current_edge_hold = (
                    current_p - current_quote.sell_price
                    if current_quote is not None and current_p is not None
                    else -1.0
                )
                if position.cost_basis_usdc > 0.0:
                    continue
                if edge_buy - current_edge_hold < args.flip_edge_gap:
                    continue
            if position.open and position.side == selected_side:
                current_cost_basis = position.cost_basis_usdc
            elif not position.open:
                position = SimPosition(side=selected_side)
                current_cost_basis = 0.0
            else:
                continue
            delta_usdc = min(args.round_cap_usdc, target_buy) - current_cost_basis
            if delta_usdc + 1e-9 < args.min_rebalance_usdc:
                continue
            shares = delta_usdc / selected_quote.buy_price
            position.shares += shares
            position.cost_basis_usdc += delta_usdc
            buy_notional += delta_usdc
            entry_count += 1
            final_side = position.side
            fills.append(
                SimFill(
                    split=signal.split,
                    strategy="dynamic_exposure_log_replay",
                    round_slug=round_slug,
                    feature_ts=signal.signal_ts_ms,
                    action="BUY",
                    side=selected_side,
                    price=selected_quote.buy_price,
                    shares=shares,
                    notional_usdc=delta_usdc,
                    realized_pnl=0.0,
                    p_up=p_up,
                    p_down=p_down,
                    p_side=selected_p_side,
                    edge_buy=edge_buy,
                    edge_hold=None,
                    target_usdc=target_buy,
                    reason="increase_to_shadow_buy_edge_target",
                )
            )

        true_label = outcomes.get(round_slug)
        if position.open:
            final_side = position.side
            if true_label is None:
                unresolved.append(
                    {
                        "round_slug": round_slug,
                        "side": position.side,
                        "shares": position.shares,
                        "avg_price": position.avg_price,
                        "realized_pnl_before_settlement": realized_pnl,
                    }
                )
                continue
            settlement = _settlement_pnl(
                side=position.side,
                true_label=true_label,
                shares=position.shares,
                avg_price=position.avg_price,
            )
            total_pnl = realized_pnl + settlement
            results.append(
                RoundResult(
                    split=args.shadow_log_label,
                    strategy="dynamic_exposure_log_replay",
                    round_slug=round_slug,
                    side=final_side,
                    true_label=true_label,
                    pnl=total_pnl,
                    buy_notional_usdc=buy_notional,
                    sell_notional_usdc=sell_notional,
                    fill_count=entry_count + exit_count,
                    entry_count=entry_count,
                    exit_count=exit_count,
                    final_shares=position.shares,
                )
            )
        elif realized_pnl != 0.0:
            results.append(
                RoundResult(
                    split=args.shadow_log_label,
                    strategy="dynamic_exposure_log_replay",
                    round_slug=round_slug,
                    side=final_side,
                    true_label=true_label or "EXITED",
                    pnl=realized_pnl,
                    buy_notional_usdc=buy_notional,
                    sell_notional_usdc=sell_notional,
                    fill_count=entry_count + exit_count,
                    entry_count=entry_count,
                    exit_count=exit_count,
                    final_shares=0.0,
                )
            )
    return results, fills, unresolved


def _shadow_quote_for_side(
    signal: ShadowSignal,
    side: str,
    *,
    args: argparse.Namespace,
) -> Quote | None:
    if side == signal.quote_side:
        ask = signal.ask
        bid = signal.bid
        buy_price = signal.worst_price if signal.can_enter and side == signal.side else signal.ask
    elif side in {"UP", "DOWN"}:
        ask = _clamp_price(1.0 - signal.bid)
        bid = _clamp_price(1.0 - signal.ask)
        buy_price = _clamp_price(ask + args.buy_slippage)
    else:
        return None
    sell_price = _clamp_price(bid - args.sell_slippage)
    return Quote(
        ask=ask,
        bid=min(ask, bid),
        buy_price=_clamp_price(buy_price),
        sell_price=min(_clamp_price(buy_price), sell_price),
    )


def _shadow_side_probability(
    signal: ShadowSignal,
    side: str,
    *,
    args: argparse.Namespace,
) -> float | None:
    if side == "UP":
        return _effective_probability(signal.p_up, shrinkage=args.probability_shrinkage)
    if side == "DOWN":
        return _effective_probability(signal.p_down, shrinkage=args.probability_shrinkage)
    return None


def _entry_timing_skip(replay: Any, row: dict[str, Any], args: argparse.Namespace) -> str | None:
    seconds_to_expiry = replay._seconds_to_expiry(row)
    seconds_since_start = replay._seconds_since_round_start(row)
    if seconds_to_expiry is None or seconds_since_start is None:
        return "missing_round_timing"
    if seconds_since_start < 0.0:
        return "before_round_start"
    if seconds_to_expiry < args.no_new_entry_before_expiry_seconds:
        return "no_new_entry_window"
    if seconds_to_expiry < args.min_seconds_to_expiry:
        return "below_min_seconds_to_expiry"
    if seconds_to_expiry > args.max_seconds_to_expiry:
        return "above_max_seconds_to_expiry"
    return None


def _settlement_side(
    payload: dict[str, float | str],
    *,
    args: argparse.Namespace,
) -> tuple[str | None, float, float, float]:
    p_up = _effective_probability(float(payload["p_up"]), shrinkage=args.probability_shrinkage)
    p_down = _effective_probability(float(payload["p_down"]), shrinkage=args.probability_shrinkage)
    if p_up >= p_down and p_up >= args.settlement_threshold:
        return "UP", p_up, p_up, p_down
    if p_down > p_up and p_down >= args.settlement_threshold:
        return "DOWN", p_down, p_up, p_down
    return None, 0.0, p_up, p_down


def _side_probability(
    payload: dict[str, float | str],
    side: str,
    *,
    args: argparse.Namespace,
) -> float | None:
    if side == "UP":
        return _effective_probability(float(payload["p_up"]), shrinkage=args.probability_shrinkage)
    if side == "DOWN":
        return _effective_probability(float(payload["p_down"]), shrinkage=args.probability_shrinkage)
    return None


def _opposite_side(side: str) -> str:
    if side == "UP":
        return "DOWN"
    if side == "DOWN":
        return "UP"
    return ""


def _risk_exit_target(
    *,
    position: SimPosition,
    current_p_side: float,
    opposite_p_side: float,
    quote: Quote,
    args: argparse.Namespace,
) -> tuple[float, str, float]:
    edge_hold = current_p_side - quote.sell_price
    unrealized_pnl = (quote.sell_price - position.avg_price) * position.shares
    if quote.sell_price <= position.avg_price - args.exit_price_stop_delta:
        return 0.0, "exit_price_stop", edge_hold
    if unrealized_pnl <= -args.exit_stop_loss_usdc:
        return 0.0, "exit_stop_loss", edge_hold
    opposite_side = _opposite_side(position.side)
    opposite_confirmed = (
        opposite_side
        and opposite_p_side >= args.exit_opposite_min_confidence
        and opposite_p_side >= current_p_side + args.exit_opposite_gap
    )
    if opposite_confirmed:
        if position.exit_opposite_candidate_side == opposite_side:
            position.exit_opposite_candidate_count += 1
        else:
            position.exit_opposite_candidate_side = opposite_side
            position.exit_opposite_candidate_count = 1
    else:
        position.exit_opposite_candidate_side = ""
        position.exit_opposite_candidate_count = 0
    if position.exit_opposite_candidate_count >= args.exit_opposite_hysteresis_bars:
        return 0.0, "exit_opposite_reversal_confirmed", edge_hold
    if current_p_side < args.exit_confidence_floor:
        return 0.0, "exit_confidence_floor", edge_hold
    if edge_hold <= args.exit_negative_edge_threshold:
        return 0.0, "exit_negative_hold_edge", edge_hold
    return position.cost_basis_usdc, "hold_risk_ok", edge_hold


def _effective_probability(raw: float, *, shrinkage: float) -> float:
    shrinkage = max(0.0, min(1.0, float(shrinkage)))
    return max(0.0, min(1.0, 0.5 + (float(raw) - 0.5) * (1.0 - shrinkage)))


def _quote_for_side(row: dict[str, Any], side: str, *, args: argparse.Namespace) -> Quote | None:
    up_ask_raw = row.get("entry_ask_price")
    if up_ask_raw is None:
        return None
    up_ask = _clamp_price(float(up_ask_raw))
    mid_raw = row.get("mid_price")
    if mid_raw is None:
        up_bid = up_ask - args.fallback_spread
    else:
        up_bid = 2.0 * float(mid_raw) - up_ask
    up_bid = min(up_ask, _clamp_price(up_bid))
    if side == "UP":
        ask = up_ask
        bid = up_bid
    elif side == "DOWN":
        ask = _clamp_price(1.0 - up_bid)
        bid = _clamp_price(1.0 - up_ask)
        bid = min(ask, bid)
    else:
        return None
    return Quote(
        ask=ask,
        bid=bid,
        buy_price=_clamp_price(ask + args.buy_slippage),
        sell_price=_clamp_price(bid - args.sell_slippage),
    )


def _target_from_edge(
    edge: float,
    *,
    min_edge: float,
    full_edge: float,
    cap_usdc: float,
) -> float:
    if edge <= min_edge:
        return 0.0
    if full_edge <= min_edge:
        return cap_usdc
    if edge >= full_edge:
        return cap_usdc
    return cap_usdc * (edge - min_edge) / (full_edge - min_edge)


def _sell_to_cost_basis(
    position: SimPosition,
    *,
    target_cost_basis: float,
    sell_price: float,
) -> tuple[float, float, float]:
    target_cost_basis = max(0.0, min(position.cost_basis_usdc, target_cost_basis))
    avg_price = position.avg_price
    cost_basis_to_sell = position.cost_basis_usdc - target_cost_basis
    if cost_basis_to_sell <= 0.0 or avg_price <= 0.0:
        return 0.0, 0.0, 0.0
    shares_to_sell = min(position.shares, cost_basis_to_sell / avg_price)
    realized_pnl = (sell_price - avg_price) * shares_to_sell
    notional = shares_to_sell * sell_price
    position.shares -= shares_to_sell
    position.cost_basis_usdc -= shares_to_sell * avg_price
    if position.shares <= 1e-12 or position.cost_basis_usdc <= 1e-12:
        position.side = ""
        position.shares = 0.0
        position.cost_basis_usdc = 0.0
    return realized_pnl, notional, shares_to_sell


def _settlement_pnl(*, side: str, true_label: str, shares: float, avg_price: float) -> float:
    exit_price = 1.0 if side == true_label else 0.0
    return (exit_price - avg_price) * shares


def _summarize_results(results: list[RoundResult], fills: list[SimFill]) -> dict[str, Any]:
    pnl_by_result = [float(result.pnl) for result in results]
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnl_by_result:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    wins = sum(1 for result in results if result.pnl > 0)
    buy_notional = sum(result.buy_notional_usdc for result in results)
    sell_notional = sum(result.sell_notional_usdc for result in results)
    return {
        "rounds_traded": len(results),
        "fill_count": len(fills),
        "entry_count": sum(1 for fill in fills if fill.action == "BUY"),
        "exit_count": sum(1 for fill in fills if fill.action == "SELL"),
        "pnl": cumulative,
        "avg_pnl_per_traded_round": cumulative / len(results) if results else None,
        "hit_rate": wins / len(results) if results else None,
        "max_drawdown": max_drawdown,
        "buy_notional_usdc": buy_notional,
        "sell_notional_usdc": sell_notional,
        "turnover_usdc": buy_notional + sell_notional,
        "avg_buy_notional_per_round": buy_notional / len(results) if results else None,
        "sample_rounds": [asdict(result) for result in results[:12]],
        "sample_fills": [asdict(fill) for fill in fills[:16]],
    }


def _markdown_report(report: dict[str, Any], *, args: argparse.Namespace) -> str:
    lines = [
        "# v6 Settlement Dynamic Exposure Replay",
        "",
        f"Model: `{report['model_json_path']}`",
        f"Dataset: `{report['dataset_dir']}`",
        f"Splits: `{','.join(report['splits'])}`",
        "",
        "## Controller",
        "",
        "- Entry sizing maps `p_side - buy_price` to target settlement exposure.",
        "- Holding exposure exits on negative hold edge, confidence loss, opposite reversal, or stop risk.",
        "- This is offline diagnostic evidence only; it does not prove live fill quality.",
        "",
        "| Parameter | Value |",
        "|---|---:|",
    ]
    for key, value in report["controller_config"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Combined Results",
            "",
            "| Strategy | Rounds | Fills | Entries | Exits | PnL | Hit rate | Max DD | Buy notional | Turnover |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in ("fixed_one_shot", "dynamic_exposure"):
        summary = report["strategies"][strategy]
        lines.append(_summary_row(strategy, summary))
    lines.extend(["", "## Split Results", ""])
    for split, payload in report["splits_detail"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                "| Strategy | Rounds | Fills | Entries | Exits | PnL | Hit rate | Max DD | Buy notional | Turnover |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                _summary_row("fixed_one_shot", payload["fixed_one_shot"]),
                _summary_row("dynamic_exposure", payload["dynamic_exposure"]),
                "",
            ]
        )
    dynamic = report["strategies"]["dynamic_exposure"]
    fixed = report["strategies"]["fixed_one_shot"]
    lines.extend(
        [
            "## Readout",
            "",
            f"- Dynamic minus fixed PnL: `{dynamic['pnl'] - fixed['pnl']:.6f}`.",
            f"- Dynamic turnover / fixed turnover: `{_safe_ratio(dynamic['turnover_usdc'], fixed['turnover_usdc'])}`.",
            "- If dynamic improves PnL without exploding turnover, the next step is paper replay on JSONL/orderbook logs.",
            "- If dynamic mainly reduces notional and loses PnL, the target curve is too conservative or the model edge is still not price-calibrated.",
            "",
            "## Sample Dynamic Fills",
            "",
            "| Split | Round | Time | Action | Side | Price | Notional | p_side | edge_buy | edge_hold | Target | Reason |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for fill in dynamic["sample_fills"][:24]:
        lines.append(
            f"| {fill['split']} | {fill['round_slug']} | {fill['feature_ts']} | "
            f"{fill['action']} | {fill['side']} | {fill['price']:.4f} | "
            f"{fill['notional_usdc']:.4f} | {fill['p_side']:.4f} | "
            f"{_fmt_float(fill['edge_buy'])} | {_fmt_float(fill['edge_hold'])} | "
            f"{fill['target_usdc']:.4f} | {fill['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_shadow_report(report: dict[str, Any]) -> str:
    actual = report["strategies"]["shadow_actual_executor"]
    dynamic = report["strategies"]["dynamic_exposure_log_replay"]
    lines = [
        "# v6 Settlement Dynamic Exposure Replay From Shadow Log",
        "",
        f"Shadow log: `{report['shadow_log_jsonl']}`",
        "",
        "## Input",
        "",
        f"- Settlement observations used: `{report['settlement_signal_count']}`.",
        f"- Executable entry-gate observations: `{report['settlement_entry_signal_count']}`.",
        f"- Hold/rebalance observations from confidence and stop checks: `{report['settlement_hold_observation_count']}`.",
        f"- Settlement outcomes from log: `{report['log_settlement_outcome_count']}`.",
        f"- Settlement outcomes resolved from Gamma: `{report['gamma_resolved_outcome_count']}`.",
        f"- Known settlement outcomes total: `{report['known_settlement_outcome_count']}`.",
        "- Settlement `entry_gate_evaluated` events and open-position stop checks with executable quote fields are replayed.",
        "- Signals in `no_new_entry_window` without bid/ask are not simulated as executable trades.",
        "",
        "## Controller",
        "",
        "| Parameter | Value |",
        "|---|---:|",
    ]
    for key, value in report["controller_config"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Strategy | Rounds | Fills | Entries | Exits | PnL | Hit rate | Max DD | Buy notional | Turnover |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _summary_row("shadow_actual_executor", actual),
            _summary_row("dynamic_exposure_log_replay", dynamic),
            "",
            "## Readout",
            "",
            f"- Dynamic minus actual executor PnL: `{dynamic['pnl'] - actual['pnl']:.6f}`.",
            f"- Dynamic turnover / actual turnover: `{_safe_ratio(dynamic['turnover_usdc'], actual['turnover_usdc'])}`.",
            f"- Dynamic unresolved open rounds due missing settlement outcomes in log: `{len(report['unresolved_dynamic_open_rounds'])}`.",
            "",
            "## Event Counts",
            "",
            "| Event | Count |",
            "|---|---:|",
        ]
    )
    interesting = {
        "entry_gate_evaluated",
        "paper_entry_filled",
        "paper_exit_filled",
        "paper_settlement_resolved",
        "phase4_summary",
        "settlement_stop_exit_filled",
    }
    for event_name, count in report["shadow_log_counts"].items():
        if event_name in interesting:
            lines.append(f"| `{event_name}` | {count} |")
    lines.extend(
        [
            "",
            "## Sample Dynamic Fills",
            "",
            "| Round | Time | Action | Side | Price | Notional | p_side | edge_buy | edge_hold | Target | Reason |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for fill in dynamic["sample_fills"][:32]:
        lines.append(
            f"| {fill['round_slug']} | {fill['feature_ts']} | {fill['action']} | "
            f"{fill['side']} | {fill['price']:.4f} | {fill['notional_usdc']:.4f} | "
            f"{fill['p_side']:.4f} | {_fmt_float(fill['edge_buy'])} | "
            f"{_fmt_float(fill['edge_hold'])} | {fill['target_usdc']:.4f} | "
            f"{fill['reason']} |"
        )
    if report["unresolved_dynamic_open_rounds"]:
        lines.extend(
            [
                "",
                "## Unresolved Dynamic Open Rounds",
                "",
                "| Round | Side | Shares | Avg price | Realized before settlement |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for item in report["unresolved_dynamic_open_rounds"][:20]:
            lines.append(
                f"| {item['round_slug']} | {item['side']} | {item['shares']:.6f} | "
                f"{item['avg_price']:.4f} | {item['realized_pnl_before_settlement']:.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _summary_row(strategy: str, summary: dict[str, Any]) -> str:
    return (
        f"| `{strategy}` | {summary['rounds_traded']} | {summary['fill_count']} | "
        f"{summary['entry_count']} | {summary['exit_count']} | {summary['pnl']:.6f} | "
        f"{_fmt_optional(summary['hit_rate'])} | {summary['max_drawdown']:.6f} | "
        f"{summary['buy_notional_usdc']:.4f} | {summary['turnover_usdc']:.4f} |"
    )


def _safe_ratio(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.4f}"


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{value:.4f}"


def _clamp_price(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _epoch_millis(value: Any) -> int:
    parsed = _optional_float(value)
    if parsed is None:
        return 0
    if parsed > 10_000_000_000:
        return int(parsed)
    return int(parsed * 1000)


def _iso_to_epoch_millis(value: Any) -> int | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp() * 1000)


def _load_replay_module() -> Any:
    spec = importlib.util.spec_from_file_location("replay_v6_btc15m_execution_restricted", REPLAY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import replay script: {REPLAY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
