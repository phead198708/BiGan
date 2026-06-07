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
            "exit_hold_edge": args.exit_hold_edge,
            "exit_hysteresis_bars": args.exit_hysteresis_bars,
            "reversal_min_confidence": args.reversal_min_confidence,
            "reversal_min_edge": args.reversal_min_edge,
            "reversal_hysteresis_bars": args.reversal_hysteresis_bars,
            "min_rebalance_usdc": args.min_rebalance_usdc,
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
    parser.add_argument("--exit-hold-edge", type=float, default=-0.02)
    parser.add_argument("--exit-hysteresis-bars", type=int, default=2)
    parser.add_argument("--reversal-min-confidence", type=float, default=0.75)
    parser.add_argument("--reversal-min-edge", type=float, default=0.04)
    parser.add_argument("--reversal-hysteresis-bars", type=int, default=2)
    parser.add_argument("--min-rebalance-usdc", type=float, default=0.05)
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
    token_probability = _float(payload.get("token_probability"))
    if token_probability is None:
        token_probability = p_up if side == "UP" else p_down
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
                "outcome": outcome,
                "open": position.open,
                "exited": position.exited,
                "remaining_shares": position.shares,
                "remaining_cost_basis_usdc": position.cost_basis_usdc,
                "avg_price": position.avg_price,
                "realized_pnl": position.realized_pnl,
                "settlement_pnl": settlement_pnl,
                "total_pnl_if_known": total_pnl if outcome in {"UP", "DOWN"} else None,
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
    p_side = side_signal.token_probability if side_signal is not None else None
    p_opposite = opposite_signal.token_probability if opposite_signal is not None else None
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

    prior_cost = position.cost_basis_usdc
    action = "HOLD"
    reason = "ev_hold"
    target_cost = prior_cost
    if position.reversal_count >= args.reversal_hysteresis_bars:
        action = "EXIT"
        reason = "confirmed_opposite_ev_reversal"
        target_cost = 0.0
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
    elif add_edge is not None and add_edge >= args.add_edge_min:
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
    )


def _apply_decision(position: SimPosition, decision: Decision) -> None:
    if decision.action in {"EXIT", "REDUCE"}:
        position.shares = max(0.0, position.shares + decision.shares_delta)
        position.cost_basis_usdc = max(0.0, decision.target_cost_basis_usdc)
        position.realized_pnl += decision.realized_pnl_delta
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
