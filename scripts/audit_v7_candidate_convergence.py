#!/usr/bin/env python3
"""Audit whether v7 candidate entries converge after the signal.

v7 entry is supposed to exploit a mismatch between model probability and the
executable market price.  This audit treats a signal as a candidate when its
model probability beats the executable ask by the configured edge threshold,
then checks whether the market price later moves toward that model probability.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
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
class Signal:
    line_number: int
    event_id: str
    round_slug: str
    canonical_symbol: str
    side: str
    feature_ts_ms: int
    created_at_ms: int
    bridged_at_ms: int
    round_end_ts_ms: int
    model_probability: float
    polymarket_price: float | None
    mispricing_edge: float | None


@dataclass(frozen=True, slots=True)
class Candidate:
    signal: Signal
    entry_ts_ms: int
    entry_bid: float
    entry_ask: float
    entry_mid: float
    seconds_to_expiry: float
    executable_edge: float


def main() -> int:
    args = _parse_args()
    events = _load_jsonl(Path(args.executor_jsonl)) if args.executor_jsonl else []
    config = _config_from_events(events)
    min_line_number = args.min_line_number
    if min_line_number is None:
        min_line_number = _signal_start_line(events)
    run_min_created_at_ms, run_max_created_at_ms = _run_time_bounds(events)
    if args.min_signal_created_at_ms is not None:
        run_min_created_at_ms = args.min_signal_created_at_ms
    if args.max_signal_created_at_ms is not None:
        run_max_created_at_ms = args.max_signal_created_at_ms
    entry_edge_min = _resolve_float(
        args.entry_edge_min,
        _nested(config, "v7_settlement_position_policy", "add_edge_min"),
        config.get("settlement_edge_threshold"),
        0.08,
    )
    min_seconds_to_expiry = _resolve_float(
        args.min_seconds_to_expiry,
        config.get("no_new_entry_before_expiry_seconds"),
        config.get("min_seconds_to_expiry"),
        300.0,
    )
    max_seconds_to_expiry = _resolve_float(
        args.max_seconds_to_expiry,
        config.get("max_seconds_to_expiry"),
        900.0,
    )
    min_entry_price = _resolve_float(args.min_entry_price, config.get("min_entry_price"), 0.01)

    signals = _merge_signals(
        _load_signals(Path(args.signal_jsonl), min_line_number=min_line_number),
        _signals_from_executor_events(events),
    )
    signals = _filter_signals_by_created_at(
        signals,
        min_created_at_ms=run_min_created_at_ms,
        max_created_at_ms=run_max_created_at_ms,
    )
    symbols = {signal.canonical_symbol for signal in signals}
    quote_bounds = _quote_bounds(signals, args=args)
    quotes = _load_quotes(
        Path(args.raw_jsonl),
        allowed_symbols=symbols,
        min_ts_ms=quote_bounds["min_ts_ms"],
        max_ts_ms=quote_bounds["max_ts_ms"],
    )
    candidates, skipped = _candidate_entries(
        signals=signals,
        quotes=quotes,
        entry_time_source=args.entry_time_source,
        entry_edge_min=entry_edge_min,
        min_seconds_to_expiry=min_seconds_to_expiry,
        max_seconds_to_expiry=max_seconds_to_expiry,
        min_entry_price=min_entry_price,
    )
    audits = [
        _audit_candidate(
            candidate,
            quotes=quotes.get(candidate.signal.canonical_symbol, []),
            signals_by_symbol=signals,
            horizons_seconds=args.horizons_seconds,
            min_toward_move=args.min_toward_move,
            residual_convergence_delta=args.residual_convergence_delta,
            take_profit_delta=args.take_profit_delta,
            model_decay_tolerance=args.model_decay_tolerance,
        )
        for candidate in candidates
    ]
    report = {
        "inputs": {
            "executor_jsonl": args.executor_jsonl,
            "signal_jsonl": args.signal_jsonl,
            "raw_jsonl": args.raw_jsonl,
        },
        "config": {
            "min_line_number": min_line_number,
            "min_signal_created_at_ms": run_min_created_at_ms,
            "max_signal_created_at_ms": run_max_created_at_ms,
            "entry_time_source": args.entry_time_source,
            "entry_edge_min": entry_edge_min,
            "min_seconds_to_expiry": min_seconds_to_expiry,
            "max_seconds_to_expiry": max_seconds_to_expiry,
            "min_entry_price": min_entry_price,
            "min_toward_move": args.min_toward_move,
            "residual_convergence_delta": args.residual_convergence_delta,
            "take_profit_delta": args.take_profit_delta,
            "model_decay_tolerance": args.model_decay_tolerance,
            "horizons_seconds": args.horizons_seconds,
        },
        "counts": {
            "signals": len(signals),
            "quote_symbols": len(quotes),
            "candidates": len(candidates),
            "candidate_rounds": len({item["round_slug"] for item in audits}),
            "skipped": dict(sorted(skipped.items())),
        },
        "summary": _summary(audits),
        "round_summary": _round_summary(audits),
        "candidates": audits,
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
    parser.add_argument("--executor-jsonl", default="")
    parser.add_argument("--signal-jsonl", required=True)
    parser.add_argument("--raw-jsonl", required=True)
    parser.add_argument("--min-line-number", type=int)
    parser.add_argument("--min-signal-created-at-ms", type=int)
    parser.add_argument("--max-signal-created-at-ms", type=int)
    parser.add_argument("--entry-time-source", choices=["created_at", "bridged_at"], default="created_at")
    parser.add_argument("--entry-edge-min", type=float)
    parser.add_argument("--min-seconds-to-expiry", type=float)
    parser.add_argument("--max-seconds-to-expiry", type=float)
    parser.add_argument("--min-entry-price", type=float)
    parser.add_argument("--min-toward-move", type=float, default=0.02)
    parser.add_argument("--residual-convergence-delta", type=float, default=0.02)
    parser.add_argument("--take-profit-delta", type=float, default=0.02)
    parser.add_argument("--model-decay-tolerance", type=float, default=0.10)
    parser.add_argument("--quote-lookback-seconds", type=float, default=30.0)
    parser.add_argument("--quote-lookahead-seconds", type=float, default=1200.0)
    parser.add_argument("--horizons-seconds", default="30,60,120,240,expiry")
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()
    args.horizons_seconds = _parse_horizons(args.horizons_seconds)
    if args.min_toward_move < 0 or args.residual_convergence_delta < 0:
        raise ValueError("convergence thresholds must be non-negative")
    if args.take_profit_delta < 0:
        raise ValueError("--take-profit-delta must be non-negative")
    if args.model_decay_tolerance < 0:
        raise ValueError("--model-decay-tolerance must be non-negative")
    return args


def _parse_horizons(text: str) -> list[int | str]:
    horizons: list[int | str] = []
    for part in text.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value == "expiry":
            horizons.append(value)
        else:
            horizons.append(int(float(value)))
    return horizons


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _config_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("event") == "phase4_started" and isinstance(event.get("config"), dict):
            return dict(event["config"])
    return {}


def _signal_start_line(events: list[dict[str, Any]]) -> int:
    for event in events:
        if event.get("event") != "phase4_started":
            continue
        cursor = event.get("cursor") or {}
        line_number = cursor.get("line_number")
        if isinstance(line_number, int):
            return line_number
    return 0


def _run_time_bounds(events: list[dict[str, Any]]) -> tuple[int, int]:
    min_ms = 0
    max_ms = 0
    for event in events:
        name = event.get("event")
        if name == "phase4_started" and min_ms <= 0:
            min_ms = _millis(event.get("started_at") or event.get("ts"))
        elif name == "phase4_summary":
            max_ms = _millis(event.get("finished_at") or event.get("ts"))
    return min_ms, max_ms


def _filter_signals_by_created_at(
    signals: list[Signal],
    *,
    min_created_at_ms: int,
    max_created_at_ms: int,
) -> list[Signal]:
    filtered: list[Signal] = []
    for signal in signals:
        if min_created_at_ms and signal.created_at_ms < min_created_at_ms:
            continue
        if max_created_at_ms and signal.created_at_ms > max_created_at_ms:
            continue
        filtered.append(signal)
    return filtered


def _load_signals(path: Path, *, min_line_number: int = 0) -> list[Signal]:
    signals: list[Signal] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number <= min_line_number or not line.strip():
                continue
            payload = json.loads(line)
            signal = _signal_from_payload(payload, line_number=line_number)
            if signal is None:
                continue
            key = (signal.event_id, signal.canonical_symbol)
            if signal.event_id and key in seen:
                continue
            signals.append(signal)
            if signal.event_id:
                seen.add(key)
    signals.sort(key=lambda item: (item.round_slug, item.feature_ts_ms, item.created_at_ms, item.side))
    return signals


def _signals_from_executor_events(events: list[dict[str, Any]]) -> list[Signal]:
    signals: list[Signal] = []
    for event in events:
        line_number = -int(event.get("_line_number") or 0)
        signal = event.get("signal")
        if isinstance(signal, dict):
            row = _signal_from_payload(signal, line_number=line_number)
            if row is not None:
                signals.append(row)
        nested = event.get("signals")
        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, dict):
                    continue
                row = _signal_from_payload(item, line_number=line_number)
                if row is not None:
                    signals.append(row)
    return signals


def _merge_signals(*groups: list[Signal]) -> list[Signal]:
    signals: list[Signal] = []
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
            signals.append(signal)
            if signal.event_id:
                seen.add(key)
    signals.sort(key=lambda item: (item.round_slug, item.feature_ts_ms, item.created_at_ms, item.side))
    return signals


def _signal_from_payload(payload: dict[str, Any], *, line_number: int) -> Signal | None:
    side = str(payload.get("outcome_side") or payload.get("selected_side") or "").upper()
    if side not in {"UP", "DOWN"}:
        return None
    model_probability = _float(payload.get("model_probability"))
    if model_probability is None:
        model_probability = _float(payload.get("token_expected_win_probability"))
    if model_probability is None:
        return None
    row = Signal(
        line_number=line_number,
        event_id=str(payload.get("event_id") or ""),
        round_slug=str(payload.get("round_slug") or ""),
        canonical_symbol=str(payload.get("canonical_symbol") or ""),
        side=side,
        feature_ts_ms=_millis(payload.get("ts")),
        created_at_ms=_millis(payload.get("created_at")),
        bridged_at_ms=_millis(payload.get("bridged_at")),
        round_end_ts_ms=_millis(payload.get("round_end_ts")),
        model_probability=model_probability,
        polymarket_price=_float(payload.get("polymarket_price") or payload.get("market_implied_prob")),
        mispricing_edge=_float(payload.get("mispricing_edge") or payload.get("selected_expected_edge") or payload.get("edge")),
    )
    if not row.round_slug or not row.canonical_symbol or row.created_at_ms <= 0:
        return None
    return row


def _quote_bounds(signals: list[Signal], *, args: argparse.Namespace) -> dict[str, int]:
    times = [item.created_at_ms for item in signals if item.created_at_ms > 0]
    if not times:
        return {"min_ts_ms": 0, "max_ts_ms": 0}
    lookback_ms = int(args.quote_lookback_seconds * 1000)
    lookahead_ms = int(args.quote_lookahead_seconds * 1000)
    return {
        "min_ts_ms": max(0, min(times) - lookback_ms),
        "max_ts_ms": max(times) + lookahead_ms,
    }


def _load_quotes(
    path: Path,
    *,
    allowed_symbols: set[str],
    min_ts_ms: int,
    max_ts_ms: int,
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
            if symbol not in allowed_symbols:
                continue
            ts_ms = _millis(
                row.get("ts")
                or row.get("message_ts")
                or row.get("capture_timestamp_ms")
                or row.get("ingest_ts")
                or payload.get("published_at_ms")
            )
            if ts_ms <= 0 or (min_ts_ms and ts_ms < min_ts_ms) or (max_ts_ms and ts_ms > max_ts_ms):
                continue
            bid = _float(row.get("bid_price"))
            ask = _float(row.get("ask_price"))
            if bid is None or ask is None or bid < 0 or ask <= 0 or ask < bid:
                continue
            quotes[symbol].append(Quote(ts_ms=ts_ms, bid=bid, ask=ask))
    for symbol, items in quotes.items():
        items.sort(key=lambda item: item.ts_ms)
        compacted: list[Quote] = []
        for item in items:
            if compacted and compacted[-1].ts_ms == item.ts_ms:
                compacted[-1] = item
            else:
                compacted.append(item)
        quotes[symbol] = compacted
    return dict(quotes)


def _candidate_entries(
    *,
    signals: list[Signal],
    quotes: dict[str, list[Quote]],
    entry_time_source: str,
    entry_edge_min: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
    min_entry_price: float,
) -> tuple[list[Candidate], Counter[str]]:
    candidates: list[Candidate] = []
    skipped: Counter[str] = Counter()
    for signal in signals:
        entry_ts_ms = signal.bridged_at_ms if entry_time_source == "bridged_at" else signal.created_at_ms
        if entry_ts_ms <= 0:
            skipped["missing_entry_ts"] += 1
            continue
        seconds_to_expiry = (signal.round_end_ts_ms - entry_ts_ms) / 1000.0
        if seconds_to_expiry < min_seconds_to_expiry:
            skipped["too_close_to_expiry"] += 1
            continue
        if seconds_to_expiry > max_seconds_to_expiry:
            skipped["too_far_from_expiry"] += 1
            continue
        quote = _quote_at(quotes.get(signal.canonical_symbol, []), entry_ts_ms)
        if quote is None:
            skipped["missing_entry_quote"] += 1
            continue
        if quote.ask < min_entry_price:
            skipped["entry_price_below_min"] += 1
            continue
        executable_edge = signal.model_probability - quote.ask
        if executable_edge < entry_edge_min:
            skipped["edge_below_threshold"] += 1
            continue
        candidates.append(
            Candidate(
                signal=signal,
                entry_ts_ms=entry_ts_ms,
                entry_bid=quote.bid,
                entry_ask=quote.ask,
                entry_mid=quote.mid,
                seconds_to_expiry=seconds_to_expiry,
                executable_edge=executable_edge,
            )
        )
    candidates.sort(key=lambda item: (item.entry_ts_ms, item.signal.side))
    return candidates, skipped


def _audit_candidate(
    candidate: Candidate,
    *,
    quotes: list[Quote],
    signals_by_symbol: list[Signal],
    horizons_seconds: list[int | str],
    min_toward_move: float,
    residual_convergence_delta: float,
    take_profit_delta: float,
    model_decay_tolerance: float,
) -> dict[str, Any]:
    signal = candidate.signal
    end_ms = signal.round_end_ts_ms if signal.round_end_ts_ms > 0 else candidate.entry_ts_ms
    future_quotes = [
        quote
        for quote in _quotes_between(quotes, candidate.entry_ts_ms, end_ms)
        if quote.ts_ms > candidate.entry_ts_ms
    ]
    direction = 1.0 if signal.model_probability >= candidate.entry_ask else -1.0
    entry_residual = signal.model_probability - candidate.entry_ask
    max_favorable_sell_move = None
    max_price_toward_model = None
    max_residual_reduction = None
    min_sell_pnl = None
    first_significant_move = ""
    first_significant_move_ts_ms = 0
    take_profit_ts_ms = 0
    toward_ts_ms = 0
    convergence_ts_ms = 0

    for quote in future_quotes:
        sell_move = quote.bid - candidate.entry_ask
        price_toward_model = (quote.mid - candidate.entry_ask) * direction
        current_residual = signal.model_probability - quote.mid
        residual_reduction = abs(entry_residual) - abs(current_residual)
        max_favorable_sell_move = _max_optional(max_favorable_sell_move, sell_move)
        min_sell_pnl = _min_optional(min_sell_pnl, sell_move)
        max_price_toward_model = _max_optional(max_price_toward_model, price_toward_model)
        max_residual_reduction = _max_optional(max_residual_reduction, residual_reduction)
        if not first_significant_move:
            if price_toward_model >= min_toward_move:
                first_significant_move = "toward"
                first_significant_move_ts_ms = quote.ts_ms
            elif price_toward_model <= -min_toward_move:
                first_significant_move = "away"
                first_significant_move_ts_ms = quote.ts_ms
        if not take_profit_ts_ms and sell_move >= take_profit_delta:
            take_profit_ts_ms = quote.ts_ms
        if not toward_ts_ms and price_toward_model >= min_toward_move:
            toward_ts_ms = quote.ts_ms
        if not convergence_ts_ms and residual_reduction >= residual_convergence_delta:
            convergence_ts_ms = quote.ts_ms

    future_model_probs = [
        item.model_probability
        for item in signals_by_symbol
        if item.canonical_symbol == signal.canonical_symbol
        and item.side == signal.side
        and item.created_at_ms > signal.created_at_ms
        and item.created_at_ms <= end_ms
    ]
    min_future_model_prob = min(future_model_probs) if future_model_probs else None
    max_future_model_prob = max(future_model_probs) if future_model_probs else None
    model_degraded = (
        min_future_model_prob is not None
        and min_future_model_prob <= signal.model_probability - model_decay_tolerance
    )
    horizons = {
        str(horizon): _horizon_snapshot(
            horizon,
            entry_ts_ms=candidate.entry_ts_ms,
            end_ms=end_ms,
            quotes=quotes,
            entry_ask=candidate.entry_ask,
            model_probability=signal.model_probability,
            direction=direction,
        )
        for horizon in horizons_seconds
    }
    return {
        "line_number": signal.line_number,
        "event_id": signal.event_id,
        "round_slug": signal.round_slug,
        "side": signal.side,
        "feature_ts": _format_utc(signal.feature_ts_ms),
        "created_at": _format_utc(signal.created_at_ms),
        "bridged_at": _format_utc(signal.bridged_at_ms),
        "entry_ts": _format_utc(candidate.entry_ts_ms),
        "seconds_to_expiry": candidate.seconds_to_expiry,
        "model_probability": signal.model_probability,
        "signal_polymarket_price": signal.polymarket_price,
        "signal_mispricing_edge": signal.mispricing_edge,
        "entry_bid": candidate.entry_bid,
        "entry_ask": candidate.entry_ask,
        "entry_mid": candidate.entry_mid,
        "entry_residual_vs_ask": entry_residual,
        "executable_edge": candidate.executable_edge,
        "future_quote_count": len(future_quotes),
        "max_favorable_sell_move": max_favorable_sell_move,
        "max_price_toward_model": max_price_toward_model,
        "max_residual_reduction": max_residual_reduction,
        "min_sell_pnl": min_sell_pnl,
        "take_profit_hit": bool(take_profit_ts_ms),
        "take_profit_ts": _format_utc(take_profit_ts_ms),
        "price_moved_toward_model": bool(toward_ts_ms),
        "price_moved_toward_model_ts": _format_utc(toward_ts_ms),
        "residual_converged": bool(convergence_ts_ms),
        "residual_converged_ts": _format_utc(convergence_ts_ms),
        "first_significant_move": first_significant_move or "none",
        "first_significant_move_ts": _format_utc(first_significant_move_ts_ms),
        "future_same_side_signal_count": len(future_model_probs),
        "min_future_model_probability": min_future_model_prob,
        "max_future_model_probability": max_future_model_prob,
        "model_probability_degraded": model_degraded,
        "horizons": horizons,
    }


def _horizon_snapshot(
    horizon: int | str,
    *,
    entry_ts_ms: int,
    end_ms: int,
    quotes: list[Quote],
    entry_ask: float,
    model_probability: float,
    direction: float,
) -> dict[str, Any]:
    target_ms = end_ms if horizon == "expiry" else min(end_ms, entry_ts_ms + int(horizon) * 1000)
    quote = _quote_at(quotes, target_ms)
    if quote is None or quote.ts_ms <= entry_ts_ms:
        return {"available": False}
    residual = model_probability - quote.mid
    return {
        "available": True,
        "ts": _format_utc(quote.ts_ms),
        "bid": quote.bid,
        "ask": quote.ask,
        "mid": quote.mid,
        "sell_move": quote.bid - entry_ask,
        "price_toward_model": (quote.mid - entry_ask) * direction,
        "residual": residual,
    }


def _quotes_between(quotes: list[Quote], start_ms: int, end_ms: int) -> list[Quote]:
    if not quotes:
        return []
    timestamps = [item.ts_ms for item in quotes]
    left = bisect.bisect_left(timestamps, start_ms)
    right = bisect.bisect_right(timestamps, end_ms)
    return quotes[left:right]


def _quote_at(quotes: list[Quote], ts_ms: int) -> Quote | None:
    if not quotes:
        return None
    timestamps = [item.ts_ms for item in quotes]
    index = bisect.bisect_right(timestamps, ts_ms) - 1
    if index < 0:
        return None
    return quotes[index]


def _summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
    side_counts = Counter(item["side"] for item in audits)
    first_move_counts = Counter(item["first_significant_move"] for item in audits)
    return {
        "candidate_count": len(audits),
        "candidate_round_count": len({item["round_slug"] for item in audits}),
        "side_counts": dict(sorted(side_counts.items())),
        "take_profit_hit_count": sum(1 for item in audits if item["take_profit_hit"]),
        "price_moved_toward_model_count": sum(
            1 for item in audits if item["price_moved_toward_model"]
        ),
        "residual_converged_count": sum(1 for item in audits if item["residual_converged"]),
        "first_significant_move_counts": dict(sorted(first_move_counts.items())),
        "model_probability_degraded_count": sum(
            1 for item in audits if item["model_probability_degraded"]
        ),
        "avg_max_favorable_sell_move": _avg(
            item["max_favorable_sell_move"] for item in audits
        ),
        "avg_max_price_toward_model": _avg(item["max_price_toward_model"] for item in audits),
        "avg_max_residual_reduction": _avg(item["max_residual_reduction"] for item in audits),
    }


def _round_summary(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_round: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in audits:
        by_round[str(item["round_slug"])].append(item)
    rows: list[dict[str, Any]] = []
    for round_slug, items in sorted(by_round.items()):
        first_move_counts = Counter(item["first_significant_move"] for item in items)
        side_counts = Counter(item["side"] for item in items)
        rows.append(
            {
                "round_slug": round_slug,
                "candidate_count": len(items),
                "side_counts": dict(sorted(side_counts.items())),
                "take_profit_hit_count": sum(1 for item in items if item["take_profit_hit"]),
                "price_moved_toward_model_count": sum(
                    1 for item in items if item["price_moved_toward_model"]
                ),
                "residual_converged_count": sum(
                    1 for item in items if item["residual_converged"]
                ),
                "first_significant_move_counts": dict(sorted(first_move_counts.items())),
                "model_probability_degraded_count": sum(
                    1 for item in items if item["model_probability_degraded"]
                ),
                "avg_max_favorable_sell_move": _avg(
                    item["max_favorable_sell_move"] for item in items
                ),
                "avg_max_price_toward_model": _avg(
                    item["max_price_toward_model"] for item in items
                ),
                "avg_max_residual_reduction": _avg(
                    item["max_residual_reduction"] for item in items
                ),
            }
        )
    return rows


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    counts = report["counts"]
    lines = [
        "# xgboost-v7 Candidate Convergence Audit",
        "",
        "## Summary",
        "",
        f"- Signals scanned: `{counts['signals']}`",
        f"- Candidate entries: `{summary['candidate_count']}` across `{summary['candidate_round_count']}` rounds",
        f"- Side counts: `{json.dumps(summary['side_counts'], sort_keys=True)}`",
        f"- Take-profit hits: `{summary['take_profit_hit_count']}`",
        f"- Price moved toward model: `{summary['price_moved_toward_model_count']}`",
        f"- Residual converged: `{summary['residual_converged_count']}`",
        f"- First significant moves: `{json.dumps(summary['first_significant_move_counts'], sort_keys=True)}`",
        f"- Model probability degraded later: `{summary['model_probability_degraded_count']}`",
        f"- Avg max favorable sell move: `{_fmt(summary['avg_max_favorable_sell_move'])}`",
        f"- Avg max price-toward-model move: `{_fmt(summary['avg_max_price_toward_model'])}`",
        f"- Avg max residual reduction: `{_fmt(summary['avg_max_residual_reduction'])}`",
        "",
        "## By Round",
        "",
        "|Round|Candidates|Sides|TP|Toward|Converged|First moves|Model degraded|Avg max sell move|",
        "|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for item in report["round_summary"]:
        lines.append(
            "|{round_slug}|{count}|`{sides}`|{tp}|{toward}|{converged}|`{first_moves}`|{degraded}|{sell}|".format(
                round_slug=item["round_slug"],
                count=item["candidate_count"],
                sides=json.dumps(item["side_counts"], sort_keys=True),
                tp=item["take_profit_hit_count"],
                toward=item["price_moved_toward_model_count"],
                converged=item["residual_converged_count"],
                first_moves=json.dumps(item["first_significant_move_counts"], sort_keys=True),
                degraded=item["model_probability_degraded_count"],
                sell=_fmt(item["avg_max_favorable_sell_move"]),
            )
        )
    lines.extend(
        [
            "",
        "## Candidate Entries",
        "",
        "|#|Created UTC|Round|Side|model_prob|entry_ask|edge|sec_left|max_sell_move|max_toward|max_resid_reduction|first_move|TP|toward|converged|model_degraded|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for index, item in enumerate(report["candidates"], start=1):
        lines.append(
            "|{index}|{created}|{round_slug}|{side}|{model_probability}|{entry_ask}|{edge}|{seconds}|{sell}|{toward}|{resid}|{first}|{tp}|{toward_flag}|{conv}|{degraded}|".format(
                index=index,
                created=item["created_at"],
                round_slug=item["round_slug"],
                side=item["side"],
                model_probability=_fmt(item["model_probability"]),
                entry_ask=_fmt(item["entry_ask"]),
                edge=_fmt(item["executable_edge"]),
                seconds=_fmt(item["seconds_to_expiry"]),
                sell=_fmt(item["max_favorable_sell_move"]),
                toward=_fmt(item["max_price_toward_model"]),
                resid=_fmt(item["max_residual_reduction"]),
                first=item["first_significant_move"],
                tp=str(bool(item["take_profit_hit"])),
                toward_flag=str(bool(item["price_moved_toward_model"])),
                conv=str(bool(item["residual_converged"])),
                degraded=str(bool(item["model_probability_degraded"])),
            )
        )
    lines.extend(
        [
            "",
            "## Label Meaning",
            "",
            "- `price_moved_toward_model`: after candidate entry, market mid moved at least the configured tolerance toward the entry model probability.",
            "- `residual_converged`: absolute residual between entry model probability and later market mid shrank by at least the configured tolerance.",
            "- `take_profit_hit`: executable sell bid moved above entry ask by at least the configured take-profit delta.",
            "- `first_significant_move`: whether the first material post-entry price move went toward or away from the model probability.",
            "- `model_probability_degraded`: later same-side model probability fell by at least the configured decay tolerance.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_float(*values: Any) -> float:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    raise ValueError("no float value available")


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _avg(values: Any) -> float | None:
    parsed = [float(value) for value in values if value is not None]
    if not parsed:
        return None
    return sum(parsed) / len(parsed)


def _max_optional(left: float | None, right: float) -> float:
    return right if left is None else max(left, right)


def _min_optional(left: float | None, right: float) -> float:
    return right if left is None else min(left, right)


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


def _format_utc(ms: int) -> str:
    if ms <= 0:
        return ""
    return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.UTC).strftime("%Y-%m-%d %H:%M:%S")


def _fmt(value: Any) -> str:
    parsed = _float(value)
    if parsed is None:
        return ""
    return f"{parsed:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
