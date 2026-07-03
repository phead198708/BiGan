#!/usr/bin/env python3
"""Analyze Phase 4 signals for volatility-exit and settlement opportunity."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.labels.v6 import VolatilityLabelConfig, compute_volatility_path_label


@dataclass(frozen=True, slots=True)
class Signal:
    event_id: str
    ts: int
    created_at: int
    bridged_at: int
    model_version: str
    canonical_symbol: str
    token_id: str
    outcome_side: str
    round_slug: str
    round_end_ts: int
    market_implied_prob: float
    token_probability: float
    edge: float

    @property
    def decision_ts(self) -> int:
        return self.bridged_at or self.created_at or self.ts


@dataclass(frozen=True, slots=True)
class Quote:
    ts: int
    bid: float | None
    ask: float | None


@dataclass(frozen=True, slots=True)
class OpportunityRow:
    event_id: str
    ts: int
    decision_ts: int
    round_slug: str
    outcome_side: str
    canonical_symbol: str
    edge: float
    token_probability: float
    signal_market_implied_prob: float
    seconds_to_expiry_at_decision: float
    policy_gate: str
    entry_quote_ts: int | None
    entry_ask: float | None
    entry_worst_price: float | None
    exit_deadline_ts: int
    max_exit_bid_ts: int | None
    max_exit_bid: float | None
    max_exit_worst_price: float | None
    max_exit_gain: float | None
    max_exit_return_per_usdc: float | None
    first_profitable_exit_ts: int | None
    first_profitable_exit_bid: float | None
    first_profitable_exit_worst_price: float | None
    first_profitable_exit_seconds: float | None
    first_min_gain_exit_ts: int | None
    first_min_gain_exit_bid: float | None
    first_min_gain_exit_seconds: float | None
    min_exit_worst_before_profit: float | None
    max_drawdown_before_profit: float | None
    soft_exit_best_bid_ts: int | None
    soft_exit_best_bid: float | None
    soft_exit_gain: float | None
    soft_exit_profitable: bool
    hard_exit_best_bid_ts: int | None
    hard_exit_best_bid: float | None
    hard_exit_gain: float | None
    hard_exit_profitable: bool
    final_quote_ts: int | None
    final_bid: float | None
    final_ask: float | None
    inferred_settlement_label: bool | None
    effective_settlement_label: bool | None
    volatility_exit_opportunity: bool
    outcome_known: bool
    realized_label: bool | None
    settlement_pnl_per_share: float | None
    settlement_return_per_usdc: float | None
    settlement_hold_opportunity: bool
    opportunity_class: str


def main() -> int:
    args = _parse_args()
    signals = load_signals_jsonl(Path(args.signals_jsonl_path))
    outcomes = (
        load_outcomes_from_duckdb(Path(args.monitoring_db_path), {signal.event_id for signal in signals})
        if args.monitoring_db_path
        else {}
    )
    quotes = load_quotes(
        [Path(path) for path in args.raw_jsonl_path],
        signals=signals,
        assume_time_ordered=not args.no_assume_time_ordered,
    )
    rows = analyze_signals(
        signals,
        quotes_by_symbol=quotes,
        outcomes_by_event_id=outcomes,
        max_entry_wait_ms=int(args.max_entry_wait_seconds * 1000),
        min_exit_seconds_before_expiry=args.min_exit_seconds_before_expiry,
        min_exit_gain=args.min_exit_gain,
        edge_threshold=args.edge_threshold,
        min_entry_price=args.min_entry_price,
        min_seconds_to_expiry=args.min_seconds_to_expiry,
        max_seconds_to_expiry=args.max_seconds_to_expiry,
        no_new_entry_before_expiry_seconds=args.no_new_entry_before_expiry_seconds,
        buy_slippage=args.buy_slippage,
        sell_slippage=args.sell_slippage,
        soft_exit_before_expiry_seconds=args.soft_exit_before_expiry_seconds,
        hard_exit_before_expiry_seconds=args.hard_exit_before_expiry_seconds,
        infer_settlement_from_final_book=args.infer_settlement_from_final_book,
        settlement_win_bid_threshold=args.settlement_win_bid_threshold,
        settlement_loss_ask_threshold=args.settlement_loss_ask_threshold,
    )
    summary = summarize(
        rows,
        edge_thresholds=_parse_thresholds(args.edge_threshold_sweep),
        min_entry_price=args.min_entry_price,
        min_seconds_to_expiry=args.min_seconds_to_expiry,
        max_seconds_to_expiry=args.max_seconds_to_expiry,
        no_new_entry_before_expiry_seconds=args.no_new_entry_before_expiry_seconds,
    )
    payload = {
        "summary": summary,
        "rows": [asdict(row) for row in rows],
        "inputs": {
            "signals_jsonl_path": args.signals_jsonl_path,
            "raw_jsonl_path": args.raw_jsonl_path,
            "monitoring_db_path": args.monitoring_db_path,
            "max_entry_wait_seconds": args.max_entry_wait_seconds,
            "min_exit_seconds_before_expiry": args.min_exit_seconds_before_expiry,
            "min_exit_gain": args.min_exit_gain,
            "edge_threshold": args.edge_threshold,
            "min_entry_price": args.min_entry_price,
            "min_seconds_to_expiry": args.min_seconds_to_expiry,
            "max_seconds_to_expiry": args.max_seconds_to_expiry,
            "no_new_entry_before_expiry_seconds": args.no_new_entry_before_expiry_seconds,
            "buy_slippage": args.buy_slippage,
            "sell_slippage": args.sell_slippage,
            "soft_exit_before_expiry_seconds": args.soft_exit_before_expiry_seconds,
            "hard_exit_before_expiry_seconds": args.hard_exit_before_expiry_seconds,
            "infer_settlement_from_final_book": args.infer_settlement_from_final_book,
            "settlement_win_bid_threshold": args.settlement_win_bid_threshold,
            "settlement_loss_ask_threshold": args.settlement_loss_ask_threshold,
        },
    }
    if args.output_json_path:
        output_json_path = Path(args.output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            markdown_report(payload, max_rows=args.max_report_rows),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-jsonl-path", required=True)
    parser.add_argument(
        "--raw-jsonl-path",
        action="append",
        required=True,
        help="Low-latency raw queue JSONL path; may be repeated.",
    )
    parser.add_argument("--monitoring-db-path", default="data/mlops/champion_catalog.duckdb")
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--max-report-rows", type=int, default=30)
    parser.add_argument("--max-entry-wait-seconds", type=float, default=60.0)
    parser.add_argument("--min-exit-seconds-before-expiry", type=float, default=300.0)
    parser.add_argument("--min-exit-gain", type=float, default=0.15)
    parser.add_argument("--edge-threshold", type=float, default=0.45)
    parser.add_argument("--min-entry-price", type=float, default=0.35)
    parser.add_argument("--min-seconds-to-expiry", type=float, default=300.0)
    parser.add_argument("--max-seconds-to-expiry", type=float, default=1200.0)
    parser.add_argument("--no-new-entry-before-expiry-seconds", type=float, default=300.0)
    parser.add_argument("--buy-slippage", type=float, default=0.02)
    parser.add_argument("--sell-slippage", type=float, default=0.02)
    parser.add_argument("--soft-exit-before-expiry-seconds", type=float, default=240.0)
    parser.add_argument("--hard-exit-before-expiry-seconds", type=float, default=120.0)
    parser.add_argument(
        "--no-infer-settlement-from-final-book",
        action="store_false",
        dest="infer_settlement_from_final_book",
        help="Only use explicit prediction_outcomes rows for settlement labels.",
    )
    parser.set_defaults(infer_settlement_from_final_book=True)
    parser.add_argument("--settlement-win-bid-threshold", type=float, default=0.98)
    parser.add_argument("--settlement-loss-ask-threshold", type=float, default=0.02)
    parser.add_argument(
        "--edge-threshold-sweep",
        default="-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30,0.45,0.60",
        help="Comma-separated edge thresholds for counterfactual gating evaluation.",
    )
    parser.add_argument(
        "--no-assume-time-ordered",
        action="store_true",
        help="Scan all raw rows instead of stopping after the signal time window.",
    )
    return parser.parse_args()


def load_signals_jsonl(path: Path) -> list[Signal]:
    signals: list[Signal] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                signals.append(
                    Signal(
                        event_id=str(payload["event_id"]),
                        ts=int(payload["ts"]),
                        created_at=int(payload.get("created_at") or 0),
                        bridged_at=int(payload.get("bridged_at") or 0),
                        model_version=str(payload.get("model_version") or ""),
                        canonical_symbol=str(payload["canonical_symbol"]),
                        token_id=str(payload.get("token_id") or ""),
                        outcome_side=str(payload["outcome_side"]).upper(),
                        round_slug=str(payload["round_slug"]),
                        round_end_ts=int(payload["round_end_ts"]),
                        market_implied_prob=float(payload["market_implied_prob"]),
                        token_probability=float(payload["token_probability"]),
                        edge=float(payload["edge"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} missing required signal field {exc}") from exc
    return signals


def load_outcomes_from_duckdb(path: Path, event_ids: set[str]) -> dict[str, bool]:
    if not path.exists() or not event_ids:
        return {}
    import duckdb

    placeholders = ",".join("?" for _ in event_ids)
    query = f"""
        SELECT event_id, realized_label
        FROM prediction_outcomes
        WHERE event_id IN ({placeholders})
    """
    with duckdb.connect(str(path), read_only=True) as conn:
        rows = conn.execute(query, sorted(event_ids)).fetchall()
    return {str(event_id): bool(realized_label) for event_id, realized_label in rows}


def load_quotes(
    raw_jsonl_paths: list[Path],
    *,
    signals: list[Signal],
    assume_time_ordered: bool = True,
) -> dict[str, list[Quote]]:
    if not signals:
        return {}
    canonical_symbols = {signal.canonical_symbol for signal in signals}
    symbols_by_token = {signal.token_id: signal.canonical_symbol for signal in signals if signal.token_id}
    min_ts = min(signal.decision_ts for signal in signals)
    max_ts = max(signal.round_end_ts for signal in signals)
    quotes: dict[str, list[Quote]] = {symbol: [] for symbol in canonical_symbols}
    for path in raw_jsonl_paths:
        with _open_text(path) as handle:
            for line in handle:
                if "raw_top_of_book" in line:
                    parsed_quotes = _quotes_from_raw_queue_line(line)
                    if (
                        parsed_quotes
                        and assume_time_ordered
                        and not path.name.endswith(".gz")
                        and parsed_quotes[0][1].ts > max_ts
                    ):
                        break
                    for canonical_symbol, quote in parsed_quotes:
                        if quote.ts < min_ts or quote.ts > max_ts:
                            continue
                        if canonical_symbol not in canonical_symbols:
                            continue
                        quotes[canonical_symbol].append(quote)
                    continue
                if "price_change" not in line and '"book"' not in line:
                    continue
                for canonical_symbol, quote in _quotes_from_ws_line(
                    line,
                    symbols_by_token=symbols_by_token,
                ):
                    if quote.ts is None:
                        continue
                    if quote.ts < min_ts or quote.ts > max_ts:
                        continue
                    quotes[canonical_symbol].append(quote)
    for symbol in quotes:
        quotes[symbol].sort(key=lambda quote: quote.ts)
    return quotes


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _quotes_from_raw_queue_line(line: str) -> list[tuple[str, Quote]]:
    payload = json.loads(line)
    row = payload.get("row") or {}
    ts = _optional_int(row.get("ts"))
    if ts is None:
        return []
    canonical_symbol = str(row.get("canonical_symbol") or "")
    if not canonical_symbol:
        return []
    return [
        (
            canonical_symbol,
            Quote(
                ts=ts,
                bid=_optional_float(row.get("bid_price")),
                ask=_optional_float(row.get("ask_price")),
            ),
        )
    ]


def _quotes_from_ws_line(
    line: str,
    *,
    symbols_by_token: dict[str, str],
) -> list[tuple[str, Quote]]:
    payload = json.loads(line)
    raw = payload.get("raw") or payload
    event_type = raw.get("event_type") or payload.get("event_type")
    ts = _optional_int(raw.get("timestamp")) or _optional_int(
        payload.get("receive_timestamp_ms") or payload.get("capture_timestamp_ms")
    )
    if ts is None:
        return []
    if event_type == "book":
        asset_id = str(raw.get("asset_id") or "")
        canonical_symbol = symbols_by_token.get(asset_id)
        if canonical_symbol is None:
            return []
        bid = _best_bid(raw.get("bids") or [])
        ask = _best_ask(raw.get("asks") or [])
        return [(canonical_symbol, Quote(ts=ts, bid=bid, ask=ask))]
    if event_type == "price_change":
        quotes: list[tuple[str, Quote]] = []
        for change in raw.get("price_changes") or []:
            asset_id = str(change.get("asset_id") or "")
            canonical_symbol = symbols_by_token.get(asset_id)
            if canonical_symbol is None:
                continue
            quotes.append(
                (
                    canonical_symbol,
                    Quote(
                        ts=ts,
                        bid=_optional_float(change.get("best_bid")),
                        ask=_optional_float(change.get("best_ask")),
                    ),
                )
            )
        return quotes
    return []


def analyze_signals(
    signals: list[Signal],
    *,
    quotes_by_symbol: dict[str, list[Quote]],
    outcomes_by_event_id: dict[str, bool],
    max_entry_wait_ms: int,
    min_exit_seconds_before_expiry: float,
    min_exit_gain: float,
    edge_threshold: float,
    min_entry_price: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
    no_new_entry_before_expiry_seconds: float,
    buy_slippage: float,
    sell_slippage: float,
    soft_exit_before_expiry_seconds: float,
    hard_exit_before_expiry_seconds: float,
    infer_settlement_from_final_book: bool,
    settlement_win_bid_threshold: float,
    settlement_loss_ask_threshold: float,
) -> list[OpportunityRow]:
    rows: list[OpportunityRow] = []
    for signal in signals:
        quotes = quotes_by_symbol.get(signal.canonical_symbol, [])
        decision_ts = signal.decision_ts
        seconds_to_expiry = (signal.round_end_ts - decision_ts) / 1000.0
        entry_quote = _entry_quote(quotes, decision_ts, max_entry_wait_ms)
        entry_ask = entry_quote.ask if entry_quote is not None else None
        entry_worst = (
            min(0.99, entry_ask + buy_slippage)
            if entry_ask is not None and math.isfinite(entry_ask)
            else None
        )
        policy_gate = _policy_gate(
            signal,
            seconds_to_expiry=seconds_to_expiry,
            entry_worst_price=entry_worst,
            edge_threshold=edge_threshold,
            min_entry_price=min_entry_price,
            min_seconds_to_expiry=min_seconds_to_expiry,
            max_seconds_to_expiry=max_seconds_to_expiry,
            no_new_entry_before_expiry_seconds=no_new_entry_before_expiry_seconds,
        )
        volatility_path = compute_volatility_path_label(
            quotes,
            decision_ts=decision_ts,
            round_end_ts=signal.round_end_ts,
            config=VolatilityLabelConfig(
                min_exit_gain=min_exit_gain,
                buy_slippage=buy_slippage,
                sell_slippage=sell_slippage,
                max_entry_wait_ms=max_entry_wait_ms,
                min_exit_seconds_before_expiry=min_exit_seconds_before_expiry,
                min_entry_price=min_entry_price,
            ),
        )
        exit_deadline_ts = volatility_path.exit_deadline_ts
        max_exit_bid = volatility_path.best_exit_bid
        max_exit_worst = volatility_path.best_exit_price
        max_exit_gain = volatility_path.max_exit_gain
        max_exit_return_per_usdc = volatility_path.max_exit_return_per_usdc
        first_profitable_exit_quote = (
            _first_exit_quote(
                quotes,
                entry_quote.ts,
                signal.round_end_ts,
                entry_worst=entry_worst,
                sell_slippage=sell_slippage,
                min_gain=0.0,
            )
            if entry_quote is not None and entry_worst is not None
            else None
        )
        first_min_gain_exit_quote = (
            _first_exit_quote(
                quotes,
                entry_quote.ts,
                signal.round_end_ts,
                entry_worst=entry_worst,
                sell_slippage=sell_slippage,
                min_gain=min_exit_gain,
            )
            if entry_quote is not None and entry_worst is not None
            else None
        )
        first_profitable_exit_worst = (
            _exit_worst(first_profitable_exit_quote.bid, sell_slippage)
            if first_profitable_exit_quote is not None and first_profitable_exit_quote.bid is not None
            else None
        )
        drawdown_cutoff_ts = (
            first_profitable_exit_quote.ts
            if first_profitable_exit_quote is not None
            else signal.round_end_ts
        )
        min_exit_worst_before_profit = (
            _min_exit_worst(quotes, entry_quote.ts, drawdown_cutoff_ts, sell_slippage=sell_slippage)
            if entry_quote is not None and entry_worst is not None
            else None
        )
        max_drawdown_before_profit = (
            max(0.0, entry_worst - min_exit_worst_before_profit)
            if entry_worst is not None and min_exit_worst_before_profit is not None
            else None
        )
        soft_exit_quote = _best_window_exit_quote(
            quotes,
            signal.round_end_ts - int(soft_exit_before_expiry_seconds * 1000),
            signal.round_end_ts,
        )
        hard_exit_quote = _best_window_exit_quote(
            quotes,
            signal.round_end_ts - int(hard_exit_before_expiry_seconds * 1000),
            signal.round_end_ts,
        )
        soft_exit_worst = (
            _exit_worst(soft_exit_quote.bid, sell_slippage)
            if soft_exit_quote is not None and soft_exit_quote.bid is not None
            else None
        )
        hard_exit_worst = (
            _exit_worst(hard_exit_quote.bid, sell_slippage)
            if hard_exit_quote is not None and hard_exit_quote.bid is not None
            else None
        )
        soft_exit_gain = (
            soft_exit_worst - entry_worst
            if soft_exit_worst is not None and entry_worst is not None
            else None
        )
        hard_exit_gain = (
            hard_exit_worst - entry_worst
            if hard_exit_worst is not None and entry_worst is not None
            else None
        )
        final_quote = _final_quote_before(quotes, signal.round_end_ts)
        inferred_settlement_label = (
            _infer_settlement_label(
                final_quote,
                win_bid_threshold=settlement_win_bid_threshold,
                loss_ask_threshold=settlement_loss_ask_threshold,
            )
            if infer_settlement_from_final_book
            else None
        )
        volatility_exit_opportunity = (
            max_exit_gain is not None and max_exit_gain + 1e-12 >= min_exit_gain
        )
        realized_label = outcomes_by_event_id.get(signal.event_id)
        effective_settlement_label = (
            realized_label if realized_label is not None else inferred_settlement_label
        )
        settlement_pnl_per_share = (
            (1.0 - entry_worst)
            if effective_settlement_label is True and entry_worst is not None
            else (
                -entry_worst
                if effective_settlement_label is False and entry_worst is not None
                else None
            )
        )
        settlement_return_per_usdc = (
            ((1.0 / entry_worst) - 1.0)
            if effective_settlement_label is True and entry_worst is not None and entry_worst > 0
            else (
                -1.0
                if effective_settlement_label is False and entry_worst is not None
                else None
            )
        )
        settlement_hold_opportunity = (
            effective_settlement_label is True and entry_worst is not None
        )
        opportunity_class = _opportunity_class(
            volatility_exit_opportunity=volatility_exit_opportunity,
            settlement_hold_opportunity=settlement_hold_opportunity,
            realized_label=effective_settlement_label,
        )
        rows.append(
            OpportunityRow(
                event_id=signal.event_id,
                ts=signal.ts,
                decision_ts=decision_ts,
                round_slug=signal.round_slug,
                outcome_side=signal.outcome_side,
                canonical_symbol=signal.canonical_symbol,
                edge=signal.edge,
                token_probability=signal.token_probability,
                signal_market_implied_prob=signal.market_implied_prob,
                seconds_to_expiry_at_decision=seconds_to_expiry,
                policy_gate=policy_gate,
                entry_quote_ts=entry_quote.ts if entry_quote is not None else None,
                entry_ask=entry_ask,
                entry_worst_price=entry_worst,
                exit_deadline_ts=exit_deadline_ts,
                max_exit_bid_ts=volatility_path.best_exit_ts,
                max_exit_bid=max_exit_bid,
                max_exit_worst_price=max_exit_worst,
                max_exit_gain=max_exit_gain,
                max_exit_return_per_usdc=max_exit_return_per_usdc,
                first_profitable_exit_ts=(
                    first_profitable_exit_quote.ts
                    if first_profitable_exit_quote is not None
                    else None
                ),
                first_profitable_exit_bid=(
                    first_profitable_exit_quote.bid
                    if first_profitable_exit_quote is not None
                    else None
                ),
                first_profitable_exit_worst_price=first_profitable_exit_worst,
                first_profitable_exit_seconds=(
                    (first_profitable_exit_quote.ts - entry_quote.ts) / 1000.0
                    if first_profitable_exit_quote is not None and entry_quote is not None
                    else None
                ),
                first_min_gain_exit_ts=(
                    first_min_gain_exit_quote.ts
                    if first_min_gain_exit_quote is not None
                    else None
                ),
                first_min_gain_exit_bid=(
                    first_min_gain_exit_quote.bid
                    if first_min_gain_exit_quote is not None
                    else None
                ),
                first_min_gain_exit_seconds=(
                    (first_min_gain_exit_quote.ts - entry_quote.ts) / 1000.0
                    if first_min_gain_exit_quote is not None and entry_quote is not None
                    else None
                ),
                min_exit_worst_before_profit=min_exit_worst_before_profit,
                max_drawdown_before_profit=max_drawdown_before_profit,
                soft_exit_best_bid_ts=soft_exit_quote.ts if soft_exit_quote is not None else None,
                soft_exit_best_bid=soft_exit_quote.bid if soft_exit_quote is not None else None,
                soft_exit_gain=soft_exit_gain,
                soft_exit_profitable=bool(soft_exit_gain is not None and soft_exit_gain > 0),
                hard_exit_best_bid_ts=hard_exit_quote.ts if hard_exit_quote is not None else None,
                hard_exit_best_bid=hard_exit_quote.bid if hard_exit_quote is not None else None,
                hard_exit_gain=hard_exit_gain,
                hard_exit_profitable=bool(hard_exit_gain is not None and hard_exit_gain > 0),
                final_quote_ts=final_quote.ts if final_quote is not None else None,
                final_bid=final_quote.bid if final_quote is not None else None,
                final_ask=final_quote.ask if final_quote is not None else None,
                inferred_settlement_label=inferred_settlement_label,
                effective_settlement_label=effective_settlement_label,
                volatility_exit_opportunity=volatility_exit_opportunity,
                outcome_known=effective_settlement_label is not None,
                realized_label=realized_label,
                settlement_pnl_per_share=settlement_pnl_per_share,
                settlement_return_per_usdc=settlement_return_per_usdc,
                settlement_hold_opportunity=settlement_hold_opportunity,
                opportunity_class=opportunity_class,
            )
        )
    return rows


def summarize(
    rows: list[OpportunityRow],
    *,
    edge_thresholds: list[float] | None = None,
    min_entry_price: float = 0.35,
    min_seconds_to_expiry: float = 300.0,
    max_seconds_to_expiry: float = 1200.0,
    no_new_entry_before_expiry_seconds: float = 300.0,
) -> dict[str, Any]:
    gates: dict[str, int] = {}
    classes: dict[str, int] = {}
    gate_opportunities: dict[str, int] = {}
    gate_volatility_opportunities: dict[str, int] = {}
    gate_settlement_opportunities: dict[str, int] = {}
    gate_gain_sum: dict[str, float] = {}
    for row in rows:
        gates[row.policy_gate] = gates.get(row.policy_gate, 0) + 1
        classes[row.opportunity_class] = classes.get(row.opportunity_class, 0) + 1
        if row.volatility_exit_opportunity:
            gate_volatility_opportunities[row.policy_gate] = (
                gate_volatility_opportunities.get(row.policy_gate, 0) + 1
            )
        if row.settlement_hold_opportunity:
            gate_settlement_opportunities[row.policy_gate] = (
                gate_settlement_opportunities.get(row.policy_gate, 0) + 1
            )
        if _has_opportunity(row):
            gate_opportunities[row.policy_gate] = gate_opportunities.get(row.policy_gate, 0) + 1
            gate_gain_sum[row.policy_gate] = gate_gain_sum.get(row.policy_gate, 0.0) + (
                row.max_exit_gain or 0.0
            )
    executor_candidates = [row for row in rows if row.policy_gate == "executor_candidate"]
    executor_opportunities = [row for row in executor_candidates if _has_opportunity(row)]
    blocked = [row for row in rows if row.policy_gate != "executor_candidate"]
    blocked_opportunities = [row for row in blocked if _has_opportunity(row)]
    opportunity_count = sum(1 for row in rows if _has_opportunity(row))
    no_opportunity_count = len(rows) - opportunity_count
    blocked_with_volatility = sum(
        1
        for row in rows
        if row.policy_gate != "executor_candidate" and row.volatility_exit_opportunity
    )
    gate_table = {}
    for gate, count in sorted(gates.items()):
        opportunities = gate_opportunities.get(gate, 0)
        volatility_opportunities = gate_volatility_opportunities.get(gate, 0)
        settlement_opportunities = gate_settlement_opportunities.get(gate, 0)
        gate_table[gate] = {
            "signals": count,
            "opportunities": opportunities,
            "volatility_exit_opportunities": volatility_opportunities,
            "settlement_hold_opportunities": settlement_opportunities,
            "opportunity_rate": _safe_ratio(opportunities, count),
            "volatility_opportunity_rate": _safe_ratio(volatility_opportunities, count),
            "settlement_opportunity_rate": _safe_ratio(settlement_opportunities, count),
            "avg_opportunity_gain": _safe_ratio(gate_gain_sum.get(gate, 0.0), opportunities),
        }
    confusion = {
        "true_positive_allowed_opportunity": len(executor_opportunities),
        "false_positive_allowed_no_opportunity": len(executor_candidates) - len(executor_opportunities),
        "false_negative_blocked_opportunity": len(blocked_opportunities),
        "true_negative_blocked_no_opportunity": len(blocked) - len(blocked_opportunities),
        "candidate_precision": _safe_ratio(len(executor_opportunities), len(executor_candidates)),
        "opportunity_recall": _safe_ratio(len(executor_opportunities), opportunity_count),
        "overfilter_rate_among_blocked": _safe_ratio(len(blocked_opportunities), len(blocked)),
        "underfilter_rate_among_candidates": _safe_ratio(
            len(executor_candidates) - len(executor_opportunities),
            len(executor_candidates),
        ),
    }
    return {
        "signals": len(rows),
        "with_entry_quote": sum(1 for row in rows if row.entry_ask is not None),
        "with_exit_quote": sum(1 for row in rows if row.max_exit_bid is not None),
        "opportunities": opportunity_count,
        "no_opportunities": no_opportunity_count,
        "volatility_exit_opportunities": sum(
            1 for row in rows if row.volatility_exit_opportunity
        ),
        "settlement_outcomes_explicit": sum(1 for row in rows if row.realized_label is not None),
        "settlement_outcomes_inferred": sum(
            1 for row in rows if row.realized_label is None and row.inferred_settlement_label is not None
        ),
        "settlement_outcomes_known": sum(1 for row in rows if row.outcome_known),
        "settlement_hold_opportunities": sum(
            1 for row in rows if row.settlement_hold_opportunity
        ),
        "wrong_outcome_but_volatility_exit": sum(
            1
            for row in rows
            if row.effective_settlement_label is False and row.volatility_exit_opportunity
        ),
        "blocked_by_policy_but_volatility_exit": blocked_with_volatility,
        "policy_gates": dict(sorted(gates.items())),
        "policy_gate_table": gate_table,
        "gating_confusion": confusion,
        "gating_confusion_by_opportunity_type": {
            "volatility_exit": _gating_confusion_for(rows, "volatility_exit_opportunity"),
            "settlement_hold": _gating_confusion_for(rows, "settlement_hold_opportunity"),
        },
        "edge_threshold_sweep": _edge_threshold_sweep(
            rows,
            edge_thresholds=edge_thresholds or [],
            min_entry_price=min_entry_price,
            min_seconds_to_expiry=min_seconds_to_expiry,
            max_seconds_to_expiry=max_seconds_to_expiry,
            no_new_entry_before_expiry_seconds=no_new_entry_before_expiry_seconds,
        ),
        "opportunity_classes": dict(sorted(classes.items())),
    }


def markdown_report(payload: dict[str, Any], *, max_rows: int) -> str:
    summary = payload["summary"]
    rows = [OpportunityRow(**row) for row in payload["rows"]]
    lines = [
        "# Phase 4 Signal Opportunity Analysis",
        "",
        "## Summary",
        "",
        f"- Signals: {summary['signals']}",
        f"- Signals with entry quote: {summary['with_entry_quote']}",
        f"- Signals with future exit quote: {summary['with_exit_quote']}",
        f"- Opportunities: {summary['opportunities']}",
        f"- Volatility-exit opportunities: {summary['volatility_exit_opportunities']}",
        f"- Settlement outcomes known: {summary['settlement_outcomes_known']}",
        f"- Settlement outcomes explicit: {summary['settlement_outcomes_explicit']}",
        f"- Settlement outcomes inferred from final book: {summary['settlement_outcomes_inferred']}",
        f"- Settlement-hold opportunities: {summary['settlement_hold_opportunities']}",
        f"- Wrong outcome but volatility-exit opportunity: {summary['wrong_outcome_but_volatility_exit']}",
        f"- Blocked by policy but volatility-exit opportunity: {summary['blocked_by_policy_but_volatility_exit']}",
        "",
        "## Post-Signal Trading Labels",
        "",
        "- `first_profitable_exit_seconds`: seconds from entry quote to first bid-minus-slippage exit above entry worst price.",
        "- `max_drawdown_before_profit`: worst bid-minus-slippage adverse move before the first profitable exit, or through expiry if no profitable exit appears.",
        "- `soft_exit_profitable` / `hard_exit_profitable`: whether the best bid in the configured exit window could still exit above entry worst price.",
        "- `effective_settlement_label`: explicit `prediction_outcomes` label when available, otherwise inferred from final top-of-book near expiry.",
        "",
        "## Policy Gates",
        "",
    ]
    for gate, count in summary["policy_gates"].items():
        gate_row = summary["policy_gate_table"][gate]
        lines.append(
            f"- {gate}: {count} signals, "
            f"{gate_row['opportunities']} opportunities, "
            f"{gate_row['volatility_exit_opportunities']} volatility, "
            f"{gate_row['settlement_hold_opportunities']} settlement, "
            f"opportunity_rate={_fmt(gate_row['opportunity_rate'])}"
        )
    confusion = summary["gating_confusion"]
    lines.extend(
        [
            "",
            "## Gating Confusion",
            "",
            f"- True positive, allowed opportunity: {confusion['true_positive_allowed_opportunity']}",
            f"- False positive, allowed no opportunity: {confusion['false_positive_allowed_no_opportunity']}",
            f"- False negative, blocked opportunity: {confusion['false_negative_blocked_opportunity']}",
            f"- True negative, blocked no opportunity: {confusion['true_negative_blocked_no_opportunity']}",
            f"- Candidate precision: {_fmt(confusion['candidate_precision'])}",
            f"- Opportunity recall: {_fmt(confusion['opportunity_recall'])}",
            f"- Over-filter rate among blocked: {_fmt(confusion['overfilter_rate_among_blocked'])}",
            f"- Under-filter rate among candidates: {_fmt(confusion['underfilter_rate_among_candidates'])}",
            "",
            "## Gating Confusion By Opportunity Type",
            "",
        ]
    )
    for label, typed_confusion in summary["gating_confusion_by_opportunity_type"].items():
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Opportunities: {typed_confusion['opportunity_count']}",
                f"- Allowed opportunities: {typed_confusion['true_positive_allowed_opportunity']}",
                f"- Blocked opportunities: {typed_confusion['false_negative_blocked_opportunity']}",
                f"- Candidate precision: {_fmt(typed_confusion['candidate_precision'])}",
                f"- Opportunity recall: {_fmt(typed_confusion['opportunity_recall'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Edge Threshold Sweep",
            "",
            "| Edge Threshold | Candidates | Opps Allowed | Precision | Recall | Vol Allowed | Vol Recall | Settle Allowed | Settle Recall |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["edge_threshold_sweep"]:
        lines.append(
            "| "
            f"{item['edge_threshold']:.2f} | "
            f"{item['candidates']} | "
            f"{item['opportunities_allowed']} | "
            f"{_fmt(item['precision'])} | "
            f"{_fmt(item['recall'])} | "
            f"{item['volatility_opportunities_allowed']} | "
            f"{_fmt(item['volatility_recall'])} | "
            f"{item['settlement_opportunities_allowed']} | "
            f"{_fmt(item['settlement_recall'])} |"
        )
    lines.extend(["", "## Opportunity Classes", ""])
    for klass, count in summary["opportunity_classes"].items():
        lines.append(f"- {klass}: {count}")
    lines.extend(
        [
            "",
            "## Top Signals By Max Exit Gain",
            "",
            "| Event | Round | Side | Gate | Entry | Max Exit | Gain | Return/1 USDC | First Profit s | Drawdown Before Profit | Soft Exit PnL | Hard Exit PnL | Settlement | Class | Decision UTC | Max Exit UTC |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    ranked = sorted(
        rows,
        key=lambda row: row.max_exit_gain if row.max_exit_gain is not None else -999.0,
        reverse=True,
    )
    for row in ranked[:max_rows]:
        lines.append(
            "| "
            f"{row.event_id} | "
            f"{row.round_slug} | "
            f"{row.outcome_side} | "
            f"{row.policy_gate} | "
            f"{_fmt(row.entry_worst_price)} | "
            f"{_fmt(row.max_exit_worst_price)} | "
            f"{_fmt(row.max_exit_gain)} | "
            f"{_fmt(row.max_exit_return_per_usdc)} | "
            f"{_fmt(row.first_profitable_exit_seconds)} | "
            f"{_fmt(row.max_drawdown_before_profit)} | "
            f"{_fmt(row.soft_exit_gain)} | "
            f"{_fmt(row.hard_exit_gain)} | "
            f"{_label(row.effective_settlement_label)} | "
            f"{row.opportunity_class} | "
            f"{_iso(row.decision_ts)} | "
            f"{_iso(row.max_exit_bid_ts) if row.max_exit_bid_ts is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "Entry uses ask plus configured buy slippage; exit uses bid minus configured sell slippage.",
            "Volatility opportunity means max executable exit gain reached the configured min-exit-gain before the safe exit deadline.",
            "",
        ]
    )
    return "\n".join(lines)


def _entry_quote(quotes: list[Quote], decision_ts: int, max_entry_wait_ms: int) -> Quote | None:
    deadline = decision_ts + max_entry_wait_ms
    for quote in quotes:
        if quote.ts < decision_ts:
            continue
        if quote.ts > deadline:
            break
        if quote.ask is not None:
            return quote
    return None


def _max_exit_bid_quote(quotes: list[Quote], start_ts: int, deadline_ts: int) -> Quote | None:
    best: Quote | None = None
    for quote in quotes:
        if quote.ts < start_ts:
            continue
        if quote.ts > deadline_ts:
            break
        if quote.bid is None:
            continue
        if best is None or (best.bid is not None and quote.bid > best.bid):
            best = quote
    return best


def _first_exit_quote(
    quotes: list[Quote],
    start_ts: int,
    deadline_ts: int,
    *,
    entry_worst: float,
    sell_slippage: float,
    min_gain: float,
) -> Quote | None:
    target = entry_worst + min_gain
    for quote in quotes:
        if quote.ts < start_ts:
            continue
        if quote.ts > deadline_ts:
            break
        if quote.bid is None:
            continue
        if _exit_worst(quote.bid, sell_slippage) > target:
            return quote
    return None


def _min_exit_worst(
    quotes: list[Quote],
    start_ts: int,
    deadline_ts: int,
    *,
    sell_slippage: float,
) -> float | None:
    worst: float | None = None
    for quote in quotes:
        if quote.ts < start_ts:
            continue
        if quote.ts > deadline_ts:
            break
        if quote.bid is None:
            continue
        value = _exit_worst(quote.bid, sell_slippage)
        if worst is None or value < worst:
            worst = value
    return worst


def _best_window_exit_quote(
    quotes: list[Quote],
    start_ts: int,
    deadline_ts: int,
) -> Quote | None:
    best: Quote | None = None
    for quote in quotes:
        if quote.ts < start_ts:
            continue
        if quote.ts > deadline_ts:
            break
        if quote.bid is None:
            continue
        if best is None or (best.bid is not None and quote.bid > best.bid):
            best = quote
    return best


def _final_quote_before(quotes: list[Quote], deadline_ts: int) -> Quote | None:
    final: Quote | None = None
    for quote in quotes:
        if quote.ts > deadline_ts:
            break
        if quote.bid is None and quote.ask is None:
            continue
        final = quote
    return final


def _exit_worst(bid: float, sell_slippage: float) -> float:
    return max(0.01, bid - sell_slippage)


def _infer_settlement_label(
    quote: Quote | None,
    *,
    win_bid_threshold: float,
    loss_ask_threshold: float,
) -> bool | None:
    if quote is None:
        return None
    if quote.bid is not None and quote.bid >= win_bid_threshold:
        return True
    if quote.ask is not None and quote.ask <= loss_ask_threshold:
        return False
    return None


def _policy_gate(
    signal: Signal,
    *,
    seconds_to_expiry: float,
    entry_worst_price: float | None,
    edge_threshold: float,
    min_entry_price: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
    no_new_entry_before_expiry_seconds: float,
) -> str:
    if seconds_to_expiry < no_new_entry_before_expiry_seconds:
        return "no_new_entry_window"
    if seconds_to_expiry < min_seconds_to_expiry:
        return "near_or_past_expiry"
    if seconds_to_expiry > max_seconds_to_expiry:
        return "too_far_from_expiry"
    if signal.edge < edge_threshold:
        return "below_edge_threshold"
    if entry_worst_price is None:
        return "missing_entry_quote"
    if entry_worst_price < min_entry_price:
        return "entry_price_below_min"
    return "executor_candidate"


def _opportunity_class(
    *,
    volatility_exit_opportunity: bool,
    settlement_hold_opportunity: bool,
    realized_label: bool | None,
) -> str:
    if volatility_exit_opportunity and settlement_hold_opportunity:
        return "volatility_and_settlement"
    if volatility_exit_opportunity and realized_label is False:
        return "wrong_outcome_but_volatility_exit"
    if volatility_exit_opportunity and realized_label is None:
        return "volatility_exit_pending_settlement"
    if volatility_exit_opportunity:
        return "volatility_exit_only"
    if settlement_hold_opportunity:
        return "settlement_hold_only"
    if realized_label is False:
        return "no_opportunity_wrong_outcome"
    if realized_label is None:
        return "no_volatility_exit_pending_settlement"
    return "no_opportunity"


def _has_opportunity(row: OpportunityRow) -> bool:
    return row.volatility_exit_opportunity or row.settlement_hold_opportunity


def _gating_confusion_for(rows: list[OpportunityRow], opportunity_attr: str) -> dict[str, Any]:
    executor_candidates = [row for row in rows if row.policy_gate == "executor_candidate"]
    executor_opportunities = [
        row for row in executor_candidates if bool(getattr(row, opportunity_attr))
    ]
    blocked = [row for row in rows if row.policy_gate != "executor_candidate"]
    blocked_opportunities = [row for row in blocked if bool(getattr(row, opportunity_attr))]
    opportunity_count = sum(1 for row in rows if bool(getattr(row, opportunity_attr)))
    return {
        "opportunity_count": opportunity_count,
        "true_positive_allowed_opportunity": len(executor_opportunities),
        "false_positive_allowed_no_opportunity": len(executor_candidates)
        - len(executor_opportunities),
        "false_negative_blocked_opportunity": len(blocked_opportunities),
        "true_negative_blocked_no_opportunity": len(blocked) - len(blocked_opportunities),
        "candidate_precision": _safe_ratio(len(executor_opportunities), len(executor_candidates)),
        "opportunity_recall": _safe_ratio(len(executor_opportunities), opportunity_count),
        "overfilter_rate_among_blocked": _safe_ratio(len(blocked_opportunities), len(blocked)),
        "underfilter_rate_among_candidates": _safe_ratio(
            len(executor_candidates) - len(executor_opportunities),
            len(executor_candidates),
        ),
    }


def _edge_threshold_sweep(
    rows: list[OpportunityRow],
    *,
    edge_thresholds: list[float],
    min_entry_price: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
    no_new_entry_before_expiry_seconds: float,
) -> list[dict[str, Any]]:
    opportunity_count = sum(1 for row in rows if _has_opportunity(row))
    volatility_opportunity_count = sum(1 for row in rows if row.volatility_exit_opportunity)
    settlement_opportunity_count = sum(1 for row in rows if row.settlement_hold_opportunity)
    sweep: list[dict[str, Any]] = []
    for threshold in edge_thresholds:
        candidates = [
            row
            for row in rows
            if _counterfactual_candidate(
                row,
                edge_threshold=threshold,
                min_entry_price=min_entry_price,
                min_seconds_to_expiry=min_seconds_to_expiry,
                max_seconds_to_expiry=max_seconds_to_expiry,
                no_new_entry_before_expiry_seconds=no_new_entry_before_expiry_seconds,
            )
        ]
        opportunities_allowed = sum(1 for row in candidates if _has_opportunity(row))
        volatility_opportunities_allowed = sum(
            1 for row in candidates if row.volatility_exit_opportunity
        )
        settlement_opportunities_allowed = sum(
            1 for row in candidates if row.settlement_hold_opportunity
        )
        sweep.append(
            {
                "edge_threshold": threshold,
                "candidates": len(candidates),
                "opportunities_allowed": opportunities_allowed,
                "opportunities_blocked": opportunity_count - opportunities_allowed,
                "precision": _safe_ratio(opportunities_allowed, len(candidates)),
                "recall": _safe_ratio(opportunities_allowed, opportunity_count),
                "volatility_opportunities_allowed": volatility_opportunities_allowed,
                "volatility_opportunities_blocked": (
                    volatility_opportunity_count - volatility_opportunities_allowed
                ),
                "volatility_precision": _safe_ratio(
                    volatility_opportunities_allowed, len(candidates)
                ),
                "volatility_recall": _safe_ratio(
                    volatility_opportunities_allowed,
                    volatility_opportunity_count,
                ),
                "settlement_opportunities_allowed": settlement_opportunities_allowed,
                "settlement_opportunities_blocked": (
                    settlement_opportunity_count - settlement_opportunities_allowed
                ),
                "settlement_precision": _safe_ratio(
                    settlement_opportunities_allowed, len(candidates)
                ),
                "settlement_recall": _safe_ratio(
                    settlement_opportunities_allowed,
                    settlement_opportunity_count,
                ),
            }
        )
    return sweep


def _counterfactual_candidate(
    row: OpportunityRow,
    *,
    edge_threshold: float,
    min_entry_price: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
    no_new_entry_before_expiry_seconds: float,
) -> bool:
    if row.seconds_to_expiry_at_decision < no_new_entry_before_expiry_seconds:
        return False
    if row.seconds_to_expiry_at_decision < min_seconds_to_expiry:
        return False
    if row.seconds_to_expiry_at_decision > max_seconds_to_expiry:
        return False
    if row.entry_worst_price is None or row.entry_worst_price < min_entry_price:
        return False
    return row.edge >= edge_threshold


def _parse_thresholds(raw: str) -> list[float]:
    thresholds: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    return thresholds


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _best_bid(levels: list[dict[str, Any]]) -> float | None:
    prices = [
        price
        for level in levels
        if (price := _optional_float(level.get("price"))) is not None
    ]
    return max(prices) if prices else None


def _best_ask(levels: list[dict[str, Any]]) -> float | None:
    prices = [
        price
        for level in levels
        if (price := _optional_float(level.get("price"))) is not None
    ]
    return min(prices) if prices else None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _label(value: bool | None) -> str:
    if value is True:
        return "win"
    if value is False:
        return "loss"
    return "-"


if __name__ == "__main__":
    raise SystemExit(main())
