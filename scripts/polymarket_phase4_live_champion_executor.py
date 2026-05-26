#!/usr/bin/env python3
"""Bounded Phase 4 executor for real champion-signal dry-runs.

This script intentionally keeps the blast radius small:
- consumes live prediction_events from DuckDB,
- re-checks the current CLOB book before every entry,
- uses FOK orders only, so it should not leave resting orders,
- caps max entry spend, concurrent positions, total entries, and realized loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from bigan.execution.position_manager import PositionManager


@dataclass(frozen=True, slots=True)
class SignalEvent:
    event_id: str
    ts: int
    created_at: int
    prob_up_15m: float
    canonical_symbol: str
    token_id: str
    outcome_side: str
    round_slug: str
    round_end_ts: int
    market_implied_prob: float
    token_probability: float
    edge: float
    bridged_at: int = 0


@dataclass(slots=True)
class LivePosition:
    event_id: str
    round_slug: str
    side: str
    token_id: str
    entry_price: float
    fill_price: float
    size: float
    order_id: str
    opened_at: int
    entry_signal_event_id: str
    entry_signal_ts: int
    entry_signal_created_at: int
    entry_signal_bridged_at: int
    entry_order_posted_at: int


@dataclass(slots=True)
class RoundLifecycleState:
    """In-memory execution state for one bounded Phase 4 run."""

    processed_event_ids: set[str] = field(default_factory=set)
    attempted_entry_event_ids: set[str] = field(default_factory=set)
    filled_rounds: set[str] = field(default_factory=set)
    closed_rounds: set[str] = field(default_factory=set)
    open_positions: dict[str, LivePosition] = field(default_factory=dict)

    def mark_event_seen(self, event_id: str) -> bool:
        """Return false when an event was already processed."""

        if not event_id:
            return True
        if event_id in self.processed_event_ids:
            return False
        self.processed_event_ids.add(event_id)
        return True

    def mark_entry_attempted(self, event_id: str) -> None:
        if event_id:
            self.attempted_entry_event_ids.add(event_id)

    def mark_entry_result(self, event: SignalEvent, position: LivePosition | None) -> None:
        """Only confirmed fills lock a round."""

        if position is None:
            return
        self.filled_rounds.add(event.round_slug)
        self.open_positions[event.round_slug] = position

    def mark_position_closed(self, round_slug: str) -> None:
        self.open_positions.pop(round_slug, None)
        self.closed_rounds.add(round_slug)


class OrderBookUnavailable(RuntimeError):
    """Raised when the CLOB no longer exposes an orderbook for a token."""

    def __init__(self, token_id: str, exc: BaseException) -> None:
        self.token_id = token_id
        self.error_type = type(exc).__name__
        self.error = str(exc)
        super().__init__(f"orderbook unavailable for token_id={token_id}: {self.error}")

    def to_log_payload(self) -> dict[str, str]:
        return {
            "token_id": self.token_id,
            "error_type": self.error_type,
            "error": self.error,
        }


STOP_REQUESTED = False


def main() -> int:
    args = _parse_args()
    _install_signal_handlers()
    log_path = Path(args.log_path)
    summary_path = Path(args.summary_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _now_ms()
    client = _build_clob_client()
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(client, heartbeat_stop, log_path),
        daemon=True,
    )
    heartbeat_thread.start()
    position_manager = PositionManager(args.monitoring_db_path)

    lifecycle = RoundLifecycleState()
    entries_attempted = 0
    entries_filled = 0
    closes_filled = 0
    realized_pnl = 0.0
    skipped: dict[str, int] = {}
    errors = 0

    signal_jsonl_path = Path(args.signal_jsonl_path) if args.signal_jsonl_path else None
    cursor_created_at = 0
    cursor_event_id = ""
    cursor_line_number = 0
    if signal_jsonl_path is None:
        cursor_created_at, cursor_event_id = _latest_cursor(
            args.monitoring_db_path,
            args.model_version,
        )
        cursor_payload: dict[str, Any] = {
            "created_at": cursor_created_at,
            "event_id": cursor_event_id,
        }
    else:
        cursor_line_number = _latest_signal_jsonl_cursor(
            signal_jsonl_path,
            start=args.signal_jsonl_start,
        )
        cursor_payload = {
            "line_number": cursor_line_number,
            "signal_jsonl_path": str(signal_jsonl_path),
            "signal_jsonl_start": args.signal_jsonl_start,
        }
    _log(
        log_path,
        "phase4_started",
        config={
            "model_version": args.model_version,
            "signal_source": "jsonl" if signal_jsonl_path is not None else "duckdb",
            "signal_jsonl_path": str(signal_jsonl_path) if signal_jsonl_path is not None else None,
            "edge_threshold": args.edge_threshold,
            "exit_edge_threshold": args.exit_edge_threshold,
            "opposite_exit_edge_threshold": args.opposite_exit_edge_threshold,
            "opposite_exit_min_seconds_to_expiry": args.opposite_exit_min_seconds_to_expiry,
            "max_rounds": args.max_rounds,
            "max_position_size_usdc": args.max_position_size_usdc,
            "daily_loss_limit_usdc": args.daily_loss_limit_usdc,
            "max_concurrent_positions": args.max_concurrent_positions,
            "min_seconds_to_expiry": args.min_seconds_to_expiry,
            "max_seconds_to_expiry": args.max_seconds_to_expiry,
            "poll_seconds": args.poll_seconds,
            "max_runtime_minutes": args.max_runtime_minutes,
        },
        cursor=cursor_payload,
    )

    try:
        while not STOP_REQUESTED:
            now_ms = _now_ms()
            if (now_ms - started_at) >= args.max_runtime_minutes * 60_000:
                _log(log_path, "stop_max_runtime")
                break
            if entries_filled >= args.max_rounds and not lifecycle.open_positions:
                _log(log_path, "stop_max_rounds_closed")
                break
            if realized_pnl <= -args.daily_loss_limit_usdc:
                _log(log_path, "stop_daily_loss_limit", realized_pnl=realized_pnl)
                break

            try:
                if signal_jsonl_path is None:
                    events = _read_events_after(
                        args.monitoring_db_path,
                        model_version=args.model_version,
                        after_created_at=cursor_created_at,
                        after_event_id=cursor_event_id,
                        limit=args.event_limit,
                    )
                    if events:
                        cursor_created_at = events[-1].created_at
                        cursor_event_id = events[-1].event_id
                else:
                    events, cursor_line_number = _read_signal_jsonl_after(
                        signal_jsonl_path,
                        after_line_number=cursor_line_number,
                        model_version=args.model_version,
                        limit=args.event_limit,
                    )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                _log(log_path, "db_read_error", error=str(exc), error_type=type(exc).__name__)
                time.sleep(args.poll_seconds)
                continue

            if events:
                received_at = _now_ms()
                _log(
                    log_path,
                    "signal_batch_received",
                    source="jsonl" if signal_jsonl_path is not None else "duckdb",
                    count=len(events),
                    cursor_line_number=cursor_line_number if signal_jsonl_path is not None else None,
                    cursor_created_at=cursor_created_at if signal_jsonl_path is None else None,
                    cursor_event_id=cursor_event_id if signal_jsonl_path is None else None,
                    signals=[
                        {
                            "event_id": event.event_id,
                            "round_slug": event.round_slug,
                            "side": event.outcome_side,
                            "edge": event.edge,
                            "latency_ms": _signal_latency_ms(event, received_at),
                            "timestamps": _signal_timestamps(event),
                        }
                        for event in events
                    ],
                )

            for event in events:
                if not lifecycle.mark_event_seen(event.event_id):
                    _bump(skipped, "duplicate_event_id")
                    continue

                if event.round_slug in lifecycle.open_positions:
                    position = lifecycle.open_positions[event.round_slug]
                    if event.outcome_side == position.side:
                        maybe_pnl = _maybe_exit(
                            client=client,
                            position_manager=position_manager,
                            position=position,
                            signal=event,
                            log_path=log_path,
                            exit_edge_threshold=args.exit_edge_threshold,
                            profit_target=args.profit_target,
                            sell_slippage=args.sell_slippage,
                        )
                        if maybe_pnl is not None:
                            realized_pnl += maybe_pnl
                            closes_filled += 1
                            lifecycle.mark_position_closed(event.round_slug)
                            if realized_pnl <= -args.daily_loss_limit_usdc:
                                _log(log_path, "daily_loss_limit_reached", realized_pnl=realized_pnl)
                                break
                    else:
                        maybe_pnl = _maybe_exit_opposite_correction(
                            client=client,
                            position_manager=position_manager,
                            position=position,
                            signal=event,
                            log_path=log_path,
                            opposite_exit_edge_threshold=args.opposite_exit_edge_threshold,
                            opposite_exit_min_seconds_to_expiry=args.opposite_exit_min_seconds_to_expiry,
                            sell_slippage=args.sell_slippage,
                        )
                        if maybe_pnl is not None:
                            realized_pnl += maybe_pnl
                            closes_filled += 1
                            lifecycle.mark_position_closed(event.round_slug)
                            if realized_pnl <= -args.daily_loss_limit_usdc:
                                _log(log_path, "daily_loss_limit_reached", realized_pnl=realized_pnl)
                                break
                    continue

                if entries_filled >= args.max_rounds:
                    _bump(skipped, "max_rounds")
                    continue
                if len(lifecycle.open_positions) >= args.max_concurrent_positions:
                    _bump(skipped, "max_concurrent_positions")
                    continue
                if event.round_slug in lifecycle.filled_rounds:
                    _bump(skipped, "round_already_filled")
                    continue
                seconds_to_expiry = (event.round_end_ts - now_ms) / 1000
                if seconds_to_expiry < args.min_seconds_to_expiry:
                    _bump(skipped, "near_or_past_expiry")
                    continue
                if seconds_to_expiry > args.max_seconds_to_expiry:
                    _bump(skipped, "too_far_from_expiry")
                    continue
                if event.edge < args.edge_threshold:
                    _bump(skipped, "below_edge_threshold")
                    continue

                entries_attempted += 1
                lifecycle.mark_entry_attempted(event.event_id)
                position = _try_entry(
                    client=client,
                    position_manager=position_manager,
                    signal=event,
                    log_path=log_path,
                    max_position_size_usdc=args.max_position_size_usdc,
                    edge_threshold=args.edge_threshold,
                    buy_slippage=args.buy_slippage,
                )
                lifecycle.mark_entry_result(event, position)
                if position is not None:
                    entries_filled += 1
                if entries_filled >= args.max_rounds:
                    break

            time.sleep(args.poll_seconds)
    finally:
        heartbeat_stop.set()
        shutdown_closed, shutdown_pnl = _close_remaining_positions(
            client=client,
            position_manager=position_manager,
            positions=lifecycle.open_positions,
            log_path=log_path,
            sell_slippage=args.sell_slippage,
        )
        closes_filled += shutdown_closed
        realized_pnl += shutdown_pnl
        summary = {
            "phase": "phase4_real_champion_signal",
            "started_at": _iso(started_at),
            "finished_at": _iso(_now_ms()),
            "status": "PASS" if errors == 0 and entries_filled > 0 else "CHECK",
            "entries_attempted": entries_attempted,
            "entries_filled": entries_filled,
            "closes_filled": closes_filled,
            "realized_pnl_usdc": round(realized_pnl, 8),
            "open_positions_at_shutdown": len(lifecycle.open_positions),
            "processed_event_count": len(lifecycle.processed_event_ids),
            "attempted_entry_event_count": len(lifecycle.attempted_entry_event_ids),
            "filled_round_count": len(lifecycle.filled_rounds),
            "closed_round_count": len(lifecycle.closed_rounds),
            "skipped": skipped,
            "errors": errors,
            "execution_log_path": str(log_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _log(log_path, "phase4_summary", **summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitoring-db-path", default="data/mlops/champion_catalog.duckdb")
    parser.add_argument(
        "--signal-jsonl-path",
        default="",
        help=(
            "Optional append-only SignalEvent JSONL queue. When set, the executor "
            "reads local/bridged signal rows from this file instead of scanning DuckDB."
        ),
    )
    parser.add_argument(
        "--signal-jsonl-start",
        choices=("tail", "beginning"),
        default="tail",
        help="Where to start reading --signal-jsonl-path on startup.",
    )
    parser.add_argument("--model-version", default="xgboost-v4")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-position-size-usdc", type=float, default=1.0)
    parser.add_argument("--daily-loss-limit-usdc", type=float, default=3.0)
    parser.add_argument("--max-concurrent-positions", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.45)
    parser.add_argument("--exit-edge-threshold", type=float, default=0.10)
    parser.add_argument("--opposite-exit-edge-threshold", type=float, default=0.45)
    parser.add_argument("--opposite-exit-min-seconds-to-expiry", type=float, default=120.0)
    parser.add_argument("--profit-target", type=float, default=0.15)
    parser.add_argument("--min-seconds-to-expiry", type=float, default=180.0)
    parser.add_argument("--max-seconds-to-expiry", type=float, default=1200.0)
    parser.add_argument("--buy-slippage", type=float, default=0.02)
    parser.add_argument("--sell-slippage", type=float, default=0.02)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--max-runtime-minutes", type=float, default=240.0)
    parser.add_argument("--log-path", default="logs/remote_dry_run_phase4_real_champion.jsonl")
    parser.add_argument("--summary-path", default="logs/remote_dry_run_phase4_real_champion_summary.json")
    return parser.parse_args()


def _install_signal_handlers() -> None:
    def _handler(_signum: int, _frame: Any) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _build_clob_client() -> Any:
    from py_clob_client_v2 import ClobClient, SignatureTypeV2

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("POLYMARKET_PRIVATE_KEY is required")
    signature_type_name = os.getenv("POLYMARKET_SIGNATURE_TYPE", "POLY_PROXY")
    signature_type = getattr(SignatureTypeV2, signature_type_name)
    kwargs: dict[str, Any] = {
        "host": os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com"),
        "key": private_key,
        "chain_id": int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
        "signature_type": signature_type,
    }
    funder = os.getenv("POLYMARKET_FUNDER")
    if funder:
        kwargs["funder"] = funder
    client = ClobClient(**kwargs)
    auth_mode = os.getenv("POLYMARKET_CLOB_AUTH_MODE", "derive").strip().lower()
    if auth_mode in {"", "derive", "derived"}:
        client.set_api_creds(client.create_or_derive_api_key())
    elif auth_mode in {"env", "static"}:
        from py_clob_client_v2.clob_types import ApiCreds

        client.set_api_creds(
            ApiCreds(
                api_key=os.environ["POLYMARKET_API_KEY"],
                api_secret=os.environ["POLYMARKET_API_SECRET"],
                api_passphrase=os.environ["POLYMARKET_API_PASSPHRASE"],
            )
        )
    else:
        raise RuntimeError(f"unsupported POLYMARKET_CLOB_AUTH_MODE={auth_mode}")
    return client


def _heartbeat_loop(client: Any, stop: threading.Event, log_path: Path) -> None:
    heartbeat_id = ""
    while not stop.wait(5):
        try:
            response = client.post_heartbeat(heartbeat_id)
            if isinstance(response, dict):
                heartbeat_id = str(response.get("heartbeat_id") or heartbeat_id)
            _log(log_path, "heartbeat_ok")
        except Exception as exc:  # noqa: BLE001 - best-effort keepalive.
            _log(log_path, "heartbeat_error", error=str(exc), error_type=type(exc).__name__)


def _latest_cursor(db_path: str, model_version: str) -> tuple[int, str]:
    for _ in range(10):
        try:
            with duckdb.connect(db_path, read_only=True) as conn:
                row = conn.execute(
                    """
                    SELECT created_at, event_id
                    FROM prediction_events
                    WHERE model_version = ?
                    ORDER BY created_at DESC, event_id DESC
                    LIMIT 1
                    """,
                    [model_version],
                ).fetchone()
            if row is None:
                return 0, ""
            return int(row[0]), str(row[1])
        except Exception:
            time.sleep(0.5)
    return 0, ""


def _read_events_after(
    db_path: str,
    *,
    model_version: str,
    after_created_at: int,
    after_event_id: str,
    limit: int,
) -> list[SignalEvent]:
    with duckdb.connect(db_path, read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT event_id, ts, created_at, prob_up_15m, feature_snapshot_json
            FROM prediction_events
            WHERE model_version = ?
              AND (
                    created_at > ?
                 OR (created_at = ? AND event_id > ?)
              )
            ORDER BY created_at ASC, event_id ASC
            LIMIT ?
            """,
            [model_version, after_created_at, after_created_at, after_event_id, limit],
        ).fetchall()
    events: list[SignalEvent] = []
    for row in rows:
        parsed = _event_from_row(row)
        if parsed is not None:
            events.append(parsed)
    return _best_event_per_round(events)


def _latest_signal_jsonl_cursor(path: Path, *, start: str) -> int:
    if start == "beginning" or not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _read_signal_jsonl_after(
    path: Path,
    *,
    after_line_number: int,
    model_version: str,
    limit: int,
) -> tuple[list[SignalEvent], int]:
    if not path.exists():
        return [], after_line_number
    events: list[SignalEvent] = []
    last_line_number = after_line_number
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if line_number <= after_line_number:
                continue
            last_line_number = line_number
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = _event_from_signal_payload(payload, model_version=model_version)
            if event is not None:
                events.append(event)
            if len(events) >= limit:
                break
    return _best_event_per_round(events), last_line_number


def _event_from_signal_payload(payload: Any, *, model_version: str) -> SignalEvent | None:
    if not isinstance(payload, dict):
        return None
    payload_model_version = str(payload.get("model_version") or model_version)
    if payload_model_version != model_version:
        return None
    round_slug = str(payload.get("round_slug") or "")
    side = str(payload.get("outcome_side") or "").upper()
    token_id = str(payload.get("token_id") or payload.get("source_symbol") or "")
    round_end_ts = _optional_int(payload.get("round_end_ts")) or _round_end_ts(round_slug)
    market = _optional_float(payload.get("market_implied_prob"))
    token_probability = _optional_float(payload.get("token_probability"))
    prob_up_15m = _optional_float(payload.get("prob_up_15m"))
    if (
        not round_slug
        or side not in {"UP", "DOWN"}
        or not token_id
        or round_end_ts is None
        or market is None
        or token_probability is None
        or prob_up_15m is None
    ):
        return None
    edge = _optional_float(payload.get("edge"))
    if edge is None:
        edge = token_probability - market
    canonical_symbol = str(payload.get("canonical_symbol") or f"BTC-15M:{round_slug}:{side}")
    return SignalEvent(
        event_id=str(payload.get("event_id") or ""),
        ts=int(_optional_int(payload.get("ts")) or 0),
        created_at=int(_optional_int(payload.get("created_at")) or 0),
        prob_up_15m=float(prob_up_15m),
        canonical_symbol=canonical_symbol,
        token_id=token_id,
        outcome_side=side,
        round_slug=round_slug,
        round_end_ts=int(round_end_ts),
        market_implied_prob=float(market),
        token_probability=float(token_probability),
        edge=float(edge),
        bridged_at=int(_optional_int(payload.get("bridged_at")) or 0),
    )


def _event_from_row(row: tuple[Any, ...]) -> SignalEvent | None:
    event_id, ts, created_at, prob_up_15m, snapshot_json = row
    snapshot = json.loads(snapshot_json)
    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    parts = canonical_symbol.split(":")
    if len(parts) < 3:
        return None
    family, round_slug, side = parts[0], parts[-2], parts[-1].upper()
    if family != "BTC-15M" or side not in {"UP", "DOWN"}:
        return None
    token_id = str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
    market = _optional_float(snapshot.get("market_implied_prob"))
    if not token_id or market is None:
        return None
    prob = float(prob_up_15m)
    token_probability = 1.0 - prob if side == "DOWN" else prob
    round_end_ts = _round_end_ts(round_slug)
    if round_end_ts is None:
        return None
    return SignalEvent(
        event_id=str(event_id),
        ts=int(ts),
        created_at=int(created_at),
        prob_up_15m=prob,
        canonical_symbol=canonical_symbol,
        token_id=token_id,
        outcome_side=side,
        round_slug=round_slug,
        round_end_ts=round_end_ts,
        market_implied_prob=market,
        token_probability=token_probability,
        edge=token_probability - market,
        bridged_at=0,
    )


def _best_event_per_round(events: list[SignalEvent]) -> list[SignalEvent]:
    best: dict[str, SignalEvent] = {}
    for event in events:
        previous = best.get(event.round_slug)
        if previous is None or event.edge > previous.edge:
            best[event.round_slug] = event
    return sorted(best.values(), key=lambda item: (item.created_at, item.event_id))


def _try_entry(
    *,
    client: Any,
    position_manager: PositionManager,
    signal: SignalEvent,
    log_path: Path,
    max_position_size_usdc: float,
    edge_threshold: float,
    buy_slippage: float,
) -> LivePosition | None:
    from py_clob_client_v2 import MarketOrderArgs, OrderType
    from py_clob_client_v2.clob_types import PartialCreateOrderOptions
    from py_clob_client_v2.order_builder.constants import BUY

    try:
        bid, ask = _best_bid_ask(client, signal.token_id)
    except OrderBookUnavailable as exc:
        _log(
            log_path,
            "entry_skipped",
            reason="orderbook_unavailable",
            signal=asdict(signal),
            **exc.to_log_payload(),
        )
        return None
    if ask is None:
        _log(log_path, "entry_skipped", reason="missing_ask", signal=asdict(signal))
        return None
    tick_size = client.get_tick_size(signal.token_id)
    neg_risk = client.get_neg_risk(signal.token_id)
    worst_price = min(0.99, _round_price(float(ask) + buy_slippage, tick_size))
    fresh_edge_at_worst = signal.token_probability - worst_price
    if fresh_edge_at_worst < edge_threshold:
        _log(
            log_path,
            "entry_skipped",
            reason="fresh_edge_below_threshold",
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            fresh_edge_at_worst=fresh_edge_at_worst,
        )
        return None
    order = client.create_market_order(
        order_args=MarketOrderArgs(
            token_id=signal.token_id,
            side=BUY,
            amount=max_position_size_usdc,
            price=worst_price,
        ),
        options=PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk),
    )
    order_submit_started_at = _now_ms()
    try:
        response = client.post_order(order, OrderType.FOK)
    except Exception as exc:  # noqa: BLE001 - Polymarket returns FOK kills as API exceptions.
        order_failed_at = _now_ms()
        _log(
            log_path,
            "entry_order_post_failed",
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            error=str(exc),
            error_type=type(exc).__name__,
            order_submit_latency_ms=order_failed_at - order_submit_started_at,
            latency_ms={
                **_signal_latency_ms(signal, order_failed_at),
                "signal_created_to_order_failure_ms": _delta_ms(signal.created_at, order_failed_at),
                "bridge_to_order_failure_ms": _delta_ms(signal.bridged_at, order_failed_at),
            },
            timestamps={
                **_signal_timestamps(signal),
                "order_submitted_at": _iso(order_submit_started_at),
                "order_failed_at": _iso(order_failed_at),
            },
        )
        return None
    order_posted_at = _now_ms()
    order_id = str(response.get("orderID") or "")
    order_matched = bool(response.get("success")) and response.get("status") == "matched" and bool(order_id)
    _log(
        log_path,
        "entry_order_posted",
        signal=asdict(signal),
        bid=bid,
        ask=ask,
        worst_price=worst_price,
        response=response,
        order_submit_latency_ms=order_posted_at - order_submit_started_at,
        latency_ms={
            **_signal_latency_ms(signal, order_posted_at),
            "signal_created_to_order_success_ms": (
                _delta_ms(signal.created_at, order_posted_at) if order_matched else None
            ),
            "bridge_to_order_success_ms": (
                _delta_ms(signal.bridged_at, order_posted_at) if order_matched else None
            ),
        },
        timestamps={
            **_signal_timestamps(signal),
            "order_submitted_at": _iso(order_submit_started_at),
            "order_posted_at": _iso(order_posted_at),
        },
    )
    if not order_matched:
        return None
    fill = _fill_for_order(client, order_id, wanted_side="BUY")
    fill_checked_at = _now_ms()
    fill_price = _optional_float(fill.get("price")) or float(ask)
    fill_size = _optional_float(fill.get("size")) or _optional_float(response.get("takingAmount")) or 0.0
    if fill_size <= 0:
        _log(
            log_path,
            "entry_fill_missing_or_unconfirmed",
            order_id=order_id,
            response=response,
            fill=fill,
            latency_ms={
                **_signal_latency_ms(signal, fill_checked_at),
                "order_success_to_fill_check_ms": fill_checked_at - order_posted_at,
            },
            timestamps={
                **_signal_timestamps(signal),
                "order_posted_at": _iso(order_posted_at),
                "fill_checked_at": _iso(fill_checked_at),
            },
        )
        return None
    event_id = f"phase4-{signal.round_slug}-{signal.outcome_side}-{order_id[-8:]}"
    position_manager.open_position(
        event_id=event_id,
        symbol=signal.canonical_symbol,
        side=signal.outcome_side,
        entry_price=fill_price,
        fill_price=fill_price,
        size=fill_size,
        order_id=order_id,
    )
    position = LivePosition(
        event_id=event_id,
        round_slug=signal.round_slug,
        side=signal.outcome_side,
        token_id=signal.token_id,
        entry_price=fill_price,
        fill_price=fill_price,
        size=fill_size,
        order_id=order_id,
        opened_at=_now_ms(),
        entry_signal_event_id=signal.event_id,
        entry_signal_ts=signal.ts,
        entry_signal_created_at=signal.created_at,
        entry_signal_bridged_at=signal.bridged_at,
        entry_order_posted_at=order_posted_at,
    )
    _log(
        log_path,
        "entry_filled",
        position=asdict(position),
        fill=fill,
        latency_ms={
            **_signal_latency_ms(signal, position.opened_at),
            "signal_created_to_fill_confirmed_ms": _delta_ms(signal.created_at, position.opened_at),
            "bridge_to_fill_confirmed_ms": _delta_ms(signal.bridged_at, position.opened_at),
            "order_success_to_fill_confirmed_ms": position.opened_at - order_posted_at,
        },
        timestamps={
            **_signal_timestamps(signal),
            "order_posted_at": _iso(order_posted_at),
            "fill_confirmed_at": _iso(position.opened_at),
        },
    )
    return position


def _maybe_exit(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    signal: SignalEvent,
    log_path: Path,
    exit_edge_threshold: float,
    profit_target: float,
    sell_slippage: float,
) -> float | None:
    try:
        bid, _ask = _best_bid_ask(client, position.token_id)
    except OrderBookUnavailable as exc:
        _log(
            log_path,
            "exit_hold",
            reason="orderbook_unavailable",
            position=asdict(position),
            signal=asdict(signal),
            **exc.to_log_payload(),
        )
        return None
    if bid is None:
        _log(log_path, "exit_hold", reason="missing_bid", position=asdict(position), signal=asdict(signal))
        return None
    unrealized = float(bid) - position.fill_price
    seconds_to_expiry = (signal.round_end_ts - _now_ms()) / 1000
    should_exit = (
        signal.edge <= exit_edge_threshold
        or unrealized >= profit_target
        or seconds_to_expiry <= 60
    )
    if not should_exit:
        try:
            position_manager.update_price(position.event_id, float(bid))
        except Exception as exc:  # noqa: BLE001
            _log(log_path, "position_mark_error", error=str(exc), event_id=position.event_id)
        return None
    return _sell_position(
        client=client,
        position_manager=position_manager,
        position=position,
        log_path=log_path,
        bid=float(bid),
        sell_slippage=sell_slippage,
        reason="exit_signal",
        signal=signal,
    )


def _maybe_exit_opposite_correction(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    signal: SignalEvent,
    log_path: Path,
    opposite_exit_edge_threshold: float,
    opposite_exit_min_seconds_to_expiry: float,
    sell_slippage: float,
) -> float | None:
    """Exit an open position when the opposite side becomes strongly favored."""

    seconds_to_expiry = (signal.round_end_ts - _now_ms()) / 1000
    if seconds_to_expiry < opposite_exit_min_seconds_to_expiry:
        _log(
            log_path,
            "opposite_exit_hold",
            reason="insufficient_time_remaining",
            position=asdict(position),
            signal=asdict(signal),
            old_side=position.side,
            new_side=signal.outcome_side,
            edge=signal.edge,
            seconds_to_expiry=seconds_to_expiry,
            opposite_exit_min_seconds_to_expiry=opposite_exit_min_seconds_to_expiry,
        )
        return None
    if signal.edge < opposite_exit_edge_threshold:
        _log(
            log_path,
            "opposite_exit_hold",
            reason="opposite_edge_below_threshold",
            position=asdict(position),
            signal=asdict(signal),
            old_side=position.side,
            new_side=signal.outcome_side,
            edge=signal.edge,
            seconds_to_expiry=seconds_to_expiry,
            opposite_exit_edge_threshold=opposite_exit_edge_threshold,
        )
        return None
    try:
        bid, _ask = _best_bid_ask(client, position.token_id)
    except OrderBookUnavailable as exc:
        _log(
            log_path,
            "opposite_exit_hold",
            reason="orderbook_unavailable",
            position=asdict(position),
            signal=asdict(signal),
            old_side=position.side,
            new_side=signal.outcome_side,
            edge=signal.edge,
            seconds_to_expiry=seconds_to_expiry,
            **exc.to_log_payload(),
        )
        return None
    if bid is None:
        _log(
            log_path,
            "opposite_exit_hold",
            reason="missing_bid",
            position=asdict(position),
            signal=asdict(signal),
            old_side=position.side,
            new_side=signal.outcome_side,
            edge=signal.edge,
            seconds_to_expiry=seconds_to_expiry,
        )
        return None
    return _sell_position(
        client=client,
        position_manager=position_manager,
        position=position,
        log_path=log_path,
        bid=float(bid),
        sell_slippage=sell_slippage,
        reason="opposite_side_exit_correction",
        signal=signal,
    )


def _close_remaining_positions(
    *,
    client: Any,
    position_manager: PositionManager,
    positions: dict[str, LivePosition],
    log_path: Path,
    sell_slippage: float,
) -> tuple[int, float]:
    closed_count = 0
    realized_pnl = 0.0
    for round_slug, position in list(positions.items()):
        try:
            bid, _ask = _best_bid_ask(client, position.token_id)
        except OrderBookUnavailable as exc:
            _log(
                log_path,
                "shutdown_close_skipped",
                reason="orderbook_unavailable",
                position=asdict(position),
                **exc.to_log_payload(),
            )
            continue
        if bid is None:
            _log(log_path, "shutdown_close_skipped", reason="missing_bid", position=asdict(position))
            continue
        pnl = _sell_position(
            client=client,
            position_manager=position_manager,
            position=position,
            log_path=log_path,
            bid=float(bid),
            sell_slippage=sell_slippage,
            reason="shutdown",
            signal=None,
        )
        if pnl is not None:
            closed_count += 1
            realized_pnl += pnl
            del positions[round_slug]
    return closed_count, realized_pnl


def _sell_position(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    log_path: Path,
    bid: float,
    sell_slippage: float,
    reason: str,
    signal: SignalEvent | None,
) -> float | None:
    from py_clob_client_v2 import MarketOrderArgs, OrderType
    from py_clob_client_v2.clob_types import PartialCreateOrderOptions
    from py_clob_client_v2.order_builder.constants import SELL

    tick_size = client.get_tick_size(position.token_id)
    neg_risk = client.get_neg_risk(position.token_id)
    worst_price = max(0.01, _round_price(bid - sell_slippage, tick_size))
    sell_size = _round_sell_size(position.size)
    dust_amount = max(0.0, float(position.size) - sell_size)
    dust_value_usd = dust_amount * float(bid)
    if sell_size <= 0:
        _log(
            log_path,
            "exit_skipped",
            reason="sell_size_too_small",
            position=asdict(position),
            sell_size=sell_size,
            dust_amount=dust_amount,
            dust_value_usd=dust_value_usd,
        )
        return None
    order = client.create_market_order(
        order_args=MarketOrderArgs(
            token_id=position.token_id,
            side=SELL,
            amount=sell_size,
            price=worst_price,
        ),
        options=PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk),
    )
    response = client.post_order(order, OrderType.FOK)
    order_id = str(response.get("orderID") or "")
    _log(
        log_path,
        "exit_order_posted",
        reason=reason,
        position=asdict(position),
        signal=None if signal is None else asdict(signal),
        bid=bid,
        worst_price=worst_price,
        sell_size=sell_size,
        dust_amount=dust_amount,
        dust_value_usd=dust_value_usd,
        response=response,
    )
    if not response.get("success") or response.get("status") != "matched" or not order_id:
        return None
    fill = _fill_for_order(client, order_id, wanted_side="SELL")
    fill_price = _optional_float(fill.get("price")) or bid
    if not fill:
        _log(
            log_path,
            "exit_fill_missing_or_unconfirmed",
            reason=reason,
            position=asdict(position),
            signal=None if signal is None else asdict(signal),
            sell_order_id=order_id,
            response=response,
        )
        return None
    closed = position_manager.close_position(position.event_id, fill_price)
    pnl = float(closed.realized_pnl or 0.0)
    _log(
        log_path,
        "exit_filled",
        reason=reason,
        position=asdict(position),
        sell_order_id=order_id,
        fill=fill,
        exit_price=fill_price,
        realized_pnl=pnl,
        realized_pnl_source="position_manager_fill_price",
        account_cashflow_reconciliation_required=True,
        dust_amount=dust_amount,
        dust_value_usd=dust_value_usd,
    )
    return pnl


def _fill_for_order(client: Any, order_id: str, *, wanted_side: str) -> dict[str, Any]:
    time.sleep(2)
    try:
        trades = client.get_trades()
    except Exception:
        return {}
    for trade in trades:
        if (
            str(trade.get("taker_order_id") or "") == order_id
            and str(trade.get("side") or "").upper() == wanted_side
            and _trade_is_confirmed(trade)
        ):
            return dict(trade)
    return {}


def _trade_is_confirmed(trade: dict[str, Any]) -> bool:
    return str(trade.get("status") or "").upper() in {"MINED", "CONFIRMED"}


def _best_bid_ask(client: Any, token_id: str) -> tuple[float | None, float | None]:
    try:
        book = client.get_order_book(token_id)
    except Exception as exc:  # noqa: BLE001
        raise OrderBookUnavailable(token_id, exc) from exc
    raw = book if isinstance(book, dict) else getattr(book, "__dict__", {})
    bids = raw.get("bids") or []
    asks = raw.get("asks") or []
    bid = _best_price(bids, want_max=True)
    ask = _best_price(asks, want_max=False)
    return bid, ask


def _best_price(levels: Any, *, want_max: bool) -> float | None:
    prices: list[float] = []
    for level in levels if isinstance(levels, list) else []:
        value = level.get("price") if isinstance(level, dict) else getattr(level, "price", None)
        parsed = _optional_float(value)
        if parsed is not None:
            prices.append(parsed)
    if not prices:
        return None
    return max(prices) if want_max else min(prices)


def _round_price(price: float, tick_size: Any) -> float:
    tick = float(tick_size)
    if tick <= 0:
        return round(price, 4)
    return round(round(price / tick) * tick, 4)


def _round_sell_size(size: float) -> float:
    return math.floor(float(size) * 1000) / 1000


def _round_end_ts(round_slug: str) -> int | None:
    try:
        start_ts = int(round_slug.rsplit("-", 1)[-1]) * 1000
    except ValueError:
        return None
    if "updown-15m-" in round_slug:
        return start_ts + 15 * 60_000
    return start_ts


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _signal_latency_ms(signal: SignalEvent, at_ms: int) -> dict[str, int | None]:
    return {
        "event_ts_to_at_ms": _delta_ms(signal.ts, at_ms),
        "signal_created_to_at_ms": _delta_ms(signal.created_at, at_ms),
        "bridge_to_at_ms": _delta_ms(signal.bridged_at, at_ms),
    }


def _signal_timestamps(signal: SignalEvent) -> dict[str, str | None]:
    return {
        "event_ts": _iso(signal.ts) if signal.ts > 0 else None,
        "signal_created_at": _iso(signal.created_at) if signal.created_at > 0 else None,
        "signal_bridged_at": _iso(signal.bridged_at) if signal.bridged_at > 0 else None,
    }


def _delta_ms(start_ms: int, end_ms: int) -> int | None:
    if start_ms <= 0:
        return None
    return max(0, end_ms - start_ms)


def _log(log_path: Path, event: str, **payload: Any) -> None:
    row = {"event": event, "ts": _iso(_now_ms()), **payload}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
