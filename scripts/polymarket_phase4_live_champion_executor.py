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
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import duckdb

from bigan.execution.cash_legs import (
    leg_from_clob_fill,
    read_execution_cash_legs,
    record_execution_cash_legs,
)
from bigan.execution.db import connect_mlops_db
from bigan.execution.phase4_policy import (
    DEFAULT_MIN_ENTRY_PRICE,
    DEFAULT_NEAR_MIN_FRESH_EDGE_THRESHOLD,
    DEFAULT_NEAR_MIN_PRICE_BAND,
    DEFAULT_NEAR_MIN_SECONDS_TO_EXPIRY,
    DEFAULT_SETTLEMENT_EDGE_THRESHOLD,
    DEFAULT_SOFT_FORCE_EXIT_MIN_BID,
    DEFAULT_VOLATILITY_MIN_ENTRY_PRICE,
    DEFAULT_VOLATILITY_MIN_ORDER_SIZE_USDC,
    DEFAULT_VOLATILITY_MIN_SECONDS_TO_EXPIRY,
    DEFAULT_VOLATILITY_ROUND_BANKROLL_USDC,
    DEFAULT_VOLATILITY_ROUND_TRIP_COST,
    DEFAULT_VOLATILITY_SAFETY_MARGIN,
    DEFAULT_VOLATILITY_SCORE_THRESHOLD,
    Phase4EntryPolicy,
    VolatilitySleeveBudget,
    entry_price_skip_reason,
    evaluate_entry_gates,
    phase4_lifecycle_complete,
    phase4_summary_status,
    settlement_cost_edge_skip_reason,
    soft_force_exit_deferred,
)
from bigan.execution.position_manager import PositionManager
from bigan.execution.v6_gate import (
    V6JointGateConfig,
    build_v6_signal_fields,
    is_v6_model_version,
    v6_joint_gate_config_from_model,
    v6_selection_score,
)
from bigan.modeling.families import market_family_from_symbol

DEFAULT_GAMMA_API_BASE = "https://gamma-api.polymarket.com"


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
    opposite_token_id: str = ""
    p_up: float | None = None
    p_down: float | None = None
    p_neutral: float | None = None
    p_vol_up: float | None = None
    p_vol_down: float | None = None
    v6_joint_side: str | None = None


@dataclass(frozen=True, slots=True)
class SignalReadBatch:
    events: list[SignalEvent]
    cursor_created_at: int
    cursor_event_id: str
    rows_scanned: int
    rows_filtered: int
    filter_reasons: dict[str, int] = field(default_factory=dict)


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
    lifecycle_state: str = "OPEN"
    exit_attempt_count: int = 0
    last_exit_attempt_at: int = 0
    last_lifecycle_reason: str = ""
    sleeve: str = "settlement"
    paper: bool = False


@dataclass(frozen=True, slots=True)
class SellResult:
    status: str
    realized_pnl: float = 0.0
    account_cash_pnl: float | None = None


@dataclass(frozen=True, slots=True)
class PaperSettlementResolverConfig:
    enabled: bool = True
    gamma_api_base: str = DEFAULT_GAMMA_API_BASE
    request_timeout_seconds: float = 10.0
    max_wait_after_expiry_seconds: float = 180.0


@dataclass(frozen=True, slots=True)
class PaperSettlementResolution:
    result: str | None
    source: str
    market: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class RoundLifecycleState:
    """In-memory execution state for one bounded Phase 4 run."""

    processed_event_ids: set[str] = field(default_factory=set)
    attempted_entry_event_ids: set[str] = field(default_factory=set)
    observed_rounds: list[str] = field(default_factory=list)
    observed_round_set: set[str] = field(default_factory=set)
    filled_rounds: set[str] = field(default_factory=set)
    closed_rounds: set[str] = field(default_factory=set)
    position_event_ids: set[str] = field(default_factory=set)
    open_positions: dict[str, LivePosition] = field(default_factory=dict)
    volatility_filled_count_by_round: dict[str, int] = field(default_factory=dict)
    filled_count_by_sleeve_round_side: dict[str, int] = field(default_factory=dict)

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

    def mark_round_seen(self, round_slug: str, *, max_rounds: int) -> bool:
        """Track capped market windows; return false for new rounds beyond the cap."""

        if not round_slug or round_slug in self.observed_round_set:
            return True
        if max_rounds <= 0 or len(self.observed_rounds) >= max_rounds:
            return False
        self.observed_round_set.add(round_slug)
        self.observed_rounds.append(round_slug)
        return True

    def max_rounds_reached(self, max_rounds: int) -> bool:
        return max_rounds <= 0 or len(self.observed_rounds) >= max_rounds

    def mark_entry_result(self, event: SignalEvent, position: LivePosition | None) -> None:
        """Only confirmed fills lock a round."""

        if position is None:
            return
        if position.sleeve == "settlement":
            self.filled_rounds.add(event.round_slug)
        elif position.sleeve == "volatility":
            self.volatility_filled_count_by_round[event.round_slug] = (
                self.volatility_filled_count_by_round.get(event.round_slug, 0) + 1
            )
        side_key = _sleeve_side_key(position.round_slug, position.sleeve, position.side)
        self.filled_count_by_sleeve_round_side[side_key] = (
            self.filled_count_by_sleeve_round_side.get(side_key, 0) + 1
        )
        self.position_event_ids.add(position.event_id)
        self.open_positions[_position_key(position.round_slug, position.sleeve)] = position

    def mark_position_closed(self, round_slug: str, sleeve: str = "settlement") -> None:
        self.open_positions.pop(_position_key(round_slug, sleeve), None)
        self.closed_rounds.add(round_slug)

    def open_position(self, round_slug: str, sleeve: str) -> LivePosition | None:
        return self.open_positions.get(_position_key(round_slug, sleeve))

    def has_open_sleeve(self, round_slug: str, sleeve: str) -> bool:
        return self.open_position(round_slug, sleeve) is not None

    def filled_count_for_side(self, *, round_slug: str, sleeve: str, side: str) -> int:
        return self.filled_count_by_sleeve_round_side.get(
            _sleeve_side_key(round_slug, sleeve, side),
            0,
        )


def _position_key(round_slug: str, sleeve: str) -> str:
    return round_slug if sleeve == "settlement" else f"{sleeve}:{round_slug}"


def _sleeve_side_key(round_slug: str, sleeve: str, side: str) -> str:
    return f"{sleeve}:{round_slug}:{side.upper()}"


def _sleeve_side_cap_skip_reason(
    lifecycle: RoundLifecycleState,
    *,
    round_slug: str,
    sleeve: str,
    side: str,
    max_filled_per_side_per_round: int,
) -> str | None:
    if max_filled_per_side_per_round <= 0:
        raise ValueError("max_filled_per_side_per_round must be positive")
    filled = lifecycle.filled_count_for_side(round_slug=round_slug, sleeve=sleeve, side=side)
    if filled >= max_filled_per_side_per_round:
        return f"{sleeve}_side_cap"
    return None


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


def _event_family_allowed(event: SignalEvent, allowed_families: frozenset[str]) -> bool:
    """Return True when the signal's market family is permitted to trade.

    An empty ``allowed_families`` set means all families are allowed.
    """

    if not allowed_families:
        return True
    return market_family_from_symbol(event.canonical_symbol) in allowed_families


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    _install_signal_handlers()
    allowed_families = frozenset(
        family.strip().upper() for family in args.market_families.split(",") if family.strip()
    )
    log_path = Path(args.log_path)
    summary_path = Path(args.summary_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _now_ms()
    entry_policy = _entry_policy_from_args(args)
    v6_joint_config = _v6_joint_config_from_args(args)
    paper_settlement_config = PaperSettlementResolverConfig(
        enabled=args.paper and not args.disable_paper_settlement_resolution,
        gamma_api_base=args.paper_settlement_gamma_api_base,
        request_timeout_seconds=args.paper_settlement_timeout_seconds,
        max_wait_after_expiry_seconds=args.paper_settlement_max_wait_after_expiry_seconds,
    )
    volatility_budget = VolatilitySleeveBudget(
        round_cap_usdc=args.volatility_round_bankroll_usdc,
        per_bet_cap_usdc=args.volatility_per_bet_cap_usdc,
        min_order_size_usdc=args.volatility_min_order_size_usdc,
    )
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
    exits_pending_confirmation = 0
    exits_pending_settlement = 0
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
            "market_families": sorted(allowed_families) or None,
            "signal_source": "jsonl" if signal_jsonl_path is not None else "duckdb",
            "signal_jsonl_path": str(signal_jsonl_path) if signal_jsonl_path is not None else None,
            "edge_threshold": args.edge_threshold,
            "settlement_edge_threshold": entry_policy.effective_settlement_edge_threshold,
            "settlement_price_gate_mode": (
                "cost_edge_only" if args.entry_gate_mode == "v6-joint" else "legacy_min_entry"
            ),
            "settlement_min_entry_price": (
                None if args.entry_gate_mode == "v6-joint" else args.min_entry_price
            ),
            "volatility_score_threshold": entry_policy.volatility_score_threshold,
            "volatility_min_entry_price": entry_policy.volatility_min_entry_price,
            "volatility_min_seconds_to_expiry": entry_policy.volatility_min_seconds_to_expiry,
            "volatility_round_trip_cost": entry_policy.volatility_round_trip_cost,
            "volatility_safety_margin": entry_policy.volatility_safety_margin,
            "volatility_round_bankroll_usdc": volatility_budget.round_cap_usdc,
            "volatility_per_bet_cap_usdc": volatility_budget.per_bet_cap_usdc,
            "volatility_min_order_size_usdc": volatility_budget.min_order_size_usdc,
            "enable_volatility_live_entries": entry_policy.enable_volatility_live_entries,
            "enable_volatility_sleeve": args.enable_volatility_sleeve,
            "volatility_live_ordering_enabled": False,
            "volatility_ordering_mode": "paper_only",
            "paper": args.paper,
            "paper_settlement_resolution_enabled": paper_settlement_config.enabled,
            "paper_settlement_gamma_api_base": paper_settlement_config.gamma_api_base,
            "paper_settlement_timeout_seconds": (
                paper_settlement_config.request_timeout_seconds
            ),
            "paper_settlement_max_wait_after_expiry_seconds": (
                paper_settlement_config.max_wait_after_expiry_seconds
            ),
            "entry_gate_mode": args.entry_gate_mode,
            "v6_joint_rule": (
                None if v6_joint_config is None else v6_joint_config.joint_rule()
            ),
            "phase4_v5_role": "diagnostic_and_opportunity_analysis",
            "exit_edge_threshold": args.exit_edge_threshold,
            "opposite_exit_edge_threshold": args.opposite_exit_edge_threshold,
            "opposite_exit_min_seconds_to_expiry": args.opposite_exit_min_seconds_to_expiry,
            "no_new_entry_before_expiry_seconds": args.no_new_entry_before_expiry_seconds,
            "soft_force_exit_before_expiry_seconds": args.soft_force_exit_before_expiry_seconds,
            "hard_force_exit_before_expiry_seconds": args.hard_force_exit_before_expiry_seconds,
            "exit_retry_seconds": args.exit_retry_seconds,
            "exit_order_timeout_seconds": args.exit_order_timeout_seconds,
            "max_exit_attempts_per_position": args.max_exit_attempts_per_position,
            "max_rounds": args.max_rounds,
            "max_position_size_usdc": args.max_position_size_usdc,
            "min_entry_price": args.min_entry_price,
            "max_combined_concurrent_positions": args.max_combined_concurrent_positions,
            "settlement_max_filled_per_side_per_round": (
                args.settlement_max_filled_per_side_per_round
            ),
            "near_min_price_band": args.near_min_price_band,
            "near_min_fresh_edge_threshold": args.near_min_fresh_edge_threshold,
            "near_min_seconds_to_expiry": args.near_min_seconds_to_expiry,
            "soft_force_exit_min_bid": args.soft_force_exit_min_bid,
            "daily_loss_limit_usdc": args.daily_loss_limit_usdc,
            "max_concurrent_positions": args.max_concurrent_positions,
            "min_seconds_to_expiry": args.min_seconds_to_expiry,
            "max_seconds_to_expiry": args.max_seconds_to_expiry,
            "poll_seconds": args.poll_seconds,
            "max_runtime_minutes": args.max_runtime_minutes,
            "continue_after_max_rounds_until_runtime": args.continue_after_max_rounds_until_runtime,
        },
        cursor=cursor_payload,
    )

    try:
        while not STOP_REQUESTED:
            now_ms = _now_ms()
            if (now_ms - started_at) >= args.max_runtime_minutes * 60_000:
                _log(log_path, "stop_max_runtime")
                break
            if (
                lifecycle.max_rounds_reached(args.max_rounds)
                and not lifecycle.open_positions
                and not args.continue_after_max_rounds_until_runtime
            ):
                _log(log_path, "stop_max_rounds_closed")
                break
            if realized_pnl <= -args.daily_loss_limit_usdc:
                _log(log_path, "stop_daily_loss_limit", realized_pnl=realized_pnl)
                break

            tick_closed, tick_pending, tick_settlement, tick_pnl = _tick_open_positions(
                client=client,
                position_manager=position_manager,
                lifecycle=lifecycle,
                log_path=log_path,
                now_ms=now_ms,
                soft_force_exit_before_expiry_seconds=args.soft_force_exit_before_expiry_seconds,
                hard_force_exit_before_expiry_seconds=args.hard_force_exit_before_expiry_seconds,
                soft_force_exit_min_bid=args.soft_force_exit_min_bid,
                exit_retry_seconds=args.exit_retry_seconds,
                exit_order_timeout_seconds=args.exit_order_timeout_seconds,
                max_exit_attempts_per_position=args.max_exit_attempts_per_position,
                sell_slippage=args.sell_slippage,
                monitoring_db_path=args.monitoring_db_path,
                paper_settlement_config=paper_settlement_config,
            )
            closes_filled += tick_closed
            exits_pending_confirmation += tick_pending
            exits_pending_settlement += tick_settlement
            realized_pnl += tick_pnl
            if (
                lifecycle.max_rounds_reached(args.max_rounds)
                and not lifecycle.open_positions
                and not args.continue_after_max_rounds_until_runtime
            ):
                _log(log_path, "stop_max_rounds_closed")
                break
            if realized_pnl <= -args.daily_loss_limit_usdc:
                _log(log_path, "daily_loss_limit_reached", realized_pnl=realized_pnl)
                break

            try:
                if signal_jsonl_path is None:
                    batch = _read_event_batch_after(
                        args.monitoring_db_path,
                        model_version=args.model_version,
                        after_created_at=cursor_created_at,
                        after_event_id=cursor_event_id,
                        limit=args.event_limit,
                        v6_joint_config=v6_joint_config,
                    )
                    events = batch.events
                    if batch.rows_scanned:
                        cursor_created_at = batch.cursor_created_at
                        cursor_event_id = batch.cursor_event_id
                        if batch.rows_filtered:
                            _log(
                                log_path,
                                "signal_rows_filtered",
                                source="duckdb",
                                rows_scanned=batch.rows_scanned,
                                rows_filtered=batch.rows_filtered,
                                filter_reasons=batch.filter_reasons,
                                cursor_created_at=cursor_created_at,
                                cursor_event_id=cursor_event_id,
                            )
                else:
                    events, cursor_line_number = _read_signal_jsonl_after(
                        signal_jsonl_path,
                        after_line_number=cursor_line_number,
                        model_version=args.model_version,
                        limit=args.event_limit,
                        v6_joint_config=v6_joint_config,
                        entry_gate_mode=args.entry_gate_mode,
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

                settlement_position = lifecycle.open_position(event.round_slug, "settlement")
                if settlement_position is not None:
                    _log(
                        log_path,
                        "settlement_sleeve_hold",
                        reason="hold_to_settlement",
                        position=asdict(settlement_position),
                        signal=asdict(event),
                    )

                volatility_position = lifecycle.open_position(event.round_slug, "volatility")
                if volatility_position is not None:
                    if event.outcome_side == volatility_position.side:
                        sell_result = _maybe_exit(
                            client=client,
                            position_manager=position_manager,
                            position=volatility_position,
                            signal=event,
                            log_path=log_path,
                            exit_edge_threshold=args.exit_edge_threshold,
                            profit_target=args.profit_target,
                            sell_slippage=args.sell_slippage,
                            exit_order_timeout_seconds=args.exit_order_timeout_seconds,
                            monitoring_db_path=args.monitoring_db_path,
                            paper_settlement_config=paper_settlement_config,
                        )
                        if sell_result is not None:
                            realized_pnl += sell_result.realized_pnl
                            if sell_result.status == "filled":
                                closes_filled += 1
                            elif sell_result.status == "settled":
                                pass
                            elif sell_result.status == "pending_settlement":
                                exits_pending_settlement += 1
                            else:
                                exits_pending_confirmation += 1
                            lifecycle.mark_position_closed(event.round_slug, "volatility")
                            volatility_budget.apply_account_pnl(
                                event.round_slug,
                                _cash_pnl_for_budget(sell_result),
                            )
                            if realized_pnl <= -args.daily_loss_limit_usdc:
                                _log(log_path, "daily_loss_limit_reached", realized_pnl=realized_pnl)
                                break
                    else:
                        sell_result = _maybe_exit_opposite_correction(
                            client=client,
                            position_manager=position_manager,
                            position=volatility_position,
                            signal=event,
                            log_path=log_path,
                            opposite_exit_edge_threshold=args.opposite_exit_edge_threshold,
                            opposite_exit_min_seconds_to_expiry=args.opposite_exit_min_seconds_to_expiry,
                            sell_slippage=args.sell_slippage,
                            exit_order_timeout_seconds=args.exit_order_timeout_seconds,
                            monitoring_db_path=args.monitoring_db_path,
                            paper_settlement_config=paper_settlement_config,
                        )
                        if sell_result is not None:
                            realized_pnl += sell_result.realized_pnl
                            if sell_result.status == "filled":
                                closes_filled += 1
                            elif sell_result.status == "settled":
                                pass
                            elif sell_result.status == "pending_settlement":
                                exits_pending_settlement += 1
                            else:
                                exits_pending_confirmation += 1
                            lifecycle.mark_position_closed(event.round_slug, "volatility")
                            volatility_budget.apply_account_pnl(
                                event.round_slug,
                                _cash_pnl_for_budget(sell_result),
                            )
                            if realized_pnl <= -args.daily_loss_limit_usdc:
                                _log(log_path, "daily_loss_limit_reached", realized_pnl=realized_pnl)
                                break
                    continue

                if not _event_family_allowed(event, allowed_families):
                    _bump(skipped, "market_family_not_allowed")
                    continue
                seconds_to_expiry = (event.round_end_ts - now_ms) / 1000
                time_skip_reason = _entry_time_window_skip_reason(
                    seconds_to_expiry,
                    no_new_entry_before_expiry_seconds=args.no_new_entry_before_expiry_seconds,
                    min_seconds_to_expiry=args.min_seconds_to_expiry,
                    max_seconds_to_expiry=args.max_seconds_to_expiry,
                )
                if time_skip_reason is not None:
                    _bump(skipped, time_skip_reason)
                    if time_skip_reason == "no_new_entry_window":
                        _log(
                            log_path,
                            "entry_skipped",
                            reason=time_skip_reason,
                            signal=asdict(event),
                            seconds_to_expiry=seconds_to_expiry,
                            no_new_entry_before_expiry_seconds=args.no_new_entry_before_expiry_seconds,
                        )
                    continue
                if not lifecycle.mark_round_seen(event.round_slug, max_rounds=args.max_rounds):
                    _bump(skipped, "max_rounds")
                    _log(
                        log_path,
                        "entry_skipped",
                        reason="max_rounds",
                        signal=asdict(event),
                        observed_round_count=len(lifecycle.observed_rounds),
                        max_rounds=args.max_rounds,
                    )
                    continue
                if len(lifecycle.open_positions) >= args.max_combined_concurrent_positions:
                    _bump(skipped, "max_combined_concurrent_positions")
                    continue

                if event.round_slug not in lifecycle.filled_rounds and settlement_position is None:
                    side_skip_reason = _sleeve_side_cap_skip_reason(
                        lifecycle,
                        round_slug=event.round_slug,
                        sleeve="settlement",
                        side=event.outcome_side,
                        max_filled_per_side_per_round=(
                            args.settlement_max_filled_per_side_per_round
                        ),
                    )
                    if side_skip_reason is not None:
                        _bump(skipped, side_skip_reason)
                        _log(
                            log_path,
                            "entry_skipped",
                            reason=side_skip_reason,
                            sleeve="settlement",
                            signal=asdict(event),
                            filled_side_count=lifecycle.filled_count_for_side(
                                round_slug=event.round_slug,
                                sleeve="settlement",
                                side=event.outcome_side,
                            ),
                            max_filled_per_side_per_round=(
                                args.settlement_max_filled_per_side_per_round
                            ),
                        )
                        continue
                    entries_attempted += 1
                    lifecycle.mark_entry_attempted(event.event_id)
                    position = _try_entry(
                        client=client,
                        position_manager=position_manager,
                        signal=event,
                        log_path=log_path,
                        max_position_size_usdc=args.max_position_size_usdc,
                        entry_policy=entry_policy,
                        seconds_to_expiry=seconds_to_expiry,
                        buy_slippage=args.buy_slippage,
                        monitoring_db_path=args.monitoring_db_path,
                        sleeve="settlement",
                        paper=args.paper,
                        entry_gate_mode=args.entry_gate_mode,
                    )
                    lifecycle.mark_entry_result(event, position)
                    if position is not None:
                        entries_filled += 1
                        continue

                if (
                    args.enable_volatility_sleeve
                    and not lifecycle.has_open_sleeve(event.round_slug, "volatility")
                    and len(lifecycle.open_positions) < args.max_combined_concurrent_positions
                ):
                    budget_decision = volatility_budget.next_entry_decision(event.round_slug)
                    if not budget_decision.allowed:
                        _bump(skipped, budget_decision.reason)
                        _log(
                            log_path,
                            "entry_skipped",
                            reason=budget_decision.reason,
                            sleeve="volatility",
                            signal=asdict(event),
                            volatility_budget=asdict(budget_decision),
                        )
                        continue
                    volatility_position = _try_entry(
                        client=client,
                        position_manager=position_manager,
                        signal=event,
                        log_path=log_path,
                        max_position_size_usdc=budget_decision.size_usdc,
                        entry_policy=entry_policy,
                        seconds_to_expiry=seconds_to_expiry,
                        buy_slippage=args.buy_slippage,
                        monitoring_db_path=args.monitoring_db_path,
                        sleeve="volatility",
                        paper=args.paper,
                        entry_gate_mode=args.entry_gate_mode,
                    )
                    lifecycle.mark_entry_result(event, volatility_position)
                    if volatility_position is not None:
                        entries_filled += 1

            time.sleep(args.poll_seconds)
    finally:
        heartbeat_stop.set()
        shutdown_closed, shutdown_pending, shutdown_settlement, shutdown_pnl = _close_remaining_positions(
            client=client,
            position_manager=position_manager,
            positions=lifecycle.open_positions,
            log_path=log_path,
            sell_slippage=args.sell_slippage,
            exit_order_timeout_seconds=args.exit_order_timeout_seconds,
            monitoring_db_path=args.monitoring_db_path,
            paper_settlement_config=paper_settlement_config,
        )
        closes_filled += shutdown_closed
        exits_pending_confirmation += shutdown_pending
        exits_pending_settlement += shutdown_settlement
        realized_pnl += shutdown_pnl
        theoretical_pnl_usdc = _theoretical_pnl_from_positions(
            position_manager,
            event_ids=lifecycle.position_event_ids,
        )
        open_positions_at_shutdown = len(lifecycle.open_positions)
        lifecycle_complete = phase4_lifecycle_complete(
            errors=errors,
            entries_filled=entries_filled,
            open_positions_at_shutdown=open_positions_at_shutdown,
            exits_pending_confirmation=exits_pending_confirmation,
            exits_pending_settlement=exits_pending_settlement,
        )
        summary = {
            "phase": "phase4_real_champion_signal",
            "started_at": _iso(started_at),
            "finished_at": _iso(_now_ms()),
            "model_version": args.model_version,
            "market_families": sorted(allowed_families) or None,
            "edge_threshold": args.edge_threshold,
            "settlement_edge_threshold": entry_policy.effective_settlement_edge_threshold,
            "settlement_price_gate_mode": (
                "cost_edge_only" if args.entry_gate_mode == "v6-joint" else "legacy_min_entry"
            ),
            "settlement_min_entry_price": (
                None if args.entry_gate_mode == "v6-joint" else args.min_entry_price
            ),
            "volatility_score_threshold": entry_policy.volatility_score_threshold,
            "volatility_min_entry_price": entry_policy.volatility_min_entry_price,
            "volatility_min_seconds_to_expiry": entry_policy.volatility_min_seconds_to_expiry,
            "volatility_round_trip_cost": entry_policy.volatility_round_trip_cost,
            "volatility_safety_margin": entry_policy.volatility_safety_margin,
            "volatility_round_bankroll_usdc": volatility_budget.round_cap_usdc,
            "volatility_per_bet_cap_usdc": volatility_budget.per_bet_cap_usdc,
            "volatility_min_order_size_usdc": volatility_budget.min_order_size_usdc,
            "enable_volatility_live_entries": entry_policy.enable_volatility_live_entries,
            "enable_volatility_sleeve": args.enable_volatility_sleeve,
            "volatility_live_ordering_enabled": False,
            "volatility_ordering_mode": "paper_only",
            "paper": args.paper,
            "paper_settlement_resolution_enabled": paper_settlement_config.enabled,
            "paper_settlement_gamma_api_base": paper_settlement_config.gamma_api_base,
            "paper_settlement_timeout_seconds": (
                paper_settlement_config.request_timeout_seconds
            ),
            "paper_settlement_max_wait_after_expiry_seconds": (
                paper_settlement_config.max_wait_after_expiry_seconds
            ),
            "phase4_v5_role": "diagnostic_and_opportunity_analysis",
            "status": phase4_summary_status(
                errors=errors,
                entries_filled=entries_filled,
                lifecycle_complete=lifecycle_complete,
            ),
            "lifecycle_complete": lifecycle_complete,
            "entries_attempted": entries_attempted,
            "entries_filled": entries_filled,
            "max_combined_concurrent_positions": args.max_combined_concurrent_positions,
            "settlement_max_filled_per_side_per_round": (
                args.settlement_max_filled_per_side_per_round
            ),
            "volatility_budget_balances": dict(volatility_budget.balances or {}),
            "volatility_filled_count_by_round": lifecycle.volatility_filled_count_by_round,
            "filled_count_by_sleeve_round_side": (
                lifecycle.filled_count_by_sleeve_round_side
            ),
            "closes_filled": closes_filled,
            "exits_pending_confirmation": exits_pending_confirmation,
            "exits_pending_settlement": exits_pending_settlement,
            "realized_pnl_usdc": round(realized_pnl, 8),
            "theoretical_pnl_usdc": round(theoretical_pnl_usdc, 8),
            "account_cash_pnl_usdc": None,
            "pnl_reconciliation_status": "theoretical_only",
            "promotion_or_capital_sizing_evidence": False,
            "account_cashflow_reconciliation_required": True,
            "decision_evidence_allowed": False,
            "decision_evidence_blockers": _decision_evidence_blockers(
                lifecycle_complete=lifecycle_complete,
                open_positions_at_shutdown=open_positions_at_shutdown,
                exits_pending_confirmation=exits_pending_confirmation,
                exits_pending_settlement=exits_pending_settlement,
            ),
            "open_positions_at_shutdown": open_positions_at_shutdown,
            "processed_event_count": len(lifecycle.processed_event_ids),
            "attempted_entry_event_count": len(lifecycle.attempted_entry_event_ids),
            "observed_round_count": len(lifecycle.observed_rounds),
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
    parser.add_argument(
        "--market-families",
        default="",
        help=(
            "Comma-separated market families to trade (e.g. BTC-15M,ETH-15M). "
            "Empty trades all families. Signals from other families are skipped."
        ),
    )
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-position-size-usdc", type=float, default=1.0)
    parser.add_argument(
        "--min-entry-price",
        type=float,
        default=DEFAULT_MIN_ENTRY_PRICE,
        help="Skip entries whose fresh CLOB ask/worst fill price is below this token price.",
    )
    parser.add_argument(
        "--near-min-price-band",
        type=float,
        default=DEFAULT_NEAR_MIN_PRICE_BAND,
        help="Apply stricter near-min gating when ask/worst price is within this band above min_entry_price.",
    )
    parser.add_argument(
        "--near-min-fresh-edge-threshold",
        type=float,
        default=DEFAULT_NEAR_MIN_FRESH_EDGE_THRESHOLD,
        help="Required fresh edge for entries in the near-min price band.",
    )
    parser.add_argument(
        "--near-min-seconds-to-expiry",
        type=float,
        default=DEFAULT_NEAR_MIN_SECONDS_TO_EXPIRY,
        help="Minimum seconds-to-expiry for near-min price entries.",
    )
    parser.add_argument(
        "--soft-force-exit-min-bid",
        type=float,
        default=DEFAULT_SOFT_FORCE_EXIT_MIN_BID,
        help="Defer soft force exits when the bid is below this price.",
    )
    parser.add_argument("--daily-loss-limit-usdc", type=float, default=3.0)
    parser.add_argument("--max-concurrent-positions", type=int, default=2)
    parser.add_argument(
        "--max-combined-concurrent-positions",
        type=int,
        default=2,
        help="Top-level cap across settlement and volatility sleeves.",
    )
    parser.add_argument(
        "--settlement-max-filled-per-side-per-round",
        type=int,
        default=1,
        help="Cap filled settlement entries per round and side.",
    )
    parser.add_argument(
        "--volatility-max-filled-per-side-per-round",
        type=int,
        default=1,
        help=(
            "Deprecated no-op. Volatility re-entry is controlled by open-position "
            "state plus per-round bankroll/per-bet sizing, not same-side fill count."
        ),
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=DEFAULT_SETTLEMENT_EDGE_THRESHOLD,
        help=(
            "Legacy alias for --settlement-edge-threshold. Kept for runbook compatibility; "
            "new Phase 4 v5 configs should set the settlement threshold explicitly."
        ),
    )
    parser.add_argument(
        "--settlement-edge-threshold",
        type=float,
        default=None,
        help=(
            "Minimum settlement confidence edge for live entries. Defaults to "
            "--edge-threshold when omitted."
        ),
    )
    parser.add_argument(
        "--volatility-score-threshold",
        type=float,
        default=DEFAULT_VOLATILITY_SCORE_THRESHOLD,
        help="Diagnostic volatility score threshold; does not enable live volatility entries.",
    )
    parser.add_argument(
        "--volatility-min-entry-price",
        type=float,
        default=DEFAULT_VOLATILITY_MIN_ENTRY_PRICE,
        help="Diagnostic volatility gate ask/worst price floor.",
    )
    parser.add_argument(
        "--volatility-min-seconds-to-expiry",
        type=float,
        default=DEFAULT_VOLATILITY_MIN_SECONDS_TO_EXPIRY,
        help="Diagnostic volatility gate minimum seconds to expiry.",
    )
    parser.add_argument(
        "--volatility-round-trip-cost",
        type=float,
        default=DEFAULT_VOLATILITY_ROUND_TRIP_COST,
        help="Minimum expected volatility exit gain consumed by buy+sell cost drag.",
    )
    parser.add_argument(
        "--volatility-safety-margin",
        type=float,
        default=DEFAULT_VOLATILITY_SAFETY_MARGIN,
        help="Safety margin added on top of volatility round-trip cost.",
    )
    parser.add_argument(
        "--volatility-round-bankroll-usdc",
        type=float,
        default=DEFAULT_VOLATILITY_ROUND_BANKROLL_USDC,
        help="Per-round volatility sleeve bankroll reset.",
    )
    parser.add_argument(
        "--volatility-per-bet-cap-usdc",
        type=float,
        default=DEFAULT_VOLATILITY_ROUND_BANKROLL_USDC,
        help="Per-entry cap for the volatility sleeve.",
    )
    parser.add_argument(
        "--volatility-min-order-size-usdc",
        type=float,
        default=DEFAULT_VOLATILITY_MIN_ORDER_SIZE_USDC,
        help="Stop volatility entries for a round below this remaining bankroll.",
    )
    parser.add_argument(
        "--enable-volatility-sleeve",
        action="store_true",
        help="Enable volatility sleeve mechanics. Phase 4 v5 volatility entries are paper-only.",
    )
    parser.add_argument(
        "--enable-volatility-live-entries",
        action="store_true",
        help=(
            "Intent flag for a future promoted volatility path. In Phase 4 v5 this still does not "
            "place live volatility orders; the path remains paper/orderbook-only."
        ),
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run orderbook-only paper execution for all sleeves; no CLOB orders are posted.",
    )
    parser.add_argument(
        "--disable-paper-settlement-resolution",
        action="store_true",
        help=(
            "Disable paper-mode post-expiry Gamma settlement resolution. When disabled, "
            "expired paper positions remain pending settlement for offline reconciliation."
        ),
    )
    parser.add_argument(
        "--paper-settlement-gamma-api-base",
        default=os.getenv("POLYMARKET_GAMMA_API_BASE", DEFAULT_GAMMA_API_BASE),
        help="Gamma API base URL used to resolve expired paper positions.",
    )
    parser.add_argument(
        "--paper-settlement-timeout-seconds",
        type=float,
        default=10.0,
        help="Timeout for paper settlement Gamma API requests.",
    )
    parser.add_argument(
        "--paper-settlement-max-wait-after-expiry-seconds",
        type=float,
        default=180.0,
        help=(
            "Maximum grace period to keep an expired paper settlement position open "
            "while Gamma has not published a final result yet."
        ),
    )
    parser.add_argument("--exit-edge-threshold", type=float, default=0.10)
    parser.add_argument("--opposite-exit-edge-threshold", type=float, default=0.45)
    parser.add_argument("--opposite-exit-min-seconds-to-expiry", type=float, default=120.0)
    parser.add_argument("--no-new-entry-before-expiry-seconds", type=float, default=300.0)
    parser.add_argument("--soft-force-exit-before-expiry-seconds", type=float, default=240.0)
    parser.add_argument("--hard-force-exit-before-expiry-seconds", type=float, default=120.0)
    parser.add_argument("--exit-retry-seconds", type=float, default=10.0)
    parser.add_argument("--exit-order-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-exit-attempts-per-position", type=int, default=6)
    parser.add_argument("--profit-target", type=float, default=0.15)
    parser.add_argument("--min-seconds-to-expiry", type=float, default=180.0)
    parser.add_argument("--max-seconds-to-expiry", type=float, default=1200.0)
    parser.add_argument("--buy-slippage", type=float, default=0.02)
    parser.add_argument("--sell-slippage", type=float, default=0.02)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--max-runtime-minutes", type=float, default=240.0)
    parser.add_argument(
        "--continue-after-max-rounds-until-runtime",
        action="store_true",
        help=(
            "Keep the executor alive until max runtime after the round cap is reached. "
            "New rounds are still skipped by max_rounds; this only preserves the "
            "monitoring/execution window."
        ),
    )
    parser.add_argument("--log-path", default="logs/remote_dry_run_phase4_real_champion.jsonl")
    parser.add_argument("--summary-path", default="logs/remote_dry_run_phase4_real_champion_summary.json")
    parser.add_argument(
        "--entry-gate-mode",
        choices=("v5-edge", "v6-joint"),
        default="v5-edge",
        help="v5-edge uses settlement edge threshold; v6-joint uses explicit p_up/p_down/p_vol gate.",
    )
    parser.add_argument(
        "--v6-model-json-path",
        default="",
        help="Optional xgboost-v6 model.json used to load volatility gain priors for v6-joint mode.",
    )
    parser.add_argument("--v6-settlement-threshold", type=float, default=0.50)
    parser.add_argument("--v6-neutral-cap", type=float, default=0.25)
    parser.add_argument("--v6-volatility-threshold", type=float, default=0.60)
    parser.add_argument("--v6-round-trip-cost", type=float, default=0.072)
    parser.add_argument("--v6-ev-margin", type=float, default=0.01)
    parser.add_argument(
        "--v6-settlement-min-edge-after-cost",
        type=float,
        default=None,
        help=(
            "Minimum p_side - worst_price required for v6 settlement entries. "
            "Defaults to --v6-round-trip-cost + --v6-ev-margin."
        ),
    )
    return parser.parse_args()


def _entry_policy_from_args(args: argparse.Namespace) -> Phase4EntryPolicy:
    v6_settlement_edge_threshold = (
        args.v6_settlement_min_edge_after_cost
        if args.v6_settlement_min_edge_after_cost is not None
        else args.v6_round_trip_cost + args.v6_ev_margin
    )
    disable_settlement_edge = args.entry_gate_mode == "v6-joint"
    return Phase4EntryPolicy(
        min_entry_price=args.min_entry_price,
        near_min_price_band=args.near_min_price_band,
        near_min_fresh_edge_threshold=args.near_min_fresh_edge_threshold,
        near_min_seconds_to_expiry=args.near_min_seconds_to_expiry,
        edge_threshold=-999.0 if disable_settlement_edge else args.edge_threshold,
        settlement_edge_threshold=(
            v6_settlement_edge_threshold if disable_settlement_edge else args.settlement_edge_threshold
        ),
        volatility_score_threshold=args.volatility_score_threshold,
        volatility_min_entry_price=args.volatility_min_entry_price,
        volatility_min_seconds_to_expiry=args.volatility_min_seconds_to_expiry,
        volatility_round_trip_cost=args.volatility_round_trip_cost,
        volatility_safety_margin=args.volatility_safety_margin,
        enable_volatility_live_entries=args.enable_volatility_live_entries,
    )


def _v6_joint_config_from_args(args: argparse.Namespace) -> V6JointGateConfig | None:
    if args.entry_gate_mode != "v6-joint":
        return None
    if not is_v6_model_version(args.model_version):
        raise ValueError("--entry-gate-mode v6-joint requires model_version xgboost-v6")
    model_json_path = (
        Path(args.v6_model_json_path)
        if args.v6_model_json_path
        else None
    )
    if model_json_path is not None and model_json_path.is_file():
        return v6_joint_gate_config_from_model(
            model_json_path,
            settlement_threshold=args.v6_settlement_threshold,
            neutral_cap=args.v6_neutral_cap,
            volatility_threshold=args.v6_volatility_threshold,
            round_trip_cost=args.v6_round_trip_cost,
            ev_margin=args.v6_ev_margin,
        )
    return V6JointGateConfig(
        settlement_threshold=args.v6_settlement_threshold,
        neutral_cap=args.v6_neutral_cap,
        volatility_threshold=args.v6_volatility_threshold,
        round_trip_cost=args.v6_round_trip_cost,
        ev_margin=args.v6_ev_margin,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.settlement_max_filled_per_side_per_round <= 0:
        raise ValueError("--settlement-max-filled-per-side-per-round must be positive")
    if args.paper_settlement_timeout_seconds <= 0:
        raise ValueError("--paper-settlement-timeout-seconds must be positive")
    if args.entry_gate_mode == "v6-joint" and not is_v6_model_version(args.model_version):
        raise ValueError("--entry-gate-mode v6-joint requires model_version xgboost-v6")


def _decision_evidence_blockers(
    *,
    lifecycle_complete: bool,
    open_positions_at_shutdown: int,
    exits_pending_confirmation: int,
    exits_pending_settlement: int,
) -> list[str]:
    blockers = ["account_cashflow_reconciliation_required"]
    if not lifecycle_complete:
        blockers.append("lifecycle_not_complete")
    if open_positions_at_shutdown:
        blockers.append("open_positions_at_shutdown")
    if exits_pending_confirmation:
        blockers.append("exits_pending_confirmation")
    if exits_pending_settlement:
        blockers.append("exits_pending_settlement")
    return blockers


def _cash_pnl_for_budget(result: SellResult) -> float:
    return result.account_cash_pnl if result.account_cash_pnl is not None else result.realized_pnl


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
    v6_joint_config: V6JointGateConfig | None = None,
) -> list[SignalEvent]:
    return _read_event_batch_after(
        db_path,
        model_version=model_version,
        after_created_at=after_created_at,
        after_event_id=after_event_id,
        limit=limit,
        v6_joint_config=v6_joint_config,
    ).events


def _read_event_batch_after(
    db_path: str,
    *,
    model_version: str,
    after_created_at: int,
    after_event_id: str,
    limit: int,
    v6_joint_config: V6JointGateConfig | None = None,
) -> SignalReadBatch:
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
        filter_reasons: dict[str, int] = {}
        cursor_created_at = after_created_at
        cursor_event_id = after_event_id
        for row in rows:
            cursor_event_id = str(row[0])
            cursor_created_at = int(row[2])
            snapshot_json = str(row[4])
            try:
                snapshot = json.loads(snapshot_json)
            except json.JSONDecodeError:
                _bump(filter_reasons, "invalid_feature_snapshot_json")
                continue
            if not isinstance(snapshot, dict):
                _bump(filter_reasons, "invalid_feature_snapshot")
                continue
            canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
            parts = canonical_symbol.split(":")
            if len(parts) < 3:
                _bump(filter_reasons, "invalid_canonical_symbol")
                continue
            round_slug = parts[-2]
            outcome_side = parts[-1].upper()
            opposite_token_id = _opposite_token_id(
                conn,
                model_version=model_version,
                round_slug=round_slug,
                outcome_side=outcome_side if outcome_side in {"UP", "DOWN"} else "UP",
            )
            parsed = _event_from_row(
                row,
                model_version=model_version,
                v6_joint_config=v6_joint_config,
                opposite_token_id=opposite_token_id,
            )
            if parsed is not None:
                events.append(
                    replace(
                        parsed,
                        opposite_token_id=parsed.opposite_token_id or opposite_token_id,
                    )
                )
            else:
                _bump(
                    filter_reasons,
                    _event_filter_reason(
                        row,
                        model_version=model_version,
                        v6_joint_config=v6_joint_config,
                        opposite_token_id=opposite_token_id,
                    ),
                )
    best_events = _best_event_per_round(
        events,
        entry_gate_mode="v6-joint" if v6_joint_config else "v5-edge",
    )
    return SignalReadBatch(
        events=best_events,
        cursor_created_at=cursor_created_at,
        cursor_event_id=cursor_event_id,
        rows_scanned=len(rows),
        rows_filtered=max(0, len(rows) - len(events)),
        filter_reasons=filter_reasons,
    )


def _event_filter_reason(
    row: tuple[Any, ...],
    *,
    model_version: str,
    v6_joint_config: V6JointGateConfig | None,
    opposite_token_id: str,
) -> str:
    _event_id, _ts, _created_at, _prob_up_15m, snapshot_json = row
    try:
        snapshot = json.loads(str(snapshot_json))
    except json.JSONDecodeError:
        return "invalid_feature_snapshot_json"
    if not isinstance(snapshot, dict):
        return "invalid_feature_snapshot"
    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    parts = canonical_symbol.split(":")
    if len(parts) < 3:
        return "invalid_canonical_symbol"
    if _round_end_ts(parts[-2]) is None:
        return "invalid_round_slug"
    if v6_joint_config is not None and is_v6_model_version(model_version):
        fields = build_v6_signal_fields(
            event_id="probe",
            ts=0,
            created_at=0,
            snapshot=snapshot,
            model_version=model_version,
            config=v6_joint_config,
            round_end_ts=1,
            opposite_token_id=opposite_token_id,
        )
        if fields is None:
            return "v6_settlement_gate_miss"
    return "unparseable_signal"


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
    v6_joint_config: V6JointGateConfig | None = None,
    entry_gate_mode: str = "v5-edge",
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
            event = _event_from_signal_payload(
                payload,
                model_version=model_version,
                v6_joint_config=v6_joint_config,
            )
            if event is not None:
                events.append(event)
            if len(events) >= limit:
                break
    return _best_event_per_round(events, entry_gate_mode=entry_gate_mode), last_line_number


def _event_from_signal_payload(
    payload: Any,
    *,
    model_version: str,
    v6_joint_config: V6JointGateConfig | None = None,
) -> SignalEvent | None:
    if not isinstance(payload, dict):
        return None
    payload_model_version = str(payload.get("model_version") or model_version)
    if payload_model_version != model_version:
        return None
    round_slug = str(payload.get("round_slug") or "")
    round_end_ts = _optional_int(payload.get("round_end_ts")) or _round_end_ts(round_slug)
    market = _optional_float(payload.get("market_implied_prob"))
    prob_up_15m = _optional_float(payload.get("prob_up_15m"))
    if not round_slug or round_end_ts is None or market is None or prob_up_15m is None:
        return None
    if v6_joint_config is not None and is_v6_model_version(model_version):
        if payload.get("v6_joint_side"):
            side = str(payload.get("v6_joint_side")).upper()
            token_id = str(payload.get("token_id") or "")
            token_probability = _optional_float(payload.get("token_probability"))
            if side not in {"UP", "DOWN"} or not token_id or token_probability is None:
                return None
            edge = _optional_float(payload.get("edge")) or token_probability - market
            return SignalEvent(
                event_id=str(payload.get("event_id") or ""),
                ts=int(_optional_int(payload.get("ts")) or 0),
                created_at=int(_optional_int(payload.get("created_at")) or 0),
                prob_up_15m=float(prob_up_15m),
                canonical_symbol=str(
                    payload.get("canonical_symbol") or f"BTC-15M:{round_slug}:{side}"
                ),
                token_id=token_id,
                outcome_side=side,
                round_slug=round_slug,
                round_end_ts=int(round_end_ts),
                market_implied_prob=float(market),
                token_probability=float(token_probability),
                edge=float(edge),
                bridged_at=int(_optional_int(payload.get("bridged_at")) or 0),
                opposite_token_id=str(payload.get("opposite_token_id") or ""),
                p_up=_optional_float(payload.get("p_up")),
                p_down=_optional_float(payload.get("p_down")),
                p_neutral=_optional_float(payload.get("p_neutral")),
                p_vol_up=_optional_float(payload.get("p_vol_up")),
                p_vol_down=_optional_float(payload.get("p_vol_down")),
                v6_joint_side=side,
            )
        snapshot = {
            "canonical_symbol": payload.get("canonical_symbol"),
            "source_symbol": payload.get("token_id") or payload.get("source_symbol"),
            "market_implied_prob": market,
            "p_up": payload.get("p_up"),
            "p_down": payload.get("p_down"),
            "p_neutral": payload.get("p_neutral"),
            "p_vol_up": payload.get("p_vol_up"),
            "p_vol_down": payload.get("p_vol_down"),
        }
        fields = build_v6_signal_fields(
            event_id=str(payload.get("event_id") or ""),
            ts=int(_optional_int(payload.get("ts")) or 0),
            created_at=int(_optional_int(payload.get("created_at")) or 0),
            snapshot=snapshot,
            model_version=model_version,
            config=v6_joint_config,
            round_end_ts=int(round_end_ts),
            bridged_at=int(_optional_int(payload.get("bridged_at")) or 0),
            opposite_token_id=str(payload.get("opposite_token_id") or ""),
        )
        return SignalEvent(**fields) if fields is not None else None
    side = str(payload.get("outcome_side") or "").upper()
    token_id = str(payload.get("token_id") or payload.get("source_symbol") or "")
    token_probability = _optional_float(payload.get("token_probability"))
    if side not in {"UP", "DOWN"} or not token_id or token_probability is None:
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
        opposite_token_id=str(payload.get("opposite_token_id") or ""),
    )


def _event_from_row(
    row: tuple[Any, ...],
    *,
    model_version: str,
    v6_joint_config: V6JointGateConfig | None = None,
    opposite_token_id: str = "",
) -> SignalEvent | None:
    event_id, ts, created_at, prob_up_15m, snapshot_json = row
    try:
        snapshot = json.loads(snapshot_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(snapshot, dict):
        return None
    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    parts = canonical_symbol.split(":")
    if len(parts) < 3:
        return None
    round_slug = parts[-2]
    round_end_ts = _round_end_ts(round_slug)
    if round_end_ts is None:
        return None
    if v6_joint_config is not None and is_v6_model_version(model_version):
        fields = build_v6_signal_fields(
            event_id=str(event_id),
            ts=int(ts),
            created_at=int(created_at),
            snapshot=snapshot,
            model_version=model_version,
            config=v6_joint_config,
            round_end_ts=int(round_end_ts),
            opposite_token_id=opposite_token_id,
        )
        return SignalEvent(**fields) if fields is not None else None
    _family, _round_slug, side = parts[0], parts[-2], parts[-1].upper()
    if side not in {"UP", "DOWN"}:
        return None
    token_id = str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
    market = _optional_float(snapshot.get("market_implied_prob"))
    if not token_id or market is None:
        return None
    prob = float(prob_up_15m)
    token_probability = 1.0 - prob if side == "DOWN" else prob
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
        opposite_token_id="",
    )


def _opposite_token_id(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    round_slug: str,
    outcome_side: str,
) -> str:
    opposite_side = "DOWN" if outcome_side == "UP" else "UP"
    canonical_symbol = f"BTC-15M:{round_slug}:{opposite_side}"
    row = conn.execute(
        """
        SELECT feature_snapshot_json
        FROM prediction_events
        WHERE model_version = ?
          AND json_extract_string(feature_snapshot_json, '$.canonical_symbol') = ?
        ORDER BY created_at DESC, event_id DESC
        LIMIT 1
        """,
        [model_version, canonical_symbol],
    ).fetchone()
    if row is None:
        return ""
    try:
        snapshot = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return ""
    if not isinstance(snapshot, dict):
        return ""
    return str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")


def _best_event_per_round(
    events: list[SignalEvent],
    *,
    entry_gate_mode: str = "v5-edge",
) -> list[SignalEvent]:
    best: dict[str, SignalEvent] = {}
    for event in events:
        best_key = event.round_slug
        if entry_gate_mode == "v6-joint" and event.p_up is not None and event.p_vol_up is not None:
            best_key = (
                f"settlement:{event.round_slug}"
                if event.v6_joint_side is not None
                else f"volatility:{event.round_slug}"
            )
        previous = best.get(best_key)
        if previous is None:
            best[best_key] = event
            continue
        if entry_gate_mode == "v6-joint" and event.p_up is not None and event.p_vol_up is not None:
            event_score = _v6_event_selection_score(event)
            previous_score = _v6_event_selection_score(previous)
            if event_score > previous_score:
                best[best_key] = event
        elif event.edge > previous.edge:
            best[best_key] = event
    return sorted(best.values(), key=lambda item: (item.created_at, item.event_id))


def _v6_event_selection_score(event: SignalEvent) -> float:
    if event.v6_joint_side is None:
        if event.outcome_side == "UP":
            return float(event.p_vol_up or 0.0)
        return float(event.p_vol_down or 0.0)
    payload = {
        "p_up": event.p_up or 0.0,
        "p_down": event.p_down or 0.0,
        "p_vol_up": event.p_vol_up or 0.0,
        "p_vol_down": event.p_vol_down or 0.0,
    }
    return v6_selection_score(payload, event.outcome_side)


def _entry_time_window_skip_reason(
    seconds_to_expiry: float,
    *,
    no_new_entry_before_expiry_seconds: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
) -> str | None:
    if seconds_to_expiry < no_new_entry_before_expiry_seconds:
        return "no_new_entry_window"
    if seconds_to_expiry < min_seconds_to_expiry:
        return "near_or_past_expiry"
    if seconds_to_expiry > max_seconds_to_expiry:
        return "too_far_from_expiry"
    return None


def _tick_open_positions(
    *,
    client: Any,
    position_manager: PositionManager,
    lifecycle: RoundLifecycleState,
    log_path: Path,
    now_ms: int,
    soft_force_exit_before_expiry_seconds: float,
    hard_force_exit_before_expiry_seconds: float,
    soft_force_exit_min_bid: float,
    exit_retry_seconds: float,
    max_exit_attempts_per_position: int,
    sell_slippage: float,
    exit_order_timeout_seconds: float = 20.0,
    monitoring_db_path: str = "data/mlops/champion_catalog.duckdb",
    paper_settlement_config: PaperSettlementResolverConfig | None = None,
) -> tuple[int, int, int, float]:
    closed_count = 0
    pending_count = 0
    settlement_count = 0
    realized_pnl = 0.0

    for _round_slug, position in list(lifecycle.open_positions.items()):
        round_end_ts = _round_end_ts(position.round_slug)
        if round_end_ts is None:
            _log(
                log_path,
                "position_lifecycle_hold",
                reason="unknown_round_end",
                position=asdict(position),
            )
            continue
        seconds_to_expiry = (round_end_ts - now_ms) / 1000
        if position.sleeve == "settlement" and seconds_to_expiry > 0:
            _log(
                log_path,
                "settlement_sleeve_hold",
                reason="hold_to_settlement",
                position=asdict(position),
                seconds_to_expiry=seconds_to_expiry,
            )
            continue
        if position.lifecycle_state == "EXIT_REQUIRED":
            exit_reason = position.last_lifecycle_reason or "risk_exit"
        elif seconds_to_expiry <= 0:
            exit_reason = "expired_position_monitor"
        elif seconds_to_expiry <= hard_force_exit_before_expiry_seconds:
            exit_reason = "hard_force_exit"
        elif seconds_to_expiry <= soft_force_exit_before_expiry_seconds:
            exit_reason = "soft_force_exit"
        else:
            continue

        sell_result = _attempt_lifecycle_exit(
            client=client,
            position_manager=position_manager,
            position=position,
            log_path=log_path,
            now_ms=now_ms,
            seconds_to_expiry=seconds_to_expiry,
            exit_reason=exit_reason,
            soft_force_exit_min_bid=soft_force_exit_min_bid,
            exit_retry_seconds=exit_retry_seconds,
            exit_order_timeout_seconds=exit_order_timeout_seconds,
            max_exit_attempts_per_position=max_exit_attempts_per_position,
            sell_slippage=sell_slippage,
            monitoring_db_path=monitoring_db_path,
            paper_settlement_config=paper_settlement_config,
        )
        if sell_result is None:
            continue
        if sell_result.status == "filled":
            closed_count += 1
        elif sell_result.status == "settled":
            pass
        elif sell_result.status == "pending_settlement":
            settlement_count += 1
        else:
            pending_count += 1
        realized_pnl += sell_result.realized_pnl
        lifecycle.mark_position_closed(position.round_slug, position.sleeve)

    return closed_count, pending_count, settlement_count, realized_pnl


def _attempt_lifecycle_exit(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    log_path: Path,
    now_ms: int,
    seconds_to_expiry: float,
    exit_reason: str,
    soft_force_exit_min_bid: float,
    exit_retry_seconds: float,
    max_exit_attempts_per_position: int,
    sell_slippage: float,
    exit_order_timeout_seconds: float = 20.0,
    monitoring_db_path: str = "data/mlops/champion_catalog.duckdb",
    paper_settlement_config: PaperSettlementResolverConfig | None = None,
) -> SellResult | None:
    if seconds_to_expiry <= 0:
        resolved = _resolve_expired_paper_position(
            position_manager=position_manager,
            position=position,
            log_path=log_path,
            signal=None,
            reason=exit_reason,
            seconds_to_expiry=seconds_to_expiry,
            paper_settlement_config=paper_settlement_config,
        )
        if resolved is not None:
            return resolved
        if _paper_settlement_should_wait(
            position=position,
            now_ms=now_ms,
            paper_settlement_config=paper_settlement_config,
        ):
            position.lifecycle_state = "AWAITING_SETTLEMENT"
            position.last_lifecycle_reason = exit_reason
            _log(
                log_path,
                "paper_settlement_resolution_waiting",
                reason=exit_reason,
                position=asdict(position),
                seconds_to_expiry=seconds_to_expiry,
                max_wait_after_expiry_seconds=(
                    None
                    if paper_settlement_config is None
                    else paper_settlement_config.max_wait_after_expiry_seconds
                ),
            )
            return None
        position.lifecycle_state = "AWAITING_SETTLEMENT"
        position.last_lifecycle_reason = exit_reason
        return _mark_pending_settlement(
            log_path=log_path,
            position=position,
            signal=None,
            reason=exit_reason,
            seconds_to_expiry=seconds_to_expiry,
        )

    retry_wait_ms = int(max(0.0, exit_retry_seconds) * 1000)
    if position.last_exit_attempt_at > 0 and now_ms - position.last_exit_attempt_at < retry_wait_ms:
        _log(
            log_path,
            "position_lifecycle_hold",
            reason="exit_retry_wait",
            position=asdict(position),
            exit_reason=exit_reason,
            seconds_to_expiry=seconds_to_expiry,
            retry_wait_ms=retry_wait_ms,
            next_retry_at=_iso(position.last_exit_attempt_at + retry_wait_ms),
        )
        return None

    if (
        position.exit_attempt_count >= max_exit_attempts_per_position
        and exit_reason != "hard_force_exit"
    ):
        if position.lifecycle_state != "MANUAL_INTERVENTION_REQUIRED":
            position.lifecycle_state = "MANUAL_INTERVENTION_REQUIRED"
            position.last_lifecycle_reason = "max_exit_attempts_reached"
            _log(
                log_path,
                "position_lifecycle_transition",
                reason="max_exit_attempts_reached",
                lifecycle_state=position.lifecycle_state,
                position=asdict(position),
                exit_reason=exit_reason,
                seconds_to_expiry=seconds_to_expiry,
                max_exit_attempts_per_position=max_exit_attempts_per_position,
            )
        return None

    position.exit_attempt_count += 1
    position.last_exit_attempt_at = now_ms
    position.lifecycle_state = "EXIT_PENDING"
    position.last_lifecycle_reason = exit_reason
    _log(
        log_path,
        "position_lifecycle_transition",
        reason=exit_reason,
        lifecycle_state=position.lifecycle_state,
        position=asdict(position),
        seconds_to_expiry=seconds_to_expiry,
        exit_attempt_count=position.exit_attempt_count,
    )

    try:
        bid, _ask = _best_bid_ask(client, position.token_id)
    except OrderBookUnavailable as exc:
        _log(
            log_path,
            "force_exit_hold",
            reason="orderbook_unavailable",
            exit_reason=exit_reason,
            position=asdict(position),
            seconds_to_expiry=seconds_to_expiry,
            exit_attempt_count=position.exit_attempt_count,
            **exc.to_log_payload(),
        )
        return None
    if bid is None:
        _log(
            log_path,
            "force_exit_hold",
            reason="missing_bid",
            exit_reason=exit_reason,
            position=asdict(position),
            seconds_to_expiry=seconds_to_expiry,
            exit_attempt_count=position.exit_attempt_count,
        )
        return None
    if soft_force_exit_deferred(
        exit_reason=exit_reason,
        bid=float(bid),
        soft_force_exit_min_bid=soft_force_exit_min_bid,
    ):
        _log(
            log_path,
            "force_exit_hold",
            reason="soft_force_exit_bid_too_low",
            exit_reason=exit_reason,
            position=asdict(position),
            bid=float(bid),
            soft_force_exit_min_bid=soft_force_exit_min_bid,
            seconds_to_expiry=seconds_to_expiry,
            exit_attempt_count=position.exit_attempt_count,
        )
        return None

    return _sell_position(
        client=client,
        position_manager=position_manager,
        position=position,
        log_path=log_path,
        bid=float(bid),
        sell_slippage=sell_slippage,
        fill_confirm_timeout_seconds=exit_order_timeout_seconds,
        reason=exit_reason,
        signal=None,
        monitoring_db_path=monitoring_db_path,
    )


def _try_entry(
    *,
    client: Any,
    position_manager: PositionManager,
    signal: SignalEvent,
    log_path: Path,
    max_position_size_usdc: float,
    entry_policy: Phase4EntryPolicy,
    seconds_to_expiry: float,
    buy_slippage: float,
    monitoring_db_path: str,
    sleeve: str = "settlement",
    paper: bool = False,
    entry_gate_mode: str = "v5-edge",
) -> LivePosition | None:
    from py_clob_client_v2 import MarketOrderArgs, OrderType
    from py_clob_client_v2.clob_types import PartialCreateOrderOptions
    from py_clob_client_v2.order_builder.constants import BUY

    v6_settlement_cost_edge_only = sleeve == "settlement" and entry_gate_mode == "v6-joint"
    no_quote_gate_evaluation = evaluate_entry_gates(
        settlement_edge=signal.edge,
        ask=None,
        worst_price=None,
        token_probability=signal.token_probability,
        seconds_to_expiry=seconds_to_expiry,
        policy=entry_policy,
    )
    try:
        bid, ask = _best_bid_ask(client, signal.token_id)
    except OrderBookUnavailable as exc:
        _log(
            log_path,
            "entry_skipped",
            reason="orderbook_unavailable",
            signal=asdict(signal),
            gate_evaluation=asdict(no_quote_gate_evaluation),
            **exc.to_log_payload(),
        )
        return None
    if ask is None:
        _log(
            log_path,
            "entry_skipped",
            reason="missing_ask",
            signal=asdict(signal),
            gate_evaluation=asdict(no_quote_gate_evaluation),
        )
        return None
    tick_size = client.get_tick_size(signal.token_id)
    neg_risk = client.get_neg_risk(signal.token_id)
    worst_price = min(0.99, _round_price(float(ask) + buy_slippage, tick_size))
    fresh_edge_at_worst = signal.token_probability - worst_price
    settlement_edge_for_gate = (
        fresh_edge_at_worst
        if sleeve == "settlement" and entry_gate_mode == "v6-joint"
        else signal.edge
    )
    gate_evaluation = evaluate_entry_gates(
        settlement_edge=settlement_edge_for_gate,
        ask=float(ask),
        bid=None if bid is None else float(bid),
        worst_price=worst_price,
        token_probability=signal.token_probability,
        seconds_to_expiry=seconds_to_expiry,
        policy=entry_policy,
    )
    gate_payload = asdict(gate_evaluation)
    _log(
        log_path,
        "entry_gate_evaluated",
        signal=asdict(signal),
        bid=bid,
        ask=ask,
        worst_price=worst_price,
        fresh_edge_at_worst=fresh_edge_at_worst,
        raw_settlement_edge=signal.edge,
        seconds_to_expiry=seconds_to_expiry,
        gate_evaluation=gate_payload,
    )
    if sleeve == "settlement" and entry_gate_mode != "v6-joint":
        if not gate_evaluation.settlement_gate_passed:
            _log(
                log_path,
                "entry_skipped",
                reason="below_edge_threshold",
                legacy_reason="below_edge_threshold",
                sleeve=sleeve,
                signal=asdict(signal),
                bid=bid,
                ask=ask,
                worst_price=worst_price,
                fresh_edge_at_worst=fresh_edge_at_worst,
                seconds_to_expiry=seconds_to_expiry,
                gate_evaluation=gate_payload,
            )
            return None
    elif sleeve == "settlement" and signal.v6_joint_side is None:
        _log(
            log_path,
            "entry_skipped",
            reason="v6_settlement_gate_miss",
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            seconds_to_expiry=seconds_to_expiry,
            gate_evaluation=gate_payload,
        )
        return None
    if sleeve == "volatility":
        if not gate_evaluation.volatility_gate_passed:
            _log(
                log_path,
                "entry_skipped",
                reason="volatility_gate_below_cost",
                sleeve=sleeve,
                signal=asdict(signal),
                bid=bid,
                ask=ask,
                worst_price=worst_price,
                fresh_edge_at_worst=fresh_edge_at_worst,
                seconds_to_expiry=seconds_to_expiry,
                gate_evaluation=gate_payload,
            )
            return None
        if not paper:
            _log(
                log_path,
                "entry_skipped",
                reason=(
                    "volatility_live_requires_paper_evidence"
                    if entry_policy.enable_volatility_live_entries
                    else "volatility_live_disabled"
                ),
                sleeve=sleeve,
                signal=asdict(signal),
                bid=bid,
                ask=ask,
                worst_price=worst_price,
                seconds_to_expiry=seconds_to_expiry,
                gate_evaluation=gate_payload,
            )
            return None
        if paper:
            return _open_paper_position(
                position_manager=position_manager,
                signal=signal,
                log_path=log_path,
                sleeve=sleeve,
                fill_price=worst_price,
                size_usdc=max_position_size_usdc,
                order_posted_at=_now_ms(),
                gate_payload=gate_payload,
            )
    skip_reason = (
        settlement_cost_edge_skip_reason(
            fresh_edge_at_worst=fresh_edge_at_worst,
            policy=entry_policy,
        )
        if v6_settlement_cost_edge_only
        else entry_price_skip_reason(
            ask=float(ask),
            worst_price=worst_price,
            fresh_edge_at_worst=fresh_edge_at_worst,
            seconds_to_expiry=seconds_to_expiry,
            policy=entry_policy,
        )
    )
    if skip_reason is not None:
        _log(
            log_path,
            "entry_skipped",
            reason=skip_reason,
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            fresh_edge_at_worst=fresh_edge_at_worst,
            seconds_to_expiry=seconds_to_expiry,
            min_entry_price=entry_policy.min_entry_price,
            near_min_price_band=entry_policy.near_min_price_band,
            near_min_fresh_edge_threshold=entry_policy.near_min_fresh_edge_threshold,
            near_min_seconds_to_expiry=entry_policy.near_min_seconds_to_expiry,
            settlement_edge_threshold=entry_policy.effective_settlement_edge_threshold,
            settlement_price_gate_mode=(
                "cost_edge_only" if v6_settlement_cost_edge_only else "legacy_min_entry"
            ),
            gate_evaluation=gate_payload,
        )
        return None
    if paper:
        return _open_paper_position(
            position_manager=position_manager,
            signal=signal,
            log_path=log_path,
            sleeve=sleeve,
            fill_price=worst_price,
            size_usdc=max_position_size_usdc,
            order_posted_at=_now_ms(),
            gate_payload=gate_payload,
        )
    if not signal.opposite_token_id:
        _log(
            log_path,
            "entry_skipped",
            reason="missing_opposite_token_id",
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            min_entry_price=entry_policy.min_entry_price,
            gate_evaluation=gate_payload,
        )
        return None
    try:
        complement_bid, _complement_ask = _best_bid_ask(client, signal.opposite_token_id)
    except OrderBookUnavailable as exc:
        _log(
            log_path,
            "entry_skipped",
            reason="opposite_orderbook_unavailable",
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            opposite_token_id=signal.opposite_token_id,
            gate_evaluation=gate_payload,
            **exc.to_log_payload(),
        )
        return None
    if complement_bid is None:
        _log(
            log_path,
            "entry_skipped",
            reason="missing_opposite_bid",
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            opposite_token_id=signal.opposite_token_id,
            gate_evaluation=gate_payload,
        )
        return None
    complement_entry_price = _round_price(1.0 - float(complement_bid), tick_size)
    complement_fresh_edge = signal.token_probability - complement_entry_price
    complement_skip_reason = (
        settlement_cost_edge_skip_reason(
            fresh_edge_at_worst=complement_fresh_edge,
            policy=entry_policy,
        )
        if v6_settlement_cost_edge_only
        else entry_price_skip_reason(
            ask=complement_entry_price,
            worst_price=complement_entry_price,
            fresh_edge_at_worst=complement_fresh_edge,
            seconds_to_expiry=seconds_to_expiry,
            policy=entry_policy,
        )
    )
    if complement_skip_reason is not None:
        _log(
            log_path,
            "entry_skipped",
            reason=f"complement_{complement_skip_reason}",
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            complement_bid=complement_bid,
            complement_entry_price=complement_entry_price,
            complement_fresh_edge_at_price=complement_fresh_edge,
            opposite_token_id=signal.opposite_token_id,
            seconds_to_expiry=seconds_to_expiry,
            min_entry_price=entry_policy.min_entry_price,
            near_min_price_band=entry_policy.near_min_price_band,
            near_min_fresh_edge_threshold=entry_policy.near_min_fresh_edge_threshold,
            near_min_seconds_to_expiry=entry_policy.near_min_seconds_to_expiry,
            settlement_edge_threshold=entry_policy.effective_settlement_edge_threshold,
            settlement_price_gate_mode=(
                "cost_edge_only" if v6_settlement_cost_edge_only else "legacy_min_entry"
            ),
            gate_evaluation=gate_payload,
            complement_gate_evaluation=asdict(
                evaluate_entry_gates(
                    settlement_edge=(
                        complement_fresh_edge
                        if sleeve == "settlement" and entry_gate_mode == "v6-joint"
                        else signal.edge
                    ),
                    ask=complement_entry_price,
                    worst_price=complement_entry_price,
                    token_probability=signal.token_probability,
                    seconds_to_expiry=seconds_to_expiry,
                    policy=entry_policy,
                )
            ),
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
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            error=str(exc),
            error_type=type(exc).__name__,
            gate_evaluation=gate_payload,
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
        sleeve=sleeve,
        signal=asdict(signal),
        bid=bid,
        ask=ask,
        worst_price=worst_price,
        gate_evaluation=gate_payload,
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
    fill = _fill_for_order(
        client,
        order_id,
        wanted_side="BUY",
        asset_id=signal.token_id,
        after_ts_ms=order_submit_started_at,
        transaction_hashes=response.get("transactionsHashes"),
    )
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
            gate_evaluation=gate_payload,
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
        sleeve=sleeve,
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
        sleeve=sleeve,
        paper=paper,
    )
    if fill_price < entry_policy.min_entry_price and not v6_settlement_cost_edge_only:
        position.lifecycle_state = "EXIT_REQUIRED"
        position.last_lifecycle_reason = "under_min_fill_exit"
    _persist_cash_leg(
        monitoring_db_path=monitoring_db_path,
        event_id=event_id,
        round_slug=signal.round_slug,
        action="BUY",
        fill=fill,
        order_id=order_id,
        sleeve=sleeve,
    )
    _log(
        log_path,
        "entry_filled",
        sleeve=sleeve,
        position=asdict(position),
        fill=fill,
        gate_evaluation=gate_payload,
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
    if fill_price < entry_policy.min_entry_price and not v6_settlement_cost_edge_only:
        _log(
            log_path,
            "entry_fill_below_min",
            position=asdict(position),
            fill=fill,
            min_entry_price=entry_policy.min_entry_price,
            exit_required=True,
            reason="under_min_fill_exit",
        )
    return position


def _open_paper_position(
    *,
    position_manager: PositionManager,
    signal: SignalEvent,
    log_path: Path,
    sleeve: str,
    fill_price: float,
    size_usdc: float,
    order_posted_at: int,
    gate_payload: dict[str, Any],
) -> LivePosition:
    fill_size = size_usdc / fill_price if fill_price > 0 else 0.0
    event_suffix = (signal.event_id or str(order_posted_at))[-8:]
    event_id = f"phase4-paper-{sleeve}-{signal.round_slug}-{signal.outcome_side}-{event_suffix}"
    order_id = f"paper-{sleeve}-{event_suffix}"
    position_manager.open_position(
        event_id=event_id,
        symbol=signal.canonical_symbol,
        side=signal.outcome_side,
        sleeve=sleeve,
        entry_price=fill_price,
        fill_price=fill_price,
        size=fill_size,
        order_id=order_id,
        entry_time=order_posted_at,
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
        opened_at=order_posted_at,
        entry_signal_event_id=signal.event_id,
        entry_signal_ts=signal.ts,
        entry_signal_created_at=signal.created_at,
        entry_signal_bridged_at=signal.bridged_at,
        entry_order_posted_at=order_posted_at,
        sleeve=sleeve,
        paper=True,
    )
    _log(
        log_path,
        "paper_entry_filled",
        sleeve=sleeve,
        position=asdict(position),
        signal=asdict(signal),
        size_usdc=size_usdc,
        gate_evaluation=gate_payload,
        timestamps={
            **_signal_timestamps(signal),
            "paper_entry_at": _iso(order_posted_at),
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
    exit_order_timeout_seconds: float = 20.0,
    monitoring_db_path: str = "data/mlops/champion_catalog.duckdb",
    paper_settlement_config: PaperSettlementResolverConfig | None = None,
) -> SellResult | None:
    now_ms = _now_ms()
    seconds_to_expiry = (signal.round_end_ts - now_ms) / 1000
    try:
        bid, _ask = _best_bid_ask(client, position.token_id)
    except OrderBookUnavailable as exc:
        if seconds_to_expiry <= 0:
            resolved = _resolve_expired_paper_position(
                position_manager=position_manager,
                position=position,
                log_path=log_path,
                signal=signal,
                reason="expired_orderbook_unavailable",
                seconds_to_expiry=seconds_to_expiry,
                paper_settlement_config=paper_settlement_config,
            )
            if resolved is not None:
                return resolved
            if _paper_settlement_should_wait(
                position=position,
                now_ms=now_ms,
                paper_settlement_config=paper_settlement_config,
            ):
                position.lifecycle_state = "AWAITING_SETTLEMENT"
                position.last_lifecycle_reason = "expired_orderbook_unavailable"
                _log(
                    log_path,
                    "paper_settlement_resolution_waiting",
                    reason="expired_orderbook_unavailable",
                    position=asdict(position),
                    signal=asdict(signal),
                    seconds_to_expiry=seconds_to_expiry,
                    max_wait_after_expiry_seconds=(
                        None
                        if paper_settlement_config is None
                        else paper_settlement_config.max_wait_after_expiry_seconds
                    ),
                    **exc.to_log_payload(),
                )
                return None
            return _mark_pending_settlement(
                log_path=log_path,
                position=position,
                signal=signal,
                reason="expired_orderbook_unavailable",
                seconds_to_expiry=seconds_to_expiry,
                error_payload=exc.to_log_payload(),
            )
        _log(
            log_path,
            "exit_hold",
            reason="orderbook_unavailable",
            position=asdict(position),
            signal=asdict(signal),
            seconds_to_expiry=seconds_to_expiry,
            **exc.to_log_payload(),
        )
        return None
    if bid is None:
        if seconds_to_expiry <= 0:
            resolved = _resolve_expired_paper_position(
                position_manager=position_manager,
                position=position,
                log_path=log_path,
                signal=signal,
                reason="expired_missing_bid",
                seconds_to_expiry=seconds_to_expiry,
                paper_settlement_config=paper_settlement_config,
            )
            if resolved is not None:
                return resolved
            if _paper_settlement_should_wait(
                position=position,
                now_ms=now_ms,
                paper_settlement_config=paper_settlement_config,
            ):
                position.lifecycle_state = "AWAITING_SETTLEMENT"
                position.last_lifecycle_reason = "expired_missing_bid"
                _log(
                    log_path,
                    "paper_settlement_resolution_waiting",
                    reason="expired_missing_bid",
                    position=asdict(position),
                    signal=asdict(signal),
                    seconds_to_expiry=seconds_to_expiry,
                    max_wait_after_expiry_seconds=(
                        None
                        if paper_settlement_config is None
                        else paper_settlement_config.max_wait_after_expiry_seconds
                    ),
                )
                return None
            return _mark_pending_settlement(
                log_path=log_path,
                position=position,
                signal=signal,
                reason="expired_missing_bid",
                seconds_to_expiry=seconds_to_expiry,
            )
        _log(
            log_path,
            "exit_hold",
            reason="missing_bid",
            position=asdict(position),
            signal=asdict(signal),
            seconds_to_expiry=seconds_to_expiry,
        )
        return None
    unrealized = float(bid) - position.fill_price
    if position.sleeve == "volatility":
        should_exit = unrealized >= profit_target or seconds_to_expiry <= 60
    else:
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
        fill_confirm_timeout_seconds=exit_order_timeout_seconds,
        reason="exit_signal",
        signal=signal,
        monitoring_db_path=monitoring_db_path,
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
    exit_order_timeout_seconds: float = 20.0,
    monitoring_db_path: str = "data/mlops/champion_catalog.duckdb",
    paper_settlement_config: PaperSettlementResolverConfig | None = None,
) -> SellResult | None:
    """Exit an open position when the opposite side becomes strongly favored."""

    seconds_to_expiry = (signal.round_end_ts - _now_ms()) / 1000
    if seconds_to_expiry <= 0:
        resolved = _resolve_expired_paper_position(
            position_manager=position_manager,
            position=position,
            log_path=log_path,
            signal=signal,
            reason="expired_before_opposite_exit",
            seconds_to_expiry=seconds_to_expiry,
            paper_settlement_config=paper_settlement_config,
        )
        if resolved is not None:
            return resolved
        return _mark_pending_settlement(
            log_path=log_path,
            position=position,
            signal=signal,
            reason="expired_before_opposite_exit",
            seconds_to_expiry=seconds_to_expiry,
        )
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
        fill_confirm_timeout_seconds=exit_order_timeout_seconds,
        reason="opposite_side_exit_correction",
        signal=signal,
        monitoring_db_path=monitoring_db_path,
    )


def _close_remaining_positions(
    *,
    client: Any,
    position_manager: PositionManager,
    positions: dict[str, LivePosition],
    log_path: Path,
    sell_slippage: float,
    exit_order_timeout_seconds: float = 20.0,
    monitoring_db_path: str = "data/mlops/champion_catalog.duckdb",
    paper_settlement_config: PaperSettlementResolverConfig | None = None,
) -> tuple[int, int, int, float]:
    closed_count = 0
    pending_count = 0
    settlement_count = 0
    realized_pnl = 0.0
    for position_key, position in list(positions.items()):
        round_end_ts = _round_end_ts(position.round_slug)
        seconds_to_expiry = None
        is_expired = False
        if round_end_ts is not None:
            seconds_to_expiry = (round_end_ts - _now_ms()) / 1000
            is_expired = seconds_to_expiry <= 0
        if position.sleeve == "settlement":
            if is_expired:
                sell_result = _resolve_expired_paper_position(
                    position_manager=position_manager,
                    position=position,
                    log_path=log_path,
                    signal=None,
                    reason="shutdown_settlement_hold_expired",
                    seconds_to_expiry=seconds_to_expiry,
                    paper_settlement_config=paper_settlement_config,
                )
                if sell_result is None:
                    sell_result = _mark_pending_settlement(
                        log_path=log_path,
                        position=position,
                        signal=None,
                        reason="shutdown_settlement_hold_expired",
                        seconds_to_expiry=seconds_to_expiry,
                    )
                if sell_result.status == "pending_settlement":
                    settlement_count += 1
                realized_pnl += sell_result.realized_pnl
                del positions[position_key]
                continue
            _log(
                log_path,
                "shutdown_close_skipped",
                reason="settlement_sleeve_hold_to_redeem",
                position=asdict(position),
                seconds_to_expiry=seconds_to_expiry,
            )
            continue
        try:
            bid, _ask = _best_bid_ask(client, position.token_id)
        except OrderBookUnavailable as exc:
            if is_expired:
                sell_result = _resolve_expired_paper_position(
                    position_manager=position_manager,
                    position=position,
                    log_path=log_path,
                    signal=None,
                    reason="shutdown_expired_orderbook_unavailable",
                    seconds_to_expiry=seconds_to_expiry,
                    paper_settlement_config=paper_settlement_config,
                )
                if sell_result is None:
                    sell_result = _mark_pending_settlement(
                        log_path=log_path,
                        position=position,
                        signal=None,
                        reason="shutdown_expired_orderbook_unavailable",
                        seconds_to_expiry=seconds_to_expiry,
                        error_payload=exc.to_log_payload(),
                    )
                if sell_result.status == "pending_settlement":
                    settlement_count += 1
                realized_pnl += sell_result.realized_pnl
                del positions[position_key]
                continue
            _log(
                log_path,
                "shutdown_close_skipped",
                reason="orderbook_unavailable",
                position=asdict(position),
                seconds_to_expiry=seconds_to_expiry,
                **exc.to_log_payload(),
            )
            continue
        if bid is None:
            if is_expired:
                sell_result = _resolve_expired_paper_position(
                    position_manager=position_manager,
                    position=position,
                    log_path=log_path,
                    signal=None,
                    reason="shutdown_expired_missing_bid",
                    seconds_to_expiry=seconds_to_expiry,
                    paper_settlement_config=paper_settlement_config,
                )
                if sell_result is None:
                    sell_result = _mark_pending_settlement(
                        log_path=log_path,
                        position=position,
                        signal=None,
                        reason="shutdown_expired_missing_bid",
                        seconds_to_expiry=seconds_to_expiry,
                    )
                if sell_result.status == "pending_settlement":
                    settlement_count += 1
                realized_pnl += sell_result.realized_pnl
                del positions[position_key]
                continue
            _log(
                log_path,
                "shutdown_close_skipped",
                reason="missing_bid",
                position=asdict(position),
                seconds_to_expiry=seconds_to_expiry,
            )
            continue
        sell_result = _sell_position(
            client=client,
            position_manager=position_manager,
            position=position,
            log_path=log_path,
            bid=float(bid),
            sell_slippage=sell_slippage,
            fill_confirm_timeout_seconds=exit_order_timeout_seconds,
            reason="shutdown",
            signal=None,
            monitoring_db_path=monitoring_db_path,
        )
        if sell_result is not None:
            if sell_result.status == "filled":
                closed_count += 1
            elif sell_result.status == "settled":
                pass
            elif sell_result.status == "pending_settlement":
                settlement_count += 1
            else:
                pending_count += 1
            realized_pnl += sell_result.realized_pnl
            del positions[position_key]
    return closed_count, pending_count, settlement_count, realized_pnl


def _resolve_expired_paper_position(
    *,
    position_manager: PositionManager,
    position: LivePosition,
    log_path: Path,
    signal: SignalEvent | None,
    reason: str,
    seconds_to_expiry: float | None,
    paper_settlement_config: PaperSettlementResolverConfig | None,
) -> SellResult | None:
    if (
        paper_settlement_config is None
        or not paper_settlement_config.enabled
        or not position.paper
    ):
        return None
    resolution = _fetch_paper_settlement_resolution(
        position.round_slug,
        config=paper_settlement_config,
    )
    if resolution.result is None:
        _log(
            log_path,
            "paper_settlement_resolution_pending",
            reason=reason,
            position=asdict(position),
            signal=None if signal is None else asdict(signal),
            seconds_to_expiry=seconds_to_expiry,
            resolution_source=resolution.source,
            resolution_error=resolution.error,
            market=resolution.market,
        )
        return None

    round_end_ts = _round_end_ts(position.round_slug)
    settled = position_manager.settle_position(
        position.event_id,
        resolution.result,
        settlement_time=round_end_ts,
    )
    pnl = float(settled.realized_pnl or 0.0)
    position.lifecycle_state = "SETTLED"
    position.last_lifecycle_reason = reason
    _log(
        log_path,
        "paper_settlement_resolved",
        reason=reason,
        position=asdict(position),
        signal=None if signal is None else asdict(signal),
        seconds_to_expiry=seconds_to_expiry,
        settlement_result=settled.settlement_result,
        exit_price=settled.exit_price,
        realized_pnl=pnl,
        realized_account_pnl=pnl,
        realized_pnl_source="paper_gamma_settlement",
        resolution_source=resolution.source,
        market=resolution.market,
        account_cashflow_reconciliation_required=False,
        settlement_reconciliation_required=False,
    )
    return SellResult(status="settled", realized_pnl=pnl, account_cash_pnl=pnl)


def _paper_settlement_should_wait(
    *,
    position: LivePosition,
    now_ms: int,
    paper_settlement_config: PaperSettlementResolverConfig | None,
) -> bool:
    if (
        paper_settlement_config is None
        or not paper_settlement_config.enabled
        or not position.paper
    ):
        return False
    round_end_ts = _round_end_ts(position.round_slug)
    if round_end_ts is None:
        return False
    seconds_after_expiry = (now_ms - round_end_ts) / 1000
    return seconds_after_expiry < paper_settlement_config.max_wait_after_expiry_seconds


def _fetch_paper_settlement_resolution(
    round_slug: str,
    *,
    config: PaperSettlementResolverConfig,
) -> PaperSettlementResolution:
    market, error_text = _fetch_gamma_market_by_slug(
        config.gamma_api_base,
        round_slug,
        timeout_seconds=config.request_timeout_seconds,
    )
    if market is None:
        return PaperSettlementResolution(
            result=None,
            source="gamma_market",
            market=None,
            error=error_text,
        )
    slim_market = _gamma_market_log_payload(market)
    if not _gamma_market_is_resolved(market):
        return PaperSettlementResolution(
            result=None,
            source="gamma_market",
            market=slim_market,
            error="market_not_resolved",
        )
    outcome = _winning_outcome_from_gamma_market(market)
    return PaperSettlementResolution(
        result=outcome,
        source="gamma_market",
        market=slim_market,
        error=None if outcome is not None else "winner_unavailable",
    )


def _fetch_gamma_market_by_slug(
    gamma_api_base: str,
    slug: str,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, str | None]:
    base = str(gamma_api_base or DEFAULT_GAMMA_API_BASE).rstrip("/")
    errors: list[str] = []
    param_sets = (
        {"slug": slug, "closed": "true", "limit": "1"},
        {"slug": slug, "active": "true", "closed": "false", "limit": "1"},
        {"slug": slug, "limit": "1"},
    )
    for params in param_sets:
        url = f"{base}/markets?{parse.urlencode(params)}"
        req = request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "BiGan-phase4-paper-settlement/1.0",
            },
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, error.URLError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return dict(payload[0]), None
    event_url = f"{base}/events/slug/{parse.quote(slug, safe='')}"
    event_req = request.Request(
        event_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "BiGan-phase4-paper-settlement/1.0",
        },
    )
    try:
        with request.urlopen(event_req, timeout=timeout_seconds) as resp:
            event_payload = json.loads(resp.read().decode("utf-8"))
    except (TimeoutError, error.URLError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    else:
        event_market = _gamma_market_from_event_payload(event_payload, slug)
        if event_market is not None:
            return event_market, None
    return None, errors[-1] if errors else "market_not_found"


def _gamma_market_from_event_payload(
    payload: Any,
    slug: str,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    markets = payload.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if isinstance(market, dict) and str(market.get("slug") or "") == slug:
            return dict(market)
    return None


def _gamma_market_is_resolved(market: dict[str, Any]) -> bool:
    if bool(market.get("closed")):
        return True
    status = str(market.get("umaResolutionStatus") or "").strip().lower()
    return status == "resolved"


def _winning_outcome_from_gamma_market(market: dict[str, Any]) -> str | None:
    outcomes = _json_list(market.get("outcomes"))
    prices = _json_list(market.get("outcomePrices"))
    if len(outcomes) != len(prices) or not outcomes:
        return None
    parsed: dict[str, float] = {}
    for outcome, price in zip(outcomes, prices, strict=True):
        side = str(outcome).strip().upper()
        if side not in {"UP", "DOWN"}:
            continue
        parsed_price = _optional_float(price)
        if parsed_price is None:
            return None
        parsed[side] = parsed_price
    up = parsed.get("UP")
    down = parsed.get("DOWN")
    if up is None or down is None or up == down:
        return None
    return "UP" if up > down else "DOWN"


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []
    return []


def _gamma_market_log_payload(market: dict[str, Any]) -> dict[str, Any]:
    return {
        key: market.get(key)
        for key in (
            "slug",
            "conditionId",
            "closed",
            "umaResolutionStatus",
            "outcomes",
            "outcomePrices",
            "endDate",
            "eventStartTime",
        )
    }


def _mark_pending_settlement(
    *,
    log_path: Path,
    position: LivePosition,
    signal: SignalEvent | None,
    reason: str,
    seconds_to_expiry: float | None,
    error_payload: dict[str, str] | None = None,
) -> SellResult:
    position.lifecycle_state = "AWAITING_SETTLEMENT"
    position.last_lifecycle_reason = reason
    _log(
        log_path,
        "exit_pending_settlement",
        reason=reason,
        position=asdict(position),
        signal=None if signal is None else asdict(signal),
        seconds_to_expiry=seconds_to_expiry,
        position_assumed_closed_to_prevent_duplicate_sell=True,
        account_cashflow_reconciliation_required=True,
        settlement_reconciliation_required=True,
        **(error_payload or {}),
    )
    return SellResult(status="pending_settlement")


def _sell_position(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    log_path: Path,
    bid: float,
    sell_slippage: float,
    fill_confirm_timeout_seconds: float = 20.0,
    reason: str,
    signal: SignalEvent | None,
    monitoring_db_path: str = "data/mlops/champion_catalog.duckdb",
) -> SellResult | None:
    if position.paper:
        fill_price = max(0.01, bid - sell_slippage)
        closed = position_manager.close_position(position.event_id, fill_price)
        pnl = float(closed.realized_pnl or 0.0)
        position.lifecycle_state = "EXIT_FILLED"
        position.last_lifecycle_reason = reason
        _log(
            log_path,
            "paper_exit_filled",
            reason=reason,
            sleeve=position.sleeve,
            position=asdict(position),
            signal=None if signal is None else asdict(signal),
            bid=bid,
            exit_price=fill_price,
            realized_account_pnl=pnl,
            realized_pnl=pnl,
            realized_pnl_source="paper_orderbook_bid_minus_slippage",
        )
        return SellResult(status="filled", realized_pnl=pnl, account_cash_pnl=pnl)

    from py_clob_client_v2 import MarketOrderArgs, OrderType
    from py_clob_client_v2.clob_types import PartialCreateOrderOptions
    from py_clob_client_v2.order_builder.constants import SELL

    position.lifecycle_state = "EXIT_PENDING"
    position.last_lifecycle_reason = reason
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
    try:
        response = client.post_order(order, OrderType.FOK)
    except Exception as exc:  # noqa: BLE001 - Polymarket returns failed FOK sells as API exceptions.
        _log(
            log_path,
            "exit_order_post_failed",
            reason=reason,
            position=asdict(position),
            signal=None if signal is None else asdict(signal),
            bid=bid,
            worst_price=worst_price,
            sell_size=sell_size,
            dust_amount=dust_amount,
            dust_value_usd=dust_value_usd,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None
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
    fill = _fill_for_order(
        client,
        order_id,
        wanted_side="SELL",
        asset_id=position.token_id,
        after_ts_ms=position.last_exit_attempt_at or _now_ms(),
        transaction_hashes=response.get("transactionsHashes"),
        max_wait_seconds=fill_confirm_timeout_seconds,
        poll_seconds=2.0,
    )
    fill_price = _optional_float(fill.get("price")) or bid
    if not fill:
        _log(
            log_path,
            "exit_pending_confirmation",
            reason=reason,
            position=asdict(position),
            signal=None if signal is None else asdict(signal),
            sell_order_id=order_id,
            response=response,
            position_assumed_closed_to_prevent_duplicate_sell=True,
            account_cashflow_reconciliation_required=True,
        )
        return SellResult(status="pending_confirmation")
    closed = position_manager.close_position(position.event_id, fill_price)
    pnl = float(closed.realized_pnl or 0.0)
    position.lifecycle_state = "EXIT_FILLED"
    position.last_lifecycle_reason = reason
    _persist_cash_leg(
        monitoring_db_path=monitoring_db_path,
        event_id=position.event_id,
        round_slug=position.round_slug,
        action="SELL",
        fill=fill,
        order_id=order_id,
        sleeve=position.sleeve,
        dust_token_amount=dust_amount,
    )
    account_cash_pnl = _account_cash_pnl_for_position(
        monitoring_db_path=monitoring_db_path,
        event_id=position.event_id,
    )
    _log(
        log_path,
        "exit_filled",
        reason=reason,
        position=asdict(position),
        sell_order_id=order_id,
        fill=fill,
        exit_price=fill_price,
        account_cash_pnl=account_cash_pnl,
        realized_pnl=pnl,
        realized_pnl_source="position_manager_fill_price",
        account_cashflow_reconciliation_required=True,
        dust_amount=dust_amount,
        dust_value_usd=dust_value_usd,
    )
    return SellResult(status="filled", realized_pnl=pnl, account_cash_pnl=account_cash_pnl)


def _persist_cash_leg(
    *,
    monitoring_db_path: str,
    event_id: str,
    round_slug: str,
    action: str,
    fill: dict[str, Any],
    order_id: str | None,
    sleeve: str = "settlement",
    dust_token_amount: float = 0.0,
) -> None:
    try:
        leg = leg_from_clob_fill(
            event_id=event_id,
            round_slug=round_slug,
            sleeve=sleeve,
            action=action,  # type: ignore[arg-type]
            fill=fill,
            order_id=order_id,
            dust_token_amount=dust_token_amount,
        )
        with connect_mlops_db(monitoring_db_path) as conn:
            record_execution_cash_legs(conn, [leg], replace=True)
    except Exception:  # noqa: BLE001 - cash-leg persistence must not stop trading.
        return


def _account_cash_pnl_for_position(
    *,
    monitoring_db_path: str,
    event_id: str,
) -> float | None:
    try:
        with connect_mlops_db(monitoring_db_path) as conn:
            rows = read_execution_cash_legs(conn, event_id=event_id)
    except Exception:  # noqa: BLE001 - unavailable cash-leg accounting should not stop trading.
        return None
    actions = {str(row.get("action") or "").upper() for row in rows}
    if "BUY" not in actions or not ({"SELL", "REDEEM"} & actions):
        return None
    return float(sum(float(row.get("cash_delta") or 0.0) for row in rows))


def _theoretical_pnl_from_positions(
    position_manager: PositionManager,
    *,
    event_ids: set[str] | frozenset[str] | None = None,
) -> float:
    total = 0.0
    scoped_event_ids = set(event_ids) if event_ids is not None else None
    for position in position_manager.list_positions():
        if scoped_event_ids is not None and position.event_id not in scoped_event_ids:
            continue
        if position.realized_pnl is not None and position.status in {"closed", "expired"}:
            total += float(position.realized_pnl)
    return total


def _fill_for_order(
    client: Any,
    order_id: str,
    *,
    wanted_side: str,
    asset_id: str | None = None,
    after_ts_ms: int | None = None,
    transaction_hashes: Any = None,
    max_wait_seconds: float = 2.0,
    poll_seconds: float = 2.0,
) -> dict[str, Any]:
    """Find a confirmed CLOB trade for a just-posted FOK order.

    Matched FOK responses can precede the authenticated trade feed by a few
    seconds, and busy accounts can push the target trade beyond the first
    unfiltered page. Retry briefly and prefer a token/time-filtered paginated
    query when the client supports it.
    """

    attempts = _confirmation_attempt_count(max_wait_seconds, poll_seconds)
    hashes = _normalise_hashes(transaction_hashes)
    for attempt in range(attempts):
        if attempt > 0:
            time.sleep(max(0.0, poll_seconds))
        for trade in _candidate_trades_for_order(client, asset_id=asset_id, after_ts_ms=after_ts_ms):
            if _trade_matches_order(trade, order_id, wanted_side=wanted_side, transaction_hashes=hashes):
                return dict(trade)
    return {}


def _confirmation_attempt_count(max_wait_seconds: float, poll_seconds: float) -> int:
    if max_wait_seconds <= 0 or poll_seconds <= 0:
        return 1
    return max(1, int(math.ceil(max_wait_seconds / poll_seconds)) + 1)


def _candidate_trades_for_order(
    client: Any,
    *,
    asset_id: str | None,
    after_ts_ms: int | None,
) -> list[dict[str, Any]]:
    try:
        from py_clob_client_v2.clob_types import TradeParams

        params = TradeParams(
            asset_id=asset_id,
            after=max(0, int(after_ts_ms / 1000) - 120) if after_ts_ms is not None else None,
        )
        return list(client.get_trades(params=params, only_first_page=False))
    except TypeError:
        try:
            return list(client.get_trades())
        except Exception:
            return []
    except Exception:
        try:
            return list(client.get_trades())
        except Exception:
            return []


def _trade_matches_order(
    trade: dict[str, Any],
    order_id: str,
    *,
    wanted_side: str,
    transaction_hashes: set[str],
) -> bool:
    side = str(trade.get("side") or "").upper()
    if side != wanted_side.upper():
        return False
    taker_order_id = str(trade.get("taker_order_id") or trade.get("takerOrderId") or "")
    trade_hash = str(trade.get("transaction_hash") or trade.get("transactionHash") or "")
    has_order_match = bool(order_id) and taker_order_id == order_id
    has_hash_match = bool(trade_hash) and trade_hash.lower() in transaction_hashes
    return (has_order_match or has_hash_match) and _trade_is_confirmed(trade)


def _normalise_hashes(raw_hashes: Any) -> set[str]:
    if raw_hashes is None:
        return set()
    if isinstance(raw_hashes, str):
        return {raw_hashes.lower()}
    if isinstance(raw_hashes, (list, tuple, set)):
        return {str(value).lower() for value in raw_hashes if value}
    return {str(raw_hashes).lower()}


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
