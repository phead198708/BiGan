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
from collections import Counter
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
from bigan.execution.low_latency_overlay import (
    LowLatencyEntryOverlay,
    LowLatencyOverlayConfig,
)
from bigan.execution.phase4_policy import (
    DEFAULT_MAX_SIGNAL_AGE_SECONDS,
    DEFAULT_MIN_ENTRY_PRICE,
    DEFAULT_NEAR_MIN_FRESH_EDGE_THRESHOLD,
    DEFAULT_NEAR_MIN_PRICE_BAND,
    DEFAULT_NEAR_MIN_SECONDS_TO_EXPIRY,
    DEFAULT_SETTLEMENT_EDGE_THRESHOLD,
    DEFAULT_SETTLEMENT_MIN_CONFIDENCE,
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
    entry_price_drift_skip_reason,
    entry_price_skip_reason,
    evaluate_entry_gates,
    phase4_lifecycle_complete,
    phase4_summary_status,
    settlement_cost_edge_skip_reason,
    soft_force_exit_deferred,
    v7_raw_side_agreement_skip_reason,
    v7_raw_side_required_min_probability,
)
from bigan.execution.position_manager import PositionManager
from bigan.execution.signal_queue import (
    JsonlSignalSource,
    KafkaSignalSource,
    SignalCursor,
    SignalSource,
)
from bigan.execution.v6_gate import (
    V6JointGateConfig,
    build_v6_signal_fields,
    evaluate_v6_settlement_side,
    is_v6_model_version,
    v6_joint_gate_config_from_model,
    v6_payload_from_values,
    v6_selection_score,
)
from bigan.execution.v7_convergence_calibration import (
    V7ConvergenceCalibrationConfig,
    V7ConvergenceCalibrationGate,
)
from bigan.modeling.families import market_family_from_symbol

DEFAULT_GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DEFAULT_SETTLEMENT_REVERSAL_MIN_CONFIDENCE = 0.75
DEFAULT_SETTLEMENT_REVERSAL_HYSTERESIS_BARS = 2
DEFAULT_SIGNAL_JSONL_STALE_WARN_SECONDS = 900.0
DEFAULT_NO_NEW_OBSERVED_ROUND_WARN_SECONDS = 1800.0


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
    model_probability: float | None = None
    polymarket_price: float | None = None
    mispricing_edge: float | None = None
    p_up: float | None = None
    p_down: float | None = None
    p_neutral: float | None = None
    p_vol_up: float | None = None
    p_vol_down: float | None = None
    v6_joint_side: str | None = None
    settlement_residual: float | None = None
    token_expected_win_probability: float | None = None
    p_up_residual_adjusted: float | None = None
    p_down_residual_adjusted: float | None = None
    expected_edge_up: float | None = None
    expected_edge_down: float | None = None
    residual_expected_edge_up: float | None = None
    residual_expected_edge_down: float | None = None
    p_up_hit_5c_before_loss_10c: float | None = None
    p_up_hit_10c_before_loss_10c: float | None = None
    p_up_loss_10c_before_hit_5c: float | None = None
    p_down_hit_5c_before_loss_10c: float | None = None
    p_down_hit_10c_before_loss_10c: float | None = None
    p_down_loss_10c_before_hit_5c: float | None = None
    selected_hit_5c_before_loss_10c: float | None = None
    selected_hit_10c_before_loss_10c: float | None = None
    selected_loss_10c_before_hit_5c: float | None = None
    selected_confidence_score: float | None = None
    selected_side: str | None = None
    selected_expected_edge: float | None = None
    entry_worst_price: float | None = None
    should_enter_settlement: bool | None = None


@dataclass(slots=True)
class SignalJsonlWatchdogState:
    path: Path | None
    stale_warn_seconds: float
    checks: int = 0
    stale_checks: int = 0
    stale_events: int = 0
    recovered_events: int = 0
    stale_active: bool = False
    max_age_seconds: float | None = None
    first_stale_at: str | None = None
    last_stale_at: str | None = None
    last_recovered_at: str | None = None
    last_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.path is not None and self.stale_warn_seconds > 0

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": None if self.path is None else str(self.path),
            "stale_warn_seconds": self.stale_warn_seconds,
            "checks": self.checks,
            "stale_checks": self.stale_checks,
            "stale_events": self.stale_events,
            "recovered_events": self.recovered_events,
            "stale_active": self.stale_active,
            "max_age_seconds": self.max_age_seconds,
            "first_stale_at": self.first_stale_at,
            "last_stale_at": self.last_stale_at,
            "last_recovered_at": self.last_recovered_at,
            "last_snapshot": self.last_snapshot,
        }


@dataclass(slots=True)
class NoNewObservedRoundWatchdogState:
    warn_seconds: float
    last_observed_round_at: int
    last_observed_round_slug: str | None = None
    checks: int = 0
    stale_checks: int = 0
    stale_events: int = 0
    recovered_events: int = 0
    stale_active: bool = False
    max_age_seconds: float | None = None
    last_age_seconds: float | None = None
    first_stale_at: str | None = None
    last_stale_at: str | None = None
    last_recovered_at: str | None = None

    @property
    def enabled(self) -> bool:
        return self.warn_seconds > 0

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warn_seconds": self.warn_seconds,
            "checks": self.checks,
            "stale_checks": self.stale_checks,
            "stale_events": self.stale_events,
            "recovered_events": self.recovered_events,
            "stale_active": self.stale_active,
            "max_age_seconds": self.max_age_seconds,
            "last_age_seconds": self.last_age_seconds,
            "first_stale_at": self.first_stale_at,
            "last_stale_at": self.last_stale_at,
            "last_recovered_at": self.last_recovered_at,
            "last_observed_round_at": _iso(self.last_observed_round_at),
            "last_observed_round_slug": self.last_observed_round_slug,
        }


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
    entry_p_up: float | None = None
    entry_p_down: float | None = None
    entry_p_neutral: float | None = None
    entry_model_probability: float | None = None
    entry_polymarket_price: float | None = None
    entry_mispricing_edge: float | None = None
    lifecycle_state: str = "OPEN"
    exit_attempt_count: int = 0
    last_exit_attempt_at: int = 0
    last_lifecycle_reason: str = ""
    sleeve: str = "settlement"
    paper: bool = False
    settlement_reversal_candidate_side: str = ""
    settlement_reversal_candidate_count: int = 0
    settlement_decay_candidate_count: int = 0
    settlement_same_side_confirmation_event_id: str = ""
    settlement_same_side_confirmation_created_at: int = 0
    settlement_same_side_confirmation_confidence: float = 0.0
    v7_position_reversal_candidate_count: int = 0
    v7_position_weak_hold_candidate_count: int = 0
    v7_position_divergence_candidate_count: int = 0
    v7_position_take_profit_candidate_count: int = 0
    v7_position_take_profit_candidate_reason: str = ""
    v7_position_adverse_confidence_candidate_count: int = 0
    v7_position_last_divergence_reduce_at: int = 0
    v7_position_adverse_confidence_reduce_count: int = 0
    v7_position_last_adverse_confidence_reduce_at: int = 0
    v7_position_realized_pnl_usdc: float = 0.0


@dataclass(frozen=True, slots=True)
class SellResult:
    status: str
    realized_pnl: float = 0.0
    account_cash_pnl: float | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PaperSettlementResolverConfig:
    enabled: bool = True
    gamma_api_base: str = DEFAULT_GAMMA_API_BASE
    request_timeout_seconds: float = 10.0
    max_wait_after_expiry_seconds: float = 86_400.0


@dataclass(frozen=True, slots=True)
class PaperSettlementResolution:
    result: str | None
    source: str
    market: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SettlementExitConfig:
    allow_mid_round_exit: bool = False
    reversal_min_confidence: float = DEFAULT_SETTLEMENT_REVERSAL_MIN_CONFIDENCE
    reversal_hysteresis_bars: int = DEFAULT_SETTLEMENT_REVERSAL_HYSTERESIS_BARS
    confidence_decay_enabled: bool = False
    decay_floor: float = 0.55
    decay_delta: float = 0.25
    decay_hysteresis_bars: int = DEFAULT_SETTLEMENT_REVERSAL_HYSTERESIS_BARS
    decay_opposite_min_confidence: float | None = DEFAULT_SETTLEMENT_REVERSAL_MIN_CONFIDENCE
    price_stop_enabled: bool = False
    stop_price_delta: float = 0.15
    stop_loss_usdc: float = 0.50
    stop_min_seconds_to_expiry: float = 120.0
    price_stop_same_side_confirmation_veto_enabled: bool = False
    price_stop_same_side_confirmation_min_confidence: float = DEFAULT_SETTLEMENT_MIN_CONFIDENCE
    price_stop_same_side_confirmation_max_age_seconds: float | None = (
        DEFAULT_MAX_SIGNAL_AGE_SECONDS
    )


@dataclass(frozen=True, slots=True)
class V7SettlementPositionConfig:
    enabled: bool = False
    paper_execute: bool = False
    round_cap_usdc: float = 1.0
    add_edge_min: float = 0.08
    full_add_edge: float = 0.20
    weak_hold_edge: float = 0.02
    reduce_fraction: float = 0.50
    divergence_reduce_max_hold_edge: float = 0.08
    exit_hold_edge: float = -0.02
    exit_hysteresis_bars: int = 2
    reversal_min_confidence: float = 0.75
    reversal_min_edge: float = 0.04
    reversal_hysteresis_bars: int = 2
    min_rebalance_usdc: float = 0.05
    convergence_price_tolerance: float = 0.02
    convergence_model_decay_tolerance: float = 0.10
    divergence_hysteresis_bars: int = 2
    add_cooldown_after_divergence_reduce_seconds: float = 120.0
    convergence_take_profit_enabled: bool = False
    take_profit_hold_edge: float = 0.03
    take_profit_residual_ratio: float = 0.40
    take_profit_price_convergence_move: float = 0.10
    take_profit_price_convergence_hold_edge_ratio: float = 0.50
    take_profit_force_exit_seconds: float = 180.0
    take_profit_hysteresis_bars: int = 2
    take_profit_up_hold_edge_tighten: float = 0.01
    take_profit_min_profit_delta: float = 0.10
    take_profit_min_profit_return: float = 0.35
    low_confidence_scalp_enabled: bool = False
    low_confidence_scalp_max_confidence_score: float = 0.0
    low_confidence_scalp_take_profit_min_profit_delta: float = 0.05
    low_confidence_scalp_take_profit_min_profit_return: float = 0.10
    low_confidence_scalp_take_profit_hysteresis_bars: int = 1
    low_confidence_scalp_adverse_full_exit_enabled: bool = False
    adverse_confidence_decay_enabled: bool = False
    adverse_confidence_price_delta_start: float = 0.10
    adverse_confidence_base_allowed_decay: float = 0.08
    adverse_confidence_price_decay_slope: float = 0.30
    adverse_confidence_min_allowed_decay: float = 0.015
    adverse_confidence_max_required_probability: float = 0.97
    adverse_confidence_exit_probability_buffer: float = 0.03
    adverse_confidence_full_exit_min_model_decay: float = 0.06
    adverse_confidence_full_exit_max_hold_edge: float = 0.25
    adverse_confidence_reduce_min_model_decay: float = 0.06
    adverse_confidence_dust_exit_max_cost: float = 0.15
    adverse_confidence_dust_exit_min_candidate_count: int = 3
    adverse_confidence_hysteresis_bars: int = 2
    adverse_confidence_max_reduces: int = 0
    adverse_confidence_post_reduce_full_exit_enabled: bool = False
    adverse_confidence_post_reduce_full_exit_bars: int = 1
    adverse_confidence_post_reduce_full_exit_min_model_decay: float = 0.06
    adverse_confidence_post_reduce_full_exit_max_hold_edge: float = -1.0
    block_add_after_adverse_confidence_reduce: bool = False
    post_take_profit_reentry_quality_enabled: bool = False
    post_take_profit_reentry_min_model_probability_improvement: float = 0.03
    post_take_profit_reentry_min_raw_probability_improvement: float = 0.02
    post_take_profit_reentry_min_seconds_to_expiry: float = 420.0


@dataclass(frozen=True, slots=True)
class V7EntryCandidateBufferConfig:
    enabled: bool = False
    max_wait_seconds: float = 45.0
    min_price: float = 0.40
    max_price: float = 0.70
    min_edge: float = 0.04
    min_seconds_to_expiry: float = 330.0
    immediate_confidence_score: float | None = None
    max_candidates_per_round: int = 64


@dataclass(frozen=True, slots=True)
class V7EntryCandidateBufferAction:
    action: str
    reason: str
    entry_event: SignalEvent | None = None
    candidate_count: int = 0
    stale_candidate_count: int = 0
    best_event_id: str = ""
    best_score: float | None = None
    first_seen_at: int | None = None
    release_after_ms: int | None = None

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "stale_candidate_count": self.stale_candidate_count,
            "best_event_id": self.best_event_id,
            "best_score": self.best_score,
            "first_seen_at": None if self.first_seen_at is None else _iso(self.first_seen_at),
            "release_after": None if self.release_after_ms is None else _iso(self.release_after_ms),
            "entry_event": (
                None if self.entry_event is None else _v7_entry_candidate_summary(self.entry_event)
            ),
        }


@dataclass(slots=True)
class V7EntryCandidateBucket:
    first_seen_at: int
    candidates: list[SignalEvent] = field(default_factory=list)


@dataclass(slots=True)
class V7EntryCandidateBuffer:
    config: V7EntryCandidateBufferConfig
    buckets: dict[str, V7EntryCandidateBucket] = field(default_factory=dict)

    def observe(
        self,
        event: SignalEvent,
        *,
        now_ms: int,
        seconds_to_expiry: float,
        max_signal_age_seconds: float | None = None,
    ) -> V7EntryCandidateBufferAction:
        skip_reason = _v7_entry_candidate_skip_reason(
            event,
            config=self.config,
            seconds_to_expiry=seconds_to_expiry,
        )
        if skip_reason is not None:
            return V7EntryCandidateBufferAction(
                action="skipped",
                reason=skip_reason,
                entry_event=None,
            )

        bucket = self.buckets.get(event.round_slug)
        if bucket is None:
            bucket = V7EntryCandidateBucket(first_seen_at=now_ms)
            self.buckets[event.round_slug] = bucket
        _v7_add_or_replace_candidate(
            bucket,
            event,
            max_candidates_per_round=self.config.max_candidates_per_round,
        )
        fresh_candidates, stale_candidate_count = _fresh_v7_entry_candidates(
            bucket.candidates,
            now_ms=now_ms,
            max_signal_age_seconds=max_signal_age_seconds,
        )
        bucket.candidates[:] = fresh_candidates
        if not fresh_candidates:
            self.buckets.pop(event.round_slug, None)
            return V7EntryCandidateBufferAction(
                action="skipped",
                reason="v7_entry_candidate_signal_age_above_threshold",
                entry_event=None,
                stale_candidate_count=stale_candidate_count,
            )
        best = _select_v7_entry_candidate(fresh_candidates)
        best_score = _v7_entry_candidate_score(best)
        release_after_ms = bucket.first_seen_at + int(self.config.max_wait_seconds * 1000)
        immediate_release = (
            self.config.immediate_confidence_score is not None
            and best_score is not None
            and best_score >= self.config.immediate_confidence_score
        )
        release_due = now_ms >= release_after_ms
        expiry_due = seconds_to_expiry <= self.config.min_seconds_to_expiry
        if immediate_release or release_due or expiry_due:
            self.buckets.pop(event.round_slug, None)
            if immediate_release:
                reason = "v7_entry_candidate_immediate_confidence"
            elif expiry_due:
                reason = "v7_entry_candidate_expiry_release"
            else:
                reason = "v7_entry_candidate_wait_elapsed"
            return V7EntryCandidateBufferAction(
                action="released",
                reason=reason,
                entry_event=best,
                candidate_count=len(bucket.candidates),
                stale_candidate_count=stale_candidate_count,
                best_event_id=best.event_id,
                best_score=best_score,
                first_seen_at=bucket.first_seen_at,
                release_after_ms=release_after_ms,
            )
        return V7EntryCandidateBufferAction(
            action="buffered",
            reason="v7_entry_candidate_waiting",
            entry_event=None,
            candidate_count=len(bucket.candidates),
            stale_candidate_count=stale_candidate_count,
            best_event_id=best.event_id,
            best_score=best_score,
            first_seen_at=bucket.first_seen_at,
            release_after_ms=release_after_ms,
        )


V7_LATE_FORCE_EXIT_REASON = "convergence_force_exit_before_expiry"
V7_LATE_FORCE_EXIT_HYSTERESIS_WAIT_REASON = (
    "convergence_force_exit_before_expiry_hysteresis_wait"
)
V7_PROFIT_LOCK_BEFORE_EXPIRY_REASON = "convergence_profit_lock_before_expiry"
V7_LOSS_SALVAGE_BEFORE_EXPIRY_REASON = "convergence_loss_salvage_before_expiry"
V7_SLOT_RELEASE_BEFORE_EXPIRY_REASON = "convergence_slot_release_before_expiry"
V7_TAKE_PROFIT_EXIT_REASONS = frozenset(
    {
        "profit_protect_take_profit",
        "low_confidence_scalp_take_profit",
        "convergence_edge_captured_take_profit",
        "convergence_gap_filled_take_profit",
        "convergence_price_move_take_profit",
        "convergence_fake_convergence_model_decay",
        V7_PROFIT_LOCK_BEFORE_EXPIRY_REASON,
    }
)


@dataclass(frozen=True, slots=True)
class PositionAdjustmentResult:
    action: str
    status: str
    reason: str = ""
    realized_pnl: float = 0.0
    closed: bool = False


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
    settlement_peak_confidence_by_round_side: dict[str, float] = field(default_factory=dict)
    last_closed_settlement_by_round: dict[str, dict[str, Any]] = field(default_factory=dict)

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

    def mark_position_closed(
        self,
        round_slug: str,
        sleeve: str = "settlement",
        *,
        position: LivePosition | None = None,
        reason: str = "",
        realized_pnl: float | None = None,
        closed_at: int | None = None,
    ) -> None:
        self.open_positions.pop(_position_key(round_slug, sleeve), None)
        self.closed_rounds.add(round_slug)
        if sleeve == "settlement" and position is not None:
            self.last_closed_settlement_by_round[round_slug] = {
                "event_id": position.event_id,
                "side": position.side,
                "entry_model_probability": position.entry_model_probability,
                "entry_raw_probability": _entry_raw_probability_for_position(position),
                "entry_polymarket_price": position.entry_polymarket_price,
                "entry_mispricing_edge": position.entry_mispricing_edge,
                "reason": reason or position.last_lifecycle_reason,
                "realized_pnl": realized_pnl,
                "closed_at": closed_at,
            }

    def open_position(self, round_slug: str, sleeve: str) -> LivePosition | None:
        return self.open_positions.get(_position_key(round_slug, sleeve))

    def has_open_sleeve(self, round_slug: str, sleeve: str) -> bool:
        return self.open_position(round_slug, sleeve) is not None

    def filled_count_for_side(self, *, round_slug: str, sleeve: str, side: str) -> int:
        return self.filled_count_by_sleeve_round_side.get(
            _sleeve_side_key(round_slug, sleeve, side),
            0,
        )

    def mark_settlement_signal_confidence(self, event: SignalEvent) -> float | None:
        """Track the strongest observed settlement confidence for this round/side."""

        if event.token_probability is None:
            return None
        key = _sleeve_side_key(event.round_slug, "settlement", event.outcome_side)
        previous = self.settlement_peak_confidence_by_round_side.get(key)
        current = float(event.token_probability)
        peak = current if previous is None else max(previous, current)
        self.settlement_peak_confidence_by_round_side[key] = peak
        return peak


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


def _v7_settlement_reentry_after_exit_enabled(
    *,
    entry_gate_mode: str,
    allow_reentry_after_exit: bool,
) -> bool:
    return bool(allow_reentry_after_exit and entry_gate_mode == "v7-pnl")


def _can_attempt_settlement_entry(
    lifecycle: RoundLifecycleState,
    *,
    round_slug: str,
    settlement_position: LivePosition | None,
    entry_gate_mode: str,
    allow_reentry_after_exit: bool,
) -> bool:
    if settlement_position is not None:
        return False
    if round_slug not in lifecycle.filled_rounds:
        return True
    return (
        round_slug in lifecycle.closed_rounds
        and _v7_settlement_reentry_after_exit_enabled(
            entry_gate_mode=entry_gate_mode,
            allow_reentry_after_exit=allow_reentry_after_exit,
        )
    )


def _settlement_side_cap_skip_reason_for_entry(
    lifecycle: RoundLifecycleState,
    *,
    round_slug: str,
    side: str,
    max_filled_per_side_per_round: int,
    entry_gate_mode: str,
    allow_reentry_after_exit: bool,
) -> str | None:
    if (
        round_slug in lifecycle.closed_rounds
        and _v7_settlement_reentry_after_exit_enabled(
            entry_gate_mode=entry_gate_mode,
            allow_reentry_after_exit=allow_reentry_after_exit,
        )
    ):
        return None
    return _sleeve_side_cap_skip_reason(
        lifecycle,
        round_slug=round_slug,
        sleeve="settlement",
        side=side,
        max_filled_per_side_per_round=max_filled_per_side_per_round,
    )


def _entry_raw_probability_for_position(position: LivePosition) -> float | None:
    if position.side == "UP":
        return position.entry_p_up
    if position.side == "DOWN":
        return position.entry_p_down
    return None


def _v7_post_take_profit_reentry_skip_payload(
    *,
    lifecycle: RoundLifecycleState,
    signal: SignalEvent,
    config: V7SettlementPositionConfig | None,
    seconds_to_expiry: float | None,
) -> dict[str, Any] | None:
    if config is None or not config.post_take_profit_reentry_quality_enabled:
        return None
    previous = lifecycle.last_closed_settlement_by_round.get(signal.round_slug)
    if previous is None:
        return None
    previous_reason = str(previous.get("reason") or "")
    previous_pnl = previous.get("realized_pnl")
    if previous_reason not in V7_TAKE_PROFIT_EXIT_REASONS:
        return None
    if previous_pnl is not None and float(previous_pnl) <= 0.0:
        return None
    if (
        seconds_to_expiry is not None
        and seconds_to_expiry < config.post_take_profit_reentry_min_seconds_to_expiry
    ):
        return {
            "reason": "post_take_profit_reentry_too_close_to_expiry",
            "previous_exit": previous,
            "seconds_to_expiry": seconds_to_expiry,
            "min_seconds_to_expiry": (
                config.post_take_profit_reentry_min_seconds_to_expiry
            ),
        }
    selected_side = signal.selected_side or signal.outcome_side
    current_raw_probability = _signal_probability_for_side(signal, selected_side)
    current_model_probability = signal.token_probability
    previous_model_probability = previous.get("entry_model_probability")
    previous_raw_probability = previous.get("entry_raw_probability")
    min_model_improvement = (
        config.post_take_profit_reentry_min_model_probability_improvement
    )
    min_raw_improvement = config.post_take_profit_reentry_min_raw_probability_improvement
    model_improvement = None
    raw_improvement = None
    model_passed = False
    raw_passed = False
    if previous_model_probability is not None and current_model_probability is not None:
        model_improvement = float(current_model_probability) - float(previous_model_probability)
        model_passed = model_improvement >= min_model_improvement
    if previous_raw_probability is not None and current_raw_probability is not None:
        raw_improvement = float(current_raw_probability) - float(previous_raw_probability)
        raw_passed = raw_improvement >= min_raw_improvement
    if model_passed or raw_passed:
        return None
    return {
        "reason": "post_take_profit_reentry_quality_below_threshold",
        "previous_exit": previous,
        "selected_side": selected_side,
        "current_model_probability": current_model_probability,
        "previous_model_probability": previous_model_probability,
        "model_probability_improvement": model_improvement,
        "min_model_probability_improvement": min_model_improvement,
        "current_raw_probability": current_raw_probability,
        "previous_raw_probability": previous_raw_probability,
        "raw_probability_improvement": raw_improvement,
        "min_raw_probability_improvement": min_raw_improvement,
        "seconds_to_expiry": seconds_to_expiry,
    }


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
    low_latency_overlay_config = _low_latency_overlay_config_from_args(args)
    low_latency_overlay = _low_latency_overlay_from_args(args)
    settlement_exit_config = _settlement_exit_config_from_args(args)
    v7_position_config = _v7_settlement_position_config_from_args(args)
    v7_entry_candidate_buffer_config = _v7_entry_candidate_buffer_config_from_args(args)
    v7_entry_candidate_buffer = (
        V7EntryCandidateBuffer(v7_entry_candidate_buffer_config)
        if v7_entry_candidate_buffer_config.enabled
        else None
    )
    v7_convergence_calibration_config = _v7_convergence_calibration_config_from_args(
        args
    )
    v7_convergence_calibration_gate = (
        V7ConvergenceCalibrationGate.from_json_path(
            v7_convergence_calibration_config.path,
            config=v7_convergence_calibration_config,
        )
        if v7_convergence_calibration_config.enabled
        else None
    )
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
    client = _build_clob_client(require_api_creds=not args.paper)
    heartbeat_stop = threading.Event()
    if args.disable_heartbeat:
        _log(log_path, "heartbeat_disabled", paper=args.paper)
    else:
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
    settlement_exit_counts: dict[str, int] = {}
    errors = 0

    signal_jsonl_path = Path(args.signal_jsonl_path) if args.signal_jsonl_path else None
    signal_source: SignalSource | None = None
    signal_source_name = "duckdb"
    if args.signal_kafka_bootstrap_servers and args.signal_kafka_topic:
        signal_source = KafkaSignalSource(
            bootstrap_servers=args.signal_kafka_bootstrap_servers,
            topic=args.signal_kafka_topic,
            group_id=args.signal_kafka_group_id,
            poll_timeout_seconds=args.signal_kafka_poll_timeout_seconds,
            max_records=args.signal_kafka_max_records,
        )
        signal_source_name = "kafka"
    elif signal_jsonl_path is not None:
        signal_source = JsonlSignalSource(signal_jsonl_path)
        signal_source_name = "jsonl"
    signal_jsonl_watchdog = SignalJsonlWatchdogState(
        path=signal_jsonl_path,
        stale_warn_seconds=args.signal_jsonl_stale_warn_seconds,
    )
    no_new_round_watchdog = NoNewObservedRoundWatchdogState(
        warn_seconds=args.no_new_observed_round_warn_seconds,
        last_observed_round_at=started_at,
    )
    cursor_created_at = 0
    cursor_event_id = ""
    signal_cursor = SignalCursor()
    cursor_line_number = 0
    cursor_line_signature = ""
    if signal_source is None:
        cursor_created_at, cursor_event_id = _latest_cursor(
            args.monitoring_db_path,
            args.model_version,
        )
        cursor_payload: dict[str, Any] = {
            "created_at": cursor_created_at,
            "event_id": cursor_event_id,
        }
    else:
        signal_cursor = signal_source.latest_cursor(
            start=(
                args.signal_kafka_start
                if signal_source_name == "kafka"
                else args.signal_jsonl_start
            )
        )
        cursor_line_number = signal_cursor.position
        cursor_line_signature = signal_cursor.signature
        cursor_payload = {
            "line_number": cursor_line_number,
            "line_signature": cursor_line_signature,
            "signal_jsonl_path": str(signal_jsonl_path),
            "signal_jsonl_start": args.signal_jsonl_start,
            "signal_kafka_topic": args.signal_kafka_topic or None,
            "signal_kafka_group_id": (
                args.signal_kafka_group_id if signal_source_name == "kafka" else None
            ),
            "signal_kafka_start": (
                args.signal_kafka_start if signal_source_name == "kafka" else None
            ),
        }
    _log(
        log_path,
        "phase4_started",
        config={
            "model_version": args.model_version,
            "market_families": sorted(allowed_families) or None,
            "signal_source": signal_source_name,
            "signal_jsonl_path": str(signal_jsonl_path) if signal_jsonl_path is not None else None,
            "signal_jsonl_stale_warn_seconds": (
                args.signal_jsonl_stale_warn_seconds
                if signal_jsonl_path is not None
                else None
            ),
            "signal_kafka_topic": args.signal_kafka_topic or None,
            "signal_kafka_group_id": args.signal_kafka_group_id
            if signal_source_name == "kafka"
            else None,
            "no_new_observed_round_warn_seconds": (
                args.no_new_observed_round_warn_seconds
                if args.no_new_observed_round_warn_seconds > 0
                else None
            ),
            "edge_threshold": args.edge_threshold,
            "settlement_edge_threshold": entry_policy.effective_settlement_edge_threshold,
            "settlement_min_confidence": entry_policy.settlement_min_confidence,
            "settlement_peak_confidence_drop_tolerance": (
                entry_policy.settlement_peak_confidence_drop_tolerance
            ),
            "max_signal_age_seconds": entry_policy.max_signal_age_seconds,
            "entry_max_price_drift_from_signal": entry_policy.entry_max_price_drift_from_signal,
            "settlement_exit_policy": asdict(settlement_exit_config),
            "v7_settlement_position_policy": asdict(v7_position_config),
            "v7_entry_candidate_buffer": asdict(v7_entry_candidate_buffer_config),
            "v7_convergence_calibration_gate": (
                v7_convergence_calibration_gate.to_log_payload()
                if v7_convergence_calibration_gate is not None
                else {"enabled": False, "config": asdict(v7_convergence_calibration_config)}
            ),
            "settlement_price_gate_mode": _settlement_price_gate_mode_name(
                args.entry_gate_mode
            ),
            "low_latency_overlay": {
                **asdict(low_latency_overlay_config),
                "raw_jsonl_path": (
                    args.low_latency_overlay_raw_jsonl_path
                    if args.low_latency_overlay_raw_jsonl_path
                    else None
                ),
                "start": args.low_latency_overlay_start,
            },
            "settlement_min_entry_price": (
                None if _settlement_cost_edge_mode(args.entry_gate_mode) else args.min_entry_price
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
            "v7_settlement_allow_reentry_after_exit": (
                args.v7_settlement_allow_reentry_after_exit
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
            _check_signal_jsonl_watchdog(
                signal_jsonl_watchdog,
                log_path=log_path,
                now_ms=now_ms,
                cursor_line_number=cursor_line_number,
                open_positions=len(lifecycle.open_positions),
            )
            _check_no_new_observed_round_watchdog(
                no_new_round_watchdog,
                log_path=log_path,
                now_ms=now_ms,
                observed_round_count=len(lifecycle.observed_rounds),
                open_positions=len(lifecycle.open_positions),
                signal_source_name=signal_source_name,
            )

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
                settlement_exit_config=settlement_exit_config,
                v7_position_config=v7_position_config,
                settlement_exit_counts=settlement_exit_counts,
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

            if low_latency_overlay is not None:
                try:
                    overlay_report = low_latency_overlay.refresh()
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    _log(
                        log_path,
                        "low_latency_overlay_refresh_error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        raw_jsonl_path=args.low_latency_overlay_raw_jsonl_path,
                    )
                    time.sleep(args.poll_seconds)
                    continue
                if overlay_report.rows_read:
                    _log(
                        log_path,
                        "low_latency_overlay_refreshed",
                        report=overlay_report.to_dict(),
                        raw_jsonl_path=args.low_latency_overlay_raw_jsonl_path,
                    )

            try:
                if signal_source is None:
                    batch = _read_event_batch_after(
                        args.monitoring_db_path,
                        model_version=args.model_version,
                        after_created_at=cursor_created_at,
                        after_event_id=cursor_event_id,
                        limit=args.event_limit,
                        v6_joint_config=v6_joint_config,
                        entry_gate_mode=args.entry_gate_mode,
                        selection_now_ms=_now_ms(),
                        max_signal_age_seconds=entry_policy.max_signal_age_seconds,
                        collapse_best_per_round=v7_entry_candidate_buffer is None,
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
                    events, signal_cursor = _read_signal_source_after(
                        signal_source,
                        cursor=signal_cursor,
                        model_version=args.model_version,
                        limit=args.event_limit,
                        v6_joint_config=v6_joint_config,
                        entry_gate_mode=args.entry_gate_mode,
                        selection_now_ms=_now_ms(),
                        max_signal_age_seconds=entry_policy.max_signal_age_seconds,
                        collapse_best_per_round=v7_entry_candidate_buffer is None,
                    )
                    cursor_line_number = signal_cursor.position
                    cursor_line_signature = signal_cursor.signature
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
                    source=signal_source_name,
                    count=len(events),
                    cursor_line_number=cursor_line_number if signal_source is not None else None,
                    cursor_created_at=cursor_created_at if signal_source is None else None,
                    cursor_event_id=cursor_event_id if signal_source is None else None,
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

                event_now_ms = _now_ms()
                signal_age_seconds = (
                    max(0.0, (event_now_ms - event.ts) / 1000)
                    if event.ts > 0
                    else None
                )
                seconds_to_expiry = (event.round_end_ts - event_now_ms) / 1000
                settlement_position = lifecycle.open_position(event.round_slug, "settlement")
                if settlement_position is not None and _event_family_allowed(
                    event,
                    allowed_families,
                ):
                    _record_settlement_same_side_confirmation(
                        position=settlement_position,
                        signal=event,
                        log_path=log_path,
                        config=settlement_exit_config,
                        signal_age_seconds=signal_age_seconds,
                        max_signal_age_seconds=entry_policy.max_signal_age_seconds,
                    )
                    v7_adjustment = _maybe_v7_settlement_position_adjustment(
                        client=client,
                        position_manager=position_manager,
                        position=settlement_position,
                        signal=event,
                        log_path=log_path,
                        config=v7_position_config,
                        paper=args.paper,
                        sell_slippage=args.sell_slippage,
                        exit_order_timeout_seconds=args.exit_order_timeout_seconds,
                        monitoring_db_path=args.monitoring_db_path,
                    )
                    if v7_adjustment is not None:
                        realized_pnl += v7_adjustment.realized_pnl
                        if v7_adjustment.closed:
                            if v7_adjustment.status == "filled":
                                closes_filled += 1
                            elif v7_adjustment.status == "pending_settlement":
                                exits_pending_settlement += 1
                            elif v7_adjustment.status not in {"recommended", "hold"}:
                                exits_pending_confirmation += 1
                            lifecycle.mark_position_closed(
                                event.round_slug,
                                "settlement",
                                position=settlement_position,
                                reason=v7_adjustment.reason,
                                realized_pnl=v7_adjustment.realized_pnl,
                                closed_at=event.created_at,
                            )
                            continue
                    settlement_exit = _maybe_settlement_signal_exit(
                        client=client,
                        position_manager=position_manager,
                        position=settlement_position,
                        signal=event,
                        log_path=log_path,
                        config=settlement_exit_config,
                        v6_joint_config=v6_joint_config,
                        signal_age_seconds=signal_age_seconds,
                        max_signal_age_seconds=entry_policy.max_signal_age_seconds,
                        seconds_to_expiry=seconds_to_expiry,
                        opposite_exit_min_seconds_to_expiry=(
                            args.opposite_exit_min_seconds_to_expiry
                        ),
                        sell_slippage=args.sell_slippage,
                        exit_order_timeout_seconds=args.exit_order_timeout_seconds,
                        monitoring_db_path=args.monitoring_db_path,
                    )
                    if settlement_exit is not None:
                        settlement_exit_reason, sell_result = settlement_exit
                        _bump(settlement_exit_counts, settlement_exit_reason)
                        realized_pnl += sell_result.realized_pnl
                        if sell_result.status == "filled":
                            closes_filled += 1
                        elif sell_result.status == "settled":
                            pass
                        elif sell_result.status == "pending_settlement":
                            exits_pending_settlement += 1
                        else:
                            exits_pending_confirmation += 1
                        lifecycle.mark_position_closed(
                            event.round_slug,
                            "settlement",
                            position=settlement_position,
                            reason=settlement_exit_reason,
                            realized_pnl=sell_result.realized_pnl,
                            closed_at=event.created_at,
                        )
                        if realized_pnl <= -args.daily_loss_limit_usdc:
                            _log(log_path, "daily_loss_limit_reached", realized_pnl=realized_pnl)
                            break
                        continue
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
                if (
                    entry_policy.max_signal_age_seconds is not None
                    and signal_age_seconds is not None
                    and signal_age_seconds > entry_policy.max_signal_age_seconds
                ):
                    _bump(skipped, "signal_age_above_threshold")
                    _log(
                        log_path,
                        "entry_skipped",
                        reason="signal_age_above_threshold",
                        signal=asdict(event),
                        signal_age_seconds=signal_age_seconds,
                        max_signal_age_seconds=entry_policy.max_signal_age_seconds,
                        signal_latency_ms=_signal_latency_ms(event, now_ms),
                    )
                    continue
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
                new_observed_round = event.round_slug not in lifecycle.observed_round_set
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
                if new_observed_round:
                    _record_no_new_observed_round_watchdog_round_seen(
                        no_new_round_watchdog,
                        log_path=log_path,
                        now_ms=now_ms,
                        round_slug=event.round_slug,
                        observed_round_count=len(lifecycle.observed_rounds),
                        open_positions=len(lifecycle.open_positions),
                        signal_source_name=signal_source_name,
                    )
                if len(lifecycle.open_positions) >= args.max_combined_concurrent_positions:
                    _bump(skipped, "max_combined_concurrent_positions")
                    continue

                if _can_attempt_settlement_entry(
                    lifecycle,
                    round_slug=event.round_slug,
                    settlement_position=settlement_position,
                    entry_gate_mode=args.entry_gate_mode,
                    allow_reentry_after_exit=args.v7_settlement_allow_reentry_after_exit,
                ):
                    entry_event = event
                    entry_signal_age_seconds = signal_age_seconds
                    entry_seconds_to_expiry = seconds_to_expiry
                    if v7_entry_candidate_buffer is not None:
                        buffer_action = v7_entry_candidate_buffer.observe(
                            event,
                            now_ms=event_now_ms,
                            seconds_to_expiry=seconds_to_expiry,
                            max_signal_age_seconds=entry_policy.max_signal_age_seconds,
                        )
                        _log(
                            log_path,
                            "v7_entry_candidate_buffer",
                            signal=_v7_entry_candidate_summary(event),
                            **buffer_action.to_log_payload(),
                        )
                        if buffer_action.entry_event is None:
                            _bump(skipped, buffer_action.reason)
                            continue
                        entry_event = buffer_action.entry_event
                        entry_now_ms = _now_ms()
                        entry_signal_age_seconds = (
                            max(0.0, (entry_now_ms - entry_event.ts) / 1000)
                            if entry_event.ts > 0
                            else None
                        )
                        entry_seconds_to_expiry = (
                            entry_event.round_end_ts - entry_now_ms
                        ) / 1000
                        if (
                            entry_policy.max_signal_age_seconds is not None
                            and entry_signal_age_seconds is not None
                            and entry_signal_age_seconds > entry_policy.max_signal_age_seconds
                        ):
                            _bump(skipped, "v7_entry_candidate_signal_age_above_threshold")
                            _log(
                                log_path,
                                "entry_skipped",
                                reason="v7_entry_candidate_signal_age_above_threshold",
                                signal=asdict(entry_event),
                                signal_age_seconds=entry_signal_age_seconds,
                                max_signal_age_seconds=entry_policy.max_signal_age_seconds,
                            )
                            continue
                        release_time_skip_reason = _entry_time_window_skip_reason(
                            entry_seconds_to_expiry,
                            no_new_entry_before_expiry_seconds=(
                                args.no_new_entry_before_expiry_seconds
                            ),
                            min_seconds_to_expiry=args.min_seconds_to_expiry,
                            max_seconds_to_expiry=args.max_seconds_to_expiry,
                        )
                        if release_time_skip_reason is not None:
                            _bump(skipped, release_time_skip_reason)
                            _log(
                                log_path,
                                "entry_skipped",
                                reason=release_time_skip_reason,
                                signal=asdict(entry_event),
                                seconds_to_expiry=entry_seconds_to_expiry,
                            )
                            continue
                    settlement_peak_confidence = lifecycle.mark_settlement_signal_confidence(
                        entry_event
                    )
                    side_skip_reason = _settlement_side_cap_skip_reason_for_entry(
                        lifecycle,
                        round_slug=entry_event.round_slug,
                        side=entry_event.outcome_side,
                        max_filled_per_side_per_round=(
                            args.settlement_max_filled_per_side_per_round
                        ),
                        entry_gate_mode=args.entry_gate_mode,
                        allow_reentry_after_exit=args.v7_settlement_allow_reentry_after_exit,
                    )
                    if side_skip_reason is not None:
                        _bump(skipped, side_skip_reason)
                        _log(
                            log_path,
                            "entry_skipped",
                            reason=side_skip_reason,
                            sleeve="settlement",
                            signal=asdict(entry_event),
                            filled_side_count=lifecycle.filled_count_for_side(
                                round_slug=entry_event.round_slug,
                                sleeve="settlement",
                                side=entry_event.outcome_side,
                            ),
                            max_filled_per_side_per_round=(
                                args.settlement_max_filled_per_side_per_round
                            ),
                        )
                        continue
                    post_tp_reentry_skip = _v7_post_take_profit_reentry_skip_payload(
                        lifecycle=lifecycle,
                        signal=entry_event,
                        config=v7_position_config,
                        seconds_to_expiry=entry_seconds_to_expiry,
                    )
                    if post_tp_reentry_skip is not None:
                        post_tp_reason = str(post_tp_reentry_skip["reason"])
                        _bump(skipped, post_tp_reason)
                        _log(
                            log_path,
                            "entry_skipped",
                            sleeve="settlement",
                            signal=asdict(entry_event),
                            **post_tp_reentry_skip,
                        )
                        continue
                    entries_attempted += 1
                    lifecycle.mark_entry_attempted(entry_event.event_id)
                    position = _try_entry(
                        client=client,
                        position_manager=position_manager,
                        signal=entry_event,
                        log_path=log_path,
                        max_position_size_usdc=args.max_position_size_usdc,
                        entry_policy=entry_policy,
                        seconds_to_expiry=entry_seconds_to_expiry,
                        buy_slippage=args.buy_slippage,
                        monitoring_db_path=args.monitoring_db_path,
                        sleeve="settlement",
                        paper=args.paper,
                        entry_gate_mode=args.entry_gate_mode,
                        signal_age_seconds=entry_signal_age_seconds,
                        settlement_peak_confidence=settlement_peak_confidence,
                        low_latency_overlay=low_latency_overlay,
                        v7_position_config=v7_position_config,
                        v7_convergence_calibration_gate=v7_convergence_calibration_gate,
                    )
                    lifecycle.mark_entry_result(entry_event, position)
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
                        signal_age_seconds=signal_age_seconds,
                        low_latency_overlay=low_latency_overlay,
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
        pnl_reconciliation = _phase4_pnl_reconciliation_summary(
            log_path,
            position_manager=position_manager,
            event_ids=lifecycle.position_event_ids,
            runtime_realized_pnl=realized_pnl,
            paper=args.paper,
        )
        account_cashflow_reconciliation_required = (
            bool(pnl_reconciliation.get("account_cashflow_reconciliation_required"))
        )
        open_positions_at_shutdown = len(lifecycle.open_positions)
        lifecycle_complete = phase4_lifecycle_complete(
            errors=errors,
            entries_filled=entries_filled,
            open_positions_at_shutdown=open_positions_at_shutdown,
            exits_pending_confirmation=exits_pending_confirmation,
            exits_pending_settlement=exits_pending_settlement,
        )
        v7_pm_monitoring = _phase4_v7_pm_monitoring_summary(log_path)
        summary = {
            "phase": "phase4_real_champion_signal",
            "started_at": _iso(started_at),
            "finished_at": _iso(_now_ms()),
            "model_version": args.model_version,
            "market_families": sorted(allowed_families) or None,
            "entry_gate_mode": args.entry_gate_mode,
            "edge_threshold": args.edge_threshold,
            "settlement_edge_threshold": entry_policy.effective_settlement_edge_threshold,
            "settlement_min_confidence": entry_policy.settlement_min_confidence,
            "settlement_peak_confidence_drop_tolerance": (
                entry_policy.settlement_peak_confidence_drop_tolerance
            ),
            "max_signal_age_seconds": entry_policy.max_signal_age_seconds,
                "entry_max_price_drift_from_signal": entry_policy.entry_max_price_drift_from_signal,
                "settlement_exit_policy": asdict(settlement_exit_config),
                "v7_settlement_position_policy": asdict(v7_position_config),
                "v7_convergence_calibration_gate": (
                    v7_convergence_calibration_gate.to_log_payload()
                    if v7_convergence_calibration_gate is not None
                    else {"enabled": False, "config": asdict(v7_convergence_calibration_config)}
                ),
                "settlement_price_gate_mode": _settlement_price_gate_mode_name(
                    args.entry_gate_mode
                ),
            "settlement_min_entry_price": (
                None if _settlement_cost_edge_mode(args.entry_gate_mode) else args.min_entry_price
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
            "v7_settlement_allow_reentry_after_exit": (
                args.v7_settlement_allow_reentry_after_exit
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
            "event_incremental_pnl_usdc": pnl_reconciliation.get(
                "event_incremental_pnl_usdc"
            ),
            "exit_only_pnl_usdc": pnl_reconciliation.get("exit_only_pnl_usdc"),
            "partial_reduce_pnl_usdc": pnl_reconciliation.get("partial_reduce_pnl_usdc"),
            "close_and_settlement_pnl_usdc": pnl_reconciliation.get(
                "close_and_settlement_pnl_usdc"
            ),
            "account_cash_pnl_usdc": None,
            "pnl_reconciliation_status": pnl_reconciliation["status"],
            "pnl_reconciliation": pnl_reconciliation,
            "promotion_or_capital_sizing_evidence": False,
            "account_cashflow_reconciliation_required": (
                account_cashflow_reconciliation_required
            ),
            "decision_evidence_allowed": False,
            "decision_evidence_blockers": _decision_evidence_blockers(
                lifecycle_complete=lifecycle_complete,
                open_positions_at_shutdown=open_positions_at_shutdown,
                exits_pending_confirmation=exits_pending_confirmation,
                exits_pending_settlement=exits_pending_settlement,
                account_cashflow_reconciliation_required=(
                    account_cashflow_reconciliation_required
                ),
                paper=args.paper,
            ),
            "open_positions_at_shutdown": open_positions_at_shutdown,
            "processed_event_count": len(lifecycle.processed_event_ids),
            "attempted_entry_event_count": len(lifecycle.attempted_entry_event_ids),
            "observed_round_count": len(lifecycle.observed_rounds),
            "filled_round_count": len(lifecycle.filled_rounds),
            "closed_round_count": len(lifecycle.closed_rounds),
            "skipped": skipped,
            "settlement_exit_counts": settlement_exit_counts,
            "v7_pm_monitoring": v7_pm_monitoring,
            "signal_jsonl_watchdog": signal_jsonl_watchdog.summary(),
            "no_new_observed_round_watchdog": no_new_round_watchdog.summary(),
            "errors": errors,
            "execution_log_path": str(log_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _log(log_path, "phase4_summary", **summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase4_v7_pm_monitoring_summary(log_path: Path) -> dict[str, Any]:
    hold_edges: list[float] = []
    take_profit_candidate_evaluations = 0
    take_profit_exit_evaluations = 0
    take_profit_unexecuted_evaluations = 0
    take_profit_reason_counts: Counter[str] = Counter()
    if not log_path.exists():
        return {
            "divergence_reduce_hold_edge": _numeric_distribution([]),
            "take_profit_candidates": {
                "evaluations": 0,
                "exits": 0,
                "unexecuted": 0,
                "reason_counts": {},
            },
        }
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "v7_settlement_position_management_evaluated":
                continue
            evaluation = payload.get("evaluation")
            if not isinstance(evaluation, dict):
                continue
            reason = str(evaluation.get("reason") or "")
            action = str(evaluation.get("action") or "").upper()
            if reason == "residual_divergence_reduce":
                hold_edge = _float_or_none(evaluation.get("hold_edge"))
                if hold_edge is not None:
                    hold_edges.append(hold_edge)
            take_profit_count = _float_or_none(evaluation.get("take_profit_count")) or 0.0
            take_profit_reason = str(evaluation.get("take_profit_reason") or "")
            if take_profit_count > 0 or take_profit_reason:
                take_profit_candidate_evaluations += 1
                if action == "EXIT":
                    take_profit_exit_evaluations += 1
                else:
                    take_profit_unexecuted_evaluations += 1
                take_profit_reason_counts[take_profit_reason or reason or "unknown"] += 1
    return {
        "divergence_reduce_hold_edge": _numeric_distribution(hold_edges),
        "take_profit_candidates": {
            "evaluations": take_profit_candidate_evaluations,
            "exits": take_profit_exit_evaluations,
            "unexecuted": take_profit_unexecuted_evaluations,
            "reason_counts": dict(sorted(take_profit_reason_counts.items())),
        },
    }


def _phase4_pnl_reconciliation_summary(
    log_path: Path,
    *,
    position_manager: PositionManager,
    event_ids: set[str] | frozenset[str] | None,
    runtime_realized_pnl: float,
    paper: bool,
) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    event_pnl: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    reason_pnl: Counter[str] = Counter()
    if log_path.exists():
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = str(payload.get("event") or "")
                pnl = _phase4_event_incremental_pnl(payload)
                if pnl is None:
                    continue
                event_counts[event] += 1
                event_pnl[event] += pnl
                reason = str(payload.get("reason") or "unknown")
                reason_key = f"{event}:{reason}"
                reason_counts[reason_key] += 1
                reason_pnl[reason_key] += pnl

    partial_reduce_pnl = event_pnl["paper_v7_settlement_position_reduced"]
    close_and_settlement_pnl = sum(
        event_pnl[event]
        for event in (
            "paper_exit_filled",
            "paper_settlement_resolved",
            "paper_v7_settlement_position_reduced_to_exit",
            "exit_filled",
        )
    )
    event_incremental_pnl = sum(event_pnl.values())
    exit_only_pnl = _theoretical_pnl_from_positions(
        position_manager,
        event_ids=event_ids,
    )
    runtime_delta = runtime_realized_pnl - event_incremental_pnl
    exit_only_delta = event_incremental_pnl - exit_only_pnl
    status = _phase4_pnl_reconciliation_status(
        paper=paper,
        runtime_delta=runtime_delta,
        exit_only_delta=exit_only_delta,
    )
    account_cashflow_required = not paper
    return {
        "status": status,
        "metric_of_record": (
            "event_incremental_pnl_usdc" if paper else "account_cash_pnl_usdc"
        ),
        "runtime_realized_pnl_usdc": round(runtime_realized_pnl, 8),
        "event_incremental_pnl_usdc": round(event_incremental_pnl, 8),
        "exit_only_pnl_usdc": round(exit_only_pnl, 8),
        "close_and_settlement_pnl_usdc": round(close_and_settlement_pnl, 8),
        "partial_reduce_pnl_usdc": round(partial_reduce_pnl, 8),
        "runtime_vs_event_delta_usdc": round(runtime_delta, 8),
        "event_vs_exit_only_delta_usdc": round(exit_only_delta, 8),
        "exit_only_excludes_partial_reduces": bool(abs(partial_reduce_pnl) > 1e-9),
        "account_cashflow_reconciliation_required": account_cashflow_required,
        "event_bucket_counts": dict(sorted(event_counts.items())),
        "event_bucket_pnl_usdc": {
            key: round(value, 8) for key, value in sorted(event_pnl.items())
        },
        "reason_bucket_counts": dict(sorted(reason_counts.items())),
        "reason_bucket_pnl_usdc": {
            key: round(value, 8) for key, value in sorted(reason_pnl.items())
        },
    }


def _phase4_event_incremental_pnl(payload: dict[str, Any]) -> float | None:
    event = payload.get("event")
    if event == "paper_v7_settlement_position_reduced":
        return _float_or_none(payload.get("realized_pnl_delta"))
    if event in {
        "paper_exit_filled",
        "paper_settlement_resolved",
        "paper_v7_settlement_position_reduced_to_exit",
        "exit_filled",
    }:
        return _float_or_none(payload.get("realized_pnl"))
    return None


def _phase4_pnl_reconciliation_status(
    *,
    paper: bool,
    runtime_delta: float,
    exit_only_delta: float,
) -> str:
    if not paper:
        return "account_cashflow_required"
    if abs(runtime_delta) > 1e-6:
        return "paper_event_incremental_mismatch"
    if abs(exit_only_delta) > 1e-6:
        return "paper_event_incremental_reconciled_exit_only_differs"
    return "paper_event_incremental_reconciled"


def _check_signal_jsonl_watchdog(
    state: SignalJsonlWatchdogState,
    *,
    log_path: Path,
    now_ms: int,
    cursor_line_number: int,
    open_positions: int,
) -> None:
    if not state.enabled or state.path is None:
        return
    snapshot = _signal_jsonl_freshness_snapshot(
        state.path,
        now_ms=now_ms,
        stale_warn_seconds=state.stale_warn_seconds,
    )
    state.checks += 1
    state.last_snapshot = snapshot
    age_seconds = _float_or_none(snapshot.get("age_seconds"))
    if age_seconds is not None:
        if state.max_age_seconds is None:
            state.max_age_seconds = age_seconds
        else:
            state.max_age_seconds = max(state.max_age_seconds, age_seconds)

    if snapshot.get("fresh") is False:
        state.stale_checks += 1
        stale_at = _iso(now_ms)
        state.last_stale_at = stale_at
        if state.first_stale_at is None:
            state.first_stale_at = stale_at
        if not state.stale_active:
            state.stale_active = True
            state.stale_events += 1
            _log(
                log_path,
                "signal_jsonl_stale",
                **snapshot,
                cursor_line_number=cursor_line_number,
                open_positions=open_positions,
            )
        return

    if state.stale_active:
        state.stale_active = False
        state.recovered_events += 1
        state.last_recovered_at = _iso(now_ms)
        _log(
            log_path,
            "signal_jsonl_recovered",
            **snapshot,
            cursor_line_number=cursor_line_number,
            open_positions=open_positions,
        )


def _check_no_new_observed_round_watchdog(
    state: NoNewObservedRoundWatchdogState,
    *,
    log_path: Path,
    now_ms: int,
    observed_round_count: int,
    open_positions: int,
    signal_source_name: str,
) -> None:
    if not state.enabled:
        return
    age_seconds = max(0.0, (now_ms - state.last_observed_round_at) / 1000.0)
    state.checks += 1
    state.last_age_seconds = age_seconds
    state.max_age_seconds = (
        age_seconds
        if state.max_age_seconds is None
        else max(state.max_age_seconds, age_seconds)
    )
    if age_seconds <= state.warn_seconds:
        if state.stale_active:
            state.stale_active = False
            state.recovered_events += 1
            state.last_recovered_at = _iso(now_ms)
            _log(
                log_path,
                "no_new_observed_round_recovered",
                age_seconds=age_seconds,
                warn_seconds=state.warn_seconds,
                observed_round_count=observed_round_count,
                last_observed_round_slug=state.last_observed_round_slug,
                open_positions=open_positions,
                signal_source=signal_source_name,
                fresh=True,
            )
        return

    state.stale_checks += 1
    stale_at = _iso(now_ms)
    state.last_stale_at = stale_at
    if state.first_stale_at is None:
        state.first_stale_at = stale_at
    if state.stale_active:
        return
    state.stale_active = True
    state.stale_events += 1
    _log(
        log_path,
        "no_new_observed_round_stale",
        reason="no_new_observed_round",
        age_seconds=age_seconds,
        warn_seconds=state.warn_seconds,
        observed_round_count=observed_round_count,
        last_observed_round_slug=state.last_observed_round_slug,
        open_positions=open_positions,
        signal_source=signal_source_name,
    )


def _record_no_new_observed_round_watchdog_round_seen(
    state: NoNewObservedRoundWatchdogState,
    *,
    log_path: Path,
    now_ms: int,
    round_slug: str,
    observed_round_count: int,
    open_positions: int,
    signal_source_name: str,
) -> None:
    if not state.enabled:
        return
    previous_age_seconds = max(0.0, (now_ms - state.last_observed_round_at) / 1000.0)
    state.last_observed_round_at = now_ms
    state.last_observed_round_slug = round_slug
    state.last_age_seconds = 0.0
    if not state.stale_active:
        return
    state.stale_active = False
    state.recovered_events += 1
    state.last_recovered_at = _iso(now_ms)
    _log(
        log_path,
        "no_new_observed_round_recovered",
        age_seconds=0.0,
        previous_age_seconds=previous_age_seconds,
        warn_seconds=state.warn_seconds,
        observed_round_count=observed_round_count,
        last_observed_round_slug=round_slug,
        open_positions=open_positions,
        signal_source=signal_source_name,
        fresh=True,
    )


def _signal_jsonl_freshness_snapshot(
    path: Path,
    *,
    now_ms: int,
    stale_warn_seconds: float,
) -> dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {
            "path": str(path),
            "exists": False,
            "fresh": False,
            "stale_warn_seconds": stale_warn_seconds,
            "age_seconds": None,
            "mtime": None,
            "size_bytes": None,
            "reason": "missing_signal_jsonl",
        }
    mtime_ms = int(stat.st_mtime_ns / 1_000_000)
    age_seconds = max(0.0, (now_ms - mtime_ms) / 1000.0)
    fresh = age_seconds <= stale_warn_seconds
    return {
        "path": str(path),
        "exists": True,
        "fresh": fresh,
        "stale_warn_seconds": stale_warn_seconds,
        "age_seconds": age_seconds,
        "mtime": _iso(mtime_ms),
        "size_bytes": stat.st_size,
        "reason": None if fresh else "signal_jsonl_mtime_stale",
    }


def _numeric_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1],
    }


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if not ordered_values:
        raise ValueError("ordered_values must be non-empty")
    if len(ordered_values) == 1:
        return ordered_values[0]
    index = percentile * (len(ordered_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered_values[lower]
    fraction = index - lower
    return ordered_values[lower] + (ordered_values[upper] - ordered_values[lower]) * fraction


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    parser.add_argument(
        "--signal-jsonl-stale-warn-seconds",
        type=float,
        default=DEFAULT_SIGNAL_JSONL_STALE_WARN_SECONDS,
        help=(
            "Log signal_jsonl_stale when the executor queue file has not been "
            "modified for this many seconds. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--no-new-observed-round-warn-seconds",
        type=float,
        default=DEFAULT_NO_NEW_OBSERVED_ROUND_WARN_SECONDS,
        help=(
            "Log no_new_observed_round_stale when no new executor-observed round "
            "has appeared for this many seconds. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--signal-kafka-bootstrap-servers",
        default="",
        help="Optional Kafka bootstrap servers for executor-ready signal consumption.",
    )
    parser.add_argument(
        "--signal-kafka-topic",
        default="",
        help="Optional Kafka topic for executor-ready signal consumption.",
    )
    parser.add_argument(
        "--signal-kafka-group-id",
        default="bigan-phase4-paper-shadow",
        help="Kafka consumer group id for executor-ready signal consumption.",
    )
    parser.add_argument(
        "--signal-kafka-start",
        choices=("tail", "beginning"),
        default="tail",
        help="Kafka offset reset policy when the consumer group has no committed offset.",
    )
    parser.add_argument(
        "--signal-kafka-poll-timeout-seconds",
        type=float,
        default=0.25,
        help="Kafka poll timeout per record fetch.",
    )
    parser.add_argument(
        "--signal-kafka-max-records",
        type=int,
        default=500,
        help="Maximum Kafka records to poll per executor cycle.",
    )
    parser.add_argument(
        "--low-latency-overlay-enabled",
        action="store_true",
        help=(
            "Enable executor-side 5s/10s overlay vetoes from the raw low-latency "
            "JSONL queue. This only blocks entries; it never relaxes the base gate."
        ),
    )
    parser.add_argument(
        "--low-latency-overlay-raw-jsonl-path",
        default="",
        help="Raw queue path written by the scorer low-latency feature path.",
    )
    parser.add_argument(
        "--low-latency-overlay-start",
        choices=("tail", "beginning"),
        default="beginning",
        help="Where to start reading the raw overlay queue on executor startup.",
    )
    parser.add_argument(
        "--low-latency-overlay-max-quote-age-seconds",
        type=float,
        default=10.0,
        help="Skip otherwise-valid entries when the latest side-token quote is older than this.",
    )
    parser.add_argument(
        "--low-latency-overlay-window-seconds",
        type=float,
        default=10.0,
        help="Rolling side-token quote window used for the adverse velocity veto.",
    )
    parser.add_argument(
        "--low-latency-overlay-max-spread",
        type=float,
        default=0.05,
        help="Skip otherwise-valid entries when the latest side-token spread is wider than this.",
    )
    parser.add_argument(
        "--low-latency-overlay-adverse-velocity-threshold",
        type=float,
        default=0.04,
        help="Skip when side-token mid falls by at least this amount over the overlay window.",
    )
    parser.add_argument(
        "--low-latency-overlay-max-price-drift-from-signal",
        type=float,
        default=0.08,
        help="Skip when latest side-token ask is this much above the signal-time implied price.",
    )
    parser.add_argument(
        "--low-latency-overlay-missing-quote-action",
        choices=("pass", "skip"),
        default="pass",
        help="Whether a missing raw queue quote should pass or skip an otherwise-valid entry.",
    )
    parser.add_argument(
        "--low-latency-overlay-max-records-per-refresh",
        type=int,
        default=20_000,
        help="Maximum raw queue records consumed by each overlay refresh.",
    )
    parser.add_argument(
        "--disable-heartbeat",
        action="store_true",
        help="Disable CLOB heartbeat keepalive, mainly for orderbook-only paper shadow runs.",
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
        default=86_400.0,
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
        choices=("v5-edge", "v6-joint", "v7-pnl"),
        default="v5-edge",
        help=(
            "v5-edge uses settlement edge threshold; v6-joint uses explicit "
            "p_up/p_down/p_vol gate; v7-pnl uses selected_side plus executable "
            "fresh p_side - price edge."
        ),
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
    parser.add_argument(
        "--v7-settlement-min-edge-after-cost",
        type=float,
        default=None,
        help=(
            "Minimum fresh p_side - executable price required for xgboost-v7 "
            "settlement entries. Defaults to --settlement-edge-threshold when set, "
            "otherwise 0.04 from the current v7 PnL-stability gate."
        ),
    )
    parser.add_argument(
        "--settlement-min-confidence",
        type=float,
        default=DEFAULT_SETTLEMENT_MIN_CONFIDENCE,
        help="Minimum selected p_up/p_down required for settlement entries.",
    )
    parser.add_argument(
        "--settlement-peak-confidence-drop-tolerance",
        type=float,
        default=None,
        help=(
            "Optional v6 settlement momentum guard. Skip same-round/side entries "
            "when current p_side is more than this amount below the observed peak."
        ),
    )
    parser.add_argument(
        "--settlement-allow-mid-round-exit",
        action="store_true",
        help="Allow settlement sleeve to sell before expiry on high-confidence opposite flips.",
    )
    parser.add_argument(
        "--allow-live-settlement-mid-round-exit",
        action="store_true",
        help=(
            "Explicitly allow live settlement sleeve mid-round exits. Without this "
            "second guard, live settlement positions hold to expiry/settlement even "
            "if reversal, confidence-decay, or price-stop flags are passed."
        ),
    )
    parser.add_argument(
        "--settlement-reversal-min-confidence",
        type=float,
        default=None,
        help=(
            "Minimum opposite p_up/p_down for settlement reversal exits; defaults "
            "to data-tuned reversal confidence."
        ),
    )
    parser.add_argument(
        "--settlement-reversal-hysteresis-bars",
        type=int,
        default=DEFAULT_SETTLEMENT_REVERSAL_HYSTERESIS_BARS,
        help="Consecutive fresh opposite settlement admissions required before reversal exit.",
    )
    parser.add_argument(
        "--settlement-confidence-decay-enabled",
        action="store_true",
        help="Allow settlement sleeve to exit when same-side confidence decays on fresh signals.",
    )
    parser.add_argument("--settlement-decay-floor", type=float, default=0.55)
    parser.add_argument("--settlement-decay-delta", type=float, default=0.25)
    parser.add_argument(
        "--settlement-decay-hysteresis-bars",
        type=int,
        default=DEFAULT_SETTLEMENT_REVERSAL_HYSTERESIS_BARS,
        help=(
            "Consecutive fresh signals that must satisfy the confidence-decay exit "
            "condition before selling. Raise above 1 to ignore single-bar noise."
        ),
    )
    parser.add_argument(
        "--settlement-decay-opposite-min-confidence",
        type=float,
        default=None,
        help=(
            "Minimum opposite p_up/p_down required for confidence-decay exits; "
            "defaults to the reversal/entry confidence threshold."
        ),
    )
    parser.add_argument(
        "--settlement-price-stop-enabled",
        action="store_true",
        help="Allow settlement sleeve to exit when current bid breaches the stop-loss policy.",
    )
    parser.add_argument("--settlement-stop-price-delta", type=float, default=0.15)
    parser.add_argument("--settlement-stop-loss-usdc", type=float, default=0.50)
    parser.add_argument("--settlement-stop-min-seconds-to-expiry", type=float, default=120.0)
    parser.add_argument(
        "--settlement-price-stop-same-side-confirmation-veto-enabled",
        action="store_true",
        help=(
            "Skip a settlement price-stop exit when a fresh, post-entry same-side "
            "settlement confidence confirmation is still active."
        ),
    )
    parser.add_argument(
        "--settlement-price-stop-same-side-confirmation-min-confidence",
        type=float,
        default=None,
        help=(
            "Minimum same-side p_up/p_down needed to veto a settlement price stop; "
            "defaults to --settlement-min-confidence."
        ),
    )
    parser.add_argument(
        "--settlement-price-stop-same-side-confirmation-max-age-seconds",
        type=float,
        default=DEFAULT_MAX_SIGNAL_AGE_SECONDS,
        help="Maximum age for same-side confirmation veto; <=0 disables the age cap.",
    )
    parser.add_argument(
        "--v7-settlement-position-management-enabled",
        action="store_true",
        help="Enable v7 settlement EV position management for open settlement positions.",
    )
    parser.add_argument(
        "--v7-settlement-allow-reentry-after-exit",
        action="store_true",
        help=(
            "Allow v7-pnl settlement sleeve to open a new entry in the same round "
            "after the previous settlement position exits."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-paper-execute",
        action="store_true",
        help="Allow v7 EV position ADD/REDUCE/EXIT simulation in paper mode.",
    )
    parser.add_argument("--v7-settlement-position-round-cap-usdc", type=float, default=1.0)
    parser.add_argument("--v7-settlement-position-add-edge-min", type=float, default=0.08)
    parser.add_argument("--v7-settlement-position-full-add-edge", type=float, default=0.20)
    parser.add_argument("--v7-settlement-position-weak-hold-edge", type=float, default=0.02)
    parser.add_argument("--v7-settlement-position-reduce-fraction", type=float, default=0.50)
    parser.add_argument(
        "--v7-settlement-position-divergence-reduce-max-hold-edge",
        type=float,
        default=0.08,
        help=(
            "Only allow residual_divergence_reduce when hold_edge is below this "
            "threshold. Set negative to disable the hold-edge guard."
        ),
    )
    parser.add_argument("--v7-settlement-position-exit-hold-edge", type=float, default=-0.02)
    parser.add_argument("--v7-settlement-position-exit-hysteresis-bars", type=int, default=2)
    parser.add_argument(
        "--v7-settlement-position-reversal-min-confidence",
        type=float,
        default=0.75,
    )
    parser.add_argument("--v7-settlement-position-reversal-min-edge", type=float, default=0.04)
    parser.add_argument(
        "--v7-settlement-position-reversal-hysteresis-bars",
        type=int,
        default=2,
    )
    parser.add_argument("--v7-settlement-position-min-rebalance-usdc", type=float, default=0.05)
    parser.add_argument(
        "--v7-settlement-position-convergence-price-tolerance",
        type=float,
        default=0.02,
        help=(
            "Allowed adverse held-token price move before v7 position management "
            "treats residual as diverging."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-convergence-model-decay-tolerance",
        type=float,
        default=0.10,
        help=(
            "Allowed adverse model-probability decay before v7 position management "
            "treats residual as diverging."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-divergence-hysteresis-bars",
        type=int,
        default=2,
        help="Consecutive divergent held-token signals needed before reducing exposure.",
    )
    parser.add_argument(
        "--v7-settlement-position-add-cooldown-after-divergence-reduce-seconds",
        type=float,
        default=120.0,
        help="Block ADD recommendations for this long after a divergence reduce.",
    )
    parser.add_argument(
        "--v7-settlement-position-convergence-take-profit-enabled",
        action="store_true",
        help="Exit settlement positions when convergence edge is captured instead of holding to settlement.",
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-hold-edge",
        type=float,
        default=0.03,
        help="Take profit when executable hold_edge falls to this level or below.",
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-residual-ratio",
        type=float,
        default=0.40,
        help="Take profit when price_converged and residual_abs_ratio is at or below this level.",
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-price-convergence-move",
        type=float,
        default=0.10,
        help=(
            "Take profit when held-token price has moved this far toward the "
            "entry model direction and executable hold edge has compressed."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-price-convergence-hold-edge-ratio",
        type=float,
        default=0.50,
        help=(
            "Take profit on price convergence when hold_edge is at or below this "
            "fraction of the absolute entry residual."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-force-exit-seconds",
        type=float,
        default=180.0,
        help="Force take-profit exit when seconds_to_expiry is at or below this threshold.",
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-hysteresis-bars",
        type=int,
        default=2,
        help="Consecutive take-profit candidate evaluations required before EXIT.",
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-up-hold-edge-tighten",
        type=float,
        default=0.01,
        help="Tighten take-profit hold_edge threshold for UP legs by this amount.",
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-min-profit-delta",
        type=float,
        default=0.10,
        help=(
            "Take profit when executable held-token bid is this much above "
            "average entry price. Set 0 to disable this profit-protect branch."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-take-profit-min-profit-return",
        type=float,
        default=0.35,
        help=(
            "Take profit when executable held-token bid return from average "
            "entry price reaches this ratio. Set 0 to disable this branch."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-low-confidence-scalp-enabled",
        action="store_true",
        help=(
            "For weak/negative confidence signals, tighten take-profit handling and "
            "optionally upgrade adverse-confidence reduces to exits."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-low-confidence-scalp-max-confidence-score",
        type=float,
        default=0.0,
        help="Maximum selected confidence score treated as low-confidence scalp mode.",
    )
    parser.add_argument(
        "--v7-settlement-position-low-confidence-scalp-take-profit-min-profit-delta",
        type=float,
        default=0.05,
        help="Profit delta used by low-confidence scalp take-profit mode.",
    )
    parser.add_argument(
        "--v7-settlement-position-low-confidence-scalp-take-profit-min-profit-return",
        type=float,
        default=0.10,
        help="Profit return used by low-confidence scalp take-profit mode.",
    )
    parser.add_argument(
        "--v7-settlement-position-low-confidence-scalp-take-profit-hysteresis-bars",
        type=int,
        default=1,
        help="Take-profit confirmation bars used by low-confidence scalp mode.",
    )
    parser.add_argument(
        "--v7-settlement-position-low-confidence-scalp-adverse-full-exit-enabled",
        action="store_true",
        help="Upgrade low-confidence adverse-confidence reduce candidates to full exits.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-decay-enabled",
        action="store_true",
        help=(
            "When held-token price moves adversely, require current p_side to stay "
            "near entry p_side before hold_edge can protect the position."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-price-delta-start",
        type=float,
        default=0.10,
        help="Held-token bid drop from average price that starts the adverse confidence gate.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-base-allowed-decay",
        type=float,
        default=0.08,
        help="Allowed entry_p_side to current p_side decay before adverse price adjustment.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-price-decay-slope",
        type=float,
        default=0.30,
        help="Amount to reduce allowed model decay per 1.00 of adverse held-token price move.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-min-allowed-decay",
        type=float,
        default=0.015,
        help="Floor on allowed model decay once adverse price pressure is large.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-max-required-probability",
        type=float,
        default=0.97,
        help="Cap for dynamically required p_side under adverse price pressure.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-exit-probability-buffer",
        type=float,
        default=0.03,
        help="Exit instead of reduce when p_side is this far below dynamic requirement.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-full-exit-min-model-decay",
        type=float,
        default=0.06,
        help="Minimum entry_p_side to current p_side decay required before adverse-confidence full exit.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-full-exit-max-hold-edge",
        type=float,
        default=0.25,
        help=(
            "Maximum executable hold_edge that still permits adverse-confidence "
            "full exit. Set negative to disable this guard."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-reduce-min-model-decay",
        type=float,
        default=0.06,
        help=(
            "Minimum entry_p_side to current p_side decay required before "
            "adverse-confidence can partially reduce instead of holding."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-dust-exit-max-cost",
        type=float,
        default=0.15,
        help=(
            "Exit instead of repeatedly reducing when adverse-confidence has "
            "already cut remaining cost basis to this level or below. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-dust-exit-min-candidate-count",
        type=int,
        default=3,
        help="Adverse-confidence candidate count required before dust cleanup full exit.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-hysteresis-bars",
        type=int,
        default=2,
        help="Consecutive adverse-confidence failures needed before REDUCE or EXIT.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-max-reduces",
        type=int,
        default=0,
        help=(
            "Maximum adverse-confidence REDUCE fills per position. Set 0 for "
            "unlimited legacy behavior."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-post-reduce-full-exit-enabled",
        action="store_true",
        help=(
            "After the configured adverse-confidence reduce budget is exhausted, "
            "upgrade continued adverse-confidence failures to a model-based full exit."
        ),
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-post-reduce-full-exit-bars",
        type=int,
        default=1,
        help="Additional adverse-confidence bars after a reduce before the upgrade full exit.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-post-reduce-full-exit-min-model-decay",
        type=float,
        default=0.06,
        help="Minimum model decay required for the post-reduce full-exit upgrade.",
    )
    parser.add_argument(
        "--v7-settlement-position-adverse-confidence-post-reduce-full-exit-max-hold-edge",
        type=float,
        default=-1.0,
        help="Maximum hold_edge for post-reduce full exit; negative disables this guard.",
    )
    parser.add_argument(
        "--v7-settlement-position-block-add-after-adverse-confidence-reduce",
        action="store_true",
        help="Block ADD recommendations after this position has had an adverse-confidence reduce.",
    )
    parser.add_argument(
        "--v7-settlement-position-post-take-profit-reentry-quality-enabled",
        action="store_true",
        help="Require stronger model/raw confirmation for same-round re-entry after a profitable TP exit.",
    )
    parser.add_argument(
        "--v7-settlement-position-post-take-profit-reentry-min-model-probability-improvement",
        type=float,
        default=0.03,
        help="Required token_probability improvement versus the prior TP entry.",
    )
    parser.add_argument(
        "--v7-settlement-position-post-take-profit-reentry-min-raw-probability-improvement",
        type=float,
        default=0.02,
        help="Required raw p_side improvement versus the prior TP entry.",
    )
    parser.add_argument(
        "--v7-settlement-position-post-take-profit-reentry-min-seconds-to-expiry",
        type=float,
        default=420.0,
        help="Minimum time left for same-round re-entry after a profitable TP exit.",
    )
    parser.add_argument(
        "--v7-entry-candidate-buffer-enabled",
        action="store_true",
        help=(
            "For v7-pnl settlement entries, collect same-round candidates for a short "
            "window and release the best confidence/price-band candidate instead of "
            "entering the first eligible signal."
        ),
    )
    parser.add_argument(
        "--v7-entry-candidate-buffer-max-wait-seconds",
        type=float,
        default=45.0,
        help="Maximum time to wait before releasing the best v7 entry candidate.",
    )
    parser.add_argument(
        "--v7-entry-candidate-buffer-min-price",
        type=float,
        default=0.40,
        help="Minimum signal-time price admitted to the v7 entry candidate buffer.",
    )
    parser.add_argument(
        "--v7-entry-candidate-buffer-max-price",
        type=float,
        default=0.70,
        help="Maximum signal-time price admitted to the v7 entry candidate buffer.",
    )
    parser.add_argument(
        "--v7-entry-candidate-buffer-min-edge",
        type=float,
        default=0.04,
        help="Minimum signal-time edge admitted to the v7 entry candidate buffer.",
    )
    parser.add_argument(
        "--v7-entry-candidate-buffer-min-seconds-to-expiry",
        type=float,
        default=330.0,
        help=(
            "Release the best buffered v7 entry candidate once time-to-expiry reaches "
            "this value, provided the normal entry time window still allows entries."
        ),
    )
    parser.add_argument(
        "--v7-entry-candidate-buffer-immediate-confidence-score",
        type=float,
        default=None,
        help=(
            "Optional confidence score that releases the best buffered candidate "
            "immediately. Omit to always wait for the window/expiry release."
        ),
    )
    parser.add_argument(
        "--v7-entry-candidate-buffer-max-candidates-per-round",
        type=int,
        default=64,
        help="Maximum retained v7 entry candidates per round.",
    )
    parser.add_argument(
        "--max-signal-age-seconds",
        type=float,
        default=DEFAULT_MAX_SIGNAL_AGE_SECONDS,
        help="Maximum executor receive time minus signal ts for new entries; <=0 disables.",
    )
    parser.add_argument(
        "--entry-max-price-drift-from-signal",
        type=float,
        default=None,
        help=(
            "Skip v7-pnl settlement entries when worst-case entry price exceeds "
            "signal polymarket price by more than this amount."
        ),
    )
    parser.add_argument(
        "--v7-raw-side-agreement-enabled",
        action="store_true",
        help="For v7-pnl entries, require raw p_up/p_down to agree with the selected side.",
    )
    parser.add_argument(
        "--v7-raw-side-min-probability",
        type=float,
        default=None,
        help="Minimum raw p_side for the selected v7-pnl side when raw-side agreement is enabled.",
    )
    parser.add_argument(
        "--v7-raw-side-min-margin",
        type=float,
        default=None,
        help=(
            "Minimum raw p_side - p_opposite margin for the selected v7-pnl side "
            "when raw-side agreement is enabled."
        ),
    )
    parser.add_argument(
        "--v7-raw-side-max-opposite-lead",
        type=float,
        default=None,
        help=(
            "Maximum allowed raw p_opposite - p_side for the selected v7-pnl side "
            "when raw-side agreement is enabled."
        ),
    )
    parser.add_argument("--v7-raw-side-price-conviction-enabled", action="store_true")
    parser.add_argument("--v7-raw-side-price-conviction-min-price", type=float, default=0.40)
    parser.add_argument("--v7-raw-side-price-conviction-center-price", type=float, default=0.50)
    parser.add_argument("--v7-raw-side-price-conviction-max-price", type=float, default=0.70)
    parser.add_argument(
        "--v7-raw-side-price-conviction-center-min-probability",
        type=float,
        default=None,
        help="Dynamic raw p_side threshold at the center of the executable price band.",
    )
    parser.add_argument(
        "--v7-convergence-calibration-path",
        default="",
        help=(
            "Optional replay calibration artifact for v7-pnl entry convergence "
            "quality gating."
        ),
    )
    parser.add_argument(
        "--v7-convergence-calibration-min-hit-5c-rate",
        type=float,
        default=0.0,
        help="Minimum historical +5c convergence hit rate for a v7-pnl entry bucket.",
    )
    parser.add_argument(
        "--v7-convergence-calibration-min-hit-10c-rate",
        type=float,
        default=0.0,
        help="Minimum historical +10c convergence hit rate for a v7-pnl entry bucket.",
    )
    parser.add_argument(
        "--v7-convergence-calibration-max-model-over-error-p80",
        type=float,
        default=None,
        help=(
            "Maximum allowed p80 model overprediction error for a v7-pnl entry "
            "bucket. Omit to disable this calibration gate."
        ),
    )
    parser.add_argument(
        "--v7-convergence-calibration-min-adjusted-median-edge",
        type=float,
        default=None,
        help=(
            "Minimum edge after adjusting model value by bucket median value "
            "error and comparing against actual execution price."
        ),
    )
    parser.add_argument(
        "--v7-convergence-calibration-min-adjusted-p80-edge",
        type=float,
        default=None,
        help=(
            "Minimum edge after subtracting bucket p80 model overprediction "
            "error from model value and comparing against actual execution price."
        ),
    )
    parser.add_argument(
        "--v7-convergence-calibration-min-bucket-sample-count",
        type=int,
        default=20,
        help="Minimum samples required before using a non-global calibration bucket.",
    )
    return parser.parse_args()


def _entry_policy_from_args(args: argparse.Namespace) -> Phase4EntryPolicy:
    v6_settlement_edge_threshold = (
        args.v6_settlement_min_edge_after_cost
        if args.v6_settlement_min_edge_after_cost is not None
        else args.v6_round_trip_cost + args.v6_ev_margin
    )
    v7_settlement_edge_threshold = (
        getattr(args, "v7_settlement_min_edge_after_cost", None)
        if getattr(args, "v7_settlement_min_edge_after_cost", None) is not None
        else (
            args.settlement_edge_threshold
            if args.settlement_edge_threshold is not None
            else 0.04
        )
    )
    max_signal_age_seconds = getattr(
        args,
        "max_signal_age_seconds",
        DEFAULT_MAX_SIGNAL_AGE_SECONDS,
    )
    cost_edge_only = args.entry_gate_mode in {"v6-joint", "v7-pnl"}
    return Phase4EntryPolicy(
        min_entry_price=args.min_entry_price,
        near_min_price_band=args.near_min_price_band,
        near_min_fresh_edge_threshold=args.near_min_fresh_edge_threshold,
        near_min_seconds_to_expiry=args.near_min_seconds_to_expiry,
        edge_threshold=-999.0 if cost_edge_only else args.edge_threshold,
        settlement_edge_threshold=(
            v6_settlement_edge_threshold
            if args.entry_gate_mode == "v6-joint"
            else (
                v7_settlement_edge_threshold
                if args.entry_gate_mode == "v7-pnl"
                else args.settlement_edge_threshold
            )
        ),
        volatility_score_threshold=args.volatility_score_threshold,
        volatility_min_entry_price=args.volatility_min_entry_price,
        volatility_min_seconds_to_expiry=args.volatility_min_seconds_to_expiry,
        volatility_round_trip_cost=args.volatility_round_trip_cost,
        volatility_safety_margin=args.volatility_safety_margin,
        enable_volatility_live_entries=args.enable_volatility_live_entries,
        settlement_min_confidence=getattr(
            args,
            "settlement_min_confidence",
            DEFAULT_SETTLEMENT_MIN_CONFIDENCE,
        ),
        max_signal_age_seconds=(
            None if max_signal_age_seconds <= 0 else max_signal_age_seconds
        ),
        settlement_peak_confidence_drop_tolerance=getattr(
            args,
            "settlement_peak_confidence_drop_tolerance",
            None,
        ),
        entry_max_price_drift_from_signal=getattr(
            args,
            "entry_max_price_drift_from_signal",
            None,
        ),
        v7_raw_side_agreement_enabled=bool(
            getattr(args, "v7_raw_side_agreement_enabled", False)
        ),
        v7_raw_side_min_probability=getattr(
            args,
            "v7_raw_side_min_probability",
            None,
        ),
        v7_raw_side_min_margin=getattr(
            args,
            "v7_raw_side_min_margin",
            None,
        ),
        v7_raw_side_max_opposite_lead=getattr(
            args,
            "v7_raw_side_max_opposite_lead",
            None,
        ),
        v7_raw_side_price_conviction_enabled=bool(
            getattr(args, "v7_raw_side_price_conviction_enabled", False)
        ),
        v7_raw_side_price_conviction_min_price=getattr(
            args,
            "v7_raw_side_price_conviction_min_price",
            0.40,
        ),
        v7_raw_side_price_conviction_center_price=getattr(
            args,
            "v7_raw_side_price_conviction_center_price",
            0.50,
        ),
        v7_raw_side_price_conviction_max_price=getattr(
            args,
            "v7_raw_side_price_conviction_max_price",
            0.70,
        ),
        v7_raw_side_price_conviction_center_min_probability=getattr(
            args,
            "v7_raw_side_price_conviction_center_min_probability",
            None,
        ),
    )


def _v7_convergence_calibration_config_from_args(
    args: argparse.Namespace,
) -> V7ConvergenceCalibrationConfig:
    return V7ConvergenceCalibrationConfig(
        path=str(getattr(args, "v7_convergence_calibration_path", "") or ""),
        min_hit_5c_rate=float(
            getattr(args, "v7_convergence_calibration_min_hit_5c_rate", 0.0)
        ),
        min_hit_10c_rate=float(
            getattr(args, "v7_convergence_calibration_min_hit_10c_rate", 0.0)
        ),
        max_model_over_error_p80=getattr(
            args,
            "v7_convergence_calibration_max_model_over_error_p80",
            None,
        ),
        min_adjusted_median_edge=getattr(
            args,
            "v7_convergence_calibration_min_adjusted_median_edge",
            None,
        ),
        min_adjusted_p80_edge=getattr(
            args,
            "v7_convergence_calibration_min_adjusted_p80_edge",
            None,
        ),
        min_bucket_sample_count=int(
            getattr(args, "v7_convergence_calibration_min_bucket_sample_count", 20)
        ),
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


def _v7_settlement_position_config_from_args(args: argparse.Namespace) -> V7SettlementPositionConfig:
    return V7SettlementPositionConfig(
        enabled=bool(args.v7_settlement_position_management_enabled),
        paper_execute=bool(args.v7_settlement_position_paper_execute),
        round_cap_usdc=args.v7_settlement_position_round_cap_usdc,
        add_edge_min=args.v7_settlement_position_add_edge_min,
        full_add_edge=args.v7_settlement_position_full_add_edge,
        weak_hold_edge=args.v7_settlement_position_weak_hold_edge,
        reduce_fraction=args.v7_settlement_position_reduce_fraction,
        divergence_reduce_max_hold_edge=(
            args.v7_settlement_position_divergence_reduce_max_hold_edge
        ),
        exit_hold_edge=args.v7_settlement_position_exit_hold_edge,
        exit_hysteresis_bars=args.v7_settlement_position_exit_hysteresis_bars,
        reversal_min_confidence=args.v7_settlement_position_reversal_min_confidence,
        reversal_min_edge=args.v7_settlement_position_reversal_min_edge,
        reversal_hysteresis_bars=args.v7_settlement_position_reversal_hysteresis_bars,
        min_rebalance_usdc=args.v7_settlement_position_min_rebalance_usdc,
        convergence_price_tolerance=args.v7_settlement_position_convergence_price_tolerance,
        convergence_model_decay_tolerance=(
            args.v7_settlement_position_convergence_model_decay_tolerance
        ),
        divergence_hysteresis_bars=args.v7_settlement_position_divergence_hysteresis_bars,
        add_cooldown_after_divergence_reduce_seconds=(
            args.v7_settlement_position_add_cooldown_after_divergence_reduce_seconds
        ),
        convergence_take_profit_enabled=bool(
            args.v7_settlement_position_convergence_take_profit_enabled
        ),
        take_profit_hold_edge=args.v7_settlement_position_take_profit_hold_edge,
        take_profit_residual_ratio=args.v7_settlement_position_take_profit_residual_ratio,
        take_profit_price_convergence_move=(
            args.v7_settlement_position_take_profit_price_convergence_move
        ),
        take_profit_price_convergence_hold_edge_ratio=(
            args.v7_settlement_position_take_profit_price_convergence_hold_edge_ratio
        ),
        take_profit_force_exit_seconds=args.v7_settlement_position_take_profit_force_exit_seconds,
        take_profit_hysteresis_bars=args.v7_settlement_position_take_profit_hysteresis_bars,
        take_profit_up_hold_edge_tighten=args.v7_settlement_position_take_profit_up_hold_edge_tighten,
        take_profit_min_profit_delta=(
            args.v7_settlement_position_take_profit_min_profit_delta
        ),
        take_profit_min_profit_return=(
            args.v7_settlement_position_take_profit_min_profit_return
        ),
        low_confidence_scalp_enabled=bool(
            args.v7_settlement_position_low_confidence_scalp_enabled
        ),
        low_confidence_scalp_max_confidence_score=(
            args.v7_settlement_position_low_confidence_scalp_max_confidence_score
        ),
        low_confidence_scalp_take_profit_min_profit_delta=(
            args.v7_settlement_position_low_confidence_scalp_take_profit_min_profit_delta
        ),
        low_confidence_scalp_take_profit_min_profit_return=(
            args.v7_settlement_position_low_confidence_scalp_take_profit_min_profit_return
        ),
        low_confidence_scalp_take_profit_hysteresis_bars=(
            args.v7_settlement_position_low_confidence_scalp_take_profit_hysteresis_bars
        ),
        low_confidence_scalp_adverse_full_exit_enabled=bool(
            args.v7_settlement_position_low_confidence_scalp_adverse_full_exit_enabled
        ),
        adverse_confidence_decay_enabled=bool(
            args.v7_settlement_position_adverse_confidence_decay_enabled
        ),
        adverse_confidence_price_delta_start=(
            args.v7_settlement_position_adverse_confidence_price_delta_start
        ),
        adverse_confidence_base_allowed_decay=(
            args.v7_settlement_position_adverse_confidence_base_allowed_decay
        ),
        adverse_confidence_price_decay_slope=(
            args.v7_settlement_position_adverse_confidence_price_decay_slope
        ),
        adverse_confidence_min_allowed_decay=(
            args.v7_settlement_position_adverse_confidence_min_allowed_decay
        ),
        adverse_confidence_max_required_probability=(
            args.v7_settlement_position_adverse_confidence_max_required_probability
        ),
        adverse_confidence_exit_probability_buffer=(
            args.v7_settlement_position_adverse_confidence_exit_probability_buffer
        ),
        adverse_confidence_full_exit_min_model_decay=(
            args.v7_settlement_position_adverse_confidence_full_exit_min_model_decay
        ),
        adverse_confidence_full_exit_max_hold_edge=(
            args.v7_settlement_position_adverse_confidence_full_exit_max_hold_edge
        ),
        adverse_confidence_reduce_min_model_decay=(
            args.v7_settlement_position_adverse_confidence_reduce_min_model_decay
        ),
        adverse_confidence_dust_exit_max_cost=(
            args.v7_settlement_position_adverse_confidence_dust_exit_max_cost
        ),
        adverse_confidence_dust_exit_min_candidate_count=(
            args.v7_settlement_position_adverse_confidence_dust_exit_min_candidate_count
        ),
        adverse_confidence_hysteresis_bars=(
            args.v7_settlement_position_adverse_confidence_hysteresis_bars
        ),
        adverse_confidence_max_reduces=(
            args.v7_settlement_position_adverse_confidence_max_reduces
        ),
        adverse_confidence_post_reduce_full_exit_enabled=bool(
            args.v7_settlement_position_adverse_confidence_post_reduce_full_exit_enabled
        ),
        adverse_confidence_post_reduce_full_exit_bars=(
            args.v7_settlement_position_adverse_confidence_post_reduce_full_exit_bars
        ),
        adverse_confidence_post_reduce_full_exit_min_model_decay=(
            args.v7_settlement_position_adverse_confidence_post_reduce_full_exit_min_model_decay
        ),
        adverse_confidence_post_reduce_full_exit_max_hold_edge=(
            args.v7_settlement_position_adverse_confidence_post_reduce_full_exit_max_hold_edge
        ),
        block_add_after_adverse_confidence_reduce=bool(
            args.v7_settlement_position_block_add_after_adverse_confidence_reduce
        ),
        post_take_profit_reentry_quality_enabled=bool(
            args.v7_settlement_position_post_take_profit_reentry_quality_enabled
        ),
        post_take_profit_reentry_min_model_probability_improvement=(
            args.v7_settlement_position_post_take_profit_reentry_min_model_probability_improvement
        ),
        post_take_profit_reentry_min_raw_probability_improvement=(
            args.v7_settlement_position_post_take_profit_reentry_min_raw_probability_improvement
        ),
        post_take_profit_reentry_min_seconds_to_expiry=(
            args.v7_settlement_position_post_take_profit_reentry_min_seconds_to_expiry
        ),
    )


def _v7_entry_candidate_buffer_config_from_args(
    args: argparse.Namespace,
) -> V7EntryCandidateBufferConfig:
    return V7EntryCandidateBufferConfig(
        enabled=bool(args.v7_entry_candidate_buffer_enabled),
        max_wait_seconds=args.v7_entry_candidate_buffer_max_wait_seconds,
        min_price=args.v7_entry_candidate_buffer_min_price,
        max_price=args.v7_entry_candidate_buffer_max_price,
        min_edge=args.v7_entry_candidate_buffer_min_edge,
        min_seconds_to_expiry=args.v7_entry_candidate_buffer_min_seconds_to_expiry,
        immediate_confidence_score=args.v7_entry_candidate_buffer_immediate_confidence_score,
        max_candidates_per_round=args.v7_entry_candidate_buffer_max_candidates_per_round,
    )


def _is_v7_model_version(model_version: str) -> bool:
    return model_version == "xgboost-v7" or model_version.startswith("xgboost-v7:")


def _low_latency_overlay_config_from_args(
    args: argparse.Namespace,
) -> LowLatencyOverlayConfig:
    return LowLatencyOverlayConfig(
        enabled=bool(args.low_latency_overlay_enabled),
        max_quote_age_seconds=args.low_latency_overlay_max_quote_age_seconds,
        window_seconds=args.low_latency_overlay_window_seconds,
        max_spread=args.low_latency_overlay_max_spread,
        adverse_velocity_threshold=args.low_latency_overlay_adverse_velocity_threshold,
        max_price_drift_from_signal=args.low_latency_overlay_max_price_drift_from_signal,
        missing_quote_action=args.low_latency_overlay_missing_quote_action,
        max_records_per_refresh=args.low_latency_overlay_max_records_per_refresh,
    )


def _low_latency_overlay_from_args(
    args: argparse.Namespace,
) -> LowLatencyEntryOverlay | None:
    config = _low_latency_overlay_config_from_args(args)
    if not config.enabled:
        return None
    if not args.low_latency_overlay_raw_jsonl_path:
        raise ValueError(
            "--low-latency-overlay-raw-jsonl-path is required when overlay is enabled"
        )
    return LowLatencyEntryOverlay(
        args.low_latency_overlay_raw_jsonl_path,
        config=config,
        start=args.low_latency_overlay_start,
    )


def _settlement_cost_edge_mode(entry_gate_mode: str) -> bool:
    return entry_gate_mode in {"v6-joint", "v7-pnl"}


def _settlement_price_gate_mode_name(entry_gate_mode: str) -> str:
    if entry_gate_mode == "v7-pnl":
        return "v7_pnl_edge_only"
    if entry_gate_mode == "v6-joint":
        return "cost_edge_only"
    return "legacy_min_entry"


def _v7_entry_price_floor_skip_reason(
    *,
    ask: float,
    worst_price: float,
    policy: Phase4EntryPolicy,
) -> str | None:
    if ask < policy.min_entry_price or worst_price < policy.min_entry_price:
        return "entry_price_below_min"
    return None


def _v7_signal_reference_price(signal: SignalEvent) -> float | None:
    if signal.polymarket_price is not None:
        return float(signal.polymarket_price)
    if signal.market_implied_prob is not None:
        return float(signal.market_implied_prob)
    return None


def _settlement_exit_config_from_args(args: argparse.Namespace) -> SettlementExitConfig:
    reversal_min_confidence = (
        DEFAULT_SETTLEMENT_REVERSAL_MIN_CONFIDENCE
        if args.settlement_reversal_min_confidence is None
        else args.settlement_reversal_min_confidence
    )
    decay_opposite_min_confidence = (
        reversal_min_confidence
        if args.settlement_decay_opposite_min_confidence is None
        else args.settlement_decay_opposite_min_confidence
    )
    same_side_confirmation_min_confidence = (
        args.settlement_min_confidence
        if args.settlement_price_stop_same_side_confirmation_min_confidence is None
        else args.settlement_price_stop_same_side_confirmation_min_confidence
    )
    same_side_confirmation_max_age_seconds = (
        None
        if args.settlement_price_stop_same_side_confirmation_max_age_seconds <= 0
        else args.settlement_price_stop_same_side_confirmation_max_age_seconds
    )
    return SettlementExitConfig(
        allow_mid_round_exit=args.settlement_allow_mid_round_exit,
        reversal_min_confidence=reversal_min_confidence,
        reversal_hysteresis_bars=args.settlement_reversal_hysteresis_bars,
        confidence_decay_enabled=args.settlement_confidence_decay_enabled,
        decay_floor=args.settlement_decay_floor,
        decay_delta=args.settlement_decay_delta,
        decay_hysteresis_bars=args.settlement_decay_hysteresis_bars,
        decay_opposite_min_confidence=decay_opposite_min_confidence,
        price_stop_enabled=args.settlement_price_stop_enabled,
        stop_price_delta=args.settlement_stop_price_delta,
        stop_loss_usdc=args.settlement_stop_loss_usdc,
        stop_min_seconds_to_expiry=args.settlement_stop_min_seconds_to_expiry,
        price_stop_same_side_confirmation_veto_enabled=(
            args.settlement_price_stop_same_side_confirmation_veto_enabled
        ),
        price_stop_same_side_confirmation_min_confidence=(
            same_side_confirmation_min_confidence
        ),
        price_stop_same_side_confirmation_max_age_seconds=(
            same_side_confirmation_max_age_seconds
        ),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.settlement_max_filled_per_side_per_round <= 0:
        raise ValueError("--settlement-max-filled-per-side-per-round must be positive")
    if args.paper_settlement_timeout_seconds <= 0:
        raise ValueError("--paper-settlement-timeout-seconds must be positive")
    if not 0.0 <= args.settlement_min_confidence <= 1.0:
        raise ValueError("--settlement-min-confidence must be between 0 and 1")
    if (
        args.settlement_peak_confidence_drop_tolerance is not None
        and args.settlement_peak_confidence_drop_tolerance < 0
    ):
        raise ValueError("--settlement-peak-confidence-drop-tolerance must be non-negative")
    if (
        args.settlement_reversal_min_confidence is not None
        and not 0.0 <= args.settlement_reversal_min_confidence <= 1.0
    ):
        raise ValueError("--settlement-reversal-min-confidence must be between 0 and 1")
    if args.settlement_reversal_hysteresis_bars <= 0:
        raise ValueError("--settlement-reversal-hysteresis-bars must be positive")
    if not 0.0 <= args.settlement_decay_floor <= 1.0:
        raise ValueError("--settlement-decay-floor must be between 0 and 1")
    if args.settlement_decay_delta < 0:
        raise ValueError("--settlement-decay-delta must be non-negative")
    if args.settlement_decay_hysteresis_bars <= 0:
        raise ValueError("--settlement-decay-hysteresis-bars must be positive")
    if (
        args.settlement_decay_opposite_min_confidence is not None
        and not 0.0 <= args.settlement_decay_opposite_min_confidence <= 1.0
    ):
        raise ValueError("--settlement-decay-opposite-min-confidence must be between 0 and 1")
    if args.settlement_stop_price_delta < 0:
        raise ValueError("--settlement-stop-price-delta must be non-negative")
    if args.settlement_stop_loss_usdc < 0:
        raise ValueError("--settlement-stop-loss-usdc must be non-negative")
    if args.settlement_stop_min_seconds_to_expiry < 0:
        raise ValueError("--settlement-stop-min-seconds-to-expiry must be non-negative")
    if (
        args.settlement_price_stop_same_side_confirmation_min_confidence is not None
        and not 0.0
        <= args.settlement_price_stop_same_side_confirmation_min_confidence
        <= 1.0
    ):
        raise ValueError(
            "--settlement-price-stop-same-side-confirmation-min-confidence must be between 0 and 1"
        )
    if args.settlement_price_stop_same_side_confirmation_max_age_seconds < 0:
        raise ValueError(
            "--settlement-price-stop-same-side-confirmation-max-age-seconds must be non-negative"
        )
    if args.entry_gate_mode == "v6-joint" and not is_v6_model_version(args.model_version):
        raise ValueError("--entry-gate-mode v6-joint requires model_version xgboost-v6")
    if args.entry_gate_mode == "v7-pnl" and not _is_v7_model_version(args.model_version):
        raise ValueError("--entry-gate-mode v7-pnl requires model_version xgboost-v7")
    if bool(args.signal_kafka_bootstrap_servers) != bool(args.signal_kafka_topic):
        raise ValueError(
            "pass both --signal-kafka-bootstrap-servers and --signal-kafka-topic, or neither"
        )
    if args.signal_kafka_poll_timeout_seconds < 0:
        raise ValueError("--signal-kafka-poll-timeout-seconds must be non-negative")
    if args.signal_kafka_max_records <= 0:
        raise ValueError("--signal-kafka-max-records must be positive")
    if (
        args.v7_settlement_min_edge_after_cost is not None
        and args.v7_settlement_min_edge_after_cost < 0
    ):
        raise ValueError("--v7-settlement-min-edge-after-cost must be non-negative")
    if (
        args.v7_raw_side_min_probability is not None
        and not 0.0 <= args.v7_raw_side_min_probability <= 1.0
    ):
        raise ValueError("--v7-raw-side-min-probability must be between 0 and 1")
    if (
        args.v7_raw_side_max_opposite_lead is not None
        and args.v7_raw_side_max_opposite_lead < 0
    ):
        raise ValueError("--v7-raw-side-max-opposite-lead must be non-negative")
    if args.v7_raw_side_min_margin is not None and args.v7_raw_side_min_margin < 0:
        raise ValueError("--v7-raw-side-min-margin must be non-negative")
    if (
        args.v7_raw_side_price_conviction_center_min_probability is not None
        and not 0.0 <= args.v7_raw_side_price_conviction_center_min_probability <= 1.0
    ):
        raise ValueError(
            "--v7-raw-side-price-conviction-center-min-probability must be between 0 and 1"
        )
    if not (
        args.v7_raw_side_price_conviction_min_price
        < args.v7_raw_side_price_conviction_center_price
        < args.v7_raw_side_price_conviction_max_price
    ):
        raise ValueError(
            "--v7-raw-side-price-conviction prices must satisfy min < center < max"
        )
    v7_convergence_calibration_config = _v7_convergence_calibration_config_from_args(
        args
    )
    if v7_convergence_calibration_config.enabled and not Path(
        v7_convergence_calibration_config.path
    ).is_file():
        raise ValueError("--v7-convergence-calibration-path must point to a file")
    if not 0.0 <= v7_convergence_calibration_config.min_hit_5c_rate <= 1.0:
        raise ValueError(
            "--v7-convergence-calibration-min-hit-5c-rate must be between 0 and 1"
        )
    if not 0.0 <= v7_convergence_calibration_config.min_hit_10c_rate <= 1.0:
        raise ValueError(
            "--v7-convergence-calibration-min-hit-10c-rate must be between 0 and 1"
        )
    if (
        v7_convergence_calibration_config.max_model_over_error_p80 is not None
        and v7_convergence_calibration_config.max_model_over_error_p80 < 0
    ):
        raise ValueError(
            "--v7-convergence-calibration-max-model-over-error-p80 must be non-negative"
        )
    if (
        v7_convergence_calibration_config.min_adjusted_median_edge is not None
        and not math.isfinite(v7_convergence_calibration_config.min_adjusted_median_edge)
    ):
        raise ValueError(
            "--v7-convergence-calibration-min-adjusted-median-edge must be finite"
        )
    if (
        v7_convergence_calibration_config.min_adjusted_p80_edge is not None
        and not math.isfinite(v7_convergence_calibration_config.min_adjusted_p80_edge)
    ):
        raise ValueError(
            "--v7-convergence-calibration-min-adjusted-p80-edge must be finite"
        )
    if v7_convergence_calibration_config.min_bucket_sample_count <= 0:
        raise ValueError(
            "--v7-convergence-calibration-min-bucket-sample-count must be positive"
        )
    _low_latency_overlay_config_from_args(args)
    if args.low_latency_overlay_enabled and not args.low_latency_overlay_raw_jsonl_path:
        raise ValueError(
            "--low-latency-overlay-raw-jsonl-path is required when overlay is enabled"
        )
    v7_position_config = _v7_settlement_position_config_from_args(args)
    if v7_position_config.round_cap_usdc <= 0:
        raise ValueError("--v7-settlement-position-round-cap-usdc must be positive")
    if v7_position_config.add_edge_min < 0:
        raise ValueError("--v7-settlement-position-add-edge-min must be non-negative")
    if v7_position_config.full_add_edge <= v7_position_config.add_edge_min:
        raise ValueError(
            "--v7-settlement-position-full-add-edge must be greater than add-edge-min"
        )
    if not 0 < v7_position_config.reduce_fraction <= 1:
        raise ValueError("--v7-settlement-position-reduce-fraction must be in (0, 1]")
    if v7_position_config.add_cooldown_after_divergence_reduce_seconds < 0:
        raise ValueError(
            "--v7-settlement-position-add-cooldown-after-divergence-reduce-seconds "
            "must be non-negative"
        )
    if v7_position_config.exit_hysteresis_bars <= 0:
        raise ValueError("--v7-settlement-position-exit-hysteresis-bars must be positive")
    if not 0 <= v7_position_config.reversal_min_confidence <= 1:
        raise ValueError(
            "--v7-settlement-position-reversal-min-confidence must be between 0 and 1"
        )
    if v7_position_config.reversal_min_edge < 0:
        raise ValueError("--v7-settlement-position-reversal-min-edge must be non-negative")
    if v7_position_config.reversal_hysteresis_bars <= 0:
        raise ValueError(
            "--v7-settlement-position-reversal-hysteresis-bars must be positive"
        )
    if v7_position_config.min_rebalance_usdc < 0:
        raise ValueError("--v7-settlement-position-min-rebalance-usdc must be non-negative")
    if v7_position_config.take_profit_hysteresis_bars <= 0:
        raise ValueError(
            "--v7-settlement-position-take-profit-hysteresis-bars must be positive"
        )
    if v7_position_config.take_profit_force_exit_seconds < 0:
        raise ValueError(
            "--v7-settlement-position-take-profit-force-exit-seconds must be non-negative"
        )
    if not 0.0 <= v7_position_config.take_profit_residual_ratio <= 1.0:
        raise ValueError(
            "--v7-settlement-position-take-profit-residual-ratio must be between 0 and 1"
        )
    if v7_position_config.take_profit_price_convergence_move < 0:
        raise ValueError(
            "--v7-settlement-position-take-profit-price-convergence-move "
            "must be non-negative"
        )
    if not 0.0 <= v7_position_config.take_profit_price_convergence_hold_edge_ratio <= 1.0:
        raise ValueError(
            "--v7-settlement-position-take-profit-price-convergence-hold-edge-ratio "
            "must be between 0 and 1"
        )
    if v7_position_config.take_profit_up_hold_edge_tighten < 0:
        raise ValueError(
            "--v7-settlement-position-take-profit-up-hold-edge-tighten must be non-negative"
        )
    if v7_position_config.take_profit_min_profit_delta < 0:
        raise ValueError(
            "--v7-settlement-position-take-profit-min-profit-delta must be non-negative"
        )
    if v7_position_config.take_profit_min_profit_return < 0:
        raise ValueError(
            "--v7-settlement-position-take-profit-min-profit-return must be non-negative"
        )
    if v7_position_config.low_confidence_scalp_take_profit_min_profit_delta < 0:
        raise ValueError(
            "--v7-settlement-position-low-confidence-scalp-take-profit-min-profit-delta "
            "must be non-negative"
        )
    if v7_position_config.low_confidence_scalp_take_profit_min_profit_return < 0:
        raise ValueError(
            "--v7-settlement-position-low-confidence-scalp-take-profit-min-profit-return "
            "must be non-negative"
        )
    if v7_position_config.low_confidence_scalp_take_profit_hysteresis_bars < 1:
        raise ValueError(
            "--v7-settlement-position-low-confidence-scalp-take-profit-hysteresis-bars "
            "must be >= 1"
        )
    if v7_position_config.adverse_confidence_price_delta_start < 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-price-delta-start must be non-negative"
        )
    if not 0.0 <= v7_position_config.adverse_confidence_base_allowed_decay <= 1.0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-base-allowed-decay must be between 0 and 1"
        )
    if v7_position_config.adverse_confidence_price_decay_slope < 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-price-decay-slope must be non-negative"
        )
    if not 0.0 <= v7_position_config.adverse_confidence_min_allowed_decay <= 1.0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-min-allowed-decay must be between 0 and 1"
        )
    if (
        v7_position_config.adverse_confidence_min_allowed_decay
        > v7_position_config.adverse_confidence_base_allowed_decay
    ):
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-min-allowed-decay must be "
            "less than or equal to base-allowed-decay"
        )
    if not 0.0 <= v7_position_config.adverse_confidence_max_required_probability <= 1.0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-max-required-probability "
            "must be between 0 and 1"
        )
    if v7_position_config.adverse_confidence_exit_probability_buffer < 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-exit-probability-buffer "
            "must be non-negative"
        )
    if v7_position_config.adverse_confidence_full_exit_min_model_decay < 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-full-exit-min-model-decay "
            "must be non-negative"
        )
    if v7_position_config.adverse_confidence_reduce_min_model_decay < 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-reduce-min-model-decay "
            "must be non-negative"
        )
    if v7_position_config.adverse_confidence_dust_exit_max_cost < 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-dust-exit-max-cost "
            "must be non-negative"
        )
    if v7_position_config.adverse_confidence_dust_exit_min_candidate_count <= 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-dust-exit-min-candidate-count "
            "must be positive"
        )
    if v7_position_config.adverse_confidence_hysteresis_bars <= 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-hysteresis-bars must be positive"
        )
    if v7_position_config.adverse_confidence_max_reduces < 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-max-reduces must be non-negative"
        )
    if v7_position_config.adverse_confidence_post_reduce_full_exit_bars <= 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-post-reduce-full-exit-bars "
            "must be positive"
        )
    if v7_position_config.adverse_confidence_post_reduce_full_exit_min_model_decay < 0:
        raise ValueError(
            "--v7-settlement-position-adverse-confidence-post-reduce-full-exit-min-model-decay "
            "must be non-negative"
        )
    if v7_position_config.post_take_profit_reentry_min_model_probability_improvement < 0:
        raise ValueError(
            "--v7-settlement-position-post-take-profit-reentry-min-model-probability-improvement "
            "must be non-negative"
        )
    if v7_position_config.post_take_profit_reentry_min_raw_probability_improvement < 0:
        raise ValueError(
            "--v7-settlement-position-post-take-profit-reentry-min-raw-probability-improvement "
            "must be non-negative"
        )
    if v7_position_config.post_take_profit_reentry_min_seconds_to_expiry < 0:
        raise ValueError(
            "--v7-settlement-position-post-take-profit-reentry-min-seconds-to-expiry "
            "must be non-negative"
        )
    v7_candidate_buffer_config = _v7_entry_candidate_buffer_config_from_args(args)
    if v7_candidate_buffer_config.enabled and args.entry_gate_mode != "v7-pnl":
        raise ValueError(
            "--v7-entry-candidate-buffer-enabled requires --entry-gate-mode v7-pnl"
        )
    if v7_candidate_buffer_config.max_wait_seconds < 0:
        raise ValueError(
            "--v7-entry-candidate-buffer-max-wait-seconds must be non-negative"
        )
    if not 0.0 <= v7_candidate_buffer_config.min_price <= 1.0:
        raise ValueError("--v7-entry-candidate-buffer-min-price must be between 0 and 1")
    if not 0.0 <= v7_candidate_buffer_config.max_price <= 1.0:
        raise ValueError("--v7-entry-candidate-buffer-max-price must be between 0 and 1")
    if v7_candidate_buffer_config.min_price > v7_candidate_buffer_config.max_price:
        raise ValueError(
            "--v7-entry-candidate-buffer-min-price must be <= max-price"
        )
    if v7_candidate_buffer_config.min_edge < 0:
        raise ValueError("--v7-entry-candidate-buffer-min-edge must be non-negative")
    if v7_candidate_buffer_config.min_seconds_to_expiry < 0:
        raise ValueError(
            "--v7-entry-candidate-buffer-min-seconds-to-expiry must be non-negative"
        )
    if (
        v7_candidate_buffer_config.immediate_confidence_score is not None
        and not math.isfinite(v7_candidate_buffer_config.immediate_confidence_score)
    ):
        raise ValueError(
            "--v7-entry-candidate-buffer-immediate-confidence-score must be finite"
        )
    if v7_candidate_buffer_config.max_candidates_per_round <= 0:
        raise ValueError(
            "--v7-entry-candidate-buffer-max-candidates-per-round must be positive"
        )
    live_mid_round_exit_requested = (
        args.settlement_allow_mid_round_exit
        or args.settlement_confidence_decay_enabled
        or args.settlement_price_stop_enabled
    )
    if (
        not args.paper
        and live_mid_round_exit_requested
        and not args.allow_live_settlement_mid_round_exit
    ):
        raise ValueError(
            "Live settlement mid-round exits are disabled by default; pass "
            "--allow-live-settlement-mid-round-exit only after explicit risk approval."
        )


def _decision_evidence_blockers(
    *,
    lifecycle_complete: bool,
    open_positions_at_shutdown: int,
    exits_pending_confirmation: int,
    exits_pending_settlement: int,
    account_cashflow_reconciliation_required: bool,
    paper: bool,
) -> list[str]:
    blockers: list[str] = []
    if paper:
        blockers.append("paper_run_not_capital_sizing_evidence")
    if account_cashflow_reconciliation_required:
        blockers.append("account_cashflow_reconciliation_required")
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


def _build_clob_client(*, require_api_creds: bool = True) -> Any:
    from py_clob_client_v2 import ClobClient, SignatureTypeV2

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if require_api_creds and not private_key:
        raise RuntimeError("POLYMARKET_PRIVATE_KEY is required")
    signature_type_name = os.getenv("POLYMARKET_SIGNATURE_TYPE", "POLY_PROXY")
    signature_type = getattr(SignatureTypeV2, signature_type_name)
    kwargs: dict[str, Any] = {
        "host": os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com"),
        "chain_id": int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
        "signature_type": signature_type,
    }
    if private_key:
        kwargs["key"] = private_key
    funder = os.getenv("POLYMARKET_FUNDER") or os.getenv("POLYMARKET_FUNDER_ADDRESS")
    if funder:
        kwargs["funder"] = funder
    client = ClobClient(**kwargs)
    if not require_api_creds:
        return client
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
    entry_gate_mode: str = "v5-edge",
    collapse_best_per_round: bool = True,
) -> list[SignalEvent]:
    return _read_event_batch_after(
        db_path,
        model_version=model_version,
        after_created_at=after_created_at,
        after_event_id=after_event_id,
        limit=limit,
        v6_joint_config=v6_joint_config,
        entry_gate_mode=entry_gate_mode,
        collapse_best_per_round=collapse_best_per_round,
    ).events


def _read_event_batch_after(
    db_path: str,
    *,
    model_version: str,
    after_created_at: int,
    after_event_id: str,
    limit: int,
    v6_joint_config: V6JointGateConfig | None = None,
    entry_gate_mode: str = "v5-edge",
    selection_now_ms: int | None = None,
    max_signal_age_seconds: float | None = None,
    collapse_best_per_round: bool = True,
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
    effective_entry_gate_mode = (
        "v6-joint"
        if v6_joint_config is not None and entry_gate_mode == "v5-edge"
        else entry_gate_mode
    )
    best_events = (
        _best_event_per_round(
            events,
            entry_gate_mode=effective_entry_gate_mode,
            selection_now_ms=selection_now_ms,
            max_signal_age_seconds=max_signal_age_seconds,
        )
        if collapse_best_per_round
        else events
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
    if _is_v7_model_version(model_version):
        if _selected_v7_signal_side(snapshot) is None:
            return "v7_selected_side_missing"
        p_up = _optional_float(snapshot.get("p_up"))
        p_down = _optional_float(snapshot.get("p_down"))
        if p_up is None or p_down is None:
            return "v7_probability_missing"
        if _optional_float(snapshot.get("market_implied_prob")) is None:
            return "v7_market_implied_prob_missing"
    return "unparseable_signal"


def _latest_signal_jsonl_cursor(path: Path, *, start: str) -> tuple[int, str]:
    cursor = JsonlSignalSource(path).latest_cursor(start=start)
    return cursor.position, cursor.signature


def _read_signal_jsonl_after(
    path: Path,
    *,
    after_line_number: int,
    after_line_signature: str = "",
    model_version: str,
    limit: int,
    v6_joint_config: V6JointGateConfig | None = None,
    entry_gate_mode: str = "v5-edge",
    selection_now_ms: int | None = None,
    max_signal_age_seconds: float | None = None,
    collapse_best_per_round: bool = True,
) -> tuple[list[SignalEvent], int, str]:
    events, cursor = _read_signal_source_after(
        JsonlSignalSource(path),
        cursor=SignalCursor(position=after_line_number, signature=after_line_signature),
        model_version=model_version,
        limit=limit,
        v6_joint_config=v6_joint_config,
        entry_gate_mode=entry_gate_mode,
        selection_now_ms=selection_now_ms,
        max_signal_age_seconds=max_signal_age_seconds,
        collapse_best_per_round=collapse_best_per_round,
    )
    return events, cursor.position, cursor.signature


def _read_signal_source_after(
    source: SignalSource,
    *,
    cursor: SignalCursor,
    model_version: str,
    limit: int,
    v6_joint_config: V6JointGateConfig | None = None,
    entry_gate_mode: str = "v5-edge",
    selection_now_ms: int | None = None,
    max_signal_age_seconds: float | None = None,
    collapse_best_per_round: bool = True,
) -> tuple[list[SignalEvent], SignalCursor]:
    events: list[SignalEvent] = []
    batch = source.read_after(cursor, limit=limit)
    for payload in batch.payloads:
        event = _event_from_signal_payload(
            payload,
            model_version=model_version,
            v6_joint_config=v6_joint_config,
        )
        if event is not None:
            events.append(event)
        if len(events) >= limit:
            break
    if collapse_best_per_round:
        events = _best_event_per_round(
            events,
            entry_gate_mode=entry_gate_mode,
            selection_now_ms=selection_now_ms,
            max_signal_age_seconds=max_signal_age_seconds,
        )
    return events, batch.cursor


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
        payload_side = payload.get("outcome_side") or payload.get("v6_joint_side")
        if payload_side:
            side = str(payload_side).upper()
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
                v6_joint_side=(
                    str(payload["v6_joint_side"]).upper()
                    if payload.get("v6_joint_side")
                    else None
                ),
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
    model_probability = _optional_float(payload.get("model_probability"))
    token_probability = model_probability
    if token_probability is None:
        token_probability = _optional_float(payload.get("token_probability"))
    if side not in {"UP", "DOWN"} or not token_id or token_probability is None:
        return None
    is_v7 = _is_v7_model_version(model_version)
    p_up_residual = _optional_float(payload.get("p_up_residual_adjusted"))
    p_down_residual = _optional_float(payload.get("p_down_residual_adjusted"))
    if is_v7:
        if model_probability is None:
            model_probability = _optional_float(payload.get("token_expected_win_probability"))
        if model_probability is not None:
            token_probability = model_probability
    confidence_fields = _v7_confidence_fields_from_mapping(payload) if is_v7 else {}
    edge = _optional_float(payload.get("mispricing_edge"))
    if edge is None:
        edge = _optional_float(payload.get("edge"))
    if is_v7 and edge is None:
        residual_edge = _optional_float(
            payload.get("residual_expected_edge_up")
            if side == "UP"
            else payload.get("residual_expected_edge_down")
        )
        if residual_edge is not None:
            edge = residual_edge
    if edge is None:
        edge = token_probability - market
    canonical_symbol = str(payload.get("canonical_symbol") or f"BTC-15M:{round_slug}:{side}")
    token_expected_win_probability = _optional_float(
        payload.get("token_expected_win_probability")
    )
    if token_expected_win_probability is None:
        token_expected_win_probability = token_probability
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
        model_probability=model_probability,
        polymarket_price=_optional_float(payload.get("polymarket_price")),
        mispricing_edge=_optional_float(payload.get("mispricing_edge")),
        p_up=_optional_float(payload.get("p_up")),
        p_down=_optional_float(payload.get("p_down")),
        p_neutral=_optional_float(payload.get("p_neutral")),
        settlement_residual=_optional_float(payload.get("settlement_residual")),
        token_expected_win_probability=token_expected_win_probability,
        p_up_residual_adjusted=p_up_residual,
        p_down_residual_adjusted=p_down_residual,
        expected_edge_up=_optional_float(payload.get("expected_edge_up")),
        expected_edge_down=_optional_float(payload.get("expected_edge_down")),
        residual_expected_edge_up=_optional_float(payload.get("residual_expected_edge_up")),
        residual_expected_edge_down=_optional_float(payload.get("residual_expected_edge_down")),
        **confidence_fields,
        selected_side=(
            side
            if is_v7
            else str(payload["selected_side"]).upper()
            if payload.get("selected_side")
            else None
        ),
        selected_expected_edge=(
            edge
            if is_v7
            else _optional_float(payload.get("selected_expected_edge"))
        ),
        entry_worst_price=_optional_float(payload.get("entry_worst_price")),
        should_enter_settlement=_optional_bool(payload.get("should_enter_settlement")),
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
    if _is_v7_model_version(model_version):
        selected_side = _selected_v7_signal_side(snapshot, token_side=side)
        if selected_side is None:
            return None
        p_up = _optional_float(snapshot.get("p_up")) or float(prob_up_15m)
        p_down = _optional_float(snapshot.get("p_down"))
        if p_down is None:
            p_down = max(0.0, min(1.0, 1.0 - p_up))
        market = _optional_float(snapshot.get("market_implied_prob"))
        if market is None:
            return None
        selected_market = market if selected_side == side else max(0.0, min(1.0, 1.0 - market))
        polymarket_price = _optional_float(snapshot.get("polymarket_price"))
        if polymarket_price is None:
            polymarket_price = selected_market
        token_id = (
            str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
            if selected_side == side
            else opposite_token_id
        )
        if not token_id:
            return None
        p_up_residual = _optional_float(snapshot.get("p_up_residual_adjusted"))
        p_down_residual = _optional_float(snapshot.get("p_down_residual_adjusted"))
        model_probability = _optional_float(snapshot.get("model_probability"))
        if model_probability is None:
            model_probability = _optional_float(snapshot.get("token_expected_win_probability"))
        if model_probability is None:
            residual_probability = p_up_residual if selected_side == "UP" else p_down_residual
            model_probability = (
                residual_probability
                if residual_probability is not None
                else (p_up if selected_side == "UP" else p_down)
            )
        token_probability = model_probability
        edge = _v7_side_edge(
            snapshot,
            side=selected_side,
            market=selected_market,
            token_probability=token_probability,
        )
        if edge is None:
            edge = token_probability - selected_market
        confidence_fields = _v7_confidence_fields_from_mapping(snapshot)
        return SignalEvent(
            event_id=str(event_id),
            ts=int(ts),
            created_at=int(created_at),
            prob_up_15m=p_up,
            canonical_symbol=f"{parts[0]}:{round_slug}:{selected_side}",
            token_id=token_id,
            outcome_side=selected_side,
            round_slug=round_slug,
            round_end_ts=round_end_ts,
            market_implied_prob=selected_market,
            token_probability=token_probability,
            edge=edge,
            bridged_at=0,
            opposite_token_id=(
                str(snapshot.get("source_symbol") or snapshot.get("token_id") or "")
                if selected_side != side
                else opposite_token_id
            ),
            model_probability=model_probability,
            polymarket_price=polymarket_price,
            mispricing_edge=edge,
            p_up=p_up,
            p_down=p_down,
            p_neutral=_optional_float(snapshot.get("p_neutral")),
            settlement_residual=_optional_float(snapshot.get("settlement_residual")),
            token_expected_win_probability=token_probability,
            p_up_residual_adjusted=p_up_residual,
            p_down_residual_adjusted=p_down_residual,
            expected_edge_up=_optional_float(snapshot.get("expected_edge_up")),
            expected_edge_down=_optional_float(snapshot.get("expected_edge_down")),
            residual_expected_edge_up=_optional_float(snapshot.get("residual_expected_edge_up")),
            residual_expected_edge_down=_optional_float(snapshot.get("residual_expected_edge_down")),
            **confidence_fields,
            selected_side=selected_side,
            selected_expected_edge=edge,
            entry_worst_price=_optional_float(
                snapshot.get("entry_worst_price_up")
                if selected_side == "UP"
                else snapshot.get("entry_worst_price_down")
            ),
            should_enter_settlement=_optional_bool(snapshot.get("should_enter_settlement")),
        )
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


def _selected_v7_signal_side(
    snapshot: dict[str, Any],
    *,
    token_side: str | None = None,
) -> str | None:
    selected = str(snapshot.get("selected_side") or "").upper()
    if selected in {"UP", "DOWN"}:
        return selected
    if token_side in {"UP", "DOWN"}:
        edge = _v7_side_edge_from_snapshot(snapshot, side=token_side, token_side=token_side)
        if edge is not None:
            return token_side
    up_edge = _v7_side_edge_from_snapshot(snapshot, side="UP", token_side=token_side)
    down_edge = _v7_side_edge_from_snapshot(snapshot, side="DOWN", token_side=token_side)
    if up_edge is not None or down_edge is not None:
        if down_edge is None or (up_edge is not None and up_edge >= down_edge):
            return "UP"
        return "DOWN"
    up_edge = _optional_float(snapshot.get("expected_edge_up"))
    down_edge = _optional_float(snapshot.get("expected_edge_down"))
    if up_edge is None and down_edge is None:
        return None
    if down_edge is None or (up_edge is not None and up_edge >= down_edge):
        return "UP"
    return "DOWN"


def _v7_side_edge(
    snapshot: dict[str, Any],
    *,
    side: str,
    market: float,
    token_probability: float,
) -> float | None:
    mispricing = _optional_float(snapshot.get("mispricing_edge"))
    if mispricing is not None:
        return mispricing
    if _optional_float(snapshot.get("model_probability")) is not None:
        return token_probability - market
    edge = _optional_float(
        snapshot.get("expected_edge_up") if side == "UP" else snapshot.get("expected_edge_down")
    )
    residual_probability = _optional_float(
        snapshot.get("p_up_residual_adjusted")
        if side == "UP"
        else snapshot.get("p_down_residual_adjusted")
    )
    if edge is not None and residual_probability is None:
        return edge
    residual = _optional_float(
        snapshot.get("residual_expected_edge_up")
        if side == "UP"
        else snapshot.get("residual_expected_edge_down")
    )
    if residual is not None:
        return residual
    return token_probability - market


def _v7_side_edge_from_snapshot(
    snapshot: dict[str, Any],
    *,
    side: str,
    token_side: str | None,
) -> float | None:
    mispricing = _optional_float(snapshot.get("mispricing_edge"))
    if mispricing is not None and (token_side is None or side == token_side):
        return mispricing
    model_probability = _optional_float(snapshot.get("model_probability"))
    if model_probability is not None and token_side is not None and side == token_side:
        market = _v7_selected_market(snapshot, selected_side=side, token_side=token_side)
        if market is not None:
            return model_probability - market
    residual_probability = _optional_float(
        snapshot.get("p_up_residual_adjusted")
        if side == "UP"
        else snapshot.get("p_down_residual_adjusted")
    )
    if residual_probability is not None and token_side is not None:
        market = _v7_selected_market(snapshot, selected_side=side, token_side=token_side)
        if market is not None:
            return residual_probability - market
    return None


def _v7_selected_market(
    snapshot: dict[str, Any],
    *,
    selected_side: str,
    token_side: str,
) -> float | None:
    value = _optional_float(snapshot.get("market_implied_prob"))
    if value is None:
        return None
    if selected_side == token_side:
        return value
    return max(0.0, min(1.0, 1.0 - value))


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
    selection_now_ms: int | None = None,
    max_signal_age_seconds: float | None = None,
) -> list[SignalEvent]:
    grouped: dict[str, list[SignalEvent]] = {}
    for event in events:
        best_key = event.round_slug
        if entry_gate_mode == "v6-joint" and event.p_up is not None and event.p_vol_up is not None:
            best_key = (
                f"settlement:{event.round_slug}"
                if event.v6_joint_side is not None
                else f"volatility:{event.round_slug}"
            )
        grouped.setdefault(best_key, []).append(event)
    best: dict[str, SignalEvent] = {}
    for best_key, group in grouped.items():
        candidates = _fresh_selection_candidates(
            group,
            selection_now_ms=selection_now_ms,
            max_signal_age_seconds=max_signal_age_seconds,
        )
        best[best_key] = _select_best_event(candidates, entry_gate_mode=entry_gate_mode)
    return sorted(best.values(), key=lambda item: (item.created_at, item.event_id))


def _fresh_selection_candidates(
    events: list[SignalEvent],
    *,
    selection_now_ms: int | None,
    max_signal_age_seconds: float | None,
) -> list[SignalEvent]:
    if selection_now_ms is None or max_signal_age_seconds is None:
        return events
    fresh = [
        event
        for event in events
        if event.ts <= 0
        or max(0.0, (selection_now_ms - event.ts) / 1000) <= max_signal_age_seconds
    ]
    if fresh:
        return fresh
    return [
        max(
            events,
            key=lambda item: (item.ts, item.created_at, item.event_id),
        )
    ]


def _select_best_event(
    events: list[SignalEvent],
    *,
    entry_gate_mode: str,
) -> SignalEvent:
    best = events[0]
    best_score = _event_selection_score(best, entry_gate_mode=entry_gate_mode)
    for event in events[1:]:
        event_score = _event_selection_score(event, entry_gate_mode=entry_gate_mode)
        if event_score > best_score:
            best = event
            best_score = event_score
    return best


def _event_selection_score(
    event: SignalEvent,
    *,
    entry_gate_mode: str,
) -> float:
    if entry_gate_mode == "v6-joint" and event.p_up is not None and event.p_vol_up is not None:
        return _v6_event_selection_score(event)
    if entry_gate_mode == "v7-pnl":
        return _v7_event_selection_score(event)
    return event.edge


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


def _v7_event_selection_score(event: SignalEvent) -> float:
    if event.selected_expected_edge is not None:
        return event.selected_expected_edge
    return event.edge


def _v7_confidence_fields_from_mapping(mapping: dict[str, Any]) -> dict[str, float | None]:
    return {
        "p_up_hit_5c_before_loss_10c": _optional_float(
            mapping.get("p_up_hit_5c_before_loss_10c")
        ),
        "p_up_hit_10c_before_loss_10c": _optional_float(
            mapping.get("p_up_hit_10c_before_loss_10c")
        ),
        "p_up_loss_10c_before_hit_5c": _optional_float(
            mapping.get("p_up_loss_10c_before_hit_5c")
        ),
        "p_down_hit_5c_before_loss_10c": _optional_float(
            mapping.get("p_down_hit_5c_before_loss_10c")
        ),
        "p_down_hit_10c_before_loss_10c": _optional_float(
            mapping.get("p_down_hit_10c_before_loss_10c")
        ),
        "p_down_loss_10c_before_hit_5c": _optional_float(
            mapping.get("p_down_loss_10c_before_hit_5c")
        ),
        "selected_hit_5c_before_loss_10c": _optional_float(
            mapping.get("selected_hit_5c_before_loss_10c")
        ),
        "selected_hit_10c_before_loss_10c": _optional_float(
            mapping.get("selected_hit_10c_before_loss_10c")
        ),
        "selected_loss_10c_before_hit_5c": _optional_float(
            mapping.get("selected_loss_10c_before_hit_5c")
        ),
        "selected_confidence_score": _optional_float(
            mapping.get("selected_confidence_score")
        ),
    }


def _v7_entry_candidate_summary(event: SignalEvent) -> dict[str, Any]:
    price = _v7_signal_reference_price(event)
    return {
        "event_id": event.event_id,
        "round_slug": event.round_slug,
        "side": event.outcome_side,
        "created_at": _iso(event.created_at) if event.created_at else None,
        "price": price,
        "model_probability": event.model_probability,
        "token_probability": event.token_probability,
        "edge": event.edge,
        "confidence_score": _v7_entry_candidate_score(event),
        "hit_5c_before_loss_10c": _v7_entry_candidate_hit_5c(event),
        "hit_10c_before_loss_10c": _v7_entry_candidate_hit_10c(event),
        "loss_10c_before_hit_5c": _v7_entry_candidate_loss_10c(event),
    }


def _v7_entry_candidate_skip_reason(
    event: SignalEvent,
    *,
    config: V7EntryCandidateBufferConfig,
    seconds_to_expiry: float,
) -> str | None:
    if seconds_to_expiry <= 0:
        return "v7_entry_candidate_expired_round"
    price = _v7_signal_reference_price(event)
    if price is None:
        return "v7_entry_candidate_price_missing"
    if price < config.min_price:
        return "v7_entry_candidate_price_below_band"
    if price > config.max_price:
        return "v7_entry_candidate_price_above_band"
    if event.edge < config.min_edge:
        return "v7_entry_candidate_edge_below_min"
    return None


def _v7_add_or_replace_candidate(
    bucket: V7EntryCandidateBucket,
    event: SignalEvent,
    *,
    max_candidates_per_round: int,
) -> None:
    for index, candidate in enumerate(bucket.candidates):
        if candidate.event_id == event.event_id:
            bucket.candidates[index] = event
            break
    else:
        bucket.candidates.append(event)
    if len(bucket.candidates) <= max_candidates_per_round:
        return
    bucket.candidates[:] = sorted(
        bucket.candidates,
        key=_v7_entry_candidate_rank_key,
        reverse=True,
    )[:max_candidates_per_round]


def _select_v7_entry_candidate(events: list[SignalEvent]) -> SignalEvent:
    if not events:
        raise ValueError("events must be non-empty")
    return max(events, key=_v7_entry_candidate_rank_key)


def _fresh_v7_entry_candidates(
    events: list[SignalEvent],
    *,
    now_ms: int,
    max_signal_age_seconds: float | None,
) -> tuple[list[SignalEvent], int]:
    if max_signal_age_seconds is None:
        return list(events), 0
    fresh: list[SignalEvent] = []
    stale_count = 0
    for event in events:
        if event.ts <= 0:
            fresh.append(event)
            continue
        age_seconds = max(0.0, (now_ms - event.ts) / 1000)
        if age_seconds <= max_signal_age_seconds:
            fresh.append(event)
        else:
            stale_count += 1
    return fresh, stale_count


def _v7_entry_candidate_rank_key(event: SignalEvent) -> tuple[float, float, float, float, float, float, int, str]:
    confidence = _v7_entry_candidate_score(event)
    hit_5c = _v7_entry_candidate_hit_5c(event)
    hit_10c = _v7_entry_candidate_hit_10c(event)
    loss_10c = _v7_entry_candidate_loss_10c(event)
    price = _v7_signal_reference_price(event)
    price_center_penalty = -abs(float(price) - 0.50) if price is not None else -1.0
    return (
        -math.inf if confidence is None else confidence,
        -math.inf if loss_10c is None else -loss_10c,
        -math.inf if hit_5c is None else hit_5c,
        -math.inf if hit_10c is None else hit_10c,
        float(event.edge),
        price_center_penalty,
        int(event.created_at),
        event.event_id,
    )


def _v7_entry_candidate_score(event: SignalEvent) -> float | None:
    if event.selected_confidence_score is not None:
        return float(event.selected_confidence_score)
    hit_5c = _v7_entry_candidate_hit_5c(event)
    loss_10c = _v7_entry_candidate_loss_10c(event)
    if hit_5c is None and loss_10c is None:
        return None
    if hit_5c is None:
        return -float(loss_10c or 0.0)
    if loss_10c is None:
        return float(hit_5c)
    return float(hit_5c) - float(loss_10c)


def _v7_entry_candidate_hit_5c(event: SignalEvent) -> float | None:
    if event.selected_hit_5c_before_loss_10c is not None:
        return float(event.selected_hit_5c_before_loss_10c)
    if (event.selected_side or event.outcome_side) == "UP":
        return event.p_up_hit_5c_before_loss_10c
    return event.p_down_hit_5c_before_loss_10c


def _v7_entry_candidate_hit_10c(event: SignalEvent) -> float | None:
    if event.selected_hit_10c_before_loss_10c is not None:
        return float(event.selected_hit_10c_before_loss_10c)
    if (event.selected_side or event.outcome_side) == "UP":
        return event.p_up_hit_10c_before_loss_10c
    return event.p_down_hit_10c_before_loss_10c


def _v7_entry_candidate_loss_10c(event: SignalEvent) -> float | None:
    if event.selected_loss_10c_before_hit_5c is not None:
        return float(event.selected_loss_10c_before_hit_5c)
    if (event.selected_side or event.outcome_side) == "UP":
        return event.p_up_loss_10c_before_hit_5c
    return event.p_down_loss_10c_before_hit_5c


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
    settlement_exit_config: SettlementExitConfig | None = None,
    v7_position_config: V7SettlementPositionConfig | None = None,
    settlement_exit_counts: dict[str, int] | None = None,
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
            sell_result = _maybe_settlement_price_stop_exit(
                client=client,
                position_manager=position_manager,
                position=position,
                log_path=log_path,
                seconds_to_expiry=seconds_to_expiry,
                config=settlement_exit_config,
                sell_slippage=sell_slippage,
                exit_order_timeout_seconds=exit_order_timeout_seconds,
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
                if settlement_exit_counts is not None:
                    _bump(settlement_exit_counts, "settlement_price_stop_exit")
                lifecycle.mark_position_closed(
                    position.round_slug,
                    position.sleeve,
                    position=position,
                    reason=sell_result.reason or "settlement_price_stop_exit",
                    realized_pnl=sell_result.realized_pnl,
                    closed_at=now_ms,
                )
                continue
            v7_tick_exit = _maybe_v7_settlement_tick_exit(
                client=client,
                position_manager=position_manager,
                position=position,
                log_path=log_path,
                seconds_to_expiry=seconds_to_expiry,
                config=v7_position_config,
                sell_slippage=sell_slippage,
                exit_order_timeout_seconds=exit_order_timeout_seconds,
                monitoring_db_path=monitoring_db_path,
            )
            if v7_tick_exit is not None:
                if v7_tick_exit.status == "filled":
                    closed_count += 1
                elif v7_tick_exit.status == "settled":
                    pass
                elif v7_tick_exit.status == "pending_settlement":
                    settlement_count += 1
                else:
                    pending_count += 1
                realized_pnl += v7_tick_exit.realized_pnl
                lifecycle.mark_position_closed(
                    position.round_slug,
                    position.sleeve,
                    position=position,
                    reason=v7_tick_exit.reason
                    or position.v7_position_take_profit_candidate_reason
                    or V7_LATE_FORCE_EXIT_REASON,
                    realized_pnl=v7_tick_exit.realized_pnl,
                    closed_at=now_ms,
                )
                continue
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
        lifecycle.mark_position_closed(
            position.round_slug,
            position.sleeve,
            position=position,
            reason=sell_result.reason or exit_reason,
            realized_pnl=sell_result.realized_pnl,
            closed_at=now_ms,
        )

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
    signal_age_seconds: float | None = None,
    settlement_peak_confidence: float | None = None,
    low_latency_overlay: LowLatencyEntryOverlay | None = None,
    v7_position_config: V7SettlementPositionConfig | None = None,
    v7_convergence_calibration_gate: V7ConvergenceCalibrationGate | None = None,
) -> LivePosition | None:
    from py_clob_client_v2 import MarketOrderArgs, OrderType
    from py_clob_client_v2.clob_types import PartialCreateOrderOptions
    from py_clob_client_v2.order_builder.constants import BUY

    settlement_cost_edge_only = sleeve == "settlement" and _settlement_cost_edge_mode(
        entry_gate_mode
    )
    settlement_confidence_for_gate = (
        None
        if entry_gate_mode == "v7-pnl"
        else signal.token_probability if sleeve == "settlement" else None
    )
    settlement_peak_confidence_for_gate = (
        None
        if entry_gate_mode == "v7-pnl"
        else settlement_peak_confidence if sleeve == "settlement" else None
    )
    no_quote_gate_evaluation = evaluate_entry_gates(
        settlement_edge=signal.edge,
        ask=None,
        worst_price=None,
        token_probability=signal.token_probability,
        seconds_to_expiry=seconds_to_expiry,
        policy=entry_policy,
        settlement_confidence=settlement_confidence_for_gate,
        settlement_peak_confidence=settlement_peak_confidence_for_gate,
        signal_age_seconds=signal_age_seconds,
        enable_settlement_gate=sleeve == "settlement",
    )
    orderbook_source = "clob"
    low_latency_overlay_payload: dict[str, Any] | None = None
    low_latency_overlay_passed: bool | None = None
    try:
        bid, ask = _best_bid_ask(client, signal.token_id)
    except OrderBookUnavailable as exc:
        if paper and low_latency_overlay is not None:
            overlay_decision = low_latency_overlay.evaluate_entry(
                asdict(signal),
                now_ms=_now_ms(),
            )
            low_latency_overlay_payload = overlay_decision.to_dict()
            low_latency_overlay_passed = overlay_decision.passed
            if not overlay_decision.passed:
                _log(
                    log_path,
                    "entry_skipped",
                    reason=overlay_decision.reason,
                    signal=asdict(signal),
                    gate_evaluation=asdict(no_quote_gate_evaluation),
                    low_latency_overlay=low_latency_overlay_payload,
                    orderbook_source="low_latency_overlay",
                    clob_orderbook_error=exc.to_log_payload(),
                )
                return None
            if overlay_decision.latest_ask is not None:
                bid = overlay_decision.latest_bid
                ask = overlay_decision.latest_ask
                orderbook_source = "low_latency_overlay"
            else:
                _log(
                    log_path,
                    "entry_skipped",
                    reason="orderbook_unavailable",
                    signal=asdict(signal),
                    gate_evaluation=asdict(no_quote_gate_evaluation),
                    low_latency_overlay=low_latency_overlay_payload,
                    **exc.to_log_payload(),
                )
                return None
        else:
            _log(
                log_path,
                "entry_skipped",
                reason="orderbook_unavailable",
                signal=asdict(signal),
                gate_evaluation=asdict(no_quote_gate_evaluation),
                **exc.to_log_payload(),
            )
            return None
    if orderbook_source == "clob":
        low_latency_overlay_payload = None
        low_latency_overlay_passed = None
    if ask is None:
        _log(
            log_path,
            "entry_skipped",
            reason="missing_ask",
            signal=asdict(signal),
            gate_evaluation=asdict(no_quote_gate_evaluation),
            orderbook_source=orderbook_source,
            low_latency_overlay=low_latency_overlay_payload,
        )
        return None
    if orderbook_source == "low_latency_overlay":
        tick_size = 0.01
        neg_risk = False
    else:
        tick_size = client.get_tick_size(signal.token_id)
        neg_risk = client.get_neg_risk(signal.token_id)
    ask_price = float(ask)
    slippage_worst_price = min(0.99, _round_price(ask_price + buy_slippage, tick_size))
    max_acceptable_price = (
        _floor_price(
            signal.token_probability - entry_policy.effective_settlement_edge_threshold,
            tick_size,
        )
        if settlement_cost_edge_only
        else None
    )
    worst_price = _round_price(ask_price, tick_size) if settlement_cost_edge_only else slippage_worst_price
    order_limit_price = (
        max(worst_price, min(slippage_worst_price, max_acceptable_price))
        if max_acceptable_price is not None
        else worst_price
    )
    fresh_edge_at_worst = signal.token_probability - worst_price
    settlement_edge_for_gate = (
        fresh_edge_at_worst
        if settlement_cost_edge_only
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
        settlement_confidence=settlement_confidence_for_gate,
        settlement_peak_confidence=settlement_peak_confidence_for_gate,
        signal_age_seconds=signal_age_seconds,
        enable_settlement_gate=sleeve == "settlement",
    )
    gate_payload = asdict(gate_evaluation)
    v7_raw_side_skip_reason: str | None = None
    raw_selected_side: str | None = None
    raw_p_side: float | None = None
    raw_p_opposite: float | None = None
    v7_convergence_calibration_skip_reason: str | None = None
    if sleeve == "settlement" and entry_gate_mode == "v7-pnl":
        raw_selected_side = signal.selected_side or signal.outcome_side
        raw_opposite_side = _opposite_side(raw_selected_side)
        raw_p_side = _signal_probability_for_side(signal, raw_selected_side)
        raw_p_opposite = _signal_probability_for_side(signal, raw_opposite_side)
        v7_raw_side_skip_reason = v7_raw_side_agreement_skip_reason(
            selected_side=raw_selected_side,
            p_up=signal.p_up,
            p_down=signal.p_down,
            entry_price=worst_price,
            policy=entry_policy,
        )
        raw_required_min_probability = v7_raw_side_required_min_probability(
            entry_price=worst_price,
            policy=entry_policy,
        )
        gate_payload["v7_raw_side_agreement"] = {
            "enabled": entry_policy.v7_raw_side_agreement_enabled,
            "selected_side": raw_selected_side,
            "p_side": raw_p_side,
            "p_opposite": raw_p_opposite,
            "raw_margin": (
                None
                if raw_p_side is None or raw_p_opposite is None
                else raw_p_side - raw_p_opposite
            ),
            "min_probability": entry_policy.v7_raw_side_min_probability,
            "required_min_probability": raw_required_min_probability,
            "min_margin": entry_policy.v7_raw_side_min_margin,
            "price_conviction_enabled": (
                entry_policy.v7_raw_side_price_conviction_enabled
            ),
            "price_conviction_entry_price": worst_price,
            "price_conviction_center_min_probability": (
                entry_policy.v7_raw_side_price_conviction_center_min_probability
            ),
            "max_opposite_lead": entry_policy.v7_raw_side_max_opposite_lead,
            "skip_reason": v7_raw_side_skip_reason,
        }
        if v7_convergence_calibration_gate is not None:
            calibration_price = _v7_signal_reference_price(signal)
            calibration_price_source = "signal_reference_price"
            if calibration_price is None:
                calibration_price = worst_price
                calibration_price_source = "worst_price_fallback"
            calibration_evaluation = v7_convergence_calibration_gate.evaluate(
                price=float(calibration_price),
                execution_price=float(worst_price),
                model_value=float(signal.token_probability),
                edge=float(signal.edge),
                raw_p_side=raw_p_side,
            )
            calibration_payload = calibration_evaluation.to_log_payload()
            calibration_payload["price_source"] = calibration_price_source
            gate_payload["v7_convergence_calibration"] = calibration_payload
            v7_convergence_calibration_skip_reason = calibration_evaluation.skip_reason
    entry_size_usdc = max_position_size_usdc
    v7_position_entry_sizing = _v7_initial_entry_sizing(
        sleeve=sleeve,
        entry_gate_mode=entry_gate_mode,
        fresh_edge_at_worst=fresh_edge_at_worst,
        max_position_size_usdc=max_position_size_usdc,
        config=v7_position_config,
    )
    if v7_position_entry_sizing is not None:
        gate_payload["v7_position_entry_sizing"] = v7_position_entry_sizing
        entry_size_usdc = float(v7_position_entry_sizing["entry_size_usdc"])
    _log(
        log_path,
        "entry_gate_evaluated",
        signal=asdict(signal),
        bid=bid,
        ask=ask,
        worst_price=worst_price,
        slippage_worst_price=slippage_worst_price,
        max_acceptable_price=max_acceptable_price,
        order_limit_price=order_limit_price,
        fresh_edge_at_worst=fresh_edge_at_worst,
        raw_settlement_edge=signal.edge,
        model_selected_expected_edge=signal.selected_expected_edge,
        model_entry_worst_price=signal.entry_worst_price,
        entry_gate_mode=entry_gate_mode,
        seconds_to_expiry=seconds_to_expiry,
        orderbook_source=orderbook_source,
        low_latency_overlay=low_latency_overlay_payload,
        v7_position_entry_sizing=v7_position_entry_sizing,
        gate_evaluation=gate_payload,
    )
    if sleeve == "settlement" and not _settlement_cost_edge_mode(entry_gate_mode):
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
                slippage_worst_price=slippage_worst_price,
                max_acceptable_price=max_acceptable_price,
                order_limit_price=order_limit_price,
                fresh_edge_at_worst=fresh_edge_at_worst,
                seconds_to_expiry=seconds_to_expiry,
                orderbook_source=orderbook_source,
                low_latency_overlay=low_latency_overlay_payload,
                gate_evaluation=gate_payload,
            )
            return None
    elif sleeve == "settlement" and entry_gate_mode == "v6-joint" and signal.v6_joint_side is None:
        _log(
            log_path,
            "entry_skipped",
            reason="v6_settlement_gate_miss",
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
            seconds_to_expiry=seconds_to_expiry,
            orderbook_source=orderbook_source,
            low_latency_overlay=low_latency_overlay_payload,
            gate_evaluation=gate_payload,
        )
        return None
    elif (
        sleeve == "settlement"
        and entry_gate_mode == "v7-pnl"
        and signal.selected_side is not None
        and signal.selected_side != signal.outcome_side
    ):
        _log(
            log_path,
            "entry_skipped",
            reason="v7_selected_side_mismatch",
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
            seconds_to_expiry=seconds_to_expiry,
            orderbook_source=orderbook_source,
            low_latency_overlay=low_latency_overlay_payload,
            gate_evaluation=gate_payload,
        )
        return None
    elif (
        sleeve == "settlement"
        and entry_gate_mode == "v7-pnl"
        and v7_raw_side_skip_reason is not None
    ):
        _log(
            log_path,
            "entry_skipped",
            reason=v7_raw_side_skip_reason,
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
            fresh_edge_at_worst=fresh_edge_at_worst,
            seconds_to_expiry=seconds_to_expiry,
            orderbook_source=orderbook_source,
            low_latency_overlay=low_latency_overlay_payload,
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
                slippage_worst_price=slippage_worst_price,
                max_acceptable_price=max_acceptable_price,
                order_limit_price=order_limit_price,
                fresh_edge_at_worst=fresh_edge_at_worst,
                seconds_to_expiry=seconds_to_expiry,
                orderbook_source=orderbook_source,
                low_latency_overlay=low_latency_overlay_payload,
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
                slippage_worst_price=slippage_worst_price,
                max_acceptable_price=max_acceptable_price,
                order_limit_price=order_limit_price,
                seconds_to_expiry=seconds_to_expiry,
                orderbook_source=orderbook_source,
                low_latency_overlay=low_latency_overlay_payload,
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
        _v7_entry_price_floor_skip_reason(
            ask=float(ask),
            worst_price=worst_price,
            policy=entry_policy,
        )
        if sleeve == "settlement" and entry_gate_mode == "v7-pnl"
        else None
    )
    if skip_reason is None:
        skip_reason = (
            settlement_cost_edge_skip_reason(
                fresh_edge_at_worst=fresh_edge_at_worst,
                policy=entry_policy,
                settlement_confidence=settlement_confidence_for_gate,
                settlement_peak_confidence=settlement_peak_confidence_for_gate,
                signal_age_seconds=signal_age_seconds,
            )
            if settlement_cost_edge_only
            else entry_price_skip_reason(
                ask=float(ask),
                worst_price=worst_price,
                fresh_edge_at_worst=fresh_edge_at_worst,
                seconds_to_expiry=seconds_to_expiry,
                policy=entry_policy,
            )
        )
    if (
        skip_reason is None
        and sleeve == "settlement"
        and entry_gate_mode == "v7-pnl"
    ):
        skip_reason = entry_price_drift_skip_reason(
            entry_price=worst_price,
            signal_price=_v7_signal_reference_price(signal),
            policy=entry_policy,
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
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
            fresh_edge_at_worst=fresh_edge_at_worst,
            seconds_to_expiry=seconds_to_expiry,
            min_entry_price=entry_policy.min_entry_price,
            near_min_price_band=entry_policy.near_min_price_band,
            near_min_fresh_edge_threshold=entry_policy.near_min_fresh_edge_threshold,
            near_min_seconds_to_expiry=entry_policy.near_min_seconds_to_expiry,
            settlement_edge_threshold=entry_policy.effective_settlement_edge_threshold,
            settlement_min_confidence=entry_policy.settlement_min_confidence,
            settlement_peak_confidence=settlement_peak_confidence_for_gate,
            settlement_peak_confidence_drop_tolerance=(
                entry_policy.settlement_peak_confidence_drop_tolerance
            ),
            max_signal_age_seconds=entry_policy.max_signal_age_seconds,
            signal_age_seconds=signal_age_seconds,
            settlement_price_gate_mode=_settlement_price_gate_mode_name(entry_gate_mode),
            gate_evaluation=gate_payload,
        )
        return None
    if (
        sleeve == "settlement"
        and entry_gate_mode == "v7-pnl"
        and v7_convergence_calibration_skip_reason is not None
    ):
        _log(
            log_path,
            "entry_skipped",
            reason=v7_convergence_calibration_skip_reason,
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
            fresh_edge_at_worst=fresh_edge_at_worst,
            seconds_to_expiry=seconds_to_expiry,
            orderbook_source=orderbook_source,
            low_latency_overlay=low_latency_overlay_payload,
            gate_evaluation=gate_payload,
        )
        return None
    if (
        v7_position_entry_sizing is not None
        and entry_size_usdc < float(v7_position_entry_sizing["min_rebalance_usdc"])
    ):
        _log(
            log_path,
            "entry_skipped",
            reason="v7_position_target_below_min_rebalance",
            sleeve=sleeve,
            signal=asdict(signal),
            bid=bid,
            ask=ask,
            worst_price=worst_price,
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
            fresh_edge_at_worst=fresh_edge_at_worst,
            seconds_to_expiry=seconds_to_expiry,
            signal_age_seconds=signal_age_seconds,
            gate_evaluation=gate_payload,
            v7_position_entry_sizing=v7_position_entry_sizing,
        )
        return None
    if low_latency_overlay is not None:
        if low_latency_overlay_payload is None:
            overlay_decision = low_latency_overlay.evaluate_entry(
                asdict(signal),
                now_ms=_now_ms(),
            )
            low_latency_overlay_payload = overlay_decision.to_dict()
            low_latency_overlay_passed = overlay_decision.passed
        assert low_latency_overlay_passed is not None
        if not low_latency_overlay_passed:
            _log(
                log_path,
                "entry_skipped",
                reason=str(low_latency_overlay_payload["reason"]),
                sleeve=sleeve,
                signal=asdict(signal),
                bid=bid,
                ask=ask,
                worst_price=worst_price,
                slippage_worst_price=slippage_worst_price,
                max_acceptable_price=max_acceptable_price,
                order_limit_price=order_limit_price,
                fresh_edge_at_worst=fresh_edge_at_worst,
                seconds_to_expiry=seconds_to_expiry,
                settlement_edge_threshold=entry_policy.effective_settlement_edge_threshold,
                settlement_min_confidence=entry_policy.settlement_min_confidence,
                settlement_peak_confidence=settlement_peak_confidence_for_gate,
                signal_age_seconds=signal_age_seconds,
                settlement_price_gate_mode=_settlement_price_gate_mode_name(entry_gate_mode),
                gate_evaluation=gate_payload,
                orderbook_source=orderbook_source,
                low_latency_overlay=low_latency_overlay_payload,
            )
            return None
        gate_payload = {
            **gate_payload,
            "orderbook_source": orderbook_source,
            "low_latency_overlay": low_latency_overlay_payload,
        }
    if paper:
        return _open_paper_position(
            position_manager=position_manager,
            signal=signal,
            log_path=log_path,
            sleeve=sleeve,
            fill_price=worst_price,
            size_usdc=entry_size_usdc,
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
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
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
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
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
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
            opposite_token_id=signal.opposite_token_id,
            gate_evaluation=gate_payload,
        )
        return None
    complement_entry_price = _round_price(1.0 - float(complement_bid), tick_size)
    complement_fresh_edge = signal.token_probability - complement_entry_price
    complement_skip_reason = (
        _v7_entry_price_floor_skip_reason(
            ask=complement_entry_price,
            worst_price=complement_entry_price,
            policy=entry_policy,
        )
        if sleeve == "settlement" and entry_gate_mode == "v7-pnl"
        else None
    )
    if complement_skip_reason is None:
        complement_skip_reason = (
            settlement_cost_edge_skip_reason(
                fresh_edge_at_worst=complement_fresh_edge,
                policy=entry_policy,
                settlement_confidence=settlement_confidence_for_gate,
                settlement_peak_confidence=settlement_peak_confidence_for_gate,
                signal_age_seconds=signal_age_seconds,
            )
            if settlement_cost_edge_only
            else entry_price_skip_reason(
                ask=complement_entry_price,
                worst_price=complement_entry_price,
                fresh_edge_at_worst=complement_fresh_edge,
                seconds_to_expiry=seconds_to_expiry,
                policy=entry_policy,
            )
        )
    if (
        complement_skip_reason is None
        and sleeve == "settlement"
        and entry_gate_mode == "v7-pnl"
    ):
        complement_skip_reason = entry_price_drift_skip_reason(
            entry_price=complement_entry_price,
            signal_price=_v7_signal_reference_price(signal),
            policy=entry_policy,
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
            settlement_min_confidence=entry_policy.settlement_min_confidence,
            settlement_peak_confidence=settlement_peak_confidence_for_gate,
            settlement_peak_confidence_drop_tolerance=(
                entry_policy.settlement_peak_confidence_drop_tolerance
            ),
            max_signal_age_seconds=entry_policy.max_signal_age_seconds,
            signal_age_seconds=signal_age_seconds,
            settlement_price_gate_mode=_settlement_price_gate_mode_name(entry_gate_mode),
            gate_evaluation=gate_payload,
            complement_gate_evaluation=asdict(
                evaluate_entry_gates(
                    settlement_edge=(
                        complement_fresh_edge
                        if settlement_cost_edge_only
                        else signal.edge
                    ),
                    ask=complement_entry_price,
                    worst_price=complement_entry_price,
                    token_probability=signal.token_probability,
                    seconds_to_expiry=seconds_to_expiry,
                    policy=entry_policy,
                    settlement_confidence=settlement_confidence_for_gate,
                    settlement_peak_confidence=settlement_peak_confidence_for_gate,
                    signal_age_seconds=signal_age_seconds,
                    enable_settlement_gate=sleeve == "settlement",
                )
            ),
        )
        return None
    order = client.create_market_order(
        order_args=MarketOrderArgs(
            token_id=signal.token_id,
            side=BUY,
            amount=entry_size_usdc,
            price=order_limit_price,
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
            slippage_worst_price=slippage_worst_price,
            max_acceptable_price=max_acceptable_price,
            order_limit_price=order_limit_price,
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
        slippage_worst_price=slippage_worst_price,
        max_acceptable_price=max_acceptable_price,
        order_limit_price=order_limit_price,
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
        entry_p_up=signal.p_up,
        entry_p_down=signal.p_down,
        entry_p_neutral=signal.p_neutral,
        entry_model_probability=signal.model_probability,
        entry_polymarket_price=signal.polymarket_price,
        entry_mispricing_edge=signal.mispricing_edge,
        sleeve=sleeve,
        paper=paper,
    )
    if fill_price < entry_policy.min_entry_price and not settlement_cost_edge_only:
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
    if fill_price < entry_policy.min_entry_price and not settlement_cost_edge_only:
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
) -> LivePosition | None:
    fill_size = size_usdc / fill_price if fill_price > 0 else 0.0
    event_suffix = (signal.event_id or str(order_posted_at))[-8:]
    event_id = f"phase4-paper-{sleeve}-{signal.round_slug}-{signal.outcome_side}-{event_suffix}"
    order_id = f"paper-{sleeve}-{event_suffix}"
    existing_open = _find_open_position_for_round_sleeve(
        position_manager,
        round_slug=signal.round_slug,
        sleeve=sleeve,
    )
    if existing_open is not None:
        _log(
            log_path,
            "paper_entry_duplicate_open",
            reason="open_position_already_exists_for_round_sleeve",
            sleeve=sleeve,
            event_id=event_id,
            existing_position=_position_log_payload(existing_open),
            signal=asdict(signal),
            fill_price=fill_price,
            size_usdc=size_usdc,
            gate_evaluation=gate_payload,
        )
        return None
    try:
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
    except ValueError as exc:
        if "open position already exists for event_id=" not in str(exc):
            raise
        _log(
            log_path,
            "paper_entry_duplicate_open",
            reason="open_position_already_exists_for_event_id",
            sleeve=sleeve,
            event_id=event_id,
            signal=asdict(signal),
            fill_price=fill_price,
            size_usdc=size_usdc,
            gate_evaluation=gate_payload,
            error=str(exc),
        )
        return None
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
        entry_p_up=signal.p_up,
        entry_p_down=signal.p_down,
        entry_p_neutral=signal.p_neutral,
        entry_model_probability=signal.model_probability,
        entry_polymarket_price=signal.polymarket_price,
        entry_mispricing_edge=signal.mispricing_edge,
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


def _find_open_position_for_round_sleeve(
    position_manager: PositionManager,
    *,
    round_slug: str,
    sleeve: str,
) -> Any | None:
    open_positions_fn = getattr(position_manager, "get_all_open", None)
    list_positions_fn = getattr(position_manager, "list_positions", None)
    try:
        if callable(open_positions_fn):
            open_positions = open_positions_fn()
        elif callable(list_positions_fn):
            open_positions = list_positions_fn(status="open")
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    for position in open_positions:
        position_sleeve = str(getattr(position, "sleeve", "settlement"))
        if position_sleeve != sleeve:
            continue
        symbol = str(getattr(position, "symbol", ""))
        event_id = str(getattr(position, "event_id", ""))
        if round_slug in symbol or round_slug in event_id:
            return position
    return None


def _position_log_payload(position: Any) -> dict[str, Any]:
    if hasattr(position, "__dataclass_fields__"):
        return asdict(position)
    payload = getattr(position, "__dict__", None)
    return dict(payload) if isinstance(payload, dict) else {"value": str(position)}


def _maybe_v7_settlement_tick_exit(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    log_path: Path,
    seconds_to_expiry: float,
    config: V7SettlementPositionConfig | None,
    sell_slippage: float,
    exit_order_timeout_seconds: float,
    monitoring_db_path: str,
) -> SellResult | None:
    if (
        config is None
        or not config.enabled
        or not config.convergence_take_profit_enabled
        or position.sleeve != "settlement"
        or seconds_to_expiry > config.take_profit_force_exit_seconds
    ):
        return None
    position.v7_position_take_profit_candidate_count += 1
    position.v7_position_take_profit_candidate_reason = "convergence_force_exit_before_expiry"
    action = "HOLD"
    reason = "convergence_force_exit_before_expiry_hysteresis_wait"
    if position.v7_position_take_profit_candidate_count >= config.take_profit_hysteresis_bars:
        action = "EXIT"
        reason = "convergence_force_exit_before_expiry"
    evaluation = {
        "action": action,
        "reason": reason,
        "config": asdict(config),
        "take_profit_count": position.v7_position_take_profit_candidate_count,
        "take_profit_reason": position.v7_position_take_profit_candidate_reason,
        "seconds_to_expiry": seconds_to_expiry,
        "poll_loop": True,
    }
    _log(
        log_path,
        "v7_settlement_position_management_evaluated",
        position=asdict(position),
        signal=None,
        evaluation=evaluation,
    )
    if action != "EXIT":
        return None
    return _sell_settlement_policy_exit(
        client=client,
        position_manager=position_manager,
        position=position,
        signal=None,
        log_path=log_path,
        event_prefix="v7_settlement_position_exit",
        reason=reason,
        sell_slippage=sell_slippage,
        exit_order_timeout_seconds=exit_order_timeout_seconds,
        monitoring_db_path=monitoring_db_path,
    )


def _maybe_v7_settlement_position_adjustment(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    signal: SignalEvent,
    log_path: Path,
    config: V7SettlementPositionConfig,
    paper: bool,
    sell_slippage: float,
    exit_order_timeout_seconds: float,
    monitoring_db_path: str,
) -> PositionAdjustmentResult | None:
    if not config.enabled or position.sleeve != "settlement":
        return None
    if position.round_slug != signal.round_slug:
        return None
    if signal.created_at <= position.entry_signal_created_at:
        return None

    p_side, p_side_source = _v7_position_probability_for_side(signal, position.side)
    opposite_side = _opposite_side(position.side)
    p_opposite, p_opposite_source = _v7_position_probability_for_side(signal, opposite_side)
    if p_side is None and p_opposite is None:
        _log(
            log_path,
            "v7_settlement_position_management_skipped",
            reason="missing_probabilities",
            position=asdict(position),
            signal=asdict(signal),
            config=asdict(config),
        )
        return None

    try:
        hold_bid, hold_ask = _best_bid_ask(client, position.token_id)
    except OrderBookUnavailable as exc:
        _log(
            log_path,
            "v7_settlement_position_management_skipped",
            reason="position_orderbook_unavailable",
            position=asdict(position),
            signal=asdict(signal),
            config=asdict(config),
            **exc.to_log_payload(),
        )
        return None
    opposite_token_id = _opposite_token_id_for_signal(position, signal)
    opposite_ask: float | None = None
    if opposite_token_id:
        try:
            _opposite_bid, opposite_ask = _best_bid_ask(client, opposite_token_id)
        except OrderBookUnavailable as exc:
            _log(
                log_path,
                "v7_settlement_position_opposite_quote_missing",
                reason="opposite_orderbook_unavailable",
                position=asdict(position),
                signal=asdict(signal),
                opposite_token_id=opposite_token_id,
                **exc.to_log_payload(),
            )

    hold_edge = None if p_side is None or hold_bid is None else p_side - float(hold_bid)
    add_edge = None if p_side is None or hold_ask is None else p_side - float(hold_ask)
    reversal_edge = (
        None if p_opposite is None or opposite_ask is None else p_opposite - float(opposite_ask)
    )
    signal_price_for_side, signal_price_source = _v7_position_signal_price_for_side(
        signal, position.side
    )
    convergence = _v7_position_convergence_evaluation(
        position=position,
        p_side=p_side,
        signal_price_for_side=signal_price_for_side,
        signal_price_source=signal_price_source,
        config=config,
    )
    reversal_confidence_passed = (
        p_opposite is not None and p_opposite >= config.reversal_min_confidence
    )
    reversal_liquidity_blocked = reversal_confidence_passed and opposite_ask is None
    reversal_confirmed = (
        reversal_confidence_passed
        and reversal_edge is not None
        and reversal_edge >= config.reversal_min_edge
    )
    if reversal_confirmed:
        position.v7_position_reversal_candidate_count += 1
    else:
        position.v7_position_reversal_candidate_count = 0
    weak_hold = hold_edge is not None and hold_edge < config.weak_hold_edge
    if weak_hold:
        position.v7_position_weak_hold_candidate_count += 1
    else:
        position.v7_position_weak_hold_candidate_count = 0
    if convergence.get("diverged") is True:
        position.v7_position_divergence_candidate_count += 1
    elif convergence.get("diverged") is False:
        position.v7_position_divergence_candidate_count = 0
    seconds_to_expiry = (
        (signal.round_end_ts - _now_ms()) / 1000 if signal.round_end_ts else None
    )
    prior_cost = _position_cost_basis_usdc(position)
    avg_price = _position_average_price(position)
    low_confidence_scalp = _v7_low_confidence_scalp_profile(signal, config)
    take_profit_candidate, take_profit_reason = _v7_take_profit_exit_candidate(
        config=config,
        side=position.side,
        hold_edge=hold_edge,
        hold_bid=hold_bid,
        avg_price=avg_price,
        convergence=convergence,
        seconds_to_expiry=seconds_to_expiry,
        low_confidence_scalp=low_confidence_scalp,
    )
    if take_profit_candidate:
        position.v7_position_take_profit_candidate_count += 1
        position.v7_position_take_profit_candidate_reason = take_profit_reason
    else:
        position.v7_position_take_profit_candidate_count = 0
        position.v7_position_take_profit_candidate_reason = ""
    adverse_confidence = _v7_adverse_confidence_decay_evaluation(
        position=position,
        p_side=p_side,
        hold_bid=hold_bid,
        avg_price=avg_price,
        config=config,
    )
    if adverse_confidence.get("triggered") is True:
        position.v7_position_adverse_confidence_candidate_count += 1
    else:
        position.v7_position_adverse_confidence_candidate_count = 0
    adverse_confidence["candidate_count"] = (
        position.v7_position_adverse_confidence_candidate_count
    )

    target_cost = prior_cost
    divergence_reduce_allowed = _v7_divergence_reduce_allowed(
        hold_edge=hold_edge,
        config=config,
    )
    add_cooldown_remaining_seconds = _v7_add_cooldown_remaining_seconds(
        position=position,
        signal_created_at=signal.created_at,
        config=config,
    )
    adverse_confidence_reduce_allowed = _v7_adverse_confidence_reduce_allowed(
        position=position,
        model_decay=float(adverse_confidence.get("model_decay") or 0.0),
        config=config,
    )
    add_blocked_by_adverse_confidence_reduce = (
        config.block_add_after_adverse_confidence_reduce
        and position.v7_position_adverse_confidence_reduce_count > 0
    )
    take_profit_hysteresis_bars = _v7_take_profit_hysteresis_bars(
        config=config,
        low_confidence_scalp=low_confidence_scalp,
    )
    action = "HOLD"
    reason = "ev_hold"
    if position.v7_position_reversal_candidate_count >= config.reversal_hysteresis_bars:
        action = "EXIT"
        reason = "confirmed_opposite_ev_reversal"
        target_cost = 0.0
    elif (
        config.convergence_take_profit_enabled
        and position.v7_position_take_profit_candidate_count
        >= take_profit_hysteresis_bars
    ):
        action = "EXIT"
        reason = position.v7_position_take_profit_candidate_reason or "convergence_take_profit"
        target_cost = 0.0
    elif (
        config.convergence_take_profit_enabled
        and position.v7_position_take_profit_candidate_count > 0
    ):
        reason = (
            f"{position.v7_position_take_profit_candidate_reason or 'take_profit'}"
            "_hysteresis_wait"
        )
    elif (
        adverse_confidence.get("triggered") is True
        and position.v7_position_adverse_confidence_candidate_count
        >= config.adverse_confidence_hysteresis_bars
    ):
        shortfall = float(adverse_confidence.get("threshold_shortfall") or 0.0)
        model_decay = float(adverse_confidence.get("model_decay") or 0.0)
        full_exit_allowed = _v7_adverse_confidence_full_exit_allowed(
            shortfall=shortfall,
            model_decay=model_decay,
            hold_edge=hold_edge,
            config=config,
        )
        post_reduce_full_exit_allowed = (
            _v7_adverse_confidence_post_reduce_full_exit_allowed(
                position=position,
                candidate_count=position.v7_position_adverse_confidence_candidate_count,
                shortfall=shortfall,
                model_decay=model_decay,
                hold_edge=hold_edge,
                config=config,
            )
        )
        low_confidence_adverse_full_exit_allowed = (
            _v7_low_confidence_adverse_full_exit_allowed(
                low_confidence_scalp=low_confidence_scalp,
                model_decay=model_decay,
                config=config,
            )
        )
        adverse_confidence["full_exit_allowed"] = full_exit_allowed
        adverse_confidence["post_reduce_full_exit_allowed"] = (
            post_reduce_full_exit_allowed
        )
        adverse_confidence["low_confidence_full_exit_allowed"] = (
            low_confidence_adverse_full_exit_allowed
        )
        adverse_confidence["full_exit_min_model_decay"] = (
            config.adverse_confidence_full_exit_min_model_decay
        )
        adverse_confidence["full_exit_max_hold_edge"] = (
            config.adverse_confidence_full_exit_max_hold_edge
        )
        adverse_confidence["post_reduce_full_exit_bars"] = (
            config.adverse_confidence_post_reduce_full_exit_bars
        )
        adverse_confidence["post_reduce_full_exit_min_model_decay"] = (
            config.adverse_confidence_post_reduce_full_exit_min_model_decay
        )
        adverse_confidence["post_reduce_full_exit_max_hold_edge"] = (
            config.adverse_confidence_post_reduce_full_exit_max_hold_edge
        )
        dust_exit_allowed = _v7_adverse_confidence_dust_exit_allowed(
            prior_cost=prior_cost,
            projected_reduce_cost=max(0.0, prior_cost * (1.0 - config.reduce_fraction)),
            candidate_count=position.v7_position_adverse_confidence_candidate_count,
            config=config,
        )
        adverse_confidence["dust_exit_allowed"] = dust_exit_allowed
        adverse_confidence["dust_exit_max_cost"] = (
            config.adverse_confidence_dust_exit_max_cost
        )
        adverse_confidence["dust_exit_min_candidate_count"] = (
            config.adverse_confidence_dust_exit_min_candidate_count
        )
        reduce_model_decay_allowed = (
            model_decay >= config.adverse_confidence_reduce_min_model_decay
        )
        adverse_confidence["reduce_min_model_decay"] = (
            config.adverse_confidence_reduce_min_model_decay
        )
        adverse_confidence["reduce_model_decay_allowed"] = reduce_model_decay_allowed
        if full_exit_allowed:
            action = "EXIT"
            reason = "adverse_confidence_decay_exit"
            target_cost = 0.0
        elif low_confidence_adverse_full_exit_allowed:
            action = "EXIT"
            reason = "low_confidence_adverse_full_exit"
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
            target_cost = max(0.0, prior_cost * (1.0 - config.reduce_fraction))
        elif not reduce_model_decay_allowed:
            reason = "adverse_confidence_reduce_blocked_by_model_decay"
        else:
            reason = "adverse_confidence_reduce_blocked_by_max_reduces"
    elif adverse_confidence.get("triggered") is True:
        reason = "adverse_confidence_decay_hysteresis_wait"
    elif (
        hold_edge is not None
        and hold_edge <= config.exit_hold_edge
        and position.v7_position_weak_hold_candidate_count >= config.exit_hysteresis_bars
    ):
        action = "EXIT"
        reason = "confirmed_negative_hold_edge"
        target_cost = 0.0
    elif (
        hold_edge is not None
        and hold_edge < config.weak_hold_edge
        and position.v7_position_weak_hold_candidate_count >= config.exit_hysteresis_bars
    ):
        action = "REDUCE"
        reason = "weak_hold_edge_reduce"
        target_cost = max(0.0, prior_cost * (1.0 - config.reduce_fraction))
    elif (
        convergence.get("diverged") is True
        and position.v7_position_divergence_candidate_count
        >= config.divergence_hysteresis_bars
    ):
        if divergence_reduce_allowed:
            action = "REDUCE"
            reason = "residual_divergence_reduce"
            target_cost = max(0.0, prior_cost * (1.0 - config.reduce_fraction))
        else:
            reason = "residual_divergence_reduce_blocked_by_hold_edge"
    elif add_edge is not None and add_edge >= config.add_edge_min:
        if add_blocked_by_adverse_confidence_reduce:
            reason = "positive_add_edge_blocked_by_adverse_confidence_reduce"
        elif add_cooldown_remaining_seconds > 0:
            reason = "positive_add_edge_blocked_by_divergence_reduce_cooldown"
        elif convergence.get("diverged") is True:
            reason = "positive_add_edge_blocked_by_residual_divergence"
        else:
            target_cost = _v7_target_cost_from_edge(
                add_edge=add_edge,
                config=config,
            )
            if target_cost >= prior_cost + config.min_rebalance_usdc:
                action = "ADD"
                reason = "positive_add_edge"
            else:
                target_cost = prior_cost
    elif reversal_liquidity_blocked:
        reason = "exit_desired_but_liquidity_blocked"

    evaluation = {
        "action": action,
        "reason": reason,
        "p_side": p_side,
        "p_opposite": p_opposite,
        "p_side_source": p_side_source,
        "p_opposite_source": p_opposite_source,
        "hold_bid": hold_bid,
        "hold_ask": hold_ask,
        "opposite_token_id": opposite_token_id,
        "opposite_ask": opposite_ask,
        "hold_edge": hold_edge,
        "add_edge": add_edge,
        "reversal_edge": reversal_edge,
        "convergence": convergence,
        "reversal_confidence_passed": reversal_confidence_passed,
        "reversal_liquidity_blocked": reversal_liquidity_blocked,
        "prior_cost_basis_usdc": prior_cost,
        "target_cost_basis_usdc": target_cost,
        "avg_price": avg_price,
        "config": asdict(config),
        "reversal_count": position.v7_position_reversal_candidate_count,
        "weak_hold_count": position.v7_position_weak_hold_candidate_count,
        "divergence_count": position.v7_position_divergence_candidate_count,
        "divergence_reduce_allowed": divergence_reduce_allowed,
        "divergence_reduce_max_hold_edge": config.divergence_reduce_max_hold_edge,
        "take_profit_count": position.v7_position_take_profit_candidate_count,
        "take_profit_reason": position.v7_position_take_profit_candidate_reason,
        "take_profit_hysteresis_bars": take_profit_hysteresis_bars,
        "low_confidence_scalp": low_confidence_scalp,
        "adverse_confidence": adverse_confidence,
        "adverse_confidence_count": (
            position.v7_position_adverse_confidence_candidate_count
        ),
        "adverse_confidence_reduce_count": (
            position.v7_position_adverse_confidence_reduce_count
        ),
        "adverse_confidence_reduce_allowed": adverse_confidence_reduce_allowed,
        "adverse_confidence_max_reduces": config.adverse_confidence_max_reduces,
        "seconds_to_expiry": seconds_to_expiry,
        "add_cooldown_remaining_seconds": add_cooldown_remaining_seconds,
        "add_blocked_by_adverse_confidence_reduce": (
            add_blocked_by_adverse_confidence_reduce
        ),
        "last_divergence_reduce_at": position.v7_position_last_divergence_reduce_at,
        "last_adverse_confidence_reduce_at": (
            position.v7_position_last_adverse_confidence_reduce_at
        ),
    }
    _log(
        log_path,
        "v7_settlement_position_management_evaluated",
        position=asdict(position),
        signal=asdict(signal),
        evaluation=evaluation,
    )
    if action == "HOLD":
        return PositionAdjustmentResult(action=action, status="hold", reason=reason)
    if not paper or not config.paper_execute:
        _log(
            log_path,
            "v7_settlement_position_management_recommended",
            reason="live_recommendation_only" if not paper else "paper_execute_disabled",
            position=asdict(position),
            signal=asdict(signal),
            evaluation=evaluation,
        )
        return PositionAdjustmentResult(action=action, status="recommended", reason=reason)
    if action == "ADD":
        return _paper_v7_settlement_add(
            position_manager=position_manager,
            position=position,
            signal=signal,
            log_path=log_path,
            hold_ask=hold_ask,
            prior_cost=prior_cost,
            target_cost=target_cost,
            evaluation=evaluation,
        )
    if action == "REDUCE":
        return _paper_v7_settlement_reduce(
            position_manager=position_manager,
            position=position,
            signal=signal,
            log_path=log_path,
            hold_bid=hold_bid,
            prior_cost=prior_cost,
            target_cost=target_cost,
            evaluation=evaluation,
        )
    if action == "EXIT":
        sell_result = _sell_settlement_policy_exit(
            client=client,
            position_manager=position_manager,
            position=position,
            signal=signal,
            log_path=log_path,
            event_prefix="v7_settlement_position_exit",
            reason=reason,
            sell_slippage=sell_slippage,
            exit_order_timeout_seconds=exit_order_timeout_seconds,
            monitoring_db_path=monitoring_db_path,
            bid=float(hold_bid) if hold_bid is not None else None,
        )
        if sell_result is None:
            return None
        return PositionAdjustmentResult(
            action=action,
            status=sell_result.status,
            reason=sell_result.reason or reason,
            realized_pnl=sell_result.realized_pnl,
            closed=sell_result.status in {"filled", "pending_settlement", "pending_confirmation"},
        )
    return None


def _paper_v7_settlement_add(
    *,
    position_manager: PositionManager,
    position: LivePosition,
    signal: SignalEvent,
    log_path: Path,
    hold_ask: float | None,
    prior_cost: float,
    target_cost: float,
    evaluation: dict[str, Any],
) -> PositionAdjustmentResult | None:
    if hold_ask is None or hold_ask <= 0:
        _log(
            log_path,
            "v7_settlement_position_management_skipped",
            reason="missing_hold_ask_for_add",
            position=asdict(position),
            signal=asdict(signal),
            evaluation=evaluation,
        )
        return None
    add_usdc = max(0.0, target_cost - prior_cost)
    if add_usdc <= 0:
        return PositionAdjustmentResult(
            action="HOLD",
            status="hold",
            reason=str(evaluation.get("reason") or ""),
        )
    shares_delta = add_usdc / float(hold_ask)
    new_size = position.size + shares_delta
    new_cost = prior_cost + add_usdc
    new_avg = new_cost / new_size
    position.size = new_size
    position.fill_price = new_avg
    position_manager.adjust_open_position(
        position.event_id,
        fill_price=new_avg,
        size=new_size,
        current_price=float(hold_ask),
    )
    _log(
        log_path,
        "paper_v7_settlement_position_added",
        position=asdict(position),
        signal=asdict(signal),
        add_usdc=add_usdc,
        shares_delta=shares_delta,
        new_size=new_size,
        new_average_price=new_avg,
        evaluation=evaluation,
    )
    return PositionAdjustmentResult(
        action="ADD",
        status="filled",
        reason=str(evaluation.get("reason") or ""),
    )


def _paper_v7_settlement_reduce(
    *,
    position_manager: PositionManager,
    position: LivePosition,
    signal: SignalEvent,
    log_path: Path,
    hold_bid: float | None,
    prior_cost: float,
    target_cost: float,
    evaluation: dict[str, Any],
) -> PositionAdjustmentResult | None:
    if hold_bid is None:
        _log(
            log_path,
            "v7_settlement_position_management_skipped",
            reason="missing_hold_bid_for_reduce",
            position=asdict(position),
            signal=asdict(signal),
            evaluation=evaluation,
        )
        return None
    avg_price = _position_average_price(position)
    cost_to_sell = max(0.0, prior_cost - target_cost)
    shares_to_sell = min(position.size, cost_to_sell / max(avg_price, 1e-12))
    if shares_to_sell <= 0:
        return PositionAdjustmentResult(action="HOLD", status="hold")
    realized_delta = shares_to_sell * (float(hold_bid) - avg_price)
    new_size = max(0.0, position.size - shares_to_sell)
    if new_size <= 1e-12:
        closed = position_manager.close_position(position.event_id, float(hold_bid))
        pnl = float(closed.realized_pnl or 0.0)
        position.size = 0.0
        position.lifecycle_state = "EXIT_FILLED"
        position.last_lifecycle_reason = "v7_position_reduce_to_zero"
        if evaluation.get("reason") == "residual_divergence_reduce":
            position.v7_position_last_divergence_reduce_at = signal.created_at
        if evaluation.get("reason") == "adverse_confidence_decay_reduce":
            position.v7_position_adverse_confidence_reduce_count += 1
            position.v7_position_last_adverse_confidence_reduce_at = signal.created_at
        position.v7_position_realized_pnl_usdc += pnl
        _log(
            log_path,
            "paper_v7_settlement_position_reduced_to_exit",
            position=asdict(position),
            signal=asdict(signal),
            hold_bid=float(hold_bid),
            realized_pnl=pnl,
            evaluation=evaluation,
        )
        return PositionAdjustmentResult(
            action="EXIT",
            status="filled",
            reason=str(evaluation.get("reason") or ""),
            realized_pnl=pnl,
            closed=True,
        )
    position.size = new_size
    if evaluation.get("reason") == "residual_divergence_reduce":
        position.v7_position_last_divergence_reduce_at = signal.created_at
    if evaluation.get("reason") == "adverse_confidence_decay_reduce":
        position.v7_position_adverse_confidence_reduce_count += 1
        position.v7_position_last_adverse_confidence_reduce_at = signal.created_at
    position.v7_position_realized_pnl_usdc += realized_delta
    position_manager.adjust_open_position(
        position.event_id,
        fill_price=avg_price,
        size=new_size,
        current_price=float(hold_bid),
    )
    _log(
        log_path,
        "paper_v7_settlement_position_reduced",
        position=asdict(position),
        signal=asdict(signal),
        hold_bid=float(hold_bid),
        shares_sold=shares_to_sell,
        remaining_size=new_size,
        realized_pnl_delta=realized_delta,
        cumulative_position_realized_pnl=position.v7_position_realized_pnl_usdc,
        evaluation=evaluation,
    )
    return PositionAdjustmentResult(
        action="REDUCE",
        status="filled",
        reason=str(evaluation.get("reason") or ""),
        realized_pnl=realized_delta,
    )


def _signal_probability_for_side(signal: SignalEvent, side: str) -> float | None:
    if side == "UP":
        return signal.p_up
    if side == "DOWN":
        return signal.p_down
    return None


def _v7_position_probability_for_side(signal: SignalEvent, side: str) -> tuple[float | None, str]:
    if side not in {"UP", "DOWN"}:
        return None, "missing"
    if signal.outcome_side == side and signal.model_probability is not None:
        return signal.model_probability, "model_probability"
    if signal.outcome_side == _opposite_side(side) and signal.model_probability is not None:
        return 1.0 - signal.model_probability, "model_probability_complement"
    if signal.outcome_side == side:
        return signal.token_probability, "token_probability"
    if signal.outcome_side == _opposite_side(side):
        return 1.0 - signal.token_probability, "token_probability_complement"
    raw = _signal_probability_for_side(signal, side)
    if raw is not None:
        return raw, "raw_probability"
    return None, "missing"


def _v7_position_signal_price_for_side(
    signal: SignalEvent, side: str
) -> tuple[float | None, str]:
    if side not in {"UP", "DOWN"}:
        return None, "missing"
    raw_price = signal.polymarket_price
    source = "polymarket_price"
    if raw_price is None:
        raw_price = signal.market_implied_prob
        source = "market_implied_prob"
    if raw_price is None:
        return None, "missing"
    price = _clamp_probability(float(raw_price))
    if signal.outcome_side == side:
        return price, source
    if signal.outcome_side == _opposite_side(side):
        return _clamp_probability(1.0 - price), f"{source}_complement"
    return None, "missing"


def _v7_position_convergence_evaluation(
    *,
    position: LivePosition,
    p_side: float | None,
    signal_price_for_side: float | None,
    signal_price_source: str,
    config: V7SettlementPositionConfig,
) -> dict[str, Any]:
    entry_probability = position.entry_model_probability
    if entry_probability is None:
        return {
            "available": False,
            "reason": "missing_entry_model_probability",
            "signal_price_source": signal_price_source,
        }
    if p_side is None:
        return {
            "available": False,
            "reason": "missing_current_model_probability",
            "entry_model_probability": entry_probability,
            "signal_price_source": signal_price_source,
        }
    if signal_price_for_side is None:
        return {
            "available": False,
            "reason": "missing_current_signal_price",
            "entry_model_probability": entry_probability,
            "signal_price_source": signal_price_source,
        }
    entry_price = position.entry_price
    if entry_price <= 0:
        entry_price = position.entry_polymarket_price or position.entry_price
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
            "signal_price_source": signal_price_source,
        }
    direction = 1.0 if entry_residual > 0 else -1.0
    price_move_toward_model = (signal_price_for_side - entry_price) * direction
    model_move_toward_market = (p_side - entry_probability) * direction
    price_diverged = price_move_toward_model < -config.convergence_price_tolerance
    model_degraded = model_move_toward_market < -config.convergence_model_decay_tolerance
    price_converged = price_move_toward_model > config.convergence_price_tolerance
    residual_abs_ratio = abs(current_residual) / max(abs(entry_residual), 1e-12)
    diverged = bool(price_diverged or (model_degraded and not price_converged))
    return {
        "available": True,
        "entry_model_probability": entry_probability,
        "entry_price": entry_price,
        "entry_residual": entry_residual,
        "current_model_probability": p_side,
        "current_price": signal_price_for_side,
        "current_residual": current_residual,
        "signal_price_source": signal_price_source,
        "price_move_toward_model": price_move_toward_model,
        "model_move_toward_market": model_move_toward_market,
        "residual_abs_ratio": residual_abs_ratio,
        "price_converged": price_converged,
        "price_diverged": price_diverged,
        "model_degraded": model_degraded,
        "diverged": diverged,
    }


def _v7_adverse_confidence_decay_evaluation(
    *,
    position: LivePosition,
    p_side: float | None,
    hold_bid: float | None,
    avg_price: float | None,
    config: V7SettlementPositionConfig,
) -> dict[str, Any]:
    if not config.adverse_confidence_decay_enabled:
        return {"available": False, "reason": "disabled"}
    entry_probability = position.entry_model_probability
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
    if avg_price is None or avg_price <= 0:
        return {
            "available": False,
            "reason": "missing_average_price",
            "entry_model_probability": entry_probability,
            "current_model_probability": p_side,
            "hold_bid": hold_bid,
        }
    adverse_price_delta = max(0.0, float(avg_price) - float(hold_bid))
    model_decay = max(0.0, float(entry_probability) - float(p_side))
    raw_allowed_decay = (
        config.adverse_confidence_base_allowed_decay
        - adverse_price_delta * config.adverse_confidence_price_decay_slope
    )
    allowed_decay = max(
        config.adverse_confidence_min_allowed_decay,
        raw_allowed_decay,
    )
    required_p_side = min(
        config.adverse_confidence_max_required_probability,
        max(0.0, float(entry_probability) - allowed_decay),
    )
    threshold_shortfall = max(0.0, required_p_side - float(p_side))
    triggered = (
        adverse_price_delta + 1e-12 >= config.adverse_confidence_price_delta_start
        and threshold_shortfall > 0.0
    )
    return {
        "available": True,
        "triggered": triggered,
        "entry_model_probability": entry_probability,
        "current_model_probability": p_side,
        "avg_price": avg_price,
        "hold_bid": hold_bid,
        "adverse_price_delta": adverse_price_delta,
        "price_delta_start": config.adverse_confidence_price_delta_start,
        "model_decay": model_decay,
        "raw_allowed_decay": raw_allowed_decay,
        "allowed_decay": allowed_decay,
        "required_p_side": required_p_side,
        "threshold_shortfall": threshold_shortfall,
    }


def _v7_take_profit_exit_candidate(
    *,
    config: V7SettlementPositionConfig,
    side: str,
    hold_edge: float | None,
    hold_bid: float | None,
    avg_price: float | None,
    convergence: dict[str, Any],
    seconds_to_expiry: float | None,
    low_confidence_scalp: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if not config.convergence_take_profit_enabled:
        return False, ""
    low_confidence_scalp = low_confidence_scalp or {"active": False}
    low_confidence_active = bool(low_confidence_scalp.get("active"))
    tau = config.take_profit_hold_edge
    if side.upper() == "UP":
        tau = max(0.0, tau - config.take_profit_up_hold_edge_tighten)
    if (
        seconds_to_expiry is not None
        and seconds_to_expiry <= config.take_profit_force_exit_seconds
    ):
        return True, _v7_late_force_exit_reason(hold_bid=hold_bid, avg_price=avg_price)
    min_profit_delta = config.take_profit_min_profit_delta
    min_profit_return = config.take_profit_min_profit_return
    if low_confidence_active:
        min_profit_delta = min(
            min_profit_delta,
            config.low_confidence_scalp_take_profit_min_profit_delta,
        )
        min_profit_return = min(
            min_profit_return,
            config.low_confidence_scalp_take_profit_min_profit_return,
        )
    if _v7_profit_protect_take_profit_candidate(
        hold_bid=hold_bid,
        avg_price=avg_price,
        min_profit_delta=min_profit_delta,
        min_profit_return=min_profit_return,
    ):
        if low_confidence_active:
            return True, "low_confidence_scalp_take_profit"
        return True, "profit_protect_take_profit"
    if hold_edge is not None and hold_edge <= tau:
        return True, "convergence_edge_captured_take_profit"
    if not convergence.get("available"):
        return False, ""
    if convergence.get("price_converged") and convergence.get("model_degraded"):
        return True, "convergence_fake_convergence_model_decay"
    residual_ratio = float(convergence.get("residual_abs_ratio") or 1.0)
    if (
        convergence.get("price_converged")
        and residual_ratio <= config.take_profit_residual_ratio
    ):
        return True, "convergence_gap_filled_take_profit"
    entry_residual = abs(float(convergence.get("entry_residual") or 0.0))
    price_move_toward_model = float(convergence.get("price_move_toward_model") or 0.0)
    if (
        hold_edge is not None
        and entry_residual > 1e-12
        and price_move_toward_model >= config.take_profit_price_convergence_move
        and hold_edge
        <= entry_residual * config.take_profit_price_convergence_hold_edge_ratio
    ):
        return True, "convergence_price_move_take_profit"
    return False, ""


def _v7_low_confidence_scalp_profile(
    signal: SignalEvent,
    config: V7SettlementPositionConfig,
) -> dict[str, Any]:
    score = _v7_entry_candidate_score(signal)
    hit_5c = _v7_entry_candidate_hit_5c(signal)
    loss_10c = _v7_entry_candidate_loss_10c(signal)
    active = (
        config.low_confidence_scalp_enabled
        and score is not None
        and score <= config.low_confidence_scalp_max_confidence_score
    )
    return {
        "enabled": config.low_confidence_scalp_enabled,
        "active": active,
        "confidence_score": score,
        "max_confidence_score": config.low_confidence_scalp_max_confidence_score,
        "hit_5c_before_loss_10c": hit_5c,
        "loss_10c_before_hit_5c": loss_10c,
        "take_profit_min_profit_delta": (
            config.low_confidence_scalp_take_profit_min_profit_delta
        ),
        "take_profit_min_profit_return": (
            config.low_confidence_scalp_take_profit_min_profit_return
        ),
        "take_profit_hysteresis_bars": (
            config.low_confidence_scalp_take_profit_hysteresis_bars
        ),
        "adverse_full_exit_enabled": (
            config.low_confidence_scalp_adverse_full_exit_enabled
        ),
    }


def _v7_take_profit_hysteresis_bars(
    *,
    config: V7SettlementPositionConfig,
    low_confidence_scalp: dict[str, Any],
) -> int:
    bars = max(1, int(config.take_profit_hysteresis_bars))
    if bool(low_confidence_scalp.get("active")):
        bars = min(
            bars,
            max(1, int(config.low_confidence_scalp_take_profit_hysteresis_bars)),
        )
    return bars


def _v7_low_confidence_adverse_full_exit_allowed(
    *,
    low_confidence_scalp: dict[str, Any],
    model_decay: float,
    config: V7SettlementPositionConfig,
) -> bool:
    return (
        bool(low_confidence_scalp.get("active"))
        and config.low_confidence_scalp_adverse_full_exit_enabled
        and model_decay >= config.adverse_confidence_reduce_min_model_decay
    )


def _v7_profit_protect_take_profit_candidate(
    *,
    hold_bid: float | None,
    avg_price: float | None,
    min_profit_delta: float,
    min_profit_return: float,
) -> bool:
    if hold_bid is None or avg_price is None or avg_price <= 0:
        return False
    profit_delta = float(hold_bid) - float(avg_price)
    if min_profit_delta > 0.0 and profit_delta >= min_profit_delta:
        return True
    profit_return = profit_delta / float(avg_price)
    return min_profit_return > 0.0 and profit_return >= min_profit_return


def _v7_adverse_confidence_full_exit_allowed(
    *,
    shortfall: float,
    model_decay: float,
    hold_edge: float | None,
    config: V7SettlementPositionConfig,
) -> bool:
    if shortfall < config.adverse_confidence_exit_probability_buffer:
        return False
    if model_decay < config.adverse_confidence_full_exit_min_model_decay:
        return False
    max_hold_edge = config.adverse_confidence_full_exit_max_hold_edge
    if max_hold_edge < 0:
        return True
    return hold_edge is not None and hold_edge <= max_hold_edge


def _v7_adverse_confidence_dust_exit_allowed(
    *,
    prior_cost: float,
    projected_reduce_cost: float,
    candidate_count: int,
    config: V7SettlementPositionConfig,
) -> bool:
    max_cost = config.adverse_confidence_dust_exit_max_cost
    if max_cost <= 0.0:
        return False
    return (
        (prior_cost <= max_cost or projected_reduce_cost <= max_cost)
        and candidate_count >= config.adverse_confidence_dust_exit_min_candidate_count
    )


def _v7_adverse_confidence_post_reduce_full_exit_allowed(
    *,
    position: LivePosition,
    candidate_count: int,
    shortfall: float,
    model_decay: float,
    hold_edge: float | None,
    config: V7SettlementPositionConfig,
) -> bool:
    if not config.adverse_confidence_post_reduce_full_exit_enabled:
        return False
    if position.v7_position_adverse_confidence_reduce_count <= 0:
        return False
    max_reduces = config.adverse_confidence_max_reduces
    if max_reduces <= 0 or position.v7_position_adverse_confidence_reduce_count < max_reduces:
        return False
    required_count = (
        config.adverse_confidence_hysteresis_bars
        + config.adverse_confidence_post_reduce_full_exit_bars
    )
    if candidate_count < required_count:
        return False
    if shortfall < config.adverse_confidence_exit_probability_buffer:
        return False
    if model_decay < config.adverse_confidence_post_reduce_full_exit_min_model_decay:
        return False
    max_hold_edge = config.adverse_confidence_post_reduce_full_exit_max_hold_edge
    if max_hold_edge < 0:
        return True
    return hold_edge is not None and hold_edge <= max_hold_edge


def _v7_adverse_confidence_reduce_allowed(
    *,
    position: LivePosition,
    model_decay: float,
    config: V7SettlementPositionConfig,
) -> bool:
    if model_decay < config.adverse_confidence_reduce_min_model_decay:
        return False
    max_reduces = config.adverse_confidence_max_reduces
    return max_reduces <= 0 or position.v7_position_adverse_confidence_reduce_count < max_reduces


def _v7_divergence_reduce_allowed(
    *,
    hold_edge: float | None,
    config: V7SettlementPositionConfig,
) -> bool:
    if config.divergence_reduce_max_hold_edge < 0:
        return True
    return hold_edge is not None and hold_edge < config.divergence_reduce_max_hold_edge


def _v7_add_cooldown_remaining_seconds(
    *,
    position: LivePosition,
    signal_created_at: int,
    config: V7SettlementPositionConfig,
) -> float:
    if (
        config.add_cooldown_after_divergence_reduce_seconds <= 0
        or position.v7_position_last_divergence_reduce_at <= 0
        or signal_created_at <= 0
    ):
        return 0.0
    elapsed_seconds = (
        signal_created_at - position.v7_position_last_divergence_reduce_at
    ) / 1000.0
    return max(0.0, config.add_cooldown_after_divergence_reduce_seconds - elapsed_seconds)


def _clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _opposite_token_id_for_signal(position: LivePosition, signal: SignalEvent) -> str:
    if signal.outcome_side == _opposite_side(position.side):
        return signal.token_id
    return signal.opposite_token_id


def _position_cost_basis_usdc(position: LivePosition) -> float:
    return max(0.0, float(position.fill_price) * float(position.size))


def _position_average_price(position: LivePosition) -> float:
    if position.size <= 0:
        return 0.0
    return float(position.fill_price)


def _v7_late_force_exit_reason(
    *,
    hold_bid: float | None,
    avg_price: float | None,
) -> str:
    if hold_bid is None or avg_price is None or avg_price <= 0:
        return V7_SLOT_RELEASE_BEFORE_EXPIRY_REASON
    if hold_bid >= avg_price:
        return V7_PROFIT_LOCK_BEFORE_EXPIRY_REASON
    return V7_LOSS_SALVAGE_BEFORE_EXPIRY_REASON


def _v7_reclassify_late_force_exit_reason(
    *,
    reason: str,
    hold_bid: float | None,
    avg_price: float | None,
) -> str:
    if reason != V7_LATE_FORCE_EXIT_REASON:
        return reason
    return _v7_late_force_exit_reason(hold_bid=hold_bid, avg_price=avg_price)


def _v7_target_cost_from_edge(
    *,
    add_edge: float,
    config: V7SettlementPositionConfig,
) -> float:
    if add_edge < config.add_edge_min:
        return 0.0
    fraction = (add_edge - config.add_edge_min) / (config.full_add_edge - config.add_edge_min)
    return min(config.round_cap_usdc, max(0.0, fraction) * config.round_cap_usdc)


def _v7_initial_entry_sizing(
    *,
    sleeve: str,
    entry_gate_mode: str,
    fresh_edge_at_worst: float,
    max_position_size_usdc: float,
    config: V7SettlementPositionConfig | None,
) -> dict[str, float] | None:
    if (
        config is None
        or not config.enabled
        or sleeve != "settlement"
        or entry_gate_mode != "v7-pnl"
    ):
        return None
    target_cost_usdc = _v7_target_cost_from_edge(
        add_edge=fresh_edge_at_worst,
        config=config,
    )
    entry_size_usdc = min(max_position_size_usdc, target_cost_usdc)
    return {
        "fresh_edge_at_worst": fresh_edge_at_worst,
        "round_cap_usdc": config.round_cap_usdc,
        "max_position_size_usdc": max_position_size_usdc,
        "target_cost_usdc": target_cost_usdc,
        "entry_size_usdc": entry_size_usdc,
        "add_edge_min": config.add_edge_min,
        "full_add_edge": config.full_add_edge,
        "min_rebalance_usdc": config.min_rebalance_usdc,
    }


def _maybe_settlement_signal_exit(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    signal: SignalEvent,
    log_path: Path,
    config: SettlementExitConfig,
    v6_joint_config: V6JointGateConfig | None,
    signal_age_seconds: float | None,
    max_signal_age_seconds: float | None,
    seconds_to_expiry: float,
    opposite_exit_min_seconds_to_expiry: float,
    sell_slippage: float,
    exit_order_timeout_seconds: float = 20.0,
    monitoring_db_path: str = "data/mlops/champion_catalog.duckdb",
) -> tuple[str, SellResult] | None:
    """Evaluate fresh v6 settlement signals for mid-round settlement exits."""

    if position.sleeve != "settlement":
        return None
    track_reversal = config.allow_mid_round_exit or config.price_stop_enabled
    if not track_reversal and not config.confidence_decay_enabled:
        return None
    if (
        max_signal_age_seconds is not None
        and signal_age_seconds is not None
        and signal_age_seconds > max_signal_age_seconds
    ):
        _log(
            log_path,
            "settlement_signal_exit_skipped",
            reason="signal_age_above_threshold",
            position=asdict(position),
            signal=asdict(signal),
            signal_age_seconds=signal_age_seconds,
            max_signal_age_seconds=max_signal_age_seconds,
        )
        return None
    if seconds_to_expiry <= 0:
        return None
    payload = _settlement_payload_from_signal(signal)
    if payload is None:
        _log(
            log_path,
            "settlement_signal_exit_skipped",
            reason="missing_v6_probabilities",
            position=asdict(position),
            signal=asdict(signal),
        )
        return None

    if track_reversal and v6_joint_config is not None:
        admitted_side = evaluate_v6_settlement_side(payload, v6_joint_config)
        admitted_confidence = _settlement_side_probability(payload, admitted_side)
        _log(
            log_path,
            "settlement_reversal_exit_evaluated",
            position=asdict(position),
            signal=asdict(signal),
            admitted_side=admitted_side,
            admitted_confidence=admitted_confidence,
            position_side=position.side,
            reversal_min_confidence=config.reversal_min_confidence,
            hysteresis_bars=config.reversal_hysteresis_bars,
            exit_enabled=config.allow_mid_round_exit,
            price_stop_reversal_tracking=config.price_stop_enabled,
            seconds_to_expiry=seconds_to_expiry,
        )
        if admitted_side is None or admitted_side == position.side:
            position.settlement_reversal_candidate_side = ""
            position.settlement_reversal_candidate_count = 0
        elif seconds_to_expiry < opposite_exit_min_seconds_to_expiry:
            _log(
                log_path,
                "settlement_reversal_exit_skipped",
                reason="insufficient_time_remaining",
                position=asdict(position),
                signal=asdict(signal),
                admitted_side=admitted_side,
                seconds_to_expiry=seconds_to_expiry,
                opposite_exit_min_seconds_to_expiry=opposite_exit_min_seconds_to_expiry,
            )
        elif (
            admitted_confidence is None
            or admitted_confidence < config.reversal_min_confidence
        ):
            position.settlement_reversal_candidate_side = ""
            position.settlement_reversal_candidate_count = 0
            _log(
                log_path,
                "settlement_reversal_exit_skipped",
                reason="below_confidence",
                position=asdict(position),
                signal=asdict(signal),
                admitted_side=admitted_side,
                admitted_confidence=admitted_confidence,
                reversal_min_confidence=config.reversal_min_confidence,
            )
        else:
            if position.settlement_reversal_candidate_side == admitted_side:
                position.settlement_reversal_candidate_count += 1
            else:
                position.settlement_reversal_candidate_side = admitted_side
                position.settlement_reversal_candidate_count = 1
            if position.settlement_reversal_candidate_count < config.reversal_hysteresis_bars:
                _log(
                    log_path,
                    "settlement_reversal_exit_skipped",
                    reason="hysteresis_wait",
                    position=asdict(position),
                    signal=asdict(signal),
                    admitted_side=admitted_side,
                    admitted_confidence=admitted_confidence,
                    candidate_count=position.settlement_reversal_candidate_count,
                    hysteresis_bars=config.reversal_hysteresis_bars,
                )
            elif not config.allow_mid_round_exit:
                _log(
                    log_path,
                    "settlement_reversal_exit_skipped",
                    reason="mid_round_exit_disabled",
                    position=asdict(position),
                    signal=asdict(signal),
                    admitted_side=admitted_side,
                    admitted_confidence=admitted_confidence,
                    candidate_count=position.settlement_reversal_candidate_count,
                    hysteresis_bars=config.reversal_hysteresis_bars,
                    price_stop_reversal_tracking=config.price_stop_enabled,
                )
            else:
                sell_result = _sell_settlement_policy_exit(
                    client=client,
                    position_manager=position_manager,
                    position=position,
                    signal=signal,
                    log_path=log_path,
                    event_prefix="settlement_reversal_exit",
                    reason="settlement_reversal_exit",
                    sell_slippage=sell_slippage,
                    exit_order_timeout_seconds=exit_order_timeout_seconds,
                    monitoring_db_path=monitoring_db_path,
                )
                if sell_result is not None:
                    return "settlement_reversal_exit", sell_result

    if config.confidence_decay_enabled:
        position_confidence = _settlement_side_probability(payload, position.side)
        opposite_confidence = _settlement_side_probability(payload, _opposite_side(position.side))
        baseline = _position_entry_side_probability(position)
        below_floor = (
            position_confidence is not None
            and position_confidence < config.decay_floor
        )
        below_delta = (
            position_confidence is not None
            and baseline is not None
            and baseline - position_confidence >= config.decay_delta
        )
        regime_shift = (
            position_confidence is not None
            and opposite_confidence is not None
            and opposite_confidence > position_confidence
        )
        opposite_confidence_passed = (
            opposite_confidence is not None
            and config.decay_opposite_min_confidence is not None
            and opposite_confidence >= config.decay_opposite_min_confidence
        )
        decay_condition = (
            below_floor and below_delta and regime_shift and opposite_confidence_passed
        )
        if decay_condition:
            position.settlement_decay_candidate_count += 1
        else:
            position.settlement_decay_candidate_count = 0
        hysteresis_met = (
            position.settlement_decay_candidate_count >= config.decay_hysteresis_bars
        )
        should_exit = decay_condition and hysteresis_met
        _log(
            log_path,
            "settlement_confidence_decay_exit_evaluated",
            position=asdict(position),
            signal=asdict(signal),
            position_confidence=position_confidence,
            opposite_confidence=opposite_confidence,
            entry_baseline_confidence=baseline,
            below_floor=below_floor,
            below_delta=below_delta,
            regime_shift=regime_shift,
            opposite_confidence_passed=opposite_confidence_passed,
            decay_opposite_min_confidence=config.decay_opposite_min_confidence,
            decay_condition=decay_condition,
            decay_candidate_count=position.settlement_decay_candidate_count,
            decay_hysteresis_bars=config.decay_hysteresis_bars,
            should_exit=should_exit,
            seconds_to_expiry=seconds_to_expiry,
        )
        if seconds_to_expiry < config.stop_min_seconds_to_expiry:
            _log(
                log_path,
                "settlement_confidence_decay_exit_skipped",
                reason="insufficient_time_remaining",
                position=asdict(position),
                signal=asdict(signal),
                seconds_to_expiry=seconds_to_expiry,
                min_seconds_to_expiry=config.stop_min_seconds_to_expiry,
            )
        elif decay_condition and not hysteresis_met:
            _log(
                log_path,
                "settlement_confidence_decay_exit_skipped",
                reason="decay_hysteresis_wait",
                position=asdict(position),
                signal=asdict(signal),
                decay_candidate_count=position.settlement_decay_candidate_count,
                decay_hysteresis_bars=config.decay_hysteresis_bars,
            )
        elif should_exit:
            sell_result = _sell_settlement_policy_exit(
                client=client,
                position_manager=position_manager,
                position=position,
                signal=signal,
                log_path=log_path,
                event_prefix="settlement_confidence_decay_exit",
                reason="settlement_confidence_decay_exit",
                sell_slippage=sell_slippage,
                exit_order_timeout_seconds=exit_order_timeout_seconds,
                monitoring_db_path=monitoring_db_path,
            )
            if sell_result is not None:
                return "settlement_confidence_decay_exit", sell_result
    return None


def _record_settlement_same_side_confirmation(
    *,
    position: LivePosition,
    signal: SignalEvent,
    log_path: Path,
    config: SettlementExitConfig,
    signal_age_seconds: float | None,
    max_signal_age_seconds: float | None,
) -> bool:
    """Remember fresh post-entry same-side settlement confidence for stop vetoes."""

    if (
        position.sleeve != "settlement"
        or signal.round_slug != position.round_slug
        or not config.price_stop_same_side_confirmation_veto_enabled
    ):
        return False
    if (
        max_signal_age_seconds is not None
        and signal_age_seconds is not None
        and signal_age_seconds > max_signal_age_seconds
    ):
        return False
    if signal.created_at <= position.entry_signal_created_at:
        return False
    payload = _settlement_payload_from_signal(signal)
    if payload is None:
        return False
    confidence = _settlement_side_probability(payload, position.side)
    opposite_confidence = _settlement_side_probability(payload, _opposite_side(position.side))
    min_confidence = config.price_stop_same_side_confirmation_min_confidence
    if confidence is None or opposite_confidence is None or confidence < min_confidence:
        return False
    if position.side == "UP" and confidence < opposite_confidence:
        return False
    if position.side == "DOWN" and confidence <= opposite_confidence:
        return False
    if signal.created_at < position.settlement_same_side_confirmation_created_at:
        return False
    if (
        signal.created_at == position.settlement_same_side_confirmation_created_at
        and signal.event_id <= position.settlement_same_side_confirmation_event_id
    ):
        return False

    position.settlement_same_side_confirmation_event_id = signal.event_id
    position.settlement_same_side_confirmation_created_at = signal.created_at
    position.settlement_same_side_confirmation_confidence = confidence
    _log(
        log_path,
        "settlement_same_side_confirmation_updated",
        position=asdict(position),
        signal=asdict(signal),
        same_side_confidence=confidence,
        opposite_confidence=opposite_confidence,
        min_confidence=min_confidence,
        signal_age_seconds=signal_age_seconds,
        max_signal_age_seconds=max_signal_age_seconds,
    )
    return True


def _settlement_same_side_confirmation_veto_payload(
    *,
    position: LivePosition,
    config: SettlementExitConfig,
    now_ms: int,
) -> dict[str, Any] | None:
    if not config.price_stop_same_side_confirmation_veto_enabled:
        return None
    if position.settlement_same_side_confirmation_created_at <= position.entry_signal_created_at:
        return None
    confidence = position.settlement_same_side_confirmation_confidence
    min_confidence = config.price_stop_same_side_confirmation_min_confidence
    if confidence < min_confidence:
        return None
    confirmation_age_seconds = max(
        0.0,
        (now_ms - position.settlement_same_side_confirmation_created_at) / 1000,
    )
    max_age_seconds = config.price_stop_same_side_confirmation_max_age_seconds
    if max_age_seconds is not None and confirmation_age_seconds > max_age_seconds:
        return None
    return {
        "event_id": position.settlement_same_side_confirmation_event_id,
        "created_at": position.settlement_same_side_confirmation_created_at,
        "confidence": confidence,
        "min_confidence": min_confidence,
        "age_seconds": confirmation_age_seconds,
        "max_age_seconds": max_age_seconds,
    }


def _settlement_reversal_confirmation_payload(
    *,
    position: LivePosition,
    config: SettlementExitConfig,
) -> dict[str, Any]:
    opposite_side = _opposite_side(position.side)
    candidate_count = (
        position.settlement_reversal_candidate_count
        if position.settlement_reversal_candidate_side == opposite_side
        else 0
    )
    return {
        "required_side": opposite_side,
        "candidate_side": position.settlement_reversal_candidate_side,
        "candidate_count": candidate_count,
        "hysteresis_bars": config.reversal_hysteresis_bars,
        "confirmed": bool(
            opposite_side
            and candidate_count >= config.reversal_hysteresis_bars
        ),
    }


def _maybe_settlement_price_stop_exit(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    log_path: Path,
    seconds_to_expiry: float,
    config: SettlementExitConfig | None,
    sell_slippage: float,
    exit_order_timeout_seconds: float = 20.0,
    monitoring_db_path: str = "data/mlops/champion_catalog.duckdb",
) -> SellResult | None:
    if config is None or not config.price_stop_enabled or position.sleeve != "settlement":
        return None
    if seconds_to_expiry < config.stop_min_seconds_to_expiry:
        _log(
            log_path,
            "settlement_stop_exit_skipped",
            reason="insufficient_time_remaining",
            position=asdict(position),
            seconds_to_expiry=seconds_to_expiry,
            min_seconds_to_expiry=config.stop_min_seconds_to_expiry,
        )
        return None
    try:
        bid, _ask = _best_bid_ask(client, position.token_id)
    except OrderBookUnavailable as exc:
        _log(
            log_path,
            "settlement_stop_exit_skipped",
            reason="orderbook_unavailable",
            position=asdict(position),
            seconds_to_expiry=seconds_to_expiry,
            **exc.to_log_payload(),
        )
        return None
    if bid is None:
        _log(
            log_path,
            "settlement_stop_exit_skipped",
            reason="missing_bid",
            position=asdict(position),
            seconds_to_expiry=seconds_to_expiry,
        )
        return None
    unrealized_pnl = (float(bid) - position.fill_price) * position.size
    price_breach = float(bid) <= position.fill_price - config.stop_price_delta
    loss_breach = unrealized_pnl <= -config.stop_loss_usdc
    reversal_confirmation = _settlement_reversal_confirmation_payload(
        position=position,
        config=config,
    )
    _log(
        log_path,
        "settlement_stop_exit_evaluated",
        position=asdict(position),
        bid=float(bid),
        fill_price=position.fill_price,
        unrealized_pnl=unrealized_pnl,
        price_breach=price_breach,
        loss_breach=loss_breach,
        stop_price_delta=config.stop_price_delta,
        stop_loss_usdc=config.stop_loss_usdc,
        reversal_confirmation=reversal_confirmation,
        seconds_to_expiry=seconds_to_expiry,
    )
    if not price_breach and not loss_breach:
        return None
    if not reversal_confirmation["confirmed"]:
        _log(
            log_path,
            "settlement_stop_exit_skipped",
            reason="reversal_confirmation_required",
            position=asdict(position),
            bid=float(bid),
            fill_price=position.fill_price,
            unrealized_pnl=unrealized_pnl,
            price_breach=price_breach,
            loss_breach=loss_breach,
            stop_price_delta=config.stop_price_delta,
            stop_loss_usdc=config.stop_loss_usdc,
            reversal_confirmation=reversal_confirmation,
            seconds_to_expiry=seconds_to_expiry,
        )
        return None
    same_side_veto = _settlement_same_side_confirmation_veto_payload(
        position=position,
        config=config,
        now_ms=_now_ms(),
    )
    if same_side_veto is not None:
        _log(
            log_path,
            "settlement_stop_exit_skipped",
            reason="same_side_confirmation_veto",
            position=asdict(position),
            bid=float(bid),
            fill_price=position.fill_price,
            unrealized_pnl=unrealized_pnl,
            price_breach=price_breach,
            loss_breach=loss_breach,
            stop_price_delta=config.stop_price_delta,
            stop_loss_usdc=config.stop_loss_usdc,
            seconds_to_expiry=seconds_to_expiry,
            same_side_confirmation=same_side_veto,
        )
        return None
    return _sell_settlement_policy_exit(
        client=client,
        position_manager=position_manager,
        position=position,
        signal=None,
        log_path=log_path,
        event_prefix="settlement_stop_exit",
        reason="settlement_price_stop_exit",
        bid=float(bid),
        sell_slippage=sell_slippage,
        exit_order_timeout_seconds=exit_order_timeout_seconds,
        monitoring_db_path=monitoring_db_path,
    )


def _sell_settlement_policy_exit(
    *,
    client: Any,
    position_manager: PositionManager,
    position: LivePosition,
    signal: SignalEvent | None,
    log_path: Path,
    event_prefix: str,
    reason: str,
    sell_slippage: float,
    exit_order_timeout_seconds: float,
    monitoring_db_path: str,
    bid: float | None = None,
) -> SellResult | None:
    if bid is None:
        try:
            bid, _ask = _best_bid_ask(client, position.token_id)
        except OrderBookUnavailable as exc:
            _log(
                log_path,
                f"{event_prefix}_skipped",
                reason="orderbook_unavailable",
                position=asdict(position),
                signal=None if signal is None else asdict(signal),
                **exc.to_log_payload(),
            )
            return None
        if bid is None:
            _log(
                log_path,
                f"{event_prefix}_skipped",
                reason="missing_bid",
                position=asdict(position),
                signal=None if signal is None else asdict(signal),
            )
            return None
    legacy_reason = reason
    reason = _v7_reclassify_late_force_exit_reason(
        reason=reason,
        hold_bid=float(bid),
        avg_price=_position_average_price(position),
    )
    sell_result = _sell_position(
        client=client,
        position_manager=position_manager,
        position=position,
        log_path=log_path,
        bid=float(bid),
        sell_slippage=sell_slippage,
        fill_confirm_timeout_seconds=exit_order_timeout_seconds,
        reason=reason,
        signal=signal,
        monitoring_db_path=monitoring_db_path,
    )
    if sell_result is None:
        _log(
            log_path,
            f"{event_prefix}_skipped",
            reason="sell_not_filled",
            legacy_reason=legacy_reason if legacy_reason != reason else None,
            position=asdict(position),
            signal=None if signal is None else asdict(signal),
            bid=float(bid),
        )
    elif sell_result.status == "filled":
        _log(
            log_path,
            f"{event_prefix}_filled",
            position=asdict(position),
            signal=None if signal is None else asdict(signal),
            bid=float(bid),
            reason=reason,
            legacy_reason=legacy_reason if legacy_reason != reason else None,
            realized_pnl=sell_result.realized_pnl,
            account_cash_pnl=sell_result.account_cash_pnl,
        )
    else:
        _log(
            log_path,
            f"{event_prefix}_pending",
            status=sell_result.status,
            reason=reason,
            legacy_reason=legacy_reason if legacy_reason != reason else None,
            position=asdict(position),
            signal=None if signal is None else asdict(signal),
            bid=float(bid),
        )
    return sell_result


def _settlement_payload_from_signal(signal: SignalEvent) -> dict[str, float | str] | None:
    if signal.p_up is None or signal.p_down is None:
        return None
    return v6_payload_from_values(
        model_version="xgboost-v6",
        p_up=signal.p_up,
        p_down=signal.p_down,
        p_neutral=0.0 if signal.p_neutral is None else signal.p_neutral,
        p_vol_up=0.0 if signal.p_vol_up is None else signal.p_vol_up,
        p_vol_down=0.0 if signal.p_vol_down is None else signal.p_vol_down,
    )


def _settlement_side_probability(
    payload: dict[str, float | str],
    side: str | None,
) -> float | None:
    if side == "UP":
        return float(payload["p_up"])
    if side == "DOWN":
        return float(payload["p_down"])
    return None


def _position_entry_side_probability(position: LivePosition) -> float | None:
    if position.side == "UP":
        return position.entry_p_up
    if position.side == "DOWN":
        return position.entry_p_down
    return None


def _opposite_side(side: str) -> str:
    return "DOWN" if side == "UP" else "UP"


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
    return SellResult(status="settled", realized_pnl=pnl, account_cash_pnl=pnl, reason=reason)


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
    return SellResult(status="pending_settlement", reason=reason)


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
        return SellResult(status="filled", realized_pnl=pnl, account_cash_pnl=pnl, reason=reason)

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
        return SellResult(status="pending_confirmation", reason=reason)
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
    return SellResult(
        status="filled",
        realized_pnl=pnl,
        account_cash_pnl=account_cash_pnl,
        reason=reason,
    )


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _best_bid_ask(client: Any, token_id: str) -> tuple[float | None, float | None]:
    try:
        book = client.get_order_book(token_id)
    except Exception as exc:  # noqa: BLE001
        if not _env_bool("POLYMARKET_ORDERBOOK_REST_FALLBACK", default=False):
            raise OrderBookUnavailable(token_id, exc) from exc
        try:
            book = _load_order_book_rest(token_id)
        except Exception as fallback_exc:  # noqa: BLE001
            combined = RuntimeError(
                f"{type(exc).__name__}: {exc}; REST fallback failed: "
                f"{type(fallback_exc).__name__}: {fallback_exc}"
            )
            raise OrderBookUnavailable(token_id, combined) from fallback_exc
    raw = book if isinstance(book, dict) else getattr(book, "__dict__", {})
    bids = raw.get("bids") or []
    asks = raw.get("asks") or []
    bid = _best_price(bids, want_max=True)
    ask = _best_price(asks, want_max=False)
    return bid, ask


def _load_order_book_rest(token_id: str) -> dict[str, Any]:
    host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com").rstrip("/")
    query = parse.urlencode({"token_id": token_id})
    timeout = float(os.getenv("POLYMARKET_ORDERBOOK_REST_TIMEOUT_SECONDS", "5"))
    req = request.Request(
        f"{host}/book?{query}",
        headers={"accept": "application/json", "user-agent": "BiGan phase4 executor"},
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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


def _floor_price(price: float, tick_size: Any) -> float:
    tick = float(tick_size)
    if tick <= 0:
        return round(price, 4)
    return round(math.floor(price / tick) * tick, 4)


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


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
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
