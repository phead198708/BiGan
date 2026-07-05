"""Diagnostic-only v8 execution layer v2 for signal-to-position control.

This module intentionally stops at deterministic paper diagnostics.  It does
not place orders, mutate O source scores, or unlock paper/live execution.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields

EXECUTION_LAYER_V2_SCHEMA_VERSION = "bigan-v8-polymarket-execution-layer-v2-v1"
EXECUTION_LAYER_V2_REPORT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-report-v1"
)
EXECUTION_LAYER_V2_POLICY_NAME = (
    "v8_execution_layer_v2_signal_position_dynamic_exit"
)
EXECUTION_LAYER_V2_BASELINE_NAME = "hold_to_settlement_v1"

ExecutionLayerV2Side = Literal["UP", "DOWN", "NONE"]
ExecutionLayerV2Action = Literal[
    "NO_ACTION",
    "ENTER_POSITION",
    "HOLD_POSITION",
    "EXIT_POSITION",
    "ROTATE_POSITION",
]
ExecutionLayerV2State = Literal["NO_POSITION", "ACTIVE", "DECAYING", "EXIT"]

EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS: tuple[str, ...] = (
    "realized_pnl",
    "realized_trade_pnl",
    "settlement_pnl",
    "settlement_label",
    "oracle_action",
    "oracle_side",
    "future_return",
    "future_price",
    "future_outcome",
    "total_polymarket_pnl",
    "winning_outcome",
    "resolved_outcome",
    "action_return_target",
    "label_return",
)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2Config:
    """Paper-only configuration for dynamic v8 position management."""

    entry_ev_threshold: float = 0.02
    min_confidence: float = 0.55
    hold_ev_floor_ratio: float = 0.60
    opposite_signal_ev_margin: float = 0.02
    time_exit_threshold_seconds: float = 60.0
    execution_cost_bps: float = 10.0
    nav_usdc: float = 10_000.0
    max_nav_fraction_per_position: float = 0.05
    min_nav_fraction_per_position: float = 0.0
    kelly_time_decay_lambda: float = 0.0005
    diagnostic_lambda_grid: tuple[float, ...] = (0.0, 0.0005, 0.001)
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False
    source_model_candidate_eligible: bool = False
    freeze_ready: bool = False
    promotion_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "entry_ev_threshold",
            "min_confidence",
            "hold_ev_floor_ratio",
            "opposite_signal_ev_margin",
            "time_exit_threshold_seconds",
            "execution_cost_bps",
            "nav_usdc",
            "max_nav_fraction_per_position",
            "min_nav_fraction_per_position",
            "kelly_time_decay_lambda",
        ):
            _require_finite(field_name, float(getattr(self, field_name)))
        if self.entry_ev_threshold < 0.0:
            raise ValueError("entry_ev_threshold must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.hold_ev_floor_ratio <= 1.0:
            raise ValueError("hold_ev_floor_ratio must be in [0, 1]")
        if self.opposite_signal_ev_margin < 0.0:
            raise ValueError("opposite_signal_ev_margin must be non-negative")
        if self.time_exit_threshold_seconds < 0.0:
            raise ValueError("time_exit_threshold_seconds must be non-negative")
        if self.execution_cost_bps < 0.0:
            raise ValueError("execution_cost_bps must be non-negative")
        if self.nav_usdc <= 0.0:
            raise ValueError("nav_usdc must be positive")
        if not 0.0 <= self.min_nav_fraction_per_position <= self.max_nav_fraction_per_position:
            raise ValueError("nav fractions must satisfy 0 <= min <= max")
        if self.max_nav_fraction_per_position > 1.0:
            raise ValueError("max_nav_fraction_per_position must be <= 1")
        if self.kelly_time_decay_lambda < 0.0:
            raise ValueError("kelly_time_decay_lambda must be non-negative")
        if not self.diagnostic_lambda_grid:
            raise ValueError("diagnostic_lambda_grid is required")
        for value in self.diagnostic_lambda_grid:
            _require_finite("diagnostic_lambda_grid", float(value))
            if value < 0.0:
                raise ValueError("diagnostic_lambda_grid values must be non-negative")
        _validate_safety_flags(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2Signal:
    """One decision-time signal row available before settlement/outcome."""

    market_id: str
    decision_ts: int
    p_up: float
    ask_up: float
    ask_down: float
    p_down: float | None = None
    bid_up: float | None = None
    bid_down: float | None = None
    time_to_expiry_seconds: float | None = None
    source_signal_id: str | None = None
    model_score: float | None = None
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        if not self.market_id.strip():
            raise ValueError("market_id is required")
        if self.decision_ts < 0:
            raise ValueError("decision_ts must be non-negative")
        _validate_probability("p_up", self.p_up)
        p_down = 1.0 - self.p_up if self.p_down is None else self.p_down
        _validate_probability("p_down", p_down)
        object.__setattr__(self, "p_down", p_down)
        for field_name in ("ask_up", "ask_down"):
            _validate_price(field_name, float(getattr(self, field_name)))
        for field_name in ("bid_up", "bid_down", "time_to_expiry_seconds", "model_score"):
            value = getattr(self, field_name)
            if value is not None:
                _require_finite(field_name, float(value))
        if self.bid_up is not None:
            _validate_price("bid_up", self.bid_up, allow_zero=True)
        if self.bid_down is not None:
            _validate_price("bid_down", self.bid_down, allow_zero=True)
        if self.time_to_expiry_seconds is not None and self.time_to_expiry_seconds < 0.0:
            raise ValueError("time_to_expiry_seconds must be non-negative")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2Position:
    """Paper-only open position state used by the v2 state machine."""

    market_id: str
    side: ExecutionLayerV2Side
    entry_ts: int
    entry_price: float
    entry_probability: float
    entry_ev: float
    size_usdc: float
    shares: float
    state: ExecutionLayerV2State = "ACTIVE"
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        if not self.market_id.strip():
            raise ValueError("market_id is required")
        if self.side not in ("UP", "DOWN"):
            raise ValueError("position side must be UP or DOWN")
        if self.entry_ts < 0:
            raise ValueError("entry_ts must be non-negative")
        _validate_price("entry_price", self.entry_price)
        _validate_probability("entry_probability", self.entry_probability)
        for field_name in ("entry_ev", "size_usdc", "shares"):
            _require_finite(field_name, float(getattr(self, field_name)))
        if self.size_usdc <= 0.0 or self.shares <= 0.0:
            raise ValueError("position size_usdc and shares must be positive")
        if self.state not in ("ACTIVE", "DECAYING"):
            raise ValueError("open position state must be ACTIVE or DECAYING")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2Decision:
    """One paper-only state-machine decision."""

    market_id: str
    decision_ts: int
    action: ExecutionLayerV2Action
    target_side: ExecutionLayerV2Side
    state_before: ExecutionLayerV2State
    state_after: ExecutionLayerV2State
    selected_ev_t: float
    entry_ev_reference: float
    ev_ratio_to_entry: float | None
    confidence: float
    execution_price: float
    paper_notional: float
    shares: float
    kelly_fraction: float
    time_decay_multiplier: float
    reason_codes: tuple[str, ...]
    source_signal_id: str | None = None
    baseline_v1_action: str = EXECUTION_LAYER_V2_BASELINE_NAME
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False

    def __post_init__(self) -> None:
        if self.action not in {
            "NO_ACTION",
            "ENTER_POSITION",
            "HOLD_POSITION",
            "EXIT_POSITION",
            "ROTATE_POSITION",
        }:
            raise ValueError("unsupported execution layer v2 action")
        if self.target_side not in ("UP", "DOWN", "NONE"):
            raise ValueError("unsupported target_side")
        if self.state_before not in ("NO_POSITION", "ACTIVE", "DECAYING", "EXIT"):
            raise ValueError("unsupported state_before")
        if self.state_after not in ("NO_POSITION", "ACTIVE", "DECAYING", "EXIT"):
            raise ValueError("unsupported state_after")
        for field_name in (
            "selected_ev_t",
            "entry_ev_reference",
            "confidence",
            "execution_price",
            "paper_notional",
            "shares",
            "kelly_fraction",
            "time_decay_multiplier",
        ):
            _require_finite(field_name, float(getattr(self, field_name)))
        if self.ev_ratio_to_entry is not None:
            _require_finite("ev_ratio_to_entry", self.ev_ratio_to_entry)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.execution_price < 0.0 or self.paper_notional < 0.0 or self.shares < 0.0:
            raise ValueError("execution values must be non-negative")
        if not self.reason_codes:
            raise ValueError("reason_codes are required")
        _validate_safety_flags(self)

    @property
    def state_transition(self) -> str:
        return f"{self.state_before}->{self.state_after}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        payload["state_transition"] = self.state_transition
        payload["execution_layer_v2_decision_hash"] = canonical_json_sha256(payload)
        return payload


class ExecutionLayerV2Engine:
    """Sequential paper-only executor for v2 state-machine decisions."""

    def __init__(self, config: ExecutionLayerV2Config | None = None) -> None:
        self.config = config or ExecutionLayerV2Config()
        self._positions: dict[str, ExecutionLayerV2Position] = {}
        self._seen_keys: set[tuple[int, str]] = set()
        self._last_key: tuple[int, str] | None = None

    @property
    def positions(self) -> dict[str, ExecutionLayerV2Position]:
        return dict(self._positions)

    def decide_many(
        self,
        signals: tuple[ExecutionLayerV2Signal, ...] | list[ExecutionLayerV2Signal],
    ) -> tuple[ExecutionLayerV2Decision, ...]:
        decisions = []
        for signal in sorted(signals, key=lambda row: (row.decision_ts, row.market_id)):
            decisions.append(self.decide(signal))
        return tuple(decisions)

    def decide(self, signal: ExecutionLayerV2Signal) -> ExecutionLayerV2Decision:
        key = (int(signal.decision_ts), str(signal.market_id))
        if key in self._seen_keys:
            raise ValueError("duplicate_execution_layer_v2_decision_key")
        if self._last_key is not None and key < self._last_key:
            raise ValueError("execution_layer_v2_state_out_of_order")
        position = self._positions.get(signal.market_id)
        decision = decide_execution_layer_v2(
            signal=signal,
            position=position,
            config=self.config,
        )
        self._apply_decision(signal=signal, decision=decision)
        self._seen_keys.add(key)
        self._last_key = key
        return decision

    def _apply_decision(
        self,
        *,
        signal: ExecutionLayerV2Signal,
        decision: ExecutionLayerV2Decision,
    ) -> None:
        if decision.action == "ENTER_POSITION":
            self._positions[signal.market_id] = _position_from_decision(signal, decision)
            return
        if decision.action == "ROTATE_POSITION":
            self._positions[signal.market_id] = _position_from_decision(signal, decision)
            return
        if decision.action == "EXIT_POSITION":
            self._positions.pop(signal.market_id, None)


def decide_execution_layer_v2(
    *,
    signal: ExecutionLayerV2Signal,
    position: ExecutionLayerV2Position | None = None,
    config: ExecutionLayerV2Config | None = None,
) -> ExecutionLayerV2Decision:
    """Recalculate EV_t and emit one paper-only entry/hold/exit/rotate decision."""

    config = config or ExecutionLayerV2Config()
    up_entry_ev = _entry_ev(signal, "UP", config)
    down_entry_ev = _entry_ev(signal, "DOWN", config)
    best_side: ExecutionLayerV2Side = "UP" if up_entry_ev >= down_entry_ev else "DOWN"
    best_entry_ev = up_entry_ev if best_side == "UP" else down_entry_ev
    confidence = _probability(signal, best_side)

    if position is None:
        return _entry_or_no_action_decision(
            signal=signal,
            side=best_side,
            entry_ev=best_entry_ev,
            confidence=confidence,
            config=config,
        )
    return _active_position_decision(
        signal=signal,
        position=position,
        up_entry_ev=up_entry_ev,
        down_entry_ev=down_entry_ev,
        config=config,
    )


def build_execution_layer_v2_report(
    signals: tuple[ExecutionLayerV2Signal, ...] | list[ExecutionLayerV2Signal],
    *,
    config: ExecutionLayerV2Config | None = None,
) -> dict[str, Any]:
    """Build a deterministic v2 diagnostic report from decision-time signals."""

    config = config or ExecutionLayerV2Config()
    engine = ExecutionLayerV2Engine(config=config)
    decisions = engine.decide_many(tuple(signals))
    decision_rows = [decision.to_dict() for decision in decisions]
    action_counts = Counter(decision.action for decision in decisions)
    transition_counts = Counter(decision.state_transition for decision in decisions)
    reason_counts: Counter[str] = Counter()
    for decision in decisions:
        reason_counts.update(decision.reason_codes)
    v1_baseline = _v1_baseline_summary(tuple(signals), config=config)
    lambda_diagnostics = _lambda_grid_diagnostics(tuple(signals), config=config)
    report = {
        "schema_version": EXECUTION_LAYER_V2_REPORT_SCHEMA_VERSION,
        "execution_layer_v2_policy_name": EXECUTION_LAYER_V2_POLICY_NAME,
        "execution_layer_v2_status": "diagnostic_only_fail_closed",
        "decision_count": len(decisions),
        "entry_decision_count": action_counts.get("ENTER_POSITION", 0),
        "hold_decision_count": action_counts.get("HOLD_POSITION", 0),
        "exit_decision_count": action_counts.get("EXIT_POSITION", 0),
        "rotation_decision_count": action_counts.get("ROTATE_POSITION", 0),
        "action_counts": dict(sorted(action_counts.items())),
        "state_transition_counts": dict(sorted(transition_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "decision_rows": decision_rows,
        "open_position_count": len(engine.positions),
        "open_positions": [
            position.to_dict()
            for position in sorted(engine.positions.values(), key=lambda row: row.market_id)
        ],
        "ev_recalculation_loop_enabled": True,
        "dynamic_exit_engine_enabled": True,
        "state_machine_executor_enabled": True,
        "kelly_time_decay_sizing_enabled": True,
        "time_decay_function": "kelly_fraction * exp(-lambda * time_to_expiry_seconds)",
        "lambda_threshold_tuning_mode": "diagnostic_only_config_grid_no_outcomes",
        "uses_validation_labels_for_tuning": False,
        "uses_realized_pnl_or_settlement_outcomes": False,
        "forbidden_outcome_fields_used": [],
        "v1_baseline_comparison": v1_baseline,
        "lambda_threshold_diagnostics": lambda_diagnostics,
        "config": config.to_dict(),
        "source_scores_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(config),
    }
    report["execution_layer_v2_report_id"] = canonical_json_sha256(report)
    return report


def build_execution_layer_v2_report_from_rows(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    config: ExecutionLayerV2Config | None = None,
) -> dict[str, Any]:
    """Build a report from raw rows, fail-closed on forbidden outcome fields."""

    config = config or ExecutionLayerV2Config()
    rows = tuple(dict(row) for row in rows)
    forbidden = _forbidden_fields_by_row(rows)
    if forbidden:
        report = {
            "schema_version": EXECUTION_LAYER_V2_REPORT_SCHEMA_VERSION,
            "execution_layer_v2_policy_name": EXECUTION_LAYER_V2_POLICY_NAME,
            "execution_layer_v2_status": "blocked_fail_closed",
            "decision_count": 0,
            "decision_rows": [],
            "forbidden_outcome_fields_present": True,
            "forbidden_outcome_fields_by_row": forbidden,
            "forbidden_outcome_fields_used": sorted(
                {field for item in forbidden for field in item["forbidden_fields"]}
            ),
            "source_scores_mutated": False,
            "paper_live_unlock_changed": False,
            **_safety_report_fields(config),
        }
        report["execution_layer_v2_report_id"] = canonical_json_sha256(report)
        return report
    signals = tuple(_signal_from_row(row) for row in rows)
    return build_execution_layer_v2_report(signals, config=config)


def time_decay_multiplier(time_to_expiry_seconds: float | None, decay_lambda: float) -> float:
    """Issue #166 time-decay term: exp(-lambda * time_to_expiry)."""

    if time_to_expiry_seconds is None:
        return 1.0
    _require_finite("time_to_expiry_seconds", float(time_to_expiry_seconds))
    _require_finite("decay_lambda", float(decay_lambda))
    if time_to_expiry_seconds < 0.0 or decay_lambda < 0.0:
        raise ValueError("time_to_expiry_seconds and decay_lambda must be non-negative")
    return math.exp(-decay_lambda * time_to_expiry_seconds)


def binary_kelly_fraction(probability: float, price: float) -> float:
    """Kelly fraction for a binary $1 payout contract, clamped to [0, 1]."""

    _validate_probability("probability", probability)
    _validate_price("price", price)
    odds = (1.0 - price) / price
    if odds <= 0.0:
        return 0.0
    fraction = (odds * probability - (1.0 - probability)) / odds
    if not math.isfinite(fraction):
        return 0.0
    return min(1.0, max(0.0, fraction))


def time_decayed_kelly_notional(
    *,
    probability: float,
    price: float,
    time_to_expiry_seconds: float | None,
    config: ExecutionLayerV2Config | None = None,
) -> dict[str, float]:
    """Return deterministic paper notional from Kelly fraction and time decay."""

    config = config or ExecutionLayerV2Config()
    raw_kelly = binary_kelly_fraction(probability, price)
    decay = time_decay_multiplier(
        time_to_expiry_seconds,
        config.kelly_time_decay_lambda,
    )
    decayed_fraction = raw_kelly * decay
    clamped_fraction = min(
        config.max_nav_fraction_per_position,
        max(config.min_nav_fraction_per_position, decayed_fraction),
    )
    notional = config.nav_usdc * clamped_fraction
    return {
        "kelly_fraction": raw_kelly,
        "time_decay_multiplier": decay,
        "decayed_kelly_fraction": decayed_fraction,
        "clamped_nav_fraction": clamped_fraction,
        "paper_notional": notional,
    }


def _entry_or_no_action_decision(
    *,
    signal: ExecutionLayerV2Signal,
    side: ExecutionLayerV2Side,
    entry_ev: float,
    confidence: float,
    config: ExecutionLayerV2Config,
) -> ExecutionLayerV2Decision:
    price = _ask(signal, side)
    sizing = time_decayed_kelly_notional(
        probability=_probability(signal, side),
        price=price,
        time_to_expiry_seconds=signal.time_to_expiry_seconds,
        config=config,
    )
    if entry_ev < config.entry_ev_threshold:
        return _decision(
            signal=signal,
            action="NO_ACTION",
            target_side="NONE",
            state_before="NO_POSITION",
            state_after="NO_POSITION",
            selected_ev_t=entry_ev,
            entry_ev_reference=entry_ev,
            ev_ratio_to_entry=None,
            confidence=confidence,
            execution_price=0.0,
            paper_notional=0.0,
            shares=0.0,
            kelly_fraction=sizing["kelly_fraction"],
            time_decay_multiplier=sizing["time_decay_multiplier"],
            reason_codes=("entry_ev_threshold_not_met", "paper_only_guard"),
        )
    if confidence < config.min_confidence:
        return _decision(
            signal=signal,
            action="NO_ACTION",
            target_side="NONE",
            state_before="NO_POSITION",
            state_after="NO_POSITION",
            selected_ev_t=entry_ev,
            entry_ev_reference=entry_ev,
            ev_ratio_to_entry=None,
            confidence=confidence,
            execution_price=0.0,
            paper_notional=0.0,
            shares=0.0,
            kelly_fraction=sizing["kelly_fraction"],
            time_decay_multiplier=sizing["time_decay_multiplier"],
            reason_codes=("entry_confidence_threshold_not_met", "paper_only_guard"),
        )
    if sizing["paper_notional"] <= 0.0:
        return _decision(
            signal=signal,
            action="NO_ACTION",
            target_side="NONE",
            state_before="NO_POSITION",
            state_after="NO_POSITION",
            selected_ev_t=entry_ev,
            entry_ev_reference=entry_ev,
            ev_ratio_to_entry=None,
            confidence=confidence,
            execution_price=0.0,
            paper_notional=0.0,
            shares=0.0,
            kelly_fraction=sizing["kelly_fraction"],
            time_decay_multiplier=sizing["time_decay_multiplier"],
            reason_codes=("time_decayed_kelly_size_zero", "paper_only_guard"),
        )
    return _decision(
        signal=signal,
        action="ENTER_POSITION",
        target_side=side,
        state_before="NO_POSITION",
        state_after="ACTIVE",
        selected_ev_t=entry_ev,
        entry_ev_reference=entry_ev,
        ev_ratio_to_entry=1.0,
        confidence=confidence,
        execution_price=price,
        paper_notional=sizing["paper_notional"],
        shares=sizing["paper_notional"] / price,
        kelly_fraction=sizing["kelly_fraction"],
        time_decay_multiplier=sizing["time_decay_multiplier"],
        reason_codes=("positive_ev_entry", "time_decayed_kelly_sizing", "paper_only_guard"),
    )


def _active_position_decision(
    *,
    signal: ExecutionLayerV2Signal,
    position: ExecutionLayerV2Position,
    up_entry_ev: float,
    down_entry_ev: float,
    config: ExecutionLayerV2Config,
) -> ExecutionLayerV2Decision:
    held_side = position.side
    opposite_side: ExecutionLayerV2Side = "DOWN" if held_side == "UP" else "UP"
    held_ev_t = _probability(signal, held_side) - position.entry_price - _cost(config)
    ev_ratio = held_ev_t / position.entry_ev if position.entry_ev > 0.0 else 0.0
    opposite_ev = up_entry_ev if opposite_side == "UP" else down_entry_ev
    opposite_confidence = _probability(signal, opposite_side)
    time_to_close = signal.time_to_expiry_seconds
    if time_to_close is not None and time_to_close <= config.time_exit_threshold_seconds:
        return _exit_decision(
            signal=signal,
            position=position,
            held_ev_t=held_ev_t,
            ev_ratio=ev_ratio,
            reason_codes=("time_to_expiry_exit_threshold_crossed", "paper_only_guard"),
        )
    floor_ev = config.hold_ev_floor_ratio * position.entry_ev
    if (
        opposite_ev >= config.entry_ev_threshold
        and opposite_ev >= held_ev_t + config.opposite_signal_ev_margin
        and opposite_confidence >= config.min_confidence
    ):
        return _rotate_decision(
            signal=signal,
            position=position,
            target_side=opposite_side,
            target_entry_ev=opposite_ev,
            held_ev_t=held_ev_t,
            ev_ratio=ev_ratio,
            confidence=opposite_confidence,
            config=config,
        )
    if held_ev_t < floor_ev:
        return _exit_decision(
            signal=signal,
            position=position,
            held_ev_t=held_ev_t,
            ev_ratio=ev_ratio,
            reason_codes=("ev_t_decayed_below_hold_floor", "paper_only_guard"),
        )
    return _decision(
        signal=signal,
        action="HOLD_POSITION",
        target_side=held_side,
        state_before=position.state,
        state_after="ACTIVE",
        selected_ev_t=held_ev_t,
        entry_ev_reference=position.entry_ev,
        ev_ratio_to_entry=ev_ratio,
        confidence=_probability(signal, held_side),
        execution_price=0.0,
        paper_notional=0.0,
        shares=0.0,
        kelly_fraction=0.0,
        time_decay_multiplier=time_decay_multiplier(
            signal.time_to_expiry_seconds,
            config.kelly_time_decay_lambda,
        ),
        reason_codes=("ev_t_above_hold_floor", "hold_position", "paper_only_guard"),
    )


def _exit_decision(
    *,
    signal: ExecutionLayerV2Signal,
    position: ExecutionLayerV2Position,
    held_ev_t: float,
    ev_ratio: float,
    reason_codes: tuple[str, ...],
) -> ExecutionLayerV2Decision:
    price = _bid(signal, position.side)
    return _decision(
        signal=signal,
        action="EXIT_POSITION",
        target_side=position.side,
        state_before=position.state,
        state_after="EXIT",
        selected_ev_t=held_ev_t,
        entry_ev_reference=position.entry_ev,
        ev_ratio_to_entry=ev_ratio,
        confidence=_probability(signal, position.side),
        execution_price=price,
        paper_notional=position.shares * price,
        shares=position.shares,
        kelly_fraction=0.0,
        time_decay_multiplier=1.0,
        reason_codes=reason_codes,
    )


def _rotate_decision(
    *,
    signal: ExecutionLayerV2Signal,
    position: ExecutionLayerV2Position,
    target_side: ExecutionLayerV2Side,
    target_entry_ev: float,
    held_ev_t: float,
    ev_ratio: float,
    confidence: float,
    config: ExecutionLayerV2Config,
) -> ExecutionLayerV2Decision:
    price = _ask(signal, target_side)
    sizing = time_decayed_kelly_notional(
        probability=_probability(signal, target_side),
        price=price,
        time_to_expiry_seconds=signal.time_to_expiry_seconds,
        config=config,
    )
    return _decision(
        signal=signal,
        action="ROTATE_POSITION",
        target_side=target_side,
        state_before=position.state,
        state_after="ACTIVE",
        selected_ev_t=target_entry_ev,
        entry_ev_reference=position.entry_ev,
        ev_ratio_to_entry=ev_ratio,
        confidence=confidence,
        execution_price=price,
        paper_notional=sizing["paper_notional"],
        shares=sizing["paper_notional"] / price if price > 0.0 else 0.0,
        kelly_fraction=sizing["kelly_fraction"],
        time_decay_multiplier=sizing["time_decay_multiplier"],
        reason_codes=(
            "opposite_signal_ev_margin_crossed",
            "rotate_position",
            "paper_only_guard",
        ),
    )


def _decision(
    *,
    signal: ExecutionLayerV2Signal,
    action: ExecutionLayerV2Action,
    target_side: ExecutionLayerV2Side,
    state_before: ExecutionLayerV2State,
    state_after: ExecutionLayerV2State,
    selected_ev_t: float,
    entry_ev_reference: float,
    ev_ratio_to_entry: float | None,
    confidence: float,
    execution_price: float,
    paper_notional: float,
    shares: float,
    kelly_fraction: float,
    time_decay_multiplier: float,
    reason_codes: tuple[str, ...],
) -> ExecutionLayerV2Decision:
    return ExecutionLayerV2Decision(
        market_id=signal.market_id,
        decision_ts=signal.decision_ts,
        action=action,
        target_side=target_side,
        state_before=state_before,
        state_after=state_after,
        selected_ev_t=selected_ev_t,
        entry_ev_reference=entry_ev_reference,
        ev_ratio_to_entry=ev_ratio_to_entry,
        confidence=confidence,
        execution_price=execution_price,
        paper_notional=paper_notional,
        shares=shares,
        kelly_fraction=kelly_fraction,
        time_decay_multiplier=time_decay_multiplier,
        reason_codes=reason_codes,
        source_signal_id=signal.source_signal_id,
    )


def _position_from_decision(
    signal: ExecutionLayerV2Signal,
    decision: ExecutionLayerV2Decision,
) -> ExecutionLayerV2Position:
    return ExecutionLayerV2Position(
        market_id=decision.market_id,
        side=decision.target_side,
        entry_ts=decision.decision_ts,
        entry_price=decision.execution_price,
        entry_probability=_probability(signal, decision.target_side),
        entry_ev=decision.selected_ev_t,
        size_usdc=decision.paper_notional,
        shares=decision.shares,
        state="ACTIVE",
    )


def _signal_from_row(row: dict[str, Any]) -> ExecutionLayerV2Signal:
    return ExecutionLayerV2Signal(
        market_id=str(row["market_id"]),
        decision_ts=int(row["decision_ts"]),
        p_up=float(row["p_up"]),
        p_down=float(row["p_down"]) if row.get("p_down") is not None else None,
        ask_up=float(row["ask_up"]),
        ask_down=float(row["ask_down"]),
        bid_up=float(row["bid_up"]) if row.get("bid_up") is not None else None,
        bid_down=float(row["bid_down"]) if row.get("bid_down") is not None else None,
        time_to_expiry_seconds=(
            float(row["time_to_expiry_seconds"])
            if row.get("time_to_expiry_seconds") is not None
            else None
        ),
        source_signal_id=row.get("source_signal_id"),
        model_score=float(row["model_score"]) if row.get("model_score") is not None else None,
    )


def _forbidden_fields_by_row(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    failures = []
    forbidden_set = set(EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS)
    for index, row in enumerate(rows):
        present = sorted(forbidden_set.intersection(row))
        if present:
            failures.append(
                {
                    "row_index": index,
                    "market_id": row.get("market_id"),
                    "decision_ts": row.get("decision_ts"),
                    "forbidden_fields": present,
                }
            )
    return failures


def _v1_baseline_summary(
    signals: tuple[ExecutionLayerV2Signal, ...],
    *,
    config: ExecutionLayerV2Config,
) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    for signal in signals:
        up_ev = _entry_ev(signal, "UP", config)
        down_ev = _entry_ev(signal, "DOWN", config)
        best_ev = max(up_ev, down_ev)
        best_side = "UP" if up_ev >= down_ev else "DOWN"
        if best_ev >= config.entry_ev_threshold:
            action_counts[f"BUY_{best_side}_HOLD_TO_SETTLEMENT"] += 1
        else:
            action_counts["NO_TRADE"] += 1
    return {
        "baseline_name": EXECUTION_LAYER_V2_BASELINE_NAME,
        "baseline_assumption": "enter positive EV then hold to settlement",
        "baseline_action_counts": dict(sorted(action_counts.items())),
        "v2_differs_from_v1": True,
        "uses_realized_pnl_or_settlement_outcomes": False,
    }


def _lambda_grid_diagnostics(
    signals: tuple[ExecutionLayerV2Signal, ...],
    *,
    config: ExecutionLayerV2Config,
) -> list[dict[str, Any]]:
    rows = []
    for value in config.diagnostic_lambda_grid:
        notionals = []
        for signal in signals:
            up_ev = _entry_ev(signal, "UP", config)
            down_ev = _entry_ev(signal, "DOWN", config)
            side: ExecutionLayerV2Side = "UP" if up_ev >= down_ev else "DOWN"
            sizing_config = ExecutionLayerV2Config(
                **{
                    **config.to_dict(),
                    "kelly_time_decay_lambda": value,
                }
            )
            sizing = time_decayed_kelly_notional(
                probability=_probability(signal, side),
                price=_ask(signal, side),
                time_to_expiry_seconds=signal.time_to_expiry_seconds,
                config=sizing_config,
            )
            notionals.append(sizing["paper_notional"])
        rows.append(
            {
                "kelly_time_decay_lambda": value,
                "mean_candidate_notional": sum(notionals) / len(notionals)
                if notionals
                else 0.0,
                "max_candidate_notional": max(notionals, default=0.0),
                "selection_metric": "size_sensitivity_no_outcomes",
                "uses_validation_labels_for_tuning": False,
            }
        )
    return rows


def _entry_ev(
    signal: ExecutionLayerV2Signal,
    side: ExecutionLayerV2Side,
    config: ExecutionLayerV2Config,
) -> float:
    return _probability(signal, side) - _ask(signal, side) - _cost(config)


def _probability(signal: ExecutionLayerV2Signal, side: ExecutionLayerV2Side) -> float:
    if side == "UP":
        return signal.p_up
    if side == "DOWN":
        return float(signal.p_down)
    return 0.0


def _ask(signal: ExecutionLayerV2Signal, side: ExecutionLayerV2Side) -> float:
    if side == "UP":
        return signal.ask_up
    if side == "DOWN":
        return signal.ask_down
    return 0.0


def _bid(signal: ExecutionLayerV2Signal, side: ExecutionLayerV2Side) -> float:
    if side == "UP":
        return signal.bid_up if signal.bid_up is not None else signal.ask_up
    if side == "DOWN":
        return signal.bid_down if signal.bid_down is not None else signal.ask_down
    return 0.0


def _cost(config: ExecutionLayerV2Config) -> float:
    return config.execution_cost_bps / 10_000.0


def _safety_report_fields(config: ExecutionLayerV2Config) -> dict[str, Any]:
    fields = {
        **compact_safety_fields(),
        "polymarket_write_enabled": config.polymarket_write_enabled,
        "wallet_signing_enabled": config.wallet_signing_enabled,
        "v8_execution_handoff_allowed": config.v8_execution_handoff_allowed,
        "source_model_candidate_eligible": config.source_model_candidate_eligible,
        "freeze_ready": config.freeze_ready,
        "promotion_evidence_eligible": config.promotion_evidence_eligible,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    return fields


def _validate_safety_flags(obj: Any) -> None:
    expected = {
        **compact_safety_fields(),
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    for field_name, expected_value in expected.items():
        if getattr(obj, field_name) is not expected_value:
            raise ValueError(f"{field_name} must be {expected_value}")
    for field_name in (
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
    ):
        if hasattr(obj, field_name) and getattr(obj, field_name) is not False:
            raise ValueError(f"{field_name} must be false")


def _validate_probability(field_name: str, value: float) -> None:
    _require_finite(field_name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")


def _validate_price(field_name: str, value: float, *, allow_zero: bool = False) -> None:
    _require_finite(field_name, value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value >= 1.0:
        lower = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{field_name} must be in {lower}")


def _require_finite(field_name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
