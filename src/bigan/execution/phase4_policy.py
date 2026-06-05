"""Phase 4 BTC-15M execution policy helpers for issue #85."""

from __future__ import annotations

from dataclasses import dataclass

# Raised from the 0.30 runtime default after the 0.31 -> 0.11 soft-force-exit loss
# on btc-updown-15m-1779913800 in run 20260527T143533Z.
DEFAULT_MIN_ENTRY_PRICE = 0.35
DEFAULT_NEAR_MIN_PRICE_BAND = 0.05
DEFAULT_NEAR_MIN_FRESH_EDGE_THRESHOLD = 0.50
DEFAULT_NEAR_MIN_SECONDS_TO_EXPIRY = 420.0
DEFAULT_SOFT_FORCE_EXIT_MIN_BID = 0.15
DEFAULT_SETTLEMENT_EDGE_THRESHOLD = 0.45
DEFAULT_VOLATILITY_SCORE_THRESHOLD = 0.50
DEFAULT_VOLATILITY_MIN_ENTRY_PRICE = 0.20
DEFAULT_VOLATILITY_MIN_SECONDS_TO_EXPIRY = 420.0
DEFAULT_VOLATILITY_ROUND_TRIP_COST = 0.04
DEFAULT_VOLATILITY_SAFETY_MARGIN = 0.02
DEFAULT_VOLATILITY_ROUND_BANKROLL_USDC = 1.0
DEFAULT_VOLATILITY_MIN_ORDER_SIZE_USDC = 0.05
DEFAULT_SETTLEMENT_MIN_CONFIDENCE = 0.80
DEFAULT_MAX_SIGNAL_AGE_SECONDS = 180.0
DEFAULT_SETTLEMENT_PEAK_CONFIDENCE_DROP_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class Phase4EntryPolicy:
    """Cheap-entry and near-threshold gating for Phase 4 live execution.

    ``edge_threshold`` is kept as a legacy alias for the settlement gate. New
    Phase 4 v5 code should prefer ``settlement_edge_threshold`` so settlement
    confidence and volatility diagnostics do not share one overloaded field.
    """

    min_entry_price: float = DEFAULT_MIN_ENTRY_PRICE
    near_min_price_band: float = DEFAULT_NEAR_MIN_PRICE_BAND
    near_min_fresh_edge_threshold: float = DEFAULT_NEAR_MIN_FRESH_EDGE_THRESHOLD
    near_min_seconds_to_expiry: float = DEFAULT_NEAR_MIN_SECONDS_TO_EXPIRY
    edge_threshold: float = DEFAULT_SETTLEMENT_EDGE_THRESHOLD
    settlement_edge_threshold: float | None = None
    volatility_score_threshold: float = DEFAULT_VOLATILITY_SCORE_THRESHOLD
    volatility_min_entry_price: float = DEFAULT_VOLATILITY_MIN_ENTRY_PRICE
    volatility_min_seconds_to_expiry: float = DEFAULT_VOLATILITY_MIN_SECONDS_TO_EXPIRY
    volatility_round_trip_cost: float = DEFAULT_VOLATILITY_ROUND_TRIP_COST
    volatility_safety_margin: float = DEFAULT_VOLATILITY_SAFETY_MARGIN
    enable_volatility_live_entries: bool = False
    settlement_min_confidence: float = DEFAULT_SETTLEMENT_MIN_CONFIDENCE
    max_signal_age_seconds: float | None = DEFAULT_MAX_SIGNAL_AGE_SECONDS
    settlement_peak_confidence_drop_tolerance: float | None = None

    @property
    def effective_settlement_edge_threshold(self) -> float:
        """Return the active settlement edge threshold."""

        if self.settlement_edge_threshold is None:
            return self.edge_threshold
        return self.settlement_edge_threshold


@dataclass(frozen=True, slots=True)
class Phase4GateEvaluation:
    """Decision record for the split settlement/volatility Phase 4 gates."""

    settlement_gate_passed: bool
    settlement_edge: float
    settlement_edge_threshold: float
    settlement_confidence: float | None
    settlement_min_confidence: float
    settlement_confidence_passed: bool
    signal_age_seconds: float | None
    max_signal_age_seconds: float | None
    signal_freshness_passed: bool
    volatility_gate_passed: bool
    volatility_score: float | None
    volatility_score_threshold: float
    volatility_min_entry_price: float
    volatility_min_seconds_to_expiry: float
    volatility_round_trip_cost: float
    volatility_safety_margin: float
    expected_volatility_exit_gain: float | None
    volatility_live_entry_enabled: bool
    gate_mode: str


@dataclass(frozen=True, slots=True)
class VolatilityBudgetDecision:
    """Sizing decision for one volatility-sleeve paper/live entry."""

    round_slug: str
    balance_usdc: float
    size_usdc: float
    min_order_size_usdc: float
    allowed: bool
    reason: str


@dataclass(slots=True)
class VolatilitySleeveBudget:
    """Per-round running bankroll for the volatility sleeve.

    The balance resets at the first signal for each round, is reduced by account
    cash-flow losses, and can refill with account cash-flow profit up to the
    configured round cap. Use account-cash PnL here; theoretical executor PnL
    hides slippage, fees, and dust.
    """

    round_cap_usdc: float = DEFAULT_VOLATILITY_ROUND_BANKROLL_USDC
    per_bet_cap_usdc: float = DEFAULT_VOLATILITY_ROUND_BANKROLL_USDC
    min_order_size_usdc: float = DEFAULT_VOLATILITY_MIN_ORDER_SIZE_USDC
    balances: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.round_cap_usdc <= 0:
            raise ValueError("round_cap_usdc must be positive")
        if self.per_bet_cap_usdc <= 0:
            raise ValueError("per_bet_cap_usdc must be positive")
        if self.min_order_size_usdc < 0:
            raise ValueError("min_order_size_usdc must be non-negative")
        if self.balances is None:
            self.balances = {}

    def balance_for_round(self, round_slug: str) -> float:
        _require_round_slug(round_slug)
        assert self.balances is not None
        return self.balances.setdefault(round_slug, self.round_cap_usdc)

    def next_entry_decision(self, round_slug: str) -> VolatilityBudgetDecision:
        balance = self.balance_for_round(round_slug)
        size = min(self.per_bet_cap_usdc, max(0.0, balance))
        allowed = size >= self.min_order_size_usdc and size > 0
        return VolatilityBudgetDecision(
            round_slug=round_slug,
            balance_usdc=balance,
            size_usdc=size if allowed else 0.0,
            min_order_size_usdc=self.min_order_size_usdc,
            allowed=allowed,
            reason="ok" if allowed else "volatility_round_balance_below_min_size",
        )

    def apply_account_pnl(self, round_slug: str, realized_account_pnl: float) -> float:
        balance = self.balance_for_round(round_slug)
        updated = min(self.round_cap_usdc, max(0.0, balance + realized_account_pnl))
        assert self.balances is not None
        self.balances[round_slug] = updated
        return updated


def is_near_min_entry(*, ask: float, worst_price: float, policy: Phase4EntryPolicy) -> bool:
    """Return whether the quote sits in the band just above ``min_entry_price``."""

    ceiling = policy.min_entry_price + policy.near_min_price_band
    return ask < ceiling or worst_price < ceiling


def entry_price_skip_reason(
    *,
    ask: float,
    worst_price: float,
    fresh_edge_at_worst: float,
    seconds_to_expiry: float | None,
    policy: Phase4EntryPolicy,
) -> str | None:
    """Return a skip reason when an entry should not be attempted."""

    if ask < policy.min_entry_price or worst_price < policy.min_entry_price:
        return "entry_price_below_min"
    if is_near_min_entry(ask=ask, worst_price=worst_price, policy=policy):
        if fresh_edge_at_worst < policy.near_min_fresh_edge_threshold:
            return "near_min_entry_fresh_edge_below_threshold"
        if (
            seconds_to_expiry is not None
            and seconds_to_expiry < policy.near_min_seconds_to_expiry
        ):
            return "near_min_entry_too_close_to_expiry"
    if fresh_edge_at_worst < policy.effective_settlement_edge_threshold:
        return "fresh_edge_below_threshold"
    return None


def settlement_cost_edge_skip_reason(
    *,
    fresh_edge_at_worst: float,
    policy: Phase4EntryPolicy,
    settlement_confidence: float | None = None,
    settlement_peak_confidence: float | None = None,
    signal_age_seconds: float | None = None,
) -> str | None:
    """Return a v6 settlement skip reason using executable, fresh, confident edge."""

    if (
        policy.max_signal_age_seconds is not None
        and signal_age_seconds is not None
        and signal_age_seconds > policy.max_signal_age_seconds
    ):
        return "signal_age_above_threshold"
    if (
        settlement_confidence is not None
        and settlement_confidence < policy.settlement_min_confidence
    ):
        return "settlement_confidence_below_threshold"
    if (
        settlement_confidence is not None
        and settlement_peak_confidence is not None
        and policy.settlement_peak_confidence_drop_tolerance is not None
        and settlement_peak_confidence - settlement_confidence
        > policy.settlement_peak_confidence_drop_tolerance
    ):
        return "settlement_confidence_peak_drop"
    if fresh_edge_at_worst < policy.effective_settlement_edge_threshold:
        return "fresh_edge_below_threshold"
    return None


def settlement_gate_passed(
    *,
    edge: float,
    policy: Phase4EntryPolicy,
    settlement_confidence: float | None = None,
    settlement_peak_confidence: float | None = None,
    signal_age_seconds: float | None = None,
) -> bool:
    """Return whether the settlement-confidence gate admits the signal."""

    return (
        settlement_cost_edge_skip_reason(
            fresh_edge_at_worst=edge,
            policy=policy,
            settlement_confidence=settlement_confidence,
            settlement_peak_confidence=settlement_peak_confidence,
            signal_age_seconds=signal_age_seconds,
        )
        is None
    )


def expected_volatility_exit_gain_from_orderbook(
    *,
    bid: float | None,
    worst_price: float,
) -> float:
    """Return an orderbook-only expected exit gain for the volatility sleeve.

    This intentionally avoids model edge. With only top-of-book data available,
    the conservative paper gate requires current executable exit value to clear
    the entry worst price plus cost/margin. Richer microstructure features can
    replace this input later without changing the budget mechanics.
    """

    if bid is None:
        return 0.0
    return float(bid) - worst_price


def volatility_score_from_quote(*, token_probability: float, worst_price: float) -> float:
    """Legacy issue #89 diagnostic alias; do not use for #90 live volatility sizing."""

    return token_probability - worst_price


def volatility_gate_passed(
    *,
    ask: float,
    worst_price: float,
    expected_volatility_exit_gain: float,
    seconds_to_expiry: float | None,
    policy: Phase4EntryPolicy,
) -> bool:
    """Return whether the diagnostic volatility gate admits the quote."""

    if ask < policy.volatility_min_entry_price or worst_price < policy.volatility_min_entry_price:
        return False
    if (
        seconds_to_expiry is not None
        and seconds_to_expiry < policy.volatility_min_seconds_to_expiry
    ):
        return False
    return expected_volatility_exit_gain >= (
        policy.volatility_round_trip_cost + policy.volatility_safety_margin
    )


def evaluate_entry_gates(
    *,
    settlement_edge: float,
    ask: float | None,
    worst_price: float | None,
    token_probability: float,
    seconds_to_expiry: float | None,
    policy: Phase4EntryPolicy,
    bid: float | None = None,
    settlement_confidence: float | None = None,
    settlement_peak_confidence: float | None = None,
    signal_age_seconds: float | None = None,
    enable_settlement_gate: bool = True,
) -> Phase4GateEvaluation:
    """Evaluate settlement and volatility gates without placing an order."""

    settlement_confidence_passed = (
        (not enable_settlement_gate)
        or settlement_confidence is None
        or settlement_confidence >= policy.settlement_min_confidence
    )
    signal_freshness_passed = (
        policy.max_signal_age_seconds is None
        or signal_age_seconds is None
        or signal_age_seconds <= policy.max_signal_age_seconds
    )
    settlement_passed = (
        settlement_gate_passed(
            edge=settlement_edge,
            policy=policy,
            settlement_confidence=settlement_confidence,
            settlement_peak_confidence=settlement_peak_confidence,
            signal_age_seconds=signal_age_seconds,
        )
        if enable_settlement_gate
        else False
    )
    expected_volatility_exit_gain = (
        expected_volatility_exit_gain_from_orderbook(bid=bid, worst_price=worst_price)
        if ask is not None and worst_price is not None
        else None
    )
    volatility_passed = (
        volatility_gate_passed(
            ask=ask,
            worst_price=worst_price,
            expected_volatility_exit_gain=expected_volatility_exit_gain,
            seconds_to_expiry=seconds_to_expiry,
            policy=policy,
        )
        if ask is not None and worst_price is not None and expected_volatility_exit_gain is not None
        else False
    )
    if settlement_passed:
        gate_mode = "settlement_live_entry"
    elif volatility_passed and policy.enable_volatility_live_entries:
        gate_mode = "volatility_live_entry"
    elif volatility_passed:
        gate_mode = "volatility_diagnostic_only"
    else:
        gate_mode = "blocked"
    return Phase4GateEvaluation(
        settlement_gate_passed=settlement_passed,
        settlement_edge=settlement_edge,
        settlement_edge_threshold=policy.effective_settlement_edge_threshold,
        settlement_confidence=settlement_confidence,
        settlement_min_confidence=policy.settlement_min_confidence,
        settlement_confidence_passed=settlement_confidence_passed,
        signal_age_seconds=signal_age_seconds,
        max_signal_age_seconds=policy.max_signal_age_seconds,
        signal_freshness_passed=signal_freshness_passed,
        volatility_gate_passed=volatility_passed,
        volatility_score=expected_volatility_exit_gain,
        volatility_score_threshold=policy.volatility_score_threshold,
        volatility_min_entry_price=policy.volatility_min_entry_price,
        volatility_min_seconds_to_expiry=policy.volatility_min_seconds_to_expiry,
        volatility_round_trip_cost=policy.volatility_round_trip_cost,
        volatility_safety_margin=policy.volatility_safety_margin,
        expected_volatility_exit_gain=expected_volatility_exit_gain,
        volatility_live_entry_enabled=policy.enable_volatility_live_entries,
        gate_mode=gate_mode,
    )


def _require_round_slug(round_slug: str) -> None:
    if not str(round_slug).strip():
        raise ValueError("round_slug is required")


def phase4_lifecycle_complete(
    *,
    errors: int,
    entries_filled: int,
    open_positions_at_shutdown: int,
    exits_pending_confirmation: int,
    exits_pending_settlement: int,
) -> bool:
    """Return whether a bounded Phase 4 run finished with a clean lifecycle."""

    return (
        errors == 0
        and entries_filled > 0
        and open_positions_at_shutdown == 0
        and exits_pending_confirmation == 0
        and exits_pending_settlement == 0
    )


def phase4_summary_status(
    *,
    errors: int,
    entries_filled: int,
    lifecycle_complete: bool,
) -> str:
    """Lifecycle-only status that must not be confused with promotion readiness."""

    if errors > 0:
        return "FAIL"
    if entries_filled <= 0:
        return "CHECK"
    if lifecycle_complete:
        return "LIFECYCLE_PASS"
    return "LIFECYCLE_INCOMPLETE"


def soft_force_exit_deferred(
    *,
    exit_reason: str,
    bid: float | None,
    soft_force_exit_min_bid: float = DEFAULT_SOFT_FORCE_EXIT_MIN_BID,
) -> bool:
    """Defer a soft force exit when the bid is too weak to justify a fire sale."""

    if exit_reason != "soft_force_exit":
        return False
    if bid is None:
        return True
    return float(bid) < soft_force_exit_min_bid
