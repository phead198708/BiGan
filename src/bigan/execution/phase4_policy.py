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


@dataclass(frozen=True, slots=True)
class Phase4EntryPolicy:
    """Cheap-entry and near-threshold gating for Phase 4 live execution."""

    min_entry_price: float = DEFAULT_MIN_ENTRY_PRICE
    near_min_price_band: float = DEFAULT_NEAR_MIN_PRICE_BAND
    near_min_fresh_edge_threshold: float = DEFAULT_NEAR_MIN_FRESH_EDGE_THRESHOLD
    near_min_seconds_to_expiry: float = DEFAULT_NEAR_MIN_SECONDS_TO_EXPIRY
    edge_threshold: float = 0.45


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
    if fresh_edge_at_worst < policy.edge_threshold:
        return "fresh_edge_below_threshold"
    return None


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
