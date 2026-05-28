"""Phase 4 entry and force-exit policy tests for issue #85."""

from __future__ import annotations

import pytest

from bigan.execution.phase4_policy import (
    Phase4EntryPolicy,
    entry_price_skip_reason,
    phase4_lifecycle_complete,
    phase4_summary_status,
    soft_force_exit_deferred,
)


def test_entry_price_skip_reason_blocks_below_min() -> None:
    policy = Phase4EntryPolicy(min_entry_price=0.35, edge_threshold=0.45)

    assert (
        entry_price_skip_reason(
            ask=0.31,
            worst_price=0.33,
            fresh_edge_at_worst=0.60,
            seconds_to_expiry=600.0,
            policy=policy,
        )
        == "entry_price_below_min"
    )


def test_near_min_entry_requires_stronger_edge_and_time() -> None:
    policy = Phase4EntryPolicy(
        min_entry_price=0.35,
        near_min_price_band=0.05,
        near_min_fresh_edge_threshold=0.50,
        near_min_seconds_to_expiry=420.0,
        edge_threshold=0.45,
    )

    assert (
        entry_price_skip_reason(
            ask=0.36,
            worst_price=0.38,
            fresh_edge_at_worst=0.48,
            seconds_to_expiry=600.0,
            policy=policy,
        )
        == "near_min_entry_fresh_edge_below_threshold"
    )
    assert (
        entry_price_skip_reason(
            ask=0.36,
            worst_price=0.38,
            fresh_edge_at_worst=0.55,
            seconds_to_expiry=300.0,
            policy=policy,
        )
        == "near_min_entry_too_close_to_expiry"
    )
    assert (
        entry_price_skip_reason(
            ask=0.36,
            worst_price=0.38,
            fresh_edge_at_worst=0.55,
            seconds_to_expiry=500.0,
            policy=policy,
        )
        is None
    )


def test_soft_force_exit_deferred_only_for_weak_bids() -> None:
    assert soft_force_exit_deferred(exit_reason="soft_force_exit", bid=0.10) is True
    assert soft_force_exit_deferred(exit_reason="soft_force_exit", bid=0.20) is False
    assert soft_force_exit_deferred(exit_reason="hard_force_exit", bid=0.10) is False


def test_phase4_summary_status_distinguishes_lifecycle_from_promotion() -> None:
    assert (
        phase4_summary_status(
            errors=0,
            entries_filled=3,
            lifecycle_complete=True,
        )
        == "LIFECYCLE_PASS"
    )
    assert (
        phase4_summary_status(
            errors=0,
            entries_filled=3,
            lifecycle_complete=False,
        )
        == "LIFECYCLE_INCOMPLETE"
    )
    assert phase4_summary_status(errors=1, entries_filled=3, lifecycle_complete=True) == "FAIL"
    assert phase4_lifecycle_complete(
        errors=0,
        entries_filled=1,
        open_positions_at_shutdown=0,
        exits_pending_confirmation=0,
        exits_pending_settlement=1,
    ) is False


@pytest.mark.parametrize("ask", [0.40, 0.50])
def test_non_near_min_entry_uses_standard_edge_threshold(ask: float) -> None:
    policy = Phase4EntryPolicy(edge_threshold=0.45)

    assert (
        entry_price_skip_reason(
            ask=ask,
            worst_price=ask + 0.02,
            fresh_edge_at_worst=0.44,
            seconds_to_expiry=600.0,
            policy=policy,
        )
        == "fresh_edge_below_threshold"
    )
