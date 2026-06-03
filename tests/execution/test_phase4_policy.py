"""Phase 4 entry and force-exit policy tests for issue #85."""

from __future__ import annotations

import pytest

from bigan.execution.phase4_policy import (
    Phase4EntryPolicy,
    VolatilitySleeveBudget,
    entry_price_skip_reason,
    evaluate_entry_gates,
    phase4_lifecycle_complete,
    phase4_summary_status,
    settlement_cost_edge_skip_reason,
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


def test_settlement_cost_edge_skip_reason_ignores_low_price_and_near_min() -> None:
    policy = Phase4EntryPolicy(
        min_entry_price=0.35,
        near_min_price_band=0.05,
        near_min_fresh_edge_threshold=0.50,
        settlement_edge_threshold=0.082,
    )

    assert (
        settlement_cost_edge_skip_reason(
            fresh_edge_at_worst=0.10,
            policy=policy,
        )
        is None
    )
    assert (
        settlement_cost_edge_skip_reason(
            fresh_edge_at_worst=0.07,
            policy=policy,
        )
        == "fresh_edge_below_threshold"
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


def test_settlement_edge_threshold_overrides_legacy_edge_alias() -> None:
    policy = Phase4EntryPolicy(edge_threshold=0.45, settlement_edge_threshold=0.14)

    assert (
        entry_price_skip_reason(
            ask=0.50,
            worst_price=0.52,
            fresh_edge_at_worst=0.15,
            seconds_to_expiry=600.0,
            policy=policy,
        )
        is None
    )


def test_volatility_gate_is_diagnostic_only_by_default() -> None:
    policy = Phase4EntryPolicy(
        settlement_edge_threshold=0.45,
        volatility_score_threshold=0.50,
        volatility_min_entry_price=0.20,
        volatility_min_seconds_to_expiry=420.0,
    )

    evaluation = evaluate_entry_gates(
        settlement_edge=0.10,
        ask=0.30,
        worst_price=0.32,
        bid=0.40,
        token_probability=0.90,
        seconds_to_expiry=600.0,
        policy=policy,
    )

    assert evaluation.settlement_gate_passed is False
    assert evaluation.volatility_gate_passed is True
    assert evaluation.expected_volatility_exit_gain == pytest.approx(0.08)
    assert evaluation.gate_mode == "volatility_diagnostic_only"


def test_volatility_live_gate_requires_explicit_enable() -> None:
    policy = Phase4EntryPolicy(
        settlement_edge_threshold=0.45,
        volatility_score_threshold=0.50,
        volatility_min_entry_price=0.20,
        enable_volatility_live_entries=True,
    )

    evaluation = evaluate_entry_gates(
        settlement_edge=0.10,
        ask=0.30,
        worst_price=0.32,
        bid=0.40,
        token_probability=0.90,
        seconds_to_expiry=600.0,
        policy=policy,
    )

    assert evaluation.gate_mode == "volatility_live_entry"


def test_volatility_budget_resets_per_round_and_refills_to_cap() -> None:
    budget = VolatilitySleeveBudget(
        round_cap_usdc=1.0,
        per_bet_cap_usdc=1.0,
        min_order_size_usdc=0.05,
    )

    first = budget.next_entry_decision("round-a")
    assert first.allowed is True
    assert first.size_usdc == pytest.approx(1.0)
    assert first.balance_usdc == pytest.approx(1.0)

    budget.apply_account_pnl("round-a", -0.20)
    reduced = budget.next_entry_decision("round-a")
    assert reduced.allowed is True
    assert reduced.size_usdc == pytest.approx(0.80)
    assert reduced.balance_usdc == pytest.approx(0.80)

    budget.apply_account_pnl("round-a", 0.30)
    refilled = budget.next_entry_decision("round-a")
    assert refilled.allowed is True
    assert refilled.size_usdc == pytest.approx(1.0)
    assert refilled.balance_usdc == pytest.approx(1.0)

    budget.apply_account_pnl("round-a", -0.98)
    below_floor = budget.next_entry_decision("round-a")
    assert below_floor.allowed is False
    assert below_floor.reason == "volatility_round_balance_below_min_size"

    new_round = budget.next_entry_decision("round-b")
    assert new_round.allowed is True
    assert new_round.size_usdc == pytest.approx(1.0)
