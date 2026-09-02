from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_support_preserving_overlay_v8_2 as v82,
)


def _profile() -> dict:
    path = (
        Path(__file__).parents[2]
        / "examples/v8/polymarket_configs"
        / "execution_layer_v2_support_preserving_overlay_v8_2_profile.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(
    *,
    market_id: str = "market-1",
    action: str = "BUY_UP_SELL_BEFORE_CLOSE",
    allowed: bool = True,
    blockers: list[str] | None = None,
    rank_passed: bool | None = True,
    point_action: str = "BUY_UP_SELL_BEFORE_CLOSE",
) -> dict:
    side = "UP" if "UP" in action else "DOWN"
    if action == "NO_TRADE":
        side = "NONE"
    return {
        "market_id": market_id,
        "decision_ts": 1000,
        "selected_action": action,
        "selected_side": side,
        "execution_guard_order_allowed": allowed,
        "execution_blocking_reason_codes": blockers or [],
        "rank_abstention_passed": rank_passed,
        "point_selected_action": point_action,
    }


def test_v8_2_profile_is_preregistered_and_fail_closed() -> None:
    profile = _profile()
    v82.validate_support_preserving_overlay_v8_2_profile(profile)
    assert profile["historical_gate"]["side_quota_enabled"] is False
    assert profile["lineage"]["issue246_outcomes_allowed_for_v8_2"] is False
    assert profile["safety"]["paper_candidate_allowed"] is False
    assert profile["safety"]["v8_execution_handoff_allowed"] is False


def test_v8_2_keeps_guard_passed_v8_1_primary() -> None:
    row = v82.select_support_preserving_overlay_decision(
        candidate=_decision(),
        baseline=_decision(action="BUY_DOWN_SELL_BEFORE_CLOSE"),
    )
    assert row["selected_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert row["selection_source"] == "v8_1_primary"
    assert row["fallback_applied"] is False


def test_v8_2_falls_back_only_for_rank_threshold_abstention() -> None:
    row = v82.select_support_preserving_overlay_decision(
        candidate=_decision(
            action="NO_TRADE",
            allowed=False,
            blockers=["v8_1_veto_to_no_trade"],
            rank_passed=False,
            point_action="BUY_UP_SELL_BEFORE_CLOSE",
        ),
        baseline=_decision(action="BUY_DOWN_SELL_BEFORE_CLOSE"),
    )
    assert row["selected_action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
    assert row["selection_source"] == "v6_7_rank_abstention_fallback"
    assert row["fallback_applied"] is True
    assert row["risk_blocker_bypass_used"] is False


@pytest.mark.parametrize(
    "candidate_blocker",
    [
        "execution_spread_too_wide",
        "execution_book_stale",
        "execution_time_to_close_unsafe",
        "execution_p_up_side_disagreement",
        "execution_market_exposure_limit",
    ],
)
def test_v8_2_never_falls_back_around_risk_blocker(
    candidate_blocker: str,
) -> None:
    row = v82.select_support_preserving_overlay_decision(
        candidate=_decision(
            action="NO_TRADE",
            allowed=False,
            blockers=[candidate_blocker],
            rank_passed=False,
            point_action="BUY_UP_SELL_BEFORE_CLOSE",
        ),
        baseline=_decision(action="BUY_DOWN_SELL_BEFORE_CLOSE"),
    )
    assert row["selected_action"] == "NO_TRADE"
    assert row["execution_guard_order_allowed"] is False
    assert row["fallback_applied"] is False


def test_v8_2_baseline_guard_must_pass() -> None:
    row = v82.select_support_preserving_overlay_decision(
        candidate=_decision(
            action="NO_TRADE",
            allowed=False,
            blockers=["policy_selected_no_trade"],
            rank_passed=False,
        ),
        baseline=_decision(
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
            allowed=False,
            blockers=["execution_book_stale"],
        ),
    )
    assert row["selected_action"] == "NO_TRADE"
    assert "v6_7_independent_full_guard_failed" in row[
        "selection_reason_codes"
    ]


def _historical_row(
    *,
    market_id: str,
    candidate_allowed: bool,
    candidate_action: str,
    candidate_target: float,
    baseline_action: str,
    baseline_target: float,
) -> dict:
    candidate_side = "UP" if "UP" in candidate_action else "DOWN"
    if candidate_action == "NO_TRADE":
        candidate_side = "NONE"
    baseline_side = "UP" if "UP" in baseline_action else "DOWN"
    return {
        "market_id": market_id,
        "market_close_ts": 2000,
        "selected_action": candidate_action,
        "selected_side": candidate_side,
        "candidate_execution_guard_order_allowed": candidate_allowed,
        "candidate_execution_blocking_reason_codes": (
            [] if candidate_allowed else ["policy_selected_no_trade"]
        ),
        "rank_abstention_passed": candidate_allowed,
        "point_selected_action": (
            candidate_action
            if candidate_allowed
            else "BUY_UP_SELL_BEFORE_CLOSE"
        ),
        "selected_target_after_cost_net_pnl_per_contract": candidate_target,
        "baseline_action": baseline_action,
        "baseline_side": baseline_side,
        "baseline_decision_ts": 1000,
        "baseline_execution_guard_order_allowed": True,
        "baseline_execution_blocking_reason_codes": [],
        "baseline_target_after_cost_net_pnl_per_contract": baseline_target,
    }


def test_v8_2_historical_gate_freezes_decisions_before_targets() -> None:
    rows = [
        _historical_row(
            market_id="m1",
            candidate_allowed=True,
            candidate_action="BUY_UP_SELL_BEFORE_CLOSE",
            candidate_target=0.4,
            baseline_action="BUY_DOWN_SELL_BEFORE_CLOSE",
            baseline_target=-0.1,
        ),
        _historical_row(
            market_id="m2",
            candidate_allowed=False,
            candidate_action="NO_TRADE",
            candidate_target=99.0,
            baseline_action="BUY_DOWN_SELL_BEFORE_CLOSE",
            baseline_target=0.2,
        ),
    ]
    result = v82.build_historical_support_preserving_overlay_v8_2(
        rows,
        profile=_profile(),
    )
    report = result["report"]
    assert report["candidate_guard_accepted_market_count"] == 2
    assert report["v6_7_guard_accepted_market_count"] == 2
    assert report["candidate_total_after_cost_net_pnl_at_frozen_size"] == (
        pytest.approx(0.12)
    )
    assert report["v6_7_total_after_cost_net_pnl_at_frozen_size"] == (
        pytest.approx(0.02)
    )
    assert report["historical_noninferiority_gate_passed"] is True
    assert result["decisions"][1]["selection_source"] == (
        "v6_7_rank_abstention_fallback"
    )
    assert all(
        row["target_or_outcome_used_for_selection"] is False
        for row in result["decisions"]
    )
    assert not any(
        "pnl" in key.lower() or "outcome" in key.lower()
        for row in result["decisions"]
        for key in row
        if key != "target_or_outcome_used_for_selection"
    )


def test_v8_2_historical_gate_fails_when_primary_is_inferior() -> None:
    rows = [
        _historical_row(
            market_id="m1",
            candidate_allowed=True,
            candidate_action="BUY_UP_SELL_BEFORE_CLOSE",
            candidate_target=-0.4,
            baseline_action="BUY_DOWN_SELL_BEFORE_CLOSE",
            baseline_target=0.2,
        )
    ]
    result = v82.build_historical_support_preserving_overlay_v8_2(
        rows,
        profile=_profile(),
    )
    report = result["report"]
    assert report["historical_noninferiority_gate_passed"] is False
    assert "historical_total_after_cost_pnl_inferior_to_v6_7" in report[
        "historical_gate_blocking_reason_codes"
    ]
    assert report["future_target_free_canary_allowed"] is False


def test_v8_2_rejects_profile_drift() -> None:
    profile = copy.deepcopy(_profile())
    profile["policy_contract"]["risk_blocker_bypass_allowed"] = True
    with pytest.raises(ValueError, match="profile invalid"):
        v82.validate_support_preserving_overlay_v8_2_profile(profile)
