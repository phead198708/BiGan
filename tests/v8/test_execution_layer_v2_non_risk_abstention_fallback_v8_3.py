from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_non_risk_abstention_fallback_v8_3 as v83,
)


def _profile() -> dict:
    path = (
        Path(__file__).parents[2]
        / "examples/v8/polymarket_configs"
        / "execution_layer_v2_non_risk_abstention_fallback_v8_3_profile.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _future_plan() -> dict:
    path = (
        Path(__file__).parents[2]
        / "examples/v8/polymarket_configs"
        / "execution_layer_v2_non_risk_abstention_fallback_v8_3_future_holdout_plan.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(
    *,
    market_id: str = "m1",
    action: str = "BUY_UP_SELL_BEFORE_CLOSE",
    allowed: bool = True,
    blockers: list[str] | None = None,
    point_action: str | None = "BUY_UP_SELL_BEFORE_CLOSE",
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
        "rank_abstention_passed": None,
        "point_selected_action": point_action,
    }


def test_v8_3_profile_is_frozen_and_fail_closed() -> None:
    profile = _profile()
    v83.validate_non_risk_abstention_fallback_v8_3_profile(profile)
    assert profile["historical_gate"]["side_quota_enabled"] is False
    assert profile["lineage"]["issue246_outcomes_allowed_for_v8_3"] is False
    assert profile["safety"]["paper_candidate_allowed"] is False


def test_v8_3_future_plan_is_strictly_later_and_preregistered() -> None:
    plan = _future_plan()
    v83.validate_non_risk_abstention_fallback_v8_3_future_plan(plan)
    assert plan["collection"][
        "strictly_later_minimum_market_start_ts_exclusive"
    ] > plan["lineage"]["issue246_latest_selected_market_end_ts"]
    assert plan["collection"][
        "candidate_scoring_during_raw_capture_allowed"
    ] is False
    assert plan["per_batch_target_free_diagnostic"][
        "candidate_scoring_after_batch_seal_allowed"
    ] is True
    assert plan["target_free_decision_freeze"]["side_quota_enabled"] is False


def test_v8_3_future_plan_rejects_gate_relaxation() -> None:
    plan = copy.deepcopy(_future_plan())
    plan["target_free_decision_freeze"][
        "minimum_candidate_guard_accepted_unique_market_count"
    ] = 1
    with pytest.raises(ValueError, match="future plan invalid"):
        v83.validate_non_risk_abstention_fallback_v8_3_future_plan(plan)


def test_v8_3_uses_v8_1_primary_when_guard_passes() -> None:
    row = v83.select_non_risk_abstention_fallback_v8_3_decision(
        candidate=_decision(),
        baseline=_decision(action="BUY_DOWN_SELL_BEFORE_CLOSE"),
    )
    assert row["selection_source"] == "v8_1_primary"
    assert row["selected_action"] == "BUY_UP_SELL_BEFORE_CLOSE"


@pytest.mark.parametrize(
    ("blocker", "point_action"),
    [
        ("v8_1_veto_to_no_trade", "BUY_UP_SELL_BEFORE_CLOSE"),
        ("policy_selected_no_trade", None),
        ("v6_7_no_positive_guard_compatible_action", None),
    ],
)
def test_v8_3_falls_back_for_policy_level_abstention(
    blocker: str,
    point_action: str | None,
) -> None:
    row = v83.select_non_risk_abstention_fallback_v8_3_decision(
        candidate=_decision(
            action="NO_TRADE",
            allowed=False,
            blockers=[blocker],
            point_action=point_action,
        ),
        baseline=_decision(action="BUY_DOWN_SELL_BEFORE_CLOSE"),
    )
    assert row["selection_source"] == (
        "v6_7_non_risk_abstention_fallback"
    )
    assert row["selected_action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
    assert row["explicit_execution_risk_blocker_bypass_used"] is False


@pytest.mark.parametrize(
    "blocker",
    [
        "execution_spread_too_wide",
        "execution_book_stale",
        "execution_liquidity_too_weak",
        "execution_time_to_close_unsafe",
        "execution_p_up_side_disagreement",
        "execution_market_exposure_limit",
        "execution_duplicate_market_side_position",
        "execution_required_runtime_fields_missing",
        "execution_provenance_invalid",
        "execution_guard_blocked_other",
    ],
)
def test_v8_3_does_not_bypass_explicit_risk_blocker(blocker: str) -> None:
    row = v83.select_non_risk_abstention_fallback_v8_3_decision(
        candidate=_decision(
            action="NO_TRADE",
            allowed=False,
            blockers=[blocker],
            point_action=None,
        ),
        baseline=_decision(action="BUY_DOWN_SELL_BEFORE_CLOSE"),
    )
    assert row["selected_action"] == "NO_TRADE"
    assert row["fallback_applied"] is False


def test_v8_3_requires_independent_v6_7_guard_pass() -> None:
    row = v83.select_non_risk_abstention_fallback_v8_3_decision(
        candidate=_decision(
            action="NO_TRADE",
            allowed=False,
            blockers=["policy_selected_no_trade"],
            point_action=None,
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
    baseline_target: float,
) -> dict:
    candidate_side = "UP" if "UP" in candidate_action else "DOWN"
    if candidate_action == "NO_TRADE":
        candidate_side = "NONE"
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
        "point_selected_action": candidate_action if candidate_allowed else "NO_TRADE",
        "selected_target_after_cost_net_pnl_per_contract": candidate_target,
        "baseline_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
        "baseline_side": "DOWN",
        "baseline_decision_ts": 1000,
        "baseline_execution_guard_order_allowed": True,
        "baseline_execution_blocking_reason_codes": [],
        "baseline_target_after_cost_net_pnl_per_contract": baseline_target,
    }


def test_v8_3_historical_gate_support_and_pnl_noninferiority() -> None:
    result = v83.build_non_risk_abstention_fallback_v8_3_historical(
        [
            _historical_row(
                market_id="m1",
                candidate_allowed=True,
                candidate_action="BUY_UP_SELL_BEFORE_CLOSE",
                candidate_target=0.3,
                baseline_target=-0.1,
            ),
            _historical_row(
                market_id="m2",
                candidate_allowed=False,
                candidate_action="NO_TRADE",
                candidate_target=99.0,
                baseline_target=0.2,
            ),
        ],
        profile=_profile(),
    )
    report = result["report"]
    assert report["candidate_guard_accepted_market_count"] == 2
    assert report["v6_7_guard_accepted_market_count"] == 2
    assert report["historical_noninferiority_gate_passed"] is True
    assert report["candidate_total_after_cost_net_pnl_at_frozen_size"] == (
        pytest.approx(0.1)
    )
    assert all(
        row["target_or_outcome_used_for_selection"] is False
        for row in result["decisions"]
    )


def test_v8_3_target_free_canary_reports_support_without_targets() -> None:
    candidate_rows = [
        {
            **_decision(
                market_id=f"m{i}",
                action="NO_TRADE",
                allowed=False,
                blockers=["policy_selected_no_trade"],
                point_action=None,
            ),
            "execution_guard_order_allowed": False,
        }
        for i in range(120)
    ]
    baseline_rows = [
        _decision(
            market_id=f"m{i}",
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
        )
        for i in range(120)
    ]
    result = v83.build_non_risk_abstention_fallback_v8_3_canary(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        profile=_profile(),
    )
    report = result["report"]
    assert report["market_count"] == 120
    assert report["guard_accepted_market_count"] == 120
    assert report["target_free_canary_passed"] is True
    assert report["issue246_outcomes_opened"] is False
    assert report["new_future_holdout_collection_allowed"] is True


def test_v8_3_rejects_profile_drift() -> None:
    profile = copy.deepcopy(_profile())
    profile["policy_contract"][
        "explicit_execution_risk_blocker_bypass_allowed"
    ] = True
    with pytest.raises(ValueError, match="profile invalid"):
        v83.validate_non_risk_abstention_fallback_v8_3_profile(profile)
