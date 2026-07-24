from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_non_risk_abstention_fallback_v8_3 as v83,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_non_risk_abstention_fallback_v8_3_future_holdout as future,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_non_risk_abstention_fallback_v8_3_future_post_freeze as post,
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


def test_v8_3_batch_baseline_guard_rows_fail_closed() -> None:
    rows = v83._baseline_guard_rows(
        ["m1", "m2"],
        baseline_rows=[
            {
                "market_id": "m1",
                "action": "BUY_DOWN_SELL_BEFORE_CLOSE",
                "side": "DOWN",
                "decision_group_id": "g1",
                "decision_ts": 1000,
            }
        ],
        action_rows=[],
        v6_7_profile={"hard_execution_safety": {}},
    )
    assert rows[0]["execution_guard_order_allowed"] is False
    assert rows[0]["execution_blocking_reason_codes"] == [
        "selected_action_source_row_missing"
    ]
    assert rows[1]["selected_action"] == "NO_TRADE"
    assert rows[1]["execution_blocking_reason_codes"] == [
        "v6_7_no_positive_guard_compatible_action"
    ]
    assert all(
        row["labels_outcomes_resolution_or_pnl_opened"] is False
        for row in rows
    )


def test_v8_3_rejects_profile_drift() -> None:
    profile = copy.deepcopy(_profile())
    profile["policy_contract"][
        "explicit_execution_risk_blocker_bypass_allowed"
    ] = True
    with pytest.raises(ValueError, match="profile invalid"):
        v83.validate_non_risk_abstention_fallback_v8_3_profile(profile)


def test_v8_3_future_window_selects_earliest_exact_120() -> None:
    plan = _future_plan()
    boundary = plan["plan_created_ts"]
    rows = [
        {
            "sequence": index + 1,
            "scheduled_round_start_ts": boundary + (index + 1) * 300_000,
            "market_start_ts": boundary + (index + 1) * 300_000,
            "market_id": f"m{index:03d}",
            "slug": f"s{index:03d}",
            "decision_id": f"d{index:03d}",
            "source_row_hash": f"h{index:03d}",
            "capture_quality_valid": index != 0,
        }
        for index in range(121)
    ]
    selected, attempted, summary = (
        v83.select_non_risk_abstention_fallback_v8_3_future_window(
            rows,
            plan=plan,
            prior_market_ids=set(),
            prior_slugs=set(),
            prior_decision_ids=set(),
            prior_source_row_hashes=set(),
        )
    )
    assert len(attempted) == 121
    assert len(selected) == 120
    assert selected[0]["market_id"] == "m001"
    assert summary["exact_window_ready"] is True
    assert summary["exclusion_reason_distribution"] == {
        "capture_quality_invalid": 1
    }


def _runtime_target(
    market_id: str,
    *,
    pnl: float,
    side: str = "DOWN",
) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": 1000,
        "max_input_ts": 999,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def test_v8_3_future_pnl_gate_uses_preregistered_comparative_checks() -> None:
    market_ids = [f"m{index:03d}" for index in range(120)]
    candidate = [
        _runtime_target(market_id, pnl=0.01)
        for market_id in market_ids[:40]
    ]
    baseline = [
        _runtime_target(market_id, pnl=0.005)
        for market_id in market_ids[:40]
    ]
    report = v83.build_non_risk_abstention_fallback_v8_3_future_pnl_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=market_ids,
        settled_market_ids=market_ids,
        plan=_future_plan(),
        target_free_freeze_sha256="a" * 64,
    )
    assert report["candidate_after_cost_pnl"] == pytest.approx(0.4)
    assert report["candidate_minus_v6_7_after_cost_pnl"] == pytest.approx(0.2)
    assert report["future_pnl_gate_passed"] is True
    assert report["automatic_paper_or_live_unlock_allowed"] is False
    assert report["paper_candidate_allowed"] is False


def test_v8_3_future_pnl_gate_fails_when_candidate_is_negative() -> None:
    market_ids = [f"m{index:03d}" for index in range(120)]
    candidate = [
        _runtime_target(market_id, pnl=-0.01)
        for market_id in market_ids[:40]
    ]
    report = v83.build_non_risk_abstention_fallback_v8_3_future_pnl_gate(
        candidate,
        baseline_rows=[],
        evaluation_market_ids=market_ids,
        settled_market_ids=market_ids,
        plan=_future_plan(),
        target_free_freeze_sha256="b" * 64,
    )
    assert report["future_pnl_gate_passed"] is False
    assert "candidate_total_after_cost_pnl_not_positive" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]


def test_v8_3_settlement_adapter_preserves_fail_closed_identity() -> None:
    freeze = {
        "candidate_name": v83.CANDIDATE_NAME,
        "decision_freeze_created_ts": 1000,
        "exact_market_count": 120,
        "selected_rows": {"path": "/tmp/selected", "sha256": "1" * 64},
        "candidate_runtime": {"path": "/tmp/candidate", "sha256": "2" * 64},
        "v6_7_runtime": {"path": "/tmp/baseline", "sha256": "3" * 64},
    }
    adapter = post._settlement_engine_adapter(
        freeze=freeze,
        freeze_path=Path("/tmp/freeze.json"),
        freeze_sha256="4" * 64,
    )
    assert adapter["candidate_name"] == v83.CANDIDATE_NAME
    assert adapter["future_target_access_allowed"] is True
    assert adapter["labels_outcomes_resolution_or_pnl_opened"] is False
    assert adapter["paper_candidate_allowed"] is False
    assert adapter["polymarket_write_enabled"] is False


def test_v8_3_lineage_accepts_sealed_issue246_support_failure() -> None:
    plan = _future_plan()
    future._validate_frozen_lineage(
        plan=plan,
        profile_sha256=plan["lineage"]["candidate_profile_sha256"],
        protocol_sha256=plan["lineage"]["collector_protocol_sha256"],
        historical_gate={"historical_noninferiority_gate_passed": True},
        historical_gate_sha256=plan["lineage"][
            "historical_gate_manifest_sha256"
        ],
        issue246={
            "target_free_freeze_passed": False,
            "labels_outcomes_resolution_or_pnl_opened": False,
            "settlement_provider_called": False,
        },
        issue246_sha256=plan["lineage"]["issue246_target_free_manifest_sha256"],
        canary={
            "target_free_canary_passed": True,
            "issue246_outcomes_opened": False,
        },
        canary_report={"labels_outcomes_resolution_or_pnl_opened": False},
        canary_sha256=plan["lineage"][
            "issue246_target_free_canary_manifest_sha256"
        ],
    )
