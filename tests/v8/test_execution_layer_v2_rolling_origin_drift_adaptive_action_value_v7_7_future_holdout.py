from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout import (
    EXACT_MARKET_COUNT,
    FROZEN_PLAN_SHA256,
    MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
    SCAN_CAP,
    STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE,
    _safety_fields,
    build_v7_7_future_pnl_noninferiority_gate,
    validate_v7_7_future_holdout_plan,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def _target_row(index: int, pnl: float, *, side: str = "UP") -> dict:
    return {
        "market_id": f"market-{index:03d}",
        "decision_ts": 1_000_000 + index,
        "max_input_ts": 999_000 + index,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def _market_ids() -> list[str]:
    return [f"market-{index:03d}" for index in range(120)]


def test_plan_freezes_bounded_strictly_later_outcome_blind_collection() -> None:
    plan = _plan()

    validate_v7_7_future_holdout_plan(plan)
    assert _sha256_file(PLAN_PATH) == FROZEN_PLAN_SHA256

    collection = plan["collection"]
    assert collection["exact_quality_valid_market_count"] == EXACT_MARKET_COUNT == 120
    assert collection["maximum_attempted_market_count"] == SCAN_CAP == 180
    assert (
        collection["strictly_later_minimum_market_start_ts_exclusive"]
        == STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE
        == 1_784_760_900_000
    )
    assert collection["outcomes_resolution_labels_or_pnl_opened"] is False
    assert collection["candidate_model_scoring_during_collection_allowed"] is False


def test_plan_uses_inclusive_noninferiority_without_side_quota() -> None:
    plan = _plan()
    freeze = plan["target_free_decision_freeze"]
    gate = plan["single_use_future_pnl_gate"]

    assert freeze["minimum_v7_7_guard_accepted_unique_market_count"] == (
        MINIMUM_GUARD_ACCEPTED_MARKET_COUNT
    )
    assert freeze["side_quota_enabled"] is False
    assert gate["comparison_operator"] == "greater_than_or_equal"
    assert gate["equality_passes_noninferiority"] is True
    assert gate["candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"] == 0.0
    assert (
        gate[
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive"
        ]
        == 0.0
    )
    assert gate["candidate_total_after_cost_pnl_minimum_exclusive"] == 0.0


def test_plan_rejects_outcome_access_or_gate_drift() -> None:
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["collection"]["outcomes_resolution_labels_or_pnl_opened"] = True
    with pytest.raises(ValueError, match="collection"):
        validate_v7_7_future_holdout_plan(changed)

    changed = copy.deepcopy(plan)
    changed["single_use_future_pnl_gate"][
        "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"
    ] = -0.01
    with pytest.raises(ValueError, match="single_use_gate"):
        validate_v7_7_future_holdout_plan(changed)


def test_plan_safety_remains_fail_closed() -> None:
    assert _plan()["safety"] == _safety_fields()
    assert _safety_fields() == {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_candidate_allowed": False,
        "live_trading_enabled": False,
    }


def test_equal_positive_candidate_passes_inclusive_noninferiority() -> None:
    rows = [_target_row(index, 0.01) for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        rows,
        baseline_rows=[dict(row) for row in rows],
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="a" * 64,
    )

    assert report["candidate_minus_v6_7_after_cost_pnl"] == 0.0
    assert report["future_noninferiority_gate_passed"] is True
    assert report["future_pnl_gate_passed"] is True
    assert report["model_improvement_demonstrated"] is False
    assert report["promotion_discussion_evidence_available"] is True
    assert report["paper_candidate_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False


def test_distinct_candidate_actions_can_pass_without_side_quota() -> None:
    candidate = [_target_row(index, 0.02, side="DOWN") for index in range(40)]
    baseline = [_target_row(index, 0.01, side="UP") for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="b" * 64,
    )

    assert report["candidate_side_distribution_diagnostic"] == {"DOWN": 40}
    assert report["v6_7_side_distribution_diagnostic"] == {"UP": 40}
    assert report["side_quota_enabled"] is False
    assert report["candidate_minus_v6_7_after_cost_pnl"] == pytest.approx(0.4)
    assert report["future_pnl_gate_passed"] is True


def test_equal_negative_candidate_fails_only_absolute_pnl_check() -> None:
    rows = [_target_row(index, -0.01) for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        rows,
        baseline_rows=[dict(row) for row in rows],
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="c" * 64,
    )

    assert report["future_noninferiority_gate_passed"] is True
    assert report["future_pnl_gate_passed"] is False
    assert report["future_pnl_gate_blocking_reason_codes"] == [
        "candidate_total_after_cost_pnl_not_positive"
    ]


def test_inferior_candidate_fails_total_noninferiority() -> None:
    candidate = [_target_row(index, 0.009) for index in range(40)]
    baseline = [_target_row(index, 0.01) for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="d" * 64,
    )

    assert report["future_noninferiority_gate_passed"] is False
    assert "candidate_total_pnl_inferior_to_v6_7" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]


def test_largest_winner_removed_noninferiority_is_hard_gate() -> None:
    candidate = [_target_row(index, 0.01) for index in range(40)]
    candidate[0] = _target_row(0, 1.0)
    baseline = [_target_row(index, 0.02) for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="e" * 64,
    )

    assert report["candidate_minus_v6_7_after_cost_pnl"] > 0.0
    assert report[
        "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl"
    ] < 0.0
    assert report["future_noninferiority_gate_passed"] is False
    assert "candidate_largest_winner_removed_pnl_inferior_to_v6_7" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]


def test_support_and_complete_settlement_remain_fail_closed() -> None:
    candidate = [_target_row(index, 0.01) for index in range(39)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        candidate,
        baseline_rows=[dict(row) for row in candidate],
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="f" * 64,
    )
    assert "insufficient_v7_7_guard_accepted_unique_market_support" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]

    with pytest.raises(ValueError, match="exact settled"):
        build_v7_7_future_pnl_noninferiority_gate(
            [_target_row(index, 0.01) for index in range(40)],
            baseline_rows=[_target_row(index, 0.01) for index in range(40)],
            evaluation_market_ids=_market_ids(),
            settled_market_ids=_market_ids()[:-1],
            plan=_plan(),
            target_free_freeze_sha256="0" * 64,
        )
