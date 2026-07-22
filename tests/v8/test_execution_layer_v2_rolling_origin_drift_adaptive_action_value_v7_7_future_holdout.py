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
    build_v7_7_target_free_holdout_freeze_report,
    materialize_guard_accepted_runtime_decisions,
    select_v7_7_future_holdout_window,
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


def _index_row(index: int, *, quality_valid: bool = True) -> dict:
    market_start_ts = STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE + 300_000 * (
        index + 1
    )
    return {
        "sequence": index + 1,
        "market_id": f"market-{index:03d}",
        "slug": f"market-slug-{index:03d}",
        "decision_id": f"decision-{index:03d}",
        "source_row_hash": f"{index + 1:064x}",
        "market_start_ts": market_start_ts,
        "market_end_ts": market_start_ts + 300_000,
        "capture_quality_valid": quality_valid,
    }


def _action_rows() -> list[dict]:
    actions = (
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "NO_TRADE",
    )
    return [
        {
            "market_id": f"market-{index:03d}",
            "decision_ts": 2_000_000 + index,
            "max_input_ts": 1_999_000 + index,
            "action": action,
        }
        for index in range(120)
        for action in actions
    ]


def _guard_rows(*, accepted_count: int, side: str = "DOWN") -> list[dict]:
    return [
        {
            "market_id": f"market-{index:03d}",
            "selected_action": f"BUY_{side}_SELL_BEFORE_CLOSE"
            if index < accepted_count
            else "NO_TRADE",
            "selected_side": side if index < accepted_count else "NONE",
            "execution_guard_order_allowed": index < accepted_count,
            "source_score_mutated": False,
            "labels_outcomes_or_pnl_opened": False,
        }
        for index in range(120)
    ]


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


def test_window_selection_uses_earliest_exact_120_within_scan_cap() -> None:
    rows = [_index_row(index) for index in range(130)]
    selected, attempted, summary = select_v7_7_future_holdout_window(
        rows,
        plan=_plan(),
        prior_market_ids=set(),
        prior_slugs=set(),
        prior_decision_ids=set(),
        prior_source_row_hashes=set(),
    )

    assert len(selected) == 120
    assert len(attempted) == 130
    assert [row["sequence"] for row in selected] == list(range(1, 121))
    assert summary["exact_window_ready"] is True


def test_window_selection_excludes_invalid_and_prior_identity_rows() -> None:
    rows = [_index_row(index) for index in range(122)]
    rows[0]["capture_quality_valid"] = False
    selected, _, summary = select_v7_7_future_holdout_window(
        rows,
        plan=_plan(),
        prior_market_ids={"market-001"},
        prior_slugs=set(),
        prior_decision_ids=set(),
        prior_source_row_hashes=set(),
    )

    assert len(selected) == 120
    assert selected[0]["market_id"] == "market-002"
    assert summary["exclusion_reason_distribution"] == {
        "capture_quality_invalid": 1,
        "prior_market_id_overlap": 1,
    }


def test_target_free_freeze_passes_one_sided_support_without_outcomes() -> None:
    selected = [_index_row(index) for index in range(120)]
    report = build_v7_7_target_free_holdout_freeze_report(
        selected,
        attempted_rows=selected,
        action_rows=_action_rows(),
        candidate_guard_rows=_guard_rows(accepted_count=40, side="DOWN"),
        baseline_guard_rows=_guard_rows(accepted_count=50, side="UP"),
        selection_summary={"exact_window_ready": True},
        plan=_plan(),
        stage_started_ts=max(row["market_end_ts"] for row in selected) + 1,
        collector_index_sha256="1" * 64,
    )

    assert report["target_free_freeze_passed"] is True
    assert report["v7_7_guard_accepted_market_count"] == 40
    assert report["v7_7_guard_accepted_side_distribution_diagnostic"] == {"DOWN": 40}
    assert report["side_quota_enabled"] is False
    assert report["future_target_access_allowed"] is True
    assert report["labels_outcomes_resolution_or_pnl_opened"] is False


def test_target_free_freeze_fails_support_causality_and_target_leakage() -> None:
    selected = [_index_row(index) for index in range(120)]
    actions = _action_rows()
    actions[0]["max_input_ts"] = actions[0]["decision_ts"] + 1
    actions[1]["settlement_pnl"] = 1.0
    report = build_v7_7_target_free_holdout_freeze_report(
        selected,
        attempted_rows=selected,
        action_rows=actions,
        candidate_guard_rows=_guard_rows(accepted_count=39),
        baseline_guard_rows=_guard_rows(accepted_count=50),
        selection_summary={"exact_window_ready": True},
        plan=_plan(),
        stage_started_ts=max(row["market_end_ts"] for row in selected) + 1,
        collector_index_sha256="2" * 64,
    )

    assert report["target_free_freeze_passed"] is False
    assert set(report["target_free_blocking_reason_codes"]) >= {
        "target_free_v7_7_guard_accepted_support_insufficient",
        "target_free_five_action_grid_incomplete",
        "target_free_feature_causality_violation",
        "target_free_forbidden_target_field_present",
    }


def test_guard_accepted_runtime_decisions_bind_frozen_source_rows() -> None:
    actions = _action_rows()
    for row in actions:
        row["market_close_ts"] = row["decision_ts"] + 60_000
        row["decision_id"] = f"{row['market_id']}:{row['action']}"
        row["microstructure_snapshot"] = {"time_to_close_seconds": 60.0}
    accepted = materialize_guard_accepted_runtime_decisions(
        _guard_rows(accepted_count=40, side="DOWN"),
        action_rows=actions,
    )

    assert len(accepted) == 40
    assert {row["side"] for row in accepted} == {"DOWN"}
    assert {row["action"] for row in accepted} == {
        "BUY_DOWN_SELL_BEFORE_CLOSE"
    }
    assert all(row["max_input_ts"] <= row["decision_ts"] for row in accepted)
    assert all(row["source_score_mutated"] is False for row in accepted)
    assert all(
        row["labels_outcomes_resolution_or_pnl_opened"] is False
        for row in accepted
    )


def test_guard_accepted_runtime_decisions_fail_on_missing_source_action() -> None:
    actions = [
        row
        for row in _action_rows()
        if not (
            row["market_id"] == "market-000"
            and row["action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
        )
    ]
    with pytest.raises(ValueError, match="source identity"):
        materialize_guard_accepted_runtime_decisions(
            _guard_rows(accepted_count=40, side="DOWN"),
            action_rows=actions,
        )
