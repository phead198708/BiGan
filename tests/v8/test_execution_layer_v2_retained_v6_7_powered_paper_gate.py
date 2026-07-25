from __future__ import annotations

from copy import deepcopy

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_retained_v6_7_powered_paper_gate as powered,
)


def _gate_plan() -> dict:
    return {
        "schema_version": (
            "bigan-v8-v6-7-forward-paper-candidate-gate-plan-v1"
        ),
        "frozen": True,
        "champion": "retained_v6_7_champion",
        "exact_quality_valid_market_count": 195,
        "maximum_capture_attempt_count": 228,
        "minimum_target_free_guard_accepted_market_count": 186,
        "bounded_batch_market_count": 12,
        "strictly_later_minimum_market_start_ts_exclusive": 100,
        "required_disjoint_identity_fields": [
            "market_id",
            "slug",
            "decision_id",
            "source_row_hash",
        ],
        "target_free_decision_freeze_required": True,
        "official_read_only_settlement_on_quarantine_copies": True,
        "complete_settlement_required": True,
        "one_evaluation_only": True,
        "result_selected_rerun_allowed": False,
        "result_selected_extension_allowed": False,
        "side_quota_enabled": False,
        "paper_candidate_auto_unlock_allowed": False,
        "separate_manual_paper_authorization_issue_required": True,
        "hard_gate_checks": {
            "total_after_cost_pnl_minimum_exclusive": 0.0,
            "largest_winner_removed_after_cost_pnl_minimum_exclusive": 0.0,
            "market_bootstrap_one_sided_lower_bound_minimum_exclusive": 0.0,
            "market_bootstrap_seed": 2522026,
            "market_bootstrap_resample_count": 10000,
            "market_bootstrap_one_sided_confidence_level": 0.95,
            "runtime_safety_and_forbidden_field_checks_required": True,
        },
        **powered.SAFETY,
    }


def _index_row(sequence: int, *, quality: bool = True) -> dict:
    return {
        "sequence": sequence,
        "capture_quality_valid": quality,
        "market_start_ts": 100 + sequence,
        "market_id": f"market-{sequence}",
        "slug": f"slug-{sequence}",
        "decision_id": f"decision-{sequence}",
        "source_row_hash": f"{sequence:064x}",
    }


def _prior() -> dict[str, set[str]]:
    return {
        "market_id": set(),
        "slug": set(),
        "decision_id": set(),
        "source_row_hash": set(),
    }


def _target(index: int, pnl: float) -> dict:
    return {
        "market_id": f"market-{index}",
        "side": "UP" if index % 2 else "DOWN",
        "action": (
            "BUY_UP_SELL_BEFORE_CLOSE"
            if index % 2
            else "BUY_DOWN_SELL_BEFORE_CLOSE"
        ),
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
        "max_input_ts": 1_000 + index,
        "decision_ts": 2_000 + index,
    }


def test_powered_gate_plan_is_exact_and_fail_closed() -> None:
    plan = _gate_plan()
    powered.validate_powered_gate_plan(plan)
    assert plan["exact_quality_valid_market_count"] == 195
    assert plan["minimum_target_free_guard_accepted_market_count"] == 186
    assert plan["side_quota_enabled"] is False
    mutated = deepcopy(plan)
    mutated["result_selected_extension_allowed"] = True
    with pytest.raises(ValueError, match="single_use"):
        powered.validate_powered_gate_plan(mutated)


def test_selects_earliest_exact_195_within_attempt_cap() -> None:
    rows = [_index_row(index) for index in range(1, 201)]
    selected, attempted, summary = powered.select_powered_target_free_window(
        rows,
        gate_plan=_gate_plan(),
        prior_registries=_prior(),
    )
    assert len(selected) == 195
    assert len(attempted) == 195
    assert selected[-1]["sequence"] == 195
    assert summary["exact_window_ready"] is True
    assert summary["collection_contract_violation_count"] == 0


def test_invalid_capture_counts_toward_scan_but_not_support() -> None:
    rows = [_index_row(1, quality=False)] + [
        _index_row(index) for index in range(2, 197)
    ]
    selected, attempted, summary = powered.select_powered_target_free_window(
        rows,
        gate_plan=_gate_plan(),
        prior_registries=_prior(),
    )
    assert len(selected) == 195
    assert len(attempted) == 196
    assert summary["exclusion_reason_distribution"] == {
        "capture_quality_invalid": 1
    }


def test_prior_overlap_is_a_fail_closed_collection_violation() -> None:
    rows = [_index_row(index) for index in range(1, 196)]
    prior = _prior()
    prior["market_id"].add("market-1")
    selected, _, summary = powered.select_powered_target_free_window(
        rows,
        gate_plan=_gate_plan(),
        prior_registries=prior,
    )
    assert len(selected) == 194
    assert summary["exact_window_ready"] is False
    assert summary["collection_contract_violation_distribution"] == {
        "prior_market_id_overlap": 1
    }


def test_powered_pnl_gate_passes_with_distributed_positive_edge() -> None:
    targets = [_target(index, 0.02) for index in range(190)]
    report = powered.build_powered_pnl_gate(
        targets,
        evaluation_market_ids=[f"market-{index}" for index in range(195)],
        minimum_guard_accepted_market_count=186,
        gate_plan=_gate_plan(),
        target_free_freeze_sha256="a" * 64,
    )
    assert report["guard_accepted_unique_market_count"] == 190
    assert report["guard_blocked_no_bet_zero_market_count"] == 5
    assert report["total_after_cost_pnl"] == pytest.approx(3.8)
    assert report["largest_winner_removed_after_cost_pnl"] == pytest.approx(
        3.78
    )
    assert (
        report["market_bootstrap"]["one_sided_lower_confidence_bound"] > 0
    )
    assert report["powered_paper_candidate_readiness_gate_passed"] is True
    assert report["manual_paper_authorization_review_eligible"] is True
    assert report["paper_candidate_allowed"] is False


def test_powered_pnl_gate_rejects_insufficient_support() -> None:
    report = powered.build_powered_pnl_gate(
        [_target(index, 0.02) for index in range(185)],
        evaluation_market_ids=[f"market-{index}" for index in range(195)],
        minimum_guard_accepted_market_count=186,
        gate_plan=_gate_plan(),
        target_free_freeze_sha256="b" * 64,
    )
    assert report["powered_paper_candidate_readiness_gate_passed"] is False
    assert (
        "powered_paper_gate_minimum_guard_accepted_market_support_failed"
        in report["powered_paper_candidate_readiness_blocking_reason_codes"]
    )


def test_single_winner_does_not_pass_robustness_gate() -> None:
    targets = [_target(0, 1.0)] + [
        _target(index, 0.0) for index in range(1, 190)
    ]
    report = powered.build_powered_pnl_gate(
        targets,
        evaluation_market_ids=[f"market-{index}" for index in range(195)],
        minimum_guard_accepted_market_count=186,
        gate_plan=_gate_plan(),
        target_free_freeze_sha256="c" * 64,
    )
    assert report["total_after_cost_pnl"] == 1.0
    assert report["largest_winner_removed_after_cost_pnl"] == 0.0
    assert report["powered_paper_candidate_readiness_gate_passed"] is False
    assert (
        "powered_paper_gate_largest_winner_removed_after_cost_pnl_positive_failed"
        in report["powered_paper_candidate_readiness_blocking_reason_codes"]
    )


def test_target_causality_violation_fails_closed() -> None:
    targets = [_target(index, 0.02) for index in range(190)]
    targets[0]["max_input_ts"] = targets[0]["decision_ts"] + 1
    with pytest.raises(ValueError, match="runtime target row"):
        powered.build_powered_pnl_gate(
            targets,
            evaluation_market_ids=[
                f"market-{index}" for index in range(195)
            ],
            minimum_guard_accepted_market_count=186,
            gate_plan=_gate_plan(),
            target_free_freeze_sha256="d" * 64,
        )


def test_all_safety_flags_remain_blocked() -> None:
    report = powered.build_powered_pnl_gate(
        [_target(index, 0.02) for index in range(190)],
        evaluation_market_ids=[f"market-{index}" for index in range(195)],
        minimum_guard_accepted_market_count=186,
        gate_plan=_gate_plan(),
        target_free_freeze_sha256="e" * 64,
    )
    for field, expected in powered.SAFETY.items():
        assert report[field] == expected
    assert report["paper_candidate_auto_unlock_allowed"] is False
    assert report["result_selected_rerun_allowed"] is False
    assert report["result_selected_extension_allowed"] is False
