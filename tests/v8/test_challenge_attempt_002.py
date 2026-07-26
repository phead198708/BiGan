from __future__ import annotations

import copy

import pytest

from bigan.v8.polymarket.challenge_attempt_002 import (
    CANDIDATE_ID,
    ChallengeAttempt002Error,
    evaluate_attempt_002_future_rows,
    validate_attempt_002_preregistration,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES


def _protocol() -> dict:
    return {
        "schema_version": (
            "bigan-v8-challenge-attempt-002-preregistration-v1"
        ),
        "attempt_id": "v8-1-challenger-future-attempt-002",
        "issue": 262,
        "goal": "challenge_model_promote_to_champion_model",
        "model_version": "v8.1",
        "candidate_id": CANDIDATE_ID,
        "baseline_id": "matched_frozen_v6_7",
        "lineage": {
            "candidate_implementation_commit": "a" * 40,
            "candidate_profile_sha256": "b" * 64,
            "candidate_module_sha256": "c" * 64,
            "historical_success_standard_sha256": "d" * 64,
            "historical_iteration_003_entry_sha256": "e" * 64,
            "historical_iteration_003_result_sha256": "f" * 64,
            "historical_evidence_commit": "1" * 40,
            "attempt_001_closure_sha256": "2" * 64,
            "development_registry_sha256": "3" * 64,
        },
        "historical_eligibility": {
            "iteration_number": 3,
            "all_historical_success_criteria_passed": True,
            "attempt_002_preregistration_allowed": True,
            "historical_result_is_promotion_evidence": False,
            "development_iterations_consumed": 3,
            "development_iteration_limit": 5,
            "further_historical_iteration_allowed_after_success": False,
        },
        "future_window": {
            "market_family": "btc_updown_5m",
            "exact_quality_valid_market_count": 120,
            "selection_rule": (
                "chronological_earliest_quality_valid_after_freeze_boundary"
            ),
            "strictly_future_and_disjoint_from_all_development_data": True,
            "same_source_market_rows_for_candidate_and_baseline": True,
            "result_dependent_extension_allowed": False,
            "minimum_accepted_candidate_support": None,
            "support_mode": (
                "full_window_paired_no_minimum_accepted_support"
            ),
            "service_root": (
                "examples/v8/polymarket_live_runs/"
                "challenge-model-v8-1-attempt-002"
            ),
            "operator_collection_authorization_required": True,
            "operator_collection_authorization_granted": False,
            "collection_started": False,
            "collector_pid": None,
            "attempted_market_count": 0,
            "quality_valid_market_count": 0,
            "outcomes_resolution_labels_or_pnl_opened": False,
        },
        "decision_freeze": {
            "candidate_decisions_frozen_before_target_access": True,
            "baseline_decisions_frozen_before_target_access": True,
            "candidate_threshold_feature_controller_and_sizing_frozen": True,
            "candidate_fixed_position_size": 1.0,
            "baseline_fixed_position_size": 0.2,
            "candidate_replacement_allowed": False,
            "result_selected_rerun_allowed": False,
            "target_used_as_decision_input": False,
        },
        "settlement": {
            "official_read_only_resolution_only": True,
            "all_120_markets_settled_before_evaluation": True,
            "target_access_after_decision_freeze_and_market_close": True,
            "single_use_target_access_claim_required": True,
            "source_outcome_blind_rows_mutated": False,
            "costs_subtracted_exactly_once": True,
        },
        "target_mapping": {
            "candidate_trade_value_field": (
                "runtime_policy_after_cost_net_pnl_per_contract"
            ),
            "candidate_position_size": 1.0,
            "baseline_position_size": 0.2,
            "no_trade_after_cost_pnl": 0.0,
            "comparison_unit": "market_id",
            "all_120_markets_included": True,
        },
        "evaluation": {
            "single_use": True,
            "protocol_isomorphic_to_historical_standard": True,
            "manual_code_change_after_target_access_allowed": False,
            "all_hard_gates_required_for_success": True,
            "full_window_paired_gate": {
                "scope": "all_120_markets",
                "method": "paired_market_percentile_bootstrap",
                "resample_count": 10000,
                "seed": 26212001,
                "lower_confidence_bound_quantile": 0.025,
                "candidate_minus_baseline_lcb_minimum_exclusive": 0.0,
            },
            "absolute_candidate_gate": {
                "scope": "all_120_markets",
                "method": "market_percentile_bootstrap",
                "resample_count": 10000,
                "seed": 26212002,
                "lower_confidence_bound_quantile": 0.025,
                "candidate_lcb_minimum_exclusive": 0.0,
            },
            "robustness_gates": {
                "largest_winner_removed_candidate_pnl_minimum_exclusive": 0.0,
                "chronological_halves": {
                    "first_half_market_count": 60,
                    "second_half_market_count": 60,
                    "method": "chronological_equal_halves",
                    "bootstrap_method": "market_percentile_bootstrap",
                    "resample_count": 10000,
                    "first_half_seed": 26212003,
                    "second_half_seed": 26212004,
                    "upper_confidence_bound_quantile": 0.975,
                    "upper_confidence_bound_minimum_inclusive": 0.0,
                },
            },
            "support": {
                "hard_gate": False,
                "minimum_accepted_candidate_support": None,
                "all_market_rows_remain_in_paired_gate": True,
            },
            "concentration_diagnostics": {
                "hard_gate": False,
                "report_selected_action_distribution": True,
                "report_selected_side_distribution": True,
                "report_largest_absolute_single_market_pnl_share": True,
                "report_largest_winner_share_of_positive_pnl": True,
            },
        },
        "alpha_spending": {
            "promotion_attempt_number": 2,
            "one_sided_alpha": 0.025,
            "confidence_level": 0.975,
            "candidate_count": 1,
            "attempt_consumed_at": "first_single_use_target_access_claim",
            "attempt_consumed": False,
        },
        "promotion_evidence": {
            "historical_results_eligible": False,
            "attempt_002_future_window_only": True,
            "eligible_only_if_all_hard_gates_pass": True,
            "promotion_audit_required_after_gate_pass": True,
            "automatic_promotion_allowed": False,
        },
        "safety": SAFE_FALSES,
    }


def _row(index: int, *, selected: bool) -> dict:
    return {
        "market_id": f"future-{index:03d}",
        "market_start_ts": 2_000_000_000_000 + index * 300_000,
        "candidate_action": (
            "BUY_DOWN_SELL_BEFORE_CLOSE" if selected else "NO_TRADE"
        ),
        "candidate_side": "DOWN" if selected else "NONE",
        "candidate_after_cost_pnl": 0.2 if selected else 0.0,
        "candidate_fixed_position_size": 1.0,
        "candidate_position_size": 1.0 if selected else 0.0,
        "baseline_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
        "baseline_side": "DOWN",
        "baseline_after_cost_pnl": -0.01,
        "baseline_fixed_position_size": 0.2,
        "baseline_position_size": 0.2,
        "candidate_decision_frozen_before_target_access": True,
        "baseline_decision_frozen_before_target_access": True,
        "target_used_as_decision_input": False,
        "settled_after_market_close": True,
        "same_settled_market_for_candidate_and_baseline": True,
        "safety": SAFE_FALSES,
    }


def test_protocol_is_exact_and_collection_stays_unauthorized() -> None:
    protocol = _protocol()
    validate_attempt_002_preregistration(protocol)
    assert protocol["future_window"]["collection_started"] is False
    assert (
        protocol["future_window"][
            "operator_collection_authorization_granted"
        ]
        is False
    )
    assert protocol["safety"] == SAFE_FALSES

    tampered = copy.deepcopy(protocol)
    tampered["future_window"]["minimum_accepted_candidate_support"] = 40
    with pytest.raises(ChallengeAttempt002Error, match="future_window"):
        validate_attempt_002_preregistration(tampered)


def test_synthetic_sparse_future_window_passes_isomorphic_gate() -> None:
    selected_indices = {0, 24, 48, 72, 96}
    rows = [
        _row(index, selected=index in selected_indices)
        for index in range(120)
    ]
    result = evaluate_attempt_002_future_rows(
        rows,
        protocol=_protocol(),
    )

    assert result["market_count"] == 120
    assert result["metrics"]["accepted_market_count"] == 5
    assert result["metrics"]["candidate_total_after_cost_pnl"] == pytest.approx(
        1.0
    )
    assert result["metrics"][
        "candidate_largest_winner_removed_after_cost_pnl"
    ] == pytest.approx(0.8)
    assert result["bootstrap"]["paired_delta_97_5_lcb"] > 0.0
    assert result["bootstrap"]["candidate_absolute_97_5_lcb"] > 0.0
    assert result["all_future_success_criteria_passed"] is True
    assert result["promotion_evidence_eligible"] is True
    assert result["automatic_promotion_allowed"] is False
    assert result["safety"] == SAFE_FALSES


def test_nonzero_no_trade_pnl_fails_closed_before_bootstrap() -> None:
    rows = [_row(index, selected=False) for index in range(120)]
    rows[0]["candidate_after_cost_pnl"] = 0.01
    with pytest.raises(
        ChallengeAttempt002Error,
        match="candidate_no_trade",
    ):
        evaluate_attempt_002_future_rows(rows, protocol=_protocol())
