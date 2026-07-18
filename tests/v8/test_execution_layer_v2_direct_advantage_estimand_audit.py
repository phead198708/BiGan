from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _prospective_support_estimate,
    build_estimand_semantics_audit,
    build_gate_attrition_report,
    build_selected_policy_value_report,
    validate_direct_advantage_estimand_audit_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/execution_layer_v2_direct_advantage_estimand_audit_profile.json"
)


def test_issue199_profile_freezes_development_only_diagnostic_scope() -> None:
    profile = _load_json(PROFILE_PATH)

    validate_direct_advantage_estimand_audit_profile(profile)

    assert profile["allowed_role"] == "development_train"
    assert profile["evidence_scope"]["future_window_may_be_consumed"] is False
    assert profile["mutation_contract"]["new_candidate_fit_allowed"] is False
    assert profile["selector_contract"]["selector_or_score_mutation_allowed"] is False
    assert profile["safety"]["source_model_candidate_eligible"] is False
    assert profile["safety"]["#134_resume_allowed"] is False
    assert profile["safety"]["#146_start_allowed"] is False


def test_issue199_profile_rejects_future_or_threshold_scope_drift() -> None:
    profile = _load_json(PROFILE_PATH)
    profile["evidence_scope"]["issue_190_or_192_future_labels_may_be_opened"] = True

    with pytest.raises(ValueError, match="quarantined_evidence_sealed"):
        validate_direct_advantage_estimand_audit_profile(profile)

    profile = _load_json(PROFILE_PATH)
    profile["mutation_contract"]["threshold_mutation_allowed"] = True
    with pytest.raises(ValueError, match="no_mutation_or_fit"):
        validate_direct_advantage_estimand_audit_profile(profile)


def test_issue199_oracle_comparator_is_not_required_for_positive_value() -> None:
    rows = _decision_rows(
        returns={
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.10,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.20,
            "BUY_UP_SELL_BEFORE_CLOSE": 0.05,
            "BUY_DOWN_SELL_BEFORE_CLOSE": -0.10,
            "NO_TRADE": 0.0,
        }
    )

    report = build_estimand_semantics_audit(
        rows,
        run_id="test-semantics",
        profile_sha256="a" * 64,
    )

    assert report["estimand_identity_violation_count"] == 0
    assert report["non_oracle_positive_advantage_violation_count"] == 0
    assert report["positive_absolute_but_nonpositive_oracle_advantage_count"] == 2
    assert report["oracle_best_advantage_is_necessary_for_positive_post_cost_value"] is False
    assert (
        report["oracle_best_advantage_semantic_role"]
        == "ranking_regret_diagnostic_not_standalone_source_eligibility_hard_gate"
    )
    assert report["source_model_candidate_eligible"] is False


def test_issue199_gate_attrition_identifies_oracle_comparator_as_first_failure() -> None:
    profile = _load_json(PROFILE_PATH)
    rows = _decision_rows(
        returns={
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.10,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.20,
            "BUY_UP_SELL_BEFORE_CLOSE": 0.05,
            "BUY_DOWN_SELL_BEFORE_CLOSE": -0.10,
            "NO_TRADE": 0.0,
        }
    )
    calibration = _calibration(absolute_lcb=0.02, no_trade_lcb=0.02, oracle_lcb=-0.01)

    report = build_gate_attrition_report(
        rows,
        calibration=calibration,
        profile=profile,
        run_id="test-attrition",
    )

    assert report["trade_action_row_count"] == 4
    assert report["all_trade_rows_reconciled"] is True
    assert report["first_failing_estimand_distribution"] == {"advantage_vs_best_alternative": 4}
    assert report["gate_pass_counts"]["two_safety_estimands_passed"] == 4
    assert report["gate_pass_counts"]["all_three_estimands_passed"] == 0
    assert all(
        row["first_failing_estimand"] == "advantage_vs_best_alternative"
        for row in report["row_attrition"]
    )


def test_issue199_policy_value_uses_frozen_scores_and_keeps_strict_gate_blocked() -> None:
    profile = _load_json(PROFILE_PATH)
    rows = _decision_rows(
        returns={
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.10,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.20,
            "BUY_UP_SELL_BEFORE_CLOSE": 0.05,
            "BUY_DOWN_SELL_BEFORE_CLOSE": -0.10,
            "NO_TRADE": 0.0,
        },
        scores={
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.90,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.20,
            "BUY_UP_SELL_BEFORE_CLOSE": 0.70,
            "BUY_DOWN_SELL_BEFORE_CLOSE": 0.10,
            "NO_TRADE": 0.30,
        },
    )
    calibration = _calibration(absolute_lcb=0.02, no_trade_lcb=0.02, oracle_lcb=-0.01)

    first = build_selected_policy_value_report(
        rows,
        calibration=calibration,
        profile=profile,
        run_id="test-policy",
    )
    second = build_selected_policy_value_report(
        rows,
        calibration=calibration,
        profile=profile,
        run_id="test-policy",
    )

    assert first == second
    raw = first["policy_variants"]["raw_pairwise_selector"]
    strict = first["policy_variants"]["strict_all_three_lcb_selector"]
    two_safety = first["policy_variants"]["two_safety_estimand_selector_diagnostic_only"]
    assert raw["selected_action_distribution"] == {"BUY_UP_HOLD_TO_SETTLEMENT": 1}
    assert raw["market_level_post_cost_return"]["point_estimate"] == pytest.approx(0.10)
    assert strict["selected_action_distribution"] == {"NO_TRADE": 1}
    assert strict["market_level_post_cost_return"]["point_estimate"] == 0.0
    assert two_safety["selected_action_distribution"] == {"BUY_UP_HOLD_TO_SETTLEMENT": 1}
    assert raw["calibration_and_evaluation_share_oof_targets"] is False
    assert two_safety["calibration_and_evaluation_share_oof_targets"] is True
    assert two_safety["eligible_as_unbiased_candidate_evidence"] is False
    assert first["independent_calibration_or_nested_cross_fit_required_before_candidate_claim"]
    assert first["targets_used_for_selection"] is False
    assert first["source_model_candidate_eligible"] is False
    assert first["promotion_evidence_eligible"] is False


def test_issue199_power_estimate_distinguishes_negative_mean_from_uncertainty() -> None:
    positive = _prospective_support_estimate(
        [0.10, 0.02, 0.08, -0.01, 0.06],
        confidence_level=0.95,
        target_power=0.8,
        maximum_market_count=100_000,
    )
    negative = _prospective_support_estimate(
        [-0.10, -0.02, 0.01],
        confidence_level=0.95,
        target_power=0.8,
        maximum_market_count=100_000,
    )

    assert positive["status"] == "estimated"
    assert positive["minimum_market_count_for_expected_positive_lcb"] is not None
    assert positive["minimum_market_count_for_target_power"] is not None
    assert negative["status"] == "not_estimable_from_nonpositive_development_mean"
    assert negative["minimum_market_count_for_target_power"] is None


def _decision_rows(
    *,
    returns: dict[str, float],
    scores: dict[str, float] | None = None,
) -> list[dict]:
    selected_scores = scores or {
        action: 1.0 - index * 0.1 for index, action in enumerate(REQUIRED_ACTIONS)
    }
    rows = []
    for action in REQUIRED_ACTIONS:
        absolute = returns[action]
        best_alternative = max(value for candidate, value in returns.items() if candidate != action)
        if action == "NO_TRADE":
            family = "NO_TRADE"
            side = "NONE"
        else:
            family = (
                "HOLD_TO_SETTLEMENT"
                if action.endswith("HOLD_TO_SETTLEMENT")
                else "SELL_BEFORE_CLOSE"
            )
            side = "UP" if "_UP_" in action else "DOWN"
        rows.append(
            {
                "market_id": "market-1",
                "decision_ts": 1_000,
                "fold_index": 0,
                "action": action,
                "action_family": family,
                "side": side,
                "pairwise_group_normalized_rank_score": selected_scores[action],
                "target_net_pnl_per_contract": absolute,
                "training_target_absolute_post_cost_net_return": absolute,
                "training_target_advantage_vs_no_trade": absolute,
                "training_target_advantage_vs_best_alternative": absolute - best_alternative,
                "training_targets_include_costs": True,
                "training_targets_used_as_decision_inputs": False,
            }
        )
    return rows


def _calibration(
    *,
    absolute_lcb: float,
    no_trade_lcb: float,
    oracle_lcb: float,
) -> dict:
    actions = {}
    groups = {}
    for action in REQUIRED_ACTIONS:
        actions[action] = {
            "adaptive_score_boundaries": [],
            "adaptive_bucket_names": ["bucket_0"],
        }
        if action == "NO_TRADE":
            lcbs = dict.fromkeys(
                (
                    "absolute_post_cost_net_return",
                    "advantage_vs_no_trade",
                    "advantage_vs_best_alternative",
                ),
                0.0,
            )
        else:
            lcbs = {
                "absolute_post_cost_net_return": absolute_lcb,
                "advantage_vs_no_trade": no_trade_lcb,
                "advantage_vs_best_alternative": oracle_lcb,
            }
        groups[f"{action}|bucket_0"] = {
            "action": action,
            "bucket_name": "bucket_0",
            "estimators": {
                estimand: {
                    "point_estimate": lcb + 0.01,
                    "lower_confidence_bound": lcb,
                }
                for estimand, lcb in lcbs.items()
            },
        }
    return {"actions": actions, "calibration_groups": groups}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
