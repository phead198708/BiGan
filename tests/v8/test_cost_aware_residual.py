from __future__ import annotations

from copy import deepcopy

import pytest

from bigan.v8.polymarket.cost_aware_residual import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROTOCOL,
    FOLD_SCHEMA_VERSION,
    MARKET_RESULT_SCHEMA_VERSION,
    PROTOCOL_SCHEMA_VERSION,
    build_residual_oof_report,
    market_results_from_predictions,
    validate_residual_oof_protocol,
    verify_frozen_residual_oof,
)
from bigan.v8.polymarket.cost_aware_residual_quantile import (
    CHALLENGER_PROTOCOL_SCHEMA_VERSION,
    validate_quantile_challenger_protocol,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY


def _protocol() -> dict:
    inputs = {
        name: {"path": f"examples/{name}.json", "sha256": "a" * 64}
        for name in (
            "lineage_authorization",
            "development_data_registry",
            "raw_capture_recovery_bundle_manifest",
            "terminal_diagnostic_scored_rows",
            "confirmatory_capture_manifest",
            "confirmatory_market_evaluation_rows",
            "baseline_decision_rows",
            "matched_global_baseline_contract",
            "parent_feature_contract",
            "parent_cost_and_action_contract",
            "implementation",
        )
    }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "lineage_id": "BTC-15M-cost-aware-market-residual-v1",
        "slot_id": "residual-primary-slot-001",
        "candidate_role": "primary",
        "created_at": "2026-08-09T07:00:00Z",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "candidate_budget": {
            "maximum_total_slots": 2,
            "this_slot_ordinal": 1,
            "slots_consumed_before_run": 0,
            "slot_budget_may_be_increased": False,
        },
        "dataset": {"market_count": 800, "side_decision_row_count": 3200},
        "target": {
            "name": "direct_after_cost_action_value",
            "formula": ("settlement_payout-executable_ask-frozen_fees-slippage-liquidity_impact"),
            "execution_policy": "HOLD_TO_SETTLEMENT",
            "NO_TRADE_value": 0.0,
            "post_close_only": True,
        },
        "feature_contract": {
            "ordered_feature_count": 108,
            "base_feature_count": 54,
            "shared_side_symmetric_model": True,
            "side_identity_feature_allowed": False,
            "native_missing_value": "nan",
            "missing_values_encoded_as_zero": False,
            "feature_search_allowed": False,
            "market_horizon_seconds": 900,
        },
        "model": {
            "family": "pooled_global_xgboost_direct_regressor",
            "route_or_expert_allowed": False,
            "fixed_num_boost_round": 128,
            "parameters": {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "nthread": 1,
            },
        },
        "rolling_origin": {
            "market_order": "frozen_capture_order",
            "initial_training_market_count": 200,
            "target_block_size": 100,
            "target_block_count": 6,
            "oof_market_count": 600,
            "strictly_prior_market_labels_only": True,
            "market_grouped": True,
            "future_market_used_for_fit": False,
        },
        "action_policy": {
            "decision_order": "chronological",
            "accept_if": "highest_side_prediction>0",
            "fixed_acceptance_threshold": 0.0,
            "side_tie_break_order": ["UP", "DOWN"],
            "one_trade_maximum_per_market": True,
            "NO_TRADE_if_no_positive_prediction": True,
            "NO_TRADE_unit_pnl": 0.0,
            "threshold_search_allowed": False,
        },
        "bootstrap": {
            "method": "market_level_paired_percentile_bootstrap",
            "confidence": 0.975,
            "lower_quantile": 0.025,
            "resamples": 1000,
            "seed": 26401,
            "candidate_and_baseline_share_indices": True,
            "NO_TRADE_participates_as_zero": True,
        },
        "cost_stress": {
            "multipliers": [1.2, 1.5, 2.0],
            "action_selection_reused_from_base_cost": True,
            "formula": "gross_price_edge-multiplier*total_cost_relative_to_mid",
        },
        "gates": dict.fromkeys(
            (
                "absolute_market_bootstrap_97_5pct_lcb_gt_zero",
                "paired_delta_market_bootstrap_97_5pct_lcb_gt_zero",
                "every_chronological_block_candidate_total_gte_zero",
                "every_chronological_block_paired_delta_total_gte_zero",
                "largest_winner_removed_candidate_total_gte_zero",
                "largest_positive_delta_removed_total_gte_zero",
                "stable_score_to_realized_pnl_ordering",
                "all_cost_stress_candidate_totals_gte_zero",
                "all_cost_stress_paired_delta_totals_gte_zero",
                "prospective_power_required_market_count_lte_2000",
                "population_and_leakage_reconciliation",
            ),
            True,
        ),
        "prospective_power": {
            "confidence": 0.975,
            "target_power": 0.8,
            "effect_haircut": 0.5,
            "maximum_market_count": 2000,
            "required_n_rule": "max_absolute_and_paired_plugin_normal_approximation",
        },
        "development_discipline": {
            "one_candidate_this_slot": True,
            "hyperparameter_search_allowed": False,
            "threshold_search_allowed": False,
            "route_side_missingness_or_outlier_filtering_allowed": False,
            "post_result_mutation_allowed": False,
            "challenger_requires_separate_preregistration": True,
        },
        "inputs": inputs,
        "state": {
            "training_started": False,
            "candidate_frozen": False,
            "live_shadow_started": False,
            "fresh_confirmatory_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }


def test_protocol_freezes_direct_regression_and_two_slot_budget() -> None:
    protocol = _protocol()
    validate_residual_oof_protocol(protocol, verify_artifacts=False)

    changed = deepcopy(protocol)
    changed["candidate_budget"]["maximum_total_slots"] = 3
    with pytest.raises(ValueError, match="candidate_budget"):
        validate_residual_oof_protocol(changed, verify_artifacts=False)

    changed = deepcopy(protocol)
    changed["action_policy"]["fixed_acceptance_threshold"] = 0.01
    with pytest.raises(ValueError, match="action_policy"):
        validate_residual_oof_protocol(changed, verify_artifacts=False)


def test_market_policy_uses_first_positive_decision_and_preserves_no_trade() -> None:
    predictions = []
    for market_id, values in {
        "m1": ((-0.1, -0.2), (0.3, 0.1)),
        "m2": ((-0.1, -0.2), (-0.3, -0.4)),
    }.items():
        for decision_index, pair in enumerate(values):
            for side, prediction in zip(("UP", "DOWN"), pair, strict=True):
                predictions.append(
                    {
                        "market_id": market_id,
                        "market_start_ts": 1000 if market_id == "m1" else 2000,
                        "decision_ts": 10 + decision_index,
                        "side": side,
                        "prediction": prediction,
                        "realized_unit_net_pnl_if_action": 0.2 if side == "UP" else -0.2,
                        "cost_decomposition": {
                            "entry_ask": 0.5,
                            "entry_bid": 0.48,
                            "fees": 0.0002,
                            "slippage": 0.01,
                            "liquidity_impact": 0.00005,
                        },
                    }
                )
    baseline = {
        market_id: {
            "baseline_accepted": False,
            "baseline_selected_side": None,
            "baseline_unit_net_pnl": 0.0,
            "decision_ts": 10,
            "cost_decomposition": {"baseline": {"total_cost": 0.0}},
        }
        for market_id in ("m1", "m2")
    }
    results = market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=baseline,
        population_order=["seed", "m1", "m2"],
        initial_training_market_count=1,
        target_block_size=1,
    )
    assert results[0]["candidate_selected_side"] == "UP"
    assert results[0]["candidate_decision_ts"] == 11
    assert results[1]["candidate_accepted"] is False
    assert results[1]["candidate_unit_net_pnl"] == 0.0
    assert results[1]["NO_TRADE_participates_as_zero"] is True


def test_report_uses_shared_market_bootstrap_and_all_frozen_gates() -> None:
    protocol = _protocol()
    rows = []
    for index in range(600):
        pnl = 0.08 + index / 60000
        rows.append(
            {
                "schema_version": MARKET_RESULT_SCHEMA_VERSION,
                "lineage_id": protocol["lineage_id"],
                "market_id": f"m{index:03d}",
                "market_start_ts": index,
                "oof_position": index + 1,
                "chronological_block": index // 100 + 1,
                "chronological_half": "first" if index < 300 else "second",
                "candidate_accepted": True,
                "candidate_selected_side": "UP",
                "candidate_decision_ts": index,
                "candidate_prediction": pnl,
                "candidate_unit_net_pnl": pnl,
                "candidate_total_cost_relative_to_mid": 0.01,
                "baseline_accepted": True,
                "baseline_selected_side": "UP",
                "baseline_decision_ts": index,
                "baseline_unit_net_pnl": 0.01,
                "baseline_total_cost_relative_to_mid": 0.01,
                "paired_delta_unit_net_pnl": pnl - 0.01,
                "NO_TRADE_participates_as_zero": True,
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    folds = []
    for ordinal in range(1, 7):
        training_count = 200 + (ordinal - 1) * 100
        folds.append(
            {
                "schema_version": FOLD_SCHEMA_VERSION,
                "chronological_block": ordinal,
                "strictly_prior_training_market_count": training_count,
                "target_market_count": 100,
                "last_training_market_position": training_count,
                "first_target_market_position": training_count + 1,
                "target_or_future_label_leakage_count": 0,
                "safety": dict(SAFETY),
            }
        )
    first = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256="b" * 64,
        source_commit="c" * 40,
        market_results=rows,
        fold_audits=folds,
    )
    second = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256="b" * 64,
        source_commit="c" * 40,
        market_results=rows,
        fold_audits=folds,
    )
    assert first == second
    assert first["all_gates_passed"] is True
    assert all(first["gate_results"].values())
    assert first["overall"]["shared_bootstrap_indices_sha256"]
    assert first["population"]["candidate_market_count"] == 600
    assert first["population"]["baseline_market_count"] == 600
    assert first["population"]["paired_market_count"] == 600


def test_frozen_primary_oof_verifies_and_remains_fail_closed() -> None:
    result = verify_frozen_residual_oof(
        protocol_path=DEFAULT_PROTOCOL,
        output_dir=DEFAULT_OUTPUT_DIR,
    )
    assert result["verification_passed"] is True
    assert result["all_gates_passed"] is False
    assert result["failed_gates"] == [
        "every_chronological_block_paired_delta_total_gte_zero",
        "prospective_power_required_market_count_lte_2000",
    ]
    assert all(value is False for value in result["safety"].values())


def test_only_remaining_slot_is_structurally_distinct_lower_quantile() -> None:
    protocol = _protocol()
    protocol["schema_version"] = CHALLENGER_PROTOCOL_SCHEMA_VERSION
    protocol["candidate_role"] = "challenger"
    protocol["candidate_budget"] = {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 2,
        "slots_consumed_before_run": 1,
        "slots_remaining_after_run": 0,
        "slot_budget_may_be_increased": False,
    }
    protocol["model"]["family"] = "pooled_global_xgboost_direct_lower_quantile_regressor"
    protocol["model"]["parameters"].update(
        {
            "objective": "reg:quantileerror",
            "eval_metric": "quantile",
            "quantile_alpha": 0.35,
        }
    )
    protocol["prior_slot_result"] = {
        "manifest": {"path": "examples/primary-manifest.json", "sha256": "b" * 64},
        "report": {"path": "examples/primary-report.json", "sha256": "c" * 64},
        "failed_gates": [
            "every_chronological_block_paired_delta_total_gte_zero",
            "prospective_power_required_market_count_lte_2000",
        ],
    }
    protocol["structural_change"] = {
        "changed_component": "fixed_training_loss_only",
        "from": "conditional_mean_squared_error",
        "to": "lower_conditional_quantile_loss_alpha_0_35",
        "reason": (
            "primary accepted 570_of_600 markets and exceeded the N_max_2000 "
            "variance budget despite positive absolute and paired LCBs"
        ),
        "expected_mechanism": (
            "a positive lower conditional action-value quantile abstains unless the "
            "after-cost edge is robust across the lower tail"
        ),
        "threshold_changed": False,
        "feature_set_changed": False,
        "rolling_population_changed": False,
    }
    validate_quantile_challenger_protocol(protocol, verify_artifacts=False)

    changed = deepcopy(protocol)
    changed["model"]["parameters"]["quantile_alpha"] = 0.4
    with pytest.raises(ValueError, match="model"):
        validate_quantile_challenger_protocol(changed, verify_artifacts=False)
