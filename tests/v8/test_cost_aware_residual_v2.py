from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bigan.v8.polymarket.cost_aware_residual_v2 import (
    GATE_NAMES,
    LINEAGE_ID,
    PAIR_CLIP_EPSILON,
    PROTOCOL_SCHEMA_VERSION,
    SELECTED_MID_INDEX,
    _action_policy,
    _bootstrap_contract,
    _cost_stress_contract,
    _model_parameters,
    _power_contract,
    _rolling_contract,
    _rolling_origin_pair_anchored_predict,
    pair_anchored_action_values,
    validate_residual_v2_protocol,
    validate_v2_lineage_authorization,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY


def _descriptor(name: str) -> dict[str, str]:
    return {"path": f"examples/{name}.json", "sha256": "a" * 64}


def _protocol() -> dict:
    inputs = {
        name: _descriptor(name)
        for name in (
            "lineage_authorization",
            "development_data_registry",
            "parent_v1_terminal_review",
            "terminal_diagnostic_scored_rows",
            "confirmatory_capture_manifest",
            "confirmatory_market_evaluation_rows",
            "baseline_decision_rows",
            "matched_global_baseline_contract",
            "parent_feature_contract",
            "parent_cost_and_action_contract",
            "raw_capture_recovery_bundle_manifest",
            "candidate_implementation",
            "gate_implementation",
        )
    }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": "residual-v2-primary-slot-001",
        "candidate_role": "primary",
        "created_at": "2026-08-09T08:30:00Z",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "candidate_budget": {
            "maximum_total_slots": 2,
            "this_slot_ordinal": 1,
            "slots_consumed_before_run": 0,
            "slots_remaining_after_run": 1,
            "slot_budget_may_be_increased": False,
        },
        "target": {
            "action_value_formula": (
                "pair_normalized_clipped_selected_mid_plus_predicted_probability_"
                "residual-entry_ask-frozen_fees-slippage-liquidity_impact"
            ),
            "execution_policy": "HOLD_TO_SETTLEMENT",
            "NO_TRADE_value": 0.0,
            "post_close_training_label_only": True,
            "regression_label": "settlement_payout-selected_mid",
        },
        "pair_coherence": {
            "anchor": "decision_time_selected_mid",
            "clip_epsilon": PAIR_CLIP_EPSILON,
            "normalization": "UP_DOWN_probabilities_sum_to_one_per_decision",
            "normalization_happens_before_cost_subtraction": True,
            "missing_anchor_behavior": "fail_closed_NO_TRADE_in_runtime",
            "missing_values_encoded_as_zero": False,
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
            "source_contract_reused_without_feature_addition_or_removal": True,
        },
        "model": {
            "family": (
                "pooled_side_symmetric_market_anchored_probability_residual_xgboost"
            ),
            "route_or_expert_allowed": False,
            "fixed_num_boost_round": 128,
            "model_selection_or_early_stopping_performed": False,
            "parameters": _model_parameters(),
        },
        "action_policy": _action_policy(),
        "bootstrap": _bootstrap_contract(),
        "cost_stress": _cost_stress_contract(),
        "gates": dict.fromkeys(GATE_NAMES, True),
        "prospective_power": _power_contract(),
        "rolling_origin": _rolling_contract(),
        "dataset": {
            "market_count": 800,
            "side_decision_row_count": 3200,
            "decision_rows_per_market": 2,
            "sides_per_decision": 2,
            "development_only_forever": True,
            "population_order": "frozen_confirmatory_capture_manifest_order",
        },
        "development_discipline": {
            "one_candidate_this_slot": True,
            "hyperparameter_search_allowed": False,
            "threshold_search_allowed": False,
            "route_side_missingness_or_outlier_filtering_allowed": False,
            "post_result_mutation_allowed": False,
            "challenger_requires_separate_preregistration": True,
        },
        "state": {
            "training_started": False,
            "candidate_frozen": False,
            "live_shadow_started": False,
            "fresh_confirmatory_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
        "inputs": inputs,
    }


def _row(side: str, selected_mid: float) -> dict:
    features = np.zeros(len(FEATURE_NAMES), dtype=float)
    features[SELECTED_MID_INDEX] = selected_mid
    return {
        "market_id": "m1",
        "decision_ts": 123,
        "side": side,
        "features": features,
        "cost_decomposition": {
            "entry_ask": selected_mid + 0.01,
            "total_cost_excluding_entry_ask": 0.005,
        },
    }


def test_new_lineage_authorization_binds_user_instruction_and_parent_v1() -> None:
    result = validate_v2_lineage_authorization()
    assert result["authorization_valid"] is True
    assert result["lineage_id"] == LINEAGE_ID
    assert result["maximum_total_slots"] == 2
    assert result["parent_v1_immutable"] is True
    assert all(value is False for value in result["safety"].values())


def test_v2_protocol_keeps_every_v1_gate_threshold_and_safety_flag() -> None:
    protocol = _protocol()
    validate_residual_v2_protocol(protocol, verify_artifacts=False)

    mutations = []
    changed = deepcopy(protocol)
    changed["action_policy"]["fixed_acceptance_threshold"] = 0.01
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["gates"].pop("prospective_power_required_market_count_lte_2000")
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["candidate_budget"]["maximum_total_slots"] = 3
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["safety"]["paper_candidate_allowed"] = True
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["pair_coherence"]["missing_values_encoded_as_zero"] = True
    mutations.append(changed)

    for mutation in mutations:
        with pytest.raises(ValueError, match="residual v2 protocol invalid"):
            validate_residual_v2_protocol(mutation, verify_artifacts=False)


def test_pair_anchored_scores_are_deterministic_coherent_and_cost_aware() -> None:
    rows = [_row("UP", 0.4), _row("DOWN", 0.6)]
    first = pair_anchored_action_values(rows, [0.1, -0.2])
    second = pair_anchored_action_values(rows, [0.1, -0.2])
    assert first == second
    assert sum(row["predicted_probability"] for row in first) == pytest.approx(1.0)
    assert first[0]["predicted_probability"] == pytest.approx(5.0 / 9.0)
    assert first[1]["predicted_probability"] == pytest.approx(4.0 / 9.0)
    assert first[0]["action_value"] == pytest.approx(5.0 / 9.0 - 0.415)
    assert first[1]["action_value"] == pytest.approx(4.0 / 9.0 - 0.615)


def test_missing_anchor_fails_closed_without_zero_imputation() -> None:
    rows = [_row("UP", 0.4), _row("DOWN", 0.6)]
    rows[0]["features"][SELECTED_MID_INDEX] = np.nan
    with pytest.raises(ValueError, match="anchor is missing"):
        pair_anchored_action_values(rows, [0.1, -0.2])


def test_pair_contract_rejects_incomplete_market_pair() -> None:
    with pytest.raises(ValueError, match="UP/DOWN pair is incomplete"):
        pair_anchored_action_values([_row("UP", 0.4)], [0.1])


def test_synthetic_rolling_origin_uses_only_strictly_prior_markets() -> None:
    rows = []
    population = [f"m{index}" for index in range(4)]
    for market_index, market_id in enumerate(population):
        outcome = "UP" if market_index % 2 == 0 else "DOWN"
        for decision_offset in (1, 2):
            for side, anchor in (("UP", 0.45), ("DOWN", 0.55)):
                row = _row(side, anchor)
                row.update(
                    {
                        "market_id": market_id,
                        "market_start_ts": market_index * 1000,
                        "decision_ts": market_index * 1000 + decision_offset,
                        "resolved_outcome": outcome,
                        "target": (
                            (1.0 if side == outcome else 0.0)
                            - (anchor + 0.01)
                            - 0.005
                        ),
                        "feature_row_sha256": f"{market_index + decision_offset:064x}",
                    }
                )
                rows.append(row)
    protocol = {
        "slot_id": "synthetic-v2-slot",
        "rolling_origin": {
            "initial_training_market_count": 2,
            "target_block_size": 2,
            "target_block_count": 1,
        },
        "model": {
            "parameters": _model_parameters(),
            "fixed_num_boost_round": 2,
        },
    }
    predictions, audits = _rolling_origin_pair_anchored_predict(
        rows=rows,
        population_order=population,
        protocol=protocol,
    )
    assert len(predictions) == 8
    assert len(audits) == 1
    assert audits[0]["strictly_prior_training_market_count"] == 2
    assert audits[0]["target_or_future_label_leakage_count"] == 0
    assert {row["market_id"] for row in predictions} == {"m2", "m3"}
    for market_id in ("m2", "m3"):
        for decision_offset in (1, 2):
            pair = [
                row
                for row in predictions
                if row["market_id"] == market_id
                and row["decision_ts"]
                == int(market_id[1:]) * 1000 + decision_offset
            ]
            assert sum(row["predicted_probability"] for row in pair) == pytest.approx(
                1.0
            )
