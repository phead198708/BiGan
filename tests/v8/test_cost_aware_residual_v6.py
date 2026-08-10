from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bigan.v8.polymarket.cost_aware_residual import _load_json
from bigan.v8.polymarket.cost_aware_residual_v6 import (
    DEFAULT_PROTOCOL,
    FIXED_NUM_BOOST_ROUND,
    FIXED_PARAMETERS,
    _dmatrix,
    nested_dynamic_stopping_predict,
    validate_residual_v6_protocol,
    validate_v6_lineage_authorization,
    verify_frozen_residual_v6_oof,
)
from bigan.v8.polymarket.cost_aware_residual_v6_challenger import (
    DEFAULT_PROTOCOL as CHALLENGER_PROTOCOL,
)
from bigan.v8.polymarket.cost_aware_residual_v6_challenger import (
    FIXED_QUANTILE_ALPHA,
    _quality_features,
    prequential_lower_quantile_proposal_predict,
    validate_challenger_protocol,
    verify_frozen_challenger_oof,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY


def _fixture() -> tuple[list[dict], list[str], dict]:
    population = [f"market-{index:02d}" for index in range(8)]
    rows: list[dict] = []
    for position, market_id in enumerate(population, start=1):
        outcome = "UP" if position % 2 else "DOWN"
        for decision_offset in (300, 600):
            for side_index, side in enumerate(("UP", "DOWN")):
                features = np.asarray(
                    [
                        float((position + side_index + feature_index) % 7)
                        for feature_index in range(len(FEATURE_NAMES))
                    ],
                    dtype=float,
                )
                if position % 2:
                    features[0] = np.nan
                rows.append(
                    {
                        "market_id": market_id,
                        "market_position": position,
                        "market_start_ts": 1_700_000_000 + position * 900,
                        "decision_ts": 1_700_000_000 + position * 900 + decision_offset,
                        "side": side,
                        "features": features,
                        "feature_row_sha256": f"{position:02d}-{decision_offset}-{side}",
                        "target": 0.49 if side == outcome else -0.51,
                        "resolved_outcome": outcome,
                        "cost_decomposition": {
                            "entry_ask": 0.5,
                            "entry_bid": 0.49,
                            "fees": 0.001,
                            "slippage": 0.004,
                            "liquidity_impact": 0.0,
                            "total_cost_excluding_entry_ask": 0.005,
                        },
                    }
                )
    protocol = {
        "slot_id": "residual-v6-primary-slot-001",
        "rolling_origin": {
            "initial_training_market_count": 4,
            "target_block_size": 2,
            "target_block_count": 2,
        },
        "sequential_training": {
            "inner_initial_training_market_count": 2,
            "inner_target_block_market_count": 2,
        },
        "model": {
            "fixed_num_boost_round": 8,
            "parameters": {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "eta": 0.05,
                "max_depth": 1,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "seed": 26461,
                "nthread": 1,
            },
        },
    }
    return rows, population, protocol


def test_v6_authorization_is_exact_and_parent_v5_remains_immutable() -> None:
    result = validate_v6_lineage_authorization()

    assert result["authorization_valid"] is True
    assert result["maximum_total_slots"] == 2
    assert result["parent_v5_immutable"] is True
    assert result["narrow_main_integration_preparation_authorized"] is True
    assert result["fresh_collection_or_execution_authorized"] is False
    assert result["safety"] == SAFETY


def test_v6_slot_1_protocol_binds_exact_executor_before_oof() -> None:
    protocol_path = Path(DEFAULT_PROTOCOL)
    protocol = _load_json(protocol_path)

    validate_residual_v6_protocol(protocol)

    assert protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").strip() == (
        "a8c259fc259277567821894060aeac6e5ad67ebbd9d2ebcd37589d4529f24c76"
    )
    assert protocol["inputs"]["candidate_implementation"]["sha256"] == (
        "299befa843d8b6feee9fce7d0b51a5a4d83de722f45b9875fa1de0bb9910374b"
    )
    assert protocol["model"]["fixed_num_boost_round"] == FIXED_NUM_BOOST_ROUND
    assert protocol["model"]["parameters"] == FIXED_PARAMETERS
    assert protocol["action_policy"]["fixed_acceptance_threshold"] == 0.0
    assert protocol["prospective_power"]["maximum_market_count"] == 2000


def test_nested_dynamic_stopping_is_deterministic_and_strictly_prior() -> None:
    rows, population, protocol = _fixture()

    first = nested_dynamic_stopping_predict(
        rows=rows, population_order=population, protocol=protocol
    )
    second = nested_dynamic_stopping_predict(
        rows=deepcopy(rows), population_order=list(population), protocol=deepcopy(protocol)
    )

    assert first == second
    predictions, audits = first
    assert len(predictions) == 16
    assert len(audits) == 2
    assert [row["strictly_prior_training_market_count"] for row in audits] == [4, 6]
    assert [row["nested_continuation_training_market_count"] for row in audits] == [2, 4]
    assert all(row["target_or_future_label_used_for_fit"] is False for row in predictions)
    assert all(
        row["outer_target_late_feature_used_for_early_score"] is False for row in predictions
    )
    assert {row["score_semantics"] for row in predictions} == {
        "early_incremental_over_continuation",
        "late_direct_after_cost",
    }


def test_outer_target_late_features_cannot_change_early_scores() -> None:
    rows, population, protocol = _fixture()
    original, _ = nested_dynamic_stopping_predict(
        rows=rows, population_order=population, protocol=protocol
    )
    changed = deepcopy(rows)
    # Mutate only the final outer target block. Earlier target blocks become
    # legitimate prior training data for later folds.
    target_ids = set(population[6:])
    late_ts_by_market = {
        market_id: max(
            int(row["decision_ts"])
            for row in changed
            if row["market_id"] == market_id
        )
        for market_id in target_ids
    }
    for row in changed:
        if row["market_id"] in target_ids and int(row["decision_ts"]) == late_ts_by_market[
            row["market_id"]
        ]:
            row["features"] = np.full(len(FEATURE_NAMES), 1_000_000.0)
    modified, _ = nested_dynamic_stopping_predict(
        rows=changed, population_order=population, protocol=protocol
    )

    original_early = {
        (row["market_id"], row["side"]): row["prediction"]
        for row in original
        if row["score_semantics"] == "early_incremental_over_continuation"
    }
    modified_early = {
        (row["market_id"], row["side"]): row["prediction"]
        for row in modified
        if row["score_semantics"] == "early_incremental_over_continuation"
    }
    assert original_early == modified_early


def test_native_missing_value_is_preserved_as_nan() -> None:
    rows, _, _ = _fixture()

    matrix = _dmatrix([rows[0]], labels=[float(rows[0]["target"])])

    assert math.isnan(float(rows[0]["features"][0]))
    assert matrix.num_col() == len(FEATURE_NAMES)


def test_missing_market_action_fails_closed() -> None:
    rows, population, protocol = _fixture()
    rows.pop()

    with pytest.raises(ValueError, match="action grid changed"):
        nested_dynamic_stopping_predict(
            rows=rows, population_order=population, protocol=protocol
        )


def test_frozen_v6_slot_1_reconciles_and_fails_closed() -> None:
    result = verify_frozen_residual_v6_oof()

    assert result["verification_passed"] is True
    assert result["all_gates_passed"] is False
    assert result["failed_gates"] == [
        "absolute_market_bootstrap_97_5pct_lcb_gt_zero",
        "paired_delta_market_bootstrap_97_5pct_lcb_gt_zero",
        "every_chronological_block_candidate_total_gte_zero",
        "every_chronological_block_paired_delta_total_gte_zero",
        "stable_score_to_realized_pnl_ordering",
        "prospective_power_required_market_count_lte_2000",
    ]
    assert result["remaining_candidate_slots"] == 1
    assert result["oof_market_count"] == 600
    assert result["manifest_sha256"] == (
        "8cbd0d71ffc39c3a177381f498d6127cbac7f6257cb00b1fff4d6c6f9b7c6009"
    )
    assert result["actual_executing_module_binding_verified"] is True
    assert result["parent_v1_through_v5_immutable"] is True
    assert result["safety"] == SAFETY


def _challenger_fixture() -> tuple[list[dict], list[dict], list[dict], list[str], dict]:
    population = [f"proposal-market-{index:02d}" for index in range(8)]
    rows: list[dict] = []
    base: list[dict] = []
    results: list[dict] = []
    for position, market_id in enumerate(population, start=1):
        outcome = "UP" if position % 3 else "DOWN"
        market_start = 1_800_000_000_000 + position * 900_000
        for decision_offset in (300_000, 600_000):
            for side_index, side in enumerate(("UP", "DOWN")):
                features = {
                    name: (
                        None
                        if feature_index == 0 and position % 2
                        else float((position + side_index + feature_index) % 11)
                    )
                    for feature_index, name in enumerate(FEATURE_NAMES)
                }
                row = {
                    "market_id": market_id,
                    "market_position": position,
                    "market_start_ts": market_start,
                    "decision_ts": market_start + decision_offset,
                    "side": side,
                    "features": features,
                    "feature_row_sha256": f"{position}-{decision_offset}-{side}",
                    "target": 0.49 if side == outcome else -0.51,
                    "resolved_outcome": outcome,
                    "cost_decomposition": {
                        "entry_ask": 0.5,
                        "entry_bid": 0.49,
                        "fees": 0.001,
                        "slippage": 0.004,
                        "liquidity_impact": 0.0,
                        "total_cost_excluding_entry_ask": 0.005,
                    },
                }
                rows.append(row)
                if position > 2:
                    selected = decision_offset == 300_000 and side == "UP"
                    base.append(
                        {
                            "market_id": market_id,
                            "market_start_ts": market_start,
                            "decision_ts": row["decision_ts"],
                            "side": side,
                            "prediction": 0.1 if selected else -0.1,
                            "predicted_probability": 0.605 if selected else 0.395,
                            "realized_unit_net_pnl_if_action": row["target"],
                            "resolved_outcome": outcome,
                            "cost_decomposition": row["cost_decomposition"],
                            "feature_row_sha256": row["feature_row_sha256"],
                        }
                    )
        if position > 2:
            results.append(
                {
                    "market_id": market_id,
                    "candidate_accepted": True,
                    "candidate_decision_ts": market_start + 300_000,
                    "candidate_selected_side": "UP",
                    "candidate_unit_net_pnl": 0.49 if outcome == "UP" else -0.51,
                }
            )
    protocol = {
        "slot_id": "residual-v6-challenger-slot-002",
        "rolling_origin": {
            "initial_training_market_count": 2,
            "target_block_size": 2,
            "target_block_count": 3,
        },
        "model": {
            "fixed_num_boost_round": 8,
            "parameters": {
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.4,
                "eval_metric": "quantile",
                "eta": 0.05,
                "max_depth": 1,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "seed": 26462,
                "nthread": 1,
            },
        },
    }
    return rows, base, results, population, protocol


def test_prequential_lower_quantile_proposal_is_deterministic_and_prior_only() -> None:
    rows, base, results, population, protocol = _challenger_fixture()

    first = prequential_lower_quantile_proposal_predict(
        rows=rows,
        frozen_v5_predictions=base,
        frozen_v5_market_results=results,
        population_order=population,
        protocol=protocol,
    )
    second = prequential_lower_quantile_proposal_predict(
        rows=deepcopy(rows),
        frozen_v5_predictions=deepcopy(base),
        frozen_v5_market_results=deepcopy(results),
        population_order=list(population),
        protocol=deepcopy(protocol),
    )

    assert first == second
    predictions, audits = first
    assert len(predictions) == 24
    assert len(audits) == 3
    assert [row["meta_model_applied"] for row in audits] == [False, True, True]
    assert [row["strictly_prior_meta_training_market_count"] for row in audits] == [
        0,
        2,
        4,
    ]
    assert all(row["target_or_future_label_used_for_fit"] is False for row in predictions)
    assert all(
        row["prediction"] == 0.0
        for row in predictions
        if row["frozen_v5_proposal_selected"] is False
    )
    first_block_selected = [
        row
        for row in predictions
        if row["chronological_block"] == 1 and row["frozen_v5_proposal_selected"]
    ]
    assert [row["prediction"] for row in first_block_selected] == [0.1, 0.1]


def test_challenger_target_labels_cannot_change_same_block_scores() -> None:
    rows, base, results, population, protocol = _challenger_fixture()
    original, _ = prequential_lower_quantile_proposal_predict(
        rows=rows,
        frozen_v5_predictions=base,
        frozen_v5_market_results=results,
        population_order=population,
        protocol=protocol,
    )
    changed_base = deepcopy(base)
    changed_results = deepcopy(results)
    last_block = set(population[-2:])
    for row in changed_base:
        if row["market_id"] in last_block:
            row["realized_unit_net_pnl_if_action"] *= -1.0
    for row in changed_results:
        if row["market_id"] in last_block:
            row["candidate_unit_net_pnl"] *= -1.0
    modified, _ = prequential_lower_quantile_proposal_predict(
        rows=deepcopy(rows),
        frozen_v5_predictions=changed_base,
        frozen_v5_market_results=changed_results,
        population_order=population,
        protocol=protocol,
    )

    original_scores = [
        row["prediction"] for row in original if row["market_id"] in last_block
    ]
    modified_scores = [
        row["prediction"] for row in modified if row["market_id"] in last_block
    ]
    assert original_scores == modified_scores


def test_challenger_quality_features_keep_native_nan() -> None:
    rows, base, _, population, _ = _challenger_fixture()
    row = next(
        row
        for row in rows
        if row["market_id"] == population[2]
        and row["decision_ts"] == 1_800_000_000_000 + 3 * 900_000 + 300_000
        and row["side"] == "UP"
    )
    frozen = next(item for item in base if item["feature_row_sha256"] == row["feature_row_sha256"])

    values = _quality_features(row, frozen)

    assert math.isnan(float(values[0]))
    assert values.shape == (112,)


def test_challenger_missing_frozen_action_fails_closed() -> None:
    rows, base, results, population, protocol = _challenger_fixture()
    base.pop()

    with pytest.raises(ValueError, match="proposal population mismatch|action grid changed"):
        prequential_lower_quantile_proposal_predict(
            rows=rows,
            frozen_v5_predictions=base,
            frozen_v5_market_results=results,
            population_order=population,
            protocol=protocol,
        )


def test_v6_final_slot_protocol_binds_exact_lower_quantile_executor() -> None:
    protocol_path = Path(CHALLENGER_PROTOCOL)
    protocol = _load_json(protocol_path)

    validate_challenger_protocol(protocol)

    assert protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").strip() == (
        "23e2332c44331412996861c5366022f41cebd2510cfa6a8bfd65a903d5aec47b"
    )
    assert protocol["inputs"]["candidate_implementation"]["sha256"] == (
        "18cdef6627e456f47f7e59969798c206118569b3f62557d5d7685997fbbd1f53"
    )
    assert protocol["model"]["fixed_quantile_alpha"] == FIXED_QUANTILE_ALPHA == 0.4
    assert protocol["action_policy"]["fixed_acceptance_threshold"] == 0.0
    assert protocol["prospective_power"]["maximum_market_count"] == 2000
    assert protocol["candidate_budget"]["slots_remaining_after_run"] == 0


def test_frozen_v6_final_slot_reconciles_and_exhausts_budget() -> None:
    result = verify_frozen_challenger_oof()

    assert result["verification_passed"] is True
    assert result["all_gates_passed"] is False
    assert result["failed_gates"] == [
        "every_chronological_block_candidate_total_gte_zero",
        "prospective_power_required_market_count_lte_2000",
    ]
    assert result["candidate_budget_exhausted"] is True
    assert result["oof_market_count"] == 600
    assert result["manifest_sha256"] == (
        "d21bc40023e5a6e0d6c64f92bdfb82f8d84251a8ac948f091e6ef25597e89348"
    )
    assert result["actual_executing_module_binding_verified"] is True
    assert result["parent_v1_through_v5_and_primary_slot_immutable"] is True
    assert result["safety"] == SAFETY
