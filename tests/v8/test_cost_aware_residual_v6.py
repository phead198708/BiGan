from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest

from bigan.v8.polymarket.cost_aware_residual_v6 import (
    _dmatrix,
    nested_dynamic_stopping_predict,
    validate_v6_lineage_authorization,
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
