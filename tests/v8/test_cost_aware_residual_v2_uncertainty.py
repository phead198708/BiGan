from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from bigan.v8.polymarket.cost_aware_residual_v2 import (
    DEFAULT_PROTOCOL as PRIMARY_PROTOCOL,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    SELECTED_MID_INDEX,
    _model_parameters,
)
from bigan.v8.polymarket.cost_aware_residual_v2_uncertainty import (
    CHALLENGER_PROTOCOL_SCHEMA_VERSION,
    STRUCTURAL_CHANGE,
    _rolling_origin_uncertainty_predict,
    _uncertainty_model_contract,
    _uncertainty_target_contract,
    uncertainty_adjusted_action_values,
    validate_uncertainty_challenger_protocol,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES


def _protocol() -> dict:
    payload = json.loads(PRIMARY_PROTOCOL.read_text(encoding="utf-8"))
    payload["schema_version"] = CHALLENGER_PROTOCOL_SCHEMA_VERSION
    payload["slot_id"] = "residual-v2-challenger-slot-002"
    payload["candidate_role"] = "challenger"
    payload["candidate_budget"] = {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 2,
        "slots_consumed_before_run": 1,
        "slots_remaining_after_run": 0,
        "slot_budget_may_be_increased": False,
    }
    payload["target"] = _uncertainty_target_contract()
    payload["model"] = _uncertainty_model_contract()
    payload["structural_change"] = dict(STRUCTURAL_CHANGE)
    payload["prior_slot_result"] = {
        "manifest": {"path": "examples/manifest.json", "sha256": "a" * 64},
        "report": {"path": "examples/report.json", "sha256": "b" * 64},
        "failed_gates": ["prospective_power_required_market_count_lte_2000"],
    }
    payload["inputs"]["candidate_implementation"] = {
        "path": "src/bigan/v8/polymarket/cost_aware_residual_v2_uncertainty.py",
        "sha256": "c" * 64,
    }
    return payload


def _row(
    *, market_id: str, decision_ts: int, side: str, anchor: float, outcome: str
) -> dict:
    features = np.zeros(len(FEATURE_NAMES), dtype=float)
    features[SELECTED_MID_INDEX] = anchor
    return {
        "market_id": market_id,
        "market_start_ts": int(market_id[1:]) * 1000,
        "decision_ts": decision_ts,
        "side": side,
        "features": features,
        "resolved_outcome": outcome,
        "target": (
            (1.0 if side == outcome else 0.0) - (anchor + 0.01) - 0.005
        ),
        "cost_decomposition": {
            "entry_ask": anchor + 0.01,
            "entry_bid": anchor,
            "fees": 0.0002,
            "slippage": 0.00475,
            "liquidity_impact": 0.00005,
            "total_cost_excluding_entry_ask": 0.005,
        },
        "feature_row_sha256": f"{decision_ts:064x}",
    }


def test_challenger_is_single_structural_change_with_shared_frozen_gates() -> None:
    protocol = _protocol()
    validate_uncertainty_challenger_protocol(protocol, verify_artifacts=False)

    mutations = []
    changed = deepcopy(protocol)
    changed["action_policy"]["fixed_acceptance_threshold"] = 0.01
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["model"]["head_parameters"]["lower_q25"]["quantile_alpha"] = 0.3
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["candidate_budget"]["slots_remaining_after_run"] = 1
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["structural_change"]["threshold_changed"] = True
    mutations.append(changed)

    for mutation in mutations:
        with pytest.raises(ValueError, match="challenger protocol invalid"):
            validate_uncertainty_challenger_protocol(
                mutation, verify_artifacts=False
            )


def test_half_iqr_penalty_is_deterministic_and_never_negative() -> None:
    means = [
        {
            "action_value": 0.08,
            "predicted_probability": 0.6,
            "market_anchor_probability": 0.55,
            "predicted_probability_residual": 0.05,
            "predicted_probability_before_pair_normalization": 0.6,
            "entry_ask": 0.51,
            "non_entry_cost": 0.01,
        },
        {
            "action_value": -0.08,
            "predicted_probability": 0.4,
            "market_anchor_probability": 0.45,
            "predicted_probability_residual": -0.05,
            "predicted_probability_before_pair_normalization": 0.4,
            "entry_ask": 0.47,
            "non_entry_cost": 0.01,
        },
    ]
    first = uncertainty_adjusted_action_values(means, [-0.1, 0.2], [0.2, 0.1])
    second = uncertainty_adjusted_action_values(means, [-0.1, 0.2], [0.2, 0.1])
    assert first == second
    assert first[0]["conditional_half_IQR_uncertainty_penalty"] == pytest.approx(
        0.15
    )
    assert first[0]["action_value"] == pytest.approx(-0.07)
    assert first[1]["conditional_half_IQR_uncertainty_penalty"] == 0.0
    assert first[1]["action_value"] == -0.08


def test_synthetic_three_head_rolling_origin_has_no_future_label_leakage() -> None:
    population = [f"m{index}" for index in range(4)]
    rows = []
    for market_index, market_id in enumerate(population):
        outcome = "UP" if market_index % 2 == 0 else "DOWN"
        for decision_offset in (1, 2):
            decision_ts = market_index * 1000 + decision_offset
            rows.extend(
                [
                    _row(
                        market_id=market_id,
                        decision_ts=decision_ts,
                        side="UP",
                        anchor=0.45,
                        outcome=outcome,
                    ),
                    _row(
                        market_id=market_id,
                        decision_ts=decision_ts,
                        side="DOWN",
                        anchor=0.55,
                        outcome=outcome,
                    ),
                ]
            )
    model = _uncertainty_model_contract()
    model["fixed_num_boost_round_by_head"] = {
        "mean": 2,
        "lower_q25": 2,
        "upper_q75": 2,
    }
    protocol = {
        "slot_id": "synthetic-uncertainty-slot",
        "rolling_origin": {
            "initial_training_market_count": 2,
            "target_block_size": 2,
            "target_block_count": 1,
        },
        "model": model,
    }
    predictions, audits = _rolling_origin_uncertainty_predict(
        rows=rows,
        population_order=population,
        protocol=protocol,
    )
    assert len(predictions) == 8
    assert len(audits) == 1
    assert audits[0]["strictly_prior_training_market_count"] == 2
    assert audits[0]["target_or_future_label_leakage_count"] == 0
    assert {row["market_id"] for row in predictions} == {"m2", "m3"}
    assert all(
        row["conditional_half_IQR_uncertainty_penalty"] >= 0.0
        for row in predictions
    )
    assert all(
        row["prediction"] <= row["action_value_before_uncertainty_penalty"]
        for row in predictions
    )


def test_uncertainty_head_contract_does_not_change_mean_head() -> None:
    contract = _uncertainty_model_contract()
    assert contract["head_parameters"]["mean"] == _model_parameters()
    assert contract["head_parameters"]["lower_q25"]["quantile_alpha"] == 0.25
    assert contract["head_parameters"]["upper_q75"]["quantile_alpha"] == 0.75
