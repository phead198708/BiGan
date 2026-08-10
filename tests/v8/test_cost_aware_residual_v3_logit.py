from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from bigan.v8.polymarket.cost_aware_residual_v3 import (
    DEFAULT_PROTOCOL as PRIMARY_PROTOCOL,
)
from bigan.v8.polymarket.cost_aware_residual_v3 import SELECTED_MID_INDEX
from bigan.v8.polymarket.cost_aware_residual_v3_logit import (
    CHALLENGER_PROTOCOL_SCHEMA_VERSION,
    STRUCTURAL_CHANGE,
    _equal_weight_contract,
    _logit_feature_contract,
    _logit_model_contract,
    _logit_model_parameters,
    _logit_target_contract,
    _rolling_origin_logit_predict,
    logit_offset_action_values,
    validate_logit_challenger_protocol,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES


def _protocol() -> dict:
    payload = json.loads(PRIMARY_PROTOCOL.read_text(encoding="utf-8"))
    payload["schema_version"] = CHALLENGER_PROTOCOL_SCHEMA_VERSION
    payload["slot_id"] = "residual-v3-challenger-slot-002"
    payload["candidate_role"] = "challenger"
    payload["candidate_budget"] = {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 2,
        "slots_consumed_before_run": 1,
        "slots_remaining_after_run": 0,
        "slot_budget_may_be_increased": False,
    }
    payload["target"] = _logit_target_contract()
    payload["feature_contract"] = _logit_feature_contract()
    payload["temporal_adaptation"] = _equal_weight_contract()
    payload["model"] = _logit_model_contract()
    payload["structural_change"] = dict(STRUCTURAL_CHANGE)
    payload["prior_slot_result"] = {
        "manifest": {"path": "examples/manifest.json", "sha256": "a" * 64},
        "report": {"path": "examples/report.json", "sha256": "b" * 64},
        "failed_gates": [
            "every_chronological_block_candidate_total_gte_zero",
            "every_chronological_block_paired_delta_total_gte_zero",
            "prospective_power_required_market_count_lte_2000",
        ],
    }
    payload["inputs"]["candidate_implementation"] = {
        "path": "src/bigan/v8/polymarket/cost_aware_residual_v3_logit.py",
        "sha256": "c" * 64,
    }
    return payload


def _row(
    *, market_id: str, decision_ts: int, side: str, anchor: float, outcome: str
) -> dict:
    features = np.full(len(FEATURE_NAMES), 0.1, dtype=float)
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


def test_logit_challenger_is_one_structural_change_with_old_gates_frozen() -> None:
    protocol = _protocol()
    validate_logit_challenger_protocol(protocol, verify_artifacts=False)

    mutations = []
    changed = deepcopy(protocol)
    changed["action_policy"]["fixed_acceptance_threshold"] = 0.01
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["prospective_power"]["maximum_market_count"] = 2001
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["model"]["parameters"]["max_depth"] = 3
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["feature_contract"]["ordered_feature_count"] = 107
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["candidate_budget"]["slots_remaining_after_run"] = 1
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["structural_change"]["threshold_changed"] = True
    mutations.append(changed)

    for mutation in mutations:
        with pytest.raises(ValueError, match="logit challenger protocol invalid"):
            validate_logit_challenger_protocol(mutation, verify_artifacts=False)


def test_logit_probabilities_are_pair_coherent_cost_aware_and_deterministic() -> None:
    rows = [
        _row(market_id="m1", decision_ts=1, side="UP", anchor=0.4, outcome="UP"),
        _row(
            market_id="m1", decision_ts=1, side="DOWN", anchor=0.6, outcome="UP"
        ),
    ]
    first = logit_offset_action_values(rows, [0.6, 0.5])
    second = logit_offset_action_values(rows, [0.6, 0.5])
    assert first == second
    assert sum(row["predicted_probability"] for row in first) == pytest.approx(1.0)
    assert first[0]["predicted_probability"] == pytest.approx(6.0 / 11.0)
    assert first[0]["action_value"] == pytest.approx(6.0 / 11.0 - 0.415)
    assert first[1]["action_value"] == pytest.approx(5.0 / 11.0 - 0.615)


def test_logit_missing_anchor_fails_closed_without_zero_imputation() -> None:
    rows = [
        _row(market_id="m1", decision_ts=1, side="UP", anchor=0.4, outcome="UP"),
        _row(
            market_id="m1", decision_ts=1, side="DOWN", anchor=0.6, outcome="UP"
        ),
    ]
    rows[0]["features"][SELECTED_MID_INDEX] = np.nan
    with pytest.raises(ValueError, match="selected_mid anchor is invalid"):
        logit_offset_action_values(rows, [0.6, 0.5])


def test_synthetic_logit_rolling_origin_uses_only_strictly_prior_markets() -> None:
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
    protocol = {
        "slot_id": "synthetic-logit-slot",
        "rolling_origin": {
            "initial_training_market_count": 2,
            "target_block_size": 2,
            "target_block_count": 1,
        },
        "model": {
            "parameters": _logit_model_parameters(),
            "fixed_num_boost_round": 2,
        },
    }
    predictions, audits = _rolling_origin_logit_predict(
        rows=rows,
        population_order=population,
        protocol=protocol,
    )
    assert len(predictions) == 8
    assert len(audits) == 1
    assert audits[0]["strictly_prior_training_market_count"] == 2
    assert audits[0]["target_or_future_label_leakage_count"] == 0
    assert {row["market_id"] for row in predictions} == {"m2", "m3"}
    assert all(row["target_or_future_label_used_for_fit"] is False for row in predictions)
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
