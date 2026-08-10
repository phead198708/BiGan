from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    GATE_NAMES,
    SELECTED_MID_INDEX,
    _action_policy,
    _bootstrap_contract,
    _cost_stress_contract,
    _model_parameters,
    _power_contract,
    _rolling_contract,
)
from bigan.v8.polymarket.cost_aware_residual_v3 import (
    ENGINEERED_VALUE_FEATURE_NAMES,
    LINEAGE_ID,
    PROTOCOL_SCHEMA_VERSION,
    RECENCY_HALF_LIFE_MARKETS,
    V3_FEATURE_NAMES,
    _baseline_contract,
    _dataset_contract,
    _discipline_contract,
    _feature_contract,
    _model_contract,
    _pair_contract,
    _state_contract,
    _target_contract,
    _temporal_contract,
    engineer_causal_features,
    recency_weights,
    rolling_origin_causal_time_adaptive_predict,
    validate_residual_v3_protocol,
    validate_v3_lineage_authorization,
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
            "parent_v2_terminal_review",
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
        "slot_id": "residual-v3-primary-slot-001",
        "candidate_role": "primary",
        "created_at": "2026-08-09T10:00:00Z",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "candidate_budget": {
            "maximum_total_slots": 2,
            "this_slot_ordinal": 1,
            "slots_consumed_before_run": 0,
            "slots_remaining_after_run": 1,
            "slot_budget_may_be_increased": False,
        },
        "target": _target_contract(),
        "pair_coherence": _pair_contract(),
        "feature_contract": _feature_contract(),
        "temporal_adaptation": _temporal_contract(),
        "model": _model_contract(),
        "action_policy": _action_policy(),
        "bootstrap": _bootstrap_contract(),
        "cost_stress": _cost_stress_contract(),
        "gates": dict.fromkeys(GATE_NAMES, True),
        "prospective_power": _power_contract(),
        "rolling_origin": _rolling_contract(),
        "baseline": _baseline_contract(),
        "dataset": _dataset_contract(),
        "development_discipline": _discipline_contract(),
        "state": _state_contract(),
        "safety": dict(SAFETY),
        "inputs": inputs,
    }


def _features() -> np.ndarray:
    values = np.full(len(FEATURE_NAMES), 0.1, dtype=float)
    assignments = {
        "selected_ask": 0.44,
        "selected_bid": 0.42,
        "selected_mid": 0.43,
        "opposite_ask": 0.59,
        "opposite_bid": 0.57,
        "opposite_mid": 0.58,
        "paired_ask_sum": 1.03,
        "paired_mid_sum": 1.01,
        "selected_liquidity_depth": 30.0,
        "opposite_liquidity_depth": 20.0,
        "selected_executable_ask_notional": 15.0,
        "opposite_executable_ask_notional": 10.0,
        "selected_book_staleness_ms": 100.0,
        "opposite_book_staleness_ms": 140.0,
        "signed_btc_return_10s": 0.01,
        "signed_btc_return_1m": 0.02,
        "signed_btc_return_5m": 0.03,
        "signed_btc_return_15m": 0.04,
        "signed_btc_mid_to_chainlink_relative_distance": 0.005,
        "market_progress_fraction": 0.5,
        "provider_health_score": 0.9,
    }
    for name, value in assignments.items():
        values[FEATURE_NAMES.index(name)] = value
    return values


def _row(side: str, selected_mid: float) -> dict:
    features = _features()
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


def test_v3_authorization_binds_exact_user_grant_and_v2_terminal() -> None:
    result = validate_v3_lineage_authorization()
    assert result["authorization_valid"] is True
    assert result["lineage_id"] == LINEAGE_ID
    assert result["maximum_total_slots"] == 2
    assert result["parent_v2_immutable"] is True
    assert all(value is False for value in result["safety"].values())


def test_causal_feature_contract_is_fixed_ordered_and_deterministic() -> None:
    first = engineer_causal_features(_features())
    second = engineer_causal_features(_features())
    assert len(first) == 132
    assert len(V3_FEATURE_NAMES) == 132
    assert np.array_equal(first, second, equal_nan=True)
    assert canonical_json_sha256(list(V3_FEATURE_NAMES)) == (
        _feature_contract()["ordered_feature_names_sha256"]
    )
    assert tuple(V3_FEATURE_NAMES[-24:-12]) == ENGINEERED_VALUE_FEATURE_NAMES
    assert first[V3_FEATURE_NAMES.index("selected_spread_fraction")] == pytest.approx(
        0.02
    )
    assert first[V3_FEATURE_NAMES.index("paired_ask_overround")] == pytest.approx(
        0.03
    )
    assert first[
        V3_FEATURE_NAMES.index("signed_return_5m_x_market_progress")
    ] == pytest.approx(0.015)


def test_engineered_missingness_remains_nan_and_is_not_zero_imputed() -> None:
    features = _features()
    features[FEATURE_NAMES.index("selected_liquidity_depth")] = np.nan
    transformed = engineer_causal_features(features)
    value_index = V3_FEATURE_NAMES.index("log_liquidity_depth_ratio")
    missing_index = V3_FEATURE_NAMES.index("log_liquidity_depth_ratio__missing")
    assert np.isnan(transformed[value_index])
    assert transformed[missing_index] == 1.0
    assert transformed[FEATURE_NAMES.index("selected_liquidity_depth")] is np.nan or (
        np.isnan(transformed[FEATURE_NAMES.index("selected_liquidity_depth")])
    )


def test_recency_weights_are_market_grouped_deterministic_and_mean_one() -> None:
    population = ["m0", "m1", "m2"]
    row_markets = [market for market in population for _ in range(4)]
    first = recency_weights(row_markets, population_order=population)
    second = recency_weights(row_markets, population_order=population)
    assert first.tolist() == second.tolist()
    assert np.mean(first) == pytest.approx(1.0)
    assert len(set(first[0:4])) == 1
    assert len(set(first[4:8])) == 1
    assert len(set(first[8:12])) == 1
    assert first[0] < first[4] < first[8]
    assert first[8] / first[0] == pytest.approx(
        2.0 ** (2.0 / RECENCY_HALF_LIFE_MARKETS)
    )


def test_protocol_preserves_old_gates_zero_threshold_n_cost_population_and_safety() -> None:
    protocol = _protocol()
    validate_residual_v3_protocol(protocol, verify_artifacts=False)

    mutations = []
    changed = deepcopy(protocol)
    changed["action_policy"]["fixed_acceptance_threshold"] = 0.01
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["prospective_power"]["maximum_market_count"] = 2001
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["gates"].pop("prospective_power_required_market_count_lte_2000")
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["candidate_budget"]["maximum_total_slots"] = 3
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["cost_stress"]["multipliers"] = [1.2, 1.5]
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["dataset"]["market_count"] = 799
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["temporal_adaptation"]["half_life_markets"] = 100.0
    mutations.append(changed)
    changed = deepcopy(protocol)
    changed["safety"]["live_trading_allowed"] = True
    mutations.append(changed)

    for mutation in mutations:
        with pytest.raises(ValueError, match="residual v3 protocol invalid"):
            validate_residual_v3_protocol(mutation, verify_artifacts=False)


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
        "slot_id": "synthetic-v3-slot",
        "rolling_origin": {
            "initial_training_market_count": 2,
            "target_block_size": 2,
            "target_block_count": 1,
        },
        "temporal_adaptation": _temporal_contract(),
        "model": {
            "parameters": _model_parameters(),
            "fixed_num_boost_round": 2,
        },
    }
    predictions, audits = rolling_origin_causal_time_adaptive_predict(
        rows=rows,
        population_order=population,
        protocol=protocol,
    )
    assert len(predictions) == 8
    assert len(audits) == 1
    assert audits[0]["strictly_prior_training_market_count"] == 2
    assert audits[0]["target_or_future_label_leakage_count"] == 0
    assert audits[0]["training_weight_mean"] == pytest.approx(1.0)
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
