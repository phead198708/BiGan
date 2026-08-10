from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pytest

from bigan.v8.polymarket.cost_aware_residual import _load_json
from bigan.v8.polymarket.cost_aware_residual_v5 import (
    DEFAULT_PROTOCOL,
    FIXED_NUM_BOOST_ROUND,
    FIXED_PARAMETERS,
    _corrector_features,
    prequential_market_residual_corrector_predict,
    validate_residual_v5_protocol,
    validate_v5_lineage_authorization,
)
from bigan.v8.polymarket.cost_aware_residual_v5_challenger import (
    _canonicalize_feature_order,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY


def _fixture() -> tuple[list[dict], list[dict], list[str], dict]:
    population = [f"market-{index:02d}" for index in range(6)]
    rows: list[dict] = []
    base: list[dict] = []
    for position, market_id in enumerate(population, start=1):
        outcome = "UP" if position % 2 else "DOWN"
        for decision_offset in (300, 600):
            for side_index, side in enumerate(("UP", "DOWN")):
                values = {name: float((position + side_index) % 3) for name in FEATURE_NAMES}
                values[FEATURE_NAMES[0]] = None if position % 2 else float(position)
                row = {
                    "market_id": market_id,
                    "market_position": position,
                    "market_start_ts": 1_700_000_000 + position * 900,
                    "decision_ts": 1_700_000_000 + position * 900 + decision_offset,
                    "side": side,
                    "features": values,
                    "feature_row_sha256": f"{position:02d}-{decision_offset}-{side}",
                    "binary_payout_target": float(side == outcome),
                    "target": (0.51 if side == outcome else -0.49) - 0.01,
                    "resolved_outcome": outcome,
                    "cost_decomposition": {
                        "entry_ask": 0.49,
                        "entry_bid": 0.48,
                        "fees": 0.001,
                        "slippage": 0.004,
                        "liquidity_impact": 0.0,
                        "total_cost_excluding_entry_ask": 0.005,
                    },
                    "development_only_forever": True,
                    "promotion_evidence_eligible": False,
                    "safety": dict(SAFETY),
                }
                rows.append(row)
                if position > 2:
                    probability = 0.56 if side == "UP" else 0.44
                    base.append(
                        {
                            "market_id": market_id,
                            "decision_ts": row["decision_ts"],
                            "side": side,
                            "predicted_probability": probability,
                            "prediction": probability - 0.495,
                            "realized_unit_net_pnl_if_action": row["target"],
                            "feature_row_sha256": row["feature_row_sha256"],
                        }
                    )
    protocol = {
        "slot_id": "residual-v5-primary-slot-001",
        "rolling_origin": {
            "initial_training_market_count": 2,
            "target_block_size": 2,
            "target_block_count": 2,
        },
        "model": {
            "fixed_num_boost_round": FIXED_NUM_BOOST_ROUND,
            "parameters": dict(FIXED_PARAMETERS),
        },
    }
    return rows, base, population, protocol


def test_v5_authorization_is_exact_and_parent_is_immutable() -> None:
    result = validate_v5_lineage_authorization()

    assert result["authorization_valid"] is True
    assert result["maximum_total_slots"] == 2
    assert result["parent_v4_immutable"] is True
    assert result["safety"] == SAFETY


def test_v5_slot_1_protocol_and_executing_module_are_frozen() -> None:
    protocol_path = Path(DEFAULT_PROTOCOL)
    protocol = _load_json(protocol_path)

    validate_residual_v5_protocol(protocol)

    assert protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").strip() == (
        "f6ed1d30b42f36170e3da23f59fde19b7c8841bade9dde839757beddaa554da4"
    )
    assert protocol["inputs"]["candidate_implementation"]["sha256"] == (
        "daf9da7efeba997a077e21bca5b600aed4fe1c52b2bd37c966a7592d680080cb"
    )
    assert protocol["action_policy"]["fixed_acceptance_threshold"] == 0.0
    assert protocol["prospective_power"]["maximum_market_count"] == 2000


def test_v5_prequential_corrector_is_deterministic_and_leakage_free() -> None:
    rows, base, population, protocol = _fixture()

    first = prequential_market_residual_corrector_predict(
        rows=rows,
        frozen_base_predictions=base,
        population_order=population,
        protocol=protocol,
    )
    second = prequential_market_residual_corrector_predict(
        rows=deepcopy(rows),
        frozen_base_predictions=deepcopy(base),
        population_order=list(population),
        protocol=deepcopy(protocol),
    )

    assert first == second
    predictions, audits = first
    assert len(predictions) == 16
    assert len(audits) == 2
    assert audits[0]["strictly_prior_corrector_training_market_count"] == 0
    assert audits[0]["corrector_applied"] is False
    assert audits[1]["strictly_prior_corrector_training_market_count"] == 2
    assert audits[1]["corrector_applied"] is True
    assert all(row["current_or_future_label_used_for_corrector"] is False for row in predictions)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in predictions:
        grouped.setdefault((row["market_id"], row["decision_ts"]), []).append(row)
    assert all(
        math.isclose(sum(row["predicted_probability"] for row in pair), 1.0)
        for pair in grouped.values()
    )


def test_v5_native_missing_value_remains_nan() -> None:
    rows, base, population, _ = _fixture()
    row = next(
        row
        for row in rows
        if row["market_id"] == population[2]
        and row["features"][FEATURE_NAMES[0]] is None
    )
    frozen = next(
        item
        for item in base
        if item["feature_row_sha256"] == row["feature_row_sha256"]
    )

    values = _corrector_features(row, frozen)

    assert math.isnan(float(values[0]))
    assert values.shape == (112,)


def test_v5_fails_closed_on_missing_frozen_base_row() -> None:
    rows, base, population, protocol = _fixture()
    base.pop()

    with pytest.raises(ValueError, match="base prediction missing"):
        prequential_market_residual_corrector_predict(
            rows=rows,
            frozen_base_predictions=base,
            population_order=population,
            protocol=protocol,
        )


def test_v5_challenger_adapter_resolves_semantic_order_without_changing_values() -> None:
    rows, _, _, _ = _fixture()
    row = deepcopy(rows[0])
    row["features"] = dict(reversed(list(row["features"].items())))

    adapted = _canonicalize_feature_order(row)

    assert tuple(adapted["features"]) == tuple(FEATURE_NAMES)
    assert adapted["features"] == rows[0]["features"]
    assert tuple(row["features"]) != tuple(adapted["features"])
