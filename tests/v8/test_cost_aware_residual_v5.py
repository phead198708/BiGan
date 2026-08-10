from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pytest

from bigan.v8.polymarket import cost_aware_residual_v5 as primary_v5
from bigan.v8.polymarket import cost_aware_residual_v5_challenger as challenger_v5
from bigan.v8.polymarket.cost_aware_residual import _load_json, _load_jsonl
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
    DEFAULT_PROTOCOL as CHALLENGER_PROTOCOL,
)
from bigan.v8.polymarket.cost_aware_residual_v5_challenger import (
    _canonicalize_feature_order,
    validate_challenger_protocol,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT


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


def test_v5_final_slot_protocol_preserves_candidate_and_gate_bytes() -> None:
    protocol_path = Path(CHALLENGER_PROTOCOL)
    protocol = _load_json(protocol_path)

    validate_challenger_protocol(protocol)

    assert protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").strip() == (
        "5483aeb573517eb66443368302d2aa8d4fbcc6d3ead97e1199b7edb1c2e56730"
    )
    assert protocol["inputs"]["candidate_implementation"]["sha256"] == (
        "df936704c7d429ab11a106527c6a6355a1335ef58907d9c6a0e4f38c8de23af6"
    )
    assert protocol["structural_change"]["candidate_algorithm_changed"] is False
    assert protocol["model"] == _load_json(Path(DEFAULT_PROTOCOL))["model"]
    assert protocol["gates"] == _load_json(Path(DEFAULT_PROTOCOL))["gates"]


def test_v5_terminal_report_is_independently_reconciled_and_fail_closed() -> None:
    root = Path(REPO_ROOT)
    protocol = _load_json(Path(CHALLENGER_PROTOCOL))
    output = root / (
        "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v5/"
        "residual_v5_challenger_slot_002_oof"
    )
    manifest = _load_json(output / "residual_v5_challenger_oof_manifest.json")
    predictions = _load_jsonl(output / "residual_v5_challenger_oof_predictions.jsonl")
    folds = _load_jsonl(output / "residual_v5_challenger_oof_fold_audits.jsonl")
    markets = _load_jsonl(output / "residual_v5_challenger_oof_market_results.jsonl")
    assert all(row["current_or_future_label_used_for_corrector"] is False for row in predictions)
    compatibility_predictions = [
        challenger_v5._replace_schema(
            {**row, "target_or_future_label_used_for_fit": False},
            primary_v5.PREDICTION_SCHEMA_VERSION,
        )
        for row in predictions
    ]
    primary_v5._validate_population(
        predictions=compatibility_predictions,
        fold_audits=[
            challenger_v5._replace_schema(row, primary_v5.FOLD_SCHEMA_VERSION)
            for row in folds
        ],
        market_results=[
            challenger_v5._replace_schema(row, primary_v5.MARKET_RESULT_SCHEMA_VERSION)
            for row in markets
        ],
        protocol=protocol,
    )
    rebuilt = challenger_v5._build_report(
        protocol=protocol,
        protocol_sha256=(
            "5483aeb573517eb66443368302d2aa8d4fbcc6d3ead97e1199b7edb1c2e56730"
        ),
        source_commit=manifest["source_commit"],
        market_results=markets,
        fold_audits=folds,
    )
    frozen = _load_json(output / "residual_v5_challenger_oof_report.json")
    _assert_semantically_equal(rebuilt, frozen, path="test_v5_terminal_reconciliation")
    terminal = _load_json(
        root
        / (
            "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v5/"
            "residual_v5_development_terminal_review.json"
        )
    )
    assert frozen["failed_gates"] == ["prospective_power_required_market_count_lte_2000"]
    assert frozen["prospective_power"]["required_market_count"] == 2598
    assert terminal["phase_1_terminal_failed"] is True
    assert terminal["candidate_budget_exhausted"] is True
    assert terminal["candidate_selected"] is None
    assert terminal["candidate_freeze_allowed"] is False
    assert terminal["safety"] == SAFETY
