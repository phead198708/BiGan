from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7 as v77,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_canary as canary,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_canary_plan.json"
)


def _plan() -> dict[str, Any]:
    return json.loads(PLAN_PATH.read_text())


def _features(side: str) -> dict[str, float]:
    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    values.update(
        {
            "action_score_available": 1.0,
            "action_score": 0.1,
            "action_score_margin": 0.01,
            "selected_side_probability": 0.6 if side == "UP" else 0.4,
            "execution_price": 0.5,
            "selected_side_probability_minus_execution_price": (
                0.1 if side == "UP" else -0.1
            ),
            "side_is_up": float(side == "UP"),
        }
    )
    return values


def _canonical(market_id: str, action: str, decision_ts: int) -> dict[str, Any]:
    side = "UP" if "_UP_" in action else "DOWN"
    return {
        "market_id": market_id,
        "decision_group_id": f"{market_id}|{decision_ts}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "action": action,
        "side": side,
        "decision_time_features": _features(side),
    }


def _source(market_id: str, action: str, decision_ts: int) -> dict[str, Any]:
    side = "UP" if "_UP_" in action else "DOWN"
    probability = 0.6 if side == "UP" else 0.4
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "market_close_ts": decision_ts + 240_000,
        "action": action,
        "side": side,
        "selected_side_probability": probability,
        "p_up_action_disagreement": False,
        "decision_time_features": {
            "execution_price": 0.5,
            "selected_side_executable_ask_notional": 10.0,
            "selected_side_executable_bid_notional": 10.0,
            "selected_side_liquidity_depth": 100.0,
        },
        "microstructure_snapshot": {
            "spread_bps": 100.0,
            "book_staleness_ms": 100.0,
            "queue_fill_proxy": 0.9,
            "time_to_close_seconds": 180.0,
        },
        "reference_price_feature_provenance": {"provenance_valid": True},
    }


def _model() -> dict[str, Any]:
    seed = []
    for index in range(40):
        for side in ("UP", "DOWN"):
            seed.append(
                {
                    "market_id": f"seed-{index}",
                    "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
                    "side": side,
                    "decision_time_features": _features(side),
                    "target_after_cost_net_pnl_per_contract": 0.1,
                }
            )
    artifact = v77._fit_weighted_model(
        seed,
        market_order=[f"seed-{index}" for index in range(40)],
        profile=json.loads(
            (
                ROOT
                / "examples/v8/polymarket_configs/"
                "execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_profile.json"
            ).read_text()
        ),
    )
    return {
        "schema_version": v77.MODEL_SCHEMA_VERSION,
        "historical_noninferiority_gate_passed": True,
        "target_free_canary_collection_allowed": True,
        "final_weighted_model": artifact,
    }


def test_canary_plan_freezes_inclusive_noninferiority_and_bounded_window() -> None:
    plan = _plan()
    canary.validate_v7_7_canary_plan(
        plan,
        historical_manifest_sha256=plan["historical_gate"][
            "historical_manifest_sha256"
        ],
        model_sha256=plan["historical_gate"]["model_sha256"],
        profile_sha256=plan["historical_gate"]["profile_sha256"],
    )
    assert plan["historical_gate"]["equality_passes"] is True
    assert plan["collection"]["target_quality_valid_market_count"] == 12
    assert plan["collection"]["maximum_attempted_market_count"] == 18
    assert plan["target_free_actionability_gate"]["no_side_quota"] is True

    changed = copy.deepcopy(plan)
    changed["frozen_scoring"]["fixed_edge_buffer"] = 0.0
    with pytest.raises(ValueError, match="frozen_scoring"):
        canary.validate_v7_7_canary_plan(
            changed,
            historical_manifest_sha256=plan["historical_gate"][
                "historical_manifest_sha256"
            ],
            model_sha256=plan["historical_gate"]["model_sha256"],
            profile_sha256=plan["historical_gate"]["profile_sha256"],
        )


def test_target_free_window_preserves_v6_7_no_trade_and_safety() -> None:
    decision_ts = 2_000_000
    market_id = "market-1"
    actions = ["BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE"]
    canonical = [_canonical(market_id, action, decision_ts) for action in actions]
    sources = [_source(market_id, action, decision_ts) for action in actions]
    decisions, replay = canary._score_window(
        [market_id],
        canonical_rows=canonical,
        baseline_rows=[],
        action_rows=sources,
        model=_model(),
        v6_7_profile=json.loads(
            (
                ROOT
                / "examples/v8/polymarket_configs/"
                "execution_layer_v2_p_up_semantic_compatibility_v6_7_profile.json"
            ).read_text()
        ),
    )
    assert decisions[0]["selected_action"] == "NO_TRADE"
    assert replay[0]["execution_guard_order_allowed"] is False
    assert replay[0]["capital_at_risk"] is False
    assert replay[0]["v8_execution_handoff_allowed"] is False
    assert replay[0]["labels_outcomes_or_pnl_opened"] is False


def test_target_free_window_scores_same_decision_pair_without_outcomes() -> None:
    decision_ts = 2_000_000
    market_id = "market-2"
    actions = ["BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE"]
    canonical = [_canonical(market_id, action, decision_ts) for action in actions]
    sources = [_source(market_id, action, decision_ts) for action in actions]
    baseline = {
        **sources[0],
        "mean_ev_lower_confidence_bound": 0.1,
        "microstructure_safety_passed": True,
        "source_score_mutated": False,
    }
    decisions, replay = canary._score_window(
        [market_id],
        canonical_rows=canonical,
        baseline_rows=[baseline],
        action_rows=sources,
        model=_model(),
        v6_7_profile=json.loads(
            (
                ROOT
                / "examples/v8/polymarket_configs/"
                "execution_layer_v2_p_up_semantic_compatibility_v6_7_profile.json"
            ).read_text()
        ),
    )
    assert len(decisions) == len(replay) == 1
    assert decisions[0]["outcome_or_pnl_field_used_at_inference"] is False
    assert decisions[0]["source_score_mutated"] is False
    assert replay[0]["paper_candidate_allowed"] is False
    assert replay[0]["polymarket_write_enabled"] is False


def test_target_free_scorer_rejects_forbidden_outcome_fields() -> None:
    decision_ts = 2_000_000
    market_id = "market-3"
    baseline = _canonical(market_id, "BUY_UP_SELL_BEFORE_CLOSE", decision_ts)
    opposite = _canonical(market_id, "BUY_DOWN_SELL_BEFORE_CLOSE", decision_ts)
    baseline["settlement_pnl"] = 1.0
    result = v77.score_rolling_origin_drift_adaptive_v7_7_market(
        {
            "market_id": market_id,
            "baseline_row": baseline,
            "opposite_row": opposite,
        },
        model_artifact=_model(),
    )
    assert result["selected_action"] == "NO_TRADE"
    assert "v7_7_forbidden_outcome_field_in_inference_row" in result[
        "selection_reason_codes"
    ]
