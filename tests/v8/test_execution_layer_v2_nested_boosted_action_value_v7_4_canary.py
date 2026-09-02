from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4 import (
    MODEL_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    _canonicalize_target_free_sbc_rows,
    _score_and_guard,
)

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads((ROOT / "examples/v8/polymarket_configs" / name).read_text())


def _source_row(
    *, market_id: str, decision_ts: int, action: str, spread: float = 100.0
) -> dict:
    side = "UP" if "_UP_" in action else "DOWN"
    probability = 0.6 if side == "UP" else 0.4
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "action": action,
        "action_family": "SELL_BEFORE_CLOSE",
        "side": side,
        "selected_side_probability": probability,
        "p_up_action_disagreement": side == "DOWN",
        "decision_time_features": {
            "btc_return_30s": 0.001,
            "btc_return_1m": 0.002,
            "reference_price_to_beat_distance_at_decision": 0.003,
            "execution_price": probability - 0.02,
            "selected_side_executable_ask_notional": 10.0,
            "selected_side_executable_bid_notional": 10.0,
            "selected_side_liquidity_depth": 100.0,
        },
        "microstructure_snapshot": {
            "spread_bps": spread,
            "book_staleness_ms": 100.0,
            "queue_fill_proxy": 0.9,
            "time_to_close_seconds": 180.0,
        },
        "reference_price_feature_provenance": {"provenance_valid": True},
    }


def _scored(row: dict, score: float) -> dict:
    return {
        **row,
        "mean_ev_lower_confidence_bound": score,
        "target_fields_stripped": True,
    }


def _model() -> dict:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "historical_noninferiority_gate_passed": True,
        "final_policy_profile": {
            "name": "KEEP_V6_7",
            "head": "KEEP_V6_7",
            "edge_buffer": 0.0,
        },
        "final_head_artifact": {"head": "KEEP_V6_7", "available": True},
    }


def test_canary_keeps_positive_v6_7_action_and_p_up_is_diagnostic_only() -> None:
    v6_7 = _load("execution_layer_v2_p_up_semantic_compatibility_v6_7_profile.json")
    v7_0 = _load(
        "execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_training_profile.json"
    )
    rows = [
        _source_row(
            market_id="market-1",
            decision_ts=1000,
            action="BUY_UP_SELL_BEFORE_CLOSE",
        ),
        _source_row(
            market_id="market-1",
            decision_ts=1000,
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
        ),
    ]
    scored = [_scored(rows[0], 0.2), _scored(rows[1], 0.1)]

    canonical, summary = _canonicalize_target_free_sbc_rows(
        scored, action_rows=rows, v6_7_profile=v6_7, v7_0_profile=v7_0
    )
    decisions, replay = _score_and_guard(
        canonical, action_rows=rows, model=_model(), v6_7_profile=v6_7
    )

    assert summary["missing_scored_or_source_action_row_count"] == 0
    assert decisions[0]["selected_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert replay[0]["execution_guard_order_allowed"] is True
    assert replay[0]["p_up_action_disagreement_diagnostic_only"] is True
    assert replay[0]["capital_at_risk"] is False
    assert replay[0]["v8_execution_handoff_allowed"] is False


def test_canary_fails_closed_to_no_trade_without_positive_action() -> None:
    v6_7 = _load("execution_layer_v2_p_up_semantic_compatibility_v6_7_profile.json")
    v7_0 = _load(
        "execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_training_profile.json"
    )
    rows = [
        _source_row(
            market_id="market-2",
            decision_ts=2000,
            action="BUY_UP_SELL_BEFORE_CLOSE",
        ),
        _source_row(
            market_id="market-2",
            decision_ts=2000,
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
        ),
    ]
    canonical, _ = _canonicalize_target_free_sbc_rows(
        [_scored(rows[0], -0.1), _scored(rows[1], -0.2)],
        action_rows=rows,
        v6_7_profile=v6_7,
        v7_0_profile=v7_0,
    )
    decisions, replay = _score_and_guard(
        canonical, action_rows=rows, model=_model(), v6_7_profile=v6_7
    )

    assert decisions[0]["selected_action"] == "NO_TRADE"
    assert replay[0]["execution_guard_order_allowed"] is False
    assert replay[0]["execution_blocking_reason_codes"] == [
        "v7_4_no_positive_v6_7_baseline_action"
    ]


def test_canary_excludes_microstructure_blocked_action_before_ranking() -> None:
    v6_7 = _load("execution_layer_v2_p_up_semantic_compatibility_v6_7_profile.json")
    v7_0 = _load(
        "execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_training_profile.json"
    )
    rows = [
        _source_row(
            market_id="market-3",
            decision_ts=3000,
            action="BUY_UP_SELL_BEFORE_CLOSE",
            spread=1200.0,
        ),
        _source_row(
            market_id="market-3",
            decision_ts=3000,
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
        ),
    ]
    canonical, _ = _canonicalize_target_free_sbc_rows(
        [_scored(rows[0], 0.5), _scored(rows[1], 0.1)],
        action_rows=rows,
        v6_7_profile=v6_7,
        v7_0_profile=v7_0,
    )
    decisions, replay = _score_and_guard(
        canonical, action_rows=rows, model=_model(), v6_7_profile=v6_7
    )

    assert decisions[0]["selected_action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
    assert replay[0]["execution_guard_order_allowed"] is True
    assert all("target_after_cost" not in row for row in canonical)
