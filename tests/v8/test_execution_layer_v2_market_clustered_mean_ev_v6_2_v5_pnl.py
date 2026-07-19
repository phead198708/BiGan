from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_v5_pnl import (
    _join_pnl_evidence,
    _raw_point_scores,
    _summary,
    _validate_profile,
    _validate_target_rows,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_market_clustered_mean_ev_v6_2_v5_pnl_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def test_profile_pins_sized_cost_aware_pnl_and_no_tuning() -> None:
    profile = _profile()
    _validate_profile(profile)
    assert profile["pnl"]["target_field"] == "target_net_pnl_per_contract"
    assert profile["pnl"]["sized_pnl_formula"].endswith("proposed_order_size")
    assert profile["interpretation"]["result_used_to_tune_v6_2_or_future_gate"] is False
    assert profile["safety"]["promotion_evidence_eligible"] is False


def test_join_pnl_uses_actual_order_size_and_cost_basis() -> None:
    accepted = [
        {
            "market_id": "market",
            "decision_ts": 1_000,
            "executed_action": "BUY_UP_SELL_BEFORE_CLOSE",
            "selected_action_family": "SELL_BEFORE_CLOSE",
            "selected_side": "UP",
            "proposed_order_size": 0.2,
            "microstructure_snapshot": {
                "entry_ask": 0.7,
                "time_to_close_seconds": 180.0,
                "spread_bps": 100.0,
                "book_staleness_ms": 50.0,
                "queue_fill_proxy": 0.9,
            },
            "p_up": 0.72,
            "p_down": 0.28,
            "decision_score": 0.04,
        }
    ]
    targets = [
        {
            "market_id": "market",
            "decision_ts": 1_000,
            "action": "BUY_UP_SELL_BEFORE_CLOSE",
            "target_net_pnl_per_contract": 0.1,
            "target_resolved_outcome": "UP",
            "target_cost_components": {"fees": 0.001},
        }
    ]
    evidence = _join_pnl_evidence(
        accepted,
        target_rows=targets,
        role="development_train",
        policy_name="market_clustered_mean_ev_v6_2",
    )[0]
    assert evidence["cost_basis"] == pytest.approx(0.14)
    assert evidence["after_cost_sized_net_pnl"] == pytest.approx(0.02)
    assert evidence["target_net_pnl_per_contract"] == pytest.approx(0.1)
    assert evidence["promotion_evidence_eligible"] is False


def test_summary_reports_chronological_drawdown_and_largest_winner_robustness() -> None:
    rows = [
        _evidence("one", 1, 0.2, 0.2),
        _evidence("two", 2, -0.3, 0.2),
        _evidence("three", 3, 0.15, 0.2),
    ]
    summary = _summary(rows, profile=_profile())
    assert summary["after_cost_sized_net_pnl"] == pytest.approx(0.05)
    assert summary["chronological_max_drawdown"] == pytest.approx(0.3)
    assert summary["largest_winning_market_pnl"] == pytest.approx(0.2)
    assert summary["pnl_after_removing_largest_winner"] == pytest.approx(-0.15)
    assert summary["market_bootstrap_mean_pnl_confidence_interval"]["resamples"] == 5000


def test_raw_point_policy_masks_incompatible_trade() -> None:
    scored = _raw_point_scores(
        [
            {
                "action": "BUY_UP_HOLD_TO_SETTLEMENT",
                "raw_direct_predicted_net_return": 0.2,
                "guard_compatible_before_ranking": False,
            },
            {
                "action": "NO_TRADE",
                "raw_direct_predicted_net_return": 0.0,
                "guard_compatible_before_ranking": True,
            },
        ]
    )
    assert scored[0]["action_advantage_lcb_net_return"] == -1_000_000.0
    assert scored[1]["action_advantage_lcb_net_return"] == 0.0


def test_target_validation_rejects_future_feature_or_decision_target_usage() -> None:
    row = {
        "role": "development_train",
        "decision_ts": 1_000,
        "max_input_ts": 1_001,
        "target_net_pnl_per_contract": 0.1,
        "target_used_as_decision_input": False,
        "outcome_fields_used_as_decision_input": False,
    }
    with pytest.raises(ValueError, match="causality"):
        _validate_target_rows([row], expected_role="development_train")
    row["max_input_ts"] = 1_000
    row["target_used_as_decision_input"] = True
    with pytest.raises(ValueError, match="target used"):
        _validate_target_rows([row], expected_role="development_train")


def _evidence(market_id: str, decision_ts: int, pnl: float, cost_basis: float) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "after_cost_sized_net_pnl": pnl,
        "target_net_pnl_per_contract": pnl,
        "cost_basis": cost_basis,
    }
