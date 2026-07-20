from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_runtime_pnl_v6_6 import (
    _cross_fit,
    _fit_model,
    _select_policy_population,
    score_policy_selected_runtime_pnl_rows,
    validate_policy_selected_runtime_pnl_v6_6_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_policy_selected_runtime_pnl_v6_6_profile.json"
)


def _selected_rows() -> tuple[list[dict], list[dict]]:
    targets = []
    replay = []
    for index in range(65):
        side = "UP" if index < 22 else "DOWN"
        value = (index - 32) / 32.0
        decision_ts = 1_000_000 + index * 10_000
        market_id = f"market-{index:03d}"
        action = f"BUY_{side}_SELL_BEFORE_CLOSE"
        features = {
            "side_is_up": float(side == "UP"),
            "execution_price": 0.5 + value * 0.08,
            "current_bid": 0.48 + value * 0.08,
            "spread_bps": 240.0 - value * 20.0,
            "queue_fill_probability_proxy": 0.8 + value * 0.05,
            "time_to_close_seconds": 180.0 + value * 30.0,
            "selected_side_probability": 0.52 + value * 0.05,
            "canonical_v6_2_score": 0.03 + (value + 1.0) * 0.01,
        }
        targets.append(
            {
                "market_id": market_id,
                "role": "development_train",
                "side": side,
                "action": action,
                "decision_ts": decision_ts,
                "max_input_ts": decision_ts,
                "features": features,
                "runtime_policy_after_cost_net_pnl_per_contract": (
                    -0.04 + 0.3 * value + 0.08 * float(side == "UP")
                ),
            }
        )
        replay.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "execution_guard_order_allowed": True,
                "selected_action_family": "SELL_BEFORE_CLOSE",
                "executed_action": action,
                "selected_side": side,
            }
        )
    targets.append(
        {
            **targets[0],
            "market_id": "counterfactual",
            "decision_ts": 9_999_999,
        }
    )
    replay.append(
        {
            "market_id": "counterfactual",
            "decision_ts": 9_999_999,
            "execution_guard_order_allowed": False,
            "selected_action_family": "NO_TRADE",
            "executed_action": "NO_TRADE",
            "selected_side": "NONE",
        }
    )
    return targets, replay


def test_v6_6_profile_rejects_counterfactual_population() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    validate_policy_selected_runtime_pnl_v6_6_profile(profile)
    profile["fit_population"]["counterfactual_rows_included"] = True
    with pytest.raises(ValueError, match="population"):
        validate_policy_selected_runtime_pnl_v6_6_profile(profile)


def test_v6_6_population_selection_is_target_free() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    targets, replay = _selected_rows()
    selected, audit = _select_policy_population(
        targets, replay_rows=replay, profile=profile
    )
    assert len(selected) == 65
    assert audit["selected_side_count"] == {"DOWN": 43, "UP": 22}
    assert audit["population_support_gate_passed"] is True
    assert audit["outcome_settlement_target_or_pnl_fields_used_for_selection"] is False
    assert audit["excluded_reason_distribution"]["guard"] == 1


def test_v6_6_compact_ridge_cross_fit_and_target_free_scoring() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    targets, replay = _selected_rows()
    selected, _ = _select_policy_population(targets, replay_rows=replay, profile=profile)
    model = _fit_model(selected, profile=profile)
    cross_fit = _cross_fit(selected, profile=profile)
    assert model["feature_columns"] == profile["model"]["feature_columns"]
    assert cross_fit["cross_fit_gate_passed"] is True
    assert cross_fit["relative_mae_improvement_over_fold_train_mean"] > 0.0
    assert cross_fit["relative_mse_improvement_over_fold_train_mean"] > 0.0
    target_free = [
        {
            key: value
            for key, value in row.items()
            if key != "runtime_policy_after_cost_net_pnl_per_contract"
        }
        for row in selected[:3]
    ]
    scored = score_policy_selected_runtime_pnl_rows(target_free, model=model)
    assert all(row["target_fields_used_for_prediction"] is False for row in scored)
    assert all("runtime_expected_net_pnl_point" in row for row in scored)
