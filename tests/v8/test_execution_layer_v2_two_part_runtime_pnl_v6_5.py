from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_two_part_runtime_pnl_v6_5 import (
    _fit_two_part_model,
    _market_grouped_cross_fit,
    score_two_part_runtime_pnl_rows,
    validate_two_part_runtime_pnl_v6_5_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_two_part_runtime_pnl_v6_5_profile.json"
)


def _features(value: float, *, side: str) -> dict[str, float]:
    return {
        "side_is_up": float(side == "UP"),
        "execution_price": 0.5 + value * 0.03,
        "current_bid": 0.48 + value * 0.03,
        "spread_bps": 220.0 - value * 10.0,
        "book_staleness_ms": 180.0 - value * 5.0,
        "queue_fill_probability_proxy": 0.75 + value * 0.05,
        "liquidity_depth_log1p": 5.0 + value,
        "executable_ask_notional_log1p": 4.0 + value * 0.5,
        "executable_bid_notional_log1p": 4.2 + value * 0.6,
        "time_to_close_seconds": 210.0 + value * 20.0,
        "recent_book_update_count_1m": 5.0 + value,
        "recent_bid_depth_volatility_1m": 0.02 - value * 0.002,
        "recent_spread_stability_1m": 0.8 + value * 0.04,
        "combined_spread_bps": 420.0 - value * 20.0,
        "liquidity_imbalance": value * 0.2,
        "btc_return_30s": value * 0.002,
        "btc_return_1m": value * 0.003,
        "reference_price_to_beat_distance_at_decision": value * 0.002,
        "canonical_v6_2_score": 0.1 + value * 0.02,
        "action_score_margin": 0.05 + value * 0.01,
        "selected_side_probability": 0.5 + value * 0.04,
        "pre_entry_market_exposure": 0.0,
        "same_side_prior_entry": 0.0,
        "side_flip_prior_entry": 0.0,
    }


def _rows() -> list[dict]:
    rows = []
    for market_index in range(89):
        market_id = f"market-{market_index:03d}"
        market_value = ((market_index % 13) - 6) / 10.0
        for slot in range(4):
            for side in ("UP", "DOWN"):
                value = market_value + slot * 0.12 + (0.06 if side == "UP" else -0.04)
                closed = side == "UP"
                target = (
                    0.18 + 0.25 * value
                    if closed
                    else -0.16 + 0.18 * value
                )
                decision_ts = 1_000_000 + market_index * 100_000 + slot * 10_000
                rows.append(
                    {
                        "market_id": market_id,
                        "role": "development_train",
                        "side": side,
                        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
                        "decision_ts": decision_ts,
                        "max_input_ts": decision_ts,
                        "features": _features(value, side=side),
                        "position_lifecycle_class": (
                            "closed_before_settlement"
                            if closed
                            else "settlement_residual"
                        ),
                        "runtime_policy_after_cost_net_pnl_per_contract": target,
                    }
                )
    return rows


def test_v6_5_profile_is_frozen_and_rejects_consumed_calibration() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    validate_two_part_runtime_pnl_v6_5_profile(profile)
    profile["prohibited"]["v6_4_consumed_calibration_labels_used"] = True
    with pytest.raises(ValueError, match="prohibited"):
        validate_two_part_runtime_pnl_v6_5_profile(profile)


def test_v6_5_two_part_score_is_decision_time_only() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    rows = _rows()
    model = _fit_two_part_model(rows, profile=profile)
    target_free = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "position_lifecycle_class",
                "runtime_policy_after_cost_net_pnl_per_contract",
            }
        }
        for row in rows[:4]
    ]
    scored = score_two_part_runtime_pnl_rows(target_free, model=model)
    assert len(scored) == 4
    assert all(row["target_fields_used_for_prediction"] is False for row in scored)
    assert all(0.0 < row["runtime_exit_probability"] < 1.0 for row in scored)
    assert all("runtime_expected_net_pnl_point" in row for row in scored)


def test_v6_5_market_grouped_cross_fit_is_deterministic() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    rows = _rows()
    first = _market_grouped_cross_fit(rows, profile=profile)
    second = _market_grouped_cross_fit(rows, profile=profile)
    assert first == second
    assert first["cross_fit_gate_passed"] is True
    assert first["relative_mae_improvement_over_fold_train_mean"] > 0.0
    assert first["relative_mse_improvement_over_fold_train_mean"] > 0.0
    assert first["exit_probability_roc_auc"] >= 0.55
    assert first["validation_oof_or_future_labels_used"] is False
    assert all(report["support_passed"] for report in first["fold_reports"])
