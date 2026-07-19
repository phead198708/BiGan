from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_calibrated_side_probability_net_ev_v6_1 import (
    HTS_ACTIONS,
    SBC_ACTIONS,
    apply_probability_calibration,
    build_side_probability_rows,
    fit_probability_calibration,
    score_action_rows_from_probability,
    validate_calibrated_side_probability_net_ev_v6_1_profile,
    validate_probability_calibration_artifact,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    REQUIRED_ACTIONS,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_calibrated_side_probability_net_ev_v6_1_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def test_profile_freezes_data_roles_action_support_and_safety() -> None:
    profile = _profile()
    validate_calibrated_side_probability_net_ev_v6_1_profile(profile)
    assert profile["chronological_roles"] == {
        "side_probability_fit_market_count": 90,
        "probability_calibration_market_count": 45,
        "net_return_conformal_market_count": 60,
        "target_free_check_market_count": 50,
        "assignment": (
            "v5_development_train_then_development_calibration_then_confirmatory_then_"
            "post_issue204_target_free"
        ),
    }
    assert set(profile["probability_to_net_ev"]["enabled_actions"]) == HTS_ACTIONS
    assert set(profile["probability_to_net_ev"]["disabled_actions"]) == SBC_ACTIONS
    assert profile["target_free_check"]["minimum_full_guard_accepted_market_count"] == 10
    assert profile["safety"]["promotion_evidence_eligible"] is False


def test_profile_rejects_market_probability_as_direct_ev_or_relaxed_support() -> None:
    profile = _profile()
    profile["probability_to_net_ev"]["market_implied_probability_used_as_direct_fair_value_ev"] = (
        True
    )
    with pytest.raises(ValueError, match="ev_contract"):
        validate_calibrated_side_probability_net_ev_v6_1_profile(profile)

    profile = _profile()
    profile["target_free_check"]["minimum_full_guard_accepted_market_count_per_side"] = 0
    with pytest.raises(ValueError, match="target_free_support"):
        validate_calibrated_side_probability_net_ev_v6_1_profile(profile)


def test_side_probability_grid_uses_one_row_per_decision_and_never_action_flags() -> None:
    rows = _action_rows(market_id="market-a", decision_ts=1000, outcome="UP")
    probability_rows = build_side_probability_rows(rows, profile=_profile(), include_target=True)
    assert len(probability_rows) == 1
    row = probability_rows[0]
    assert row["target_resolved_outcome_is_up"] == 1
    assert row["max_input_ts"] <= row["decision_ts"]
    assert all(not name.startswith("action_") for name in row["model_features"])
    assert "up_execution_price" in row["model_features"]
    assert "down_execution_price" in row["model_features"]
    assert row["market_implied_probability_used_as_direct_fair_value_ev"] is False


def test_side_probability_grid_rejects_incomplete_group_and_target_free_leakage() -> None:
    rows = _action_rows(market_id="market-a", decision_ts=1000, outcome="UP")
    with pytest.raises(ValueError, match="complete five-action"):
        build_side_probability_rows(rows[:-1], profile=_profile(), include_target=True)
    with pytest.raises(ValueError, match="contain targets"):
        build_side_probability_rows(rows, profile=_profile(), include_target=False)


def test_probability_calibration_is_bounded_deterministic_and_fail_closed() -> None:
    rows = []
    raw = []
    for index in range(20):
        rows.append(
            {
                "market_id": f"market-{index // 2}",
                "target_resolved_outcome_is_up": int(index % 3 != 0),
            }
        )
        raw.append(0.2 + 0.03 * index)
    artifact = fit_probability_calibration(
        rows,
        raw,
        profile=_profile(),
        model_sha256="a" * 64,
    )
    validate_probability_calibration_artifact(artifact)
    calibrated = apply_probability_calibration(raw, artifact)
    assert all(0.0 < value < 1.0 for value in calibrated)
    assert artifact == fit_probability_calibration(
        rows,
        raw,
        profile=_profile(),
        model_sha256="a" * 64,
    )

    legacy = copy.deepcopy(artifact)
    legacy.pop("market_implied_probability_used_as_direct_fair_value_ev")
    with pytest.raises(ValueError, match="market_probability_direct_ev"):
        validate_probability_calibration_artifact(legacy)


def test_probability_to_ev_enables_hts_masks_sbc_and_strips_outcomes() -> None:
    labeled = _action_rows(market_id="market-a", decision_ts=1000, outcome="UP")
    target_free = []
    for row in labeled:
        stripped = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "target_net_pnl_per_contract",
                "target_resolved_outcome",
                "target_cost_components",
            }
        }
        target_free.append(stripped)
    target_free_probability_rows = build_side_probability_rows(
        target_free, profile=_profile(), include_target=False
    )
    predictions = score_action_rows_from_probability(
        target_free,
        probability_rows=target_free_probability_rows,
        calibrated_p_up=[0.80],
        profile=_profile(),
    )
    by_action = {row["action"]: row for row in predictions}
    assert by_action["BUY_UP_HOLD_TO_SETTLEMENT"]["raw_direct_predicted_net_return"] > 0.0
    assert by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]["raw_direct_predicted_net_return"] < 0.0
    assert all(
        by_action[action]["guard_compatible_before_ranking"] is False for action in SBC_ACTIONS
    )
    assert all(
        by_action[action]["probability_to_net_ev_action_available"] is True
        for action in HTS_ACTIONS
    )
    assert all("target_resolved_outcome" not in row for row in predictions)
    assert all(
        row["market_implied_probability_used_as_direct_fair_value_ev"] is False
        for row in predictions
    )


def _action_rows(*, market_id: str, decision_ts: int, outcome: str) -> list[dict]:
    profile = _profile()
    common_names = profile["side_probability_features"]["common_feature_names"]
    side_names = profile["side_probability_features"]["side_feature_names"]
    rows = []
    for action in REQUIRED_ACTIONS:
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if action.endswith("HOLD_TO_SETTLEMENT")
            else "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "NO_TRADE"
        )
        p_up = 0.70
        selected_probability = p_up if side == "UP" else 1.0 - p_up if side == "DOWN" else 0.0
        ask = 0.71 if side == "UP" else 0.31 if side == "DOWN" else 0.0
        bid = 0.69 if side == "UP" else 0.29 if side == "DOWN" else 0.0
        features = dict.fromkeys(common_names, 0.1)
        features.update(dict.fromkeys(side_names, 1.0))
        features.update(
            {
                "selected_side_probability": selected_probability,
                "execution_price": ask,
                "selected_side_probability_minus_execution_price": (selected_probability - ask),
                "selected_side_spread_bps": 200.0,
                "selected_side_queue_fill_probability_proxy": 0.9,
                "selected_side_book_staleness_ms": 100.0,
                "selected_side_liquidity_depth": 10.0,
            }
        )
        features.update(
            {
                "action_buy_up": float(side == "UP"),
                "action_buy_down": float(side == "DOWN"),
                "action_hold_to_settlement": float(family == "HOLD_TO_SETTLEMENT"),
                "action_sell_before_close": float(family == "SELL_BEFORE_CLOSE"),
                "action_no_trade": float(action == "NO_TRADE"),
            }
        )
        disagreement = (side == "UP" and p_up < 0.5) or (side == "DOWN" and p_up > 0.5)
        fees = 0.0002 if action != "NO_TRADE" else 0.0
        slippage = max(0.0001, (ask - bid) / 2.0) if action != "NO_TRADE" else 0.0
        impact = 0.00005 if action != "NO_TRADE" else 0.0
        payout = 1.0 if side == outcome else 0.0
        target = payout - ask - fees - slippage - impact if action != "NO_TRADE" else 0.0
        rows.append(
            {
                "market_id": market_id,
                "condition_id": market_id,
                "market_slug": market_id,
                "decision_ts": decision_ts,
                "market_close_ts": decision_ts + 240_000,
                "max_input_ts": decision_ts,
                "role": "development_train",
                "market_selection_rank": 1,
                "action": action,
                "side": side,
                "action_family": family,
                "decision_time_features": features,
                "p_up": p_up,
                "p_down": 1.0 - p_up,
                "p_up_action_disagreement": disagreement,
                "microstructure_snapshot": {
                    "entry_bid": bid,
                    "entry_ask": ask,
                    "spread_bps": 200.0,
                    "book_staleness_ms": 100.0,
                    "queue_fill_proxy": 0.9,
                    "time_to_close_seconds": 180.0,
                },
                "reference_price_feature_provenance": {
                    "provenance_valid": True,
                    "max_input_ts": decision_ts,
                },
                "target_net_pnl_per_contract": target,
                "target_resolved_outcome": outcome,
                "target_cost_components": {
                    "fees": fees,
                    "slippage": slippage,
                    "liquidity_impact": impact,
                },
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
                "paper_only": True,
                "capital_at_risk": False,
            }
        )
    return rows
