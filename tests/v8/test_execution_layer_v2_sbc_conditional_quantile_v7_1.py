from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)
from bigan.v8.polymarket.training.execution_layer_v2_sbc_conditional_quantile_v7_1 import (
    _historical_replay_superiority,
    fit_sbc_conditional_quantile_v7_1,
    score_sbc_conditional_quantile_v7_1_decision_group,
    validate_sbc_conditional_quantile_v7_1_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_sbc_conditional_quantile_v7_1_profile.json"
)
SBC_ACTIONS = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)
HTS_ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _canonical_rows(*, target: float = 1.0) -> list[dict]:
    rows = []
    for market_index in range(134):
        decision_ts = 1_000_000 + market_index * 10_000
        for action_index, action in enumerate(SBC_ACTIONS):
            side = "UP" if "_UP_" in action else "DOWN"
            score = 0.2 if side == "UP" else 0.1
            features = dict.fromkeys(FEATURE_NAMES, 0.0)
            features.update(
                {
                    "action_score_available": 1.0,
                    "action_score": score,
                    "action_score_margin": score - 0.1,
                    "selected_side_probability": 0.6 if side == "UP" else 0.4,
                    "execution_price": 0.5,
                    "selected_side_probability_minus_execution_price": (
                        0.1 if side == "UP" else -0.1
                    ),
                    "side_is_up": float(side == "UP"),
                }
            )
            rows.append(
                {
                    "source": "test",
                    "market_id": f"market-{market_index:03d}",
                    "decision_group_id": f"market-{market_index:03d}|{decision_ts}",
                    "decision_ts": decision_ts,
                    "max_input_ts": decision_ts - 1,
                    "role": "historical_development",
                    "action_family": "SELL_BEFORE_CLOSE",
                    "action": action,
                    "side": side,
                    "decision_time_features": features,
                    "target_after_cost_net_pnl_per_contract": target,
                    "target_used_as_decision_time_input": False,
                    "target_available_only_post_exit_or_official_resolution": True,
                    "row_index": action_index,
                }
            )
    return rows


def _oof_row(
    *, market_index: int, action: str, prediction: float, target: float, score: float
) -> dict:
    decision_ts = 2_000_000 + market_index * 10_000
    side = "UP" if "_UP_" in action else "DOWN"
    return {
        "market_id": f"market-{market_index:03d}",
        "decision_group_id": f"market-{market_index:03d}|{decision_ts}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "action": action,
        "side": side,
        "fold_index": 0,
        "raw_conditional_lower_quantile_after_cost_net_pnl_per_contract": prediction,
        "fold_train_constant_lower_quantile": 0.0,
        "frozen_v6_7_base_score_available": True,
        "frozen_v6_7_base_score": score,
        "frozen_v6_7_base_score_source": (
            "frozen_v6_2_market_clustered_mean_ev_lcb"
        ),
        "target_after_cost_net_pnl_per_contract": target,
        "target_used_as_decision_time_input": False,
        "target_used_for_model_parameter_threshold_or_gate_selection": False,
    }


def test_profile_freezes_same_dataset_v6_7_superiority_before_collection() -> None:
    profile = _profile()
    validate_sbc_conditional_quantile_v7_1_profile(profile)
    gate = profile["historical_replay_superiority_gate"]
    assert gate["exact_evaluation_market_count"] == 90
    assert gate["fixed_position_size"] == 0.2
    assert gate["common_selected_row_filter_allowed"] is False
    assert gate["gate_failure_stops_before_target_free_collection"] is True

    changed = copy.deepcopy(profile)
    changed["historical_replay_superiority_gate"][
        "candidate_minus_v6_7_total_pnl_minimum_exclusive"
    ] = -1.0
    with pytest.raises(ValueError, match="replay_gate"):
        validate_sbc_conditional_quantile_v7_1_profile(changed)


def test_same_dataset_replay_counts_no_bet_markets_and_beats_v6_7() -> None:
    rows = []
    for market_index in range(90):
        rows.extend(
            [
                _oof_row(
                    market_index=market_index,
                    action=SBC_ACTIONS[0],
                    prediction=0.5,
                    target=1.0,
                    score=0.1,
                ),
                _oof_row(
                    market_index=market_index,
                    action=SBC_ACTIONS[1],
                    prediction=-0.5,
                    target=-1.0,
                    score=0.2,
                ),
            ]
        )
    report = _historical_replay_superiority(
        rows, conformal_correction=0.0, profile=_profile()
    )
    assert report["evaluation_market_count"] == 90
    assert report["candidate_selected_market_count"] == 90
    assert report["v6_7_baseline_selected_market_count"] == 90
    assert report["candidate"]["total_after_cost_net_pnl_at_frozen_size"] == pytest.approx(
        18.0
    )
    assert report["v6_7_baseline"][
        "total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(-18.0)
    assert report["candidate_total_pnl_strictly_better_than_v6_7"] is True
    assert report["common_selected_row_filter_applied"] is False


def test_equal_historical_pnl_fails_closed_before_canary() -> None:
    result = fit_sbc_conditional_quantile_v7_1(
        rows=_canonical_rows(),
        profile=_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=2_000_000,
    )
    model = result["model_artifact"]
    assert (
        "historical_same_dataset_candidate_pnl_not_strictly_better_than_v6_7"
        in model["historical_gate_blocking_reason_codes"]
    )
    assert model["historical_gate_passed"] is False
    assert model["target_free_canary_collection_allowed"] is False
    assert model["paper_candidate_allowed"] is False
    assert model["v8_execution_handoff_allowed"] is False


def test_outcome_free_consumer_keeps_hts_disabled_and_safety_blocked() -> None:
    fitted = fit_sbc_conditional_quantile_v7_1(
        rows=_canonical_rows(),
        profile=_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=2_000_000,
    )["model_artifact"]
    action_rows = []
    for action in (*SBC_ACTIONS, *HTS_ACTIONS):
        side = "UP" if "_UP_" in action else "DOWN"
        source = _canonical_rows()[0 if side == "UP" else 1]
        action_rows.append(
            {
                **source,
                "action": action,
                "action_family": (
                    "SELL_BEFORE_CLOSE"
                    if action in SBC_ACTIONS
                    else "HOLD_TO_SETTLEMENT"
                ),
                "decision_group_id": "inference-group",
            }
        )
    decision = score_sbc_conditional_quantile_v7_1_decision_group(
        action_rows, model_artifact=fitted
    )
    assert decision["selected_action"] == "NO_TRADE"
    assert "v7_1_historical_gate_not_passed" in decision["selection_reason_codes"]
    assert decision["outcome_or_pnl_field_used_at_inference"] is False
    assert decision["paper_candidate_allowed"] is False
    assert decision["capital_at_risk"] is False
