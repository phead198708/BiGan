from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
    HTS_ACTIONS,
    MODEL_SCHEMA_VERSION,
    SBC_ACTIONS,
    fit_abstention_aware_v7_0,
    materialize_v7_0_hts_rows,
    materialize_v7_0_sbc_rows,
    score_v7_0_decision_group,
    validate_v7_0_training_profile,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_training_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _runtime_source_rows() -> list[dict]:
    rows = []
    roles = (("development_train", 89), ("development_calibration", 45))
    market_index = 0
    for role, count in roles:
        for role_index in range(count):
            market_id = f"sbc-{market_index:03d}"
            decision_ts = 1_000_000 + market_index * 1_000
            for side_index, side in enumerate(("UP", "DOWN")):
                action = f"BUY_{side}_SELL_BEFORE_CLOSE"
                score = -1_000_000.0 if role_index == 0 else 0.01 * (role_index + 1)
                probability = 0.6 if side == "UP" else 0.4
                price = probability - 0.05 + 0.01 * side_index
                target = (
                    0.04
                    + 0.25 * max(score, 0.0)
                    + 0.7 * (probability - price)
                    - 0.01 * side_index
                )
                rows.append(
                    {
                        "market_id": market_id,
                        "role": role,
                        "side": side,
                        "action": action,
                        "decision_ts": decision_ts,
                        "max_input_ts": decision_ts - 1,
                        "features": {
                            "canonical_v6_2_score": score,
                            "action_score_margin": score / 2,
                            "btc_return_30s": 0.001 * (1 if side == "UP" else -1),
                            "btc_return_1m": 0.002 * (1 if side == "UP" else -1),
                            "reference_price_to_beat_distance_at_decision": 0.0015,
                            "selected_side_probability": probability,
                            "execution_price": price,
                            "spread_bps": 100.0 + role_index,
                            "queue_fill_probability_proxy": 0.9,
                            "book_staleness_ms": 100.0,
                            "time_to_close_seconds": 240.0,
                            "pre_entry_market_exposure": 0.0,
                            "same_side_prior_entry": 0.0,
                            "side_flip_prior_entry": 0.0,
                        },
                        "runtime_policy_after_cost_net_pnl_per_contract": target,
                        "target_used_as_decision_time_input": False,
                        "target_available_only_post_exit_or_official_resolution": True,
                    }
                )
            market_index += 1
    return rows


def _full_grid_source_rows() -> list[dict]:
    rows = []
    all_actions = [*SBC_ACTIONS, *HTS_ACTIONS, "NO_TRADE"]
    for market_index in range(65):
        market_id = f"hts-{market_index:03d}"
        decision_ts = 2_000_000 + market_index * 1_000
        ranking = []
        targets = {}
        for action_index, action in enumerate(all_actions):
            if action == "NO_TRADE":
                side = "NONE"
                probability = 0.0
                price = 0.0
            else:
                side = "UP" if "_UP_" in action else "DOWN"
                probability = 0.65 if side == "UP" else 0.35
                price = probability - 0.04 + 0.005 * action_index
            score = 0.2 * (4 - action_index) + 0.01 * market_index
            ranking.append(
                {
                    "selected_action": action,
                    "corrected_model_score": score,
                    "microstructure_snapshot": {
                        "entry_ask": price,
                        "spread_bps": 120.0 + market_index,
                        "queue_fill_proxy": 0.92,
                        "book_staleness_ms": 90.0,
                        "time_to_close_seconds": 240.0,
                    },
                }
            )
            targets[action] = (
                0.0
                if action == "NO_TRADE"
                else 0.05 + 0.2 * score + 0.6 * (probability - price)
            )
        rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "max_input_ts": decision_ts - 1,
                "decision_time_features": {
                    "chainlink_momentum_30s": 0.001,
                    "chainlink_momentum_60s": 0.002,
                    "chainlink_momentum_120s": 0.003,
                    "reference_price_to_beat_distance_at_decision": 0.0015,
                    "cumulative_market_exposure_before_entry": 0.0,
                    "same_side_reentry": 0.0,
                    "side_flip": 0.0,
                },
                "execution_handoff_context": {
                    "p_up": 0.65,
                    "p_down": 0.35,
                    "full_5_action_ranking": ranking,
                },
                "evaluation_target_net_pnl_per_contract_by_action": targets,
                "target_outcome_available_only_post_resolution": True,
                "target_provenance": {"outcome_used_as_decision_input": False},
            }
        )
    return rows


def _inference_rows() -> list[dict]:
    rows = []
    for action in [*SBC_ACTIONS, *HTS_ACTIONS]:
        side = "UP" if "_UP_" in action else "DOWN"
        family = (
            "SELL_BEFORE_CLOSE" if action in SBC_ACTIONS else "HOLD_TO_SETTLEMENT"
        )
        rows.append(
            {
                "market_id": "future",
                "decision_group_id": "future|3000000",
                "decision_ts": 3_000_000,
                "max_input_ts": 2_999_999,
                "action": action,
                "action_family": family,
                "side": side,
                "decision_time_features": dict.fromkeys(FEATURE_NAMES, 0.0),
            }
        )
    return rows


def _scoring_model(*, intercept: float) -> dict:
    family_models = {}
    for family in ("SELL_BEFORE_CLOSE", "HOLD_TO_SETTLEMENT"):
        family_models[family] = {
            "model": {
                "feature_mean": [0.0] * len(FEATURE_NAMES),
                "feature_scale": [1.0] * len(FEATURE_NAMES),
                "intercept": intercept,
                "coefficients": [0.0] * len(FEATURE_NAMES),
            },
            "calibration": {"residual_quantile": 0.0},
        }
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "historical_development_gate_passed": True,
        "family_models": family_models,
    }


def test_training_profile_is_fixed_and_rejects_side_quota() -> None:
    validate_v7_0_training_profile(_profile())
    drift = _profile()
    drift["selection_contract"]["side_quota_allowed"] = True
    with pytest.raises(ValueError, match="selection"):
        validate_v7_0_training_profile(drift)


def test_all_historical_markets_materialize_without_positive_score_prefilter() -> None:
    profile = _profile()
    sbc_rows = materialize_v7_0_sbc_rows(_runtime_source_rows(), profile)
    hts_rows = materialize_v7_0_hts_rows(_full_grid_source_rows(), profile)
    assert len({row["market_id"] for row in sbc_rows}) == 134
    assert len({row["market_id"] for row in hts_rows}) == 65
    assert sum(
        row["decision_time_features"]["action_score_available"] == 0.0
        for row in sbc_rows
    ) == 4
    assert {row["action"] for row in hts_rows} == set(HTS_ACTIONS)
    assert all(row["max_input_ts"] <= row["decision_ts"] for row in [*sbc_rows, *hts_rows])


def test_fixed_fit_uses_no_future_rows_and_keeps_safety_blocked() -> None:
    profile = _profile()
    result = fit_abstention_aware_v7_0(
        sbc_rows=materialize_v7_0_sbc_rows(_runtime_source_rows(), profile),
        hts_rows=materialize_v7_0_hts_rows(_full_grid_source_rows(), profile),
        profile=profile,
        implementation_commit="a" * 40,
        fit_created_ts=4_000_000,
    )
    model = result["model_artifact"]
    assert model["issue229_or_issue231_rows_used_for_fit_or_tuning"] is False
    assert model["validation_or_oof_pnl_used_for_tuning_or_gate"] is False
    assert model["side_quota_applied"] is False
    assert model["paper_candidate_allowed"] is False
    assert model["live_trading_enabled"] is False
    assert len(result["oof_rows"]) > 0
    assert len(result["calibration_rows"]) > 0


def test_outcome_free_consumer_abstains_and_can_select_positive_lcb() -> None:
    rows = _inference_rows()
    no_trade = score_v7_0_decision_group(
        rows, model_artifact=_scoring_model(intercept=-0.01)
    )
    assert no_trade["selected_action"] == "NO_TRADE"
    assert no_trade["selection_reason_codes"] == [
        "v7_0_no_positive_calibrated_lower_bound"
    ]
    positive = score_v7_0_decision_group(
        rows, model_artifact=_scoring_model(intercept=0.01)
    )
    assert positive["trade_selected"] is True
    assert positive["outcome_or_pnl_field_used_at_inference"] is False
    assert positive["paper_candidate_allowed"] is False


def test_inference_target_field_fails_closed() -> None:
    rows = copy.deepcopy(_inference_rows())
    rows[0]["resolved_outcome"] = "UP"
    decision = score_v7_0_decision_group(
        rows, model_artifact=_scoring_model(intercept=0.1)
    )
    assert decision["selected_action"] == "NO_TRADE"
    assert "v7_0_forbidden_outcome_field_in_inference_row" in decision[
        "selection_reason_codes"
    ]
