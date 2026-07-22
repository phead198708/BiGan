from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    V67RelativeSafePolicyV72Config,
    fit_v6_7_relative_safe_policy_v7_2,
    score_v6_7_relative_safe_policy_v7_2_market,
    validate_v6_7_relative_safe_policy_v7_2_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_v6_7_relative_safe_policy_v7_2_profile.json"
)
ACTIONS = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _rows(*, baseline_target: float, opposite_target: float) -> list[dict]:
    rows = []
    for market_index in range(134):
        decision_ts = 1_000_000 + market_index * 10_000
        for action in ACTIONS:
            side = "UP" if "_UP_" in action else "DOWN"
            features = dict.fromkeys(FEATURE_NAMES, 0.0)
            features.update(
                {
                    "action_score_available": 1.0,
                    "action_score": 0.2 if side == "UP" else 0.1,
                    "action_score_margin": 0.1 if side == "UP" else -0.1,
                    "btc_anchor_direction": 0.2 if side == "UP" else -0.2,
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
                    "target_after_cost_net_pnl_per_contract": (
                        baseline_target if side == "UP" else opposite_target
                    ),
                    "target_used_as_decision_time_input": False,
                    "target_available_only_post_exit_or_official_resolution": True,
                }
            )
    return rows


def _fit(*, baseline_target: float, opposite_target: float) -> dict:
    return fit_v6_7_relative_safe_policy_v7_2(
        rows=_rows(
            baseline_target=baseline_target,
            opposite_target=opposite_target,
        ),
        profile=_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=2_000_000,
    )


def test_profile_freezes_baseline_relative_and_no_collection_contract() -> None:
    profile = _profile()
    validate_v6_7_relative_safe_policy_v7_2_profile(profile)
    assert profile["baseline_contract"]["baseline_is_default_action"] is True
    assert profile["action_contract"]["alternative_decision_timestamp_allowed"] is False
    assert profile["historical_replay_superiority_gate"][
        "exact_evaluation_market_count"
    ] == 90

    changed = copy.deepcopy(profile)
    changed["incremental_advantage_models"]["ridge_alpha"] = 1.0
    with pytest.raises(ValueError, match="models"):
        validate_v6_7_relative_safe_policy_v7_2_profile(changed)


def test_stable_switch_advantage_beats_v6_7_without_timestamp_change() -> None:
    result = _fit(baseline_target=-1.0, opposite_target=1.0)
    model = result["model_artifact"]
    replay = model["historical_replay_superiority_gate"]
    assert model["historical_gate_passed"] is True
    assert model["target_free_canary_collection_allowed"] is True
    assert replay["policy_decision_distribution"] == {
        "SWITCH_SAME_DECISION_SBC": 90
    }
    assert replay["candidate"]["total_after_cost_net_pnl_at_frozen_size"] == pytest.approx(
        18.0
    )
    assert replay["v6_7_baseline"][
        "total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(-18.0)
    assert all(
        row["opposite_decision_ts"] == row["baseline_decision_ts"]
        for row in result["oof_rows"]
    )
    assert all(
        fold["fit_max_decision_ts"] < fold["calibration_min_decision_ts"]
        < fold["validation_min_decision_ts"]
        and fold["calibration_max_decision_ts"]
        < fold["validation_min_decision_ts"]
        and fold["validation_labels_used_for_fold_model_or_correction"] is False
        for fold in model["fold_reports"]
    )


def test_policy_identity_fails_closed_before_new_collection() -> None:
    result = _fit(baseline_target=1.0, opposite_target=1.0)
    model = result["model_artifact"]
    assert model["historical_policy_difference_market_count"] == 0
    assert "candidate_identical_to_v6_7" in model[
        "historical_gate_blocking_reason_codes"
    ]
    assert (
        "historical_same_dataset_candidate_pnl_not_strictly_better_than_v6_7"
        in model["historical_gate_blocking_reason_codes"]
    )
    assert model["historical_gate_passed"] is False
    assert model["target_free_canary_collection_allowed"] is False
    assert model["paper_candidate_allowed"] is False


def test_outcome_free_inference_switches_only_same_decision_action() -> None:
    model = _fit(baseline_target=-1.0, opposite_target=1.0)["model_artifact"]
    market_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "target_after_cost_net_pnl_per_contract",
                "target_used_as_decision_time_input",
                "target_available_only_post_exit_or_official_resolution",
            }
        }
        for row in _rows(baseline_target=-1.0, opposite_target=1.0)
        if row["market_id"] == "market-133"
    ]
    decision = score_v6_7_relative_safe_policy_v7_2_market(
        market_rows, model_artifact=model
    )
    assert decision["baseline_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert decision["selected_policy_decision"] == "SWITCH_SAME_DECISION_SBC"
    assert decision["selected_action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
    assert decision["opposite_decision_ts"] == decision["baseline_decision_ts"]
    assert decision["outcome_or_pnl_field_used_at_inference"] is False
    assert decision["source_score_mutated"] is False
    assert decision["capital_at_risk"] is False


def test_outcome_field_fails_closed_and_config_accepts_no_issue233_path() -> None:
    assert not any(
        "issue233" in name for name in V67RelativeSafePolicyV72Config.__annotations__
    )
    model = _fit(baseline_target=-1.0, opposite_target=1.0)["model_artifact"]
    market_rows = [
        row
        for row in _rows(baseline_target=-1.0, opposite_target=1.0)
        if row["market_id"] == "market-133"
    ]
    decision = score_v6_7_relative_safe_policy_v7_2_market(
        market_rows, model_artifact=model
    )
    assert decision["selected_action"] == "NO_TRADE"
    assert "v7_2_forbidden_outcome_field_in_inference_row" in decision[
        "selection_reason_codes"
    ]
    assert decision["paper_candidate_allowed"] is False
    assert decision["v8_execution_handoff_allowed"] is False


def test_opposite_action_must_share_baseline_decision_timestamp() -> None:
    rows = _rows(baseline_target=-1.0, opposite_target=1.0)
    down = next(
        row
        for row in rows
        if row["market_id"] == "market-000" and row["side"] == "DOWN"
    )
    down["decision_group_id"] = "market-000|later"
    down["decision_ts"] += 1
    down["max_input_ts"] += 1
    with pytest.raises(ValueError, match="same-decision"):
        fit_v6_7_relative_safe_policy_v7_2(
            rows=rows,
            profile=_profile(),
            implementation_commit="a" * 40,
            fit_created_ts=2_000_000,
        )
