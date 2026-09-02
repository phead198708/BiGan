from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_expert_safe_policy_v7_3 import (
    NestedExpertSafePolicyV73Config,
    fit_nested_expert_safe_policy_v7_3,
    score_nested_expert_safe_policy_v7_3_market,
    validate_nested_expert_safe_policy_v7_3_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_nested_expert_safe_policy_v7_3_profile.json"
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
    return fit_nested_expert_safe_policy_v7_3(
        rows=_rows(
            baseline_target=baseline_target,
            opposite_target=opposite_target,
        ),
        profile=_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=2_000_000,
    )


def test_profile_freezes_nested_experts_and_historical_gate() -> None:
    profile = _profile()
    validate_nested_expert_safe_policy_v7_3_profile(profile)
    assert profile["expert_library"]["expert_names"] == [
        "KEEP_V6_7",
        "KNN_QUANTILE_RELATIVE",
        "RIDGE_RELATIVE",
    ]
    assert profile["historical_replay_superiority_gate"][
        "exact_evaluation_market_count"
    ] == 90
    assert profile["target_free_canary"][
        "historical_superiority_gate_must_pass_before_collection"
    ] is True
    assert all(
        value is False for value in profile["prior_result_exclusion"].values()
    )

    changed = copy.deepcopy(profile)
    changed["expert_library"]["knn_neighbor_count"] = 10
    with pytest.raises(ValueError, match="library"):
        validate_nested_expert_safe_policy_v7_3_profile(changed)


def test_nested_prior_only_expert_can_beat_v6_7_on_same_90_markets() -> None:
    result = _fit(baseline_target=-1.0, opposite_target=1.0)
    model = result["model_artifact"]
    replay = model["historical_replay_superiority_gate"]
    assert model["historical_gate_passed"] is True
    assert model["target_free_canary_collection_allowed"] is True
    assert model["historical_policy_difference_market_count"] == 90
    assert replay["candidate"][
        "total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(18.0)
    assert replay["v6_7_baseline"][
        "total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(-18.0)
    assert replay["evaluation_market_count"] == 90
    assert all(
        row["opposite_decision_ts"] == row["baseline_decision_ts"]
        for row in result["outer_oof_rows"]
    )


def test_nested_selection_is_strictly_prior_and_never_uses_outer_oof() -> None:
    model = _fit(baseline_target=-1.0, opposite_target=1.0)["model_artifact"]
    assert all(
        report["outer_train_max_decision_ts"]
        < report["outer_validation_min_decision_ts"]
        and report["outer_validation_targets_used_for_expert_selection_or_fit"]
        is False
        and report["nested_selection"]["outer_oof_results_used_for_selection"]
        is False
        and report["nested_selection"][
            "selection_uses_strictly_prior_training_markets_only"
        ]
        is True
        for report in model["outer_fold_reports"]
    )
    assert model["final_nested_selection"][
        "outer_oof_results_used_for_selection"
    ] is False


def test_identical_policy_fails_before_any_new_collection() -> None:
    model = _fit(baseline_target=1.0, opposite_target=1.0)["model_artifact"]
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


def test_outcome_free_consumer_and_forbidden_target_fail_closed() -> None:
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
    decision = score_nested_expert_safe_policy_v7_3_market(
        market_rows, model_artifact=model
    )
    assert decision["opposite_decision_ts"] == decision["baseline_decision_ts"]
    assert decision["outcome_or_pnl_field_used_at_inference"] is False
    assert decision["source_score_mutated"] is False
    assert decision["capital_at_risk"] is False

    blocked = score_nested_expert_safe_policy_v7_3_market(
        [*market_rows, {**market_rows[0], "settlement_pnl": 1.0}],
        model_artifact=model,
    )
    assert blocked["selected_action"] == "NO_TRADE"
    assert "v7_3_forbidden_outcome_field_in_inference_row" in blocked[
        "selection_reason_codes"
    ]
    assert blocked["v8_execution_handoff_allowed"] is False


def test_config_accepts_no_prior_result_artifact_path() -> None:
    annotations = NestedExpertSafePolicyV73Config.__annotations__
    assert not any("issue233" in name or "issue234" in name for name in annotations)
    assert "v7_2_relative_policy_source_path" in annotations
