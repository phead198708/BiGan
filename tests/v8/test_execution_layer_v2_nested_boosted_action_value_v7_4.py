from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4 import (
    NestedBoostedActionValueV74Config,
    fit_nested_boosted_action_value_v7_4,
    score_nested_boosted_action_value_v7_4_market,
    validate_nested_boosted_action_value_v7_4_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_nested_boosted_action_value_v7_4_profile.json"
)
ACTIONS = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _rows(
    *,
    baseline_target: float,
    opposite_target: float,
    reverse_after_market: int | None = None,
) -> list[dict]:
    rows = []
    for market_index in range(134):
        decision_ts = 1_000_000 + market_index * 10_000
        reverse = reverse_after_market is not None and market_index >= reverse_after_market
        up_target = opposite_target if reverse else baseline_target
        down_target = baseline_target if reverse else opposite_target
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
                        up_target if side == "UP" else down_target
                    ),
                    "target_used_as_decision_time_input": False,
                    "target_available_only_post_exit_or_official_resolution": True,
                }
            )
    return rows


def _fit(
    *,
    baseline_target: float,
    opposite_target: float,
    reverse_after_market: int | None = None,
) -> dict:
    return fit_nested_boosted_action_value_v7_4(
        rows=_rows(
            baseline_target=baseline_target,
            opposite_target=opposite_target,
            reverse_after_market=reverse_after_market,
        ),
        profile=_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=2_000_000,
    )


@pytest.fixture(scope="module")
def improved_fit() -> dict:
    return _fit(baseline_target=-1.0, opposite_target=1.0)


@pytest.fixture(scope="module")
def equal_fit() -> dict:
    return _fit(baseline_target=1.0, opposite_target=1.0)


def test_profile_freezes_boosted_library_and_noninferiority_gate() -> None:
    profile = _profile()
    validate_nested_boosted_action_value_v7_4_profile(profile)
    assert profile["xgboost"]["num_boost_round"] == 64
    assert len(profile["policy_profiles"]) == 10
    gate = profile["historical_replay_superiority_gate"]
    assert gate["candidate_minus_v6_7_total_pnl_minimum_inclusive"] == 0.0
    assert gate["policy_difference_is_diagnostic_only"] is True

    changed = copy.deepcopy(profile)
    changed["xgboost"]["max_depth"] = 3
    with pytest.raises(ValueError, match="xgboost"):
        validate_nested_boosted_action_value_v7_4_profile(changed)


def test_nested_boosted_policy_can_improve_on_same_90_markets(
    improved_fit: dict,
) -> None:
    model = improved_fit["model_artifact"]
    replay = model["historical_replay_noninferiority_gate"]
    assert model["historical_noninferiority_gate_passed"] is True
    assert model["target_free_canary_collection_allowed"] is True
    assert model["model_improvement_demonstrated"] is True
    assert model["historical_policy_difference_market_count"] == 90
    assert replay["candidate"][
        "total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(18.0)
    assert replay["v6_7_baseline"][
        "total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(-18.0)


def test_equal_keep_policy_passes_noninferiority_without_claiming_improvement(
    equal_fit: dict,
) -> None:
    model = equal_fit["model_artifact"]
    replay = model["historical_replay_noninferiority_gate"]
    assert model["historical_noninferiority_gate_passed"] is True
    assert model["historical_policy_difference_market_count"] == 0
    assert model["model_improvement_demonstrated"] is False
    assert replay[
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(0.0)
    assert replay["gate_name"] == (
        "same_dataset_historical_replay_noninferiority_to_v6_7"
    )
    assert replay["policy_difference_is_diagnostic_only"] is True
    assert model["target_free_canary_collection_allowed"] is True


def test_distribution_reversal_fails_closed_when_candidate_is_worse() -> None:
    model = _fit(
        baseline_target=-1.0,
        opposite_target=1.0,
        reverse_after_market=44,
    )["model_artifact"]
    assert model["historical_noninferiority_gate_passed"] is False
    assert "historical_same_dataset_candidate_pnl_worse_than_v6_7" in model[
        "historical_gate_blocking_reason_codes"
    ]
    assert model["target_free_canary_collection_allowed"] is False


def test_nested_selection_is_prior_only_and_same_timestamp(
    improved_fit: dict,
) -> None:
    model = improved_fit["model_artifact"]
    assert all(
        fold["outer_train_max_decision_ts"]
        < fold["outer_validation_min_decision_ts"]
        and fold["outer_validation_targets_used_for_profile_selection_or_fit"]
        is False
        and fold["nested_selection"]["outer_oof_results_used_for_selection"]
        is False
        and fold["nested_selection"][
            "selection_uses_strictly_prior_training_markets_only"
        ]
        is True
        for fold in model["outer_fold_reports"]
    )
    assert all(
        row["opposite_decision_ts"] == row["baseline_decision_ts"]
        for row in improved_fit["outer_oof_rows"]
    )


def test_outcome_free_consumer_and_forbidden_target_fail_closed(
    improved_fit: dict,
) -> None:
    model = improved_fit["model_artifact"]
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
    decision = score_nested_boosted_action_value_v7_4_market(
        market_rows, model_artifact=model
    )
    assert decision["opposite_decision_ts"] == decision["baseline_decision_ts"]
    assert decision["outcome_or_pnl_field_used_at_inference"] is False
    assert decision["source_score_mutated"] is False
    assert decision["capital_at_risk"] is False

    blocked = score_nested_boosted_action_value_v7_4_market(
        [*market_rows, {**market_rows[0], "settlement_pnl": 1.0}],
        model_artifact=model,
    )
    assert blocked["selected_action"] == "NO_TRADE"
    assert "v7_4_forbidden_outcome_field_in_inference_row" in blocked[
        "selection_reason_codes"
    ]
    assert blocked["v8_execution_handoff_allowed"] is False


def test_config_accepts_no_prior_result_artifact_path() -> None:
    annotations = NestedBoostedActionValueV74Config.__annotations__
    assert not any(
        any(issue in name for issue in ("issue233", "issue234", "issue235"))
        for name in annotations
    )
    assert "v7_2_relative_policy_source_path" in annotations
