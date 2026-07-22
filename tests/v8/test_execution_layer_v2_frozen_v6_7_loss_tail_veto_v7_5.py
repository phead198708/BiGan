from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import bigan.v8.polymarket.training.execution_layer_v2_frozen_v6_7_loss_tail_veto_v7_5 as loss_tail_module
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)
from bigan.v8.polymarket.training.execution_layer_v2_frozen_v6_7_loss_tail_veto_v7_5 import (
    FrozenV67LossTailVetoV75Config,
    fit_frozen_v6_7_loss_tail_veto_v7_5,
    score_frozen_v6_7_loss_tail_veto_v7_5_market,
    validate_frozen_v6_7_loss_tail_veto_v7_5_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_frozen_v6_7_loss_tail_veto_v7_5_profile.json"
)
ACTIONS = ("BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE")


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _rows(*, reverse_after_market: int | None = None) -> list[dict]:
    rows = []
    for market_index in range(134):
        decision_ts = 1_000_000 + market_index * 10_000
        high_risk = market_index % 5 == 0
        reverse = reverse_after_market is not None and market_index >= reverse_after_market
        baseline_target = (-1.0 if high_risk else 0.3) if not reverse else (
            0.3 if high_risk else -1.0
        )
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
                    "log1p_book_staleness_ms": 8.5 if high_risk else 2.0,
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
                        baseline_target if side == "UP" else 0.0
                    ),
                    "target_used_as_decision_time_input": False,
                    "target_available_only_post_exit_or_official_resolution": True,
                }
            )
    return rows


def _fit(*, reverse_after_market: int | None = None) -> dict:
    return fit_frozen_v6_7_loss_tail_veto_v7_5(
        rows=_rows(reverse_after_market=reverse_after_market),
        profile=_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=2_000_000,
    )


@pytest.fixture(scope="module")
def improved_fit() -> dict:
    return _fit()


def test_profile_freezes_classifier_veto_and_noninferiority_contract() -> None:
    profile = _profile()
    validate_frozen_v6_7_loss_tail_veto_v7_5_profile(profile)
    assert profile["loss_classifier"]["objective"] == "binary:logistic"
    assert [row["veto_fraction"] for row in profile["veto_profiles"]] == [
        0.0,
        0.05,
        0.1,
        0.2,
    ]
    gate = profile["historical_replay_superiority_gate"]
    assert gate["candidate_minus_v6_7_total_pnl_minimum_inclusive"] == 0.0
    assert profile["target_free_canary"][
        "historical_policy_difference_market_count_minimum"
    ] == 1

    changed = copy.deepcopy(profile)
    changed["loss_classifier"]["max_depth"] = 3
    with pytest.raises(ValueError, match="classifier"):
        validate_frozen_v6_7_loss_tail_veto_v7_5_profile(changed)


def test_loss_tail_veto_can_improve_without_side_or_action_switch(
    improved_fit: dict,
) -> None:
    model = improved_fit["model_artifact"]
    assert model["historical_noninferiority_gate_passed"] is True
    assert model["model_improvement_demonstrated"] is True
    assert model["historical_policy_difference_market_count"] > 0
    assert model["target_free_canary_collection_allowed"] is True
    assert all(
        row["selected_action"] in {row["baseline_action"], "NO_TRADE"}
        and row["selected_policy_decision"]
        in {"KEEP_V6_7", "VETO_TO_NO_TRADE"}
        for row in improved_fit["outer_oof_rows"]
    )


def test_equal_keep_policy_passes_history_but_blocks_collection() -> None:
    rows = _rows()
    for row in rows:
        row["target_after_cost_net_pnl_per_contract"] = 1.0
    model = fit_frozen_v6_7_loss_tail_veto_v7_5(
        rows=rows,
        profile=_profile(),
        implementation_commit="b" * 40,
        fit_created_ts=2_000_000,
    )["model_artifact"]
    assert model["historical_noninferiority_gate_passed"] is True
    assert model["historical_policy_difference_market_count"] == 0
    assert model["model_improvement_demonstrated"] is False
    assert model["target_free_canary_collection_allowed"] is False
    assert "historical_policy_difference_support_absent" in model[
        "target_free_canary_collection_blocking_reason_codes"
    ]


def test_worse_historical_replay_fails_noninferiority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = loss_tail_module._historical_replay

    def worse_replay(rows: list[dict], *, profile: dict) -> dict:
        replay = original(rows, profile=profile)
        replay[
            "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
        ] = -0.01
        replay[
            "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
        ] = -0.01
        return replay

    monkeypatch.setattr(loss_tail_module, "_historical_replay", worse_replay)
    model = _fit()["model_artifact"]
    assert model["historical_noninferiority_gate_passed"] is False
    assert model["target_free_canary_collection_allowed"] is False
    assert any(
        reason.startswith("historical_same_dataset_")
        for reason in model["historical_gate_blocking_reason_codes"]
    )


def test_nested_selection_is_strictly_prior_only(improved_fit: dict) -> None:
    assert all(
        fold["outer_train_max_decision_ts"]
        < fold["outer_validation_min_decision_ts"]
        and fold["outer_validation_targets_used_for_profile_selection_or_fit"]
        is False
        and fold["nested_selection"]["outer_oof_results_used_for_selection"]
        is False
        for fold in improved_fit["model_artifact"]["outer_fold_reports"]
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
        for row in _rows()
        if row["market_id"] == "market-133"
    ]
    decision = score_frozen_v6_7_loss_tail_veto_v7_5_market(
        market_rows, model_artifact=model
    )
    assert decision["selected_action"] in {
        decision["baseline_action"],
        "NO_TRADE",
    }
    assert decision["side_switch_applied"] is False
    assert decision["source_score_mutated"] is False
    assert decision["outcome_or_pnl_field_used_at_inference"] is False

    blocked = score_frozen_v6_7_loss_tail_veto_v7_5_market(
        [*market_rows, {**market_rows[0], "settlement_pnl": 1.0}],
        model_artifact=model,
    )
    assert blocked["selected_action"] == "NO_TRADE"
    assert "v7_5_forbidden_outcome_field_in_inference_row" in blocked[
        "selection_reason_codes"
    ]
    assert blocked["v8_execution_handoff_allowed"] is False


def test_config_cannot_accept_prior_result_artifact_paths() -> None:
    annotations = FrozenV67LossTailVetoV75Config.__annotations__
    assert not any(
        any(issue in name for issue in ("issue233", "issue234", "issue235", "issue236"))
        for name in annotations
    )
    assert "runtime_target_rows_path" in annotations
    assert "v7_2_relative_policy_source_path" in annotations
