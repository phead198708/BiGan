from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_recency_adaptive_nested_action_value_v7_6 as v76,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_recency_adaptive_nested_action_value_v7_6_profile.json"
)
ACTIONS = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)


def _profile() -> dict[str, Any]:
    return json.loads(PROFILE_PATH.read_text())


def _rows(*, baseline_target: float, opposite_target: float) -> list[dict[str, Any]]:
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


def _selection(profile_name: str) -> dict[str, Any]:
    return {
        "selected_policy_profile_name": profile_name,
        "outer_oof_results_used_for_selection": False,
        "selection_uses_strictly_prior_training_markets_only": True,
    }


def _fake_score(
    example: dict[str, Any],
    *,
    policy_profile: dict[str, Any],
    head_artifact: dict[str, Any],
    fold_index: int | None,
    inference: bool = False,
) -> dict[str, Any]:
    del head_artifact
    switch = policy_profile["head"] == "RELATIVE_SWITCH_VALUE"
    selected_target = None
    baseline_target = None
    if not inference:
        baseline_target = float(example["baseline_target"])
        selected_target = (
            float(example["opposite_target"]) if switch else baseline_target
        )
    return {
        **example,
        "fold_index": fold_index,
        "selected_policy_profile_name": policy_profile["name"],
        "selected_model_head": policy_profile["head"],
        "head_available": True,
        "edge_buffer": float(policy_profile["edge_buffer"]),
        "predicted_baseline_return": None,
        "predicted_opposite_return": None,
        "predicted_switch_advantage": 1.0 if switch else None,
        "selected_policy_decision": (
            "SWITCH_SAME_DECISION_SBC" if switch else "KEEP_V6_7"
        ),
        "selected_action": (
            example["opposite_action"] if switch else example["baseline_action"]
        ),
        "selected_side": (
            example["opposite_side"] if switch else example["baseline_side"]
        ),
        "selected_target_after_cost_net_pnl_per_contract": selected_target,
        "baseline_target_after_cost_net_pnl_per_contract": baseline_target,
        "target_used_as_decision_time_input": False,
        "outer_validation_target_used_for_profile_selection_or_fit": False,
    }


def _stub_fit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_profile_name: str,
    baseline_target: float,
    opposite_target: float,
) -> dict[str, Any]:
    monkeypatch.setattr(
        v76,
        "_nested_select_profile",
        lambda *args, **kwargs: _selection(selected_profile_name),
    )
    monkeypatch.setattr(
        v76,
        "_fit_profile_head",
        lambda *args, **kwargs: {
            "head": kwargs.get("policy_profile", args[0] if args else {})
            .get("head", "KEEP_V6_7"),
            "available": True,
            "fit_market_ids_hash": "a" * 64,
        },
    )
    monkeypatch.setattr(v76, "_score_example", _fake_score)
    return v76.fit_recency_adaptive_nested_action_value_v7_6(
        rows=_rows(
            baseline_target=baseline_target,
            opposite_target=opposite_target,
        ),
        profile=_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=2_000_000,
    )


def test_profile_freezes_recency_windows_and_inclusive_noninferiority() -> None:
    profile = _profile()
    v76.validate_recency_adaptive_nested_action_value_v7_6_profile(profile)
    assert profile["training_windows"]["market_counts"] == [30, 60, 0]
    assert len(profile["policy_profiles"]) == 19
    gate = profile["historical_replay_superiority_gate"]
    assert gate["comparison_operator"] == "greater_than_or_equal"
    assert gate["equality_passes_noninferiority"] is True
    assert gate["candidate_minus_v6_7_total_pnl_minimum_inclusive"] == 0.0

    changed = copy.deepcopy(profile)
    changed["training_windows"]["market_counts"] = [20, 60, 0]
    with pytest.raises(ValueError, match="windows"):
        v76.validate_recency_adaptive_nested_action_value_v7_6_profile(changed)


def test_recency_window_uses_latest_strictly_prior_markets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_fit_head(
        head: str,
        market_order: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        captured.extend(market_order)
        return {
            "head": head,
            "available": True,
            "fit_market_ids_hash": v76.canonical_json_sha256(market_order),
        }

    monkeypatch.setattr(v76, "_fit_head", fake_fit_head)
    prior = [f"market-{index:03d}" for index in range(80)]
    item = {
        "name": "BASELINE_VETO_W030_B000",
        "head": "BASELINE_LOSS_VETO",
        "training_window_market_count": 30,
        "edge_buffer": 0.0,
    }
    v76._fit_profile_head(
        item,
        prior,
        by_market={},
        profile=_profile(),
        cache={},
    )
    assert captured == prior[-30:]
    assert "market-080" not in captured


def test_equal_keep_passes_noninferiority_but_does_not_start_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _stub_fit(
        monkeypatch,
        selected_profile_name="KEEP_V6_7",
        baseline_target=1.0,
        opposite_target=1.0,
    )["model_artifact"]
    replay = model["historical_replay_noninferiority_gate"]
    assert model["historical_noninferiority_gate_passed"] is True
    assert replay[
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(0.0)
    assert replay[
        "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(0.0)
    assert model["historical_policy_difference_market_count"] == 0
    assert model["target_free_canary_collection_allowed"] is False
    assert "minimum_historical_policy_difference_met" in model[
        "historical_actionability_blocking_reason_codes"
    ]


def test_better_recency_policy_passes_and_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _stub_fit(
        monkeypatch,
        selected_profile_name="RELATIVE_SWITCH_W030_B000",
        baseline_target=-1.0,
        opposite_target=1.0,
    )["model_artifact"]
    replay = model["historical_replay_noninferiority_gate"]
    assert model["historical_noninferiority_gate_passed"] is True
    assert model["target_free_canary_collection_allowed"] is True
    assert model["historical_policy_difference_market_count"] == 90
    assert replay[
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(36.0)
    assert model["paper_candidate_allowed"] is False
    assert model["live_trading_enabled"] is False


def test_worse_recency_policy_fails_before_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _stub_fit(
        monkeypatch,
        selected_profile_name="RELATIVE_SWITCH_W030_B000",
        baseline_target=1.0,
        opposite_target=-1.0,
    )["model_artifact"]
    assert model["historical_noninferiority_gate_passed"] is False
    assert model["target_free_canary_collection_allowed"] is False
    assert "historical_same_dataset_candidate_pnl_worse_than_v6_7" in model[
        "historical_gate_blocking_reason_codes"
    ]


def test_target_free_consumer_fails_closed_on_outcome_field() -> None:
    model = {
        "schema_version": v76.MODEL_SCHEMA_VERSION,
        "historical_noninferiority_gate_passed": True,
        "target_free_canary_collection_allowed": True,
    }
    decision = v76.score_recency_adaptive_nested_action_value_v7_6_market(
        [{"market_id": "m", "settlement_pnl": 1.0}], model_artifact=model
    )
    assert decision["selected_action"] == "NO_TRADE"
    assert "v7_6_forbidden_outcome_field_in_inference_row" in decision[
        "selection_reason_codes"
    ]
    assert decision["capital_at_risk"] is False
    assert decision["polymarket_write_enabled"] is False


def test_config_cannot_accept_issue238_artifact_paths() -> None:
    annotations = v76.RecencyAdaptiveNestedActionValueV76Config.__annotations__
    assert not any("issue238" in name for name in annotations)
    assert "runtime_target_rows_path" in annotations
