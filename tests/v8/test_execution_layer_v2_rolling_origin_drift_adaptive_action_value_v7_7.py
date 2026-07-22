from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7 as v77,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_profile.json"
)


def _profile() -> dict[str, Any]:
    return json.loads(PROFILE_PATH.read_text())


def _features(*, side: str, score: float = 0.1) -> dict[str, float]:
    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    values.update(
        {
            "action_score_available": 1.0,
            "action_score": score,
            "action_score_margin": score,
            "selected_side_probability": 0.6 if side == "UP" else 0.4,
            "execution_price": 0.5,
            "selected_side_probability_minus_execution_price": (
                0.1 if side == "UP" else -0.1
            ),
            "side_is_up": float(side == "UP"),
        }
    )
    return values


def _seed_rows() -> list[dict[str, Any]]:
    output = []
    for index in range(134):
        decision_ts = 1_000_000 + index * 10_000
        for side in ("UP", "DOWN"):
            action = f"BUY_{side}_SELL_BEFORE_CLOSE"
            output.append(
                {
                    "source": "test_seed",
                    "market_id": f"seed-{index:03d}",
                    "decision_group_id": f"seed-{index:03d}|{decision_ts}",
                    "decision_ts": decision_ts,
                    "max_input_ts": decision_ts - 1,
                    "role": "historical_development",
                    "action_family": "SELL_BEFORE_CLOSE",
                    "action": action,
                    "side": side,
                    "decision_time_features": _features(side=side),
                    "target_after_cost_net_pnl_per_contract": 0.1,
                    "target_used_as_decision_time_input": False,
                    "target_available_only_post_exit_or_official_resolution": True,
                }
            )
    return output


def _feature_row(
    market_id: str, *, decision_ts: int, side: str
) -> dict[str, Any]:
    return {
        "source": "test_stream",
        "market_id": market_id,
        "decision_group_id": f"{market_id}|{decision_ts}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "role": "consumed_historical_prequential",
        "action_family": "SELL_BEFORE_CLOSE",
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "side": side,
        "microstructure_snapshot": {"time_to_close_seconds": 240.0},
        "decision_time_features": _features(side=side),
        "target_used_as_decision_time_input": False,
    }


def _stream_markets() -> list[dict[str, Any]]:
    output = []
    start = 3_000_000
    for index in range(120):
        decision_ts = start + index * 10_000
        market_id = f"stream-{index:03d}"
        output.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "market_close_ts": decision_ts + 5_000,
                "baseline_row": _feature_row(
                    market_id, decision_ts=decision_ts, side="UP"
                ),
                "opposite_row": _feature_row(
                    market_id, decision_ts=decision_ts, side="DOWN"
                ),
            }
        )
    return output


def _target(row: dict[str, Any], value: float) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "action": row["action"],
        "runtime_policy_after_cost_net_pnl_per_contract": value,
    }


def _fit_with_stubbed_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision: str,
    baseline_target: float,
    opposite_target: float,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        v77,
        "_fit_weighted_model",
        lambda *args, **kwargs: {"available": True, "booster_sha256": "a" * 64},
    )

    def score(
        market: dict[str, Any], *, artifact: dict[str, Any], profile: dict[str, Any]
    ) -> dict[str, Any]:
        del artifact, profile
        events.append(("score", market["market_id"]))
        baseline = market["baseline_row"]
        opposite = market["opposite_row"]
        selected_action = (
            opposite["action"]
            if decision == "SWITCH_SAME_DECISION_SBC"
            else "NO_TRADE"
            if decision == "VETO_TO_NO_TRADE"
            else baseline["action"]
        )
        selected_side = (
            opposite["side"]
            if decision == "SWITCH_SAME_DECISION_SBC"
            else "NONE"
            if decision == "VETO_TO_NO_TRADE"
            else baseline["side"]
        )
        return {
            "market_id": market["market_id"],
            "market_close_ts": market["market_close_ts"],
            "baseline_action": baseline["action"],
            "baseline_side": baseline["side"],
            "baseline_decision_ts": baseline["decision_ts"],
            "baseline_max_input_ts": baseline["max_input_ts"],
            "opposite_action": opposite["action"],
            "opposite_side": opposite["side"],
            "opposite_decision_ts": opposite["decision_ts"],
            "opposite_max_input_ts": opposite["max_input_ts"],
            "predicted_baseline_return": 0.0,
            "predicted_opposite_return": 0.0,
            "fixed_edge_buffer": 0.025,
            "selected_policy_decision": decision,
            "selected_action": selected_action,
            "selected_side": selected_side,
            "target_used_as_decision_time_input": False,
            "source_score_mutated": False,
        }

    monkeypatch.setattr(v77, "_score_stream_market", score)

    def load(
        baseline: dict[str, Any], opposite: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        events.append(("load", baseline["market_id"]))
        return _target(baseline, baseline_target), _target(
            opposite, opposite_target
        )

    fit = v77.fit_rolling_origin_drift_adaptive_v7_7(
        seed_rows=_seed_rows(),
        stream_markets=_stream_markets(),
        target_loader=load,
        profile=_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=4_500_000,
    )
    return fit, events


def test_profile_freezes_single_model_and_inclusive_gate() -> None:
    profile = _profile()
    v77.validate_rolling_origin_drift_adaptive_v7_7_profile(profile)
    assert profile["model_contract"]["exponential_decay_half_life_markets"] == 60.0
    assert profile["model_contract"]["fixed_edge_buffer"] == 0.025
    assert profile["model_contract"]["profile_or_hyperparameter_search_allowed"] is False
    gate = profile["historical_prequential_noninferiority_gate"]
    assert gate["comparison_operator"] == "greater_than_or_equal"
    assert gate["equality_passes_noninferiority"] is True

    changed = copy.deepcopy(profile)
    changed["model_contract"]["fixed_edge_buffer"] = 0.0
    with pytest.raises(ValueError, match="model"):
        v77.validate_rolling_origin_drift_adaptive_v7_7_profile(changed)


def test_prediction_precedes_each_current_target_load_and_equal_keep_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, events = _fit_with_stubbed_policy(
        monkeypatch,
        decision="KEEP_V6_7",
        baseline_target=0.1,
        opposite_target=-0.1,
    )
    assert all(
        events[index][0] == ("score" if index % 2 == 0 else "load")
        for index in range(len(events))
    )
    assert all(
        events[index][1] == events[index + 1][1]
        for index in range(0, len(events), 2)
    )
    model = fit["model_artifact"]
    replay = model["historical_prequential_noninferiority_gate"]
    assert model["historical_noninferiority_gate_passed"] is True
    assert replay[
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
    ] == pytest.approx(0.0)
    assert model["target_free_canary_collection_allowed"] is False
    assert model["paper_candidate_allowed"] is False


def test_improving_action_differences_allow_only_target_free_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, _ = _fit_with_stubbed_policy(
        monkeypatch,
        decision="SWITCH_SAME_DECISION_SBC",
        baseline_target=-1.0,
        opposite_target=1.0,
    )
    model = fit["model_artifact"]
    assert model["historical_noninferiority_gate_passed"] is True
    assert model["historical_policy_difference_market_count"] == 120
    assert model["target_free_canary_collection_allowed"] is True
    assert model["promotion_evidence_eligible"] is False
    assert model["live_trading_enabled"] is False


def test_worse_prequential_policy_fails_before_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, _ = _fit_with_stubbed_policy(
        monkeypatch,
        decision="SWITCH_SAME_DECISION_SBC",
        baseline_target=1.0,
        opposite_target=-1.0,
    )
    model = fit["model_artifact"]
    assert model["historical_noninferiority_gate_passed"] is False
    assert model["target_free_canary_collection_allowed"] is False
    assert "historical_same_stream_candidate_pnl_worse_than_v6_7" in model[
        "historical_gate_blocking_reason_codes"
    ]


def test_weighted_model_uses_positive_recency_weights() -> None:
    artifact = v77._fit_weighted_model(
        _seed_rows(),
        market_order=[f"seed-{index:03d}" for index in range(134)],
        profile=_profile(),
    )
    assert artifact["available"] is True
    assert artifact["training_market_count"] == 134
    assert 0.0 < artifact["minimum_row_weight"] < artifact["maximum_row_weight"]
    assert artifact["maximum_row_weight"] == pytest.approx(1.0)


def test_target_free_consumer_uses_frozen_rule_and_rejects_outcomes() -> None:
    artifact = v77._fit_weighted_model(
        _seed_rows(),
        market_order=[f"seed-{index:03d}" for index in range(134)],
        profile=_profile(),
    )
    model = {
        "schema_version": v77.MODEL_SCHEMA_VERSION,
        "historical_noninferiority_gate_passed": True,
        "target_free_canary_collection_allowed": True,
        "final_weighted_model": artifact,
    }
    market = _stream_markets()[0]
    decision = v77.score_rolling_origin_drift_adaptive_v7_7_market(
        market, model_artifact=model
    )
    assert decision["selected_action"] in {
        "NO_TRADE",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
    }
    assert decision["outcome_or_pnl_field_used_at_inference"] is False
    assert decision["source_score_mutated"] is False
    assert decision["capital_at_risk"] is False

    blocked_market = copy.deepcopy(market)
    blocked_market["baseline_row"]["settlement_pnl"] = 1.0
    blocked = v77.score_rolling_origin_drift_adaptive_v7_7_market(
        blocked_market, model_artifact=model
    )
    assert blocked["selected_action"] == "NO_TRADE"
    assert "v7_7_forbidden_outcome_field_in_inference_row" in blocked[
        "selection_reason_codes"
    ]


def test_config_rejects_issue239_result_paths_by_construction() -> None:
    annotations = v77.RollingOriginDriftAdaptiveV77Config.__annotations__
    assert not any("issue239" in name for name in annotations)
    assert "consumed_stream_settled_index_path" in annotations
