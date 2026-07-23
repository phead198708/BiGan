from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_rolling_origin_score_rank_abstention_v7_9 as v79,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_rolling_origin_score_rank_abstention_v7_9_profile.json"
)


def _profile() -> dict[str, Any]:
    return json.loads(PROFILE_PATH.read_text())


def _features(*, side: str, score: float) -> dict[str, float]:
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
    rows = []
    for index in range(134):
        decision_ts = 1_000_000 + index * 10_000
        for side in ("UP", "DOWN"):
            rows.append(
                {
                    "source": "test_seed",
                    "market_id": f"seed-{index:03d}",
                    "decision_group_id": f"seed-{index:03d}|{decision_ts}",
                    "decision_ts": decision_ts,
                    "max_input_ts": decision_ts - 1,
                    "market_close_ts": decision_ts + 5_000,
                    "role": "historical_development",
                    "action_family": "SELL_BEFORE_CLOSE",
                    "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
                    "side": side,
                    "decision_time_features": _features(side=side, score=0.0),
                    "target_after_cost_net_pnl_per_contract": 0.1,
                    "target_used_as_decision_time_input": False,
                    "target_available_only_post_exit_or_official_resolution": True,
                }
            )
    return rows


def _feature_row(
    market_id: str, *, decision_ts: int, side: str, score: float
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
        "microstructure_snapshot": {
            "spread_bps": 100.0,
            "book_staleness_ms": 100.0,
            "queue_fill_proxy": 0.9,
            "time_to_close_seconds": 240.0,
        },
        "decision_time_features": _features(side=side, score=score),
        "target_used_as_decision_time_input": False,
    }


def _guard_source(row: dict[str, Any], *, pass_guard: bool) -> dict[str, Any]:
    source = copy.deepcopy(row)
    source["decision_time_features"].update(
        {
            "selected_side_executable_ask_notional": 100.0,
            "selected_side_executable_bid_notional": 100.0,
            "selected_side_liquidity_depth": 1000.0,
        }
    )
    if not pass_guard:
        source["microstructure_snapshot"]["spread_bps"] = 10_000.0
    source["reference_price_feature_provenance"] = {"provenance_valid": True}
    return source


def _stream_markets(*, pass_guard: bool = True) -> list[dict[str, Any]]:
    rows = []
    start = 3_000_000
    for index in range(120):
        decision_ts = start + index * 10_000
        market_id = f"stream-{index:03d}"
        baseline = _feature_row(
            market_id, decision_ts=decision_ts, side="UP", score=0.0
        )
        opposite = _feature_row(
            market_id, decision_ts=decision_ts, side="DOWN", score=0.3
        )
        rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "market_close_ts": decision_ts + 5_000,
                "baseline_row": baseline,
                "opposite_row": opposite,
                "guard_source_by_action": {
                    baseline["action"]: _guard_source(
                        baseline, pass_guard=pass_guard
                    ),
                    opposite["action"]: _guard_source(
                        opposite, pass_guard=pass_guard
                    ),
                },
            }
        )
    return rows


def _target(row: dict[str, Any], value: float) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "action": row["action"],
        "runtime_policy_after_cost_net_pnl_per_contract": value,
    }


def _guard_profile() -> dict[str, Any]:
    return {
        "hard_execution_safety": {
            "max_spread_bps": 1200.0,
            "max_book_staleness_ms": 5000.0,
            "min_queue_fill_probability_proxy": 0.5,
            "min_time_to_close_seconds": 60.0,
        }
    }


def _fit(
    monkeypatch: pytest.MonkeyPatch, *, pass_guard: bool = True
) -> dict[str, Any]:
    monkeypatch.setattr(
        v79.v77,
        "_fit_weighted_model",
        lambda *args, **kwargs: {
            "available": True,
            "booster_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        v79.v77,
        "_predict",
        lambda row, artifact: float(row["decision_time_features"]["action_score"]),
    )

    def load(
        baseline: dict[str, Any], opposite: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return _target(baseline, -0.2), _target(opposite, 0.5)

    return v79.fit_rolling_origin_score_rank_abstention_v7_9(
        seed_rows=_seed_rows(),
        stream_markets=_stream_markets(pass_guard=pass_guard),
        target_loader=load,
        profile=_profile(),
        v6_7_profile=_guard_profile(),
        implementation_commit="a" * 40,
        fit_created_ts=5_000_000,
    )


def test_profile_freezes_support_derived_q60_and_last60_window() -> None:
    profile = _profile()
    v79.validate_rolling_origin_score_rank_abstention_v7_9_profile(profile)
    contract = profile["rank_abstention_contract"]
    assert contract["eligible_prior_score_window"] == 60
    assert contract["rolling_score_quantile"] == 0.6
    assert v79.finite_sample_score_quantile(list(range(10)), 0.6) == 5.0

    changed = copy.deepcopy(profile)
    changed["rank_abstention_contract"]["rolling_score_quantile"] = 0.5
    with pytest.raises(ValueError, match="rank"):
        v79.validate_rolling_origin_score_rank_abstention_v7_9_profile(changed)


def test_strictly_prior_rank_policy_passes_noninferior_historical_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(monkeypatch)
    model = fit["model_artifact"]
    assert model["historical_hard_gate_passed"] is True
    assert model["historical_candidate_guard_accepted_unique_market_count"] == 120
    assert model["historical_policy_difference_market_count"] == 120
    assert model["target_free_canary_collection_allowed"] is True
    assert model["paper_candidate_allowed"] is False
    assert model["promotion_evidence_eligible"] is False
    assert all(
        row["rank_state_max_decision_ts"] < row["baseline_decision_ts"]
        and row["current_market_score_used_for_threshold"] is False
        for row in fit["prequential_rows"]
    )
    state = model["final_rank_state"]
    assert state["eligible_prediction_score_count"] == 214
    assert len(state["eligible_prediction_scores"]) == 60
    assert state["rank_state_uses_target_outcome_or_pnl"] is False


def test_full_guard_blocks_support_without_fake_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(monkeypatch, pass_guard=False)
    model = fit["model_artifact"]
    assert model["historical_hard_gate_passed"] is False
    assert model["historical_candidate_guard_accepted_unique_market_count"] == 0
    assert "historical_candidate_guard_accepted_support_insufficient" in model[
        "historical_gate_blocking_reason_codes"
    ]
    assert model["target_free_canary_collection_allowed"] is False


def test_target_free_scoring_threads_outcome_free_rank_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v79.v77,
        "_predict",
        lambda row, artifact: float(row["decision_time_features"]["action_score"]),
    )
    prior_scores = [0.0] * 60
    model = {
        "schema_version": v79.MODEL_SCHEMA_VERSION,
        "historical_hard_gate_passed": True,
        "target_free_canary_collection_allowed": True,
        "final_weighted_model": {"available": True},
        "final_rank_state": {
            "eligible_prior_score_window": 60,
            "rolling_score_quantile": 0.6,
            "eligible_prediction_scores": prior_scores,
        },
    }
    market = _stream_markets()[0]
    decision = v79.score_rolling_origin_score_rank_abstention_v7_9_market(
        market, model_artifact=model
    )
    assert decision["selected_action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
    assert decision["outcome_or_pnl_field_used_at_inference"] is False
    assert decision["source_score_mutated"] is False
    assert decision["next_rank_state"]["eligible_prediction_scores"][-1] == 0.3
    assert decision["capital_at_risk"] is False

    blocked_market = copy.deepcopy(market)
    blocked_market["baseline_row"]["settlement_pnl"] = 1.0
    blocked = v79.score_rolling_origin_score_rank_abstention_v7_9_market(
        blocked_market, model_artifact=model
    )
    assert blocked["selected_action"] == "NO_TRADE"
    assert "v7_9_forbidden_outcome_field_in_inference_row" in blocked[
        "selection_reason_codes"
    ]


def test_issue241_and_issue242_paths_are_excluded_by_construction() -> None:
    names = {field.name for field in fields(v79.RollingOriginScoreRankAbstentionV79Config)}
    assert not any("issue241" in name or "issue242" in name for name in names)
    assert "consumed_stream_settled_index_path" in names
