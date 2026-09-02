from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_rolling_origin_support_budgeted_score_rank_v8_0 as v80,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_rolling_origin_support_budgeted_score_rank_v8_0_profile.json"
)


def _profile() -> dict[str, Any]:
    return json.loads(PROFILE_PATH.read_text())


def test_profile_freezes_support_budgeted_q40_without_model_drift() -> None:
    profile = _profile()
    v80.validate_rolling_origin_support_budgeted_score_rank_v8_0_profile(profile)
    rank = profile["rank_abstention_contract"]
    assert rank["eligible_prior_score_window"] == 60
    assert rank["rolling_score_quantile"] == 0.4
    assert rank["quantile_selected_from_outcome_free_support_budget_only"] is True
    assert profile["model_contract"]["fixed_edge_buffer"] == 0.025
    assert profile["historical_prequential_hard_gate"][
        "minimum_candidate_guard_accepted_unique_market_count"
    ] == 40

    changed = copy.deepcopy(profile)
    changed["rank_abstention_contract"]["rolling_score_quantile"] = 0.5
    with pytest.raises(ValueError, match="rank"):
        v80.validate_rolling_origin_support_budgeted_score_rank_v8_0_profile(changed)


def test_fit_wrapper_uses_v79_engine_and_rebinds_v80_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_fit(**kwargs: Any) -> dict[str, Any]:
        called.update(kwargs)
        kwargs["_profile_validator"](kwargs["profile"])
        return {
            "model_artifact": {
                "schema_version": v80.v79.MODEL_SCHEMA_VERSION,
                "candidate_name": v80.v79.CANDIDATE_NAME,
                "final_rank_state": {
                    "schema_version": (
                        "bigan-v8-rolling-origin-score-rank-state-v7-9-v1"
                    ),
                    "rolling_score_q60": 0.01,
                },
            },
            "prequential_rows": [
                {
                    "prior_rolling_score_q60": 0.01,
                    "point_selected_score_at_or_above_prior_q60": True,
                }
            ],
            "guard_replay_rows": [
                {
                    "prior_rolling_score_q60": 0.01,
                    "point_selected_score_at_or_above_prior_q60": True,
                }
            ],
        }

    monkeypatch.setattr(
        v80.v79, "fit_rolling_origin_score_rank_abstention_v7_9", fake_fit
    )
    fit = v80.fit_rolling_origin_support_budgeted_score_rank_v8_0(
        profile=_profile()
    )
    model = fit["model_artifact"]
    assert model["schema_version"] == v80.MODEL_SCHEMA_VERSION
    assert model["candidate_name"] == v80.CANDIDATE_NAME
    assert model["final_rank_state"]["rolling_score_q40"] == 0.01
    assert "rolling_score_q60" not in model["final_rank_state"]
    assert fit["prequential_rows"][0]["prior_rolling_score_q40"] == 0.01
    assert (
        fit["prequential_rows"][0][
            "point_selected_score_at_or_above_prior_q40"
        ]
        is True
    )
    assert (
        model["issue243_pnl_targets_winners_or_losers_opened_for_threshold_selection"]
        is False
    )
    assert "_profile_validator" in called


def test_target_free_wrapper_preserves_safety_and_threads_q40_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_score(
        market: dict[str, Any],
        *,
        model_artifact: dict[str, Any],
        prior_rank_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del market, prior_rank_state
        observed.update(model_artifact)
        return {
            "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
            "prior_rolling_score_q60": 0.02,
            "point_selected_score_at_or_above_prior_q60": True,
            "next_rank_state": {
                "schema_version": "bigan-v8-rolling-origin-score-rank-state-v7-9-v1",
                "rolling_score_q60": 0.03,
            },
            "outcome_or_pnl_field_used_at_inference": False,
            "source_score_mutated": False,
            "paper_only": True,
            "capital_at_risk": False,
        }

    monkeypatch.setattr(
        v80.v79,
        "score_rolling_origin_score_rank_abstention_v7_9_market",
        fake_score,
    )
    model = {
        "schema_version": v80.MODEL_SCHEMA_VERSION,
        "final_rank_state": {
            "schema_version": "bigan-v8-rolling-origin-score-rank-state-v8-0-v1",
            "rolling_score_q40": 0.01,
            "quantile_selected_from_outcome_free_support_budget_only": True,
        },
    }
    result = v80.score_rolling_origin_support_budgeted_score_rank_v8_0_market(
        {}, model_artifact=model
    )
    assert observed["schema_version"] == v80.v79.MODEL_SCHEMA_VERSION
    assert result["prior_rolling_score_q40"] == 0.02
    assert result["next_rank_state"]["rolling_score_q40"] == 0.03
    assert result["outcome_or_pnl_field_used_at_inference"] is False
    assert result["source_score_mutated"] is False
    assert result["capital_at_risk"] is False


def test_forbidden_prior_issue_paths_are_excluded_by_construction() -> None:
    names = {
        field.name
        for field in fields(v80.RollingOriginSupportBudgetedScoreRankV80Config)
    }
    assert not any(
        "issue241" in name or "issue242" in name or "issue243" in name
        for name in names
    )
    assert "consumed_stream_settled_index_path" in names
