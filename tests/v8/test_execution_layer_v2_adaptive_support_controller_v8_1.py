from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1 as v81,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_adaptive_support_controller_v8_1_profile.json"
)


def _profile() -> dict[str, Any]:
    return json.loads(PROFILE_PATH.read_text())


def test_profile_freezes_outcome_free_controller_and_unchanged_model() -> None:
    profile = _profile()
    v81.validate_adaptive_support_controller_v8_1_profile(profile)
    rank = profile["rank_abstention_contract"]
    assert rank["eligible_prior_score_window"] == 60
    assert rank["controller_observation_window"] == 20
    assert rank["controller_initial_quantile"] == 0.4
    assert rank["controller_low_support_quantile"] == 0.25
    assert rank["controller_balanced_support_quantile"] == 0.4
    assert rank["controller_high_support_quantile"] == 0.5
    assert rank["controller_target_outcome_pnl_free"] is True
    assert profile["model_contract"]["fixed_edge_buffer"] == 0.025
    assert profile["historical_prequential_hard_gate"][
        "minimum_candidate_guard_accepted_unique_market_count"
    ] == 40

    changed = copy.deepcopy(profile)
    changed["rank_abstention_contract"]["controller_low_support_quantile"] = 0.24
    with pytest.raises(ValueError, match="rank"):
        v81.validate_adaptive_support_controller_v8_1_profile(changed)


@pytest.mark.parametrize(
    ("history", "expected_band", "expected_quantile"),
    [
        ([False] * 19, "initial_q40", 0.4),
        ([True] * 6 + [False] * 14, "low_support_q25", 0.25),
        ([True] * 8 + [False] * 12, "balanced_support_q40", 0.4),
        ([True] * 10 + [False] * 10, "balanced_support_q40", 0.4),
        ([True] * 11 + [False] * 9, "high_support_q50", 0.5),
    ],
)
def test_controller_bands_use_strictly_prior_guard_support(
    history: list[bool], expected_band: str, expected_quantile: float
) -> None:
    decision = v81._support_controller_decision(tuple(history))
    assert decision["controller_band"] == expected_band
    assert decision["selected_quantile"] == expected_quantile
    assert decision["current_market_score_used"] is False
    assert decision["current_market_guard_result_used"] is False
    assert decision["target_outcome_label_or_pnl_used"] is False


def test_fit_wrapper_uses_controller_hook_and_rebinds_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_fit(**kwargs: Any) -> dict[str, Any]:
        called.update(kwargs)
        kwargs["_profile_validator"](kwargs["profile"])
        controller = kwargs["_rank_controller"]((False,) * 20)
        return {
            "model_artifact": {
                "schema_version": v81.v79.MODEL_SCHEMA_VERSION,
                "candidate_name": v81.v79.CANDIDATE_NAME,
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
                    "rank_controller_decision": controller,
                }
            ],
            "guard_replay_rows": [
                {
                    "prior_rolling_score_q60": 0.01,
                    "point_selected_score_at_or_above_prior_q60": True,
                    "rank_controller_decision": controller,
                }
            ],
            "controller_guard_acceptance_history": [False] * 20,
        }

    monkeypatch.setattr(
        v81.v79, "fit_rolling_origin_score_rank_abstention_v7_9", fake_fit
    )
    fit = v81.fit_adaptive_support_controller_v8_1(profile=_profile())
    model = fit["model_artifact"]
    assert model["schema_version"] == v81.MODEL_SCHEMA_VERSION
    assert model["candidate_name"] == v81.CANDIDATE_NAME
    assert (
        model["final_rank_state"]["next_controller_decision"]["controller_band"]
        == "low_support_q25"
    )
    assert (
        fit["prequential_rows"][0][
            "point_selected_score_at_or_above_controller_threshold"
        ]
        is True
    )
    assert "_rank_controller" in called


def test_target_free_score_requires_post_guard_controller_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_score(
        market: dict[str, Any],
        *,
        model_artifact: dict[str, Any],
        prior_rank_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del market
        observed["model"] = model_artifact
        observed["state"] = prior_rank_state
        return {
            "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
            "prior_rolling_score_q60": 0.02,
            "point_selected_score_at_or_above_prior_q60": True,
            "next_rank_state": {
                "schema_version": "bigan-v8-rolling-origin-score-rank-state-v7-9-v1",
                "rolling_score_q60": 0.03,
                "eligible_prediction_scores": [0.01] * 60,
                "eligible_prior_score_window": 60,
            },
            "outcome_or_pnl_field_used_at_inference": False,
            "source_score_mutated": False,
            "paper_only": True,
            "capital_at_risk": False,
        }

    monkeypatch.setattr(
        v81.v79,
        "score_rolling_origin_score_rank_abstention_v7_9_market",
        fake_score,
    )
    model = {
        "schema_version": v81.MODEL_SCHEMA_VERSION,
        "final_rank_state": {
            "schema_version": "bigan-v8-adaptive-support-controller-state-v8-1-v1",
            "rolling_score_controller_threshold": 0.01,
            "eligible_prediction_scores": [0.01] * 60,
            "eligible_prior_score_window": 60,
            "controller_guard_acceptance_history": [False] * 20,
        },
    }
    result = v81.score_adaptive_support_controller_v8_1_market(
        {}, model_artifact=model
    )
    assert observed["state"]["rolling_score_quantile"] == 0.25
    assert result["rank_controller_decision"]["controller_band"] == "low_support_q25"
    assert result["next_rank_state"][
        "pending_current_guard_observation_requires_post_guard_advance"
    ]
    assert result["next_rank_state"]["controller_guard_acceptance_history"] == [
        False
    ] * 20

    advanced = v81.advance_adaptive_support_controller_v8_1_state(
        result["next_rank_state"], current_guard_accepted=True
    )
    assert advanced["controller_guard_acceptance_history"][-1] is True
    assert (
        advanced["pending_current_guard_observation_requires_post_guard_advance"]
        is False
    )
    assert advanced["capital_at_risk"] is False


def test_forbidden_prior_issue_paths_are_excluded_by_construction() -> None:
    names = {field.name for field in fields(v81.AdaptiveSupportControllerV81Config)}
    assert not any(
        f"issue{issue}" in name for name in names for issue in (241, 242, 243, 244)
    )
    assert "consumed_stream_settled_index_path" in names
