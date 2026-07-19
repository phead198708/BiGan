from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_v6_on_v5_target_free_diagnostic import (
    _normalize_v5_labeled_rows,
    validate_v6_on_v5_diagnostic_profile,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/execution_layer_v2_v6_on_v5_target_free_diagnostic_v1.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _labeled_rows() -> list[dict]:
    actions = (
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    )
    return [
        {
            "market_id": "market-1",
            "decision_ts": 100,
            "max_input_ts": 99,
            "action": action,
            "role": "development_train",
            "decision_time_features": {"feature_a": float(index)},
            "target_net_pnl_per_contract": 0.1,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        for index, action in enumerate(actions)
    ]


def test_profile_freezes_v5_roles_target_free_check_and_safety() -> None:
    profile = _profile()
    validate_v6_on_v5_diagnostic_profile(profile)
    assert profile["chronological_roles"] == {
        "point_model_fit_market_count": 135,
        "policy_selected_conformal_market_count": 60,
        "target_free_check_market_count": 50,
        "assignment": "v5_fit_then_v5_calibration_then_strictly_later_post_issue204_check",
    }
    assert profile["target_free_check"]["minimum_full_guard_accepted_market_count_per_side"] == 5
    assert profile["prohibited_inputs"]["uses_204_outcomes_for_fitting"] is False
    assert profile["prohibited_inputs"]["uses_204_pnl_for_tuning"] is False
    assert profile["safety"]["promotion_evidence_eligible"] is False
    assert profile["safety"]["paper_candidate_allowed"] is False


def test_profile_rejects_relaxed_safety_or_result_dependent_extension() -> None:
    profile = _profile()
    profile["safety"]["promotion_evidence_eligible"] = True
    with pytest.raises(ValueError, match="safety"):
        validate_v6_on_v5_diagnostic_profile(profile)
    profile = _profile()
    profile["target_free_check"]["result_dependent_extension_allowed"] = True
    with pytest.raises(ValueError, match="target_free_sealed"):
        validate_v6_on_v5_diagnostic_profile(profile)


def test_v5_role_adapter_preserves_complete_grid_and_blocks_feature_leakage() -> None:
    rows = _labeled_rows()
    normalized = _normalize_v5_labeled_rows(
        rows,
        role="point_model_fit",
        expected_source_roles={"development_train"},
        feature_columns=("feature_a",),
    )
    assert {row["action"] for row in normalized} == {
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    }
    assert {row["role"] for row in normalized} == {"point_model_fit"}
    assert all(row["target_used_as_decision_input"] is False for row in normalized)

    leaking = copy.deepcopy(rows)
    leaking[0]["decision_time_features"]["target_net_pnl_per_contract"] = 0.8
    with pytest.raises(ValueError, match="leaked"):
        _normalize_v5_labeled_rows(
            leaking,
            role="point_model_fit",
            expected_source_roles={"development_train"},
            feature_columns=("feature_a",),
        )


def test_v5_role_adapter_rejects_causality_and_role_mismatch() -> None:
    rows = _labeled_rows()
    rows[0]["max_input_ts"] = 101
    with pytest.raises(ValueError, match="causality"):
        _normalize_v5_labeled_rows(
            rows,
            role="point_model_fit",
            expected_source_roles={"development_train"},
            feature_columns=("feature_a",),
        )
    rows = _labeled_rows()
    rows[0]["role"] = "confirmatory_validation"
    with pytest.raises(ValueError, match="source role"):
        _normalize_v5_labeled_rows(
            rows,
            role="point_model_fit",
            expected_source_roles={"development_train"},
            feature_columns=("feature_a",),
        )
