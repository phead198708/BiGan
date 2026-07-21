from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_evaluation import (
    apply_v6_7_side_residual_calibration,
    build_v6_7_side_only_confirmatory_gate,
    build_v6_7_side_residual_calibration,
    validate_v6_7_evaluation_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_p_up_semantic_compatibility_v6_7_evaluation_v1.json"
)


def _profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text())


def _calibration_row(index: int, *, side: str) -> dict[str, object]:
    action = f"BUY_{side}_SELL_BEFORE_CLOSE"
    return {
        "market_id": f"calibration-{index:03d}",
        "decision_ts": 1_000_000 + index,
        "max_input_ts": 999_000 + index,
        "side": side,
        "action": action,
        "v6_7_base_score": 0.20,
        "runtime_policy_after_cost_net_pnl_per_contract": 0.15,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def _evaluation_row(index: int, *, side: str, pnl: float) -> dict[str, object]:
    return {
        "market_id": f"future-{index:03d}",
        "decision_ts": 2_000_000 + index,
        "max_input_ts": 1_999_000 + index,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def test_v6_7_evaluation_profile_freezes_side_only_gate() -> None:
    profile = _profile()
    validate_v6_7_evaluation_profile(profile)

    changed = copy.deepcopy(profile)
    changed["confirmatory_side_only_pnl_gate"][
        "minimum_supported_side_unique_market_count"
    ] = 1
    with pytest.raises(ValueError, match="confirmatory"):
        validate_v6_7_evaluation_profile(changed)


def test_v6_7_side_residual_calibration_is_fixed_and_target_isolated() -> None:
    rows = [
        *[_calibration_row(index, side="UP") for index in range(20)],
        *[_calibration_row(index, side="DOWN") for index in range(20, 60)],
    ]
    artifact, calibrated = build_v6_7_side_residual_calibration(
        rows,
        profile=_profile(),
        decision_freeze_descriptor={"path": "/tmp/freeze", "sha256": "1" * 64},
        settled_index_descriptor={"path": "/tmp/index", "sha256": "2" * 64},
        runtime_policy_profile_descriptor={
            "path": "/tmp/runtime",
            "sha256": "3" * 64,
        },
    )

    assert artifact["calibration_gate_passed"] is True
    assert artifact["selected_market_count_by_side"] == {"UP": 20, "DOWN": 40}
    assert artifact["positive_calibrated_lcb_unique_market_count_by_side"] == {
        "UP": 20,
        "DOWN": 40,
    }
    assert artifact["calibration_outcomes_used_as_model_or_decision_inputs"] is False
    assert all(row["v6_7_calibrated_runtime_pnl_lcb"] > 0.0 for row in calibrated)


def test_v6_7_calibration_application_fails_closed_on_future_target_fields() -> None:
    rows = [
        *[_calibration_row(index, side="UP") for index in range(20)],
        *[_calibration_row(index, side="DOWN") for index in range(20, 60)],
    ]
    artifact, _ = build_v6_7_side_residual_calibration(
        rows,
        profile=_profile(),
        decision_freeze_descriptor={"path": "/tmp/freeze", "sha256": "1" * 64},
        settled_index_descriptor={"path": "/tmp/index", "sha256": "2" * 64},
        runtime_policy_profile_descriptor={
            "path": "/tmp/runtime",
            "sha256": "3" * 64,
        },
    )
    target_free = [
        {
            "market_id": "future-1",
            "decision_ts": 3_000,
            "max_input_ts": 2_999,
            "side": "UP",
            "action": "BUY_UP_SELL_BEFORE_CLOSE",
            "v6_7_base_score": 0.20,
        }
    ]
    selected = apply_v6_7_side_residual_calibration(
        target_free, calibration_artifact=artifact
    )
    assert len(selected) == 1
    assert selected[0]["labels_outcomes_resolution_or_pnl_opened"] is False

    target_free[0]["settlement_pnl"] = 1.0
    with pytest.raises(ValueError, match="contains target fields"):
        apply_v6_7_side_residual_calibration(
            target_free, calibration_artifact=artifact
        )


def test_v6_7_confirmatory_gate_is_side_only_and_robustness_checked() -> None:
    candidate = [
        *[_evaluation_row(index, side="UP", pnl=0.02) for index in range(20)],
        *[
            _evaluation_row(index, side="DOWN", pnl=0.02)
            for index in range(20, 40)
        ],
    ]
    baseline = [
        *[_evaluation_row(index, side="UP", pnl=-0.01) for index in range(20)],
        *[
            _evaluation_row(index, side="DOWN", pnl=-0.01)
            for index in range(20, 40)
        ],
    ]
    report = build_v6_7_side_only_confirmatory_gate(
        candidate,
        matched_legacy_rows=baseline,
        evaluation_market_ids=[row["market_id"] for row in candidate],
        profile=_profile(),
        decision_freeze_sha256="4" * 64,
    )

    assert report["confirmatory_side_only_pnl_gate_passed"] is True
    assert report["pnl_hard_gate_aggregation"] == "selected_side_buy_up_buy_down_only"
    assert report["accepted_action_family_metrics"]["SELL_BEFORE_CLOSE"][
        "diagnostic_only"
    ] is True
    assert report["candidate_minus_matched_legacy_market_bootstrap"][
        "lower_confidence_bound"
    ] > 0.0
    assert report["promotion_evidence_eligible"] is False


def test_v6_7_confirmatory_gate_blocks_when_one_supported_side_loses() -> None:
    candidate = [
        *[_evaluation_row(index, side="UP", pnl=-0.02) for index in range(20)],
        *[
            _evaluation_row(index, side="DOWN", pnl=0.05)
            for index in range(20, 40)
        ],
    ]
    report = build_v6_7_side_only_confirmatory_gate(
        candidate,
        matched_legacy_rows=[],
        evaluation_market_ids=[row["market_id"] for row in candidate],
        profile=_profile(),
        decision_freeze_sha256="5" * 64,
    )

    assert report["candidate_after_cost_pnl"] > 0.0
    assert report["confirmatory_side_only_pnl_gate_passed"] is False
    assert "supported_side_post_cost_pnl_gate_failed" in report[
        "confirmatory_side_only_pnl_gate_blocking_reason_codes"
    ]
    assert report["source_model_candidate_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
