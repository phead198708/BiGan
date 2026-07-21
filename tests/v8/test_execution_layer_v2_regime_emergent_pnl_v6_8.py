from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    build_regime_emergent_target_free_support,
    build_v6_8_pooled_residual_calibration,
    build_v6_8_regime_emergent_confirmatory_gate,
    validate_regime_emergent_pnl_v6_8_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_regime_emergent_pnl_v6_8_evaluation_v1.json"
)


def _profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text())


def _decision(index: int, *, side: str) -> dict[str, object]:
    action = f"BUY_{side}_SELL_BEFORE_CLOSE"
    decision_ts = 1_000_000 + index
    return {
        "market_id": f"market-{index}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "side": side,
        "action": action,
        "v6_7_base_score": 0.12,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "source_score_mutated": False,
    }


def _target_row(index: int, *, side: str) -> dict[str, object]:
    return {
        **_decision(index, side=side),
        "runtime_policy_after_cost_net_pnl_per_contract": 0.08,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def _evaluation_row(index: int, *, side: str, pnl: float) -> dict[str, object]:
    row = _decision(index, side=side)
    return {
        **row,
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def test_profile_preregisters_no_side_count_or_side_pnl_hard_gate() -> None:
    profile = _profile()
    validate_regime_emergent_pnl_v6_8_profile(profile)

    assert profile["sealed_calibration_window"]["side_quota_enforced"] is False
    assert profile["fresh_calibration"][
        "minimum_selected_unique_market_count_per_side"
    ] is None
    assert profile["future_confirmatory"]["required_supported_sides"] == []
    assert profile["future_confirmatory"][
        "side_count_and_side_pnl_diagnostic_only"
    ] is True

    drift = copy.deepcopy(profile)
    drift["fresh_calibration"]["minimum_selected_unique_market_count_per_side"] = 20
    with pytest.raises(ValueError, match="calibration"):
        validate_regime_emergent_pnl_v6_8_profile(drift)


def test_target_free_support_accepts_regime_emergent_15_up_45_down() -> None:
    rows = [
        *[_decision(index, side="UP") for index in range(15)],
        *[_decision(index, side="DOWN") for index in range(15, 60)],
    ]
    support = build_regime_emergent_target_free_support(
        rows,
        exact_window_market_count=60,
        required_total_market_count=60,
        score_field="v6_7_base_score",
    )

    assert support["target_free_support_gate_passed"] is True
    assert support["count_by_side"] == {"DOWN": 45, "UP": 15}
    assert support["minimum_per_side_required"] is None
    assert support["side_count_hard_gate_enabled"] is False


def test_pooled_calibration_has_no_side_support_gate() -> None:
    rows = [
        *[_target_row(index, side="UP") for index in range(15)],
        *[_target_row(index, side="DOWN") for index in range(15, 60)],
    ]
    artifact, calibrated = build_v6_8_pooled_residual_calibration(
        rows,
        profile=_profile(),
        decision_freeze_descriptor={"path": "/decision", "sha256": "a" * 64},
        settled_index_descriptor={"path": "/settled", "sha256": "b" * 64},
        runtime_policy_profile_descriptor={
            "path": "/runtime",
            "sha256": "c" * 64,
        },
    )

    assert artifact["calibration_gate_passed"] is True
    assert artifact["selected_market_count_by_side_diagnostic"] == {
        "DOWN": 45,
        "UP": 15,
    }
    assert artifact["side_count_hard_gate_enabled"] is False
    assert artifact["calibration_gate_checks"]["side_quota_disabled"] is True
    assert len(calibrated) == 60
    assert len(
        {row["pooled_residual_upper_confidence_bound"] for row in calibrated}
    ) == 1


def test_confirmatory_gate_can_pass_one_sided_regime_when_pnl_is_robust() -> None:
    candidate = [
        _evaluation_row(index, side="UP", pnl=0.10) for index in range(40)
    ]
    legacy = [
        _evaluation_row(index, side="UP", pnl=0.01) for index in range(40)
    ]
    report = build_v6_8_regime_emergent_confirmatory_gate(
        candidate,
        matched_legacy_rows=legacy,
        evaluation_market_ids=[f"market-{index}" for index in range(120)],
        profile=_profile(),
        decision_freeze_sha256="d" * 64,
    )

    assert report["confirmatory_execution_pnl_gate_passed"] is True
    assert report["accepted_side_distribution_diagnostic"] == {"UP": 40}
    assert report["side_count_hard_gate_enabled"] is False
    assert report["side_pnl_hard_gate_enabled"] is False
    assert report["largest_winner_removed_after_cost_pnl"] > 0.0
    assert report["candidate_minus_matched_legacy_market_bootstrap"][
        "lower_confidence_bound"
    ] > 0.0
    for key in (
        "v8_execution_handoff_allowed",
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "#134_resume_allowed",
        "#146_start_allowed",
    ):
        assert report[key] is False


def test_confirmatory_gate_still_fails_closed_on_negative_total_pnl() -> None:
    candidate = [
        _evaluation_row(index, side="DOWN", pnl=-0.01) for index in range(40)
    ]
    report = build_v6_8_regime_emergent_confirmatory_gate(
        candidate,
        matched_legacy_rows=[],
        evaluation_market_ids=[f"market-{index}" for index in range(120)],
        profile=_profile(),
        decision_freeze_sha256="e" * 64,
    )

    assert report["confirmatory_execution_pnl_gate_passed"] is False
    assert "accepted_total_after_cost_pnl_not_positive" in report[
        "confirmatory_execution_pnl_gate_blocking_reason_codes"
    ]
    assert report["promotion_evidence_eligible"] is False


def test_target_free_support_rejects_target_leakage() -> None:
    rows = [_decision(index, side="UP") for index in range(60)]
    rows[0]["settlement_pnl"] = 1.0
    support = build_regime_emergent_target_free_support(
        rows,
        exact_window_market_count=60,
        required_total_market_count=60,
        score_field="v6_7_base_score",
    )

    assert support["target_free_support_gate_passed"] is False
    assert "targets_sealed_gate_failed" in support["blocking_reason_codes"]
