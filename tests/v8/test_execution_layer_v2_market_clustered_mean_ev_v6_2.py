from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2 import (
    apply_market_clustered_mean_ev_scores,
    build_market_clustered_mean_risk_calibration,
    build_target_free_actionability_gate,
    validate_collector_pause_attestation,
    validate_market_clustered_mean_ev_v6_2_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    REQUIRED_ACTIONS,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/execution_layer_v2_market_clustered_mean_ev_v6_2_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def test_profile_freezes_mean_estimand_support_and_safety() -> None:
    profile = _profile()
    validate_market_clustered_mean_ev_v6_2_profile(profile)
    calibration = profile["mean_risk_calibration"]
    assert calibration["individual_outcome_quantile_subtraction_enabled"] is False
    assert calibration["minimum_selected_market_count"] == 40
    assert calibration["minimum_selected_market_count_per_side"] == 20
    assert calibration["bootstrap_resample_count"] == 5000
    assert profile["safety"]["promotion_evidence_eligible"] is False


def test_profile_rejects_individual_quantile_or_relaxed_guard_support() -> None:
    profile = _profile()
    profile["mean_risk_calibration"]["individual_outcome_quantile_subtraction_enabled"] = True
    with pytest.raises(ValueError, match="mean_estimand"):
        validate_market_clustered_mean_ev_v6_2_profile(profile)
    profile = _profile()
    profile["target_free_check"]["minimum_full_guard_accepted_unique_market_count"] = 0
    with pytest.raises(ValueError, match="target_free"):
        validate_market_clustered_mean_ev_v6_2_profile(profile)


def test_mean_risk_calibration_is_side_only_market_clustered_and_deterministic() -> None:
    predictions = []
    targets = []
    for market_index in range(40):
        side = "UP" if market_index % 2 == 0 else "DOWN"
        market_predictions, market_targets = _decision_group(
            market_id=f"market-{market_index:02d}",
            decision_ts=1000 + market_index,
            selected_side=side,
            residual=0.02 if side == "UP" else 0.03,
        )
        predictions.extend(market_predictions)
        targets.extend(market_targets)
    artifact = build_market_clustered_mean_risk_calibration(
        predictions,
        target_rows=targets,
        profile=_profile(),
        model_sha256="a" * 64,
    )
    assert artifact["calibration_gate_passed"] is True
    assert artifact["selected_calibration_market_count"] == 40
    assert artifact["selected_side_distribution"] == {"UP": 20, "DOWN": 20}
    assert artifact["sides"]["UP"]["mean_residual"] == pytest.approx(0.02)
    assert artifact["sides"]["DOWN"]["mean_residual"] == pytest.approx(0.03)
    assert artifact == build_market_clustered_mean_risk_calibration(
        predictions,
        target_rows=targets,
        profile=_profile(),
        model_sha256="a" * 64,
    )


def test_mean_risk_calibration_fails_closed_on_one_sided_support() -> None:
    predictions = []
    targets = []
    for market_index in range(40):
        market_predictions, market_targets = _decision_group(
            market_id=f"market-{market_index:02d}",
            decision_ts=1000 + market_index,
            selected_side="UP",
            residual=0.02,
        )
        predictions.extend(market_predictions)
        targets.extend(market_targets)
    artifact = build_market_clustered_mean_risk_calibration(
        predictions,
        target_rows=targets,
        profile=_profile(),
        model_sha256="a" * 64,
    )
    assert artifact["calibration_gate_passed"] is False
    assert (
        "mean_risk_selected_calibration_side_support_failed"
        in artifact["calibration_gate_blocking_reason_codes"]
    )


def test_mean_ev_score_uses_mean_ucb_and_keeps_targets_sealed() -> None:
    artifact = {
        "sides": {
            "UP": {"mean_residual": 0.01, "mean_residual_upper_confidence_bound": 0.03},
            "DOWN": {"mean_residual": -0.01, "mean_residual_upper_confidence_bound": 0.02},
        }
    }
    rows = [
        {
            "market_id": "market-a",
            "decision_ts": 1000,
            "action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "side": "UP",
            "raw_direct_predicted_net_return": 0.08,
            "guard_compatible_before_ranking": True,
        },
        {
            "market_id": "market-a",
            "decision_ts": 1000,
            "action": "NO_TRADE",
            "side": "NONE",
            "raw_direct_predicted_net_return": 0.0,
            "guard_compatible_before_ranking": True,
        },
    ]
    scored = apply_market_clustered_mean_ev_scores(rows, calibration_artifact=artifact)
    trade = scored[0]
    assert trade["bias_corrected_action_expected_net_return"] == pytest.approx(0.07)
    assert trade["mean_ev_lower_confidence_bound"] == pytest.approx(0.05)
    assert trade["individual_outcome_quantile_subtraction_enabled"] is False
    assert all("target_net_pnl_per_contract" not in row for row in scored)
    assert all(row["promotion_evidence_eligible"] is False for row in scored)


def test_target_free_gate_requires_both_sides_and_full_guard_support() -> None:
    profile = _profile()
    calibration = {
        "calibration_gate_passed": True,
        "calibration_gate_blocking_reason_codes": [],
    }
    selected = [{"market_id": f"up-{index}", "side": "UP"} for index in range(5)] + [
        {"market_id": f"down-{index}", "side": "DOWN"} for index in range(5)
    ]
    accepted = [{"market_id": row["market_id"], "selected_side": row["side"]} for row in selected]
    gate = build_target_free_actionability_gate(
        calibration_artifact=calibration,
        static_selected=selected,
        accepted=accepted,
        profile=profile,
    )
    assert gate["passed"] is True

    one_sided = [row for row in accepted if row["selected_side"] == "UP"]
    blocked = build_target_free_actionability_gate(
        calibration_artifact=calibration,
        static_selected=selected,
        accepted=one_sided,
        profile=profile,
    )
    assert blocked["passed"] is False
    assert "target_free_full_guard_side_support_failed" in blocked["reason_codes"]


def test_collector_pause_attestation_is_fail_closed() -> None:
    attestation = {
        "schema_version": "bigan-v8-persistent-collector-pause-attestation-v1",
        "collector_paused": True,
        "launchd_service_loaded": False,
        "last_completed_batch_sequence": 26,
        "last_batch_canary_passed": True,
        "labels_outcomes_or_pnl_opened": False,
    }
    validate_collector_pause_attestation(attestation)
    running = copy.deepcopy(attestation)
    running["launchd_service_loaded"] = True
    with pytest.raises(ValueError, match="service_unloaded"):
        validate_collector_pause_attestation(running)


def _decision_group(
    *,
    market_id: str,
    decision_ts: int,
    selected_side: str,
    residual: float,
) -> tuple[list[dict], list[dict]]:
    selected_action = (
        "BUY_UP_HOLD_TO_SETTLEMENT" if selected_side == "UP" else "BUY_DOWN_HOLD_TO_SETTLEMENT"
    )
    predictions = []
    targets = []
    for action in REQUIRED_ACTIONS:
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if action.endswith("HOLD_TO_SETTLEMENT")
            else "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "NO_TRADE"
        )
        raw = 0.10 if action == selected_action else 0.01 if action != "NO_TRADE" else 0.0
        predictions.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "action": action,
                "side": side,
                "action_family": family,
                "raw_direct_predicted_net_return": raw,
                "guard_compatible_before_ranking": action != "NO_TRADE",
            }
        )
        targets.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "action": action,
                "target_net_pnl_per_contract": raw - residual if action == selected_action else 0.0,
            }
        )
    return predictions, targets
