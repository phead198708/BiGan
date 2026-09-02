from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_guard_compatible_conformal_net_return_v5 as conformal_v5,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _build_future_evaluation_protocol,
    _group_quantile,
    apply_conformal_scores,
    build_market_grouped_conformal_artifact,
    validate_guard_compatible_conformal_net_return_v5_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_guard_compatible_conformal_net_return_v5_fit_profile.json"
)


def test_issue203_profile_freezes_135_fit_60_calibration_and_side_gate() -> None:
    profile = _profile()

    validate_guard_compatible_conformal_net_return_v5_profile(profile)

    assert profile["roles"]["fit_market_count"] == 135
    assert profile["roles"]["calibration_market_count"] == 60
    assert profile["roles"]["fit"] == ["development_train", "development_calibration"]
    assert profile["roles"]["calibration"] == "confirmatory_validation"
    assert profile["future_evaluation"]["pnl_hard_gate_aggregation"] == (
        "selected_side_buy_up_buy_down_only"
    )
    assert profile["future_evaluation"]["action_and_family_pnl_diagnostic_only"] is True
    assert profile["access_sequence"]["issue_202_oof_or_gate_artifacts_may_be_opened"] is False


def test_issue203_profile_rejects_action_level_hard_gate_or_search() -> None:
    profile = _profile()
    profile["future_evaluation"]["action_and_family_pnl_diagnostic_only"] = False
    with pytest.raises(ValueError, match="future"):
        validate_guard_compatible_conformal_net_return_v5_profile(profile)

    profile = _profile()
    profile["conformal_calibration"]["calibration_threshold_search_enabled"] = True
    with pytest.raises(ValueError, match="calibration"):
        validate_guard_compatible_conformal_net_return_v5_profile(profile)


def test_issue203_quantile_uses_maximum_residual_per_market() -> None:
    rows = [
        {"market_id": "a", "residual": -0.2},
        {"market_id": "a", "residual": 0.4},
        {"market_id": "b", "residual": 0.1},
        {"market_id": "b", "residual": 0.2},
    ]

    result = _group_quantile(rows, alpha=0.25, group_name="test")

    assert result["market_count"] == 2
    assert result["quantile_rank"] == 2
    assert result["quantile"] == 0.4
    assert result["empirical_market_simultaneous_coverage"] == 1.0


def test_issue203_artifact_is_action_symmetric_and_does_not_compute_policy_pnl() -> None:
    predictions, targets = _calibration_rows(market_count=30)

    artifact = build_market_grouped_conformal_artifact(
        predictions,
        target_rows=targets,
        profile=_profile(),
        feature_contract_sha256="a" * 64,
    )

    assert set(artifact["actions"]) == set(REQUIRED_ACTIONS)
    assert artifact["policy_pnl_computed_on_calibration"] is False
    assert artifact["calibration_threshold_search_enabled"] is False
    assert artifact["candidate_comparison_enabled"] is False
    assert all(
        artifact["actions"][action]["calibration_source"] == "action"
        for action in REQUIRED_ACTIONS
        if action != "NO_TRADE"
    )
    assert artifact["actions"]["NO_TRADE"]["calibration_penalty"] == 0.0


def test_issue203_nonzero_no_trade_target_fails_closed() -> None:
    predictions, targets = _calibration_rows(market_count=30)
    no_trade = next(row for row in targets if row["action"] == "NO_TRADE")
    no_trade["target_net_pnl_per_contract"] = 0.01

    with pytest.raises(ValueError, match="exact zero anchor"):
        build_market_grouped_conformal_artifact(
            predictions,
            target_rows=targets,
            profile=_profile(),
            feature_contract_sha256="a" * 64,
        )


def test_issue203_conformal_score_keeps_model_value_and_masks_incompatible_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, targets = _calibration_rows(market_count=30)
    artifact = build_market_grouped_conformal_artifact(
        predictions,
        target_rows=targets,
        profile=_profile(),
        feature_contract_sha256="b" * 64,
    )
    one_group = [row for row in predictions if row["market_id"] == "market-000"]

    def fake_universe(rows: list[dict]) -> list[dict]:
        return [
            {
                **row,
                "p_up_alignment_passed": row["action"] != "BUY_DOWN_HOLD_TO_SETTLEMENT",
                "execution_quality_only_passed": row["action"] != "NO_TRADE",
            }
            for row in rows
        ]

    monkeypatch.setattr(
        conformal_v5,
        "build_execution_compatible_action_universe",
        fake_universe,
    )

    scored = apply_conformal_scores(
        one_group,
        calibration_artifact=artifact,
        profile=_profile(),
    )
    by_action = {row["action"]: row for row in scored}

    blocked = by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]
    assert blocked["raw_direct_predicted_net_return"] == 0.25
    assert blocked["action_selection_score"] == -1_000_000.0
    assert by_action["NO_TRADE"]["action_selection_score"] == 0.0
    assert all(row["target_used_as_decision_input"] is False for row in scored)
    assert not any("target_net_pnl_per_contract" in row for row in scored)


def test_issue203_future_protocol_is_powered_side_only_and_fail_closed() -> None:
    protocol = _build_future_evaluation_protocol(
        run_id="test",
        profile=_profile(),
        candidate_freeze_created_ts=1_000,
        calibration_gate_passed=True,
        power_report={
            "recommended_quality_valid_market_count": 220,
            "recommended_required_accepted_unique_market_count": 88,
        },
    )

    assert protocol["future_evaluation_allowed"] is True
    assert protocol["required_quality_valid_market_count"] == 220
    assert protocol["required_guard_accepted_unique_market_count"] == 88
    assert protocol["required_checks"]["pnl_hard_gate_aggregation"] == (
        "selected_side_buy_up_buy_down_only"
    )
    assert protocol["required_checks"]["action_and_family_pnl_diagnostic_only"] is True
    assert protocol["source_model_candidate_eligible"] is False
    assert protocol["promotion_evidence_eligible"] is False
    assert protocol["v8_execution_handoff_allowed"] is False
    assert protocol["#134_resume_allowed"] is False
    assert protocol["#146_start_allowed"] is False


def _calibration_rows(*, market_count: int) -> tuple[list[dict], list[dict]]:
    predictions = []
    targets = []
    for market_index in range(market_count):
        market_id = f"market-{market_index:03d}"
        decision_ts = 1_000 + market_index
        for action in REQUIRED_ACTIONS:
            side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
            family = (
                "HOLD_TO_SETTLEMENT"
                if action.endswith("HOLD_TO_SETTLEMENT")
                else "SELL_BEFORE_CLOSE"
                if action.endswith("SELL_BEFORE_CLOSE")
                else "NO_TRADE"
            )
            raw = 0.0 if action == "NO_TRADE" else 0.25
            target = 0.0 if action == "NO_TRADE" else 0.1 + market_index / 1_000.0
            base = {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "action": action,
                "side": side,
                "action_family": family,
                "max_input_ts": decision_ts,
                "p_up": 0.6,
                "p_down": 0.4,
                "p_up_action_disagreement": side == "DOWN",
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
            }
            predictions.append(
                {
                    **base,
                    "raw_model_prediction": raw,
                    "raw_direct_predicted_net_return": raw,
                }
            )
            targets.append({**base, "target_net_pnl_per_contract": target})
    return predictions, targets


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
