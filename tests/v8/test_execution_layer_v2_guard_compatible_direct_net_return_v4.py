from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    _attach_predictions_and_mask,
    _build_gate_report,
    _strip_target_fields,
    validate_guard_compatible_direct_net_return_v4_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_guard_compatible_direct_net_return_v4_fit_profile.json"
)


def test_issue202_profile_freezes_one_shot_model_mask_and_evidence_scope() -> None:
    profile = _profile()

    validate_guard_compatible_direct_net_return_v4_profile(profile)

    assert profile["model"]["objective"] == "reg:squarederror"
    assert profile["model"]["hyperparameter_search_enabled"] is False
    assert profile["decision_rule"]["p_up_side_alignment_required"] is True
    assert profile["decision_rule"]["execution_guard_mutation_allowed"] is False
    assert profile["development_gate"]["pnl_hard_gate_aggregation"] == (
        "selected_side_buy_up_buy_down_only"
    )
    assert profile["development_gate"]["action_and_action_family_pnl_diagnostic_only"] is True
    assert profile["access_sequence"]["development_calibration_files_may_be_opened"] is False
    assert profile["access_sequence"]["confirmatory_files_may_be_opened"] is False
    assert profile["access_sequence"]["issue_190_or_192_future_files_may_be_opened"] is False
    assert profile["output_contract"]["strictly_later_persistent_window_required"] is True
    assert profile["safety"]["source_model_candidate_eligible"] is False


def test_issue202_profile_rejects_model_or_guard_tuning() -> None:
    profile = _profile()
    profile["model"]["eta"] = 0.04
    with pytest.raises(ValueError, match="fixed_model"):
        validate_guard_compatible_direct_net_return_v4_profile(profile)

    profile = _profile()
    profile["decision_rule"]["execution_guard_mutation_allowed"] = True
    with pytest.raises(ValueError, match="decision"):
        validate_guard_compatible_direct_net_return_v4_profile(profile)


def test_issue202_mask_is_applied_before_argmax_without_mutating_model_score() -> None:
    profile = _profile()
    rows = _decision_rows()
    predictions = [0.3, 0.9, 0.2, 0.8, 0.7]
    compatibility = {
        _key(row): row["action"] in {"BUY_UP_HOLD_TO_SETTLEMENT", "BUY_UP_SELL_BEFORE_CLOSE"}
        for row in rows
    }

    output = _attach_predictions_and_mask(
        rows,
        predictions,
        compatibility=compatibility,
        profile=profile,
        fold_index=1,
    )
    by_action = {row["action"]: row for row in output}

    assert by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]["direct_predicted_net_return"] == 0.9
    assert by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]["action_selection_score"] == -1_000_000.0
    assert by_action["BUY_UP_HOLD_TO_SETTLEMENT"]["action_selection_score"] == 0.3
    assert by_action["NO_TRADE"]["action_selection_score"] == 0.0
    assert by_action["NO_TRADE"]["direct_predicted_net_return"] == 0.7
    assert all(row["target_used_as_decision_input"] is False for row in output)


def test_issue202_target_stripping_fails_closed() -> None:
    row = {
        **_decision_rows()[0],
        "target_net_pnl_per_contract": 0.5,
        "target_resolved_outcome": "UP",
        "target_cost_components": {"fee": 0.01},
    }

    stripped = _strip_target_fields(row)

    assert "target_net_pnl_per_contract" not in stripped
    assert "target_resolved_outcome" not in stripped
    assert "target_cost_components" not in stripped
    assert stripped["target_used_as_decision_input"] is False


def test_issue202_gate_passes_research_evaluation_but_never_unlocks() -> None:
    profile = _profile()
    evaluation_rows, oof_rows, train_rows = _gate_rows(pnl_per_bet=0.02)

    report = _build_gate_report(
        run_id="test-pass",
        profile=profile,
        decision_freeze_sha256="a" * 64,
        evaluation_rows=evaluation_rows,
        oof_rows=oof_rows,
        train_rows=train_rows,
        corpus_audits=[{"blocking_reason_codes": []}],
    )

    assert report["development_gate_passed"] is True
    assert report["candidate_specific_future_evaluation_allowed"] is True
    assert report["guard_accepted_bet_count"] == 20
    assert report["guard_accepted_unique_market_count"] == 20
    assert report["accepted_side_metrics"]["UP"]["accepted_bet_net_pnl_sum"] > 0.0
    assert report["accepted_side_metrics"]["DOWN"]["accepted_bet_net_pnl_sum"] > 0.0
    assert report["all_oof_market_policy_pnl"]["lower_confidence_bound"] > 0.0
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_issue202_negative_pnl_remains_fail_closed() -> None:
    profile = _profile()
    evaluation_rows, oof_rows, train_rows = _gate_rows(pnl_per_bet=-0.02)

    report = _build_gate_report(
        run_id="test-fail",
        profile=profile,
        decision_freeze_sha256="b" * 64,
        evaluation_rows=evaluation_rows,
        oof_rows=oof_rows,
        train_rows=train_rows,
        corpus_audits=[{"blocking_reason_codes": []}],
    )

    assert report["development_gate_passed"] is False
    assert report["candidate_specific_future_evaluation_allowed"] is False
    assert "accepted_bet_total_pnl_not_positive" in report["gate_blocking_reason_codes"]
    assert "all_oof_market_policy_pnl_lcb_not_positive" in report["gate_blocking_reason_codes"]
    assert "supported_side_pnl_gate_failed" in report["gate_blocking_reason_codes"]
    assert report["paper_candidate_allowed"] is False


def test_issue202_action_loss_is_diagnostic_when_both_side_totals_are_positive() -> None:
    profile = _profile()
    evaluation_rows, oof_rows, train_rows = _gate_rows(pnl_per_bet=0.03)
    down_rows = [row for row in evaluation_rows if row["selected_side"] == "DOWN"]
    for index, row in enumerate(down_rows):
        if index < 5:
            row["executed_action"] = "BUY_DOWN_HOLD_TO_SETTLEMENT"
            row["source_selected_action"] = "BUY_DOWN_HOLD_TO_SETTLEMENT"
            row["accepted_bet_net_pnl"] = -0.01
        else:
            row["executed_action"] = "BUY_DOWN_SELL_BEFORE_CLOSE"
            row["source_selected_action"] = "BUY_DOWN_SELL_BEFORE_CLOSE"
            row["accepted_bet_net_pnl"] = 0.03

    report = _build_gate_report(
        run_id="test-side-only",
        profile=profile,
        decision_freeze_sha256="c" * 64,
        evaluation_rows=evaluation_rows,
        oof_rows=oof_rows,
        train_rows=train_rows,
        corpus_audits=[{"blocking_reason_codes": []}],
    )

    assert report["development_gate_passed"] is True
    assert report["accepted_side_metrics"]["DOWN"]["accepted_bet_net_pnl_sum"] == pytest.approx(0.1)
    assert report["accepted_action_metrics"]["BUY_DOWN_HOLD_TO_SETTLEMENT"][
        "accepted_bet_net_pnl_sum"
    ] == pytest.approx(-0.05)
    assert (
        report["accepted_action_metrics"]["BUY_DOWN_HOLD_TO_SETTLEMENT"]["diagnostic_only"] is True
    )


def _gate_rows(*, pnl_per_bet: float) -> tuple[list[dict], list[dict], list[dict]]:
    evaluations = []
    oof_rows = []
    train_rows = []
    for index in range(20):
        market_id = f"market-{index:02d}"
        decision_ts = 1_000 + index
        side = "UP" if index % 2 == 0 else "DOWN"
        selected_action = (
            "BUY_UP_HOLD_TO_SETTLEMENT" if side == "UP" else "BUY_DOWN_HOLD_TO_SETTLEMENT"
        )
        for action in REQUIRED_ACTIONS:
            target = pnl_per_bet / 0.2 if action == selected_action else 0.0
            row = {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "action": action,
                "target_net_pnl_per_contract": target,
            }
            train_rows.append(row)
            oof_rows.append(
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "action": action,
                    "max_input_ts": decision_ts,
                    "direct_predicted_net_return": target,
                }
            )
        evaluations.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "source_selected_action": selected_action,
                "executed_action": selected_action,
                "selected_side": side,
                "p_up_action_disagreement": False,
                "execution_guard_order_allowed": True,
                "accepted_bet_net_pnl": pnl_per_bet,
            }
        )
    return evaluations, oof_rows, train_rows


def _decision_rows() -> list[dict]:
    output = []
    for action in REQUIRED_ACTIONS:
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if action.endswith("HOLD_TO_SETTLEMENT")
            else "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "NO_TRADE"
        )
        output.append(
            {
                "market_id": "market-1",
                "decision_ts": 1_000,
                "action": action,
                "side": side,
                "action_family": family,
                "max_input_ts": 1_000,
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
            }
        )
    return output


def _key(row: dict) -> tuple[str, int, str]:
    return row["market_id"], row["decision_ts"], row["action"]


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
