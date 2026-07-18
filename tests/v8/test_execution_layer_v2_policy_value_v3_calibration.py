from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_value_v3_calibration import (
    apply_policy_value_v3_scores,
    build_policy_value_v3_evaluation_rows,
    build_policy_value_v3_gate_report,
    strip_policy_value_v3_targets,
    validate_policy_value_v3_calibration_profile,
    validate_policy_value_v3_target_stripped_rows,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_policy_value_lcb_v3_calibration_gate_profile.json"
)


def test_issue200_profile_freezes_independent_calibration_and_safety() -> None:
    profile = _load_json(PROFILE_PATH)

    validate_policy_value_v3_calibration_profile(profile)

    assert profile["evaluation_role"] == "development_calibration"
    assert profile["selector"]["advantage_vs_best_alternative_used_for_selection"] is False
    assert profile["development_gate"]["minimum_guard_accepted_unique_market_count"] == 10
    assert profile["access_sequence"]["confirmatory_validation_labels_may_be_opened"] is False
    assert profile["access_sequence"]["issue_190_or_192_future_labels_may_be_opened"] is False
    assert profile["safety"]["source_model_candidate_eligible"] is False
    assert profile["safety"]["promotion_evidence_eligible"] is False


def test_issue200_profile_rejects_oracle_comparator_or_threshold_drift() -> None:
    profile = _load_json(PROFILE_PATH)
    profile["selector"]["advantage_vs_best_alternative_used_for_selection"] = True
    with pytest.raises(ValueError, match="two_safety_selector"):
        validate_policy_value_v3_calibration_profile(profile)

    profile = _load_json(PROFILE_PATH)
    profile["mutation_contract"]["threshold_search_allowed"] = True
    with pytest.raises(ValueError, match="no_mutation"):
        validate_policy_value_v3_calibration_profile(profile)


def test_issue200_v3_score_ignores_oracle_comparator_lcb() -> None:
    profile = _load_json(PROFILE_PATH)
    predictions = _prediction_rows()
    first = apply_policy_value_v3_scores(
        predictions,
        calibration=_calibration(oracle_lcb=-10.0),
        profile=profile,
    )
    second = apply_policy_value_v3_scores(
        predictions,
        calibration=_calibration(oracle_lcb=10.0),
        profile=profile,
    )

    first_scores = {
        row["action"]: (
            row["policy_value_v3_score"],
            row["policy_value_v3_two_safety_estimands_passed"],
        )
        for row in first
    }
    second_scores = {
        row["action"]: (
            row["policy_value_v3_score"],
            row["policy_value_v3_two_safety_estimands_passed"],
        )
        for row in second
    }
    assert first_scores == second_scores
    assert first_scores["BUY_UP_HOLD_TO_SETTLEMENT"] == (0.05, True)
    assert first_scores["NO_TRADE"] == (0.0, False)
    assert all(row["policy_value_v3_oracle_comparator_diagnostic_only"] for row in first)


def test_issue200_target_stripping_fails_closed_on_any_target_field() -> None:
    row = _prediction_rows()[0]
    stripped = strip_policy_value_v3_targets(row)
    validate_policy_value_v3_target_stripped_rows(
        [
            {
                **candidate,
                "training_target_fields_stripped": True,
            }
            for candidate in _prediction_rows()
        ]
    )
    assert stripped["target_or_outcome_fields_used"] is False

    with pytest.raises(ValueError, match="target fields"):
        strip_policy_value_v3_targets({**row, "target_net_pnl_per_contract": 999.0})


def test_issue200_evaluation_join_uses_guard_size_times_cost_aware_target() -> None:
    replay = [
        {
            "decision_index": 1,
            "market_id": "market-1",
            "decision_ts": 1_000,
            "source_selected_action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "executed_action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "selected_side": "UP",
            "selected_action_family": "HOLD_TO_SETTLEMENT",
            "execution_guard_order_allowed": True,
            "proposed_order_size": 0.2,
            "execution_blocking_reason_codes": [],
            "viability_row_sha256": "a" * 64,
        }
    ]
    target_rows = [
        {
            "market_id": "market-1",
            "decision_ts": 1_000,
            "action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "target_net_pnl_per_contract": 0.5,
        }
    ]

    rows, join = build_policy_value_v3_evaluation_rows(replay, target_rows=target_rows)

    assert join["accepted_target_join_reconciled"] is True
    assert rows[0]["evaluation_target_net_pnl_per_contract"] == 0.5
    assert rows[0]["accepted_bet_net_pnl"] == pytest.approx(0.1)
    assert rows[0]["target_used_as_decision_input"] is False


def test_issue200_gate_passes_only_with_support_positive_lcb_and_action_pnl(
    tmp_path: Path,
) -> None:
    profile = _load_json(PROFILE_PATH)
    freeze_path = tmp_path / "decision-freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    evaluation_rows = _evaluation_rows(accepted_pnl=0.1)
    report = build_policy_value_v3_gate_report(
        run_id="test-pass",
        profile=profile,
        decision_freeze=_decision_freeze(),
        decision_freeze_path=freeze_path,
        evaluation_rows=evaluation_rows,
        join_report=_join_report(accepted_count=10),
        corpus_audits=_corpus_audits(),
        calibration_market_ids=[f"market-{index:02d}" for index in range(45)],
    )

    assert report["development_calibration_gate_passed"] is True
    assert report["candidate_specific_confirmatory_evaluation_allowed"] is True
    assert report["guard_accepted_bet_count"] == 10
    assert report["guard_accepted_unique_market_count"] == 10
    assert report["all_calibration_market_policy_pnl"]["lower_confidence_bound"] > 0.0
    assert report["development_calibration_labels_opened_after_decision_freeze"] is True
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_issue200_negative_calibration_pnl_remains_fail_closed(tmp_path: Path) -> None:
    profile = _load_json(PROFILE_PATH)
    freeze_path = tmp_path / "decision-freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    report = build_policy_value_v3_gate_report(
        run_id="test-fail",
        profile=profile,
        decision_freeze=_decision_freeze(),
        decision_freeze_path=freeze_path,
        evaluation_rows=_evaluation_rows(accepted_pnl=-0.1),
        join_report=_join_report(accepted_count=10),
        corpus_audits=_corpus_audits(),
        calibration_market_ids=[f"market-{index:02d}" for index in range(45)],
    )

    assert report["development_calibration_gate_passed"] is False
    assert report["candidate_specific_confirmatory_evaluation_allowed"] is False
    assert "accepted_bet_total_pnl_not_positive" in report["gate_blocking_reason_codes"]
    assert "all_market_policy_pnl_lcb_not_positive" in report["gate_blocking_reason_codes"]
    assert "supported_action_pnl_gate_failed" in report["gate_blocking_reason_codes"]
    assert report["source_model_candidate_eligible"] is False


def test_issue200_zero_guard_acceptance_reports_p_up_intersection(tmp_path: Path) -> None:
    profile = _load_json(PROFILE_PATH)
    freeze_path = tmp_path / "decision-freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    freeze = _decision_freeze()
    freeze["trade_candidate_p_up_disagreement_count"] = 10
    rows = _evaluation_rows(accepted_pnl=0.0)
    for row in rows:
        row["execution_guard_order_allowed"] = False
        row["accepted_bet_net_pnl"] = 0.0
        row["execution_blocking_reason_codes"] = ["execution_p_up_side_disagreement"]
    report = build_policy_value_v3_gate_report(
        run_id="test-zero-guard",
        profile=profile,
        decision_freeze=freeze,
        decision_freeze_path=freeze_path,
        evaluation_rows=rows,
        join_report=_join_report(accepted_count=0),
        corpus_audits=_corpus_audits(),
        calibration_market_ids=[f"market-{index:02d}" for index in range(45)],
    )

    assert report["guard_intersection_reason_codes"] == [
        "zero_guard_accepted_bets",
        "all_trade_candidates_p_up_side_disagreement",
    ]
    assert report["accepted_bet_pnl_evaluation_status"] == ("no_guard_accepted_bets_no_pnl_sample")
    assert report["candidate_specific_confirmatory_evaluation_allowed"] is False


def _prediction_rows() -> list[dict]:
    rows = []
    for index, action in enumerate(REQUIRED_ACTIONS):
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if action.endswith("HOLD_TO_SETTLEMENT")
            else "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "NO_TRADE"
        )
        rows.append(
            {
                "market_id": "market-1",
                "decision_ts": 1_000,
                "action": action,
                "side": side,
                "action_family": family,
                "pairwise_group_normalized_rank_score": 1.0 - index * 0.1,
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
            }
        )
    return rows


def _calibration(*, oracle_lcb: float) -> dict:
    actions = {}
    groups = {}
    for action in REQUIRED_ACTIONS:
        actions[action] = {
            "adaptive_score_boundaries": [],
            "adaptive_bucket_names": ["bucket_0"],
        }
        lcbs = {
            "absolute_post_cost_net_return": 0.05 if action != "NO_TRADE" else 0.0,
            "advantage_vs_no_trade": 0.04 if action != "NO_TRADE" else 0.0,
            "advantage_vs_best_alternative": oracle_lcb,
        }
        groups[f"{action}|bucket_0"] = {
            "estimators": {
                name: {
                    "point_estimate": value + 0.01,
                    "lower_confidence_bound": value,
                }
                for name, value in lcbs.items()
            }
        }
    return {"actions": actions, "calibration_groups": groups}


def _evaluation_rows(*, accepted_pnl: float) -> list[dict]:
    rows = []
    for market_index in range(45):
        for decision_index in range(4):
            accepted = market_index < 10 and decision_index == 0
            rows.append(
                {
                    "market_id": f"market-{market_index:02d}",
                    "decision_ts": market_index * 10_000 + decision_index,
                    "executed_action": "BUY_UP_HOLD_TO_SETTLEMENT",
                    "selected_side": "UP",
                    "selected_action_family": "HOLD_TO_SETTLEMENT",
                    "execution_guard_order_allowed": accepted,
                    "accepted_bet_net_pnl": accepted_pnl if accepted else 0.0,
                    "execution_blocking_reason_codes": []
                    if accepted
                    else ["policy_selected_no_trade"],
                }
            )
    return rows


def _decision_freeze() -> dict:
    return {
        "development_calibration_labels_opened": False,
        "two_safety_selector_trade_decision_count": 10,
        "guard_evaluated_count": 10,
        "source_selected_action_distribution": {"BUY_UP_HOLD_TO_SETTLEMENT": 10},
        "source_selected_side_distribution": {"UP": 10},
        "trade_candidate_p_up_disagreement_count": 0,
    }


def _join_report(*, accepted_count: int) -> dict:
    return {
        "accepted_target_join_reconciled": True,
        "guard_accepted_count": accepted_count,
        "accepted_target_join_count": accepted_count,
        "duplicate_target_key_count": 0,
        "missing_accepted_target_count": 0,
    }


def _corpus_audits() -> list[dict]:
    return [
        {
            "feature_causality_violation_count": 0,
            "cost_component_violation_count": 0,
        }
        for _ in range(45)
    ]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
