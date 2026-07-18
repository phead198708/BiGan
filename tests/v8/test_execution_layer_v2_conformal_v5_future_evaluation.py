from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    _selected_window_blockers,
    build_conformal_v5_side_only_future_pnl_gate,
    validate_conformal_v5_future_evaluation_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_conformal_v5_strict_future_evaluation_v1.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _evaluation_row(
    index: int,
    *,
    side: str,
    action: str,
    pnl: float,
    baseline_pnl: float = 0.001,
) -> dict:
    return {
        "market_id": f"market-{index:03d}",
        "selected_side": side,
        "executed_action": action,
        "execution_guard_order_allowed": True,
        "accepted_bet_net_pnl": pnl,
        "matched_baseline_net_pnl": baseline_pnl,
        "settlement_resolved": True,
        "target_joined_after_decision_freeze": True,
        "target_used_as_decision_input": False,
        "forbidden_outcome_field_used_for_decision": False,
        "feature_causality_violation": False,
        "provenance_violation": False,
        "runtime_state_violation": False,
    }


def _passing_rows() -> list[dict]:
    rows = [
        _evaluation_row(
            index,
            side="UP",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            pnl=0.02,
        )
        for index in range(44)
    ]
    rows.extend(
        _evaluation_row(
            index,
            side="DOWN",
            action=("BUY_DOWN_HOLD_TO_SETTLEMENT" if index < 54 else "BUY_DOWN_SELL_BEFORE_CLOSE"),
            pnl=-0.01 if index < 54 else 0.02,
        )
        for index in range(44, 88)
    )
    return rows


def test_profile_freezes_195_source_markets_and_side_only_gate() -> None:
    profile = _profile()
    validate_conformal_v5_future_evaluation_profile(profile)
    assert profile["issue_203_candidate"]["fit_market_count"] == 135
    assert profile["issue_203_candidate"]["conformal_calibration_market_count"] == 60
    assert profile["issue_203_candidate"]["source_market_count"] == 195
    gates = profile["support_and_pnl_gates"]
    assert gates["pnl_hard_gate_aggregation"] == "selected_side_buy_up_buy_down_only"
    assert gates["action_and_action_family_pnl_diagnostic_only"] is True
    assert gates["minimum_guard_accepted_unique_market_count"] == 88


def test_profile_rejects_action_level_hard_gate() -> None:
    profile = _profile()
    profile["support_and_pnl_gates"]["action_and_action_family_pnl_diagnostic_only"] = False
    with pytest.raises(ValueError, match="side_only"):
        validate_conformal_v5_future_evaluation_profile(profile)


def test_profile_rejects_prediction_before_window_binding() -> None:
    profile = _profile()
    profile["prediction_and_settlement_sequence"][
        "window_binding_before_feature_materialization"
    ] = False
    with pytest.raises(ValueError, match="access_sequence"):
        validate_conformal_v5_future_evaluation_profile(profile)


def test_negative_action_subtype_does_not_block_positive_down_side() -> None:
    report = build_conformal_v5_side_only_future_pnl_gate(
        _passing_rows(),
        profile=_profile(),
        decision_freeze_sha256="a" * 64,
    )
    assert report["future_gate_passed"] is True
    assert report["accepted_side_metrics"]["UP"]["accepted_bet_net_pnl_sum"] > 0.0
    assert report["accepted_side_metrics"]["DOWN"]["accepted_bet_net_pnl_sum"] > 0.0
    assert (
        report["accepted_action_metrics"]["BUY_DOWN_HOLD_TO_SETTLEMENT"]["accepted_bet_net_pnl_sum"]
        < 0.0
    )
    assert (
        report["accepted_action_metrics"]["BUY_DOWN_HOLD_TO_SETTLEMENT"]["diagnostic_only"] is True
    )
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False


def test_negative_supported_side_fails_closed() -> None:
    rows = _passing_rows()
    for row in rows:
        if row["selected_side"] == "UP":
            row["accepted_bet_net_pnl"] = -0.01
    report = build_conformal_v5_side_only_future_pnl_gate(
        rows,
        profile=_profile(),
        decision_freeze_sha256="b" * 64,
    )
    assert report["future_gate_passed"] is False
    assert "supported_side_post_cost_pnl_gate_failed" in report["future_gate_blocking_reason_codes"]


def test_selected_window_rejects_wrong_collector_commit_before_prediction() -> None:
    profile = _profile()
    boundary = {
        "minimum_collection_decision_ts": 2_000,
        "prior_market_ids": [],
        "prior_slugs": [],
        "prior_source_row_hashes": [],
    }
    rows = [
        {
            "market_id": f"future-{index}",
            "slug": f"future-slug-{index}",
            "source_row_hash": f"source-{index}",
            "entry_sha256": f"entry-{index}",
            "scheduled_round_start_ts": 2_001 + index,
            "collector_git_commit": profile["issue_192_collection"]["collector_commit"],
            "capture_quality_valid": True,
            "labels_outcomes_or_pnl_opened": False,
            **profile["safety"],
        }
        for index in range(220)
    ]
    index_rows = copy.deepcopy(rows)
    rows[0]["collector_git_commit"] = "0" * 40
    blockers = _selected_window_blockers(
        selected_rows=rows,
        index_rows=index_rows,
        boundary=boundary,
        profile=profile,
    )
    assert "selected_row_collector_commit_mismatch" in blockers


def test_selected_window_rejects_forbidden_outcome_fields() -> None:
    profile = _profile()
    boundary = {
        "minimum_collection_decision_ts": 2_000,
        "prior_market_ids": [],
        "prior_slugs": [],
        "prior_source_row_hashes": [],
    }
    rows = [
        {
            "market_id": f"future-{index}",
            "slug": f"future-slug-{index}",
            "source_row_hash": f"source-{index}",
            "entry_sha256": f"entry-{index}",
            "scheduled_round_start_ts": 2_001 + index,
            "collector_git_commit": profile["issue_192_collection"]["collector_commit"],
            "capture_quality_valid": True,
            "labels_outcomes_or_pnl_opened": False,
            **profile["safety"],
        }
        for index in range(220)
    ]
    index_rows = copy.deepcopy(rows)
    rows[-1]["settlement_outcome"] = "UP"
    blockers = _selected_window_blockers(
        selected_rows=rows,
        index_rows=index_rows,
        boundary=boundary,
        profile=profile,
    )
    assert "selected_rows_contain_forbidden_target_fields" in blockers
