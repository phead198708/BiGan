from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_p_up_aligned_action_value_support import (
    _claimed_descriptor,
    build_execution_compatible_action_universe,
    build_p_up_aligned_action_value_support_report,
    build_source_guard_intersection_attribution_report,
    validate_p_up_aligned_action_value_support_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_p_up_aligned_action_value_support_audit_profile.json"
)


def test_issue201_profile_freezes_train_only_access_and_all_safety() -> None:
    profile = _load_json(PROFILE_PATH)

    validate_p_up_aligned_action_value_support_profile(profile)

    assert profile["allowed_role"] == "development_train"
    assert profile["access_sequence"]["development_calibration_files_may_be_opened"] is False
    assert profile["access_sequence"]["confirmatory_files_may_be_opened"] is False
    assert profile["access_sequence"]["issue_190_or_192_future_files_may_be_opened"] is False
    assert profile["mutation_contract"]["new_candidate_fit_allowed"] is False
    assert profile["safety"]["source_model_candidate_eligible"] is False
    assert profile["safety"]["#134_resume_allowed"] is False
    assert profile["safety"]["#146_start_allowed"] is False


def test_issue201_profile_rejects_guard_or_evidence_scope_drift() -> None:
    profile = _load_json(PROFILE_PATH)
    profile["static_guard_probe"]["guard_config_mutation_allowed"] = True
    with pytest.raises(ValueError, match="probe"):
        validate_p_up_aligned_action_value_support_profile(profile)

    profile = _load_json(PROFILE_PATH)
    profile["access_sequence"]["development_calibration_files_may_be_opened"] = True
    with pytest.raises(ValueError, match="access"):
        validate_p_up_aligned_action_value_support_profile(profile)


def test_issue201_target_descriptor_claim_does_not_open_target_file(tmp_path: Path) -> None:
    missing_target_path = tmp_path / "target-bearing-rows-do-not-touch.jsonl"

    descriptor = _claimed_descriptor(
        {"path": str(missing_target_path), "sha256": "a" * 64},
        name="target rows",
    )

    assert descriptor["path"] == str(missing_target_path)
    assert missing_target_path.exists() is False


def test_issue201_static_guard_probe_separates_p_up_alignment_from_execution_quality() -> None:
    rows = _action_rows(market_id="market-1", decision_ts=1_000, p_up=0.7)

    universe = build_execution_compatible_action_universe(rows)
    by_action = {row["action"]: row for row in universe}

    assert len(universe) == 5
    assert by_action["BUY_UP_HOLD_TO_SETTLEMENT"]["p_up_alignment_passed"] is True
    assert by_action["BUY_UP_HOLD_TO_SETTLEMENT"]["execution_quality_only_passed"] is True
    assert by_action["BUY_UP_HOLD_TO_SETTLEMENT"]["full_guard_original_action_allowed"] is True
    assert by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]["p_up_alignment_passed"] is False
    assert by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]["execution_quality_only_passed"] is True
    assert by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]["full_guard_order_allowed"] is False
    assert (
        "execution_p_up_side_disagreement"
        in by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]["execution_blocking_reason_codes"]
    )
    assert all(row["target_or_outcome_fields_used"] is False for row in universe)


def test_issue201_support_report_is_cost_aware_but_not_policy_pnl() -> None:
    profile = _load_json(PROFILE_PATH)
    joined = []
    for market_index in range(10):
        rows = _action_rows(
            market_id=f"market-{market_index:02d}",
            decision_ts=1_000 + market_index,
            p_up=0.7,
        )
        for row in build_execution_compatible_action_universe(rows):
            target = 0.1 if row["p_up_alignment_passed"] else -0.2
            if row["action"] == "NO_TRADE":
                target = 0.0
            joined.append({**row, "target_net_pnl_per_contract": target})

    report = build_p_up_aligned_action_value_support_report(
        run_id="test-positive",
        rows=joined,
        profile=profile,
        universe_freeze_sha256="a" * 64,
    )
    focal = report["segment_metrics"]["p_up_aligned_execution_quality_passed_trade_actions"]

    assert focal["row_count"] == 20
    assert focal["unique_market_count"] == 10
    assert focal["target_post_cost_return_mean"] == pytest.approx(0.1)
    assert focal["market_level_post_cost_return"]["lower_confidence_bound"] > 0.0
    assert report["support_conclusion"] == (
        "positive_lcb_support_exists_for_preregistered_v4_research"
    )
    assert report["opportunity_set_target_sum_is_policy_pnl"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False


def test_issue201_attribution_never_promotes_or_relaxes_guard() -> None:
    profile = _load_json(PROFILE_PATH)
    rows = []
    for market_index in range(10):
        universe = build_execution_compatible_action_universe(
            _action_rows(
                market_id=f"market-{market_index:02d}",
                decision_ts=2_000 + market_index,
                p_up=0.7,
            )
        )
        rows.extend(
            {
                **row,
                "target_net_pnl_per_contract": (0.05 if row["p_up_alignment_passed"] else -0.1),
            }
            for row in universe
        )
    support = build_p_up_aligned_action_value_support_report(
        run_id="test-attribution",
        rows=rows,
        profile=profile,
        universe_freeze_sha256="b" * 64,
    )
    issue200 = [
        {
            "source_selected_action": "BUY_DOWN_HOLD_TO_SETTLEMENT",
            "p_up_action_disagreement": True,
            "execution_guard_order_allowed": False,
        }
        for _ in range(4)
    ]

    report = build_source_guard_intersection_attribution_report(
        run_id="test-attribution",
        issue200_replay=issue200,
        support_report=support,
    )

    assert report["issue200_all_selected_trade_candidates_p_up_disagreeing"] is True
    assert report["issue200_guard_accepted_trade_count"] == 0
    assert report["new_candidate_fit_or_guard_relaxation_performed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["v8_execution_handoff_allowed"] is False


def _action_rows(*, market_id: str, decision_ts: int, p_up: float) -> list[dict]:
    rows = []
    for action in REQUIRED_ACTIONS:
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if action.endswith("HOLD_TO_SETTLEMENT")
            else "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "NO_TRADE"
        )
        disagreement = (side == "UP" and p_up < 0.5) or (side == "DOWN" and p_up > 0.5)
        rows.append(
            {
                "market_id": market_id,
                "condition_id": market_id,
                "market_slug": market_id,
                "decision_ts": decision_ts,
                "market_close_ts": decision_ts + 240_000,
                "max_input_ts": decision_ts,
                "role": "development_train",
                "market_selection_rank": 1,
                "action": action,
                "side": side,
                "action_family": family,
                "p_up": p_up,
                "p_down": 1.0 - p_up,
                "p_up_action_disagreement": disagreement,
                "microstructure_snapshot": {
                    "entry_bid": 0.49,
                    "entry_ask": 0.50,
                    "spread_bps": 200.0,
                    "book_staleness_ms": 100.0,
                    "queue_fill_proxy": 0.9,
                    "time_to_close_seconds": 240.0,
                },
                "reference_price_feature_provenance": {
                    "provenance_valid": True,
                    "max_input_ts": decision_ts,
                },
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
                "paper_only": True,
                "capital_at_risk": False,
            }
        )
    return rows


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
