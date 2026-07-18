from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    _candidate_fit_profile_from_preregistered_lineage,
    _minimum_future_collection_ts,
    _selected_window_blockers,
    _window_binding_blockers,
    build_conformal_v5_side_only_future_pnl_gate,
    validate_conformal_v5_future_evaluation_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    WINDOW_MANIFEST_SCHEMA_VERSION,
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
    baseline = profile["frozen_matched_market_baseline"]
    assert baseline["candidate_name"] == "guard_compatible_direct_net_return_v4"
    assert baseline["selection_method"] == ("guard_compatible_direct_predicted_net_return_argmax")
    assert baseline["future_outcomes_used_to_select_baseline"] is False


def test_future_collection_starts_after_latest_preregistration_amendment() -> None:
    assert (
        _minimum_future_collection_ts(
            max_prior_decision_ts=1_000,
            candidate_freeze_ts=2_000,
            preregistration_created_ts=3_000,
        )
        == 3_001
    )


def test_missing_redundant_prereg_fit_profile_uses_pinned_candidate_lineage(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "fit-profile.json"
    fit_path.write_text("{}\n", encoding="utf-8")
    fit_descriptor = {
        "path": str(fit_path),
        "sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps({"fit_profile": fit_descriptor}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prereg = {
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        }
    }
    profile = _profile()
    profile["issue_203_candidate"]["fit_profile_sha256"] = fit_descriptor["sha256"]

    resolved = _candidate_fit_profile_from_preregistered_lineage(prereg, profile=profile)

    assert resolved == fit_descriptor
    assert "candidate_fit_profile" not in prereg

    prereg["candidate_fit_profile"] = {"path": "wrong", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="ambiguous preregistered"):
        _candidate_fit_profile_from_preregistered_lineage(prereg, profile=profile)


def test_profile_rejects_action_level_hard_gate() -> None:
    profile = _profile()
    profile["support_and_pnl_gates"]["action_and_action_family_pnl_diagnostic_only"] = False
    with pytest.raises(ValueError, match="side_only"):
        validate_conformal_v5_future_evaluation_profile(profile)


def test_window_binding_requires_immutable_collector_index_snapshot() -> None:
    profile = _profile()
    boundary_descriptor = {"path": "/frozen/boundary.json", "sha256": "a" * 64}
    prereg = {"source_boundary_manifest": boundary_descriptor}
    boundary = {"minimum_collection_decision_ts": 2_000}
    window = {
        "schema_version": WINDOW_MANIFEST_SCHEMA_VERSION,
        "window_freeze_ready": True,
        "labels_outcomes_or_pnl_opened_for_selection": False,
        "target_valid_market_count": 220,
        "maximum_scan_count": 340,
        "selected_market_count": 220,
        "selected_window_start_ts": 2_000,
        "source_boundary_manifest": boundary_descriptor,
        "protocol": {"sha256": profile["issue_192_collection"]["collector_protocol_sha256"]},
        "blocking_reason_codes": [],
        **profile["safety"],
    }

    blockers = _window_binding_blockers(
        prereg=prereg,
        profile=profile,
        boundary=boundary,
        window=window,
    )
    assert "window_collector_index_snapshot_not_immutable" in blockers

    window["collector_index_snapshot_immutable"] = True
    window["window_selection_used_immutable_index_snapshot"] = True
    assert (
        _window_binding_blockers(
            prereg=prereg,
            profile=profile,
            boundary=boundary,
            window=window,
        )
        == []
    )


def test_profile_rejects_result_selected_or_unpinned_matched_baseline() -> None:
    profile = _profile()
    profile["frozen_matched_market_baseline"]["future_outcomes_used_to_select_baseline"] = True
    profile["frozen_matched_market_baseline"]["model_sha256"] = "invalid"
    with pytest.raises(
        ValueError,
        match="baseline_identity.*baseline_not_market_implied_or_result_selected",
    ):
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


def test_matched_baseline_is_evaluated_separately_on_the_same_market_window() -> None:
    candidate_rows = _passing_rows()
    baseline_rows = []
    for row in candidate_rows:
        baseline = copy.deepcopy(row)
        baseline.pop("matched_baseline_net_pnl")
        baseline["accepted_bet_net_pnl"] = 0.001
        baseline_rows.append(baseline)
    report = build_conformal_v5_side_only_future_pnl_gate(
        candidate_rows,
        matched_baseline_evaluation_rows=baseline_rows,
        evaluation_market_ids=[f"market-{index:03d}" for index in range(100)],
        profile=_profile(),
        decision_freeze_sha256="c" * 64,
    )
    assert report["future_gate_passed"] is True
    assert report["matched_baseline_evaluated_separately_on_same_frozen_window"] is True
    assert report["matched_baseline_guard_accepted_bet_count"] == 88
    assert report["comparison_market_count"] == 100
    assert report["candidate_minus_matched_baseline_post_cost_net_pnl"] > 0.0


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


def test_selected_window_accepts_frozen_collector_safety_schema() -> None:
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
            **{
                key: value
                for key, value in profile["safety"].items()
                if key != "paper_candidate_allowed"
            },
        }
        for index in range(220)
    ]
    blockers = _selected_window_blockers(
        selected_rows=rows,
        index_rows=copy.deepcopy(rows),
        boundary=boundary,
        profile=profile,
    )
    assert blockers == []


def test_selected_window_rejects_positive_paper_candidate_source_flag() -> None:
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
    rows[0]["paper_candidate_allowed"] = True
    blockers = _selected_window_blockers(
        selected_rows=rows,
        index_rows=index_rows,
        boundary=boundary,
        profile=profile,
    )
    assert "selected_row_paper_candidate_allowed_invalid" in blockers


def test_selected_window_rejects_mutated_collector_safety_flag() -> None:
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
    rows[-1]["capital_at_risk"] = True
    blockers = _selected_window_blockers(
        selected_rows=rows,
        index_rows=index_rows,
        boundary=boundary,
        profile=profile,
    )
    assert "selected_row_safety_invalid" in blockers
