from __future__ import annotations

import hashlib
from pathlib import Path

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_manual_promotion import (
    FROZEN_HASHES,
    _build_review_report,
    _paper_handoff_plan,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blocked() -> dict[str, object]:
    return {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _inputs(tmp_path: Path, monkeypatch) -> dict[str, object]:
    model = tmp_path / "model.json"
    calibration = tmp_path / "calibration.json"
    model.write_text("model", encoding="utf-8")
    calibration.write_text("calibration", encoding="utf-8")
    monkeypatch.setitem(FROZEN_HASHES, "candidate_model", _sha(model))
    monkeypatch.setitem(FROZEN_HASHES, "candidate_calibration", _sha(calibration))
    safety = _blocked()
    selected_rows = [
        {"market_id": f"market-{index:03d}", "market_start_ts": 1_000 + index}
        for index in range(200)
    ]
    candidate = {
        "candidate_name": "market_clustered_mean_ev_v6_2",
        "research_actionability_candidate_frozen": True,
        "target_free_actionability_gate_passed": True,
        "target_free_labels_outcomes_settlement_targets_or_pnl_opened": False,
        "future_collection_minimum_created_ts_exclusive": 999,
        "source_model": {"path": str(model), "sha256": _sha(model)},
        "market_clustered_mean_risk_calibration": {
            "path": str(calibration),
            "sha256": _sha(calibration),
        },
        **safety,
    }
    freeze = {
        "decision_freeze_written_before_target_access": True,
        "labels_outcomes_or_pnl_opened": False,
        "settlement_provider_called": False,
        "resolution_artifact_opened": False,
        **safety,
    }
    freeze_report = {
        "selected_market_count": 200,
        "future_strictly_later_disjoint_and_exact_window_passed": True,
        "feature_causality_violation_count": 0,
        "complete_five_action_grid_passed": True,
        "target_free_support_gate_passed": True,
    }
    settlement_manifest = {**safety}
    settlement_report = {
        "settled_corpus_ready_market_count": 200,
        "unresolved_or_failed_market_count": 0,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "direct_training_eligibility_relaxed": False,
        **safety,
    }
    fallback = {
        "evaluation_only_settlement_fallback": True,
        "official_read_only_resolution": True,
        "direct_training_eligibility_relaxed": False,
        "evaluation_only_settlement_fallback_reason_codes": [
            "frozen_feature_equivalent_chainlink_training_gate_block"
        ],
    }
    settlement_index = {"entries": [fallback], **safety}
    claim = {
        "prediction_freeze_manifest": {
            "sha256": FROZEN_HASHES["prediction_freeze_manifest"]
        },
        "settled_corpus_index": {"sha256": FROZEN_HASHES["settled_corpus_index"]},
        "future_result_driven_rerun_allowed": False,
        **safety,
    }
    evaluation_manifest = {
        "single_use_claim": {"sha256": FROZEN_HASHES["single_use_claim"]},
        "future_results_used_for_tuning": False,
        "future_result_driven_rerun_allowed": False,
        **safety,
    }
    evaluation_report = {
        "side_only_gate_executed_exactly_once": True,
        "future_results_used_for_tuning": False,
        **safety,
    }
    side_metrics = {
        side: {
            "accepted_unique_market_count": count,
            "accepted_bet_net_pnl_sum": pnl,
            "diagnostic_only": False,
        }
        for side, count, pnl in (("UP", 38, 0.66), ("DOWN", 85, 2.0))
    }
    gate = {
        "future_gate_passed": True,
        "future_gate_blocking_reason_codes": [],
        "future_gate_checks": {"all": True},
        "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
        "action_and_action_family_pnl_diagnostic_only": True,
        "accepted_side_metrics": side_metrics,
        "guard_accepted_bet_count": 123,
        "guard_accepted_unique_market_count": 123,
        "candidate_post_cost_net_pnl": 2.66,
        "matched_v5_post_cost_net_pnl": 0.0,
        "candidate_minus_matched_v5_post_cost_net_pnl": 2.66,
        "candidate_minus_matched_v5_market_bootstrap": {
            "bootstrap_unit": "market_id",
            "market_count": 200,
            "lower_confidence_bound": 0.009,
        },
        "largest_winner_removed_candidate_pnl": 2.57,
        "future_result_driven_rerun_allowed": False,
        **safety,
    }
    historical = {
        "historical_outcome_aware_diagnostic_only": True,
        "uses_historical_pnl_for_tuning": False,
        "no_strictly_unseen_split_in_this_report": True,
        "promotion_evidence": False,
        **safety,
    }
    return {
        "run_id": "manual-review-test",
        "review_completed_ts": 1_234,
        "builder_git_commit": "a" * 40,
        "candidate": candidate,
        "freeze": freeze,
        "freeze_report": freeze_report,
        "settlement_manifest": settlement_manifest,
        "settlement_report": settlement_report,
        "settlement_index": settlement_index,
        "claim": claim,
        "evaluation_manifest": evaluation_manifest,
        "evaluation_report": evaluation_report,
        "gate": gate,
        "historical": historical,
        "selected_rows": selected_rows,
        "raw_evidence": {"passed": True},
        "settled_evidence": {"passed": True},
        "source_audit": {"passed": True},
    }


def test_manual_review_promotes_research_candidate_but_not_paper(
    tmp_path: Path, monkeypatch
) -> None:
    report = _build_review_report(**_inputs(tmp_path, monkeypatch))

    payload = dict(report)
    report_id = payload.pop("manual_promotion_review_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["manual_promotion_review_passed"] is True
    assert report["research_candidate_promoted"] is True
    assert report["source_model_candidate_eligible"] is True
    assert report["freeze_ready"] is True
    assert report["promotion_evidence_eligible"] is True
    assert report["paper_candidate_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_manual_review_fails_closed_when_one_side_loses(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    inputs["gate"]["accepted_side_metrics"]["UP"]["accepted_bet_net_pnl_sum"] = -0.01

    report = _build_review_report(**inputs)

    assert report["manual_promotion_review_passed"] is False
    assert report["research_candidate_promoted"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert (
        "manual_review_both_buy_sides_supported_and_profitable_failed"
        in report["manual_promotion_review_blocking_reason_codes"]
    )


def test_manual_review_fails_closed_on_frozen_source_change(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    inputs["source_audit"] = {"passed": False}

    report = _build_review_report(**inputs)

    assert report["manual_promotion_review_passed"] is False
    assert (
        "manual_review_frozen_scoring_guard_cost_and_gate_source_unchanged_failed"
        in report["manual_promotion_review_blocking_reason_codes"]
    )


def test_paper_handoff_remains_separate_and_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    report = _build_review_report(**_inputs(tmp_path, monkeypatch))
    plan = _paper_handoff_plan(report)

    assert plan["research_candidate_promoted"] is True
    assert plan["paper_candidate_gate_required"] is True
    assert plan["paper_candidate_allowed"] is False
    assert plan["v8_execution_handoff_allowed"] is False
    assert plan["capital_at_risk"] is False
    assert plan["polymarket_write_enabled"] is False
    assert plan["wallet_signing_enabled"] is False
