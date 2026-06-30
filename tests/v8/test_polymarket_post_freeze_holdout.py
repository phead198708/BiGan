"""Post-freeze holdout validation tests for the frozen M selector."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from bigan.v8.polymarket import (
    PolymarketCorpusBuildConfig,
    PolymarketPolicyTrainingConfig,
    build_polymarket_btc_corpus,
    run_polymarket_policy_training,
    write_deterministic_polymarket_corpus_fixtures,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256
from bigan.v8.polymarket.training.post_freeze_holdout import (
    FROZEN_M_SELECTOR_BASELINE_COMMIT,
    FROZEN_M_SELECTOR_METHOD,
    M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION,
    PolymarketPostFreezeHoldoutConfig,
    run_polymarket_m_post_freeze_holdout_validation,
)
from bigan.v8.polymarket.training.post_freeze_holdout_accumulation import (
    M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION,
    PolymarketPostFreezeHoldoutAccumulationConfig,
    run_polymarket_m_post_freeze_holdout_accumulation,
)
from bigan.v8.polymarket.training.post_freeze_promotion_readiness_audit import (
    M_POST_FREEZE_PROMOTION_READINESS_AUDIT_SCHEMA_VERSION,
    PolymarketPostFreezePromotionReadinessAuditConfig,
    run_polymarket_m_post_freeze_promotion_readiness_audit,
)
from bigan.v8.polymarket.training.post_freeze_weak_evidence_drilldown import (
    M_POST_FREEZE_WEAK_EVIDENCE_DRILLDOWN_SCHEMA_VERSION,
    PolymarketPostFreezeWeakEvidenceDrilldownConfig,
    run_polymarket_m_post_freeze_weak_evidence_drilldown,
)


def test_post_freeze_holdout_blocks_same_lineage_before_prediction(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path / "source")
    training = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "training",
        )
    )
    _patch_frozen_manifest_lineage(training.run_dir, corpus_dir, training.dataset.dataset_hash)

    result = run_polymarket_m_post_freeze_holdout_validation(
        PolymarketPostFreezeHoldoutConfig(
            frozen_model_dir=training.run_dir,
            frozen_corpus_dir=corpus_dir,
            holdout_corpus_dir=corpus_dir,
            output_dir=tmp_path / "holdout",
        )
    )

    report = result.report
    assert report["schema_version"] == M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION
    assert report["validation_status"] == "blocked_fail_closed"
    assert report["prediction_attempted"] is False
    assert report["true_post_freeze_holdout"] is False
    assert "holdout_not_strictly_after_frozen_training_window" in report[
        "reason_codes"
    ]
    assert "holdout_market_ids_overlap_frozen_training_corpus" in report[
        "reason_codes"
    ]
    assert "holdout_dataset_hash_matches_frozen_dataset" in report["reason_codes"]
    assert report["selected_entry_count"] == 0
    assert report["replay_entry_count"] == 0
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["paper_run_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert result.artifact_paths["report"].exists()
    assert result.artifact_paths["summary"].exists()


def test_post_freeze_holdout_runs_later_disjoint_corpus_with_frozen_weights(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path / "source")
    training = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "training",
        )
    )
    _patch_frozen_manifest_lineage(training.run_dir, corpus_dir, training.dataset.dataset_hash)
    frozen_max_ts = max(example.decision_ts for example in training.dataset.examples)
    holdout_corpus_dir = _copy_later_disjoint_corpus(
        source=corpus_dir,
        destination=tmp_path / "later_holdout_corpus",
        min_after_ts=frozen_max_ts,
    )

    result = run_polymarket_m_post_freeze_holdout_validation(
        PolymarketPostFreezeHoldoutConfig(
            frozen_model_dir=training.run_dir,
            frozen_corpus_dir=corpus_dir,
            holdout_corpus_dir=holdout_corpus_dir,
            output_dir=tmp_path / "holdout",
        )
    )

    report = result.report
    report_id = report["m_post_freeze_holdout_validation_report_id"]
    payload = dict(report)
    payload.pop("m_post_freeze_holdout_validation_report_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["validation_status"] == "completed"
    assert report["prediction_attempted"] is True
    assert report["true_post_freeze_holdout"] is True
    assert report["baseline_selector_commit"] == FROZEN_M_SELECTOR_BASELINE_COMMIT
    assert report["selector_method"] == FROZEN_M_SELECTOR_METHOD
    assert report["selector_weights_unchanged_from_baseline"] is True
    assert report["rank_weight_tuning_allowed"] is False
    assert report["rank_weight_tuning_performed"] is False
    assert report["holdout_feedback_used_for_tuning"] is False
    assert report["p_up_side_alignment_filter_enabled"] is False
    assert report["p_up_side_alignment_diagnostic_enabled"] is True
    assert report["provenance"]["holdout_strictly_after_frozen"] is True
    assert report["provenance"]["market_id_disjoint"] is True
    assert report["provenance"]["dataset_hash_changed"] is True
    assert report["provenance"]["frozen_model_sha256_matches_manifest"] is True
    assert report["provenance"]["frozen_dataset_hash_matches_manifest"] is True
    assert report["provenance"]["frozen_split_hash_matches_manifest"] is True
    assert report["provenance"][
        "frozen_corpus_dir_matches_frozen_training_lineage"
    ] is True
    assert report["selected_exit_decision_count"] == 0
    assert report["replay_entry_reconciliation"]["reconciled"] is True
    assert set(report["replay_pnl_by_side"]) == {"UP", "DOWN"}
    assert "rank_score_component_summary" in report
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["paper_run_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    manifest = _read_json(result.artifact_paths["manifest"])
    assert looks_like_sha256(manifest["artifact_hashes"]["report"])
    assert looks_like_sha256(manifest["artifact_hashes"]["summary"])


def test_post_freeze_holdout_blocks_wrong_frozen_corpus_before_prediction(
    tmp_path: Path,
) -> None:
    corpus_dir = _build_corpus(tmp_path / "source")
    training = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "training",
        )
    )
    _patch_frozen_manifest_lineage(training.run_dir, corpus_dir, training.dataset.dataset_hash)
    frozen_max_ts = max(example.decision_ts for example in training.dataset.examples)
    wrong_frozen_corpus_dir = _copy_later_disjoint_corpus(
        source=corpus_dir,
        destination=tmp_path / "wrong_frozen_corpus",
        min_after_ts=frozen_max_ts,
    )
    holdout_corpus_dir = _copy_later_disjoint_corpus(
        source=corpus_dir,
        destination=tmp_path / "later_holdout_corpus",
        min_after_ts=frozen_max_ts + 10_000_000,
    )

    result = run_polymarket_m_post_freeze_holdout_validation(
        PolymarketPostFreezeHoldoutConfig(
            frozen_model_dir=training.run_dir,
            frozen_corpus_dir=wrong_frozen_corpus_dir,
            holdout_corpus_dir=holdout_corpus_dir,
            output_dir=tmp_path / "holdout",
        )
    )

    report = result.report
    assert report["validation_status"] == "blocked_fail_closed"
    assert report["prediction_attempted"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["provenance"]["frozen_model_sha256_matches_manifest"] is True
    assert report["provenance"]["frozen_dataset_hash_matches_manifest"] is False
    assert report["provenance"][
        "frozen_corpus_dir_matches_frozen_training_lineage"
    ] is False
    assert "frozen_dataset_hash_mismatch_manifest" in report["reason_codes"]
    assert "frozen_corpus_dir_not_frozen_training_lineage" in report["reason_codes"]
    assert report["selected_entry_count"] == 0
    assert report["replay_entry_count"] == 0


def test_post_freeze_holdout_accumulation_counts_only_true_completed_runs(
    tmp_path: Path,
) -> None:
    true_up = _write_holdout_report(
        tmp_path / "true_up",
        _holdout_validation_report(
            run_id="true-up",
            market_ids=("market-up",),
            replay_rows=[
                _replay_row(
                    market_id="market-up",
                    decision_ts=10,
                    side="UP",
                    pnl=0.10,
                )
            ],
            replay_pnl_by_side={"UP": 0.10, "DOWN": 0.0},
            label_vs_replay_pnl_gap=0.03,
        ),
    )
    true_down = _write_holdout_report(
        tmp_path / "true_down",
        _holdout_validation_report(
            run_id="true-down",
            market_ids=("market-down",),
            replay_rows=[
                _replay_row(
                    market_id="market-down",
                    decision_ts=20,
                    side="DOWN",
                    pnl=-0.02,
                )
            ],
            replay_pnl_by_side={"UP": 0.0, "DOWN": -0.02},
            label_vs_replay_pnl_gap=0.04,
        ),
    )
    blocked = _write_holdout_report(
        tmp_path / "blocked",
        _blocked_holdout_report(
            reason_codes=("frozen_corpus_dir_not_frozen_training_lineage",),
            replay_total_pnl_sum=999.0,
        ),
    )

    result = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=(true_up.parent, true_down, blocked.parent),
            output_dir=tmp_path / "accumulation",
            min_replay_entry_support=3,
            min_unique_market_support=2,
        )
    )

    report = result.report
    payload = dict(report)
    report_id = payload.pop("m_post_freeze_holdout_accumulation_report_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["schema_version"] == M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION
    assert report["loaded_report_count"] == 3
    assert report["holdout_run_count"] == 2
    assert report["duplicate_excluded_run_count"] == 0
    assert report["candidate_market_count"] == 2
    assert report["selected_market_count"] == 2
    assert report["replay_unique_market_count"] == 2
    assert report["unique_market_count"] == 2
    assert report["failed_provenance_run_count"] == 1
    assert report["selected_entry_count"] == 2
    assert report["replay_entry_count"] == 2
    assert report["replay_entry_count_by_side"] == {"UP": 1, "DOWN": 1}
    assert report["replay_total_pnl_sum"] == 0.08
    assert report["replay_pnl_by_side"] == {"UP": 0.10, "DOWN": -0.02}
    assert report["label_vs_replay_pnl_gap"] == 0.07
    assert report["support_gate_passed"] is False
    assert report["support_gate_reason_codes"] == [
        "insufficient_replay_entry_support"
    ]
    assert report["excluded_run_count"] == 1
    assert report["blocked_provenance_runs"][0]["reason_codes"] == [
        "frozen_corpus_dir_not_frozen_training_lineage"
    ]
    assert report["top_negative_replay_entries"][0]["market_id"] == "market-down"
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert result.artifact_paths["report"].exists()
    assert result.artifact_paths["summary"].exists()


def test_post_freeze_holdout_accumulation_support_gate_requires_markets_and_sides(
    tmp_path: Path,
) -> None:
    true_up = _write_holdout_report(
        tmp_path / "true_up",
        _holdout_validation_report(
            run_id="true-up",
            market_ids=("market-up",),
            replay_rows=[
                _replay_row(
                    market_id="market-up",
                    decision_ts=10,
                    side="UP",
                    pnl=0.10,
                )
            ],
            replay_pnl_by_side={"UP": 0.10, "DOWN": 0.0},
        ),
    )

    result = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=(true_up,),
            output_dir=tmp_path / "accumulation",
            min_replay_entry_support=1,
            min_unique_market_support=2,
        )
    )

    reason_codes = result.report["support_gate_reason_codes"]
    assert result.report["support_gate_passed"] is False
    assert result.report["candidate_market_count"] == 1
    assert result.report["selected_market_count"] == 1
    assert result.report["replay_unique_market_count"] == 1
    assert "insufficient_unique_market_support" in reason_codes
    assert "missing_down_replay_entry_support" in reason_codes
    assert "missing_up_replay_entry_support" not in reason_codes
    assert result.report["source_model_candidate_eligible"] is False
    assert result.report["#146_start_allowed"] is False
    assert result.report["#134_resume_allowed"] is False


def test_post_freeze_holdout_accumulation_dedupes_same_report_before_support(
    tmp_path: Path,
) -> None:
    true_both_sides = _write_holdout_report(
        tmp_path / "true_both_sides",
        _holdout_validation_report(
            run_id="true-both-sides",
            market_ids=("market-both",),
            replay_rows=[
                _replay_row(
                    market_id="market-both",
                    decision_ts=10,
                    side="UP",
                    pnl=0.10,
                ),
                _replay_row(
                    market_id="market-both",
                    decision_ts=20,
                    side="DOWN",
                    pnl=0.10,
                ),
            ],
            replay_pnl_by_side={"UP": 0.10, "DOWN": 0.10},
        ),
    )

    result = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=(true_both_sides, true_both_sides),
            output_dir=tmp_path / "accumulation",
            min_replay_entry_support=4,
            min_unique_market_support=1,
        )
    )

    report = result.report
    assert report["loaded_report_count"] == 2
    assert report["holdout_run_count"] == 1
    assert report["duplicate_excluded_run_count"] == 1
    assert report["candidate_market_count"] == 1
    assert report["selected_market_count"] == 1
    assert report["replay_unique_market_count"] == 1
    assert report["selected_entry_count"] == 2
    assert report["replay_entry_count"] == 2
    assert report["unique_market_count"] == 1
    assert report["replay_entry_count_by_side"] == {"UP": 1, "DOWN": 1}
    assert report["replay_total_pnl_sum"] == 0.20
    assert report["support_gate_passed"] is False
    assert report["support_gate_reason_codes"] == [
        "insufficient_replay_entry_support"
    ]
    duplicate = report["duplicate_excluded_runs"][0]
    assert "duplicate_report_sha256" in duplicate["duplicate_reason_codes"]
    assert "duplicate_report_id" in duplicate["duplicate_reason_codes"]
    assert "duplicate_run_id" in duplicate["duplicate_reason_codes"]
    assert "duplicate_holdout_corpus_window_market_ids" in duplicate[
        "duplicate_reason_codes"
    ]
    assert duplicate["dedupe_identity"]["market_ids"] == ("market-both",)
    assert report["promotion_evidence_eligible"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False


def test_post_freeze_holdout_accumulation_market_support_uses_replay_entries_only(
    tmp_path: Path,
) -> None:
    replay_rows = []
    pnl_by_side = {"UP": 0.0, "DOWN": 0.0}
    for index in range(20):
        side = "UP" if index % 2 == 0 else "DOWN"
        replay_rows.append(
            _replay_row(
                market_id="single-replay-market",
                decision_ts=10_000 + index,
                side=side,
                pnl=0.01,
            )
        )
        pnl_by_side[side] += 0.01
    candidate_only_rows = [
        _replay_row(
            market_id=f"candidate-only-market-{index:02d}",
            decision_ts=20_000 + index,
            side="UP" if index % 2 == 0 else "DOWN",
            pnl=0.0,
            side_quota_selected=False,
            entry_order_opened=False,
        )
        for index in range(12)
    ]
    report_path = _write_holdout_report(
        tmp_path / "inflated_candidate_markets",
        _holdout_validation_report(
            run_id="inflated-candidate-markets",
            market_ids=tuple(
                ["single-replay-market"]
                + [f"candidate-only-market-{index:02d}" for index in range(12)]
            ),
            replay_rows=replay_rows,
            replay_pnl_by_side=pnl_by_side,
            extra_rows=candidate_only_rows,
        ),
    )

    result = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=(report_path,),
            output_dir=tmp_path / "accumulation",
            min_replay_entry_support=20,
            min_unique_market_support=10,
        )
    )

    report = result.report
    assert report["candidate_market_count"] == 13
    assert report["selected_market_count"] == 1
    assert report["replay_unique_market_count"] == 1
    assert report["unique_market_count"] == 1
    assert report["replay_entry_count"] == 20
    assert report["replay_entry_count_by_side"] == {"UP": 10, "DOWN": 10}
    assert report["replay_total_pnl_sum"] == 0.20
    assert report["support_gate_passed"] is False
    assert report["support_gate_reason_codes"] == [
        "insufficient_unique_market_support"
    ]
    assert report["promotion_evidence_eligible"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False


def test_post_freeze_holdout_accumulation_support_ready_still_does_not_unlock(
    tmp_path: Path,
) -> None:
    rows = []
    pnl_by_side = {"UP": 0.0, "DOWN": 0.0}
    for index in range(20):
        side = "UP" if index % 2 == 0 else "DOWN"
        pnl = 0.01
        market_id = f"market-{index // 2}"
        rows.append(
            _replay_row(
                market_id=market_id,
                decision_ts=1_000 + index,
                side=side,
                pnl=pnl,
            )
        )
        pnl_by_side[side] += pnl
    true_report = _write_holdout_report(
        tmp_path / "true_ready",
        _holdout_validation_report(
            run_id="true-ready",
            market_ids=tuple(f"market-{index}" for index in range(10)),
            replay_rows=rows,
            replay_pnl_by_side=pnl_by_side,
        ),
    )

    result = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=(true_report,),
            output_dir=tmp_path / "accumulation",
        )
    )

    report = result.report
    assert report["support_gate_passed"] is True
    assert report["support_gate_reason_codes"] == []
    assert report["candidate_market_count"] == 10
    assert report["selected_market_count"] == 10
    assert report["replay_unique_market_count"] == 10
    assert report["promotion_evidence_eligible"] is True
    assert report["source_model_candidate_eligible"] is False
    assert report["source_model_candidate_ineligible_reason_codes"] == [
        "accumulation_report_diagnostic_only_no_source_eligibility_unlock"
    ]
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["paper_run_resume_allowed"] is False


def test_post_freeze_promotion_readiness_audit_explains_weak_evidence(
    tmp_path: Path,
) -> None:
    up_failed = _write_holdout_report(
        tmp_path / "up_failed",
        _holdout_validation_report(
            run_id="up-failed",
            market_ids=("up-failed-market",),
            replay_rows=[
                _replay_row(
                    market_id="up-failed-market",
                    decision_ts=10,
                    side="UP",
                    pnl=-0.05,
                )
            ],
            replay_pnl_by_side={"UP": -0.05, "DOWN": 0.0},
        ),
    )
    down_passed = _write_holdout_report(
        tmp_path / "down_passed",
        _holdout_validation_report(
            run_id="down-passed",
            market_ids=("down-passed-market",),
            replay_rows=[
                _replay_row(
                    market_id="down-passed-market",
                    decision_ts=20,
                    side="DOWN",
                    pnl=0.20,
                )
            ],
            replay_pnl_by_side={"UP": 0.0, "DOWN": 0.20},
        ),
    )
    accumulation = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=(up_failed, down_passed),
            output_dir=tmp_path / "accumulation",
            min_replay_entry_support=2,
            min_unique_market_support=2,
        )
    )

    result = run_polymarket_m_post_freeze_promotion_readiness_audit(
        PolymarketPostFreezePromotionReadinessAuditConfig(
            accumulation_report_path=accumulation.artifact_paths["report"],
            output_dir=tmp_path / "audit",
        )
    )

    report = result.report
    payload = dict(report)
    report_id = payload.pop("m_post_freeze_promotion_readiness_audit_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["schema_version"] == M_POST_FREEZE_PROMOTION_READINESS_AUDIT_SCHEMA_VERSION
    assert report["support_gate_passed"] is True
    assert report["promotion_evidence_eligible"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_readiness"] == "weak"
    assert report["promotion_gate_reason_codes"] == [
        "included_holdout_validation_not_passed"
    ]
    assert "included_holdout_validation_not_passed" in report[
        "source_model_candidate_ineligible_reason_codes"
    ]
    assert report["included_holdout_validation_failed_count"] == 1
    assert report["included_runs_with_holdout_validation_passed_false"][0][
        "replay_total_pnl_sum"
    ] == -0.05
    assert report["up_vs_down_pnl_imbalance"]["up_pnl"] == -0.05
    assert report["up_vs_down_pnl_imbalance"]["down_pnl"] == 0.20
    assert report["up_side_negative_pnl_should_block_promotion_discussion"] is True
    assert report["pnl_per_replay_entry_stats"]["minimum"] == -0.05
    assert report["pnl_per_replay_entry_stats"]["median"] == 0.07500000000000001
    assert report["pnl_per_replay_entry_stats"]["mean"] == 0.07500000000000001
    assert abs(report["largest_positive_entry_removed_total_pnl"] - -0.05) < 1e-12
    assert (
        report["total_pnl_remains_positive_if_largest_positive_entry_removed"]
        is False
    )
    assert (
        abs(
            report["leave_one_out_replay_pnl_sensitivity"][
                "minimum_leave_one_out_total_pnl"
            ]
            - -0.05
        )
        < 1e-12
    )
    assert report["top_negative_replay_entries"][0]["market_id"] == "up-failed-market"
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert result.artifact_paths["report"].exists()
    assert result.artifact_paths["summary"].exists()


def test_post_freeze_weak_evidence_drilldown_explains_root_causes(
    tmp_path: Path,
) -> None:
    up_failed_payload = _holdout_validation_report(
        run_id="up-failed",
        market_ids=("up-failed-market-a", "up-failed-market-b"),
        replay_rows=[
            _replay_row(
                market_id="up-failed-market-a",
                decision_ts=10,
                side="UP",
                pnl=-0.05,
            ),
            _replay_row(
                market_id="up-failed-market-b",
                decision_ts=20,
                side="UP",
                pnl=-0.02,
            ),
        ],
        replay_pnl_by_side={"UP": -0.07, "DOWN": 0.0},
    )
    up_failed_payload["ineligible_reason_codes"] = [
        "post_freeze_holdout_validation_not_passed",
        "diagnostic_only_no_paper_live_unlock",
    ]
    up_failed_payload["m_post_freeze_holdout_validation_report_id"] = (
        canonical_json_sha256(up_failed_payload)
    )
    up_failed = _write_holdout_report(tmp_path / "up_failed", up_failed_payload)

    blocked_selected_row = _replay_row(
        market_id="blocked-selected-market",
        decision_ts=40,
        side="DOWN",
        pnl=0.0,
        side_quota_selected=True,
        entry_order_opened=False,
    )
    blocked_selected_row["replay_reason_codes"] = [
        "turnover_guard_blocked",
        "max_entry_guard_blocked",
    ]
    blocked_selected_row["attrition_reason_codes"] = [
        "turnover_guard_blocked",
        "max_entry_guard_blocked",
    ]
    down_passed = _write_holdout_report(
        tmp_path / "down_passed",
        _holdout_validation_report(
            run_id="down-passed",
            market_ids=("down-passed-market", "blocked-selected-market"),
            replay_rows=[
                _replay_row(
                    market_id="down-passed-market",
                    decision_ts=30,
                    side="DOWN",
                    pnl=0.20,
                )
            ],
            replay_pnl_by_side={"UP": 0.0, "DOWN": 0.20},
            extra_rows=[blocked_selected_row],
        ),
    )
    accumulation = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=(up_failed, down_passed),
            output_dir=tmp_path / "accumulation",
            min_replay_entry_support=3,
            min_unique_market_support=3,
        )
    )
    audit = run_polymarket_m_post_freeze_promotion_readiness_audit(
        PolymarketPostFreezePromotionReadinessAuditConfig(
            accumulation_report_path=accumulation.artifact_paths["report"],
            output_dir=tmp_path / "audit",
        )
    )

    result = run_polymarket_m_post_freeze_weak_evidence_drilldown(
        PolymarketPostFreezeWeakEvidenceDrilldownConfig(
            promotion_readiness_audit_path=audit.artifact_paths["report"],
            accumulation_report_path=accumulation.artifact_paths["report"],
            output_dir=tmp_path / "drilldown",
        )
    )

    report = result.report
    payload = dict(report)
    report_id = payload.pop("m_post_freeze_weak_evidence_drilldown_report_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["schema_version"] == M_POST_FREEZE_WEAK_EVIDENCE_DRILLDOWN_SCHEMA_VERSION
    assert report["support_gate_passed"] is True
    assert report["promotion_evidence_eligible"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["root_cause_classification"] == "mixed"
    assert report["weakness_type"] == "structural_or_mixed_not_sample_size_driven"
    assert report["failed_included_holdout_run_count"] == 1
    assert report["failed_run_ineligible_reason_code_counts"][
        "post_freeze_holdout_validation_not_passed"
    ] == 1
    assert report["failed_runs_by_replay_pnl_sign"]["negative_count"] == 1
    assert report["up_loss_entry_count"] == 2
    assert report["down_loss_entry_count"] == 0
    assert report["selected_without_replay_run_count"] == 1
    assert report["selected_without_replay_row_count"] == 1
    assert report["turnover_or_max_entry_blocked_selected_row_count"] == 1
    assert report["root_cause_indicators"]["execution_attrition"] is True
    assert report["root_cause_indicators"]["side_imbalance"] is True
    assert report["root_cause_indicators"]["winner_concentration"] is True
    assert report["root_cause_indicators"]["structural_weakness"] is True
    assert report["largest_winner_dependency"][
        "winner_concentration_detected"
    ] is True
    assert (
        abs(
            report["largest_winner_dependency"][
                "total_pnl_after_largest_positive_entry_removed"
            ]
            - -0.07
        )
        < 1e-12
    )
    assert report["median_entry_pnl_weakness"]["median_entry_pnl_non_positive"] is True
    assert report["median_entry_pnl_weakness"]["median_entry_pnl"] == -0.02
    assert report["top_positive_replay_entries"][0]["market_id"] == "down-passed-market"
    assert report["top_negative_replay_entries"][0]["market_id"] == "up-failed-market-a"
    assert report["recommended_next_action"] == "keep_blocked"
    assert "investigate_side_specific_weakness" in report["recommended_next_actions"]
    assert "investigate_execution_attrition" in report["recommended_next_actions"]
    assert "reject_promotion_for_now" in report["recommended_next_actions"]
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert result.artifact_paths["report"].exists()
    assert result.artifact_paths["summary"].exists()


def _build_corpus(root: Path) -> Path:
    raw_dir = root / "raw"
    corpus_dir = root / "corpus"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=corpus_dir,
        )
    )
    return corpus_dir


def _patch_frozen_manifest_lineage(
    run_dir: Path,
    corpus_dir: Path,
    dataset_hash: str,
) -> None:
    manifest_path = run_dir / "polymarket_policy_model_manifest.json"
    manifest = _read_json(manifest_path)
    split = _read_json(corpus_dir / "polymarket_train_shadow_split.json")
    manifest["policy_dataset_hash"] = dataset_hash
    manifest["split_hash"] = split["split_hash"]
    manifest["train_dataset_hash"] = split["train_dataset_hash"]
    manifest["shadow_dataset_hash"] = split["shadow_dataset_hash"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_later_disjoint_corpus(
    *,
    source: Path,
    destination: Path,
    min_after_ts: int,
) -> Path:
    shutil.copytree(source, destination)
    feature_rows = _read_jsonl(destination / "polymarket_feature_rows.jsonl")
    source_min_ts = min(int(row["decision_ts"]) for row in feature_rows)
    offset = int(min_after_ts) + 60_000 - source_min_ts
    suffix = "-post-freeze-holdout"
    for path in destination.glob("*.jsonl"):
        rows = [_transform_payload(row, offset=offset, suffix=suffix) for row in _read_jsonl(path)]
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
    manifest_path = destination / "polymarket_corpus_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["post_freeze_holdout_test_transform"] = {
        "timestamp_offset_ms": offset,
        "id_suffix": suffix,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _transform_payload(
    value: Any,
    *,
    offset: int,
    suffix: str,
    key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _transform_payload(
                child_value,
                offset=offset,
                suffix=suffix,
                key=child_key,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _transform_payload(item, offset=offset, suffix=suffix, key=key)
            for item in value
        ]
    if isinstance(value, str) and key in {
        "market_id",
        "condition_id",
        "slug",
        "token_id",
        "up_token_id",
        "down_token_id",
    }:
        return value + suffix
    if isinstance(value, int) and (key == "ts" or str(key).endswith("_ts")):
        return value + offset
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_holdout_report(run_dir: Path, payload: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True)
    path = run_dir / "m_post_freeze_holdout_validation_report.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _holdout_validation_report(
    *,
    run_id: str,
    market_ids: tuple[str, ...],
    replay_rows: list[dict[str, Any]],
    replay_pnl_by_side: dict[str, float],
    label_vs_replay_pnl_gap: float = 0.0,
    extra_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = [*replay_rows, *(extra_rows or [])]
    replay_total_pnl = sum(float(row["total_polymarket_pnl"]) for row in replay_rows)
    report = {
        "schema_version": M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION,
        "run_id": run_id,
        "validation_status": "completed",
        "true_post_freeze_holdout": True,
        "prediction_attempted": True,
        "holdout_validation_passed": replay_total_pnl > 0.0,
        "provenance": {
            "holdout_corpus_dir": f"/tmp/{run_id}",
            "holdout_dataset_hash": canonical_json_sha256(
                {"run_id": run_id, "kind": "dataset"}
            ),
            "holdout_corpus_manifest_sha256": canonical_json_sha256(
                {"run_id": run_id, "kind": "corpus"}
            ),
            "holdout_training_corpus_hash": canonical_json_sha256(
                {"run_id": run_id, "kind": "corpus"}
            ),
            "holdout_min_decision_ts": min(
                int(row["decision_ts"]) for row in rows
            ),
            "holdout_max_decision_ts": max(
                int(row["decision_ts"]) for row in rows
            ),
            "market_id_overlap_count": 0,
        },
        "selected_entry_count": sum(
            1 for row in rows if bool(row.get("side_quota_selected", False))
        ),
        "replay_entry_count": len(replay_rows),
        "selected_exit_decision_count": 0,
        "replay_entry_reconciliation": {
            "selected_entry_count": len(replay_rows),
            "replay_entry_count": len(replay_rows),
            "selected_without_replay_entry_count": 0,
            "selected_exit_decision_count": 0,
            "reconciled": True,
        },
        "replay_total_pnl_sum": replay_total_pnl,
        "replay_pnl_by_side": replay_pnl_by_side,
        "label_vs_replay_pnl_gap": label_vs_replay_pnl_gap,
        "rows": rows,
        "reason_codes": [],
        "ineligible_reason_codes": ["diagnostic_only_no_paper_live_unlock"],
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    report["provenance"]["holdout_market_count"] = len(market_ids)
    report["provenance"]["holdout_market_ids"] = list(market_ids)
    report["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _blocked_holdout_report(
    *,
    reason_codes: tuple[str, ...],
    replay_total_pnl_sum: float = 0.0,
) -> dict[str, Any]:
    report = {
        "schema_version": M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION,
        "validation_status": "blocked_fail_closed",
        "true_post_freeze_holdout": False,
        "prediction_attempted": False,
        "holdout_validation_passed": False,
        "selected_entry_count": 0,
        "replay_entry_count": 0,
        "selected_exit_decision_count": 0,
        "replay_total_pnl_sum": replay_total_pnl_sum,
        "replay_pnl_by_side": {"UP": replay_total_pnl_sum, "DOWN": 0.0},
        "label_vs_replay_pnl_gap": 0.0,
        "rows": [
            _replay_row(
                market_id="blocked-market",
                decision_ts=1,
                side="UP",
                pnl=replay_total_pnl_sum,
            )
        ],
        "reason_codes": list(reason_codes),
        "ineligible_reason_codes": [*reason_codes, "diagnostic_only_no_paper_live_unlock"],
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    report["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _replay_row(
    *,
    market_id: str,
    decision_ts: int,
    side: str,
    pnl: float,
    side_quota_selected: bool = True,
    entry_order_opened: bool = True,
) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "selected_side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "side_quota_selected": side_quota_selected,
        "entry_order_opened": entry_order_opened,
        "raw_calibrated_action_score": 0.10,
        "best_action_margin": 0.01,
        "candidate_rank_score": 0.50,
        "action_return_target": pnl,
        "realized_trade_pnl": pnl,
        "settlement_pnl": 0.0,
        "total_polymarket_pnl": pnl,
        "exit_reason_codes": ["test_exit"],
        "replay_reason_codes": ["test_replay"],
        "attrition_stage": "final_pnl",
        "attrition_reason_codes": [],
    }
