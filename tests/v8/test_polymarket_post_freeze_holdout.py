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
from bigan.v8.polymarket.training.post_freeze_m2_replay_parity import (
    M2_REPLAY_PARITY_SCHEMA_VERSION,
    M2_UP_ALIGNMENT_SCHEMA_VERSION,
    PolymarketM2ReplayParityConfig,
    run_polymarket_m2_replay_parity_diagnostics,
)
from bigan.v8.polymarket.training.post_freeze_n2_up_feature_proxy import (
    N2_FORBIDDEN_SELECTION_FIELDS,
    N2_NON_LEAKY_UP_FEATURE_PROXY_CANDIDATE_SCHEMA_VERSION,
    N2_NON_LEAKY_UP_FEATURE_PROXY_SCORE_OVERLAY_SCHEMA_VERSION,
    PolymarketN2UpFeatureProxyConfig,
    run_polymarket_n2_up_feature_proxy_candidate,
)
from bigan.v8.polymarket.training.post_freeze_n_up_replay_aligned import (
    N_UP_REPLAY_ALIGNED_CANDIDATE_SCHEMA_VERSION,
    N_UP_REPLAY_ALIGNED_SCORE_OVERLAY_SCHEMA_VERSION,
    PolymarketNUpReplayAlignedConfig,
    run_polymarket_n_up_replay_aligned_candidate,
)
from bigan.v8.polymarket.training.post_freeze_promotion_readiness_audit import (
    M_POST_FREEZE_PROMOTION_READINESS_AUDIT_SCHEMA_VERSION,
    PolymarketPostFreezePromotionReadinessAuditConfig,
    run_polymarket_m_post_freeze_promotion_readiness_audit,
)
from bigan.v8.polymarket.training.post_freeze_up_diagnostics import (
    UP_ACTION_VALUE_CALIBRATION_SCHEMA_VERSION,
    UP_LABEL_REPLAY_ALIGNMENT_SCHEMA_VERSION,
    PolymarketUpSellBeforeCloseDiagnosticsConfig,
    run_polymarket_up_sell_before_close_diagnostics,
)
from bigan.v8.polymarket.training.post_freeze_up_full_candidate_pool import (
    UP_FULL_CANDIDATE_POOL_DIAGNOSTIC_SCHEMA_VERSION,
    UP_FULL_CANDIDATE_POOL_FEATURE_PROXY_SCHEMA_VERSION,
    PolymarketUpFullCandidatePoolConfig,
    run_polymarket_up_full_candidate_pool_diagnostics,
)
from bigan.v8.polymarket.training.post_freeze_weak_evidence_drilldown import (
    M_POST_FREEZE_WEAK_EVIDENCE_DRILLDOWN_SCHEMA_VERSION,
    PolymarketPostFreezeWeakEvidenceDrilldownConfig,
    run_polymarket_m_post_freeze_weak_evidence_drilldown,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
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


def test_m2_replay_parity_blocks_same_market_second_entry_and_reports_up_diagnostics(
    tmp_path: Path,
) -> None:
    same_market_second = _replay_row(
        market_id="market-a",
        decision_ts=20,
        side="UP",
        pnl=0.0,
        side_quota_selected=True,
        entry_order_opened=False,
    )
    same_market_second["action_return_target"] = -0.10
    same_market_second["replay_reason_codes"] = [
        "entry_blocked_turnover_guard",
        "entry_blocked_max_entries_per_market",
    ]
    same_market_second["attrition_reason_codes"] = [
        "entry_blocked_turnover_guard",
        "entry_blocked_max_entries_per_market",
    ]
    replacement = _replay_row(
        market_id="market-b",
        decision_ts=30,
        side="UP",
        pnl=0.0,
        side_quota_selected=False,
        entry_order_opened=False,
    )
    replacement["action_return_target"] = 0.15
    report_path = _write_holdout_report(
        tmp_path / "m2_source",
        _holdout_validation_report(
            run_id="m2-source",
            market_ids=("market-a", "market-b", "market-c", "market-d"),
            replay_rows=[
                _replay_row(
                    market_id="market-a",
                    decision_ts=10,
                    side="UP",
                    pnl=-0.05,
                ),
                _replay_row(
                    market_id="market-c",
                    decision_ts=40,
                    side="UP",
                    pnl=-0.02,
                ),
                _replay_row(
                    market_id="market-d",
                    decision_ts=50,
                    side="DOWN",
                    pnl=0.30,
                ),
            ],
            replay_pnl_by_side={"UP": -0.07, "DOWN": 0.30},
            extra_rows=[same_market_second, replacement],
        ),
    )
    payload = _read_json(report_path)
    for row in payload["rows"]:
        if row["market_id"] == "market-a" and row["decision_ts"] == 10:
            row["action_return_target"] = 0.20
            row["raw_calibrated_action_score"] = 0.90
            row["attrition_reason_codes"] = [
                "closed_before_settlement_with_negative_replay_pnl"
            ]
        if row["market_id"] == "market-c":
            row["action_return_target"] = -0.20
            row["raw_calibrated_action_score"] = 0.80
            row["attrition_reason_codes"] = [
                "closed_before_settlement_with_negative_replay_pnl"
            ]
    payload["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        payload
    )
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    accumulation = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=(report_path,),
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
    drilldown = run_polymarket_m_post_freeze_weak_evidence_drilldown(
        PolymarketPostFreezeWeakEvidenceDrilldownConfig(
            promotion_readiness_audit_path=audit.artifact_paths["report"],
            accumulation_report_path=accumulation.artifact_paths["report"],
            output_dir=tmp_path / "drilldown",
        )
    )

    result = run_polymarket_m2_replay_parity_diagnostics(
        PolymarketM2ReplayParityConfig(
            weak_evidence_drilldown_report_path=drilldown.artifact_paths["report"],
            accumulation_report_path=accumulation.artifact_paths["report"],
            output_dir=tmp_path / "m2",
        )
    )

    candidate = result.candidate_report
    up = result.up_alignment_report
    payload = dict(candidate)
    report_id = payload.pop("m2_stateful_replay_parity_candidate_report_id")
    assert canonical_json_sha256(payload) == report_id
    assert candidate["schema_version"] == M2_REPLAY_PARITY_SCHEMA_VERSION
    assert candidate["candidate_name"] == SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME
    assert candidate["baseline_candidate_name"] == (
        SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME
    )
    assert candidate["current_frozen_m_promotion_status"] == "reject_promotion_for_now"
    assert candidate["current_frozen_m_evidence_status"] == "weak_mixed_structural"
    assert candidate["m2_turnover_or_max_entry_attrition_count"] == 0
    assert candidate["turnover_or_max_entry_attrition_reduced_to_zero"] is True
    assert candidate["m2_selected_without_replay_count"] == 1
    assert candidate["m2_replay_entry_reconciliation"]["reconciled"] is False
    assert candidate["m2_replay_entry_reconciliation"]["failure_reason_codes"] == [
        "m2_selected_rows_missing_replay_evidence"
    ]
    assert any(
        row["market_id"] == "market-a"
        and row["decision_ts"] == 20
        and "m2_entry_blocked_max_entries_per_market" in row["m2_reason_codes"]
        for row in candidate["m2_blocked_rows"]
    )
    assert any(
        row["market_id"] == "market-b"
        and row["m2_side_quota_selected"] is True
        for row in candidate["m2_selected_rows"]
    )
    up_payload = dict(up)
    up_report_id = up_payload.pop("m2_up_label_replay_alignment_diagnostic_id")
    assert canonical_json_sha256(up_payload) == up_report_id
    assert up["schema_version"] == M2_UP_ALIGNMENT_SCHEMA_VERSION
    assert up["m2_up_negative_label_selected_count"] >= 1
    assert up["m2_up_positive_label_replay_negative_count"] == 1
    assert up["m2_up_first_executable_exit_negative_count"] == 2
    assert up["m2_top_up_false_positives"][0]["market_id"] == "market-a"
    assert candidate["source_model_candidate_eligible"] is False
    assert candidate["promotion_evidence_eligible"] is False
    assert candidate["#146_start_allowed"] is False
    assert candidate["#134_resume_allowed"] is False
    assert up["#146_start_allowed"] is False
    assert up["#134_resume_allowed"] is False
    assert result.artifact_paths["candidate_report"].exists()
    assert result.artifact_paths["up_alignment_report"].exists()


def test_up_sell_before_close_diagnostics_explain_label_and_score_weakness(
    tmp_path: Path,
) -> None:
    up_negative_label = _replay_row(
        market_id="up-negative-label",
        decision_ts=10,
        side="UP",
        pnl=-0.12,
    )
    up_negative_label.update(
        {
            "action_return_target": -0.05,
            "raw_calibrated_action_score": 0.92,
            "candidate_rank_score": 0.10,
            "entry_quality_ask": 0.55,
            "exit_quality_bid": 0.48,
            "entry_exit_quality_spread_bps": 700.0,
            "entry_exit_quality_queue_fill": 0.70,
            "closed_before_settlement": True,
            "attrition_reason_codes": [
                "closed_before_settlement_with_negative_replay_pnl"
            ],
        }
    )
    up_positive_label_negative_replay = _replay_row(
        market_id="up-positive-label-negative-replay",
        decision_ts=20,
        side="UP",
        pnl=-0.08,
    )
    up_positive_label_negative_replay.update(
        {
            "action_return_target": 0.20,
            "raw_calibrated_action_score": 0.85,
            "candidate_rank_score": 0.20,
            "entry_quality_ask": 0.60,
            "exit_quality_bid": 0.50,
            "entry_exit_quality_spread_bps": 800.0,
            "entry_exit_quality_queue_fill": 0.60,
            "closed_before_settlement": True,
            "attrition_reason_codes": [
                "closed_before_settlement_with_negative_replay_pnl"
            ],
        }
    )
    up_low_score_positive = _replay_row(
        market_id="up-low-score-positive",
        decision_ts=30,
        side="UP",
        pnl=0.15,
    )
    up_low_score_positive.update(
        {
            "action_return_target": 0.10,
            "raw_calibrated_action_score": 0.30,
            "candidate_rank_score": -0.10,
            "entry_quality_ask": 0.40,
            "exit_quality_bid": 0.47,
            "entry_exit_quality_spread_bps": 500.0,
            "entry_exit_quality_queue_fill": 0.80,
        }
    )
    down_positive = _replay_row(
        market_id="down-positive",
        decision_ts=40,
        side="DOWN",
        pnl=0.30,
    )
    down_positive.update(
        {
            "action_return_target": 0.20,
            "raw_calibrated_action_score": 0.60,
            "candidate_rank_score": 0.15,
            "entry_quality_ask": 0.50,
            "exit_quality_bid": 0.60,
            "entry_exit_quality_spread_bps": 300.0,
            "entry_exit_quality_queue_fill": 0.90,
        }
    )
    source_report = _write_holdout_report(
        tmp_path / "source",
        _holdout_validation_report(
            run_id="up-diagnostic-source",
            market_ids=(
                "up-negative-label",
                "up-positive-label-negative-replay",
                "up-low-score-positive",
                "down-positive",
            ),
            replay_rows=[
                up_negative_label,
                up_positive_label_negative_replay,
                up_low_score_positive,
                down_positive,
            ],
            replay_pnl_by_side={"UP": -0.05, "DOWN": 0.30},
        ),
    )
    source_payload = _read_json(source_report)
    source_payload["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        source_payload
    )
    source_report.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_rows = []
    for row in source_payload["rows"]:
        row = dict(row)
        row["source_report_path"] = str(source_report)
        row["m2_side_quota_selected"] = True
        row["m2_reason_codes"] = ["m2_stateful_replay_parity_selected"]
        selected_rows.append(row)
    m2_report = {
        "schema_version": M2_REPLAY_PARITY_SCHEMA_VERSION,
        "candidate_name": SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        "baseline_candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "diagnostic_only": True,
        "current_frozen_m_promotion_status": "reject_promotion_for_now",
        "current_frozen_m_evidence_status": "weak_mixed_structural",
        "current_frozen_m_evidence_reused_for_m2_promotion": False,
        "m2_selected_rows": selected_rows,
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    m2_report["m2_stateful_replay_parity_candidate_report_id"] = canonical_json_sha256(
        m2_report
    )
    m2_report_path = tmp_path / "m2_stateful_replay_parity_candidate_report.json"
    m2_report_path.write_text(
        json.dumps(m2_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = run_polymarket_up_sell_before_close_diagnostics(
        PolymarketUpSellBeforeCloseDiagnosticsConfig(
            m2_candidate_report_path=m2_report_path,
            output_dir=tmp_path / "up_diagnostics",
        )
    )

    label = result.label_replay_report
    label_payload = dict(label)
    label_id = label_payload.pop(
        "up_sell_before_close_label_replay_alignment_report_id"
    )
    assert canonical_json_sha256(label_payload) == label_id
    assert label["schema_version"] == UP_LABEL_REPLAY_ALIGNMENT_SCHEMA_VERSION
    assert label["up_negative_label_selected_count"] == 1
    assert label["up_positive_label_replay_negative_count"] == 1
    assert label["first_executable_exit_negative_count"] == 2
    assert label["up_top_false_positives"][0]["market_id"] == "up-negative-label"
    assert "up_label_target_optimistic" in label["root_cause_codes"]
    assert "up_executable_exit_path_mismatch" in label["root_cause_codes"]
    assert label["root_cause_classification"] == "mixed"
    assert "introduce_up_executable_exit_label_correction" in label[
        "recommended_next_actions"
    ]
    assert label["#146_start_allowed"] is False
    assert label["#134_resume_allowed"] is False

    calibration = result.calibration_report
    calibration_payload = dict(calibration)
    calibration_id = calibration_payload.pop(
        "up_sell_before_close_action_value_calibration_diagnostic_id"
    )
    assert canonical_json_sha256(calibration_payload) == calibration_id
    assert (
        calibration["schema_version"]
        == UP_ACTION_VALUE_CALIBRATION_SCHEMA_VERSION
    )
    assert (
        calibration[
            "calibrated_action_score_vs_realized_up_replay_pnl_correlation"
        ]
        < 0.0
    )
    assert calibration["high_score_negative_replay_up_count"] == 2
    assert calibration["low_score_positive_replay_up_count"] == 1
    assert "up_action_value_overcalibrated" in calibration["root_cause_codes"]
    assert "up_rank_score_false_positive_bias" in calibration["root_cause_codes"]
    assert calibration["source_model_candidate_eligible"] is False
    assert calibration["promotion_evidence_eligible"] is False
    assert calibration["#146_start_allowed"] is False
    assert calibration["#134_resume_allowed"] is False
    assert result.artifact_paths["label_replay_report"].exists()
    assert result.artifact_paths["calibration_report"].exists()


def test_n_up_replay_aligned_candidate_flags_up_false_positives_fail_closed(
    tmp_path: Path,
) -> None:
    up_negative_label = _replay_row(
        market_id="n-up-negative-label",
        decision_ts=10,
        side="UP",
        pnl=-0.12,
    )
    up_negative_label.update(
        {
            "action_return_target": -0.05,
            "raw_calibrated_action_score": 0.92,
            "candidate_rank_score": 0.10,
        }
    )
    up_positive_label_negative_replay = _replay_row(
        market_id="n-up-positive-label-negative-replay",
        decision_ts=20,
        side="UP",
        pnl=-0.08,
    )
    up_positive_label_negative_replay.update(
        {
            "action_return_target": 0.20,
            "raw_calibrated_action_score": 0.85,
            "candidate_rank_score": 0.20,
        }
    )
    up_positive_replay = _replay_row(
        market_id="n-up-positive-replay",
        decision_ts=30,
        side="UP",
        pnl=0.15,
    )
    up_positive_replay.update(
        {
            "action_return_target": 0.10,
            "raw_calibrated_action_score": 0.30,
            "candidate_rank_score": -0.10,
        }
    )
    down_reference = _replay_row(
        market_id="n-down-reference",
        decision_ts=40,
        side="DOWN",
        pnl=0.30,
    )
    source_report = _write_holdout_report(
        tmp_path / "n_source",
        _holdout_validation_report(
            run_id="n-up-replay-aligned-source",
            market_ids=(
                "n-up-negative-label",
                "n-up-positive-label-negative-replay",
                "n-up-positive-replay",
                "n-down-reference",
            ),
            replay_rows=[
                up_negative_label,
                up_positive_label_negative_replay,
                up_positive_replay,
                down_reference,
            ],
            replay_pnl_by_side={"UP": -0.05, "DOWN": 0.30},
        ),
    )
    source_payload = _read_json(source_report)
    source_payload["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        source_payload
    )
    source_report.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_rows = []
    for row in source_payload["rows"]:
        row = dict(row)
        row["source_report_path"] = str(source_report)
        row["m2_side_quota_selected"] = True
        row["m2_reason_codes"] = ["m2_stateful_replay_parity_selected"]
        selected_rows.append(row)
    m2_report = {
        "schema_version": M2_REPLAY_PARITY_SCHEMA_VERSION,
        "candidate_name": SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        "baseline_candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "diagnostic_only": True,
        "current_frozen_m_promotion_status": "reject_promotion_for_now",
        "current_frozen_m_evidence_status": "weak_mixed_structural",
        "current_frozen_m_evidence_reused_for_m2_promotion": False,
        "m2_selected_rows": selected_rows,
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    m2_report["m2_stateful_replay_parity_candidate_report_id"] = canonical_json_sha256(
        m2_report
    )
    m2_report_path = tmp_path / "m2_stateful_replay_parity_candidate_report.json"
    m2_report_path.write_text(
        json.dumps(m2_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_m2_bytes = m2_report_path.read_bytes()

    result = run_polymarket_n_up_replay_aligned_candidate(
        PolymarketNUpReplayAlignedConfig(
            m2_candidate_report_path=m2_report_path,
            output_dir=tmp_path / "n_up_replay_aligned",
        )
    )

    assert m2_report_path.read_bytes() == original_m2_bytes
    candidate = result.candidate_report
    candidate_payload = dict(candidate)
    candidate_id = candidate_payload.pop("n_up_replay_aligned_candidate_report_id")
    assert canonical_json_sha256(candidate_payload) == candidate_id
    assert candidate["schema_version"] == N_UP_REPLAY_ALIGNED_CANDIDATE_SCHEMA_VERSION
    assert (
        candidate["candidate_name"]
        == SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME
    )
    assert candidate["m2_up_selected_count"] == 3
    assert candidate["n_would_selected_up_count"] == 1
    assert candidate["n_would_blocked_up_count"] == 2
    assert candidate["n_would_selected_up_replay_pnl_sum"] == 0.15
    assert candidate["n_blocked_up_false_positive_count"] == 1
    assert (
        candidate["m2_up_label_vs_replay_gap"]
        > candidate["n_label_vs_replay_gap_after_correction"]
    )
    blocked_by_market = {
        row["market_id"]: row for row in candidate["n_would_blocked_rows"]
    }
    high_score_row = blocked_by_market["n-up-positive-label-negative-replay"]
    assert high_score_row["high_score_negative_replay_guard_triggered"] is True
    assert high_score_row["positive_label_replay_negative_flagged"] is True
    assert "n_blocked_high_score_negative_replay" in high_score_row[
        "n_decision_reason_codes"
    ]
    negative_label_row = blocked_by_market["n-up-negative-label"]
    assert negative_label_row["negative_label_selected_flagged"] is True
    assert "n_flagged_negative_label_selected" in negative_label_row[
        "n_decision_reason_codes"
    ]
    assert candidate["source_model_candidate_eligible"] is False
    assert candidate["promotion_evidence_eligible"] is False
    assert candidate["#146_start_allowed"] is False
    assert candidate["#134_resume_allowed"] is False
    assert candidate["paper_only"] is True
    assert candidate["capital_at_risk"] is False

    overlay = result.score_overlay_report
    overlay_payload = dict(overlay)
    overlay_id = overlay_payload.pop("n_up_replay_aligned_score_overlay_report_id")
    assert canonical_json_sha256(overlay_payload) == overlay_id
    assert overlay["schema_version"] == N_UP_REPLAY_ALIGNED_SCORE_OVERLAY_SCHEMA_VERSION
    assert overlay["original_score_vs_replay_correlation"] < 0.0
    assert (
        overlay["replay_aligned_score_proxy_vs_replay_correlation"]
        > overlay["original_score_vs_replay_correlation"]
    )
    assert overlay["source_model_candidate_eligible"] is False
    assert overlay["promotion_evidence_eligible"] is False
    assert overlay["#146_start_allowed"] is False
    assert overlay["#134_resume_allowed"] is False
    assert result.artifact_paths["candidate_report"].exists()
    assert result.artifact_paths["score_overlay_report"].exists()


def test_n2_non_leaky_up_feature_proxy_blocks_without_future_fields(
    tmp_path: Path,
) -> None:
    up_bad_proxy_positive_future = _replay_row(
        market_id="n2-bad-proxy-positive-future",
        decision_ts=10,
        side="UP",
        pnl=1.00,
    )
    up_bad_proxy_positive_future.update(
        {
            "action_return_target": 1.00,
            "raw_calibrated_action_score": 0.95,
            "best_action_margin": 0.03,
            "execution_pnl_immediate_exit_pnl": -0.05,
            "entry_quality_ask": 0.60,
            "exit_quality_bid": 0.55,
            "entry_exit_quality_spread_bps": 1200.0,
            "entry_exit_quality_queue_fill": 0.50,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "up_recent_book_update_count_1m": 5.0,
        }
    )
    up_positive_label_negative_replay = _replay_row(
        market_id="n2-positive-label-negative-replay",
        decision_ts=20,
        side="UP",
        pnl=-0.08,
    )
    up_positive_label_negative_replay.update(
        {
            "action_return_target": 0.20,
            "raw_calibrated_action_score": 0.90,
            "best_action_margin": 0.02,
            "execution_pnl_immediate_exit_pnl": -0.03,
            "entry_quality_ask": 0.50,
            "exit_quality_bid": 0.47,
            "entry_exit_quality_spread_bps": 700.0,
            "entry_exit_quality_queue_fill": 0.70,
            "entry_exit_quality_book_staleness_ms": 700.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "up_recent_book_update_count_1m": 5.0,
        }
    )
    up_good_proxy_positive = _replay_row(
        market_id="n2-good-proxy-positive",
        decision_ts=30,
        side="UP",
        pnl=0.15,
    )
    up_good_proxy_positive.update(
        {
            "action_return_target": 0.10,
            "raw_calibrated_action_score": 0.55,
            "best_action_margin": 0.02,
            "execution_pnl_immediate_exit_pnl": 0.04,
            "entry_quality_ask": 0.40,
            "exit_quality_bid": 0.44,
            "entry_exit_quality_spread_bps": 300.0,
            "entry_exit_quality_queue_fill": 0.90,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "up_recent_book_update_count_1m": 5.0,
        }
    )
    up_negative_label_good_proxy = _replay_row(
        market_id="n2-negative-label-good-proxy",
        decision_ts=40,
        side="UP",
        pnl=0.05,
    )
    up_negative_label_good_proxy.update(
        {
            "action_return_target": -0.05,
            "raw_calibrated_action_score": 0.50,
            "best_action_margin": 0.02,
            "execution_pnl_immediate_exit_pnl": 0.04,
            "entry_quality_ask": 0.40,
            "exit_quality_bid": 0.44,
            "entry_exit_quality_spread_bps": 300.0,
            "entry_exit_quality_queue_fill": 0.90,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "up_recent_book_update_count_1m": 5.0,
        }
    )
    down_reference = _replay_row(
        market_id="n2-down-reference",
        decision_ts=50,
        side="DOWN",
        pnl=0.30,
    )
    source_report = _write_holdout_report(
        tmp_path / "n2_source",
        _holdout_validation_report(
            run_id="n2-non-leaky-source",
            market_ids=(
                "n2-bad-proxy-positive-future",
                "n2-positive-label-negative-replay",
                "n2-good-proxy-positive",
                "n2-negative-label-good-proxy",
                "n2-down-reference",
            ),
            replay_rows=[
                up_bad_proxy_positive_future,
                up_positive_label_negative_replay,
                up_good_proxy_positive,
                up_negative_label_good_proxy,
                down_reference,
            ],
            replay_pnl_by_side={"UP": 1.12, "DOWN": 0.30},
        ),
    )
    source_payload = _read_json(source_report)
    source_payload["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        source_payload
    )
    source_report.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_rows = []
    for row in source_payload["rows"]:
        row = dict(row)
        row["source_report_path"] = str(source_report)
        row["m2_side_quota_selected"] = True
        row["m2_reason_codes"] = ["m2_stateful_replay_parity_selected"]
        selected_rows.append(row)
    m2_report = {
        "schema_version": M2_REPLAY_PARITY_SCHEMA_VERSION,
        "candidate_name": SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        "baseline_candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "diagnostic_only": True,
        "current_frozen_m_promotion_status": "reject_promotion_for_now",
        "current_frozen_m_evidence_status": "weak_mixed_structural",
        "current_frozen_m_evidence_reused_for_m2_promotion": False,
        "m2_selected_rows": selected_rows,
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    m2_report["m2_stateful_replay_parity_candidate_report_id"] = canonical_json_sha256(
        m2_report
    )
    m2_report_path = tmp_path / "m2_stateful_replay_parity_candidate_report.json"
    m2_report_path.write_text(
        json.dumps(m2_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    n_result = run_polymarket_n_up_replay_aligned_candidate(
        PolymarketNUpReplayAlignedConfig(
            m2_candidate_report_path=m2_report_path,
            output_dir=tmp_path / "n_up_replay_aligned",
        )
    )
    n_report_path = n_result.artifact_paths["candidate_report"]
    original_m2_bytes = m2_report_path.read_bytes()
    original_n_bytes = n_report_path.read_bytes()

    result = run_polymarket_n2_up_feature_proxy_candidate(
        PolymarketN2UpFeatureProxyConfig(
            m2_candidate_report_path=m2_report_path,
            n_candidate_report_path=n_report_path,
            output_dir=tmp_path / "n2_up_feature_proxy",
        )
    )

    assert m2_report_path.read_bytes() == original_m2_bytes
    assert n_report_path.read_bytes() == original_n_bytes
    candidate = result.candidate_report
    candidate_payload = dict(candidate)
    candidate_id = candidate_payload.pop(
        "n2_non_leaky_up_feature_proxy_candidate_report_id"
    )
    assert canonical_json_sha256(candidate_payload) == candidate_id
    assert (
        candidate["schema_version"]
        == N2_NON_LEAKY_UP_FEATURE_PROXY_CANDIDATE_SCHEMA_VERSION
    )
    assert (
        candidate["candidate_name"]
        == SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME
    )
    assert candidate["n2_selection_uses_only_allowed_fields"] is True
    assert candidate["n_reference_report"]["provided"] is True
    assert candidate["n2_would_selected_up_count"] == 2
    assert candidate["n2_would_blocked_up_count"] == 2
    assert candidate["n2_blocked_up_false_positive_count"] == 1
    assert candidate["n2_would_selected_up_replay_pnl_sum"] == 0.20
    blocked_by_market = {
        row["market_id"]: row for row in candidate["n2_would_blocked_rows"]
    }
    positive_future = blocked_by_market["n2-bad-proxy-positive-future"]
    assert positive_future["realized_replay_pnl"] == 1.0
    assert "n2_blocked_nonpositive_immediate_exit_proxy" in positive_future[
        "n2_decision_reason_codes"
    ]
    assert "n2_blocked_spread_too_wide" in positive_future[
        "n2_decision_reason_codes"
    ]
    false_positive = blocked_by_market["n2-positive-label-negative-replay"]
    assert (
        false_positive["positive_label_replay_negative_flagged_for_evaluation"]
        is True
    )
    assert "n2_blocked_nonpositive_immediate_exit_proxy" in false_positive[
        "n2_decision_reason_codes"
    ]
    selected_by_market = {
        row["market_id"]: row for row in candidate["n2_would_selected_rows"]
    }
    negative_label = selected_by_market["n2-negative-label-good-proxy"]
    assert negative_label["negative_label_selected_flagged_for_evaluation"] is True
    assert negative_label["n2_decision_reason_codes"] == [
        "n2_non_leaky_feature_proxy_would_select"
    ]
    for row in candidate["original_up_selected_rows"]:
        assert not set(row["n2_decision_input_fields_used"]).intersection(
            N2_FORBIDDEN_SELECTION_FIELDS
        )
        assert row["n2_forbidden_fields_used_for_selection"] == []
    assert candidate["source_model_candidate_eligible"] is False
    assert candidate["promotion_evidence_eligible"] is False
    assert candidate["#146_start_allowed"] is False
    assert candidate["#134_resume_allowed"] is False
    assert candidate["paper_only"] is True
    assert candidate["capital_at_risk"] is False

    overlay = result.score_overlay_report
    overlay_payload = dict(overlay)
    overlay_id = overlay_payload.pop(
        "n2_non_leaky_up_feature_proxy_score_overlay_report_id"
    )
    assert canonical_json_sha256(overlay_payload) == overlay_id
    assert (
        overlay["schema_version"]
        == N2_NON_LEAKY_UP_FEATURE_PROXY_SCORE_OVERLAY_SCHEMA_VERSION
    )
    assert overlay["n2_selection_uses_only_allowed_fields"] is True
    assert overlay["source_model_candidate_eligible"] is False
    assert overlay["promotion_evidence_eligible"] is False
    assert overlay["#146_start_allowed"] is False
    assert overlay["#134_resume_allowed"] is False
    assert result.artifact_paths["candidate_report"].exists()
    assert result.artifact_paths["score_overlay_report"].exists()


def test_up_full_candidate_pool_diagnostic_segments_selected_and_non_selected(
    tmp_path: Path,
) -> None:
    selected_up = _replay_row(
        market_id="pool-selected-up",
        decision_ts=10,
        side="UP",
        pnl=0.10,
    )
    selected_up.update(
        {
            "action_return_target": 0.08,
            "raw_calibrated_action_score": 0.50,
            "best_action_margin": 0.02,
            "execution_pnl_immediate_exit_pnl": 0.03,
            "entry_quality_ask": 0.40,
            "exit_quality_bid": 0.43,
            "entry_exit_quality_spread_bps": 300.0,
            "entry_exit_quality_queue_fill": 0.90,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "up_recent_book_update_count_1m": 5.0,
            "guard_compatible_candidate": True,
            "pre_guard_candidate": True,
        }
    )
    non_selected_viable = _replay_row(
        market_id="pool-non-selected-viable",
        decision_ts=20,
        side="UP",
        pnl=0.12,
        side_quota_selected=False,
        entry_order_opened=False,
    )
    non_selected_viable.update(
        {
            "action_return_target": 0.10,
            "raw_calibrated_action_score": 0.55,
            "best_action_margin": 0.02,
            "execution_pnl_immediate_exit_pnl": 0.04,
            "entry_quality_ask": 0.40,
            "exit_quality_bid": 0.44,
            "entry_exit_quality_spread_bps": 280.0,
            "entry_exit_quality_queue_fill": 0.92,
            "entry_exit_quality_book_staleness_ms": 400.0,
            "entry_exit_quality_time_to_close_seconds": 210.0,
            "up_recent_book_update_count_1m": 5.0,
            "guard_compatible_candidate": True,
            "pre_guard_candidate": True,
        }
    )
    non_selected_future_winner_bad_proxy = _replay_row(
        market_id="pool-non-selected-future-winner-bad-proxy",
        decision_ts=30,
        side="UP",
        pnl=1.00,
        side_quota_selected=False,
        entry_order_opened=False,
    )
    non_selected_future_winner_bad_proxy.update(
        {
            "action_return_target": 1.00,
            "raw_calibrated_action_score": 0.95,
            "best_action_margin": 0.03,
            "execution_pnl_immediate_exit_pnl": -0.05,
            "entry_quality_ask": 0.60,
            "exit_quality_bid": 0.55,
            "entry_exit_quality_spread_bps": 1200.0,
            "entry_exit_quality_queue_fill": 0.50,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "up_recent_book_update_count_1m": 5.0,
            "guard_compatible_candidate": True,
            "pre_guard_candidate": True,
        }
    )
    guard_incompatible_up = _replay_row(
        market_id="pool-guard-incompatible-up",
        decision_ts=40,
        side="UP",
        pnl=0.50,
        side_quota_selected=False,
        entry_order_opened=False,
    )
    guard_incompatible_up.update(
        {
            "action_return_target": 0.50,
            "raw_calibrated_action_score": 0.70,
            "best_action_margin": 0.02,
            "execution_pnl_immediate_exit_pnl": 0.04,
            "entry_quality_ask": 0.40,
            "exit_quality_bid": 0.44,
            "entry_exit_quality_spread_bps": 300.0,
            "entry_exit_quality_queue_fill": 0.90,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "up_recent_book_update_count_1m": 5.0,
            "guard_compatible_candidate": False,
            "pre_guard_candidate": True,
        }
    )
    down_reference = _replay_row(
        market_id="pool-down-reference",
        decision_ts=50,
        side="DOWN",
        pnl=0.30,
    )
    down_reference.update({"guard_compatible_candidate": True})
    source_report = _write_holdout_report(
        tmp_path / "pool_source",
        _holdout_validation_report(
            run_id="up-full-pool-source",
            market_ids=(
                "pool-selected-up",
                "pool-non-selected-viable",
                "pool-non-selected-future-winner-bad-proxy",
                "pool-guard-incompatible-up",
                "pool-down-reference",
            ),
            replay_rows=[
                selected_up,
                non_selected_viable,
                non_selected_future_winner_bad_proxy,
                guard_incompatible_up,
                down_reference,
            ],
            replay_pnl_by_side={"UP": 1.72, "DOWN": 0.30},
        ),
    )
    source_payload = _read_json(source_report)
    source_payload["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        source_payload
    )
    source_report.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_payload = dict(selected_up)
    selected_payload["source_report_path"] = str(source_report)
    selected_payload["m2_side_quota_selected"] = True
    selected_payload["m2_reason_codes"] = ["m2_stateful_replay_parity_selected"]
    m2_report = {
        "schema_version": M2_REPLAY_PARITY_SCHEMA_VERSION,
        "candidate_name": SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        "baseline_candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "diagnostic_only": True,
        "current_frozen_m_promotion_status": "reject_promotion_for_now",
        "current_frozen_m_evidence_status": "weak_mixed_structural",
        "current_frozen_m_evidence_reused_for_m2_promotion": False,
        "m2_selected_rows": [selected_payload],
        "m2_blocked_rows": [],
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    m2_report["m2_stateful_replay_parity_candidate_report_id"] = canonical_json_sha256(
        m2_report
    )
    m2_report_path = tmp_path / "m2_stateful_replay_parity_candidate_report.json"
    m2_report_path.write_text(
        json.dumps(m2_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_m2_bytes = m2_report_path.read_bytes()

    result = run_polymarket_up_full_candidate_pool_diagnostics(
        PolymarketUpFullCandidatePoolConfig(
            m2_candidate_report_path=m2_report_path,
            output_dir=tmp_path / "up_full_pool",
        )
    )

    assert m2_report_path.read_bytes() == original_m2_bytes
    pool = result.candidate_pool_report
    pool_payload = dict(pool)
    pool_id = pool_payload.pop("up_full_candidate_pool_diagnostic_report_id")
    assert canonical_json_sha256(pool_payload) == pool_id
    assert pool["schema_version"] == UP_FULL_CANDIDATE_POOL_DIAGNOSTIC_SCHEMA_VERSION
    assert pool["total_up_candidate_pool_size"] == 4
    assert pool["guard_compatible_up_pool_size"] == 3
    assert pool["m2_selected_up_count"] == 1
    assert pool["m2_non_selected_guard_compatible_up_count"] == 2
    assert pool["non_selected_up_rows_viable_under_non_leaky_proxy_count"] == 1
    assert pool["up_path_should_remain_fully_blocked"] is False
    assert pool["pool_segment_metrics"]["m2_selected"]["row_count"] == 1
    assert pool["pool_segment_metrics"]["m2_non_selected"]["row_count"] == 2
    viable = pool["non_selected_up_rows_viable_under_non_leaky_proxy"][0]
    assert viable["market_id"] == "pool-non-selected-viable"
    rows_by_market = {row["market_id"]: row for row in pool["rows"]}
    bad_future = rows_by_market["pool-non-selected-future-winner-bad-proxy"]
    assert bad_future["realized_replay_pnl"] == 1.0
    assert bad_future["n2_would_select"] is False
    assert "n2_blocked_nonpositive_immediate_exit_proxy" in bad_future[
        "n2_decision_reason_codes"
    ]
    for row in pool["rows"]:
        assert not set(row["n2_decision_input_fields_used"]).intersection(
            N2_FORBIDDEN_SELECTION_FIELDS
        )
        assert row["n2_forbidden_fields_used_for_selection"] == []
    assert pool["source_model_candidate_eligible"] is False
    assert pool["promotion_evidence_eligible"] is False
    assert pool["#146_start_allowed"] is False
    assert pool["#134_resume_allowed"] is False
    assert pool["paper_only"] is True
    assert pool["capital_at_risk"] is False

    proxy = result.feature_proxy_report
    proxy_payload = dict(proxy)
    proxy_id = proxy_payload.pop("up_full_candidate_pool_feature_proxy_report_id")
    assert canonical_json_sha256(proxy_payload) == proxy_id
    assert proxy["schema_version"] == UP_FULL_CANDIDATE_POOL_FEATURE_PROXY_SCHEMA_VERSION
    assert proxy["selection_uses_only_allowed_fields"] is True
    assert proxy["n2_would_selected_full_pool_count"] == 2
    assert proxy["n2_would_selected_non_selected_pool_count"] == 1
    assert proxy["source_model_candidate_eligible"] is False
    assert proxy["promotion_evidence_eligible"] is False
    assert proxy["#146_start_allowed"] is False
    assert proxy["#134_resume_allowed"] is False
    assert result.artifact_paths["candidate_pool_report"].exists()
    assert result.artifact_paths["feature_proxy_report"].exists()


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
