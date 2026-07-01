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
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    O_DEPLOYABLE_MODEL_FEATURE_NAMES,
    O_FEATURE_AND_LABEL_LEAKAGE_AUDIT_SCHEMA_VERSION,
    O_FEATURE_SET_SELECTION_SCHEMA_VERSION,
    O_FREEZE_READINESS_SCHEMA_VERSION,
    O_HTS_P_UP_CONFIDENTLY_WRONG_FEATURE_DIAGNOSTIC_SCHEMA_VERSION,
    O_JOINT_FEATURE_CORRECTION_SELECTION_SCHEMA_VERSION,
    O_LABEL_CONSTRUCTION_SCHEMA_VERSION,
    O_LABEL_DIAGNOSTIC_VARIANTS,
    O_MODEL_PREDICTED_VARIANT,
    O_SOURCE_CANDIDATE_COMPARISON_SCHEMA_VERSION,
    O_SOURCE_MODEL_ELIGIBILITY_GATE_SCHEMA_VERSION,
    O_SOURCE_RANKING_OBJECTIVE_SCHEMA_VERSION,
    PolymarketOReplayAlignedSourceRankingConfig,
    run_polymarket_o_replay_aligned_source_ranking,
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
    REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
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


def test_o_replay_aligned_source_ranking_reports_fail_closed_without_mutation(
    tmp_path: Path,
) -> None:
    weak_up = _replay_row(
        market_id="o-market-a",
        decision_ts=10,
        side="UP",
        pnl=-0.10,
    )
    weak_up.update(
        {
            "action_return_target": 0.50,
            "raw_calibrated_action_score": 0.95,
            "best_action_margin": 0.10,
            "execution_pnl_immediate_exit_pnl": -0.02,
            "entry_quality_ask": 0.60,
            "exit_quality_bid": 0.58,
            "entry_exit_quality_spread_bps": 700.0,
            "entry_exit_quality_queue_fill": 0.70,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "p_up": 0.30,
        }
    )
    better_down = _replay_row(
        market_id="o-market-a",
        decision_ts=20,
        side="DOWN",
        pnl=0.12,
    )
    better_down.update(
        {
            "action_return_target": 0.12,
            "raw_calibrated_action_score": 0.30,
            "best_action_margin": 0.03,
            "execution_pnl_immediate_exit_pnl": 0.04,
            "entry_quality_ask": 0.40,
            "exit_quality_bid": 0.44,
            "entry_exit_quality_spread_bps": 250.0,
            "entry_exit_quality_queue_fill": 0.92,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "p_up": 0.70,
        }
    )
    weak_down = _replay_row(
        market_id="o-market-b",
        decision_ts=30,
        side="DOWN",
        pnl=-0.08,
    )
    weak_down.update(
        {
            "action_return_target": 0.20,
            "raw_calibrated_action_score": 0.80,
            "best_action_margin": 0.06,
            "execution_pnl_immediate_exit_pnl": -0.03,
            "entry_quality_ask": 0.55,
            "exit_quality_bid": 0.52,
            "entry_exit_quality_spread_bps": 600.0,
            "entry_exit_quality_queue_fill": 0.75,
            "entry_exit_quality_book_staleness_ms": 500.0,
            "entry_exit_quality_time_to_close_seconds": 180.0,
            "p_up": 0.30,
        }
    )
    source_report = _write_holdout_report(
        tmp_path / "o_source",
        _holdout_validation_report(
            run_id="o-replay-aligned-source",
            market_ids=("o-market-a", "o-market-b"),
            replay_rows=[weak_up, better_down, weak_down],
            replay_pnl_by_side={"UP": -0.10, "DOWN": 0.04},
        ),
    )
    source_payload = _read_json(source_report)
    label_rows = []
    for market_id, decision_ts, targets in (
        (
            "o-market-a",
            10,
            {
                "BUY_UP_SELL_BEFORE_CLOSE": -0.10,
                "BUY_DOWN_SELL_BEFORE_CLOSE": 0.08,
                "BUY_UP_HOLD_TO_SETTLEMENT": -0.20,
                "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.12,
                "NO_TRADE": 0.0,
            },
        ),
        (
            "o-market-a",
            20,
            {
                "BUY_UP_SELL_BEFORE_CLOSE": -0.05,
                "BUY_DOWN_SELL_BEFORE_CLOSE": 0.12,
                "BUY_UP_HOLD_TO_SETTLEMENT": -0.30,
                "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.20,
                "NO_TRADE": 0.0,
            },
        ),
        (
            "o-market-b",
            30,
            {
                "BUY_UP_SELL_BEFORE_CLOSE": -0.03,
                "BUY_DOWN_SELL_BEFORE_CLOSE": -0.08,
                "BUY_UP_HOLD_TO_SETTLEMENT": -0.10,
                "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.20,
                "NO_TRADE": 0.0,
            },
        ),
    ):
        for action, target in targets.items():
            label_rows.append(
                _o_label_row(
                    market_id=market_id,
                    decision_ts=decision_ts,
                    action=action,
                    target=target,
                )
            )
    holdout_corpus_dir = tmp_path / "o_holdout_corpus"
    holdout_corpus_dir.mkdir()
    (holdout_corpus_dir / "polymarket_label_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in label_rows) + "\n",
        encoding="utf-8",
    )
    feature_rows = [
        _o_feature_row(
            market_id="o-market-a",
            decision_ts=10,
            btc_mid_price=101.0,
            up_depth=150.0,
            down_depth=90.0,
            up_update_count=8,
            down_update_count=3,
        ),
        _o_feature_row(
            market_id="o-market-a",
            decision_ts=20,
            btc_mid_price=99.0,
            up_depth=80.0,
            down_depth=180.0,
            up_update_count=2,
            down_update_count=9,
        ),
        _o_feature_row(
            market_id="o-market-b",
            decision_ts=30,
            btc_mid_price=100.4,
            up_depth=110.0,
            down_depth=100.0,
            up_update_count=4,
            down_update_count=4,
        ),
    ]
    (holdout_corpus_dir / "polymarket_feature_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in feature_rows) + "\n",
        encoding="utf-8",
    )
    market_metadata_rows = [
        {"market_id": "o-market-a", "market_start_ts": 0},
        {"market_id": "o-market-b", "market_start_ts": 0},
    ]
    (holdout_corpus_dir / "polymarket_market_metadata.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in market_metadata_rows)
        + "\n",
        encoding="utf-8",
    )
    btc_reference_rows = [
        {
            "ts": 0,
            "available_at_ts": 60_000,
            "open_price": 100.0,
            "high_price": 100.0,
            "low_price": 100.0,
            "close_price": 100.0,
            "volume": 1.0,
            "timeframe_ms": 60_000,
            "source": "test_reference_feed",
        }
    ]
    (holdout_corpus_dir / "polymarket_btc_reference_candles.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in btc_reference_rows)
        + "\n",
        encoding="utf-8",
    )
    source_payload["provenance"] = {"holdout_corpus_dir": str(holdout_corpus_dir)}
    source_payload["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        source_payload
    )
    source_report.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_payload = dict(weak_up)
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

    result = run_polymarket_o_replay_aligned_source_ranking(
        PolymarketOReplayAlignedSourceRankingConfig(
            m2_candidate_report_path=m2_report_path,
            output_dir=tmp_path / "o_replay_aligned",
        )
    )

    assert m2_report_path.read_bytes() == original_m2_bytes
    labels = result.label_construction_report
    label_payload = dict(labels)
    label_id = label_payload.pop("o_replay_aligned_label_construction_report_id")
    assert canonical_json_sha256(label_payload) == label_id
    assert labels["schema_version"] == O_LABEL_CONSTRUCTION_SCHEMA_VERSION
    assert labels["candidate_name"] == REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME
    assert labels["row_count"] == 15
    assert labels["decision_group_count"] == 3
    market_a_groups = {
        row["decision_group_id"]
        for row in labels["label_rows"]
        if row["market_id"] == "o-market-a"
    }
    assert len(market_a_groups) == 2
    assert all(
        group_id.endswith("|10") or group_id.endswith("|20")
        for group_id in market_a_groups
    )
    assert labels["decision_group_completeness_summary"][
        "partial_decision_group_count"
    ] == 0
    assert labels["decision_group_completeness_summary"][
        "complete_decision_group_count"
    ] == 3
    assert labels["decision_group_completeness_summary"][
        "ranking_metric_scope"
    ] == "full_decision_group"
    assert labels["action_candidate_construction_summary"][
        "complete_action_candidate_grid"
    ] is True
    assert labels["action_candidate_construction_summary"]["action_counts"] == {
        "BUY_DOWN_HOLD_TO_SETTLEMENT": 3,
        "BUY_DOWN_SELL_BEFORE_CLOSE": 3,
        "BUY_UP_HOLD_TO_SETTLEMENT": 3,
        "BUY_UP_SELL_BEFORE_CLOSE": 3,
        "NO_TRADE": 3,
    }
    assert labels["action_candidate_construction_summary"][
        "candidate_label_source_counts"
    ] == {"holdout_corpus_label_rows": 15}
    assert any(row["action"] == "NO_TRADE" for row in labels["label_rows"])
    assert all(
        row["ranking_metric_scope"] == "full_decision_group"
        for row in labels["label_rows"]
    )
    assert any(
        row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
        for row in labels["label_rows"]
    )
    assert (
        labels["label_component_field_classes"][
            "first_executable_exit_bid_after_entry"
        ]
        == "replay_derived_label_only"
    )
    assert labels["source_model_candidate_eligible"] is False
    assert labels["#146_start_allowed"] is False
    assert labels["#134_resume_allowed"] is False

    ranking = result.ranking_objective_report
    ranking_payload = dict(ranking)
    ranking_id = ranking_payload.pop("o_source_ranking_objective_report_id")
    assert canonical_json_sha256(ranking_payload) == ranking_id
    assert ranking["schema_version"] == O_SOURCE_RANKING_OBJECTIVE_SCHEMA_VERSION
    assert ranking["primary_variant_name"] == O_MODEL_PREDICTED_VARIANT
    assert ranking["primary_ranking_score_source"] == "model_predicted_score"
    assert ranking["model_predicted_candidate_name"] == O_MODEL_PREDICTED_VARIANT
    assert ranking["deployable_model_score_available"] is True
    assert ranking["label_diagnostic_variants"] == list(O_LABEL_DIAGNOSTIC_VARIANTS)
    assert ranking["label_diagnostic_variants_deployable"] is False
    assert "o_replay_aligned_labels_family_priors" in ranking[
        "ranking_metric_by_variant"
    ]
    assert O_MODEL_PREDICTED_VARIANT in ranking["ranking_metric_by_variant"]
    assert ranking["ranking_metric_scope"] == "full_decision_group"
    assert ranking["full_source_model_ranking_quality_claimed"] is True
    assert ranking["o_model_training_summary"]["ranking_score_source"] == (
        "model_predicted_score"
    )
    assert (
        ranking["o_model_training_summary"]["correction_constants_source"]
        == "shadow_split_only"
    )
    assert (
        ranking["o_model_training_summary"]["probe_constants_source"]
        == "shadow_split_only"
    )
    assert looks_like_sha256(
        ranking["o_model_training_summary"]["correction_config_hash"]
    )
    assert looks_like_sha256(ranking["o_model_training_summary"]["probe_config_hash"])
    assert (
        ranking["correction_constants_source"]
        == ranking["o_model_training_summary"]["correction_constants_source"]
    )
    assert (
        ranking["correction_config_hash"]
        == ranking["o_model_training_summary"]["correction_config_hash"]
    )
    assert (
        ranking["probe_constants_source"]
        == ranking["o_model_training_summary"]["probe_constants_source"]
    )
    assert (
        ranking["probe_config_hash"]
        == ranking["o_model_training_summary"]["probe_config_hash"]
    )
    assert ranking["o_model_training_summary"][
        "deployable_model_score_available"
    ] is True
    selected_feature_names = ranking["o_model_training_summary"]["feature_names"]
    all_candidate_feature_names = ranking["o_model_training_summary"][
        "all_candidate_feature_names"
    ]
    assert ranking["o_model_training_summary"][
        "model_input_fields_decision_time_only"
    ] == selected_feature_names
    assert all_candidate_feature_names == list(O_DEPLOYABLE_MODEL_FEATURE_NAMES)
    assert ranking["selected_feature_set_name"] == ranking[
        "o_model_training_summary"
    ]["selected_feature_set_name"]
    assert ranking["selected_correction_policy_name"] == ranking[
        "o_model_training_summary"
    ]["selected_correction_policy_name"]
    assert ranking["selected_high_score_threshold_profile_name"] == ranking[
        "o_model_training_summary"
    ]["selected_high_score_threshold_profile_name"]
    assert ranking["selected_joint_candidate_name"] == ranking[
        "o_model_training_summary"
    ]["selected_joint_candidate_name"]
    assert looks_like_sha256(
        ranking["o_model_training_summary"]["selected_feature_set_config_hash"]
    )
    assert looks_like_sha256(
        ranking["o_model_training_summary"][
            "joint_feature_correction_selection_config_hash"
        ]
    )
    assert ranking["o_model_training_summary"]["post_model_ranking_correction_enabled"] is True
    assert ranking["o_model_training_summary"][
        "ranking_correction_source"
    ] == "shadow_split_only"
    assert ranking["o_model_training_summary"]["ranking_correction_config"][
        "uses_validation_labels_for_tuning"
    ] is False
    correction_config = ranking["o_model_training_summary"][
        "ranking_correction_config"
    ]
    assert correction_config["correction_constants_source"] == "shadow_split_only"
    assert correction_config["correction_constants_are_shadow_derived"] is True
    assert correction_config["probe_constants_source"] == "shadow_split_only"
    assert looks_like_sha256(correction_config["probe_config_hash"])
    assert correction_config["trade_base_score_source"] == "0.5 + shadow_p_up_edge_q75"
    assert (
        correction_config["sell_before_close_base_score_source"]
        == "0.5 + shadow_p_up_edge_q25 / 2"
    )
    assert correction_config["no_trade_base_score_source"] == (
        "0.5 + shadow_p_up_edge_median"
    )
    assert correction_config["confidence_bonus_source"] == "shadow_p_up_edge_median"
    assert (
        correction_config["weak_opportunity_trade_penalty_source"]
        == "-shadow_p_up_edge_q25"
    )
    assert correction_config["microstructure_quality_weight_source"] == (
        "shadow_microstructure_target_correlation_scaled_by_p_edge_q25"
    )
    assert correction_config["p_up_misalignment_raw_positive_penalty_source"] == (
        "shadow_candidate_search_p_up_edge_quantile_grid"
    )
    assert correction_config["p_up_misalignment_penalty_applies_to"] == (
        "buy_actions_with_negative_p_up_alignment_and_positive_raw_component"
    )
    assert correction_config["large_regret_reversal_guard_enabled"] is True
    assert correction_config["large_regret_reversal_guard_source"] == (
        "shadow_split_only_hold_to_settlement_action_pair_regret_priors"
    )
    assert correction_config["large_regret_reversal_guard_modes"] == (
        "raw_p_up_opposition_confidence_veto",
        "hold_to_settlement_high_reversal_exposure_veto",
    )
    assert correction_config["large_regret_reversal_guard_applies_to"] == (
        "hold_to_settlement_buy_actions_with_positive_raw_component_"
        "and_opposite_p_up_alignment_or_high_reversal_exposure"
    )
    assert correction_config[
        "large_regret_reversal_confidence_edge_ceiling_source"
    ] == "shadow_p_up_edge_q25_plus_q75"
    assert correction_config[
        "large_regret_reversal_pair_regret_threshold_source"
    ] == "shadow_hold_to_settlement_up_down_positive_reversal_regret_median"
    assert set(correction_config["large_regret_reversal_pair_regret_priors"]) == {
        "BUY_DOWN_HOLD_TO_SETTLEMENT->BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_UP_HOLD_TO_SETTLEMENT->BUY_DOWN_HOLD_TO_SETTLEMENT",
    }
    assert correction_config["large_regret_reversal_alignment_threshold_source"] == (
        "shadow_candidate_search_p_up_edge_quantile_grid"
    )
    assert correction_config["large_regret_reversal_penalty_source"] == (
        "shadow_candidate_search_largest_regret_reversal_grid"
    )
    assert correction_config["hts_p_up_reliability_guard_enabled"] is True
    assert correction_config["hts_p_up_reliability_guard_source"] == (
        "shadow_split_only_hts_p_up_side_bucket_regret"
    )
    assert correction_config["hts_p_up_reliability_guard_applies_to"] == (
        "hold_to_settlement_actions_matching_p_up_implied_side_in_"
        "high_shadow_regret_side_confidence_or_microstructure_bucket"
    )
    assert set(correction_config["hts_p_up_reliability_bucket_thresholds"]) == {
        "p_up_confidence",
        "queue",
        "spread",
        "staleness",
        "threshold_source",
        "time_to_close",
    }
    assert correction_config["hts_p_up_reliability_bucket_thresholds"][
        "threshold_source"
    ] == "shadow_split_decision_group_feature_quantiles"
    assert isinstance(correction_config["hts_p_up_reliability_regime_priors"], dict)
    assert set(correction_config["hts_p_up_reliability_bucket_diagnostics"]) == {
        "by_p_up_confidence_bucket",
        "by_queue_bucket",
        "by_selected_vs_oracle_side",
        "by_spread_bucket",
        "by_staleness_bucket",
        "by_time_to_close_bucket",
    }
    assert correction_config["hts_p_up_reliability_regret_threshold_source"] == (
        "shadow_hts_p_up_positive_regret_q25"
    )
    assert correction_config["hts_p_up_reliability_min_support_source"] == (
        "max(1, floor(sqrt(shadow_hts_group_count) / 2))"
    )
    assert correction_config["hts_p_up_reliability_penalty_source"] in {
        "shadow_p_up_edge_quantile_grid",
        "shadow_p_up_edge_q75_lightweight_feature_set_selection",
    }
    assert correction_config["hts_p_up_reliability_no_trade_buffer_enabled"] is True
    assert correction_config["hts_p_up_reliability_no_trade_buffer_source"] == (
        "shadow_p_up_edge_q25"
    )
    assert correction_config["hts_p_up_reliability_no_trade_buffer"] >= (
        correction_config["p_up_edge_quantiles"]["q25"]
    )
    assert correction_config["p_up_safety_target_disagreement_rate"] == 0.25
    assert correction_config["p_up_safety_target_source"] == (
        "config_hashed_stricter_than_hard_gate_target"
    )
    assert correction_config["shadow_p_up_selection_max_disagreement_rate_source"] == (
        "max_p_up_action_disagreement_rate_minus_shadow_p_up_edge_q75"
    )
    assert looks_like_sha256(correction_config["correction_config_hash"])
    assert set(correction_config["p_up_edge_quantiles"]) == {"median", "q25", "q75"}
    assert set(correction_config["shadow_component_diagnostics"]) == {
        "microstructure",
        "prior",
        "raw_model",
    }
    probe_config = correction_config["probe_score_config"]
    assert probe_config["probe_constants_source"] == "shadow_split_only"
    assert probe_config["uses_validation_labels_for_tuning"] is False
    assert looks_like_sha256(probe_config["probe_config_hash"])
    assert probe_config["probe_no_trade_base_score_source"] == (
        "0.5 + max(0, -shadow_global_target_mean)"
    )
    assert probe_config["probe_no_trade_weak_edge_cutoff_source"] == (
        "shadow_p_up_edge_median"
    )
    assert probe_config["probe_hold_to_settlement_base_score_source"] == (
        "0.5 + shadow_p_up_edge_median + positive_shadow_hold_to_settlement_prior"
    )
    assert probe_config["probe_sell_before_close_base_score_source"] == (
        "0.5 - shadow_p_up_edge_q25 + positive_shadow_sell_before_close_prior"
    )
    assert probe_config["probe_sell_before_close_alignment_weight_source"] == (
        "0.5 + shadow_p_up_edge_q25 + positive_shadow_global_target_mean"
    )
    raw_diagnostics = correction_config["shadow_component_diagnostics"]["raw_model"]
    assert raw_diagnostics["raw_weight_selection_metric_source"] in {
        "shadow_split_only",
        "shadow_split_only_lightweight_feature_set_selection",
    }
    assert "shadow_candidate_eligible" in raw_diagnostics[
        "selected_raw_weight_candidate"
    ]
    assert (
        raw_diagnostics["selected_raw_weight_candidate"][
            "candidate_weight_source"
        ]
        in {
            "shadow_p_up_edge_quantile_grid",
            "shadow_p_up_edge_q75_lightweight_feature_set_selection",
        }
    )
    assert raw_diagnostics["raw_weight_candidate_rows"]
    assert raw_diagnostics["p_up_misalignment_penalty_candidate_source"] in {
        "shadow_p_up_edge_quantile_grid",
        "shadow_p_up_edge_q25_lightweight_feature_set_selection",
    }
    assert raw_diagnostics["large_regret_reversal_guard_candidate_source"] in {
        "shadow_largest_regret_reversal_grid",
        "shadow_p_up_edge_q75_lightweight_feature_set_selection",
    }
    assert (
        raw_diagnostics["large_regret_reversal_guard_selection_metric_source"]
        in {
            "shadow_split_only",
            "shadow_split_only_lightweight_feature_set_selection",
        }
    )
    assert raw_diagnostics["hts_p_up_reliability_guard_candidate_source"] in {
        "shadow_p_up_edge_quantile_grid",
        "shadow_p_up_edge_q75_lightweight_feature_set_selection",
    }
    assert (
        raw_diagnostics["hts_p_up_reliability_guard_selection_metric_source"]
        in {
            "shadow_split_only",
            "shadow_split_only_lightweight_feature_set_selection",
        }
    )
    assert (
        raw_diagnostics[
            "hts_p_up_reliability_no_trade_buffer_excluded_from_raw_weight_search"
        ]
        is True
    )
    assert raw_diagnostics[
        "hts_p_up_reliability_no_trade_buffer_application_stage"
    ] in {
        "post_shadow_raw_weight_selection_safety_buffer",
        "post_lightweight_feature_set_selection_safety_buffer",
    }
    assert (
        raw_diagnostics["hts_p_up_reliability_bucket_thresholds"]
        == correction_config["hts_p_up_reliability_bucket_thresholds"]
    )
    assert (
        raw_diagnostics["hts_p_up_reliability_bucket_diagnostics"]
        == correction_config["hts_p_up_reliability_bucket_diagnostics"]
    )
    assert (
        raw_diagnostics["large_regret_reversal_pair_regret_priors"]
        == correction_config["large_regret_reversal_pair_regret_priors"]
    )
    assert raw_diagnostics["raw_weight_max_shadow_p_up_disagreement_rate"] == (
        correction_config["shadow_p_up_selection_max_disagreement_rate"]
    )
    assert raw_diagnostics["raw_weight_p_up_safety_buffer"] >= 0.0
    assert "shadow_largest_regret_case" in raw_diagnostics[
        "selected_raw_weight_candidate"
    ]
    assert "shadow_action_family_level_regret" in raw_diagnostics[
        "selected_raw_weight_candidate"
    ]
    assert "shadow_action_pair_regret_summary" in raw_diagnostics[
        "selected_raw_weight_candidate"
    ]
    assert "shadow_hold_to_settlement_up_down_reversal_regret" in raw_diagnostics[
        "selected_raw_weight_candidate"
    ]
    assert "candidate_hts_p_up_reliability_penalty" in raw_diagnostics[
        "selected_raw_weight_candidate"
    ]
    assert "shadow_hts_p_up_reliability_regret_summary" in raw_diagnostics[
        "selected_raw_weight_candidate"
    ]
    assert "shadow_no_trade_missed_opportunity" in raw_diagnostics[
        "selected_raw_weight_candidate"
    ]
    assert ranking["o_model_training_summary"]["ranking_correction_config"][
        "NO_TRADE_prior"
    ]["enabled"] is True
    assert "buy_up_hold_to_settlement_x_p_up" in ranking[
        "o_model_training_summary"
    ]["feature_names"]
    assert "buy_down_hold_to_settlement_x_p_down" in ranking[
        "o_model_training_summary"
    ]["feature_names"]
    assert "buy_up_sell_before_close_x_time_to_close" in ranking[
        "o_model_training_summary"
    ]["feature_names"]
    assert "buy_down_sell_before_close_x_spread" in ranking[
        "o_model_training_summary"
    ]["feature_names"]
    assert "buy_down_hold_to_settlement_x_queue" in ranking[
        "o_model_training_summary"
    ]["feature_names"]
    assert "buy_up_hold_to_settlement_x_staleness" in ranking[
        "o_model_training_summary"
    ]["feature_names"]
    assert "buy_down_hold_to_settlement_x_entry_ask" in ranking[
        "o_model_training_summary"
    ]["feature_names"]
    assert "buy_down_hold_to_settlement_x_exit_bid_proxy" in ranking[
        "o_model_training_summary"
    ]["feature_names"]
    assert "reference_price_to_beat_distance_scaled" in all_candidate_feature_names
    assert "recent_reference_price_momentum_30s_scaled" in all_candidate_feature_names
    assert "side_book_depth_imbalance" in all_candidate_feature_names
    assert "side_book_update_velocity_scaled" in all_candidate_feature_names
    assert "hts_vs_sell_before_close_exit_value_gap_proxy" in all_candidate_feature_names
    assert "p_up_bucket_calibration_residual" in all_candidate_feature_names
    coverage = ranking["o_model_training_summary"][
        "decision_time_feature_coverage"
    ]
    assert coverage["feature_row_available_count"] == 15
    assert coverage["feature_provenance_violation_count"] == 0
    assert coverage["field_coverage"][
        "reference_price_to_beat_distance_at_decision"
    ]["used_as_model_input"] is (
        "reference_price_to_beat_distance_scaled" in selected_feature_names
    )
    assert coverage["field_coverage"][
        "reference_price_to_beat_distance_at_decision"
    ]["available_count"] == 15
    reference_effect = ranking["o_model_training_summary"][
        "reference_price_feature_effect_summary"
    ]
    assert reference_effect["diagnostic_only"] is True
    assert reference_effect["uses_validation_labels_for_tuning"] is False
    assert (
        reference_effect[
            "reference_price_to_beat_distance_available_count"
        ]
        == 15
    )
    assert (
        reference_effect["final_shadow_corrected_gate_remains_fail_closed"]
        is True
    )
    assert coverage["field_coverage"]["side_book_depth_imbalance"][
        "available_count"
    ] == 12
    assert coverage["field_coverage"]["recent_reference_price_momentum_120s"][
        "missing_count"
    ] == 15
    ablation = ranking["o_model_training_summary"]["feature_ablation_diagnostics"]
    assert ablation["uses_validation_labels_for_tuning"] is False
    assert set(ablation["feature_sets"]) == {
        "old_features_only",
        "new_reference_price_features",
        "new_book_pressure_features",
        "combined_feature_set",
    }
    assert (
        ablation["feature_sets"]["combined_feature_set"]["feature_count"]
        > ablation["feature_sets"]["old_features_only"]["feature_count"]
    )
    feature_selection = result.feature_set_selection_report
    assert feature_selection["schema_version"] == O_FEATURE_SET_SELECTION_SCHEMA_VERSION
    assert feature_selection["uses_validation_labels_for_tuning"] is False
    assert feature_selection["selection_metric_source"] == "shadow_split_only"
    assert feature_selection["feature_set_selection_min_high_score_support_count"] == 5
    assert feature_selection["source_model_gate_min_high_score_support_count"] == 10
    assert feature_selection["feature_set_selection_derived_from_joint_selection"] is True
    assert feature_selection["selected_correction_policy_name"] == ranking[
        "selected_correction_policy_name"
    ]
    assert feature_selection["selected_high_score_threshold_profile_name"] == ranking[
        "selected_high_score_threshold_profile_name"
    ]
    assert feature_selection["selected_joint_candidate_name"] == ranking[
        "selected_joint_candidate_name"
    ]
    assert {row["feature_set_name"] for row in feature_selection["candidate_feature_sets"]} == {
        "old_features_only",
        "book_pressure_features",
        "reference_price_features",
        "combined_features",
        "combined_minus_reference_distance",
    }
    assert feature_selection["selected_feature_set_name"] == ranking[
        "selected_feature_set_name"
    ]
    assert feature_selection["selected_feature_names"] == selected_feature_names
    assert looks_like_sha256(
        feature_selection["feature_set_selection_config_hash"]
    )
    assert feature_selection["#146_start_allowed"] is False
    assert feature_selection["#134_resume_allowed"] is False
    assert "feature_set_selection_report" in result.artifact_paths
    joint_selection = result.joint_feature_correction_selection_report
    assert (
        joint_selection["schema_version"]
        == O_JOINT_FEATURE_CORRECTION_SELECTION_SCHEMA_VERSION
    )
    assert joint_selection["uses_validation_labels_for_tuning"] is False
    assert joint_selection["selection_metric_source"] == "shadow_split_only"
    assert joint_selection["selected_feature_set_name"] == ranking[
        "selected_feature_set_name"
    ]
    assert joint_selection["selected_correction_policy_name"] == ranking[
        "selected_correction_policy_name"
    ]
    assert joint_selection["selected_high_score_threshold_profile_name"] == ranking[
        "selected_high_score_threshold_profile_name"
    ]
    assert joint_selection["selected_joint_candidate_name"] == ranking[
        "selected_joint_candidate_name"
    ]
    assert {
        row["correction_policy_name"] for row in joint_selection["candidate_rows"]
    } == {
        "balanced_hts_sbc",
        "conservative_hts",
        "high_score_profitability_preserving",
        "no_trade_tail_risk_buffer",
        "sbc_preferred_when_hts_reliability_weak",
    }
    assert {
        row["high_score_threshold_profile_name"]
        for row in joint_selection["candidate_rows"]
    } == {
        "current_threshold",
        "high_score_profitability_threshold",
        "slightly_lower_shadow_derived_threshold",
        "support_preserving_threshold",
    }
    assert joint_selection["candidate_count"] == 100
    assert joint_selection["selected_high_score_threshold_profile"][
        "uses_validation_labels_for_tuning"
    ] is False
    assert joint_selection["selected_full_correction_rerun_diagnostics"][
        "full_correction_rerun_enabled"
    ] is True
    assert joint_selection["selected_full_correction_rerun_diagnostics"][
        "full_correction_search_source"
    ] == "shadow_split_only"
    assert joint_selection["selected_lightweight_preselection_candidate_row"][
        "joint_candidate_name"
    ] == joint_selection["selected_joint_candidate_name"]
    assert joint_selection["selected_final_full_correction_candidate_row"][
        "joint_candidate_name"
    ] == joint_selection["selected_joint_candidate_name"]
    assert all(
        "shadow_high_score_support_deficit_to_source_gate" in row
        for row in joint_selection["candidate_rows"]
    )
    assert looks_like_sha256(
        joint_selection["joint_feature_correction_selection_config_hash"]
    )
    assert joint_selection["#146_start_allowed"] is False
    assert joint_selection["#134_resume_allowed"] is False
    assert "joint_feature_correction_selection_report" in result.artifact_paths
    residual = ranking["o_model_training_summary"][
        "p_up_bucket_calibration_residual_summary"
    ]
    assert residual["residual_source"] == "shadow_split_only"
    assert residual["uses_validation_labels_for_tuning"] is False
    assert ranking["o_model_training_summary"][
        "training_target"
    ] == "replay_aligned_executable_label_target"
    assert ranking["eligibility_metric_source"] == "validation_metrics_only"
    assert set(ranking["train_shadow_metrics"]) >= {
        "decision_group_count",
        "top1_realized_best_action_hit_rate",
        "top2_realized_best_action_hit_rate",
        "top3_realized_best_action_hit_rate",
        "selected_action_realized_replay_return_sum",
        "oracle_executable_best_action_return_sum",
        "mean_regret",
        "high_score_support_count",
        "high_score_realized_return_mean",
        "high_score_realized_return_sum",
        "NO_TRADE_selection_rate",
        "action_family_selected_return_breakdown",
        "side_selected_return_breakdown",
        "largest_winner_dependency",
        "largest_regret_case",
        "action_family_level_regret",
        "side_level_regret",
        "no_trade_missed_opportunity",
        "no_trade_opportunity_cost_mean",
        "ranking_confusion_matrix",
        "action_pair_regret_summary",
        "hold_to_settlement_up_down_reversal_regret",
        "hts_p_up_reliability_regret_summary",
    }
    assert (
        ranking["train_shadow_metrics"]["decision_group_count"]
        + ranking["validation_metrics"]["decision_group_count"]
        == ranking["all_metrics"]["decision_group_count"]
    )
    assert ranking["decision_group_completeness_summary"][
        "partial_decision_group_count"
    ] == 0
    assert 0.0 <= ranking["top1_realized_best_action_hit_rate"] <= 1.0
    assert ranking["mean_regret"] >= 0.0
    assert any(
        row["oracle_executable_best_action"] == "NO_TRADE"
        for row in ranking["ranking_rows"]
    )
    assert all(
        row["ranking_score_source"] == "model_predicted_score"
        for row in ranking["ranking_rows"]
    )
    assert all(
        row["deployable_model_score_available"] is True
        for row in ranking["ranking_rows"]
    )
    assert all(
        row["o_model_predicted_score"] is not None
        for row in ranking["ranking_rows"]
    )
    assert isinstance(
        correction_config["high_score_calibration"]["high_score_threshold"],
        float,
    )
    assert correction_config["high_score_calibration"][
        "high_score_threshold_profile_source"
    ] == "shadow_split_only"
    assert correction_config["high_score_calibration"][
        "uses_validation_labels_for_threshold_tuning"
    ] is False
    assert correction_config["high_score_calibration"][
        "selected_high_score_threshold_profile_name"
    ] == ranking["selected_high_score_threshold_profile_name"]
    assert correction_config["high_score_calibration"][
        "high_score_threshold_source"
    ].endswith("_shadow_split_only")
    assert looks_like_sha256(
        correction_config["high_score_calibration"][
            "high_score_threshold_profile_config_hash"
        ]
    )
    assert {
        row["profile_name"]
        for row in correction_config["high_score_calibration"][
            "high_score_threshold_profile_candidates"
        ]
    } == {
        "current_threshold",
        "high_score_profitability_threshold",
        "slightly_lower_shadow_derived_threshold",
        "support_preserving_threshold",
    }
    assert "large_regret_reversal_penalty_adjusted" in correction_config[
        "high_score_calibration"
    ]
    assert (
        ranking["high_score_threshold"]
        == correction_config["high_score_calibration"]["high_score_threshold"]
    )
    assert correction_config["high_score_calibration"][
        "high_score_requires_corrected_model_score_gte_threshold"
    ] is True
    assert all(
        set(row["o_model_score_components"]) >= {
            "base_score",
            "p_up_side_alignment_component",
            "confidence_or_weak_opportunity_component",
            "group_normalized_raw_model_component",
            "p_up_misalignment_penalty_component",
            "large_regret_reversal_guard_component",
            "hts_p_up_reliability_guard_component",
            "hts_p_up_reliability_no_trade_buffer_component",
            "shadow_action_family_prior_component",
            "microstructure_quality_component",
        }
        for row in ranking["ranking_rows"]
    )
    assert ranking["source_model_candidate_eligible"] is False
    assert ranking["#146_start_allowed"] is False
    assert ranking["#134_resume_allowed"] is False

    leakage = result.leakage_audit_report
    leakage_payload = dict(leakage)
    leakage_id = leakage_payload.pop("o_feature_and_label_leakage_audit_report_id")
    assert canonical_json_sha256(leakage_payload) == leakage_id
    assert leakage["schema_version"] == O_FEATURE_AND_LABEL_LEAKAGE_AUDIT_SCHEMA_VERSION
    assert leakage["ranking_score_source"] == "model_predicted_score"
    assert leakage["deployable_model_score_available"] is True
    assert leakage["leakage_audit_passed"] is True
    assert leakage["model_input_forbidden_field_overlap"] == []
    assert leakage["expanded_decision_time_feature_provenance_passed"] is True
    assert leakage["expanded_feature_coverage"][
        "feature_provenance_violation_count"
    ] == 0
    assert "reference_price_to_beat_distance_at_decision" in leakage[
        "expanded_decision_time_feature_fields"
    ]
    assert "total_polymarket_pnl" not in leakage["model_input_fields_decision_time_only"]
    assert "action_return_target" not in leakage["model_input_fields_decision_time_only"]
    assert "label_pnl_target" not in leakage["model_input_fields_decision_time_only"]
    assert "realized_trade_pnl" not in leakage["model_input_fields_decision_time_only"]
    assert leakage["model_input_fields_decision_time_only"] == selected_feature_names
    assert leakage["all_candidate_model_input_fields_decision_time_only"] == list(
        O_DEPLOYABLE_MODEL_FEATURE_NAMES
    )
    assert leakage["selected_feature_set_name"] == ranking["selected_feature_set_name"]
    assert leakage["future_replay_outcomes_used_as_model_inputs"] is False
    assert leakage["future_replay_outcomes_used_as_training_labels"] is True
    assert leakage["source_model_candidate_eligible"] is False
    assert leakage["#146_start_allowed"] is False
    assert leakage["#134_resume_allowed"] is False

    comparison = result.candidate_comparison_report
    comparison_payload = dict(comparison)
    comparison_id = comparison_payload.pop("o_source_candidate_comparison_report_id")
    assert canonical_json_sha256(comparison_payload) == comparison_id
    assert (
        comparison["schema_version"]
        == O_SOURCE_CANDIDATE_COMPARISON_SCHEMA_VERSION
    )
    assert len(comparison["candidate_rows"]) >= 5
    assert comparison["eligible_candidate_count"] == 0
    assert comparison["model_predicted_candidate_name"] == O_MODEL_PREDICTED_VARIANT
    assert comparison["model_training_summary"]["model_candidate_name"] == (
        O_MODEL_PREDICTED_VARIANT
    )
    assert comparison["label_diagnostic_variants"] == list(O_LABEL_DIAGNOSTIC_VARIANTS)
    assert all(
        row["source_model_candidate_eligible"] is False
        for row in comparison["candidate_rows"]
    )
    assert all(
        row["ranking_metric_scope"] == "full_decision_group"
        for row in comparison["candidate_rows"]
    )
    comparison_by_name = {
        row["candidate_name"]: row for row in comparison["candidate_rows"]
    }
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "ranking_score_source"
    ] == "model_predicted_score"
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "deployable_model_score_available"
    ] is True
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "full_source_model_ranking_quality_claimed"
    ] is True
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "model_training_summary"
    ]["feature_names"] == selected_feature_names
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "model_training_summary"
    ]["all_candidate_feature_names"] == list(O_DEPLOYABLE_MODEL_FEATURE_NAMES)
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "correction_constants_source"
    ] == "shadow_split_only"
    assert (
        comparison_by_name[O_MODEL_PREDICTED_VARIANT]["correction_config_hash"]
        == ranking["o_model_training_summary"]["correction_config_hash"]
    )
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "probe_constants_source"
    ] == "shadow_split_only"
    assert (
        comparison_by_name[O_MODEL_PREDICTED_VARIANT]["probe_config_hash"]
        == ranking["o_model_training_summary"]["probe_config_hash"]
    )
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "eligible_for_source_model_gate"
    ] is True
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "excluded_from_eligibility_reason"
    ] is None
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "eligibility_metric_source"
    ] == "validation_metrics_only"
    assert comparison_by_name["current_source_baseline"][
        "full_source_model_ranking_quality_claimed"
    ] is False
    assert comparison_by_name["current_source_baseline"][
        "ranking_score_source"
    ] == "observed_source_score"
    assert comparison_by_name["current_source_baseline"][
        "deployable_model_score_available"
    ] is False
    assert comparison_by_name["current_source_baseline"][
        "source_score_completeness_summary"
    ]["source_score_complete"] is False
    assert all(
        row["ranking_score_source"] == "label_diagnostic_score"
        and row["deployable_model_score_available"] is False
        and row["label_diagnostic_score"] is True
        and row["eligible_for_source_model_gate"] is False
        and row["excluded_from_eligibility_reason"]
        == "label_diagnostic_score_not_model_predicted"
        and row["full_source_model_ranking_quality_claimed"] is False
        for row in comparison["candidate_rows"]
        if row["candidate_name"] in O_LABEL_DIAGNOSTIC_VARIANTS
    )
    assert all(
        row["action_family_gate_metrics"]["HOLD_TO_SETTLEMENT"][
            "paper_decision_eligible"
        ]
        is False
        for row in comparison["candidate_rows"]
    )
    assert comparison["#146_start_allowed"] is False
    assert comparison["#134_resume_allowed"] is False
    assert comparison["paper_only"] is True
    assert comparison["capital_at_risk"] is False

    gate = result.source_model_eligibility_gate_report
    gate_payload = dict(gate)
    gate_id = gate_payload.pop("o_source_model_eligibility_gate_report_id")
    assert canonical_json_sha256(gate_payload) == gate_id
    assert gate["schema_version"] == O_SOURCE_MODEL_ELIGIBILITY_GATE_SCHEMA_VERSION
    assert gate["candidate_name"] == O_MODEL_PREDICTED_VARIANT
    assert gate["ranking_score_source"] == "model_predicted_score"
    assert gate["deployable_model_score_available"] is True
    assert gate["correction_constants_source"] == "shadow_split_only"
    assert gate["probe_constants_source"] == "shadow_split_only"
    assert (
        gate["correction_config_hash"]
        == ranking["o_model_training_summary"]["correction_config_hash"]
    )
    assert (
        gate["probe_config_hash"]
        == ranking["o_model_training_summary"]["probe_config_hash"]
    )
    assert gate["correction_config_hash_verified"] is True
    assert gate["probe_config_hash_verified"] is True
    assert gate["high_score_threshold"] == ranking["high_score_threshold"]
    assert gate["high_score_threshold_source"] == correction_config[
        "high_score_calibration"
    ]["high_score_threshold_source"]
    assert gate["high_score_threshold_source"].endswith("_shadow_split_only")
    assert gate["eligible_for_source_model_gate"] is True
    assert gate["validation_metrics_only_for_eligibility"] is True
    assert gate["validation_metrics"] == ranking["validation_metrics"]
    assert gate["train_shadow_metrics"] == ranking["train_shadow_metrics"]
    assert gate["all_metrics"] == ranking["all_metrics"]
    assert gate["high_score_support_count"] == gate["validation_metrics"][
        "high_score_support_count"
    ]
    assert gate["all_metrics"]["high_score_support_count"] >= gate[
        "high_score_support_count"
    ]
    assert gate["mean_regret"] == gate["validation_metrics"]["mean_regret"]
    assert gate["NO_TRADE_selection_rate"] == gate["validation_metrics"][
        "NO_TRADE_selection_rate"
    ]
    assert gate["source_model_candidate_eligible"] is False
    assert gate["promotion_evidence_eligible"] is False
    assert gate["future_unseen_holdout_required"] is True
    assert gate["promotion_blocking_reason_codes"] == [
        "future_unseen_holdout_required"
    ]
    assert gate["p_up_action_disagreement_summary"][
        "candidate_scoped_p_up_action_disagreement_rate"
    ] <= gate["p_up_action_disagreement_summary"][
        "max_allowed_disagreement_rate"
    ]
    assert gate["gate_thresholds"]["p_up_safety_target_disagreement_rate"] == 0.25
    assert gate["gate_thresholds"]["p_up_safety_target_is_hard_gate"] is False
    assert "p_up_safety_target_met" in gate
    assert "largest_regret_case" in gate["validation_metrics"]
    assert "action_family_level_regret" in gate["validation_metrics"]
    assert "action_pair_regret_summary" in gate["validation_metrics"]
    assert "hold_to_settlement_up_down_reversal_regret" in gate[
        "validation_metrics"
    ]
    assert "no_trade_missed_opportunity" in gate["validation_metrics"]
    assert 0.0 <= gate["validation_metrics"]["NO_TRADE_selection_rate"] <= 1.0
    assert "future_unseen_holdout_required" in gate["ineligible_reason_codes"]
    assert gate["#146_start_allowed"] is False
    assert gate["#134_resume_allowed"] is False
    assert gate["paper_only"] is True
    assert gate["capital_at_risk"] is False

    freeze = result.freeze_readiness_report
    freeze_payload = dict(freeze)
    freeze_id = freeze_payload.pop("o_freeze_readiness_report_id")
    assert canonical_json_sha256(freeze_payload) == freeze_id
    assert freeze["schema_version"] == O_FREEZE_READINESS_SCHEMA_VERSION
    assert freeze["candidate_name"] == O_MODEL_PREDICTED_VARIANT
    assert freeze["ranking_score_source"] == "model_predicted_score"
    assert freeze["correction_constants_source"] == "shadow_split_only"
    assert freeze["probe_constants_source"] == "shadow_split_only"
    assert (
        freeze["correction_config_hash"]
        == ranking["o_model_training_summary"]["correction_config_hash"]
    )
    assert (
        freeze["probe_config_hash"]
        == ranking["o_model_training_summary"]["probe_config_hash"]
    )
    assert freeze["freeze_ready"] is False
    assert freeze["source_model_candidate_eligible"] is False
    assert "source_model_validation_gates_not_passed" in freeze[
        "freeze_blocking_reason_codes"
    ]
    for field in (
        "model_sha256",
        "model_manifest_sha256",
        "training_data_hash",
        "label_grid_hash",
        "feature_schema_hash",
        "split_hash",
        "candidate_config_hash",
    ):
        assert looks_like_sha256(freeze[field])
    assert freeze["future_unseen_holdout_required"] is True
    assert freeze["promotion_evidence_eligible"] is False
    assert freeze["#146_start_allowed"] is False
    assert freeze["#134_resume_allowed"] is False

    hts_diagnostic = (
        result.hts_p_up_confidently_wrong_feature_diagnostic_report
    )
    hts_payload = dict(hts_diagnostic)
    hts_id = hts_payload.pop(
        "o_hts_p_up_confidently_wrong_feature_diagnostic_report_id"
    )
    assert canonical_json_sha256(hts_payload) == hts_id
    assert hts_diagnostic["schema_version"] == (
        O_HTS_P_UP_CONFIDENTLY_WRONG_FEATURE_DIAGNOSTIC_SCHEMA_VERSION
    )
    assert hts_diagnostic["candidate_name"] == O_MODEL_PREDICTED_VARIANT
    assert hts_diagnostic["ranking_score_source"] == "model_predicted_score"
    assert hts_diagnostic["diagnostic_only"] is True
    assert hts_diagnostic["uses_validation_labels_for_tuning"] is False
    assert (
        hts_diagnostic["report_uses_replay_outcomes_for_evaluation_only"]
        is True
    )
    assert hts_diagnostic["selection_or_gate_mutation"] is False
    assert hts_diagnostic["eligibility_remains_validation_only"] is True
    assert hts_diagnostic["model_input_fields_decision_time_only"] == list(
        O_DEPLOYABLE_MODEL_FEATURE_NAMES
    )
    assert hts_diagnostic["forbidden_fields_used_for_selection"] == []
    assert hts_diagnostic["case_count"] >= 0
    assert hts_diagnostic["validation_case_count"] >= 0
    assert set(hts_diagnostic["split_summaries"]) == {
        "all",
        "shadow",
        "validation",
    }
    assert "complete_decision_time_feature_case_count" in hts_diagnostic[
        "feature_coverage_summary"
    ]
    assert "existing_features_insufficient" in hts_diagnostic[
        "feature_coverage_summary"
    ]
    assert "reference_price_to_beat_distance_at_decision" in {
        row["feature"]
        for row in hts_diagnostic["missing_or_weak_decision_time_feature_candidates"]
    }
    assert "opposite_hts_side" in hts_diagnostic[
        "alternative_comparison_summary"
    ]
    assert "best_sell_before_close_by_return" in hts_diagnostic[
        "alternative_comparison_summary"
    ]
    if hts_diagnostic["top_confidently_wrong_cases"]:
        top_case = hts_diagnostic["top_confidently_wrong_cases"][0]
        assert top_case["p_up_confidently_wrong"] is True
        assert top_case["selected_action"].endswith("HOLD_TO_SETTLEMENT")
        assert top_case["selected_side"] != top_case["oracle_side"]
        assert "p_up_confidence_bucket" in top_case
        assert "time_to_close_seconds" in top_case["feature_snapshot"]
        assert "spread_bps" in top_case["feature_snapshot"]
        assert "queue_fill" in top_case["feature_snapshot"]
        assert "book_staleness_ms" in top_case["feature_snapshot"]
        assert "entry_ask" in top_case["feature_snapshot"]
        assert "exit_bid_proxy" in top_case["feature_snapshot"]
        assert "reference_price_to_beat_distance_at_decision" in top_case[
            "feature_snapshot"
        ]
        assert "side_book_depth_imbalance" in top_case["feature_snapshot"]
        assert "side_book_update_velocity" in top_case["feature_snapshot"]
        assert "hts_vs_sell_before_close_exit_value_gap_proxy" in top_case[
            "feature_snapshot"
        ]
        assert "p_up_calibration_residual_by_time_spread_queue_bucket" in top_case[
            "feature_snapshot"
        ]
        assert "group_normalized_raw_model_component" in top_case[
            "score_components"
        ]
        assert "NO_TRADE" in top_case["alternative_actions"]
        assert "opposite_hts_side" in top_case["alternative_actions"]
    assert hts_diagnostic["recommended_next_action"] in {
        "add_new_decision_time_reference_and_book_pressure_features_before_"
        "further_hts_priority_changes",
        "collect_reference_price_to_beat_distance_before_further_hts_priority_"
        "changes",
        "lower_hts_side_bet_priority_when_reliability_is_weak",
        "continue_monitoring",
    }
    assert hts_diagnostic["gate_status_snapshot"][
        "source_model_candidate_eligible"
    ] is False
    assert hts_diagnostic["source_model_candidate_eligible"] is False
    assert hts_diagnostic["promotion_evidence_eligible"] is False
    assert hts_diagnostic["#146_start_allowed"] is False
    assert hts_diagnostic["#134_resume_allowed"] is False
    assert hts_diagnostic["paper_only"] is True
    assert hts_diagnostic["capital_at_risk"] is False

    assert result.artifact_paths["label_construction_report"].exists()
    assert result.artifact_paths["ranking_objective_report"].exists()
    assert result.artifact_paths["leakage_audit_report"].exists()
    assert result.artifact_paths["candidate_comparison_report"].exists()
    assert result.artifact_paths["source_model_eligibility_gate_report"].exists()
    assert result.artifact_paths["freeze_readiness_report"].exists()
    assert result.artifact_paths[
        "hts_p_up_confidently_wrong_feature_diagnostic_report"
    ].exists()
    assert result.artifact_paths[
        "hts_p_up_confidently_wrong_feature_diagnostic_summary"
    ].exists()

    manifest = _read_json(result.artifact_paths["manifest"])
    assert "hts_p_up_confidently_wrong_feature_diagnostic_report" in manifest[
        "artifact_hashes"
    ]
    assert "hts_p_up_confidently_wrong_feature_diagnostic_summary" in manifest[
        "artifact_hashes"
    ]


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


def _o_label_row(
    *,
    market_id: str,
    decision_ts: int,
    action: str,
    target: float,
) -> dict[str, Any]:
    sell_before_close = action.endswith("SELL_BEFORE_CLOSE")
    hold_to_settlement = action.endswith("HOLD_TO_SETTLEMENT")
    no_trade = action == "NO_TRADE"
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "action": action,
        "outcome": _side_from_o_action(action),
        "entry_ask": 0.50 if not no_trade else 0.0,
        "exit_bid": 0.55 if sell_before_close else 0.0,
        "realized_trade_return": target if sell_before_close else 0.0,
        "settlement_return": target if hold_to_settlement else 0.0,
        "total_net_return": target,
        "total_net_pnl_per_notional": target,
        "sell_before_close_execution_class": (
            "realizable_sell_before_close" if sell_before_close else "not_applicable"
        ),
        "label_uses_executable_exit_path": sell_before_close,
        "queue_fill_probability_estimate": 0.95 if sell_before_close else 0.0,
        "executable_liquidity_notional": 10.0 if sell_before_close else 0.0,
        "theoretical_terminal_bid_return": target if sell_before_close else 0.0,
        "realized_executable_sell_before_close_return": (
            target if sell_before_close else 0.0
        ),
        "execution_gap_return": 0.0,
    }


def _o_feature_row(
    *,
    market_id: str,
    decision_ts: int,
    btc_mid_price: float,
    up_depth: float,
    down_depth: float,
    up_update_count: int,
    down_update_count: int,
) -> dict[str, Any]:
    features = {
        "btc_mid_price": btc_mid_price,
        "btc_return_30s": 0.002,
        "btc_return_1m": 0.003,
        "time_to_close_seconds": 180.0,
        "up_ask": 0.41,
        "up_bid": 0.40,
        "down_ask": 0.61,
        "down_bid": 0.60,
        "up_liquidity_depth": up_depth,
        "down_liquidity_depth": down_depth,
        "up_recent_book_update_count_1m": up_update_count,
        "down_recent_book_update_count_1m": down_update_count,
        "up_book_staleness_ms": 250.0,
        "down_book_staleness_ms": 300.0,
        "up_spread_bps": 250.0,
        "down_spread_bps": 300.0,
        "combined_spread_bps": 275.0,
        "up_queue_fill_probability_proxy": 0.91,
        "down_queue_fill_probability_proxy": 0.88,
    }
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "available_at_ts": decision_ts,
        "feature_cutoff_ts": decision_ts,
        "max_input_ts": decision_ts,
        "features": features,
        "feature_provenance": {
            key: {
                "available_at_ts": decision_ts,
                "input_end_ts": decision_ts,
                "input_start_ts": decision_ts - 60_000,
                "lookback_ms": 60_000,
                "source": "test_polymarket_corpus",
            }
            for key in features
        },
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _side_from_o_action(action: str) -> str:
    if "_UP_" in action:
        return "UP"
    if "_DOWN_" in action:
        return "DOWN"
    return "NONE"
