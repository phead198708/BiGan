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
from bigan.v8.polymarket.recorder.public_provider import RealCorpusPublicProviderError
from bigan.v8.polymarket.training.o_v8_paper_candidate_unlock import (
    O_V8_PAPER_CANDIDATE_UNLOCK_MANIFEST_SCHEMA_VERSION,
    O_V8_PAPER_CANDIDATE_UNLOCK_SCHEMA_VERSION,
    O_V8_PAPER_FILL_SIMULATION_SCHEMA_VERSION,
    O_V8_PAPER_INTERNAL_EXECUTION_LOOP_SCHEMA_VERSION,
    O_V8_PAPER_RUNTIME_SAFETY_SCHEMA_VERSION,
    PINNED_ISSUE_159_ARTIFACT_FILENAMES,
    PolymarketOV8PaperCandidateUnlockConfig,
    run_polymarket_o_v8_paper_candidate_unlock,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    EXECUTION_LAYER_V2_PAPER_REMAP_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_CANONICAL_FEATURE_MAPPING_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_CANONICAL_SCORER_ALIGNMENT_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_CANONICAL_SCORER_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_CUMULATIVE_MONITORING_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_EXIT_LEDGER_UPDATE_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_EXIT_SIGNAL_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_FILL_SIMULATION_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_LEGACY_POSITION_POLICY_AUDIT_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_LOOP_MANIFEST_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_LOOP_RUN_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_MONITORING_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_NO_TRADE_DIAGNOSTIC_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_POSITION_STATE_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_PROVIDER_FEATURE_COVERAGE_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_RUNTIME_SAFETY_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_SCORE_DECOMPOSITION_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_SCORER_COMPARISON_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_SIGNAL_TRACE_SCHEMA_VERSION,
    O_V8_PAPER_FRESH_TIME_WINDOW_DIAGNOSTIC_SCHEMA_VERSION,
    O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
    O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE,
    PolymarketOV8PaperFreshLoopConfig,
    run_polymarket_o_v8_paper_fresh_loop,
)
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
    O_MAX_MEAN_REGRET,
    O_MAX_P_UP_ACTION_DISAGREEMENT_RATE,
    O_MIN_HIGH_SCORE_SUPPORT_COUNT,
    O_MIN_TOP1_HIT_RATE,
    O_MODEL_PREDICTED_VARIANT,
    O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET,
    O_REPORT_ONLY_EVALUATION_FIELDS,
    O_REQUIRED_DECISION_ACTION_FAMILIES,
    O_SHADOW_P_UP_SELECTION_BUFFER_TARGET,
    O_SOURCE_CANDIDATE_COMPARISON_SCHEMA_VERSION,
    O_SOURCE_MODEL_ELIGIBILITY_GATE_SCHEMA_VERSION,
    O_SOURCE_RANKING_OBJECTIVE_SCHEMA_VERSION,
    O_V8_ACTION_RANK_HANDOFF_SCHEMA_VERSION,
    O_V8_EXECUTION_ALLOWED_ORDER_QUALITY_SCHEMA_VERSION,
    O_V8_EXECUTION_GUARD_BLOCK_ANALYSIS_SCHEMA_VERSION,
    O_V8_EXECUTION_HANDOFF_GATE_SCHEMA_VERSION,
    O_V8_EXECUTION_POLICY_READINESS_SCHEMA_VERSION,
    O_V8_EXECUTION_RISK_GUARD_SCHEMA_VERSION,
    O_V8_EXECUTION_RUNTIME_FIELD_COVERAGE_SCHEMA_VERSION,
    O_V8_EXECUTION_RUNTIME_STATE_SCHEMA_VERSION,
    O_V8_EXECUTION_SIMULATED_ORDER_REPLAY_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_ACTION_RANK_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_COLLECTION_PLAN_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_EXECUTION_REPLAY_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_HANDOFF_GATE_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_INPUT_FREEZE_MANIFEST_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_PAPER_CANDIDATE_GATE_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_PLAN_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_POLICY_READINESS_SCHEMA_VERSION,
    O_V8_FUTURE_UNSEEN_HOLDOUT_RAW_COLLECTION_MANIFEST_SCHEMA_VERSION,
    O_V8_PAPER_CANDIDATE_GATE_DESIGN_SCHEMA_VERSION,
    PolymarketOReplayAlignedSourceRankingConfig,
    _o_relaxed_diagnostic_gate_status,
    _v8_execution_allowed_order_quality_report,
    _v8_execution_guard_block_analysis_report,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_execution_handoff_gate_report,
    _v8_execution_policy_readiness_report,
    _v8_execution_runtime_field_coverage_report,
    _v8_execution_simulated_runtime_reports,
    _v8_future_unseen_holdout_action_rank_report,
    _v8_future_unseen_holdout_collection_plan_report,
    _v8_future_unseen_holdout_execution_replay_report,
    _v8_future_unseen_holdout_handoff_gate_report,
    _v8_future_unseen_holdout_input_freeze_manifest,
    _v8_future_unseen_holdout_paper_candidate_gate_report,
    _v8_future_unseen_holdout_plan_report,
    _v8_future_unseen_holdout_policy_readiness_report,
    _v8_future_unseen_holdout_raw_collection_manifest,
    _v8_initial_runtime_state,
    _v8_paper_candidate_gate_design_report,
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
from examples.v8.generate_o_v8_future_unseen_holdout_raw_input import (
    FORBIDDEN_HOLDOUT_ROW_FIELDS,
    _filter_runtime_quality_rows,
    _prepare_runtime_quality_action_entry,
    _runtime_input_quality_rule_counts,
    _select_diversified_quality_rows,
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


def test_o_relaxed_diagnostic_gate_passes_without_strict_source_unlock() -> None:
    assert O_MAX_MEAN_REGRET == 0.15
    assert O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET == 0.25

    status = _o_relaxed_diagnostic_gate_status(
        validation_metrics={
            "mean_regret": 0.22495588235294117,
            "top1_realized_best_action_hit_rate": 0.38235294117647056,
        },
        p_up_action_disagreement_within_limit=True,
        calibration_support_passed=True,
        action_family_paper_decision_eligible=True,
        best_action_concentration_passed=True,
        high_score_return_positive=True,
        leakage_passed=True,
    )

    summary = status["strict_vs_relaxed_gate_summary"]
    assert summary["strict_max_mean_regret"] == O_MAX_MEAN_REGRET
    assert summary["relaxed_diagnostic_max_mean_regret"] == (
        O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
    )
    assert summary["strict_mean_regret_passed"] is False
    assert summary["relaxed_diagnostic_mean_regret_passed"] is True
    assert status["strict_calibration_quality_passed"] is False
    assert status["relaxed_diagnostic_calibration_quality_passed"] is True
    assert status["relaxed_diagnostic_source_candidate"] is True
    assert "strict_source_gate_remains_authoritative" in status[
        "relaxed_diagnostic_reason_codes"
    ]
    assert "diagnostic_only_no_paper_live_unlock" in status[
        "relaxed_diagnostic_reason_codes"
    ]


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
    assert correction_config["hts_p_up_reliability_no_trade_buffer"] >= 0.0
    assert correction_config["hts_p_up_reliability_no_trade_buffer"] == (
        correction_config["p_up_edge_quantiles"]["q25"]
        * correction_config["hts_p_up_reliability_no_trade_buffer_multiplier"]
    )
    assert correction_config[
        "hts_p_up_reliability_no_trade_buffer_multiplier_source"
    ] == "shadow_split_only_config_hashed_correction_policy_profile"
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
    assert feature_selection["shadow_p_up_safety_constrained_selection_enabled"] is True
    assert (
        feature_selection["shadow_p_up_safety_target_rate"]
        == O_SHADOW_P_UP_SELECTION_BUFFER_TARGET
    )
    assert feature_selection["shadow_top1_aware_selection_enabled"] is True
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
    assert joint_selection["shadow_p_up_safety_constrained_selection_enabled"] is True
    assert (
        joint_selection["shadow_p_up_safety_target_rate"]
        == O_SHADOW_P_UP_SELECTION_BUFFER_TARGET
    )
    assert joint_selection["shadow_top1_aware_selection_enabled"] is True
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
        "hts_sbc_regret_balancing",
        "high_score_profitability_preserving",
        "largest_regret_dampening",
        "no_trade_missed_opportunity_recovery",
        "no_trade_tail_risk_buffer",
        "p_up_safe_regret_reduction",
        "sbc_preferred_when_hts_reliability_weak",
        "top1_miss_regret_minimizing",
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
    assert joint_selection["candidate_count"] == 200
    assert looks_like_sha256(
        joint_selection["regret_reduction_selection_config_hash"]
    )
    regret_config = joint_selection["regret_reduction_selection_config"]
    assert regret_config["uses_validation_labels_for_tuning"] is False
    assert regret_config["selection_metric_source"] == "shadow_split_only"
    assert regret_config["p_up_safety_target_rate"] == (
        O_SHADOW_P_UP_SELECTION_BUFFER_TARGET
    )
    assert regret_config["min_top1_hit_rate"] == O_MIN_TOP1_HIT_RATE
    assert regret_config["min_high_score_support_count"] == (
        O_MIN_HIGH_SCORE_SUPPORT_COUNT
    )
    assert set(regret_config["selection_terms"]) == {
        "shadow_largest_regret_value",
        "shadow_mean_regret",
        "shadow_no_trade_missed_positive_opportunity_sum",
        "shadow_positive_regret_sum",
        "shadow_top1_miss_regret_sum",
    }
    assert joint_selection["selected_high_score_threshold_profile"][
        "uses_validation_labels_for_tuning"
    ] is False
    assert joint_selection["selected_full_correction_rerun_diagnostics"][
        "full_correction_rerun_enabled"
    ] is True
    assert joint_selection["selected_full_correction_rerun_diagnostics"][
        "full_correction_search_source"
    ] == "shadow_split_only"
    assert joint_selection["selected_full_correction_rerun_diagnostics"][
        "shadow_p_up_safety_target_rate"
    ] == O_SHADOW_P_UP_SELECTION_BUFFER_TARGET
    assert (
        "shadow_top1_quality_acceptance_path"
        in joint_selection["selected_full_correction_rerun_diagnostics"]
    )
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
    assert all(
        "shadow_p_up_safety_target_passed" in row
        and "shadow_top1_quality_target_passed" in row
        and "shadow_top1_miss_regret_sum" in row
        and "shadow_positive_regret_sum" in row
        and "shadow_no_trade_missed_positive_opportunity_sum" in row
        for row in joint_selection["candidate_rows"]
    )
    assert set(joint_selection["mean_regret_reduction_diagnostics"]) == {
        "shadow",
        "validation_report_only",
    }
    assert joint_selection["mean_regret_reduction_diagnostics"]["shadow"] == (
        ranking["train_shadow_metrics"]["mean_regret_reduction_diagnostics"]
    )
    assert joint_selection["mean_regret_reduction_diagnostics"][
        "validation_report_only"
    ] == ranking["validation_metrics"]["mean_regret_reduction_diagnostics"]
    assert joint_selection["largest_regret_case_diagnostics"][
        "validation_report_only"
    ] == ranking["validation_metrics"]["largest_regret_case"]
    assert joint_selection["top1_miss_regret_diagnostics"][
        "validation_report_only"
    ] == ranking["validation_metrics"]["top1_miss_diagnostics"]
    assert joint_selection["action_pair_regret_reduction_diagnostics"][
        "validation_report_only"
    ] == ranking["validation_metrics"]["action_pair_regret_summary"]
    assert joint_selection["no_trade_missed_opportunity_diagnostics"][
        "validation_report_only"
    ] == ranking["validation_metrics"]["no_trade_missed_opportunity"]
    assert joint_selection["gate_preservation_diagnostics"][
        "selection_metric_source"
    ] == "shadow_split_only"
    assert joint_selection["gate_preservation_diagnostics"][
        "validation_metrics_report_only"
    ] is True
    assert joint_selection["gate_preservation_diagnostics"][
        "validation_report_only"
    ]["p_up_disagreement_within_hard_gate"] == (
        joint_selection["selected_validation_metrics_report_only"][
            "p_up_disagreement_rate"
        ]
        <= O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
    )
    tradeoff = joint_selection["mean_regret_gate_tradeoff_diagnostics"]
    assert tradeoff["uses_validation_labels_for_tuning"] is False
    assert tradeoff["selection_metric_source"] == "shadow_split_only"
    assert tradeoff["validation_metrics_report_only"] is True
    assert tradeoff["selected_joint_candidate_name"] == (
        joint_selection["selected_joint_candidate_name"]
    )
    assert tradeoff["selected_shadow_mean_regret"] == (
        joint_selection["selected_shadow_metrics"]["mean_regret"]
    )
    assert tradeoff["selected_validation_mean_regret_report_only"] == (
        joint_selection["selected_validation_metrics_report_only"]["mean_regret"]
    )
    assert tradeoff["tradeoff_conclusion"] in {
        "lower_shadow_mean_regret_candidates_break_shadow_gates",
        "no_gate_preserving_lower_mean_regret_candidate_found",
        "shadow_gate_passing_lower_mean_regret_candidates_exist",
        (
            "validation_report_only_lower_mean_regret_candidates_exist_but_not_"
            "shadow_selected"
        ),
    }
    assert isinstance(tradeoff["lower_shadow_blocker_reason_counts"], dict)
    assert feature_selection["mean_regret_gate_tradeoff_diagnostics"] == tradeoff
    assert feature_selection["regret_reduction_selection_config_hash"] == (
        joint_selection["regret_reduction_selection_config_hash"]
    )
    assert feature_selection["mean_regret_reduction_diagnostics"] == (
        joint_selection["mean_regret_reduction_diagnostics"]
    )
    assert feature_selection["gate_preservation_diagnostics"] == (
        joint_selection["gate_preservation_diagnostics"]
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
        "top1_miss_diagnostics",
        "action_family_level_regret",
        "side_level_regret",
        "no_trade_missed_opportunity",
        "no_trade_opportunity_cost_mean",
        "ranking_confusion_matrix",
        "action_pair_regret_summary",
        "hold_to_settlement_up_down_reversal_regret",
        "hts_p_up_reliability_regret_summary",
        "mean_regret_reduction_diagnostics",
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
    assert ranking["validation_metrics"]["top1_miss_diagnostics"][
        "top1_miss_count"
    ] >= 0
    assert "action_pair_confusion" in ranking["validation_metrics"][
        "top1_miss_diagnostics"
    ]
    assert "regret_contribution_by_miss_type" in ranking["validation_metrics"][
        "top1_miss_diagnostics"
    ]
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
    o_training_summary = ranking["o_model_training_summary"]
    assert o_training_summary["final_scoring_source"] in {
        "full_shadow_correction_search",
        "lightweight_preselection_shadow_ranker",
    }
    assert (
        o_training_summary["final_scoring_source"]
        != "model_predicted_score_with_auxiliary_risk_guard"
    )
    assert "large_regret_risk_model_enabled" not in o_training_summary
    assert "large_regret_risk_model_report" not in o_training_summary
    assert "selective_action_guard_enabled" not in o_training_summary
    assert "selective_action_guard_report" not in o_training_summary
    assert "risk_head_replaces_action_signal" not in o_training_summary
    assert "base_action_value_signal_preserved" not in o_training_summary
    assert all(
        "o_large_regret_risk_score" not in row
        and "o_selective_action_guard_mode" not in row
        and "o_selective_guard_final_selected_action" not in row
        and "execution_guarded_score" not in row
        and "execution_guarded_action" not in row
        for row in ranking["ranking_rows"]
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
    assert comparison["v8_scope"] == (
        "action_rank_signal_and_execution_layer_handoff_only"
    )
    assert comparison["model_layer_regret_risk_selection_enabled"] is False
    assert comparison["model_layer_regret_risk_selection_deferred_to_issue"] == "#158"
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
    assert "strict_calibration_quality_passed" in comparison_by_name[
        O_MODEL_PREDICTED_VARIANT
    ]
    assert "relaxed_diagnostic_calibration_quality_passed" in comparison_by_name[
        O_MODEL_PREDICTED_VARIANT
    ]
    assert "relaxed_diagnostic_source_candidate" in comparison_by_name[
        O_MODEL_PREDICTED_VARIANT
    ]
    assert "relaxed_diagnostic_reason_codes" in comparison_by_name[
        O_MODEL_PREDICTED_VARIANT
    ]
    assert "strict_vs_relaxed_gate_summary" in comparison_by_name[
        O_MODEL_PREDICTED_VARIANT
    ]
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
    assert comparison["relaxed_diagnostic_no_paper_live_unlock"] is True
    assert "strict_vs_relaxed_gate_summary" in comparison
    assert "relaxed_diagnostic_source_candidate" in comparison
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
    assert gate["v8_scope"] == "action_rank_signal_and_execution_layer_handoff_only"
    assert gate["model_layer_regret_risk_selection_enabled"] is False
    assert gate["model_layer_regret_risk_selection_deferred_to_issue"] == "#158"
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
    assert gate["p_up_action_disagreement_summary"][
        "max_allowed_disagreement_rate"
    ] == O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
    assert gate["gate_thresholds"]["p_up_safety_target_disagreement_rate"] == 0.25
    assert gate["gate_thresholds"]["p_up_safety_target_is_hard_gate"] is False
    assert "p_up_safety_target_met" in gate
    assert gate["top1_realized_best_action_hit_rate"] >= 0.0
    assert gate["gate_thresholds"][
        "min_top1_realized_best_action_hit_rate"
    ] == O_MIN_TOP1_HIT_RATE
    assert O_MAX_MEAN_REGRET == 0.15
    assert O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET == 0.25
    assert gate["gate_thresholds"]["max_mean_regret"] == O_MAX_MEAN_REGRET
    assert gate["gate_thresholds"]["relaxed_diagnostic_max_mean_regret"] == (
        O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
    )
    assert gate["gate_thresholds"][
        "min_high_score_support_count"
    ] == O_MIN_HIGH_SCORE_SUPPORT_COUNT
    strict_quality_from_metrics = (
        gate["top1_realized_best_action_hit_rate"] >= O_MIN_TOP1_HIT_RATE
        and gate["mean_regret"] <= O_MAX_MEAN_REGRET
    )
    relaxed_quality_from_metrics = (
        gate["top1_realized_best_action_hit_rate"] >= O_MIN_TOP1_HIT_RATE
        and gate["mean_regret"] <= O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
    )
    assert gate["strict_calibration_quality_passed"] == (
        strict_quality_from_metrics
    )
    assert gate["calibration_quality_passed"] == strict_quality_from_metrics
    assert gate["relaxed_diagnostic_calibration_quality_passed"] == (
        relaxed_quality_from_metrics
    )
    relaxed_summary = gate["strict_vs_relaxed_gate_summary"]
    assert relaxed_summary["strict_source_gate_remains_authoritative"] is True
    assert relaxed_summary["relaxed_diagnostic_gate_is_diagnostic_only"] is True
    assert relaxed_summary["relaxed_diagnostic_no_paper_live_unlock"] is True
    assert relaxed_summary["future_unseen_holdout_required"] is True
    assert relaxed_summary["strict_max_mean_regret"] == O_MAX_MEAN_REGRET
    assert relaxed_summary["relaxed_diagnostic_max_mean_regret"] == (
        O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
    )
    assert relaxed_summary["strict_calibration_quality_passed"] == (
        gate["strict_calibration_quality_passed"]
    )
    assert relaxed_summary[
        "relaxed_diagnostic_calibration_quality_passed"
    ] == gate["relaxed_diagnostic_calibration_quality_passed"]
    assert relaxed_summary["relaxed_diagnostic_source_candidate"] == gate[
        "relaxed_diagnostic_source_candidate"
    ]
    assert "strict_source_gate_remains_authoritative" in gate[
        "relaxed_diagnostic_reason_codes"
    ]
    assert "diagnostic_only_no_paper_live_unlock" in gate[
        "relaxed_diagnostic_reason_codes"
    ]
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "strict_calibration_quality_passed"
    ] == gate["strict_calibration_quality_passed"]
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "relaxed_diagnostic_calibration_quality_passed"
    ] == gate["relaxed_diagnostic_calibration_quality_passed"]
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "relaxed_diagnostic_source_candidate"
    ] == gate["relaxed_diagnostic_source_candidate"]
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "strict_vs_relaxed_gate_summary"
    ] == gate["strict_vs_relaxed_gate_summary"]
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "v8_action_rank_quality_passed"
    ] == gate["v8_action_rank_quality_passed"]
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "v8_action_rank_candidate_eligible"
    ] == gate["v8_action_rank_candidate_eligible"]
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "v8_execution_risk_control_required"
    ] is True
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "v8_execution_handoff_allowed"
    ] is False
    assert comparison_by_name[O_MODEL_PREDICTED_VARIANT][
        "strict_source_gate_remains_failed"
    ] is True
    assert comparison["v8_action_rank_candidate_eligible"] == gate[
        "v8_action_rank_candidate_eligible"
    ]
    assert comparison["v8_execution_handoff_allowed"] is False
    assert comparison["strict_source_gate_remains_failed"] is True
    assert gate["top1_miss_diagnostics"] == gate["validation_metrics"][
        "top1_miss_diagnostics"
    ]
    assert gate["gate_reason_code_consistency_passed"] is True
    assert gate["gate_reason_code_consistency"][
        "gate_reason_code_consistency_passed"
    ] is True
    assert gate["gate_reason_code_consistency"]["unexpected_reason_codes"] == []
    assert gate["gate_reason_code_consistency"]["missing_reason_codes"] == []
    assert gate["v8_full_decision_grid_summary"][
        "required_action_families"
    ] == list(O_REQUIRED_DECISION_ACTION_FAMILIES)
    validation_grid_has_rows = (
        gate["v8_full_decision_grid_summary"]["decision_group_count"] > 0
    )
    if validation_grid_has_rows:
        assert gate["v8_full_decision_grid_summary"][
            "complete_5_action_decision_grid"
        ] is True
        assert gate["v8_action_rank_gate_summary"]["required_checks"][
            "full_5_action_decision_grid_complete"
        ] is True
    else:
        assert gate["v8_full_decision_grid_summary"][
            "complete_5_action_decision_grid"
        ] is False
        assert gate["v8_action_rank_gate_summary"]["required_checks"][
            "full_5_action_decision_grid_complete"
        ] is False
    assert gate["v8_action_rank_gate_summary"]["required_checks"][
        "full_5_action_decision_grid_complete"
    ] == gate["v8_full_decision_grid_summary"]["complete_5_action_decision_grid"]
    assert gate["v8_action_rank_gate_summary"]["strict_max_mean_regret"] == (
        O_MAX_MEAN_REGRET
    )
    assert gate["v8_action_rank_gate_summary"][
        "relaxed_diagnostic_max_mean_regret"
    ] == O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
    assert gate["v8_action_rank_quality_passed"] == all(
        gate["v8_action_rank_gate_summary"]["required_checks"].values()
    )
    assert gate["v8_action_rank_candidate_eligible"] == gate[
        "v8_action_rank_quality_passed"
    ]
    assert gate["v8_execution_risk_control_required"] is True
    assert gate["v8_execution_handoff_allowed"] is False
    assert "execution_layer_runtime_risk_control_not_validated" in gate[
        "v8_execution_handoff_blocking_reason_codes"
    ]
    assert "paper_live_unlock_prohibited" in gate[
        "v8_execution_handoff_blocking_reason_codes"
    ]
    assert gate["strict_source_gate_remains_failed"] == (
        not gate["source_model_candidate_eligible"]
    )
    if (
        gate["high_score_support_count"] >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
        and gate["high_score_realized_return_mean"] > 0.0
        and gate["high_score_realized_return_sum"] > 0.0
    ):
        assert (
            "validation_high_score_return_gate_failed"
            not in gate["ineligible_reason_codes"]
        )
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
    assert freeze["v8_scope"] == "action_rank_signal_and_execution_layer_handoff_only"
    assert freeze["model_layer_regret_risk_selection_enabled"] is False
    assert freeze["model_layer_regret_risk_selection_deferred_to_issue"] == "#158"
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
    assert freeze["strict_calibration_quality_passed"] == gate[
        "strict_calibration_quality_passed"
    ]
    assert freeze["relaxed_diagnostic_calibration_quality_passed"] == gate[
        "relaxed_diagnostic_calibration_quality_passed"
    ]
    assert freeze["relaxed_diagnostic_source_candidate"] == gate[
        "relaxed_diagnostic_source_candidate"
    ]
    assert freeze["relaxed_diagnostic_reason_codes"] == gate[
        "relaxed_diagnostic_reason_codes"
    ]
    assert freeze["strict_vs_relaxed_gate_summary"] == gate[
        "strict_vs_relaxed_gate_summary"
    ]
    assert freeze["relaxed_diagnostic_no_freeze_unlock"] is True
    assert freeze["relaxed_diagnostic_no_paper_live_unlock"] is True
    assert freeze["v8_action_rank_candidate_eligible"] == gate[
        "v8_action_rank_candidate_eligible"
    ]
    assert freeze["v8_action_rank_quality_passed"] == gate[
        "v8_action_rank_quality_passed"
    ]
    assert freeze["v8_execution_risk_control_required"] is True
    assert freeze["v8_execution_handoff_allowed"] is False
    assert freeze["strict_source_gate_remains_failed"] == gate[
        "strict_source_gate_remains_failed"
    ]
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

    handoff = result.v8_action_rank_handoff_report
    handoff_payload = dict(handoff)
    handoff_id = handoff_payload.pop("o_v8_action_rank_handoff_report_id")
    assert canonical_json_sha256(handoff_payload) == handoff_id
    assert handoff["schema_version"] == O_V8_ACTION_RANK_HANDOFF_SCHEMA_VERSION
    assert handoff["candidate_name"] == O_MODEL_PREDICTED_VARIANT
    assert handoff["ranking_score_source"] == "model_predicted_score"
    assert handoff["diagnostic_only"] is True
    assert handoff["v8_scope"] == "action_rank_signal_and_execution_layer_handoff_only"
    assert handoff["model_layer_regret_risk_selection_enabled"] is False
    assert handoff["model_layer_regret_risk_selection_deferred_to_issue"] == "#158"
    assert handoff["strict_calibration_quality_passed"] == gate[
        "strict_calibration_quality_passed"
    ]
    assert handoff["relaxed_diagnostic_source_candidate"] == gate[
        "relaxed_diagnostic_source_candidate"
    ]
    assert handoff["v8_action_rank_quality_passed"] == gate[
        "v8_action_rank_quality_passed"
    ]
    assert handoff["v8_action_rank_candidate_eligible"] == gate[
        "v8_action_rank_candidate_eligible"
    ]
    assert handoff["v8_execution_risk_control_required"] is True
    assert handoff["v8_execution_handoff_allowed"] is False
    assert handoff["strict_source_gate_remains_failed"] == gate[
        "strict_source_gate_remains_failed"
    ]
    assert handoff["source_model_candidate_eligible"] is False
    assert handoff["freeze_ready"] is False
    assert handoff["promotion_evidence_eligible"] is False
    assert handoff["#146_start_allowed"] is False
    assert handoff["#134_resume_allowed"] is False
    assert handoff["paper_only"] is True
    assert handoff["capital_at_risk"] is False
    assert looks_like_sha256(handoff["handoff_contract_hash"])
    assert looks_like_sha256(handoff["model_sha256"])
    assert looks_like_sha256(handoff["feature_schema_hash"])
    assert looks_like_sha256(handoff["split_hash"])
    assert "full_5_action_ranking" in handoff["execution_handoff_contract"][
        "required_fields"
    ]
    assert handoff["selected_action_handoff_row_count"] == gate[
        "validation_metrics"
    ]["decision_group_count"]
    assert len(handoff["selected_action_handoff_rows"]) == handoff[
        "selected_action_handoff_row_count"
    ]
    for row in handoff["selected_action_handoff_rows"]:
        assert set(row) >= {
            "decision_group_id",
            "market_id",
            "decision_ts",
            "selected_action",
            "selected_side",
            "selected_action_family",
            "corrected_model_score",
            "raw_model_score",
            "score_components",
            "high_score_flag",
            "p_up",
            "p_down",
            "p_up_action_disagreement",
            "microstructure_snapshot",
            "reference_price_to_beat_distance_at_decision",
            "reference_price_feature_provenance",
            "full_5_action_ranking",
        }
        assert len(row["full_5_action_ranking"]) == len(
            O_REQUIRED_DECISION_ACTION_FAMILIES
        )
        assert {
            ranked["selected_action"]
            for ranked in row["full_5_action_ranking"]
        } == set(O_REQUIRED_DECISION_ACTION_FAMILIES)
        assert set(row["microstructure_snapshot"]) >= {
            "book_staleness_ms",
            "spread_bps",
            "queue_fill_proxy",
            "time_to_close_seconds",
            "entry_ask",
            "executable_exit_bid_proxy",
        }
        assert set(row["reference_price_feature_provenance"]) >= {
            "source_fields_used",
            "max_input_ts",
            "decision_ts",
            "provenance_valid",
        }
    assert handoff["no_paper_live_unlock_from_v8_action_rank_gate"] is True
    assert handoff["no_source_freeze_unlock_from_v8_action_rank_gate"] is True

    execution_guard = result.v8_execution_risk_guard_report
    execution_guard_payload = dict(execution_guard)
    execution_guard_id = execution_guard_payload.pop(
        "o_v8_execution_risk_guard_report_id"
    )
    assert canonical_json_sha256(execution_guard_payload) == execution_guard_id
    assert (
        execution_guard["schema_version"]
        == O_V8_EXECUTION_RISK_GUARD_SCHEMA_VERSION
    )
    assert execution_guard["candidate_name"] == O_MODEL_PREDICTED_VARIANT
    assert execution_guard["report_type"] == "o_v8_execution_risk_guard"
    assert execution_guard["diagnostic_only"] is True
    assert execution_guard["v8_scope"] == (
        "execution_layer_risk_guarded_action_selection_only"
    )
    assert execution_guard["source_action_rank_signal_available"] is True
    assert execution_guard["source_action_rank_signal_report_id"] == handoff[
        "o_v8_action_rank_handoff_report_id"
    ]
    assert execution_guard["model_layer_regret_risk_selection_enabled"] is False
    assert execution_guard[
        "model_layer_regret_risk_selection_deferred_to_issue"
    ] == "#158"
    assert execution_guard["trains_regret_model"] is False
    assert execution_guard["trains_risk_head"] is False
    assert execution_guard["mutates_o_model_predicted_score"] is False
    assert execution_guard["mutates_source_ranking_scores"] is False
    assert execution_guard["uses_replay_regret_labels_for_guard_tuning"] is False
    assert (
        execution_guard["uses_validation_realized_outcomes_for_guard_tuning"]
        is False
    )
    assert execution_guard["runtime_risk_control_validation_passed"] is False
    assert execution_guard["v8_action_rank_candidate_eligible"] == handoff[
        "v8_action_rank_candidate_eligible"
    ]
    assert execution_guard["v8_execution_risk_control_required"] is True
    assert execution_guard["v8_execution_handoff_allowed"] is False
    assert "execution_layer_runtime_risk_control_not_validated" in execution_guard[
        "v8_execution_handoff_blocking_reason_codes"
    ]
    assert "future_unseen_holdout_required" in execution_guard[
        "v8_execution_handoff_blocking_reason_codes"
    ]
    assert "paper_live_unlock_prohibited" in execution_guard[
        "v8_execution_handoff_blocking_reason_codes"
    ]
    assert execution_guard["source_model_candidate_eligible"] is False
    assert execution_guard["freeze_ready"] is False
    assert execution_guard["promotion_evidence_eligible"] is False
    assert execution_guard["#146_start_allowed"] is False
    assert execution_guard["#134_resume_allowed"] is False
    assert execution_guard["paper_only"] is True
    assert execution_guard["capital_at_risk"] is False
    assert execution_guard["polymarket_write_enabled"] is False
    assert execution_guard["wallet_signing_enabled"] is False
    assert "runtime_exposure_state" in execution_guard["required_runtime_fields"]
    assert "realized_trade_pnl" in execution_guard["forbidden_guard_input_fields"]
    assert looks_like_sha256(execution_guard["execution_guard_config_hash"])
    assert execution_guard["execution_guard_decision_count"] == handoff[
        "selected_action_handoff_row_count"
    ]
    assert len(execution_guard["execution_guard_decision_rows"]) == execution_guard[
        "execution_guard_decision_count"
    ]
    assert execution_guard["order_allowed_count"] == 0
    assert execution_guard["proposed_order_size_total"] == 0.0
    assert execution_guard["no_paper_live_unlock_from_execution_guard"] is True
    assert execution_guard["no_source_freeze_unlock_from_execution_guard"] is True
    handoff_by_group = {
        row["decision_group_id"]: row for row in handoff["selected_action_handoff_rows"]
    }
    trade_guard_rows = [
        row
        for row in execution_guard["execution_guard_decision_rows"]
        if row["source_selected_action"] != "NO_TRADE"
    ]
    for row in execution_guard["execution_guard_decision_rows"]:
        assert set(row) >= {
            "decision_group_id",
            "market_id",
            "decision_ts",
            "source_selected_action",
            "source_selected_side",
            "source_selected_family",
            "source_model_score",
            "source_high_score_flag",
            "top_k_action_ranking",
            "execution_guarded_action",
            "execution_guarded_side",
            "execution_guarded_family",
            "execution_guarded_score",
            "order_allowed",
            "proposed_order_size",
            "execution_guard_reason_codes",
            "execution_blocking_reason_codes",
            "required_runtime_fields_present",
            "missing_runtime_field_codes",
            "fail_closed",
        }
        source_handoff = handoff_by_group[row["decision_group_id"]]
        assert row["source_model_score"] == source_handoff["corrected_model_score"]
        assert row["source_score_mutated"] is False
        assert row["o_model_predicted_score_mutated"] is False
        assert row["order_allowed"] is False
        assert row["proposed_order_size"] == 0.0
        assert len(row["top_k_action_ranking"]) == len(
            O_REQUIRED_DECISION_ACTION_FAMILIES
        )
        if row["source_selected_action"] != "NO_TRADE":
            assert row["fail_closed"] is True
            assert row["required_runtime_fields_present"] is False
            assert "execution_exposure_state_missing" in row[
                "missing_runtime_field_codes"
            ]
            assert "execution_exposure_state_missing" in row[
                "execution_blocking_reason_codes"
            ]
            assert "execution_required_runtime_fields_missing" in row[
                "execution_blocking_reason_codes"
            ]
            assert "execution_blocked_size_zero" in row["sizing_reason_codes"]
    if trade_guard_rows:
        assert execution_guard["execution_guard_summary"][
            "execution_blocking_reason_counts"
        ]["execution_exposure_state_missing"] == len(trade_guard_rows)

    runtime_state = result.v8_execution_runtime_state_report
    runtime_payload = dict(runtime_state)
    runtime_id = runtime_payload.pop("o_v8_execution_runtime_state_report_id")
    assert canonical_json_sha256(runtime_payload) == runtime_id
    assert (
        runtime_state["schema_version"]
        == O_V8_EXECUTION_RUNTIME_STATE_SCHEMA_VERSION
    )
    assert runtime_state["report_type"] == "o_v8_execution_runtime_state"
    assert runtime_state["diagnostic_only"] is True
    assert runtime_state["simulation_only"] is True
    assert runtime_state["risk_state_source"] == "simulated_diagnostic_ledger"
    assert runtime_state["source_action_rank_signal_report_id"] == handoff[
        "o_v8_action_rank_handoff_report_id"
    ]
    assert runtime_state["execution_guard_report_id"] == execution_guard[
        "o_v8_execution_risk_guard_report_id"
    ]
    assert looks_like_sha256(runtime_state["runtime_state_config_hash"])
    assert runtime_state["runtime_state_validation_passed"] is True
    assert runtime_state["runtime_risk_control_validation_passed"] is True
    assert runtime_state["v8_execution_handoff_allowed"] is False
    assert runtime_state["source_model_candidate_eligible"] is False
    assert runtime_state["freeze_ready"] is False
    assert runtime_state["promotion_evidence_eligible"] is False
    assert runtime_state["#146_start_allowed"] is False
    assert runtime_state["#134_resume_allowed"] is False
    assert runtime_state["paper_only"] is True
    assert runtime_state["capital_at_risk"] is False
    assert runtime_state["polymarket_write_enabled"] is False
    assert runtime_state["wallet_signing_enabled"] is False
    assert runtime_state["initial_state"]["current_total_exposure"] == 0.0
    assert runtime_state["final_state"][
        "executed_simulated_order_count"
    ] == len(runtime_state["executed_simulated_orders"])
    assert runtime_state["final_state"][
        "blocked_simulated_order_count"
    ] == len(runtime_state["blocked_simulated_orders"])

    simulated_replay = result.v8_execution_simulated_order_replay_report
    replay_payload = dict(simulated_replay)
    replay_id = replay_payload.pop("o_v8_execution_simulated_order_replay_report_id")
    assert canonical_json_sha256(replay_payload) == replay_id
    assert (
        simulated_replay["schema_version"]
        == O_V8_EXECUTION_SIMULATED_ORDER_REPLAY_SCHEMA_VERSION
    )
    assert simulated_replay["report_type"] == "o_v8_execution_simulated_order_replay"
    assert simulated_replay["diagnostic_only"] is True
    assert simulated_replay["simulation_only"] is True
    assert simulated_replay["replay_source_report_id"] == handoff[
        "o_v8_action_rank_handoff_report_id"
    ]
    assert simulated_replay["execution_guard_report_id"] == execution_guard[
        "o_v8_execution_risk_guard_report_id"
    ]
    assert simulated_replay["runtime_state_report_id"] == runtime_state[
        "o_v8_execution_runtime_state_report_id"
    ]
    assert simulated_replay["decision_count"] == handoff[
        "selected_action_handoff_row_count"
    ]
    assert len(simulated_replay["simulated_decision_rows"]) == simulated_replay[
        "decision_count"
    ]
    assert simulated_replay["runtime_risk_control_validation_passed"] is True
    assert simulated_replay["v8_execution_handoff_allowed"] is False
    assert simulated_replay["source_model_candidate_eligible"] is False
    assert simulated_replay["freeze_ready"] is False
    assert simulated_replay["promotion_evidence_eligible"] is False
    assert simulated_replay["#146_start_allowed"] is False
    assert simulated_replay["#134_resume_allowed"] is False
    assert simulated_replay["paper_only"] is True
    assert simulated_replay["capital_at_risk"] is False
    assert simulated_replay["polymarket_write_enabled"] is False
    assert simulated_replay["wallet_signing_enabled"] is False
    assert looks_like_sha256(simulated_replay["deterministic_replay_hash"])
    assert simulated_replay["total_proposed_notional"] == runtime_state[
        "current_total_exposure"
    ]
    for row in simulated_replay["simulated_decision_rows"]:
        assert "pre_decision_exposure_state" in row
        assert "post_decision_exposure_state" in row
        assert "exposure_delta" in row
        assert "exposure_reason_codes" in row
        assert "execution_exposure_state_missing" not in row[
            "execution_blocking_reason_codes"
        ]
        if row["order_allowed"]:
            assert row["simulated_order_id"]
            assert row["exposure_delta"] == row["proposed_order_size"]
            assert "execution_simulated_order_allowed" in row[
                "exposure_reason_codes"
            ]
        else:
            assert row["simulated_order_id"] is None
            assert row["exposure_delta"] == 0.0

    allowed_quality = result.v8_execution_allowed_order_quality_report
    quality_payload = dict(allowed_quality)
    quality_id = quality_payload.pop(
        "o_v8_execution_allowed_order_quality_report_id"
    )
    assert canonical_json_sha256(quality_payload) == quality_id
    assert (
        allowed_quality["schema_version"]
        == O_V8_EXECUTION_ALLOWED_ORDER_QUALITY_SCHEMA_VERSION
    )
    assert (
        allowed_quality["report_type"]
        == "o_v8_execution_allowed_order_quality"
    )
    assert allowed_quality["diagnostic_only"] is True
    assert allowed_quality["simulation_only"] is True
    assert allowed_quality["uses_validation_outcomes_for_tuning"] is False
    assert allowed_quality["thresholds_tuned"] is False
    assert allowed_quality["mutates_o_model_predicted_score"] is False
    assert allowed_quality["mutates_source_ranking_scores"] is False
    assert allowed_quality["uses_realized_pnl_or_labels_for_analysis"] is False
    assert allowed_quality["forbidden_outcome_fields_used"] == []
    assert allowed_quality["simulated_order_replay_report_id"] == simulated_replay[
        "o_v8_execution_simulated_order_replay_report_id"
    ]
    assert allowed_quality["decision_count"] == simulated_replay["decision_count"]
    assert allowed_quality["allowed_order_count"] == simulated_replay[
        "simulated_allowed_order_count"
    ]
    assert allowed_quality["blocked_decision_count"] == simulated_replay[
        "blocked_decision_count"
    ]
    assert len(allowed_quality["allowed_order_quality_rows"]) == allowed_quality[
        "allowed_order_count"
    ]
    assert len(allowed_quality["residual_blocked_decision_rows"]) == allowed_quality[
        "blocked_decision_count"
    ]
    for row in allowed_quality["allowed_order_quality_rows"]:
        assert set(row) >= {
            "decision_group_id",
            "market_id",
            "decision_ts",
            "simulated_order_id",
            "execution_guarded_action",
            "execution_guarded_family",
            "execution_guarded_side",
            "order_origin",
            "source_model_score",
            "execution_guarded_score",
            "spread_bps",
            "book_staleness_ms",
            "queue_fill_proxy",
            "time_to_close_seconds",
            "pre_decision_exposure",
            "post_decision_exposure",
            "proposed_order_size",
            "sizing_reason_codes",
            "p_up_agreement_status",
        }
        assert row["source_score_mutated"] is False
        assert row["o_model_predicted_score_mutated"] is False
    for row in allowed_quality["residual_blocked_decision_rows"]:
        assert set(row) >= {
            "decision_group_id",
            "market_id",
            "decision_ts",
            "execution_blocking_reason_codes",
            "minimal_blocking_set",
            "deterministic_recommendation_codes",
            "primary_deterministic_recommendation",
            "recommendation_reason_codes",
        }
        assert row["source_score_mutated"] is False
        assert row["o_model_predicted_score_mutated"] is False
    assert set(allowed_quality["residual_blocker_summary"]) >= {
        "exposure_limit_blocked_decision_count",
        "p_up_disagreement_blocked_decision_count",
        "duplicate_market_side_position_count",
        "time_to_close_unsafe_count",
    }
    assert set(allowed_quality["deterministic_recommendation_counts"]).issubset(
        {
            "keep_blocked",
            "needs_exposure_policy_review",
            "needs_p_up_action_rank_review",
            "needs_time_to_close_policy_review",
        }
    )
    assert allowed_quality["v8_execution_handoff_allowed"] is False
    assert allowed_quality["source_model_candidate_eligible"] is False
    assert allowed_quality["freeze_ready"] is False
    assert allowed_quality["promotion_evidence_eligible"] is False
    assert allowed_quality["#146_start_allowed"] is False
    assert allowed_quality["#134_resume_allowed"] is False
    assert allowed_quality["paper_only"] is True
    assert allowed_quality["capital_at_risk"] is False

    policy_readiness = result.v8_execution_policy_readiness_report
    readiness_payload = dict(policy_readiness)
    readiness_id = readiness_payload.pop(
        "o_v8_execution_policy_readiness_report_id"
    )
    assert canonical_json_sha256(readiness_payload) == readiness_id
    assert (
        policy_readiness["schema_version"]
        == O_V8_EXECUTION_POLICY_READINESS_SCHEMA_VERSION
    )
    assert (
        policy_readiness["report_type"]
        == "o_v8_execution_policy_readiness"
    )
    assert policy_readiness["diagnostic_only"] is True
    assert policy_readiness["simulation_only"] is True
    assert policy_readiness["uses_validation_outcomes_for_tuning"] is False
    assert policy_readiness["thresholds_tuned"] is False
    assert policy_readiness["mutates_o_model_predicted_score"] is False
    assert policy_readiness["mutates_source_ranking_scores"] is False
    assert policy_readiness["uses_realized_pnl_or_labels_for_analysis"] is False
    assert policy_readiness["forbidden_outcome_fields_used"] == []
    assert policy_readiness["simulated_order_replay_report_id"] == simulated_replay[
        "o_v8_execution_simulated_order_replay_report_id"
    ]
    assert policy_readiness["allowed_order_quality_report_id"] == allowed_quality[
        "o_v8_execution_allowed_order_quality_report_id"
    ]
    assert policy_readiness["allowed_order_count"] == allowed_quality[
        "allowed_order_count"
    ]
    assert policy_readiness["blocked_decision_count"] == allowed_quality[
        "blocked_decision_count"
    ]
    assert set(policy_readiness["execution_policy_readiness_required_checks"]) == {
        "all_allowed_orders_original_or_safe_downgrade",
        "all_allowed_orders_p_up_agreement",
        "allowed_order_exposure_within_limits",
        "allowed_order_microstructure_quality_passed",
        "min_allowed_order_count",
        "no_paper_live_write_or_capital_flags",
        "zero_missing_runtime_fields",
        "zero_provenance_violations",
    }
    assert all(
        set(check) >= {"passed", "observed", "required", "reason_code"}
        for check in policy_readiness[
            "execution_policy_readiness_required_checks"
        ].values()
    )
    expected_readiness_blockers = sorted(
        check["reason_code"]
        for check in policy_readiness[
            "execution_policy_readiness_required_checks"
        ].values()
        if check["passed"] is not True
    )
    assert policy_readiness[
        "execution_policy_readiness_blocking_reason_codes"
    ] == expected_readiness_blockers
    assert (
        policy_readiness["execution_policy_readiness_diagnostic_passed"]
        == (expected_readiness_blockers == [])
    )
    assert policy_readiness[
        "future_explicit_execution_handoff_gate_required"
    ] is True
    assert policy_readiness["v8_execution_handoff_allowed"] is False
    assert policy_readiness["source_model_candidate_eligible"] is False
    assert policy_readiness["freeze_ready"] is False
    assert policy_readiness["promotion_evidence_eligible"] is False
    assert policy_readiness["#146_start_allowed"] is False
    assert policy_readiness["#134_resume_allowed"] is False
    assert policy_readiness["paper_only"] is True
    assert policy_readiness["capital_at_risk"] is False

    handoff_gate = result.v8_execution_handoff_gate_report
    handoff_gate_payload = dict(handoff_gate)
    handoff_gate_id = handoff_gate_payload.pop(
        "o_v8_execution_handoff_gate_report_id"
    )
    assert canonical_json_sha256(handoff_gate_payload) == handoff_gate_id
    assert handoff_gate["schema_version"] == O_V8_EXECUTION_HANDOFF_GATE_SCHEMA_VERSION
    assert handoff_gate["report_type"] == "o_v8_execution_handoff_gate"
    assert handoff_gate["diagnostic_only"] is True
    assert handoff_gate["simulation_only"] is True
    assert (
        handoff_gate["explicit_execution_handoff_gate_mode"]
        == "diagnostic_only_fail_closed"
    )
    assert handoff_gate["uses_validation_outcomes_for_tuning"] is False
    assert handoff_gate["thresholds_tuned"] is False
    assert handoff_gate["mutates_o_model_predicted_score"] is False
    assert handoff_gate["mutates_source_ranking_scores"] is False
    assert handoff_gate["uses_realized_pnl_or_labels_for_analysis"] is False
    assert handoff_gate["uses_oracle_actions_for_analysis"] is False
    assert handoff_gate["forbidden_outcome_fields_used"] == []
    assert handoff_gate["policy_readiness_report_id"] == policy_readiness[
        "o_v8_execution_policy_readiness_report_id"
    ]
    assert handoff_gate["allowed_order_quality_report_id"] == allowed_quality[
        "o_v8_execution_allowed_order_quality_report_id"
    ]
    assert handoff_gate["simulated_order_replay_report_id"] == simulated_replay[
        "o_v8_execution_simulated_order_replay_report_id"
    ]
    assert set(handoff_gate["explicit_execution_handoff_required_checks"]) == {
        "allowed_order_exposure_within_limits",
        "allowed_order_microstructure_quality_passed",
        "allowed_orders_origin_safe",
        "all_allowed_orders_p_up_agreement",
        "future_unseen_holdout_required",
        "min_allowed_order_count_met",
        "no_model_layer_regret_risk_selection_enabled",
        "no_paper_live_write_or_capital_flags",
        "no_source_score_mutation",
        "policy_readiness_diagnostic_passed",
        "runtime_state_validation_passed",
        "simulated_runtime_risk_control_validation_passed",
        "source_freeze_promotion_remain_blocked",
        "zero_missing_runtime_fields",
        "zero_provenance_violations",
    }
    expected_handoff_blockers = sorted(
        check["reason_code"]
        for check in handoff_gate["explicit_execution_handoff_required_checks"].values()
        if check["passed"] is not True
    )
    assert handoff_gate[
        "explicit_execution_handoff_blocking_reason_codes"
    ] == expected_handoff_blockers
    assert (
        handoff_gate["explicit_execution_handoff_gate_passed"]
        == (expected_handoff_blockers == [])
    )
    assert handoff_gate["future_unseen_holdout_required"] is True
    assert handoff_gate["future_paper_candidate_gate_required"] is True
    assert handoff_gate["v8_execution_handoff_allowed"] is False
    assert handoff_gate["source_model_candidate_eligible"] is False
    assert handoff_gate["freeze_ready"] is False
    assert handoff_gate["promotion_evidence_eligible"] is False
    assert handoff_gate["#146_start_allowed"] is False
    assert handoff_gate["#134_resume_allowed"] is False
    assert handoff_gate["paper_only"] is True
    assert handoff_gate["capital_at_risk"] is False
    assert handoff_gate["polymarket_write_enabled"] is False
    assert handoff_gate["wallet_signing_enabled"] is False

    holdout_plan = result.v8_future_unseen_holdout_plan_report
    holdout_plan_payload = dict(holdout_plan)
    holdout_plan_id = holdout_plan_payload.pop(
        "o_v8_future_unseen_holdout_plan_report_id"
    )
    assert canonical_json_sha256(holdout_plan_payload) == holdout_plan_id
    assert (
        holdout_plan["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_PLAN_SCHEMA_VERSION
    )
    assert holdout_plan["report_type"] == "o_v8_future_unseen_holdout_plan"
    assert holdout_plan["diagnostic_only"] is True
    assert holdout_plan["simulation_only"] is True
    assert holdout_plan["future_unseen_holdout_plan_ready"] is True
    assert holdout_plan["future_unseen_holdout_blocking_reason_codes"] == []
    assert set(holdout_plan["future_unseen_holdout_required_checks"]) >= {
        "allowed_order_count_threshold",
        "allowed_order_origin_safety_requirement",
        "deterministic_report_hashes_frozen_before_holdout_evaluation",
        "exposure_microstructure_pass_requirement",
        "input_reports_do_not_use_forbidden_outcomes",
        "missing_runtime_fields_threshold",
        "no_overlap_with_shadow_validation_decisions",
        "p_up_agreement_requirement",
        "provenance_violation_threshold",
        "residual_blocker_classification_requirement",
        "same_execution_guard_config",
        "same_o_model_action_rank_config",
        "same_runtime_field_cleanup_backfill_rules",
        "same_simulated_ledger_rules",
        "unseen_date_window_definition",
    }
    assert holdout_plan["future_unseen_holdout_required"] is True
    assert holdout_plan["paper_candidate_allowed"] is False
    assert holdout_plan["v8_execution_handoff_allowed"] is False
    assert holdout_plan["source_model_candidate_eligible"] is False
    assert holdout_plan["freeze_ready"] is False
    assert holdout_plan["promotion_evidence_eligible"] is False
    assert holdout_plan["#146_start_allowed"] is False
    assert holdout_plan["#134_resume_allowed"] is False
    assert holdout_plan["paper_only"] is True
    assert holdout_plan["capital_at_risk"] is False

    paper_gate = result.v8_paper_candidate_gate_design_report
    paper_gate_payload = dict(paper_gate)
    paper_gate_id = paper_gate_payload.pop(
        "o_v8_paper_candidate_gate_design_report_id"
    )
    assert canonical_json_sha256(paper_gate_payload) == paper_gate_id
    assert (
        paper_gate["schema_version"]
        == O_V8_PAPER_CANDIDATE_GATE_DESIGN_SCHEMA_VERSION
    )
    assert paper_gate["report_type"] == "o_v8_paper_candidate_gate_design"
    assert paper_gate["diagnostic_only"] is True
    assert paper_gate["simulation_only"] is True
    assert paper_gate["paper_candidate_gate_design_ready"] is True
    assert paper_gate["paper_candidate_gate_blocking_reason_codes"] == []
    assert set(paper_gate["paper_candidate_required_checks"]) >= {
        "capital_at_risk_false",
        "explicit_execution_handoff_gate_passed_on_holdout",
        "explicit_manual_approval_required",
        "future_unseen_holdout_passed",
        "input_report_hashes_available",
        "no_model_layer_regret_risk_selection_enabled",
        "paper_only_flags_enforced",
        "polymarket_writes_disabled",
        "source_freeze_promotion_gates_remain_separate",
        "wallet_signing_disabled",
        "zero_forbidden_outcome_field_usage",
        "zero_provenance_violations",
        "zero_source_score_mutation",
    }
    assert paper_gate["paper_candidate_allowed"] is False
    assert paper_gate["future_unseen_holdout_required"] is True
    assert paper_gate["future_paper_candidate_gate_required"] is True
    assert paper_gate["v8_execution_handoff_allowed"] is False
    assert paper_gate["source_model_candidate_eligible"] is False
    assert paper_gate["freeze_ready"] is False
    assert paper_gate["promotion_evidence_eligible"] is False
    assert paper_gate["#146_start_allowed"] is False
    assert paper_gate["#134_resume_allowed"] is False
    assert paper_gate["paper_only"] is True
    assert paper_gate["capital_at_risk"] is False
    assert paper_gate["polymarket_write_enabled"] is False
    assert paper_gate["wallet_signing_enabled"] is False

    collection_plan = result.v8_future_unseen_holdout_collection_plan_report
    collection_payload = dict(collection_plan)
    collection_id = collection_payload.pop(
        "o_v8_future_unseen_holdout_collection_plan_report_id"
    )
    assert canonical_json_sha256(collection_payload) == collection_id
    assert (
        collection_plan["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_COLLECTION_PLAN_SCHEMA_VERSION
    )
    assert (
        collection_plan["report_type"]
        == "o_v8_future_unseen_holdout_collection_plan"
    )
    assert collection_plan["diagnostic_only"] is True
    assert collection_plan["simulation_only"] is True
    assert collection_plan["future_unseen_holdout_collection_plan_ready"] is True
    assert collection_plan[
        "future_unseen_holdout_collection_blocking_reason_codes"
    ] == []
    assert collection_plan["collection_status"] == "not_started"
    assert collection_plan["future_outcome_evaluation_generated"] is False
    assert collection_plan["future_outcome_evaluation_artifacts_generated"] == []
    assert collection_plan["paper_candidate_allowed"] is False
    assert collection_plan["v8_execution_handoff_allowed"] is False
    assert collection_plan["source_model_candidate_eligible"] is False
    assert collection_plan["freeze_ready"] is False
    assert collection_plan["promotion_evidence_eligible"] is False
    assert collection_plan["#146_start_allowed"] is False
    assert collection_plan["#134_resume_allowed"] is False
    assert collection_plan["paper_only"] is True
    assert collection_plan["capital_at_risk"] is False
    assert collection_plan["polymarket_write_enabled"] is False
    assert collection_plan["wallet_signing_enabled"] is False

    block_analysis = result.v8_execution_guard_block_analysis_report
    block_payload = dict(block_analysis)
    block_id = block_payload.pop("o_v8_execution_guard_block_analysis_report_id")
    assert canonical_json_sha256(block_payload) == block_id
    assert (
        block_analysis["schema_version"]
        == O_V8_EXECUTION_GUARD_BLOCK_ANALYSIS_SCHEMA_VERSION
    )
    assert block_analysis["report_type"] == "o_v8_execution_guard_block_analysis"
    assert block_analysis["diagnostic_only"] is True
    assert block_analysis["simulation_only"] is True
    assert block_analysis["uses_validation_outcomes_for_tuning"] is False
    assert block_analysis["thresholds_tuned"] is False
    assert block_analysis["mutates_o_model_predicted_score"] is False
    assert block_analysis["mutates_source_ranking_scores"] is False
    assert block_analysis["safe_order_discovery_uses_realized_pnl"] is False
    assert block_analysis["simulated_order_replay_report_id"] == simulated_replay[
        "o_v8_execution_simulated_order_replay_report_id"
    ]
    assert block_analysis["decision_count"] == simulated_replay["decision_count"]
    assert block_analysis["blocked_decision_count"] == simulated_replay[
        "blocked_decision_count"
    ]
    assert block_analysis["allowed_decision_count"] == simulated_replay[
        "simulated_allowed_order_count"
    ]
    assert len(block_analysis["blocked_decision_analysis_rows"]) == block_analysis[
        "blocked_decision_count"
    ]
    assert isinstance(block_analysis["primary_blocker_categories"], list)
    discovery_summary = block_analysis["safe_order_discovery_summary"]
    assert discovery_summary["safe_order_candidate_count"] >= 0
    assert discovery_summary["fundamentally_unsafe_count"] >= 0
    assert block_analysis["v8_execution_handoff_allowed"] is False
    assert block_analysis["source_model_candidate_eligible"] is False
    assert block_analysis["freeze_ready"] is False
    assert block_analysis["promotion_evidence_eligible"] is False
    assert block_analysis["#146_start_allowed"] is False
    assert block_analysis["#134_resume_allowed"] is False
    assert block_analysis["paper_only"] is True
    assert block_analysis["capital_at_risk"] is False
    for row in block_analysis["blocked_decision_analysis_rows"]:
        assert set(row) >= {
            "decision_group_id",
            "market_id",
            "decision_ts",
            "source_selected_action",
            "source_selected_family",
            "source_selected_side",
            "execution_guarded_action",
            "execution_guarded_family",
            "execution_guarded_side",
            "minimal_blocking_set",
            "safe_order_discovery_classification",
            "safe_order_discovery_reason_codes",
            "time_to_close_bucket",
        }
        assert row["source_score_mutated"] is False
        assert row["o_model_predicted_score_mutated"] is False

    field_coverage = result.v8_execution_runtime_field_coverage_report
    coverage_payload = dict(field_coverage)
    coverage_id = coverage_payload.pop(
        "o_v8_execution_runtime_field_coverage_report_id"
    )
    assert canonical_json_sha256(coverage_payload) == coverage_id
    assert (
        field_coverage["schema_version"]
        == O_V8_EXECUTION_RUNTIME_FIELD_COVERAGE_SCHEMA_VERSION
    )
    assert field_coverage["report_type"] == "o_v8_execution_runtime_field_coverage"
    assert field_coverage["diagnostic_only"] is True
    assert field_coverage["simulation_only"] is True
    assert field_coverage["uses_validation_outcomes_for_tuning"] is False
    assert field_coverage["thresholds_tuned"] is False
    assert field_coverage["backfill_rules_applied"] == (
        field_coverage["applied_runtime_field_backfill_count"] > 0
    )
    assert field_coverage["proposed_backfill_rules_only"] == (
        field_coverage["applied_runtime_field_backfill_count"] == 0
    )
    assert field_coverage["mutates_o_model_predicted_score"] is False
    assert field_coverage["mutates_source_ranking_scores"] is False
    assert "applied_runtime_field_backfill_count" in field_coverage
    assert "applied_runtime_field_backfill_rule_counts" in field_coverage
    assert "runtime_field_backfill_provenance_validity_summary" in field_coverage
    assert field_coverage["simulated_order_replay_report_id"] == simulated_replay[
        "o_v8_execution_simulated_order_replay_report_id"
    ]
    assert field_coverage["block_analysis_report_id"] == block_analysis[
        "o_v8_execution_guard_block_analysis_report_id"
    ]
    assert field_coverage["decision_count"] == simulated_replay["decision_count"]
    expected_missing_rows = [
        row
        for row in simulated_replay["simulated_decision_rows"]
        if row["missing_runtime_field_codes"]
    ]
    assert field_coverage["missing_runtime_field_decision_count"] == len(
        expected_missing_rows
    )
    assert len(field_coverage["runtime_field_coverage_decision_rows"]) == len(
        expected_missing_rows
    )
    assert set(field_coverage["classification_counts"]) >= {
        "true_data_coverage_gap",
        "derived_backfill_from_existing_handoff_fields",
        "too_strict_for_simulation_only_mode",
    }
    if (
        field_coverage["applied_runtime_field_backfill_rule_counts"].get(
            "make_non_order_runtime_fields_optional_for_no_trade",
            0,
        )
        == 0
    ):
        assert "optional_for_no_trade" in field_coverage["classification_counts"]
    assert isinstance(field_coverage["primary_missing_runtime_fields"], list)
    assert field_coverage["proposed_deterministic_backfill_rules"]
    assert all(
        rule["applied_now"] == (rule["applied_count"] > 0)
        for rule in field_coverage["proposed_deterministic_backfill_rules"]
    )
    for row in field_coverage["runtime_field_coverage_decision_rows"]:
        assert set(row) >= {
            "decision_group_id",
            "market_id",
            "decision_ts",
            "source_selected_action",
            "source_selected_family",
            "source_selected_side",
            "runtime_field_backfill_candidates",
        }
        for candidate in row["runtime_field_backfill_candidates"]:
            assert set(candidate) >= {
                "missing_field_code",
                "runtime_field_name",
                "field_gap_classification",
                "proposed_rule_id",
                "backfill_source_class",
                "can_backfill_in_later_commit",
                "requires_required_field_policy_change",
                "backfill_rule_applied_now",
                "existing_handoff_evidence",
            }
            assert candidate["backfill_rule_applied_now"] is False
    assert field_coverage["v8_execution_handoff_allowed"] is False
    assert field_coverage["source_model_candidate_eligible"] is False
    assert field_coverage["freeze_ready"] is False
    assert field_coverage["promotion_evidence_eligible"] is False
    assert field_coverage["#146_start_allowed"] is False
    assert field_coverage["#134_resume_allowed"] is False
    assert field_coverage["paper_only"] is True
    assert field_coverage["capital_at_risk"] is False
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
    assert "large_regret_risk_model_report" not in result.artifact_paths
    assert "large_regret_risk_model_summary" not in result.artifact_paths
    assert "selective_action_guard_report" not in result.artifact_paths
    assert "selective_action_guard_summary" not in result.artifact_paths
    assert result.artifact_paths["v8_action_rank_handoff_report"].exists()
    assert result.artifact_paths["v8_action_rank_handoff_summary"].exists()
    assert result.artifact_paths["v8_execution_risk_guard_report"].exists()
    assert result.artifact_paths["v8_execution_risk_guard_summary"].exists()
    assert result.artifact_paths["v8_execution_runtime_state_report"].exists()
    assert result.artifact_paths["v8_execution_runtime_state_summary"].exists()
    assert result.artifact_paths[
        "v8_execution_simulated_order_replay_report"
    ].exists()
    assert result.artifact_paths[
        "v8_execution_simulated_order_replay_summary"
    ].exists()
    assert result.artifact_paths["v8_execution_allowed_order_quality_report"].exists()
    assert result.artifact_paths["v8_execution_allowed_order_quality_summary"].exists()
    assert result.artifact_paths["v8_execution_policy_readiness_report"].exists()
    assert result.artifact_paths["v8_execution_policy_readiness_summary"].exists()
    assert result.artifact_paths["v8_execution_guard_block_analysis_report"].exists()
    assert result.artifact_paths["v8_execution_guard_block_analysis_summary"].exists()
    assert result.artifact_paths["v8_execution_runtime_field_coverage_report"].exists()
    assert result.artifact_paths["v8_execution_runtime_field_coverage_summary"].exists()
    assert result.artifact_paths["v8_execution_handoff_gate_report"].exists()
    assert result.artifact_paths["v8_execution_handoff_gate_summary"].exists()
    assert result.artifact_paths["v8_future_unseen_holdout_plan_report"].exists()
    assert result.artifact_paths["v8_future_unseen_holdout_plan_summary"].exists()
    assert result.artifact_paths["v8_paper_candidate_gate_design_report"].exists()
    assert result.artifact_paths["v8_paper_candidate_gate_design_summary"].exists()
    assert result.artifact_paths[
        "v8_future_unseen_holdout_collection_plan_report"
    ].exists()
    assert result.artifact_paths[
        "v8_future_unseen_holdout_collection_plan_summary"
    ].exists()
    for artifact_name in (
        "v8_future_unseen_holdout_raw_collection_manifest",
        "v8_future_unseen_holdout_raw_collection_summary",
        "v8_future_unseen_holdout_input_freeze_manifest",
        "v8_future_unseen_holdout_input_freeze_summary",
        "v8_future_unseen_holdout_action_rank_report",
        "v8_future_unseen_holdout_action_rank_summary",
        "v8_future_unseen_holdout_execution_replay_report",
        "v8_future_unseen_holdout_execution_replay_summary",
        "v8_future_unseen_holdout_policy_readiness_report",
        "v8_future_unseen_holdout_policy_readiness_summary",
        "v8_future_unseen_holdout_handoff_gate_report",
        "v8_future_unseen_holdout_handoff_gate_summary",
        "v8_future_unseen_holdout_paper_candidate_gate_report",
        "v8_future_unseen_holdout_paper_candidate_gate_summary",
    ):
        assert result.artifact_paths[artifact_name].exists()

    manifest = _read_json(result.artifact_paths["manifest"])
    assert "hts_p_up_confidently_wrong_feature_diagnostic_report" in manifest[
        "artifact_hashes"
    ]
    assert "hts_p_up_confidently_wrong_feature_diagnostic_summary" in manifest[
        "artifact_hashes"
    ]
    assert "large_regret_risk_model_report" not in manifest["artifact_hashes"]
    assert "large_regret_risk_model_summary" not in manifest["artifact_hashes"]
    assert "selective_action_guard_report" not in manifest["artifact_hashes"]
    assert "selective_action_guard_summary" not in manifest["artifact_hashes"]
    assert "v8_action_rank_handoff_report" in manifest["artifact_hashes"]
    assert "v8_action_rank_handoff_summary" in manifest["artifact_hashes"]
    assert "v8_execution_risk_guard_report" in manifest["artifact_hashes"]
    assert "v8_execution_risk_guard_summary" in manifest["artifact_hashes"]
    assert "v8_execution_runtime_state_report" in manifest["artifact_hashes"]
    assert "v8_execution_runtime_state_summary" in manifest["artifact_hashes"]
    assert (
        "v8_execution_simulated_order_replay_report" in manifest["artifact_hashes"]
    )
    assert (
        "v8_execution_simulated_order_replay_summary" in manifest["artifact_hashes"]
    )
    assert "v8_execution_allowed_order_quality_report" in manifest["artifact_hashes"]
    assert "v8_execution_allowed_order_quality_summary" in manifest["artifact_hashes"]
    assert "v8_execution_policy_readiness_report" in manifest["artifact_hashes"]
    assert "v8_execution_policy_readiness_summary" in manifest["artifact_hashes"]
    assert "v8_execution_guard_block_analysis_report" in manifest["artifact_hashes"]
    assert "v8_execution_guard_block_analysis_summary" in manifest["artifact_hashes"]
    assert "v8_execution_runtime_field_coverage_report" in manifest["artifact_hashes"]
    assert "v8_execution_runtime_field_coverage_summary" in manifest["artifact_hashes"]
    assert "v8_execution_handoff_gate_report" in manifest["artifact_hashes"]
    assert "v8_execution_handoff_gate_summary" in manifest["artifact_hashes"]
    assert "v8_future_unseen_holdout_plan_report" in manifest["artifact_hashes"]
    assert "v8_future_unseen_holdout_plan_summary" in manifest["artifact_hashes"]
    assert "v8_paper_candidate_gate_design_report" in manifest["artifact_hashes"]
    assert "v8_paper_candidate_gate_design_summary" in manifest["artifact_hashes"]
    assert (
        "v8_future_unseen_holdout_collection_plan_report"
        in manifest["artifact_hashes"]
    )
    assert (
        "v8_future_unseen_holdout_collection_plan_summary"
        in manifest["artifact_hashes"]
    )
    for artifact_name in (
        "v8_future_unseen_holdout_raw_collection_manifest",
        "v8_future_unseen_holdout_raw_collection_summary",
        "v8_future_unseen_holdout_input_freeze_manifest",
        "v8_future_unseen_holdout_input_freeze_summary",
        "v8_future_unseen_holdout_action_rank_report",
        "v8_future_unseen_holdout_action_rank_summary",
        "v8_future_unseen_holdout_execution_replay_report",
        "v8_future_unseen_holdout_execution_replay_summary",
        "v8_future_unseen_holdout_policy_readiness_report",
        "v8_future_unseen_holdout_policy_readiness_summary",
        "v8_future_unseen_holdout_handoff_gate_report",
        "v8_future_unseen_holdout_handoff_gate_summary",
        "v8_future_unseen_holdout_paper_candidate_gate_report",
        "v8_future_unseen_holdout_paper_candidate_gate_summary",
    ):
        assert artifact_name in manifest["artifact_hashes"]
    assert manifest["large_regret_risk_model_report_available"] is False
    assert manifest["selective_action_guard_report_available"] is False
    assert manifest["large_regret_risk_model_enabled"] is False
    assert manifest["selective_action_guard_enabled"] is False
    assert manifest["model_layer_regret_risk_selection_deferred_to_issue"] == "#158"
    assert manifest["v8_action_rank_handoff_report_available"] is True
    assert manifest["v8_execution_risk_guard_report_available"] is True
    assert manifest["v8_execution_risk_guard_report_id"] == execution_guard[
        "o_v8_execution_risk_guard_report_id"
    ]
    assert manifest["v8_execution_guard_runtime_validation_passed"] is False
    assert manifest["v8_execution_runtime_state_report_available"] is True
    assert manifest["v8_execution_runtime_state_report_id"] == runtime_state[
        "o_v8_execution_runtime_state_report_id"
    ]
    assert manifest["v8_execution_runtime_state_validation_passed"] is True
    assert manifest["v8_execution_simulated_order_replay_report_available"] is True
    assert manifest["v8_execution_simulated_order_replay_report_id"] == (
        simulated_replay["o_v8_execution_simulated_order_replay_report_id"]
    )
    assert manifest["v8_execution_simulated_allowed_order_count"] == (
        simulated_replay["simulated_allowed_order_count"]
    )
    assert manifest["v8_execution_simulated_blocked_decision_count"] == (
        simulated_replay["blocked_decision_count"]
    )
    assert (
        manifest["v8_execution_simulated_runtime_risk_control_validation_passed"]
        is True
    )
    assert manifest["v8_execution_allowed_order_quality_report_available"] is True
    assert manifest["v8_execution_allowed_order_quality_report_id"] == (
        allowed_quality["o_v8_execution_allowed_order_quality_report_id"]
    )
    assert manifest["v8_execution_allowed_order_quality_allowed_order_count"] == (
        allowed_quality["allowed_order_count"]
    )
    assert manifest["v8_execution_allowed_order_quality_blocked_decision_count"] == (
        allowed_quality["blocked_decision_count"]
    )
    assert manifest["v8_execution_allowed_order_quality_recommendation_counts"] == (
        allowed_quality["deterministic_recommendation_counts"]
    )
    assert manifest["v8_execution_policy_readiness_report_available"] is True
    assert manifest["v8_execution_policy_readiness_report_id"] == (
        policy_readiness["o_v8_execution_policy_readiness_report_id"]
    )
    assert manifest["v8_execution_policy_readiness_diagnostic_passed"] == (
        policy_readiness["execution_policy_readiness_diagnostic_passed"]
    )
    assert manifest["v8_execution_policy_readiness_required_checks"] == (
        policy_readiness["execution_policy_readiness_required_checks"]
    )
    assert manifest["v8_execution_policy_readiness_blocking_reason_codes"] == (
        policy_readiness["execution_policy_readiness_blocking_reason_codes"]
    )
    assert manifest["future_explicit_execution_handoff_gate_required"] is True
    assert manifest["v8_execution_guard_block_analysis_report_available"] is True
    assert manifest["v8_execution_guard_block_analysis_report_id"] == block_analysis[
        "o_v8_execution_guard_block_analysis_report_id"
    ]
    assert (
        manifest["v8_execution_guard_block_analysis_safe_order_candidate_count"]
        == discovery_summary["safe_order_candidate_count"]
    )
    assert (
        manifest["v8_execution_guard_block_analysis_fundamentally_unsafe_count"]
        == discovery_summary["fundamentally_unsafe_count"]
    )
    assert (
        manifest["v8_execution_guard_block_analysis_primary_blocker_categories"]
        == block_analysis["primary_blocker_categories"]
    )
    assert manifest["v8_execution_runtime_field_coverage_report_available"] is True
    assert manifest["v8_execution_runtime_field_coverage_report_id"] == field_coverage[
        "o_v8_execution_runtime_field_coverage_report_id"
    ]
    assert manifest["v8_execution_runtime_field_missing_decision_count"] == (
        field_coverage["missing_runtime_field_decision_count"]
    )
    assert manifest["v8_execution_runtime_field_true_data_gap_count"] == (
        field_coverage["classification_counts"]["true_data_coverage_gap"]
    )
    assert (
        manifest["v8_execution_runtime_field_safe_backfill_candidate_count"]
        == field_coverage["safe_backfill_candidate_count"]
    )
    assert (
        manifest[
            "v8_execution_runtime_field_existing_handoff_backfill_candidate_count"
        ]
        == field_coverage["existing_handoff_backfill_candidate_count"]
    )
    assert (
        manifest[
            "v8_execution_runtime_field_decision_time_data_join_backfill_candidate_count"
        ]
        == field_coverage["decision_time_data_join_backfill_candidate_count"]
    )
    assert manifest["v8_execution_runtime_field_optional_for_no_trade_count"] == (
        field_coverage["classification_counts"]["optional_for_no_trade"]
    )
    assert (
        manifest["v8_execution_runtime_field_simulation_policy_too_strict_count"]
        == field_coverage["classification_counts"][
            "too_strict_for_simulation_only_mode"
        ]
    )
    assert manifest["v8_execution_runtime_field_primary_missing_fields"] == (
        field_coverage["primary_missing_runtime_fields"]
    )
    assert manifest["v8_execution_runtime_field_backfill_rules_applied"] == (
        field_coverage["runtime_field_backfill_rules_applied"]
    )
    assert manifest["v8_execution_runtime_field_applied_backfill_count"] == (
        field_coverage["applied_runtime_field_backfill_count"]
    )
    assert manifest["v8_execution_runtime_field_applied_backfill_rule_counts"] == (
        field_coverage["applied_runtime_field_backfill_rule_counts"]
    )
    assert (
        manifest["v8_execution_runtime_field_backfill_provenance_validity_summary"]
        == field_coverage["runtime_field_backfill_provenance_validity_summary"]
    )
    assert manifest["v8_execution_handoff_gate_report_available"] is True
    assert manifest["v8_execution_handoff_gate_report_id"] == (
        handoff_gate["o_v8_execution_handoff_gate_report_id"]
    )
    assert manifest["explicit_execution_handoff_gate_passed"] == (
        handoff_gate["explicit_execution_handoff_gate_passed"]
    )
    assert manifest["explicit_execution_handoff_blocking_reason_codes"] == (
        handoff_gate["explicit_execution_handoff_blocking_reason_codes"]
    )
    assert (
        manifest["explicit_execution_handoff_gate_mode"]
        == "diagnostic_only_fail_closed"
    )
    assert manifest["explicit_execution_handoff_allowed"] is False
    assert manifest["future_unseen_holdout_required"] is True
    assert manifest["future_paper_candidate_gate_required"] is True
    assert manifest["v8_future_unseen_holdout_plan_report_available"] is True
    assert manifest["v8_future_unseen_holdout_plan_report_id"] == (
        holdout_plan["o_v8_future_unseen_holdout_plan_report_id"]
    )
    assert manifest["future_unseen_holdout_plan_ready"] == (
        holdout_plan["future_unseen_holdout_plan_ready"]
    )
    assert manifest["future_unseen_holdout_blocking_reason_codes"] == (
        holdout_plan["future_unseen_holdout_blocking_reason_codes"]
    )
    assert manifest["v8_paper_candidate_gate_design_report_available"] is True
    assert manifest["v8_paper_candidate_gate_design_report_id"] == (
        paper_gate["o_v8_paper_candidate_gate_design_report_id"]
    )
    assert manifest["paper_candidate_gate_design_ready"] == (
        paper_gate["paper_candidate_gate_design_ready"]
    )
    assert manifest["paper_candidate_gate_blocking_reason_codes"] == (
        paper_gate["paper_candidate_gate_blocking_reason_codes"]
    )
    assert manifest["paper_candidate_allowed"] is False
    assert (
        manifest["v8_future_unseen_holdout_collection_plan_report_available"]
        is True
    )
    assert manifest["v8_future_unseen_holdout_collection_plan_report_id"] == (
        collection_plan["o_v8_future_unseen_holdout_collection_plan_report_id"]
    )
    assert manifest["future_unseen_holdout_collection_plan_ready"] == (
        collection_plan["future_unseen_holdout_collection_plan_ready"]
    )
    assert manifest["future_unseen_holdout_collection_blocking_reason_codes"] == (
        collection_plan[
            "future_unseen_holdout_collection_blocking_reason_codes"
        ]
    )
    assert (
        manifest["v8_future_unseen_holdout_raw_collection_manifest_available"]
        is True
    )
    assert manifest["future_unseen_holdout_raw_collection_ready"] is False
    assert (
        "future_holdout_raw_collection_manifest_missing"
        in manifest["future_unseen_holdout_raw_collection_blocking_reason_codes"]
    )
    assert manifest["future_window_time_validation_passed"] is False
    assert looks_like_sha256(manifest["future_holdout_prior_reference_hash"])
    assert manifest["future_holdout_prior_reference_sources"]
    assert manifest["future_holdout_collection_plan_created_ts"] is not None
    assert manifest["future_holdout_raw_manifest_created_ts"] is None
    assert (
        manifest["v8_future_unseen_holdout_input_freeze_manifest_available"]
        is True
    )
    assert manifest["future_unseen_holdout_input_freeze_ready"] is False
    assert manifest["v8_future_unseen_holdout_action_rank_report_available"] is True
    assert manifest["future_unseen_holdout_action_rank_ready"] is False
    assert (
        manifest["v8_future_unseen_holdout_execution_replay_report_available"]
        is True
    )
    assert manifest["future_unseen_holdout_execution_replay_ready"] is False
    assert manifest["future_unseen_holdout_simulated_allowed_order_count"] == 0
    assert (
        manifest["v8_future_unseen_holdout_policy_readiness_report_available"]
        is True
    )
    assert manifest["future_unseen_holdout_policy_readiness_passed"] is False
    assert manifest["v8_future_unseen_holdout_handoff_gate_report_available"] is True
    assert manifest["future_unseen_holdout_handoff_gate_passed"] is False
    assert (
        manifest["v8_future_unseen_holdout_paper_candidate_gate_report_available"]
        is True
    )
    assert manifest["future_unseen_holdout_paper_candidate_gate_passed"] is False
    assert manifest["strict_calibration_quality_passed"] == gate[
        "strict_calibration_quality_passed"
    ]
    assert manifest["relaxed_diagnostic_calibration_quality_passed"] == gate[
        "relaxed_diagnostic_calibration_quality_passed"
    ]
    assert manifest["relaxed_diagnostic_source_candidate"] == gate[
        "relaxed_diagnostic_source_candidate"
    ]
    assert manifest["v8_action_rank_quality_passed"] == gate[
        "v8_action_rank_quality_passed"
    ]
    assert manifest["v8_action_rank_candidate_eligible"] == gate[
        "v8_action_rank_candidate_eligible"
    ]
    assert manifest["v8_execution_risk_control_required"] is True
    assert manifest["v8_execution_handoff_allowed"] is False
    assert manifest["source_model_candidate_eligible"] is False
    assert manifest["freeze_ready"] is False
    assert manifest["promotion_evidence_eligible"] is False
    assert manifest["strict_source_gate_remains_failed"] == gate[
        "strict_source_gate_remains_failed"
    ]
    assert manifest["strict_vs_relaxed_gate_summary"] == gate[
        "strict_vs_relaxed_gate_summary"
    ]
    assert manifest["relaxed_diagnostic_no_paper_live_unlock"] is True


def test_o_v8_execution_guard_blocks_trade_when_exposure_state_missing() -> None:
    def _ranked_action(action: str, score: float, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "selected_action": action,
            "selected_side": "UP" if "BUY_UP" in action else "DOWN",
            "selected_action_family": "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "HOLD_TO_SETTLEMENT",
            "corrected_model_score": score,
            "raw_model_score": score - 0.05,
            "high_score_flag": score >= 0.75,
            "p_up_action_disagreement": False,
            "microstructure_snapshot": {
                "book_staleness_ms": 500.0,
                "spread_bps": 200.0,
                "queue_fill_proxy": 0.90,
                "time_to_close_seconds": 240.0,
                "entry_ask": 0.45,
                "executable_exit_bid_proxy": 0.47,
            },
        }

    row = {
        "decision_group_id": "source|market|123",
        "market_id": "market",
        "decision_ts": 123,
        "selected_action": "BUY_UP_HOLD_TO_SETTLEMENT",
        "selected_side": "UP",
        "selected_action_family": "HOLD_TO_SETTLEMENT",
        "corrected_model_score": 0.82,
        "raw_model_score": 0.72,
        "score_components": {"base_score": 0.50},
        "high_score_flag": True,
        "p_up": 0.70,
        "p_down": 0.30,
        "p_up_action_disagreement": False,
        "microstructure_snapshot": {
            "book_staleness_ms": 500.0,
            "spread_bps": 200.0,
            "queue_fill_proxy": 0.90,
            "time_to_close_seconds": 240.0,
            "entry_ask": 0.45,
            "executable_exit_bid_proxy": 0.47,
        },
        "reference_price_feature_provenance": {
            "source_fields_used": ["price_to_beat", "reference_mid"],
            "max_input_ts": 120,
            "decision_ts": 123,
            "provenance_valid": True,
        },
        "full_5_action_ranking": [
            _ranked_action("BUY_UP_HOLD_TO_SETTLEMENT", 0.82, 1),
            _ranked_action("BUY_UP_SELL_BEFORE_CLOSE", 0.79, 2),
            _ranked_action("BUY_DOWN_HOLD_TO_SETTLEMENT", 0.40, 3),
            _ranked_action("BUY_DOWN_SELL_BEFORE_CLOSE", 0.35, 4),
            {
                **_ranked_action("NO_TRADE", 0.10, 5),
                "selected_side": "NONE",
                "selected_action_family": "NO_TRADE",
            },
        ],
    }

    guarded = _v8_execution_guard_decision(
        row,
        guard_config=_v8_execution_guard_config(),
    )

    assert guarded["source_selected_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert guarded["source_model_score"] == row["corrected_model_score"]
    assert guarded["source_score_mutated"] is False
    assert guarded["o_model_predicted_score_mutated"] is False
    assert guarded["execution_guarded_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert "execution_hts_downgraded_to_same_side_sbc" in guarded[
        "execution_guard_reason_codes"
    ]
    assert guarded["order_allowed"] is False
    assert guarded["proposed_order_size"] == 0.0
    assert guarded["fail_closed"] is True
    assert guarded["required_runtime_fields_present"] is False
    assert "execution_exposure_state_missing" in guarded[
        "missing_runtime_field_codes"
    ]
    assert "execution_exposure_state_missing" in guarded[
        "execution_blocking_reason_codes"
    ]
    assert "execution_required_runtime_fields_missing" in guarded[
        "execution_blocking_reason_codes"
    ]
    assert "execution_blocked_size_zero" in guarded["sizing_reason_codes"]


def test_o_v8_execution_guard_applies_no_trade_optional_runtime_policy() -> None:
    row = {
        "decision_group_id": "source|market|no-trade",
        "market_id": "market",
        "decision_ts": 123,
        "selected_action": "NO_TRADE",
        "selected_side": "NONE",
        "selected_action_family": "NO_TRADE",
        "corrected_model_score": 0.82,
        "raw_model_score": 0.72,
        "score_components": {"base_score": 0.50},
        "high_score_flag": True,
        "p_up": 0.70,
        "p_down": 0.30,
        "p_up_action_disagreement": None,
        "microstructure_snapshot": {},
        "reference_price_feature_provenance": {},
        "full_5_action_ranking": [
            {
                "rank": 1,
                "selected_action": "NO_TRADE",
                "selected_side": "NONE",
                "selected_action_family": "NO_TRADE",
                "corrected_model_score": 0.82,
                "raw_model_score": 0.72,
                "high_score_flag": True,
                "p_up_action_disagreement": None,
            }
        ],
    }

    guarded = _v8_execution_guard_decision(
        row,
        guard_config=_v8_execution_guard_config(),
    )

    assert guarded["source_selected_action"] == "NO_TRADE"
    assert guarded["required_runtime_fields_present"] is True
    assert guarded["missing_runtime_field_codes"] == []
    assert "execution_required_runtime_fields_missing" not in guarded[
        "execution_blocking_reason_codes"
    ]
    assert guarded["runtime_field_backfill_rules_applied"] is True
    assert guarded["runtime_field_backfill_rule_counts"] == {
        "make_non_order_runtime_fields_optional_for_no_trade": 6
    }
    assert len(guarded["runtime_field_applied_backfill_rows"]) == 6
    assert all(
        row["application_type"] == "required_field_policy_relaxation"
        for row in guarded["runtime_field_applied_backfill_rows"]
    )
    assert guarded["runtime_field_backfill_provenance_valid"] is True
    assert guarded["order_allowed"] is False
    assert guarded["source_score_mutated"] is False
    assert guarded["o_model_predicted_score_mutated"] is False


def test_o_v8_execution_guard_backfills_time_to_close_with_valid_provenance() -> None:
    row = {
        "decision_group_id": "source|market|valid-backfill",
        "market_id": "market",
        "decision_ts": 123,
        "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
        "selected_side": "UP",
        "selected_action_family": "SELL_BEFORE_CLOSE",
        "corrected_model_score": 0.82,
        "raw_model_score": 0.72,
        "score_components": {"base_score": 0.50},
        "high_score_flag": True,
        "p_up": 0.70,
        "p_down": 0.30,
        "p_up_action_disagreement": False,
        "microstructure_snapshot": {
            "book_staleness_ms": 500.0,
            "spread_bps": 200.0,
            "queue_fill_proxy": 0.90,
            "time_to_close_seconds": None,
        },
        "reference_price_feature_provenance": {
            "source_fields_used": ["price_to_beat", "reference_mid"],
            "max_input_ts": 120,
            "decision_ts": 123,
            "provenance_valid": True,
        },
        "runtime_field_backfill_sources": {
            "microstructure_snapshot.time_to_close_seconds": {
                "field": "microstructure_snapshot.time_to_close_seconds",
                "value": 180.0,
                "source_field_name": (
                    "polymarket_feature_rows.features.time_to_close_seconds"
                ),
                "source_timestamp": 120,
                "max_input_ts": 120,
                "decision_ts": 123,
                "deterministic_rule_id": (
                    "backfill_time_to_close_from_decision_time_feature_or_market_schedule"
                ),
                "provenance_valid": True,
                "reason_codes": ["decision_time_time_to_close_source_available"],
            }
        },
        "full_5_action_ranking": [
            {
                "rank": 1,
                "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
                "selected_side": "UP",
                "selected_action_family": "SELL_BEFORE_CLOSE",
                "corrected_model_score": 0.82,
                "raw_model_score": 0.72,
                "high_score_flag": True,
                "p_up_action_disagreement": False,
            },
            {
                "rank": 2,
                "selected_action": "NO_TRADE",
                "selected_side": "NONE",
                "selected_action_family": "NO_TRADE",
                "corrected_model_score": 0.10,
                "raw_model_score": 0.05,
                "high_score_flag": False,
                "p_up_action_disagreement": None,
            },
        ],
    }

    guarded = _v8_execution_guard_decision(
        row,
        guard_config=_v8_execution_guard_config(),
        runtime_state={
            "runtime_state_validation_passed": True,
            "current_total_exposure": 0.0,
            "current_market_exposure_by_market_id": {},
            "current_side_exposure_by_side": {"DOWN": 0.0, "NONE": 0.0, "UP": 0.0},
            "open_position_by_market_id": {},
            "open_position_by_market_side": {},
            "cooldown_state": {},
        },
        runtime_mode="simulated_runtime_state",
    )

    assert guarded["microstructure_snapshot"]["time_to_close_seconds"] == 180.0
    assert guarded["required_runtime_fields_present"] is True
    assert guarded["missing_runtime_field_codes"] == []
    assert "execution_time_to_close_unsafe" not in guarded[
        "execution_blocking_reason_codes"
    ]
    assert guarded["runtime_field_backfill_rules_applied"] is True
    assert guarded["runtime_field_backfill_rule_counts"] == {
        "backfill_time_to_close_from_decision_time_feature_or_market_schedule": 1
    }
    applied = guarded["runtime_field_applied_backfill_rows"][0]
    assert applied["runtime_field_name"] == "microstructure_snapshot.time_to_close_seconds"
    assert applied["source_field_name"] == (
        "polymarket_feature_rows.features.time_to_close_seconds"
    )
    assert applied["source_timestamp"] == 120
    assert applied["max_input_ts"] == 120
    assert applied["provenance_valid"] is True
    assert guarded["runtime_field_backfill_provenance_valid"] is True
    assert guarded["order_allowed"] is True
    assert guarded["source_score_mutated"] is False
    assert guarded["o_model_predicted_score_mutated"] is False


def test_o_v8_execution_guard_rejects_invalid_time_to_close_backfill() -> None:
    row = {
        "decision_group_id": "source|market|invalid-backfill",
        "market_id": "market",
        "decision_ts": 123,
        "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
        "selected_side": "UP",
        "selected_action_family": "SELL_BEFORE_CLOSE",
        "corrected_model_score": 0.82,
        "raw_model_score": 0.72,
        "score_components": {"base_score": 0.50},
        "high_score_flag": True,
        "p_up": 0.70,
        "p_down": 0.30,
        "p_up_action_disagreement": False,
        "microstructure_snapshot": {
            "book_staleness_ms": 500.0,
            "spread_bps": 200.0,
            "queue_fill_proxy": 0.90,
            "time_to_close_seconds": None,
        },
        "reference_price_feature_provenance": {
            "source_fields_used": ["price_to_beat", "reference_mid"],
            "max_input_ts": 120,
            "decision_ts": 123,
            "provenance_valid": True,
        },
        "runtime_field_backfill_sources": {
            "microstructure_snapshot.time_to_close_seconds": {
                "field": "microstructure_snapshot.time_to_close_seconds",
                "value": 180.0,
                "source_field_name": (
                    "polymarket_feature_rows.features.time_to_close_seconds"
                ),
                "source_timestamp": 124,
                "max_input_ts": 124,
                "decision_ts": 123,
                "deterministic_rule_id": (
                    "backfill_time_to_close_from_decision_time_feature_or_market_schedule"
                ),
                "provenance_valid": False,
                "reason_codes": ["time_to_close_source_provenance_invalid"],
            }
        },
        "full_5_action_ranking": [
            {
                "rank": 1,
                "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
                "selected_side": "UP",
                "selected_action_family": "SELL_BEFORE_CLOSE",
                "corrected_model_score": 0.82,
                "raw_model_score": 0.72,
                "high_score_flag": True,
                "p_up_action_disagreement": False,
            }
        ],
    }

    guarded = _v8_execution_guard_decision(
        row,
        guard_config=_v8_execution_guard_config(),
        runtime_state={
            "runtime_state_validation_passed": True,
            "current_total_exposure": 0.0,
            "current_market_exposure_by_market_id": {},
            "current_side_exposure_by_side": {"DOWN": 0.0, "NONE": 0.0, "UP": 0.0},
            "open_position_by_market_id": {},
            "open_position_by_market_side": {},
            "cooldown_state": {},
        },
        runtime_mode="simulated_runtime_state",
    )

    assert guarded["microstructure_snapshot"]["time_to_close_seconds"] is None
    assert guarded["required_runtime_fields_present"] is False
    assert "missing_microstructure_time_to_close_seconds" in guarded[
        "missing_runtime_field_codes"
    ]
    assert "execution_required_runtime_fields_missing" in guarded[
        "execution_blocking_reason_codes"
    ]
    assert "execution_time_to_close_unsafe" in guarded[
        "execution_blocking_reason_codes"
    ]
    assert guarded["runtime_field_backfill_rules_applied"] is False
    assert guarded["runtime_field_backfill_provenance_valid"] is False
    assert len(guarded["runtime_field_backfill_provenance_violations"]) == 1
    violation = guarded["runtime_field_backfill_provenance_violations"][0]
    assert violation["source_timestamp"] == 124
    assert violation["max_input_ts"] == 124
    assert violation["provenance_valid"] is False
    assert guarded["order_allowed"] is False
    assert guarded["source_score_mutated"] is False
    assert guarded["o_model_predicted_score_mutated"] is False


def test_o_v8_simulated_runtime_replay_updates_only_allowed_exposure(
    tmp_path: Path,
) -> None:
    def _handoff_row(
        *,
        market_id: str,
        decision_ts: int,
        action: str,
        score: float,
        p_up: float,
    ) -> dict[str, Any]:
        side = "UP" if "BUY_UP" in action else "DOWN"
        family = "SELL_BEFORE_CLOSE"
        ranked_actions = [
            action,
            "BUY_UP_HOLD_TO_SETTLEMENT",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
            "BUY_DOWN_HOLD_TO_SETTLEMENT",
            "NO_TRADE",
        ]
        unique_ranked_actions = []
        for candidate in ranked_actions:
            if candidate not in unique_ranked_actions:
                unique_ranked_actions.append(candidate)
        if len(unique_ranked_actions) < len(O_REQUIRED_DECISION_ACTION_FAMILIES):
            unique_ranked_actions.extend(
                candidate
                for candidate in O_REQUIRED_DECISION_ACTION_FAMILIES
                if candidate not in unique_ranked_actions
            )
        full_ranking = []
        for rank, candidate in enumerate(unique_ranked_actions, start=1):
            candidate_side = "UP" if "BUY_UP" in candidate else "DOWN"
            if candidate == "NO_TRADE":
                candidate_side = "NONE"
            full_ranking.append(
                {
                    "rank": rank,
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "selected_action": candidate,
                    "selected_side": candidate_side,
                    "selected_action_family": "NO_TRADE"
                    if candidate == "NO_TRADE"
                    else "SELL_BEFORE_CLOSE"
                    if candidate.endswith("SELL_BEFORE_CLOSE")
                    else "HOLD_TO_SETTLEMENT",
                    "corrected_model_score": score - (rank - 1) * 0.05,
                    "raw_model_score": score - 0.10,
                    "high_score_flag": rank == 1,
                    "p_up_action_disagreement": False,
                    "microstructure_snapshot": {
                        "book_staleness_ms": 500.0,
                        "spread_bps": 200.0,
                        "queue_fill_proxy": 0.90,
                        "time_to_close_seconds": 240.0,
                        "entry_ask": 0.45,
                        "executable_exit_bid_proxy": 0.47,
                    },
                }
            )
        return {
            "decision_group_id": f"source|{market_id}|{decision_ts}",
            "market_id": market_id,
            "decision_ts": decision_ts,
            "selected_action": action,
            "selected_side": side,
            "selected_action_family": family,
            "corrected_model_score": score,
            "raw_model_score": score - 0.10,
            "score_components": {"base_score": 0.50},
            "high_score_flag": True,
            "p_up": p_up,
            "p_down": 1.0 - p_up,
            "p_up_action_disagreement": False,
            "microstructure_snapshot": {
                "book_staleness_ms": 500.0,
                "spread_bps": 200.0,
                "queue_fill_proxy": 0.90,
                "time_to_close_seconds": 240.0,
                "entry_ask": 0.45,
                "executable_exit_bid_proxy": 0.47,
            },
            "reference_price_feature_provenance": {
                "source_fields_used": ["price_to_beat", "reference_mid"],
                "max_input_ts": decision_ts - 1,
                "decision_ts": decision_ts,
                "provenance_valid": True,
            },
            "full_5_action_ranking": full_ranking,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    handoff_report = {
        "report_type": "o_v8_action_rank_handoff",
        "o_v8_action_rank_handoff_report_id": "handoff-test",
        "selected_action_handoff_rows": [
            _handoff_row(
                market_id="market-a",
                decision_ts=1,
                action="BUY_UP_SELL_BEFORE_CLOSE",
                score=0.90,
                p_up=0.70,
            ),
            _handoff_row(
                market_id="market-a",
                decision_ts=2,
                action="BUY_UP_SELL_BEFORE_CLOSE",
                score=0.88,
                p_up=0.70,
            ),
            _handoff_row(
                market_id="market-b",
                decision_ts=3,
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                score=0.86,
                p_up=0.30,
            ),
        ],
        "v8_execution_handoff_blocking_reason_codes": [
            "future_unseen_holdout_required",
            "paper_live_unlock_prohibited",
        ],
    }
    execution_guard_report = {
        "o_v8_execution_risk_guard_report_id": "guard-test",
    }

    runtime_report, replay_report = _v8_execution_simulated_runtime_reports(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report=handoff_report,
        execution_guard_report=execution_guard_report,
    )

    assert runtime_report["runtime_state_validation_passed"] is True
    assert replay_report["runtime_risk_control_validation_passed"] is True
    assert replay_report["v8_execution_handoff_allowed"] is False
    assert replay_report["simulated_allowed_order_count"] == 2
    assert replay_report["blocked_decision_count"] == 1
    assert replay_report["total_proposed_notional"] == 0.4
    rows = replay_report["simulated_decision_rows"]
    assert rows[0]["order_allowed"] is True
    assert rows[0]["simulated_order_id"] == "sim-v8-o-000001"
    assert rows[0]["post_decision_exposure_state"][
        "current_total_exposure"
    ] == 0.2
    assert rows[1]["order_allowed"] is False
    assert "execution_duplicate_market_side_position" in rows[1][
        "execution_blocking_reason_codes"
    ]
    assert rows[1]["pre_decision_exposure_state"][
        "current_total_exposure"
    ] == rows[1]["post_decision_exposure_state"]["current_total_exposure"]
    assert rows[2]["order_allowed"] is True
    assert rows[2]["simulated_order_id"] == "sim-v8-o-000003"
    assert runtime_report["current_total_exposure"] == 0.4
    assert runtime_report["exposure_by_side"]["UP"] == 0.2
    assert runtime_report["exposure_by_side"]["DOWN"] == 0.2
    assert runtime_report["source_model_candidate_eligible"] is False
    assert runtime_report["freeze_ready"] is False
    assert runtime_report["promotion_evidence_eligible"] is False
    assert runtime_report["#146_start_allowed"] is False
    assert runtime_report["#134_resume_allowed"] is False
    assert runtime_report["paper_only"] is True
    assert runtime_report["capital_at_risk"] is False
    assert replay_report["source_model_candidate_eligible"] is False
    assert replay_report["freeze_ready"] is False
    assert replay_report["promotion_evidence_eligible"] is False
    assert replay_report["#146_start_allowed"] is False
    assert replay_report["#134_resume_allowed"] is False
    assert replay_report["paper_only"] is True
    assert replay_report["capital_at_risk"] is False


def test_o_v8_execution_allowed_order_quality_report_is_deterministic(
    tmp_path: Path,
) -> None:
    def _replay_row(
        *,
        decision_id: str,
        action: str,
        side: str,
        order_allowed: bool,
        blocking_reasons: list[str] | None = None,
        guard_reasons: list[str] | None = None,
        exposure_reasons: list[str] | None = None,
        guarded_action: str | None = None,
        score: float = 0.80,
        time_to_close: float = 240.0,
        p_up_disagreement: bool = False,
        simulated_order_id: str | None = None,
    ) -> dict[str, Any]:
        guarded_action = guarded_action or action
        guarded_side = "UP" if "BUY_UP" in guarded_action else "DOWN"
        guarded_family = (
            "SELL_BEFORE_CLOSE"
            if guarded_action.endswith("SELL_BEFORE_CLOSE")
            else "HOLD_TO_SETTLEMENT"
        )
        exposure_delta = 0.2 if order_allowed else 0.0
        return {
            "decision_group_id": f"source|market-{decision_id}|{decision_id}",
            "market_id": f"market-{decision_id}",
            "decision_ts": int(decision_id),
            "source_selected_action": action,
            "source_selected_family": "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "HOLD_TO_SETTLEMENT",
            "source_selected_side": side,
            "source_model_score": score,
            "source_raw_model_score": score - 0.10,
            "source_high_score_flag": True,
            "p_up": 0.70 if side == "UP" else 0.30,
            "p_down": 0.30 if side == "UP" else 0.70,
            "p_up_action_disagreement": p_up_disagreement,
            "microstructure_snapshot": {
                "book_staleness_ms": 500.0,
                "spread_bps": 200.0,
                "queue_fill_proxy": 0.90,
                "time_to_close_seconds": time_to_close,
            },
            "execution_guarded_action": guarded_action,
            "execution_guarded_family": guarded_family,
            "execution_guarded_side": guarded_side,
            "execution_guarded_score": score - 0.01,
            "execution_score_penalties": {"spread_penalty": 0.01},
            "order_allowed": order_allowed,
            "proposed_order_size": 0.2 if order_allowed else 0.0,
            "uncapped_proposed_order_size": 0.2,
            "sizing_reason_codes": ["execution_size_high_score_default"]
            if order_allowed
            else ["execution_blocked_size_zero"],
            "exposure_reason_codes": exposure_reasons or [],
            "execution_guard_reason_codes": guard_reasons or [],
            "execution_blocking_reason_codes": blocking_reasons or [],
            "missing_runtime_field_codes": [],
            "pre_decision_exposure_state": {
                "current_total_exposure": 0.0,
                "current_market_exposure_by_market_id": {},
                "current_side_exposure_by_side": {"DOWN": 0.0, "UP": 0.0},
                "executed_simulated_order_count": 0,
                "blocked_simulated_order_count": 0,
            },
            "post_decision_exposure_state": {
                "current_total_exposure": exposure_delta,
                "current_market_exposure_by_market_id": {
                    f"market-{decision_id}": exposure_delta
                }
                if order_allowed
                else {},
                "current_side_exposure_by_side": {
                    "DOWN": exposure_delta if guarded_side == "DOWN" else 0.0,
                    "UP": exposure_delta if guarded_side == "UP" else 0.0,
                },
                "executed_simulated_order_count": 1 if order_allowed else 0,
                "blocked_simulated_order_count": 0 if order_allowed else 1,
            },
            "exposure_delta": exposure_delta,
            "simulated_order_id": simulated_order_id,
            "source_score_mutated": False,
            "o_model_predicted_score_mutated": False,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    replay_report = {
        "o_v8_execution_simulated_order_replay_report_id": "replay-test",
        "simulated_decision_rows": [
            _replay_row(
                decision_id="1",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                side="UP",
                order_allowed=True,
                exposure_reasons=["execution_simulated_order_allowed"],
                simulated_order_id="sim-v8-o-000001",
            ),
            _replay_row(
                decision_id="2",
                action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                guarded_action="BUY_DOWN_SELL_BEFORE_CLOSE",
                side="DOWN",
                order_allowed=True,
                guard_reasons=["execution_hts_downgraded_to_same_side_sbc"],
                exposure_reasons=["execution_simulated_order_allowed"],
                simulated_order_id="sim-v8-o-000002",
            ),
            _replay_row(
                decision_id="3",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                side="UP",
                order_allowed=False,
                blocking_reasons=["execution_p_up_side_disagreement"],
                p_up_disagreement=True,
            ),
            _replay_row(
                decision_id="4",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                side="DOWN",
                order_allowed=False,
                blocking_reasons=["execution_duplicate_market_side_position"],
                exposure_reasons=["execution_simulated_order_blocked"],
            ),
            _replay_row(
                decision_id="5",
                action="BUY_UP_HOLD_TO_SETTLEMENT",
                side="UP",
                order_allowed=False,
                blocking_reasons=["execution_time_to_close_unsafe"],
                guard_reasons=["execution_hts_guard_failed"],
                time_to_close=20.0,
            ),
        ],
    }

    report = _v8_execution_allowed_order_quality_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
    )

    payload = dict(report)
    report_id = payload.pop("o_v8_execution_allowed_order_quality_report_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["schema_version"] == O_V8_EXECUTION_ALLOWED_ORDER_QUALITY_SCHEMA_VERSION
    assert report["report_type"] == "o_v8_execution_allowed_order_quality"
    assert report["allowed_order_count"] == 2
    assert report["blocked_decision_count"] == 3
    assert report["allowed_order_origin_distribution"] == {
        "hts_to_sbc_downgrade": 1,
        "original_selected_action": 1,
    }
    assert report["allowed_order_side_distribution"] == {"DOWN": 1, "UP": 1}
    assert report["allowed_order_p_up_agreement_distribution"] == {"p_up_agrees": 2}
    assert report["allowed_order_metric_summary"]["proposed_order_size"][
        "count"
    ] == 2
    assert report["residual_blocker_summary"] == {
        "duplicate_market_side_position_count": 1,
        "exposure_limit_blocked_decision_count": 1,
        "hts_guard_failed_count": 1,
        "p_up_disagreement_blocked_decision_count": 1,
        "time_to_close_unsafe_count": 1,
    }
    assert report["deterministic_recommendation_counts"] == {
        "keep_blocked": 1,
        "needs_exposure_policy_review": 1,
        "needs_p_up_action_rank_review": 1,
        "needs_time_to_close_policy_review": 1,
    }
    assert report["primary_deterministic_recommendation_counts"] == {
        "needs_exposure_policy_review": 1,
        "needs_p_up_action_rank_review": 1,
        "needs_time_to_close_policy_review": 1,
    }
    assert report["uses_validation_outcomes_for_tuning"] is False
    assert report["thresholds_tuned"] is False
    assert report["uses_realized_pnl_or_labels_for_analysis"] is False
    assert report["mutates_o_model_predicted_score"] is False
    assert report["mutates_source_ranking_scores"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False


def test_o_v8_execution_policy_readiness_passes_diagnostic_without_unlock(
    tmp_path: Path,
) -> None:
    def _allowed_row(index: int) -> dict[str, Any]:
        side = "UP" if index % 2 else "DOWN"
        action = f"BUY_{side}_SELL_BEFORE_CLOSE"
        post_total = index * 0.1
        return {
            "decision_group_id": f"source|market-{index}|{index}",
            "market_id": f"market-{index}",
            "decision_ts": index,
            "source_selected_action": action,
            "source_selected_family": "SELL_BEFORE_CLOSE",
            "source_selected_side": side,
            "source_model_score": 0.90,
            "source_raw_model_score": 0.80,
            "p_up": 0.70 if side == "UP" else 0.30,
            "p_down": 0.30 if side == "UP" else 0.70,
            "p_up_action_disagreement": False,
            "microstructure_snapshot": {
                "book_staleness_ms": 500.0,
                "spread_bps": 200.0,
                "queue_fill_proxy": 0.90,
                "time_to_close_seconds": 180.0,
            },
            "runtime_field_backfill_provenance_violations": [],
            "execution_guarded_action": action,
            "execution_guarded_family": "SELL_BEFORE_CLOSE",
            "execution_guarded_side": side,
            "execution_guarded_score": 0.89,
            "execution_score_penalties": {},
            "order_allowed": True,
            "proposed_order_size": 0.1,
            "uncapped_proposed_order_size": 0.1,
            "sizing_reason_codes": ["execution_base_size_applied"],
            "exposure_reason_codes": ["execution_simulated_order_allowed"],
            "execution_guard_reason_codes": [],
            "execution_blocking_reason_codes": [],
            "missing_runtime_field_codes": [],
            "pre_decision_exposure_state": {
                "current_total_exposure": post_total - 0.1,
                "current_market_exposure_by_market_id": {},
                "current_side_exposure_by_side": {"DOWN": 0.0, "UP": 0.0},
                "executed_simulated_order_count": index - 1,
                "blocked_simulated_order_count": 0,
            },
            "post_decision_exposure_state": {
                "current_total_exposure": post_total,
                "current_market_exposure_by_market_id": {f"market-{index}": 0.1},
                "current_side_exposure_by_side": {"DOWN": 0.1, "UP": 0.1},
                "executed_simulated_order_count": index,
                "blocked_simulated_order_count": 0,
            },
            "exposure_delta": 0.1,
            "simulated_order_id": f"sim-v8-o-{index:06d}",
            "source_score_mutated": False,
            "o_model_predicted_score_mutated": False,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    replay_report = {
        "o_v8_execution_simulated_order_replay_report_id": "replay-test",
        "simulated_decision_rows": [_allowed_row(index) for index in range(1, 6)],
    }
    allowed_quality = _v8_execution_allowed_order_quality_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
    )

    report = _v8_execution_policy_readiness_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
        allowed_order_quality_report=allowed_quality,
    )

    payload = dict(report)
    report_id = payload.pop("o_v8_execution_policy_readiness_report_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["schema_version"] == O_V8_EXECUTION_POLICY_READINESS_SCHEMA_VERSION
    assert report["execution_policy_readiness_diagnostic_passed"] is True
    assert report["execution_policy_readiness_blocking_reason_codes"] == []
    assert all(
        check["passed"] is True
        for check in report["execution_policy_readiness_required_checks"].values()
    )
    assert report["future_explicit_execution_handoff_gate_required"] is True
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False


def test_o_v8_execution_policy_readiness_fails_closed_on_quality_gaps(
    tmp_path: Path,
) -> None:
    def _allowed_row(index: int) -> dict[str, Any]:
        return {
            "decision_group_id": f"source|market-{index}|{index}",
            "market_id": f"market-{index}",
            "decision_ts": index,
            "source_selected_action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "source_selected_family": "HOLD_TO_SETTLEMENT",
            "source_selected_side": "UP",
            "source_model_score": 0.90,
            "source_raw_model_score": 0.80,
            "p_up": 0.30 if index == 1 else 0.70,
            "p_down": 0.70 if index == 1 else 0.30,
            "p_up_action_disagreement": index == 1,
            "microstructure_snapshot": {
                "book_staleness_ms": 2500.0 if index == 2 else 500.0,
                "spread_bps": 1200.0 if index == 2 else 200.0,
                "queue_fill_proxy": 0.40 if index == 2 else 0.90,
                "time_to_close_seconds": 30.0 if index == 2 else 180.0,
            },
            "runtime_field_backfill_provenance_violations": [
                {"field": "microstructure_snapshot.time_to_close_seconds"}
            ]
            if index == 3
            else [],
            "execution_guarded_action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "execution_guarded_family": "HOLD_TO_SETTLEMENT",
            "execution_guarded_side": "UP",
            "execution_guarded_score": 0.89,
            "execution_score_penalties": {},
            "order_allowed": True,
            "proposed_order_size": 0.3 if index == 4 else 0.1,
            "uncapped_proposed_order_size": 0.3 if index == 4 else 0.1,
            "sizing_reason_codes": ["execution_base_size_applied"],
            "exposure_reason_codes": ["execution_simulated_order_allowed"],
            "execution_guard_reason_codes": [],
            "execution_blocking_reason_codes": [],
            "missing_runtime_field_codes": ["missing_selected_side"]
            if index == 3
            else [],
            "pre_decision_exposure_state": {
                "current_total_exposure": 0.0,
                "current_market_exposure_by_market_id": {},
                "current_side_exposure_by_side": {"UP": 0.0},
                "executed_simulated_order_count": 0,
                "blocked_simulated_order_count": 0,
            },
            "post_decision_exposure_state": {
                "current_total_exposure": 1.2 if index == 4 else 0.1,
                "current_market_exposure_by_market_id": {f"market-{index}": 0.3},
                "current_side_exposure_by_side": {"UP": 1.2}
                if index == 4
                else {"UP": 0.1},
                "executed_simulated_order_count": index,
                "blocked_simulated_order_count": 0,
            },
            "exposure_delta": 0.3 if index == 4 else 0.1,
            "simulated_order_id": f"sim-v8-o-{index:06d}",
            "source_score_mutated": False,
            "o_model_predicted_score_mutated": False,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    replay_report = {
        "o_v8_execution_simulated_order_replay_report_id": "replay-test",
        "simulated_decision_rows": [_allowed_row(index) for index in range(1, 5)],
    }
    allowed_quality = _v8_execution_allowed_order_quality_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
    )

    report = _v8_execution_policy_readiness_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
        allowed_order_quality_report=allowed_quality,
    )

    assert report["execution_policy_readiness_diagnostic_passed"] is False
    assert set(report["execution_policy_readiness_blocking_reason_codes"]) >= {
        "execution_policy_allowed_order_exposure_limit_failed",
        "execution_policy_allowed_order_microstructure_quality_failed",
        "execution_policy_allowed_order_p_up_disagreement_present",
        "execution_policy_min_allowed_order_count_not_met",
        "execution_policy_provenance_violations_present",
        "execution_policy_runtime_missing_fields_present",
    }
    assert report["execution_policy_readiness_required_checks"][
        "no_paper_live_write_or_capital_flags"
    ]["passed"] is True
    assert report["future_explicit_execution_handoff_gate_required"] is True
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False


def test_o_v8_execution_handoff_gate_passes_diagnostic_but_stays_closed(
    tmp_path: Path,
) -> None:
    def _allowed_row(index: int) -> dict[str, Any]:
        side = "UP" if index % 2 else "DOWN"
        action = f"BUY_{side}_SELL_BEFORE_CLOSE"
        return {
            "decision_group_id": f"source|market-{index}|{index}",
            "market_id": f"market-{index}",
            "decision_ts": index,
            "source_selected_action": action,
            "source_selected_family": "SELL_BEFORE_CLOSE",
            "source_selected_side": side,
            "source_model_score": 0.90,
            "source_raw_model_score": 0.80,
            "p_up": 0.70 if side == "UP" else 0.30,
            "p_down": 0.30 if side == "UP" else 0.70,
            "p_up_action_disagreement": False,
            "microstructure_snapshot": {
                "book_staleness_ms": 500.0,
                "spread_bps": 200.0,
                "queue_fill_proxy": 0.90,
                "time_to_close_seconds": 180.0,
            },
            "runtime_field_backfill_provenance_violations": [],
            "execution_guarded_action": action,
            "execution_guarded_family": "SELL_BEFORE_CLOSE",
            "execution_guarded_side": side,
            "execution_guarded_score": 0.89,
            "execution_score_penalties": {},
            "order_allowed": True,
            "proposed_order_size": 0.1,
            "uncapped_proposed_order_size": 0.1,
            "sizing_reason_codes": ["execution_base_size_applied"],
            "exposure_reason_codes": ["execution_simulated_order_allowed"],
            "execution_guard_reason_codes": [],
            "execution_blocking_reason_codes": [],
            "missing_runtime_field_codes": [],
            "pre_decision_exposure_state": {
                "current_total_exposure": (index - 1) * 0.1,
                "current_market_exposure_by_market_id": {},
                "current_side_exposure_by_side": {"DOWN": 0.0, "UP": 0.0},
                "executed_simulated_order_count": index - 1,
                "blocked_simulated_order_count": 0,
            },
            "post_decision_exposure_state": {
                "current_total_exposure": index * 0.1,
                "current_market_exposure_by_market_id": {f"market-{index}": 0.1},
                "current_side_exposure_by_side": {"DOWN": 0.1, "UP": 0.1},
                "executed_simulated_order_count": index,
                "blocked_simulated_order_count": 0,
            },
            "exposure_delta": 0.1,
            "simulated_order_id": f"sim-v8-o-{index:06d}",
            "source_score_mutated": False,
            "o_model_predicted_score_mutated": False,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    replay_report = {
        "o_v8_execution_simulated_order_replay_report_id": "replay-test",
        "simulated_decision_rows": [_allowed_row(index) for index in range(1, 6)],
        "final_exposure": {"runtime_state_validation_passed": True},
        "runtime_risk_control_validation_passed": True,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    allowed_quality = _v8_execution_allowed_order_quality_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
    )
    policy_readiness = _v8_execution_policy_readiness_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
        allowed_order_quality_report=allowed_quality,
    )
    block_analysis = _v8_execution_guard_block_analysis_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report={"o_v8_action_rank_handoff_report_id": "handoff-test"},
        execution_guard_report={"o_v8_execution_risk_guard_report_id": "guard-test"},
        runtime_state_report={"o_v8_execution_runtime_state_report_id": "state-test"},
        simulated_order_replay_report=replay_report,
    )
    field_coverage = _v8_execution_runtime_field_coverage_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report={"o_v8_action_rank_handoff_report_id": "handoff-test"},
        execution_guard_report={"o_v8_execution_risk_guard_report_id": "guard-test"},
        runtime_state_report={"o_v8_execution_runtime_state_report_id": "state-test"},
        simulated_order_replay_report=replay_report,
        block_analysis_report=block_analysis,
    )

    report = _v8_execution_handoff_gate_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        policy_readiness_report=policy_readiness,
        allowed_order_quality_report=allowed_quality,
        simulated_order_replay_report=replay_report,
        runtime_field_coverage_report=field_coverage,
        guard_block_analysis_report=block_analysis,
    )

    payload = dict(report)
    report_id = payload.pop("o_v8_execution_handoff_gate_report_id")
    assert canonical_json_sha256(payload) == report_id
    assert report["schema_version"] == O_V8_EXECUTION_HANDOFF_GATE_SCHEMA_VERSION
    assert report["explicit_execution_handoff_gate_passed"] is True
    assert report["explicit_execution_handoff_blocking_reason_codes"] == []
    assert all(
        check["passed"] is True
        for check in report["explicit_execution_handoff_required_checks"].values()
    )
    assert report["explicit_execution_handoff_gate_mode"] == (
        "diagnostic_only_fail_closed"
    )
    assert report["future_unseen_holdout_required"] is True
    assert report["future_paper_candidate_gate_required"] is True
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False


def test_o_v8_execution_handoff_gate_fails_when_readiness_or_runtime_fails(
    tmp_path: Path,
) -> None:
    def _row(index: int) -> dict[str, Any]:
        return {
            "decision_group_id": f"source|market-{index}|{index}",
            "market_id": f"market-{index}",
            "decision_ts": index,
            "source_selected_action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "source_selected_family": "HOLD_TO_SETTLEMENT",
            "source_selected_side": "UP",
            "source_model_score": 0.90,
            "source_raw_model_score": 0.80,
            "p_up": 0.30 if index == 1 else 0.70,
            "p_down": 0.70 if index == 1 else 0.30,
            "p_up_action_disagreement": index == 1,
            "microstructure_snapshot": {
                "book_staleness_ms": 2500.0 if index == 2 else 500.0,
                "spread_bps": 1200.0 if index == 2 else 200.0,
                "queue_fill_proxy": 0.40 if index == 2 else 0.90,
                "time_to_close_seconds": 30.0 if index == 2 else 180.0,
            },
            "runtime_field_backfill_provenance_violations": [
                {"field": "microstructure_snapshot.time_to_close_seconds"}
            ]
            if index == 3
            else [],
            "execution_guarded_action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "execution_guarded_family": "HOLD_TO_SETTLEMENT",
            "execution_guarded_side": "UP",
            "execution_guarded_score": 0.89,
            "execution_score_penalties": {},
            "order_allowed": True,
            "proposed_order_size": 0.3 if index == 4 else 0.1,
            "uncapped_proposed_order_size": 0.3 if index == 4 else 0.1,
            "sizing_reason_codes": ["execution_base_size_applied"],
            "exposure_reason_codes": ["execution_simulated_order_allowed"],
            "execution_guard_reason_codes": [],
            "execution_blocking_reason_codes": [],
            "missing_runtime_field_codes": ["missing_selected_side"]
            if index == 3
            else [],
            "pre_decision_exposure_state": {
                "current_total_exposure": 0.0,
                "current_market_exposure_by_market_id": {},
                "current_side_exposure_by_side": {"UP": 0.0},
                "executed_simulated_order_count": 0,
                "blocked_simulated_order_count": 0,
            },
            "post_decision_exposure_state": {
                "current_total_exposure": 1.2 if index == 4 else 0.1,
                "current_market_exposure_by_market_id": {f"market-{index}": 0.3},
                "current_side_exposure_by_side": {"UP": 1.2}
                if index == 4
                else {"UP": 0.1},
                "executed_simulated_order_count": index,
                "blocked_simulated_order_count": 0,
            },
            "exposure_delta": 0.3 if index == 4 else 0.1,
            "simulated_order_id": f"sim-v8-o-{index:06d}",
            "source_score_mutated": index == 4,
            "o_model_predicted_score_mutated": False,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    replay_report = {
        "o_v8_execution_simulated_order_replay_report_id": "replay-test",
        "simulated_decision_rows": [_row(index) for index in range(1, 5)],
        "final_exposure": {"runtime_state_validation_passed": False},
        "runtime_risk_control_validation_passed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    allowed_quality = _v8_execution_allowed_order_quality_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
    )
    policy_readiness = _v8_execution_policy_readiness_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
        allowed_order_quality_report=allowed_quality,
    )
    block_analysis = _v8_execution_guard_block_analysis_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report={"o_v8_action_rank_handoff_report_id": "handoff-test"},
        execution_guard_report={"o_v8_execution_risk_guard_report_id": "guard-test"},
        runtime_state_report={"o_v8_execution_runtime_state_report_id": "state-test"},
        simulated_order_replay_report=replay_report,
    )
    field_coverage = _v8_execution_runtime_field_coverage_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report={"o_v8_action_rank_handoff_report_id": "handoff-test"},
        execution_guard_report={"o_v8_execution_risk_guard_report_id": "guard-test"},
        runtime_state_report={"o_v8_execution_runtime_state_report_id": "state-test"},
        simulated_order_replay_report=replay_report,
        block_analysis_report=block_analysis,
    )

    report = _v8_execution_handoff_gate_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        policy_readiness_report=policy_readiness,
        allowed_order_quality_report=allowed_quality,
        simulated_order_replay_report=replay_report,
        runtime_field_coverage_report=field_coverage,
        guard_block_analysis_report=block_analysis,
    )

    assert report["explicit_execution_handoff_gate_passed"] is False
    assert set(report["explicit_execution_handoff_blocking_reason_codes"]) >= {
        "execution_handoff_allowed_order_exposure_limit_failed",
        "execution_handoff_allowed_order_microstructure_quality_failed",
        "execution_handoff_allowed_order_p_up_disagreement_present",
        "execution_handoff_min_allowed_order_count_not_met",
        "execution_handoff_policy_readiness_not_passed",
        "execution_handoff_provenance_violations_present",
        "execution_handoff_runtime_missing_fields_present",
        "execution_handoff_runtime_risk_control_validation_failed",
        "execution_handoff_runtime_state_validation_failed",
        "execution_handoff_source_score_mutation_detected",
    }
    assert report["explicit_execution_handoff_required_checks"][
        "no_paper_live_write_or_capital_flags"
    ]["passed"] is True
    assert report["future_unseen_holdout_required"] is True
    assert report["future_paper_candidate_gate_required"] is True
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False


def _build_v8_future_gate_design_fixture(
    tmp_path: Path,
    *,
    fail_closed: bool = False,
) -> dict[str, Any]:
    def _row(index: int) -> dict[str, Any]:
        side = "UP" if index % 2 else "DOWN"
        action = f"BUY_{side}_SELL_BEFORE_CLOSE"
        return {
            "decision_group_id": f"source|market-{index}|{index}",
            "market_id": f"market-{index}",
            "decision_ts": index,
            "source_selected_action": action,
            "source_selected_family": "SELL_BEFORE_CLOSE",
            "source_selected_side": side,
            "source_model_score": 0.90,
            "source_raw_model_score": 0.80,
            "p_up": 0.70 if side == "UP" else 0.30,
            "p_down": 0.30 if side == "UP" else 0.70,
            "p_up_action_disagreement": False,
            "microstructure_snapshot": {
                "book_staleness_ms": 500.0,
                "spread_bps": 200.0,
                "queue_fill_proxy": 0.90,
                "time_to_close_seconds": 180.0,
            },
            "runtime_field_backfill_provenance_violations": [],
            "execution_guarded_action": action,
            "execution_guarded_family": "SELL_BEFORE_CLOSE",
            "execution_guarded_side": side,
            "execution_guarded_score": 0.89,
            "execution_score_penalties": {},
            "order_allowed": True,
            "proposed_order_size": 0.1,
            "uncapped_proposed_order_size": 0.1,
            "sizing_reason_codes": ["execution_base_size_applied"],
            "exposure_reason_codes": ["execution_simulated_order_allowed"],
            "execution_guard_reason_codes": [],
            "execution_blocking_reason_codes": [],
            "missing_runtime_field_codes": [],
            "pre_decision_exposure_state": {
                "current_total_exposure": (index - 1) * 0.1,
                "current_market_exposure_by_market_id": {},
                "current_side_exposure_by_side": {"DOWN": 0.0, "UP": 0.0},
                "executed_simulated_order_count": index - 1,
                "blocked_simulated_order_count": 0,
            },
            "post_decision_exposure_state": {
                "current_total_exposure": index * 0.1,
                "current_market_exposure_by_market_id": {f"market-{index}": 0.1},
                "current_side_exposure_by_side": {"DOWN": 0.1, "UP": 0.1},
                "executed_simulated_order_count": index,
                "blocked_simulated_order_count": 0,
            },
            "exposure_delta": 0.1,
            "runtime_field_applied_backfill_rows": [
                {
                    "runtime_field_name": "microstructure_snapshot.time_to_close_seconds",
                    "deterministic_rule_id": (
                        "backfill_time_to_close_from_decision_time_feature_or_market_schedule"
                    ),
                    "application_type": "decision_time_data_join_backfill",
                    "provenance_valid": True,
                }
            ],
            "runtime_field_backfill_rules_applied": True,
            "runtime_field_backfill_rule_counts": {
                "backfill_time_to_close_from_decision_time_feature_or_market_schedule": 1
            },
            "runtime_field_backfill_provenance_valid": True,
            "simulated_order_id": f"sim-v8-o-{index:06d}",
            "source_score_mutated": fail_closed and index == 5,
            "o_model_predicted_score_mutated": False,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    replay_report = {
        "o_v8_execution_simulated_order_replay_report_id": "replay-test",
        "report_type": "o_v8_execution_simulated_order_replay",
        "simulated_decision_rows": [_row(index) for index in range(1, 6)],
        "final_exposure": {"runtime_state_validation_passed": True},
        "runtime_risk_control_validation_passed": True,
        "deterministic_replay_hash": "replay-hash",
        "applied_runtime_field_backfill_rule_counts": {
            "backfill_time_to_close_from_decision_time_feature_or_market_schedule": 5
        },
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    allowed_quality = _v8_execution_allowed_order_quality_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
    )
    policy_readiness = _v8_execution_policy_readiness_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        simulated_order_replay_report=replay_report,
        allowed_order_quality_report=allowed_quality,
    )
    block_analysis = _v8_execution_guard_block_analysis_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report={"o_v8_action_rank_handoff_report_id": "handoff-test"},
        execution_guard_report={"o_v8_execution_risk_guard_report_id": "guard-test"},
        runtime_state_report={"o_v8_execution_runtime_state_report_id": "state-test"},
        simulated_order_replay_report=replay_report,
    )
    field_coverage = _v8_execution_runtime_field_coverage_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report={"o_v8_action_rank_handoff_report_id": "handoff-test"},
        execution_guard_report={"o_v8_execution_risk_guard_report_id": "guard-test"},
        runtime_state_report={"o_v8_execution_runtime_state_report_id": "state-test"},
        simulated_order_replay_report=replay_report,
        block_analysis_report=block_analysis,
    )
    handoff_gate = _v8_execution_handoff_gate_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        policy_readiness_report=policy_readiness,
        allowed_order_quality_report=allowed_quality,
        simulated_order_replay_report=replay_report,
        runtime_field_coverage_report=field_coverage,
        guard_block_analysis_report=block_analysis,
    )
    action_rank_handoff = {
        "o_v8_action_rank_handoff_report_id": "action-rank-test",
        "report_type": "o_v8_action_rank_handoff",
        "model_sha256": None if fail_closed else "a" * 64,
        "split_hash": "b" * 64,
        "feature_schema_hash": "c" * 64,
        "handoff_contract_hash": "d" * 64,
        "ranking_score_source": "model_predicted_score",
        "model_layer_regret_risk_selection_enabled": False,
        "strict_source_gate_remains_failed": True,
        "uses_validation_outcomes_for_tuning": False,
        "forbidden_outcome_fields_used": [],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    execution_guard = {
        "o_v8_execution_risk_guard_report_id": "guard-test",
        "report_type": "o_v8_execution_risk_guard",
        "execution_guard_config_hash": "guard-config-hash",
        "model_layer_regret_risk_selection_enabled": fail_closed,
        "trains_regret_model": False,
        "trains_risk_head": False,
        "uses_validation_realized_outcomes_for_guard_tuning": fail_closed,
        "uses_replay_regret_labels_for_guard_tuning": False,
        "forbidden_outcome_fields_used": [],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    block_analysis["execution_guard_config_hash"] = "other-guard-hash" if fail_closed else "guard-config-hash"
    if fail_closed:
        block_analysis["safe_order_discovery_uses_realized_pnl"] = True
    holdout_plan = _v8_future_unseen_holdout_plan_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        action_rank_handoff_report=action_rank_handoff,
        execution_guard_report=execution_guard,
        simulated_order_replay_report=replay_report,
        allowed_order_quality_report=allowed_quality,
        policy_readiness_report=policy_readiness,
        handoff_gate_report=handoff_gate,
        runtime_field_coverage_report=field_coverage,
        guard_block_analysis_report=block_analysis,
    )
    paper_gate = _v8_paper_candidate_gate_design_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        action_rank_handoff_report=action_rank_handoff,
        execution_guard_report=execution_guard,
        simulated_order_replay_report=replay_report,
        allowed_order_quality_report=allowed_quality,
        policy_readiness_report=policy_readiness,
        handoff_gate_report=handoff_gate,
        runtime_field_coverage_report=field_coverage,
        guard_block_analysis_report=block_analysis,
        holdout_plan_report=holdout_plan,
    )
    collection_plan = _v8_future_unseen_holdout_collection_plan_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        action_rank_handoff_report=action_rank_handoff,
        execution_guard_report=execution_guard,
        simulated_order_replay_report=replay_report,
        allowed_order_quality_report=allowed_quality,
        policy_readiness_report=policy_readiness,
        handoff_gate_report=handoff_gate,
        runtime_field_coverage_report=field_coverage,
        guard_block_analysis_report=block_analysis,
        holdout_plan_report=holdout_plan,
        paper_candidate_gate_design_report=paper_gate,
    )
    return {
        "m2_report_path": m2_report_path,
        "m2_report": m2_report,
        "action_rank_handoff": action_rank_handoff,
        "execution_guard": execution_guard,
        "simulated_order_replay": replay_report,
        "allowed_quality": allowed_quality,
        "policy_readiness": policy_readiness,
        "runtime_field_coverage": field_coverage,
        "block_analysis": block_analysis,
        "handoff_gate": handoff_gate,
        "holdout_plan": holdout_plan,
        "paper_gate": paper_gate,
        "collection_plan": collection_plan,
    }


def test_o_v8_future_unseen_holdout_and_paper_gate_design_ready_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path)
    holdout_plan = fixture["holdout_plan"]
    paper_gate = fixture["paper_gate"]

    holdout_payload = dict(holdout_plan)
    holdout_id = holdout_payload.pop("o_v8_future_unseen_holdout_plan_report_id")
    assert canonical_json_sha256(holdout_payload) == holdout_id
    assert (
        holdout_plan["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_PLAN_SCHEMA_VERSION
    )
    assert holdout_plan["future_unseen_holdout_plan_ready"] is True
    assert holdout_plan["future_unseen_holdout_blocking_reason_codes"] == []
    assert all(
        check["passed"] is True
        for check in holdout_plan["future_unseen_holdout_required_checks"].values()
    )
    assert holdout_plan["paper_candidate_allowed"] is False
    assert holdout_plan["v8_execution_handoff_allowed"] is False
    assert holdout_plan["#146_start_allowed"] is False
    assert holdout_plan["#134_resume_allowed"] is False

    paper_payload = dict(paper_gate)
    paper_id = paper_payload.pop("o_v8_paper_candidate_gate_design_report_id")
    assert canonical_json_sha256(paper_payload) == paper_id
    assert (
        paper_gate["schema_version"]
        == O_V8_PAPER_CANDIDATE_GATE_DESIGN_SCHEMA_VERSION
    )
    assert paper_gate["paper_candidate_gate_design_ready"] is True
    assert paper_gate["paper_candidate_gate_blocking_reason_codes"] == []
    assert all(
        check["passed"] is True
        for check in paper_gate["paper_candidate_required_checks"].values()
    )
    assert paper_gate["paper_candidate_allowed"] is False
    assert paper_gate["v8_execution_handoff_allowed"] is False
    assert paper_gate["source_model_candidate_eligible"] is False
    assert paper_gate["freeze_ready"] is False
    assert paper_gate["promotion_evidence_eligible"] is False
    assert paper_gate["#146_start_allowed"] is False
    assert paper_gate["#134_resume_allowed"] is False
    assert paper_gate["paper_only"] is True
    assert paper_gate["capital_at_risk"] is False
    assert paper_gate["polymarket_write_enabled"] is False
    assert paper_gate["wallet_signing_enabled"] is False


def test_o_v8_future_unseen_holdout_and_paper_gate_design_fail_closed_on_bad_inputs(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path, fail_closed=True)
    holdout_plan = fixture["holdout_plan"]
    paper_gate = fixture["paper_gate"]

    assert holdout_plan["future_unseen_holdout_plan_ready"] is False
    assert set(holdout_plan["future_unseen_holdout_blocking_reason_codes"]) >= {
        "future_holdout_execution_guard_config_not_frozen",
        "future_holdout_input_report_forbidden_outcome_usage_detected",
        "future_holdout_o_action_rank_config_not_frozen",
    }
    assert holdout_plan["paper_candidate_allowed"] is False
    assert holdout_plan["v8_execution_handoff_allowed"] is False
    assert holdout_plan["#146_start_allowed"] is False
    assert holdout_plan["#134_resume_allowed"] is False

    assert paper_gate["paper_candidate_gate_design_ready"] is False
    assert set(paper_gate["paper_candidate_gate_blocking_reason_codes"]) >= {
        "paper_candidate_forbidden_outcome_usage_detected",
        "paper_candidate_future_unseen_holdout_plan_not_ready",
        "paper_candidate_model_layer_regret_risk_selection_enabled",
        "paper_candidate_source_score_mutation_detected",
    }
    assert paper_gate["paper_candidate_allowed"] is False
    assert paper_gate["v8_execution_handoff_allowed"] is False
    assert paper_gate["source_model_candidate_eligible"] is False
    assert paper_gate["freeze_ready"] is False
    assert paper_gate["promotion_evidence_eligible"] is False
    assert paper_gate["#146_start_allowed"] is False
    assert paper_gate["#134_resume_allowed"] is False


def test_o_v8_future_unseen_holdout_collection_plan_ready_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path)
    report = fixture["collection_plan"]

    payload = dict(report)
    report_id = payload.pop(
        "o_v8_future_unseen_holdout_collection_plan_report_id"
    )
    assert canonical_json_sha256(payload) == report_id
    assert (
        report["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_COLLECTION_PLAN_SCHEMA_VERSION
    )
    assert report["report_type"] == "o_v8_future_unseen_holdout_collection_plan"
    assert report["diagnostic_only"] is True
    assert report["simulation_only"] is True
    assert report["future_unseen_holdout_collection_plan_ready"] is True
    assert report["future_unseen_holdout_collection_blocking_reason_codes"] == []
    assert all(
        check["passed"] is True
        for check in report[
            "future_unseen_holdout_collection_required_checks"
        ].values()
    )
    assert set(report["frozen_current_v8_o_config_references"]) == {
        "action_rank_config",
        "execution_guard_config",
        "runtime_field_cleanup_rules",
        "simulated_ledger_rules",
        "handoff_gate_design",
        "future_holdout_and_paper_gate_design",
    }
    assert report["holdout_window_requirements"]["unseen_future_dates_only"] is True
    assert report["holdout_window_requirements"][
        "no_overlap_with_validation_shadow_or_replay_data"
    ] is True
    assert report["collection_status"] == "not_started"
    assert report["future_outcome_evaluation_generated"] is False
    assert report["future_outcome_evaluation_artifacts_generated"] == []
    assert report["paper_candidate_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False


def test_o_v8_future_unseen_holdout_collection_plan_fails_on_bad_inputs(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path, fail_closed=True)
    report = fixture["collection_plan"]

    assert report["future_unseen_holdout_collection_plan_ready"] is False
    assert set(report["future_unseen_holdout_collection_blocking_reason_codes"]) >= {
        "future_collection_action_rank_config_not_frozen",
        "future_collection_execution_guard_config_not_frozen",
        "future_collection_forbidden_outcome_usage_detected",
    }
    assert report["collection_status"] == "not_started"
    assert report["future_outcome_evaluation_generated"] is False
    assert report["paper_candidate_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False


def _future_holdout_raw_row(index: int, *, decision_ts: int) -> dict[str, Any]:
    side = "UP" if index % 2 else "DOWN"
    action = f"BUY_{side}_SELL_BEFORE_CLOSE"
    p_up = 0.70 if side == "UP" else 0.30
    p_down = 0.30 if side == "UP" else 0.70
    microstructure = {
        "book_staleness_ms": 500.0,
        "spread_bps": 200.0,
        "queue_fill_proxy": 0.90,
        "time_to_close_seconds": 180.0,
    }
    return {
        "decision_group_id": f"future-source|future-market-{index}|{decision_ts}",
        "market_id": f"future-market-{index}",
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": side,
        "selected_action_family": "SELL_BEFORE_CLOSE",
        "full_5_action_ranking": [
            {
                "rank": 1,
                "selected_action": action,
                "selected_side": side,
                "selected_action_family": "SELL_BEFORE_CLOSE",
                "corrected_model_score": 0.90,
                "raw_model_score": 0.85,
                "high_score_flag": True,
                "p_up_action_disagreement": False,
                "microstructure_snapshot": microstructure,
            },
            {
                "rank": 2,
                "selected_action": "NO_TRADE",
                "selected_side": "NONE",
                "selected_action_family": "NO_TRADE",
                "corrected_model_score": 0.70,
                "raw_model_score": 0.70,
                "high_score_flag": False,
                "p_up_action_disagreement": False,
                "microstructure_snapshot": microstructure,
            },
        ],
        "corrected_model_score": 0.90,
        "raw_model_score": 0.85,
        "score_components": {"model_predicted_score": 0.90},
        "high_score_flag": True,
        "p_up": p_up,
        "p_down": p_down,
        "p_up_action_disagreement": False,
        "microstructure_snapshot": microstructure,
        "reference_price_to_beat_distance_at_decision": 12.5,
        "reference_price_feature_provenance": {
            "provenance_valid": True,
            "decision_ts": decision_ts,
            "max_input_ts": decision_ts - 1,
            "source_field_name": "future_holdout_reference_mid",
        },
        "decision_time_feature_max_input_ts": decision_ts - 1,
        "runtime_exposure_state": {},
        "configured_execution_limits": {
            "max_order_size": 0.20,
            "max_total_exposure": 1.00,
        },
        "source_score_mutated": False,
        "o_model_predicted_score_mutated": False,
    }


def _write_future_holdout_raw_manifest(
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]],
    window_start_ts: int,
    input_freeze_created_ts: int = 8_000,
    raw_manifest_created_ts: int = 9_000,
) -> Path:
    path = tmp_path / "future_holdout_raw_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "test-future-holdout-raw-input-v1",
                "market_family": "btc_updown_5m",
                "window_start_ts": window_start_ts,
                "window_end_ts": window_start_ts + 300_000,
                "input_freeze_created_ts": input_freeze_created_ts,
                "raw_manifest_created_ts": raw_manifest_created_ts,
                "holdout_decision_rows": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _contains_forbidden_holdout_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_HOLDOUT_ROW_FIELDS
            or _contains_forbidden_holdout_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_holdout_field(child) for child in value)
    return False


def test_o_v8_future_holdout_generator_prepares_runtime_quality_fields() -> None:
    guard_config = _v8_execution_guard_config()
    initial_state = _v8_initial_runtime_state(guard_config)
    entry = {
        "decision_group_id": "future-source|market|10000",
        "market_id": "market",
        "decision_ts": 10_000,
        "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
        "selected_side": "UP",
        "selected_action_family": "SELL_BEFORE_CLOSE",
        "corrected_model_score": 0.90,
        "raw_model_score": 0.85,
        "score_components": {"model_predicted_score": 0.90},
        "high_score_flag": True,
        "p_up": 0.70,
        "p_down": 0.30,
        "p_up_action_disagreement": False,
        "microstructure_snapshot": {
            "book_staleness_ms": 500.0,
            "spread_bps": 200.0,
            "queue_fill_proxy": 0.90,
            "time_to_close_seconds": None,
        },
        "reference_price_feature_provenance": {
            "provenance_valid": True,
            "decision_ts": 10_000,
            "max_input_ts": 9_999,
            "source_field_name": "future_holdout_reference_mid",
        },
        "runtime_field_backfill_sources": {
            "microstructure_snapshot.time_to_close_seconds": {
                "field": "microstructure_snapshot.time_to_close_seconds",
                "value": 180.0,
                "source_field_name": "polymarket_feature_rows.features.time_to_close_seconds",
                "source_timestamp": 9_999,
                "max_input_ts": 9_999,
                "decision_ts": 10_000,
                "deterministic_rule_id": (
                    "backfill_time_to_close_from_decision_time_feature_or_market_schedule"
                ),
                "provenance_valid": True,
                "reason_codes": ["decision_time_time_to_close_source_available"],
            }
        },
        "full_5_action_ranking": [
            {
                "selected_action": "NO_TRADE",
                "realized_trade_pnl": 1.0,
                "oracle_action": "BUY_UP_SELL_BEFORE_CLOSE",
            }
        ],
        "oracle_executable_best_action": "BUY_UP_SELL_BEFORE_CLOSE",
        "realized_replay_return_report_only": 1.0,
        "future_return": 1.0,
        "settlement_outcome": "UP",
    }

    prepared = _prepare_runtime_quality_action_entry(
        entry,
        guard_config=guard_config,
        initial_runtime_state=initial_state,
    )

    assert {
        *O_REPORT_ONLY_EVALUATION_FIELDS,
        "oracle_action",
        "oracle_executable_best_action",
        "future_return",
        "settlement_outcome",
    } <= FORBIDDEN_HOLDOUT_ROW_FIELDS
    assert _contains_forbidden_holdout_field(prepared) is False
    assert prepared["microstructure_snapshot"]["time_to_close_seconds"] == 180.0
    assert prepared["runtime_exposure_state"]["runtime_state_validation_passed"] is True
    assert prepared["runtime_exposure_state"]["current_total_exposure"] == 0.0
    assert prepared["configured_execution_limits"]["max_total_exposure"] == 1.0
    assert prepared["runtime_input_quality_rules_applied"] is True
    assert _runtime_input_quality_rule_counts([prepared]) == {
        "attach_initial_simulated_runtime_exposure_state": 1,
        "backfill_time_to_close_from_decision_time_feature_or_market_schedule": 1,
    }


def test_o_v8_future_holdout_generator_filters_late_or_stale_rows() -> None:
    guard_config = _v8_execution_guard_config()
    initial_state = _v8_initial_runtime_state(guard_config)
    safe = _prepare_runtime_quality_action_entry(
        _future_holdout_raw_row(1, decision_ts=10_001),
        guard_config=guard_config,
        initial_runtime_state=initial_state,
    )
    late = _prepare_runtime_quality_action_entry(
        _future_holdout_raw_row(2, decision_ts=10_002),
        guard_config=guard_config,
        initial_runtime_state=initial_state,
    )
    late["microstructure_snapshot"]["time_to_close_seconds"] = 45.0
    stale = _prepare_runtime_quality_action_entry(
        _future_holdout_raw_row(3, decision_ts=10_003),
        guard_config=guard_config,
        initial_runtime_state=initial_state,
    )
    stale["microstructure_snapshot"]["book_staleness_ms"] = 3_500.0

    accepted, rejected = _filter_runtime_quality_rows(
        [safe, late, stale],
        min_selected_time_to_close_seconds=120.0,
        max_selected_book_staleness_ms=2_000.0,
    )

    assert [row["decision_group_id"] for row in accepted] == [
        safe["decision_group_id"]
    ]
    rejected_by_id = {row["decision_group_id"]: row for row in rejected}
    assert rejected_by_id[late["decision_group_id"]][
        "runtime_input_quality_reason_codes"
    ] == ["runtime_quality_time_to_close_below_execution_threshold"]
    assert rejected_by_id[stale["decision_group_id"]][
        "runtime_input_quality_reason_codes"
    ] == ["runtime_quality_book_stale"]


def test_o_v8_future_holdout_generator_selects_diversified_markets() -> None:
    guard_config = _v8_execution_guard_config()
    initial_state = _v8_initial_runtime_state(guard_config)

    def _prepared(index: int, *, market_id: str, staleness: float) -> dict[str, Any]:
        row = _future_holdout_raw_row(index, decision_ts=10_000 + index)
        row["market_id"] = market_id
        row["decision_group_id"] = f"future-source|{market_id}|{10_000 + index}"
        row["microstructure_snapshot"]["book_staleness_ms"] = staleness
        return _prepare_runtime_quality_action_entry(
            row,
            guard_config=guard_config,
            initial_runtime_state=initial_state,
        )

    same_market_best = _prepared(1, market_id="market-a", staleness=50.0)
    same_market_duplicate = _prepared(2, market_id="market-a", staleness=75.0)
    same_market_opposite_side = _prepared(3, market_id="market-a", staleness=25.0)
    same_market_opposite_side["selected_side"] = "DOWN"
    same_market_opposite_side["selected_action"] = "BUY_DOWN_SELL_BEFORE_CLOSE"
    independent_b = _prepared(4, market_id="market-b", staleness=100.0)
    independent_c = _prepared(5, market_id="market-c", staleness=150.0)

    selected = _select_diversified_quality_rows(
        [
            same_market_best,
            same_market_duplicate,
            same_market_opposite_side,
            independent_b,
            independent_c,
        ],
        target_unique_market_count=3,
        max_rows_per_market=1,
        max_rows_per_market_side=1,
    )

    assert [row["market_id"] for row in selected] == [
        "market-a",
        "market-b",
        "market-c",
    ]
    assert len({row["market_id"] for row in selected}) == 3
    assert all(row["runtime_input_quality_rules_applied"] is True for row in selected)
    assert selected[0]["diversified_holdout_selection_rank"] == 1
    assert "independent_market_window_preferred" in selected[0][
        "diversified_holdout_selection_reason_codes"
    ]


def test_o_v8_future_unseen_holdout_reports_pass_diagnostic_but_stay_closed(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path)
    rows = [
        _future_holdout_raw_row(index, decision_ts=10_000 + index)
        for index in range(1, 6)
    ]
    raw_path = _write_future_holdout_raw_manifest(
        tmp_path,
        rows=rows,
        window_start_ts=10_000,
    )
    config = PolymarketOReplayAlignedSourceRankingConfig(
        m2_candidate_report_path=fixture["m2_report_path"],
        output_dir=tmp_path,
        future_holdout_raw_manifest_path=raw_path,
    )
    raw = _v8_future_unseen_holdout_raw_collection_manifest(
        config=config,
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        action_rank_handoff_report=fixture["action_rank_handoff"],
        simulated_order_replay_report=fixture["simulated_order_replay"],
        collection_plan_report=fixture["collection_plan"],
    )
    payload = dict(raw)
    raw_id = payload.pop("o_v8_future_unseen_holdout_raw_collection_manifest_id")
    assert canonical_json_sha256(payload) == raw_id
    assert (
        raw["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_RAW_COLLECTION_MANIFEST_SCHEMA_VERSION
    )
    assert raw["future_unseen_holdout_raw_collection_ready"] is True
    assert raw["future_unseen_holdout_raw_collection_blocking_reason_codes"] == []
    assert raw["collection_status"] == "collected"
    assert raw["holdout_decision_count"] == 5
    assert raw["future_window_time_validation_passed"] is True
    assert raw["collection_plan_created_ts"] is not None
    assert raw["raw_manifest_created_ts"] == 9_000
    assert looks_like_sha256(raw["prior_reference_hash"])
    assert {
        source["source_name"] for source in raw["prior_reference_sources"]
    } >= {"simulated_replay_rows", "m2_selected_rows", "m2_blocked_rows"}

    freeze = _v8_future_unseen_holdout_input_freeze_manifest(
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        raw_collection_manifest=raw,
        collection_plan_report=fixture["collection_plan"],
        action_rank_handoff_report=fixture["action_rank_handoff"],
        execution_guard_report=fixture["execution_guard"],
        simulated_order_replay_report=fixture["simulated_order_replay"],
        runtime_field_coverage_report=fixture["runtime_field_coverage"],
        handoff_gate_report=fixture["handoff_gate"],
        paper_candidate_gate_design_report=fixture["paper_gate"],
    )
    assert (
        freeze["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_INPUT_FREEZE_MANIFEST_SCHEMA_VERSION
    )
    assert freeze["future_unseen_holdout_input_freeze_ready"] is True
    assert looks_like_sha256(freeze["frozen_input_manifest_hash"])
    assert looks_like_sha256(freeze["frozen_current_v8_o_config_hash"])

    action_rank = _v8_future_unseen_holdout_action_rank_report(
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        raw_collection_manifest=raw,
        input_freeze_manifest=freeze,
        source_action_rank_handoff_report=fixture["action_rank_handoff"],
    )
    assert (
        action_rank["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_ACTION_RANK_SCHEMA_VERSION
    )
    assert action_rank["future_unseen_holdout_action_rank_ready"] is True
    assert action_rank["prediction_attempted"] is True
    assert action_rank["selected_action_handoff_row_count"] == 5

    execution = _v8_future_unseen_holdout_execution_replay_report(
        config=config,
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        action_rank_report=action_rank,
    )
    assert (
        execution["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_EXECUTION_REPLAY_SCHEMA_VERSION
    )
    assert execution["future_unseen_holdout_execution_replay_ready"] is True
    assert execution["execution_replay_attempted"] is True
    assert execution["simulated_allowed_order_count"] == 5
    assert execution["zero_missing_runtime_fields"] is True
    assert execution["zero_provenance_violations"] is True

    policy = _v8_future_unseen_holdout_policy_readiness_report(
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        execution_replay_report=execution,
    )
    assert (
        policy["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_POLICY_READINESS_SCHEMA_VERSION
    )
    assert policy["future_unseen_holdout_policy_readiness_passed"] is True

    handoff = _v8_future_unseen_holdout_handoff_gate_report(
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        input_freeze_manifest=freeze,
        action_rank_report=action_rank,
        execution_replay_report=execution,
        policy_readiness_report=policy,
    )
    assert (
        handoff["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_HANDOFF_GATE_SCHEMA_VERSION
    )
    assert handoff["future_unseen_holdout_handoff_gate_passed"] is True
    assert handoff["v8_execution_handoff_allowed"] is False

    paper_gate = _v8_future_unseen_holdout_paper_candidate_gate_report(
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        input_freeze_manifest=freeze,
        action_rank_report=action_rank,
        execution_replay_report=execution,
        policy_readiness_report=policy,
        handoff_gate_report=handoff,
    )
    assert (
        paper_gate["schema_version"]
        == O_V8_FUTURE_UNSEEN_HOLDOUT_PAPER_CANDIDATE_GATE_SCHEMA_VERSION
    )
    assert paper_gate["future_unseen_holdout_paper_candidate_gate_passed"] is True
    assert paper_gate["paper_candidate_allowed"] is False
    assert paper_gate["v8_execution_handoff_allowed"] is False
    assert paper_gate["paper_only"] is True
    assert paper_gate["capital_at_risk"] is False
    assert paper_gate["polymarket_write_enabled"] is False
    assert paper_gate["wallet_signing_enabled"] is False
    assert paper_gate["source_model_candidate_eligible"] is False
    assert paper_gate["freeze_ready"] is False
    assert paper_gate["promotion_evidence_eligible"] is False
    assert paper_gate["#146_start_allowed"] is False
    assert paper_gate["#134_resume_allowed"] is False


def test_o_v8_future_unseen_holdout_raw_collection_fails_on_overlap_past_or_outcomes(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path)
    bad_row = _future_holdout_raw_row(1, decision_ts=1)
    bad_row["market_id"] = "market-1"
    bad_row["decision_group_id"] = "source|market-1|1"
    bad_row["realized_trade_pnl"] = 1.0
    raw_path = _write_future_holdout_raw_manifest(
        tmp_path,
        rows=[bad_row],
        window_start_ts=1,
    )
    config = PolymarketOReplayAlignedSourceRankingConfig(
        m2_candidate_report_path=fixture["m2_report_path"],
        output_dir=tmp_path,
        future_holdout_raw_manifest_path=raw_path,
    )
    raw = _v8_future_unseen_holdout_raw_collection_manifest(
        config=config,
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        action_rank_handoff_report=fixture["action_rank_handoff"],
        simulated_order_replay_report=fixture["simulated_order_replay"],
        collection_plan_report=fixture["collection_plan"],
    )
    assert raw["future_unseen_holdout_raw_collection_ready"] is False
    assert set(raw["future_unseen_holdout_raw_collection_blocking_reason_codes"]) >= {
        "future_holdout_window_not_future_unseen",
        "future_holdout_overlap_with_prior_data",
        "future_holdout_forbidden_outcome_fields_present",
    }
    assert raw["holdout_decision_rows"] == []
    assert raw["paper_candidate_allowed"] is False
    assert raw["v8_execution_handoff_allowed"] is False

    freeze = _v8_future_unseen_holdout_input_freeze_manifest(
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        raw_collection_manifest=raw,
        collection_plan_report=fixture["collection_plan"],
        action_rank_handoff_report=fixture["action_rank_handoff"],
        execution_guard_report=fixture["execution_guard"],
        simulated_order_replay_report=fixture["simulated_order_replay"],
        runtime_field_coverage_report=fixture["runtime_field_coverage"],
        handoff_gate_report=fixture["handoff_gate"],
        paper_candidate_gate_design_report=fixture["paper_gate"],
    )
    action_rank = _v8_future_unseen_holdout_action_rank_report(
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        raw_collection_manifest=raw,
        input_freeze_manifest=freeze,
        source_action_rank_handoff_report=fixture["action_rank_handoff"],
    )
    execution = _v8_future_unseen_holdout_execution_replay_report(
        config=config,
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        action_rank_report=action_rank,
    )
    assert freeze["future_unseen_holdout_input_freeze_ready"] is False
    assert action_rank["prediction_attempted"] is False
    assert execution["execution_replay_attempted"] is False
    assert execution["future_unseen_holdout_execution_replay_ready"] is False


def test_o_v8_future_unseen_holdout_raw_collection_detects_m2_row_overlap(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path)
    fixture["m2_report"]["m2_selected_rows"] = [
        {
            "market_id": "m2-selected-market",
            "decision_group_id": "m2-selected|m2-selected-market|7",
            "decision_ts": 7,
        }
    ]
    fixture["m2_report"]["m2_blocked_rows"] = [
        {
            "market_id": "m2-blocked-market",
            "decision_group_id": "m2-blocked|m2-blocked-market|8",
            "decision_ts": 8,
        }
    ]
    row = _future_holdout_raw_row(1, decision_ts=10_001)
    row["market_id"] = "m2-selected-market"
    raw_path = _write_future_holdout_raw_manifest(
        tmp_path,
        rows=[row],
        window_start_ts=10_000,
    )
    config = PolymarketOReplayAlignedSourceRankingConfig(
        m2_candidate_report_path=fixture["m2_report_path"],
        output_dir=tmp_path,
        future_holdout_raw_manifest_path=raw_path,
    )
    raw = _v8_future_unseen_holdout_raw_collection_manifest(
        config=config,
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        action_rank_handoff_report=fixture["action_rank_handoff"],
        simulated_order_replay_report=fixture["simulated_order_replay"],
        collection_plan_report=fixture["collection_plan"],
    )
    assert raw["future_unseen_holdout_raw_collection_ready"] is False
    assert "future_holdout_overlap_with_prior_data" in raw[
        "future_unseen_holdout_raw_collection_blocking_reason_codes"
    ]
    sources = {source["source_name"]: source for source in raw["prior_reference_sources"]}
    assert sources["m2_selected_rows"]["row_count"] == 1
    assert sources["m2_blocked_rows"]["row_count"] == 1
    assert looks_like_sha256(raw["prior_reference_hash"])


def test_o_v8_future_unseen_holdout_raw_collection_detects_source_report_overlap(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path)
    source_report_path = tmp_path / "source_report.json"
    source_report_path.write_text(
        json.dumps(
            {
                "report_type": "source-overlap-fixture",
                "rows": [
                    {
                        "market_id": "source-only-market",
                        "decision_group_id": "source-only|source-only-market|9",
                        "decision_ts": 9,
                        "split": "validation",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fixture["m2_report"]["m2_blocked_rows"] = [
        {
            "market_id": "m2-pointer-market",
            "decision_ts": 8,
            "source_report_path": str(source_report_path),
        }
    ]
    row = _future_holdout_raw_row(1, decision_ts=10_001)
    row["market_id"] = "source-only-market"
    raw_path = _write_future_holdout_raw_manifest(
        tmp_path,
        rows=[row],
        window_start_ts=10_000,
    )
    config = PolymarketOReplayAlignedSourceRankingConfig(
        m2_candidate_report_path=fixture["m2_report_path"],
        output_dir=tmp_path,
        future_holdout_raw_manifest_path=raw_path,
    )
    raw = _v8_future_unseen_holdout_raw_collection_manifest(
        config=config,
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        action_rank_handoff_report=fixture["action_rank_handoff"],
        simulated_order_replay_report=fixture["simulated_order_replay"],
        collection_plan_report=fixture["collection_plan"],
    )
    assert raw["future_unseen_holdout_raw_collection_ready"] is False
    assert "future_holdout_overlap_with_prior_data" in raw[
        "future_unseen_holdout_raw_collection_blocking_reason_codes"
    ]
    source_report_sources = [
        source
        for source in raw["prior_reference_sources"]
        if source["source_name"].startswith("source_report_rows_")
    ]
    assert source_report_sources
    assert source_report_sources[0]["row_count"] == 1
    assert source_report_sources[0]["split_counts"] == {"validation": 1}


def test_o_v8_future_unseen_holdout_raw_collection_fails_invalid_time_windows(
    tmp_path: Path,
) -> None:
    fixture = _build_v8_future_gate_design_fixture(tmp_path)
    row = _future_holdout_raw_row(1, decision_ts=10_001)
    raw_path = _write_future_holdout_raw_manifest(
        tmp_path,
        rows=[row],
        window_start_ts=5,
        input_freeze_created_ts=9_000,
        raw_manifest_created_ts=8_000,
    )
    config = PolymarketOReplayAlignedSourceRankingConfig(
        m2_candidate_report_path=fixture["m2_report_path"],
        output_dir=tmp_path,
        future_holdout_raw_manifest_path=raw_path,
    )
    raw = _v8_future_unseen_holdout_raw_collection_manifest(
        config=config,
        m2_report_path=fixture["m2_report_path"],
        m2_report=fixture["m2_report"],
        action_rank_handoff_report=fixture["action_rank_handoff"],
        simulated_order_replay_report=fixture["simulated_order_replay"],
        collection_plan_report=fixture["collection_plan"],
    )
    assert raw["future_unseen_holdout_raw_collection_ready"] is False
    assert raw["future_window_time_validation_passed"] is False
    check = raw["future_unseen_holdout_raw_collection_required_checks"][
        "future_only_window"
    ]
    assert check["observed"]["window_start_after_prior_decision_ts"] is False
    assert check["observed"]["window_start_after_collection_plan_created_ts"] is False
    assert check["observed"]["raw_manifest_created_after_input_freeze"] is False


def test_o_v8_execution_guard_block_analysis_classifies_safe_order_discovery(
    tmp_path: Path,
) -> None:
    def _blocked_row(
        *,
        decision_id: str,
        action: str,
        blocking_reasons: list[str],
        guard_reasons: list[str] | None = None,
        exposure_reasons: list[str] | None = None,
        missing_fields: list[str] | None = None,
        time_to_close: float = 180.0,
        p_up_disagreement: bool = False,
    ) -> dict[str, Any]:
        side = "UP" if "BUY_UP" in action else "DOWN"
        return {
            "decision_group_id": decision_id,
            "market_id": f"market-{decision_id}",
            "decision_ts": 100 + len(decision_id),
            "source_selected_action": action,
            "source_selected_side": side,
            "source_selected_family": "SELL_BEFORE_CLOSE",
            "source_model_score": 0.82,
            "p_up_action_disagreement": p_up_disagreement,
            "microstructure_snapshot": {
                "book_staleness_ms": 500.0,
                "spread_bps": 200.0,
                "queue_fill_proxy": 0.90,
                "time_to_close_seconds": time_to_close,
            },
            "execution_guarded_action": action,
            "execution_guarded_side": side,
            "execution_guarded_family": "SELL_BEFORE_CLOSE",
            "execution_guarded_score": 0.80,
            "order_allowed": False,
            "proposed_order_size": 0.0,
            "execution_blocking_reason_codes": blocking_reasons,
            "execution_guard_reason_codes": guard_reasons or [],
            "exposure_reason_codes": exposure_reasons
            or ["execution_simulated_order_blocked"],
            "missing_runtime_field_codes": missing_fields or [],
            "simulated_order_id": None,
            "source_score_mutated": False,
            "o_model_predicted_score_mutated": False,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    replay_report = {
        "o_v8_execution_simulated_order_replay_report_id": "replay-test",
        "decision_count": 3,
        "blocked_decision_count": 3,
        "simulated_allowed_order_count": 0,
        "simulated_decision_rows": [
            _blocked_row(
                decision_id="missing",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                blocking_reasons=[
                    "execution_required_runtime_fields_missing",
                    "execution_exposure_state_missing",
                ],
                missing_fields=["execution_exposure_state_missing"],
            ),
            _blocked_row(
                decision_id="threshold",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                blocking_reasons=["execution_time_to_close_unsafe"],
                time_to_close=20.0,
            ),
            _blocked_row(
                decision_id="pup",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                blocking_reasons=["execution_p_up_side_disagreement"],
                p_up_disagreement=True,
            ),
        ],
    }

    report = _v8_execution_guard_block_analysis_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report={"o_v8_action_rank_handoff_report_id": "handoff-test"},
        execution_guard_report={"o_v8_execution_risk_guard_report_id": "guard-test"},
        runtime_state_report={"o_v8_execution_runtime_state_report_id": "state-test"},
        simulated_order_replay_report=replay_report,
    )

    assert (
        report["schema_version"]
        == O_V8_EXECUTION_GUARD_BLOCK_ANALYSIS_SCHEMA_VERSION
    )
    assert report["decision_count"] == 3
    assert report["blocked_decision_count"] == 3
    assert report["allowed_decision_count"] == 0
    by_id = {
        row["decision_group_id"]: row
        for row in report["blocked_decision_analysis_rows"]
    }
    assert by_id["missing"]["safe_order_discovery_classification"] == (
        "blocked_only_by_missing_runtime_fields"
    )
    assert by_id["threshold"]["safe_order_discovery_classification"] == (
        "blocked_only_by_configurable_thresholds"
    )
    assert by_id["pup"]["safe_order_discovery_classification"] == (
        "fundamentally_unsafe"
    )
    summary = report["safe_order_discovery_summary"]
    assert summary["safe_order_candidate_count"] == 2
    assert summary["blocked_only_by_missing_runtime_fields_count"] == 1
    assert summary["blocked_only_by_configurable_thresholds_count"] == 1
    assert summary["fundamentally_unsafe_count"] == 1
    assert report["why_simulated_allowed_order_count_zero"]
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False


def test_o_v8_execution_runtime_field_coverage_classifies_backfill_plan(
    tmp_path: Path,
) -> None:
    def _row(
        *,
        decision_id: str,
        action: str,
        missing_code: str,
        runtime_mode: str = "simulated_runtime_state",
        runtime_available: bool = True,
    ) -> dict[str, Any]:
        side = "UP" if "BUY_UP" in action else "NONE"
        family = "NO_TRADE" if action == "NO_TRADE" else "SELL_BEFORE_CLOSE"
        return {
            "decision_group_id": decision_id,
            "market_id": f"market-{decision_id}",
            "decision_ts": 200 + len(decision_id),
            "source_selected_action": action,
            "source_selected_side": side,
            "source_selected_family": family,
            "source_model_score": 0.81,
            "source_high_score_flag": True,
            "p_up": 0.70,
            "p_down": 0.30,
            "p_up_action_disagreement": False,
            "microstructure_snapshot": {
                "book_staleness_ms": 500.0,
                "spread_bps": 200.0,
                "queue_fill_proxy": 0.90,
                "time_to_close_seconds": None
                if "time_to_close" in missing_code
                else 180.0,
            },
            "top_k_action_ranking": [
                {"selected_action": candidate, "corrected_model_score": 0.50}
                for candidate in O_REQUIRED_DECISION_ACTION_FAMILIES
            ],
            "reference_price_feature_provenance": {"provenance_valid": True},
            "execution_guarded_action": action,
            "execution_guarded_side": side,
            "execution_guarded_family": family,
            "execution_guarded_score": 0.80,
            "order_allowed": False,
            "proposed_order_size": 0.0,
            "execution_blocking_reason_codes": [
                "execution_required_runtime_fields_missing"
            ],
            "execution_guard_reason_codes": [],
            "exposure_reason_codes": ["execution_simulated_order_blocked"],
            "missing_runtime_field_codes": [missing_code],
            "runtime_mode": runtime_mode,
            "runtime_exposure_state_available": runtime_available,
            "pre_decision_exposure_state": {"current_total_exposure": 0.0},
            "post_decision_exposure_state": {"current_total_exposure": 0.0},
            "simulated_order_id": None,
            "source_score_mutated": False,
            "o_model_predicted_score_mutated": False,
        }

    m2_report_path = tmp_path / "m2.json"
    m2_report = {"m2_stateful_replay_parity_candidate_report_id": "m2-test"}
    m2_report_path.write_text(json.dumps(m2_report, sort_keys=True), encoding="utf-8")
    replay_rows = [
        _row(
            decision_id="derived",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            missing_code="missing_selected_side",
        ),
        _row(
            decision_id="notrade",
            action="NO_TRADE",
            missing_code="missing_microstructure_time_to_close_seconds",
        ),
        _row(
            decision_id="gap",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            missing_code="missing_microstructure_time_to_close_seconds",
        ),
        _row(
            decision_id="simulation",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            missing_code="execution_exposure_state_missing",
        ),
    ]
    replay_report = {
        "o_v8_execution_simulated_order_replay_report_id": "replay-test",
        "decision_count": len(replay_rows),
        "blocked_decision_count": len(replay_rows),
        "simulated_allowed_order_count": 0,
        "simulated_decision_rows": replay_rows,
    }
    block_analysis_report = {
        "o_v8_execution_guard_block_analysis_report_id": "block-test",
    }

    report = _v8_execution_runtime_field_coverage_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report={"o_v8_action_rank_handoff_report_id": "handoff-test"},
        execution_guard_report={"o_v8_execution_risk_guard_report_id": "guard-test"},
        runtime_state_report={"o_v8_execution_runtime_state_report_id": "state-test"},
        simulated_order_replay_report=replay_report,
        block_analysis_report=block_analysis_report,
    )

    assert (
        report["schema_version"]
        == O_V8_EXECUTION_RUNTIME_FIELD_COVERAGE_SCHEMA_VERSION
    )
    assert report["decision_count"] == 4
    assert report["missing_runtime_field_decision_count"] == 4
    assert report["missing_runtime_field_occurrence_count"] == 4
    assert report["classification_counts"][
        "derived_backfill_from_existing_handoff_fields"
    ] == 1
    assert report["classification_counts"]["optional_for_no_trade"] == 1
    assert report["classification_counts"]["true_data_coverage_gap"] == 1
    assert report["classification_counts"][
        "too_strict_for_simulation_only_mode"
    ] == 1
    assert report["safe_backfill_candidate_count"] == 2
    assert report["existing_handoff_backfill_candidate_count"] == 1
    assert report["decision_time_data_join_backfill_candidate_count"] == 1
    assert report["required_field_policy_relaxation_candidate_count"] == 2
    by_id = {
        row["decision_group_id"]: row
        for row in report["runtime_field_coverage_decision_rows"]
    }
    assert by_id["derived"]["runtime_field_backfill_candidates"][0][
        "proposed_rule_id"
    ] == "derive_selected_side_from_action"
    assert by_id["derived"]["runtime_field_backfill_candidates"][0][
        "backfill_source_class"
    ] == "existing_handoff_fields"
    assert by_id["derived"]["runtime_field_backfill_candidates"][0][
        "can_backfill_in_later_commit"
    ] is True
    assert by_id["notrade"]["runtime_field_backfill_candidates"][0][
        "field_gap_classification"
    ] == "optional_for_no_trade"
    assert by_id["notrade"]["runtime_field_backfill_candidates"][0][
        "requires_required_field_policy_change"
    ] is True
    assert by_id["gap"]["runtime_field_backfill_candidates"][0][
        "field_gap_classification"
    ] == "true_data_coverage_gap"
    assert by_id["gap"]["runtime_field_backfill_candidates"][0][
        "backfill_source_class"
    ] == "decision_time_data_join_required"
    assert by_id["gap"]["runtime_field_backfill_candidates"][0][
        "can_backfill_in_later_commit"
    ] is True
    assert by_id["simulation"]["runtime_field_backfill_candidates"][0][
        "field_gap_classification"
    ] == "too_strict_for_simulation_only_mode"
    assert by_id["simulation"]["runtime_field_backfill_candidates"][0][
        "requires_required_field_policy_change"
    ] is True
    assert all(
        rule["applied_now"] is False
        for rule in report["proposed_deterministic_backfill_rules"]
    )
    assert report["backfill_rules_applied"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False


def test_o_v8_paper_candidate_unlock_allows_local_paper_loop_only(
    tmp_path: Path,
) -> None:
    issue159_dir, expected_hashes = _write_issue159_paper_unlock_fixture(
        tmp_path / "issue159"
    )

    result = run_polymarket_o_v8_paper_candidate_unlock(
        PolymarketOV8PaperCandidateUnlockConfig(
            run_id="paper-unlock-pass",
            output_dir=tmp_path / "out",
            issue_159_eval_dir=issue159_dir,
            expected_issue_159_hashes=expected_hashes,
            manual_approval_approved=True,
            manual_approval_id="manual-approval-test",
            manual_approval_operator="pytest",
        )
    )

    unlock = result.paper_candidate_unlock_report
    unlock_payload = dict(unlock)
    unlock_id = unlock_payload.pop("o_v8_paper_candidate_unlock_report_id")
    assert canonical_json_sha256(unlock_payload) == unlock_id
    assert unlock["schema_version"] == O_V8_PAPER_CANDIDATE_UNLOCK_SCHEMA_VERSION
    assert unlock["paper_candidate_allowed"] is True
    assert unlock["paper_candidate_blocking_reason_codes"] == []
    assert unlock["pinned_artifact_hashes_verified"] is True
    assert unlock["manual_approval_payload"]["manual_approval_approved"] is True
    assert unlock["v8_execution_handoff_allowed"] is False
    assert unlock["paper_only"] is True
    assert unlock["capital_at_risk"] is False
    assert unlock["polymarket_write_enabled"] is False
    assert unlock["wallet_signing_enabled"] is False
    assert unlock["source_model_candidate_eligible"] is False
    assert unlock["freeze_ready"] is False
    assert unlock["promotion_evidence_eligible"] is False
    assert unlock["#146_start_allowed"] is False
    assert unlock["#134_resume_allowed"] is False

    loop = result.paper_internal_execution_loop_report
    loop_payload = dict(loop)
    loop_id = loop_payload.pop("o_v8_paper_internal_execution_loop_report_id")
    assert canonical_json_sha256(loop_payload) == loop_id
    assert loop["schema_version"] == O_V8_PAPER_INTERNAL_EXECUTION_LOOP_SCHEMA_VERSION
    assert loop["paper_internal_execution_loop_enabled"] is True
    assert loop["v8_paper_internal_handoff_allowed"] is True
    assert loop["v8_execution_handoff_allowed"] is False
    assert loop["paper_order_intent_count"] == 2
    assert loop["paper_fill_count"] == 2
    assert loop["paper_ledger_entry_count"] == 2

    fill_report = result.paper_fill_simulation_report
    fill_payload = dict(fill_report)
    fill_id = fill_payload.pop("o_v8_paper_fill_simulation_report_id")
    assert canonical_json_sha256(fill_payload) == fill_id
    assert (
        fill_report["schema_version"] == O_V8_PAPER_FILL_SIMULATION_SCHEMA_VERSION
    )
    assert fill_report["fill_count"] == 2
    assert fill_report["outcome_pnl_used"] is False
    assert fill_report["realized_pnl_used"] is False

    safety = result.paper_runtime_safety_report
    safety_payload = dict(safety)
    safety_id = safety_payload.pop("o_v8_paper_runtime_safety_report_id")
    assert canonical_json_sha256(safety_payload) == safety_id
    assert safety["schema_version"] == O_V8_PAPER_RUNTIME_SAFETY_SCHEMA_VERSION
    assert safety["paper_runtime_safety_passed"] is True
    assert safety["ledger_updates_only_accepted_intents"] is True
    assert safety["v8_execution_handoff_allowed"] is False

    manifest = result.manifest
    manifest_payload = dict(manifest)
    manifest_id = manifest_payload.pop("o_v8_paper_candidate_unlock_manifest_id")
    assert canonical_json_sha256(manifest_payload) == manifest_id
    assert (
        manifest["schema_version"]
        == O_V8_PAPER_CANDIDATE_UNLOCK_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["paper_candidate_allowed"] is True
    assert manifest["paper_internal_execution_loop_enabled"] is True
    assert manifest["v8_paper_internal_handoff_allowed"] is True
    assert manifest["v8_execution_handoff_allowed"] is False
    assert manifest["artifact_hashes"]["paper_order_intent_log"]

    intents = _read_jsonl(result.artifact_paths["paper_order_intent_log"])
    assert len(intents) == 2
    assert {row["simulated_order_id"] for row in intents} == {
        "sim-v8-o-000001",
        "sim-v8-o-000002",
    }
    assert all(row["paper_only"] is True for row in intents)
    assert all(row["capital_at_risk"] is False for row in intents)
    assert all(row["polymarket_write_enabled"] is False for row in intents)
    assert all(row["wallet_signing_enabled"] is False for row in intents)


def test_o_v8_paper_candidate_unlock_fails_without_manual_approval(
    tmp_path: Path,
) -> None:
    issue159_dir, expected_hashes = _write_issue159_paper_unlock_fixture(
        tmp_path / "issue159"
    )

    result = run_polymarket_o_v8_paper_candidate_unlock(
        PolymarketOV8PaperCandidateUnlockConfig(
            run_id="paper-unlock-no-approval",
            output_dir=tmp_path / "out",
            issue_159_eval_dir=issue159_dir,
            expected_issue_159_hashes=expected_hashes,
            manual_approval_approved=False,
            manual_approval_id="manual-approval-test",
            manual_approval_operator="pytest",
        )
    )

    assert result.paper_candidate_unlock_report["paper_candidate_allowed"] is False
    assert "manual_approval_required_before_paper_candidate" in result.paper_candidate_unlock_report[
        "paper_candidate_blocking_reason_codes"
    ]
    assert (
        result.paper_internal_execution_loop_report[
            "paper_internal_execution_loop_enabled"
        ]
        is False
    )
    assert result.paper_internal_execution_loop_report["paper_order_intent_count"] == 0
    assert _read_jsonl(result.artifact_paths["paper_order_intent_log"]) == []
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_candidate_unlock_fails_on_pinned_hash_mismatch(
    tmp_path: Path,
) -> None:
    issue159_dir, expected_hashes = _write_issue159_paper_unlock_fixture(
        tmp_path / "issue159"
    )
    expected_hashes["execution_replay_report"] = "0" * 64

    result = run_polymarket_o_v8_paper_candidate_unlock(
        PolymarketOV8PaperCandidateUnlockConfig(
            run_id="paper-unlock-hash-mismatch",
            output_dir=tmp_path / "out",
            issue_159_eval_dir=issue159_dir,
            expected_issue_159_hashes=expected_hashes,
            manual_approval_approved=True,
            manual_approval_id="manual-approval-test",
            manual_approval_operator="pytest",
        )
    )

    assert result.paper_candidate_unlock_report["paper_candidate_allowed"] is False
    assert "pinned_issue_159_artifact_hash_mismatch" in result.paper_candidate_unlock_report[
        "paper_candidate_blocking_reason_codes"
    ]
    assert result.manifest["paper_internal_execution_loop_enabled"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_candidate_unlock_fails_on_upstream_safety_flag(
    tmp_path: Path,
) -> None:
    issue159_dir, expected_hashes = _write_issue159_paper_unlock_fixture(
        tmp_path / "issue159",
        upstream_capital_at_risk=True,
    )

    result = run_polymarket_o_v8_paper_candidate_unlock(
        PolymarketOV8PaperCandidateUnlockConfig(
            run_id="paper-unlock-safety-flag",
            output_dir=tmp_path / "out",
            issue_159_eval_dir=issue159_dir,
            expected_issue_159_hashes=expected_hashes,
            manual_approval_approved=True,
            manual_approval_id="manual-approval-test",
            manual_approval_operator="pytest",
        )
    )

    assert result.paper_candidate_unlock_report["paper_candidate_allowed"] is False
    assert "paper_candidate_live_safety_flags_not_blocked" in result.paper_candidate_unlock_report[
        "paper_candidate_blocking_reason_codes"
    ]
    assert result.manifest["paper_candidate_allowed"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_loop_single_cycle_success(tmp_path: Path) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-single",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=(
                (
                    _paper_fresh_public_row(
                        index=1,
                        market_id="fresh-market-up",
                        action="BUY_UP_HOLD_TO_SETTLEMENT",
                        side="UP",
                        p_up=0.82,
                    ),
                    _paper_fresh_public_row(
                        index=2,
                        market_id="fresh-market-down",
                        action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                        side="DOWN",
                        p_up=0.22,
                    ),
                ),
            ),
        )
    )

    run_report = result.fresh_loop_run_report
    run_payload = dict(run_report)
    run_id = run_payload.pop("o_v8_paper_fresh_loop_run_report_id")
    assert canonical_json_sha256(run_payload) == run_id
    assert run_report["schema_version"] == O_V8_PAPER_FRESH_LOOP_RUN_SCHEMA_VERSION
    assert run_report["paper_fresh_loop_enabled"] is True
    assert run_report["paper_fresh_loop_mode"] == "single_cycle"
    assert (
        run_report["paper_fresh_loop_public_data_source"]
        == O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE
    )
    assert run_report["uses_paper_intent_logs_as_fresh_public_data"] is False
    assert run_report["paper_fresh_loop_cycle_count"] == 1
    assert run_report["candidate_decision_count"] == 2
    assert run_report["guard_allowed_decision_count"] == 2
    assert run_report["guard_blocked_decision_count"] == 0
    assert run_report["paper_fresh_order_intent_count"] == 2
    assert run_report["paper_fresh_fill_count"] == 2
    assert run_report["paper_fresh_ledger_entry_count"] == 2
    assert run_report["thresholds_tuned"] is False
    assert run_report["forbidden_outcome_fields_used"] == []
    assert run_report["v8_paper_internal_handoff_allowed"] is True
    assert run_report["v8_execution_handoff_allowed"] is False
    assert run_report["source_model_candidate_eligible"] is False
    assert run_report["freeze_ready"] is False
    assert run_report["promotion_evidence_eligible"] is False
    assert run_report["#146_start_allowed"] is False
    assert run_report["#134_resume_allowed"] is False

    fill_report = result.fill_simulation_report
    fill_payload = dict(fill_report)
    fill_id = fill_payload.pop("o_v8_paper_fresh_fill_simulation_report_id")
    assert canonical_json_sha256(fill_payload) == fill_id
    assert (
        fill_report["schema_version"]
        == O_V8_PAPER_FRESH_FILL_SIMULATION_SCHEMA_VERSION
    )
    assert fill_report["paper_fresh_fill_count"] == 2
    assert fill_report["outcome_pnl_used"] is False
    assert fill_report["realized_pnl_used"] is False

    safety = result.runtime_safety_report
    safety_payload = dict(safety)
    safety_id = safety_payload.pop("o_v8_paper_fresh_runtime_safety_report_id")
    assert canonical_json_sha256(safety_payload) == safety_id
    assert (
        safety["schema_version"] == O_V8_PAPER_FRESH_RUNTIME_SAFETY_SCHEMA_VERSION
    )
    assert safety["paper_fresh_runtime_safety_passed"] is True
    assert safety["v8_execution_handoff_allowed"] is False

    monitoring = result.monitoring_report
    monitoring_payload = dict(monitoring)
    monitoring_id = monitoring_payload.pop("o_v8_paper_fresh_monitoring_report_id")
    assert canonical_json_sha256(monitoring_payload) == monitoring_id
    assert monitoring["schema_version"] == O_V8_PAPER_FRESH_MONITORING_SCHEMA_VERSION
    assert monitoring["paper_fresh_monitoring_passed"] is True
    assert monitoring["cycle_count"] == 1
    assert monitoring["cycle_monitoring_reports"][0]["public_data_source"] == (
        O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE
    )
    assert monitoring["cycle_monitoring_reports"][0]["unique_market_count"] == 2

    cumulative = result.cumulative_monitoring_report
    cumulative_payload = dict(cumulative)
    cumulative_id = cumulative_payload.pop(
        "o_v8_paper_fresh_cumulative_monitoring_report_id"
    )
    assert canonical_json_sha256(cumulative_payload) == cumulative_id
    assert (
        cumulative["schema_version"]
        == O_V8_PAPER_FRESH_CUMULATIVE_MONITORING_SCHEMA_VERSION
    )
    assert cumulative["total_cycles"] == 1
    assert cumulative["total_paper_intents"] == 2
    assert cumulative["total_paper_fills"] == 2
    assert cumulative["ledger_updates_only_accepted_intents"] is True

    no_trade = result.no_trade_diagnostic_report
    no_trade_payload = dict(no_trade)
    no_trade_id = no_trade_payload.pop(
        "o_v8_paper_fresh_no_trade_diagnostic_report_id"
    )
    assert canonical_json_sha256(no_trade_payload) == no_trade_id
    assert (
        no_trade["schema_version"]
        == O_V8_PAPER_FRESH_NO_TRADE_DIAGNOSTIC_SCHEMA_VERSION
    )
    assert no_trade["rank_blocked_by_no_trade_count"] == 0
    assert no_trade["execution_guard_blocked_count"] == 0
    assert no_trade["canonical_frozen_o_scorer_used"] is False
    assert no_trade["scoring_rule_id"] == "fresh_provider_simplified_score"
    assert no_trade["v8_execution_handoff_allowed"] is False

    decomposition = result.score_decomposition_report
    decomposition_payload = dict(decomposition)
    decomposition_id = decomposition_payload.pop(
        "o_v8_paper_fresh_score_decomposition_report_id"
    )
    assert canonical_json_sha256(decomposition_payload) == decomposition_id
    assert (
        decomposition["schema_version"]
        == O_V8_PAPER_FRESH_SCORE_DECOMPOSITION_SCHEMA_VERSION
    )
    assert decomposition["score_decomposition_action_row_count"] == 10
    assert decomposition["canonical_frozen_o_scorer_used"] is False
    assert decomposition["mutates_source_ranking_scores"] is False

    coverage = result.provider_feature_coverage_report
    coverage_payload = dict(coverage)
    coverage_id = coverage_payload.pop(
        "o_v8_paper_fresh_provider_feature_coverage_report_id"
    )
    assert canonical_json_sha256(coverage_payload) == coverage_id
    assert (
        coverage["schema_version"]
        == O_V8_PAPER_FRESH_PROVIDER_FEATURE_COVERAGE_SCHEMA_VERSION
    )
    assert coverage["public_feature_row_count"] == 2
    assert coverage["missing_runtime_field_count"] == 0
    assert coverage["provenance_invalid_count"] == 0

    alignment = result.canonical_scorer_alignment_report
    alignment_payload = dict(alignment)
    alignment_id = alignment_payload.pop(
        "o_v8_paper_fresh_canonical_scorer_alignment_report_id"
    )
    assert canonical_json_sha256(alignment_payload) == alignment_id
    assert (
        alignment["schema_version"]
        == O_V8_PAPER_FRESH_CANONICAL_SCORER_ALIGNMENT_SCHEMA_VERSION
    )
    assert alignment["canonical_frozen_o_scorer_used"] is False
    assert alignment["canonical_alignment_diagnostic_status"] == "blocked_fail_closed"
    assert "missing_frozen_model_summary" in alignment[
        "canonical_alignment_blocking_reason_codes"
    ]

    manifest = result.manifest
    manifest_payload = dict(manifest)
    manifest_id = manifest_payload.pop("o_v8_paper_fresh_loop_manifest_id")
    assert canonical_json_sha256(manifest_payload) == manifest_id
    assert manifest["schema_version"] == O_V8_PAPER_FRESH_LOOP_MANIFEST_SCHEMA_VERSION
    assert manifest["paper_fresh_loop_enabled"] is True
    assert manifest["paper_fresh_monitoring_passed"] is True
    assert (
        manifest["paper_fresh_loop_public_data_source"]
        == O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE
    )
    assert manifest["v8_paper_internal_handoff_allowed"] is True
    assert manifest["v8_execution_handoff_allowed"] is False
    assert manifest["rank_blocked_by_no_trade_count"] == 0
    assert manifest["canonical_frozen_o_scorer_used"] is False
    assert manifest["canonical_alignment_diagnostic_status"] == "blocked_fail_closed"
    assert manifest["sparse_provider_row_flag"] is True
    assert manifest["paper_only"] is True
    assert manifest["capital_at_risk"] is False
    assert manifest["polymarket_write_enabled"] is False
    assert manifest["wallet_signing_enabled"] is False

    intents = _read_jsonl(result.artifact_paths["fresh_order_intent_log"])
    assert len(intents) == 2
    assert {row["paper_fresh_order_intent_status"] for row in intents} == {
        "accepted_for_fresh_paper_loop"
    }
    assert all(row["paper_only"] is True for row in intents)
    assert all(row["capital_at_risk"] is False for row in intents)
    assert all(row["polymarket_write_enabled"] is False for row in intents)
    assert all(row["wallet_signing_enabled"] is False for row in intents)


def test_o_v8_paper_fresh_no_trade_diagnostic_rank_blocked_zero_intent(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-no-trade-rank-blocked",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=(
                (
                    _paper_fresh_public_row(
                        index=1,
                        market_id="fresh-no-trade-market",
                        action="NO_TRADE",
                        side="NONE",
                        p_up=0.50,
                    ),
                ),
            ),
        )
    )

    report = result.no_trade_diagnostic_report
    assert report["candidate_decision_count"] == 1
    assert report["rank_blocked_by_no_trade_count"] == 1
    assert report["execution_guard_blocked_count"] == 0
    assert report["paper_fresh_order_intent_count"] == 0
    assert report["zero_intent_behavior_classification"] == (
        "rank_blocked_by_no_trade_under_simplified_provider_score"
    )
    row = report["decision_rows"][0]
    assert row["selected_action"] == "NO_TRADE"
    assert row["rank_blocked_by_no_trade"] is True
    assert row["execution_guard_blocked"] is False
    assert row["missing_runtime_fields"] == []
    assert row["provenance_violations"] == []
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_distinguishes_guard_blocked_from_rank_blocked(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-p-up-disagreement",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.20,
    )
    row["p_up_action_disagreement"] = True

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-guard-blocked-not-rank-blocked",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
        )
    )

    report = result.no_trade_diagnostic_report
    assert report["rank_blocked_by_no_trade_count"] == 0
    assert report["execution_guard_blocked_count"] == 1
    decision = report["decision_rows"][0]
    assert decision["selected_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert decision["rank_blocked_by_no_trade"] is False
    assert decision["execution_guard_blocked"] is True
    assert "execution_p_up_side_disagreement" in decision[
        "execution_guard_blocking_reasons"
    ]
    assert result.fresh_loop_run_report["paper_fresh_order_intent_count"] == 0
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_score_decomposition_provider_components(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-provider-score-decomposition",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_provider=_FakeFreshPublicProvider(),
        )
    )

    report = result.score_decomposition_report
    assert report["score_decomposition_action_row_count"] == 5
    assert report["scoring_rule_id"] == "fresh_provider_simplified_score"
    assert report["canonical_frozen_o_scorer_used"] is False
    up_hts = next(
        row
        for row in report["score_decomposition_rows"]
        if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    )
    assert up_hts["p_side_contribution"] > 0.0
    assert up_hts["p_up_contribution"] > 0.0
    assert up_hts["ask_contribution"] < 0.0
    assert up_hts["spread_penalty"] > 0.0
    assert up_hts["queue_fill_term_used"] is False
    assert up_hts["book_staleness_term_used"] is False
    assert up_hts["time_to_close_term_used"] is False
    assert up_hts["canonical_frozen_o_scorer_used"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_canonical_alignment_unavailable_fail_closed(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-canonical-alignment-unavailable",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_provider=_FakeFreshPublicProvider(),
        )
    )

    report = result.canonical_scorer_alignment_report
    assert report["canonical_frozen_o_scorer_invoked"] is False
    assert report["canonical_frozen_o_scorer_used"] is False
    assert report["canonical_alignment_diagnostic_status"] == "blocked_fail_closed"
    assert "missing_frozen_model_summary" in report[
        "canonical_alignment_blocking_reason_codes"
    ]
    assert "missing_feature_schema" in report[
        "canonical_alignment_blocking_reason_codes"
    ]
    assert report["source_o_score_mutated"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["#146_start_allowed"] is False
    assert report["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_canonical_scorer_invoked_with_complete_mapping(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    canonical_manifest_path = _write_canonical_o_source_fixture(
        tmp_path,
        prefer_no_trade=False,
    )

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-canonical-complete",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            canonical_o_source_manifest_path=canonical_manifest_path,
            public_data_cycles=(
                (
                    _paper_fresh_public_row(
                        index=1,
                        market_id="fresh-canonical-market",
                        action="NO_TRADE",
                        side="NONE",
                        p_up=0.82,
                    ),
                ),
            ),
        )
    )

    mapping = result.canonical_feature_mapping_report
    assert (
        mapping["schema_version"]
        == O_V8_PAPER_FRESH_CANONICAL_FEATURE_MAPPING_SCHEMA_VERSION
    )
    assert mapping["canonical_feature_mapping_complete"] is True
    assert mapping["canonical_action_row_count"] == len(O_REQUIRED_DECISION_ACTION_FAMILIES)
    assert mapping["missing_canonical_feature_names"] == []
    assert mapping["provenance_invalid_count"] == 0

    action_rows = _read_jsonl(result.artifact_paths["fresh_canonical_action_rows"])
    assert len(action_rows) == len(O_REQUIRED_DECISION_ACTION_FAMILIES)
    assert {row["action"] for row in action_rows} == set(
        O_REQUIRED_DECISION_ACTION_FAMILIES
    )
    assert all(
        set(row["canonical_feature_values"]) == set(O_DEPLOYABLE_MODEL_FEATURE_NAMES)
        for row in action_rows
    )

    scorer = result.canonical_scorer_report
    assert scorer["schema_version"] == O_V8_PAPER_FRESH_CANONICAL_SCORER_SCHEMA_VERSION
    assert scorer["canonical_frozen_o_scorer_invoked"] is True
    assert scorer["canonical_frozen_o_scorer_used"] is True
    assert scorer["canonical_scored_action_row_count"] == len(
        O_REQUIRED_DECISION_ACTION_FAMILIES
    )
    assert scorer["canonical_selected_decision_count"] == 1
    assert scorer["canonical_selected_decision_rows"][0]["action"] == (
        "BUY_UP_HOLD_TO_SETTLEMENT"
    )

    comparison = result.scorer_comparison_report
    assert comparison["schema_version"] == O_V8_PAPER_FRESH_SCORER_COMPARISON_SCHEMA_VERSION
    assert comparison["scorer_comparison_complete"] is True
    assert comparison["selected_action_agreement_count"] == 0
    assert comparison["no_trade_agreement_count"] == 0
    row = comparison["comparison_decision_rows"][0]
    assert row["simplified_provider_selected_action"] == "NO_TRADE"
    assert row["canonical_selected_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert row["no_trade_selection_agrees"] is False

    assert result.fresh_loop_run_report["canonical_frozen_o_scorer_used"] is True
    assert result.fresh_loop_run_report["scoring_rule_id"] == (
        "canonical_frozen_o_model_predicted_score_with_frozen_shadow_correction"
    )
    assert result.no_trade_diagnostic_report[
        "canonical_frozen_o_scorer_used"
    ] is True
    assert result.no_trade_diagnostic_report["decision_rows"][0][
        "selected_action"
    ] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert result.score_decomposition_report[
        "canonical_frozen_o_scorer_used"
    ] is True
    alignment = result.canonical_scorer_alignment_report
    assert alignment["canonical_alignment_diagnostic_status"] == "passed"
    assert alignment["canonical_frozen_o_scorer_invoked"] is True
    assert alignment["canonical_frozen_o_scorer_used"] is True
    assert alignment["source_o_score_mutated"] is False
    assert alignment["v8_execution_handoff_allowed"] is False
    assert result.manifest["canonical_frozen_o_scorer_used"] is True
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_canonical_scorer_no_trade_agreement(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    canonical_manifest_path = _write_canonical_o_source_fixture(
        tmp_path,
        prefer_no_trade=True,
    )

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-canonical-no-trade",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            canonical_o_source_manifest_path=canonical_manifest_path,
            public_data_cycles=(
                (
                    _paper_fresh_public_row(
                        index=1,
                        market_id="fresh-canonical-no-trade",
                        action="NO_TRADE",
                        side="NONE",
                        p_up=0.51,
                    ),
                ),
            ),
        )
    )

    comparison = result.scorer_comparison_report
    assert comparison["scorer_comparison_complete"] is True
    assert comparison["selected_action_agreement_count"] == 1
    assert comparison["no_trade_agreement_count"] == 1
    assert comparison["comparison_decision_rows"][0]["canonical_selected_action"] == (
        "NO_TRADE"
    )
    assert result.canonical_scorer_alignment_report[
        "canonical_alignment_diagnostic_status"
    ] == "passed"
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_signal_trace_canonical_time_windows(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    canonical_manifest_path = _write_canonical_o_source_fixture(
        tmp_path,
        prefer_no_trade=False,
        preferred_action="BUY_UP_HOLD_TO_SETTLEMENT",
    )
    early_row = _paper_fresh_public_row(
        index=3,
        market_id="fresh-trace-early",
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    early_row["microstructure_snapshot"]["time_to_close_seconds"] = 260.0
    hts_allowed_row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-trace-hts",
        action="NO_TRADE",
        side="NONE",
        p_up=0.83,
    )
    hts_allowed_row["microstructure_snapshot"]["time_to_close_seconds"] = 180.0
    sbc_only_row = _paper_fresh_public_row(
        index=4,
        market_id="fresh-trace-sbc-only",
        action="NO_TRADE",
        side="NONE",
        p_up=0.84,
    )
    sbc_only_row["microstructure_snapshot"]["time_to_close_seconds"] = 90.0
    final_row = _paper_fresh_public_row(
        index=2,
        market_id="fresh-trace-final",
        action="NO_TRADE",
        side="NONE",
        p_up=0.85,
    )
    final_row["microstructure_snapshot"]["time_to_close_seconds"] = 45.0

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-signal-trace-canonical",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            canonical_o_source_manifest_path=canonical_manifest_path,
            public_data_cycles=((early_row, hts_allowed_row, sbc_only_row, final_row),),
        )
    )

    trace = result.signal_trace_report
    trace_payload = dict(trace)
    trace_id = trace_payload.pop("o_v8_paper_fresh_signal_trace_report_id")
    assert canonical_json_sha256(trace_payload) == trace_id
    assert trace["schema_version"] == O_V8_PAPER_FRESH_SIGNAL_TRACE_SCHEMA_VERSION
    assert trace["trace_row_count"] == 4
    assert trace["trace_rows_sorted_by_decision_ts"] is True
    assert [row["decision_ts"] for row in trace["trace_rows"]] == sorted(
        row["decision_ts"] for row in trace["trace_rows"]
    )
    assert trace["canonical_selected_decision_count"] == 4
    assert trace["rows_by_lifecycle_window"] == {
        "early_window": 1,
        "final_no_trade_window": 1,
        "hts_allowed_window": 1,
        "sbc_only_window": 1,
    }
    by_market = {row["market_id"]: row for row in trace["trace_rows"]}
    assert by_market["fresh-trace-sbc-only"]["selected_action_is_hts"] is True
    assert by_market["fresh-trace-sbc-only"]["required_min_time_to_close_seconds"] == 120.0
    assert by_market["fresh-trace-sbc-only"]["time_to_close_gate_passed"] is False
    assert by_market["fresh-trace-sbc-only"]["time_to_close_shortfall_seconds"] == 30.0
    assert by_market["fresh-trace-sbc-only"]["signal_outcome_classification"] == (
        "paper_intent_created"
    )
    assert by_market["fresh-trace-sbc-only"]["execution_blocking_reason_codes"] == []
    assert (
        by_market["fresh-trace-sbc-only"]["execution_guarded_action"]
        == "BUY_UP_SELL_BEFORE_CLOSE"
    )
    assert by_market["fresh-trace-sbc-only"]["hts_time_window_remap_applied"] is True
    assert (
        by_market["fresh-trace-sbc-only"]["remapped_action"]
        == "BUY_UP_SELL_BEFORE_CLOSE"
    )
    assert "same_side_sbc_guard_passed" in by_market["fresh-trace-sbc-only"][
        "remap_reason_codes"
    ]
    assert by_market["fresh-trace-final"]["lifecycle_window"] == (
        "final_no_trade_window"
    )
    assert by_market["fresh-trace-final"]["is_in_final_no_trade_window"] is True
    assert by_market["fresh-trace-early"]["lifecycle_window"] == "early_window"
    assert by_market["fresh-trace-hts"]["lifecycle_window"] == "hts_allowed_window"
    assert trace["rows_blocked_by_time_to_close"] == 2
    assert trace["rows_with_missing_runtime_fields"] == 0
    assert trace["rows_with_provenance_violations"] == 0
    assert trace["paper_only"] is True
    assert trace["capital_at_risk"] is False
    assert trace["v8_execution_handoff_allowed"] is False

    time_window = result.time_window_diagnostic_report
    time_window_payload = dict(time_window)
    time_window_id = time_window_payload.pop(
        "o_v8_paper_fresh_time_window_diagnostic_report_id"
    )
    assert canonical_json_sha256(time_window_payload) == time_window_id
    assert (
        time_window["schema_version"]
        == O_V8_PAPER_FRESH_TIME_WINDOW_DIAGNOSTIC_SCHEMA_VERSION
    )
    assert time_window["rows_blocked_by_time_to_close"] == 2
    assert time_window["hts_selected_after_hts_window_expired_count"] == 2
    assert result.manifest["signal_trace_row_count"] == 4
    assert result.manifest["signal_trace_rows_sorted_by_decision_ts"] is True
    assert "fresh_signal_trace_report" in result.manifest["artifact_hashes"]
    assert "fresh_time_window_diagnostic_report" in result.manifest["artifact_hashes"]
    assert result.artifact_hashes["fresh_signal_trace_report"] == (
        _file_sha256_for_test(result.artifact_paths["fresh_signal_trace_report"])
    )
    assert result.artifact_hashes["fresh_time_window_diagnostic_report"] == (
        _file_sha256_for_test(
            result.artifact_paths["fresh_time_window_diagnostic_report"]
        )
    )
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_signal_trace_sbc_window_can_pass(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    canonical_manifest_path = _write_canonical_o_source_fixture(
        tmp_path,
        prefer_no_trade=False,
        preferred_action="BUY_UP_SELL_BEFORE_CLOSE",
    )
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-trace-sbc-pass",
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["time_to_close_seconds"] = 90.0

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-signal-trace-sbc-pass",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            canonical_o_source_manifest_path=canonical_manifest_path,
            public_data_cycles=((row,),),
        )
    )

    trace_row = result.signal_trace_report["trace_rows"][0]
    assert trace_row["canonical_scorer_used"] is True
    assert trace_row["canonical_selected_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert trace_row["selected_action_family"] == "SELL_BEFORE_CLOSE"
    assert trace_row["lifecycle_window"] == "sbc_only_window"
    assert trace_row["required_min_time_to_close_seconds"] == 60.0
    assert trace_row["time_to_close_gate_passed"] is True
    assert trace_row["time_to_close_shortfall_seconds"] == 0.0
    assert trace_row["order_allowed"] is True
    assert trace_row["paper_intent_id"]
    assert trace_row["signal_outcome_classification"] == "paper_intent_created"
    assert result.signal_trace_report["paper_intent_count"] == 1
    assert result.time_window_diagnostic_report[
        "real_action_selected_inside_executable_window_count"
    ] == 1
    assert result.manifest["paper_fresh_order_intent_count"] == 1
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_hts_time_window_remap_creates_paper_intent(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-hts-remap-up",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["time_to_close_seconds"] = 90.0

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-hts-remap-up",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
        )
    )

    remap = result.execution_layer_v2_paper_remap_report
    remap_payload = dict(remap)
    remap_id = remap_payload.pop("execution_layer_v2_paper_remap_report_id")
    assert canonical_json_sha256(remap_payload) == remap_id
    assert remap["schema_version"] == EXECUTION_LAYER_V2_PAPER_REMAP_SCHEMA_VERSION
    assert remap["hts_time_window_blocked_count"] == 1
    assert remap["same_side_sbc_alternative_available_count"] == 1
    assert remap["same_side_sbc_calibrated_ev_available_count"] == 1
    assert remap["remap_candidate_count"] == 1
    assert remap["remap_guard_passed_count"] == 1
    assert remap["paper_intent_remap_applied_count"] == 1
    assert remap["paper_only"] is True
    assert remap["capital_at_risk"] is False
    assert remap["polymarket_write_enabled"] is False
    assert remap["wallet_signing_enabled"] is False
    assert remap["v8_execution_handoff_allowed"] is False
    assert remap["source_scores_mutated"] is False
    assert remap["o_score_mutated"] is False

    intents = _read_jsonl(result.artifact_paths["fresh_order_intent_log"])
    assert len(intents) == 1
    intent = intents[0]
    assert intent["source_selected_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert intent["execution_guarded_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert intent["original_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert intent["remapped_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert intent["hts_time_window_remap_applied"] is True
    assert "same_side_sbc_guard_passed" in intent["remap_reason_codes"]
    assert intent["paper_only"] is True
    assert intent["capital_at_risk"] is False
    assert intent["polymarket_write_enabled"] is False
    assert intent["wallet_signing_enabled"] is False

    trace_row = result.signal_trace_report["trace_rows"][0]
    assert trace_row["canonical_selected_action"] is None
    assert trace_row["selected_action_is_hts"] is True
    assert trace_row["original_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert trace_row["remapped_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert trace_row["hts_time_window_remap_applied"] is True
    assert trace_row["execution_guarded_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert trace_row["order_allowed"] is True
    assert trace_row["paper_intent_id"]

    assert result.fresh_loop_run_report[
        "execution_layer_v2_paper_remap_applied_count"
    ] == 1
    assert result.manifest["execution_layer_v2_paper_remap_applied_count"] == 1
    assert "execution_layer_v2_paper_remap_report" in result.manifest[
        "artifact_hashes"
    ]
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_hts_time_window_remap_missing_sbc_fails_closed(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-hts-remap-missing-sbc",
        action="BUY_DOWN_HOLD_TO_SETTLEMENT",
        side="DOWN",
        p_up=0.20,
    )
    row["microstructure_snapshot"]["time_to_close_seconds"] = 90.0
    row["full_5_action_ranking"] = [
        candidate
        for candidate in row["full_5_action_ranking"]
        if candidate["selected_action"] != "BUY_DOWN_SELL_BEFORE_CLOSE"
    ]

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-hts-remap-missing-sbc",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
        )
    )

    remap = result.execution_layer_v2_paper_remap_report
    assert remap["hts_time_window_blocked_count"] == 1
    assert remap["same_side_sbc_alternative_available_count"] == 0
    assert remap["remap_candidate_count"] == 0
    assert remap["remap_guard_passed_count"] == 0
    assert "same_side_sbc_alternative_missing" in remap[
        "remap_reason_distribution"
    ]
    assert result.fresh_loop_run_report["paper_fresh_order_intent_count"] == 0
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_hts_time_window_remap_guard_blocked_fails_closed(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-hts-remap-wide-sbc",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["time_to_close_seconds"] = 90.0
    for candidate in row["full_5_action_ranking"]:
        if candidate["selected_action"] == "BUY_UP_SELL_BEFORE_CLOSE":
            candidate["microstructure_snapshot"] = {
                **row["microstructure_snapshot"],
                "spread_bps": 9_999.0,
            }

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-hts-remap-wide-sbc",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
        )
    )

    remap = result.execution_layer_v2_paper_remap_report
    assert remap["hts_time_window_blocked_count"] == 1
    assert remap["same_side_sbc_alternative_available_count"] == 1
    assert remap["same_side_sbc_calibrated_ev_available_count"] == 1
    assert remap["remap_candidate_count"] == 1
    assert remap["remap_guard_passed_count"] == 0
    assert "execution_spread_too_wide" in remap["remap_reason_distribution"]
    assert result.fresh_loop_run_report["paper_fresh_order_intent_count"] == 0
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_exit_adapter_no_position_emits_no_exit(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-exit-no-position",
        action="NO_TRADE",
        side="NONE",
        p_up=0.55,
    )

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-exit-no-position",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
        )
    )

    position_report = result.paper_position_state_report
    position_payload = dict(position_report)
    position_id = position_payload.pop("o_v8_paper_fresh_position_state_report_id")
    assert canonical_json_sha256(position_payload) == position_id
    assert (
        position_report["schema_version"]
        == O_V8_PAPER_FRESH_POSITION_STATE_SCHEMA_VERSION
    )
    assert position_report["open_paper_position_count"] == 0
    assert position_report["exit_threshold_profile_name"] == (
        "paper_only_adapter_heuristic_v1"
    )
    assert position_report["exit_threshold_source"] == (
        "static_code_constants_for_paper_only_diagnostic_adapter_not_legacy_tuned"
    )
    assert position_report["exit_thresholds_tuned"] is False
    assert position_report["legacy_state_manager_reused"] is True
    assert position_report["legacy_decision_policy_reused"] is False
    assert position_report["exit_decision_policy_source"] == (
        "paper_only_adapter_heuristic_v1"
    )

    audit = result.legacy_position_policy_audit_report
    audit_payload = dict(audit)
    audit_id = audit_payload.pop(
        "o_v8_paper_fresh_legacy_position_policy_audit_report_id"
    )
    assert canonical_json_sha256(audit_payload) == audit_id
    assert (
        audit["schema_version"]
        == O_V8_PAPER_FRESH_LEGACY_POSITION_POLICY_AUDIT_SCHEMA_VERSION
    )
    assert audit["legacy_state_manager_reused"] is True
    assert audit["legacy_decision_policy_reused"] is False
    assert audit["reusable_legacy_decision_policy_found"] is False
    assert audit["exit_decision_policy_source"] == "paper_only_adapter_heuristic_v1"
    assert "src/bigan/execution/position_manager.py" in {
        row["module_or_script"] for row in audit["discovered_modules_and_functions"]
    }

    exit_report = result.paper_exit_signal_report
    exit_payload = dict(exit_report)
    exit_id = exit_payload.pop("o_v8_paper_fresh_exit_signal_report_id")
    assert canonical_json_sha256(exit_payload) == exit_id
    assert exit_report["schema_version"] == O_V8_PAPER_FRESH_EXIT_SIGNAL_SCHEMA_VERSION
    assert exit_report["paper_exit_signal_rows"][0]["paper_exit_decision"] == "NO_EXIT"
    assert exit_report["sell_position_intent_count"] == 0
    assert _read_jsonl(
        result.artifact_paths["fresh_paper_sell_position_intent_log"]
    ) == []
    assert result.manifest["paper_sell_position_intent_count"] == 0
    assert "fresh_legacy_position_policy_audit_report" in result.manifest[
        "artifact_hashes"
    ]
    assert result.manifest["fresh_legacy_position_policy_audit_report_id"] == (
        result.legacy_position_policy_audit_report[
            "o_v8_paper_fresh_legacy_position_policy_audit_report_id"
        ]
    )
    assert result.manifest["exit_thresholds_tuned"] is False
    assert result.manifest["legacy_decision_policy_reused"] is False
    assert result.manifest["paper_exit_adapter_mutates_o_entry_scorer"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_exit_adapter_holds_open_position(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-exit-hold",
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["executable_exit_bid_proxy"] = 0.41

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-exit-hold",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
            initial_paper_position_rows=(
                _paper_fresh_initial_position_row(
                    market_id="fresh-exit-hold",
                    side="UP",
                    entry_price=0.40,
                    entry_time=2_999_000,
                ),
            ),
        )
    )

    exit_rows = result.paper_exit_signal_report["paper_exit_signal_rows"]
    assert len(exit_rows) == 1
    assert exit_rows[0]["paper_exit_decision"] == "HOLD_POSITION"
    assert "paper_adapter_hold_position" in exit_rows[0]["exit_reason_codes"]
    assert "p_up" in exit_rows[0]["legacy_consumed_signal_fields"]
    assert "entry_price" in exit_rows[0]["legacy_consumed_position_fields"]
    assert exit_rows[0]["legacy_decision_policy_reused"] is False
    assert exit_rows[0]["exit_thresholds_tuned"] is False
    assert result.paper_exit_signal_report["sell_position_signal_count"] == 0
    assert result.synthetic_ledger_update_report["synthetic_ledger_update_count"] == 0
    assert result.runtime_safety_report["paper_fresh_runtime_safety_passed"] is True
    assert result.manifest["paper_exit_adapter_mutates_o_entry_scorer"] is False


def test_o_v8_paper_fresh_exit_adapter_sells_open_position_paper_only(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-exit-sell",
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["executable_exit_bid_proxy"] = 0.50

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-exit-sell",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
            initial_paper_position_rows=(
                _paper_fresh_initial_position_row(
                    market_id="fresh-exit-sell",
                    side="UP",
                    entry_price=0.40,
                    entry_time=2_999_000,
                ),
            ),
        )
    )

    exit_row = result.paper_exit_signal_report["paper_exit_signal_rows"][0]
    assert exit_row["paper_exit_decision"] == "SELL_POSITION"
    assert "paper_adapter_profit_target_crossed" in exit_row["exit_reason_codes"]
    assert exit_row["accepted_for_paper_exit_intent"] is True
    assert exit_row["exit_decision_policy_source"] == "paper_only_adapter_heuristic_v1"
    assert exit_row["uses_settlement_oracle_future_return_fields"] is False
    sell_intents = _read_jsonl(
        result.artifact_paths["fresh_paper_sell_position_intent_log"]
    )
    assert len(sell_intents) == 1
    assert sell_intents[0]["exit_thresholds_tuned"] is False
    assert sell_intents[0]["legacy_decision_policy_reused"] is False
    assert sell_intents[0]["paper_only"] is True
    assert sell_intents[0]["capital_at_risk"] is False
    assert sell_intents[0]["polymarket_write_enabled"] is False
    assert sell_intents[0]["wallet_signing_enabled"] is False
    ledger_report = result.synthetic_ledger_update_report
    assert (
        ledger_report["schema_version"]
        == O_V8_PAPER_FRESH_EXIT_LEDGER_UPDATE_SCHEMA_VERSION
    )
    assert ledger_report["synthetic_ledger_update_count"] == 1
    assert ledger_report["ledger_updates_only_for_accepted_paper_exit_intents"] is True
    assert result.runtime_safety_report[
        "paper_fresh_runtime_safety_checks"
    ]["exit_ledger_updates_only_accepted_sell_position_intents"]["passed"] is True
    assert result.manifest["paper_sell_position_intent_count"] == 1
    assert result.manifest["synthetic_exit_ledger_update_count"] == 1
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_exit_adapter_forbidden_outcome_fails_closed(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-exit-forbidden",
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["executable_exit_bid_proxy"] = 0.50
    initial_position = _paper_fresh_initial_position_row(
        market_id="fresh-exit-forbidden",
        side="UP",
        entry_price=0.40,
        entry_time=2_999_000,
    )
    initial_position["future_return"] = 0.25

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-exit-forbidden",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
            initial_paper_position_rows=(initial_position,),
        )
    )

    assert result.paper_position_state_report["position_state_adapter_status"] == (
        "blocked_fail_closed"
    )
    assert result.paper_exit_signal_report["forbidden_outcome_fields_present"] is True
    assert result.paper_exit_signal_report["sell_position_intent_count"] == 0
    assert result.synthetic_ledger_update_report["synthetic_ledger_update_count"] == 0
    assert _read_jsonl(
        result.artifact_paths["fresh_paper_sell_position_intent_log"]
    ) == []
    assert result.paper_exit_signal_report["v8_execution_handoff_allowed"] is False
    assert result.manifest["paper_sell_position_intent_count"] == 0
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_exit_adapter_invalid_initial_side_fails_closed(
    tmp_path: Path,
) -> None:
    result = _run_paper_fresh_exit_adapter_with_initial_positions(
        tmp_path,
        run_id="fresh-exit-invalid-side",
        initial_positions=(
            _paper_fresh_initial_position_row(
                market_id="fresh-exit-invalid-side",
                side="MAYBE",
                entry_price=0.40,
                entry_time=2_999_000,
            ),
        ),
    )

    assert result.paper_position_state_report["position_state_adapter_status"] == (
        "blocked_fail_closed"
    )
    assert result.paper_position_state_report["position_open_failed_count"] == 1
    failure = result.paper_position_state_report["position_open_failure_rows"][0]
    assert "position_open_invalid_side" in failure[
        "position_open_failure_reason_codes"
    ]
    assert result.paper_exit_signal_report["sell_position_intent_count"] == 0
    assert "position_open_failure_present_fail_closed" in result.paper_exit_signal_report[
        "paper_exit_signal_rows"
    ][0]["exit_reason_codes"]
    assert result.manifest["position_open_failed_count"] == 1
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_exit_adapter_invalid_entry_price_fails_closed(
    tmp_path: Path,
) -> None:
    result = _run_paper_fresh_exit_adapter_with_initial_positions(
        tmp_path,
        run_id="fresh-exit-invalid-price",
        initial_positions=(
            _paper_fresh_initial_position_row(
                market_id="fresh-exit-invalid-price",
                side="UP",
                entry_price=0.0,
                entry_time=2_999_000,
            ),
        ),
    )

    failure = result.paper_position_state_report["position_open_failure_rows"][0]
    assert "position_open_non_positive_entry_price" in failure[
        "position_open_failure_reason_codes"
    ]
    assert result.paper_position_state_report["open_paper_position_count"] == 0
    assert result.paper_exit_signal_report["sell_position_intent_count"] == 0
    assert result.synthetic_ledger_update_report["synthetic_ledger_update_count"] == 0
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_exit_adapter_duplicate_initial_event_id_fails_closed(
    tmp_path: Path,
) -> None:
    first = _paper_fresh_initial_position_row(
        market_id="fresh-exit-duplicate",
        side="UP",
        entry_price=0.40,
        entry_time=2_999_000,
    )
    second = dict(first)
    second["entry_price"] = 0.41
    result = _run_paper_fresh_exit_adapter_with_initial_positions(
        tmp_path,
        run_id="fresh-exit-duplicate",
        initial_positions=(first, second),
    )

    assert result.paper_position_state_report["position_open_failed_count"] == 1
    failure = result.paper_position_state_report["position_open_failure_rows"][0]
    assert "position_open_duplicate_event_id" in failure[
        "position_open_failure_reason_codes"
    ]
    assert result.paper_exit_signal_report["sell_position_intent_count"] == 0
    assert result.synthetic_ledger_update_report[
        "ledger_updates_only_for_accepted_paper_exit_intents"
    ] is True
    assert result.manifest["position_state_adapter_status"] == "blocked_fail_closed"
    assert result.manifest["source_model_candidate_eligible"] is False


def test_o_v8_paper_fresh_canonical_mapping_invalid_provenance_fails_closed(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    canonical_manifest_path = _write_canonical_o_source_fixture(
        tmp_path,
        prefer_no_trade=False,
    )
    bad_row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-canonical-bad-provenance",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    bad_row["reference_price_feature_provenance"] = {
        "provenance_valid": True,
        "decision_ts": bad_row["decision_ts"],
        "max_input_ts": bad_row["decision_ts"] + 1,
        "source_fields_used": ["future_reference_field"],
    }

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-canonical-invalid-provenance",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            canonical_o_source_manifest_path=canonical_manifest_path,
            public_data_cycles=((bad_row,),),
        )
    )

    mapping = result.canonical_feature_mapping_report
    assert mapping["canonical_feature_mapping_complete"] is False
    assert mapping["provenance_invalid_count"] == len(O_REQUIRED_DECISION_ACTION_FAMILIES)
    assert "provenance_invalid_for_mapped_features" in mapping[
        "canonical_feature_mapping_blocking_reason_codes"
    ]
    assert result.canonical_scorer_report["canonical_frozen_o_scorer_invoked"] is False
    assert result.canonical_scorer_alignment_report[
        "canonical_alignment_diagnostic_status"
    ] == "blocked_fail_closed"
    assert result.manifest["canonical_frozen_o_scorer_used"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_provider_feature_coverage_sparse_flag(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-sparse-feature-coverage",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            loop_mode="bounded_recurring",
            max_cycles=3,
            sleep_seconds=0.0,
            public_data_cycles=(
                (
                    _paper_fresh_public_row(
                        index=1,
                        market_id="fresh-sparse-market",
                        action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                        side="DOWN",
                        p_up=0.20,
                    ),
                ),
                (),
                (),
            ),
        )
    )

    report = result.provider_feature_coverage_report
    assert report["cycle_count"] == 3
    assert report["cycles_with_rows"] == 1
    assert report["idle_cycles"] == 2
    assert report["rows_per_cycle"] == [1, 0, 0]
    assert report["public_feature_row_count"] == 1
    assert report["sparse_provider_row_flag"] is True
    assert report["missing_runtime_field_count"] == 0
    assert report["provenance_invalid_count"] == 0
    assert "provider_feature_rows_below_minimum_diagnostic_density" in report[
        "sparse_provider_row_reason_codes"
    ]
    assert result.manifest["sparse_provider_row_flag"] is True
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_o_v8_paper_fresh_loop_bounded_recurring_cumulative_monitoring(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-bounded",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            loop_mode="bounded_recurring",
            max_cycles=2,
            sleep_seconds=0.0,
            public_data_cycles=(
                (
                    _paper_fresh_public_row(
                        index=1,
                        market_id="fresh-bounded-up",
                        action="BUY_UP_HOLD_TO_SETTLEMENT",
                        side="UP",
                        p_up=0.83,
                    ),
                    _paper_fresh_public_row(
                        index=2,
                        market_id="fresh-bounded-down",
                        action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                        side="DOWN",
                        p_up=0.24,
                    ),
                ),
                (
                    _paper_fresh_public_row(
                        index=3,
                        market_id="fresh-bounded-up-2",
                        action="BUY_UP_HOLD_TO_SETTLEMENT",
                        side="UP",
                        p_up=0.81,
                    ),
                ),
            ),
        )
    )

    assert result.fresh_loop_run_report["paper_fresh_loop_mode"] == "bounded_recurring"
    assert result.fresh_loop_run_report["paper_fresh_loop_cycle_count"] == 2
    assert result.fresh_loop_run_report["paper_fresh_order_intent_count"] == 3
    assert result.monitoring_report["cycle_count"] == 2
    assert result.monitoring_report["cycle_failure_count"] == 0
    assert len(result.monitoring_report["cycle_monitoring_reports"]) == 2
    assert result.cumulative_monitoring_report["total_cycles"] == 2
    assert result.cumulative_monitoring_report["total_paper_intents"] == 3
    assert result.cumulative_monitoring_report["total_paper_fills"] == 3
    assert result.cumulative_monitoring_report["safety_violation_count"] == 0
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False


def test_o_v8_paper_fresh_loop_default_uses_read_only_public_provider(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-provider",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_provider=_FakeFreshPublicProvider(),
        )
    )

    run_report = result.fresh_loop_run_report
    assert (
        run_report["paper_fresh_loop_public_data_source"]
        == O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
    )
    assert run_report["public_data_collection_report"][
        "public_provider_safety_passed"
    ] is True
    assert run_report["public_data_collection_report"][
        "paper_fresh_provider_collection_failed"
    ] is False
    assert run_report["public_data_collection_report"]["public_market_count"] == 1
    assert run_report["public_data_collection_report"]["public_orderbook_row_count"] == 2
    assert run_report["public_data_collection_report"]["public_trade_row_count"] == 1
    assert (
        run_report["public_data_collection_report"][
            "public_btc_feature_candle_row_count"
        ]
        == 1
    )
    assert run_report["uses_paper_intent_logs_as_fresh_public_data"] is False
    assert run_report["candidate_decision_count"] == 1
    assert run_report["guard_allowed_decision_count"] == 1
    assert run_report["paper_fresh_order_intent_count"] == 1
    assert result.monitoring_report["cycle_monitoring_reports"][0][
        "public_data_source"
    ] == O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
    assert result.manifest["paper_fresh_loop_public_data_source"] == (
        O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
    )
    assert result.manifest["uses_paper_intent_logs_as_fresh_public_data"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["capital_at_risk"] is False
    assert result.manifest["polymarket_write_enabled"] is False
    assert result.manifest["wallet_signing_enabled"] is False

    intents = _read_jsonl(result.artifact_paths["fresh_order_intent_log"])
    assert len(intents) == 1
    assert intents[0]["market_id"] == "provider-market-1"
    assert intents[0]["paper_only"] is True
    assert intents[0]["capital_at_risk"] is False


def test_o_v8_paper_fresh_loop_fails_closed_on_unlock_hash_mismatch(
    tmp_path: Path,
) -> None:
    unlock_dir, _unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-hash-mismatch",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256="0" * 64,
            public_data_cycles=(
                (
                    _paper_fresh_public_row(
                        index=1,
                        market_id="fresh-market-up",
                        action="BUY_UP_HOLD_TO_SETTLEMENT",
                        side="UP",
                        p_up=0.82,
                    ),
                ),
            ),
        )
    )

    assert result.fresh_loop_run_report["paper_fresh_loop_enabled"] is False
    assert "paper_candidate_unlock_manifest_hash_mismatch" in result.fresh_loop_run_report[
        "paper_fresh_loop_blocking_reason_codes"
    ]
    assert result.fresh_loop_run_report["paper_fresh_order_intent_count"] == 0
    assert result.monitoring_report["paper_fresh_monitoring_passed"] is False
    assert result.manifest["v8_paper_internal_handoff_allowed"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert _read_jsonl(result.artifact_paths["fresh_order_intent_log"]) == []


def test_o_v8_paper_fresh_loop_fails_closed_on_public_provider_error(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-provider-error",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_provider=_FailingFreshPublicProvider(),
        )
    )

    run_report = result.fresh_loop_run_report
    assert (
        run_report["paper_fresh_loop_public_data_source"]
        == O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
    )
    assert run_report["paper_fresh_provider_collection_failed"] is True
    assert "paper_fresh_public_provider_collection_failed" in run_report[
        "paper_fresh_loop_blocking_reason_codes"
    ]
    assert "test_public_provider_unavailable" in run_report[
        "paper_fresh_loop_blocking_reason_codes"
    ]
    assert run_report["candidate_decision_count"] == 0
    assert run_report["paper_fresh_order_intent_count"] == 0
    assert result.monitoring_report["paper_fresh_monitoring_passed"] is False
    assert result.manifest["paper_fresh_provider_collection_failed"] is True
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["source_model_candidate_eligible"] is False
    assert result.manifest["freeze_ready"] is False
    assert result.manifest["promotion_evidence_eligible"] is False
    assert _read_jsonl(result.artifact_paths["fresh_order_intent_log"]) == []


def test_o_v8_paper_fresh_loop_records_failed_public_data_cycle_fail_closed(
    tmp_path: Path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    bad_row = _paper_fresh_public_row(
        index=1,
        market_id="fresh-bad-row",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    bad_row["realized_pnl"] = 1.0

    result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id="fresh-forbidden-field",
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((bad_row,),),
        )
    )

    assert result.fresh_loop_run_report["paper_fresh_loop_enabled"] is True
    assert "paper_fresh_public_data_cycle_failed" in result.fresh_loop_run_report[
        "paper_fresh_loop_blocking_reason_codes"
    ]
    assert result.fresh_loop_run_report["paper_fresh_order_intent_count"] == 0
    assert result.monitoring_report["cycle_failure_count"] == 1
    assert result.monitoring_report["cycle_monitoring_reports"][0][
        "cycle_failure_reason_codes"
    ] == ["fresh_public_data_forbidden_outcome_fields_present"]
    assert result.cumulative_monitoring_report["safety_violation_count"] == 1
    assert result.manifest["paper_fresh_monitoring_passed"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["source_model_candidate_eligible"] is False
    assert result.manifest["freeze_ready"] is False
    assert result.manifest["promotion_evidence_eligible"] is False


class _FakeFreshPublicProvider:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def market_rows(self, config: Any) -> list[dict[str, Any]]:
        del config
        return [
            {
                "market_id": "provider-market-1",
                "condition_id": "provider-market-1",
                "slug": "btc-updown-5m-3000",
                "market_family": "btc_updown_5m",
                "horizon_ms": 300_000,
                "market_start_ts": 3_000_000,
                "market_end_ts": 3_300_000,
                "settlement_ts": 3_360_000,
                "up_token_id": "provider-up-token",
                "down_token_id": "provider-down-token",
                "reference_price_source": "polymarket_official_btc_usd_reference",
                "reference_price_start": 65_000.0,
                "reference_price_at_start": 65_000.0,
                "settlement_rule": "BTC 5m UP/DOWN public provider fixture",
                "raw_market_sha256": "1" * 64,
                "paper_only": True,
                "capital_at_risk": False,
                "broker_exchange_write_enabled": False,
                "live_exchange_write_enabled": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        ]

    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: Any,
    ) -> list[dict[str, Any]]:
        del markets, config
        common = {
            "market_id": "provider-market-1",
            "ts": 3_060_000,
            "available_at_ts": 3_060_000,
            "bid_size": 2.0,
            "ask_size": 2.0,
            "liquidity_depth": 4.0,
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }
        return [
            {
                **common,
                "token_id": "provider-up-token",
                "outcome": "UP",
                "bid_price": 0.64,
                "ask_price": 0.66,
                "mid_price": 0.65,
            },
            {
                **common,
                "token_id": "provider-down-token",
                "outcome": "DOWN",
                "bid_price": 0.10,
                "ask_price": 0.12,
                "mid_price": 0.11,
            },
        ]

    def trade_rows(
        self,
        markets: list[dict[str, Any]],
        config: Any,
    ) -> list[dict[str, Any]]:
        del markets, config
        return [
            {
                "market_id": "provider-market-1",
                "token_id": "provider-up-token",
                "outcome": "UP",
                "ts": 3_050_000,
                "available_at_ts": 3_050_000,
                "price": 0.63,
                "size": 1.0,
                "side": "BUY",
                "paper_only": True,
                "capital_at_risk": False,
                "broker_exchange_write_enabled": False,
                "live_exchange_write_enabled": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        ]

    def btc_feature_candle_rows(
        self,
        markets: list[dict[str, Any]],
        config: Any,
    ) -> list[dict[str, Any]]:
        del markets, config
        return [
            {
                "ts": 3_000_000,
                "close_time": 3_060_000,
                "available_at_ts": 3_060_000,
                "open_price": 65_000.0,
                "high_price": 65_100.0,
                "low_price": 64_900.0,
                "close_price": 65_050.0,
                "volume": 10.0,
                "timeframe_ms": 60_000,
                "source": "coinbase_btc_usd",
            }
        ]

    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: Any,
    ) -> list[dict[str, Any]]:
        del markets, config
        return []


class _FailingFreshPublicProvider(_FakeFreshPublicProvider):
    def market_rows(self, config: Any) -> list[dict[str, Any]]:
        del config
        raise RealCorpusPublicProviderError(
            "test provider unavailable",
            reason_codes=("test_public_provider_unavailable",),
        )


def _build_issue160_unlock_fixture(tmp_path: Path) -> tuple[Path, str]:
    issue159_dir, expected_hashes = _write_issue159_paper_unlock_fixture(
        tmp_path / "issue159"
    )
    unlock = run_polymarket_o_v8_paper_candidate_unlock(
        PolymarketOV8PaperCandidateUnlockConfig(
            run_id="issue160-unlock-fixture",
            output_dir=tmp_path / "issue160",
            issue_159_eval_dir=issue159_dir,
            expected_issue_159_hashes=expected_hashes,
            manual_approval_approved=True,
            manual_approval_id="manual-approval-test",
            manual_approval_operator="pytest",
        )
    )
    return unlock.output_dir, unlock.artifact_hashes["manifest"]


def _paper_fresh_public_row(
    *,
    index: int,
    market_id: str,
    action: str,
    side: str,
    p_up: float,
) -> dict[str, Any]:
    decision_ts = 3_000_000 + index
    score = 1.2 + index / 100.0
    family = (
        "NO_TRADE"
        if action == "NO_TRADE"
        else (
            "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "HOLD_TO_SETTLEMENT"
        )
    )
    return {
        "decision_group_id": f"fresh-public|{market_id}|{decision_ts}",
        "market_id": market_id,
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": side,
        "selected_action_family": family,
        "corrected_model_score": score,
        "raw_model_score": 20.0 + index,
        "high_score_flag": True,
        "p_up": p_up,
        "p_down": 1.0 - p_up,
        "p_up_action_disagreement": False,
        "microstructure_snapshot": {
            "entry_ask": 0.42 if side == "UP" else 0.58,
            "executable_exit_bid_proxy": 0.41 if side == "UP" else 0.57,
            "spread_bps": 120.0,
            "book_staleness_ms": 250.0,
            "queue_fill_proxy": 0.92,
            "time_to_close_seconds": 240.0,
        },
        "reference_price_feature_provenance": {
            "provenance_valid": True,
            "decision_ts": decision_ts,
            "max_input_ts": decision_ts - 100,
            "source_fields_used": ["test_public_read_only_provider"],
        },
        "decision_time_feature_max_input_ts": decision_ts - 100,
        "full_5_action_ranking": [
            {
                "selected_action": candidate,
                "selected_side": "NONE"
                if candidate == "NO_TRADE"
                else ("UP" if "BUY_UP" in candidate else "DOWN"),
                "selected_action_family": (
                    "NO_TRADE"
                    if candidate == "NO_TRADE"
                    else (
                        "SELL_BEFORE_CLOSE"
                        if candidate.endswith("SELL_BEFORE_CLOSE")
                        else "HOLD_TO_SETTLEMENT"
                    )
                ),
                "corrected_model_score": score
                if candidate == action
                else score - 0.20 - 0.01 * candidate_index,
                "raw_model_score": 20.0 + index,
            }
            for candidate_index, candidate in enumerate(
                O_REQUIRED_DECISION_ACTION_FAMILIES
            )
        ],
    }


def _paper_fresh_initial_position_row(
    *,
    market_id: str,
    side: str,
    entry_price: float,
    entry_time: int,
) -> dict[str, Any]:
    return {
        "event_id": f"initial-{market_id}-{side}",
        "market_id": market_id,
        "symbol": f"POLYMARKET:{market_id}:{side}",
        "side": side,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "fill_price": entry_price,
        "size": 0.2,
        "order_id": f"initial-order-{market_id}-{side}",
        "sleeve": "volatility",
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _run_paper_fresh_exit_adapter_with_initial_positions(
    tmp_path: Path,
    *,
    run_id: str,
    initial_positions: tuple[dict[str, Any], ...],
):
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id=str(initial_positions[0].get("market_id") or run_id),
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["executable_exit_bid_proxy"] = 0.50
    return run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
            run_id=run_id,
            output_dir=tmp_path / "fresh",
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
            initial_paper_position_rows=initial_positions,
        )
    )


def _write_canonical_o_source_fixture(
    tmp_path: Path,
    *,
    prefer_no_trade: bool,
    preferred_action: str | None = None,
) -> Path:
    source_dir = tmp_path / (
        "canonical-o-no-trade" if prefer_no_trade else "canonical-o-buy-up"
    )
    source_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(O_DEPLOYABLE_MODEL_FEATURE_NAMES)
    coefficients = dict.fromkeys(feature_names, 0.0)
    selected_action = preferred_action or (
        "NO_TRADE" if prefer_no_trade else "BUY_UP_HOLD_TO_SETTLEMENT"
    )
    for action in O_REQUIRED_DECISION_ACTION_FAMILIES:
        feature_name = f"action_{action.lower()}"
        coefficients[feature_name] = 3.0 if action == selected_action else -1.0
    if selected_action != "NO_TRADE":
        coefficients["p_up"] = 0.25
    ranking_correction_config = _canonical_o_source_fixture_correction_config(
        prefer_no_trade=prefer_no_trade,
    )
    ranking_report = {
        "o_model_training_summary": {
            "feature_names": feature_names,
            "coefficients_by_feature": coefficients,
            "ranking_correction_config": ranking_correction_config,
            "correction_config_hash": ranking_correction_config[
                "correction_config_hash"
            ],
            "selected_feature_set_name": "pytest_canonical_feature_schema",
            "selected_correction_policy_name": "pytest_frozen_correction",
            "selected_high_score_threshold_profile_name": "pytest_threshold",
            "deployable_model_score_available": True,
        }
    }
    ranking_report_path = source_dir / "o_source_ranking_objective_report.json"
    ranking_report_path.write_text(
        json.dumps(ranking_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_manifest = {
        "run_id": "pytest-canonical-o-source",
        "artifact_paths": {
            "ranking_objective_report": ranking_report_path.name,
        },
        "artifact_hashes": {
            "ranking_objective_report": _file_sha256_for_test(ranking_report_path),
        },
    }
    source_manifest_path = source_dir / "o_replay_aligned_source_ranking_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source_manifest_path


def _canonical_o_source_fixture_correction_config(
    *,
    prefer_no_trade: bool,
) -> dict[str, Any]:
    correction_config_hash = (
        "pytest-no-trade-correction"
        if prefer_no_trade
        else "pytest-buy-up-correction"
    )
    no_trade_base = 3.0 if prefer_no_trade else -1.0
    trade_base = -1.0 if prefer_no_trade else 0.0
    return {
        "correction_config_hash": correction_config_hash,
        "correction_constants_source": "pytest_shadow_split_only",
        "probe_constants_source": "pytest_shadow_split_only",
        "weak_opportunity_p_edge_cutoff": 0.10,
        "no_trade_base_score": no_trade_base,
        "trade_base_score": trade_base,
        "sell_before_close_base_score": -0.25,
        "confidence_bonus": 0.0,
        "weak_opportunity_trade_penalty": 0.0,
        "sell_before_close_confidence_bonus": 0.0,
        "sell_before_close_weak_penalty": 0.0,
        "action_shadow_priors": dict.fromkeys(O_REQUIRED_DECISION_ACTION_FAMILIES, 0.0),
        "action_family_shadow_priors": {
            "HOLD_TO_SETTLEMENT": 0.0,
            "SELL_BEFORE_CLOSE": 0.0,
            "NO_TRADE": 0.0,
        },
        "group_normalized_raw_model_weight": 1.0,
        "p_up_misalignment_raw_positive_penalty": 0.0,
        "large_regret_reversal_guard_enabled": False,
        "large_regret_reversal_penalty": 0.0,
        "hts_p_up_reliability_guard_enabled": False,
        "hts_p_up_reliability_penalty": 0.0,
        "hts_p_up_reliability_no_trade_buffer_enabled": False,
        "hts_p_up_reliability_no_trade_buffer": 0.0,
        "shadow_action_family_prior_weight": 0.0,
        "microstructure_quality_weight": 0.0,
        "high_score_calibration": {"high_score_threshold": -999.0},
        "hts_p_up_reliability_bucket_thresholds": {
            "p_up_confidence": {"q25": 0.05, "median": 0.10, "q75": 0.20},
            "time_to_close": {"q25": 1.0, "median": 3.0, "q75": 5.0},
            "spread": {"q25": 0.01, "median": 0.05, "q75": 0.10},
            "queue": {"q25": 0.25, "median": 0.50, "q75": 0.75},
            "staleness": {"q25": 1.0, "median": 3.0, "q75": 5.0},
        },
    }


def _write_issue159_paper_unlock_fixture(
    run_dir: Path,
    *,
    upstream_capital_at_risk: bool = False,
) -> tuple[Path, dict[str, str]]:
    run_dir.mkdir(parents=True)
    common = {
        "paper_only": True,
        "capital_at_risk": upstream_capital_at_risk,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "thresholds_tuned": False,
        "uses_validation_outcomes_for_tuning": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
    }
    reports = {
        "source_manifest": {
            **common,
            "run_id": "issue159-fixture",
            "future_unseen_holdout_simulated_allowed_order_count": 2,
            "future_unseen_holdout_raw_collection_ready": True,
            "future_window_time_validation_passed": True,
            "future_unseen_holdout_policy_readiness_passed": True,
            "future_unseen_holdout_handoff_gate_passed": True,
            "future_unseen_holdout_paper_candidate_gate_passed": True,
            "paper_candidate_allowed": False,
        },
        "raw_collection_manifest": {
            **common,
            "future_unseen_holdout_raw_collection_ready": True,
            "future_window_time_validation_passed": True,
            "future_unseen_holdout_raw_collection_required_checks": {
                "no_overlap_with_prior_replay_validation_shadow": {
                    "passed": True,
                    "observed": {"decision_group_overlap": [], "market_overlap": []},
                    "reason_code": "future_holdout_overlap_with_prior_data",
                }
            },
            "paper_candidate_allowed": False,
        },
        "execution_replay_report": {
            **common,
            "zero_missing_runtime_fields": True,
            "zero_provenance_violations": True,
            "simulated_allowed_order_count": 2,
            "blocked_decision_count": 1,
            "future_unseen_holdout_execution_replay_ready": True,
            "paper_candidate_allowed": False,
            "derived_reports": {
                "simulated_order_replay": {
                    "simulated_decision_rows": [
                        _paper_unlock_simulated_row(
                            index=1,
                            market_id="market-up",
                            action="BUY_UP_HOLD_TO_SETTLEMENT",
                            side="UP",
                            p_up=0.82,
                            order_allowed=True,
                        ),
                        _paper_unlock_simulated_row(
                            index=2,
                            market_id="market-down",
                            action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                            side="DOWN",
                            p_up=0.21,
                            order_allowed=True,
                        ),
                        _paper_unlock_simulated_row(
                            index=3,
                            market_id="market-blocked",
                            action="BUY_UP_HOLD_TO_SETTLEMENT",
                            side="UP",
                            p_up=0.78,
                            order_allowed=False,
                        ),
                    ],
                }
            },
        },
        "policy_readiness_report": {
            **common,
            "future_unseen_holdout_policy_readiness_passed": True,
            "allowed_order_count": 2,
            "min_allowed_order_count": 2,
            "paper_candidate_allowed": False,
        },
        "handoff_gate_report": {
            **common,
            "future_unseen_holdout_handoff_gate_passed": True,
            "derived_explicit_handoff_gate_passed": True,
            "paper_candidate_allowed": False,
        },
        "paper_candidate_gate_report": {
            **common,
            "future_unseen_holdout_paper_candidate_gate_passed": True,
            "future_paper_candidate_gate_required": True,
            "paper_candidate_allowed": False,
        },
    }
    expected_hashes: dict[str, str] = {}
    for name, filename in PINNED_ISSUE_159_ARTIFACT_FILENAMES.items():
        path = run_dir / filename
        path.write_text(
            json.dumps(reports[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected_hashes[name] = _file_sha256_for_test(path)
    return run_dir, expected_hashes


def _paper_unlock_simulated_row(
    *,
    index: int,
    market_id: str,
    action: str,
    side: str,
    p_up: float,
    order_allowed: bool,
) -> dict[str, Any]:
    decision_ts = 2_000_000 + index
    return {
        "decision_group_id": f"fixture|{market_id}|{decision_ts}",
        "market_id": market_id,
        "decision_ts": decision_ts,
        "simulated_order_id": f"sim-v8-o-{index:06d}" if order_allowed else None,
        "source_selected_action": action,
        "source_selected_family": "HOLD_TO_SETTLEMENT",
        "source_selected_side": side,
        "execution_guarded_action": action,
        "execution_guarded_family": "HOLD_TO_SETTLEMENT",
        "execution_guarded_side": side,
        "source_model_score": 1.0 + index / 10.0,
        "execution_guarded_score": 1.0 + index / 10.0,
        "source_raw_model_score": 10.0 + index,
        "p_up": p_up,
        "p_down": 1.0 - p_up,
        "p_up_action_disagreement": False,
        "order_allowed": order_allowed,
        "fail_closed": not order_allowed,
        "proposed_order_size": 0.2 if order_allowed else 0.0,
        "microstructure_snapshot": {
            "entry_ask": 0.44 if side == "UP" else 0.56,
            "executable_exit_bid_proxy": 0.43 if side == "UP" else 0.55,
            "spread_bps": 100.0 + index,
            "book_staleness_ms": 200.0,
            "queue_fill_proxy": 0.9,
            "time_to_close_seconds": 180.0,
        },
        "pre_decision_exposure_state": {
            "current_total_exposure": 0.0,
            "current_side_exposure_by_side": {"UP": 0.0, "DOWN": 0.0},
        },
        "post_decision_exposure_state": {
            "current_total_exposure": 0.2 if order_allowed else 0.0,
            "current_side_exposure_by_side": {
                "UP": 0.2 if order_allowed and side == "UP" else 0.0,
                "DOWN": 0.2 if order_allowed and side == "DOWN" else 0.0,
            },
        },
        "execution_guard_reason_codes": [],
        "execution_blocking_reason_codes": []
        if order_allowed
        else ["execution_total_exposure_limit_reached"],
        "sizing_reason_codes": ["execution_base_size_applied"]
        if order_allowed
        else ["execution_blocked_size_zero"],
        "source_score_mutated": False,
        "o_model_predicted_score_mutated": False,
    }


def _file_sha256_for_test(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
