"""Run diagnostic O replay-aligned source-ranking reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (  # noqa: E402
    PolymarketOReplayAlignedSourceRankingConfig,
    run_polymarket_o_replay_aligned_source_ranking,
)


def run_polymarket_o_replay_aligned_source_ranking_cli(
    *,
    m2_candidate_report_path: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_o_replay_aligned_source_ranking",
    overwrite_existing: bool = False,
    future_holdout_raw_manifest_path: Path | str | None = None,
) -> dict:
    result = run_polymarket_o_replay_aligned_source_ranking(
        PolymarketOReplayAlignedSourceRankingConfig(
            m2_candidate_report_path=m2_candidate_report_path,
            output_dir=output_dir,
            run_id=run_id,
            overwrite_existing=overwrite_existing,
            future_holdout_raw_manifest_path=future_holdout_raw_manifest_path,
        )
    )
    labels = result.label_construction_report
    ranking = result.ranking_objective_report
    leakage = result.leakage_audit_report
    comparison = result.candidate_comparison_report
    gate = result.source_model_eligibility_gate_report
    freeze = result.freeze_readiness_report
    handoff = result.v8_action_rank_handoff_report
    execution_guard = result.v8_execution_risk_guard_report
    runtime_state = result.v8_execution_runtime_state_report
    simulated_replay = result.v8_execution_simulated_order_replay_report
    allowed_quality = result.v8_execution_allowed_order_quality_report
    policy_readiness = result.v8_execution_policy_readiness_report
    block_analysis = result.v8_execution_guard_block_analysis_report
    field_coverage = result.v8_execution_runtime_field_coverage_report
    handoff_gate = result.v8_execution_handoff_gate_report
    holdout_plan = result.v8_future_unseen_holdout_plan_report
    paper_candidate_gate = result.v8_paper_candidate_gate_design_report
    collection_plan = result.v8_future_unseen_holdout_collection_plan_report
    future_raw = result.v8_future_unseen_holdout_raw_collection_manifest
    future_freeze = result.v8_future_unseen_holdout_input_freeze_manifest
    future_action_rank = result.v8_future_unseen_holdout_action_rank_report
    future_execution = result.v8_future_unseen_holdout_execution_replay_report
    future_policy = result.v8_future_unseen_holdout_policy_readiness_report
    future_handoff = result.v8_future_unseen_holdout_handoff_gate_report
    future_paper_gate = result.v8_future_unseen_holdout_paper_candidate_gate_report
    return {
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "candidate_name": labels["candidate_name"],
        "label_row_count": labels["row_count"],
        "label_gap_before": labels["label_gap_before"],
        "label_gap_after": labels["label_gap_after"],
        "label_gap_delta": labels["label_gap_delta"],
        "primary_variant_name": ranking["primary_variant_name"],
        "selected_feature_set_name": ranking["selected_feature_set_name"],
        "selected_correction_policy_name": ranking[
            "selected_correction_policy_name"
        ],
        "selected_high_score_threshold_profile_name": ranking[
            "selected_high_score_threshold_profile_name"
        ],
        "selected_joint_candidate_name": ranking["selected_joint_candidate_name"],
        "top1_hit_rate": ranking["top1_realized_best_action_hit_rate"],
        "top2_hit_rate": ranking["top2_realized_best_action_hit_rate"],
        "top3_hit_rate": ranking["top3_realized_best_action_hit_rate"],
        "mean_regret": ranking["mean_regret"],
        "validation_top1_hit_rate": gate["top1_realized_best_action_hit_rate"],
        "validation_mean_regret": gate["mean_regret"],
        "validation_high_score_support_count": gate["high_score_support_count"],
        "validation_p_up_disagreement_rate": gate[
            "p_up_action_disagreement_summary"
        ]["candidate_scoped_p_up_action_disagreement_rate"],
        "gate_reason_code_consistency_passed": gate[
            "gate_reason_code_consistency_passed"
        ],
        "v8_action_rank_quality_passed": gate["v8_action_rank_quality_passed"],
        "v8_action_rank_candidate_eligible": gate[
            "v8_action_rank_candidate_eligible"
        ],
        "v8_execution_risk_control_required": gate[
            "v8_execution_risk_control_required"
        ],
        "v8_execution_handoff_allowed": gate["v8_execution_handoff_allowed"],
        "strict_source_gate_remains_failed": gate[
            "strict_source_gate_remains_failed"
        ],
        "v8_selected_action_handoff_row_count": handoff[
            "selected_action_handoff_row_count"
        ],
        "v8_execution_risk_guard_decision_count": execution_guard[
            "execution_guard_decision_count"
        ],
        "v8_execution_guard_order_allowed_count": execution_guard[
            "order_allowed_count"
        ],
        "v8_execution_guard_fail_closed_decision_count": execution_guard[
            "fail_closed_decision_count"
        ],
        "v8_execution_guard_runtime_validation_passed": execution_guard[
            "runtime_risk_control_validation_passed"
        ],
        "v8_execution_runtime_state_validation_passed": runtime_state[
            "runtime_state_validation_passed"
        ],
        "v8_execution_simulated_allowed_order_count": simulated_replay[
            "simulated_allowed_order_count"
        ],
        "v8_execution_simulated_blocked_decision_count": simulated_replay[
            "blocked_decision_count"
        ],
        "v8_execution_simulated_total_proposed_notional": simulated_replay[
            "total_proposed_notional"
        ],
        "v8_execution_simulated_runtime_risk_control_validation_passed": (
            simulated_replay["runtime_risk_control_validation_passed"]
        ),
        "v8_execution_allowed_order_quality_allowed_order_count": allowed_quality[
            "allowed_order_count"
        ],
        "v8_execution_allowed_order_quality_blocked_decision_count": allowed_quality[
            "blocked_decision_count"
        ],
        "v8_execution_allowed_order_quality_recommendation_counts": allowed_quality[
            "deterministic_recommendation_counts"
        ],
        "v8_execution_allowed_order_quality_origin_distribution": allowed_quality[
            "allowed_order_origin_distribution"
        ],
        "v8_execution_policy_readiness_diagnostic_passed": policy_readiness[
            "execution_policy_readiness_diagnostic_passed"
        ],
        "v8_execution_policy_readiness_blocking_reason_codes": policy_readiness[
            "execution_policy_readiness_blocking_reason_codes"
        ],
        "v8_execution_policy_readiness_required_checks": policy_readiness[
            "execution_policy_readiness_required_checks"
        ],
        "future_explicit_execution_handoff_gate_required": policy_readiness[
            "future_explicit_execution_handoff_gate_required"
        ],
        "explicit_execution_handoff_gate_passed": handoff_gate[
            "explicit_execution_handoff_gate_passed"
        ],
        "explicit_execution_handoff_gate_mode": handoff_gate[
            "explicit_execution_handoff_gate_mode"
        ],
        "explicit_execution_handoff_blocking_reason_codes": handoff_gate[
            "explicit_execution_handoff_blocking_reason_codes"
        ],
        "future_unseen_holdout_required": handoff_gate[
            "future_unseen_holdout_required"
        ],
        "future_paper_candidate_gate_required": handoff_gate[
            "future_paper_candidate_gate_required"
        ],
        "future_unseen_holdout_plan_ready": holdout_plan[
            "future_unseen_holdout_plan_ready"
        ],
        "future_unseen_holdout_blocking_reason_codes": holdout_plan[
            "future_unseen_holdout_blocking_reason_codes"
        ],
        "paper_candidate_gate_design_ready": paper_candidate_gate[
            "paper_candidate_gate_design_ready"
        ],
        "paper_candidate_gate_blocking_reason_codes": paper_candidate_gate[
            "paper_candidate_gate_blocking_reason_codes"
        ],
        "paper_candidate_allowed": paper_candidate_gate["paper_candidate_allowed"],
        "future_unseen_holdout_collection_plan_ready": collection_plan[
            "future_unseen_holdout_collection_plan_ready"
        ],
        "future_unseen_holdout_collection_blocking_reason_codes": collection_plan[
            "future_unseen_holdout_collection_blocking_reason_codes"
        ],
        "future_unseen_holdout_collection_status": collection_plan[
            "collection_status"
        ],
        "future_outcome_evaluation_generated": collection_plan[
            "future_outcome_evaluation_generated"
        ],
        "future_unseen_holdout_raw_collection_status": future_raw[
            "collection_status"
        ],
        "future_unseen_holdout_raw_collection_ready": future_raw[
            "future_unseen_holdout_raw_collection_ready"
        ],
        "future_unseen_holdout_raw_collection_blocking_reason_codes": future_raw[
            "future_unseen_holdout_raw_collection_blocking_reason_codes"
        ],
        "future_unseen_holdout_input_freeze_ready": future_freeze[
            "future_unseen_holdout_input_freeze_ready"
        ],
        "future_unseen_holdout_input_freeze_blocking_reason_codes": future_freeze[
            "future_unseen_holdout_input_freeze_blocking_reason_codes"
        ],
        "future_unseen_holdout_action_rank_ready": future_action_rank[
            "future_unseen_holdout_action_rank_ready"
        ],
        "future_unseen_holdout_prediction_attempted": future_action_rank[
            "prediction_attempted"
        ],
        "future_unseen_holdout_execution_replay_ready": future_execution[
            "future_unseen_holdout_execution_replay_ready"
        ],
        "future_unseen_holdout_execution_replay_attempted": future_execution[
            "execution_replay_attempted"
        ],
        "future_unseen_holdout_simulated_allowed_order_count": future_execution[
            "simulated_allowed_order_count"
        ],
        "future_unseen_holdout_policy_readiness_passed": future_policy[
            "future_unseen_holdout_policy_readiness_passed"
        ],
        "future_unseen_holdout_handoff_gate_passed": future_handoff[
            "future_unseen_holdout_handoff_gate_passed"
        ],
        "future_unseen_holdout_paper_candidate_gate_passed": future_paper_gate[
            "future_unseen_holdout_paper_candidate_gate_passed"
        ],
        "future_unseen_holdout_paper_candidate_gate_blocking_reason_codes": (
            future_paper_gate[
                "future_unseen_holdout_paper_candidate_gate_blocking_reason_codes"
            ]
        ),
        "v8_execution_block_analysis_safe_order_candidate_count": (
            block_analysis["safe_order_discovery_summary"][
                "safe_order_candidate_count"
            ]
        ),
        "v8_execution_block_analysis_fundamentally_unsafe_count": (
            block_analysis["safe_order_discovery_summary"][
                "fundamentally_unsafe_count"
            ]
        ),
        "v8_execution_block_analysis_primary_blocker_categories": block_analysis[
            "primary_blocker_categories"
        ],
        "v8_execution_runtime_field_missing_decision_count": field_coverage[
            "missing_runtime_field_decision_count"
        ],
        "v8_execution_runtime_field_true_data_gap_count": field_coverage[
            "classification_counts"
        ]["true_data_coverage_gap"],
        "v8_execution_runtime_field_safe_backfill_candidate_count": field_coverage[
            "safe_backfill_candidate_count"
        ],
        "v8_execution_runtime_field_existing_handoff_backfill_candidate_count": (
            field_coverage["existing_handoff_backfill_candidate_count"]
        ),
        "v8_execution_runtime_field_decision_time_data_join_backfill_candidate_count": (
            field_coverage["decision_time_data_join_backfill_candidate_count"]
        ),
        "v8_execution_runtime_field_optional_for_no_trade_count": field_coverage[
            "classification_counts"
        ]["optional_for_no_trade"],
        "v8_execution_runtime_field_simulation_policy_too_strict_count": (
            field_coverage["classification_counts"][
                "too_strict_for_simulation_only_mode"
            ]
        ),
        "v8_execution_runtime_field_primary_missing_fields": field_coverage[
            "primary_missing_runtime_fields"
        ],
        "v8_execution_runtime_field_backfill_rules_applied": field_coverage[
            "runtime_field_backfill_rules_applied"
        ],
        "v8_execution_runtime_field_applied_backfill_count": field_coverage[
            "applied_runtime_field_backfill_count"
        ],
        "v8_execution_runtime_field_applied_backfill_rule_counts": field_coverage[
            "applied_runtime_field_backfill_rule_counts"
        ],
        "v8_execution_runtime_field_backfill_provenance_validity_summary": (
            field_coverage["runtime_field_backfill_provenance_validity_summary"]
        ),
        "source_model_candidate_eligible": gate["source_model_candidate_eligible"],
        "freeze_ready": freeze["freeze_ready"],
        "leakage_audit_passed": leakage["leakage_audit_passed"],
        "eligible_candidate_count": comparison["eligible_candidate_count"],
        "#146_start_allowed": labels["#146_start_allowed"],
        "#134_resume_allowed": labels["#134_resume_allowed"],
        "label_construction_report_path": str(
            result.artifact_paths["label_construction_report"]
        ),
        "ranking_objective_report_path": str(
            result.artifact_paths["ranking_objective_report"]
        ),
        "leakage_audit_report_path": str(result.artifact_paths["leakage_audit_report"]),
        "candidate_comparison_report_path": str(
            result.artifact_paths["candidate_comparison_report"]
        ),
        "source_model_eligibility_gate_report_path": str(
            result.artifact_paths["source_model_eligibility_gate_report"]
        ),
        "freeze_readiness_report_path": str(
            result.artifact_paths["freeze_readiness_report"]
        ),
        "feature_set_selection_report_path": str(
            result.artifact_paths["feature_set_selection_report"]
        ),
        "joint_feature_correction_selection_report_path": str(
            result.artifact_paths["joint_feature_correction_selection_report"]
        ),
        "v8_action_rank_handoff_report_path": str(
            result.artifact_paths["v8_action_rank_handoff_report"]
        ),
        "v8_execution_risk_guard_report_path": str(
            result.artifact_paths["v8_execution_risk_guard_report"]
        ),
        "v8_execution_runtime_state_report_path": str(
            result.artifact_paths["v8_execution_runtime_state_report"]
        ),
        "v8_execution_simulated_order_replay_report_path": str(
            result.artifact_paths["v8_execution_simulated_order_replay_report"]
        ),
        "v8_execution_allowed_order_quality_report_path": str(
            result.artifact_paths["v8_execution_allowed_order_quality_report"]
        ),
        "v8_execution_policy_readiness_report_path": str(
            result.artifact_paths["v8_execution_policy_readiness_report"]
        ),
        "v8_execution_guard_block_analysis_report_path": str(
            result.artifact_paths["v8_execution_guard_block_analysis_report"]
        ),
        "v8_execution_runtime_field_coverage_report_path": str(
            result.artifact_paths["v8_execution_runtime_field_coverage_report"]
        ),
        "v8_execution_handoff_gate_report_path": str(
            result.artifact_paths["v8_execution_handoff_gate_report"]
        ),
        "v8_future_unseen_holdout_plan_report_path": str(
            result.artifact_paths["v8_future_unseen_holdout_plan_report"]
        ),
        "v8_paper_candidate_gate_design_report_path": str(
            result.artifact_paths["v8_paper_candidate_gate_design_report"]
        ),
        "v8_future_unseen_holdout_collection_plan_report_path": str(
            result.artifact_paths["v8_future_unseen_holdout_collection_plan_report"]
        ),
        "v8_future_unseen_holdout_raw_collection_manifest_path": str(
            result.artifact_paths["v8_future_unseen_holdout_raw_collection_manifest"]
        ),
        "v8_future_unseen_holdout_input_freeze_manifest_path": str(
            result.artifact_paths["v8_future_unseen_holdout_input_freeze_manifest"]
        ),
        "v8_future_unseen_holdout_action_rank_report_path": str(
            result.artifact_paths["v8_future_unseen_holdout_action_rank_report"]
        ),
        "v8_future_unseen_holdout_execution_replay_report_path": str(
            result.artifact_paths["v8_future_unseen_holdout_execution_replay_report"]
        ),
        "v8_future_unseen_holdout_policy_readiness_report_path": str(
            result.artifact_paths["v8_future_unseen_holdout_policy_readiness_report"]
        ),
        "v8_future_unseen_holdout_handoff_gate_report_path": str(
            result.artifact_paths["v8_future_unseen_holdout_handoff_gate_report"]
        ),
        "v8_future_unseen_holdout_paper_candidate_gate_report_path": str(
            result.artifact_paths[
                "v8_future_unseen_holdout_paper_candidate_gate_report"
            ]
        ),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-candidate-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_o_replay_aligned_source_ranking",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--future-holdout-raw-manifest")
    args = parser.parse_args(argv)
    summary = run_polymarket_o_replay_aligned_source_ranking_cli(
        m2_candidate_report_path=args.m2_candidate_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
        future_holdout_raw_manifest_path=args.future_holdout_raw_manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
