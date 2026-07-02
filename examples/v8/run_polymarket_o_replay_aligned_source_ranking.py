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
) -> dict:
    result = run_polymarket_o_replay_aligned_source_ranking(
        PolymarketOReplayAlignedSourceRankingConfig(
            m2_candidate_report_path=m2_candidate_report_path,
            output_dir=output_dir,
            run_id=run_id,
            overwrite_existing=overwrite_existing,
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
    block_analysis = result.v8_execution_guard_block_analysis_report
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
        "v8_execution_guard_block_analysis_report_path": str(
            result.artifact_paths["v8_execution_guard_block_analysis_report"]
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
    args = parser.parse_args(argv)
    summary = run_polymarket_o_replay_aligned_source_ranking_cli(
        m2_candidate_report_path=args.m2_candidate_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
