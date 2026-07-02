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
    risk = result.large_regret_risk_model_report
    guard = result.selective_action_guard_report
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
        "final_scoring_source": ranking["final_scoring_source"],
        "selected_guard_mode": guard["selected_guard_mode"],
        "large_regret_risk_model_config_hash": risk["risk_model_config_hash"],
        "selective_action_guard_selection_config_hash": guard[
            "selection_config_hash"
        ],
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
        "large_regret_risk_model_report_path": str(
            result.artifact_paths["large_regret_risk_model_report"]
        ),
        "selective_action_guard_report_path": str(
            result.artifact_paths["selective_action_guard_report"]
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
