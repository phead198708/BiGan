"""Freeze one #227 v6.7 target-free calibration or confirmatory window."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_pipeline import (
    V67TargetFreeWindowFreezeConfig,
    freeze_v6_7_target_free_window,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        required=True,
        choices=("fresh_calibration", "future_confirmatory"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--evaluation-profile", type=Path, required=True)
    parser.add_argument("--expected-evaluation-profile-sha256", required=True)
    parser.add_argument("--candidate-freeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-freeze-manifest-sha256", required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--expected-collection-plan-sha256", required=True)
    parser.add_argument("--collection-plan-correction", type=Path, required=True)
    parser.add_argument("--expected-collection-plan-correction-sha256", required=True)
    parser.add_argument("--collector-index", type=Path, required=True)
    parser.add_argument("--expected-collector-index-sha256", required=True)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--expected-calibration-artifact-sha256")
    parser.add_argument("--calibration-prediction-freeze-manifest", type=Path)
    parser.add_argument("--expected-calibration-prediction-freeze-manifest-sha256")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--decision-freeze-created-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    head = _head()
    if args.implementation_commit is not None and args.implementation_commit != head:
        raise ValueError("implementation commit does not match current HEAD")
    result = freeze_v6_7_target_free_window(
        V67TargetFreeWindowFreezeConfig(
            role=args.role,
            run_id=args.run_id,
            output_dir=args.output_dir,
            evaluation_profile_path=args.evaluation_profile,
            expected_evaluation_profile_sha256=(
                args.expected_evaluation_profile_sha256
            ),
            candidate_freeze_manifest_path=args.candidate_freeze_manifest,
            expected_candidate_freeze_manifest_sha256=(
                args.expected_candidate_freeze_manifest_sha256
            ),
            collection_plan_path=args.collection_plan,
            expected_collection_plan_sha256=args.expected_collection_plan_sha256,
            collection_plan_correction_path=args.collection_plan_correction,
            expected_collection_plan_correction_sha256=(
                args.expected_collection_plan_correction_sha256
            ),
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.expected_collector_index_sha256,
            calibration_artifact_path=args.calibration_artifact,
            expected_calibration_artifact_sha256=(
                args.expected_calibration_artifact_sha256
            ),
            calibration_prediction_freeze_manifest_path=(
                args.calibration_prediction_freeze_manifest
            ),
            expected_calibration_prediction_freeze_manifest_sha256=(
                args.expected_calibration_prediction_freeze_manifest_sha256
            ),
            implementation_commit=head,
            decision_freeze_created_ts=(
                args.decision_freeze_created_ts or int(time.time() * 1000)
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "role": report["role"],
                "selected_window_market_count": report[
                    "selected_window_market_count"
                ],
                "final_selected_side_count": report["final_selected_side_count"],
                "target_free_support_gate_passed": report[
                    "target_free_support_gate_passed"
                ],
                "future_target_access_allowed": report[
                    "future_target_access_allowed"
                ],
                "labels_outcomes_resolution_or_pnl_opened": False,
                **{
                    key: report[key]
                    for key in (
                        "paper_only",
                        "capital_at_risk",
                        "polymarket_write_enabled",
                        "wallet_signing_enabled",
                        "v8_execution_handoff_allowed",
                        "source_model_candidate_eligible",
                        "freeze_ready",
                        "promotion_evidence_eligible",
                        "#134_resume_allowed",
                        "#146_start_allowed",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
