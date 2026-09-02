#!/usr/bin/env python3
"""Initialize immutable v8 pre-promotion remediation evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pre_promotion_remediation import (
    ExecutionLayerV2PrePromotionRemediationConfig,
    initialize_pre_promotion_remediation_goal,
    utc_now_iso,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--starting-branch", required=True)
    parser.add_argument("--starting-commit", required=True)
    parser.add_argument("--prior-blocked-bundle-dir", required=True, type=Path)
    parser.add_argument("--prior-corpus-rows-path", required=True, type=Path)
    parser.add_argument("--prior-split-report-path", required=True, type=Path)
    parser.add_argument("--prior-calibration-report-path", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--created-at", default=None)
    args = parser.parse_args()

    result = initialize_pre_promotion_remediation_goal(
        ExecutionLayerV2PrePromotionRemediationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            created_at=args.created_at or utc_now_iso(),
            starting_branch=args.starting_branch,
            starting_commit=args.starting_commit,
            prior_blocked_bundle_dir=args.prior_blocked_bundle_dir,
            prior_corpus_rows_path=args.prior_corpus_rows_path,
            prior_split_report_path=args.prior_split_report_path,
            prior_calibration_report_path=args.prior_calibration_report_path,
        )
    )
    print(f"goal_dir={result.goal_dir}")
    print(f"initial_goal_configuration={result.configuration_path}")
    print(
        "initial_goal_configuration_sha256="
        f"{result.configuration_sha256_path.read_text(encoding='utf-8').strip()}"
    )
    print(f"initial_excluded_evidence_manifest={result.exclusions_path}")
    print(f"initial_goal_state={result.state_path}")
    print("goal_status=IN_PROGRESS")
    print("promotion_evidence_stage_started=false")
    print("promotion_evidence_eligible=false")
    print("live_evidence_stage_started=false")
    print("live_evidence_allowed=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
