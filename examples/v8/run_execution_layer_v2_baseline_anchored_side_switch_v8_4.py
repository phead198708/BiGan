#!/usr/bin/env python3
"""Freeze and replay the issue #251 baseline-anchored v8.4 policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_baseline_anchored_side_switch_v8_4 import (
    BaselineAnchoredSideSwitchV84Config,
    run_baseline_anchored_side_switch_v8_4,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--v8-3-historical-manifest", type=Path, required=True)
    parser.add_argument("--v8-3-historical-manifest-sha256", required=True)
    parser.add_argument(
        "--issue250-target-free-selected-rows",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--issue250-target-free-selected-rows-sha256",
        required=True,
    )
    parser.add_argument("--implementation-commit")
    parser.add_argument("--evidence-created-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if args.implementation_commit and args.implementation_commit != head:
        raise ValueError("implementation commit does not match current HEAD")
    result = run_baseline_anchored_side_switch_v8_4(
        BaselineAnchoredSideSwitchV84Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.profile_sha256,
            v8_3_historical_manifest_path=args.v8_3_historical_manifest,
            expected_v8_3_historical_manifest_sha256=(
                args.v8_3_historical_manifest_sha256
            ),
            issue250_target_free_selected_rows_path=(
                args.issue250_target_free_selected_rows
            ),
            expected_issue250_target_free_selected_rows_sha256=(
                args.issue250_target_free_selected_rows_sha256
            ),
            implementation_commit=head,
            evidence_created_ts=args.evidence_created_ts
            or int(time.time() * 1000),
            overwrite_existing=args.overwrite_existing,
        )
    )
    evidence = result["evidence_artifact"]
    report = result["historical_report"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "evidence_artifact_path": str(
                    result["evidence_artifact_path"]
                ),
                "evidence_artifact_sha256": result[
                    "evidence_artifact_sha256"
                ],
                "evidence_report_path": str(result["evidence_report_path"]),
                "evidence_report_sha256": result["evidence_report_sha256"],
                "historical_report_path": str(
                    result["historical_report_path"]
                ),
                "historical_report_sha256": result[
                    "historical_report_sha256"
                ],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "switch_example_count": evidence["switch_example_count"],
                "eligible_switch_classes": evidence[
                    "eligible_switch_classes"
                ],
                "candidate_minus_v6_7_total_after_cost_pnl": report[
                    "candidate_minus_v6_7_total_after_cost_pnl"
                ],
                "historical_noninferiority_gate_passed": report[
                    "historical_noninferiority_gate_passed"
                ],
                "model_improvement_demonstrated": report[
                    "model_improvement_demonstrated"
                ],
                "new_future_challenger_collection_justified": report[
                    "new_future_challenger_collection_justified"
                ],
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
