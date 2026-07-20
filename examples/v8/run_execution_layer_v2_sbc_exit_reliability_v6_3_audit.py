"""Freeze or audit the preregistered #223 SBC exit-reliability corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_sbc_exit_reliability_v6_3 import (
    SBCExitReliabilityV63AuditConfig,
    run_sbc_exit_reliability_v6_3_audit,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze_lineage", "audit_labels"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_sbc_exit_reliability_v6_3_profile.json"
        ),
    )
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--role-assignment-manifest", type=Path, required=True)
    parser.add_argument("--lineage-manifest", type=Path)
    parser.add_argument("--expected-lineage-manifest-sha256")
    parser.add_argument("--implementation-commit")
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
    result = run_sbc_exit_reliability_v6_3_audit(
        SBCExitReliabilityV63AuditConfig(
            stage=args.stage,
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            role_assignment_manifest_path=args.role_assignment_manifest,
            implementation_commit=args.implementation_commit or _head(),
            lineage_manifest_path=args.lineage_manifest,
            expected_lineage_manifest_sha256=args.expected_lineage_manifest_sha256,
            overwrite_existing=args.overwrite_existing,
        )
    )
    summary = {
        "run_dir": str(result["run_dir"]),
        "stage": args.stage,
    }
    if args.stage == "freeze_lineage":
        summary.update(
            {
                "lineage_manifest_path": str(result["lineage_manifest_path"]),
                "lineage_manifest_sha256": result["lineage_manifest_sha256"],
                "eligible_market_count": result["manifest"]["eligible_market_count"],
                "label_file_content_opened": False,
            }
        )
    else:
        summary.update(
            {
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "eligible_market_count": result["manifest"]["eligible_market_count"],
                "label_audit_gate_passed": result["label_audit"][
                    "label_audit_gate_passed"
                ],
                "feature_coverage_gate_passed": result["feature_coverage"][
                    "feature_coverage_gate_passed"
                ],
                "fit_allowed": result["manifest"]["fit_allowed"],
                "blocking_reason_codes": result["manifest"]["blocking_reason_codes"],
            }
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
