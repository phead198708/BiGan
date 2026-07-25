#!/usr/bin/env python3
"""Build the issue #252 retained-v6.7 paper-readiness design."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_retained_v6_7_paper_readiness import (
    RetainedV67PaperReadinessConfig,
    run_retained_v6_7_paper_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--issue238-manifest", type=Path, required=True)
    parser.add_argument("--issue238-manifest-sha256", required=True)
    parser.add_argument("--issue250-manifest", type=Path, required=True)
    parser.add_argument("--issue250-manifest-sha256", required=True)
    parser.add_argument("--issue251-manifest", type=Path, required=True)
    parser.add_argument("--issue251-manifest-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument(
        "--report-created-ts",
        type=int,
        default=None,
        help="Unix milliseconds; defaults to current wall-clock time.",
    )
    args = parser.parse_args()
    result = run_retained_v6_7_paper_readiness(
        RetainedV67PaperReadinessConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.profile_sha256,
            issue238_manifest_path=args.issue238_manifest,
            expected_issue238_manifest_sha256=args.issue238_manifest_sha256,
            issue250_manifest_path=args.issue250_manifest,
            expected_issue250_manifest_sha256=args.issue250_manifest_sha256,
            issue251_manifest_path=args.issue251_manifest,
            expected_issue251_manifest_sha256=args.issue251_manifest_sha256,
            implementation_commit=args.implementation_commit,
            report_created_ts=(
                args.report_created_ts
                if args.report_created_ts is not None
                else int(time.time() * 1000)
            ),
        )
    )
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
                if key
                in {
                    "run_dir",
                    "inventory_path",
                    "inventory_sha256",
                    "power_report_path",
                    "power_report_sha256",
                    "gate_plan_path",
                    "gate_plan_sha256",
                    "manifest_path",
                    "manifest_sha256",
                }
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
