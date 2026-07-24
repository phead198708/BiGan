#!/usr/bin/env python3
"""Freeze v8.3 and v6.7 decisions on the issue #249 future holdout."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_non_risk_abstention_fallback_v8_3_future_holdout import (
    NonRiskAbstentionFallbackV83FutureFreezeConfig,
    run_non_risk_abstention_fallback_v8_3_future_target_free_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    for name in (
        "plan",
        "profile",
        "collector-protocol",
        "collector-index",
        "historical-gate-manifest",
        "issue246-target-free-manifest",
        "target-free-canary-manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument(
        "--development-batch-manifest", action="append", type=Path, required=True
    )
    parser.add_argument(
        "--development-batch-manifest-sha256", action="append", required=True
    )
    parser.add_argument(
        "--v6-2-batch-manifest", action="append", type=Path, required=True
    )
    parser.add_argument(
        "--v6-2-batch-manifest-sha256", action="append", required=True
    )
    parser.add_argument("--implementation-commit")
    parser.add_argument("--stage-started-ts", type=int)
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
    result = run_non_risk_abstention_fallback_v8_3_future_target_free_freeze(
        NonRiskAbstentionFallbackV83FutureFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            plan_path=args.plan,
            expected_plan_sha256=args.plan_sha256,
            profile_path=args.profile,
            expected_profile_sha256=args.profile_sha256,
            collector_protocol_path=args.collector_protocol,
            expected_collector_protocol_sha256=args.collector_protocol_sha256,
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.collector_index_sha256,
            historical_gate_manifest_path=args.historical_gate_manifest,
            expected_historical_gate_manifest_sha256=(
                args.historical_gate_manifest_sha256
            ),
            issue246_target_free_manifest_path=args.issue246_target_free_manifest,
            expected_issue246_target_free_manifest_sha256=(
                args.issue246_target_free_manifest_sha256
            ),
            target_free_canary_manifest_path=args.target_free_canary_manifest,
            expected_target_free_canary_manifest_sha256=(
                args.target_free_canary_manifest_sha256
            ),
            development_batch_manifest_paths=tuple(
                args.development_batch_manifest
            ),
            expected_development_batch_manifest_sha256s=tuple(
                args.development_batch_manifest_sha256
            ),
            v6_2_batch_manifest_paths=tuple(args.v6_2_batch_manifest),
            expected_v6_2_batch_manifest_sha256s=tuple(
                args.v6_2_batch_manifest_sha256
            ),
            implementation_commit=head,
            stage_started_ts=args.stage_started_ts or int(time.time() * 1000),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "selected_market_count": report["selected_market_count"],
                "candidate_guard_accepted_market_count": report[
                    "candidate_guard_accepted_market_count"
                ],
                "target_free_freeze_passed": report[
                    "target_free_freeze_passed"
                ],
                "target_free_blocking_reason_codes": report[
                    "target_free_blocking_reason_codes"
                ],
                "labels_outcomes_resolution_or_pnl_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
