#!/usr/bin/env python3
"""Freeze issue #241 target-free v7.7/v6.7 decisions before settlement."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout_pipeline import (
    V77FutureTargetFreeFreezeConfig,
    run_v7_7_future_target_free_freeze,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--collector-protocol", type=Path, required=True)
    parser.add_argument("--expected-collector-protocol-sha256", required=True)
    parser.add_argument("--collector-index", type=Path, required=True)
    parser.add_argument("--expected-collector-index-sha256", required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--expected-historical-manifest-sha256", required=True)
    parser.add_argument("--prior-lineage-rows", type=Path, required=True)
    parser.add_argument("--expected-prior-lineage-rows-sha256", required=True)
    parser.add_argument("--prior-canary-index", type=Path, required=True)
    parser.add_argument("--expected-prior-canary-index-sha256", required=True)
    parser.add_argument(
        "--development-batch-manifest", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--expected-development-batch-manifest-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--v6-2-batch-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-v6-2-batch-manifest-sha256", action="append", required=True
    )
    parser.add_argument("--implementation-commit")
    parser.add_argument("--stage-started-ts", type=int)
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
    if args.implementation_commit and args.implementation_commit != head:
        raise ValueError("implementation commit does not match current HEAD")
    result = run_v7_7_future_target_free_freeze(
        V77FutureTargetFreeFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            plan_path=args.plan,
            expected_plan_sha256=args.expected_plan_sha256,
            collector_protocol_path=args.collector_protocol,
            expected_collector_protocol_sha256=(
                args.expected_collector_protocol_sha256
            ),
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.expected_collector_index_sha256,
            historical_manifest_path=args.historical_manifest,
            expected_historical_manifest_sha256=(
                args.expected_historical_manifest_sha256
            ),
            prior_lineage_rows_path=args.prior_lineage_rows,
            expected_prior_lineage_rows_sha256=(
                args.expected_prior_lineage_rows_sha256
            ),
            prior_canary_index_path=args.prior_canary_index,
            expected_prior_canary_index_sha256=(
                args.expected_prior_canary_index_sha256
            ),
            development_batch_manifest_paths=tuple(args.development_batch_manifest),
            expected_development_batch_manifest_sha256s=tuple(
                args.expected_development_batch_manifest_sha256
            ),
            v6_2_batch_manifest_paths=tuple(args.v6_2_batch_manifest),
            expected_v6_2_batch_manifest_sha256s=tuple(
                args.expected_v6_2_batch_manifest_sha256
            ),
            implementation_commit=head,
            stage_started_ts=args.stage_started_ts or int(time.time() * 1000),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "target_free_freeze_passed": result["report"][
                    "target_free_freeze_passed"
                ],
                "future_target_access_allowed": result["report"][
                    "future_target_access_allowed"
                ],
                "side_quota_enabled": False,
                "labels_outcomes_resolution_or_pnl_opened": False,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "v8_execution_handoff_allowed": False,
                "source_model_candidate_eligible": False,
                "freeze_ready": False,
                "promotion_evidence_eligible": False,
                "#134_resume_allowed": False,
                "#146_start_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
