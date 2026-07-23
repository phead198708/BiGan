#!/usr/bin/env python3
"""Freeze v8.1 and v6.7 decisions on the issue #246 future holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_pipeline import (
    AdaptiveSupportControllerV81FutureFreezeConfig,
    run_adaptive_support_controller_v8_1_future_target_free_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--collector-protocol", type=Path, required=True)
    parser.add_argument("--collector-protocol-sha256", required=True)
    parser.add_argument("--collector-index", type=Path, required=True)
    parser.add_argument("--collector-index-sha256", required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest-sha256", required=True)
    parser.add_argument("--prior-canary-index", type=Path, required=True)
    parser.add_argument("--prior-canary-index-sha256", required=True)
    parser.add_argument("--prior-canary-manifest", type=Path, required=True)
    parser.add_argument("--prior-canary-manifest-sha256", required=True)
    parser.add_argument(
        "--development-batch-manifest",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--development-batch-manifest-sha256",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--v6-2-batch-manifest",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--v6-2-batch-manifest-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--stage-started-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_adaptive_support_controller_v8_1_future_target_free_freeze(
        AdaptiveSupportControllerV81FutureFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            plan_path=args.plan,
            expected_plan_sha256=args.plan_sha256,
            collector_protocol_path=args.collector_protocol,
            expected_collector_protocol_sha256=args.collector_protocol_sha256,
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.collector_index_sha256,
            historical_manifest_path=args.historical_manifest,
            expected_historical_manifest_sha256=args.historical_manifest_sha256,
            prior_canary_index_path=args.prior_canary_index,
            expected_prior_canary_index_sha256=args.prior_canary_index_sha256,
            prior_canary_manifest_path=args.prior_canary_manifest,
            expected_prior_canary_manifest_sha256=(
                args.prior_canary_manifest_sha256
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
            implementation_commit=args.implementation_commit,
            stage_started_ts=args.stage_started_ts,
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
