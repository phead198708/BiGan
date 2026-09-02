#!/usr/bin/env python3
"""Run the frozen issue #246 v8.1 outcome-blind target-free canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_canary import (
    AdaptiveSupportControllerV81CanaryConfig,
    run_adaptive_support_controller_v8_1_target_free_canary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--canary-plan-sha256", required=True)
    parser.add_argument("--development-batch-canary-manifest", type=Path, required=True)
    parser.add_argument("--development-batch-canary-manifest-sha256", required=True)
    parser.add_argument("--v6-2-batch-canary-manifest", type=Path, required=True)
    parser.add_argument("--v6-2-batch-canary-manifest-sha256", required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_adaptive_support_controller_v8_1_target_free_canary(
        AdaptiveSupportControllerV81CanaryConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            canary_plan_path=args.canary_plan,
            expected_canary_plan_sha256=args.canary_plan_sha256,
            development_batch_canary_manifest_path=(
                args.development_batch_canary_manifest
            ),
            expected_development_batch_canary_manifest_sha256=(
                args.development_batch_canary_manifest_sha256
            ),
            v6_2_batch_canary_manifest_path=args.v6_2_batch_canary_manifest,
            expected_v6_2_batch_canary_manifest_sha256=(
                args.v6_2_batch_canary_manifest_sha256
            ),
            historical_manifest_path=args.historical_manifest,
            expected_historical_manifest_sha256=args.historical_manifest_sha256,
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
                "guard_accepted_bet_count": report["guard_accepted_bet_count"],
                "guard_accepted_policy_difference_market_count": report[
                    "guard_accepted_policy_difference_market_count"
                ],
                "target_free_canary_passed": report["target_free_canary_passed"],
                "target_free_canary_blocking_reason_codes": report[
                    "target_free_canary_blocking_reason_codes"
                ],
                "labels_outcomes_resolution_or_pnl_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
