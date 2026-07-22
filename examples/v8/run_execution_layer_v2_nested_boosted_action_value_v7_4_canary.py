#!/usr/bin/env python3
"""Run the frozen issue #236 v7.4 outcome-blind target-free canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    NestedBoostedActionValueV74CanaryConfig,
    run_nested_boosted_action_value_v7_4_target_free_canary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--v6-2-batch-canary-manifest", type=Path, required=True)
    parser.add_argument("--v6-2-batch-canary-manifest-sha256", required=True)
    parser.add_argument("--v7-4-historical-manifest", type=Path, required=True)
    parser.add_argument("--v7-4-historical-manifest-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_nested_boosted_action_value_v7_4_target_free_canary(
        NestedBoostedActionValueV74CanaryConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            v6_2_batch_canary_manifest_path=args.v6_2_batch_canary_manifest,
            expected_v6_2_batch_canary_manifest_sha256=(
                args.v6_2_batch_canary_manifest_sha256
            ),
            v7_4_historical_manifest_path=args.v7_4_historical_manifest,
            expected_v7_4_historical_manifest_sha256=(
                args.v7_4_historical_manifest_sha256
            ),
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
                "quality_valid_market_count": report["quality_valid_market_count"],
                "guard_accepted_bet_count": report["guard_accepted_bet_count"],
                "guard_accepted_by_side": report["guard_accepted_by_side"],
                "target_free_canary_passed": report["target_free_canary_passed"],
                "target_free_canary_blocking_reason_codes": report[
                    "target_free_canary_blocking_reason_codes"
                ],
                "labels_outcomes_or_pnl_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
