#!/usr/bin/env python3
"""Run the preregistered issue #247 historical non-inferiority gate."""

from __future__ import annotations

import argparse
import json

from bigan.v8.polymarket.training.execution_layer_v2_support_preserving_overlay_v8_2 import (
    SupportPreservingOverlayV82Config,
    run_support_preserving_overlay_v8_2_historical_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--historical-manifest", required=True)
    parser.add_argument("--historical-manifest-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--evaluation-started-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_support_preserving_overlay_v8_2_historical_gate(
        SupportPreservingOverlayV82Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.profile_sha256,
            historical_manifest_path=args.historical_manifest,
            expected_historical_manifest_sha256=(
                args.historical_manifest_sha256
            ),
            implementation_commit=args.implementation_commit,
            evaluation_started_ts=args.evaluation_started_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "historical_noninferiority_gate_passed": report[
                    "historical_noninferiority_gate_passed"
                ],
                "historical_gate_blocking_reason_codes": report[
                    "historical_gate_blocking_reason_codes"
                ],
                "candidate_guard_accepted_market_count": report[
                    "candidate_guard_accepted_market_count"
                ],
                "v6_7_guard_accepted_market_count": report[
                    "v6_7_guard_accepted_market_count"
                ],
                "candidate_total_after_cost_pnl": report[
                    "candidate_total_after_cost_net_pnl_at_frozen_size"
                ],
                "v6_7_total_after_cost_pnl": report[
                    "v6_7_total_after_cost_net_pnl_at_frozen_size"
                ],
                "candidate_minus_v6_7_total_after_cost_pnl": report[
                    "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
                ],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
