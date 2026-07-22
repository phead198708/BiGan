#!/usr/bin/env python3
"""Fit and replay the issue #234 frozen-v6.7-relative v7.2 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    V67RelativeSafePolicyV72Config,
    run_v6_7_relative_safe_policy_v7_2_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--v7-0-training-profile", required=True, type=Path)
    parser.add_argument("--v6-7-candidate-profile", required=True, type=Path)
    parser.add_argument("--runtime-target-rows", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--fit-created-ts", required=True, type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_v6_7_relative_safe_policy_v7_2_fit(
        V67RelativeSafePolicyV72Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            v7_0_training_profile_path=args.v7_0_training_profile,
            v6_7_candidate_profile_path=args.v6_7_candidate_profile,
            runtime_target_rows_path=args.runtime_target_rows,
            implementation_commit=args.implementation_commit,
            fit_created_ts=args.fit_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    replay = result["model"]["historical_replay_superiority_gate"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "historical_gate_passed": result["model"][
                    "historical_gate_passed"
                ],
                "historical_gate_blocking_reason_codes": result["model"][
                    "historical_gate_blocking_reason_codes"
                ],
                "historical_policy_difference_market_count": result["model"][
                    "historical_policy_difference_market_count"
                ],
                "candidate_total_after_cost_net_pnl_at_frozen_size": replay[
                    "candidate"
                ]["total_after_cost_net_pnl_at_frozen_size"],
                "v6_7_total_after_cost_net_pnl_at_frozen_size": replay[
                    "v6_7_baseline"
                ]["total_after_cost_net_pnl_at_frozen_size"],
                "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": (
                    replay[
                        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
                    ]
                ),
                "target_free_canary_collection_allowed": result["model"][
                    "target_free_canary_collection_allowed"
                ],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "paper_candidate_allowed": False,
                "live_trading_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
