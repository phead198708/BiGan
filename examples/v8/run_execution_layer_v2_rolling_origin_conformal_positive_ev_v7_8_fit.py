#!/usr/bin/env python3
"""Run the preregistered issue #242 historical conformal positive-EV gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_conformal_positive_ev_v7_8 import (
    RollingOriginConformalPositiveEVV78Config,
    run_rolling_origin_conformal_positive_ev_v7_8_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--v7-7-profile", required=True, type=Path)
    parser.add_argument("--v7-0-training-profile", required=True, type=Path)
    parser.add_argument("--v6-7-candidate-profile", required=True, type=Path)
    parser.add_argument("--runtime-policy-profile", required=True, type=Path)
    parser.add_argument("--seed-runtime-target-rows", required=True, type=Path)
    parser.add_argument("--consumed-stream-five-action-rows", required=True, type=Path)
    parser.add_argument("--consumed-stream-v6-7-candidate-rows", required=True, type=Path)
    parser.add_argument("--consumed-stream-v6-7-baseline-rows", required=True, type=Path)
    parser.add_argument("--consumed-stream-settled-index", required=True, type=Path)
    parser.add_argument(
        "--consumed-stream-target-free-freeze-manifest", required=True, type=Path
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--fit-created-ts", required=True, type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_rolling_origin_conformal_positive_ev_v7_8_fit(
        RollingOriginConformalPositiveEVV78Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            v7_7_profile_path=args.v7_7_profile,
            v7_0_training_profile_path=args.v7_0_training_profile,
            v6_7_candidate_profile_path=args.v6_7_candidate_profile,
            runtime_policy_profile_path=args.runtime_policy_profile,
            seed_runtime_target_rows_path=args.seed_runtime_target_rows,
            consumed_stream_five_action_rows_path=(
                args.consumed_stream_five_action_rows
            ),
            consumed_stream_v6_7_candidate_rows_path=(
                args.consumed_stream_v6_7_candidate_rows
            ),
            consumed_stream_v6_7_baseline_rows_path=(
                args.consumed_stream_v6_7_baseline_rows
            ),
            consumed_stream_settled_index_path=args.consumed_stream_settled_index,
            consumed_stream_target_free_freeze_manifest_path=(
                args.consumed_stream_target_free_freeze_manifest
            ),
            implementation_commit=args.implementation_commit,
            fit_created_ts=args.fit_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    model = result["model"]
    replay = model["historical_prequential_hard_gate"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "historical_hard_gate_passed": model["historical_hard_gate_passed"],
                "historical_gate_blocking_reason_codes": model[
                    "historical_gate_blocking_reason_codes"
                ],
                "candidate_guard_accepted_unique_market_count": model[
                    "historical_candidate_guard_accepted_unique_market_count"
                ],
                "historical_policy_difference_market_count": model[
                    "historical_policy_difference_market_count"
                ],
                "candidate_total_after_cost_net_pnl_at_frozen_size": replay[
                    "candidate"
                ]["total_after_cost_net_pnl_at_frozen_size"],
                "v6_7_total_after_cost_net_pnl_at_frozen_size": replay[
                    "v6_7_baseline"
                ]["total_after_cost_net_pnl_at_frozen_size"],
                "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": replay[
                    "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
                ],
                "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": replay[
                    "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
                ],
                "target_free_canary_collection_allowed": model[
                    "target_free_canary_collection_allowed"
                ],
                "target_free_canary_started": False,
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
