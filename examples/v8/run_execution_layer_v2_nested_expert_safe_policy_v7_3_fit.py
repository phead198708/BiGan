#!/usr/bin/env python3
"""Fit and replay the issue #235 nested training-only v7.3 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_nested_expert_safe_policy_v7_3 import (
    NestedExpertSafePolicyV73Config,
    run_nested_expert_safe_policy_v7_3_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--v7-0-training-profile", required=True, type=Path)
    parser.add_argument("--v6-7-candidate-profile", required=True, type=Path)
    parser.add_argument("--v7-2-relative-policy-source", required=True, type=Path)
    parser.add_argument("--runtime-target-rows", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--fit-created-ts", required=True, type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_nested_expert_safe_policy_v7_3_fit(
        NestedExpertSafePolicyV73Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            v7_0_training_profile_path=args.v7_0_training_profile,
            v6_7_candidate_profile_path=args.v6_7_candidate_profile,
            v7_2_relative_policy_source_path=args.v7_2_relative_policy_source,
            runtime_target_rows_path=args.runtime_target_rows,
            implementation_commit=args.implementation_commit,
            fit_created_ts=args.fit_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    model = result["model"]
    replay = model["historical_replay_superiority_gate"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "historical_gate_passed": model["historical_gate_passed"],
                "historical_gate_blocking_reason_codes": model[
                    "historical_gate_blocking_reason_codes"
                ],
                "outer_selected_expert_names": [
                    row["nested_selection"]["selected_expert_name"]
                    for row in model["outer_fold_reports"]
                ],
                "final_selected_expert_name": model["final_nested_selection"][
                    "selected_expert_name"
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
                "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": (
                    replay[
                        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
                    ]
                ),
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
