#!/usr/bin/env python3
"""Run the preregistered issue #237 v7.5 historical fit and replay once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_frozen_v6_7_loss_tail_veto_v7_5 import (
    FrozenV67LossTailVetoV75Config,
    run_frozen_v6_7_loss_tail_veto_v7_5_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--v7-0-training-profile", required=True)
    parser.add_argument("--v6-7-candidate-profile", required=True)
    parser.add_argument("--v7-2-relative-policy-source", required=True)
    parser.add_argument("--runtime-target-rows", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--fit-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_frozen_v6_7_loss_tail_veto_v7_5_fit(
        FrozenV67LossTailVetoV75Config(
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
            profile_path=Path(args.profile),
            expected_profile_sha256=args.profile_sha256,
            v7_0_training_profile_path=Path(args.v7_0_training_profile),
            v6_7_candidate_profile_path=Path(args.v6_7_candidate_profile),
            v7_2_relative_policy_source_path=Path(args.v7_2_relative_policy_source),
            runtime_target_rows_path=Path(args.runtime_target_rows),
            implementation_commit=args.implementation_commit,
            fit_created_ts=args.fit_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    replay = report["historical_replay_noninferiority_gate"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "historical_noninferiority_gate_passed": report[
                    "historical_noninferiority_gate_passed"
                ],
                "historical_policy_difference_market_count": report[
                    "historical_policy_difference_market_count"
                ],
                "model_improvement_demonstrated": report[
                    "model_improvement_demonstrated"
                ],
                "candidate_pnl": replay["candidate"][
                    "total_after_cost_net_pnl_at_frozen_size"
                ],
                "v6_7_pnl": replay["v6_7_baseline"][
                    "total_after_cost_net_pnl_at_frozen_size"
                ],
                "target_free_canary_collection_allowed": report[
                    "target_free_canary_collection_allowed"
                ],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
