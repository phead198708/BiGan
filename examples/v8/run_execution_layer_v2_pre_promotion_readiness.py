#!/usr/bin/env python3
"""Initialize the bounded v8 pre-promotion readiness goal."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pre_promotion_readiness import (
    ExecutionLayerV2PrePromotionGoalConfig,
    initialize_pre_promotion_readiness_goal,
    utc_now_iso,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze pre-promotion budgets, statistical gates, exclusions, and "
            "safety boundaries before fresh data collection."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--starting-commit", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--maximum-wall-clock-seconds", type=int, default=18_000)
    parser.add_argument(
        "--historical-collection-window-seconds", type=int, default=3_600
    )
    parser.add_argument("--maximum-historical-collection-windows", type=int, default=4)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    result = initialize_pre_promotion_readiness_goal(
        ExecutionLayerV2PrePromotionGoalConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            evidence_root=args.evidence_root,
            created_at=args.created_at or utc_now_iso(),
            starting_commit=args.starting_commit,
            maximum_wall_clock_seconds=args.maximum_wall_clock_seconds,
            historical_collection_window_seconds=(
                args.historical_collection_window_seconds
            ),
            maximum_historical_collection_windows=(
                args.maximum_historical_collection_windows
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(f"run_id={args.run_id}")
    print(f"goal_dir={result.goal_dir}")
    print(f"goal_configuration={result.goal_configuration_path}")
    print(f"goal_configuration_sha256={result.goal_configuration_sha256}")
    print(f"excluded_evidence_manifest={result.excluded_evidence_manifest_path}")
    print(f"goal_state={result.goal_state_path}")
    print("goal_status=IN_PROGRESS")
    print("promotion_evidence_stage_started=false")
    print("promotion_evidence_eligible=false")
    print("live_evidence_stage_started=false")
    print("live_evidence_allowed=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
