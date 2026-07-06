#!/usr/bin/env python3
"""Run the v8 Execution Layer v2 one-hour paper-only remap goal."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal import (
    DEFAULT_FROZEN_EV_CALIBRATION_ARTIFACT,
    DEFAULT_ONE_HOUR_UNLOCK_DIR,
    PINNED_ISSUE_160_MANIFEST_SHA256,
    ExecutionLayerV2OneHourRemapPaperGoalConfig,
    run_execution_layer_v2_one_hour_remap_paper_goal,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 1-hour read-only-provider paper goal diagnostic for the "
            "Execution Layer v2 HTS-to-SBC remap path."
        )
    )
    parser.add_argument(
        "--run-id",
        default=f"execution-layer-v2-one-hour-remap-paper-goal-{_utc_stamp()}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--settlement-poll-max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument(
        "--max-consecutive-orderbook-failure-rounds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--allow-short-diagnostic-run",
        action="store_true",
        help="Allow duration-seconds below 3600 for bounded smoke diagnostics.",
    )
    parser.add_argument(
        "--paper-candidate-unlock-dir",
        type=Path,
        default=DEFAULT_ONE_HOUR_UNLOCK_DIR,
    )
    parser.add_argument(
        "--paper-candidate-unlock-manifest-sha256",
        default=PINNED_ISSUE_160_MANIFEST_SHA256,
    )
    parser.add_argument("--allow-unpinned-unlock-manifest", action="store_true")
    parser.add_argument(
        "--frozen-ev-calibration-artifact",
        type=Path,
        default=DEFAULT_FROZEN_EV_CALIBRATION_ARTIFACT,
    )
    parser.add_argument(
        "--canonical-o-source-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    expected_unlock_manifest_sha256 = (
        None
        if args.allow_unpinned_unlock_manifest
        else args.paper_candidate_unlock_manifest_sha256
    )

    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            duration_seconds=args.duration_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            paper_candidate_unlock_dir=args.paper_candidate_unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=(
                expected_unlock_manifest_sha256
            ),
            frozen_ev_calibration_artifact_path=args.frozen_ev_calibration_artifact,
            canonical_o_source_manifest_path=args.canonical_o_source_manifest,
            settlement_poll_max_wait_seconds=args.settlement_poll_max_wait_seconds,
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            max_consecutive_orderbook_failure_rounds=(
                args.max_consecutive_orderbook_failure_rounds
            ),
            allow_short_diagnostic_run=args.allow_short_diagnostic_run,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result.goal_report
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"duration_seconds={report['duration_seconds']}")
    print(f"complete_round_count={report['complete_round_count']}")
    print(
        "complete_rounds_with_bet_count="
        f"{report['complete_rounds_with_bet_count']}"
    )
    print(f"missing_bet_round_count={report['missing_bet_round_count']}")
    print(f"normal_policy_bet_count={report['normal_policy_bet_count']}")
    print(f"remap_paper_bet_count={report['remap_paper_bet_count']}")
    print(f"forced_coverage_bet_count={report['forced_coverage_bet_count']}")
    print(f"settled_pnl={report['settled_pnl']}")
    print(f"unresolved_pnl={report['unresolved_pnl']}")
    print(f"settlement_poll_attempt_count={report['settlement_poll_attempt_count']}")
    print(f"settlement_evaluation_row_count={report['settlement_evaluation_row_count']}")
    print(f"settlement_resolution_reason_codes={report['settlement_resolution_reason_codes']}")
    print(f"provider_fail_fast_stop_triggered={str(report['provider_fail_fast_stop_triggered']).lower()}")
    print(f"provider_fail_fast_reason_codes={report['provider_fail_fast_reason_codes']}")
    print(
        "consecutive_orderbook_failure_count_at_stop="
        f"{report['consecutive_orderbook_failure_count_at_stop']}"
    )
    print(f"final_goal_success={str(report['final_goal_success']).lower()}")
    print(f"goal_failure_reason_codes={report['goal_failure_reason_codes']}")
    print(
        "one_hour_remap_paper_goal_report="
        f"{result.artifact_paths['one_hour_remap_paper_goal_report']}"
    )
    print(
        "one_hour_remap_paper_goal_report_sha256="
        f"{result.artifact_hashes['one_hour_remap_paper_goal_report']}"
    )
    print(
        "one_hour_remap_paper_goal_manifest="
        f"{result.artifact_paths['one_hour_remap_paper_goal_manifest']}"
    )
    print(
        "one_hour_remap_paper_goal_manifest_sha256="
        f"{result.artifact_hashes['one_hour_remap_paper_goal_manifest']}"
    )
    print("paper_only=true")
    print("capital_at_risk=false")
    print("polymarket_write_enabled=false")
    print("wallet_signing_enabled=false")
    print("v8_execution_handoff_allowed=false")


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    main()
