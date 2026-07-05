#!/usr/bin/env python3
"""Run a deterministic paper-only v8 execution layer v2 backtest bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2 import (
    ExecutionLayerV2BacktestConfig,
    ExecutionLayerV2Config,
    run_execution_layer_v2_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v8 Execution Layer v2 diagnostic-only backtest.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
        type=Path,
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--entry-ev-threshold", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--hold-ev-floor-ratio", type=float, default=0.60)
    parser.add_argument("--opposite-signal-ev-margin", type=float, default=0.02)
    parser.add_argument("--time-exit-threshold-seconds", type=float, default=60.0)
    parser.add_argument("--execution-cost-bps", type=float, default=10.0)
    parser.add_argument("--nav-usdc", type=float, default=10_000.0)
    parser.add_argument("--max-nav-fraction-per-position", type=float, default=0.05)
    parser.add_argument("--kelly-time-decay-lambda", type=float, default=0.0005)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    execution_config = ExecutionLayerV2Config(
        entry_ev_threshold=args.entry_ev_threshold,
        min_confidence=args.min_confidence,
        hold_ev_floor_ratio=args.hold_ev_floor_ratio,
        opposite_signal_ev_margin=args.opposite_signal_ev_margin,
        time_exit_threshold_seconds=args.time_exit_threshold_seconds,
        execution_cost_bps=args.execution_cost_bps,
        nav_usdc=args.nav_usdc,
        max_nav_fraction_per_position=args.max_nav_fraction_per_position,
        kelly_time_decay_lambda=args.kelly_time_decay_lambda,
    )
    result = run_execution_layer_v2_backtest(
        ExecutionLayerV2BacktestConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            input_path=args.input_path,
            max_rows=args.max_rows,
            execution_config=execution_config,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result.report
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"status={report['execution_layer_v2_status']}")
    print(f"decision_count={report.get('decision_count', 0)}")
    print(f"entry_decision_count={report.get('entry_decision_count', 0)}")
    print(f"hold_decision_count={report.get('hold_decision_count', 0)}")
    print(f"exit_decision_count={report.get('exit_decision_count', 0)}")
    print(f"rotation_decision_count={report.get('rotation_decision_count', 0)}")
    print(f"report_path={result.artifact_paths['execution_layer_v2_backtest_report']}")
    print(f"report_sha256={result.artifact_hashes['execution_layer_v2_backtest_report']}")
    print(f"manifest_path={result.artifact_paths['execution_layer_v2_backtest_manifest']}")
    print(f"manifest_sha256={result.artifact_hashes['execution_layer_v2_backtest_manifest']}")
    print("paper_only=true")
    print("capital_at_risk=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
