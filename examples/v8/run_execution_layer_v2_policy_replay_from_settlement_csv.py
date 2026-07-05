#!/usr/bin/env python3
"""Run v8 Execution Layer v2 settlement-CSV policy replay diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    ExecutionLayerV2PolicyReplayConfig,
    run_execution_layer_v2_policy_replay_from_settlement_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay diagnostic-only v8 execution policy variants from a settlement PnL CSV.",
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
        type=Path,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    try:
        result = run_execution_layer_v2_policy_replay_from_settlement_csv(
            ExecutionLayerV2PolicyReplayConfig(
                run_id=args.run_id,
                input_csv=args.input_csv,
                output_dir=args.output_dir,
                overwrite_existing=args.overwrite_existing,
            )
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    report = result.report
    variants = report["policy_variants"]
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"row_count={report['row_count']}")
    print(f"ev_mapping_status={report['signal_to_ev_diagnostic']['ev_mapping_status']}")
    for name, metrics in variants.items():
        print(
            f"variant={name} rows={metrics['row_count']} "
            f"pnl={metrics['settlement_pnl']:.6f} roi={metrics['roi']:.6f} "
            f"win_rate={metrics['win_rate']:.6f}"
        )
    print(f"report_path={result.artifact_paths['execution_layer_v2_policy_replay_report']}")
    print(f"report_sha256={result.artifact_hashes['execution_layer_v2_policy_replay_report']}")
    print(f"manifest_path={result.artifact_paths['execution_layer_v2_policy_replay_manifest']}")
    print(f"manifest_sha256={result.artifact_hashes['execution_layer_v2_policy_replay_manifest']}")
    print("paper_only=true")
    print("capital_at_risk=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
