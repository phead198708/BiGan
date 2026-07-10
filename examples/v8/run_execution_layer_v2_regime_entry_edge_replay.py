#!/usr/bin/env python3
"""Run diagnostic-only regime entry-edge and exposure replay."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    ExecutionLayerV2RegimeEntryEdgeReplayConfig,
    run_execution_layer_v2_regime_entry_edge_replay,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze regime-conditioned entry edge and repeated-market exposure "
            "from a settled paper run without changing execution gates."
        )
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
        type=Path,
    )
    parser.add_argument(
        "--frozen-ev-calibration-artifact",
        type=Path,
        default=None,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    try:
        result = run_execution_layer_v2_regime_entry_edge_replay(
            ExecutionLayerV2RegimeEntryEdgeReplayConfig(
                run_id=args.run_id,
                input_path=args.input_path,
                output_dir=args.output_dir,
                frozen_ev_calibration_artifact=(
                    args.frozen_ev_calibration_artifact
                ),
                overwrite_existing=args.overwrite_existing,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    report = result.report
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"row_count={report['row_count']}")
    print(f"unique_market_count={report['unique_market_count']}")
    for name, metrics in report["policy_variants"].items():
        print(f"{name}_fill_count={metrics['fill_count']}")
        print(f"{name}_settled_pnl={metrics['settled_pnl']:.12f}")
    print(
        "correlated_momentum_reference_counted_as_independent_votes="
        f"{str(report['correlated_momentum_reference_counted_as_independent_votes']).lower()}"
    )
    print(
        "report_path="
        f"{result.artifact_paths['execution_layer_v2_regime_entry_edge_replay_report']}"
    )
    print(
        "report_sha256="
        f"{result.artifact_hashes['execution_layer_v2_regime_entry_edge_replay_report']}"
    )
    print(
        "manifest_path="
        f"{result.artifact_paths['execution_layer_v2_regime_entry_edge_replay_manifest']}"
    )
    print(
        "manifest_sha256="
        f"{result.artifact_hashes['execution_layer_v2_regime_entry_edge_replay_manifest']}"
    )
    print("paper_only=true")
    print("capital_at_risk=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
