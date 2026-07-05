#!/usr/bin/env python3
"""Run diagnostic-only v8 Execution Layer v2 forward-shadow policy validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    ExecutionLayerV2ForwardShadowConfig,
    run_execution_layer_v2_forward_shadow_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline, bucket-aware, and calibrated-EV v2 policies on "
            "fresh decision-time signal trace artifacts without outcome labels."
        ),
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
        type=Path,
    )
    parser.add_argument("--entry-ev-threshold", default=0.02, type=float)
    parser.add_argument("--default-execution-cost", default=0.001, type=float)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    try:
        result = run_execution_layer_v2_forward_shadow_policy(
            ExecutionLayerV2ForwardShadowConfig(
                run_id=args.run_id,
                input_path=args.input_path,
                output_dir=args.output_dir,
                entry_ev_threshold=args.entry_ev_threshold,
                default_execution_cost=args.default_execution_cost,
                max_rows=args.max_rows,
                overwrite_existing=args.overwrite_existing,
            )
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    ev = result.ev_mapping_report
    shadow = result.forward_shadow_report
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"raw_row_count={shadow['raw_row_count']}")
    print(f"accepted_signal_row_count={shadow['accepted_signal_row_count']}")
    print(f"ev_mapping_status={ev['ev_mapping_status']}")
    print(f"calibrated_ev_available={str(ev['calibrated_ev_available']).lower()}")
    for name, metrics in shadow["policy_variants"].items():
        print(
            f"variant={name} allowed={metrics['allowed_decision_count']} "
            f"entries={metrics['entry_count']} exits={metrics['exit_count']} "
            f"holds={metrics['hold_count']} rejected={metrics['rejected_decision_count']}"
        )
    print(
        "ev_mapping_report_path="
        f"{result.artifact_paths['execution_layer_v2_calibrated_ev_mapping_report']}"
    )
    print(
        "ev_mapping_report_sha256="
        f"{result.artifact_hashes['execution_layer_v2_calibrated_ev_mapping_report']}"
    )
    print(
        "forward_shadow_report_path="
        f"{result.artifact_paths['execution_layer_v2_forward_shadow_policy_report']}"
    )
    print(
        "forward_shadow_report_sha256="
        f"{result.artifact_hashes['execution_layer_v2_forward_shadow_policy_report']}"
    )
    print(
        "manifest_path="
        f"{result.artifact_paths['execution_layer_v2_forward_shadow_manifest']}"
    )
    print(
        "manifest_sha256="
        f"{result.artifact_hashes['execution_layer_v2_forward_shadow_manifest']}"
    )
    print("paper_only=true")
    print("capital_at_risk=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
