#!/usr/bin/env python3
"""Run outcome-free regime-conditioned EV forward-shadow diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev import (
    ExecutionLayerV2RegimeConditionedEVForwardShadowConfig,
    run_execution_layer_v2_regime_conditioned_ev_forward_shadow,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a separately fitted frozen regime-conditioned EV artifact to "
            "fresh decision-time rows and intersect candidates with the existing "
            "execution guard. This runner is diagnostic-only and fail-closed."
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
        "--frozen-regime-conditioned-ev-artifact",
        required=True,
        type=Path,
    )
    parser.add_argument("--entry-ev-threshold", default=0.02, type=float)
    parser.add_argument("--default-execution-cost", default=0.001, type=float)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    try:
        result = run_execution_layer_v2_regime_conditioned_ev_forward_shadow(
            ExecutionLayerV2RegimeConditionedEVForwardShadowConfig(
                run_id=args.run_id,
                input_path=args.input_path,
                output_dir=args.output_dir,
                frozen_regime_conditioned_ev_artifact=(
                    args.frozen_regime_conditioned_ev_artifact
                ),
                entry_ev_threshold=args.entry_ev_threshold,
                default_execution_cost=args.default_execution_cost,
                max_rows=args.max_rows,
                overwrite_existing=args.overwrite_existing,
            )
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    report = result.forward_shadow_report
    validation = result.artifact_validation_report
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"artifact_valid={str(validation['artifact_valid']).lower()}")
    print(f"artifact_status={validation['artifact_status']}")
    print(f"raw_row_count={report['raw_row_count']}")
    print(f"accepted_signal_row_count={report['accepted_signal_row_count']}")
    print(
        "regime_conditioned_ev_produced_count="
        f"{report['regime_conditioned_ev_produced_count']}"
    )
    print(
        "regime_conditioned_ev_missing_count="
        f"{report['regime_conditioned_ev_missing_count']}"
    )
    print(f"candidate_count={report['candidate_count']}")
    print(f"full_guard_passed_count={report['full_guard_passed_count']}")
    print(f"executable_shadow_count={report['executable_shadow_count']}")
    print("market_implied_probability_used_as_direct_fair_value_ev=false")
    print("market_implied_probability_used_as_conditioning_feature=true")
    print("market_implied_probability_used_as_regime_direction_vote=false")
    print(
        "future_v2_probability_value_contract_status="
        f"{report['future_v2_probability_value_contract_recommendation']['status']}"
    )
    print(
        "rejection_reason_distribution="
        f"{report['rejection_reason_distribution']}"
    )
    print(
        "forward_shadow_report_path="
        f"{result.artifact_paths['execution_layer_v2_regime_conditioned_ev_forward_shadow_report']}"
    )
    print(
        "forward_shadow_report_sha256="
        f"{result.artifact_hashes['execution_layer_v2_regime_conditioned_ev_forward_shadow_report']}"
    )
    print(
        "manifest_path="
        f"{result.artifact_paths['execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest']}"
    )
    print(
        "manifest_sha256="
        f"{result.artifact_hashes['execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest']}"
    )
    print("diagnostic_only=true")
    print("paper_only=true")
    print("capital_at_risk=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
