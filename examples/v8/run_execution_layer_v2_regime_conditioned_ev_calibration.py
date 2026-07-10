#!/usr/bin/env python3
"""Run the fail-closed regime-conditioned EV v2 calibration protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev_calibration import (
    ExecutionLayerV2RegimeConditionedEVCalibrationConfig,
    run_execution_layer_v2_regime_conditioned_ev_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a regularized regime-conditioned EV baseline on a historical "
            "market-disjoint split, evaluate on later validation rows, and "
            "optionally run an outcome-free future shadow. Diagnostic only."
        )
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", default="examples/v8/polymarket_runs", type=Path
    )
    parser.add_argument("--future-shadow-input-path", type=Path)
    parser.add_argument("--validation-fraction", default=0.25, type=float)
    parser.add_argument("--ridge-alpha", default=1.0, type=float)
    parser.add_argument("--entry-ev-threshold", default=0.02, type=float)
    parser.add_argument("--min-fit-rows", default=100, type=int)
    parser.add_argument("--min-validation-rows", default=30, type=int)
    parser.add_argument("--min-fit-markets", default=20, type=int)
    parser.add_argument("--min-validation-markets", default=10, type=int)
    parser.add_argument("--max-abs-coefficient", default=2.0, type=float)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    try:
        result = run_execution_layer_v2_regime_conditioned_ev_calibration(
            ExecutionLayerV2RegimeConditionedEVCalibrationConfig(
                run_id=args.run_id,
                input_path=args.input_path,
                output_dir=args.output_dir,
                future_shadow_input_path=args.future_shadow_input_path,
                validation_fraction=args.validation_fraction,
                ridge_alpha=args.ridge_alpha,
                entry_ev_threshold=args.entry_ev_threshold,
                min_fit_rows=args.min_fit_rows,
                min_validation_rows=args.min_validation_rows,
                min_fit_markets=args.min_fit_markets,
                min_validation_markets=args.min_validation_markets,
                max_abs_coefficient=args.max_abs_coefficient,
                overwrite_existing=args.overwrite_existing,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    split = result.split_report
    report = result.calibration_report
    shadow = report["future_shadow"]
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"fit_row_count={split['fit_row_count']}")
    print(f"validation_row_count={split['validation_row_count']}")
    print(f"fit_market_count={split['fit_market_count']}")
    print(f"validation_market_count={split['validation_market_count']}")
    print(f"leakage_checks_passed={str(split['leakage_checks_passed']).lower()}")
    print(f"artifact_created={str(report['artifact_created']).lower()}")
    print(f"artifact_sha256={report['artifact_sha256']}")
    print(f"future_shadow_status={shadow['status']}")
    print(
        "regime_conditioned_ev_produced_count="
        f"{shadow.get('regime_conditioned_ev_produced_count', 0)}"
    )
    print(f"candidate_count={shadow.get('candidate_count', 0)}")
    print(f"full_guard_passed_count={shadow.get('full_guard_passed_count', 0)}")
    print(f"executable_shadow_count={shadow.get('executable_shadow_count', 0)}")
    print(f"blocking_reason_codes={report['blocking_reason_codes']}")
    print("diagnostic_only=true")
    print("paper_only=true")
    print("capital_at_risk=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
