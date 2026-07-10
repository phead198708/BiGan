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
    parser.add_argument("--probability-price-tolerance", default=1e-9, type=float)
    parser.add_argument("--min-relative-mae-improvement", default=0.05, type=float)
    parser.add_argument("--min-relative-mse-improvement", default=0.05, type=float)
    parser.add_argument("--bootstrap-samples", default=1_000, type=int)
    parser.add_argument("--bootstrap-confidence-level", default=0.95, type=float)
    parser.add_argument(
        "--min-bootstrap-improvement-lower-bound", default=0.0, type=float
    )
    parser.add_argument(
        "--max-lomo-coefficient-absolute-deviation", default=0.50, type=float
    )
    parser.add_argument(
        "--min-lomo-coefficient-sign-agreement", default=0.75, type=float
    )
    parser.add_argument("--min-validation-rows-per-side", default=5, type=int)
    parser.add_argument(
        "--min-validation-rows-per-action-family", default=5, type=int
    )
    parser.add_argument(
        "--min-validation-rows-per-resolved-outcome", default=5, type=int
    )
    parser.add_argument(
        "--min-validation-markets-per-category", default=2, type=int
    )
    parser.add_argument("--statistical-random-seed", default=17_029, type=int)
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
                probability_price_tolerance=args.probability_price_tolerance,
                min_relative_mae_improvement=args.min_relative_mae_improvement,
                min_relative_mse_improvement=args.min_relative_mse_improvement,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_confidence_level=args.bootstrap_confidence_level,
                min_bootstrap_improvement_lower_bound=(
                    args.min_bootstrap_improvement_lower_bound
                ),
                max_lomo_coefficient_absolute_deviation=(
                    args.max_lomo_coefficient_absolute_deviation
                ),
                min_lomo_coefficient_sign_agreement=(
                    args.min_lomo_coefficient_sign_agreement
                ),
                min_validation_rows_per_side=(
                    args.min_validation_rows_per_side
                ),
                min_validation_rows_per_action_family=(
                    args.min_validation_rows_per_action_family
                ),
                min_validation_rows_per_resolved_outcome=(
                    args.min_validation_rows_per_resolved_outcome
                ),
                min_validation_markets_per_category=(
                    args.min_validation_markets_per_category
                ),
                statistical_random_seed=args.statistical_random_seed,
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
    print(
        "schema_runtime_validation_agreement_passed="
        f"{str(split['schema_runtime_validation_agreement_passed']).lower()}"
    )
    print(f"invalid_row_reason_distribution={split['invalid_row_reason_distribution']}")
    print(
        "statistical_eligibility_passed="
        f"{str(report['statistical_eligibility_passed']).lower()}"
    )
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
