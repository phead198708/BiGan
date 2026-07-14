#!/usr/bin/env python3
"""Run the development-only HTS residual-edge and power analysis."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_edge import (
    HTSResidualEdgePowerConfig,
    run_hts_residual_edge_power_analysis,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-estimand-goal-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=Path("examples/v8/polymarket_runs"), type=Path)
    parser.add_argument("--repository-root", default=Path.cwd(), type=Path)
    parser.add_argument("--bootstrap-samples", default=2_000, type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_hts_residual_edge_power_analysis(
        HTSResidualEdgePowerConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            source_estimand_goal_dir=args.source_estimand_goal_dir,
            created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            bootstrap_samples=args.bootstrap_samples,
        )
    )
    print(f"analysis_dir={result['analysis_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"selected_candidate_name={result['selected_candidate_name']}")
    print(
        "development_candidate_gate_passed="
        f"{str(result['development_candidate_gate_passed']).lower()}"
    )
    print(f"incremental_signal_status={result['incremental_signal_status']}")
    print(
        "recommended_minimum_fresh_confirmatory_markets="
        f"{result['recommended_minimum_fresh_confirmatory_markets']}"
    )
    print("fresh_confirmatory_validation_start_allowed=false")
    print("paper_live_promotion_unlock=false")


if __name__ == "__main__":
    main()
