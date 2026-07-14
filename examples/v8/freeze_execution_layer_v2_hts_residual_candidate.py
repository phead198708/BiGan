#!/usr/bin/env python3
"""Freeze one development-gated HTS residual candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_confirmatory import (
    HTSResidualCandidateFreezeConfig,
    freeze_hts_residual_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--development-oof-manifest", required=True, type=Path)
    parser.add_argument("--minimum-confirmatory-market-count", type=int, default=283)
    parser.add_argument(
        "--minimum-confirmatory-source-run-count", type=int, default=24
    )
    parser.add_argument("--minimum-input-source-market-count", type=int, default=283)
    parser.add_argument("--minimum-input-hts-market-count", type=int, default=283)
    parser.add_argument("--minimum-relative-brier-improvement", type=float, default=0.03)
    parser.add_argument(
        "--minimum-relative-log-loss-improvement", type=float, default=0.03
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = freeze_hts_residual_candidate(
        HTSResidualCandidateFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            development_oof_manifest_path=args.development_oof_manifest,
            minimum_confirmatory_market_count=args.minimum_confirmatory_market_count,
            minimum_confirmatory_source_run_count=(
                args.minimum_confirmatory_source_run_count
            ),
            minimum_input_source_market_count=args.minimum_input_source_market_count,
            minimum_input_hts_market_count=args.minimum_input_hts_market_count,
            minimum_relative_brier_improvement=(
                args.minimum_relative_brier_improvement
            ),
            minimum_relative_log_loss_improvement=(
                args.minimum_relative_log_loss_improvement
            ),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_confidence_level=args.bootstrap_confidence_level,
            bootstrap_seed=args.bootstrap_seed,
        )
    )
    report = result["report"]
    print(f"output_dir={result['output_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"candidate_name={report['candidate_name']}")
    print("candidate_frozen=true")
    print("confirmatory_validation_started=false")
    print("pre_promotion_ready=false")


if __name__ == "__main__":
    main()
