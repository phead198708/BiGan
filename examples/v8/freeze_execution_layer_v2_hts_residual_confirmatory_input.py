#!/usr/bin/env python3
"""Freeze future HTS residual confirmatory inputs without reading outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_confirmatory import (
    HTSResidualConfirmatoryInputConfig,
    freeze_hts_residual_confirmatory_input,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--candidate-freeze-manifest", required=True, type=Path)
    parser.add_argument("--source-corpus-dir", action="append", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = freeze_hts_residual_confirmatory_input(
        HTSResidualConfirmatoryInputConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            candidate_freeze_manifest_path=args.candidate_freeze_manifest,
            source_corpus_dirs=tuple(args.source_corpus_dir),
        )
    )
    manifest = result["manifest"]
    print(f"output_dir={result['output_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"source_market_count={manifest['source_market_count']}")
    print(f"input_gate_passed={str(manifest['input_gate_passed']).lower()}")
    print("outcome_values_inspected_during_input_freeze=false")
    print("confirmatory_evaluation_started=false")
    print("pre_promotion_ready=false")


if __name__ == "__main__":
    main()
