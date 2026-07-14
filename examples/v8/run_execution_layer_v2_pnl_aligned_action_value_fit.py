#!/usr/bin/env python3
"""Fit the frozen research-only PnL-aligned v8 action-value candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    PnLAlignedActionValueFitConfig,
    fit_frozen_pnl_aligned_action_value_model,
)

DEFAULT_PROTOCOL = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pnl_aligned_action_value_v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--historical-corpus-manifest", required=True, type=Path
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = fit_frozen_pnl_aligned_action_value_model(
        PnLAlignedActionValueFitConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol_path,
            expected_protocol_sha256=args.expected_protocol_sha256,
            historical_corpus_manifest_path=args.historical_corpus_manifest,
        )
    )
    print(f"run_dir={result['run_dir']}")
    print(f"model_path={result['model_path']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print("research_artifact_frozen=true")
    print("future_unseen_evaluation_required=true")
    print("source_model_candidate_eligible=false")
    print("promotion_evidence_eligible=false")


if __name__ == "__main__":
    main()
