#!/usr/bin/env python3
"""Consume one frozen HTS residual confirmatory input exactly once."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_confirmatory import (
    HTSResidualConfirmatoryEvaluationConfig,
    evaluate_hts_residual_confirmatory_once,
)

DEFAULT_UNLOCK_DIR = Path(
    "examples/v8/polymarket_runs/o-v8-paper-candidate-unlock-20260703T073000Z"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmatory-input-manifest", required=True, type=Path)
    parser.add_argument(
        "--paper-candidate-unlock-dir", type=Path, default=DEFAULT_UNLOCK_DIR
    )
    parser.add_argument("--canonical-o-source-manifest-path", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_hts_residual_confirmatory_once(
        HTSResidualConfirmatoryEvaluationConfig(
            confirmatory_input_manifest_path=args.confirmatory_input_manifest,
            paper_candidate_unlock_dir=args.paper_candidate_unlock_dir,
            canonical_o_source_manifest_path=args.canonical_o_source_manifest_path,
        )
    )
    report = result["report"]
    print(f"output_dir={result['output_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"confirmatory_market_count={report['confirmatory_market_count']}")
    print(f"confirmatory_gate_passed={str(report['confirmatory_gate_passed']).lower()}")
    print(f"pre_promotion_status={report['status']}")
    print(f"pre_promotion_ready={str(report['pre_promotion_ready']).lower()}")
    print("v8_execution_handoff_allowed=false")
    print("paper_run_resume_allowed=false")


if __name__ == "__main__":
    main()
