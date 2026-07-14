#!/usr/bin/env python3
"""Run the frozen, development-only HTS residual forward OOF evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_development_corpus import (
    HTSResidualForwardOOFConfig,
    run_hts_residual_development_forward_oof,
)

DEFAULT_PROTOCOL = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_hts_residual_development_protocol_v2.json"
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
        "--development-corpus-manifest", action="append", required=True, type=Path
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_hts_residual_development_forward_oof(
        HTSResidualForwardOOFConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol_path,
            expected_protocol_sha256=args.expected_protocol_sha256,
            development_corpus_manifest_paths=tuple(
                args.development_corpus_manifest
            ),
        )
    )
    report = result["report"]
    print(f"output_dir={result['output_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"selected_candidate_name={report['selected_candidate_name']}")
    print(
        "development_candidate_gate_passed="
        f"{str(report['development_candidate_gate_passed']).lower()}"
    )
    print("candidate_frozen=false")
    print("confirmatory_validation_started=false")


if __name__ == "__main__":
    main()
