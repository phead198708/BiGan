#!/usr/bin/env python3
"""Build a post-protocol, development-only HTS residual corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_development_corpus import (
    HTSResidualDevelopmentCorpusConfig,
    build_hts_residual_development_corpus,
)

DEFAULT_PROTOCOL = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_hts_residual_development_protocol_v3.json"
)
DEFAULT_PRIOR_ROWS = Path(
    "examples/v8/polymarket_runs/"
    "v8-hts-residual-edge-power-20260714T055843Z/"
    "hts_residual_edge_power_analysis/hts_post_validation_development_rows.jsonl"
)
DEFAULT_UNLOCK_DIR = Path(
    "examples/v8/polymarket_runs/o-v8-paper-candidate-unlock-20260703T073000Z"
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
        "--source-corpus-dir", action="append", required=True, type=Path
    )
    parser.add_argument(
        "--prior-development-rows-path", type=Path, default=DEFAULT_PRIOR_ROWS
    )
    parser.add_argument(
        "--paper-candidate-unlock-dir", type=Path, default=DEFAULT_UNLOCK_DIR
    )
    parser.add_argument("--canonical-o-source-manifest-path", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_hts_residual_development_corpus(
        HTSResidualDevelopmentCorpusConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol_path,
            expected_protocol_sha256=args.expected_protocol_sha256,
            source_corpus_dirs=tuple(args.source_corpus_dir),
            prior_development_rows_path=args.prior_development_rows_path,
            paper_candidate_unlock_dir=args.paper_candidate_unlock_dir,
            canonical_o_source_manifest_path=args.canonical_o_source_manifest_path,
        )
    )
    report = result["report"]
    print(f"output_dir={result['output_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"residual_market_count={report['residual_market_count']}")
    print(
        "forward_oof_evaluation_ready="
        f"{str(report['forward_oof_evaluation_ready']).lower()}"
    )
    print("candidate_fit_attempted=false")
    print("confirmatory_validation_started=false")


if __name__ == "__main__":
    main()
