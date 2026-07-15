#!/usr/bin/env python3
"""Freeze #172 market roles and exclusions before public-data collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_cross_fitted_family_lcb import (
    CrossFittedFamilyLCBPrecollectionFreezeConfig,
    freeze_cross_fitted_family_lcb_precollection,
)

DEFAULT_PROTOCOL = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_cross_fitted_family_lcb_v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument(
        "--prior-market-registry-pin",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    parser.add_argument(
        "--prior-evidence-artifact-pin",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    parser.add_argument("--expected-prior-unique-market-count", type=int, default=95)
    return parser


def _pins(values: list[str]) -> tuple[tuple[Path, str], ...]:
    pins = []
    for value in values:
        path, separator, digest = value.rpartition("=")
        if not separator or not path or not digest:
            raise ValueError("artifact pins must use PATH=SHA256")
        pins.append((Path(path), digest))
    return tuple(pins)


def main() -> None:
    args = _parser().parse_args()
    result = freeze_cross_fitted_family_lcb_precollection(
        CrossFittedFamilyLCBPrecollectionFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol_path,
            expected_protocol_sha256=args.expected_protocol_sha256,
            git_commit=args.git_commit,
            prior_market_registry_pins=_pins(args.prior_market_registry_pin),
            prior_evidence_artifact_pins=_pins(args.prior_evidence_artifact_pin),
            expected_prior_unique_market_count=args.expected_prior_unique_market_count,
        )
    )
    print(f"output_dir={result['output_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"descriptor_path={result['descriptor_path']}")
    print(f"descriptor_sha256={result['descriptor_sha256']}")
    print("collection_started=false")
    print("paper_only=true")
    print("capital_at_risk=false")


if __name__ == "__main__":
    main()
