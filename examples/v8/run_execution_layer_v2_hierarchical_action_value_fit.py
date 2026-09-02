#!/usr/bin/env python3
"""Fit the #171 historical-only hierarchical action-value candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_action_value import (
    HierarchicalActionValueFitConfig,
    fit_historical_hierarchical_action_value,
)

DEFAULT_PROTOCOL = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_hierarchical_action_value_v2.json"
)
DEFAULT_SOURCE_ACTION_PROTOCOL = Path(
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
        "--source-action-protocol-path",
        type=Path,
        default=DEFAULT_SOURCE_ACTION_PROTOCOL,
    )
    parser.add_argument("--expected-source-action-protocol-sha256", required=True)
    parser.add_argument("--historical-corpus-manifest", type=Path, required=True)
    parser.add_argument("--excluded-future-decision-rows", type=Path, required=True)
    parser.add_argument(
        "--expected-excluded-future-decision-rows-sha256", required=True
    )
    parser.add_argument(
        "--excluded-future-artifact-pin",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    return parser


def _artifact_pins(values: list[str]) -> tuple[tuple[Path, str], ...]:
    pins = []
    for value in values:
        path, separator, digest = value.rpartition("=")
        if not separator or not path or not digest:
            raise ValueError("artifact pins must use PATH=SHA256")
        pins.append((Path(path), digest))
    return tuple(pins)


def main() -> None:
    args = _parser().parse_args()
    result = fit_historical_hierarchical_action_value(
        HierarchicalActionValueFitConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol_path,
            expected_protocol_sha256=args.expected_protocol_sha256,
            source_action_protocol_path=args.source_action_protocol_path,
            expected_source_action_protocol_sha256=(
                args.expected_source_action_protocol_sha256
            ),
            historical_corpus_manifest_path=args.historical_corpus_manifest,
            excluded_future_decision_rows_path=args.excluded_future_decision_rows,
            expected_excluded_future_decision_rows_sha256=(
                args.expected_excluded_future_decision_rows_sha256
            ),
            excluded_future_artifact_pins=_artifact_pins(
                args.excluded_future_artifact_pin
            ),
        )
    )
    report = result["validation_report"]
    print(f"run_dir={result['run_dir']}")
    print(f"freeze_manifest={result['freeze_manifest_path']}")
    print(f"freeze_manifest_sha256={result['freeze_manifest_sha256']}")
    print(
        "historical_validation_gate_passed="
        f"{str(report['historical_validation_gate_passed']).lower()}"
    )
    print(
        "historical_validation_gate_blocking_reason_codes="
        f"{report['historical_validation_gate_blocking_reason_codes']}"
    )
    print("source_model_candidate_eligible=false")
    print("promotion_evidence_eligible=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
