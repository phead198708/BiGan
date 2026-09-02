#!/usr/bin/env python3
"""Freeze #169 report-only diagnostics before future reconciliation."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_diagnostics import (
    PnLAlignedFutureDiagnosticsFreezeConfig,
    freeze_pnl_aligned_future_supplemental_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--evaluation-freeze-manifest", required=True, type=Path)
    parser.add_argument("--expected-evaluation-freeze-manifest-sha256", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    result = freeze_pnl_aligned_future_supplemental_diagnostics(
        PnLAlignedFutureDiagnosticsFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=args.protocol,
            expected_protocol_sha256=args.expected_protocol_sha256,
            evaluation_freeze_manifest_path=args.evaluation_freeze_manifest,
            expected_evaluation_freeze_manifest_sha256=(
                args.expected_evaluation_freeze_manifest_sha256
            ),
            git_commit=args.git_commit,
        )
    )
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print("report_only=true")
    print("primary_future_evidence_gate_mutation_allowed=false")
    print("future_outcome_targets_loaded=false")


if __name__ == "__main__":
    main()
