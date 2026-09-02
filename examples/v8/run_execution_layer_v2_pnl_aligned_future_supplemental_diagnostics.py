#!/usr/bin/env python3
"""Run frozen #169 report-only accepted-bet diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_diagnostics import (
    run_pnl_aligned_future_supplemental_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--diagnostics-freeze-manifest", required=True, type=Path)
    parser.add_argument("--expected-diagnostics-freeze-manifest-sha256", required=True)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--expected-evaluation-manifest-sha256", required=True)
    args = parser.parse_args()
    result = run_pnl_aligned_future_supplemental_diagnostics(
        run_id=args.run_id,
        output_dir=args.output_dir,
        diagnostics_freeze_manifest_path=args.diagnostics_freeze_manifest,
        expected_diagnostics_freeze_manifest_sha256=(
            args.expected_diagnostics_freeze_manifest_sha256
        ),
        evaluation_manifest_path=args.evaluation_manifest,
        expected_evaluation_manifest_sha256=args.expected_evaluation_manifest_sha256,
    )
    report = result["report"]
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"status={report['status']}")
    print(
        "primary_future_evidence_gate_passed="
        f"{str(report['primary_future_evidence_gate_passed']).lower()}"
    )
    print("supplemental_diagnostics_report_only=true")
    print("supplemental_diagnostics_can_mutate_primary_gate=false")


if __name__ == "__main__":
    main()
