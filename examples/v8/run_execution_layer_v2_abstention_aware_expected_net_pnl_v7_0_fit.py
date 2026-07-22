#!/usr/bin/env python3
"""Fit the issue #232 historical-only abstention-aware v7.0 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    AbstentionAwareV70FitConfig,
    run_abstention_aware_v7_0_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--training-profile", required=True, type=Path)
    parser.add_argument("--expected-training-profile-sha256", required=True)
    parser.add_argument("--lineage-audit-manifest", required=True, type=Path)
    parser.add_argument("--runtime-target-rows", required=True, type=Path)
    parser.add_argument("--full-action-grid-rows", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--fit-created-ts", required=True, type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_abstention_aware_v7_0_fit(
        AbstentionAwareV70FitConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            training_profile_path=args.training_profile,
            expected_training_profile_sha256=(
                args.expected_training_profile_sha256
            ),
            lineage_audit_manifest_path=args.lineage_audit_manifest,
            runtime_target_rows_path=args.runtime_target_rows,
            full_action_grid_rows_path=args.full_action_grid_rows,
            implementation_commit=args.implementation_commit,
            fit_created_ts=args.fit_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "historical_development_gate_passed": result["model"][
                    "historical_development_gate_passed"
                ],
                "historical_development_blocking_reason_codes": result["model"][
                    "historical_development_blocking_reason_codes"
                ],
                "fit_leakage_audit_passed": result["leakage_audit"][
                    "fit_leakage_audit_passed"
                ],
                "family_gate_results": result["model"]["family_gate_results"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "paper_candidate_allowed": False,
                "live_trading_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
