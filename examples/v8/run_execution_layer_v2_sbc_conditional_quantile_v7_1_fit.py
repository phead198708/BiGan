#!/usr/bin/env python3
"""Fit the issue #233 SBC conditional-quantile v7.1 candidate once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_sbc_conditional_quantile_v7_1 import (
    SbcConditionalQuantileV71Config,
    run_sbc_conditional_quantile_v7_1_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--v7-0-training-profile", required=True, type=Path)
    parser.add_argument("--runtime-target-rows", required=True, type=Path)
    parser.add_argument("--v7-0-lineage-audit-manifest", required=True, type=Path)
    parser.add_argument(
        "--v7-0-fit-manifest-rejection-evidence", required=True, type=Path
    )
    parser.add_argument("--v7-0-model-rejection-evidence", required=True, type=Path)
    parser.add_argument(
        "--v7-0-fit-report-rejection-evidence", required=True, type=Path
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--fit-created-ts", required=True, type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_sbc_conditional_quantile_v7_1_fit(
        SbcConditionalQuantileV71Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            v7_0_training_profile_path=args.v7_0_training_profile,
            runtime_target_rows_path=args.runtime_target_rows,
            v7_0_lineage_audit_manifest_path=args.v7_0_lineage_audit_manifest,
            v7_0_fit_manifest_rejection_evidence_path=(
                args.v7_0_fit_manifest_rejection_evidence
            ),
            v7_0_model_rejection_evidence_path=(
                args.v7_0_model_rejection_evidence
            ),
            v7_0_fit_report_rejection_evidence_path=(
                args.v7_0_fit_report_rejection_evidence
            ),
            implementation_commit=args.implementation_commit,
            fit_created_ts=args.fit_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    replay = result["model"]["historical_replay_superiority_gate"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "historical_gate_passed": result["model"][
                    "historical_gate_passed"
                ],
                "historical_gate_blocking_reason_codes": result["model"][
                    "historical_gate_blocking_reason_codes"
                ],
                "candidate_total_after_cost_net_pnl_at_frozen_size": replay[
                    "candidate_total_after_cost_net_pnl_at_frozen_size"
                ],
                "v6_7_baseline_total_after_cost_net_pnl_at_frozen_size": replay[
                    "v6_7_baseline_total_after_cost_net_pnl_at_frozen_size"
                ],
                "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": (
                    replay[
                        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
                    ]
                ),
                "target_free_canary_collection_allowed": result["model"][
                    "target_free_canary_collection_allowed"
                ],
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
