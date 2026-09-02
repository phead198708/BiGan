#!/usr/bin/env python3
"""Run the issue #232 historical-only lineage and exclusion audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    AbstentionAwareV70LineageAuditConfig,
    run_v7_0_lineage_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--runtime-target-manifest", required=True, type=Path)
    parser.add_argument("--runtime-target-rows", required=True, type=Path)
    parser.add_argument("--full-action-grid-manifest", required=True, type=Path)
    parser.add_argument("--full-action-grid-rows", required=True, type=Path)
    parser.add_argument(
        "--issue229-target-free-freeze-manifest", required=True, type=Path
    )
    parser.add_argument(
        "--issue231-target-free-freeze-manifest", required=True, type=Path
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--audit-created-ts", required=True, type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_v7_0_lineage_audit(
        AbstentionAwareV70LineageAuditConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            runtime_target_manifest_path=args.runtime_target_manifest,
            runtime_target_rows_path=args.runtime_target_rows,
            full_action_grid_manifest_path=args.full_action_grid_manifest,
            full_action_grid_rows_path=args.full_action_grid_rows,
            issue229_target_free_freeze_manifest_path=(
                args.issue229_target_free_freeze_manifest
            ),
            issue231_target_free_freeze_manifest_path=(
                args.issue231_target_free_freeze_manifest
            ),
            implementation_commit=args.implementation_commit,
            audit_created_ts=args.audit_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "lineage_audit_passed": result["audit"]["lineage_audit_passed"],
                "lineage_audit_blocking_reason_codes": result["audit"][
                    "lineage_audit_blocking_reason_codes"
                ],
                "historical_unique_market_count": result["audit"][
                    "historical_unique_market_count"
                ],
                "excluded_future_unique_market_count": result["audit"][
                    "excluded_future_unique_market_count"
                ],
                "audit_path": str(result["audit_path"]),
                "audit_sha256": result["audit_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "paper_candidate_allowed": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
