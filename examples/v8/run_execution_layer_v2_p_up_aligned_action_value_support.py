#!/usr/bin/env python3
"""Run the frozen #201 p_up-aligned action-value support audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_p_up_aligned_action_value_support import (
    PUpAlignedActionValueSupportConfig,
    run_p_up_aligned_action_value_support_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--audit-profile", type=Path, required=True)
    parser.add_argument("--audit-profile-sha256", required=True)
    parser.add_argument("--issue198-candidate-manifest", type=Path, required=True)
    parser.add_argument("--issue198-candidate-manifest-sha256", required=True)
    parser.add_argument("--issue200-manifest", type=Path, required=True)
    parser.add_argument("--issue200-manifest-sha256", required=True)
    parser.add_argument("--role-assignment-manifest", type=Path, required=True)
    parser.add_argument("--role-assignment-manifest-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_p_up_aligned_action_value_support_audit(
        PUpAlignedActionValueSupportConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            audit_profile_path=args.audit_profile,
            expected_audit_profile_sha256=args.audit_profile_sha256,
            issue198_candidate_manifest_path=args.issue198_candidate_manifest,
            expected_issue198_candidate_manifest_sha256=(args.issue198_candidate_manifest_sha256),
            issue200_manifest_path=args.issue200_manifest,
            expected_issue200_manifest_sha256=args.issue200_manifest_sha256,
            role_assignment_manifest_path=args.role_assignment_manifest,
            expected_role_assignment_manifest_sha256=(args.role_assignment_manifest_sha256),
            overwrite_existing=args.overwrite_existing,
        )
    )
    focal = result["support_report"]["segment_metrics"][
        "p_up_aligned_execution_quality_passed_trade_actions"
    ]
    interval = focal.get("market_level_post_cost_return") or {}
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "p_up_aligned_execution_quality_row_count": focal["row_count"],
                "p_up_aligned_execution_quality_unique_market_count": focal["unique_market_count"],
                "p_up_aligned_execution_quality_mean": focal["target_post_cost_return_mean"],
                "p_up_aligned_execution_quality_market_lcb": interval.get("lower_confidence_bound"),
                "support_conclusion": result["support_report"]["support_conclusion"],
                "root_cause_classification": result["attribution_report"][
                    "root_cause_classification"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
