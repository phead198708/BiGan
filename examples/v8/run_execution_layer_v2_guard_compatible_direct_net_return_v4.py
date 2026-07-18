#!/usr/bin/env python3
"""Run the one-shot #202 guard-compatible direct net-return v4 fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    GuardCompatibleDirectNetReturnV4Config,
    fit_guard_compatible_direct_net_return_v4,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--fit-profile", type=Path, required=True)
    parser.add_argument("--fit-profile-sha256", required=True)
    parser.add_argument("--issue198-candidate-manifest", type=Path, required=True)
    parser.add_argument("--issue198-candidate-manifest-sha256", required=True)
    parser.add_argument("--issue201-manifest", type=Path, required=True)
    parser.add_argument("--issue201-manifest-sha256", required=True)
    parser.add_argument("--role-assignment-manifest", type=Path, required=True)
    parser.add_argument("--role-assignment-manifest-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = fit_guard_compatible_direct_net_return_v4(
        GuardCompatibleDirectNetReturnV4Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            fit_profile_path=args.fit_profile,
            expected_fit_profile_sha256=args.fit_profile_sha256,
            issue198_candidate_manifest_path=args.issue198_candidate_manifest,
            expected_issue198_candidate_manifest_sha256=(args.issue198_candidate_manifest_sha256),
            issue201_manifest_path=args.issue201_manifest,
            expected_issue201_manifest_sha256=args.issue201_manifest_sha256,
            role_assignment_manifest_path=args.role_assignment_manifest,
            expected_role_assignment_manifest_sha256=(args.role_assignment_manifest_sha256),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["gate_report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "oof_market_count": report["oof_market_count"],
                "guard_accepted_bet_count": report["guard_accepted_bet_count"],
                "guard_accepted_unique_market_count": report["guard_accepted_unique_market_count"],
                "accepted_bet_net_pnl_sum": report["accepted_bet_net_pnl_sum"],
                "all_oof_market_policy_pnl_lcb": report["all_oof_market_policy_pnl"][
                    "lower_confidence_bound"
                ],
                "development_gate_passed": report["development_gate_passed"],
                "candidate_specific_future_evaluation_allowed": report[
                    "candidate_specific_future_evaluation_allowed"
                ],
                "gate_blocking_reason_codes": report["gate_blocking_reason_codes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
