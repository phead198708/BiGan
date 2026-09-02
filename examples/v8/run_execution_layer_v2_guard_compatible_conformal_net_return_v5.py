#!/usr/bin/env python3
"""Fit and freeze the prospective #203 conformal net-return v5 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    GuardCompatibleConformalNetReturnV5Config,
    fit_guard_compatible_conformal_net_return_v5,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--fit-profile", type=Path, required=True)
    parser.add_argument("--fit-profile-sha256", required=True)
    parser.add_argument("--role-assignment-manifest", type=Path, required=True)
    parser.add_argument("--role-assignment-manifest-sha256", required=True)
    parser.add_argument("--accepted-bet-power-manifest", type=Path, required=True)
    parser.add_argument("--accepted-bet-power-manifest-sha256", required=True)
    parser.add_argument("--accepted-bet-power-report", type=Path, required=True)
    parser.add_argument("--accepted-bet-power-report-sha256", required=True)
    parser.add_argument("--issue201-manifest", type=Path, required=True)
    parser.add_argument("--issue201-manifest-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--candidate-freeze-created-ts", required=True, type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = fit_guard_compatible_conformal_net_return_v5(
        GuardCompatibleConformalNetReturnV5Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            fit_profile_path=args.fit_profile,
            expected_fit_profile_sha256=args.fit_profile_sha256,
            role_assignment_manifest_path=args.role_assignment_manifest,
            expected_role_assignment_manifest_sha256=args.role_assignment_manifest_sha256,
            accepted_bet_power_manifest_path=args.accepted_bet_power_manifest,
            expected_accepted_bet_power_manifest_sha256=(args.accepted_bet_power_manifest_sha256),
            accepted_bet_power_report_path=args.accepted_bet_power_report,
            expected_accepted_bet_power_report_sha256=(args.accepted_bet_power_report_sha256),
            issue201_manifest_path=args.issue201_manifest,
            expected_issue201_manifest_sha256=args.issue201_manifest_sha256,
            implementation_commit=args.implementation_commit,
            candidate_freeze_created_ts=args.candidate_freeze_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["calibration_report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "fit_market_count": 135,
                "calibration_market_count": 60,
                "calibration_gate_passed": report["calibration_gate_passed"],
                "calibration_gate_blocking_reason_codes": report[
                    "calibration_gate_blocking_reason_codes"
                ],
                "candidate_specific_future_evaluation_allowed": result["manifest"][
                    "candidate_specific_future_evaluation_allowed"
                ],
                "eligible_future_collection": result["manifest"]["eligible_future_collection"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
