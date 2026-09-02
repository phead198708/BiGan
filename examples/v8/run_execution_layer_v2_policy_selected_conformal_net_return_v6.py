#!/usr/bin/env python3
"""Fit and freeze the preregistered #207 policy-selected conformal v6 candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (  # noqa: E402
    PolicySelectedConformalNetReturnV6FitConfig,
    fit_policy_selected_conformal_net_return_v6,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--development-settlement-manifest", required=True)
    parser.add_argument("--development-settlement-manifest-sha256", required=True)
    parser.add_argument("--feature-contract", required=True)
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--candidate-freeze-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = fit_policy_selected_conformal_net_return_v6(
        PolicySelectedConformalNetReturnV6FitConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.profile_sha256,
            development_settlement_manifest_path=args.development_settlement_manifest,
            expected_development_settlement_manifest_sha256=(
                args.development_settlement_manifest_sha256
            ),
            feature_contract_path=args.feature_contract,
            expected_feature_contract_sha256=args.feature_contract_sha256,
            implementation_commit=args.implementation_commit,
            candidate_freeze_created_ts=args.candidate_freeze_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["calibration_report"]
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "point_model_fit_market_count": report["point_model_fit_market_count"],
                "conformal_calibration_market_count": report[
                    "conformal_calibration_market_count"
                ],
                "calibration_check_market_count": report[
                    "calibration_check_market_count"
                ],
                "policy_selected_calibration_market_count": report[
                    "policy_selected_calibration"
                ]["selected_calibration_market_count"],
                "target_free_calibration_check_support": report[
                    "target_free_calibration_check_support"
                ],
                "calibration_gate_passed": report["calibration_gate_passed"],
                "calibration_gate_blocking_reason_codes": report[
                    "calibration_gate_blocking_reason_codes"
                ],
                "research_candidate_frozen": manifest["research_candidate_frozen"],
                "candidate_specific_future_evaluation_allowed": manifest[
                    "candidate_specific_future_evaluation_allowed"
                ],
                "model_sha256": manifest["model_sha256"],
                "policy_dataset_hash": manifest["policy_dataset_hash"],
                "split_hash": manifest["split_hash"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "uses_204_outcomes_for_fitting": False,
                "policy_pnl_computed": False,
                "paper_candidate_allowed": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["calibration_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
