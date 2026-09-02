"""Fit/calibrate or evaluate the preregistered #223 v6.3 candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_sbc_exit_reliability_v6_3_fit import (
    SBCExitReliabilityV63FitConfig,
    run_sbc_exit_reliability_v6_3_fit,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit_calibrate", "evaluate_oof"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument(
        "--fit-profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_sbc_exit_reliability_v6_3_fit_profile.json"
        ),
    )
    parser.add_argument("--expected-fit-profile-sha256", required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--v6-2-historical-manifest", type=Path, required=True)
    parser.add_argument("--threshold-freeze-manifest", type=Path)
    parser.add_argument("--expected-threshold-freeze-manifest-sha256")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    result = run_sbc_exit_reliability_v6_3_fit(
        SBCExitReliabilityV63FitConfig(
            stage=args.stage,
            run_id=args.run_id,
            output_dir=args.output_dir,
            fit_profile_path=args.fit_profile,
            expected_fit_profile_sha256=args.expected_fit_profile_sha256,
            audit_manifest_path=args.audit_manifest,
            v6_2_historical_manifest_path=args.v6_2_historical_manifest,
            implementation_commit=args.implementation_commit or _head(),
            threshold_freeze_manifest_path=args.threshold_freeze_manifest,
            expected_threshold_freeze_manifest_sha256=(
                args.expected_threshold_freeze_manifest_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    if args.stage == "fit_calibrate":
        payload = {
            "run_dir": str(result["run_dir"]),
            "threshold_freeze_manifest_path": str(
                result["threshold_freeze_manifest_path"]
            ),
            "threshold_freeze_manifest_sha256": result[
                "threshold_freeze_manifest_sha256"
            ],
            "calibration_gate_passed": result["calibration"][
                "calibration_gate_passed"
            ],
            "calibration_gate_reason_codes": result["calibration"][
                "calibration_gate_reason_codes"
            ],
            "selected_threshold": result["calibration"]["selected_threshold"],
            "oof_evaluation_allowed": result["freeze"]["oof_evaluation_allowed"],
        }
    else:
        payload = {
            "run_dir": str(result["run_dir"]),
            "candidate_manifest_path": str(result["candidate_manifest_path"]),
            "candidate_manifest_sha256": result["candidate_manifest_sha256"],
            "historical_side_only_oof_gate_passed": result["report"][
                "historical_side_only_oof_gate_passed"
            ],
            "historical_side_only_oof_gate_reason_codes": result["report"][
                "historical_side_only_oof_gate_reason_codes"
            ],
            "future_candidate_freeze_step_allowed": result["manifest"][
                "future_candidate_freeze_step_allowed"
            ],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
