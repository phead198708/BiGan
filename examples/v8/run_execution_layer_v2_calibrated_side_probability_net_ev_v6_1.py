"""Run the #210 calibrated side-probability net-EV v6.1 candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_calibrated_side_probability_net_ev_v6_1 import (
    CalibratedSideProbabilityNetEVV61Config,
    run_calibrated_side_probability_net_ev_v6_1,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_calibrated_side_probability_net_ev_v6_1_profile.json"
        ),
    )
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--v5-freeze-manifest", type=Path, required=True)
    parser.add_argument("--v6-profile", type=Path, required=True)
    parser.add_argument("--v6-preregistration-manifest", type=Path, required=True)
    parser.add_argument("--collector-index", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--freeze-created-ts", type=int)
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
    result = run_calibrated_side_probability_net_ev_v6_1(
        CalibratedSideProbabilityNetEVV61Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            v5_freeze_manifest_path=args.v5_freeze_manifest,
            v6_profile_path=args.v6_profile,
            v6_preregistration_manifest_path=args.v6_preregistration_manifest,
            collector_index_path=args.collector_index,
            feature_contract_path=args.feature_contract,
            implementation_commit=args.implementation_commit or _head(),
            freeze_created_ts=args.freeze_created_ts or int(time.time() * 1000),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "target_free_action_viability_passed": report[
                    "target_free_action_viability_passed"
                ],
                "blocking_reason_codes": report[
                    "target_free_action_viability_blocking_reason_codes"
                ],
                "positive_lcb_selected_side_distribution": report[
                    "positive_lcb_selected_side_distribution"
                ],
                "full_guard_accepted_side_market_count": report[
                    "full_guard_accepted_side_market_count"
                ],
                "collector_resume_allowed": report["collector_resume_allowed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
