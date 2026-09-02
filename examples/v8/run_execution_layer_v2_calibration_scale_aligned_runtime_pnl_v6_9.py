#!/usr/bin/env python3
"""Run the issue #231 v6.9 scale-aligned mapping and liveness freeze."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9 import (
    CalibrationScaleAlignedV69Config,
    run_calibration_scale_aligned_v6_9,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--runtime-target-manifest", required=True)
    parser.add_argument("--runtime-target-rows", required=True)
    parser.add_argument("--failed-v6-8-calibration-artifact", required=True)
    parser.add_argument("--issue229-target-free-freeze-manifest", required=True)
    parser.add_argument("--issue229-v6-7-base-selected-rows", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument(
        "--candidate-freeze-created-ts",
        type=int,
        default=None,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_calibration_scale_aligned_v6_9(
        CalibrationScaleAlignedV69Config(
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
            profile_path=Path(args.profile),
            expected_profile_sha256=args.expected_profile_sha256,
            runtime_target_manifest_path=Path(args.runtime_target_manifest),
            runtime_target_rows_path=Path(args.runtime_target_rows),
            failed_v6_8_calibration_artifact_path=Path(
                args.failed_v6_8_calibration_artifact
            ),
            issue229_target_free_freeze_manifest_path=Path(
                args.issue229_target_free_freeze_manifest
            ),
            issue229_v6_7_base_selected_rows_path=Path(
                args.issue229_v6_7_base_selected_rows
            ),
            implementation_commit=args.implementation_commit,
            candidate_freeze_created_ts=(
                args.candidate_freeze_created_ts or int(time.time() * 1000)
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "mapping_gate_passed": result["mapping_artifact"][
                    "mapping_gate_passed"
                ],
                "target_free_liveness_gate_passed": result["report"][
                    "target_free_liveness_gate_passed"
                ],
                "positive_mapped_score_unique_market_count": result["report"][
                    "positive_mapped_score_unique_market_count"
                ],
                "guard_accepted_unique_market_count": result["report"][
                    "guard_accepted_unique_market_count"
                ],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
