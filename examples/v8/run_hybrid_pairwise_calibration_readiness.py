#!/usr/bin/env python3
"""Evaluate #184 calibration readiness without opening labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration import (  # noqa: E402
    HybridPairwiseCalibrationReadinessConfig,
    evaluate_hybrid_pairwise_calibration_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--hybrid-protocol", required=True)
    parser.add_argument("--hybrid-protocol-sha256", required=True)
    parser.add_argument("--historical-ranker-descriptor", required=True)
    parser.add_argument(
        "--historical-ranker-descriptor-sha256",
        required=True,
    )
    parser.add_argument("--historical-ranker-manifest", required=True)
    parser.add_argument(
        "--historical-ranker-manifest-sha256",
        required=True,
    )
    parser.add_argument("--upstream-terminal-freeze-state", required=True)
    parser.add_argument(
        "--upstream-terminal-freeze-state-sha256",
        required=True,
    )
    parser.add_argument("--fresh-role-assignment-manifest")
    parser.add_argument("--fresh-role-assignment-manifest-sha256")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = evaluate_hybrid_pairwise_calibration_readiness(
        HybridPairwiseCalibrationReadinessConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            hybrid_protocol_path=Path(args.hybrid_protocol),
            expected_hybrid_protocol_sha256=args.hybrid_protocol_sha256,
            historical_ranker_descriptor_path=Path(
                args.historical_ranker_descriptor
            ),
            expected_historical_ranker_descriptor_sha256=(
                args.historical_ranker_descriptor_sha256
            ),
            historical_ranker_manifest_path=Path(
                args.historical_ranker_manifest
            ),
            expected_historical_ranker_manifest_sha256=(
                args.historical_ranker_manifest_sha256
            ),
            upstream_terminal_freeze_state_path=Path(
                args.upstream_terminal_freeze_state
            ),
            expected_upstream_terminal_freeze_state_sha256=(
                args.upstream_terminal_freeze_state_sha256
            ),
            fresh_role_assignment_manifest_path=(
                Path(args.fresh_role_assignment_manifest)
                if args.fresh_role_assignment_manifest
                else None
            ),
            expected_fresh_role_assignment_manifest_sha256=(
                args.fresh_role_assignment_manifest_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "readiness_status": report["readiness_status"],
                "calibration_readiness_passed": report[
                    "calibration_readiness_passed"
                ],
                "calibration_start_allowed": report[
                    "calibration_start_allowed"
                ],
                "blocking_reason_codes": report[
                    "blocking_reason_codes"
                ],
                "report_path": str(result["report_path"]),
                "manifest_path": str(result["manifest_path"]),
                "model_prediction_attempted": False,
                "label_or_outcome_artifacts_opened": False,
                "paper_only": True,
                "capital_at_risk": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
