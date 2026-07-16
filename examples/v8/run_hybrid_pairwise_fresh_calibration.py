#!/usr/bin/env python3
"""Freeze fresh LCB calibration for the #182 historical ranker."""

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
    HybridPairwiseFreshCalibrationConfig,
    freeze_hybrid_pairwise_fresh_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    for name in (
        "hybrid-protocol",
        "source-pairwise-protocol",
        "feature-contract",
        "historical-ranker-descriptor",
        "historical-ranker-manifest",
        "fresh-role-assignment-manifest",
    ):
        parser.add_argument(f"--{name}", required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = freeze_hybrid_pairwise_fresh_calibration(
        HybridPairwiseFreshCalibrationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            hybrid_protocol_path=Path(args.hybrid_protocol),
            expected_hybrid_protocol_sha256=args.hybrid_protocol_sha256,
            source_pairwise_protocol_path=Path(
                args.source_pairwise_protocol
            ),
            expected_source_pairwise_protocol_sha256=(
                args.source_pairwise_protocol_sha256
            ),
            feature_contract_path=Path(args.feature_contract),
            expected_feature_contract_sha256=args.feature_contract_sha256,
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
            fresh_role_assignment_manifest_path=Path(
                args.fresh_role_assignment_manifest
            ),
            expected_fresh_role_assignment_manifest_sha256=(
                args.fresh_role_assignment_manifest_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "development_gate_passed": result[
                    "development_gate_passed"
                ],
                "freeze_manifest_path": str(
                    result["freeze_manifest_path"]
                ),
                "freeze_manifest_sha256": result[
                    "freeze_manifest_sha256"
                ],
                "confirmatory_evaluation_started": False,
                "ranker_retrained": False,
                "ranker_score_mutated": False,
                "paper_only": True,
                "capital_at_risk": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
