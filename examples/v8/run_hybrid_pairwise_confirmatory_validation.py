#!/usr/bin/env python3
"""Run the one-shot hybrid pairwise confirmatory validation."""

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
    HybridPairwiseConfirmatoryConfig,
    evaluate_hybrid_pairwise_confirmatory_once,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--calibration-freeze-manifest", required=True)
    parser.add_argument(
        "--calibration-freeze-manifest-sha256",
        required=True,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = evaluate_hybrid_pairwise_confirmatory_once(
        HybridPairwiseConfirmatoryConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            calibration_freeze_manifest_path=Path(
                args.calibration_freeze_manifest
            ),
            expected_calibration_freeze_manifest_sha256=(
                args.calibration_freeze_manifest_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "confirmatory_gate_passed": result[
                    "confirmatory_gate_passed"
                ],
                "candidate_freeze_path": str(
                    result["candidate_freeze_path"]
                ),
                "candidate_freeze_sha256": result[
                    "candidate_freeze_sha256"
                ],
                "source_model_candidate_eligible": False,
                "promotion_evidence_eligible": False,
                "future_unseen_execution_holdout_required": True,
                "paper_only": True,
                "capital_at_risk": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
