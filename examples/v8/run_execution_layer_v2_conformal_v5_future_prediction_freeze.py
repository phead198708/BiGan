"""Freeze #204 v5 and matched-v4 target-free future guard decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (  # noqa: E402
    ConformalV5FuturePredictionFreezeConfig,
    freeze_conformal_v5_future_predictions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--binding-manifest", required=True)
    parser.add_argument("--binding-manifest-sha256", required=True)
    parser.add_argument("--feature-contract", required=True)
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--decision-freeze-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = freeze_conformal_v5_future_predictions(
        ConformalV5FuturePredictionFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            binding_manifest_path=args.binding_manifest,
            expected_binding_manifest_sha256=args.binding_manifest_sha256,
            feature_contract_path=args.feature_contract,
            expected_feature_contract_sha256=args.feature_contract_sha256,
            builder_git_commit=args.builder_git_commit,
            decision_freeze_created_ts=args.decision_freeze_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "prediction_and_decision_freeze_passed": result["report"][
                    "prediction_and_decision_freeze_passed"
                ],
                "candidate_guard_accepted_bet_count": result["report"][
                    "candidate_guard_accepted_bet_count"
                ],
                "matched_baseline_guard_accepted_bet_count": result["report"][
                    "matched_baseline_guard_accepted_bet_count"
                ],
                "decision_freeze_path": str(result["decision_freeze_path"]),
                "decision_freeze_sha256": result["decision_freeze_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "future_labels_outcomes_or_pnl_opened": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
