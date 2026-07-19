#!/usr/bin/env python3
"""Freeze #207 v6 future window, target-free predictions, and guard decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_future_prediction import (  # noqa: E402
    PolicySelectedConformalV6FuturePredictionConfig,
    freeze_policy_selected_conformal_v6_future_predictions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--future-preregistration-manifest", required=True)
    parser.add_argument("--future-preregistration-manifest-sha256", required=True)
    parser.add_argument("--collector-index", required=True)
    parser.add_argument("--collector-index-sha256", required=True)
    parser.add_argument("--feature-contract", required=True)
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--decision-freeze-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = freeze_policy_selected_conformal_v6_future_predictions(
        PolicySelectedConformalV6FuturePredictionConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            future_preregistration_manifest_path=args.future_preregistration_manifest,
            expected_future_preregistration_manifest_sha256=(
                args.future_preregistration_manifest_sha256
            ),
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.collector_index_sha256,
            feature_contract_path=args.feature_contract,
            expected_feature_contract_sha256=args.feature_contract_sha256,
            builder_git_commit=args.builder_git_commit,
            decision_freeze_created_ts=args.decision_freeze_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "selected_market_count": report["selected_market_count"],
                "prediction_attempted": report.get("prediction_attempted", True),
                "candidate_guard_accepted_unique_market_count": report.get(
                    "candidate_guard_accepted_unique_market_count", 0
                ),
                "candidate_guard_accepted_side_distribution": report.get(
                    "candidate_guard_accepted_side_distribution", {}
                ),
                "future_target_free_support_gate_passed": report[
                    "future_target_free_support_gate_passed"
                ],
                "future_target_access_allowed_after_decision_freeze": report[
                    "future_target_access_allowed_after_decision_freeze"
                ],
                "blocking_reason_codes": report["blocking_reason_codes"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "future_labels_outcomes_or_pnl_opened": False,
                "paper_candidate_allowed": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["future_target_free_support_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
