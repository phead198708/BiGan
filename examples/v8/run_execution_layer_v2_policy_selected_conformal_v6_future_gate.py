#!/usr/bin/env python3
"""Join frozen #207 v6 future decisions to targets and run the side-only gate once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_future_settlement import (  # noqa: E402
    PolicySelectedConformalV6FutureGateConfig,
    reconcile_policy_selected_conformal_v6_future_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--prediction-freeze-manifest", required=True)
    parser.add_argument("--prediction-freeze-manifest-sha256", required=True)
    parser.add_argument("--settled-corpus-index", required=True)
    parser.add_argument("--settled-corpus-index-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--reconciliation-started-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = reconcile_policy_selected_conformal_v6_future_gate(
        PolicySelectedConformalV6FutureGateConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            prediction_freeze_manifest_path=args.prediction_freeze_manifest,
            expected_prediction_freeze_manifest_sha256=(
                args.prediction_freeze_manifest_sha256
            ),
            settled_corpus_index_path=args.settled_corpus_index,
            expected_settled_corpus_index_sha256=args.settled_corpus_index_sha256,
            builder_git_commit=args.builder_git_commit,
            reconciliation_started_ts=args.reconciliation_started_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    gate = result["gate"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "pnl_hard_gate_aggregation": gate["pnl_hard_gate_aggregation"],
                "action_and_action_family_pnl_diagnostic_only": gate[
                    "action_and_action_family_pnl_diagnostic_only"
                ],
                "guard_accepted_bet_count": gate["guard_accepted_bet_count"],
                "guard_accepted_unique_market_count": gate[
                    "guard_accepted_unique_market_count"
                ],
                "accepted_side_metrics": gate["accepted_side_metrics"],
                "candidate_post_cost_net_pnl": gate["candidate_post_cost_net_pnl"],
                "matched_baseline_post_cost_net_pnl": gate[
                    "matched_baseline_post_cost_net_pnl"
                ],
                "future_gate_passed": gate["future_gate_passed"],
                "future_gate_blocking_reason_codes": gate[
                    "future_gate_blocking_reason_codes"
                ],
                "gate_path": str(result["gate_path"]),
                "gate_sha256": result["gate_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "future_results_used_for_tuning": False,
                "source_model_candidate_eligible": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if gate["future_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
