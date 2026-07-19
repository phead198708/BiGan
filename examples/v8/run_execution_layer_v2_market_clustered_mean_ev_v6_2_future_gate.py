"""Run the single-use #212 v6.2 side-only future PnL gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation import (
    SINGLE_USE_CLAIM_FILENAME,
    MarketClusteredMeanEVV62FutureGateConfig,
    run_market_clustered_mean_ev_v6_2_future_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--prediction-freeze-manifest", type=Path, required=True)
    parser.add_argument("--prediction-freeze-manifest-sha256", required=True)
    parser.add_argument("--settled-corpus-index", type=Path, required=True)
    parser.add_argument("--settled-corpus-index-sha256", required=True)
    parser.add_argument("--evaluation-profile", type=Path, required=True)
    parser.add_argument("--evaluation-profile-sha256", required=True)
    parser.add_argument("--single-use-claim", type=Path)
    parser.add_argument("--evaluation-started-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_market_clustered_mean_ev_v6_2_future_gate(
        MarketClusteredMeanEVV62FutureGateConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            prediction_freeze_manifest_path=args.prediction_freeze_manifest,
            expected_prediction_freeze_manifest_sha256=(
                args.prediction_freeze_manifest_sha256
            ),
            settled_corpus_index_path=args.settled_corpus_index,
            expected_settled_corpus_index_sha256=args.settled_corpus_index_sha256,
            evaluation_profile_path=args.evaluation_profile,
            expected_evaluation_profile_sha256=args.evaluation_profile_sha256,
            single_use_claim_path=(
                args.single_use_claim
                or args.prediction_freeze_manifest.resolve().parent
                / SINGLE_USE_CLAIM_FILENAME
            ),
            builder_git_commit=_head(),
            evaluation_started_ts=args.evaluation_started_ts or int(time.time() * 1000),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "manifest_sha256": result["manifest_sha256"],
                "future_gate_passed": result["report"]["future_gate_passed"],
                "future_gate_blocking_reason_codes": result["report"][
                    "future_gate_blocking_reason_codes"
                ],
                "candidate_post_cost_net_pnl": result["report"][
                    "candidate_post_cost_net_pnl"
                ],
                "matched_v5_post_cost_net_pnl": result["report"][
                    "matched_v5_post_cost_net_pnl"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


if __name__ == "__main__":
    main()
