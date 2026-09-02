"""Freeze the exact #212 v6.2 future window before target access."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation import (
    MarketClusteredMeanEVV62FutureFreezeConfig,
    freeze_market_clustered_mean_ev_v6_2_future_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--evaluation-profile", type=Path, required=True)
    parser.add_argument("--evaluation-profile-sha256", required=True)
    parser.add_argument("--collection-profile", type=Path, required=True)
    parser.add_argument("--collection-profile-sha256", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--cumulative-canary-manifest", type=Path, required=True)
    parser.add_argument("--cumulative-canary-manifest-sha256", required=True)
    parser.add_argument("--collector-index", type=Path, required=True)
    parser.add_argument("--collector-index-sha256", required=True)
    parser.add_argument("--decision-freeze-created-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = freeze_market_clustered_mean_ev_v6_2_future_predictions(
        MarketClusteredMeanEVV62FutureFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            evaluation_profile_path=args.evaluation_profile,
            expected_evaluation_profile_sha256=args.evaluation_profile_sha256,
            collection_profile_path=args.collection_profile,
            expected_collection_profile_sha256=args.collection_profile_sha256,
            candidate_manifest_path=args.candidate_manifest,
            expected_candidate_manifest_sha256=args.candidate_manifest_sha256,
            cumulative_canary_manifest_path=args.cumulative_canary_manifest,
            expected_cumulative_canary_manifest_sha256=(
                args.cumulative_canary_manifest_sha256
            ),
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.collector_index_sha256,
            builder_git_commit=_head(),
            decision_freeze_created_ts=(
                args.decision_freeze_created_ts or int(time.time() * 1000)
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(_summary(result))


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _summary(result: dict) -> str:
    report = result["report"]
    return json.dumps(
        {
            "run_dir": str(result["run_dir"]),
            "manifest_sha256": result["manifest_sha256"],
            "selected_market_count": report["selected_market_count"],
            "guard_accepted_unique_market_count": report[
                "candidate_guard_accepted_unique_market_count"
            ],
            "guard_accepted_unique_market_count_by_side": report[
                "candidate_guard_accepted_unique_market_count_by_side"
            ],
            "target_free_support_gate_passed": report[
                "target_free_support_gate_passed"
            ],
            "future_target_access_allowed": report["future_target_access_allowed"],
        },
        indent=2,
        sort_keys=True,
    )


if __name__ == "__main__":
    main()
