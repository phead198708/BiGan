"""Build the #212 v6.2 settled corpus on quarantine copies."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation import (
    MarketClusteredMeanEVV62FutureSettlementConfig,
    build_market_clustered_mean_ev_v6_2_future_settled_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--prediction-freeze-manifest", type=Path, required=True)
    parser.add_argument("--prediction-freeze-manifest-sha256", required=True)
    parser.add_argument("--target-access-started-ts", type=int)
    parser.add_argument("--settlement-max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--provider-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--provider-http-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = build_market_clustered_mean_ev_v6_2_future_settled_corpus(
        MarketClusteredMeanEVV62FutureSettlementConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            prediction_freeze_manifest_path=args.prediction_freeze_manifest,
            expected_prediction_freeze_manifest_sha256=(
                args.prediction_freeze_manifest_sha256
            ),
            builder_git_commit=_head(),
            target_access_started_ts=(
                args.target_access_started_ts or int(time.time() * 1000)
            ),
            provider_timeout_seconds=args.provider_timeout_seconds,
            provider_http_timeout_seconds=args.provider_http_timeout_seconds,
            settlement_max_wait_seconds=args.settlement_max_wait_seconds,
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            max_workers=args.max_workers,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "manifest_sha256": result["manifest_sha256"],
                "settled_corpus_index_ready": result["report"][
                    "settled_corpus_index_ready"
                ],
                "settled_market_count": result["report"][
                    "settled_corpus_ready_market_count"
                ],
                "unresolved_market_count": result["report"][
                    "unresolved_or_failed_market_count"
                ],
                "settled_corpus_index_sha256": result["index_sha256"],
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
