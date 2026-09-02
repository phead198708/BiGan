"""Run one pinned #212 v6.2 future batch action canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_batch_canary import (
    MarketClusteredMeanEVV62FutureBatchCanaryConfig,
    run_market_clustered_mean_ev_v6_2_future_batch_canary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--development-batch-canary-manifest", type=Path, required=True)
    parser.add_argument("--development-batch-canary-manifest-sha256", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_market_clustered_mean_ev_v6_2_future_batch_canary(
        MarketClusteredMeanEVV62FutureBatchCanaryConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            development_batch_canary_manifest_path=(
                args.development_batch_canary_manifest
            ),
            expected_development_batch_canary_manifest_sha256=(
                args.development_batch_canary_manifest_sha256
            ),
            candidate_manifest_path=args.candidate_manifest,
            expected_candidate_manifest_sha256=args.candidate_manifest_sha256,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "positive_mean_ev_lcb_unique_market_count": result["report"][
                    "positive_mean_ev_lcb_unique_market_count"
                ],
                "guard_accepted_unique_market_count": result["report"][
                    "guard_accepted_unique_market_count"
                ],
                "guard_accepted_by_side": result["report"]["guard_accepted_by_side"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
