"""Run the #216 frozen-v5 historical PnL diagnostic for v6.2."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_historical_pnl import (
    MarketClusteredMeanEVV62HistoricalPnlConfig,
    run_market_clustered_mean_ev_v6_2_historical_pnl,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--v5-freeze-manifest", type=Path, required=True)
    parser.add_argument("--v5-freeze-manifest-sha256", required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    result = run_market_clustered_mean_ev_v6_2_historical_pnl(
        MarketClusteredMeanEVV62HistoricalPnlConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            candidate_manifest_path=args.candidate_manifest,
            expected_candidate_manifest_sha256=args.candidate_manifest_sha256,
            v5_freeze_manifest_path=args.v5_freeze_manifest,
            expected_v5_freeze_manifest_sha256=args.v5_freeze_manifest_sha256,
            feature_contract_path=args.feature_contract,
            expected_feature_contract_sha256=args.feature_contract_sha256,
            implementation_commit=_head(),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "historical_market_count": report["historical_market_count"],
                "candidate_post_cost_net_pnl": report[
                    "final_combined_candidate_post_cost_net_pnl"
                ],
                "matched_v5_post_cost_net_pnl": report[
                    "final_combined_matched_v5_post_cost_net_pnl"
                ],
                "candidate_minus_matched_v5_post_cost_net_pnl": report[
                    "final_combined_candidate_minus_matched_v5_post_cost_net_pnl"
                ],
                "promotion_evidence": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
