"""Run #213 split-aware v6.2 retrospective PnL on frozen v5 data."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_v5_pnl import (
    MarketClusteredMeanEVV62V5PnLConfig,
    run_market_clustered_mean_ev_v6_2_v5_pnl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_market_clustered_mean_ev_v6_2_v5_pnl_profile.json"
        ),
    )
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--v5-freeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-v5-freeze-manifest-sha256", required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--expected-feature-contract-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_market_clustered_mean_ev_v6_2_v5_pnl(
        MarketClusteredMeanEVV62V5PnLConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            candidate_manifest_path=args.candidate_manifest,
            expected_candidate_manifest_sha256=(
                args.expected_candidate_manifest_sha256
            ),
            v5_freeze_manifest_path=args.v5_freeze_manifest,
            expected_v5_freeze_manifest_sha256=(
                args.expected_v5_freeze_manifest_sha256
            ),
            feature_contract_path=args.feature_contract,
            expected_feature_contract_sha256=args.expected_feature_contract_sha256,
            implementation_commit=_head(),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "candidate_accepted_unique_market_count": report[
                    "candidate_accepted_unique_market_count"
                ],
                "candidate_after_cost_sized_net_pnl": report[
                    "candidate_after_cost_sized_net_pnl"
                ],
                "candidate_cost_basis": report["candidate_cost_basis"],
                "candidate_roi": report["candidate_roi"],
                "development_retrospective_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
