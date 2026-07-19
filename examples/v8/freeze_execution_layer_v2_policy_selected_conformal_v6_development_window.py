"""Freeze the earliest 260 post-#204 development markets and chronological roles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (  # noqa: E402
    PolicySelectedConformalV6DevelopmentWindowConfig,
    freeze_policy_selected_conformal_v6_development_window,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--preregistration-manifest", required=True)
    parser.add_argument("--preregistration-manifest-sha256", required=True)
    parser.add_argument("--collector-index", required=True)
    parser.add_argument("--collector-index-sha256", required=True)
    parser.add_argument("--feature-contract", required=True)
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--freeze-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = freeze_policy_selected_conformal_v6_development_window(
        PolicySelectedConformalV6DevelopmentWindowConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            preregistration_manifest_path=args.preregistration_manifest,
            expected_preregistration_manifest_sha256=args.preregistration_manifest_sha256,
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.collector_index_sha256,
            feature_contract_path=args.feature_contract,
            expected_feature_contract_sha256=args.feature_contract_sha256,
            builder_git_commit=args.builder_git_commit,
            freeze_created_ts=args.freeze_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "development_window_freeze_ready": result["report"][
                    "development_window_freeze_ready"
                ],
                "selected_market_count": result["report"]["selected_market_count"],
                "role_market_counts": result["report"]["role_market_counts"],
                "blocking_reason_codes": result["report"]["blocking_reason_codes"],
                "labels_outcomes_or_pnl_opened": False,
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "paper_candidate_allowed": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["report"]["development_window_freeze_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
