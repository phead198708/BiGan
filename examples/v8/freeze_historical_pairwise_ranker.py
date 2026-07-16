#!/usr/bin/env python3
"""Fit and freeze the historical-train-only pairwise ranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.historical_pairwise_ranker_freeze import (  # noqa: E402
    HistoricalPairwiseRankerFreezeConfig,
    freeze_historical_pairwise_ranker,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--registry-descriptor", required=True)
    parser.add_argument("--registry-descriptor-sha256", required=True)
    parser.add_argument("--registry-manifest", required=True)
    parser.add_argument("--registry-manifest-sha256", required=True)
    parser.add_argument("--registry-report", required=True)
    parser.add_argument("--registry-report-sha256", required=True)
    parser.add_argument("--registry-rows", required=True)
    parser.add_argument("--registry-rows-sha256", required=True)
    parser.add_argument(
        "--protocol-path",
        default=(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
        ),
    )
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument(
        "--feature-contract-path",
        default=(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
        ),
    )
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = freeze_historical_pairwise_ranker(
        HistoricalPairwiseRankerFreezeConfig(
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
            registry_descriptor_path=Path(args.registry_descriptor),
            expected_registry_descriptor_sha256=args.registry_descriptor_sha256,
            registry_manifest_path=Path(args.registry_manifest),
            expected_registry_manifest_sha256=args.registry_manifest_sha256,
            registry_report_path=Path(args.registry_report),
            expected_registry_report_sha256=args.registry_report_sha256,
            registry_rows_path=Path(args.registry_rows),
            expected_registry_rows_sha256=args.registry_rows_sha256,
            protocol_path=Path(args.protocol_path),
            expected_protocol_sha256=args.protocol_sha256,
            feature_contract_path=Path(args.feature_contract_path),
            expected_feature_contract_sha256=args.feature_contract_sha256,
            overwrite_existing=args.overwrite_existing,
        )
    )
    manifest = result["freeze_manifest"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "freeze_status": manifest["freeze_status"],
                "training_market_count": manifest["training_market_count"],
                "oof_market_count": manifest["oof_market_count"],
                "model_sha256": manifest["model_sha256"],
                "dataset_hash": manifest["dataset_hash"],
                "split_hash": manifest["split_hash"],
                "model_config_hash": manifest["model_config_hash"],
                "fresh_calibration_required": manifest[
                    "fresh_calibration_required"
                ],
                "rank_scores_execution_eligible": manifest[
                    "rank_scores_execution_eligible"
                ],
                "manifest_path": str(result["freeze_manifest_path"]),
                "descriptor_path": str(result["descriptor_path"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
