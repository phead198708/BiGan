#!/usr/bin/env python3
"""Evaluate readiness for hybrid fresh calibration precollection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.hybrid_pairwise_precollection_readiness import (  # noqa: E402
    HybridPairwisePrecollectionReadinessConfig,
    evaluate_hybrid_pairwise_precollection_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--hybrid-protocol", required=True)
    parser.add_argument("--hybrid-protocol-sha256", required=True)
    parser.add_argument("--source-pairwise-protocol", required=True)
    parser.add_argument("--source-pairwise-protocol-sha256", required=True)
    parser.add_argument("--source-feature-contract", required=True)
    parser.add_argument("--source-feature-contract-sha256", required=True)
    parser.add_argument("--historical-registry-descriptor", required=True)
    parser.add_argument("--historical-registry-descriptor-sha256", required=True)
    parser.add_argument("--historical-ranker-descriptor", required=True)
    parser.add_argument("--historical-ranker-descriptor-sha256", required=True)
    parser.add_argument("--historical-ranker-manifest", required=True)
    parser.add_argument("--historical-ranker-manifest-sha256", required=True)
    parser.add_argument("--freeze-created-at-ts", required=True, type=int)
    parser.add_argument("--active-lineage-state", action="append", default=[])
    parser.add_argument("--final-prior-quarantine")
    parser.add_argument("--final-prior-quarantine-sha256")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = evaluate_hybrid_pairwise_precollection_readiness(
        HybridPairwisePrecollectionReadinessConfig(
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
            hybrid_protocol_path=Path(args.hybrid_protocol),
            expected_hybrid_protocol_sha256=args.hybrid_protocol_sha256,
            source_pairwise_protocol_path=Path(args.source_pairwise_protocol),
            expected_source_pairwise_protocol_sha256=(
                args.source_pairwise_protocol_sha256
            ),
            source_feature_contract_path=Path(args.source_feature_contract),
            expected_source_feature_contract_sha256=(
                args.source_feature_contract_sha256
            ),
            historical_registry_descriptor_path=Path(
                args.historical_registry_descriptor
            ),
            expected_historical_registry_descriptor_sha256=(
                args.historical_registry_descriptor_sha256
            ),
            historical_ranker_descriptor_path=Path(
                args.historical_ranker_descriptor
            ),
            expected_historical_ranker_descriptor_sha256=(
                args.historical_ranker_descriptor_sha256
            ),
            historical_ranker_manifest_path=Path(args.historical_ranker_manifest),
            expected_historical_ranker_manifest_sha256=(
                args.historical_ranker_manifest_sha256
            ),
            freeze_created_at_ts=args.freeze_created_at_ts,
            active_lineage_state_paths=tuple(
                Path(value) for value in args.active_lineage_state
            ),
            final_prior_quarantine_path=(
                Path(args.final_prior_quarantine)
                if args.final_prior_quarantine
                else None
            ),
            expected_final_prior_quarantine_sha256=(
                args.final_prior_quarantine_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "readiness_status": report["readiness_status"],
                "precollection_readiness_passed": report[
                    "precollection_readiness_passed"
                ],
                "precollection_freeze_created": report[
                    "precollection_freeze_created"
                ],
                "collection_start_allowed": report["collection_start_allowed"],
                "collection_start_command_generated": report[
                    "collection_start_command_generated"
                ],
                "blocking_reason_codes": report["blocking_reason_codes"],
                "report_path": str(result["report_path"]),
                "manifest_path": str(result["manifest_path"]),
                "freeze_manifest_path": (
                    str(result["freeze_manifest_path"])
                    if result["freeze_manifest_path"]
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
