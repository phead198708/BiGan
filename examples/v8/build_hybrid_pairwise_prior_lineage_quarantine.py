#!/usr/bin/env python3
"""Build the final outcome-blind prior-lineage quarantine for #183."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.hybrid_pairwise_prior_lineage_quarantine import (  # noqa: E402
    HybridPairwisePriorLineageQuarantineConfig,
    build_hybrid_pairwise_prior_lineage_quarantine,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--created-at-ts", required=True, type=int)
    parser.add_argument("--historical-registry-descriptor", required=True)
    parser.add_argument(
        "--historical-registry-descriptor-sha256",
        required=True,
    )
    parser.add_argument("--historical-registry-rows", required=True)
    parser.add_argument("--historical-registry-rows-sha256", required=True)
    parser.add_argument("--terminal-lineage-state", required=True)
    parser.add_argument("--terminal-lineage-state-sha256", required=True)
    parser.add_argument("--final-support-gate-manifest", required=True)
    parser.add_argument(
        "--final-support-gate-manifest-sha256",
        required=True,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = build_hybrid_pairwise_prior_lineage_quarantine(
        HybridPairwisePriorLineageQuarantineConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            created_at_ts=args.created_at_ts,
            historical_registry_descriptor_path=Path(
                args.historical_registry_descriptor
            ),
            expected_historical_registry_descriptor_sha256=(
                args.historical_registry_descriptor_sha256
            ),
            historical_registry_rows_path=Path(
                args.historical_registry_rows
            ),
            expected_historical_registry_rows_sha256=(
                args.historical_registry_rows_sha256
            ),
            terminal_lineage_state_path=Path(
                args.terminal_lineage_state
            ),
            expected_terminal_lineage_state_sha256=(
                args.terminal_lineage_state_sha256
            ),
            final_support_gate_manifest_path=Path(
                args.final_support_gate_manifest
            ),
            expected_final_support_gate_manifest_sha256=(
                args.final_support_gate_manifest_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    quarantine = result["quarantine"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": quarantine["status"],
                "total_prior_unique_market_count": quarantine[
                    "total_prior_unique_market_count"
                ],
                "prior_market_ids_sha256": quarantine[
                    "prior_market_ids_sha256"
                ],
                "maximum_prior_decision_ts": quarantine[
                    "maximum_prior_decision_ts"
                ],
                "minimum_future_decision_ts": quarantine[
                    "minimum_future_decision_ts"
                ],
                "quarantine_path": str(result["quarantine_path"]),
                "quarantine_sha256": result["quarantine_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "outcome_label_or_pnl_artifacts_opened": False,
                "resolution_artifacts_opened": False,
                "collection_start_allowed": False,
                "paper_only": True,
                "capital_at_risk": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
