#!/usr/bin/env python3
"""Freeze #207 v6 candidate and collector prefix before future collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_future import (  # noqa: E402
    PolicySelectedConformalV6FuturePreRegistrationConfig,
    pre_register_policy_selected_conformal_v6_future_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-manifest-sha256", required=True)
    parser.add_argument("--collector-protocol", required=True)
    parser.add_argument("--collector-protocol-sha256", required=True)
    parser.add_argument("--collector-index", required=True)
    parser.add_argument("--collector-index-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--preregistration-created-ts", type=int, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = pre_register_policy_selected_conformal_v6_future_evaluation(
        PolicySelectedConformalV6FuturePreRegistrationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            candidate_manifest_path=args.candidate_manifest,
            expected_candidate_manifest_sha256=args.candidate_manifest_sha256,
            baseline_manifest_path=args.baseline_manifest,
            expected_baseline_manifest_sha256=args.baseline_manifest_sha256,
            collector_protocol_path=args.collector_protocol,
            expected_collector_protocol_sha256=args.collector_protocol_sha256,
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.collector_index_sha256,
            builder_git_commit=args.builder_git_commit,
            preregistration_created_ts=args.preregistration_created_ts,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "future_preregistration_ready": report["future_preregistration_ready"],
                "collector_index_prefix_entry_count": report[
                    "collector_index_prefix_entry_count"
                ],
                "minimum_collection_index_sequence": report[
                    "minimum_collection_index_sequence"
                ],
                "minimum_collection_decision_ts": report[
                    "minimum_collection_decision_ts"
                ],
                "target_quality_valid_market_count": report[
                    "target_quality_valid_market_count"
                ],
                "maximum_index_scan_count": report["maximum_index_scan_count"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "future_labels_outcomes_or_pnl_opened": False,
                "paper_candidate_allowed": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
