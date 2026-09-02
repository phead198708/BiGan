#!/usr/bin/env python3
"""Create the pre-registered #188 supplemental support freeze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_pairwise_supplemental_support import (  # noqa: E402
    PairwiseSupplementalSupportFreezeConfig,
    create_pairwise_supplemental_support_freeze,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
    )
    parser.add_argument("--freeze-created-ts", type=int, required=True)
    parser.add_argument("--parent-precollection-freeze", required=True)
    parser.add_argument(
        "--parent-precollection-freeze-sha256",
        required=True,
    )
    parser.add_argument(
        "--parent-terminal-reconciliation-report",
        required=True,
    )
    parser.add_argument(
        "--parent-terminal-reconciliation-report-sha256",
        required=True,
    )
    parser.add_argument(
        "--parent-terminal-reconciliation-manifest",
        required=True,
    )
    parser.add_argument(
        "--parent-terminal-reconciliation-manifest-sha256",
        required=True,
    )
    parser.add_argument("--parent-support-report", required=True)
    parser.add_argument(
        "--parent-support-report-sha256",
        required=True,
    )
    parser.add_argument("--parent-support-manifest", required=True)
    parser.add_argument(
        "--parent-support-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--successor-freeze-builder-git-commit",
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = create_pairwise_supplemental_support_freeze(
        PairwiseSupplementalSupportFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            freeze_created_ts=args.freeze_created_ts,
            parent_precollection_freeze_path=(
                args.parent_precollection_freeze
            ),
            parent_precollection_freeze_sha256=(
                args.parent_precollection_freeze_sha256
            ),
            parent_terminal_reconciliation_report_path=(
                args.parent_terminal_reconciliation_report
            ),
            parent_terminal_reconciliation_report_sha256=(
                args.parent_terminal_reconciliation_report_sha256
            ),
            parent_terminal_reconciliation_manifest_path=(
                args.parent_terminal_reconciliation_manifest
            ),
            parent_terminal_reconciliation_manifest_sha256=(
                args.parent_terminal_reconciliation_manifest_sha256
            ),
            parent_support_report_path=args.parent_support_report,
            parent_support_report_sha256=(
                args.parent_support_report_sha256
            ),
            parent_support_manifest_path=args.parent_support_manifest,
            parent_support_manifest_sha256=(
                args.parent_support_manifest_sha256
            ),
            successor_freeze_builder_git_commit=(
                args.successor_freeze_builder_git_commit
            ),
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "supplemental_support_freeze_ready": report[
                    "supplemental_support_freeze_ready"
                ],
                "supplemental_capture_attempt_count": report[
                    "supplemental_capture_attempt_count"
                ],
                "parent_selected_market_count": report[
                    "parent_selected_market_count"
                ],
                "target_valid_market_count": report[
                    "target_valid_market_count"
                ],
                "supplemental_minimum_collection_decision_ts": report[
                    "supplemental_minimum_collection_decision_ts"
                ],
                "freeze_path": str(result["freeze_path"]),
                "freeze_sha256": result["freeze_sha256"],
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "descriptor_path": str(result["descriptor_path"]),
                "descriptor_sha256": result["descriptor_sha256"],
                "labels_or_outcomes_opened_for_support_planning": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
