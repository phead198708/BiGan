"""Assign #174 fresh confirmatory markets without opening labels or outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_side_family_lcb import (  # noqa: E402
    HierarchicalSideFamilyLCBConfirmatoryAssignmentConfig,
    assign_hierarchical_side_family_lcb_confirmatory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--precollection-freeze-manifest", required=True)
    parser.add_argument("--precollection-freeze-manifest-sha256", required=True)
    parser.add_argument("--batch-progress", action="append", required=True)
    parser.add_argument("--batch-progress-sha256", action="append", required=True)
    parser.add_argument("--training-corpus-root", default="/Volumes/PHILIPS/v8")
    args = parser.parse_args(argv)
    if len(args.batch_progress) != len(args.batch_progress_sha256):
        parser.error("batch progress paths and SHA-256 values must align")
    result = assign_hierarchical_side_family_lcb_confirmatory(
        HierarchicalSideFamilyLCBConfirmatoryAssignmentConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            precollection_freeze_manifest_path=args.precollection_freeze_manifest,
            expected_precollection_freeze_manifest_sha256=(
                args.precollection_freeze_manifest_sha256
            ),
            batch_progress_pins=tuple(
                zip(
                    args.batch_progress,
                    args.batch_progress_sha256,
                    strict=True,
                )
            ),
            training_corpus_root=args.training_corpus_root,
        )
    )
    report = result["report"]
    summary = {
        "run_id": args.run_id,
        "status": report["status"],
        "assignment_ready": report["assignment_ready"],
        "selected_market_count": report["selected_market_count"],
        "excluded_capture_count": report["excluded_capture_count"],
        "excluded_reason_distribution": report["excluded_reason_distribution"],
        "blocking_reason_codes": report["blocking_reason_codes"],
        "manifest_path": str(result["manifest_path"]),
        "manifest_sha256": result["manifest_sha256"],
        "labels_or_outcomes_opened_for_assignment": False,
        "paper_only": True,
        "capital_at_risk": False,
        "v8_execution_handoff_allowed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["assignment_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
