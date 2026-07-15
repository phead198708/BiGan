"""Assign the frozen #172 40/20/30 corpus roles without reading outcome values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_cross_fitted_family_lcb import (  # noqa: E402
    CrossFittedFamilyLCBRoleAssignmentConfig,
    assign_cross_fitted_family_lcb_roles,
)


def run_role_assignment(
    *,
    run_id: str,
    output_dir: Path | str,
    precollection_freeze_manifest: Path | str,
    precollection_freeze_manifest_sha256: str,
    batch_progress_paths: tuple[Path | str, ...],
    batch_progress_sha256: tuple[str, ...],
    training_corpus_root: Path | str = "/Volumes/PHILIPS/v8",
) -> dict:
    if len(batch_progress_paths) != len(batch_progress_sha256):
        raise ValueError("batch progress paths and SHA-256 values must align")
    result = assign_cross_fitted_family_lcb_roles(
        CrossFittedFamilyLCBRoleAssignmentConfig(
            run_id=run_id,
            output_dir=output_dir,
            precollection_freeze_manifest_path=precollection_freeze_manifest,
            expected_precollection_freeze_manifest_sha256=(
                precollection_freeze_manifest_sha256
            ),
            batch_progress_pins=tuple(
                zip(batch_progress_paths, batch_progress_sha256, strict=True)
            ),
            training_corpus_root=training_corpus_root,
        )
    )
    report = result["report"]
    return {
        "run_id": run_id,
        "status": report["status"],
        "role_assignment_ready": report["role_assignment_ready"],
        "selected_market_count": report["selected_market_count"],
        "role_market_counts": report["role_market_counts"],
        "blocking_reason_codes": report["blocking_reason_codes"],
        "excluded_reason_distribution": report["excluded_reason_distribution"],
        "manifest_path": str(result["manifest_path"]),
        "manifest_sha256": result["manifest_sha256"],
        "labels_or_outcomes_opened_for_role_assignment": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
    }


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
    summary = run_role_assignment(
        run_id=args.run_id,
        output_dir=args.output_dir,
        precollection_freeze_manifest=args.precollection_freeze_manifest,
        precollection_freeze_manifest_sha256=(
            args.precollection_freeze_manifest_sha256
        ),
        batch_progress_paths=tuple(args.batch_progress),
        batch_progress_sha256=tuple(args.batch_progress_sha256),
        training_corpus_root=args.training_corpus_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["role_assignment_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
