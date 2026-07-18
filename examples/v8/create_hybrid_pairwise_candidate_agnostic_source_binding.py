#!/usr/bin/env python3
"""Freeze a #190 candidate-agnostic raw source for the #184 hybrid protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.hybrid_pairwise_candidate_agnostic_source_binding import (
    HybridCandidateAgnosticSourceBindingConfig,
    create_hybrid_candidate_agnostic_source_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--readiness-manifest-sha256", required=True)
    parser.add_argument("--precollection-freeze-manifest", type=Path, required=True)
    parser.add_argument("--precollection-freeze-manifest-sha256", required=True)
    parser.add_argument("--source-pre-registration-manifest", type=Path, required=True)
    parser.add_argument("--source-pre-registration-manifest-sha256", required=True)
    parser.add_argument("--source-collection-freeze-manifest", type=Path, required=True)
    parser.add_argument("--source-collection-freeze-manifest-sha256", required=True)
    parser.add_argument("--source-batch-progress", type=Path, required=True)
    parser.add_argument("--source-batch-progress-sha256", required=True)
    parser.add_argument("--source-batch-id", required=True)
    parser.add_argument("--source-raw-root", type=Path, required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = create_hybrid_candidate_agnostic_source_binding(
        HybridCandidateAgnosticSourceBindingConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            readiness_manifest_path=args.readiness_manifest,
            expected_readiness_manifest_sha256=args.readiness_manifest_sha256,
            precollection_freeze_manifest_path=args.precollection_freeze_manifest,
            expected_precollection_freeze_manifest_sha256=(
                args.precollection_freeze_manifest_sha256
            ),
            source_pre_registration_manifest_path=args.source_pre_registration_manifest,
            expected_source_pre_registration_manifest_sha256=(
                args.source_pre_registration_manifest_sha256
            ),
            source_collection_freeze_manifest_path=(
                args.source_collection_freeze_manifest
            ),
            expected_source_collection_freeze_manifest_sha256=(
                args.source_collection_freeze_manifest_sha256
            ),
            source_batch_progress_path=args.source_batch_progress,
            expected_source_batch_progress_sha256=args.source_batch_progress_sha256,
            source_batch_id=args.source_batch_id,
            source_raw_root=args.source_raw_root,
            builder_git_commit=args.builder_git_commit,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "source_binding_ready": result["report"]["source_binding_ready"],
                "source_collection_terminal": result["report"][
                    "source_collection_terminal"
                ],
                "source_current_capture_count": result["report"][
                    "source_current_capture_count"
                ],
                "blocking_reason_codes": result["report"][
                    "blocking_reason_codes"
                ],
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["report"]["source_binding_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
