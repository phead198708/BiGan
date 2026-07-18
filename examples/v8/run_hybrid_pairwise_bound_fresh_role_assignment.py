#!/usr/bin/env python3
"""Assign #184 fresh roles from a finalized, frozen first-150 source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bigan.v8.polymarket.training.hybrid_pairwise_candidate_agnostic_source_binding import (
    HybridBoundFreshRoleAssignmentConfig,
    assign_bound_hybrid_fresh_roles,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--finalized-batch-progress", type=Path, required=True)
    parser.add_argument("--finalized-batch-progress-sha256", required=True)
    parser.add_argument(
        "--training-corpus-root",
        type=Path,
        default=Path("/Volumes/PHILIPS/v8"),
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = assign_bound_hybrid_fresh_roles(
        HybridBoundFreshRoleAssignmentConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            snapshot_manifest_path=args.snapshot_manifest,
            expected_snapshot_manifest_sha256=args.snapshot_manifest_sha256,
            finalized_batch_progress_path=args.finalized_batch_progress,
            expected_finalized_batch_progress_sha256=(
                args.finalized_batch_progress_sha256
            ),
            training_corpus_root=args.training_corpus_root,
            overwrite_existing=args.overwrite_existing,
        )
    )
    role_result = result["role_assignment_result"]
    report = role_result["report"]
    role_manifest_path = Path(role_result["manifest_path"])
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "role_assignment_ready": report["role_assignment_ready"],
                "selected_market_count": report["selected_market_count"],
                "role_market_counts": report["role_market_counts"],
                "blocking_reason_codes": report["blocking_reason_codes"],
                "bounded_batch_progress_path": str(
                    result["bounded_batch_progress_path"]
                ),
                "bounded_batch_progress_sha256": result[
                    "bounded_batch_progress_sha256"
                ],
                "role_assignment_manifest_path": str(role_manifest_path),
                "role_assignment_manifest_sha256": hashlib.sha256(
                    role_manifest_path.read_bytes()
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["role_assignment_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
