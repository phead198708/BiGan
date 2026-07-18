#!/usr/bin/env python3
"""Freeze first-150 source attempts after the bound collector is terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.hybrid_pairwise_candidate_agnostic_source_binding import (
    HybridCandidateAgnosticSourceSnapshotConfig,
    freeze_hybrid_candidate_agnostic_source_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--binding-manifest", type=Path, required=True)
    parser.add_argument("--binding-manifest-sha256", required=True)
    parser.add_argument("--terminal-batch-progress", type=Path, required=True)
    parser.add_argument("--terminal-batch-progress-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = freeze_hybrid_candidate_agnostic_source_snapshot(
        HybridCandidateAgnosticSourceSnapshotConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            binding_manifest_path=args.binding_manifest,
            expected_binding_manifest_sha256=args.binding_manifest_sha256,
            terminal_batch_progress_path=args.terminal_batch_progress,
            expected_terminal_batch_progress_sha256=(
                args.terminal_batch_progress_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "source_snapshot_ready": result["report"]["source_snapshot_ready"],
                "bounded_capture_attempt_count": result["report"][
                    "bounded_capture_attempt_count"
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
    return 0 if result["report"]["source_snapshot_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
