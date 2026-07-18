#!/usr/bin/env python3
"""Prepare an exact-first-150 filesystem view for the frozen #190 finalizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bigan.v8.polymarket.training.hybrid_pairwise_candidate_agnostic_source_binding import (
    HybridBoundedFinalizationViewConfig,
    prepare_hybrid_bounded_finalization_view,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/v8/polymarket_runs"),
    )
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--finalizer-script", type=Path, required=True)
    parser.add_argument("--finalizer-script-sha256", required=True)
    parser.add_argument("--finalizer-git-commit", required=True)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--training-corpus-root",
        type=Path,
        default=Path("/Volumes/PHILIPS/v8"),
    )
    parser.add_argument(
        "--settlement-poll-interval-seconds",
        type=float,
        default=15.0,
    )
    parser.add_argument("--settlement-grace-seconds", type=float, default=1_200.0)
    parser.add_argument(
        "--public-provider-http-timeout-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = prepare_hybrid_bounded_finalization_view(
        HybridBoundedFinalizationViewConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            snapshot_manifest_path=args.snapshot_manifest,
            expected_snapshot_manifest_sha256=args.snapshot_manifest_sha256,
            finalizer_script_path=args.finalizer_script,
            expected_finalizer_script_sha256=args.finalizer_script_sha256,
            finalizer_git_commit=args.finalizer_git_commit,
            python_executable=args.python_executable,
            training_corpus_root=args.training_corpus_root,
            settlement_poll_interval_seconds=(
                args.settlement_poll_interval_seconds
            ),
            settlement_grace_seconds=args.settlement_grace_seconds,
            public_provider_http_timeout_seconds=(
                args.public_provider_http_timeout_seconds
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "bounded_finalization_view_ready": result["report"][
                    "bounded_finalization_view_ready"
                ],
                "bounded_capture_attempt_count": result["report"][
                    "bounded_capture_attempt_count"
                ],
                "source_attempts_after_150_included": result["report"][
                    "source_attempts_after_150_included"
                ],
                "view_root": str(result["view_root"]),
                "finalizer_command_argv": result["manifest"][
                    "finalizer_command_argv"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
