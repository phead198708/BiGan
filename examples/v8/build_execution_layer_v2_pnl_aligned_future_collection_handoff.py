#!/usr/bin/env python3
"""Freeze exact #169/#170 collector batches without opening outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    PnLAlignedFutureCollectionHandoffConfig,
    build_pnl_aligned_future_collection_handoff,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--batch-progress", required=True, type=Path)
    parser.add_argument("--expected-batch-progress-sha256", required=True)
    parser.add_argument(
        "--additional-batch-progress",
        action="append",
        default=[],
        type=Path,
        help="Additional replacement batch progress path; repeat for multiple batches.",
    )
    parser.add_argument(
        "--additional-expected-batch-progress-sha256",
        action="append",
        default=[],
        help="SHA-256 paired by position with --additional-batch-progress.",
    )
    parser.add_argument("--collection-freeze-manifest", required=True, type=Path)
    parser.add_argument("--expected-collection-freeze-manifest-sha256", required=True)
    parser.add_argument(
        "--training-corpus-root",
        type=Path,
        default=Path("/Volumes/PHILIPS/v8/polymarket"),
    )
    args = parser.parse_args()
    if len(args.additional_batch_progress) != len(
        args.additional_expected_batch_progress_sha256
    ):
        parser.error("additional batch progress paths and SHA-256 values must align")
    result = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            batch_progress_path=args.batch_progress,
            expected_batch_progress_sha256=args.expected_batch_progress_sha256,
            collection_freeze_manifest_path=args.collection_freeze_manifest,
            expected_collection_freeze_manifest_sha256=(
                args.expected_collection_freeze_manifest_sha256
            ),
            training_corpus_root=args.training_corpus_root,
            additional_batch_progress_paths=tuple(args.additional_batch_progress),
            additional_expected_batch_progress_sha256=tuple(
                args.additional_expected_batch_progress_sha256
            ),
        )
    )
    report = result["report"]
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"status={report['status']}")
    print(f"capture_count={report['capture_count']}")
    print(f"capture_quality_eligible_count={report['capture_quality_eligible_count']}")
    print(f"excluded_capture_count={report['excluded_capture_count']}")
    print(f"exported_round_count={report['exported_round_count']}")
    print(f"source_corpus_count={report['source_corpus_count']}")
    print(f"source_unique_market_count={report['source_unique_market_count']}")
    print("future_outcome_targets_loaded=false")
    if report["status"] != "OUTCOME_BLIND_COLLECTION_HANDOFF_READY":
        raise SystemExit(
            "future collection handoff failed closed: " + ",".join(report["blocking_reason_codes"])
        )


if __name__ == "__main__":
    main()
