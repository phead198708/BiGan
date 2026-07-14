#!/usr/bin/env python3
"""Build frozen #169 outcome-blind future decision inputs from Phase 2 corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    PnLAlignedFutureDecisionInputConfig,
    build_pnl_aligned_future_outcome_blind_decision_inputs,
    load_pnl_aligned_future_collection_handoff_source_dirs,
)

DEFAULT_UNLOCK_DIR = Path(
    "examples/v8/polymarket_runs/o-v8-paper-candidate-unlock-20260703T073000Z"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--collection-freeze-manifest", required=True, type=Path)
    parser.add_argument("--expected-collection-freeze-manifest-sha256", required=True)
    parser.add_argument("--source-corpus-dir", action="append", type=Path)
    parser.add_argument("--collection-handoff-manifest", type=Path)
    parser.add_argument("--expected-collection-handoff-manifest-sha256")
    parser.add_argument("--paper-candidate-unlock-dir", type=Path, default=DEFAULT_UNLOCK_DIR)
    parser.add_argument("--expected-unlock-manifest-sha256")
    parser.add_argument("--canonical-o-source-manifest-path", type=Path)
    args = parser.parse_args()
    handoff_requested = args.collection_handoff_manifest is not None
    if handoff_requested != bool(args.expected_collection_handoff_manifest_sha256):
        parser.error("collection handoff manifest and SHA-256 must be provided together")
    if handoff_requested and args.source_corpus_dir:
        parser.error("do not combine collection handoff with explicit source corpus directories")
    if not handoff_requested and not args.source_corpus_dir:
        parser.error("provide a collection handoff or at least one source corpus directory")
    source_corpus_dirs = (
        load_pnl_aligned_future_collection_handoff_source_dirs(
            args.collection_handoff_manifest,
            expected_sha256=args.expected_collection_handoff_manifest_sha256,
        )
        if handoff_requested
        else tuple(args.source_corpus_dir)
    )
    kwargs = {}
    if args.expected_unlock_manifest_sha256:
        kwargs["expected_unlock_manifest_sha256"] = args.expected_unlock_manifest_sha256
    result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            collection_freeze_manifest_path=args.collection_freeze_manifest,
            expected_collection_freeze_manifest_sha256=(
                args.expected_collection_freeze_manifest_sha256
            ),
            source_corpus_dirs=source_corpus_dirs,
            paper_candidate_unlock_dir=args.paper_candidate_unlock_dir,
            canonical_o_source_manifest_path=args.canonical_o_source_manifest_path,
            collection_handoff_manifest_path=args.collection_handoff_manifest,
            expected_collection_handoff_manifest_sha256=(
                args.expected_collection_handoff_manifest_sha256
            ),
            **kwargs,
        )
    )
    report = result["report"]
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"status={report['status']}")
    print(f"source_unique_market_count={report['source_unique_market_count']}")
    print(f"outcome_blind_decision_row_count={report['outcome_blind_decision_row_count']}")
    print("future_outcome_targets_loaded=false")
    if report["status"] != "OUTCOME_BLIND_FUTURE_DECISION_INPUT_READY":
        raise SystemExit(
            "future decision input build failed closed: "
            + ",".join(report["blocking_reason_codes"])
        )


if __name__ == "__main__":
    main()
