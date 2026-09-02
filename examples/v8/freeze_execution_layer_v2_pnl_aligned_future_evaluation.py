#!/usr/bin/env python3
"""Freeze #169 accepted-bet evaluator inputs before outcome reconciliation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    PnLAlignedFutureEvaluationFreezeConfig,
    freeze_pnl_aligned_future_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--evaluation-protocol", required=True, type=Path)
    parser.add_argument("--expected-evaluation-protocol-sha256", required=True)
    parser.add_argument("--collection-freeze-manifest", required=True, type=Path)
    parser.add_argument("--expected-collection-freeze-manifest-sha256", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit")
    args = parser.parse_args()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if args.expected_git_commit and git_commit != args.expected_git_commit.lower():
        raise SystemExit("repository HEAD does not match --expected-git-commit")
    result = freeze_pnl_aligned_future_evaluation(
        PnLAlignedFutureEvaluationFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            evaluation_protocol_path=args.evaluation_protocol,
            expected_evaluation_protocol_sha256=(
                args.expected_evaluation_protocol_sha256
            ),
            collection_freeze_manifest_path=args.collection_freeze_manifest,
            expected_collection_freeze_manifest_sha256=(
                args.expected_collection_freeze_manifest_sha256
            ),
            model_dir=args.model_dir,
            git_commit=git_commit,
        )
    )
    manifest = result["manifest"]
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"evaluation_freeze_id={manifest['evaluation_freeze_id']}")
    print(f"git_commit={manifest['git_commit']}")
    print("future_outcome_targets_loaded=false")
    print("promotion_evidence_eligible=false")


if __name__ == "__main__":
    main()
