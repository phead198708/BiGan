#!/usr/bin/env python3
"""Freeze issue #169 model/config lineage before future collection."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    PnLAlignedFutureCollectionFreezeConfig,
    freeze_pnl_aligned_future_collection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-git-commit",
        help="Fail closed unless the repository HEAD equals this SHA-1.",
    )
    parser.add_argument("--expected-round-count", type=int, default=30)
    return parser


def main() -> None:
    args = _parser().parse_args()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if args.expected_git_commit and git_commit != args.expected_git_commit.lower():
        raise SystemExit("repository HEAD does not match --expected-git-commit")
    result = freeze_pnl_aligned_future_collection(
        PnLAlignedFutureCollectionFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            git_commit=git_commit,
            expected_round_count=args.expected_round_count,
        )
    )
    manifest = result["manifest"]
    print(f"output_dir={result['output_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"collection_freeze_id={manifest['collection_freeze_id']}")
    print(f"git_commit={manifest['git_commit']}")
    print(
        "execution_guard_config_sha256="
        f"{manifest['execution_guard_config_sha256']}"
    )
    print(f"max_prior_decision_ts={manifest['max_prior_decision_ts']}")
    print(
        "minimum_future_window_start_ts="
        f"{manifest['minimum_future_window_start_ts']}"
    )
    print(f"expected_round_count={manifest['expected_round_count']}")
    print("collection_started=false")
    print("promotion_evidence_eligible=false")


if __name__ == "__main__":
    main()
