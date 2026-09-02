#!/usr/bin/env python3
"""Run one exact-195 retained-v6.7 powered paper-gate stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_retained_v6_7_powered_paper_gate import (
    RetainedV67PoweredPaperGateConfig,
    run_retained_v6_7_powered_paper_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("freeze_target_free", "settle", "evaluate_powered_pnl"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--readiness-manifest-sha256", required=True)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--stage-started-ts", type=int)
    parser.add_argument("--collector-protocol", type=Path)
    parser.add_argument("--collector-protocol-sha256")
    parser.add_argument("--collector-index", type=Path)
    parser.add_argument("--collector-index-sha256")
    parser.add_argument("--v6-7-profile", type=Path)
    parser.add_argument("--v6-7-profile-sha256")
    parser.add_argument(
        "--development-batch-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--development-batch-manifest-sha256",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--v6-2-batch-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--v6-2-batch-manifest-sha256",
        action="append",
        default=[],
    )
    parser.add_argument("--target-free-freeze-manifest", type=Path)
    parser.add_argument("--target-free-freeze-manifest-sha256")
    parser.add_argument("--settled-corpus-index", type=Path)
    parser.add_argument("--settled-corpus-index-sha256")
    parser.add_argument("--runtime-policy-profile", type=Path)
    parser.add_argument("--runtime-policy-profile-sha256")
    parser.add_argument("--provider-timeout-seconds", type=float, default=15.0)
    parser.add_argument(
        "--provider-http-timeout-seconds", type=float, default=5.0
    )
    parser.add_argument(
        "--settlement-max-wait-seconds", type=float, default=600.0
    )
    parser.add_argument(
        "--settlement-poll-interval-seconds", type=float, default=15.0
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> int:
    args = _parser().parse_args()
    config = RetainedV67PoweredPaperGateConfig(
        stage=args.stage,
        run_id=args.run_id,
        output_dir=args.output_dir,
        readiness_manifest_path=args.readiness_manifest,
        expected_readiness_manifest_sha256=args.readiness_manifest_sha256,
        implementation_commit=args.implementation_commit or _git_commit(),
        stage_started_ts=args.stage_started_ts or int(time.time() * 1000),
        collector_protocol_path=args.collector_protocol,
        expected_collector_protocol_sha256=args.collector_protocol_sha256,
        collector_index_path=args.collector_index,
        expected_collector_index_sha256=args.collector_index_sha256,
        v6_7_profile_path=args.v6_7_profile,
        expected_v6_7_profile_sha256=args.v6_7_profile_sha256,
        development_batch_manifest_paths=tuple(
            args.development_batch_manifest
        ),
        expected_development_batch_manifest_sha256s=tuple(
            args.development_batch_manifest_sha256
        ),
        v6_2_batch_manifest_paths=tuple(args.v6_2_batch_manifest),
        expected_v6_2_batch_manifest_sha256s=tuple(
            args.v6_2_batch_manifest_sha256
        ),
        target_free_freeze_manifest_path=args.target_free_freeze_manifest,
        expected_target_free_freeze_manifest_sha256=(
            args.target_free_freeze_manifest_sha256
        ),
        settled_corpus_index_path=args.settled_corpus_index,
        expected_settled_corpus_index_sha256=(
            args.settled_corpus_index_sha256
        ),
        runtime_policy_profile_path=args.runtime_policy_profile,
        expected_runtime_policy_profile_sha256=(
            args.runtime_policy_profile_sha256
        ),
        provider_timeout_seconds=args.provider_timeout_seconds,
        provider_http_timeout_seconds=args.provider_http_timeout_seconds,
        settlement_max_wait_seconds=args.settlement_max_wait_seconds,
        settlement_poll_interval_seconds=(
            args.settlement_poll_interval_seconds
        ),
        max_workers=args.max_workers,
        overwrite_existing=args.overwrite_existing,
    )
    result = run_retained_v6_7_powered_paper_gate(config)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
