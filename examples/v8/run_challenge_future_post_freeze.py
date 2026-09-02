#!/usr/bin/env python3
"""Run official challenge settlement and the single-use parallel gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.challenge_future_post_freeze import (
    ChallengeFutureEvaluationConfig,
    ChallengeFutureSettlementConfig,
    build_challenge_future_settled_index,
    evaluate_challenge_parallel_future_gate,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _sha256_file,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent.parent
CONFIG_DIR = EXAMPLES_DIR / "polymarket_configs"
DEFAULT_OUTPUT_DIR = EXAMPLES_DIR / "polymarket_runs"
DEFAULT_POST_FREEZE_PROTOCOL = (
    CONFIG_DIR / "challenge_future_post_freeze_protocol.json"
)
DEFAULT_RUNTIME_PROFILE = (
    CONFIG_DIR
    / "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_profile.json"
)


def _git_head_and_clean() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(
            "post-freeze execution requires a clean committed implementation"
        )
    return head


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError(f"SHA-256 sidecar missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if _sha256_file(path) != expected.lower():
        raise ValueError(f"SHA-256 sidecar mismatch: {path}")
    return expected.lower()


def _settle(args: argparse.Namespace) -> dict:
    head = _git_head_and_clean()
    return build_challenge_future_settled_index(
        ChallengeFutureSettlementConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            target_free_freeze_manifest_path=args.freeze_manifest,
            expected_target_free_freeze_manifest_sha256=(
                args.freeze_manifest_sha256
            ),
            post_freeze_protocol_path=args.post_freeze_protocol,
            expected_post_freeze_protocol_sha256=_sidecar_digest(
                args.post_freeze_protocol.resolve()
            ),
            implementation_commit=head,
            target_access_started_ts=(
                args.target_access_started_ts
                if args.target_access_started_ts is not None
                else time.time_ns() // 1_000_000
            ),
            provider_timeout_seconds=args.provider_timeout_seconds,
            provider_http_timeout_seconds=(
                args.provider_http_timeout_seconds
            ),
            settlement_max_wait_seconds=args.settlement_max_wait_seconds,
            settlement_poll_interval_seconds=(
                args.settlement_poll_interval_seconds
            ),
            max_workers=args.max_workers,
            overwrite_existing=args.overwrite_existing,
        )
    )


def _evaluate(args: argparse.Namespace) -> dict:
    head = _git_head_and_clean()
    return evaluate_challenge_parallel_future_gate(
        ChallengeFutureEvaluationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            target_free_freeze_manifest_path=args.freeze_manifest,
            expected_target_free_freeze_manifest_sha256=(
                args.freeze_manifest_sha256
            ),
            settled_index_path=args.settled_index,
            expected_settled_index_sha256=args.settled_index_sha256,
            post_freeze_protocol_path=args.post_freeze_protocol,
            expected_post_freeze_protocol_sha256=_sidecar_digest(
                args.post_freeze_protocol.resolve()
            ),
            runtime_policy_profile_path=args.runtime_profile,
            expected_runtime_policy_profile_sha256=_sha256_file(
                args.runtime_profile.resolve()
            ),
            implementation_commit=head,
            evaluation_started_ts=(
                args.evaluation_started_ts
                if args.evaluation_started_ts is not None
                else time.time_ns() // 1_000_000
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest-sha256", required=True)
    parser.add_argument(
        "--post-freeze-protocol",
        type=Path,
        default=DEFAULT_POST_FREEZE_PROTOCOL,
    )
    parser.add_argument("--overwrite-existing", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Settle and evaluate the exact frozen challenge future window."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    settle = subparsers.add_parser("settle")
    _shared_arguments(settle)
    settle.add_argument("--target-access-started-ts", type=int)
    settle.add_argument("--provider-timeout-seconds", type=float, default=15.0)
    settle.add_argument(
        "--provider-http-timeout-seconds",
        type=float,
        default=5.0,
    )
    settle.add_argument(
        "--settlement-max-wait-seconds",
        type=float,
        default=21_600.0,
    )
    settle.add_argument(
        "--settlement-poll-interval-seconds",
        type=float,
        default=30.0,
    )
    settle.add_argument("--max-workers", type=int, default=8)

    evaluate = subparsers.add_parser("evaluate")
    _shared_arguments(evaluate)
    evaluate.add_argument("--settled-index", type=Path, required=True)
    evaluate.add_argument("--settled-index-sha256", required=True)
    evaluate.add_argument(
        "--runtime-profile",
        type=Path,
        default=DEFAULT_RUNTIME_PROFILE,
    )
    evaluate.add_argument("--evaluation-started-ts", type=int)

    args = parser.parse_args(argv)
    result = _settle(args) if args.command == "settle" else _evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
