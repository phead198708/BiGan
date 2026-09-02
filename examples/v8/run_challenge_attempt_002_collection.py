#!/usr/bin/env python3
"""Preflight or run the authorization-gated attempt-002 collection."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.challenge_attempt_002_collection import (
    Attempt002CollectionConfig,
    preflight_attempt_002_collection,
    run_attempt_002_collection,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLES_DIR.parent.parent
CONFIG_DIR = EXAMPLES_DIR / "polymarket_configs"
DEFAULT_PROTOCOL = CONFIG_DIR / "challenge_attempt_002_preregistration.json"
DEFAULT_COLLECTOR_PROTOCOL = (
    CONFIG_DIR / "execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
DEFAULT_FEATURE_CONTRACT = (
    CONFIG_DIR
    / "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)


def _git_head_and_clean() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(
            "attempt-002 collection requires a clean committed worktree"
        )
    return head


def _config(args: argparse.Namespace) -> Attempt002CollectionConfig:
    return Attempt002CollectionConfig(
        repository_root=REPOSITORY_ROOT,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.protocol_sha256,
        operator_authorization_path=args.operator_authorization,
        expected_operator_authorization_sha256=(
            args.operator_authorization_sha256
        ),
        collector_protocol_path=args.collector_protocol,
        expected_collector_protocol_sha256=args.collector_protocol_sha256,
        feature_contract_path=args.feature_contract,
        expected_feature_contract_sha256=args.feature_contract_sha256,
        service_root=args.service_root,
        implementation_commit=_git_head_and_clean(),
        run_id=args.run_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        command.add_argument("--protocol-sha256", required=True)
        command.add_argument(
            "--operator-authorization",
            required=True,
            type=Path,
        )
        command.add_argument(
            "--operator-authorization-sha256",
            required=True,
        )
        command.add_argument(
            "--collector-protocol",
            type=Path,
            default=DEFAULT_COLLECTOR_PROTOCOL,
        )
        command.add_argument("--collector-protocol-sha256", required=True)
        command.add_argument(
            "--feature-contract",
            type=Path,
            default=DEFAULT_FEATURE_CONTRACT,
        )
        command.add_argument("--feature-contract-sha256", required=True)
        command.add_argument("--service-root", required=True, type=Path)
    args = parser.parse_args(argv)
    config = _config(args)
    result = (
        preflight_attempt_002_collection(config)
        if args.command == "preflight"
        else run_attempt_002_collection(config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
