#!/usr/bin/env python3
"""Freeze attempt-002 v8.1 and v6.7 decisions after exact-120 collection."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.challenge_attempt_002_target_freeze import (
    Attempt002TargetFreezeConfig,
    run_attempt_002_target_freeze,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLES_DIR.parent.parent
CONFIG_DIR = EXAMPLES_DIR / "polymarket_configs"
DEFAULT_PROTOCOL = CONFIG_DIR / "challenge_attempt_002_preregistration.json"
DEFAULT_FEATURE_CONTRACT = CONFIG_DIR / (
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)
DEFAULT_BINDING = CONFIG_DIR / "parallel_frozen_v8_1_model_binding.json"
DEFAULT_V8_1_CONTRACT = (
    CONFIG_DIR / "parallel_candidate_v8_1_primary_no_fallback_contract.json"
)
DEFAULT_ENTRY_PROFILE = (
    CONFIG_DIR / "challenge_v8_1_entry_price_floor_0_30_profile.json"
)
DEFAULT_SIZING_PROFILE = CONFIG_DIR / (
    "challenge_v8_1_entry_price_floor_0_30_sized_1_0_profile.json"
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
            "attempt-002 target freeze requires a clean committed worktree"
        )
    return head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--service-root", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--supervisor-state", required=True, type=Path)
    parser.add_argument("--supervisor-state-sha256", required=True)
    parser.add_argument("--collector-index", required=True, type=Path)
    parser.add_argument("--collector-index-sha256", required=True)
    parser.add_argument(
        "--feature-contract",
        type=Path,
        default=DEFAULT_FEATURE_CONTRACT,
    )
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument(
        "--v6-2-candidate-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--v6-2-candidate-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--historical-fit-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--historical-fit-manifest-sha256", required=True)
    parser.add_argument(
        "--frozen-model-binding",
        type=Path,
        default=DEFAULT_BINDING,
    )
    parser.add_argument("--frozen-model-binding-sha256", required=True)
    parser.add_argument(
        "--v8-1-candidate-contract",
        type=Path,
        default=DEFAULT_V8_1_CONTRACT,
    )
    parser.add_argument("--v8-1-candidate-contract-sha256", required=True)
    parser.add_argument(
        "--entry-price-floor-profile",
        type=Path,
        default=DEFAULT_ENTRY_PROFILE,
    )
    parser.add_argument("--entry-price-floor-profile-sha256", required=True)
    parser.add_argument(
        "--sizing-profile",
        type=Path,
        default=DEFAULT_SIZING_PROFILE,
    )
    parser.add_argument("--sizing-profile-sha256", required=True)
    parser.add_argument(
        "--decision-freeze-created-ts",
        required=True,
        type=int,
    )
    args = parser.parse_args(argv)
    result = run_attempt_002_target_freeze(
        Attempt002TargetFreezeConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            service_root=args.service_root,
            protocol_path=args.protocol,
            expected_protocol_sha256=args.protocol_sha256,
            supervisor_state_path=args.supervisor_state,
            expected_supervisor_state_sha256=(
                args.supervisor_state_sha256
            ),
            collector_index_path=args.collector_index,
            expected_collector_index_sha256=args.collector_index_sha256,
            feature_contract_path=args.feature_contract,
            expected_feature_contract_sha256=(
                args.feature_contract_sha256
            ),
            v6_2_candidate_manifest_path=args.v6_2_candidate_manifest,
            expected_v6_2_candidate_manifest_sha256=(
                args.v6_2_candidate_manifest_sha256
            ),
            historical_fit_manifest_path=args.historical_fit_manifest,
            expected_historical_fit_manifest_sha256=(
                args.historical_fit_manifest_sha256
            ),
            frozen_model_binding_path=args.frozen_model_binding,
            expected_frozen_model_binding_sha256=(
                args.frozen_model_binding_sha256
            ),
            v8_1_candidate_contract_path=args.v8_1_candidate_contract,
            expected_v8_1_candidate_contract_sha256=(
                args.v8_1_candidate_contract_sha256
            ),
            entry_price_floor_profile_path=(
                args.entry_price_floor_profile
            ),
            expected_entry_price_floor_profile_sha256=(
                args.entry_price_floor_profile_sha256
            ),
            sizing_profile_path=args.sizing_profile,
            expected_sizing_profile_sha256=(
                args.sizing_profile_sha256
            ),
            implementation_commit=_git_head_and_clean(),
            decision_freeze_created_ts=args.decision_freeze_created_ts,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
