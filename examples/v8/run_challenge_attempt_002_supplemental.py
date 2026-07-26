#!/usr/bin/env python3
"""Generate hash-bound #257/#258/#256 evidence for attempt-002."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.challenge_attempt_002_supplemental import (
    Attempt002SupplementalConfig,
    run_attempt_002_supplemental_evidence,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLES_DIR.parent.parent
CONFIG_DIR = EXAMPLES_DIR / "polymarket_configs"
DEFAULT_OUTPUT_DIR = EXAMPLES_DIR / "polymarket_runs"


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
            "attempt-002 supplemental evidence requires a clean worktree"
        )
    return head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--future-manifest", required=True, type=Path)
    parser.add_argument("--future-manifest-sha256", required=True)
    parser.add_argument("--operator-authorization", required=True, type=Path)
    parser.add_argument("--operator-authorization-sha256", required=True)
    parser.add_argument("--shared-source-rows", required=True, type=Path)
    parser.add_argument("--shared-source-rows-sha256", required=True)
    parser.add_argument("--feature-rows", required=True, type=Path)
    parser.add_argument("--feature-rows-sha256", required=True)
    parser.add_argument("--native-decisions", required=True, type=Path)
    parser.add_argument("--native-decisions-sha256", required=True)
    parser.add_argument(
        "--regime-contract",
        type=Path,
        default=CONFIG_DIR / "regime_definition_contract.json",
    )
    parser.add_argument("--regime-contract-sha256", required=True)
    parser.add_argument(
        "--policy-manifest",
        type=Path,
        default=CONFIG_DIR / "policy_candidate_manifest.json",
    )
    parser.add_argument("--policy-manifest-sha256", required=True)
    parser.add_argument(
        "--compatibility-manifest",
        type=Path,
        default=CONFIG_DIR / "source_execution_compatibility_manifest.json",
    )
    parser.add_argument("--compatibility-manifest-sha256", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    result = run_attempt_002_supplemental_evidence(
        Attempt002SupplementalConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            repository_root=REPOSITORY_ROOT,
            future_manifest_path=args.future_manifest,
            expected_future_manifest_sha256=args.future_manifest_sha256,
            operator_authorization_path=args.operator_authorization,
            expected_operator_authorization_sha256=(
                args.operator_authorization_sha256
            ),
            shared_source_rows_path=args.shared_source_rows,
            expected_shared_source_rows_sha256=(
                args.shared_source_rows_sha256
            ),
            feature_rows_path=args.feature_rows,
            expected_feature_rows_sha256=args.feature_rows_sha256,
            native_decisions_path=args.native_decisions,
            expected_native_decisions_sha256=args.native_decisions_sha256,
            regime_contract_path=args.regime_contract,
            expected_regime_contract_sha256=args.regime_contract_sha256,
            policy_manifest_path=args.policy_manifest,
            expected_policy_manifest_sha256=args.policy_manifest_sha256,
            compatibility_manifest_path=args.compatibility_manifest,
            expected_compatibility_manifest_sha256=(
                args.compatibility_manifest_sha256
            ),
            implementation_commit=_git_head_and_clean(),
            generated_at=args.generated_at,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "supplemental_runtime_evidence": str(
                    result["runtime_evidence_path"]
                ),
                "supplemental_runtime_evidence_sha256": result[
                    "runtime_evidence_sha256"
                ],
                "manifest": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "promotion_decision_emitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
