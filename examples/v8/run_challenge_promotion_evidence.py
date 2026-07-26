#!/usr/bin/env python3
"""Build exact-lineage regime, policy, and promotion evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.challenge_promotion_evidence import (
    ChallengePromotionEvidenceConfig,
    run_challenge_promotion_evidence,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _sha256_file,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent.parent
CONFIG_DIR = EXAMPLES_DIR / "polymarket_configs"
DEFAULT_OUTPUT_DIR = EXAMPLES_DIR / "polymarket_runs"
DEFAULT_PROMOTION_EVIDENCE_PROTOCOL = CONFIG_DIR / "challenge_promotion_evidence_protocol.json"
DEFAULT_REGIME_CONTRACT = CONFIG_DIR / "regime_definition_contract.json"
DEFAULT_EXECUTION_POLICY_CONTRACT = CONFIG_DIR / "execution_policy_contract.json"
DEFAULT_POLICY_CANDIDATE_MANIFEST = CONFIG_DIR / "policy_candidate_manifest.json"
DEFAULT_COMPATIBILITY_MANIFEST = CONFIG_DIR / "source_execution_compatibility_manifest.json"


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
        raise ValueError("promotion evidence requires a clean committed implementation")
    return head


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError(f"SHA-256 sidecar missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if _sha256_file(path) != expected.lower():
        raise ValueError(f"SHA-256 sidecar mismatch: {path}")
    return expected.lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bind regime and execution-policy evidence to the exact "
            "multiplicity-aware challenge winner."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--freeze-manifest-sha256", required=True)
    parser.add_argument(
        "--promotion-evidence-protocol",
        type=Path,
        default=DEFAULT_PROMOTION_EVIDENCE_PROTOCOL,
    )
    parser.add_argument(
        "--regime-contract",
        type=Path,
        default=DEFAULT_REGIME_CONTRACT,
    )
    parser.add_argument(
        "--execution-policy-contract",
        type=Path,
        default=DEFAULT_EXECUTION_POLICY_CONTRACT,
    )
    parser.add_argument(
        "--policy-candidate-manifest",
        type=Path,
        default=DEFAULT_POLICY_CANDIDATE_MANIFEST,
    )
    parser.add_argument(
        "--compatibility-manifest",
        type=Path,
        default=DEFAULT_COMPATIBILITY_MANIFEST,
    )
    parser.add_argument("--generated-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)

    implementation_commit = _git_head_and_clean()
    result = run_challenge_promotion_evidence(
        ChallengePromotionEvidenceConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            repository_root=REPO_ROOT,
            parallel_evaluation_manifest_path=args.evaluation_manifest,
            expected_parallel_evaluation_manifest_sha256=(args.evaluation_manifest_sha256),
            target_free_freeze_manifest_path=args.freeze_manifest,
            expected_target_free_freeze_manifest_sha256=(args.freeze_manifest_sha256),
            promotion_evidence_protocol_path=(args.promotion_evidence_protocol),
            expected_promotion_evidence_protocol_sha256=_sidecar_digest(
                args.promotion_evidence_protocol.resolve()
            ),
            regime_definition_contract_path=args.regime_contract,
            expected_regime_definition_contract_sha256=_sidecar_digest(
                args.regime_contract.resolve()
            ),
            execution_policy_contract_path=args.execution_policy_contract,
            expected_execution_policy_contract_sha256=_sidecar_digest(
                args.execution_policy_contract.resolve()
            ),
            policy_candidate_manifest_path=args.policy_candidate_manifest,
            expected_policy_candidate_manifest_sha256=_sidecar_digest(
                args.policy_candidate_manifest.resolve()
            ),
            source_execution_compatibility_manifest_path=(args.compatibility_manifest),
            expected_source_execution_compatibility_manifest_sha256=(
                _sidecar_digest(args.compatibility_manifest.resolve())
            ),
            implementation_commit=implementation_commit,
            generated_ts=(
                args.generated_ts if args.generated_ts is not None else time.time_ns() // 1_000_000
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
