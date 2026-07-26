#!/usr/bin/env python3
"""Freeze, claim, and evaluate challenge attempt-002 without collection control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_attempt_002_pipeline import (
    ZERO_SHA256,
    Attempt002EvaluationConfig,
    build_attempt_002_target_access_claim,
    build_attempt_002_target_free_pairs,
    run_attempt_002_future_evaluation,
    validate_attempt_002_operator_authorization,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent.parent
CONFIG_DIR = EXAMPLES_DIR / "polymarket_configs"
DEFAULT_PROTOCOL = CONFIG_DIR / "challenge_attempt_002_preregistration.json"
DEFAULT_OUTPUT_DIR = EXAMPLES_DIR / "polymarket_runs"


def _git_head_and_clean() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("attempt-002 execution requires a clean worktree")
    return head


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256(path)
    if actual != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError(f"SHA-256 sidecar missing: {sidecar}")
    digest = sidecar.read_text(encoding="ascii").strip().split()[0]
    _verify(path, digest, label=str(path))
    return digest.lower()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL objects required: {path}")
    return rows


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _freeze_pairs(args: argparse.Namespace) -> dict[str, Any]:
    implementation_commit = _git_head_and_clean()
    protocol_path = args.protocol.resolve()
    shared_path = args.shared_source_rows.resolve()
    candidate_path = args.candidate_decisions.resolve()
    baseline_path = args.baseline_decisions.resolve()
    protocol_sha256 = _sidecar_digest(protocol_path)
    for path, expected, label in (
        (shared_path, args.shared_source_rows_sha256, "shared source rows"),
        (
            candidate_path,
            args.candidate_decisions_sha256,
            "candidate decisions",
        ),
        (
            baseline_path,
            args.baseline_decisions_sha256,
            "baseline decisions",
        ),
    ):
        _verify(path, expected, label=label)
    pairs = build_attempt_002_target_free_pairs(
        shared_source_rows=_jsonl(shared_path),
        candidate_decisions=_jsonl(candidate_path),
        baseline_decisions=_jsonl(baseline_path),
        protocol=_json(protocol_path),
    )
    run_dir = args.output_dir.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pairs_path = run_dir / "attempt_002_target_free_pairs.jsonl"
    _write_jsonl_exclusive(pairs_path, pairs)
    manifest = {
        "schema_version": (
            "bigan-v8-challenge-attempt-002-target-free-freeze-manifest-v1"
        ),
        "run_id": args.run_id,
        "implementation_commit": implementation_commit,
        "protocol_sha256": protocol_sha256,
        "shared_source_rows_sha256": _sha256(shared_path),
        "candidate_decisions_sha256": _sha256(candidate_path),
        "baseline_decisions_sha256": _sha256(baseline_path),
        "target_free_pairs_sha256": _sha256(pairs_path),
        "market_count": len(pairs),
        "all_decisions_frozen_before_target_access": True,
        "outcomes_resolution_labels_or_pnl_opened": False,
        "collection_control_invoked": False,
    }
    manifest_path = run_dir / "attempt_002_target_free_freeze_manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    return {
        "run_dir": str(run_dir),
        "pairs_path": str(pairs_path),
        "pairs_sha256": _sha256(pairs_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "market_count": len(pairs),
        "outcomes_opened": False,
        "collection_control_invoked": False,
    }


def _claim(args: argparse.Namespace) -> dict[str, Any]:
    _git_head_and_clean()
    protocol_path = args.protocol.resolve()
    pairs_path = args.target_free_pairs.resolve()
    protocol_sha256 = _sidecar_digest(protocol_path)
    _verify(
        pairs_path,
        args.target_free_pairs_sha256,
        label="target-free pairs",
    )
    authorization_sha256 = ZERO_SHA256
    if args.synthetic_only:
        if args.operator_authorization is not None:
            raise ValueError(
                "synthetic claim must not use operator authorization"
            )
    else:
        if args.operator_authorization is None:
            raise ValueError(
                "real claim requires --operator-authorization"
            )
        authorization_path = args.operator_authorization.resolve()
        authorization_sha256 = _sidecar_digest(authorization_path)
        validate_attempt_002_operator_authorization(
            _json(authorization_path),
            protocol=_json(protocol_path),
            protocol_sha256=protocol_sha256,
        )
    claim = build_attempt_002_target_access_claim(
        target_free_pairs=_jsonl(pairs_path),
        protocol=_json(protocol_path),
        protocol_sha256=protocol_sha256,
        target_access_started_ts=(
            args.target_access_started_ts
            if args.target_access_started_ts is not None
            else time.time_ns() // 1_000_000
        ),
        operator_authorization_sha256=authorization_sha256,
        synthetic_only=args.synthetic_only,
    )
    output_path = args.output.resolve()
    _write_json_exclusive(output_path, claim)
    return {
        "claim_path": str(output_path),
        "claim_sha256": _sha256(output_path),
        "synthetic_only": args.synthetic_only,
        "attempt_and_promotion_alpha_consumed": claim[
            "attempt_and_promotion_alpha_consumed"
        ],
    }


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    implementation_commit = _git_head_and_clean()
    protocol_path = args.protocol.resolve()
    result = run_attempt_002_future_evaluation(
        Attempt002EvaluationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            protocol_path=protocol_path,
            expected_protocol_sha256=_sidecar_digest(protocol_path),
            target_free_pairs_path=args.target_free_pairs,
            expected_target_free_pairs_sha256=(
                args.target_free_pairs_sha256
            ),
            target_access_claim_path=args.target_access_claim,
            expected_target_access_claim_sha256=(
                args.target_access_claim_sha256
            ),
            settlement_targets_path=args.settlement_targets,
            expected_settlement_targets_sha256=(
                args.settlement_targets_sha256
            ),
            implementation_commit=implementation_commit,
            evaluated_at=args.evaluated_at,
            operator_authorization_path=args.operator_authorization,
            expected_operator_authorization_sha256=(
                _sidecar_digest(args.operator_authorization.resolve())
                if args.operator_authorization is not None
                else ZERO_SHA256
            ),
        )
    )
    return {
        "run_dir": str(result["run_dir"]),
        "comparison_sha256": result["comparison_sha256"],
        "result_sha256": result["result_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "synthetic_only": result["result"]["synthetic_only"],
        "all_future_success_criteria_passed": result["result"][
            "all_future_success_criteria_passed"
        ],
        "promotion_evidence_eligible": result["result"][
            "promotion_evidence_eligible"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-pairs")
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument("--shared-source-rows", required=True, type=Path)
    freeze.add_argument("--shared-source-rows-sha256", required=True)
    freeze.add_argument("--candidate-decisions", required=True, type=Path)
    freeze.add_argument("--candidate-decisions-sha256", required=True)
    freeze.add_argument("--baseline-decisions", required=True, type=Path)
    freeze.add_argument("--baseline-decisions-sha256", required=True)
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    claim = subparsers.add_parser("claim")
    claim.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    claim.add_argument("--target-free-pairs", required=True, type=Path)
    claim.add_argument("--target-free-pairs-sha256", required=True)
    claim.add_argument("--operator-authorization", type=Path)
    claim.add_argument("--target-access-started-ts", type=int)
    claim.add_argument("--synthetic-only", action="store_true")
    claim.add_argument("--output", required=True, type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    evaluate.add_argument("--target-free-pairs", required=True, type=Path)
    evaluate.add_argument("--target-free-pairs-sha256", required=True)
    evaluate.add_argument("--target-access-claim", required=True, type=Path)
    evaluate.add_argument("--target-access-claim-sha256", required=True)
    evaluate.add_argument("--settlement-targets", required=True, type=Path)
    evaluate.add_argument("--settlement-targets-sha256", required=True)
    evaluate.add_argument("--operator-authorization", type=Path)
    evaluate.add_argument("--evaluated-at", required=True)
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args(argv)
    if args.command == "freeze-pairs":
        result = _freeze_pairs(args)
    elif args.command == "claim":
        result = _claim(args)
    else:
        result = _evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
