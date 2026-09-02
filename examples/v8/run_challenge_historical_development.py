#!/usr/bin/env python3
"""Run one preregistered exact-195 historical-development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_historical_development import (
    ZERO_SHA256,
    HistoricalDevelopmentEvaluationConfig,
    run_historical_development_evaluation,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent.parent
CONFIG_DIR = EXAMPLES_DIR / "polymarket_configs"
DEFAULT_OUTPUT_DIR = EXAMPLES_DIR / "polymarket_runs"
DEFAULT_CLOSURE = CONFIG_DIR / "challenge_attempt_001_closure.json"
DEFAULT_REGISTRY = (
    CONFIG_DIR / "challenge_historical_development_data_registry.json"
)
DEFAULT_STANDARD = (
    CONFIG_DIR / "challenge_historical_development_success_standard_v2.json"
)
DEFAULT_LEDGER = (
    CONFIG_DIR / "challenge_historical_development_iteration_ledger.json"
)


def _run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_preregistered_state(
    preregistration_path: Path,
) -> dict[str, Any]:
    if _run_git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("historical replay requires a clean committed worktree")
    head = _run_git("rev-parse", "HEAD")
    path = preregistration_path.resolve()
    try:
        relative_path = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("preregistration must be inside this repository") from error
    tracked_path = relative_path.as_posix()
    _run_git("ls-files", "--error-unmatch", tracked_path)
    preregistration_commit = _run_git(
        "log",
        "-1",
        "--format=%H",
        "--",
        tracked_path,
    )
    if not preregistration_commit:
        raise ValueError("preregistration has no introducing commit")
    committed_bytes = subprocess.run(
        ["git", "show", f"{preregistration_commit}:{tracked_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if committed_bytes != path.read_bytes():
        raise ValueError("preregistration bytes changed after its introducing commit")

    preregistration = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(preregistration, dict):
        raise ValueError("preregistration must be a JSON object")
    implementation_base_commit = str(
        preregistration.get("implementation_commit") or ""
    )
    if not implementation_base_commit:
        raise ValueError("preregistration implementation_commit is required")
    actual_parent = _run_git("rev-parse", f"{preregistration_commit}^")
    if implementation_base_commit != actual_parent:
        raise ValueError(
            "preregistration implementation_commit must be its prechange parent"
        )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", preregistration_commit, head],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    preregistration_changes = set(
        _run_git(
            "diff",
            "--name-only",
            f"{implementation_base_commit}..{preregistration_commit}",
        )
        .splitlines()
    )
    allowed = {tracked_path, str(Path(tracked_path).with_suffix(".sha256"))}
    if not preregistration_changes or not preregistration_changes <= allowed:
        raise ValueError(
            "only the committed preregistration and its sidecar may differ "
            "from its prechange base commit"
        )
    implementation_changes = set(
        _run_git(
            "diff",
            "--name-only",
            f"{preregistration_commit}..{head}",
        ).splitlines()
    )
    if (
        not implementation_changes
        or tracked_path in implementation_changes
        or str(Path(tracked_path).with_suffix(".sha256"))
        in implementation_changes
    ):
        raise ValueError(
            "candidate implementation must follow preregistration without "
            "rewriting preregistration bytes"
        )
    return {
        "preregistration": preregistration,
        "implementation_base_commit": implementation_base_commit,
        "preregistration_commit": preregistration_commit,
        "implementation_commit": head,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError(f"SHA-256 sidecar missing: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    if _sha256(path) != expected:
        raise ValueError(f"SHA-256 sidecar mismatch: {path}")
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iteration-number", required=True, type=int)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--comparison-rows", required=True, type=Path)
    parser.add_argument("--comparison-rows-sha256", required=True)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--previous-entry", type=Path)
    parser.add_argument("--previous-entry-sha256", default=ZERO_SHA256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attempt-closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--development-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--success-standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)

    git_state = _require_clean_preregistered_state(args.preregistration)
    if _sha256(args.preregistration.resolve()) != args.preregistration_sha256.lower():
        raise ValueError("preregistration SHA-256 mismatch")
    result = run_historical_development_evaluation(
        HistoricalDevelopmentEvaluationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            iteration_number=args.iteration_number,
            candidate_id=args.candidate_id,
            comparison_rows_path=args.comparison_rows,
            expected_comparison_rows_sha256=args.comparison_rows_sha256,
            preregistration_path=args.preregistration,
            expected_preregistration_sha256=args.preregistration_sha256,
            success_standard_path=args.success_standard,
            expected_success_standard_sha256=_sidecar_digest(
                args.success_standard.resolve()
            ),
            registry_path=args.development_registry,
            expected_registry_sha256=_sidecar_digest(
                args.development_registry.resolve()
            ),
            ledger_root_path=args.ledger_root,
            expected_ledger_root_sha256=_sidecar_digest(
                args.ledger_root.resolve()
            ),
            attempt_closure_path=args.attempt_closure,
            expected_attempt_closure_sha256=_sidecar_digest(
                args.attempt_closure.resolve()
            ),
            implementation_base_commit=git_state[
                "implementation_base_commit"
            ],
            preregistration_commit=git_state["preregistration_commit"],
            implementation_commit=git_state["implementation_commit"],
            evaluated_at=args.evaluated_at,
            previous_iteration_entry_sha256=args.previous_entry_sha256,
            previous_iteration_entry_path=args.previous_entry,
        )
    )
    summary = {
        "run_dir": str(result["run_dir"]),
        "report_sha256": result["report_sha256"],
        "iteration_entry_sha256": result["iteration_entry_sha256"],
        "iteration_entry_file_sha256": result[
            "iteration_entry_file_sha256"
        ],
        "manifest_sha256": result["manifest_sha256"],
        "all_historical_success_criteria_passed": result["report"][
            "all_historical_success_criteria_passed"
        ],
        "attempt_002_preregistration_allowed": result["report"][
            "attempt_002_preregistration_allowed"
        ],
        "replacement_future_attempt_preregistration_allowed": result["report"][
            "replacement_future_attempt_preregistration_allowed"
        ],
        "promotion_evidence_eligible": False,
        "collection_started": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_historical_success_criteria_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
