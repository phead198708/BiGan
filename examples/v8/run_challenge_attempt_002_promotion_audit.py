#!/usr/bin/env python3
"""Run the fail-closed attempt-002 promotion audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_attempt_002_promotion import (
    attempt_002_promotion_readiness_markdown,
    audit_attempt_002_promotion,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLES_DIR.parent.parent


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
            "attempt-002 promotion audit requires a clean committed worktree"
        )
    return head


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve()
    actual = _sha256(resolved)
    if actual != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {resolved}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return {
        "path": str(resolved),
        "sha256": actual,
        "size_bytes": resolved.stat().st_size,
    }


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-manifest", required=True, type=Path)
    parser.add_argument("--future-manifest-sha256", required=True)
    parser.add_argument(
        "--supplemental-runtime-evidence",
        required=True,
        type=Path,
        help=(
            "Hash-indexed JSON descriptors for operator authorization and "
            "#257/#258/#256 runtime reports."
        ),
    )
    parser.add_argument("--supplemental-runtime-evidence-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)

    implementation_commit = _git_head_and_clean()
    future_descriptor = _descriptor(
        args.future_manifest,
        args.future_manifest_sha256,
    )
    runtime_descriptor = _descriptor(
        args.supplemental_runtime_evidence,
        args.supplemental_runtime_evidence_sha256,
    )
    runtime = _json(Path(runtime_descriptor["path"]))
    report = audit_attempt_002_promotion(
        repository_root=REPOSITORY_ROOT,
        future_evidence_manifest=future_descriptor,
        supplemental_runtime_evidence=runtime,
    )
    report["audit_implementation_commit"] = implementation_commit
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    markdown_output = (
        args.markdown_output.resolve()
        if args.markdown_output is not None
        else output.with_suffix(".md")
    )
    markdown_output.write_text(
        attempt_002_promotion_readiness_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "promotion_eligible": report[
                    "challenge_model_promotion_eligible"
                ],
                "selected_champion_candidate": report[
                    "selected_champion_candidate"
                ],
                "output": str(output),
                "output_sha256": _sha256(output),
                "markdown_output": str(markdown_output),
                "markdown_sha256": _sha256(markdown_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
