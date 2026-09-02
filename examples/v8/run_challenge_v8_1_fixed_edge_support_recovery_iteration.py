#!/usr/bin/env python3
"""Run preregistered historical-development iteration 4 exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_historical_development import (
    SAFE_FALSES,
    HistoricalDevelopmentEvaluationConfig,
    run_historical_development_evaluation,
)
from bigan.v8.polymarket.challenge_v8_1_fixed_edge_support_recovery import (
    CANDIDATE_ID,
    build_fixed_edge_support_recovery_comparison,
    materialize_fixed_edge_support_recovery_decisions,
    validate_fixed_edge_support_recovery_profile,
)
from examples.v8.run_challenge_historical_development import (
    CONFIG_DIR,
    DEFAULT_CLOSURE,
    DEFAULT_LEDGER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REGISTRY,
    DEFAULT_STANDARD,
    _require_clean_preregistered_state,
    _sidecar_digest,
)

DEFAULT_PROFILE = (
    CONFIG_DIR
    / "challenge_v8_1_fixed_edge_support_recovery_0_025_profile.json"
)
DEFAULT_PREREGISTRATION = (
    CONFIG_DIR
    / "challenge_historical_development_iteration_004_preregistration.json"
)
DEFAULT_MARKET_IDS = (
    CONFIG_DIR / "challenge_historical_development_exact_195_market_ids.txt"
)
DEFAULT_PREVIOUS_ENTRY = (
    CONFIG_DIR / "challenge_historical_development_iteration_003_entry.json"
)
PREVIOUS_ENTRY_SHA256 = (
    "abe1eb3b6e03530c15cb51326801e783454a0be4d5885edd7dcca8dab22779ae"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, expected_sha256: str, *, label: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _descriptor(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--base-guard-rows", required=True, type=Path)
    parser.add_argument("--base-comparison-rows", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument("--market-ids", type=Path, default=DEFAULT_MARKET_IDS)
    parser.add_argument(
        "--previous-entry",
        type=Path,
        default=DEFAULT_PREVIOUS_ENTRY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attempt-closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--development-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--success-standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)

    git_state = _require_clean_preregistered_state(args.preregistration)
    profile_path = args.profile.resolve()
    preregistration_path = args.preregistration.resolve()
    market_ids_path = args.market_ids.resolve()
    previous_entry_path = args.previous_entry.resolve()
    profile = _json(profile_path)
    validate_fixed_edge_support_recovery_profile(profile)
    preregistration = _json(preregistration_path)
    if (
        preregistration.get("candidate_id") != CANDIDATE_ID
        or preregistration.get("previous_iteration_entry_sha256")
        != PREVIOUS_ENTRY_SHA256
    ):
        raise ValueError("iteration-4 preregistration mismatch")
    _verify(profile_path, _sidecar_digest(profile_path), label="candidate profile")
    _verify(
        preregistration_path,
        _sidecar_digest(preregistration_path),
        label="iteration preregistration",
    )
    lineage = profile["lineage"]

    # Freeze target-free decisions before reading any outcome comparison.
    _verify(
        args.base_guard_rows.resolve(),
        lineage["base_v8_1_guard_replay_sha256"],
        label="base v8.1 guard replay",
    )
    _verify(
        market_ids_path,
        lineage["exact_195_market_ids_sha256"],
        label="frozen market ids",
    )
    frozen_market_ids = market_ids_path.read_text(encoding="utf-8").splitlines()
    candidate_decisions = materialize_fixed_edge_support_recovery_decisions(
        base_guard_rows=_jsonl(args.base_guard_rows.resolve()),
        frozen_market_ids=frozen_market_ids,
        profile=profile,
    )
    run_dir = args.output_dir.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    decisions_path = run_dir / "candidate_target_free_decisions.jsonl"
    _write_jsonl(decisions_path, candidate_decisions)
    accepted_count = sum(
        row["selected_action"] != "NO_TRADE" for row in candidate_decisions
    )
    target_free_manifest = {
        "schema_version": (
            "bigan-v8-challenge-fixed-edge-support-recovery-"
            "target-free-manifest-v1"
        ),
        "run_id": args.run_id,
        "candidate_id": CANDIDATE_ID,
        "implementation_base_commit": git_state[
            "implementation_base_commit"
        ],
        "preregistration_commit": git_state["preregistration_commit"],
        "implementation_commit": git_state["implementation_commit"],
        "profile": _descriptor(profile_path),
        "preregistration": _descriptor(preregistration_path),
        "frozen_market_ids": _descriptor(market_ids_path),
        "base_guard_rows": _descriptor(args.base_guard_rows.resolve()),
        "candidate_decisions": _descriptor(decisions_path),
        "target_free_accepted_market_count": accepted_count,
        "outcomes_labels_settlement_or_pnl_opened": False,
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "collection_started": False,
        "safety": SAFE_FALSES,
    }
    target_free_manifest_path = run_dir / "candidate_target_free_manifest.json"
    _write_json(target_free_manifest_path, target_free_manifest)

    _verify(
        args.base_comparison_rows.resolve(),
        lineage["base_v8_1_market_comparison_sha256"],
        label="base v8.1 market comparison",
    )
    comparison = build_fixed_edge_support_recovery_comparison(
        candidate_decisions=candidate_decisions,
        base_comparison_rows=_jsonl(args.base_comparison_rows.resolve()),
        frozen_market_ids=frozen_market_ids,
    )
    comparison_path = run_dir / "candidate_market_comparison.jsonl"
    _write_jsonl(comparison_path, comparison)
    result = run_historical_development_evaluation(
        HistoricalDevelopmentEvaluationConfig(
            run_id="evaluation",
            output_dir=run_dir,
            iteration_number=4,
            candidate_id=CANDIDATE_ID,
            comparison_rows_path=comparison_path,
            expected_comparison_rows_sha256=_sha256(comparison_path),
            preregistration_path=preregistration_path,
            expected_preregistration_sha256=_sha256(preregistration_path),
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
            previous_iteration_entry_sha256=PREVIOUS_ENTRY_SHA256,
            previous_iteration_entry_path=previous_entry_path,
        )
    )
    manifest = {
        "schema_version": (
            "bigan-v8-challenge-fixed-edge-support-recovery-"
            "development-manifest-v1"
        ),
        "run_id": args.run_id,
        "candidate_id": CANDIDATE_ID,
        "target_free_manifest": _descriptor(target_free_manifest_path),
        "candidate_comparison": _descriptor(comparison_path),
        "historical_evaluation_manifest": _descriptor(
            result["manifest_path"]
        ),
        "historical_evaluation_report": _descriptor(result["report_path"]),
        "iteration_entry": _descriptor(result["iteration_entry_path"]),
        "all_historical_success_criteria_passed": result["report"][
            "all_historical_success_criteria_passed"
        ],
        "replacement_future_attempt_preregistration_allowed": result[
            "report"
        ]["replacement_future_attempt_preregistration_allowed"],
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "collection_started": False,
        "safety": SAFE_FALSES,
    }
    manifest_path = run_dir / "candidate_development_manifest.json"
    _write_json(manifest_path, manifest)
    summary = {
        "run_dir": str(run_dir),
        "target_free_manifest_sha256": _sha256(target_free_manifest_path),
        "candidate_comparison_sha256": _sha256(comparison_path),
        "evaluation_report_sha256": result["report_sha256"],
        "iteration_entry_sha256": result["iteration_entry_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "accepted_market_count": accepted_count,
        "all_historical_success_criteria_passed": result["report"][
            "all_historical_success_criteria_passed"
        ],
        "replacement_future_attempt_preregistration_allowed": result[
            "report"
        ]["replacement_future_attempt_preregistration_allowed"],
        "promotion_evidence_eligible": False,
        "collection_started": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_historical_success_criteria_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
