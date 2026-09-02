#!/usr/bin/env python3
"""Recompute the strict v8.5 pre-collection historical replay gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.historical_replay_gate import (
    audit_historical_replay_superiority,
    validate_exact_historical_model_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-contract", required=True, type=Path)
    parser.add_argument("--candidate-contract", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--leakage-audit", required=True, type=Path)
    parser.add_argument("--candidate-rows", required=True, type=Path)
    parser.add_argument("--baseline-rows", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--frozen-model-binding", required=True, type=Path)
    parser.add_argument("--frozen-model-artifact", required=True, type=Path)
    parser.add_argument("--candidate-profile", required=True, type=Path)
    parser.add_argument("--expected-gate-contract-sha256", required=True)
    parser.add_argument("--expected-candidate-contract-sha256", required=True)
    parser.add_argument("--expected-source-report-sha256", required=True)
    parser.add_argument("--expected-leakage-audit-sha256", required=True)
    parser.add_argument("--expected-candidate-rows-sha256", required=True)
    parser.add_argument("--expected-baseline-rows-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-frozen-model-binding-sha256", required=True)
    parser.add_argument("--expected-frozen-model-artifact-sha256", required=True)
    parser.add_argument("--expected-candidate-profile-sha256", required=True)
    parser.add_argument("--evaluation-completed-ts", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = {
        "gate_contract": args.gate_contract.resolve(),
        "candidate_contract": args.candidate_contract.resolve(),
        "source_report": args.source_report.resolve(),
        "leakage_audit": args.leakage_audit.resolve(),
        "candidate_rows": args.candidate_rows.resolve(),
        "baseline_rows": args.baseline_rows.resolve(),
        "source_manifest": args.source_manifest.resolve(),
        "frozen_model_binding": args.frozen_model_binding.resolve(),
        "frozen_model_artifact": args.frozen_model_artifact.resolve(),
        "candidate_profile": args.candidate_profile.resolve(),
    }
    expected = {
        "gate_contract": args.expected_gate_contract_sha256,
        "candidate_contract": args.expected_candidate_contract_sha256,
        "source_report": args.expected_source_report_sha256,
        "leakage_audit": args.expected_leakage_audit_sha256,
        "candidate_rows": args.expected_candidate_rows_sha256,
        "baseline_rows": args.expected_baseline_rows_sha256,
        "source_manifest": args.expected_source_manifest_sha256,
        "frozen_model_binding": (
            args.expected_frozen_model_binding_sha256
        ),
        "frozen_model_artifact": (
            args.expected_frozen_model_artifact_sha256
        ),
        "candidate_profile": args.expected_candidate_profile_sha256,
    }
    for name, path in paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected[name].lower():
            raise ValueError(f"{name} SHA-256 mismatch")

    source_manifest = _json(paths["source_manifest"])
    _validate_source_manifest(source_manifest, paths=paths, expected=expected)
    candidate_contract = _json(paths["candidate_contract"])
    binding_summary = validate_exact_historical_model_binding(
        candidate_contract=candidate_contract,
        frozen_model_binding=_json(paths["frozen_model_binding"]),
        frozen_model_artifact=_json(paths["frozen_model_artifact"]),
        source_manifest=source_manifest,
        expected_binding_sha256=expected["frozen_model_binding"],
        expected_model_artifact_sha256=expected["frozen_model_artifact"],
        expected_source_manifest_sha256=expected["source_manifest"],
        expected_candidate_profile_sha256=expected["candidate_profile"],
    )
    report = audit_historical_replay_superiority(
        gate_contract=_json(paths["gate_contract"]),
        candidate_contract=candidate_contract,
        source_report=_json(paths["source_report"]),
        leakage_audit=_json(paths["leakage_audit"]),
        candidate_rows=_jsonl(paths["candidate_rows"]),
        baseline_rows=_jsonl(paths["baseline_rows"]),
        lineage_sha256s=expected,
        evaluation_completed_ts=args.evaluation_completed_ts,
        exact_model_binding_summary=binding_summary,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["historical_superiority_gate_passed"] else 2


def _validate_source_manifest(
    manifest: dict[str, Any],
    *,
    paths: dict[str, Path],
    expected: dict[str, str],
) -> None:
    descriptors = {
        "report": "source_report",
        "leakage_audit": "leakage_audit",
        "candidate_selected_rows": "candidate_rows",
        "v6_7_baseline_selected_rows": "baseline_rows",
        "model": "frozen_model_artifact",
        "profile": "candidate_profile",
    }
    for manifest_name, local_name in descriptors.items():
        descriptor = dict(manifest.get(manifest_name) or {})
        if descriptor.get("sha256") != expected[local_name]:
            raise ValueError(f"source manifest {manifest_name} SHA-256 mismatch")
        if Path(str(descriptor.get("path") or "")).name != paths[local_name].name:
            raise ValueError(f"source manifest {manifest_name} filename mismatch")
    if manifest.get("historical_hard_gate_passed") is not True:
        raise ValueError("source manifest historical hard gate did not pass")
    if manifest.get("fit_leakage_audit_passed") is not True:
        raise ValueError("source manifest leakage audit did not pass")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL objects required: {path}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
