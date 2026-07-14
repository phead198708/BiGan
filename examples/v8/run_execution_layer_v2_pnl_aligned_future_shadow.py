#!/usr/bin/env python3
"""Write candidate and baseline outcome-blind #169 future shadows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    run_pnl_aligned_future_outcome_blind_shadow_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--decision-rows-jsonl", required=True, type=Path)
    parser.add_argument("--evaluation-freeze-manifest", required=True, type=Path)
    parser.add_argument("--expected-evaluation-freeze-manifest-sha256", required=True)
    args = parser.parse_args()
    if _sha256(args.evaluation_freeze_manifest) != args.expected_evaluation_freeze_manifest_sha256:
        raise SystemExit("evaluation freeze manifest SHA-256 mismatch")
    freeze = _load_json(args.evaluation_freeze_manifest)
    if freeze.get("future_outcome_targets_loaded") is not False:
        raise SystemExit("evaluation freeze is not outcome blind")
    fit_manifest = _load_json(
        args.model_dir / "pnl_aligned_action_value_fit_manifest.json"
    )
    if fit_manifest.get("model") != freeze.get("model"):
        raise SystemExit("model lineage differs from evaluation freeze")
    decision_rows = _load_jsonl(args.decision_rows_jsonl)
    rows, report = run_pnl_aligned_future_outcome_blind_shadow_comparison(
        model_dir=args.model_dir,
        decision_rows=decision_rows,
    )
    if report["status"] != "OUTCOME_BLIND_COMPARISON_SHADOW_COMPLETE":
        raise SystemExit("outcome-blind shadow failed closed")
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = run_dir / "pnl_aligned_future_candidate_shadow_rows.jsonl"
    baseline_path = run_dir / "pnl_aligned_future_baseline_shadow_rows.jsonl"
    _write_jsonl(candidate_path, rows["candidate"])
    _write_jsonl(baseline_path, rows["baseline"])
    report["candidate_shadow_rows"] = _descriptor(candidate_path)
    report["baseline_shadow_rows"] = _descriptor(baseline_path)
    report_path = run_dir / "pnl_aligned_future_shadow_comparison_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema_version": (
            "bigan-v8-execution-layer-v2-pnl-aligned-future-shadow-manifest-v1"
        ),
        "run_id": args.run_id,
        "evaluation_freeze_manifest": _descriptor(args.evaluation_freeze_manifest),
        "input_decision_rows": _descriptor(args.decision_rows_jsonl),
        "candidate_shadow_rows": _descriptor(candidate_path),
        "baseline_shadow_rows": _descriptor(baseline_path),
        "shadow_report": _descriptor(report_path),
        "future_outcome_targets_loaded": False,
        "outcome_reconciliation_started": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    manifest_path = run_dir / "pnl_aligned_future_shadow_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"manifest_path={manifest_path}")
    print(f"manifest_sha256={_sha256(manifest_path)}")
    print(f"decision_count={report['decision_count']}")
    print(
        "candidate_executable_shadow_bet_count="
        f"{report['candidate_shadow_report']['executable_shadow_bet_count']}"
    )
    print(
        "baseline_executable_shadow_bet_count="
        f"{report['baseline_shadow_report']['executable_shadow_bet_count']}"
    )
    print("future_outcome_targets_loaded=false")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
