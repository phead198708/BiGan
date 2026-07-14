#!/usr/bin/env python3
"""Reconcile frozen #169 shadows to post-close targets exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    evaluate_pnl_aligned_future_accepted_bets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--shadow-manifest", required=True, type=Path)
    parser.add_argument("--expected-shadow-manifest-sha256", required=True)
    parser.add_argument("--settlement-target-manifest", required=True, type=Path)
    parser.add_argument("--expected-settlement-target-manifest-sha256", required=True)
    args = parser.parse_args()
    if _sha256(args.shadow_manifest) != args.expected_shadow_manifest_sha256:
        raise SystemExit("shadow manifest SHA-256 mismatch")
    shadow_manifest = _load_json(args.shadow_manifest)
    if shadow_manifest.get("future_outcome_targets_loaded") is not False:
        raise SystemExit("shadow manifest is not outcome blind")
    if _sha256(args.settlement_target_manifest) != (
        args.expected_settlement_target_manifest_sha256
    ):
        raise SystemExit("settlement target manifest SHA-256 mismatch")
    target_manifest = _load_json(args.settlement_target_manifest)
    if target_manifest.get("shadow_manifest") != _descriptor(args.shadow_manifest):
        raise SystemExit("settlement target shadow lineage mismatch")
    if not (
        target_manifest.get("future_outcome_targets_loaded") is True
        and target_manifest.get("outcome_reconciliation_started") is True
        and target_manifest.get("identity_reconciliation_passed") is True
    ):
        raise SystemExit("settlement target manifest failed closed")
    target_descriptor = _verified_descriptor(
        target_manifest["settled_evaluation_targets"],
        name="settled evaluation targets",
    )
    target_report_descriptor = _verified_descriptor(
        target_manifest["settlement_target_report"],
        name="settlement target report",
    )
    target_report = _load_json(Path(target_report_descriptor["path"]))
    if not (
        target_report.get("status") == "SETTLED_EVALUATION_TARGETS_READY"
        and target_report.get("shadow_manifest_sha256") == args.expected_shadow_manifest_sha256
        and target_report.get("identity_reconciliation_passed") is True
    ):
        raise SystemExit("settlement target report is not ready")
    _require_fail_closed_safety(target_manifest, name="settlement target manifest")
    _require_fail_closed_safety(target_report, name="settlement target report")
    freeze_descriptor = _verified_descriptor(
        shadow_manifest["evaluation_freeze_manifest"], name="evaluation freeze"
    )
    freeze = _load_json(Path(freeze_descriptor["path"]))
    protocol_descriptor = _verified_descriptor(
        freeze["evaluation_protocol"], name="evaluation protocol"
    )
    collection_descriptor = _verified_descriptor(
        freeze["collection_freeze_manifest"], name="collection freeze"
    )
    candidate_descriptor = _verified_descriptor(
        shadow_manifest["candidate_shadow_rows"], name="candidate shadow"
    )
    baseline_descriptor = _verified_descriptor(
        shadow_manifest["baseline_shadow_rows"], name="baseline shadow"
    )
    run_dir = args.output_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"evaluation output directory already exists: {run_dir}")
    evaluation_marker_path = (
        args.shadow_manifest.resolve().parent
        / "pnl_aligned_future_accepted_bet_evaluation_started.json"
    )
    if evaluation_marker_path.exists():
        raise SystemExit("accepted-bet evaluation already started for this shadow")
    _write_json_exclusive(
        evaluation_marker_path,
        {
            "schema_version": (
                "bigan-v8-execution-layer-v2-pnl-aligned-future-accepted-bet-evaluation-start-v1"
            ),
            "run_id": args.run_id,
            "evaluation_started_ts": int(time.time() * 1000),
            "shadow_manifest": _descriptor(args.shadow_manifest),
            "settlement_target_manifest": _descriptor(args.settlement_target_manifest),
            "exactly_once": True,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "source_model_candidate_eligible": False,
            "freeze_ready": False,
            "promotion_evidence_eligible": False,
            "v8_execution_handoff_allowed": False,
            "#134_resume_allowed": False,
            "#146_start_allowed": False,
        },
    )
    report, rows = evaluate_pnl_aligned_future_accepted_bets(
        evaluation_protocol=_load_json(Path(protocol_descriptor["path"])),
        collection_freeze_manifest=_load_json(Path(collection_descriptor["path"])),
        candidate_shadow_rows=_load_jsonl(Path(candidate_descriptor["path"])),
        baseline_shadow_rows=_load_jsonl(Path(baseline_descriptor["path"])),
        settled_evaluation_rows=_load_jsonl(Path(target_descriptor["path"])),
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    rows_path = run_dir / "pnl_aligned_future_accepted_bet_pnl_rows.jsonl"
    _write_jsonl(rows_path, rows)
    report["accepted_bet_pnl_rows"] = _descriptor(rows_path)
    report_path = run_dir / "pnl_aligned_future_accepted_bet_pnl_report.json"
    _write_json(report_path, report)
    markdown_path = run_dir / "pnl_aligned_future_accepted_bet_pnl_report.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": ("bigan-v8-execution-layer-v2-pnl-aligned-future-evaluation-manifest-v1"),
        "run_id": args.run_id,
        "shadow_manifest": _descriptor(args.shadow_manifest),
        "settlement_target_manifest": _descriptor(args.settlement_target_manifest),
        "settled_evaluation_rows": target_descriptor,
        "accepted_bet_evaluation_start_marker": _descriptor(evaluation_marker_path),
        "accepted_bet_pnl_rows": _descriptor(rows_path),
        "accepted_bet_pnl_report": _descriptor(report_path),
        "accepted_bet_pnl_markdown": _descriptor(markdown_path),
        "future_evidence_gate_passed": report["future_evidence_gate_passed"],
        "future_evidence_gate_blocking_reason_codes": report[
            "future_evidence_gate_blocking_reason_codes"
        ],
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
    manifest_path = run_dir / "pnl_aligned_future_evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"manifest_path={manifest_path}")
    print(f"manifest_sha256={_sha256(manifest_path)}")
    print(f"future_evidence_gate_passed={str(report['future_evidence_gate_passed']).lower()}")
    print(f"candidate_net_pnl={report['candidate_policy_metrics']['settled_net_pnl_sum']}")
    print(f"baseline_net_pnl={report['baseline_policy_metrics']['settled_net_pnl_sum']}")
    print("promotion_evidence_eligible=false")


def _markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate_policy_metrics"]
    baseline = report["baseline_policy_metrics"]
    return "\n".join(
        [
            "# PnL-Aligned Future Accepted-Bet Evaluation",
            "",
            f"- status: `{report['status']}`",
            f"- gate passed: `{str(report['future_evidence_gate_passed']).lower()}`",
            f"- blockers: `{report['future_evidence_gate_blocking_reason_codes']}`",
            f"- candidate accepted bets: `{candidate['accepted_bet_count']}`",
            f"- candidate net PnL: `{candidate['settled_net_pnl_sum']}`",
            f"- candidate ROI: `{candidate['roi']}`",
            f"- baseline accepted bets: `{baseline['accepted_bet_count']}`",
            f"- baseline net PnL: `{baseline['settled_net_pnl_sum']}`",
            f"- candidate minus baseline: `{report['candidate_minus_baseline_net_pnl']}`",
            f"- bootstrap: `{report['market_bootstrap_interval']}`",
            "",
            "Diagnostic evidence only. No source, freeze, promotion, handoff, paper, or live unlock.",
            "",
        ]
    )


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    descriptor = dict(value or {})
    path = Path(str(descriptor.get("path") or ""))
    if not path.is_file() or descriptor.get("sha256") != _sha256(path):
        raise SystemExit(f"{name} descriptor hash mismatch")
    return {"path": str(path), "sha256": str(descriptor["sha256"])}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_fail_closed_safety(payload: dict[str, Any], *, name: str) -> None:
    expected = {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "future_results_used_for_tuning": False,
        "future_results_used_for_unlock": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    failed = sorted(key for key, value in expected.items() if payload.get(key) is not value)
    if failed:
        raise SystemExit(f"{name} safety fields failed closed: {', '.join(failed)}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
