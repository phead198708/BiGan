#!/usr/bin/env python3
"""Run outcome-blind shadow execution for the frozen PnL-aligned model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    run_pnl_aligned_action_value_outcome_blind_shadow,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--decision-rows-jsonl", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    rows = [
        json.loads(line)
        for line in args.decision_rows_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    replay_rows, report = run_pnl_aligned_action_value_outcome_blind_shadow(
        model_dir=args.model_dir,
        decision_rows=rows,
    )
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    rows_path = run_dir / "pnl_aligned_action_value_shadow_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in replay_rows),
        encoding="utf-8",
    )
    report["shadow_rows"] = _descriptor(rows_path)
    report_path = run_dir / "pnl_aligned_action_value_shadow_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": (
            "bigan-v8-execution-layer-v2-pnl-aligned-action-value-"
            "shadow-manifest-v1"
        ),
        "run_id": args.run_id,
        "input_decision_rows": _descriptor(args.decision_rows_jsonl),
        "shadow_rows": _descriptor(rows_path),
        "shadow_report": _descriptor(report_path),
        "model_dir": str(args.model_dir.resolve()),
        "future_unseen_outcome_reconciliation_required": True,
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
    manifest_path = run_dir / "pnl_aligned_action_value_shadow_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"run_dir={run_dir}")
    print(f"manifest_path={manifest_path}")
    print(f"manifest_sha256={_sha256(manifest_path)}")
    print(f"decision_count={report.get('decision_count', 0)}")
    print(f"model_trade_candidate_count={report.get('model_trade_candidate_count', 0)}")
    print(f"executable_shadow_bet_count={report.get('executable_shadow_bet_count', 0)}")
    print("promotion_evidence_eligible=false")


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
