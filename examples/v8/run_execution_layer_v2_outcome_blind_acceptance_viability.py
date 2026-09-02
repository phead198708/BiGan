"""Run the #196 outcome-blind frozen accepted-bet viability audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (  # noqa: E402
    OutcomeBlindAcceptanceViabilityConfig,
    run_outcome_blind_acceptance_viability_audit,
)


def run_audit(
    *,
    run_id: str,
    output_dir: Path | str,
    candidate_freeze_manifest: Path | str,
    candidate_freeze_manifest_sha256: str,
    role_assignment_manifest: Path | str,
    role_assignment_manifest_sha256: str,
    overwrite_existing: bool = False,
) -> dict:
    result = run_outcome_blind_acceptance_viability_audit(
        OutcomeBlindAcceptanceViabilityConfig(
            run_id=run_id,
            output_dir=output_dir,
            candidate_freeze_manifest_path=candidate_freeze_manifest,
            expected_candidate_freeze_manifest_sha256=(candidate_freeze_manifest_sha256),
            role_assignment_manifest_path=role_assignment_manifest,
            expected_role_assignment_manifest_sha256=(role_assignment_manifest_sha256),
            overwrite_existing=overwrite_existing,
        )
    )
    report = result["report"]
    return {
        "run_id": run_id,
        "audit_status": report["status"],
        "audited_market_count": report["audited_market_count"],
        "decision_group_count": report["decision_group_count"],
        "materialized_action_row_count": report["materialized_action_row_count"],
        "selected_action_distribution": report["selected_action_distribution"],
        "raw_ranker_top_action_distribution": report["raw_ranker_top_action_distribution"],
        "best_trade_action_distribution": report["best_trade_action_distribution"],
        "execution_guard_allowed_count": report["execution_guard_allowed_count"],
        "accepted_bet_support_shortfall": report["accepted_bet_support_shortfall"],
        "first_terminal_stage_distribution": report["first_terminal_stage_distribution"],
        "zero_accepted_bet_explanation": report["zero_accepted_bet_explanation"],
        "target_or_outcome_files_opened": report["target_or_outcome_files_opened"],
        "current_oof_or_validation_pnl_used": report["current_oof_or_validation_pnl_used"],
        "report_path": str(result["report_path"]),
        "report_sha256": result["report_sha256"],
        "rows_path": str(result["rows_path"]),
        "rows_sha256": result["rows_sha256"],
        "manifest_path": str(result["manifest_path"]),
        "manifest_sha256": result["manifest_sha256"],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "paper_candidate_allowed": False,
        "live_trading_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "paper_only": True,
        "capital_at_risk": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--candidate-freeze-manifest", required=True)
    parser.add_argument("--candidate-freeze-manifest-sha256", required=True)
    parser.add_argument("--role-assignment-manifest", required=True)
    parser.add_argument("--role-assignment-manifest-sha256", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_audit(
        run_id=args.run_id,
        output_dir=args.output_dir,
        candidate_freeze_manifest=args.candidate_freeze_manifest,
        candidate_freeze_manifest_sha256=args.candidate_freeze_manifest_sha256,
        role_assignment_manifest=args.role_assignment_manifest,
        role_assignment_manifest_sha256=args.role_assignment_manifest_sha256,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
