"""Run a diagnostic post-freeze M promotion-readiness audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_promotion_readiness_audit import (  # noqa: E402
    PolymarketPostFreezePromotionReadinessAuditConfig,
    run_polymarket_m_post_freeze_promotion_readiness_audit,
)


def run_polymarket_post_freeze_promotion_readiness_audit_cli(
    *,
    accumulation_report_path: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_m_post_freeze_promotion_readiness_audit",
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_m_post_freeze_promotion_readiness_audit(
        PolymarketPostFreezePromotionReadinessAuditConfig(
            accumulation_report_path=accumulation_report_path,
            output_dir=output_dir,
            run_id=run_id,
            overwrite_existing=overwrite_existing,
        )
    )
    report = result.report
    return {
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "promotion_readiness": report["promotion_readiness"],
        "support_gate_passed": report["support_gate_passed"],
        "promotion_evidence_eligible": report["promotion_evidence_eligible"],
        "source_model_candidate_eligible": report["source_model_candidate_eligible"],
        "promotion_gate_reason_codes": report["promotion_gate_reason_codes"],
        "source_model_candidate_ineligible_reason_codes": (
            report["source_model_candidate_ineligible_reason_codes"]
        ),
        "holdout_run_count": report["holdout_run_count"],
        "replay_entry_count": report["replay_entry_count"],
        "replay_unique_market_count": report["replay_unique_market_count"],
        "replay_total_pnl_sum": report["replay_total_pnl_sum"],
        "replay_pnl_by_side": report["replay_pnl_by_side"],
        "up_side_negative_pnl_should_block_promotion_discussion": report[
            "up_side_negative_pnl_should_block_promotion_discussion"
        ],
        "#146_start_allowed": report["#146_start_allowed"],
        "#134_resume_allowed": report["#134_resume_allowed"],
        "report_path": str(result.artifact_paths["report"]),
        "summary_path": str(result.artifact_paths["summary"]),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accumulation-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_m_post_freeze_promotion_readiness_audit",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_polymarket_post_freeze_promotion_readiness_audit_cli(
        accumulation_report_path=args.accumulation_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
