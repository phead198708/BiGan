"""Run a diagnostic post-freeze M weak-evidence root-cause drilldown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_weak_evidence_drilldown import (  # noqa: E402
    PolymarketPostFreezeWeakEvidenceDrilldownConfig,
    run_polymarket_m_post_freeze_weak_evidence_drilldown,
)


def run_polymarket_post_freeze_weak_evidence_drilldown_cli(
    *,
    promotion_readiness_audit_path: Path | str,
    accumulation_report_path: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_m_post_freeze_weak_evidence_drilldown",
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_m_post_freeze_weak_evidence_drilldown(
        PolymarketPostFreezeWeakEvidenceDrilldownConfig(
            promotion_readiness_audit_path=promotion_readiness_audit_path,
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
        "root_cause_classification": report["root_cause_classification"],
        "weakness_type": report["weakness_type"],
        "recommended_next_actions": report["recommended_next_actions"],
        "promotion_readiness": report["promotion_readiness"],
        "support_gate_passed": report["support_gate_passed"],
        "promotion_evidence_eligible": report["promotion_evidence_eligible"],
        "source_model_candidate_eligible": report["source_model_candidate_eligible"],
        "failed_included_holdout_run_count": report[
            "failed_included_holdout_run_count"
        ],
        "up_loss_entry_count": report["up_loss_entry_count"],
        "down_loss_entry_count": report["down_loss_entry_count"],
        "turnover_or_max_entry_blocked_selected_row_count": report[
            "turnover_or_max_entry_blocked_selected_row_count"
        ],
        "#146_start_allowed": report["#146_start_allowed"],
        "#134_resume_allowed": report["#134_resume_allowed"],
        "report_path": str(result.artifact_paths["report"]),
        "summary_path": str(result.artifact_paths["summary"]),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-readiness-audit", required=True)
    parser.add_argument("--accumulation-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_m_post_freeze_weak_evidence_drilldown",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_polymarket_post_freeze_weak_evidence_drilldown_cli(
        promotion_readiness_audit_path=args.promotion_readiness_audit,
        accumulation_report_path=args.accumulation_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
