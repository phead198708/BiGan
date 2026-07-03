"""Run diagnostic M2 replay-parity selection reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_m2_replay_parity import (  # noqa: E402
    PolymarketM2ReplayParityConfig,
    run_polymarket_m2_replay_parity_diagnostics,
)


def run_polymarket_m2_replay_parity_cli(
    *,
    weak_evidence_drilldown_report_path: Path | str,
    accumulation_report_path: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_m2_stateful_replay_parity",
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_m2_replay_parity_diagnostics(
        PolymarketM2ReplayParityConfig(
            weak_evidence_drilldown_report_path=weak_evidence_drilldown_report_path,
            accumulation_report_path=accumulation_report_path,
            output_dir=output_dir,
            run_id=run_id,
            overwrite_existing=overwrite_existing,
        )
    )
    candidate = result.candidate_report
    up = result.up_alignment_report
    return {
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "candidate_name": candidate["candidate_name"],
        "current_frozen_m_promotion_status": candidate[
            "current_frozen_m_promotion_status"
        ],
        "current_frozen_m_evidence_status": candidate[
            "current_frozen_m_evidence_status"
        ],
        "m2_selected_entry_count": candidate["m2_selected_entry_count"],
        "m2_known_replay_entry_count": candidate["m2_known_replay_entry_count"],
        "m2_selected_without_replay_count": candidate[
            "m2_selected_without_replay_count"
        ],
        "current_m_turnover_or_max_entry_attrition_count": candidate[
            "current_m_turnover_or_max_entry_attrition_count"
        ],
        "m2_turnover_or_max_entry_attrition_count": candidate[
            "m2_turnover_or_max_entry_attrition_count"
        ],
        "m2_replay_reconciliation": candidate["m2_replay_entry_reconciliation"][
            "reconciled"
        ],
        "m2_up_selected_entry_count": up["m2_up_selected_entry_count"],
        "m2_up_negative_replay_pnl_count": up[
            "m2_up_negative_replay_pnl_count"
        ],
        "m2_up_positive_label_replay_negative_count": up[
            "m2_up_positive_label_replay_negative_count"
        ],
        "#146_start_allowed": candidate["#146_start_allowed"],
        "#134_resume_allowed": candidate["#134_resume_allowed"],
        "candidate_report_path": str(result.artifact_paths["candidate_report"]),
        "up_alignment_report_path": str(result.artifact_paths["up_alignment_report"]),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-evidence-drilldown", required=True)
    parser.add_argument("--accumulation-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_m2_stateful_replay_parity",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_polymarket_m2_replay_parity_cli(
        weak_evidence_drilldown_report_path=args.weak_evidence_drilldown,
        accumulation_report_path=args.accumulation_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
