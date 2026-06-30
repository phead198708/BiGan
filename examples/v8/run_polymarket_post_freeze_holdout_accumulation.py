"""Run post-freeze M holdout evidence accumulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_holdout_accumulation import (  # noqa: E402
    PolymarketPostFreezeHoldoutAccumulationConfig,
    run_polymarket_m_post_freeze_holdout_accumulation,
)


def run_polymarket_post_freeze_holdout_accumulation_cli(
    *,
    holdout_report_paths: tuple[Path | str, ...],
    output_dir: Path | str,
    run_id: str = "polymarket_m_post_freeze_holdout_accumulation",
    min_replay_entry_support: int = 20,
    min_unique_market_support: int = 10,
    require_both_side_replay_entries: bool = True,
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_m_post_freeze_holdout_accumulation(
        PolymarketPostFreezeHoldoutAccumulationConfig(
            holdout_report_paths=holdout_report_paths,
            output_dir=output_dir,
            run_id=run_id,
            min_replay_entry_support=min_replay_entry_support,
            min_unique_market_support=min_unique_market_support,
            require_both_side_replay_entries=require_both_side_replay_entries,
            overwrite_existing=overwrite_existing,
        )
    )
    report = result.report
    return {
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "holdout_run_count": report["holdout_run_count"],
        "unique_market_count": report["unique_market_count"],
        "selected_entry_count": report["selected_entry_count"],
        "replay_entry_count": report["replay_entry_count"],
        "replay_entry_count_by_side": report["replay_entry_count_by_side"],
        "replay_total_pnl_sum": report["replay_total_pnl_sum"],
        "replay_pnl_by_side": report["replay_pnl_by_side"],
        "mean_pnl_per_entry": report["mean_pnl_per_entry"],
        "label_vs_replay_pnl_gap": report["label_vs_replay_pnl_gap"],
        "failed_provenance_run_count": report["failed_provenance_run_count"],
        "duplicate_excluded_run_count": report["duplicate_excluded_run_count"],
        "support_gate_passed": report["support_gate_passed"],
        "support_gate_reason_codes": report["support_gate_reason_codes"],
        "source_model_candidate_eligible": report["source_model_candidate_eligible"],
        "#146_start_allowed": report["#146_start_allowed"],
        "#134_resume_allowed": report["#134_resume_allowed"],
        "report_path": str(result.artifact_paths["report"]),
        "summary_path": str(result.artifact_paths["summary"]),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-report",
        action="append",
        required=True,
        dest="holdout_reports",
        help=(
            "Path to a m_post_freeze_holdout_validation_report.json file or "
            "a run directory containing one. Repeatable."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_m_post_freeze_holdout_accumulation",
    )
    parser.add_argument("--min-replay-entry-support", type=int, default=20)
    parser.add_argument("--min-unique-market-support", type=int, default=10)
    parser.add_argument(
        "--allow-single-side-replay-entries",
        action="store_true",
        help="Disable the default both-side replay support requirement.",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_polymarket_post_freeze_holdout_accumulation_cli(
        holdout_report_paths=tuple(args.holdout_reports),
        output_dir=args.output_dir,
        run_id=args.run_id,
        min_replay_entry_support=args.min_replay_entry_support,
        min_unique_market_support=args.min_unique_market_support,
        require_both_side_replay_entries=not args.allow_single_side_replay_entries,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
