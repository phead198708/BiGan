"""Run UP SELL_BEFORE_CLOSE label/replay and calibration diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_up_diagnostics import (  # noqa: E402
    PolymarketUpSellBeforeCloseDiagnosticsConfig,
    run_polymarket_up_sell_before_close_diagnostics,
)


def run_polymarket_up_sell_before_close_diagnostics_cli(
    *,
    m2_candidate_report_path: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_up_sell_before_close_diagnostics",
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_up_sell_before_close_diagnostics(
        PolymarketUpSellBeforeCloseDiagnosticsConfig(
            m2_candidate_report_path=m2_candidate_report_path,
            output_dir=output_dir,
            run_id=run_id,
            overwrite_existing=overwrite_existing,
        )
    )
    label = result.label_replay_report
    calibration = result.calibration_report
    return {
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "root_cause_classification": label["root_cause_classification"],
        "root_cause_codes": label["root_cause_codes"],
        "recommended_next_actions": label["recommended_next_actions"],
        "up_selected_entry_count": label["up_selected_entry_count"],
        "up_known_replay_entry_count": label["up_known_replay_entry_count"],
        "up_label_target_sum": label["up_label_target_sum"],
        "up_replay_pnl_sum": label["up_replay_pnl_sum"],
        "up_label_vs_replay_gap": label["up_label_vs_replay_gap"],
        "score_vs_replay_correlation": calibration[
            "calibrated_action_score_vs_realized_up_replay_pnl_correlation"
        ],
        "rank_score_vs_replay_correlation": calibration[
            "rank_score_vs_realized_up_replay_pnl_correlation"
        ],
        "high_score_negative_replay_up_count": calibration[
            "high_score_negative_replay_up_count"
        ],
        "#146_start_allowed": label["#146_start_allowed"],
        "#134_resume_allowed": label["#134_resume_allowed"],
        "label_replay_report_path": str(result.artifact_paths["label_replay_report"]),
        "calibration_report_path": str(result.artifact_paths["calibration_report"]),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-candidate-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_up_sell_before_close_diagnostics",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_polymarket_up_sell_before_close_diagnostics_cli(
        m2_candidate_report_path=args.m2_candidate_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
