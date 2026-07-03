"""Run diagnostic N UP replay-aligned action-value candidate reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_n_up_replay_aligned import (  # noqa: E402
    PolymarketNUpReplayAlignedConfig,
    run_polymarket_n_up_replay_aligned_candidate,
)


def run_polymarket_n_up_replay_aligned_cli(
    *,
    m2_candidate_report_path: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_n_up_replay_aligned_candidate",
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_n_up_replay_aligned_candidate(
        PolymarketNUpReplayAlignedConfig(
            m2_candidate_report_path=m2_candidate_report_path,
            output_dir=output_dir,
            run_id=run_id,
            overwrite_existing=overwrite_existing,
        )
    )
    candidate = result.candidate_report
    overlay = result.score_overlay_report
    return {
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "candidate_name": candidate["candidate_name"],
        "m2_up_selected_count": candidate["m2_up_selected_count"],
        "m2_up_replay_pnl_sum": candidate["m2_up_replay_pnl_sum"],
        "m2_up_label_vs_replay_gap": candidate["m2_up_label_vs_replay_gap"],
        "n_would_selected_up_count": candidate["n_would_selected_up_count"],
        "n_would_blocked_up_count": candidate["n_would_blocked_up_count"],
        "n_would_selected_up_replay_pnl_sum": candidate[
            "n_would_selected_up_replay_pnl_sum"
        ],
        "n_label_vs_replay_gap_after_correction": candidate[
            "n_label_vs_replay_gap_after_correction"
        ],
        "n_blocked_up_false_positive_count": candidate[
            "n_blocked_up_false_positive_count"
        ],
        "original_score_vs_replay_correlation": overlay[
            "original_score_vs_replay_correlation"
        ],
        "replay_aligned_score_proxy_vs_replay_correlation": overlay[
            "replay_aligned_score_proxy_vs_replay_correlation"
        ],
        "#146_start_allowed": candidate["#146_start_allowed"],
        "#134_resume_allowed": candidate["#134_resume_allowed"],
        "candidate_report_path": str(result.artifact_paths["candidate_report"]),
        "score_overlay_report_path": str(
            result.artifact_paths["score_overlay_report"]
        ),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-candidate-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_n_up_replay_aligned_candidate",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_polymarket_n_up_replay_aligned_cli(
        m2_candidate_report_path=args.m2_candidate_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
