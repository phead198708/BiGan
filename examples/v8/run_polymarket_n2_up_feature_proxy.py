"""Run diagnostic N2 non-leaky UP feature-proxy candidate reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_n2_up_feature_proxy import (  # noqa: E402
    PolymarketN2UpFeatureProxyConfig,
    run_polymarket_n2_up_feature_proxy_candidate,
)


def run_polymarket_n2_up_feature_proxy_cli(
    *,
    m2_candidate_report_path: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_n2_non_leaky_up_feature_proxy",
    n_candidate_report_path: Path | str | None = None,
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_n2_up_feature_proxy_candidate(
        PolymarketN2UpFeatureProxyConfig(
            m2_candidate_report_path=m2_candidate_report_path,
            n_candidate_report_path=n_candidate_report_path,
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
        "n2_selection_uses_only_allowed_fields": candidate[
            "n2_selection_uses_only_allowed_fields"
        ],
        "m2_up_selected_count": candidate["m2_up_selected_count"],
        "m2_up_replay_pnl_sum": candidate["m2_up_replay_pnl_sum"],
        "m2_up_label_vs_replay_gap": candidate["m2_up_label_vs_replay_gap"],
        "n2_would_selected_up_count": candidate["n2_would_selected_up_count"],
        "n2_would_blocked_up_count": candidate["n2_would_blocked_up_count"],
        "n2_would_selected_up_replay_pnl_sum": candidate[
            "n2_would_selected_up_replay_pnl_sum"
        ],
        "n2_selected_label_vs_replay_gap_after_feature_proxy": candidate[
            "n2_selected_label_vs_replay_gap_after_feature_proxy"
        ],
        "n2_blocked_up_false_positive_count": candidate[
            "n2_blocked_up_false_positive_count"
        ],
        "original_score_vs_replay_correlation": overlay[
            "original_score_vs_replay_correlation"
        ],
        "n2_feature_proxy_score_vs_replay_correlation": overlay[
            "n2_feature_proxy_score_vs_replay_correlation"
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
    parser.add_argument("--n-candidate-report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_n2_non_leaky_up_feature_proxy",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_polymarket_n2_up_feature_proxy_cli(
        m2_candidate_report_path=args.m2_candidate_report,
        n_candidate_report_path=args.n_candidate_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
