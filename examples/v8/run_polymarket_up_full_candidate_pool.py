"""Run UP SELL_BEFORE_CLOSE full candidate-pool diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.post_freeze_up_full_candidate_pool import (  # noqa: E402
    PolymarketUpFullCandidatePoolConfig,
    run_polymarket_up_full_candidate_pool_diagnostics,
)


def run_polymarket_up_full_candidate_pool_cli(
    *,
    m2_candidate_report_path: Path | str,
    output_dir: Path | str,
    run_id: str = "polymarket_up_full_candidate_pool_diagnostic",
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_up_full_candidate_pool_diagnostics(
        PolymarketUpFullCandidatePoolConfig(
            m2_candidate_report_path=m2_candidate_report_path,
            output_dir=output_dir,
            run_id=run_id,
            overwrite_existing=overwrite_existing,
        )
    )
    pool = result.candidate_pool_report
    proxy = result.feature_proxy_report
    return {
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "total_up_candidate_pool_size": pool["total_up_candidate_pool_size"],
        "guard_compatible_up_pool_size": pool["guard_compatible_up_pool_size"],
        "m2_selected_up_count": pool["m2_selected_up_count"],
        "m2_non_selected_guard_compatible_up_count": pool[
            "m2_non_selected_guard_compatible_up_count"
        ],
        "non_selected_up_rows_viable_under_non_leaky_proxy_count": pool[
            "non_selected_up_rows_viable_under_non_leaky_proxy_count"
        ],
        "up_path_should_remain_fully_blocked": pool[
            "up_path_should_remain_fully_blocked"
        ],
        "selection_uses_only_allowed_fields": proxy[
            "selection_uses_only_allowed_fields"
        ],
        "original_score_vs_replay_correlation": proxy[
            "original_score_vs_replay_correlation"
        ],
        "feature_proxy_score_vs_replay_correlation": proxy[
            "feature_proxy_score_vs_replay_correlation"
        ],
        "#146_start_allowed": pool["#146_start_allowed"],
        "#134_resume_allowed": pool["#134_resume_allowed"],
        "candidate_pool_report_path": str(
            result.artifact_paths["candidate_pool_report"]
        ),
        "feature_proxy_report_path": str(
            result.artifact_paths["feature_proxy_report"]
        ),
        "manifest_path": str(result.artifact_paths["manifest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-candidate-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="polymarket_up_full_candidate_pool_diagnostic",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    summary = run_polymarket_up_full_candidate_pool_cli(
        m2_candidate_report_path=args.m2_candidate_report,
        output_dir=args.output_dir,
        run_id=args.run_id,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
