#!/usr/bin/env python3
"""Run the exact #218 v6.2 manual promotion review."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_manual_promotion import (
    MarketClusteredMeanEVV62ManualPromotionConfig,
    run_market_clustered_mean_ev_v6_2_manual_promotion_review,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--prediction-freeze-manifest", type=Path, required=True)
    parser.add_argument("--settlement-manifest", type=Path, required=True)
    parser.add_argument("--settled-corpus-index", type=Path, required=True)
    parser.add_argument("--single-use-claim", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--side-only-gate-report", type=Path, required=True)
    parser.add_argument("--historical-diagnostic-report", type=Path, required=True)
    parser.add_argument("--builder-git-commit")
    parser.add_argument("--review-completed-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    builder = args.builder_git_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
    ).strip()
    result = run_market_clustered_mean_ev_v6_2_manual_promotion_review(
        MarketClusteredMeanEVV62ManualPromotionConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            candidate_manifest_path=args.candidate_manifest,
            prediction_freeze_manifest_path=args.prediction_freeze_manifest,
            settlement_manifest_path=args.settlement_manifest,
            settled_corpus_index_path=args.settled_corpus_index,
            single_use_claim_path=args.single_use_claim,
            evaluation_manifest_path=args.evaluation_manifest,
            side_only_gate_report_path=args.side_only_gate_report,
            historical_diagnostic_report_path=args.historical_diagnostic_report,
            builder_git_commit=builder,
            review_completed_ts=args.review_completed_ts or int(time.time() * 1000),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "manual_promotion_review_passed": report[
                    "manual_promotion_review_passed"
                ],
                "research_candidate_promoted": report["research_candidate_promoted"],
                "candidate_post_cost_net_pnl": report["future_evidence_summary"][
                    "candidate_post_cost_net_pnl"
                ],
                "promotion_evidence_eligible": report["promotion_evidence_eligible"],
                "paper_candidate_allowed": report["paper_candidate_allowed"],
                "v8_execution_handoff_allowed": report[
                    "v8_execution_handoff_allowed"
                ],
                "blocking_reason_codes": report[
                    "manual_promotion_review_blocking_reason_codes"
                ],
                "report_path": str(result["report_path"]),
                "promotion_manifest_path": str(result["promotion_manifest_path"]),
                "bundle_manifest_path": str(result["bundle_manifest_path"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["manual_promotion_review_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
