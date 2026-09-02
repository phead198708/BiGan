#!/usr/bin/env python3
"""Build the dedicated #219 v6.2 bounded-paper-candidate gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_paper_candidate import (
    MANUAL_APPROVAL_SCOPE,
    MarketClusteredMeanEVV62PaperCandidateConfig,
    run_market_clustered_mean_ev_v6_2_paper_candidate_gate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument("--promoted-candidate-manifest", type=Path, required=True)
    parser.add_argument("--paper-handoff-plan", type=Path, required=True)
    parser.add_argument("--manual-approval", action="store_true")
    parser.add_argument("--manual-approval-id", default="issue-219-v6-2-paper-canary-approval")
    parser.add_argument("--manual-approval-operator", default="codex")
    parser.add_argument("--manual-approval-scope", default=MANUAL_APPROVAL_SCOPE)
    parser.add_argument("--manual-approval-ts", type=int)
    parser.add_argument("--builder-git-commit")
    parser.add_argument("--bounded-complete-round-count", type=int, default=12)
    parser.add_argument("--maximum-paper-order-notional", type=float, default=0.2)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    builder = args.builder_git_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    result = run_market_clustered_mean_ev_v6_2_paper_candidate_gate(
        MarketClusteredMeanEVV62PaperCandidateConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            promoted_candidate_manifest_path=args.promoted_candidate_manifest,
            paper_handoff_plan_path=args.paper_handoff_plan,
            manual_approval_approved=args.manual_approval,
            manual_approval_id=args.manual_approval_id,
            manual_approval_operator=args.manual_approval_operator,
            manual_approval_scope=args.manual_approval_scope,
            manual_approval_ts=args.manual_approval_ts or int(time.time() * 1000),
            builder_git_commit=builder,
            bounded_complete_round_count=args.bounded_complete_round_count,
            maximum_paper_order_notional=args.maximum_paper_order_notional,
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "paper_candidate_allowed": report["paper_candidate_allowed"],
                "paper_candidate_allowed_scope": report[
                    "paper_candidate_allowed_scope"
                ],
                "paper_canary_handoff_allowed": report[
                    "paper_canary_handoff_allowed"
                ],
                "v8_execution_handoff_allowed": report[
                    "v8_execution_handoff_allowed"
                ],
                "blocking_reason_codes": report[
                    "paper_candidate_blocking_reason_codes"
                ],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": _sha256(result["manifest_path"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["paper_candidate_allowed"] else 2


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
