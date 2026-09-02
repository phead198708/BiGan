"""Finalize frozen #207 development rounds on quarantine copies using official outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (  # noqa: E402
    PolicySelectedConformalV6DevelopmentSettlementConfig,
    build_policy_selected_conformal_v6_development_settled_corpus_index,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--development-window-manifest", required=True)
    parser.add_argument("--development-window-manifest-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--target-access-started-ts", type=int, required=True)
    parser.add_argument("--provider-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--provider-http-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--settlement-max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = build_policy_selected_conformal_v6_development_settled_corpus_index(
        PolicySelectedConformalV6DevelopmentSettlementConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            development_window_manifest_path=args.development_window_manifest,
            expected_development_window_manifest_sha256=(
                args.development_window_manifest_sha256
            ),
            builder_git_commit=args.builder_git_commit,
            target_access_started_ts=args.target_access_started_ts,
            provider_timeout_seconds=args.provider_timeout_seconds,
            provider_http_timeout_seconds=args.provider_http_timeout_seconds,
            settlement_max_wait_seconds=args.settlement_max_wait_seconds,
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            max_workers=args.max_workers,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "development_settled_corpus_ready": result["report"][
                    "development_settled_corpus_ready"
                ],
                "settled_market_count": result["report"]["settled_market_count"],
                "unresolved_market_count": result["report"]["unresolved_market_count"],
                "role_market_counts": result["report"]["role_market_counts"],
                "blocking_reason_codes": result["report"]["blocking_reason_codes"],
                "policy_pnl_computed": False,
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "paper_candidate_allowed": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["report"]["development_settled_corpus_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
