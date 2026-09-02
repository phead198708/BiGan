"""Build a quarantined post-close settled-corpus index for frozen #204 decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (  # noqa: E402
    ConformalV5FutureSettlementCorpusIndexConfig,
    build_conformal_v5_future_settled_corpus_index,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--prediction-freeze-manifest", required=True)
    parser.add_argument("--prediction-freeze-manifest-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--target-access-started-ts", type=int, required=True)
    parser.add_argument("--provider-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--provider-http-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--settlement-max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = build_conformal_v5_future_settled_corpus_index(
        ConformalV5FutureSettlementCorpusIndexConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            prediction_freeze_manifest_path=args.prediction_freeze_manifest,
            expected_prediction_freeze_manifest_sha256=(args.prediction_freeze_manifest_sha256),
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
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "selected_market_count": report["selected_market_count"],
                "settled_corpus_ready_market_count": report["settled_corpus_ready_market_count"],
                "unresolved_or_failed_market_count": report["unresolved_or_failed_market_count"],
                "unresolved_or_failed_reason_distribution": report[
                    "unresolved_or_failed_reason_distribution"
                ],
                "settled_corpus_index_ready": report["settled_corpus_index_ready"],
                "index_path": None if result["index_path"] is None else str(result["index_path"]),
                "index_sha256": result["index_sha256"],
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "source_outcome_blind_rounds_mutated": False,
                "direct_training_corpus_exported": False,
                "future_results_used_for_tuning": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["settled_corpus_index_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
