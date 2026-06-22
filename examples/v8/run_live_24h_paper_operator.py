"""Run a v8 24h paper operator workflow with real read-only live data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from examples.v8.run_24h_paper_operator import (  # noqa: E402
    run_24h_paper_operator_cli,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "gh-command", "direct-comment"),
        default="gh-command",
    )
    parser.add_argument(
        "--feed-mode",
        choices=("live-readonly",),
        default="live-readonly",
        help="Explicit live feed mode; replay fallback is not supported here.",
    )
    parser.add_argument("--provider", default="coinbase_public_ticker")
    parser.add_argument(
        "--provider-endpoint",
        default="https://api.exchange.coinbase.com/products/BTC-USD/ticker",
    )
    parser.add_argument("--instrument", default="BTC-USD")
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=60)
    parser.add_argument("--summary-interval-seconds", type=int, default=300)
    parser.add_argument("--request-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-reconnect-attempts", type=int, default=3)
    parser.add_argument("--max-stale-seconds", type=float, default=120.0)
    parser.add_argument(
        "--github-round-progress-comments",
        action="store_true",
        help=(
            "Post/write per-round GitHub progress comment evidence. "
            "Default is off to avoid noisy issue threads."
        ),
    )
    parser.add_argument(
        "--round-progress-comment-mode",
        choices=("dry-run", "gh-command", "direct-comment"),
        help="Override per-round comment mode; defaults to --mode when enabled.",
    )
    parser.add_argument(
        "--round-progress-comment-interval-rounds",
        type=int,
        default=1,
        help="Emit one progress comment every N rounds when enabled.",
    )
    parser.add_argument(
        "--max-round-progress-comments",
        type=int,
        default=0,
        help="Maximum progress comments to emit; 0 means unlimited.",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)

    summary = run_24h_paper_operator_cli(
        run_id=args.run_id,
        output_dir=args.output_dir,
        repo=args.repo,
        issue_number=args.issue_number,
        mode=args.mode,
        duration_hours=args.duration_hours,
        feed_event_interval_seconds=args.poll_interval_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        summary_interval_seconds=args.summary_interval_seconds,
        feed_mode=args.feed_mode,
        provider=args.provider,
        provider_endpoint=args.provider_endpoint,
        instrument=args.instrument,
        request_timeout_seconds=args.request_timeout_seconds,
        max_reconnect_attempts=args.max_reconnect_attempts,
        max_stale_seconds=args.max_stale_seconds,
        github_round_progress_comments=args.github_round_progress_comments,
        round_progress_comment_mode=args.round_progress_comment_mode,
        round_progress_comment_interval_rounds=(
            args.round_progress_comment_interval_rounds
        ),
        max_round_progress_comments=args.max_round_progress_comments,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
