"""Run the v8 24h-capable paper operator workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.paper import (  # noqa: E402
    PaperOperatorCLIError,
    PaperOperatorRunConfig,
    run_24h_paper_operator,
)

_MODE_MAP = {
    "dry-run": "dry_run",
    "gh-command": "gh_command",
    "direct-comment": "direct_comment",
}


def run_24h_paper_operator_cli(
    *,
    run_id: str,
    output_dir: Path | str,
    repo: str,
    issue_number: int,
    mode: str = "dry-run",
    duration_seconds: int | None = None,
    duration_hours: float | None = None,
    heartbeat_interval_seconds: int = 60,
    summary_interval_seconds: int = 300,
    feed_event_interval_seconds: int = 60,
    feed_mode: str = "deterministic-replay",
    provider: str = "coinbase_public_ticker",
    provider_endpoint: str = (
        "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
    ),
    instrument: str = "BTC-USD",
    request_timeout_seconds: float = 10.0,
    max_reconnect_attempts: int = 3,
    max_stale_seconds: float = 120.0,
    github_round_progress_comments: bool = False,
    round_progress_comment_mode: str | None = None,
    round_progress_comment_interval_rounds: int = 1,
    max_round_progress_comments: int = 0,
    overwrite_existing: bool = False,
    stop_after_events: int | None = None,
    inject_degradation: bool = False,
) -> dict[str, object]:
    """Run the operator workflow and return the final console summary."""

    resolved_duration_seconds = _resolve_duration_seconds(
        duration_seconds=duration_seconds,
        duration_hours=duration_hours,
    )
    result = run_24h_paper_operator(
        config=PaperOperatorRunConfig(
            run_id=run_id,
            output_dir=output_dir,
            repo_full_name=repo,
            issue_number=issue_number,
            post_mode=_MODE_MAP[mode],  # type: ignore[arg-type]
            duration_seconds=resolved_duration_seconds,
            feed_event_interval_seconds=feed_event_interval_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            summary_interval_seconds=summary_interval_seconds,
            feed_mode=feed_mode,  # type: ignore[arg-type]
            provider_name=provider,
            provider_endpoint=provider_endpoint,
            instrument_id=instrument,
            request_timeout_seconds=request_timeout_seconds,
            max_reconnect_attempts=max_reconnect_attempts,
            max_stale_seconds=max_stale_seconds,
            round_progress_comments_enabled=github_round_progress_comments,
            round_progress_comment_post_mode=None
            if round_progress_comment_mode is None
            else _MODE_MAP[round_progress_comment_mode],  # type: ignore[arg-type]
            round_progress_comment_interval_rounds=(
                round_progress_comment_interval_rounds
            ),
            max_round_progress_comments=max_round_progress_comments,
            overwrite_existing=overwrite_existing,
            stop_after_events=stop_after_events,
            inject_degradation=inject_degradation,
        )
    )
    return result.console_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=tuple(_MODE_MAP),
        default="dry-run",
        help="Delivery mode. direct-comment posts via gh and must be explicit.",
    )
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--duration-hours", type=float)
    parser.add_argument("--feed-event-interval-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=60)
    parser.add_argument("--summary-interval-seconds", type=int, default=300)
    parser.add_argument(
        "--feed-mode",
        choices=("deterministic-replay", "live-readonly"),
        default="deterministic-replay",
    )
    parser.add_argument("--provider", default="coinbase_public_ticker")
    parser.add_argument(
        "--provider-endpoint",
        default="https://api.exchange.coinbase.com/products/BTC-USD/ticker",
    )
    parser.add_argument("--instrument", default="BTC-USD")
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
        choices=tuple(_MODE_MAP),
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
    parser.add_argument("--stop-after-events", type=int)
    parser.add_argument(
        "--inject-degradation",
        action="store_true",
        help="Inject deterministic paper degradation for fail-closed rehearsal.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace an existing operator run directory for the same run id.",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_24h_paper_operator_cli(
            run_id=args.run_id,
            output_dir=args.output_dir,
            repo=args.repo,
            issue_number=args.issue_number,
            mode=args.mode,
            duration_seconds=args.duration_seconds,
            duration_hours=args.duration_hours,
            feed_event_interval_seconds=args.feed_event_interval_seconds,
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
            stop_after_events=args.stop_after_events,
            inject_degradation=args.inject_degradation,
            overwrite_existing=args.overwrite_existing,
        )
    except (PaperOperatorCLIError, ValueError, FileExistsError) as exc:
        print(json.dumps({"status": "failed_fail_closed", "error": str(exc)}))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_duration_seconds(
    *,
    duration_seconds: int | None,
    duration_hours: float | None,
) -> int:
    if duration_seconds is not None and duration_hours is not None:
        raise ValueError("set duration_seconds or duration_hours, not both")
    if duration_seconds is not None:
        return duration_seconds
    if duration_hours is not None:
        return int(duration_hours * 60 * 60)
    return 24 * 60 * 60


if __name__ == "__main__":
    raise SystemExit(main())
