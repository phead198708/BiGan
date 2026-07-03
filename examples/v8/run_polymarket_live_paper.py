"""Run a Polymarket BTC UP/DOWN live-data paper-only operator pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket import (  # noqa: E402
    V8_TRAINING_CORPUS_ROOT,
    PolymarketLivePaperConfig,
    run_polymarket_live_paper,
)
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS  # noqa: E402


def run_polymarket_live_paper_cli(
    *,
    run_id: str,
    output_dir: Path | str,
    repo: str = "phead198708/BiGan",
    issue_number: int = 134,
    mode: str = "dry-run",
    mock_live: bool = True,
    market_families: tuple[str, ...] = tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
    model_manifest: Path | str | None = None,
    model_path: Path | str | None = None,
    duration_seconds: int = 300,
    poll_interval_seconds: int = 5,
    summary_interval_seconds: int = 300,
    stream_observability: bool = False,
    status_interval_seconds: int = 15,
    heartbeat_interval_seconds: int = 60,
    flush_event_files: bool = False,
    settlement_mode: str = "resolved",
    settlement_wait_timeout_seconds: int = 600,
    settlement_poll_interval_seconds: int = 15,
    export_training_corpus: bool = False,
    training_corpus_root: Path | str = V8_TRAINING_CORPUS_ROOT,
    stop_requested: bool = False,
    overwrite_existing: bool = False,
) -> dict:
    result = run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id=run_id,
            output_dir=output_dir,
            repo_full_name=repo,
            issue_number=issue_number,
            mode=mode,  # type: ignore[arg-type]
            mock_live=mock_live,
            market_families=market_families,
            model_manifest=model_manifest,
            model_path=model_path,
            duration_seconds=duration_seconds,
            poll_interval_seconds=poll_interval_seconds,
            summary_interval_seconds=summary_interval_seconds,
            stream_observability=stream_observability,
            status_interval_seconds=status_interval_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            flush_event_files=flush_event_files,
            settlement_mode=settlement_mode,  # type: ignore[arg-type]
            settlement_wait_timeout_seconds=settlement_wait_timeout_seconds,
            settlement_poll_interval_seconds=settlement_poll_interval_seconds,
            export_training_corpus=export_training_corpus,
            training_corpus_root=training_corpus_root,
            stop_requested=stop_requested,
            overwrite_existing=overwrite_existing,
        )
    )
    manifest = result.operator_manifest
    return {
        "run_id": manifest["run_id"],
        "run_dir": str(result.run_dir),
        "operator_status": manifest["operator_status"],
        "operator_recommendation": manifest["operator_recommendation"],
        "critical_alert_count": manifest["critical_alert_count"],
        "prediction_count": manifest["prediction_count"],
        "decision_count": manifest["decision_count"],
        "trade_count": manifest["trade_count"],
        "resolved_market_count": manifest["resolved_market_count"],
        "unresolved_market_count": manifest["unresolved_market_count"],
        "settlement_wait_enabled": manifest["settlement_wait_enabled"],
        "settlement_wait_timeout_seconds": manifest["settlement_wait_timeout_seconds"],
        "settlement_wait_poll_count": manifest["settlement_wait_poll_count"],
        "settlement_wait_timed_out": manifest["settlement_wait_timed_out"],
        "export_training_corpus_enabled": manifest["export_training_corpus_enabled"],
        "exported_training_corpus_count": manifest["exported_training_corpus_count"],
        "exported_training_corpus_dirs": manifest["exported_training_corpus_dirs"],
        "training_corpus_root": manifest["training_corpus_root"],
        "total_polymarket_pnl": manifest["total_polymarket_pnl"],
        "live_polymarket_data": manifest["live_polymarket_data"],
        "live_binance_reference_data": manifest["live_binance_reference_data"],
        "deterministic_replay": manifest["deterministic_replay"],
        "paper_only": manifest["paper_only"],
        "capital_at_risk": manifest["capital_at_risk"],
        "polymarket_write_enabled": manifest["polymarket_write_enabled"],
        "wallet_signing_enabled": manifest["wallet_signing_enabled"],
        "operator_manifest_path": str(
            result.artifact_paths["polymarket_live_operator_manifest"]
        ),
        "github_comment_path": str(result.artifact_paths["github_paper_comment_md"]),
        "live_status_path": str(result.artifact_paths.get("live_status", "")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo", default="phead198708/BiGan")
    parser.add_argument("--issue-number", type=int, default=134)
    parser.add_argument("--mode", choices=("dry-run", "gh-command"), default="dry-run")
    parser.add_argument("--mock-live", action="store_true")
    parser.add_argument("--no-mock-live", action="store_true")
    parser.add_argument(
        "--market-family",
        action="append",
        choices=tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
        dest="market_families",
    )
    parser.add_argument("--model-manifest")
    parser.add_argument("--model-path")
    parser.add_argument("--duration-hours", type=float)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--summary-interval-seconds", type=int, default=300)
    parser.add_argument("--stream-observability", action="store_true")
    parser.add_argument("--status-interval-seconds", type=int, default=15)
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=60)
    parser.add_argument("--flush-event-files", action="store_true")
    parser.add_argument("--settlement-mode", choices=("resolved", "delayed"), default="resolved")
    parser.add_argument("--settlement-wait-timeout-seconds", type=int, default=600)
    parser.add_argument("--settlement-poll-interval-seconds", type=int, default=15)
    parser.add_argument(
        "--export-training-corpus",
        dest="export_training_corpus",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-export-training-corpus",
        dest="export_training_corpus",
        action="store_false",
    )
    parser.add_argument("--training-corpus-root", default=str(V8_TRAINING_CORPUS_ROOT))
    parser.add_argument("--stop-requested", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    duration_seconds = (
        int(args.duration_hours * 3600)
        if args.duration_hours is not None
        else args.duration_seconds
    )
    mock_live = True
    if args.no_mock_live:
        mock_live = False
    if args.mock_live:
        mock_live = True
    export_training_corpus = (
        not mock_live
        if args.export_training_corpus is None
        else bool(args.export_training_corpus)
    )
    summary = run_polymarket_live_paper_cli(
        run_id=args.run_id,
        output_dir=args.output_dir,
        repo=args.repo,
        issue_number=args.issue_number,
        mode=args.mode,
        mock_live=mock_live,
        market_families=tuple(args.market_families or BTC_UPDOWN_MARKET_HORIZONS_MS),
        model_manifest=args.model_manifest,
        model_path=args.model_path,
        duration_seconds=duration_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        summary_interval_seconds=args.summary_interval_seconds,
        stream_observability=args.stream_observability,
        status_interval_seconds=args.status_interval_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        flush_event_files=args.flush_event_files,
        settlement_mode=args.settlement_mode,
        settlement_wait_timeout_seconds=args.settlement_wait_timeout_seconds,
        settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
        export_training_corpus=export_training_corpus,
        training_corpus_root=args.training_corpus_root,
        stop_requested=args.stop_requested,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
