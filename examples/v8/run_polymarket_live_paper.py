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
    settlement_mode: str = "resolved",
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
            settlement_mode=settlement_mode,  # type: ignore[arg-type]
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
    parser.add_argument("--settlement-mode", choices=("resolved", "delayed"), default="resolved")
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
        settlement_mode=args.settlement_mode,
        stop_requested=args.stop_requested,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
