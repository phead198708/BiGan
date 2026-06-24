"""Run non-blocking round capture with asynchronous settlement finalization."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket import (  # noqa: E402
    DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    V8_TRAINING_CORPUS_ROOT,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
    capture_polymarket_pending_round,
    finalize_polymarket_pending_round,
)
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS  # noqa: E402


def run_polymarket_async_round_collector_cli(
    *,
    batch_id: str,
    output_dir: Path | str,
    round_count: int,
    market_family: str = "btc_updown_5m",
    public_provider_timeout_seconds: float = 330.0,
    public_provider_http_timeout_seconds: float = 15.0,
    orderbook_snapshot_interval_seconds: float = 1.0,
    settlement_poll_interval_seconds: float = 15.0,
    settlement_grace_seconds: float = 0.0,
    training_corpus_root: Path | str = V8_TRAINING_CORPUS_ROOT,
    clob_ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    max_round_start_lag_seconds: float = 30.0,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    if round_count <= 0:
        raise ValueError("round_count must be positive")
    if public_provider_timeout_seconds <= 0:
        raise ValueError("public_provider_timeout_seconds must be positive")
    if public_provider_http_timeout_seconds <= 0:
        raise ValueError("public_provider_http_timeout_seconds must be positive")
    if settlement_poll_interval_seconds <= 0:
        raise ValueError("settlement_poll_interval_seconds must be positive")
    if settlement_grace_seconds < 0:
        raise ValueError("settlement_grace_seconds must be non-negative")
    if max_round_start_lag_seconds < 0:
        raise ValueError("max_round_start_lag_seconds must be non-negative")

    root = Path(output_dir).expanduser().resolve()
    batch_dir = root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    lock = threading.Lock()
    captures: list[dict[str, Any]] = []
    finalizations: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def finalizer_loop() -> None:
        while not stop_event.is_set():
            _finalize_pending_once(
                output_dir=root,
                destination_root=Path(training_corpus_root),
                clob_ws_url=clob_ws_url,
                overwrite_existing=overwrite_existing,
                batch_id_prefix=batch_id,
                finalizations=finalizations,
                errors=errors,
                lock=lock,
            )
            stop_event.wait(settlement_poll_interval_seconds)

    finalizer = threading.Thread(
        target=finalizer_loop,
        name=f"{batch_id}-settlement-finalizer",
        daemon=True,
    )
    finalizer.start()

    try:
        for index in range(1, round_count + 1):
            _sleep_until_round_start_window(
                market_family=market_family,
                max_round_start_lag_seconds=max_round_start_lag_seconds,
            )
            run_id = f"{batch_id}-round{index:02d}-{_utc_stamp()}"
            provider = PolymarketPublicHTTPRealCorpusProvider(
                max_markets=1,
                clob_ws_url=clob_ws_url,
                timeout_seconds=public_provider_timeout_seconds,
                http_timeout_seconds=public_provider_http_timeout_seconds,
                orderbook_snapshot_interval_seconds=orderbook_snapshot_interval_seconds,
            )
            config = PolymarketRealCorpusRecorderConfig(
                run_id=run_id,
                output_dir=root,
                market_families=(market_family,),
                mock_public_data=False,
                overwrite_existing=overwrite_existing,
            )
            capture = capture_polymarket_pending_round(config, public_provider=provider)
            with lock:
                captures.append(
                    {
                        "run_id": run_id,
                        "run_dir": str(capture.run_dir),
                        "capture_status": capture.report["capture_status"],
                        "pending_resolution": capture.report["pending_resolution"],
                        "raw_polymarket_market_count": capture.report[
                            "raw_polymarket_market_count"
                        ],
                        "raw_orderbook_row_count": capture.report["raw_orderbook_row_count"],
                        "raw_trade_row_count": capture.report["raw_trade_row_count"],
                        "raw_btc_candle_row_count": capture.report["raw_btc_candle_row_count"],
                        "reject_reason_counts": capture.report["reject_reason_counts"],
                    }
                )
            _write_json(batch_dir / "batch_progress.json", _summary(batch_id, captures, finalizations, errors))
        if settlement_grace_seconds:
            deadline = time.monotonic() + settlement_grace_seconds
            while time.monotonic() < deadline:
                time.sleep(min(settlement_poll_interval_seconds, deadline - time.monotonic()))
    finally:
        stop_event.set()
        finalizer.join(timeout=max(1.0, settlement_poll_interval_seconds))

    _finalize_pending_once(
        output_dir=root,
        destination_root=Path(training_corpus_root),
        clob_ws_url=clob_ws_url,
        overwrite_existing=overwrite_existing,
        batch_id_prefix=batch_id,
        finalizations=finalizations,
        errors=errors,
        lock=lock,
    )
    summary = _summary(batch_id, captures, finalizations, errors)
    summary_path = batch_dir / "batch_summary.json"
    _write_json(summary_path, summary)
    summary["batch_summary_path"] = str(summary_path)
    return summary


def run_polymarket_async_finalizer_cli(
    *,
    batch_id: str,
    output_dir: Path | str,
    settlement_poll_interval_seconds: float = 15.0,
    settlement_grace_seconds: float = 0.0,
    training_corpus_root: Path | str = V8_TRAINING_CORPUS_ROOT,
    clob_ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    if settlement_poll_interval_seconds <= 0:
        raise ValueError("settlement_poll_interval_seconds must be positive")
    if settlement_grace_seconds < 0:
        raise ValueError("settlement_grace_seconds must be non-negative")
    root = Path(output_dir).expanduser().resolve()
    batch_dir = root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    finalizations: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    deadline = time.monotonic() + settlement_grace_seconds
    while True:
        _finalize_pending_once(
            output_dir=root,
            destination_root=Path(training_corpus_root),
            clob_ws_url=clob_ws_url,
            overwrite_existing=overwrite_existing,
            batch_id_prefix=None,
            finalizations=finalizations,
            errors=errors,
            lock=lock,
        )
        if settlement_grace_seconds <= 0 or time.monotonic() >= deadline:
            break
        time.sleep(min(settlement_poll_interval_seconds, deadline - time.monotonic()))
    summary = {
        "batch_id": batch_id,
        "paper_only": True,
        "capital_at_risk": False,
        "finalize_only": True,
        "finalization_attempt_count": len(finalizations),
        "exported_round_count": sum(
            1 for item in finalizations if item.get("finalization_status") == "exported"
        ),
        "pending_resolution_count": sum(
            1
            for item in finalizations
            if item.get("finalization_status") == "pending_resolution"
        ),
        "error_count": len(errors),
        "finalizations": finalizations,
        "errors": errors,
    }
    summary_path = batch_dir / "finalizer_summary.json"
    _write_json(summary_path, summary)
    summary["finalizer_summary_path"] = str(summary_path)
    return summary


def _finalize_pending_once(
    *,
    output_dir: Path,
    destination_root: Path,
    clob_ws_url: str,
    overwrite_existing: bool,
    batch_id_prefix: str | None,
    finalizations: list[dict[str, Any]],
    errors: list[dict[str, str]],
    lock: threading.Lock,
) -> None:
    seen_exported = {
        item["run_id"]
        for item in finalizations
        if item.get("finalization_status") == "exported"
    }
    for manifest_path in sorted(output_dir.glob("*/pending_round_capture_manifest.json")):
        run_dir = manifest_path.parent
        if batch_id_prefix is not None and not run_dir.name.startswith(
            f"{batch_id_prefix}-round"
        ):
            continue
        if run_dir.name in seen_exported:
            continue
        capture_manifest = _read_json(manifest_path)
        if not capture_manifest.get("pending_resolution"):
            continue
        finalization_report_path = run_dir / "pending_round_finalization_report.json"
        if finalization_report_path.exists():
            previous = _read_json(finalization_report_path)
            if previous.get("finalization_status") == "exported":
                continue
        try:
            provider = PolymarketPublicHTTPRealCorpusProvider(
                max_markets=1,
                clob_ws_url=clob_ws_url,
                timeout_seconds=15.0,
            )
            result = finalize_polymarket_pending_round(
                run_dir,
                public_provider=provider,
                destination_root=destination_root,
                overwrite_existing=overwrite_existing,
            )
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append({"run_dir": str(run_dir), "error": str(exc)})
            continue
        with lock:
            _upsert_by_run_id(
                finalizations,
                {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "finalization_status": result.report["finalization_status"],
                    "pending_resolution": result.report["pending_resolution"],
                    "training_eligible": result.report["training_eligible"],
                    "exported_training_corpus_dir": result.report[
                        "exported_training_corpus_dir"
                    ],
                    "raw_resolution_count": result.report["raw_resolution_count"],
                    "reject_reason_counts": result.report["reject_reason_counts"],
                },
            )


def _summary(
    batch_id: str,
    captures: list[dict[str, Any]],
    finalizations: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    exported = [
        item for item in finalizations if item.get("finalization_status") == "exported"
    ]
    pending = [
        item
        for item in finalizations
        if item.get("finalization_status") == "pending_resolution"
    ]
    return {
        "batch_id": batch_id,
        "paper_only": True,
        "capital_at_risk": False,
        "capture_count": len(captures),
        "capture_pending_resolution_count": sum(
            1 for item in captures if item.get("pending_resolution") is True
        ),
        "finalization_attempt_count": len(finalizations),
        "exported_round_count": len(exported),
        "pending_resolution_count": len(pending),
        "error_count": len(errors),
        "captures": captures,
        "finalizations": finalizations,
        "errors": errors,
    }


def _upsert_by_run_id(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    run_id = item["run_id"]
    for index, existing in enumerate(items):
        if existing.get("run_id") == run_id:
            items[index] = item
            return
    items.append(item)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sleep_until_round_start_window(
    *,
    market_family: str,
    max_round_start_lag_seconds: float,
) -> None:
    sleep_seconds = _round_start_alignment_sleep_seconds(
        market_family=market_family,
        max_round_start_lag_seconds=max_round_start_lag_seconds,
        now_epoch_seconds=time.time(),
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def _round_start_alignment_sleep_seconds(
    *,
    market_family: str,
    max_round_start_lag_seconds: float,
    now_epoch_seconds: float,
) -> float:
    horizon_seconds = BTC_UPDOWN_MARKET_HORIZONS_MS[market_family] / 1000.0
    elapsed = now_epoch_seconds % horizon_seconds
    if elapsed <= max_round_start_lag_seconds:
        return 0.0
    return horizon_seconds - elapsed + 1.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=f"async-round-collector-{_utc_stamp()}")
    parser.add_argument("--output-dir", default="examples/v8/operator_runs")
    parser.add_argument("--round-count", type=int, default=1)
    parser.add_argument(
        "--market-family",
        choices=tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
        default="btc_updown_5m",
    )
    parser.add_argument("--public-provider-timeout-seconds", type=float, default=330.0)
    parser.add_argument("--public-provider-http-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--orderbook-snapshot-interval-seconds", type=float, default=1.0)
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--settlement-grace-seconds", type=float, default=0.0)
    parser.add_argument("--training-corpus-root", default=str(V8_TRAINING_CORPUS_ROOT))
    parser.add_argument("--clob-ws-url", default=DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL)
    parser.add_argument("--max-round-start-lag-seconds", type=float, default=30.0)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only scan pending round captures and try settlement finalization.",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize_only:
        summary = run_polymarket_async_finalizer_cli(
            batch_id=args.batch_id,
            output_dir=args.output_dir,
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            settlement_grace_seconds=args.settlement_grace_seconds,
            training_corpus_root=args.training_corpus_root,
            clob_ws_url=args.clob_ws_url,
            overwrite_existing=args.overwrite_existing,
        )
    else:
        summary = run_polymarket_async_round_collector_cli(
            batch_id=args.batch_id,
            output_dir=args.output_dir,
            round_count=args.round_count,
            market_family=args.market_family,
            public_provider_timeout_seconds=args.public_provider_timeout_seconds,
            public_provider_http_timeout_seconds=args.public_provider_http_timeout_seconds,
            orderbook_snapshot_interval_seconds=args.orderbook_snapshot_interval_seconds,
            settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
            settlement_grace_seconds=args.settlement_grace_seconds,
            training_corpus_root=args.training_corpus_root,
            clob_ws_url=args.clob_ws_url,
            overwrite_existing=args.overwrite_existing,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
