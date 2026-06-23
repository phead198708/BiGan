"""Record Polymarket BTC UP/DOWN facts into the v8 Phase 2 raw corpus contract."""

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
    DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    V8_TRAINING_CORPUS_ROOT,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
    export_trainable_corpus,
    record_polymarket_real_corpus,
    round_corpus_id_from_corpus_dir,
)
from bigan.v8.polymarket.corpus import (  # noqa: E402
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    DEFAULT_CORPUS_CREATED_AT,
)
from bigan.v8.polymarket.recorder import (  # noqa: E402
    DEFAULT_RECORDER_ENDED_AT,
    DEFAULT_RECORDER_STARTED_AT,
)


def run_record_polymarket_real_corpus_cli(
    *,
    run_id: str,
    output_dir: Path | str,
    market_families: tuple[str, ...] = tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
    created_at: str = DEFAULT_CORPUS_CREATED_AT,
    started_at: str = DEFAULT_RECORDER_STARTED_AT,
    ended_at: str = DEFAULT_RECORDER_ENDED_AT,
    build_phase2_corpus: bool = True,
    mock_public_data: bool = True,
    use_public_http_provider: bool = False,
    market_slugs: tuple[str, ...] = (),
    max_markets: int = 3,
    clob_ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
    public_provider_timeout_seconds: float = 15.0,
    orderbook_snapshot_interval_seconds: float = 1.0,
    seed_rest_orderbooks_before_stream: bool = True,
    use_rest_orderbooks: bool = False,
    export_training_corpus: bool = True,
    training_corpus_root: Path | str = V8_TRAINING_CORPUS_ROOT,
    overwrite_existing: bool = False,
) -> dict:
    public_provider = (
        PolymarketPublicHTTPRealCorpusProvider(
            market_slugs=market_slugs,
            max_markets=max_markets,
            clob_ws_url=clob_ws_url,
            timeout_seconds=public_provider_timeout_seconds,
            orderbook_snapshot_interval_seconds=orderbook_snapshot_interval_seconds,
            seed_rest_orderbooks_before_stream=seed_rest_orderbooks_before_stream,
            use_rest_orderbooks=use_rest_orderbooks,
        )
        if use_public_http_provider
        else None
    )
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id=run_id,
            output_dir=output_dir,
            market_families=market_families,
            created_at=created_at,
            started_at=started_at,
            ended_at=ended_at,
            build_phase2_corpus=build_phase2_corpus,
            mock_public_data=mock_public_data,
            overwrite_existing=overwrite_existing,
        ),
        public_provider=public_provider,
    )
    report = result.report
    exported_training_corpus_dir = None
    exported_training_corpus_id = None
    exported_training_round_slug = None
    if (
        export_training_corpus
        and report["real_historical_training_eligible"] is True
        and result.corpus_dir is not None
    ):
        exported_training_round_slug = round_corpus_id_from_corpus_dir(result.corpus_dir)
        exported_training_corpus_id = exported_training_round_slug
        exported_training_corpus_dir = export_trainable_corpus(
            corpus_dir=result.corpus_dir,
            corpus_id=exported_training_corpus_id,
            destination_root=training_corpus_root,
            overwrite_existing=overwrite_existing,
            provenance={
                "source": "polymarket_real_corpus_recorder",
                "run_id": run_id,
                "round_slug": exported_training_round_slug,
                "corpus_id": exported_training_corpus_id,
                "round_scoped_export": True,
                "recorder_manifest_path": str(
                    result.artifact_paths["real_corpus_recorder_manifest"]
                ),
                "recorder_report_path": str(result.artifact_paths["real_corpus_recorder_report"]),
                "phase2_corpus_manifest_sha256": report["phase2_corpus_manifest_sha256"],
                "real_historical_corpus_used": report["real_historical_corpus_used"],
                "manual_live_evidence_eligible": report["manual_live_evidence_eligible"],
                "mock_public_data_used": report["mock_public_data_used"],
                "synthetic_public_data_used": report["synthetic_public_data_used"],
                "synthetic_corpus_used": report["synthetic_corpus_used"],
            },
        )
    return {
        "run_id": report["run_id"],
        "run_dir": str(result.run_dir),
        "raw_dir": str(result.raw_dir),
        "corpus_dir": None if result.corpus_dir is None else str(result.corpus_dir),
        "exported_training_corpus_dir": (
            None if exported_training_corpus_dir is None else str(exported_training_corpus_dir)
        ),
        "exported_training_corpus_id": exported_training_corpus_id,
        "exported_training_round_slug": exported_training_round_slug,
        "training_corpus_root": str(training_corpus_root),
        "recorder_manifest_path": str(result.artifact_paths["real_corpus_recorder_manifest"]),
        "recorder_report_path": str(result.artifact_paths["real_corpus_recorder_report"]),
        "rejected_rows_path": str(result.artifact_paths["real_corpus_rejected_rows"]),
        "raw_polymarket_market_count": report["raw_polymarket_market_count"],
        "raw_orderbook_row_count": report["raw_orderbook_row_count"],
        "raw_trade_row_count": report["raw_trade_row_count"],
        "raw_btc_candle_row_count": report["raw_btc_candle_row_count"],
        "raw_resolution_count": report["raw_resolution_count"],
        "rejected_row_count": report["rejected_row_count"],
        "reject_reason_counts": report["reject_reason_counts"],
        "training_eligible": report["training_eligible"],
        "phase2_corpus_build_eligible": report["phase2_corpus_build_eligible"],
        "real_historical_training_eligible": report["real_historical_training_eligible"],
        "manual_live_evidence_eligible": report["manual_live_evidence_eligible"],
        "phase2_corpus_built": report["phase2_corpus_built"],
        "phase2_corpus_manifest_sha256": report["phase2_corpus_manifest_sha256"],
        "mock_public_data_used": report["mock_public_data_used"],
        "synthetic_public_data_used": report["synthetic_public_data_used"],
        "synthetic_corpus_used": report["synthetic_corpus_used"],
        "real_historical_corpus_used": report["real_historical_corpus_used"],
        "fixture_corpus_used": report["fixture_corpus_used"],
        "requested_live_public_collection": report["requested_live_public_collection"],
        "public_collection_status": report["public_collection_status"],
        "public_collection_reason_codes": report["public_collection_reason_codes"],
        "live_polymarket_data_read": report["live_polymarket_data_read"],
        "live_btc_reference_data_read": report["live_btc_reference_data_read"],
        "live_polymarket_data": report["live_polymarket_data"],
        "live_btc_reference_data": report["live_btc_reference_data"],
        "paper_only": report["paper_only"],
        "capital_at_risk": report["capital_at_risk"],
        "polymarket_write_enabled": report["polymarket_write_enabled"],
        "wallet_signing_enabled": report["wallet_signing_enabled"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--created-at", default=DEFAULT_CORPUS_CREATED_AT)
    parser.add_argument("--started-at", default=DEFAULT_RECORDER_STARTED_AT)
    parser.add_argument("--ended-at", default=DEFAULT_RECORDER_ENDED_AT)
    parser.add_argument(
        "--market-family",
        action="append",
        choices=tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
        dest="market_families",
    )
    parser.add_argument("--no-build-phase2-corpus", action="store_true")
    parser.add_argument("--mock-public-data", action="store_true")
    parser.add_argument("--no-mock-public-data", action="store_true")
    parser.add_argument(
        "--use-public-http-provider",
        action="store_true",
        help="Use read-only public Gamma/Data/CLOB WS/Binance provider.",
    )
    parser.add_argument(
        "--use-rest-orderbooks",
        action="store_true",
        help="Use legacy CLOB REST /book for orderbooks instead of the default websocket source.",
    )
    parser.add_argument(
        "--clob-ws-url",
        default=DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
        help="Read-only Polymarket CLOB market-channel websocket URL.",
    )
    parser.add_argument(
        "--public-provider-timeout-seconds",
        type=float,
        default=15.0,
        help="Timeout for public HTTP reads and websocket orderbook collection.",
    )
    parser.add_argument(
        "--orderbook-snapshot-interval-seconds",
        type=float,
        default=1.0,
        help="Interval for read-only CLOB websocket orderbook snapshots.",
    )
    parser.add_argument(
        "--no-rest-orderbook-seed",
        action="store_true",
        help="Disable the initial read-only REST /book seed before websocket snapshots.",
    )
    parser.add_argument(
        "--market-slug",
        action="append",
        dest="market_slugs",
        help="Specific BTC UP/DOWN slug to record, repeatable.",
    )
    parser.add_argument("--max-markets", type=int, default=3)
    parser.add_argument(
        "--training-corpus-root",
        default=str(V8_TRAINING_CORPUS_ROOT),
        help=(
            "External root for validated direct-training corpora only. "
            "Defaults to /Volumes/PHILIPS/v8."
        ),
    )
    parser.add_argument("--no-export-training-corpus", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)

    mock_public_data = True
    if args.no_mock_public_data:
        mock_public_data = False
    if args.use_public_http_provider:
        mock_public_data = False
    if args.mock_public_data:
        mock_public_data = True
    summary = run_record_polymarket_real_corpus_cli(
        run_id=args.run_id,
        output_dir=args.output_dir,
        market_families=tuple(args.market_families or BTC_UPDOWN_MARKET_HORIZONS_MS),
        created_at=args.created_at,
        started_at=args.started_at,
        ended_at=args.ended_at,
        build_phase2_corpus=not args.no_build_phase2_corpus,
        mock_public_data=mock_public_data,
        use_public_http_provider=args.use_public_http_provider,
        market_slugs=tuple(args.market_slugs or ()),
        max_markets=args.max_markets,
        clob_ws_url=args.clob_ws_url,
        public_provider_timeout_seconds=args.public_provider_timeout_seconds,
        orderbook_snapshot_interval_seconds=args.orderbook_snapshot_interval_seconds,
        seed_rest_orderbooks_before_stream=not args.no_rest_orderbook_seed,
        use_rest_orderbooks=args.use_rest_orderbooks,
        export_training_corpus=not args.no_export_training_corpus,
        training_corpus_root=args.training_corpus_root,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
