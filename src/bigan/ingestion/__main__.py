"""CLI entry point: ``python -m bigan.ingestion`` or ``bigan-ingest``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path

import structlog
import typer
from prometheus_client import start_http_server

from bigan.canonical.etl import run_etl_batch
from bigan.canonical.query import open_warehouse, warehouse_summary
from bigan.canonical.symbols import SymbolMapper

from .backfill import BackfillService, GapWindow
from .clob_rest import PolymarketRestClient
from .config import IngestionSettings
from .metrics import REGISTRY
from .price_readers import (
    ChainlinkOracleReader,
    ChainlinkReaderConfig,
    CoinbaseTickerReader,
    KrakenTickerReader,
    WarehousePriceSink,
    WsPriceReaderConfig,
)
from .runner import IngestionRunner
from .sink import NdjsonGzipSink

app = typer.Typer(add_completion=False, help="BiGan ingestion service")
SYMBOL_MAPPING_PATH_OPTION = typer.Option(
    None,
    help="Optional CSV, JSON, JSONL, or directory of symbol_mapping rows.",
)
TIMESTAMP_FUTURE_GRACE_SECONDS_OPTION = typer.Option(
    None,
    help="Override BIGAN_TIMESTAMP_FUTURE_GRACE_SECONDS for this ETL run.",
)
TIMESTAMP_STALE_THRESHOLD_SECONDS_OPTION = typer.Option(
    None,
    help="Override BIGAN_TIMESTAMP_STALE_THRESHOLD_SECONDS for this ETL run.",
)


def _configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=log_level,
        stream=sys.stderr,
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
    )


@app.command("serve")
def serve() -> None:
    """Run the WebSocket ingestion service (long-running)."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    runner = IngestionRunner(settings)
    asyncio.run(runner.serve())


@app.command("smoke")
def smoke(seconds: int = typer.Option(30, help="How long to run before exiting.")) -> None:
    """Run for ``seconds`` then exit cleanly. Used for live smoke tests."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    runner = IngestionRunner(settings)

    async def main() -> None:
        task = asyncio.create_task(runner.serve())
        await asyncio.sleep(seconds)
        runner.stop()
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(main())


@app.command("reference-prices")
def reference_prices(
    symbol_mapping_path: Path | None = SYMBOL_MAPPING_PATH_OPTION,
    max_rows_per_partition: int = typer.Option(
        1,
        help="Flush reference-price warehouse partitions after this many rows.",
    ),
) -> None:
    """Run Coinbase, Kraken, and Chainlink BTC/USD reference-price readers."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    if not settings.chainlink_rpc_url:
        raise typer.BadParameter(
            "BIGAN_CHAINLINK_RPC_URL must be set to run the Chainlink reader"
        )

    async def main() -> None:
        if settings.metrics_enabled:
            start_http_server(settings.metrics_port, registry=REGISTRY)
        symbol_mapper = (
            SymbolMapper.from_path(symbol_mapping_path)
            if symbol_mapping_path is not None
            else None
        )
        sink = WarehousePriceSink(
            settings.warehouse_dir,
            symbol_mapper=symbol_mapper,
            max_rows_per_partition=max_rows_per_partition,
        )
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(
                    CoinbaseTickerReader(
                        WsPriceReaderConfig(
                            url=settings.coinbase_ws_url,
                            symbol=settings.coinbase_product_id,
                            reconnect_min_seconds=settings.price_reader_reconnect_min_seconds,
                            reconnect_max_seconds=settings.price_reader_reconnect_max_seconds,
                        ),
                        sink,
                    ).run(),
                    name="coinbase-ticker",
                )
                tg.create_task(
                    KrakenTickerReader(
                        WsPriceReaderConfig(
                            url=settings.kraken_ws_url,
                            symbol=settings.kraken_symbol,
                            reconnect_min_seconds=settings.price_reader_reconnect_min_seconds,
                            reconnect_max_seconds=settings.price_reader_reconnect_max_seconds,
                        ),
                        sink,
                    ).run(),
                    name="kraken-ticker",
                )
                tg.create_task(
                    ChainlinkOracleReader(
                        ChainlinkReaderConfig(
                            rpc_url=settings.chainlink_rpc_url,
                            feed_address=settings.chainlink_feed_address,
                            symbol=settings.chainlink_symbol,
                            poll_interval_seconds=settings.chainlink_poll_interval_seconds,
                            request_timeout_seconds=settings.chainlink_request_timeout_seconds,
                        ),
                        sink,
                    ).run(),
                    name="chainlink-oracle",
                )
        finally:
            await sink.close()

    asyncio.run(main())


@app.command("etl-batch")
def etl_batch(
    lag_seconds: float = typer.Option(
        60.0, help="Skip NDJSON.gz files whose mtime is within this many seconds."
    ),
    max_rows_per_partition: int = typer.Option(
        50_000, help="Flush a partition buffer when it exceeds this size."
    ),
    symbol_mapping_path: Path | None = SYMBOL_MAPPING_PATH_OPTION,
    timestamp_future_grace_seconds: float | None = TIMESTAMP_FUTURE_GRACE_SECONDS_OPTION,
    timestamp_stale_threshold_seconds: float | None = TIMESTAMP_STALE_THRESHOLD_SECONDS_OPTION,
) -> None:
    """Convert raw NDJSON archive into the canonical Parquet warehouse."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_etl_batch(
        raw_dir=settings.raw_dir,
        warehouse_dir=settings.warehouse_dir,
        lag_seconds=lag_seconds,
        max_rows_per_partition=max_rows_per_partition,
        symbol_mapping_path=symbol_mapping_path,
        timestamp_future_grace_seconds=(
            settings.timestamp_future_grace_seconds
            if timestamp_future_grace_seconds is None
            else timestamp_future_grace_seconds
        ),
        timestamp_stale_threshold_seconds=(
            settings.timestamp_stale_threshold_seconds
            if timestamp_stale_threshold_seconds is None
            else timestamp_stale_threshold_seconds
        ),
    )
    typer.echo(
        json.dumps(
            {
                "files_processed": report.files_processed,
                "records_read": report.records_read,
                "rows_per_table": report.rows_per_table,
                "quarantined_by_rule": report.quarantined_by_rule,
                "quarantined_total": report.quarantined_total,
                "cross_batch_duplicates_skipped": report.cross_batch_duplicates_skipped,
            },
            indent=2,
        )
    )


@app.command("warehouse-stats")
def warehouse_stats() -> None:
    """Print row counts for each canonical table via DuckDB."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    summary = warehouse_summary(settings.warehouse_dir)
    typer.echo(json.dumps(summary, indent=2))


@app.command("quarantine-report")
def quarantine_report(
    limit: int = typer.Option(50, help="Max rows of detail to display."),
) -> None:
    """Summarise the quarantine table: counts by rule + recent samples."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    out: dict = {"by_rule": {}, "by_target_table": {}, "samples": []}
    with open_warehouse(settings.warehouse_dir) as conn:
        try:
            rows = conn.execute(
                "SELECT rule, COUNT(*) AS n FROM quarantine GROUP BY rule ORDER BY n DESC"
            ).fetchall()
            out["by_rule"] = {r[0]: r[1] for r in rows}

            rows = conn.execute(
                "SELECT target_table, COUNT(*) AS n FROM quarantine "
                "GROUP BY target_table ORDER BY n DESC"
            ).fetchall()
            out["by_target_table"] = {r[0]: r[1] for r in rows}

            samples = conn.execute(
                "SELECT ts, source, source_symbol, target_table, rule, detail "
                "FROM quarantine ORDER BY ts DESC LIMIT ?",
                [limit],
            ).fetchall()
            out["samples"] = [
                {
                    "ts": r[0],
                    "source": r[1],
                    "source_symbol": r[2],
                    "target_table": r[3],
                    "rule": r[4],
                    "detail": r[5],
                }
                for r in samples
            ]

            try:
                dupes = conn.execute(
                    "SELECT COUNT(*) - COUNT(DISTINCT trade_id) AS duplicate_rows "
                    "FROM raw_trades"
                ).fetchone()
                out["raw_trade_duplicate_rows"] = int(dupes[0])
            except Exception:  # noqa: BLE001
                out["raw_trade_duplicate_rows"] = 0
        except Exception:  # noqa: BLE001
            # Empty warehouse / no quarantine partition yet.
            pass
    typer.echo(json.dumps(out, indent=2))


@app.command("backfill")
def backfill(
    asset_id: str = typer.Option(..., help="CLOB token id (asset_id) to backfill."),
    market: str = typer.Option(
        ..., help="Polymarket condition_id (market hash) for trade lookup."
    ),
    since_ms: int = typer.Option(..., help="Gap start in epoch ms (UTC)."),
    until_ms: int = typer.Option(..., help="Gap end in epoch ms (UTC)."),
) -> None:
    """Manually run a REST backfill for a known [since_ms, until_ms] gap.

    Synthesised NDJSON records are written into the same raw sink the
    live WS pipeline uses. The next ETL run will pick them up.
    """
    settings = IngestionSettings()
    _configure_logging(settings.log_level)

    async def _run() -> dict:
        sink = NdjsonGzipSink(
            settings.raw_dir,
            flush_interval_seconds=settings.sink_flush_interval_seconds,
            max_buffer_records=settings.sink_max_buffer_records,
        )
        await sink.start_background_flusher()
        try:
            async with PolymarketRestClient(
                settings.clob_rest_url,
                timeout_seconds=settings.backfill_rest_timeout_seconds,
            ) as rest:
                async def resolver(_: str) -> str:
                    return market

                service = BackfillService(rest, sink, resolver)
                report = await service.handle_gap(
                    GapWindow(
                        asset_id=asset_id,
                        gap_start_ms=since_ms,
                        gap_end_ms=until_ms,
                    )
                )
                return {
                    "asset_id": report.asset_id,
                    "market": report.market,
                    "gap_start_ms": report.gap_start_ms,
                    "gap_end_ms": report.gap_end_ms,
                    "trades_replayed": report.trades_replayed,
                    "orderbook_replayed": report.orderbook_replayed,
                    "errors": report.errors,
                    "total_records": report.total_records,
                }
        finally:
            await sink.close()

    result = asyncio.run(_run())
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
