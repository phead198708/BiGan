"""CLI entry point: ``python -m bigan.ingestion`` or ``bigan-ingest``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import typer
from prometheus_client import start_http_server
from typer import Exit

from bigan.backtest import (
    load_backtest_config,
    run_oracle_label_sanity_backtest,
    run_prediction_threshold_backtest,
)
from bigan.canonical.etl import run_etl_batch
from bigan.canonical.query import open_warehouse, warehouse_summary
from bigan.canonical.symbols import SymbolMapper
from bigan.features import run_feature_batch, run_feature_quality_sql_checks
from bigan.labels import run_label_batch
from bigan.modeling import (
    BootstrapCandidateInput,
    LogisticBaselineConfig,
    SplitConfig,
    XGBoostV1Config,
    assemble_training_dataset,
    evaluate_bootstrap_champion,
    evaluate_model_promotion,
    fit_probability_calibration,
    run_prediction_batch,
    train_logistic_baseline,
    train_xgboost_v1,
    train_xgboost_v2,
)

from .backfill import BackfillService, GapWindow
from .clob_rest import PolymarketRestClient
from .config import IngestionSettings
from .gamma_client import GammaClient
from .market_compare import compare_market_coverage
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
from .soak import (
    SoakThresholds,
    finalize_soak_rollup,
    read_soak_samples,
    record_soak_samples,
    summarize_soak,
    write_soak_summary,
)

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
SOAK_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/soak"),
    help="Directory for soak sample NDJSON and summary JSON evidence.",
)
SOAK_SAMPLES_PATH_OPTION = typer.Option(
    ...,
    help="Soak sample NDJSON emitted by soak.",
)
SOAK_RAW_DIR_OPTION = typer.Option(
    None,
    help="Raw NDJSON directory. Defaults to BIGAN_DATA_DIR/BIGAN_RAW_SUBDIR.",
)
SOAK_ROLLUP_DIR_OPTION = typer.Option(
    None,
    help="Rollup Parquet directory. Defaults to BIGAN_DATA_DIR/BIGAN_ROLLUP_SUBDIR.",
)
SOAK_SUMMARY_PATH_OPTION = typer.Option(
    None,
    help="Optional path to write the JSON summary.",
)
BACKTEST_CONFIG_PATH_ARGUMENT = typer.Argument(
    ...,
    help="Backtest YAML or JSON config path.",
)
TRAINING_DATASET_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Directory for train.parquet, val.parquet, test.parquet, and manifest.json.",
)
LOGISTIC_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory produced by training-dataset-v1.",
)
LOGISTIC_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/logreg-baseline-v1"),
    help="Directory for logistic baseline artifacts.",
)
XGBOOST_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory produced by training-dataset-v1.",
)
XGBOOST_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1"),
    help="Directory for XGBoost-v1 artifacts.",
)
XGBOOST_V2_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v2"),
    help="Directory for XGBoost-v2 artifacts.",
)
CALIBRATION_MODEL_PATH_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1/model.json"),
    help="Path to a saved model.json produced by xgboost-v1.",
)
CALIBRATION_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory with validation split.",
)
CALIBRATION_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1-calibration"),
    help="Directory for calibration artifacts.",
)
PROMOTION_BASELINE_DIR_OPTION = typer.Option(
    Path("data/model-runs/logreg-baseline-v1"),
    help="Baseline model run directory.",
)
PROMOTION_CANDIDATE_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1"),
    help="Candidate model run directory.",
)
PROMOTION_CALIBRATION_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1-calibration"),
    help="Calibration artifact directory.",
)
PROMOTION_BACKTEST_SUMMARY_OPTION = typer.Option(
    Path("data/backtests/summary.json"),
    help="Threshold-backtest summary JSON.",
)
PROMOTION_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/promotion-v1"),
    help="Directory for promotion report artifacts.",
)
BOOTSTRAP_BASELINE_BACKTEST_SUMMARY_OPTION = typer.Option(
    None,
    help="Optional baseline threshold-backtest summary JSON.",
)
BOOTSTRAP_SERVING_READINESS_PATH_OPTION = typer.Option(
    None,
    help="Optional serving readiness JSON with latency/error evidence.",
)
BOOTSTRAP_FEATURE_SCHEMA_PATH_OPTION = typer.Option(
    None,
    help="Optional candidate feature_schema.json path; defaults to candidate_dir/feature_schema.json.",
)
BOOTSTRAP_ROLLBACK_RUNBOOK_PATH_OPTION = typer.Option(
    Path("docs/runbooks/model_rollback.md"),
    help="Rollback/fallback runbook path.",
)
BOOTSTRAP_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/bootstrap-champion-v1"),
    help="Directory for first-champion bootstrap decision artifacts.",
)
PREDICTION_MODEL_PATH_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1/model.json"),
    help="Path to saved XGBoost-v1 model.json.",
)
PREDICTION_CALIBRATION_PATH_OPTION = typer.Option(
    None,
    help="Optional path to calibration.json.",
)
ORACLE_BACKTEST_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory with train.parquet, val.parquet, and test.parquet.",
)
ORACLE_BACKTEST_WAREHOUSE_DIR_OPTION = typer.Option(
    None,
    help="Warehouse root containing raw_top_of_book parquet. Defaults to BIGAN_DATA_DIR/BIGAN_WAREHOUSE_SUBDIR.",
)
ORACLE_BACKTEST_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/backtests/oracle-label-sanity-v1"),
    help="Directory for oracle-label sanity backtest artifacts.",
)
PREDICTION_BACKTEST_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/backtests/predictions-threshold-v1"),
    help="Directory for prediction threshold backtest artifacts.",
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


@app.command("soak")
def soak(
    seconds: int = typer.Option(
        86_400,
        help="How long to run ingestion before producing the soak summary.",
    ),
    sample_interval_seconds: float = typer.Option(
        60.0,
        help="How often to append in-process Prometheus metric samples.",
    ),
    output_dir: Path = SOAK_OUTPUT_DIR_OPTION,
    min_duration_seconds: float | None = typer.Option(
        None,
        help="Minimum observed duration required to pass. Defaults to --seconds.",
    ),
    max_reconnects: float = typer.Option(
        24.0,
        help="Maximum allowed WebSocket reconnects during the run.",
    ),
    max_last_event_lag_seconds: float = typer.Option(
        60.0,
        help="Maximum allowed lag of bigan_last_event_receive_time_seconds.",
    ),
    max_hash_mismatches: float = typer.Option(
        0.0,
        help="Maximum allowed WebSocket book hash mismatches.",
    ),
    max_rss_growth_mb: float = typer.Option(
        256.0,
        help="Maximum allowed increase in process max RSS.",
    ),
    final_rollup: bool = typer.Option(
        True,
        help="Run one final NDJSON-to-Parquet rollup after stopping ingestion.",
    ),
    market_coverage: bool = typer.Option(
        True,
        "--market-coverage/--no-market-coverage",
        help="Run Gamma/CLOB REST coverage verification after stopping ingestion.",
    ),
    coverage_max_stale_seconds: float | None = typer.Option(
        None,
        help=(
            "Optional per-asset raw event freshness threshold for market coverage. "
            "Defaults to disabled for completed soak runs."
        ),
    ),
    coverage_require_hash_match: bool = typer.Option(
        False,
        help=(
            "Require latest raw WS book hashes to match CLOB REST during coverage "
            "verification."
        ),
    ),
    coverage_raw_end_grace_seconds: float = typer.Option(
        120.0,
        help="Grace window for ignoring markets opened after the raw archive ended.",
    ),
    coverage_rest_concurrency: int = typer.Option(
        12,
        min=1,
        help="Maximum concurrent CLOB REST /book requests for coverage verification.",
    ),
) -> None:
    """Run ingestion for a soak window and write validation evidence.

    The default duration is 24h for issue #25. For a local proof pass, use a
    shorter ``--seconds`` plus matching ``--min-duration-seconds``.
    """

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    started_label = _soak_timestamp_label()
    samples_path = output_dir / f"soak-{started_label}.ndjson"
    summary_path = output_dir / f"soak-{started_label}-summary.json"
    thresholds = SoakThresholds(
        min_duration_seconds=(
            float(seconds)
            if min_duration_seconds is None
            else min_duration_seconds
        ),
        max_reconnects=max_reconnects,
        max_last_event_lag_seconds=max_last_event_lag_seconds,
        max_hash_mismatches=max_hash_mismatches,
        max_rss_growth_mb=max_rss_growth_mb,
    )

    async def main() -> dict:
        runner = IngestionRunner(settings)
        started_at = asyncio.get_running_loop().time()
        wall_started_at = _now_seconds()
        stop_samples = asyncio.Event()
        serve_task = asyncio.create_task(runner.serve(), name="soak-serve")
        sampler_task = asyncio.create_task(
            record_soak_samples(
                samples_path,
                started_at_seconds=wall_started_at,
                interval_seconds=sample_interval_seconds,
                stop_event=stop_samples,
            ),
            name="soak-sampler",
        )
        fatal_exit: str | None = None
        try:
            while True:
                elapsed = asyncio.get_running_loop().time() - started_at
                remaining = seconds - elapsed
                if remaining <= 0:
                    break
                done, _ = await asyncio.wait(
                    {serve_task},
                    timeout=min(5.0, remaining),
                )
                if serve_task in done:
                    exc = serve_task.exception()
                    fatal_exit = repr(exc) if exc is not None else "serve exited early"
                    break
        finally:
            runner.stop()
            stop_samples.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(sampler_task, timeout=10.0)
            if not serve_task.done():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(serve_task, timeout=30.0)
            if not serve_task.done():
                serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await serve_task

        samples = read_soak_samples(samples_path)
        final_rollup_result = (
            finalize_soak_rollup(settings.raw_dir, settings.rollup_dir)
            if final_rollup
            else {"files": 0, "records": 0, "errors": []}
        )
        market_coverage_result = (
            await _run_market_coverage_check(
                settings=settings,
                raw_dir=settings.raw_dir,
                max_stale_seconds=coverage_max_stale_seconds,
                require_hash_match=coverage_require_hash_match,
                ignore_markets_opened_after_raw_end=True,
                raw_end_grace_seconds=coverage_raw_end_grace_seconds,
                rest_concurrency=coverage_rest_concurrency,
                max_examples=20,
            )
            if market_coverage
            else None
        )
        summary = summarize_soak(
            samples,
            raw_dir=settings.raw_dir,
            rollup_dir=settings.rollup_dir,
            thresholds=thresholds,
            fatal_exit=fatal_exit,
            market_coverage=market_coverage_result,
        )
        summary["final_rollup"] = final_rollup_result
        write_soak_summary(summary_path, summary)
        summary["samples_path"] = str(samples_path)
        summary["summary_path"] = str(summary_path)
        return summary

    summary = asyncio.run(main())
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise Exit(code=1)


@app.command("soak-report")
def soak_report(
    samples_path: Path = SOAK_SAMPLES_PATH_OPTION,
    raw_dir: Path | None = SOAK_RAW_DIR_OPTION,
    rollup_dir: Path | None = SOAK_ROLLUP_DIR_OPTION,
    summary_path: Path | None = SOAK_SUMMARY_PATH_OPTION,
    min_duration_seconds: float = typer.Option(
        86_400.0,
        help="Minimum observed duration required to pass.",
    ),
    max_reconnects: float = typer.Option(24.0),
    max_last_event_lag_seconds: float = typer.Option(60.0),
    max_hash_mismatches: float = typer.Option(0.0),
    max_rss_growth_mb: float = typer.Option(256.0),
    market_coverage: bool = typer.Option(
        False,
        "--market-coverage/--no-market-coverage",
        help="Also run Gamma/CLOB REST coverage verification for this raw archive.",
    ),
    coverage_max_stale_seconds: float | None = typer.Option(
        None,
        help=(
            "Optional per-asset raw event freshness threshold for market coverage. "
            "Defaults to disabled for completed soak runs."
        ),
    ),
    coverage_require_hash_match: bool = typer.Option(
        False,
        help="Require latest raw WS book hashes to match CLOB REST.",
    ),
    coverage_raw_end_grace_seconds: float = typer.Option(
        120.0,
        help="Grace window for ignoring markets opened after the raw archive ended.",
    ),
    coverage_rest_concurrency: int = typer.Option(
        12,
        min=1,
        help="Maximum concurrent CLOB REST /book requests for coverage verification.",
    ),
) -> None:
    """Validate a soak sample file after an operational run."""

    settings = IngestionSettings()
    thresholds = SoakThresholds(
        min_duration_seconds=min_duration_seconds,
        max_reconnects=max_reconnects,
        max_last_event_lag_seconds=max_last_event_lag_seconds,
        max_hash_mismatches=max_hash_mismatches,
        max_rss_growth_mb=max_rss_growth_mb,
    )
    report_raw_dir = settings.raw_dir if raw_dir is None else raw_dir
    market_coverage_result = (
        asyncio.run(
            _run_market_coverage_check(
                settings=settings,
                raw_dir=report_raw_dir,
                max_stale_seconds=coverage_max_stale_seconds,
                require_hash_match=coverage_require_hash_match,
                ignore_markets_opened_after_raw_end=True,
                raw_end_grace_seconds=coverage_raw_end_grace_seconds,
                rest_concurrency=coverage_rest_concurrency,
                max_examples=20,
            )
        )
        if market_coverage
        else None
    )
    summary = summarize_soak(
        read_soak_samples(samples_path),
        raw_dir=report_raw_dir,
        rollup_dir=settings.rollup_dir if rollup_dir is None else rollup_dir,
        thresholds=thresholds,
        market_coverage=market_coverage_result,
    )
    if summary_path is not None:
        write_soak_summary(summary_path, summary)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise Exit(code=1)


@app.command("market-coverage-report")
def market_coverage_report(
    raw_dir: Path | None = SOAK_RAW_DIR_OPTION,
    summary_path: Path | None = SOAK_SUMMARY_PATH_OPTION,
    max_stale_seconds: float = typer.Option(
        120.0,
        help="Fail if an expected token has no raw event within this many seconds.",
    ),
    disable_stale_check: bool = typer.Option(
        False,
        help="Skip freshness checks; useful when reporting on a completed short soak.",
    ),
    require_hash_match: bool = typer.Option(
        False,
        help="Fail if latest raw wire hash differs from current REST book hash.",
    ),
    ignore_markets_opened_after_raw_end: bool = typer.Option(
        False,
        help="Ignore Gamma markets created after the latest raw receive_time.",
    ),
    raw_end_grace_seconds: float = typer.Option(
        120.0,
        help="Grace window for --ignore-markets-opened-after-raw-end.",
    ),
    rest_concurrency: int = typer.Option(
        12,
        min=1,
        help="Maximum concurrent CLOB REST /book requests.",
    ),
    max_examples: int = typer.Option(
        20,
        min=1,
        help="Maximum example assets included for each failed bucket.",
    ),
) -> None:
    """Compare Gamma active markets with raw WS coverage and CLOB REST books."""

    settings = IngestionSettings()
    report_raw_dir = settings.raw_dir if raw_dir is None else raw_dir

    async def main() -> dict:
        return await _run_market_coverage_check(
            settings=settings,
            raw_dir=report_raw_dir,
            max_stale_seconds=None if disable_stale_check else max_stale_seconds,
            require_hash_match=require_hash_match,
            ignore_markets_opened_after_raw_end=ignore_markets_opened_after_raw_end,
            raw_end_grace_seconds=raw_end_grace_seconds,
            rest_concurrency=rest_concurrency,
            max_examples=max_examples,
        )

    report = asyncio.run(main())
    if summary_path is not None:
        write_soak_summary(summary_path, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise Exit(code=1)


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


@app.command("features-15m-v1")
def features_15m_v1(
    max_rows_per_partition: int = typer.Option(
        50_000,
        help="Flush feature partitions after this many rows.",
    ),
) -> None:
    """Generate minute-grain features_15m_v1 rows from canonical raw tables."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_feature_batch(
        settings.warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
    )
    typer.echo(
        json.dumps(
            {
                "feature_version": report.feature_version,
                "rows_generated": report.rows_generated,
                "rows_written": report.rows_written,
            },
            indent=2,
        )
    )


@app.command("feature-quality-report")
def feature_quality_report() -> None:
    """Run SQL quality checks against generated features_15m_v1 rows."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_feature_quality_sql_checks(settings.warehouse_dir)
    typer.echo(json.dumps(report.to_dict(), indent=2))
    if not report.passed:
        raise Exit(code=1)


@app.command("labels-15m-v1")
def labels_15m_v1(
    max_rows_per_partition: int = typer.Option(
        50_000,
        help="Flush label partitions after this many rows.",
    ),
    fee_bps: float = typer.Option(
        0.0,
        help="Entry fee assumption, in basis points, for profitability labels.",
    ),
    request_timeout_seconds: float = typer.Option(
        10.0,
        help="Per-request timeout when fetching Polymarket round metadata.",
    ),
) -> None:
    """Generate independent UP-token profitability labels_15m_v1 rows."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_label_batch(
        settings.warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
        fee_bps=fee_bps,
        gamma_api_base=settings.gamma_api_base,
        market_slug_prefix=settings.market_slug_prefix,
        request_timeout_seconds=request_timeout_seconds,
    )
    typer.echo(
        json.dumps(
            {
                "label_version": report.label_version,
                "rows_generated": report.rows_generated,
                "rows_written": report.rows_written,
                "fee_bps": fee_bps,
            },
            indent=2,
        )
    )


@app.command("training-dataset-v1")
def training_dataset_v1(
    output_dir: Path = TRAINING_DATASET_OUTPUT_DIR_OPTION,
    min_completeness_score: float = typer.Option(
        0.80,
        help="Minimum feature completeness_score accepted for training samples.",
    ),
    train_fraction: float = typer.Option(
        0.60,
        help="Oldest fraction of rows assigned to train.",
    ),
    val_fraction: float = typer.Option(
        0.20,
        help="Next fraction of rows assigned to validation; the remainder is test.",
    ),
) -> None:
    """Assemble train/val/test samples from features_15m_v1 and labels_15m_v1."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = assemble_training_dataset(
        settings.warehouse_dir,
        output_dir,
        split_config=SplitConfig(
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        ),
        min_completeness_score=min_completeness_score,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("logistic-baseline-v1")
def logistic_baseline_v1(
    dataset_dir: Path = LOGISTIC_DATASET_DIR_OPTION,
    output_dir: Path = LOGISTIC_OUTPUT_DIR_OPTION,
    epochs: int = typer.Option(500, help="Number of full-batch gradient descent epochs."),
    learning_rate: float = typer.Option(0.10, help="Full-batch gradient descent learning rate."),
    l2_penalty: float = typer.Option(0.0, help="L2 coefficient penalty."),
) -> None:
    """Train deterministic logistic regression baseline artifacts."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = train_logistic_baseline(
        dataset_dir,
        output_dir,
        config=LogisticBaselineConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            l2_penalty=l2_penalty,
        ),
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("xgboost-v1")
def xgboost_v1(
    dataset_dir: Path = XGBOOST_DATASET_DIR_OPTION,
    output_dir: Path = XGBOOST_OUTPUT_DIR_OPTION,
    rounds_grid: str = typer.Option("100,200,300", help="Comma-separated boosting-round grid."),
    learning_rate_grid: str = typer.Option("0.01,0.05,0.10", help="Comma-separated learning-rate grid."),
    l2_penalty_grid: str = typer.Option("0.10,1.0,5.0", help="Comma-separated L2 penalty grid."),
    max_depth_grid: str = typer.Option("3,4,5", help="Comma-separated max-depth grid."),
    subsample_grid: str = typer.Option("0.70,0.80,1.0", help="Comma-separated row-subsample grid."),
    colsample_bytree_grid: str = typer.Option(
        "0.70,0.80,1.0",
        help="Comma-separated column-subsample grid.",
    ),
) -> None:
    """Train deterministic XGBoost-v1 candidate artifacts."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = train_xgboost_v1(
        dataset_dir,
        output_dir,
        config=XGBoostV1Config(
            rounds_grid=_parse_int_grid(rounds_grid),
            learning_rate_grid=_parse_float_grid(learning_rate_grid),
            l2_penalty_grid=_parse_float_grid(l2_penalty_grid),
            max_depth_grid=_parse_int_grid(max_depth_grid),
            subsample_grid=_parse_float_grid(subsample_grid),
            colsample_bytree_grid=_parse_float_grid(colsample_bytree_grid),
        ),
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("xgboost-v2")
def xgboost_v2(
    dataset_dir: Path = XGBOOST_DATASET_DIR_OPTION,
    output_dir: Path = XGBOOST_V2_OUTPUT_DIR_OPTION,
) -> None:
    """Train conservative XGBoost-v2 candidate artifacts."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = train_xgboost_v2(dataset_dir, output_dir)
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("calibration-v1")
def calibration_v1(
    model_path: Path = CALIBRATION_MODEL_PATH_OPTION,
    dataset_dir: Path = CALIBRATION_DATASET_DIR_OPTION,
    output_dir: Path = CALIBRATION_OUTPUT_DIR_OPTION,
) -> None:
    """Fit probability calibration for a saved XGBoost-v1 model."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = fit_probability_calibration(model_path, dataset_dir, output_dir)
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("promotion-report-v1")
def promotion_report_v1(
    baseline_dir: Path = PROMOTION_BASELINE_DIR_OPTION,
    candidate_dir: Path = PROMOTION_CANDIDATE_DIR_OPTION,
    calibration_dir: Path = PROMOTION_CALIBRATION_DIR_OPTION,
    backtest_summary_path: Path = PROMOTION_BACKTEST_SUMMARY_OPTION,
    output_dir: Path = PROMOTION_OUTPUT_DIR_OPTION,
) -> None:
    """Evaluate model promotion rules and write a checklist/report."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = evaluate_model_promotion(
        baseline_dir,
        candidate_dir,
        calibration_dir,
        backtest_summary_path,
        output_dir,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("bootstrap-champion-v1")
def bootstrap_champion_v1(
    baseline_dir: Path = PROMOTION_BASELINE_DIR_OPTION,
    candidate_dir: Path = PROMOTION_CANDIDATE_DIR_OPTION,
    calibration_dir: Path = PROMOTION_CALIBRATION_DIR_OPTION,
    candidate_backtest_summary_path: Path = PROMOTION_BACKTEST_SUMMARY_OPTION,
    output_dir: Path = BOOTSTRAP_OUTPUT_DIR_OPTION,
    baseline_backtest_summary_path: Path | None = BOOTSTRAP_BASELINE_BACKTEST_SUMMARY_OPTION,
    serving_readiness_path: Path | None = BOOTSTRAP_SERVING_READINESS_PATH_OPTION,
    feature_schema_path: Path | None = BOOTSTRAP_FEATURE_SCHEMA_PATH_OPTION,
    rollback_runbook_path: Path | None = BOOTSTRAP_ROLLBACK_RUNBOOK_PATH_OPTION,
    baseline_type: str = typer.Option("logistic regression baseline", help="Human-readable baseline type."),
    baseline_explicit: bool = typer.Option(
        True,
        "--baseline-explicit/--baseline-inferred",
        help="Whether the supplied baseline is an explicit project baseline.",
    ),
) -> None:
    """Evaluate whether a candidate is ready to become the first champion."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = evaluate_bootstrap_champion(
        baseline_dir=baseline_dir,
        candidates=(
            BootstrapCandidateInput(
                candidate_dir=candidate_dir,
                calibration_dir=calibration_dir,
                candidate_backtest_summary_path=candidate_backtest_summary_path,
                serving_readiness_path=serving_readiness_path,
                feature_schema_path=feature_schema_path,
            ),
        ),
        baseline_backtest_summary_path=baseline_backtest_summary_path,
        rollback_runbook_path=rollback_runbook_path,
        baseline_type=baseline_type,
        baseline_explicit=baseline_explicit,
        output_dir=output_dir,
    )
    typer.echo(report.to_markdown())


@app.command("predictions-v1")
def predictions_v1(
    model_path: Path = PREDICTION_MODEL_PATH_OPTION,
    calibration_path: Path | None = PREDICTION_CALIBRATION_PATH_OPTION,
    max_rows_per_partition: int = typer.Option(
        50_000,
        help="Flush prediction partitions after this many rows.",
    ),
) -> None:
    """Generate predictions table rows from features_15m_v1."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_prediction_batch(
        settings.warehouse_dir,
        model_path,
        calibration_path=calibration_path,
        max_rows_per_partition=max_rows_per_partition,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("backtest-oracle-sanity-v1")
def backtest_oracle_sanity_v1(
    dataset_dir: Path = ORACLE_BACKTEST_DATASET_DIR_OPTION,
    warehouse_dir: Path | None = ORACLE_BACKTEST_WAREHOUSE_DIR_OPTION,
    output_dir: Path = ORACLE_BACKTEST_OUTPUT_DIR_OPTION,
    thresholds: str = typer.Option(
        "0.00,0.03,0.05",
        help="Comma-separated edge thresholds for the oracle sweep.",
    ),
    use_label_target_ts: bool = typer.Option(
        True,
        "--label-target-ts/--fixed-hold",
        help="Exit at each label target_ts instead of feature_ts + 15m.",
    ),
    required_outcome_side: str = typer.Option(
        "UP",
        help="Required outcome side encoded in canonical_symbol. Use an empty string to include all outcomes.",
    ),
) -> None:
    """Run a perfect-label sanity backtest before trusting model promotion evidence."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_oracle_label_sanity_backtest(
        dataset_dir=dataset_dir,
        warehouse_dir=settings.warehouse_dir if warehouse_dir is None else warehouse_dir,
        output_dir=output_dir,
        thresholds=_parse_float_grid(thresholds),
        use_label_target_ts=use_label_target_ts,
        required_outcome_side=required_outcome_side or None,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("backtest-predictions-v1")
def backtest_predictions_v1(
    warehouse_dir: Path | None = ORACLE_BACKTEST_WAREHOUSE_DIR_OPTION,
    output_dir: Path = PREDICTION_BACKTEST_OUTPUT_DIR_OPTION,
    model_version: str | None = typer.Option(
        None,
        help="Optional model_version filter for the predictions table.",
    ),
    thresholds: str = typer.Option(
        "0.00,0.03,0.05",
        help="Comma-separated edge thresholds for the prediction sweep.",
    ),
    required_outcome_side: str = typer.Option(
        "UP",
        help="Required outcome side encoded in canonical_symbol. Use an empty string to include all outcomes.",
    ),
) -> None:
    """Run a grouped threshold backtest from warehouse predictions."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_prediction_threshold_backtest(
        warehouse_dir=settings.warehouse_dir if warehouse_dir is None else warehouse_dir,
        output_dir=output_dir,
        model_version=model_version,
        thresholds=_parse_float_grid(thresholds),
        required_outcome_side=required_outcome_side or None,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("backtest-config")
def backtest_config(
    config_path: Path = BACKTEST_CONFIG_PATH_ARGUMENT,
    preserve_run_id: bool = typer.Option(
        False,
        help="Preserve output.run_id from the file instead of generating a fresh one.",
    ),
) -> None:
    """Validate and print normalized backtest config JSON."""
    config = load_backtest_config(config_path, new_run_id=not preserve_run_id)
    typer.echo(json.dumps(config.to_script_dict(), indent=2, sort_keys=True))


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


def _soak_timestamp_label() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _now_seconds() -> float:
    return time.time()


async def _run_market_coverage_check(
    *,
    settings: IngestionSettings,
    raw_dir: Path,
    max_stale_seconds: float | None,
    require_hash_match: bool,
    ignore_markets_opened_after_raw_end: bool,
    raw_end_grace_seconds: float,
    rest_concurrency: int,
    max_examples: int,
) -> dict[str, Any]:
    try:
        async with GammaClient(
            settings.gamma_api_base,
            settings.market_slug_prefix,
        ) as gamma:
            markets = await gamma.list_active_markets()
        async with PolymarketRestClient(
            settings.clob_rest_url,
            data_api_base_url=settings.polymarket_data_api_url,
            timeout_seconds=settings.backfill_rest_timeout_seconds,
        ) as rest:
            return await compare_market_coverage(
                markets=markets,
                raw_dir=raw_dir,
                rest=rest,
                max_stale_seconds=max_stale_seconds,
                require_hash_match=require_hash_match,
                ignore_markets_opened_after_raw_end=ignore_markets_opened_after_raw_end,
                raw_end_grace_seconds=raw_end_grace_seconds,
                max_concurrency=rest_concurrency,
                max_examples=max_examples,
            )
    except Exception as exc:  # noqa: BLE001
        return {
            "passed": False,
            "error": repr(exc),
            "raw": {"dir": str(raw_dir)},
        }


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
                data_api_base_url=settings.polymarket_data_api_url,
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


def _parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated integers") from exc
    if not parsed:
        raise typer.BadParameter("expected at least one integer")
    return parsed


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated numbers") from exc
    if not parsed:
        raise typer.BadParameter("expected at least one number")
    return parsed


if __name__ == "__main__":
    app()
