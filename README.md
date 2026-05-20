# BiGan

Polymarket prediction-market microstructure signal pipeline for the
`btc-updown-15m-*` family of binary 15-minute markets.

See `docs/adr/0001-market-data-source.md` for the data-source decision record.

## Layout

```
src/bigan/                 Python package
  ingestion/               Realtime WebSocket ingestion service (issue #2)
    config.py              Settings loaded from env / .env
    message_types.py       Pydantic models for CLOB market-channel events
    gamma_client.py        Gamma REST client (active market discovery)
    clob_ws.py             Async WebSocket client w/ reconnect + sub mgmt
    book_state.py          Local order-book replica + hash verification
    sink.py                NDJSON gzip sink (date-partitioned)
    rollup.py              NDJSON -> Parquet hourly rollup worker
    metrics.py             Prometheus counters / gauges / histograms
    runner.py              Orchestrator (Gamma poll + WS + sink + rollup)
    __main__.py            CLI entry point
tests/ingestion/           pytest unit tests
docs/adr/                  Architecture decision records
data/                      Local raw / rollup data (gitignored)
```

## Quick start

Requires Python 3.11+ (TaskGroup, StrEnum). Local dev assumes pyenv.

```bash
# create venv & install in editable mode with dev deps
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

# unit tests
pytest

# lint
ruff check src tests

# live REST schema checks (touches public Polymarket APIs)
BIGAN_RUN_LIVE_TESTS=1 pytest -m live tests/integration/ -s

# run ingestion (long-running)
bigan-ingest serve

# or for a 30-second live smoke test
bigan-ingest smoke --seconds 30

# 24h operational soak evidence for issue #25
bigan-ingest soak
```

## Configuration

All settings are env vars prefixed `BIGAN_` (or a `.env` file in cwd). See
`src/bigan/ingestion/config.py` for the full set with defaults.

Common knobs:

| Env var | Default | Purpose |
|---|---|---|
| `BIGAN_CLOB_WS_URL` | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | Polymarket market channel |
| `BIGAN_COINBASE_WS_URL` | `wss://advanced-trade-ws.coinbase.com` | Coinbase Advanced Trade market-data WebSocket |
| `BIGAN_KRAKEN_WS_URL` | `wss://ws.kraken.com/v2` | Kraken WebSocket v2 endpoint |
| `BIGAN_CHAINLINK_RPC_URL` | empty | Ethereum JSON-RPC URL for Chainlink BTC/USD reads |
| `BIGAN_CHAINLINK_FEED_ADDRESS` | `0xF403...E88c` | Chainlink BTC/USD AggregatorV3 proxy |
| `BIGAN_GAMMA_API_BASE` | `https://gamma-api.polymarket.com` | Gamma REST API |
| `BIGAN_POLYMARKET_DATA_API_URL` | `https://data-api.polymarket.com` | Public trade-history API used by REST backfill |
| `BIGAN_MARKET_SLUG_PREFIX` | `btc-updown-15m-` | Markets to subscribe |
| `BIGAN_COINBASE_PRODUCT_ID` | `BTC-USD` | Coinbase reference-price symbol |
| `BIGAN_KRAKEN_SYMBOL` | `BTC/USD` | Kraken reference-price symbol |
| `BIGAN_CHAINLINK_SYMBOL` | `BTC/USD` | Chainlink source symbol for mapping |
| `BIGAN_GAMMA_POLL_INTERVAL_SECONDS` | `60.0` | Active-set refresh cadence |
| `BIGAN_DATA_DIR` | `data` | Output root |
| `BIGAN_METRICS_PORT` | `9101` | Prometheus scrape port |
| `BIGAN_INGEST_LAG_WARN_SECONDS` | `0.5` | Warn when `ingest_ts - message_ts` exceeds this SLA |
| `BIGAN_TIMESTAMP_FUTURE_GRACE_SECONDS` | `5.0` | ETL quarantine grace for upstream clocks ahead of local ingest |
| `BIGAN_TIMESTAMP_STALE_THRESHOLD_SECONDS` | `600.0` | ETL quarantine threshold for stale/replayed event time |
| `BIGAN_ROLLUP_ENABLED` | `true` | Enable hourly NDJSON->Parquet |
| `BIGAN_LOG_LEVEL` | `INFO` | Standard logging levels |

## Storage layout

```
data/
  raw/ws_market/
    YYYY-MM-DD.ndjson.gz          # one append-only file per UTC date
    _done/                         # files already rolled up
  rollup/ws_market/
    date=YYYY-MM-DD/
      event_type=book/             # Hive-partitioned Parquet
      event_type=price_change/
      event_type=best_bid_ask/
      ...
```

The NDJSON record schema is intentionally minimal:

```json
{"receive_time": 1778423400123, "raw": {"event_type": "book", ...}}
```

`raw` is the verbatim CLOB payload preserved as a black box for replay /
re-parse. The rollup worker projects a small set of top-level fields plus the
verbatim JSON string into Parquet.

## Metrics

A Prometheus endpoint is exposed at `:${BIGAN_METRICS_PORT}/metrics`:

- `bigan_ws_messages_total{event_type}` — message throughput
- `bigan_ws_reconnects_total` — reconnect cycles
- `bigan_ws_subscribed_markets` — current subscription set size
- `bigan_ws_hash_mismatch_total{asset_id}` — book/delta hash failures
- `bigan_sink_records_written_total` — persisted record count
- `bigan_last_event_receive_time_seconds` — gauge for liveness alarms
- `bigan_ingest_lag_seconds{source,event_type}` — `ingest_ts - message_ts` latency histogram
- `bigan_gamma_polls_total{outcome}` — Gamma poll outcomes
- `bigan_rollup_files_total{outcome}` — rollup outcomes
- `bigan_gap_detected_total{asset_id}` — silence detections (#5)
- `bigan_gap_resolved_total{asset_id}` — silence resolutions (#5)
- `bigan_gap_silence_duration_seconds` — histogram of resolved gap durations
- `bigan_backfill_invocations_total{outcome}` — backfill outcomes (`ok` / `partial` / `error`)
- `bigan_backfill_records_total{kind}` — replayed record counts (`trade` / `orderbook`)
- `bigan_backfill_in_flight` — REST backfill invocations inside the concurrency guard
- `bigan_backfill_circuit_state` — circuit state (`0=closed`, `1=open`, `2=half_open`)
- `bigan_backfill_throttled_total{reason}` — backpressure events (`semaphore`, `rate_limiter`, `circuit_open`)
- `bigan_price_reader_up{source,reader}` — Coinbase/Kraken/Chainlink reader liveness
- `bigan_price_reader_last_success_time_seconds{source,reader}` — latest successfully written reference-price row
- `bigan_price_reader_messages_total{source,reader}` — reference-price rows written
- `bigan_price_reader_errors_total{source,reader,kind}` — reconnect/polling failures

## Reference price readers (issue #24)

Coinbase, Kraken, and Chainlink BTC/USD readers can run together and write
directly into the canonical warehouse:

```bash
BIGAN_CHAINLINK_RPC_URL=https://... \
bigan-ingest reference-prices --symbol-mapping-path path/to/mappings.csv
```

Rows land in `raw_spot_price` (`source=coinbase` / `source=kraken`) and
`raw_oracle_price` (`source=chainlink`). The same symbol mapping layer fills
`canonical_symbol` when mappings are available.

## Gap detection & REST backfill (issue #5)

When a previously-active asset goes silent for longer than
`BIGAN_GAP_SILENCE_THRESHOLD_SECONDS` (default `30s`), the ingestion
service marks the asset as **in-gap** and emits a `gap.detected`
structured log. Once activity resumes, it computes the missed window
`[gap_start_ms, gap_end_ms]`, fetches missed trades via the public Data
API plus a fresh orderbook snapshot via the CLOB REST API, and
re-injects them into the NDJSON sink with
`provenance="polymarket-rest-backfill"`. Downstream ETL preserves that
tag in the canonical Parquet so models / features can filter or weight
backfilled rows differently from realtime ones.

```
src/bigan/ingestion/
  gap_detector.py — per-asset silence state machine (sync, deterministic)
  clob_rest.py    — async PolymarketRestClient (CLOB book + public trade history)
  backfill.py     — orchestrates fetch -> synth -> sink replay
```

### Running live integration tests

Default `pytest` runs never touch the public internet. The live REST smoke
tests are marked `live` and stay skipped unless explicitly enabled:

```bash
BIGAN_RUN_LIVE_TESTS=1 pytest -m live tests/integration/ -s
# or
pytest --run-live -m live tests/integration/ -s
```

These checks discover an active market through Gamma, call public CLOB
`/book`, call public Data API `/trades`, and print compact schema
fingerprints. The nightly GitHub Actions workflow
`.github/workflows/nightly-live.yml` runs the same command on a schedule and
can also be triggered manually with `workflow_dispatch`.

### Recovery log events

Structured logs emitted by the recovery flow (use `BIGAN_LOG_LEVEL=INFO`
or above to capture):

| Event | Trigger |
|---|---|
| `gap.detected` | asset's silence first crosses the threshold |
| `gap.resolved` | activity resumes after a detected gap |
| `backfill.start` | REST recovery begins for a resolved gap |
| `backfill.done` | REST recovery finished (counts in payload) |
| `backfill.no_market` | gap asset unknown to Gamma cache; trades skipped |
| `backfill.trades_fetch_failed` / `backfill.book_fetch_failed` | per-leg errors |
| `backfill.circuit_open` | REST circuit is open; the gap recovery is skipped |

Each event carries `asset_id`, the gap window, and (for `done`) the
number of trades + orderbook records replayed. Combined with the
provenance column in the warehouse this makes every backfilled row
trivially traceable to the gap that produced it.

### Manual replay

For one-off historical recovery, the manual CLI bypasses the live
runner:

```bash
bigan-ingest backfill \
  --asset-id <token_id> \
  --market <condition_id> \
  --since-ms 1700000000000 \
  --until-ms 1700000060000
```

Synthesised records flow through the same NDJSON sink the live WS
pipeline uses, so the next ETL run picks them up automatically.

### Backfill configuration

| Env var | Default | Purpose |
|---|---|---|
| `BIGAN_GAP_DETECTION_ENABLED` | `true` | Toggle the watchdog + auto-backfill |
| `BIGAN_GAP_SILENCE_THRESHOLD_SECONDS` | `30` | Silence to declare a gap |
| `BIGAN_GAP_MIN_RESUME_SECONDS` | `1` | Min delta to honour a resume packet |
| `BIGAN_GAP_CHECK_INTERVAL_SECONDS` | `5` | Watchdog tick rate |
| `BIGAN_CLOB_REST_URL` | `https://clob.polymarket.com` | CLOB orderbook REST base |
| `BIGAN_POLYMARKET_DATA_API_URL` | `https://data-api.polymarket.com` | Public trade-history REST base |
| `BIGAN_BACKFILL_REST_TIMEOUT_SECONDS` | `10` | Per-request timeout |
| `BIGAN_BACKFILL_MAX_PAGES` | `20` | Trade-history page cap per gap |
| `BIGAN_BACKFILL_MAX_CONCURRENCY` | `4` | Global concurrent backfill invocation cap |
| `BIGAN_BACKFILL_RATE_LIMIT_PER_SECOND` | `10.0` | Global CLOB REST token-bucket rate |
| `BIGAN_BACKFILL_CIRCUIT_FAILURE_THRESHOLD` | `5` | Consecutive REST failures before opening circuit |
| `BIGAN_BACKFILL_CIRCUIT_COOL_DOWN_SECONDS` | `30.0` | Open-circuit cooldown before half-open probe |

### Backfill failure protection

When many assets resolve gaps at once (#28), the runner protects both CLOB REST
and the local sink with three layers:

- A global semaphore caps concurrent `BackfillService` invocations.
- A token-bucket limiter gates individual REST calls before trade/orderbook fetches.
- A circuit breaker opens after repeated REST failures, skips new backfills while
  open, then allows one half-open probe after cooldown.

## 24h soak validation (issue #25)

The soak command runs the normal ingestion service, samples its in-process
Prometheus metrics to NDJSON, and writes a JSON pass/fail summary at the end.
The default duration and thresholds match the issue #25 acceptance criteria:

```bash
# Use a fresh data dir so raw counts and rollup evidence are scoped to this run.
BIGAN_DATA_DIR=data/soak-run \
BIGAN_ROLLUP_INTERVAL_SECONDS=3600 \
BIGAN_ROLLUP_LAG_SECONDS=300 \
bigan-ingest soak
```

For a short rehearsal:

```bash
bigan-ingest soak --seconds 300 --min-duration-seconds 300
```

Evidence lands under `data/soak/`:

- `soak-<timestamp>.ndjson` — metric samples captured every 60 seconds
- `soak-<timestamp>-summary.json` — threshold checks, NDJSON decode counts,
  rollup file counts, reconnect totals, liveness lag, hash mismatches, market
  coverage against Gamma/CLOB REST, and RSS growth

To re-check a completed run without running ingestion again:

```bash
bigan-ingest soak-report --samples-path data/soak/soak-<timestamp>.ndjson
```

Add `--market-coverage` to re-run the Gamma/CLOB REST coverage comparison for
the completed raw archive. Completed-run coverage ignores markets opened after
the raw archive ended, so newly created rounds do not make old evidence fail.

`bigan-ingest soak` runs one final NDJSON-to-Parquet rollup after stopping
ingestion by default, so the summary can include immediate Parquet evidence.
Pass `--no-final-rollup` when preserving the raw top-level files in place is
more important than producing the final rollup artifact during shutdown.

The host should still run Prometheus against `:9101/metrics` for dashboard
screenshots and longer retention. Minimal scrape config:

```yaml
scrape_configs:
  - job_name: bigan-ingestion
    static_configs:
      - targets: ["localhost:9101"]
```

## Canonical warehouse (issue #3)

The `src/bigan/canonical/` module provides an ETL pipeline that transforms raw NDJSON
into canonical Parquet tables with Hive partitioning, plus a validation/quarantine
layer (issue #4) that isolates anomalous rows before they reach the main tables.

The locked v1 feature contract for downstream training and online inference is
documented in `docs/features/feature_dictionary_v1.md` and exposed through
`bigan.features.registry` as `FEATURE_VERSION = "bigan-mvp-v1.0.0"`.

```
src/bigan/canonical/
  schemas.py      — PyArrow schemas for raw_* tables + quarantine
  symbols.py      — Source-symbol -> canonical-symbol mapping lookup
  transform.py    — Convert WS event payloads to canonical row dicts
  validation.py   — Per-row rules (crossed book, negative size, dup trade_id, ...)
  writer.py       — Buffered Parquet writer with Hive partitioning
  candles.py      — 1-minute candle aggregation from top_of_book + trades
  etl.py          — Batch ETL runner (raw NDJSON -> canonical Parquet)
  query.py        — DuckDB helpers for querying the warehouse
tests/canonical/  — pytest unit tests
```

### Warehouse storage layout

```
data/warehouse/
  raw_top_of_book/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
  raw_orderbook_snapshot/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
  raw_trades/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
  raw_candles_1m/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
  raw_spot_price/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
  raw_oracle_price/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
  symbol_mapping/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
  features_15m_v1/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
  quarantine/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
```

All tables share a common identity contract:

- **Timestamps**: `ts` (event time), `message_ts` (persisted `source_timestamp_ms`), `ingest_ts` (receive time), nullable `capture_timestamp_ms` (raw sink capture time)
- **Symbol identity** (#22): `source`, `source_symbol`, `source_market`, `canonical_symbol`
- **Source channel** (#30): nullable `source_channel` keeps transport (`clob-ws`, `clob-rest`) separate from provenance
- **Provenance** (#5): `provenance` — `ws` for realtime, `polymarket-rest-seed` for initial REST snapshots, `polymarket-rest-backfill` for recovered, `manual` for CLI replays
- **Append-only**: ETL writes new `part-*.parquet` files; existing files are never mutated

`source_timestamp_ms` is not stored as a separate Parquet column because it is
the normalized value in `message_ts`; `capture_timestamp_ms` is persisted
explicitly so WS and REST snapshot capture times remain queryable after ETL.

`canonical_symbol` is filled by the optional issue #22 symbol mapping layer.
Pass `--symbol-mapping-path path/to/mappings.csv` (also supports JSON, JSONL,
or a directory of mapping files) to `bigan-ingest etl-batch`. Unknown mappings
remain `NULL` in raw tables so ingestion can continue safely.

### Table schemas

- **raw_top_of_book**: One row per `best_bid_ask` event with `bid_price`, `ask_price`, `spread`, plus `source_channel` and `capture_timestamp_ms`
- **raw_orderbook_snapshot**: Long-format orderbook with `side`, `level`, `price`, `size` per level, plus `source_channel` and `capture_timestamp_ms`
- **raw_trades**: One row per `last_trade_price` event with `price`, `size`, `side`, `trade_id`, plus `source_channel` and `capture_timestamp_ms`
- **raw_candles_1m**: Derived 1-minute OHLC candles with bid/ask/trade OHLC, VWAP, volume, counts
- **raw_spot_price**: Coinbase/Kraken BTC spot reference ticks with `price`, `bid_price`, `ask_price`
- **raw_oracle_price**: Chainlink BTC/USD latest-round rows with `price`, `answer`, `decimals`, `round_id`
- **symbol_mapping**: Temporal source-symbol lookup with `effective_from_ts`, optional `effective_to_ts`, `symbol_kind`, and `metadata_json`
- **features_15m_v1**: Minute-close, strictly backward-looking model feature rows with `feature_ts`, `symbol`, `feature_version`, quality fields, and `market_implied_prob`
- **labels_15m_v1**: Independent 15-minute UP-token profitability labels aligned to `feature_ts`, with Polymarket round prices, `direction_up_15m`, entry ask/cost, settlement price, realized return, and `label_profit_up_15m`
- **quarantine**: Anomalous rows isolated by the validation layer with `target_table`, `rule`, `detail`, `payload_json`

## Feature aggregation (issue #7)

`bigan-ingest features-15m-v1` reads canonical raw tables and appends
minute-grain rows to `features_15m_v1`. The aggregation timestamp
`feature_ts` is a minute boundary; every point-in-time lookup uses
`ts <= feature_ts`, and every rolling window uses `(feature_ts - window,
feature_ts]`.

The v1 table includes `feature_ts`, `symbol`, `feature_version`, spread,
`market_implied_prob` from the current UP-token ask,
mid-price, microprice, OBI L1/L5/L10, signed 1-minute volume, 1-minute trade
imbalance, `ret_1m`/`ret_5m`/`ret_15m`, and realized volatility over
1/5/15-minute windows.

`labels_15m_v1` separates the underlying BTC direction from the executable
UP-token trade target. `direction_up_15m` records whether the round settled UP;
`label_profit_up_15m` is true only when `settlement_price - entry_cost > 0`.
The legacy `label_up_15m` column is retained as a compatibility alias for the
profitability target.

Issue #8 adds data-quality fields to every feature row:

- `quote_age_ms`, `depth_age_ms`, `trade_age_ms` — age of the latest as-of inputs
- `completeness_score` — weighted score in `[0, 1]`
- `data_gap_flag` — true when required market-state inputs are missing or stale
- `quality_filter_pass` — true when the row is trainable under the default v1 filter

Run `bigan-ingest feature-quality-report` after feature generation to validate
row count, duplicate keys, minute alignment, identity fields, score bounds,
gap/filter consistency, and the presence of trainable rows. The SQL-only
verification recipe lives in `docs/features/feature_sql_quality_verification.md`.

## Backtest config (issue #10)

Backtest jobs read a fixed `backtest_config_v1` YAML or JSON file. The v1
contract covers the long-entry threshold, fees, slippage, latency, dataset
version, model version, output directory, and generated run id:

```yaml
schema_version: backtest_config_v1
strategy:
  long_threshold: 0.6
costs:
  fee_bps: 2.0
  slippage_bps: 1.0
execution:
  latency_ms: 500
dataset:
  dataset_version: features-labels-v1
  feature_table: features_15m_v1
  label_table: labels_15m_v1
model:
  model_version: baseline-v0
output:
  output_dir: data/backtests
```

Validate and normalize a config for scripts with:

```bash
bigan-ingest backtest-config path/to/backtest.yaml
```

The command prints JSON and generates a fresh `output.run_id` for each run by
default. Use `--preserve-run-id` only for intentional replay/debug workflows.

### Validation rules (issue #4)

Every transformed row is checked against these rules before being written. If any
rule fires, the row is redirected to `quarantine` (one quarantine row per rule
violation) and the main `raw_*` table is unaffected:

| Rule | Trigger |
|---|---|
| `empty_symbol` | `source_symbol` is null, empty string, or whitespace |
| `empty_time` | `ts` is null or non-positive |
| `crossed_book` | top-of-book row with `bid_price > ask_price` |
| `negative_price` | top-of-book / snapshot / trade with `price < 0` (or `bid_price`/`ask_price` < 0) |
| `negative_size` | snapshot / trade row with `size < 0` |
| `duplicate_trade_id` | `trade_id` already seen in the same ETL batch |
| `ts_in_future` | `ts` leads `ingest_ts` by more than `BIGAN_TIMESTAMP_FUTURE_GRACE_SECONDS` |
| `ts_too_stale` | `ingest_ts` lags `ts` by more than `BIGAN_TIMESTAMP_STALE_THRESHOLD_SECONDS` |

Trade-id dedup is **per ETL run**: re-ingesting the same NDJSON archive will
catch duplicate trade IDs before they reach the writer. Cross-batch trade
dedup (#27) also read-checks the target `raw_trades/source=<source>/dt=...`
partition before writing; duplicates already present in the warehouse are
skipped and counted as `cross_batch_duplicates_skipped` in the ETL report.

### ETL workflow

```bash
# Run ETL on a specific date's raw NDJSON archive
bigan-ingest etl-batch --date 2025-01-15

# Or run on today's data
bigan-ingest etl-batch

# Query warehouse stats
bigan-ingest warehouse-stats

# Generate minute-grain model features
bigan-ingest features-15m-v1

# Inspect anomaly counts + recent quarantine samples
bigan-ingest quarantine-report --limit 50
```

### DuckDB queries

```python
from bigan.canonical.query import open_warehouse

with open_warehouse("data/warehouse") as conn:
    # Row counts per table
    for table in ["raw_top_of_book", "raw_orderbook_snapshot", "raw_trades", "raw_candles_1m"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count}")

    # Latest top-of-book per symbol
    df = conn.execute("""
        SELECT source_symbol, bid_price, ask_price, spread
        FROM raw_top_of_book
        WHERE (source_symbol, ts) IN (
            SELECT source_symbol, MAX(ts)
            FROM raw_top_of_book
            GROUP BY source_symbol
        )
    """).fetchdf()

    # 1-minute candles for a specific symbol
    df = conn.execute("""
        SELECT *
        FROM raw_candles_1m
        WHERE source_symbol = ?
        ORDER BY bucket_ts DESC
        LIMIT 100
    """, ["some-asset-id"]).fetchdf()

    # Quarantine breakdown by rule + target table
    df = conn.execute("""
        SELECT target_table, rule, COUNT(*) AS n
        FROM quarantine
        GROUP BY target_table, rule
        ORDER BY n DESC
    """).fetchdf()

    # Backfill coverage: how many trades came from REST recovery?
    df = conn.execute("""
        SELECT provenance, COUNT(*) AS n
        FROM raw_trades
        GROUP BY provenance
        ORDER BY n DESC
    """).fetchdf()
```
