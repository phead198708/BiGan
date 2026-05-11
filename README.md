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

# run ingestion (long-running)
bigan-ingest serve

# or for a 30-second live smoke test
bigan-ingest smoke --seconds 30
```

## Configuration

All settings are env vars prefixed `BIGAN_` (or a `.env` file in cwd). See
`src/bigan/ingestion/config.py` for the full set with defaults.

Common knobs:

| Env var | Default | Purpose |
|---|---|---|
| `BIGAN_CLOB_WS_URL` | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | Polymarket market channel |
| `BIGAN_GAMMA_API_BASE` | `https://gamma-api.polymarket.com` | Gamma REST API |
| `BIGAN_MARKET_SLUG_PREFIX` | `btc-updown-15m-` | Markets to subscribe |
| `BIGAN_GAMMA_POLL_INTERVAL_SECONDS` | `60.0` | Active-set refresh cadence |
| `BIGAN_DATA_DIR` | `data` | Output root |
| `BIGAN_METRICS_PORT` | `9101` | Prometheus scrape port |
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
- `bigan_gamma_polls_total{outcome}` — Gamma poll outcomes
- `bigan_rollup_files_total{outcome}` — rollup outcomes

## Canonical warehouse (issue #3)

The `src/bigan/canonical/` module provides an ETL pipeline that transforms raw NDJSON
into canonical Parquet tables with Hive partitioning, plus a validation/quarantine
layer (issue #4) that isolates anomalous rows before they reach the main tables.

```
src/bigan/canonical/
  schemas.py      — PyArrow schemas for raw_* tables + quarantine
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
  quarantine/
    source=<source>/dt=YYYY-MM-DD/
      part-*.parquet
```

All tables share a common identity contract:

- **Timestamps**: `ts` (event time), `message_ts` (protocol timestamp), `ingest_ts` (receive time)
- **Symbol identity**: `source`, `source_symbol`, `source_market`, `canonical_symbol`
- **Append-only**: ETL writes new `part-*.parquet` files; existing files are never mutated

### Table schemas

- **raw_top_of_book**: One row per `best_bid_ask` event with `bid_price`, `ask_price`, `spread`
- **raw_orderbook_snapshot**: Long-format orderbook with `side`, `level`, `price`, `size` per level
- **raw_trades**: One row per `last_trade_price` event with `price`, `size`, `side`, `trade_id`
- **raw_candles_1m**: Derived 1-minute OHLC candles with bid/ask/trade OHLC, VWAP, volume, counts
- **quarantine**: Anomalous rows isolated by the validation layer with `target_table`, `rule`, `detail`, `payload_json`

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

Trade-id dedup is **per ETL run**: re-ingesting the same NDJSON archive will
not flag historical duplicates from a previous run. Cross-batch dedup is a
follow-up if/when re-ETL becomes a routine workflow.

### ETL workflow

```bash
# Run ETL on a specific date's raw NDJSON archive
bigan-ingest etl-batch --date 2025-01-15

# Or run on today's data
bigan-ingest etl-batch

# Query warehouse stats
bigan-ingest warehouse-stats

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
```
