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
