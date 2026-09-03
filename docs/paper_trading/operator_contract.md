# Paper Trading Operator Contract (PR-B)

## Scope and safety boundary

The operator is a long-running orchestration layer around the fixed-window,
auditable `PaperTradingSession` delivered by PR-A. It discovers a market,
consumes public market data, creates or resumes one session, waits for final
resolution, and rolls to the next window. It does not implement a dashboard.

The following values are invariants, not feature flags:

```json
{
  "paper_only": true,
  "capital_at_risk": false,
  "broker_exchange_write_enabled": false,
  "live_exchange_write_enabled": false,
  "polymarket_write_enabled": false,
  "wallet_signing_enabled": false
}
```

Configuration containing wallet, credential, authorization, private-key,
signature, live-trading, approval, or order-write fields is rejected before an
output run is opened. Endpoint validation uses explicit public host/path
allowlists. No environment-variable overlay exists, so an environment variable
cannot turn a paper process into a live process. The code contains no order
creation, cancellation, allowance, wallet, or signing adapter.

## Components and data flow

1. `OperatorConfig` performs strict TOML schema, range, safety, and endpoint
   validation. Every effective setting participates in `config_sha256`.
2. `GammaDiscoveryClient` performs public GET requests. Parsing and business
   selection are separate. `parse_gamma_markets` validates identities and
   structure; `select_market_windows` applies exact family filters and a stable
   ordering.
3. `PublicWebSocketTransport` owns public subscription, PING, bounded
   drop-oldest queue, bounded exponential reconnect, and connection generation.
4. `BinanceDepthSynchronizer` applies REST snapshot plus `U/u` deltas. A gap,
   reconnect, symbol mismatch, future/out-of-order event, or buffer overflow
   invalidates alpha and requires a new snapshot.
5. `PolymarketBookSynchronizer` requires full books for both discovered token
   IDs. A missing/stale token, sequence gap, window mismatch, or old generation
   cannot produce a tradable snapshot.
6. `RollingPricingInputsProvider` keeps independent bounded Binance spot and
   Polymarket RTDS Chainlink oracle samples. It supplies the existing
   `StrategyRunner` only after TWAP, volatility, timestamp, and freshness gates
   pass.
7. `PaperTradingOperator` serializes snapshot processing, settlement, rollover,
   and shutdown under one async lock. It is the only caller of the
   `PaperTradingSession` processing boundary.
8. `operator_status.json` is an atomic derived projection.
   `OperatorReadRepository` exposes bounded status/account/history reads for
   PR-C. PR-A JSONL and SQLite remain authoritative.

The decision path is:

```text
public Binance depth ──> OFI + spot ─┐
public Chainlink RTDS ──> oracle/TWAP/volatility ─┼─> freshness gate
public Polymarket YES + NO books ────────────────┘
                                                   │
                                                   v
                                     PaperTradingSession -> PR-A ledger
```

The Polymarket book mid is never used as oracle truth.

## Discovery contract

Candidate rows must contain a market ID, condition ID, exact start/end,
distinct YES/NO (or UP/DOWN) token IDs, public resolution source, and active,
closed, and accepting-order flags. Native Gamma JSON string arrays are
supported. For the known family, classification may be derived only from an
exact structural slug (`<underlying>-updown-(5m|15m|1h)-<epoch>`), never a
fuzzy title substring. Explicit classification fields, when supplied, are
validated against configured filters.

Eligible rows must match underlying, market type, exact duration, optional
full-match slug/title regex, and the active/pre-open interval. Current windows
sort by earliest end, then latest start, then stable IDs. Future windows sort by
start, end, and stable IDs. Two rows with the same winning time rank are an
observable ambiguity and fail closed.

Provenance records endpoint, discovery time, market/condition/token IDs, slug,
window times, resolution identity, optional authoritative start reference, and
the SHA-256 of the parsed source row.

## Pricing input conventions

- Spot source: Binance depth best-bid/best-ask midpoint after a synchronized
  public REST + WebSocket book.
- Oracle source: public Polymarket RTDS `crypto_prices_chainlink` for the exact
  configured symbol.
- TWAP: arithmetic mean of accepted oracle event-time samples in
  `twap_window_ms`.
- Returns: log returns using samples at least
  `volatility_return_interval_ms` apart.
- Volatility: population standard deviation over `volatility_window_ms`,
  annualized by `sqrt(annualization_seconds * 1000 / return_interval_ms)`.
- Warm-up: at least `volatility_min_samples` returns. Reconnect clears rolling
  state, so TWAP or volatility is never fabricated.
- Invalid/outlier handling: non-positive/non-finite, wrong source, out-of-order,
  future, stale, and over-bound log-return samples fail closed.

Every provider parameter and source identity is included in the PR-A manifest
hash through the runner and operator configuration identity.

## State machine

```text
STARTING -> DISCOVERING -> SYNCING -> RUNNING
                  |           |         |
                  v           v         v
              DEGRADED <------+   SETTLEMENT_PENDING
                                      |
                                      v
                                ROLLING_OVER -> SYNCING

any authoritative ledger/persistence/cash failure -> FAILED
shutdown: any non-failed state -> STOPPING -> STOPPED
```

`DEGRADED` is retryable for discovery, feed freshness, unavailable resolution,
or projection output. `FAILED` is permanent for an authoritative
Session/ledger/persistence/cash-consistency failure. No automatic rollover is
allowed from `FAILED`.

Only `RUNNING` with fresh synchronized Binance, fresh dynamic pricing inputs,
fresh complete YES/NO books, an active window, and the current window
generation may submit a snapshot. At `now >= end_ts_ms`, acceptance is fenced
before resolution polling. Final resolution must match market, condition,
window, token, resolution identity, and an exact binary 1/0 payout. A missing
or non-final response remains `SETTLEMENT_PENDING`; the last quote is never a
settlement substitute.

## Run identity, recovery, and rollover

The stable run ID is SHA-256 over strategy ID, market ID, window ID, and paper
account ID. Process start time is not part of the identity. If that directory
does not exist, the operator creates a session. If it exists, it calls the
strict PR-A resume path, which verifies the manifest, validates authoritative
JSONL, rebuilds the derived SQLite idempotency index, restores cash/positions/
sequence, and does not restore an unprovable WebSocket cursor or resting order.

A settled window is never settled or traded again. Rollover occurs only after
the settlement event and account snapshot are durably persisted. The old
session is then replaced with a fresh runner/session for the next stable run.
The operator window generation increments. Connection generations are nested
under it, so callbacks from either an old connection or the old window are
dropped before reaching OMS. One lock prevents settlement and a decision from
mutating the ledger concurrently. Shutdown first fences new callbacks, then
waits for the lock holder and disconnects public feeds.

## Dashboard read model

`operator_status.json` has schema version `1.0` and deterministic strict JSON.
It contains operator/strategy/run identity, state and reason, process/update
timestamps, source commit, hard safety flags, market and token provenance,
seconds to expiry, both feed health records, pricing and alpha timestamp/age/
freshness, session health, account/PnL/fees/positions, bounded counters, last
decision/fill, settlement status, and reconnect/error counters.

Example (abridged values, complete top-level schema):

```json
{
  "schema_version": "1.0",
  "operator_id": "btc-paper-operator",
  "strategy_id": "alpha-pricing-v1",
  "run_id": "paper-0123456789abcdef01234567",
  "state": "RUNNING",
  "state_reason": "all_sources_fresh",
  "process_started_at_ms": 1788390000000,
  "updated_at_ms": 1788390001000,
  "source_commit": "abc123",
  "paper_only": true,
  "safety": {},
  "active_market": {},
  "feeds": {"binance": {}, "polymarket": {}, "chainlink": {}},
  "pricing_inputs": {},
  "alpha": {},
  "session": {"healthy": true, "failure_reason": null},
  "account": {"cash": 1000.0, "equity": 1000.0, "open_positions": []},
  "counters": {},
  "last_decision": null,
  "last_fill": null,
  "settlement": {"status": "OPEN", "source_reference": null}
}
```

The writer uses a same-directory temporary file and `os.replace`; NaN and
Infinity are rejected. Projection failure does not modify authoritative JSONL.
It increments `projection_errors` and moves a non-failed operator to observable
`DEGRADED`. The read repository accepts only a configured finite `N` up to a
hard maximum and reverse-reads JSONL without loading full history. It exposes
current status, current account snapshot, recent decisions, recent fills, and
settlements.

## Operation

Validate the complete, secret-free example configuration:

```bash
python -m bigan.paper_trading.operator \
  --config config/paper_operator.example.toml --check
```

Run a deterministic local demo without network:

```bash
python -m bigan.paper_trading.operator \
  --config config/paper_operator.example.toml --mock-demo
```

For public live data, copy the file, set `mock=false` and `dry_run=false`, and
set `source_commit` to the deployed revision. This changes only data sources;
all executions remain simulated and paper-only.

## Known limitations

- Gamma does not consistently publish the authoritative price-to-beat for an
  already-running up/down window. The operator accepts an explicit
  `referencePriceAtStart`/`priceToBeat`; when it is absent it stays degraded
  rather than substituting current spot or CLOB mid. A later integration can
  add a separately proven historical Chainlink start-price lookup.
- The process is single strategy, single paper account, and one active window.
- Metrics are bounded in-memory counters projected to JSON; no new monitoring
  service or large database is introduced.
- The projection is local-file read-only infrastructure for PR-C, not a Web
  API, login system, or UI.
