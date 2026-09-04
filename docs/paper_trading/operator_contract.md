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

`operator_id` is a portable ASCII path component: 1–128 characters, starting
with a letter or digit, followed only by letters, digits, `_`, `-`, or `.`.
`.` and `..`, separators, and the case-insensitive reserved `paper-` run
namespace are rejected during configuration validation, before directory
creation. Existing deployments with a reserved ID require explicit account
identity migration; renaming only the configuration or directory does not
bypass checkpoint and manifest identity validation.

## Components and data flow

1. `OperatorConfig` performs strict TOML schema, range, safety, and endpoint
   validation. Every effective setting participates in `config_sha256`.
2. `GammaDiscoveryClient` performs public GET requests. Parsing and business
   selection are separate. `parse_gamma_markets` validates identities and
   structure; `select_market_windows` applies exact family filters and a stable
   ordering.
3. `PublicWebSocketTransport` sends the public subscription before bootstrap,
   then owns PING, a bounded drop-oldest queue, bounded exponential reconnect,
   and connection generation. Binance deltas are buffered while REST depth is
   in flight.
4. `BinanceDepthSynchronizer` builds a local price-level book from the REST
   snapshot and applies absolute-quantity `U/u` deltas, including one-sided
   updates and zero-quantity deletes. Both sides have the fixed
   `binance_book_level_limit`; reaching it fails closed instead of evicting
   information required to identify the true top. A gap, reconnect, symbol
   mismatch, future/out-of-order event, or buffer/level overflow invalidates
   alpha and requests an immediate reconnect/bootstrap.
   REST receipt time is not an exchange event timestamp: a standalone snapshot
   initializes only the local book. The first valid WS delta seeds OFI and spot
   using its exchange timestamp, including when it was buffered before the REST
   response arrived. Volatility warm-up starts from that event-time baseline.
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
exact structural slug (`<underlying>-updown-(5m|15m)-<epoch>`), never a
fuzzy title substring. Explicit classification fields, when supplied, are
validated against configured filters. This operator currently rejects all
durations except 5m and 15m; 1h is not silently mapped onto 15m pricing.

Source identities are coupled: BTC requires `BTCUSDT` and `btc/usd`; ETH
requires `ETHUSDT` and `eth/usd`. Other mappings are rejected. The operator's
slug pattern is the canonical asset/duration family; omission derives it from
those settings, while an empty or mismatched pattern is rejected. The title
regex can add a narrower filter. The Binance per-side level cap must be at
least 1,000, matching the public REST bootstrap request.

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
- TWAP: left-continuous event-time integral over the known oracle path, bounded
  by both `twap_window_ms` and the active market start. A pre-open sample may
  establish the price at the start boundary, but pre-open duration is excluded.
- Returns: log returns using samples at least
  `volatility_return_interval_ms` apart, with each actual elapsed interval
  retained. Both endpoints must lie inside `volatility_window_ms`; a return
  crossing the cutoff is discarded in both pricing and health checks. A gap
  longer than the window resets the return baseline and requires new warm-up.
- Volatility: elapsed-time-demeaned realized variance over
  `volatility_window_ms`, normalized by total observed milliseconds and then
  annualized. Irregular source cadence therefore does not inherit the configured
  minimum interval as a false sampling frequency.
- Warm-up: at least `volatility_min_samples` returns. Reconnect clears rolling
  state, so TWAP or volatility is never fabricated.
- Invalid/outlier handling: non-positive/non-finite, wrong source, out-of-order,
  future, stale, and over-bound log-return samples fail closed.
  A rejected synchronized Binance spot sample invalidates alpha and pricing
  warm state and requests re-bootstrap; cached spot is not paired with new OFI.
  Full CLOB `book` events must validate both sides, every price/quantity and the
  uncrossed top before advancing token sequence/freshness. Malformed full books
  invalidate the synchronized pair instead of borrowing a previous bid/ask.

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

settled cash == 0 -> EXHAUSTED (no new window, no capital reinjection)
any authoritative ledger/persistence/cash failure -> FAILED
shutdown: non-terminal state -> STOPPING -> STOPPED
```

`DEGRADED` is retryable for discovery, feed freshness, unavailable resolution,
or projection output. `FAILED` is permanent for an authoritative
Session/ledger/persistence/cash-consistency failure. No automatic rollover is
allowed from `FAILED`.

`EXHAUSTED` is a normal terminal economic outcome, not a persistence error.
The final settled ledger and its positive original opening balance remain the
account frontier; no zero-opening-cash successor is created. Startup derives
`EXHAUSTED` again from that ledger, polling/OMS remain fenced, and the live
supervisor shuts down feeds and releases ownership without changing this state.

Only `RUNNING` with fresh synchronized Binance, a fresh OFI timestamp, an
actively connected Chainlink stream, fresh dynamic pricing inputs, fresh
complete YES/NO books, an active window, and the current window generation may
submit a snapshot. At `now >= end_ts_ms`, acceptance is fenced before
resolution polling. Final resolution must match market, condition, window,
token, resolution identity, and an exact binary 1/0 payout. A missing or
non-final response remains `SETTLEMENT_PENDING`; the last quote is never a
settlement substitute.

## Run identity, recovery, and rollover

The stable run ID is SHA-256 over strategy ID, market ID, window ID, and paper
account ID. Process start time is not part of the identity. A new run directory
is created only for an explicit `ACTIVATING` intent. An `ACTIVE` run with a
missing directory fails closed. An existing directory uses the
strict PR-A resume path, which verifies the manifest, validates authoritative
JSONL, rebuilds the derived SQLite idempotency index, restores cash/positions/
sequence, and does not restore an unprovable WebSocket cursor or resting order.

Before reading checkpoint/ledger data or writing shared status, startup takes
non-blocking POSIX `flock` locks on both the paper account (under
`<output_dir>/.account-locks/`) and `<output_dir>/<operator_id>/.operator.lock`.
The account lock also prevents a differently named operator from writing the
same account in that output directory. A contender raises
`AccountOwnershipError` without touching checkpoint, ledger, SQLite or status.
Ownership is held through discovery retries, decisions and rollover and released
in shutdown's `finally` block or by OS process exit. Lock files are never
unlinked. Run on a POSIX filesystem supporting `flock`; these are local-storage
locks, not distributed cross-host/account locks for independent output roots.

Each operator-owned session is bound once to a per-activation identity token.
Its public snapshot, settlement, and feed-lifecycle methods reject calls with
`PaperSessionOwnershipError`; only the operator's tokenized internal entry
points may mutate it, and every entry also checks current account-lock ownership.
Decision persistence rechecks the authorized processing context. Shutdown
irrevocably closes the session before releasing the process locks; rollover
and permanent operator failure also revoke the old session. Retained session
references (including a previously valid token) cannot write after revocation
or takeover. Read-only snapshots/history remain available. Standalone PR-A
sessions retain their public API.

`<output_dir>/<operator_id>/account_checkpoint.json` is the durable account
frontier, not a dashboard projection. It records the active market metadata,
stable run ID, opening cash, original account bankroll, cumulative prior-run
realized PnL/fees, run index, configuration hash, and predecessor settlement
run/window/cash. Checkpoint schema version 2 uses two-phase activation:

1. Persist and fsync the explicit `ACTIVATING` intent.
2. Completely create (or validate/recover) the empty run; persist its immutable
   `operator_account_link.json`, then fsync the run and parent directory.
3. Atomically persist `ACTIVE` before exposing the session for decisions.

Only `ACTIVATING` may create an absent run. It cannot contain trading or
settlement events; a partial or conflicting run is not replaced. Crashing after
complete creation but before `ACTIVE` publication can safely finish activation.
`ACTIVE` must agree with its existing run link and pass strict ledger recovery.
Operator-owned PR-A ledgers always
fsync authoritative writes (the `fsync` setting controls optional projection
fsync). Startup recovers this frontier before calling discovery, including an
expired or already-settled old window. The recovered ledger determines carried
cash; a successor checkpoint is checked against its predecessor's settled
ledger. A crash before the checkpoint resumes the old run; a crash after the
checkpoint but before creation creates the intended successor with the same
cash. No funds are reinitialized from `initial_bankroll` at rollover.

Missing checkpoints alongside manifest-backed run directories (regardless of
prefix), incomplete canonical `paper-<24 hex digits>` run paths, corrupt or
configuration-mismatched checkpoints, and ledger/cash mismatches fail closed.
Existing manifests are parsed and their run identity checked, not inferred from
a directory prefix; unrelated folders such as `paper-notes` are not run data.
Pre-checkpoint and schema-version-1 deployments require explicit account-frontier migration; the
operator never guesses a balance from today's discovered market. Use a
dedicated output directory per operator/account, and retain its checkpoint
together with all predecessor ledgers. A partially created/corrupt run requires
repair from authoritative artifacts rather than silently replacing it.

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

The status `account` object is account-lifetime scoped: `initial_bankroll` never
resets on rollover, and `realized_pnl` / `total_fees` add current ledger values
to checkpointed predecessor totals. `total_pnl` is current equity minus the
original account bankroll. The explicit `current_run_initial_bankroll`,
`current_run_realized_pnl`, and `current_run_fees` fields expose the window view.
`account_summary()` returns these cumulative totals; `current_account()` retains
its existing current-window `PaperAccountSnapshot` contract. Counters are still
current-run operational counters, not lifetime accounting values.

Immutable per-run links make the predecessor chain traversable without keeping
an unbounded run list in memory. `recent_decisions`, `recent_fills`, and
`settlements` now include prior runs and return the newest N events in
chronological order. Each query caps both returned rows and scanned runs at
`recent_query_max`. `recent_runs` returns a newest-first bounded run page;
passing its last `run_id` as `before_run_id` to these methods pages to earlier
windows (exclusive). A page may contain fewer events when its scanned windows
have no matching activity; it does not imply that older account history ended.
Keep every predecessor run and run link with the account checkpoint.

Projection failure also closes execution immediately: even fresh snapshots
cannot reach Session/OMS unless the gate transition successfully publishes
`RUNNING`. Feed ingestion continues for recovery, and a successful status
publication can reopen execution without changing authoritative ledger state.

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

For public live data, install a verified regular wheel (see
[live stack build instructions](live_stack.md)), copy the live example, set
`mock=false` and `dry_run=false`, and set `source_commit` to the verified SHA
from `python -m bigan.build_provenance`. The standalone command is:

```bash
python -m bigan.paper_trading.operator --config config/paper_operator.live.toml
```

Before any live network connection or account/session creation, the CLI always
verifies its installed package against that SHA. It does not need a Supervisor
or hidden expected-source argument to enforce this gate. Missing/modified build
provenance and mismatched SHAs exit with status 2. Config-only checks and offline
mock execution remain available from source/editable installs; a successful
`--check` validates configuration, not live build readiness. All executions
remain simulated and paper-only.

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
