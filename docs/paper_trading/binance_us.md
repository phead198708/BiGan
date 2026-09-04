# Binance.US paper market data

The live example now uses **Binance.US**, explicitly selected by the operator.
This is a different trading venue, not a mirror or failover host for Binance
Global. Use it only where applicable under the service's terms. Public market
data uses no API key, wallet or real exchange orders.

## Paired configuration

```toml
binance_venue = "us"
binance_depth_endpoint = "https://api.binance.us/api/v3/depth"
binance_ws_url = "wss://stream.binance.us:9443/ws"
binance_symbol = "BTCUSDT"
binance_clock_ahead_tolerance_ms = 50
```

Both hosts must agree with the explicit venue. US/Global mixtures, an unsupported
venue, credentials, query overrides, write paths or unsupported ports fail
configuration validation before any network or account mutation. There is no
automatic fallback. Historical configurations that omit `binance_venue` continue
to select `global` with the original Global defaults; the offline mock example
also retains an explicit Global identity.

The public stream is `btcusdt@depth@100ms`. The existing receiver subscribes and
buffers before fetching `GET /api/v3/depth?symbol=BTCUSDT&limit=1000`; it aligns
the snapshot/update IDs, applies absolute quantities/deletions to its bounded
local book and re-bootstraps on gaps. No changes to OFI math, trading thresholds,
freshness cutoffs, risk limits or readiness measurement are made to accommodate
the new venue. A quiet/stale US market must still close the execution gate.

The live example permits up to 50ms of exchange-clock lead relative to the
original local receipt. Events are boundedly delayed until local processing time
reaches the **unchanged** exchange timestamp, and the synchronizer re-checks this
before updating the book/OFI. Both original event and arrival timestamps are
retained; no future quote is used early. Leads beyond the configured bound,
failure of the clock to catch up, ordering errors and gaps still fail closed.
The default is zero (legacy strict behavior); the absolute configurable ceiling
is 1,000ms. This is a receive-time buffer, not a widened freshness or trading gate.

API contracts: [Binance.US official documentation](https://docs.binance.us/).
The 1,000-level bootstrap is a limited snapshot; it does not represent liquidity
outside the retained book. Existing bounded-book overflow/recovery rules remain.

## Identity and migration

- US spot and alpha source: `binance_us_depth:BTCUSDT`; Global retains
  `binance_depth:BTCUSDT`. The pricing provider rejects the other source identity.
- OFI `config_identity()` includes `venue`. Operator configuration includes the
  venue and both endpoints, and the session manifest's hash binds the full
  operator and strategy identities. Native depth messages identify the symbol,
  **not the venue**; venue binding is provided by the validated transport pair.
- Status retains the `feeds.binance` key for API compatibility and adds explicit
  venue, symbol, source and endpoints. Alpha and current spot carry venue/source.
  The Dashboard labels both spot and feed health **Binance.US**.
- Preflight and JSON/Markdown soak reports identify the market-data venue. A US
  soak is not evidence that Global works, and Global OOS/calibration results do
  not validate the US alpha distribution or expected strategy performance.
- Changing venue is not a resume of the same experiment. Stop the old writer
  gracefully, preserve its output/report directory, build and install a newly
  sealed wheel, and use a **new paper account/output directory and report path**.
  Do not replace checkpoint/manifest hashes or reuse historical caches to force
  a migration. The explicit OFI/config identity fields also require a new run
  for pre-change artifacts rather than bypassing their hash checks.

## Live verification

Copy `config/paper_operator.live.example.toml` to a deployment-specific file,
set `source_commit` to the verified installed wheel's SHA, and choose a new
account/output directory. The bundled placeholder must still fail preflight.
Follow [the live stack guide](live_stack.md) for a preflight and `--duration 30m`
soak. Reachable REST/WS or a populated Dashboard is **not** a 30-minute PASS;
measurement starts only once all required inputs are ready and fresh.

No real trading, regional proxy, authentication bypass, or automatic venue
failover is introduced.
