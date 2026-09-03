# Paper trading event and ledger contract (PR-A)

## Scope and boundary

PR-A adds the durable data boundary between the existing fixed-window strategy
runner and future paper-trading operators or dashboards. It contains decision
events, a deterministic BUY-only account ledger, append-only persistence,
recovery, and a fixed-window session. It does **not** contain market discovery,
online Binance/oracle clients, automatic rollover, an HTTP/WebSocket service,
a Web UI, wallet/signing code, or any exchange write path.

Every top-level paper artifact carries the following immutable values:

```text
paper_only=true
capital_at_risk=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
polymarket_write_enabled=false
wallet_signing_enabled=false
```

Constructors reject any attempt to relax those values. `PaperTradingSession`
only calls the existing simulated `PolymarketOMS`; storage contains no network
or execution client.

## Strategy decision schema

`StrategyDecisionEvent` is a frozen, slotted pipeline contract with schema
version `1.0`. One event is emitted for every snapshot that returns normally
from `StrategyRunner.process_snapshot_sync()`. It captures:

- run-independent time, window identity, and the complete immutable YES/NO
  top-of-book;
- alpha event time, age, freshness, missing/stale reason, and effective `z_ofi`;
- point-in-time pricing inputs and their freshness;
- the pricing signal and selected direction;
- the OMS result, fill fee, and rejection text;
- cash before/after and a stable disposition/reason code.

Dispositions are `DROPPED`, `HOLD`, `NO_ORDER`, `FILLED`, and `REJECTED`.
Reason codes are `window_mismatch`, `pricing_inputs_missing`,
`pricing_inputs_stale`, `alpha_missing`, `alpha_stale`, `signal_hold`,
`oms_no_result`, `oms_filled`, and `oms_rejected`. Missing or stale alpha is
explicitly distinguishable from a real zero Z-score. Missing/stale pricing
inputs produce no fabricated signal and fail closed as `DROPPED`.

Callbacks are synchronous and isolated. A failing subscriber increments
`decision_callback_errors`, is logged, and cannot undo an OMS result or prevent
later subscribers. Unhandled pricing or OMS exceptions retain their prior
exception behavior.

All contracts serialize through `to_dict()` to JSON-native values. Enum and
dataclass instances never escape into the payload, schemas have fixed field
sets, and non-finite floats are rejected.

`StrategyRunner.execution_history` is a bounded deque (10,000 records by
default). `execution_count` is monotonic; complete execution history belongs
in the JSONL artifacts.

## Accounting rules

The ledger starts with cash and equity equal to the initial bankroll and no
positions. A fill creates an immutable `PaperLot` and applies:

```text
notional = shares * fill_price
fee = notional * fee_bps / 10_000
cash_after = cash_before - notional - fee
```

The OMS reserves both notional and fee before a fill, so even a 100% allocation
satisfies `notional + fee <= cash`; position, cash, and fee are then committed
as one in-memory mutation. Fees are charged once on entry. `StrategyRunner.current_bankroll`,
`PolymarketOMS.bankroll`, decision `cash_after`, and ledger cash must agree;
the session raises on any mismatch.

Open lots aggregate into a `PaperPosition` per `(window_id, side)`. Marks use
executable bids only:

```text
market_value = yes_shares * yes_bid + no_shares * no_bid
equity = cash + market_value
unrealized_pnl = market_value - entry_notional - entry_fees
drawdown = (peak_equity - equity) / peak_equity
```

`HOLD`, `DROPPED`, `NO_ORDER`, and `REJECTED` update the observation/mark but do
not change cash, lots, positions, realized PnL, or commission. There is no sell
or synthetic close in PR-A.

Settlement requires an explicit `PaperSettlementInput` with provenance. It
must name a registered window, occur no earlier than its end, and provide a
finite `yes_payout` in `[0, 1]`; `no_payout = 1 - yes_payout`. A post-expiry
order-book quote is never accepted as settlement. Per lot:

```text
proceeds = shares * side_payout
realized_pnl = proceeds - shares * entry_price - entry_fee
```

Settled lots are removed. If no other window remains open, equity equals cash.

## Identity, idempotency, and replay

Authoritative `PaperDecisionEvent` and `PaperSettlementEvent` records contain
`run_id`, `event_id`, and a strictly increasing `event_sequence`. New events
must be exactly `last_event_sequence + 1`. Replaying an identical decision ID
does nothing; a duplicate ID with different content fails. Repeating identical
settlement truth is idempotent even if requested again, while a payout or
provenance conflict fails. Replay executes the same accounting functions as
the online session and checks persisted derived ledger events byte-for-value
against regenerated contracts.

Each decision also persists `source_snapshot_id`, a canonical SHA-256 over the
complete immutable market snapshot. Recovery rebuilds this identity set and
`PaperTradingSession` checks it before calling the pricing engine or OMS. A
redelivered snapshot therefore cannot create another fill after reconnect or
restart. The decision event ID is derived from the run ID and snapshot hash.

## Artifacts and recovery

An explicit `<output_dir>/<run_id>/` contains:

```text
paper_run_manifest.json
signal_events.jsonl
execution_events.jsonl
ledger_events.jsonl
position_snapshots.jsonl
pnl_snapshots.jsonl
settlement_events.jsonl
paper_snapshot.json
```

The manifest fixes source commit, initial bankroll, fee, registered window,
configuration SHA-256, and the safety boundary. The hash covers complete OFI,
pricing, OMS, Runner/static pricing input settings, retention bounds, and an
explicit dynamic pricing-provider identity. A session with a dynamic provider
but no stable provider identity is rejected. JSONL is canonical UTF-8 with
one complete sorted object per line, flush on every append, optional `fsync`,
and no NaN/Infinity. `create_new()` refuses an existing run. `resume_existing()`
requires the explicit output directory and run ID, compares manifest/config
identity, rejects malformed or truncated lines, replays all authoritative
events, verifies derived streams, and requires the current snapshot to match
the replayed result exactly.

`paper_snapshot.json` is written to a same-directory exclusive temporary file,
flushed, and atomically installed with `os.replace()`. A ledger or storage
failure permanently marks its `PaperTradingSession` failed; later processing
is refused instead of continuing with unaudited state. Feed-driven integrations
must use `PaperTradingSession.start()` (not `StrategyRunner.start()`) so every
snapshot enters through that fail-closed session boundary.

## Follow-on integration

PR-B can register more `PaperWindowRegistration` values and build a live,
read-only multi-window operator around `StrategyDecisionEvent`,
`PaperAccountLedger`, `PaperRunStore`, and `PaperTradingSession`. It must add
market discovery, source freshness, lifecycle, and rollover explicitly rather
than weakening this boundary. PR-C can read `paper_snapshot.json` and the
append-only contracts for a dashboard; it should remain a read-only consumer
and must not become an execution path.
