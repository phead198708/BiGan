# v8 Polymarket Rules and Settlement

This document covers the paper-only Polymarket Phase 1 layer for BTC UP/DOWN
binary outcome-token markets.

The layer models two distinct PnL sources:

```text
pre-resolution trade PnL
settlement redemption PnL
```

No real orders, wallet signing, private keys, or CLOB write APIs are present in
this path.

## Resolution Rules

The normalized rule contract is `PolymarketResolutionRule`.

Supported comparators:

```text
close_gt_open
close_gte_open
```

Tie behavior is explicit:

```text
up
down
unknown
```

When `unknown_50_50_enabled=true` and the market resolves as unknown, the payout
vector is:

```text
UP = 0.5
DOWN = 0.5
```

Unknown / 50-50 is not inferred from an ordinary price tie. Resolution status is
explicit:

```text
resolution_status=normal
resolution_status=unknown_50_50
```

Normal price ties follow the comparator:

```text
close_gt_open:
  close == open => DOWN

close_gte_open:
  close == open => UP
```

## Ledger Semantics

The position ledger records paper-only outcome-token events:

```text
BUY
SELL
HOLD
SETTLE
NO_TRADE
MERGE_COMPLETE_SET
```

Execution prices are conservative:

```text
BUY uses ask price
SELL uses bid price
mid price is not executable PnL
```

Polymarket paper decisions carry explicit action semantics:

```text
BUY_UP
BUY_DOWN
SELL_UP
SELL_DOWN
HOLD
NO_TRADE
```

The settlement engine dispatches these actions directly:

```text
BUY_*  -> ledger.buy(... ask_price)
SELL_* -> ledger.sell(... bid_price)
HOLD   -> ledger.hold(...)
NO_TRADE -> ledger.no_trade(...)
```

SELL fails closed if the requested paper quantity exceeds the open paper
position.

Ledger events do not contain real order ids, wallet signatures, private keys, or
broker/CLOB write handles.

## Settlement Artifacts

The Polymarket settlement engine writes these run-scoped artifacts:

```text
polymarket_position_ledger.jsonl
polymarket_settlement_events.jsonl
polymarket_position_summary.json
polymarket_pnl_breakdown.json
```

`polymarket_pnl_breakdown.json` reconciles:

```text
realized_trade_pnl
  + settlement_pnl
  + complete_set_pnl
  - fees
  - slippage
  = total_polymarket_pnl
```

The pipeline also surfaces key Polymarket PnL fields in:

```text
paper_run_summary.json
paper_bundle_manifest.json
paper_observability_report.json
github_paper_comment_payload.json
```

## Safety Flags

All Polymarket rule, ledger, settlement, and summary artifacts preserve:

```text
paper_only=true
capital_at_risk=false
polymarket_write_enabled=false
wallet_signing_enabled=false
broker_exchange_write_enabled=false
live_exchange_write_enabled=false
```

## Validation

Focused tests:

```bash
PYTHONPATH=src python -m pytest \
  tests/v8/test_polymarket_rules.py \
  tests/v8/test_polymarket_ledger.py \
  tests/v8/test_polymarket_settlement_engine.py -q
```

Pipeline tests:

```bash
PYTHONPATH=src python -m pytest \
  tests/v8/test_polymarket_contracts.py \
  tests/v8/test_polymarket_btc15m_adapter.py \
  tests/v8/test_polymarket_paper_pipeline.py -q
```
