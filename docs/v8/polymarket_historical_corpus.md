# v8 Polymarket Historical BTC Up/Down Corpus

This document defines the Phase 2 offline corpus builder for Polymarket BTC Up/Down
markets. The builder is deterministic, paper-only, and uses local JSON/JSONL inputs.
It does not call live APIs, submit orders, sign wallet payloads, or put capital at risk.

## Scope

The corpus covers these BTC binary market families:

- `btc_updown_5m`
- `btc_updown_15m`
- `btc_updown_1h`

Each corpus build combines market metadata, normalized resolution rules, token book
snapshots, token trades, Binance BTCUSDT reference candles, and settlement metadata.
The output is a training-ready offline bundle with point-in-time features,
settlement-aware labels, and a temporal train/shadow split.

## Raw Inputs

The builder accepts local deterministic raw files:

- `raw_polymarket_markets.jsonl`
- `raw_polymarket_orderbooks.jsonl`
- `raw_polymarket_trades.jsonl`
- `raw_binance_btcusdt_klines.jsonl` (legacy BTC feature-candle contract name;
  row-level `source` records the actual provider)
- `raw_polymarket_resolutions.jsonl`

The example runner can generate a small fixture corpus for CI and local smoke tests.
Future live/public ingestion may write these same raw schemas, but CI must stay local
and deterministic.

## Normalized Outputs

The builder writes:

- `polymarket_corpus_manifest.json`
- `polymarket_market_rules.jsonl`
- `polymarket_market_metadata.jsonl`
- `polymarket_token_book_snapshots.jsonl`
- `polymarket_token_trades.jsonl`
- `polymarket_btc_reference_candles.jsonl`
- `polymarket_resolution_events.jsonl`
- `polymarket_feature_rows.jsonl`
- `polymarket_label_rows.jsonl`
- `polymarket_train_shadow_split.json`
- `polymarket_corpus_summary.json`

The manifest records raw artifact hashes, normalized artifact hashes, rule hashes,
resolution hashes, row counts, market family counts, sample config, and paper-only
safety flags.

## Causality Rules

Feature construction is strictly point-in-time:

- `feature_cutoff_ts <= decision_ts`
- `max_input_ts <= decision_ts`
- `available_at_ts <= decision_ts`

Book snapshots, trades, and candles are eligible for a feature row only when their
own `available_at_ts` is not later than the row `decision_ts`. Future settlement
data is never available to feature construction.

For `raw_binance_btcusdt_klines.jsonl`, `ts` is treated as the candle open
timestamp regardless of whether the row came from Coinbase, Kraken, or Binance.
OHLC close-derived fields, including `close_price`, are usable only after the
candle has closed:

- `available_at_ts >= ts + timeframe_ms`
- if no availability field is supplied, the builder derives `available_at_ts`
  as `ts + timeframe_ms`
- a candle whose open timestamp equals `decision_ts` is not eligible at that
  same `decision_ts`

If intra-candle BTC reference prices are required, they must come from a separate
tick or snapshot input rather than from final kline close fields.

For Polymarket token rows, `token_id` is the source of truth when present. The
builder fails closed when:

- `token_id` is unknown for the market
- `token_id` maps to UP but `outcome=DOWN`
- `token_id` maps to DOWN but `outcome=UP`

Rows without `token_id` may use `outcome` as a fallback and are normalized to the
canonical market token id.

Labels are the only stage allowed to use future information. They may use the
settlement outcome and the final eligible pre-close book snapshot to compute
future-aware returns.

## Feature Rows

Minimum features include market timing, BTC reference price movement, token book
state, spreads, liquidity depth, liquidity imbalance, and recent token trade volume:

- `market_family`
- `horizon_ms`
- `time_to_close_seconds`
- `market_age_seconds`
- `btc_mid_price`
- `btc_return_10s`
- `btc_return_30s`
- `btc_return_1m`
- `btc_return_5m`
- `btc_return_15m`
- `btc_volatility_1m`
- `btc_volatility_5m`
- `btc_volatility_15m`
- `up_bid`, `up_ask`, `up_mid`
- `down_bid`, `down_ask`, `down_mid`
- `up_down_mid_sum`, `up_down_bid_sum`, `up_down_ask_sum`
- `up_spread_bps`, `down_spread_bps`, `combined_spread_bps`
- `up_liquidity_depth`, `down_liquidity_depth`
- `liquidity_imbalance`
- `recent_up_trade_volume`, `recent_down_trade_volume`

Every feature has provenance with an input window and availability timestamp.

## Labels

The builder emits labels for:

- `NO_TRADE`
- `BUY_UP_HOLD_TO_SETTLEMENT`
- `BUY_DOWN_HOLD_TO_SETTLEMENT`
- `BUY_UP_SELL_BEFORE_CLOSE`
- `BUY_DOWN_SELL_BEFORE_CLOSE`

Entry labels use the token ask price because a buy must cross the ask. Sell-before-close
labels use the future eligible bid price because an exit must hit the bid. Hold labels
use the Phase 1 settlement engine payout semantics:

- winning token pays `1.0`
- losing token pays `0.0`
- unknown 50-50 settlement pays `0.5` to both sides

Each label separates `realized_trade_return`, `settlement_return`, and
`total_net_return`. The total return subtracts fees, slippage, and liquidity impact.

## Split

`polymarket_train_shadow_split.json` is a temporal split. It fails closed unless:

- both train and shadow splits are non-empty
- `max_train_decision_ts < min_shadow_decision_ts`
- split and dataset hashes are SHA-256 hex digests
- paper-only safety flags are preserved

No random shuffling is used.

## Safety Flags

Every Polymarket corpus artifact preserves:

- `paper_only=true`
- `capital_at_risk=false`
- `polymarket_write_enabled=false`
- `wallet_signing_enabled=false`

Where applicable, artifacts also preserve:

- `broker_exchange_write_enabled=false`
- `live_exchange_write_enabled=false`

## Local Smoke

```bash
PYTHONPATH=src python examples/v8/build_polymarket_btc_corpus.py \
  --output-dir /tmp/bigan-v8-polymarket-corpus \
  --overwrite-existing
```

The command writes raw deterministic fixtures under the output directory and builds
the normalized corpus under `corpus/`.

## CI Gate

The v8 hard-gate workflow runs:

```bash
python -m pytest \
  tests/v8/test_polymarket_corpus_contracts.py \
  tests/v8/test_polymarket_corpus_builder.py \
  tests/v8/test_polymarket_corpus_point_in_time.py -q
```
