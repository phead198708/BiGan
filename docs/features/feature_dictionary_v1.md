# Feature Dictionary v1

Issue: #6
Milestone: mvp-v1
Feature set id: `bigan-mvp-v1`
Feature version: `bigan-mvp-v1.0.0`
Status: locked
Source of truth: `src/bigan/features/registry.py`

This document locks the first production feature contract for training,
backtesting, and online inference. The matching Python registry exposes the
same feature names, formulas, windows, units, and source tables so later
feature computation code can depend on a single versioned contract.

## Time Contract

- Prediction timestamp is `t`, an epoch-milliseconds UTC event time.
- All rolling windows are trailing half-open intervals: `(t - window, t]`.
- Point-in-time values use an as-of join: latest row with `ts <= t`.
- Feature code must not read rows with `ts > t`.
- `ingest_ts` is diagnostic only and must not be used as model time.
- `source`, `source_symbol`, and `source_market` scope every feature to one
  Polymarket outcome token unless the formula explicitly joins BTC reference
  price or market metadata.
- Missing inputs produce `NULL`; imputation is a separate training/inference
  policy and is not part of this dictionary.

## Versioning Rules

- `bigan-mvp-v1.0.0` is immutable once used for training or online inference.
- Changing a feature name, formula, window, unit, null policy, or source table
  requires a new feature version.
- Additive features may use a new minor version such as `bigan-mvp-v1.1.0`,
  but the exact `v1.0.0` feature list remains frozen.
- Bug fixes that only correct documentation typos and do not alter semantics
  may use a patch version.
- Model artifacts must record both `FEATURE_SET_ID` and `FEATURE_VERSION`.

## Source Tables

- `raw_top_of_book`: best bid/ask, spread, event time.
- `raw_orderbook_snapshot`: per-side depth levels from full book snapshots.
- `raw_trades`: `last_trade_price` rows with side, price, size, and trade id.
- `raw_spot_price`: Coinbase/Kraken BTC/USD spot ticks.
- `raw_oracle_price`: Chainlink BTC/USD oracle reads.
- `market_meta`: logical market metadata from Gamma (`start_ts`, `end_ts`,
  `target_price`, token/outcome mapping). The v1 dictionary reserves these
  features even though metadata materialisation is implemented downstream.

## Order Book Features

| Name | Formula | Window | Unit | Source |
|---|---|---|---|---|
| `ob_bid_price` | latest `raw_top_of_book.bid_price` at or before `t` | point-in-time as-of | probability [0,1] | `raw_top_of_book` |
| `ob_ask_price` | latest `raw_top_of_book.ask_price` at or before `t` | point-in-time as-of | probability [0,1] | `raw_top_of_book` |
| `ob_mid_price` | (`ob_bid_price` + `ob_ask_price`) / 2 | point-in-time as-of | probability [0,1] | `raw_top_of_book` |
| `ob_spread` | `ob_ask_price` - `ob_bid_price` | point-in-time as-of | probability points | `raw_top_of_book` |
| `ob_spread_bps` | 10000 * `ob_spread` / `ob_mid_price` | point-in-time as-of | basis points | `raw_top_of_book` |
| `ob_l1_bid_size` | latest BID `level=0` size | point-in-time as-of | contracts | `raw_orderbook_snapshot` |
| `ob_l1_ask_size` | latest ASK `level=0` size | point-in-time as-of | contracts | `raw_orderbook_snapshot` |
| `ob_l1_size_imbalance` | (`ob_l1_bid_size` - `ob_l1_ask_size`) / (`ob_l1_bid_size` + `ob_l1_ask_size`) | point-in-time as-of | ratio [-1,1] | `raw_orderbook_snapshot` |
| `ob_microprice_l1` | (`ob_ask_price` * `ob_l1_bid_size` + `ob_bid_price` * `ob_l1_ask_size`) / (`ob_l1_bid_size` + `ob_l1_ask_size`) | point-in-time as-of | probability [0,1] | `raw_top_of_book`, `raw_orderbook_snapshot` |
| `ob_depth_bid_size_3` | sum BID sizes for latest levels 0..2 | point-in-time as-of | contracts | `raw_orderbook_snapshot` |
| `ob_depth_ask_size_3` | sum ASK sizes for latest levels 0..2 | point-in-time as-of | contracts | `raw_orderbook_snapshot` |
| `ob_depth_imbalance_3` | (`ob_depth_bid_size_3` - `ob_depth_ask_size_3`) / (`ob_depth_bid_size_3` + `ob_depth_ask_size_3`) | point-in-time as-of | ratio [-1,1] | `raw_orderbook_snapshot` |
| `ob_depth_bid_size_5` | sum BID sizes for latest levels 0..4 | point-in-time as-of | contracts | `raw_orderbook_snapshot` |
| `ob_depth_ask_size_5` | sum ASK sizes for latest levels 0..4 | point-in-time as-of | contracts | `raw_orderbook_snapshot` |
| `ob_depth_imbalance_5` | (`ob_depth_bid_size_5` - `ob_depth_ask_size_5`) / (`ob_depth_bid_size_5` + `ob_depth_ask_size_5`) | point-in-time as-of | ratio [-1,1] | `raw_orderbook_snapshot` |
| `ob_quote_update_count_60s` | count top-of-book rows in `(t - 60s, t]` | trailing 60s | events | `raw_top_of_book` |
| `ob_quote_age_ms` | `t` - latest top-of-book `ts` | point-in-time as-of | milliseconds | `raw_top_of_book` |

## Trade Flow Features

| Name | Formula | Window | Unit | Source |
|---|---|---|---|---|
| `trade_last_price` | latest `raw_trades.price` at or before `t` | point-in-time as-of | probability [0,1] | `raw_trades` |
| `trade_last_size` | latest `raw_trades.size` at or before `t` | point-in-time as-of | contracts | `raw_trades` |
| `trade_time_since_last_ms` | `t` - latest trade `ts` | point-in-time as-of | milliseconds | `raw_trades` |
| `trade_count_15s` | count trades in `(t - 15s, t]` | trailing 15s | trades | `raw_trades` |
| `trade_count_60s` | count trades in `(t - 60s, t]` | trailing 60s | trades | `raw_trades` |
| `trade_count_300s` | count trades in `(t - 300s, t]` | trailing 300s | trades | `raw_trades` |
| `trade_volume_15s` | sum trade `size` in `(t - 15s, t]` | trailing 15s | contracts | `raw_trades` |
| `trade_volume_60s` | sum trade `size` in `(t - 60s, t]` | trailing 60s | contracts | `raw_trades` |
| `trade_volume_300s` | sum trade `size` in `(t - 300s, t]` | trailing 300s | contracts | `raw_trades` |
| `trade_buy_volume_60s` | sum `size` where `side='BUY'` in `(t - 60s, t]` | trailing 60s | contracts | `raw_trades` |
| `trade_sell_volume_60s` | sum `size` where `side='SELL'` in `(t - 60s, t]` | trailing 60s | contracts | `raw_trades` |
| `trade_flow_imbalance_60s` | (`trade_buy_volume_60s` - `trade_sell_volume_60s`) / (`trade_buy_volume_60s` + `trade_sell_volume_60s`) | trailing 60s | ratio [-1,1] | `raw_trades` |
| `trade_vwap_60s` | sum(`price` * `size`) / sum(`size`) over trailing trades | trailing 60s | probability [0,1] | `raw_trades` |
| `trade_max_size_300s` | max trade `size` in `(t - 300s, t]` | trailing 300s | contracts | `raw_trades` |
| `trade_buy_count_ratio_60s` | count BUY trades / count all trades in `(t - 60s, t]` | trailing 60s | ratio [0,1] | `raw_trades` |

## Price / Return Features

| Name | Formula | Window | Unit | Source |
|---|---|---|---|---|
| `mid_return_15s` | ln(`ob_mid_price_t` / `ob_mid_price_{t-15s}`) | trailing 15s | log return | `raw_top_of_book` |
| `mid_return_60s` | ln(`ob_mid_price_t` / `ob_mid_price_{t-60s}`) | trailing 60s | log return | `raw_top_of_book` |
| `mid_return_300s` | ln(`ob_mid_price_t` / `ob_mid_price_{t-300s}`) | trailing 300s | log return | `raw_top_of_book` |
| `mid_momentum_60s_300s` | `mid_return_60s` - `mid_return_300s` | trailing 300s | log return difference | `raw_top_of_book` |
| `spot_price` | latest `raw_spot_price.price` at or before `t` | point-in-time as-of | USD per BTC | `raw_spot_price` |
| `spot_return_60s` | ln(`spot_price_t` / `spot_price_{t-60s}`) | trailing 60s | log return | `raw_spot_price` |
| `spot_return_300s` | ln(`spot_price_t` / `spot_price_{t-300s}`) | trailing 300s | log return | `raw_spot_price` |
| `oracle_price` | latest `raw_oracle_price.price` at or before `t` | point-in-time as-of | USD per BTC | `raw_oracle_price` |
| `spot_oracle_basis_bps` | 10000 * (`spot_price` / `oracle_price` - 1) | point-in-time as-of | basis points | `raw_spot_price`, `raw_oracle_price` |
| `target_distance_bps` | 10000 * (`spot_price` / `target_price` - 1) | point-in-time as-of | basis points | `raw_spot_price`, `market_meta` |
| `market_progress` | clip((`t` - `market_start_ts`) / (`market_end_ts` - `market_start_ts`), 0, 1) | point-in-time as-of | ratio [0,1] | `market_meta` |
| `time_to_expiry_seconds` | max(0, (`market_end_ts` - `t`) / 1000) | point-in-time as-of | seconds | `market_meta` |

## Volatility / Regime Features

| Name | Formula | Window | Unit | Source |
|---|---|---|---|---|
| `mid_realized_vol_60s` | sqrt(sum(delta ln(`ob_mid_price`)^2)) over top-of-book updates | trailing 60s | realized log-return volatility | `raw_top_of_book` |
| `mid_realized_vol_300s` | sqrt(sum(delta ln(`ob_mid_price`)^2)) over top-of-book updates | trailing 300s | realized log-return volatility | `raw_top_of_book` |
| `spot_realized_vol_300s` | sqrt(sum(delta ln(`spot_price`)^2)) over spot ticks | trailing 300s | realized log-return volatility | `raw_spot_price` |
| `spread_mean_60s` | avg(`ob_spread`) over top-of-book rows | trailing 60s | probability points | `raw_top_of_book` |
| `spread_p95_300s` | percentile_cont(0.95) of `ob_spread` | trailing 300s | probability points | `raw_top_of_book` |
| `depth_total_3_mean_60s` | avg(`ob_depth_bid_size_3` + `ob_depth_ask_size_3`) | trailing 60s | contracts | `raw_orderbook_snapshot` |
| `depth_total_3_zscore_300s` | (`current_depth_total_3` - avg(`depth_total_3`)) / stddev_pop(`depth_total_3`) | trailing 300s | z-score | `raw_orderbook_snapshot` |
| `trade_intensity_60s` | `trade_count_60s` / 60 | trailing 60s | trades per second | `raw_trades` |
| `quote_intensity_60s` | `ob_quote_update_count_60s` / 60 | trailing 60s | updates per second | `raw_top_of_book` |
| `liquidity_spread_depth_score` | `ob_spread_bps` / ln(1 + `ob_depth_bid_size_3` + `ob_depth_ask_size_3`) | point-in-time as-of | score | `raw_top_of_book`, `raw_orderbook_snapshot` |
| `regime_vol_ratio_60s_300s` | `mid_realized_vol_60s` / `mid_realized_vol_300s` | trailing 300s | ratio | `raw_top_of_book` |
| `quiet_time_ratio_300s` | fraction of one-second bins in `(t - 300s, t]` with zero top-of-book updates | trailing 300s | ratio [0,1] | `raw_top_of_book` |

## Locked v1 Counts

- Order book: 17 features
- Trade flow: 15 features
- Price / return: 12 features
- Volatility / regime: 12 features
- Total: 56 features
