"""Locked v1 feature dictionary.

This registry is intentionally declarative: feature computation lands in later
tickets, while issue #6 fixes the names, formulas, windows, and units that
training and online inference must share.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

FeatureGroup = Literal[
    "order_book",
    "trade_flow",
    "price_return",
    "volatility_regime",
]

FEATURE_SET_ID = "bigan-mvp-v1"
FEATURE_VERSION = "bigan-mvp-v1.0.0"
FEATURE_VERSION_STATUS = "locked"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature contract entry."""

    name: str
    group: FeatureGroup
    formula: str
    window: str
    unit: str
    source_tables: tuple[str, ...]
    dtype: str = "float64"
    null_policy: str = "NULL until all required as-of inputs are available."


FEATURES: tuple[FeatureSpec, ...] = (
    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------
    FeatureSpec(
        "ob_bid_price",
        "order_book",
        "Latest raw_top_of_book.bid_price at or before prediction_ts.",
        "point-in-time as-of",
        "probability [0,1]",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "ob_ask_price",
        "order_book",
        "Latest raw_top_of_book.ask_price at or before prediction_ts.",
        "point-in-time as-of",
        "probability [0,1]",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "ob_mid_price",
        "order_book",
        "(ob_bid_price + ob_ask_price) / 2.",
        "point-in-time as-of",
        "probability [0,1]",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "ob_spread",
        "order_book",
        "ob_ask_price - ob_bid_price.",
        "point-in-time as-of",
        "probability points",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "ob_spread_bps",
        "order_book",
        "10000 * ob_spread / ob_mid_price.",
        "point-in-time as-of",
        "basis points",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "ob_l1_bid_size",
        "order_book",
        "Latest BID level=0 size from raw_orderbook_snapshot.",
        "point-in-time as-of",
        "contracts",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_l1_ask_size",
        "order_book",
        "Latest ASK level=0 size from raw_orderbook_snapshot.",
        "point-in-time as-of",
        "contracts",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_l1_size_imbalance",
        "order_book",
        "(ob_l1_bid_size - ob_l1_ask_size) / (ob_l1_bid_size + ob_l1_ask_size).",
        "point-in-time as-of",
        "ratio [-1,1]",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_microprice_l1",
        "order_book",
        "(ob_ask_price * ob_l1_bid_size + ob_bid_price * ob_l1_ask_size) / "
        "(ob_l1_bid_size + ob_l1_ask_size).",
        "point-in-time as-of",
        "probability [0,1]",
        ("raw_top_of_book", "raw_orderbook_snapshot"),
    ),
    FeatureSpec(
        "ob_depth_bid_size_3",
        "order_book",
        "sum(size) over latest BID levels 0..2.",
        "point-in-time as-of",
        "contracts",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_depth_ask_size_3",
        "order_book",
        "sum(size) over latest ASK levels 0..2.",
        "point-in-time as-of",
        "contracts",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_depth_imbalance_3",
        "order_book",
        "(ob_depth_bid_size_3 - ob_depth_ask_size_3) / "
        "(ob_depth_bid_size_3 + ob_depth_ask_size_3).",
        "point-in-time as-of",
        "ratio [-1,1]",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_depth_bid_size_5",
        "order_book",
        "sum(size) over latest BID levels 0..4.",
        "point-in-time as-of",
        "contracts",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_depth_ask_size_5",
        "order_book",
        "sum(size) over latest ASK levels 0..4.",
        "point-in-time as-of",
        "contracts",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_depth_imbalance_5",
        "order_book",
        "(ob_depth_bid_size_5 - ob_depth_ask_size_5) / "
        "(ob_depth_bid_size_5 + ob_depth_ask_size_5).",
        "point-in-time as-of",
        "ratio [-1,1]",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "ob_quote_update_count_60s",
        "order_book",
        "count(raw_top_of_book rows with prediction_ts - 60s < ts <= prediction_ts).",
        "trailing 60s",
        "events",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "ob_quote_age_ms",
        "order_book",
        "prediction_ts - max(raw_top_of_book.ts <= prediction_ts).",
        "point-in-time as-of",
        "milliseconds",
        ("raw_top_of_book",),
    ),
    # ------------------------------------------------------------------
    # Trade flow
    # ------------------------------------------------------------------
    FeatureSpec(
        "trade_last_price",
        "trade_flow",
        "Latest raw_trades.price at or before prediction_ts.",
        "point-in-time as-of",
        "probability [0,1]",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_last_size",
        "trade_flow",
        "Latest raw_trades.size at or before prediction_ts.",
        "point-in-time as-of",
        "contracts",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_time_since_last_ms",
        "trade_flow",
        "prediction_ts - max(raw_trades.ts <= prediction_ts).",
        "point-in-time as-of",
        "milliseconds",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_count_15s",
        "trade_flow",
        "count(raw_trades rows with prediction_ts - 15s < ts <= prediction_ts).",
        "trailing 15s",
        "trades",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_count_60s",
        "trade_flow",
        "count(raw_trades rows with prediction_ts - 60s < ts <= prediction_ts).",
        "trailing 60s",
        "trades",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_count_300s",
        "trade_flow",
        "count(raw_trades rows with prediction_ts - 300s < ts <= prediction_ts).",
        "trailing 300s",
        "trades",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_volume_15s",
        "trade_flow",
        "sum(size) for raw_trades rows with prediction_ts - 15s < ts <= prediction_ts.",
        "trailing 15s",
        "contracts",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_volume_60s",
        "trade_flow",
        "sum(size) for raw_trades rows with prediction_ts - 60s < ts <= prediction_ts.",
        "trailing 60s",
        "contracts",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_volume_300s",
        "trade_flow",
        "sum(size) for raw_trades rows with prediction_ts - 300s < ts <= prediction_ts.",
        "trailing 300s",
        "contracts",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_buy_volume_60s",
        "trade_flow",
        "sum(size) where side='BUY' and prediction_ts - 60s < ts <= prediction_ts.",
        "trailing 60s",
        "contracts",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_sell_volume_60s",
        "trade_flow",
        "sum(size) where side='SELL' and prediction_ts - 60s < ts <= prediction_ts.",
        "trailing 60s",
        "contracts",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_flow_imbalance_60s",
        "trade_flow",
        "(trade_buy_volume_60s - trade_sell_volume_60s) / "
        "(trade_buy_volume_60s + trade_sell_volume_60s).",
        "trailing 60s",
        "ratio [-1,1]",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_vwap_60s",
        "trade_flow",
        "sum(price * size) / sum(size) over trailing 60s trades.",
        "trailing 60s",
        "probability [0,1]",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_max_size_300s",
        "trade_flow",
        "max(size) over trailing 300s trades.",
        "trailing 300s",
        "contracts",
        ("raw_trades",),
    ),
    FeatureSpec(
        "trade_buy_count_ratio_60s",
        "trade_flow",
        "count(side='BUY') / count(*) over trailing 60s trades.",
        "trailing 60s",
        "ratio [0,1]",
        ("raw_trades",),
    ),
    # ------------------------------------------------------------------
    # Price / return
    # ------------------------------------------------------------------
    FeatureSpec(
        "mid_return_15s",
        "price_return",
        "ln(ob_mid_price_t / ob_mid_price_{t-15s}) using as-of mid prices.",
        "trailing 15s",
        "log return",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "mid_return_60s",
        "price_return",
        "ln(ob_mid_price_t / ob_mid_price_{t-60s}) using as-of mid prices.",
        "trailing 60s",
        "log return",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "mid_return_300s",
        "price_return",
        "ln(ob_mid_price_t / ob_mid_price_{t-300s}) using as-of mid prices.",
        "trailing 300s",
        "log return",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "mid_momentum_60s_300s",
        "price_return",
        "mid_return_60s - mid_return_300s.",
        "trailing 300s",
        "log return difference",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "spot_price",
        "price_return",
        "Latest raw_spot_price.price at or before prediction_ts.",
        "point-in-time as-of",
        "USD per BTC",
        ("raw_spot_price",),
    ),
    FeatureSpec(
        "spot_return_60s",
        "price_return",
        "ln(spot_price_t / spot_price_{t-60s}) using as-of spot prices.",
        "trailing 60s",
        "log return",
        ("raw_spot_price",),
    ),
    FeatureSpec(
        "spot_return_300s",
        "price_return",
        "ln(spot_price_t / spot_price_{t-300s}) using as-of spot prices.",
        "trailing 300s",
        "log return",
        ("raw_spot_price",),
    ),
    FeatureSpec(
        "oracle_price",
        "price_return",
        "Latest raw_oracle_price.price at or before prediction_ts.",
        "point-in-time as-of",
        "USD per BTC",
        ("raw_oracle_price",),
    ),
    FeatureSpec(
        "spot_oracle_basis_bps",
        "price_return",
        "10000 * (spot_price / oracle_price - 1).",
        "point-in-time as-of",
        "basis points",
        ("raw_spot_price", "raw_oracle_price"),
    ),
    FeatureSpec(
        "target_distance_bps",
        "price_return",
        "10000 * (spot_price / target_price - 1).",
        "point-in-time as-of",
        "basis points",
        ("raw_spot_price", "market_meta"),
    ),
    FeatureSpec(
        "market_progress",
        "price_return",
        "clip((prediction_ts - market_start_ts) / (market_end_ts - market_start_ts), 0, 1).",
        "point-in-time as-of",
        "ratio [0,1]",
        ("market_meta",),
    ),
    FeatureSpec(
        "time_to_expiry_seconds",
        "price_return",
        "max(0, (market_end_ts - prediction_ts) / 1000).",
        "point-in-time as-of",
        "seconds",
        ("market_meta",),
    ),
    # ------------------------------------------------------------------
    # Volatility / regime
    # ------------------------------------------------------------------
    FeatureSpec(
        "mid_realized_vol_60s",
        "volatility_regime",
        "sqrt(sum(power(delta ln(ob_mid_price), 2))) over top-of-book updates.",
        "trailing 60s",
        "realized log-return volatility",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "mid_realized_vol_300s",
        "volatility_regime",
        "sqrt(sum(power(delta ln(ob_mid_price), 2))) over top-of-book updates.",
        "trailing 300s",
        "realized log-return volatility",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "spot_realized_vol_300s",
        "volatility_regime",
        "sqrt(sum(power(delta ln(spot_price), 2))) over spot ticks.",
        "trailing 300s",
        "realized log-return volatility",
        ("raw_spot_price",),
    ),
    FeatureSpec(
        "spread_mean_60s",
        "volatility_regime",
        "avg(ob_spread) over top-of-book rows in the trailing 60s window.",
        "trailing 60s",
        "probability points",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "spread_p95_300s",
        "volatility_regime",
        "percentile_cont(0.95) of ob_spread over trailing 300s top-of-book rows.",
        "trailing 300s",
        "probability points",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "depth_total_3_mean_60s",
        "volatility_regime",
        "avg(ob_depth_bid_size_3 + ob_depth_ask_size_3) over trailing 60s snapshots.",
        "trailing 60s",
        "contracts",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "depth_total_3_zscore_300s",
        "volatility_regime",
        "(current_depth_total_3 - avg(depth_total_3)) / stddev_pop(depth_total_3) "
        "over trailing 300s snapshots.",
        "trailing 300s",
        "z-score",
        ("raw_orderbook_snapshot",),
    ),
    FeatureSpec(
        "trade_intensity_60s",
        "volatility_regime",
        "trade_count_60s / 60.",
        "trailing 60s",
        "trades per second",
        ("raw_trades",),
    ),
    FeatureSpec(
        "quote_intensity_60s",
        "volatility_regime",
        "ob_quote_update_count_60s / 60.",
        "trailing 60s",
        "updates per second",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "liquidity_spread_depth_score",
        "volatility_regime",
        "ob_spread_bps / ln(1 + ob_depth_bid_size_3 + ob_depth_ask_size_3).",
        "point-in-time as-of",
        "score",
        ("raw_top_of_book", "raw_orderbook_snapshot"),
    ),
    FeatureSpec(
        "regime_vol_ratio_60s_300s",
        "volatility_regime",
        "mid_realized_vol_60s / mid_realized_vol_300s.",
        "trailing 300s",
        "ratio",
        ("raw_top_of_book",),
    ),
    FeatureSpec(
        "quiet_time_ratio_300s",
        "volatility_regime",
        "fraction of one-second bins in trailing 300s with zero top-of-book updates.",
        "trailing 300s",
        "ratio [0,1]",
        ("raw_top_of_book",),
    ),
)


def feature_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in FEATURES)


def features_by_group() -> dict[FeatureGroup, tuple[FeatureSpec, ...]]:
    grouped: dict[FeatureGroup, list[FeatureSpec]] = defaultdict(list)
    for spec in FEATURES:
        grouped[spec.group].append(spec)
    return {group: tuple(items) for group, items in grouped.items()}


def get_feature(name: str) -> FeatureSpec:
    for spec in FEATURES:
        if spec.name == name:
            return spec
    raise KeyError(name)
