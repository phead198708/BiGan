"""Causal feature mapping for Polymarket BTC 15m markets."""

from __future__ import annotations

import math

from bigan.v8.phase0 import FeatureProvenance, FeatureVector, MarketData
from bigan.v8.polymarket.contracts import (
    POLYMARKET_BTC15M_HORIZON_MS,
    POLYMARKET_SOURCE,
    PolymarketAdapterError,
    PolymarketBinaryMarket,
    PolymarketFeatureRow,
    PolymarketTokenSnapshot,
)


def build_polymarket_feature_rows(
    *,
    market: PolymarketBinaryMarket,
    token_snapshots: tuple[PolymarketTokenSnapshot, ...],
    btc_market_data: tuple[MarketData, ...],
) -> tuple[PolymarketFeatureRow, ...]:
    """Build causal v8-compatible features from mocked Polymarket snapshots."""

    if not token_snapshots:
        raise PolymarketAdapterError("missing_token_snapshots")
    if not btc_market_data:
        raise PolymarketAdapterError("missing_btc_market_data")
    rows: list[PolymarketFeatureRow] = []
    snapshots_by_ts = _snapshots_by_ts(token_snapshots)
    btc_rows = tuple(sorted(btc_market_data, key=lambda row: row.ts))
    for decision_ts in sorted(snapshots_by_ts):
        pair = snapshots_by_ts[decision_ts]
        if "UP" not in pair or "DOWN" not in pair:
            continue
        btc_now = _latest_btc_row(btc_rows, decision_ts)
        if btc_now is None:
            continue
        lookback_start_ts = max(market.market_start_ts, decision_ts - 15 * 60_000)
        features = _features_for_ts(
            market=market,
            up=pair["UP"],
            down=pair["DOWN"],
            btc_rows=btc_rows,
            btc_now=btc_now,
            decision_ts=decision_ts,
        )
        provenance = {
            name: _provenance(
                feature_name=name,
                input_start_ts=lookback_start_ts,
                input_end_ts=decision_ts,
                available_at_ts=decision_ts,
                lookback_ms=decision_ts - lookback_start_ts,
            )
            for name in features
        }
        v8_feature = FeatureVector(
            decision_ts=decision_ts,
            feature_cutoff_ts=decision_ts,
            lookback_start_ts=lookback_start_ts,
            max_input_ts=max(
                decision_ts,
                btc_now.available_at_ts or btc_now.ts,
                pair["UP"].ts,
                pair["DOWN"].ts,
            ),
            source=POLYMARKET_SOURCE,
            instrument_id=market.slug,
            features=features,
            provenance=provenance,
        )
        rows.append(
            PolymarketFeatureRow(
                market_id=market.market_id,
                condition_id=market.condition_id,
                slug=market.slug,
                decision_ts=decision_ts,
                feature_cutoff_ts=decision_ts,
                max_input_ts=v8_feature.max_input_ts,
                horizon_ms=market.horizon_ms,
                features=features,
                v8_feature=v8_feature,
            )
        )
    if not rows:
        raise PolymarketAdapterError("no_causal_feature_rows")
    return tuple(rows)


def _features_for_ts(
    *,
    market: PolymarketBinaryMarket,
    up: PolymarketTokenSnapshot,
    down: PolymarketTokenSnapshot,
    btc_rows: tuple[MarketData, ...],
    btc_now: MarketData,
    decision_ts: int,
) -> dict[str, float | int | None]:
    btc_mid = btc_now.effective_mid_price
    return_1m = _return_over_window(btc_rows, decision_ts, 60_000)
    return_5m = _return_over_window(btc_rows, decision_ts, 5 * 60_000)
    return_15m = _return_over_window(btc_rows, decision_ts, 15 * 60_000)
    vol_5m = _volatility(btc_rows, decision_ts, 5 * 60_000)
    vol_15m = _volatility(btc_rows, decision_ts, 15 * 60_000)
    total_liquidity = up.liquidity_depth + down.liquidity_depth
    imbalance = (
        (up.liquidity_depth - down.liquidity_depth) / total_liquidity
        if total_liquidity > 0.0
        else 0.0
    )
    up_down_spread = up.spread_bps + down.spread_bps
    return {
        "btc_mid_price": btc_mid,
        "btc_return_1m": return_1m,
        "btc_return_5m": return_5m,
        "btc_return_15m": return_15m,
        "btc_volatility_5m": vol_5m,
        "btc_volatility_15m": vol_15m,
        "up_token_mid_price": up.mid_price,
        "down_token_mid_price": down.mid_price,
        "up_down_price_sum": up.mid_price + down.mid_price,
        "up_down_spread_bps": up_down_spread,
        "market_time_to_close_seconds": max(0, market.market_end_ts - decision_ts)
        / 1000.0,
        "market_age_seconds": max(0, decision_ts - market.market_start_ts) / 1000.0,
        "up_liquidity_depth": up.liquidity_depth,
        "down_liquidity_depth": down.liquidity_depth,
        "up_down_liquidity_imbalance": imbalance,
        # Phase 0-compatible core columns for downstream shared tooling.
        "mid_price": btc_mid,
        "spread": (up.ask_price - up.bid_price) + (down.ask_price - down.bid_price),
        "spread_bps": up_down_spread,
        "return_1m": return_1m,
        "return_5m": return_5m,
        "return_15m": return_15m,
        "volatility_5m": vol_5m,
        "volatility_15m": vol_15m,
        "volume_1m": up.volume + down.volume,
        "volume_5m": up.volume + down.volume,
        "trade_count_1m": up.trade_count + down.trade_count,
        "trade_count_5m": up.trade_count + down.trade_count,
        "orderbook_imbalance_l1": imbalance,
        "liquidity_depth": total_liquidity,
        "minute_of_day": (decision_ts // 60_000) % (24 * 60),
        "day_of_week": (decision_ts // (24 * 60 * 60_000) + 3) % 7,
    }


def _snapshots_by_ts(
    snapshots: tuple[PolymarketTokenSnapshot, ...],
) -> dict[int, dict[str, PolymarketTokenSnapshot]]:
    grouped: dict[int, dict[str, PolymarketTokenSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.ts, {})[snapshot.outcome] = snapshot
    return grouped


def _latest_btc_row(
    rows: tuple[MarketData, ...],
    decision_ts: int,
) -> MarketData | None:
    latest: MarketData | None = None
    for row in rows:
        available_at = row.available_at_ts or row.ts
        if available_at <= decision_ts:
            latest = row
        if row.ts > decision_ts:
            break
    return latest


def _return_over_window(
    rows: tuple[MarketData, ...],
    decision_ts: int,
    window_ms: int,
) -> float:
    now = _latest_btc_row(rows, decision_ts)
    prev = _latest_btc_row(rows, max(0, decision_ts - window_ms))
    if now is None or prev is None:
        return 0.0
    return now.effective_mid_price / prev.effective_mid_price - 1.0


def _volatility(
    rows: tuple[MarketData, ...],
    decision_ts: int,
    window_ms: int,
) -> float:
    prices = [
        row.effective_mid_price
        for row in rows
        if decision_ts - window_ms <= row.ts <= decision_ts
        and (row.available_at_ts or row.ts) <= decision_ts
    ]
    if len(prices) < 2:
        return 0.0
    returns = [
        prices[index] / prices[index - 1] - 1.0
        for index in range(1, len(prices))
        if prices[index - 1] > 0.0
    ]
    if not returns:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return math.sqrt(variance)


def _provenance(
    *,
    feature_name: str,
    input_start_ts: int,
    input_end_ts: int,
    available_at_ts: int,
    lookback_ms: int,
) -> FeatureProvenance:
    return FeatureProvenance(
        feature_name=feature_name,
        input_start_ts=input_start_ts,
        input_end_ts=input_end_ts,
        available_at_ts=available_at_ts,
        lookback_ms=lookback_ms,
        source_timeframe_ms=60_000
        if lookback_ms < POLYMARKET_BTC15M_HORIZON_MS
        else POLYMARKET_BTC15M_HORIZON_MS,
    )
