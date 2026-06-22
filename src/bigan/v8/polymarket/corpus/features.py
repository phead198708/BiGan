"""Point-in-time feature construction for Polymarket BTC corpus rows."""

from __future__ import annotations

from statistics import pstdev

from bigan.v8.polymarket.corpus.contracts import (
    BinanceBTCCandle,
    PolymarketCorpusBookSnapshot,
    PolymarketCorpusBuildConfig,
    PolymarketCorpusFeatureRow,
    PolymarketCorpusMarket,
    PolymarketCorpusTrade,
)


def build_polymarket_corpus_feature_rows(
    *,
    markets: tuple[PolymarketCorpusMarket, ...],
    book_snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    trades: tuple[PolymarketCorpusTrade, ...],
    btc_candles: tuple[BinanceBTCCandle, ...],
    config: PolymarketCorpusBuildConfig,
) -> tuple[PolymarketCorpusFeatureRow, ...]:
    """Build strictly point-in-time feature rows for configured markets."""

    snapshots_by_market = _snapshots_by_market(book_snapshots)
    trades_by_market = _trades_by_market(trades)
    candles = tuple(sorted(btc_candles, key=lambda item: item.ts))
    rows: list[PolymarketCorpusFeatureRow] = []
    for market in sorted(markets, key=lambda item: (item.market_start_ts, item.market_id)):
        if market.market_family not in config.market_families:
            continue
        interval_ms = config.resolved_sample_intervals()[market.market_family] * 1000
        for decision_ts in _sample_times(market=market, interval_ms=interval_ms, config=config):
            up_snapshot = _last_snapshot(
                snapshots_by_market.get((market.market_id, "UP"), ()),
                decision_ts,
            )
            down_snapshot = _last_snapshot(
                snapshots_by_market.get((market.market_id, "DOWN"), ()),
                decision_ts,
            )
            candle = _last_candle(candles, decision_ts)
            if up_snapshot is None or down_snapshot is None or candle is None:
                continue
            market_trades = trades_by_market.get(market.market_id, ())
            rows.append(
                _feature_row(
                    market=market,
                    decision_ts=decision_ts,
                    up_snapshot=up_snapshot,
                    down_snapshot=down_snapshot,
                    market_trades=market_trades,
                    candles=candles,
                    current_candle=candle,
                )
            )
    if not rows:
        raise ValueError("no point-in-time Polymarket corpus feature rows")
    return tuple(sorted(rows, key=lambda item: (item.decision_ts, item.market_id)))


def _feature_row(
    *,
    market: PolymarketCorpusMarket,
    decision_ts: int,
    up_snapshot: PolymarketCorpusBookSnapshot,
    down_snapshot: PolymarketCorpusBookSnapshot,
    market_trades: tuple[PolymarketCorpusTrade, ...],
    candles: tuple[BinanceBTCCandle, ...],
    current_candle: BinanceBTCCandle,
) -> PolymarketCorpusFeatureRow:
    recent_up_volume = _recent_trade_volume(
        trades=market_trades,
        outcome="UP",
        decision_ts=decision_ts,
        lookback_ms=60_000,
    )
    recent_down_volume = _recent_trade_volume(
        trades=market_trades,
        outcome="DOWN",
        decision_ts=decision_ts,
        lookback_ms=60_000,
    )
    up_spread_bps = _spread_bps(up_snapshot)
    down_spread_bps = _spread_bps(down_snapshot)
    liquidity_total = up_snapshot.liquidity_depth + down_snapshot.liquidity_depth
    features: dict[str, float | int | str | None] = {
        "market_family": market.market_family,
        "horizon_ms": market.horizon_ms,
        "time_to_close_seconds": (market.market_end_ts - decision_ts) / 1000.0,
        "market_age_seconds": (decision_ts - market.market_start_ts) / 1000.0,
        "btc_mid_price": current_candle.close_price,
        "btc_return_10s": _return(candles, decision_ts=decision_ts, lookback_ms=10_000),
        "btc_return_30s": _return(candles, decision_ts=decision_ts, lookback_ms=30_000),
        "btc_return_1m": _return(candles, decision_ts=decision_ts, lookback_ms=60_000),
        "btc_return_5m": _return(candles, decision_ts=decision_ts, lookback_ms=300_000),
        "btc_return_15m": _return(candles, decision_ts=decision_ts, lookback_ms=900_000),
        "btc_volatility_1m": _volatility(candles, decision_ts=decision_ts, lookback_ms=60_000),
        "btc_volatility_5m": _volatility(candles, decision_ts=decision_ts, lookback_ms=300_000),
        "btc_volatility_15m": _volatility(candles, decision_ts=decision_ts, lookback_ms=900_000),
        "up_bid": up_snapshot.bid_price,
        "up_ask": up_snapshot.ask_price,
        "up_mid": up_snapshot.mid_price,
        "down_bid": down_snapshot.bid_price,
        "down_ask": down_snapshot.ask_price,
        "down_mid": down_snapshot.mid_price,
        "up_down_mid_sum": up_snapshot.mid_price + down_snapshot.mid_price,
        "up_down_bid_sum": up_snapshot.bid_price + down_snapshot.bid_price,
        "up_down_ask_sum": up_snapshot.ask_price + down_snapshot.ask_price,
        "up_spread_bps": up_spread_bps,
        "down_spread_bps": down_spread_bps,
        "combined_spread_bps": up_spread_bps + down_spread_bps,
        "up_liquidity_depth": up_snapshot.liquidity_depth,
        "down_liquidity_depth": down_snapshot.liquidity_depth,
        "liquidity_imbalance": 0.0
        if liquidity_total <= 0.0
        else (up_snapshot.liquidity_depth - down_snapshot.liquidity_depth) / liquidity_total,
        "recent_up_trade_volume": recent_up_volume,
        "recent_down_trade_volume": recent_down_volume,
    }
    max_trade_ts = max(
        (
            trade.available_at_ts
            for trade in market_trades
            if decision_ts - 60_000 <= trade.available_at_ts <= decision_ts
        ),
        default=0,
    )
    max_input_ts = max(
        up_snapshot.ts,
        down_snapshot.ts,
        current_candle.ts,
        max_trade_ts,
    )
    available_at_ts = max(
        up_snapshot.available_at_ts,
        down_snapshot.available_at_ts,
        current_candle.available_at_ts,
        max_trade_ts,
    )
    provenance = {
        name: {
            "source": "polymarket_corpus",
            "input_start_ts": max(0, decision_ts - _feature_lookback_ms(name)),
            "input_end_ts": max_input_ts,
            "available_at_ts": available_at_ts,
            "lookback_ms": _feature_lookback_ms(name),
        }
        for name in features
    }
    return PolymarketCorpusFeatureRow(
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        market_family=market.market_family,
        horizon_ms=market.horizon_ms,
        decision_ts=decision_ts,
        feature_cutoff_ts=decision_ts,
        max_input_ts=max_input_ts,
        available_at_ts=available_at_ts,
        features=features,
        feature_provenance=provenance,
    )


def _sample_times(
    *,
    market: PolymarketCorpusMarket,
    interval_ms: int,
    config: PolymarketCorpusBuildConfig,
) -> tuple[int, ...]:
    times: list[int] = []
    ts = market.market_start_ts
    while ts < market.market_end_ts:
        time_to_close_seconds = (market.market_end_ts - ts) // 1000
        if time_to_close_seconds >= config.min_time_to_close_seconds and (
            config.max_time_to_close_seconds is None
            or time_to_close_seconds <= config.max_time_to_close_seconds
        ):
            times.append(ts)
        ts += interval_ms
    return tuple(times)


def _snapshots_by_market(
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
) -> dict[tuple[str, str], tuple[PolymarketCorpusBookSnapshot, ...]]:
    grouped: dict[tuple[str, str], list[PolymarketCorpusBookSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault((snapshot.market_id, snapshot.outcome), []).append(snapshot)
    return {
        key: tuple(sorted(value, key=lambda item: item.ts))
        for key, value in grouped.items()
    }


def _trades_by_market(
    trades: tuple[PolymarketCorpusTrade, ...],
) -> dict[str, tuple[PolymarketCorpusTrade, ...]]:
    grouped: dict[str, list[PolymarketCorpusTrade]] = {}
    for trade in trades:
        grouped.setdefault(trade.market_id, []).append(trade)
    return {
        key: tuple(sorted(value, key=lambda item: item.ts))
        for key, value in grouped.items()
    }


def _last_snapshot(
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    decision_ts: int,
) -> PolymarketCorpusBookSnapshot | None:
    eligible = [
        snapshot
        for snapshot in snapshots
        if snapshot.ts <= decision_ts and snapshot.available_at_ts <= decision_ts
    ]
    return eligible[-1] if eligible else None


def _last_candle(
    candles: tuple[BinanceBTCCandle, ...],
    decision_ts: int,
) -> BinanceBTCCandle | None:
    eligible = [
        candle
        for candle in candles
        if candle.ts <= decision_ts and candle.available_at_ts <= decision_ts
    ]
    return eligible[-1] if eligible else None


def _return(
    candles: tuple[BinanceBTCCandle, ...],
    *,
    decision_ts: int,
    lookback_ms: int,
) -> float:
    current = _last_candle(candles, decision_ts)
    previous = _last_candle(candles, max(0, decision_ts - lookback_ms))
    if current is None or previous is None:
        return 0.0
    return current.close_price / previous.close_price - 1.0


def _volatility(
    candles: tuple[BinanceBTCCandle, ...],
    *,
    decision_ts: int,
    lookback_ms: int,
) -> float:
    window = [
        candle
        for candle in candles
        if decision_ts - lookback_ms <= candle.ts <= decision_ts
        and candle.available_at_ts <= decision_ts
    ]
    if len(window) < 2:
        return 0.0
    returns = [
        window[index].close_price / window[index - 1].close_price - 1.0
        for index in range(1, len(window))
    ]
    return pstdev(returns) if len(returns) > 1 else abs(returns[0])


def _spread_bps(snapshot: PolymarketCorpusBookSnapshot) -> float:
    if snapshot.mid_price <= 0.0:
        return 0.0
    return (snapshot.ask_price - snapshot.bid_price) / snapshot.mid_price * 10_000.0


def _recent_trade_volume(
    *,
    trades: tuple[PolymarketCorpusTrade, ...],
    outcome: str,
    decision_ts: int,
    lookback_ms: int,
) -> float:
    return sum(
        trade.size
        for trade in trades
        if trade.outcome == outcome
        and decision_ts - lookback_ms <= trade.available_at_ts <= decision_ts
    )


def _feature_lookback_ms(name: str) -> int:
    if name.endswith("15m"):
        return 900_000
    if name.endswith("5m"):
        return 300_000
    if name.endswith("1m") or name.startswith("recent_"):
        return 60_000
    if name.endswith("30s"):
        return 30_000
    if name.endswith("10s"):
        return 10_000
    if name in {"market_age_seconds", "time_to_close_seconds"}:
        return 0
    if name in {"market_family", "horizon_ms"}:
        return 0
    return 0
