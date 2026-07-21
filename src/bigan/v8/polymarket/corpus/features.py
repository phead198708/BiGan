"""Point-in-time feature construction for Polymarket BTC corpus rows."""

from __future__ import annotations

from statistics import pstdev
from typing import Any

from bigan.v8.polymarket.corpus.contracts import (
    BinanceBTCCandle,
    PolymarketChainlinkPrice,
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
    chainlink_prices: tuple[PolymarketChainlinkPrice, ...] = (),
    config: PolymarketCorpusBuildConfig,
) -> tuple[PolymarketCorpusFeatureRow, ...]:
    """Build strictly point-in-time feature rows for configured markets."""

    snapshots_by_market = _snapshots_by_market(book_snapshots)
    trades_by_market = _trades_by_market(trades)
    candles = tuple(sorted(btc_candles, key=lambda item: item.ts))
    chainlink = tuple(
        sorted(chainlink_prices, key=lambda item: (item.source_ts, item.available_at_ts))
    )
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
                    up_snapshots=snapshots_by_market.get((market.market_id, "UP"), ()),
                    down_snapshots=snapshots_by_market.get((market.market_id, "DOWN"), ()),
                    market_trades=market_trades,
                    candles=candles,
                    current_candle=candle,
                    chainlink_prices=chainlink,
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
    up_snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    down_snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    market_trades: tuple[PolymarketCorpusTrade, ...],
    candles: tuple[BinanceBTCCandle, ...],
    current_candle: BinanceBTCCandle,
    chainlink_prices: tuple[PolymarketChainlinkPrice, ...],
) -> PolymarketCorpusFeatureRow:
    reference_context = _chainlink_reference_price_context(
        market=market,
        chainlink_prices=chainlink_prices,
        decision_ts=decision_ts,
    ) or _reference_price_to_beat_context(
        market=market, candles=candles, decision_ts=decision_ts
    )
    reference_price_to_beat = (
        float(reference_context["reference_price_to_beat"])
        if reference_context is not None
        else None
    )
    reference_current_price = (
        float(reference_context["current_price_at_decision"])
        if reference_context is not None
        and reference_context.get("current_price_at_decision") is not None
        else current_candle.close_price
    )
    reference_distance = (
        (reference_current_price - reference_price_to_beat) / reference_price_to_beat
        if reference_price_to_beat is not None and reference_price_to_beat > 0.0
        else None
    )
    trade_volume_coverage = _recent_trade_volume_coverage(
        market=market,
        decision_ts=decision_ts,
        lookback_ms=60_000,
    )
    recent_up_volume = (
        _recent_trade_volume(
            trades=market_trades,
            outcome="UP",
            decision_ts=decision_ts,
            lookback_ms=60_000,
        )
        if trade_volume_coverage["use_volume"]
        else None
    )
    recent_down_volume = (
        _recent_trade_volume(
            trades=market_trades,
            outcome="DOWN",
            decision_ts=decision_ts,
            lookback_ms=60_000,
        )
        if trade_volume_coverage["use_volume"]
        else None
    )
    up_spread_bps = _spread_bps(up_snapshot)
    down_spread_bps = _spread_bps(down_snapshot)
    liquidity_total = up_snapshot.liquidity_depth + down_snapshot.liquidity_depth
    up_executable_bid_notional = up_snapshot.bid_price * up_snapshot.bid_size
    down_executable_bid_notional = down_snapshot.bid_price * down_snapshot.bid_size
    up_executable_ask_notional = up_snapshot.ask_price * up_snapshot.ask_size
    down_executable_ask_notional = down_snapshot.ask_price * down_snapshot.ask_size
    features: dict[str, float | int | str | None] = {
        "market_family": market.market_family,
        "horizon_ms": market.horizon_ms,
        "time_to_close_seconds": (market.market_end_ts - decision_ts) / 1000.0,
        "market_age_seconds": (decision_ts - market.market_start_ts) / 1000.0,
        "btc_mid_price": current_candle.close_price,
        "reference_price_to_beat": reference_price_to_beat,
        "reference_price_to_beat_distance_at_decision": reference_distance,
        "chainlink_price_at_decision": (
            reference_current_price
            if reference_context is not None
            and reference_context.get("source_type")
            == "polymarket_rtds_chainlink_market_start"
            else None
        ),
        "chainlink_reference_price_at_market_start": (
            reference_price_to_beat
            if reference_context is not None
            and reference_context.get("source_type")
            == "polymarket_rtds_chainlink_market_start"
            else None
        ),
        "chainlink_reference_distance_at_decision": (
            reference_distance
            if reference_context is not None
            and reference_context.get("source_type")
            == "polymarket_rtds_chainlink_market_start"
            else None
        ),
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
        "up_bid_size": up_snapshot.bid_size,
        "up_ask_size": up_snapshot.ask_size,
        "down_bid": down_snapshot.bid_price,
        "down_ask": down_snapshot.ask_price,
        "down_mid": down_snapshot.mid_price,
        "down_bid_size": down_snapshot.bid_size,
        "down_ask_size": down_snapshot.ask_size,
        "up_down_mid_sum": up_snapshot.mid_price + down_snapshot.mid_price,
        "up_down_bid_sum": up_snapshot.bid_price + down_snapshot.bid_price,
        "up_down_ask_sum": up_snapshot.ask_price + down_snapshot.ask_price,
        "up_spread_bps": up_spread_bps,
        "down_spread_bps": down_spread_bps,
        "combined_spread_bps": up_spread_bps + down_spread_bps,
        "up_liquidity_depth": up_snapshot.liquidity_depth,
        "down_liquidity_depth": down_snapshot.liquidity_depth,
        "up_executable_bid_notional": up_executable_bid_notional,
        "down_executable_bid_notional": down_executable_bid_notional,
        "up_executable_ask_notional": up_executable_ask_notional,
        "down_executable_ask_notional": down_executable_ask_notional,
        "up_queue_fill_probability_proxy": _queue_fill_probability_proxy(up_snapshot),
        "down_queue_fill_probability_proxy": _queue_fill_probability_proxy(down_snapshot),
        "up_book_staleness_ms": decision_ts - up_snapshot.ts,
        "down_book_staleness_ms": decision_ts - down_snapshot.ts,
        "up_book_update_lag_ms": decision_ts - up_snapshot.available_at_ts,
        "down_book_update_lag_ms": decision_ts - down_snapshot.available_at_ts,
        "book_snapshot_pair_ts_delta_ms": abs(up_snapshot.ts - down_snapshot.ts),
        "up_recent_book_update_count_1m": _recent_book_update_count(
            snapshots=up_snapshots,
            decision_ts=decision_ts,
            lookback_ms=60_000,
        ),
        "down_recent_book_update_count_1m": _recent_book_update_count(
            snapshots=down_snapshots,
            decision_ts=decision_ts,
            lookback_ms=60_000,
        ),
        "up_recent_bid_depth_volatility_1m": _recent_depth_volatility(
            snapshots=up_snapshots,
            decision_ts=decision_ts,
            lookback_ms=60_000,
        ),
        "down_recent_bid_depth_volatility_1m": _recent_depth_volatility(
            snapshots=down_snapshots,
            decision_ts=decision_ts,
            lookback_ms=60_000,
        ),
        "up_recent_spread_stability_1m": _recent_spread_stability(
            snapshots=up_snapshots,
            decision_ts=decision_ts,
            lookback_ms=60_000,
        ),
        "down_recent_spread_stability_1m": _recent_spread_stability(
            snapshots=down_snapshots,
            decision_ts=decision_ts,
            lookback_ms=60_000,
        ),
        "up_empty_book_flag": float(
            up_snapshot.bid_size <= 0.0 or up_snapshot.ask_size <= 0.0
        ),
        "down_empty_book_flag": float(
            down_snapshot.bid_size <= 0.0 or down_snapshot.ask_size <= 0.0
        ),
        "up_crossed_or_locked_book_flag": float(
            up_snapshot.bid_price >= up_snapshot.ask_price
        ),
        "down_crossed_or_locked_book_flag": float(
            down_snapshot.bid_price >= down_snapshot.ask_price
        ),
        "liquidity_imbalance": 0.0
        if liquidity_total <= 0.0
        else (up_snapshot.liquidity_depth - down_snapshot.liquidity_depth) / liquidity_total,
        "recent_up_trade_volume": recent_up_volume,
        "recent_down_trade_volume": recent_down_volume,
        "recent_trade_volume_coverage_complete": trade_volume_coverage[
            "coverage_complete"
        ],
        "recent_trade_volume_censored": trade_volume_coverage["censored"],
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
        int(reference_context["max_input_ts"]) if reference_context else 0,
        int(reference_context.get("current_source_ts") or 0)
        if reference_context
        else 0,
        max_trade_ts,
    )
    available_at_ts = max(
        up_snapshot.available_at_ts,
        down_snapshot.available_at_ts,
        current_candle.available_at_ts,
        int(reference_context["available_at_ts"]) if reference_context else 0,
        int(reference_context.get("current_available_at_ts") or 0)
        if reference_context
        else 0,
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
    for feature_name in (
        "recent_up_trade_volume",
        "recent_down_trade_volume",
        "recent_trade_volume_coverage_complete",
        "recent_trade_volume_censored",
    ):
        provenance[feature_name].update(
            {
                "trade_coverage_required_start_ts": trade_volume_coverage[
                    "required_start_ts"
                ],
                "trade_stream_started_at_ts": market.trade_stream_started_at_ts,
                "trade_stream_ended_at_ts": market.trade_stream_ended_at_ts,
                "trade_stream_continuity_passed": (
                    market.trade_stream_continuity_passed
                ),
                "trade_tape_censored": market.trade_tape_censored,
                "trade_collection_reason_codes": list(
                    market.trade_collection_reason_codes
                ),
            }
        )
    if reference_context is not None:
        reference_provenance = {
            "source": "polymarket_corpus",
            "input_start_ts": int(reference_context["input_start_ts"]),
            "input_end_ts": max(
                int(reference_context["input_end_ts"]),
                int(reference_context.get("current_source_ts") or current_candle.ts),
            ),
            "available_at_ts": max(
                int(reference_context["available_at_ts"]),
                int(
                    reference_context.get("current_available_at_ts")
                    or current_candle.available_at_ts
                ),
            ),
            "lookback_ms": max(0, decision_ts - int(reference_context["input_start_ts"])),
            "source_fields_used": "|".join(
                (
                    str(reference_context["source_fields_used"]),
                    str(
                        reference_context.get("current_source_fields_used")
                        or "polymarket_btc_reference_candles.close_price_at_decision"
                    ),
                )
            ),
            "max_input_ts": max(
                int(reference_context["max_input_ts"]),
                int(reference_context.get("current_source_ts") or current_candle.ts),
            ),
            "decision_ts": decision_ts,
            "provenance_valid": (
                max(
                    int(reference_context["available_at_ts"]),
                    int(
                        reference_context.get("current_available_at_ts")
                        or current_candle.available_at_ts
                    ),
                )
                <= decision_ts
            ),
            "reference_price_to_beat_source": str(reference_context["source_type"]),
        }
        provenance["reference_price_to_beat"] = {
            **reference_provenance,
            "source_fields_used": str(reference_context["source_fields_used"]),
            "input_end_ts": int(reference_context["input_end_ts"]),
            "available_at_ts": int(reference_context["available_at_ts"]),
            "max_input_ts": int(reference_context["max_input_ts"]),
        }
        provenance["reference_price_to_beat_distance_at_decision"] = (
            reference_provenance
        )
        if (
            reference_context.get("source_type")
            == "polymarket_rtds_chainlink_market_start"
        ):
            for feature_name in (
                "chainlink_price_at_decision",
                "chainlink_reference_price_at_market_start",
                "chainlink_reference_distance_at_decision",
            ):
                provenance[feature_name] = dict(reference_provenance)
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


def _reference_price_to_beat_context(
    *,
    market: PolymarketCorpusMarket,
    candles: tuple[BinanceBTCCandle, ...],
    decision_ts: int,
) -> dict[str, float | int | str] | None:
    if market.reference_price_start is not None:
        return {
            "reference_price_to_beat": market.reference_price_start,
            "input_start_ts": market.market_start_ts,
            "input_end_ts": market.market_start_ts,
            "available_at_ts": market.market_start_ts,
            "max_input_ts": market.market_start_ts,
            "source_fields_used": "polymarket_market_metadata.reference_price_start",
            "source_type": "market_metadata_reference_price_start",
        }
    start_open = _market_start_open_candle(candles, market.market_start_ts)
    if start_open is not None and start_open.ts <= decision_ts:
        return {
            "reference_price_to_beat": start_open.open_price,
            "input_start_ts": start_open.ts,
            "input_end_ts": start_open.ts,
            "available_at_ts": start_open.ts,
            "max_input_ts": start_open.ts,
            "source_fields_used": "polymarket_btc_reference_candles.open_price_at_market_start",
            "source_type": "market_start_reference_candle_open_price",
        }
    eligible_prior = [
        candle
        for candle in candles
        if candle.ts <= market.market_start_ts
        and candle.available_at_ts <= decision_ts
    ]
    if not eligible_prior:
        return None
    prior = eligible_prior[-1]
    return {
        "reference_price_to_beat": prior.close_price,
        "input_start_ts": prior.ts,
        "input_end_ts": prior.ts,
        "available_at_ts": prior.available_at_ts,
        "max_input_ts": prior.ts,
        "source_fields_used": "polymarket_btc_reference_candles.close_price_before_market_start",
        "source_type": "prior_available_reference_candle_close_price",
    }


def _chainlink_reference_price_context(
    *,
    market: PolymarketCorpusMarket,
    chainlink_prices: tuple[PolymarketChainlinkPrice, ...],
    decision_ts: int,
) -> dict[str, float | int | str] | None:
    reference_rows = [
        row
        for row in chainlink_prices
        if row.source_ts <= market.market_start_ts
        and row.available_at_ts <= decision_ts
    ]
    current_rows = [
        row
        for row in chainlink_prices
        if row.source_ts <= decision_ts and row.available_at_ts <= decision_ts
    ]
    if not reference_rows or not current_rows:
        return None
    reference = reference_rows[-1]
    current = current_rows[-1]
    return {
        "reference_price_to_beat": reference.price,
        "current_price_at_decision": current.price,
        "input_start_ts": reference.source_ts,
        "input_end_ts": reference.source_ts,
        "available_at_ts": reference.available_at_ts,
        "max_input_ts": reference.source_ts,
        "current_source_ts": current.source_ts,
        "current_available_at_ts": current.available_at_ts,
        "source_fields_used": (
            "raw_polymarket_chainlink_prices.price_at_or_before_market_start"
        ),
        "current_source_fields_used": (
            "raw_polymarket_chainlink_prices.price_at_or_before_decision"
        ),
        "source_type": "polymarket_rtds_chainlink_market_start",
    }


def _market_start_open_candle(
    candles: tuple[BinanceBTCCandle, ...],
    market_start_ts: int,
) -> BinanceBTCCandle | None:
    for candle in candles:
        if candle.ts == market_start_ts and candle.open_price > 0.0:
            return candle
        if candle.ts > market_start_ts:
            break
    return None


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


def _recent_trade_volume_coverage(
    *,
    market: PolymarketCorpusMarket,
    decision_ts: int,
    lookback_ms: int,
) -> dict[str, Any]:
    metadata_available = any(
        value is not None
        for value in (
            market.trade_stream_started_at_ts,
            market.trade_stream_ended_at_ts,
            market.trade_stream_continuity_passed,
            market.trade_tape_censored,
        )
    )
    required_start_ts = max(market.market_start_ts, decision_ts - lookback_ms)
    if not metadata_available:
        return {
            "use_volume": True,
            "coverage_complete": None,
            "censored": None,
            "required_start_ts": required_start_ts,
        }
    coverage_complete = bool(
        market.trade_stream_continuity_passed is True
        and market.trade_stream_started_at_ts is not None
        and market.trade_stream_ended_at_ts is not None
        and market.trade_stream_started_at_ts <= required_start_ts
        and market.trade_stream_ended_at_ts >= decision_ts
    )
    return {
        "use_volume": coverage_complete,
        "coverage_complete": float(coverage_complete),
        "censored": float(not coverage_complete),
        "required_start_ts": required_start_ts,
    }


def _recent_book_update_count(
    *,
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    decision_ts: int,
    lookback_ms: int,
) -> int:
    return sum(
        1
        for snapshot in snapshots
        if decision_ts - lookback_ms <= snapshot.available_at_ts <= decision_ts
        and snapshot.ts <= decision_ts
    )


def _recent_depth_volatility(
    *,
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    decision_ts: int,
    lookback_ms: int,
) -> float:
    depths = [
        snapshot.bid_size
        for snapshot in snapshots
        if decision_ts - lookback_ms <= snapshot.available_at_ts <= decision_ts
        and snapshot.ts <= decision_ts
    ]
    if len(depths) < 2:
        return 0.0
    return pstdev(depths)


def _recent_spread_stability(
    *,
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    decision_ts: int,
    lookback_ms: int,
) -> float:
    spreads = [
        _spread_bps(snapshot)
        for snapshot in snapshots
        if decision_ts - lookback_ms <= snapshot.available_at_ts <= decision_ts
        and snapshot.ts <= decision_ts
    ]
    if len(spreads) < 2:
        return 1.0
    return 1.0 / (1.0 + pstdev(spreads))


def _queue_fill_probability_proxy(snapshot: PolymarketCorpusBookSnapshot) -> float:
    executable_notional = snapshot.bid_price * snapshot.bid_size
    size_score = min(1.0, executable_notional)
    depth_score = min(1.0, snapshot.liquidity_depth / 2.0)
    spread_score = max(0.0, 1.0 - _spread_bps(snapshot) / 2_000.0)
    return max(0.0, min(1.0, 0.60 * size_score + 0.30 * depth_score + 0.10 * spread_score))


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
