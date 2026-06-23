"""BTC feature candle capture helpers for the raw corpus recorder."""

from __future__ import annotations

from typing import Any

from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig


def mock_btc_feature_candle_rows(
    markets: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> list[dict[str, Any]]:
    """Build deterministic closed BTC feature candles with causal availability."""

    if not markets:
        return []
    timeframe_ms = config.candle_timeframe_ms
    min_ts = min(int(market["market_start_ts"]) for market in markets) - 15 * 60_000
    max_ts = max(int(market["market_end_ts"]) for market in markets)
    rows: list[dict[str, Any]] = []
    sequence = 0
    ts = min_ts
    while ts <= max_ts:
        close = 65_000.0 + sequence * 2.5 + (sequence % 7) * 0.25
        rows.append(
            {
                "ts": ts,
                "close_time": ts + timeframe_ms,
                "available_at_ts": ts + timeframe_ms,
                "open_price": close - 0.75,
                "high_price": close + 1.25,
                "low_price": close - 1.25,
                "close_price": close,
                "volume": 100.0 + sequence,
                "timeframe_ms": timeframe_ms,
                "source": config.btc_feature_candle_source,
            }
        )
        sequence += 1
        ts += timeframe_ms
    return rows


def validate_btc_feature_candles(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: set[str] = set()
    valid: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        ts = int(row["ts"])
        timeframe_ms = int(row.get("timeframe_ms") or 60_000)
        available_at_ts = int(row.get("available_at_ts") or row.get("close_time") or 0)
        if available_at_ts < ts + timeframe_ms:
            reasons.add("future_candle_close_leakage")
            continue
        if ts in seen:
            continue
        seen.add(ts)
        valid.append(row)
    if not valid:
        reasons.add("missing_btc_feature_candles")
    return sorted(valid, key=lambda item: int(item["ts"])), sorted(reasons)
