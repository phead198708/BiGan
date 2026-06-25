"""Wall-clock read-only Polymarket + BTC reference polling for live paper runs."""

from __future__ import annotations

import json
import math
import time
import urllib.request
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from bigan.v8.polymarket.live.binance_reference_feed import BinanceBTCReferenceHTTPFeed
from bigan.v8.polymarket.live.contracts import PolymarketLivePaperConfig


def load_real_live_feed_rows(
    config: PolymarketLivePaperConfig,
    *,
    streaming_writer: Any | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Poll public read-only feeds until duration elapses, then finalize market rows."""

    from bigan.v8.polymarket.recorder.public_provider import (
        PolymarketPublicHTTPRealCorpusProvider,
    )

    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    object.__setattr__(config, "started_at", started_at)

    recorder_config = _recorder_config(config, started_at=started_at)
    provider = PolymarketPublicHTTPRealCorpusProvider(
        max_markets=len(config.market_families),
        timeout_seconds=max(10.0, float(config.poll_interval_seconds)),
        http_timeout_seconds=max(10.0, float(config.poll_interval_seconds)),
        orderbook_snapshot_interval_seconds=float(config.poll_interval_seconds),
        use_rest_orderbooks=True,
    )
    binance_feed = BinanceBTCReferenceHTTPFeed(
        timeout_seconds=max(10.0, float(config.poll_interval_seconds)),
    )
    coinbase_ticker_url = (
        "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
    )

    markets_by_id: dict[str, dict[str, Any]] = {}
    orderbook_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    candle_rows: list[dict[str, Any]] = []

    deadline = time.monotonic() + float(config.duration_seconds)
    if streaming_writer is not None:
        streaming_writer.record_feed_checkpoint(
            stage="waiting_for_feed",
            market_count=0,
            latest_market_id=None,
            orderbook_count=0,
            trade_count=0,
            tick_count=0,
            candle_count=0,
            force=True,
        )
    while time.monotonic() < deadline:
        poll_started = time.monotonic()
        now_ms = int(time.time() * 1000)
        try:
            recorder_markets = provider.market_rows(recorder_config)
            for row in recorder_markets:
                markets_by_id[str(row["market_id"])] = row
        except Exception:
            recorder_markets = []
        if recorder_markets:
            snapshot_ts = now_ms
            try:
                for row in provider.orderbook_rows(recorder_markets, recorder_config):
                    orderbook_rows.append(_live_orderbook_row(row, snapshot_ts=snapshot_ts))
            except Exception:
                pass
            try:
                for row in provider.trade_rows(recorder_markets, recorder_config):
                    trade_rows.append(_live_trade_row(row))
            except Exception:
                pass
        try:
            tick_rows.append(
                _live_tick_row(_fetch_json(coinbase_ticker_url), now_ms=now_ms)
            )
        except Exception:
            with suppress(Exception):
                tick_rows.append(
                    _live_tick_row(binance_feed.fetch_tick_payload(), now_ms=now_ms)
                )

        if streaming_writer is not None:
            streaming_writer.record_feed_checkpoint(
                stage="collecting_feed",
                market_count=len(markets_by_id),
                latest_market_id=(
                    sorted(markets_by_id)[-1] if markets_by_id else None
                ),
                orderbook_count=len(orderbook_rows),
                trade_count=len(trade_rows),
                tick_count=len(tick_rows),
                candle_count=len(candle_rows),
            )

        sleep_seconds = float(config.poll_interval_seconds) - (time.monotonic() - poll_started)
        if sleep_seconds > 0 and time.monotonic() + sleep_seconds < deadline:
            time.sleep(sleep_seconds)

    recorder_markets = list(markets_by_id.values())
    if not recorder_markets:
        raise RuntimeError("real live polling collected zero Polymarket markets")

    resolutions_by_market = {
        str(row["market_id"]): row
        for row in provider.resolution_rows(recorder_markets, recorder_config)
    }
    for market in recorder_markets:
        resolution = resolutions_by_market.get(str(market["market_id"]))
        if resolution is not None:
            candle_rows.append(_live_candle_row(market, resolution))

    market_rows = [
        _live_market_row(
            market,
            resolution=resolutions_by_market.get(str(market["market_id"])),
            now_ms=int(time.time() * 1000),
        )
        for market in recorder_markets
    ]
    if not tick_rows:
        for _url, fetcher in (
            (coinbase_ticker_url, lambda: _fetch_json(coinbase_ticker_url)),
            ("binance", binance_feed.fetch_tick_payload),
        ):
            try:
                tick_rows.append(
                    _live_tick_row(fetcher(), now_ms=int(time.time() * 1000))
                )
                break
            except Exception:
                continue
    if not candle_rows:
        for market in recorder_markets:
            candle_rows.append(_fallback_candle_row(market, tick_rows[-1]))

    return market_rows, orderbook_rows, trade_rows, tick_rows, candle_rows


def _recorder_config(
    config: PolymarketLivePaperConfig,
    *,
    started_at: str,
) -> Any:
    from bigan.v8.polymarket.recorder.contracts import (
        DEFAULT_RECORDER_ENDED_AT,
        PolymarketRealCorpusRecorderConfig,
    )

    return PolymarketRealCorpusRecorderConfig(
        run_id=config.run_id,
        output_dir=config.run_dir,
        created_at=started_at,
        started_at=started_at,
        ended_at=DEFAULT_RECORDER_ENDED_AT,
        market_families=config.market_families,
        mock_public_data=False,
        overwrite_existing=True,
    )


def _live_market_row(
    market: dict[str, Any],
    *,
    resolution: dict[str, Any] | None,
    now_ms: int,
) -> dict[str, Any]:
    end_ts = int(market["market_end_ts"])
    resolved = resolution is not None
    if resolved:
        status = "resolved"
    elif now_ms >= end_ts:
        status = "closed"
    else:
        status = "open"
    reference_price_at_start = _market_reference_price_start(market) or 0.0
    if resolution is not None:
        reference_price_at_start = (
            _optional_positive_float(resolution.get("reference_price_start"))
            or reference_price_at_start
        )
    if reference_price_at_start <= 0.0:
        reference_price_at_start = 1.0
    return {
        "market_id": str(market["market_id"]),
        "condition_id": str(market["condition_id"]),
        "slug": str(market["slug"]),
        "market_family": str(market["market_family"]),
        "horizon_ms": int(market["horizon_ms"]),
        "market_start_ts": int(market["market_start_ts"]),
        "market_end_ts": end_ts,
        "settlement_ts": int(market["settlement_ts"]),
        "up_token_id": str(market["up_token_id"]),
        "down_token_id": str(market["down_token_id"]),
        "reference_price_source": str(
            market.get("reference_price_source") or "binance_btcusdt"
        ),
        "settlement_rule": str(market.get("settlement_rule") or ""),
        "reference_price_at_start": reference_price_at_start,
        "status": status,
        "resolution_available": resolved,
        "raw_market_sha256": str(market.get("raw_market_sha256") or ""),
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _market_reference_price_start(market: dict[str, Any]) -> float | None:
    raw_payload = market.get("raw_public_payload")
    candidates = [market]
    if isinstance(raw_payload, dict):
        candidates.append(raw_payload)
    for candidate in candidates:
        start = _first_positive_float(
            candidate,
            "priceToBeat",
            "price_to_beat",
            "reference_price_start",
            "reference_price_at_start",
            "referencePriceStart",
        )
        if start is not None:
            return start
    return None


def _live_orderbook_row(row: dict[str, Any], *, snapshot_ts: int | None = None) -> dict[str, Any]:
    ts = int(snapshot_ts if snapshot_ts is not None else row["ts"])
    received_ts = int(row.get("available_at_ts") or ts)
    return {
        "market_id": str(row["market_id"]),
        "token_id": str(row["token_id"]),
        "outcome": str(row["outcome"]).upper(),
        "ts": ts,
        "received_ts": received_ts,
        "bid_price": float(row["bid_price"]),
        "ask_price": float(row["ask_price"]),
        "mid_price": float(row["mid_price"]),
        "bid_size": float(row.get("bid_size") or 0.0),
        "ask_size": float(row.get("ask_size") or 0.0),
        "liquidity_depth": float(row.get("liquidity_depth") or 0.0),
        "source": "polymarket_public",
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _live_trade_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": str(row["market_id"]),
        "token_id": str(row["token_id"]),
        "outcome": str(row["outcome"]).upper(),
        "ts": int(row["ts"]),
        "price": float(row["price"]),
        "size": float(row.get("size") or 0.0),
        "side": str(row.get("side") or "UNKNOWN"),
        "source": "polymarket_public",
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bigan-v8-polymarket-live-readonly/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=15.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _first_positive_float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _optional_positive_float(payload.get(key))
        if value is not None:
            return value
    return None


def _optional_positive_float(value: Any) -> float | None:
    numeric = _optional_float(value)
    if numeric is None or numeric <= 0.0 or not math.isfinite(numeric):
        return None
    return numeric


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _live_tick_row(payload: dict[str, Any], *, now_ms: int) -> dict[str, Any]:
    bid = float(payload.get("bid") or payload.get("bidPrice") or 0.0)
    ask = float(payload.get("ask") or payload.get("askPrice") or 0.0)
    if bid <= 0.0 or ask <= 0.0:
        last = float(payload.get("price") or payload.get("last") or 0.0)
        bid = last - 0.5
        ask = last + 0.5
    mid = (bid + ask) / 2.0
    return {
        "ts": now_ms,
        "received_ts": now_ms,
        "bid_price": bid,
        "ask_price": ask,
        "mid_price": mid,
        "last_price": mid,
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _live_candle_row(market: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
    start = float(resolution.get("reference_price_start") or 0.0)
    end = float(resolution.get("reference_price_end") or start)
    return {
        "market_id": str(market["market_id"]),
        "market_family": str(market["market_family"]),
        "open_ts": int(market["market_start_ts"]),
        "close_ts": int(market["market_end_ts"]),
        "open_price": start,
        "close_price": end,
        "high_price": max(start, end),
        "low_price": min(start, end),
        "source": str(resolution.get("reference_price_source") or market["reference_price_source"]),
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _fallback_candle_row(market: dict[str, Any], tick: dict[str, Any]) -> dict[str, Any]:
    price = float(tick["mid_price"])
    return {
        "market_id": str(market["market_id"]),
        "market_family": str(market["market_family"]),
        "open_ts": int(market["market_start_ts"]),
        "close_ts": int(market["market_end_ts"]),
        "open_price": price,
        "close_price": price,
        "high_price": price,
        "low_price": price,
        "source": "binance_btcusdt",
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
