"""Read-only provider contracts for real Polymarket corpus recording."""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

import orjson
import websockets

from bigan.ingestion.message_types import (
    BestBidAskEvent,
    BookEvent,
    PriceChangeEvent,
    UnknownEvent,
    parse_event,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.corpus.contracts import safety_fields
from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig

BTC_UPDOWN_SLUG_PATTERN = re.compile(r"^btc-updown-(5m|15m|1h)-(\d+)$")
BTC_UPDOWN_FAMILY_BY_SLUG = {
    "5m": "btc_updown_5m",
    "15m": "btc_updown_15m",
    "1h": "btc_updown_1h",
}
DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_CLOB_WS_SOURCE_CHANNEL = "polymarket_clob_ws_market"


class RealCorpusPublicProviderError(RuntimeError):
    """Raised when a public read-only provider cannot safely normalize rows."""

    def __init__(self, message: str, *, reason_codes: tuple[str, ...]) -> None:
        super().__init__(message)
        self.reason_codes = reason_codes


class PolymarketOrderBookSource(Protocol):
    """Read-only source for CLOB book snapshots keyed by token id."""

    def book_payloads(self, token_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """Return CLOB-like book payloads keyed by token id."""


class PolymarketRealCorpusPublicProvider(Protocol):
    """Normalized read-only public-data provider for the recorder operator.

    Implementations must return rows already normalized to the recorder raw
    contracts. The operator still validates every row and fails closed on
    provider exceptions or unsafe provider flags.
    """

    read_only: bool
    write_capable: bool
    paper_only: bool
    capital_at_risk: bool
    broker_exchange_write_enabled: bool
    live_exchange_write_enabled: bool
    polymarket_write_enabled: bool
    wallet_signing_enabled: bool

    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized Polymarket Gamma market metadata rows."""

    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized Polymarket CLOB orderbook rows."""

    def trade_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized Polymarket CLOB trade rows."""

    def btc_feature_candle_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized causal BTC feature candle rows."""

    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        """Return normalized official Polymarket resolution/reference rows."""


class PolymarketPublicHTTPRealCorpusProvider:
    """Read-only public provider for Polymarket BTC UP/DOWN corpus facts.

    This provider reads Gamma/Data API/CLOB websocket/Binance endpoints and
    normalizes what those endpoints actually expose. It does not synthesize
    historical bid/ask orderbooks from price history, and it does not use
    Binance as the official Polymarket settlement source.
    """

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(
        self,
        *,
        market_slugs: tuple[str, ...] = (),
        gamma_markets_endpoint: str = "https://gamma-api.polymarket.com/markets",
        clob_book_endpoint: str = "https://clob.polymarket.com/book",
        clob_ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
        data_trades_endpoint: str = "https://data-api.polymarket.com/trades",
        binance_klines_endpoint: str = "https://api.binance.com/api/v3/klines",
        max_markets: int = 3,
        recent_trade_limit: int = 250,
        timeout_seconds: float = 15.0,
        current_time_ms: int | None = None,
        fetch_json: Callable[[str], Any] | None = None,
        orderbook_source: PolymarketOrderBookSource | None = None,
        use_rest_orderbooks: bool = False,
    ) -> None:
        if max_markets <= 0:
            raise ValueError("max_markets must be positive")
        if recent_trade_limit <= 0:
            raise ValueError("recent_trade_limit must be positive")
        self.market_slugs = tuple(dict.fromkeys(slug.strip() for slug in market_slugs if slug.strip()))
        self.gamma_markets_endpoint = gamma_markets_endpoint
        self.clob_book_endpoint = clob_book_endpoint
        self.clob_ws_url = clob_ws_url
        self.data_trades_endpoint = data_trades_endpoint
        self.binance_klines_endpoint = binance_klines_endpoint
        self.max_markets = max_markets
        self.recent_trade_limit = recent_trade_limit
        self.timeout_seconds = timeout_seconds
        self.current_time_ms = current_time_ms
        self._fetch_json = fetch_json
        if orderbook_source is not None:
            self.orderbook_source = orderbook_source
        elif use_rest_orderbooks:
            self.orderbook_source = None
        else:
            self.orderbook_source = PolymarketCLOBWebSocketOrderBookSource(
                ws_url=clob_ws_url,
                timeout_seconds=timeout_seconds,
            )

    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        slugs = self.market_slugs or self._discover_recent_btc_updown_slugs()
        rows = []
        for slug in slugs[: self.max_markets]:
            payloads = self._fetch_gamma_market_payloads(slug)
            rows.extend(
                row
                for row in (
                    self._normalize_gamma_market_payload(payload, config)
                    for payload in payloads
                )
                if row is not None
            )
        if not rows:
            raise RealCorpusPublicProviderError(
                "No BTC UP/DOWN Gamma markets could be normalized from public data.",
                reason_codes=("real_public_collection_empty_market_discovery",),
            )
        return rows[: self.max_markets]

    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        if self.orderbook_source is not None:
            return self._orderbook_rows_from_source(markets)
        return self._orderbook_rows_from_rest(markets)

    def _orderbook_rows_from_source(self, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payloads = self.orderbook_source.book_payloads(_token_ids_for_markets(markets))
        rows: list[dict[str, Any]] = []
        for market in markets:
            for outcome, token_id in (
                ("UP", str(market["up_token_id"])),
                ("DOWN", str(market["down_token_id"])),
            ):
                payload = payloads.get(token_id)
                if payload is None:
                    continue
                row = self._normalize_book_payload(
                    market=market,
                    outcome=outcome,
                    token_id=token_id,
                    payload=payload,
                )
                if row is not None:
                    rows.append(row)
        return rows

    def _orderbook_rows_from_rest(self, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for market in markets:
            for outcome, token_id in (
                ("UP", str(market["up_token_id"])),
                ("DOWN", str(market["down_token_id"])),
            ):
                payload = self._fetch_clob_book(token_id)
                row = self._normalize_book_payload(
                    market=market,
                    outcome=outcome,
                    token_id=token_id,
                    payload=payload,
                )
                if row is not None:
                    rows.append(row)
        return rows

    def trade_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        rows: list[dict[str, Any]] = []
        by_condition = {str(market["condition_id"]): market for market in markets}
        for condition_id, market in by_condition.items():
            params = urllib.parse.urlencode(
                {"market": condition_id, "limit": self.recent_trade_limit}
            )
            payload = self._get_json(f"{self.data_trades_endpoint}?{params}")
            trades = payload if isinstance(payload, list) else []
            for trade in trades:
                row = self._normalize_trade_payload(market=market, payload=trade)
                if row is not None:
                    rows.append(row)
        return rows

    def btc_feature_candle_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        if not markets:
            return []
        timeframe_ms = config.candle_timeframe_ms
        interval = _binance_interval(timeframe_ms)
        min_ts = min(int(market["market_start_ts"]) for market in markets) - 15 * 60_000
        max_ts = max(int(market["market_end_ts"]) for market in markets)
        end_ts = min(max_ts + timeframe_ms, self._current_time_ms())
        if end_ts <= min_ts:
            return []
        params = urllib.parse.urlencode(
            {
                "symbol": "BTCUSDT",
                "interval": interval,
                "startTime": min_ts,
                "endTime": end_ts,
                "limit": 1000,
            }
        )
        payload = self._get_json(f"{self.binance_klines_endpoint}?{params}")
        if not isinstance(payload, list):
            raise RealCorpusPublicProviderError(
                "Invalid Binance klines public payload.",
                reason_codes=("invalid_btc_feature_candle_payload",),
            )
        return [
            row
            for row in (
                self._normalize_binance_kline(row, timeframe_ms, config) for row in payload
            )
            if row is not None
        ]

    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        rows = []
        for market in markets:
            raw_payload = dict(market.get("raw_public_payload") or {})
            start = _optional_float(
                raw_payload.get("referencePriceStart")
                or raw_payload.get("reference_price_start")
                or raw_payload.get("reference_price_at_start")
            )
            end = _optional_float(
                raw_payload.get("referencePriceEnd")
                or raw_payload.get("reference_price_end")
                or raw_payload.get("reference_price_at_end")
            )
            if start is None or end is None:
                continue
            rows.append(
                {
                    "market_id": market["market_id"],
                    "reference_price_start": start,
                    "reference_price_end": end,
                    "reference_price_source": market["reference_price_source"],
                    "resolution_status": _resolution_status_from_payload(raw_payload),
                    "raw_resolution_text": str(raw_payload.get("description") or ""),
                    **safety_fields(),
                }
            )
        return rows

    def _discover_recent_btc_updown_slugs(self) -> tuple[str, ...]:
        params = urllib.parse.urlencode({"limit": self.recent_trade_limit})
        payload = self._get_json(f"{self.data_trades_endpoint}?{params}")
        if not isinstance(payload, list):
            raise RealCorpusPublicProviderError(
                "Invalid Polymarket Data API trades payload.",
                reason_codes=("invalid_polymarket_trade_payload",),
            )
        slugs: list[str] = []
        for trade in payload:
            slug = str(dict(trade).get("slug") or "")
            if BTC_UPDOWN_SLUG_PATTERN.match(slug) and slug not in slugs:
                slugs.append(slug)
        if not slugs:
            raise RealCorpusPublicProviderError(
                "No recent BTC UP/DOWN trade slugs were found in public data.",
                reason_codes=("real_public_collection_empty_market_discovery",),
            )
        return tuple(slugs)

    def _fetch_gamma_market_payloads(self, slug: str) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({"slug": slug})
        payload = self._get_json(f"{self.gamma_markets_endpoint}?{params}")
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("markets", []))
            if isinstance(data, list):
                return [dict(row) for row in data]
        raise RealCorpusPublicProviderError(
            "Invalid Gamma markets payload.",
            reason_codes=("invalid_gamma_market_payload",),
        )

    def _fetch_clob_book(self, token_id: str) -> dict[str, Any]:
        params = urllib.parse.urlencode({"token_id": token_id})
        payload = self._get_json(f"{self.clob_book_endpoint}?{params}")
        if not isinstance(payload, dict):
            raise RealCorpusPublicProviderError(
                "Invalid CLOB orderbook payload.",
                reason_codes=("invalid_polymarket_orderbook_payload",),
            )
        return dict(payload)

    def _normalize_gamma_market_payload(
        self,
        payload: dict[str, Any],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> dict[str, Any] | None:
        slug = str(payload.get("slug") or payload.get("market_slug") or "")
        match = BTC_UPDOWN_SLUG_PATTERN.match(slug)
        if not match:
            return None
        family = BTC_UPDOWN_FAMILY_BY_SLUG[match.group(1)]
        if family not in config.market_families:
            return None
        start_ts = int(match.group(2)) * 1000
        horizon_ms = BTC_UPDOWN_MARKET_HORIZONS_MS[family]
        end_ts = start_ts + horizon_ms
        outcomes = _json_list(payload.get("outcomes"))
        token_ids = _json_list(payload.get("clobTokenIds"))
        token_by_outcome = _token_by_up_down_outcome(outcomes=outcomes, token_ids=token_ids)
        if token_by_outcome is None:
            return None
        condition_id = str(payload.get("conditionId") or payload.get("condition_id") or "")
        if not condition_id:
            return None
        reference_source = str(payload.get("resolutionSource") or "").strip()
        return {
            "market_id": condition_id,
            "condition_id": condition_id,
            "slug": slug,
            "market_family": family,
            "horizon_ms": horizon_ms,
            "market_start_ts": start_ts,
            "market_end_ts": end_ts,
            "settlement_ts": _settlement_ts(payload, default=end_ts + 60_000),
            "up_token_id": token_by_outcome["UP"],
            "down_token_id": token_by_outcome["DOWN"],
            "reference_price_source": reference_source,
            "settlement_rule": str(payload.get("description") or payload.get("question") or ""),
            "raw_market_sha256": canonical_json_sha256(payload),
            "raw_public_payload": payload,
            **safety_fields(),
        }

    def _normalize_book_payload(
        self,
        *,
        market: dict[str, Any],
        outcome: str,
        token_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        bids = _price_levels(payload.get("bids"))
        asks = _price_levels(payload.get("asks") if "asks" in payload else payload.get("ask"))
        if not bids or not asks:
            return None
        best_bid_price, best_bid_size = max(bids, key=lambda level: level[0])
        best_ask_price, best_ask_size = min(asks, key=lambda level: level[0])
        timestamp = int(payload.get("timestamp") or self._current_time_ms())
        return {
            "market_id": market["market_id"],
            "token_id": token_id,
            "outcome": outcome,
            "ts": timestamp,
            "available_at_ts": timestamp,
            "bid_price": best_bid_price,
            "ask_price": best_ask_price,
            "mid_price": round((best_bid_price + best_ask_price) / 2.0, 8),
            "bid_size": best_bid_size,
            "ask_size": best_ask_size,
            "liquidity_depth": round(sum(size for _, size in bids + asks), 8),
            **safety_fields(),
        }

    def _normalize_trade_payload(
        self,
        *,
        market: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        token_id = str(payload.get("asset") or payload.get("token_id") or "")
        outcome = _expected_outcome_for_token(market=market, token_id=token_id)
        if outcome is None:
            return None
        price = _optional_float(payload.get("price"))
        size = _optional_float(payload.get("size"))
        if price is None or size is None:
            return None
        timestamp = int(float(payload.get("timestamp") or 0) * 1000)
        if timestamp <= 0:
            return None
        return {
            "market_id": market["market_id"],
            "token_id": token_id,
            "outcome": outcome,
            "ts": timestamp,
            "available_at_ts": timestamp,
            "price": price,
            "size": size,
            "side": str(payload.get("side") or "").upper(),
            **safety_fields(),
        }

    def _normalize_binance_kline(
        self,
        row: Any,
        timeframe_ms: int,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> dict[str, Any] | None:
        if not isinstance(row, list | tuple) or len(row) < 6:
            return None
        ts = int(row[0])
        close_time = ts + timeframe_ms
        return {
            "ts": ts,
            "close_time": close_time,
            "available_at_ts": close_time,
            "open_price": float(row[1]),
            "high_price": float(row[2]),
            "low_price": float(row[3]),
            "close_price": float(row[4]),
            "volume": float(row[5]),
            "timeframe_ms": timeframe_ms,
            "source": config.btc_feature_candle_source,
        }

    def _get_json(self, url: str) -> Any:
        if self._fetch_json is not None:
            return self._fetch_json(url)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "bigan-v8-polymarket-real-corpus-readonly/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _current_time_ms(self) -> int:
        return self.current_time_ms if self.current_time_ms is not None else int(time.time() * 1000)


class PolymarketCLOBWebSocketOrderBookSource:
    """Short-lived read-only collector for CLOB market-channel book snapshots."""

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    _KEEPALIVE_TOKENS = (b"PONG", b"PING", b"pong", b"ping")

    def __init__(
        self,
        *,
        ws_url: str = DEFAULT_POLYMARKET_CLOB_WS_MARKET_URL,
        timeout_seconds: float = 15.0,
        custom_feature_enabled: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not ws_url.strip():
            raise ValueError("ws_url is required")
        self.ws_url = ws_url
        self.timeout_seconds = timeout_seconds
        self.custom_feature_enabled = custom_feature_enabled

    def book_payloads(self, token_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        token_ids = tuple(dict.fromkeys(str(token_id) for token_id in token_ids if str(token_id)))
        if not token_ids:
            return {}
        try:
            return asyncio.run(self._collect_book_payloads(token_ids))
        except RealCorpusPublicProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RealCorpusPublicProviderError(
                f"CLOB websocket orderbook collection failed: {exc}",
                reason_codes=("polymarket_clob_ws_orderbook_collection_failed",),
            ) from exc

    async def _collect_book_payloads(
        self,
        token_ids: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        target_tokens = set(token_ids)
        book_payloads: dict[str, dict[str, Any]] = {}
        fallback_payloads: dict[str, dict[str, Any]] = {}
        deadline = time.monotonic() + self.timeout_seconds
        try:
            async with websockets.connect(
                self.ws_url,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
                max_size=2**24,
            ) as ws:
                await ws.send(
                    orjson.dumps(
                        {
                            "assets_ids": sorted(target_tokens),
                            "type": "market",
                            "custom_feature_enabled": self.custom_feature_enabled,
                        }
                    )
                )
                while time.monotonic() < deadline and set(book_payloads) != target_tokens:
                    timeout = max(0.001, deadline - time.monotonic())
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except TimeoutError:
                        break
                    receive_time_ms = int(time.time() * 1000)
                    for payload in _decode_market_ws_payloads(raw):
                        self._update_payload_maps(
                            payload=payload,
                            receive_time_ms=receive_time_ms,
                            target_tokens=target_tokens,
                            book_payloads=book_payloads,
                            fallback_payloads=fallback_payloads,
                        )
        except Exception as exc:  # noqa: BLE001
            raise RealCorpusPublicProviderError(
                f"CLOB websocket orderbook collection failed: {exc}",
                reason_codes=("polymarket_clob_ws_orderbook_collection_failed",),
            ) from exc

        merged = dict(fallback_payloads)
        merged.update(book_payloads)
        if not merged:
            raise RealCorpusPublicProviderError(
                "CLOB websocket emitted no orderbook payloads before timeout.",
                reason_codes=("polymarket_clob_ws_no_orderbooks",),
            )
        return {token_id: merged[token_id] for token_id in token_ids if token_id in merged}

    def _update_payload_maps(
        self,
        *,
        payload: dict[str, Any],
        receive_time_ms: int,
        target_tokens: set[str],
        book_payloads: dict[str, dict[str, Any]],
        fallback_payloads: dict[str, dict[str, Any]],
    ) -> None:
        normalized_payload = _market_ws_payload_for_parse(
            payload=payload,
            receive_time_ms=receive_time_ms,
        )
        if normalized_payload is None:
            return
        try:
            event = parse_event(normalized_payload, receive_time_ms=receive_time_ms)
        except UnknownEvent:
            return
        except Exception:
            return
        if isinstance(event, BookEvent) and event.asset_id in target_tokens:
            book_payloads[event.asset_id] = _book_event_payload(event)
            return
        if isinstance(event, BestBidAskEvent) and event.asset_id in target_tokens:
            top_payload = _top_of_book_payload(
                asset_id=event.asset_id,
                market=event.market,
                timestamp=int(event.timestamp),
                best_bid=event.best_bid,
                best_ask=event.best_ask,
                receive_time=event.receive_time,
                source_event_type="best_bid_ask",
            )
            if top_payload is not None:
                fallback_payloads[event.asset_id] = top_payload
            return
        if isinstance(event, PriceChangeEvent):
            for change in event.price_changes:
                if change.asset_id not in target_tokens:
                    continue
                top_payload = _top_of_book_payload(
                    asset_id=change.asset_id,
                    market=event.market,
                    timestamp=int(event.timestamp),
                    best_bid=change.best_bid,
                    best_ask=change.best_ask,
                    receive_time=event.receive_time,
                    source_event_type="price_change",
                )
                if top_payload is not None:
                    fallback_payloads[change.asset_id] = top_payload


def _token_ids_for_markets(markets: list[dict[str, Any]]) -> tuple[str, ...]:
    token_ids: list[str] = []
    for market in markets:
        token_ids.extend([str(market["up_token_id"]), str(market["down_token_id"])])
    return tuple(dict.fromkeys(token_ids))


def _decode_market_ws_payloads(raw: bytes | str) -> list[dict[str, Any]]:
    raw_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    stripped = raw_bytes.strip()
    if stripped in PolymarketCLOBWebSocketOrderBookSource._KEEPALIVE_TOKENS:
        return []
    try:
        payload = orjson.loads(raw_bytes)
    except orjson.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _market_ws_payload_for_parse(
    *,
    payload: dict[str, Any],
    receive_time_ms: int,
) -> dict[str, Any] | None:
    normalized = dict(payload)
    if "event_type" not in normalized:
        inferred = _infer_market_ws_event_type(normalized)
        if inferred is None:
            return None
        normalized["event_type"] = inferred
    if "timestamp" not in normalized:
        normalized["timestamp"] = str(receive_time_ms)
    if "asks" not in normalized and "ask" in normalized:
        normalized["asks"] = normalized["ask"]
    return normalized


def _infer_market_ws_event_type(payload: dict[str, Any]) -> str | None:
    if "price_changes" in payload:
        return "price_change"
    if "asset_id" in payload and "bids" in payload and ("asks" in payload or "ask" in payload):
        return "book"
    if "asset_id" in payload and ("best_bid" in payload or "best_ask" in payload):
        return "best_bid_ask"
    return None


def _book_event_payload(event: BookEvent) -> dict[str, Any]:
    return {
        "market": event.market,
        "asset_id": event.asset_id,
        "timestamp": int(event.timestamp),
        "receive_time": event.receive_time,
        "source_channel": POLYMARKET_CLOB_WS_SOURCE_CHANNEL,
        "source_event_type": "book",
        "hash": event.hash,
        "bids": _price_level_dicts(event.bids),
        "asks": _price_level_dicts(event.asks),
    }


def _price_level_dicts(levels: list[Any]) -> list[dict[str, str]]:
    return [{"price": str(level.price), "size": str(level.size)} for level in levels]


def _top_of_book_payload(
    *,
    asset_id: str,
    market: str,
    timestamp: int,
    best_bid: Any,
    best_ask: Any,
    receive_time: int | None,
    source_event_type: str,
) -> dict[str, Any] | None:
    bid = _optional_float(best_bid)
    ask = _optional_float(best_ask)
    if bid is None or ask is None:
        return None
    return {
        "market": market,
        "asset_id": asset_id,
        "timestamp": timestamp,
        "receive_time": receive_time,
        "source_channel": POLYMARKET_CLOB_WS_SOURCE_CHANNEL,
        "source_event_type": source_event_type,
        "bids": [{"price": str(bid), "size": "0"}],
        "asks": [{"price": str(ask), "size": "0"}],
    }


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _token_by_up_down_outcome(
    *,
    outcomes: list[Any],
    token_ids: list[Any],
) -> dict[str, str] | None:
    if len(outcomes) != len(token_ids):
        return None
    token_by_outcome: dict[str, str] = {}
    for outcome, token_id in zip(outcomes, token_ids, strict=True):
        normalized = str(outcome).strip().upper()
        if normalized == "UP":
            token_by_outcome["UP"] = str(token_id)
        elif normalized == "DOWN":
            token_by_outcome["DOWN"] = str(token_id)
    return token_by_outcome if set(token_by_outcome) == {"UP", "DOWN"} else None


def _settlement_ts(payload: dict[str, Any], *, default: int) -> int:
    for key in ("closedTime", "closed_time", "endDate", "end_date_iso"):
        value = payload.get(key)
        if value:
            parsed = _parse_iso_millis(str(value))
            if parsed is not None:
                return max(parsed, default)
    return default


def _parse_iso_millis(value: str) -> int | None:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _price_levels(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    levels = []
    for row in value:
        if not isinstance(row, dict):
            continue
        price = _optional_float(row.get("price"))
        size = _optional_float(row.get("size"))
        if price is not None and size is not None:
            levels.append((price, size))
    return levels


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _expected_outcome_for_token(*, market: dict[str, Any], token_id: str) -> str | None:
    if token_id == str(market.get("up_token_id")):
        return "UP"
    if token_id == str(market.get("down_token_id")):
        return "DOWN"
    return None


def _binance_interval(timeframe_ms: int) -> str:
    mapping = {
        60_000: "1m",
        300_000: "5m",
        900_000: "15m",
        3_600_000: "1h",
    }
    if timeframe_ms not in mapping:
        raise RealCorpusPublicProviderError(
            f"Unsupported BTC feature candle timeframe_ms={timeframe_ms}",
            reason_codes=("unsupported_btc_feature_candle_timeframe",),
        )
    return mapping[timeframe_ms]


def _resolution_status_from_payload(payload: dict[str, Any]) -> str:
    prices = _json_list(payload.get("outcomePrices"))
    if len(prices) == 2 and all(str(price) == "0.5" for price in prices):
        return "unknown_50_50"
    return "normal"
