"""Reference-price readers for Coinbase, Kraken, and Chainlink (issue #24)."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import aiohttp
import orjson
import websockets

from bigan.canonical.symbols import SymbolMapper
from bigan.canonical.writer import WarehouseWriter

from .metrics import (
    PRICE_READER_ERRORS_TOTAL,
    PRICE_READER_LAST_SUCCESS_TIME,
    PRICE_READER_MESSAGES_TOTAL,
    PRICE_READER_UP,
)

logger = logging.getLogger(__name__)

SOURCE_COINBASE = "coinbase"
SOURCE_KRAKEN = "kraken"
SOURCE_CHAINLINK = "chainlink"

PROVENANCE_WS = "ws"
PROVENANCE_JSON_RPC = "json-rpc"

DEFAULT_COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
DEFAULT_COINBASE_PRODUCT_ID = "BTC-USD"
DEFAULT_KRAKEN_WS_URL = "wss://ws.kraken.com/v2"
DEFAULT_KRAKEN_SYMBOL = "BTC/USD"
DEFAULT_CHAINLINK_SYMBOL = "BTC/USD"
DEFAULT_CHAINLINK_BTC_USD_FEED = "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c"

DECIMALS_SELECTOR = "0x313ce567"
LATEST_ROUND_DATA_SELECTOR = "0xfeaf968c"

_RFC3339_FRACTION_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"\.(?P<fraction>\d+)(?P<tz>Z|[+-]\d{2}:\d{2})?$"
)


class PriceRowSink(Protocol):
    async def write_price_row(self, table: str, row: Mapping[str, Any]) -> None:
        """Persist one canonical reference-price row."""


class WarehousePriceSink:
    """Async facade over :class:`WarehouseWriter` for long-running readers."""

    def __init__(
        self,
        warehouse_dir: Path | str,
        *,
        symbol_mapper: SymbolMapper | None = None,
        max_rows_per_partition: int = 1,
    ) -> None:
        self._writer = WarehouseWriter(
            warehouse_dir,
            max_rows_per_partition=max_rows_per_partition,
        )
        self._symbol_mapper = symbol_mapper

    async def write_price_row(self, table: str, row: Mapping[str, Any]) -> None:
        out = dict(row)
        if self._symbol_mapper is not None:
            out = self._symbol_mapper.enrich_row(out)
        self._writer.append_rows(table, [out])

    async def close(self) -> None:
        self._writer.flush()


@dataclass(frozen=True, slots=True)
class WsPriceReaderConfig:
    url: str
    symbol: str
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    ping_interval_seconds: float = 20.0
    ping_timeout_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class ChainlinkReaderConfig:
    rpc_url: str
    feed_address: str = DEFAULT_CHAINLINK_BTC_USD_FEED
    symbol: str = DEFAULT_CHAINLINK_SYMBOL
    poll_interval_seconds: float = 5.0
    request_timeout_seconds: float = 10.0


class CoinbaseTickerReader:
    """Coinbase Advanced Trade ticker reader for one product."""

    source = SOURCE_COINBASE
    reader = "coinbase_ticker"

    def __init__(self, config: WsPriceReaderConfig, sink: PriceRowSink) -> None:
        self._cfg = config
        self._sink = sink

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        await _run_reconnecting(
            source=self.source,
            reader=self.reader,
            reconnect_min_seconds=self._cfg.reconnect_min_seconds,
            reconnect_max_seconds=self._cfg.reconnect_max_seconds,
            stop_event=stop_event,
            run_once=lambda: self._run_once(stop_event),
        )

    async def _run_once(self, stop_event: asyncio.Event | None) -> None:
        async with websockets.connect(
            self._cfg.url,
            ping_interval=self._cfg.ping_interval_seconds,
            ping_timeout=self._cfg.ping_timeout_seconds,
        ) as ws:
            PRICE_READER_UP.labels(source=self.source, reader=self.reader).set(1)
            await ws.send(
                orjson.dumps(
                    {
                        "type": "subscribe",
                        "product_ids": [self._cfg.symbol],
                        "channel": "ticker",
                    }
                )
            )
            async for raw in ws:
                if _is_stopped(stop_event):
                    return
                payload = orjson.loads(raw if isinstance(raw, bytes) else raw.encode())
                ingest_ts = _now_ms()
                rows = parse_coinbase_ticker_message(payload, ingest_ts=ingest_ts)
                for row in rows:
                    await self._sink.write_price_row("raw_spot_price", row)
                    _record_reader_success(self.source, self.reader)


class KrakenTickerReader:
    """Kraken WebSocket v2 ticker reader for one symbol."""

    source = SOURCE_KRAKEN
    reader = "kraken_ticker"

    def __init__(self, config: WsPriceReaderConfig, sink: PriceRowSink) -> None:
        self._cfg = config
        self._sink = sink

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        await _run_reconnecting(
            source=self.source,
            reader=self.reader,
            reconnect_min_seconds=self._cfg.reconnect_min_seconds,
            reconnect_max_seconds=self._cfg.reconnect_max_seconds,
            stop_event=stop_event,
            run_once=lambda: self._run_once(stop_event),
        )

    async def _run_once(self, stop_event: asyncio.Event | None) -> None:
        async with websockets.connect(
            self._cfg.url,
            ping_interval=self._cfg.ping_interval_seconds,
            ping_timeout=self._cfg.ping_timeout_seconds,
        ) as ws:
            PRICE_READER_UP.labels(source=self.source, reader=self.reader).set(1)
            await ws.send(
                orjson.dumps(
                    {
                        "method": "subscribe",
                        "params": {
                            "channel": "ticker",
                            "symbol": [self._cfg.symbol],
                        },
                    }
                )
            )
            async for raw in ws:
                if _is_stopped(stop_event):
                    return
                payload = orjson.loads(raw if isinstance(raw, bytes) else raw.encode())
                ingest_ts = _now_ms()
                rows = parse_kraken_ticker_message(payload, ingest_ts=ingest_ts)
                for row in rows:
                    await self._sink.write_price_row("raw_spot_price", row)
                    _record_reader_success(self.source, self.reader)


class ChainlinkOracleReader:
    """Poll Chainlink AggregatorV3Interface.latestRoundData via JSON-RPC."""

    source = SOURCE_CHAINLINK
    reader = "chainlink_oracle"

    def __init__(self, config: ChainlinkReaderConfig, sink: PriceRowSink) -> None:
        if not config.rpc_url:
            raise ValueError("Chainlink reader requires a JSON-RPC URL")
        self._cfg = config
        self._sink = sink

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        timeout = aiohttp.ClientTimeout(total=self._cfg.request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not _is_stopped(stop_event):
                try:
                    row = await self.fetch_row(session)
                    await self._sink.write_price_row("raw_oracle_price", row)
                    PRICE_READER_UP.labels(source=self.source, reader=self.reader).set(1)
                    _record_reader_success(self.source, self.reader)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    PRICE_READER_UP.labels(source=self.source, reader=self.reader).set(0)
                    PRICE_READER_ERRORS_TOTAL.labels(
                        source=self.source,
                        reader=self.reader,
                        kind=type(exc).__name__,
                    ).inc()
                    logger.warning("price_reader.error", extra={"reader": self.reader, "err": str(exc)})

                await _sleep_or_stop(stop_event, self._cfg.poll_interval_seconds)

    async def fetch_row(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        ingest_ts = _now_ms()
        decimals_hex = await _eth_call(
            session,
            rpc_url=self._cfg.rpc_url,
            to=self._cfg.feed_address,
            data=DECIMALS_SELECTOR,
        )
        round_data_hex = await _eth_call(
            session,
            rpc_url=self._cfg.rpc_url,
            to=self._cfg.feed_address,
            data=LATEST_ROUND_DATA_SELECTOR,
        )
        decimals = decode_decimals(decimals_hex)
        round_data = decode_latest_round_data(round_data_hex)
        return build_chainlink_oracle_row(
            source_symbol=self._cfg.symbol,
            feed_address=self._cfg.feed_address,
            decimals=decimals,
            round_data=round_data,
            ingest_ts=ingest_ts,
        )


def parse_coinbase_ticker_message(
    payload: Mapping[str, Any],
    *,
    ingest_ts: int,
) -> list[dict[str, Any]]:
    """Parse Coinbase Advanced Trade ticker messages into raw_spot_price rows."""

    if payload.get("channel") not in {"ticker", "ticker_batch"}:
        return []
    message_ts = _as_epoch_ms(payload.get("timestamp")) or ingest_ts
    rows: list[dict[str, Any]] = []
    for event in _coinbase_events(payload):
        event_ts = _as_epoch_ms(event.get("time") or event.get("event_time"))
        for ticker in _coinbase_tickers(event):
            source_symbol = _optional_str(ticker.get("product_id"))
            price = _as_float(ticker.get("price") or ticker.get("last_price"))
            if source_symbol is None or price is None:
                continue
            ts = (
                _as_epoch_ms(ticker.get("time") or ticker.get("event_time"))
                or event_ts
                or message_ts
            )
            rows.append(
                build_spot_price_row(
                    source=SOURCE_COINBASE,
                    source_symbol=source_symbol,
                    ts=ts,
                    message_ts=message_ts,
                    ingest_ts=ingest_ts,
                    price=price,
                    bid_price=_as_float(ticker.get("best_bid") or ticker.get("best_bid_price")),
                    ask_price=_as_float(ticker.get("best_ask") or ticker.get("best_ask_price")),
                )
            )
    return rows


def parse_kraken_ticker_message(
    payload: Mapping[str, Any],
    *,
    ingest_ts: int,
) -> list[dict[str, Any]]:
    """Parse Kraken WebSocket v2 ticker messages into raw_spot_price rows."""

    if payload.get("channel") != "ticker":
        return []
    rows: list[dict[str, Any]] = []
    for item in payload.get("data") or []:
        if not isinstance(item, Mapping):
            continue
        source_symbol = _optional_str(item.get("symbol"))
        price = _as_float(item.get("last") or item.get("price"))
        if source_symbol is None or price is None:
            continue
        ts = _as_epoch_ms(item.get("timestamp") or payload.get("timestamp")) or ingest_ts
        rows.append(
            build_spot_price_row(
                source=SOURCE_KRAKEN,
                source_symbol=source_symbol,
                ts=ts,
                message_ts=ts,
                ingest_ts=ingest_ts,
                price=price,
                bid_price=_as_float(item.get("bid")),
                ask_price=_as_float(item.get("ask")),
            )
        )
    return rows


def build_spot_price_row(
    *,
    source: str,
    source_symbol: str,
    ts: int,
    message_ts: int,
    ingest_ts: int,
    price: float,
    bid_price: float | None = None,
    ask_price: float | None = None,
) -> dict[str, Any]:
    return {
        "ts": int(ts),
        "message_ts": int(message_ts),
        "ingest_ts": int(ingest_ts),
        "source": source,
        "source_symbol": source_symbol,
        "source_market": None,
        "canonical_symbol": None,
        "provenance": PROVENANCE_WS,
        "price": float(price),
        "bid_price": bid_price,
        "ask_price": ask_price,
    }


def build_chainlink_oracle_row(
    *,
    source_symbol: str,
    feed_address: str,
    decimals: int,
    round_data: Mapping[str, int],
    ingest_ts: int,
) -> dict[str, Any]:
    answer = int(round_data["answer"])
    updated_at = int(round_data["updated_at"])
    ts = updated_at * 1000
    price = answer / float(10**int(decimals))
    return {
        "ts": ts,
        "message_ts": ts,
        "ingest_ts": int(ingest_ts),
        "source": SOURCE_CHAINLINK,
        "source_symbol": source_symbol,
        "source_market": feed_address,
        "canonical_symbol": None,
        "provenance": PROVENANCE_JSON_RPC,
        "price": price,
        "answer": answer,
        "decimals": int(decimals),
        "round_id": str(round_data.get("round_id")) if round_data.get("round_id") is not None else None,
        "answered_in_round": (
            str(round_data.get("answered_in_round"))
            if round_data.get("answered_in_round") is not None
            else None
        ),
    }


def decode_decimals(result_hex: str) -> int:
    return _decode_uint(_only_hex(result_hex))


def decode_latest_round_data(result_hex: str) -> dict[str, int]:
    raw = _only_hex(result_hex)
    if len(raw) < 64 * 5:
        raise ValueError("latestRoundData result is too short")
    words = [raw[i : i + 64] for i in range(0, 64 * 5, 64)]
    return {
        "round_id": _decode_uint(words[0]),
        "answer": _decode_int(words[1]),
        "started_at": _decode_uint(words[2]),
        "updated_at": _decode_uint(words[3]),
        "answered_in_round": _decode_uint(words[4]),
    }


async def _eth_call(
    session: aiohttp.ClientSession,
    *,
    rpc_url: str,
    to: str,
    data: str,
) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }
    async with session.post(rpc_url, json=payload) as resp:
        body = await resp.json()
    if "error" in body:
        raise RuntimeError(f"eth_call failed: {body['error']}")
    result = body.get("result")
    if not isinstance(result, str):
        raise RuntimeError("eth_call response missing result")
    return result


async def _run_reconnecting(
    *,
    source: str,
    reader: str,
    reconnect_min_seconds: float,
    reconnect_max_seconds: float,
    stop_event: asyncio.Event | None,
    run_once: Callable[[], Awaitable[None]],
) -> None:
    backoff = reconnect_min_seconds
    while not _is_stopped(stop_event):
        try:
            await run_once()
            backoff = reconnect_min_seconds
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            PRICE_READER_UP.labels(source=source, reader=reader).set(0)
            PRICE_READER_ERRORS_TOTAL.labels(
                source=source,
                reader=reader,
                kind=type(exc).__name__,
            ).inc()
            logger.warning("price_reader.reconnect", extra={"reader": reader, "err": str(exc)})
            await _sleep_or_stop(stop_event, backoff)
            backoff = min(backoff * 2, reconnect_max_seconds)


async def _sleep_or_stop(stop_event: asyncio.Event | None, delay: float) -> None:
    if stop_event is None:
        await asyncio.sleep(delay)
        return
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=delay)
    except TimeoutError:
        return


def _record_reader_success(source: str, reader: str) -> None:
    PRICE_READER_MESSAGES_TOTAL.labels(source=source, reader=reader).inc()
    PRICE_READER_LAST_SUCCESS_TIME.labels(source=source, reader=reader).set(time.time())


def _is_stopped(stop_event: asyncio.Event | None) -> bool:
    return stop_event is not None and stop_event.is_set()


def _coinbase_events(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = payload.get("events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, Mapping)]
    if payload.get("product_id") is not None:
        return [payload]
    return []


def _coinbase_tickers(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tickers = event.get("tickers")
    if isinstance(tickers, list):
        return [ticker for ticker in tickers if isinstance(ticker, Mapping)]
    if event.get("product_id") is not None:
        return [event]
    return []


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_epoch_ms(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return _seconds_or_ms(value)
    if isinstance(value, float):
        return _seconds_or_ms(value)
    text = str(value).strip()
    try:
        return _seconds_or_ms(float(text))
    except ValueError:
        return _rfc3339_to_ms(text)


def _seconds_or_ms(value: float) -> int:
    return int(value * 1000) if abs(value) < 100_000_000_000 else int(value)


def _rfc3339_to_ms(text: str) -> int:
    match = _RFC3339_FRACTION_RE.match(text)
    if match is not None:
        fraction = match.group("fraction")[:6].ljust(6, "0")
        tz = match.group("tz") or "Z"
        text = f"{match.group('head')}.{fraction}{tz}"
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _only_hex(result_hex: str) -> str:
    raw = result_hex[2:] if result_hex.startswith("0x") else result_hex
    if not raw:
        raise ValueError("empty hex result")
    return raw.rjust(64, "0")


def _decode_uint(word: str) -> int:
    return int(word, 16)


def _decode_int(word: str) -> int:
    value = int(word, 16)
    if value >= 2**255:
        value -= 2**256
    return value


def _now_ms() -> int:
    return int(time.time() * 1000)
