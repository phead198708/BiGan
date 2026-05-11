"""Minimal async client for the Polymarket CLOB REST API (issue #5).

The REST client only exists to backfill missing data when the WebSocket
stream goes silent. It is **not** the primary data path; correctness of
the live pipeline does not depend on it.

Endpoints used (all on ``https://clob.polymarket.com``):

- ``GET /book?token_id=<asset_id>``      — current orderbook snapshot
- ``GET /trades?market=<condition_id>``  — historical trades, paginated

Both endpoints return JSON. We project their responses into the same
shape as the WebSocket events so the existing transform layer (and the
canonical ETL) accept backfilled records without any new branches.

The client is permissive: HTTP errors return ``None`` / ``[]`` rather
than raising, so a transient REST outage does not crash the runner.
Each call is logged with structured fields for ops triage.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


DEFAULT_CLOB_REST_BASE = "https://clob.polymarket.com"


@dataclass(slots=True)
class RestTrade:
    """One trade as returned by ``/trades``, lifted to a typed shape.

    We only project the fields the backfill flow actually consumes; all
    other fields stay in :attr:`raw` for forensic logging.
    """

    asset_id: str
    market: str
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    match_time_ms: int
    raw: dict[str, Any]


@dataclass(slots=True)
class RestOrderbook:
    """One orderbook snapshot as returned by ``/book``."""

    asset_id: str
    market: str
    timestamp_ms: int
    hash: str | None
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PolymarketRestClient:
    """Thin async wrapper around the Polymarket CLOB REST endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_CLOB_REST_BASE,
        *,
        timeout_seconds: float = 10.0,
        page_size: int = 100,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._page_size = page_size
        self._owned_session = session is None
        self._session = session

    # ------------------------------------------------------------------
    # Async lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> PolymarketRestClient:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owned_session and self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_orderbook(self, asset_id: str) -> RestOrderbook | None:
        """Return the current orderbook for ``asset_id`` (token id)."""
        url = f"{self._base_url}/book"
        params = {"token_id": asset_id}
        data = await self._get_json(url, params)
        if not isinstance(data, dict):
            return None
        return _parse_orderbook(data)

    async def iter_trades(
        self,
        market_condition_id: str,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        max_pages: int = 50,
    ) -> AsyncIterator[RestTrade]:
        """Yield trades for ``market_condition_id``, optionally bounded by
        the [since_ms, until_ms] inclusive window.

        Pagination uses the API's ``next_cursor`` token. We stop early
        once a page reports a trade older than ``since_ms`` to avoid
        scanning the entire trade history of busy markets.
        """
        url = f"{self._base_url}/trades"
        cursor: str | None = None
        for _ in range(max_pages):
            params: dict[str, Any] = {
                "market": market_condition_id,
                "limit": self._page_size,
            }
            if cursor is not None:
                params["next_cursor"] = cursor
            data = await self._get_json(url, params)
            if not isinstance(data, dict):
                return
            trades = data.get("data") or []
            saw_old = False
            for entry in trades:
                if not isinstance(entry, dict):
                    continue
                trade = _parse_trade(entry)
                if trade is None:
                    continue
                if since_ms is not None and trade.match_time_ms < since_ms:
                    saw_old = True
                    continue
                if until_ms is not None and trade.match_time_ms > until_ms:
                    continue
                yield trade
            cursor = data.get("next_cursor") or None
            if cursor is None or saw_old:
                return

    async def fetch_trades(
        self,
        market_condition_id: str,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        max_pages: int = 50,
    ) -> list[RestTrade]:
        """List form of :meth:`iter_trades` for callers that want a
        materialised result."""
        out: list[RestTrade] = []
        async for trade in self.iter_trades(
            market_condition_id,
            since_ms=since_ms,
            until_ms=until_ms,
            max_pages=max_pages,
        ):
            out.append(trade)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError(
                "PolymarketRestClient is not active; use 'async with'."
            )
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "clob_rest.non_200",
                        extra={"url": url, "status": resp.status},
                    )
                    return None
                return await resp.json()
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.warning(
                "clob_rest.request_failed",
                extra={"url": url, "err": str(exc)},
            )
            return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int_ms(v: Any) -> int | None:
    """Coerce a Polymarket REST timestamp into UTC ms epoch.

    Polymarket sometimes returns seconds-as-string and sometimes
    milliseconds-as-string. We normalise to ms by detecting the
    magnitude (anything < 10^12 is treated as seconds).
    """
    if v is None:
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    if n < 0:
        return None
    return n * 1000 if n < 1_000_000_000_000 else n


def _parse_trade(raw: dict[str, Any]) -> RestTrade | None:
    asset_id = raw.get("asset_id") or raw.get("token_id")
    market = raw.get("market") or raw.get("condition_id")
    price = _as_float(raw.get("price"))
    size = _as_float(raw.get("size"))
    side = raw.get("side")
    match_time = _as_int_ms(raw.get("match_time") or raw.get("timestamp"))
    if (
        not asset_id
        or not market
        or price is None
        or size is None
        or side not in ("BUY", "SELL")
        or match_time is None
    ):
        return None
    return RestTrade(
        asset_id=str(asset_id),
        market=str(market),
        price=price,
        size=size,
        side=str(side),
        match_time_ms=match_time,
        raw=dict(raw),
    )


def _parse_orderbook(raw: dict[str, Any]) -> RestOrderbook | None:
    asset_id = raw.get("asset_id") or raw.get("token_id")
    market = raw.get("market") or raw.get("condition_id")
    timestamp = _as_int_ms(raw.get("timestamp"))
    if not asset_id or not market or timestamp is None:
        return None
    bids = _coerce_levels(raw.get("bids"))
    # Polymarket's payload uses both ``ask`` and ``asks`` historically.
    asks_payload = raw.get("asks") if "asks" in raw else raw.get("ask")
    asks = _coerce_levels(asks_payload)
    return RestOrderbook(
        asset_id=str(asset_id),
        market=str(market),
        timestamp_ms=timestamp,
        hash=str(raw["hash"]) if raw.get("hash") is not None else None,
        bids=bids,
        asks=asks,
        raw=dict(raw),
    )


def _coerce_levels(raw_levels: Any) -> list[tuple[float, float]]:
    if not isinstance(raw_levels, list):
        return []
    out: list[tuple[float, float]] = []
    for lvl in raw_levels:
        if not isinstance(lvl, dict):
            continue
        price = _as_float(lvl.get("price"))
        size = _as_float(lvl.get("size"))
        if price is None or size is None:
            continue
        out.append((price, size))
    return out
