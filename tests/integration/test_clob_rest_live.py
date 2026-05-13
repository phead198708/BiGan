"""Live Polymarket REST schema smoke tests for issue #26.

These tests intentionally call public upstream APIs. They are skipped by
default through ``tests/conftest.py`` and run only when the live marker is
explicitly selected or ``BIGAN_RUN_LIVE_TESTS=1`` is set.
"""

from __future__ import annotations

import json
from typing import Any

import aiohttp
import pytest

from bigan.ingestion.clob_rest import (
    PolymarketRestClient,
    RestTrade,
    _as_int_ms,
    _parse_orderbook,
    _parse_trade,
)
from bigan.ingestion.config import IngestionSettings
from bigan.ingestion.gamma_client import ActiveMarket, GammaClient

pytestmark = pytest.mark.live


async def _get_json(
    base_url: str,
    path: str,
    params: dict[str, Any],
) -> Any:
    timeout = aiohttp.ClientTimeout(total=15.0)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.get(f"{base_url.rstrip('/')}{path}", params=params) as resp,
    ):
        body = await resp.text()
        assert resp.status == 200, body[:500]
        return await resp.json(content_type=None)


async def _discover_active_market(settings: IngestionSettings) -> ActiveMarket:
    async with GammaClient(
        settings.gamma_api_base,
        settings.market_slug_prefix,
        request_timeout_seconds=15.0,
    ) as gamma:
        markets = await gamma.list_active_markets(page_limit=100, max_pages=5)
    if not markets:
        pytest.skip(f"no active markets found for slug prefix {settings.market_slug_prefix!r}")
    return markets[0]


def _field_fingerprint(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "fields": sorted(raw),
        "bid_fields": sorted(raw["bids"][0]) if raw.get("bids") else [],
        "ask_fields": sorted(raw["asks"][0]) if raw.get("asks") else [],
    }


async def test_live_book_schema_accepts_current_parser() -> None:
    settings = IngestionSettings()
    market = await _discover_active_market(settings)

    last_raw: dict[str, Any] | None = None
    for asset_id in market.asset_ids:
        raw = await _get_json(settings.clob_rest_url, "/book", {"token_id": asset_id})
        assert isinstance(raw, dict)
        last_raw = raw
        book = _parse_orderbook(raw)
        if book is None:
            continue

        assert {"asset_id", "market", "timestamp", "bids", "asks"}.issubset(raw)
        assert book.asset_id == str(raw["asset_id"])
        assert book.market == str(raw["market"])
        assert book.timestamp_ms >= 1_000_000_000_000
        print("book_schema_fingerprint", json.dumps(_field_fingerprint(raw), sort_keys=True))
        return

    assert last_raw is not None
    pytest.fail(
        "no active-market book payload was accepted by _parse_orderbook; "
        f"last fields={sorted(last_raw)}"
    )


async def test_live_trade_history_schema_accepts_current_parser_and_paginates() -> None:
    settings = IngestionSettings()
    recent = await _get_json(settings.polymarket_data_api_url, "/trades", {"limit": 100})
    assert isinstance(recent, list)
    if not recent:
        pytest.skip("public trade history returned no recent trades")

    parsed_candidates: list[tuple[dict[str, Any], RestTrade]] = []
    for entry in recent:
        if not isinstance(entry, dict):
            continue
        trade = _parse_trade(entry)
        if trade is not None:
            parsed_candidates.append((entry, trade))
    if not parsed_candidates:
        pytest.fail("no recent public trade payload was accepted by _parse_trade")

    raw_trade, parsed_trade = next(
        (
            candidate
            for candidate in parsed_candidates
            if str(candidate[0].get("slug", "")).startswith("btc-updown-")
        ),
        parsed_candidates[0],
    )
    assert {"asset", "conditionId", "price", "side", "size", "timestamp"}.issubset(
        raw_trade
    )

    async with PolymarketRestClient(
        settings.clob_rest_url,
        data_api_base_url=settings.polymarket_data_api_url,
        page_size=1,
    ) as client:
        first_two_pages = await client.fetch_trades(parsed_trade.market, max_pages=2)

    assert first_two_pages
    assert all(trade.market == parsed_trade.market for trade in first_two_pages)
    ts_ms = _as_int_ms(raw_trade["timestamp"])
    assert ts_ms is not None
    assert ts_ms >= 1_000_000_000_000
    assert _as_int_ms(ts_ms) == ts_ms
    assert parsed_trade.match_time_ms == ts_ms

    fingerprint = {
        "fields": sorted(raw_trade),
        "market": parsed_trade.market,
        "pages_checked": 2,
        "timestamp_digits": len(str(raw_trade["timestamp"])),
        "parsed_timestamp_digits": len(str(parsed_trade.match_time_ms)),
    }
    print("trade_schema_fingerprint", json.dumps(fingerprint, sort_keys=True))
