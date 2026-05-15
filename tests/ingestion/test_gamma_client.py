"""Unit tests for :mod:`bigan.ingestion.gamma_client`."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from bigan.ingestion.gamma_client import (
    GammaClient,
    _market_from_gamma,
    _parse_iso8601_to_ms,
    diff_subscription_sets,
)


def test_parse_iso8601_with_z_suffix() -> None:
    ms = _parse_iso8601_to_ms("2026-05-10T14:30:00Z")
    assert ms == 1778423400000  # known epoch for that UTC moment


def test_parse_iso8601_with_offset() -> None:
    ms = _parse_iso8601_to_ms("2026-05-10T14:30:00+00:00")
    assert ms == 1778423400000


def test_parse_iso8601_garbage_returns_zero() -> None:
    assert _parse_iso8601_to_ms("not a date") == 0
    assert _parse_iso8601_to_ms(None) == 0
    assert _parse_iso8601_to_ms("") == 0


def test_market_from_gamma_with_string_encoded_arrays() -> None:
    record = {
        "slug": "btc-updown-15m-1778423700",
        "conditionId": "0xabc",
        "clobTokenIds": '["111", "222"]',
        "outcomes": '["Up", "Down"]',
        "startDate": "2026-05-10T14:30:00Z",
        "endDate": "2026-05-10T14:45:00Z",
        "orderPriceMinTickSize": "0.01",
    }
    market = _market_from_gamma(record)
    assert market is not None
    assert market.slug == "btc-updown-15m-1778423700"
    assert market.condition_id == "0xabc"
    assert market.asset_id_up == "111"
    assert market.asset_id_down == "222"
    assert market.tick_size == "0.01"
    assert market.start_ts_ms == 1778423400000
    assert market.end_ts_ms == 1778424300000


def test_market_from_gamma_handles_reversed_outcome_order() -> None:
    """If Gamma ever returns ["Down", "Up"] we still attribute tokens correctly."""
    record = {
        "slug": "btc-updown-15m-x",
        "conditionId": "0xabc",
        "clobTokenIds": ["111", "222"],
        "outcomes": ["Down", "Up"],
        "startDate": "2026-05-10T14:30:00Z",
        "endDate": "2026-05-10T14:45:00Z",
    }
    market = _market_from_gamma(record)
    assert market is not None
    assert market.asset_id_down == "111"
    assert market.asset_id_up == "222"


def test_market_from_gamma_drops_invalid_records() -> None:
    assert _market_from_gamma({}) is None
    assert _market_from_gamma({"slug": "x"}) is None
    assert (
        _market_from_gamma(
            {
                "slug": "x",
                "conditionId": "y",
                "clobTokenIds": ["1"],  # only 1 token
                "outcomes": ["Up", "Down"],
            }
        )
        is None
    )


def test_diff_subscription_sets() -> None:
    add, remove = diff_subscription_sets(current=["a", "b"], desired=["b", "c"])
    assert add == ["c"]
    assert remove == ["a"]


def test_diff_subscription_sets_empty_current() -> None:
    add, remove = diff_subscription_sets(current=[], desired=["x", "y"])
    assert add == ["x", "y"]
    assert remove == []


def test_diff_subscription_sets_no_op() -> None:
    add, remove = diff_subscription_sets(current=["a"], desired=["a"])
    assert add == []
    assert remove == []


def test_list_active_markets_handles_gamma_limit_cap() -> None:
    page0 = [
        _gamma_record(
            slug="btc-updown-15m-4102444800",
            condition_id="0xbtc1",
            up="111",
            down="222",
        ),
        *[
            _gamma_record(
                slug=f"other-market-{idx}",
                condition_id=f"0xother{idx}",
                up=f"up-{idx}",
                down=f"down-{idx}",
            )
            for idx in range(99)
        ],
    ]
    page1 = [
        _gamma_record(
            slug="btc-updown-15m-4102445700",
            condition_id="0xbtc2",
            up="333",
            down="444",
        )
    ]
    session = _FakeGammaSession(pages={0: page0, 100: page1})

    async def go() -> list[str]:
        client = GammaClient("https://gamma.test", "btc-updown-15m-")
        client._session = session  # type: ignore[attr-defined]  # test fake
        markets = await client.list_active_markets(
            page_limit=200,
            max_pages=3,
            empty_page_streak_limit=99,
        )
        return [market.slug for market in markets]

    assert asyncio.run(go()) == [
        "btc-updown-15m-4102444800",
        "btc-updown-15m-4102445700",
    ]
    assert [call["limit"] for call in session.calls] == [100, 100]
    assert [call["offset"] for call in session.calls] == [0, 100]


def test_list_active_markets_keeps_scanning_after_empty_target_pages() -> None:
    page0 = [
        _gamma_record(
            slug="btc-updown-15m-4102444800",
            condition_id="0xbtc1",
            up="111",
            down="222",
        ),
        _gamma_record(
            slug="other-market-before-gap",
            condition_id="0xother-before",
            up="up-before",
            down="down-before",
        ),
    ]
    page1 = [
        _gamma_record(
            slug=f"other-market-{idx}",
            condition_id=f"0xother{idx}",
            up=f"up-{idx}",
            down=f"down-{idx}",
        )
        for idx in range(2)
    ]
    page2 = [
        _gamma_record(
            slug="btc-updown-15m-4102445700",
            condition_id="0xbtc2",
            up="333",
            down="444",
        )
    ]
    session = _FakeGammaSession(pages={0: page0, 2: page1, 4: page2})

    async def go() -> list[str]:
        client = GammaClient("https://gamma.test", "btc-updown-15m-")
        client._session = session  # type: ignore[attr-defined]  # test fake
        markets = await client.list_active_markets(
            page_limit=2,
            max_pages=3,
            empty_page_streak_limit=1,
        )
        return [market.slug for market in markets]

    assert asyncio.run(go()) == [
        "btc-updown-15m-4102444800",
        "btc-updown-15m-4102445700",
    ]
    assert [call["offset"] for call in session.calls] == [0, 2, 4]


class _FakeGammaSession:
    def __init__(self, pages: dict[int, list[dict[str, Any]]]) -> None:
        self._pages = pages
        self.calls: list[dict[str, Any]] = []

    def get(self, _url: str, *, params: dict[str, Any]) -> _FakeGammaResponse:
        self.calls.append(dict(params))
        return _FakeGammaResponse(self._pages.get(int(params["offset"]), []))


class _FakeGammaResponse:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    async def __aenter__(self) -> _FakeGammaResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        return json.dumps(self._records).encode("utf-8")


def _gamma_record(
    *,
    slug: str,
    condition_id: str,
    up: str,
    down: str,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "conditionId": condition_id,
        "clobTokenIds": [up, down],
        "outcomes": ["Up", "Down"],
        "startDate": "2099-01-01T00:00:00Z",
        "endDate": "2099-01-01T00:15:00Z",
        "orderPriceMinTickSize": "0.01",
    }
