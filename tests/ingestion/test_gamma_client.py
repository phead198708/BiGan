"""Unit tests for :mod:`bigan.ingestion.gamma_client`."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from bigan.ingestion.gamma_client import (
    GammaClient,
    MarketDiscoverySpec,
    _candidate_slugs_for_now,
    _gamma_fetch_error_context,
    _log_gamma_fetch_failed,
    _market_from_gamma,
    _parse_iso8601_to_ms,
    active_market_symbol_mapping_rows,
    diff_subscription_sets,
    parse_market_specs_json,
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


def test_active_market_symbol_mapping_rows_encode_outcome_side() -> None:
    market = _market_from_gamma(
        {
            "slug": "btc-updown-15m-1778423700",
            "conditionId": "0xabc",
            "clobTokenIds": ["111", "222"],
            "outcomes": ["Up", "Down"],
            "startDate": "2026-05-10T14:30:00Z",
            "endDate": "2026-05-10T14:45:00Z",
            "orderPriceMinTickSize": "0.01",
        }
    )
    assert market is not None

    rows = active_market_symbol_mapping_rows([market], ingest_ts=1_800_000_000_000)

    by_token = {row["source_symbol"]: row for row in rows}
    assert by_token["111"]["canonical_symbol"] == "BTC-15M:btc-updown-15m-1778423700:UP"
    assert by_token["222"]["canonical_symbol"] == "BTC-15M:btc-updown-15m-1778423700:DOWN"
    assert by_token["111"]["source_market"] == "0xabc"
    assert by_token["111"]["symbol_kind"] == "btc_15m_outcome"
    assert '"outcome_side":"UP"' in by_token["111"]["metadata_json"]


def test_active_market_symbol_mapping_rows_encode_eth_horizon() -> None:
    spec = MarketDiscoverySpec(
        slug_prefix="eth-updown-5m-",
        underlying="ETH",
        horizon_ms=5 * 60_000,
        symbol_kind="eth_5m_outcome",
    )
    market = _market_from_gamma(
        {
            "slug": "eth-updown-5m-1778423700",
            "conditionId": "0xeth",
            "clobTokenIds": ["333", "444"],
            "outcomes": ["Up", "Down"],
            "startDate": "2026-05-10T14:30:00Z",
            "endDate": "2026-05-10T14:35:00Z",
            "orderPriceMinTickSize": "0.01",
        },
        spec,
    )
    assert market is not None

    rows = active_market_symbol_mapping_rows([market], ingest_ts=1_800_000_000_000)

    by_token = {row["source_symbol"]: row for row in rows}
    assert by_token["333"]["canonical_symbol"] == "ETH-5M:eth-updown-5m-1778423700:UP"
    assert by_token["333"]["symbol_kind"] == "eth_5m_outcome"
    assert '"horizon_ms":300000' in by_token["333"]["metadata_json"]


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
    page_calls = [call for call in session.calls if "offset" in call]
    assert [call["limit"] for call in page_calls] == [100, 100, 100]
    assert [call["offset"] for call in page_calls] == [0, 100, 200]
    assert {call["order"] for call in page_calls} == {"endDate"}
    assert {call["ascending"] for call in page_calls} == {"true"}


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
    page_calls = [call for call in session.calls if "offset" in call]
    assert [call["offset"] for call in page_calls] == [0, 2, 4]


def test_candidate_slugs_for_now_include_current_and_upcoming_rounds() -> None:
    now_ms = 1_779_461_234_000

    slugs = _candidate_slugs_for_now(
        "btc-updown-15m-",
        now_ms=now_ms,
        horizon_ms=15 * 60_000,
        lookback_intervals=1,
        lookahead_intervals=2,
    )

    assert slugs == (
        "btc-updown-15m-1779460200",
        "btc-updown-15m-1779461100",
        "btc-updown-15m-1779462000",
        "btc-updown-15m-1779462900",
    )


def test_list_active_markets_directly_fetches_near_expiry_slug() -> None:
    now_ms = 1_779_461_234_000
    session = _FakeGammaSession(
        pages={0: []},
        slug_records={
            "btc-updown-15m-1779461100": [
                _gamma_record(
                    slug="btc-updown-15m-1779461100",
                    condition_id="0xcurrent",
                    up="111",
                    down="222",
                    end_date="2026-05-22T15:00:00Z",
                )
            ]
        },
    )

    async def go() -> list[str]:
        client = GammaClient("https://gamma.test", "btc-updown-15m-")
        client._session = session  # type: ignore[attr-defined]  # test fake
        markets = await client.list_active_markets(
            page_limit=2,
            max_pages=1,
            direct_slug_lookback_intervals=1,
            direct_slug_lookahead_intervals=2,
            now_ms=now_ms,
        )
        return [market.slug for market in markets]

    assert asyncio.run(go()) == ["btc-updown-15m-1779461100"]


def test_list_active_markets_keeps_direct_slug_when_pagination_fails() -> None:
    now_ms = 1_779_461_234_000
    session = _FakeGammaSession(
        pages={
            0: [
                _gamma_record(
                    slug=f"other-market-{idx}",
                    condition_id=f"0xother{idx}",
                    up=f"up-{idx}",
                    down=f"down-{idx}",
                )
                for idx in range(2)
            ],
            2: TimeoutError("gamma page failed"),
        },
        slug_records={
            "btc-updown-15m-1779461100": [
                _gamma_record(
                    slug="btc-updown-15m-1779461100",
                    condition_id="0xcurrent",
                    up="111",
                    down="222",
                    end_date="2026-05-22T15:00:00Z",
                )
            ]
        },
    )

    async def go() -> list[str]:
        client = GammaClient("https://gamma.test", "btc-updown-15m-")
        client._session = session  # type: ignore[attr-defined]  # test fake
        markets = await client.list_active_markets(
            page_limit=2,
            max_pages=2,
            direct_slug_lookback_intervals=1,
            direct_slug_lookahead_intervals=2,
            now_ms=now_ms,
        )
        return [market.slug for market in markets]

    assert asyncio.run(go()) == ["btc-updown-15m-1779461100"]


def test_list_active_markets_drops_round_at_expiry_and_keeps_next_round() -> None:
    now_ms = 1_779_462_000_000
    session = _FakeGammaSession(
        pages={0: []},
        slug_records={
            "btc-updown-15m-1779461100": [
                _gamma_record(
                    slug="btc-updown-15m-1779461100",
                    condition_id="0xexpired",
                    up="111",
                    down="222",
                    end_date="2026-05-22T15:00:00Z",
                )
            ],
            "btc-updown-15m-1779462000": [
                _gamma_record(
                    slug="btc-updown-15m-1779462000",
                    condition_id="0xnext",
                    up="333",
                    down="444",
                    end_date="2026-05-22T15:15:00Z",
                )
            ],
        },
    )

    async def go() -> list[str]:
        client = GammaClient("https://gamma.test", "btc-updown-15m-")
        client._session = session  # type: ignore[attr-defined]  # test fake
        markets = await client.list_active_markets(
            page_limit=2,
            max_pages=1,
            direct_slug_lookback_intervals=1,
            direct_slug_lookahead_intervals=1,
            now_ms=now_ms,
        )
        return [market.slug for market in markets]

    assert asyncio.run(go()) == ["btc-updown-15m-1779462000"]


def test_list_active_markets_supports_multiple_market_specs() -> None:
    now_ms = 1_779_461_234_000
    specs = parse_market_specs_json(
        """
        [
          {"slug_prefix": "btc-updown-15m-", "underlying": "BTC", "horizon_minutes": 15},
          {"slug_prefix": "eth-updown-5m-", "underlying": "ETH", "horizon_minutes": 5}
        ]
        """
    )
    session = _FakeGammaSession(
        pages={
            0: [
                _gamma_record(
                    slug="btc-updown-15m-1779461100",
                    condition_id="0xbtc",
                    up="111",
                    down="222",
                ),
                _gamma_record(
                    slug="eth-updown-5m-1779461100",
                    condition_id="0xeth",
                    up="333",
                    down="444",
                ),
                _gamma_record(
                    slug="sol-updown-5m-1779461100",
                    condition_id="0xsol",
                    up="555",
                    down="666",
                ),
            ]
        },
    )

    async def go() -> list[tuple[str, str, int]]:
        client = GammaClient("https://gamma.test", "btc-updown-15m-", market_specs=specs)
        client._session = session  # type: ignore[attr-defined]  # test fake
        markets = await client.list_active_markets(
            page_limit=3,
            max_pages=1,
            direct_slug_lookback_intervals=0,
            direct_slug_lookahead_intervals=0,
            now_ms=now_ms,
        )
        return [(market.slug, market.underlying, market.horizon_ms) for market in markets]

    assert asyncio.run(go()) == [
        ("btc-updown-15m-1779461100", "BTC", 15 * 60_000),
        ("eth-updown-5m-1779461100", "ETH", 5 * 60_000),
    ]


def test_parse_market_specs_json_infers_defaults() -> None:
    specs = parse_market_specs_json(
        '[{"slug_prefix":"eth-updown-5m-"},{"slug_prefix":"btc-updown-15m-"}]'
    )

    assert specs[0] == MarketDiscoverySpec(
        slug_prefix="eth-updown-5m-",
        underlying="ETH",
        horizon_ms=5 * 60_000,
        symbol_kind="eth_5m_outcome",
    )
    assert specs[1].underlying == "BTC"
    assert specs[1].horizon_ms == 15 * 60_000


def test_gamma_fetch_error_context_includes_page_diagnostics() -> None:
    cause = TimeoutError("read timed out")
    exc = TimeoutError()
    exc.__cause__ = cause

    context = _gamma_fetch_error_context(
        exc,
        url="https://gamma.test/markets",
        limit=100,
        offset=200,
        timeout_s=10.0,
    )

    assert context == {
        "err_type": "TimeoutError",
        "err": "",
        "url": "https://gamma.test/markets",
        "limit": 100,
        "offset": 200,
        "timeout_s": 10.0,
        "cause_type": "TimeoutError",
        "cause": "read timed out",
    }


def test_gamma_poll_failed_log_message_includes_plain_diagnostics(caplog) -> None:
    exc = TimeoutError("read timed out")

    with caplog.at_level(logging.WARNING):
        _log_gamma_fetch_failed(
            exc,
            url="https://gamma.test/markets",
            limit=100,
            offset=200,
            timeout_s=10.0,
        )

    message = caplog.records[-1].getMessage()
    assert "gamma.poll_failed" in message
    assert "err_type=TimeoutError" in message
    assert "err='read timed out'" in message
    assert "url=https://gamma.test/markets" in message
    assert "limit=100" in message
    assert "offset=200" in message
    assert "timeout_s=10.0" in message


class _FakeGammaSession:
    def __init__(
        self,
        pages: dict[int, list[dict[str, Any]] | Exception],
        slug_records: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._pages = pages
        self._slug_records = slug_records or {}
        self.calls: list[dict[str, Any]] = []

    def get(self, _url: str, *, params: dict[str, Any]) -> _FakeGammaResponse:
        self.calls.append(dict(params))
        if "slug" in params:
            return _FakeGammaResponse(self._slug_records.get(str(params["slug"]), []))
        return _FakeGammaResponse(self._pages.get(int(params["offset"]), []))


class _FakeGammaResponse:
    def __init__(self, records: list[dict[str, Any]] | Exception) -> None:
        self._records = records

    async def __aenter__(self) -> _FakeGammaResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if isinstance(self._records, Exception):
            raise self._records
        return None

    async def read(self) -> bytes:
        if isinstance(self._records, Exception):
            raise self._records
        return json.dumps(self._records).encode("utf-8")


def _gamma_record(
    *,
    slug: str,
    condition_id: str,
    up: str,
    down: str,
    end_date: str = "2099-01-01T00:15:00Z",
) -> dict[str, Any]:
    return {
        "slug": slug,
        "conditionId": condition_id,
        "clobTokenIds": [up, down],
        "outcomes": ["Up", "Down"],
        "startDate": "2099-01-01T00:00:00Z",
        "endDate": end_date,
        "orderPriceMinTickSize": "0.01",
    }
