"""Unit tests for :mod:`bigan.ingestion.gamma_client`."""

from __future__ import annotations

from bigan.ingestion.gamma_client import (
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
