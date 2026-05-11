"""Unit tests for :mod:`bigan.ingestion.book_state`."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bigan.ingestion.book_state import BookRegistry, OrderBook
from bigan.ingestion.message_types import BookEvent, PriceChange, PriceLevel, Side


def _snapshot(asset_id: str = "asset-1") -> BookEvent:
    return BookEvent(
        asset_id=asset_id,
        market="market-1",
        bids=[
            PriceLevel(price=Decimal("0.48"), size=Decimal("30")),
            PriceLevel(price=Decimal("0.50"), size=Decimal("15")),
        ],
        asks=[
            PriceLevel(price=Decimal("0.52"), size=Decimal("25")),
            PriceLevel(price=Decimal("0.54"), size=Decimal("10")),
        ],
        timestamp=1,
        hash="snap-hash",
    )


def test_from_snapshot_populates_book() -> None:
    ob = OrderBook.from_snapshot(_snapshot())
    assert ob.bids == {"0.48": "30", "0.5": "15"}
    assert ob.asks == {"0.52": "25", "0.54": "10"}
    assert ob.last_hash == "snap-hash"


def test_best_bid_best_ask() -> None:
    ob = OrderBook.from_snapshot(_snapshot())
    bid_price, bid_size = ob.best_bid()
    ask_price, ask_size = ob.best_ask()
    assert bid_price == Decimal("0.5")
    assert bid_size == Decimal("15")
    assert ask_price == Decimal("0.52")
    assert ask_size == Decimal("25")


def test_apply_price_change_adds_level() -> None:
    ob = OrderBook.from_snapshot(_snapshot())
    ob.apply_price_change(
        PriceChange(
            asset_id="asset-1",
            price=Decimal("0.49"),
            size=Decimal("100"),
            side=Side.BUY,
            hash="delta-hash",
        )
    )
    assert ob.bids["0.49"] == "100"


def test_apply_price_change_removes_level_when_size_zero() -> None:
    ob = OrderBook.from_snapshot(_snapshot())
    ob.apply_price_change(
        PriceChange(
            asset_id="asset-1",
            price=Decimal("0.48"),
            size=Decimal("0"),
            side=Side.BUY,
            hash="delta-hash",
        )
    )
    assert "0.48" not in ob.bids


def test_compute_hash_is_deterministic() -> None:
    a = OrderBook.from_snapshot(_snapshot())
    b = OrderBook.from_snapshot(_snapshot())
    assert a.compute_hash() == b.compute_hash()
    # State change must change the hash.
    a.apply_price_change(
        PriceChange(
            asset_id="asset-1",
            price=Decimal("0.49"),
            size=Decimal("100"),
            side=Side.BUY,
            hash="delta-hash",
        )
    )
    assert a.compute_hash() != b.compute_hash()


def test_update_wire_hash_returns_changed_flag() -> None:
    ob = OrderBook.from_snapshot(_snapshot())
    assert ob.last_hash == "snap-hash"
    assert ob.update_wire_hash("snap-hash") is False
    assert ob.update_wire_hash("new-hash") is True
    assert ob.last_hash == "new-hash"


def test_registry_upsert_and_apply() -> None:
    reg = BookRegistry()
    reg.upsert_snapshot(_snapshot("a"))
    assert "a" in reg
    ob = reg.apply_price_change(
        "a",
        PriceChange(
            asset_id="a",
            price=Decimal("0.51"),
            size=Decimal("5"),
            side=Side.SELL,
            hash="delta",
        ),
    )
    assert ob is not None
    assert ob.asks["0.51"] == "5"
    assert ob.last_hash == "delta"


def test_registry_apply_without_snapshot_returns_none() -> None:
    reg = BookRegistry()
    result = reg.apply_price_change(
        "ghost",
        PriceChange(
            asset_id="ghost",
            price=Decimal("0.5"),
            size=Decimal("1"),
            side=Side.BUY,
            hash="x",
        ),
    )
    assert result is None


def test_registry_drop_removes_asset() -> None:
    reg = BookRegistry()
    reg.upsert_snapshot(_snapshot("a"))
    reg.upsert_snapshot(_snapshot("b"))
    reg.drop(["a"])
    assert "a" not in reg
    assert "b" in reg


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.50", "0.5"),
        ("0.500", "0.5"),
        ("0.49", "0.49"),
        ("1", "1"),
        ("0", "0"),
    ],
)
def test_price_normalisation(raw: str, expected: str) -> None:
    from bigan.ingestion.book_state import _normalize_price_str

    assert _normalize_price_str(Decimal(raw)) == expected
