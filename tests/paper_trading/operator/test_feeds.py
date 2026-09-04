"""Read-only feed synchronization, freshness, and fencing tests."""

from __future__ import annotations

import pytest

from bigan.features.binance_ofi import BinanceOFICalculator
from bigan.paper_trading.operator.feeds import (
    BinanceDepthSynchronizer,
    BoundedEventQueue,
    FakeReadOnlyTransport,
    FeedConnectionState,
    PolymarketBookSynchronizer,
)


def _delta(
    first: int,
    final: int,
    *,
    ts_ms: int,
    symbol: str = "BTCUSDT",
    bid: str = "100.0",
    ask: str = "101.0",
) -> dict[str, object]:
    return {
        "e": "depthUpdate",
        "E": ts_ms,
        "s": symbol,
        "U": first,
        "u": final,
        "b": [[bid, "10"]],
        "a": [[ask, "8"]],
    }


def _snapshot(update_id: int = 10) -> dict[str, object]:
    return {
        "lastUpdateId": update_id,
        "bids": [["99.0", "9"], ["98.0", "6"]],
        "asks": [["102.0", "7"], ["103.0", "5"]],
    }


def _binance(
    *,
    buffer_size: int = 10,
    book_level_limit: int = 5_000,
) -> BinanceDepthSynchronizer:
    return BinanceDepthSynchronizer(
        calculator=BinanceOFICalculator(
            symbol="BTCUSDT",
            zscore_min_samples=1,
            ema_alpha=1.0,
        ),
        symbol="BTCUSDT",
        max_age_ms=500,
        delta_buffer_size=buffer_size,
        book_level_limit=book_level_limit,
    )


def test_binance_snapshot_then_buffered_delta_synchronizes_ofi() -> None:
    feed = _binance()
    feed.begin_generation(1)
    assert feed.ingest_delta(_delta(11, 11, ts_ms=1_100), generation=1) is False

    assert feed.ingest_snapshot(_snapshot(10), generation=1, received_at_ms=1_000)
    assert feed.state is FeedConnectionState.READY
    assert feed.last_update_id == 11
    assert feed.calculator.last_timestamp_ms == 1_100
    assert feed.health(now_ms=1_200).fresh is True


def test_binance_gap_invalidates_alpha_until_new_bootstrap() -> None:
    feed = _binance()
    feed.begin_generation(1)
    assert feed.ingest_snapshot(_snapshot(10), generation=1, received_at_ms=1_000)

    assert feed.ingest_delta(_delta(12, 12, ts_ms=1_100), generation=1) is False
    assert feed.state is FeedConnectionState.SYNCING
    assert feed.needs_bootstrap is True
    assert feed.gap_count == 1
    assert feed.calculator.last_timestamp_ms is None
    assert feed.health(now_ms=1_100).fresh is False

    feed.begin_generation(2)
    assert feed.ingest_snapshot(_snapshot(20), generation=2, received_at_ms=1_200)
    assert feed.state is FeedConnectionState.READY


def test_binance_diff_depth_updates_a_local_book_one_side_at_a_time() -> None:
    feed = _binance()
    feed.begin_generation(1)
    assert feed.ingest_snapshot(_snapshot(10), generation=1, received_at_ms=1_000)

    assert feed.ingest_delta(
        {
            "s": "BTCUSDT",
            "E": 1_100,
            "U": 11,
            "u": 11,
            "b": [["98", "12"]],
            "a": [],
        },
        generation=1,
    )
    assert feed.last_update_id == 11
    assert feed.mid_price == pytest.approx(100.5)
    assert feed.last_top_changed is False
    assert feed.calculator.last_timestamp_ms == 1_000

    assert feed.ingest_delta(
        {
            "s": "BTCUSDT",
            "E": 1_200,
            "U": 12,
            "u": 12,
            "b": [["99", "0"]],
            "a": [],
        },
        generation=1,
    )
    assert feed.last_update_id == 12
    assert feed.last_bid_price == 98.0
    assert feed.last_ask_price == 102.0
    assert feed.mid_price == pytest.approx(100.0)
    assert feed.last_top_changed is True
    assert feed.calculator.last_timestamp_ms == 1_200


def test_binance_first_delta_can_predate_rest_snapshot_receipt() -> None:
    feed = _binance()
    feed.begin_generation(1)
    assert feed.ingest_snapshot(_snapshot(10), generation=1, received_at_ms=1_200)

    assert feed.ingest_delta(_delta(11, 11, ts_ms=1_100), generation=1, received_at_ms=1_250)
    assert feed.last_update_id == 11
    assert feed.calculator.last_timestamp_ms == 1_100
    assert feed.needs_bootstrap is False


def test_binance_subscription_ack_is_not_treated_as_a_depth_failure() -> None:
    feed = _binance()
    feed.begin_generation(1)
    assert feed.ingest_snapshot(_snapshot(10), generation=1, received_at_ms=1_000)

    assert feed.ingest_delta({"result": None, "id": 1}, generation=1) is False
    assert feed.needs_bootstrap is False
    assert feed.last_update_id == 10
    assert feed.error_count == 0


def test_binance_book_level_limit_forces_fail_closed_rebootstrap() -> None:
    feed = _binance(book_level_limit=3)
    feed.begin_generation(1)
    assert feed.ingest_snapshot(_snapshot(10), generation=1, received_at_ms=1_000)

    assert feed.ingest_delta(
        {
            "s": "BTCUSDT",
            "E": 1_100,
            "U": 11,
            "u": 11,
            "b": [["97", "1"]],
            "a": [],
        },
        generation=1,
    )
    assert feed.bid_level_count == 3
    assert feed.ingest_delta(
        {
            "s": "BTCUSDT",
            "E": 1_200,
            "U": 12,
            "u": 12,
            "b": [["96", "1"]],
            "a": [],
        },
        generation=1,
    ) is False

    assert feed.book_overflow_count == 1
    assert feed.needs_bootstrap is True
    assert feed.bid_level_count == 0
    assert feed.ask_level_count == 0


def test_binance_symbol_time_generation_and_buffer_bounds_fail_closed() -> None:
    feed = _binance(buffer_size=1)
    feed.begin_generation(3)
    assert feed.ingest_delta(_delta(11, 11, ts_ms=1_100), generation=3) is False
    assert feed.ingest_delta(_delta(12, 12, ts_ms=1_200), generation=3) is False
    assert feed.buffer_overflow_count == 1
    assert feed.ingest_snapshot(_snapshot(10), generation=3, received_at_ms=1_000) is False
    assert feed.needs_bootstrap is True

    feed.begin_generation(4)
    assert feed.ingest_snapshot(_snapshot(20), generation=4, received_at_ms=2_000)
    assert feed.ingest_delta(
        _delta(21, 21, ts_ms=2_100, symbol="ETHUSDT"),
        generation=4,
    ) is False
    assert feed.symbol_mismatch_count == 1
    assert feed.ingest_delta(_delta(21, 21, ts_ms=2_100), generation=3) is False
    assert feed.dropped_generation_count == 1
    assert feed.ingest_delta(_delta(21, 21, ts_ms=2_100), generation=4) is True
    assert feed.ingest_delta(_delta(22, 22, ts_ms=1_900), generation=4) is False
    assert feed.out_of_order_count == 1
    assert (
        feed.ingest_delta(
            _delta(21, 21, ts_ms=2_200),
            generation=4,
            received_at_ms=2_100,
        )
        is False
    )
    assert feed.out_of_order_count == 2


def _book(
    token_id: str,
    sequence: int,
    timestamp_ms: int,
    *,
    bid: str,
    ask: str,
    event_type: str = "book",
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "asset_id": token_id,
        "window_id": "window-a",
        "timestamp": timestamp_ms,
        "sequence": sequence,
        "bids": [{"price": bid, "size": "10"}],
        "asks": [{"price": ask, "size": "12"}],
    }


def _polymarket() -> PolymarketBookSynchronizer:
    return PolymarketBookSynchronizer(
        window_id="window-a",
        yes_token_id="yes-token",
        no_token_id="no-token",
        max_age_ms=500,
    )


def test_polymarket_requires_fresh_full_books_for_both_tokens() -> None:
    feed = _polymarket()
    feed.begin_generation(1)
    assert (
        feed.ingest(
            _book("yes-token", 10, 1_000, bid="0.39", ask="0.40"),
            generation=1,
        )
        is None
    )
    snapshot = feed.ingest(
        _book("no-token", 20, 1_000, bid="0.59", ask="0.60"),
        generation=1,
    )

    assert snapshot is not None
    assert snapshot.yes_bid == 0.39
    assert snapshot.no_ask == 0.60
    assert snapshot.yes_ask_size == 12.0
    assert feed.state is FeedConnectionState.READY
    assert feed.health(now_ms=1_100).fresh is True


@pytest.mark.parametrize("mutation", [
    "missing_bids", "missing_asks", "empty_bids", "malformed_deep_level",
    "invalid_size", "crossed", "out_of_range", "null_asks",
])
def test_malformed_full_book_invalidates_without_reusing_old_top(mutation) -> None:
    feed = _polymarket()
    feed.begin_generation(1)
    feed.ingest(_book("yes-token", 1, 1_000, bid="0.39", ask="0.40"), generation=1)
    assert feed.ingest(
        _book("no-token", 1, 1_000, bid="0.59", ask="0.60"), generation=1
    ) is not None
    payload = _book("yes-token", 2, 1_100, bid="0.39", ask="0.40")
    if mutation == "missing_bids":
        payload.pop("bids")
    elif mutation == "missing_asks":
        payload.pop("asks")
    elif mutation == "empty_bids":
        payload["bids"] = []
    elif mutation == "malformed_deep_level":
        payload["bids"].append({"price": "NaN", "size": "1"})
    elif mutation == "invalid_size":
        payload["asks"][0]["size"] = "-1"
    elif mutation == "crossed":
        payload["bids"][0]["price"] = "0.5"
    elif mutation == "out_of_range":
        payload["asks"][0]["price"] = "1.2"
    else:
        payload["asks"] = None
    assert feed.ingest(payload, generation=1) is None
    assert feed.state is FeedConnectionState.SYNCING
    assert feed.health(now_ms=1_100).fresh is False
    assert feed.token_health(now_ms=1_100)["yes"]["timestamp_ms"] is None
    assert feed.ingest(
        _book("no-token", 2, 1_100, bid="0.59", ask="0.60"), generation=1
    ) is None
    assert feed.ingest(
        _book("yes-token", 2, 1_100, bid="0.39", ask="0.40"), generation=1
    ) is not None


def test_polymarket_one_stale_token_or_sequence_gap_blocks_snapshot() -> None:
    feed = _polymarket()
    feed.begin_generation(1)
    feed.ingest(
        _book("yes-token", 10, 1_000, bid="0.39", ask="0.40"),
        generation=1,
    )
    feed.ingest(
        _book("no-token", 20, 1_000, bid="0.59", ask="0.60"),
        generation=1,
    )
    stale = feed.ingest(
        _book(
            "no-token",
            21,
            1_700,
            bid="0.58",
            ask="0.59",
            event_type="price_change",
        ),
        generation=1,
    )
    assert stale is None
    assert feed.health(now_ms=1_700).fresh is False

    gap = feed.ingest(
        _book(
            "no-token",
            23,
            1_800,
            bid="0.57",
            ask="0.58",
            event_type="price_change",
        ),
        generation=1,
    )
    assert gap is None
    assert feed.state is FeedConnectionState.SYNCING
    assert feed.gap_count == 1


def test_polymarket_reconnect_resubscribe_and_old_generation_are_fenced() -> None:
    feed = _polymarket()
    feed.begin_generation(1)
    first_subscription = feed.subscription_message()
    assert first_subscription["assets_ids"] == ["yes-token", "no-token"]

    feed.begin_generation(2)
    assert feed.reconnect_count == 1
    assert (
        feed.ingest(
            _book("yes-token", 1, 1_000, bid="0.39", ask="0.40"),
            generation=1,
        )
        is None
    )
    assert feed.dropped_generation_count == 1
    assert feed.state is FeedConnectionState.SYNCING


def test_polymarket_real_sequence_less_full_books_and_price_changes() -> None:
    feed = PolymarketBookSynchronizer(
        window_id="window-a",
        yes_token_id="yes-token",
        no_token_id="no-token",
        condition_id="condition-a",
        max_age_ms=500,
    )
    feed.begin_generation(1)
    book = {
        "event_type": "book",
        "market": "condition-a",
        "timestamp": 1_000,
        "bids": [{"price": "0.4", "size": "5"}],
        "asks": [{"price": "0.5", "size": "6"}],
    }
    assert feed.ingest({**book, "asset_id": "yes-token"}, generation=1) is None
    assert feed.ingest({**book, "asset_id": "no-token"}, generation=1) is not None
    updated = feed.ingest(
        {
            "event_type": "price_change",
            "market": "condition-a",
            "timestamp": 1_001,
            "price_changes": [
                {
                    "asset_id": "yes-token",
                    "price": "0.41",
                    "size": "7",
                    "side": "BUY",
                    "best_bid": "0.41",
                    "best_ask": "0.5",
                },
                {
                    "asset_id": "no-token",
                    "price": "0.48",
                    "size": "8",
                    "side": "SELL",
                    "best_bid": "0.4",
                    "best_ask": "0.48",
                },
            ],
        },
        generation=1,
    )
    assert updated is not None
    assert updated.yes_bid == 0.41
    assert updated.no_ask == 0.48
    assert (
        feed.ingest(
            {**book, "market": "wrong-condition", "asset_id": "yes-token"},
            generation=1,
        )
        is None
    )
    assert feed.token_mismatch_count == 1
    assert (
        feed.ingest(
            {**book, "asset_id": "yes-token", "timestamp": 2_000},
            generation=1,
            received_at_ms=1_999,
        )
        is None
    )
    assert feed.out_of_order_count == 1


@pytest.mark.asyncio
async def test_bounded_queue_drops_oldest_and_transport_resubscribes() -> None:
    queue: BoundedEventQueue[int] = BoundedEventQueue(maxsize=2)
    queue.put_nowait(1)
    queue.put_nowait(2)
    queue.put_nowait(3)
    assert queue.dropped_count == 1
    assert await queue.get() == 2
    assert await queue.get() == 3

    transport = FakeReadOnlyTransport(
        subscription={"type": "market", "assets_ids": ["yes", "no"]},
        queue_size=2,
    )
    first_generation = await transport.connect()
    await transport.disconnect()
    second_generation = await transport.connect()
    assert (first_generation, second_generation) == (1, 2)
    assert transport.subscription_count == 2
    assert transport.reconnect_count == 1
    assert transport.write_capable is False
    assert transport.paper_only is True
