"""Polymarket live feed contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bigan.v8.polymarket import (
    BinanceBTCReferenceTick,
    MockBinanceBTCReferenceFeed,
    MockPolymarketLiveFeed,
    PolymarketLiveOrderBook,
    PolymarketLivePaperConfig,
)


def test_mock_polymarket_feed_emits_market_metadata_and_up_down_books(
    tmp_path: Path,
) -> None:
    config = PolymarketLivePaperConfig(
        run_id="live-feed-contracts",
        output_dir=tmp_path,
    )
    feed = MockPolymarketLiveFeed(config)

    markets = feed.markets()
    orderbooks = feed.orderbooks(markets)

    assert {market.market_family for market in markets} == {
        "btc_updown_5m",
        "btc_updown_15m",
        "btc_updown_1h",
    }
    assert len(orderbooks) == len(markets) * 3 * 2
    for market in markets:
        timestamps = {book.ts for book in orderbooks if book.market_id == market.market_id}
        for ts in timestamps:
            outcomes = {
                book.outcome
                for book in orderbooks
                if book.market_id == market.market_id and book.ts == ts
            }
            assert outcomes == {"UP", "DOWN"}
    for row in (*markets, *orderbooks):
        _assert_readonly_safe(row.to_dict())


def test_mock_binance_reference_feed_emits_ticks_and_candles(tmp_path: Path) -> None:
    config = PolymarketLivePaperConfig(
        run_id="binance-reference-contracts",
        output_dir=tmp_path,
    )
    markets = MockPolymarketLiveFeed(config).markets()
    feed = MockBinanceBTCReferenceFeed(config)

    ticks = feed.ticks(markets)
    candles = feed.candles(markets)

    assert len(ticks) == len(markets) * 3
    assert len(candles) == len(markets)
    assert {candle.market_id for candle in candles} == {
        market.market_id for market in markets
    }
    for row in (*ticks, *candles):
        _assert_readonly_safe(row.to_dict())


def test_write_capable_or_wallet_enabled_feed_payloads_are_rejected(
    tmp_path: Path,
) -> None:
    config = PolymarketLivePaperConfig(
        run_id="unsafe-feed-contracts",
        output_dir=tmp_path,
    )
    market = MockPolymarketLiveFeed(config).markets()[0]

    with pytest.raises(ValueError, match="write-capable"):
        PolymarketLiveOrderBook(
            market_id=market.market_id,
            token_id=market.up_token_id,
            outcome="UP",
            ts=market.market_start_ts,
            received_ts=market.market_start_ts,
            bid_price=0.48,
            ask_price=0.50,
            mid_price=0.49,
            bid_size=1.0,
            ask_size=1.0,
            liquidity_depth=2.0,
            write_capable=True,
        )

    with pytest.raises(ValueError, match="wallet_signing_enabled"):
        BinanceBTCReferenceTick(
            ts=market.market_start_ts,
            received_ts=market.market_start_ts,
            bid_price=65_000.0,
            ask_price=65_001.0,
            mid_price=65_000.5,
            last_price=65_000.5,
            wallet_signing_enabled=True,
        )


def _assert_readonly_safe(payload: dict) -> None:
    assert payload["read_only"] is True
    assert payload["write_capable"] is False
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["broker_exchange_write_enabled"] is False
    assert payload["live_exchange_write_enabled"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
