"""Source observations remain inspectable when the execution gate is closed."""

from dataclasses import replace

import pytest

from bigan.paper_trading.operator.opening_reference import bind_opening_reference
from bigan.paper_trading.operator.pricing_inputs import ReferencePriceSample
from bigan.paper_trading.operator.read_model import OperatorStatus
from bigan.paper_trading.operator.runtime import PaperTradingOperator
from tests.paper_trading.operator.test_feeds import _book, _polymarket
from tests.paper_trading.operator.test_market_depth import change, ready
from tests.paper_trading.operator.test_read_model import _status
from tests.paper_trading.operator.test_runtime import (
    FakeClock,
    FakeDiscovery,
    FakeResolution,
    _config,
    _final,
    _market,
    _ready_operator,
    _selection,
)


def test_legacy_status_has_no_invented_prices():
    payload = _status().to_dict()
    del payload["market_data"]
    assert OperatorStatus.from_dict(payload).market_data == {}
    del payload["feeds"]
    with pytest.raises(ValueError, match="fields"):
        OperatorStatus.from_dict(payload)


def test_per_token_quotes_do_not_invent_missing_side_or_refresh_its_time():
    feed = _polymarket()
    feed.begin_generation(1)
    feed.ingest(_book("yes-token", 1, 1000, bid="0.40", ask="0.50"), generation=1)
    values = feed.quote_observations(now_ms=1100)
    assert values["yes"]["ask"] == 0.50
    assert values["yes"]["ask_size"] == 12
    assert values["yes"]["fresh"] is True
    assert values["no"]["ask"] is None
    assert values["no"]["fresh"] is False
    feed.ingest(_book("no-token", 1, 1400, bid="0.41", ask="0.51"), generation=1)
    values = feed.quote_observations(now_ms=1550)
    assert values["yes"]["timestamp_ms"] == 1000
    assert values["yes"]["fresh"] is False
    assert values["no"]["timestamp_ms"] == 1400
    assert values["no"]["fresh"] is True


def test_unreconciled_depth_is_hidden_and_reconnect_clears_quotes():
    feed = ready()
    feed.ingest({"event_type": "best_bid_ask", "asset_id": "yes-token", "timestamp": 1100,
                 "best_bid": "0.39", "best_ask": "0.50"}, generation=1)
    pending = feed.quote_observations(now_ms=1100)["yes"]
    assert pending["confirmed"] is False
    assert pending["ask"] is None and pending["bid_size"] is None
    feed.ingest(change(), generation=1)
    confirmed = feed.quote_observations(now_ms=1100)["yes"]
    assert (confirmed["bid"], confirmed["bid_size"]) == (0.39, 7)
    assert confirmed["confirmed"] is True
    # Return values are independent dictionaries, not mutable feed state.
    confirmed["bid"] = 0.99
    assert feed.quote_observations(now_ms=1100)["yes"]["bid"] == 0.39
    feed.disconnect()
    for quote in feed.quote_observations(now_ms=1100).values():
        assert quote["ask"] is None and quote["fresh"] is False
    feed.begin_generation(2)
    assert feed.quote_observations(now_ms=1100)["yes"]["timestamp_ms"] is None


async def test_oracle_opening_and_clob_visible_without_binance_or_any_decision(tmp_path):
    clock = FakeClock(10_000)
    market = replace(_market(1), oracle_twap_lookback_seconds=60, reference_price_at_start=None,
                     resolution_source="https://data.chain.link/streams/btc-usd-twap-60s-streams")
    market = bind_opening_reference(market, {
        "openPrice": 100.25, "closePrice": None, "timestamp": 9000,
        "completed": False, "incomplete": True, "cached": False,
    }, requested_at_ms=9000, received_at_ms=9500)
    operator = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([None]), clock_ms=clock,
    )
    await operator.start()
    try:
        sample = ReferencePriceSample(9900, 10_000, 101.123456,
                                      "polymarket_rtds_chainlink_twap:btc/usd:60s")
        assert await operator.ingest_oracle(sample, generation=operator.generation)
        for token in (market.yes_token_id, market.no_token_id):
            book = _book(token, 1, 10_000, bid="0.40", ask="0.50")
            book["window_id"] = market.window_id
            await operator.ingest_market_message(book, generation=operator.generation)
        status = operator.status()
        assert status.last_decision is None
        assert status.active_market["reference_price_at_start"] == 100.25
        assert status.active_market["opening_reference"]["price"] == 100.25
        prices = status.market_data
        assert prices["spot"]["value"] is None
        assert prices["oracle"]["value"] == sample.price
        assert prices["oracle"]["kind"] == "published_twap"
        assert prices["oracle"]["lookback_seconds"] == 60
        assert prices["oracle"]["timestamp_ms"] == 9900
        assert prices["oracle"]["received_at_ms"] == 10_000
        assert prices["oracle"]["age_ms"] == 100
        assert prices["oracle"]["fresh"] is True
        assert prices["up"]["ask"] == prices["down"]["ask"] == 0.5
        assert not status.pricing_inputs["ready"]
        before = operator.pricing_provider.oracle_sample_count
        for _ in range(10):
            assert operator.status().market_data == prices
        assert operator.pricing_provider.oracle_sample_count == before
        assert operator.session.runner.decision_count == 0
        await operator.disconnect_feed("chainlink", window_generation=operator.generation)
        assert operator.status().market_data["oracle"]["fresh"] is False
    finally:
        await operator.shutdown()
    assert operator.status().market_data["oracle"]["fresh"] is False


async def test_rollover_does_not_reuse_previous_market_prices(tmp_path):
    clock = FakeClock(10_000)
    first, second = _market(1), _market(2, start=900_000)
    operator = PaperTradingOperator(
        config=_config(tmp_path), discovery=FakeDiscovery([_selection(first), _selection(second)]),
        resolution=FakeResolution([_final(first)]), clock_ms=clock,
    )
    await operator.start()
    try:
        await _ready_operator(operator, clock)
        assert operator.status().market_data["spot"]["value"] == 102
        # A recent price is not fresh when depth synchronization has a gap.
        await operator.ingest_binance_delta(
            {"s": "BTCUSDT", "E": 10_000, "U": 13, "u": 13, "b": [], "a": []},
            generation=operator.generation, received_at_ms=10_000,
        )
        assert operator.status().market_data["spot"]["fresh"] is False
        clock.now_ms = first.end_ts_ms + 1
        await operator.poll()
        values = operator.status().market_data
        assert values["window_id"] == second.window_id
        assert values["spot"]["value"] is None
        assert values["oracle"]["value"] is None
        assert values["up"]["ask"] is None
        assert values["up"]["token_id"] == second.yes_token_id
    finally:
        await operator.shutdown()
