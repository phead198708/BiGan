from __future__ import annotations

import pytest

from bigan.paper_trading.operator.market_depth import MarketDepth
from tests.paper_trading.operator.test_feeds import _book, _polymarket


def ready():
    feed = _polymarket()
    feed.begin_generation(1)
    yes = _book("yes-token", 1, 1000, bid="0.40", ask="0.50")
    yes["bids"].append({"price": "0.39", "size": "7"})
    yes["asks"].append({"price": "0.51", "size": "9"})
    feed.ingest(yes, generation=1)
    assert feed.ingest(_book("no-token", 1, 1000, bid="0.40", ask="0.50"), generation=1)
    return feed


def change(**kwargs):
    return {"event_type": "price_change", "asset_id": "yes-token", "timestamp": 1100,
            "price": "0.40", "size": "0", "side": "BUY", "best_bid": "0.39", "best_ask": "0.50", **kwargs}


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_delete_best_uses_known_next_level_quantity(side):
    feed = ready()
    payload = change()
    if side == "SELL":
        payload.update(price="0.50", side="SELL", best_bid="0.40", best_ask="0.51")
    snapshot = feed.ingest(payload, generation=1)
    assert snapshot is not None
    assert (snapshot.yes_bid, snapshot.yes_bid_size) == ((0.39, 7) if side == "BUY" else (0.40, 10))
    assert (snapshot.yes_ask, snapshot.yes_ask_size) == ((0.51, 9) if side == "SELL" else (0.50, 12))
    assert not feed.needs_bootstrap


def test_advisory_before_depth_blocks_execution_without_erasing_book():
    feed = ready()
    advisory = {"event_type": "best_bid_ask", "asset_id": "yes-token", "timestamp": 1100,
                "best_bid": "0.39", "best_ask": "0.50"}
    assert feed.ingest(advisory, generation=1) is None
    assert feed.token_health(now_ms=1100)["yes"]["timestamp_ms"] == 1000
    assert not feed.health(now_ms=1100).fresh
    assert not feed.needs_bootstrap
    assert feed.ingest(_book("no-token", 2, 1100, bid="0.40", ask="0.50"), generation=1) is None
    snapshot = feed.ingest(change(), generation=1)
    assert snapshot is not None and snapshot.yes_bid_size == 7
    assert feed.health(now_ms=1100).fresh
    # No artificial liquidity timestamp refresh from an unchanged advisory.
    assert feed.ingest({**advisory, "timestamp": 1200}, generation=1) is None
    assert feed.token_health(now_ms=1200)["yes"]["timestamp_ms"] == 1100


def test_deeper_updates_are_retained_when_they_later_become_best():
    feed = ready()
    assert feed.ingest(change(price="0.39", size="25", best_bid="0.40"), generation=1)
    snapshot = feed.ingest(change(timestamp=1101), generation=1)
    assert snapshot is not None and snapshot.yes_bid_size == 25


def test_unreconciled_advisory_resubscribes_after_freshness_deadline():
    feed = ready()
    feed.ingest({"event_type": "best_bid_ask", "asset_id": "yes-token", "timestamp": 1100,
                 "best_bid": "0.39", "best_ask": "0.50"}, generation=1)
    assert feed.ingest(change(timestamp=1700), generation=1) is None
    assert feed.needs_bootstrap and feed.gap_count == 1


@pytest.mark.parametrize("mutation", [
    {"best_bid": "NaN"}, {"size": "NaN"}, {"size": "-1"}, {"side": "BAD"},
    {"best_bid": "0.38", "best_ask": "NaN"},
    {"price": True}, {"size": True}, {"price": "1.2"},
])
def test_ambiguous_or_malformed_depth_requests_resubscription(mutation):
    feed = ready()
    assert feed.ingest(change(**mutation), generation=1) is None
    assert feed.needs_bootstrap
    assert not feed.health(now_ms=1100).fresh
    assert not feed._depth


@pytest.mark.parametrize("later", [change(asset_id="no-token", size="NaN"), None])
def test_later_invalid_batch_change_cannot_return_an_earlier_snapshot(later):
    feed = ready()
    assert feed.ingest({"event_type": "price_change", "timestamp": 1100, "price_changes": [
        change(), later,
    ]}, generation=1) is None
    assert feed.needs_bootstrap


def test_depth_limit_is_enforced_before_publishing_or_mutating(monkeypatch):
    monkeypatch.setattr("bigan.paper_trading.operator.market_depth.MAX_MARKET_LEVELS", 2)
    payload = _book("yes-token", 1, 1000, bid="0.40", ask="0.50")
    book = MarketDepth.from_payload(payload)
    book = book.updated(change(price="0.39", size="7", best_bid="0.40"))
    with pytest.raises(ValueError, match="memory bound"):
        book.updated(change(price="0.38", size="3", best_bid="0.40"))
    assert book.bids == {0.40: 10, 0.39: 7}
    payload["bids"] *= 3
    with pytest.raises(ValueError, match="oversized"):
        MarketDepth.from_payload(payload)


def test_reconnect_cannot_reuse_old_depth_or_pending_advisories():
    feed = ready()
    feed.disconnect()
    assert not feed._depth
    feed.begin_generation(2)
    assert feed.ingest(change(), generation=2) is None
    assert feed.needs_bootstrap


def test_interleaved_token_timestamps_are_not_a_global_sequence_gap():
    feed = ready()
    assert feed.ingest(_book("no-token", 2, 1200, bid="0.40", ask="0.50"), generation=1)
    snapshot = feed.ingest(_book("yes-token", 2, 1100, bid="0.39", ask="0.50"), generation=1)
    assert snapshot is not None and snapshot.timestamp_ms == 1200
    assert feed.token_health(now_ms=1200)["yes"]["timestamp_ms"] == 1100
    assert feed.token_health(now_ms=1200)["no"]["timestamp_ms"] == 1200
    assert feed.health(now_ms=1200).fresh
    assert not feed.token_health(now_ms=1601)["yes"]["fresh"]
    # An actual older event for the same token still cannot change its depth.
    assert feed.ingest(_book("yes-token", 3, 1099, bid="0.10", ask="0.20"), generation=1) is None
    assert feed._depth["yes-token"].top() == (0.39, 0.50)


def test_unconfirmed_delta_waits_for_trade_full_book_without_reconnect_loop():
    feed = ready()
    assert feed.ingest(change(best_bid="0.38"), generation=1) is None
    assert not feed.needs_bootstrap and not feed.health(now_ms=1100).fresh
    # Price-only confirmation is still insufficient evidence of quantity.
    assert feed.ingest({"event_type": "best_bid_ask", "asset_id": "yes-token", "timestamp": 1100,
                        "best_bid": "0.39", "best_ask": "0.50"}, generation=1) is None
    assert not feed.health(now_ms=1100).fresh
    snapshot = feed.ingest(_book("yes-token", 2, 1101, bid="0.38", ask="0.50"), generation=1)
    assert snapshot is not None and snapshot.yes_bid_size == 10
    assert feed.health(now_ms=1101).fresh
    assert not feed.needs_bootstrap
