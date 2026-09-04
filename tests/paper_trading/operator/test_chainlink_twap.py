from __future__ import annotations

import asyncio
import json

import pytest

from bigan.paper_trading.operator.chainlink_twap import TWAP_TOPICS, parse_twap_sample
from bigan.paper_trading.operator.transports import (
    ChainlinkReadonlyFeed,
    PublicWebSocketTransport,
    chainlink_subscription,
)
from tests.paper_trading.operator.test_transports import FakeSocket


def message(*, symbol="btc/usd", lookback=60):
    return {"topic": TWAP_TOPICS[lookback], "type": "update", "timestamp": 1100,
            "payload": {"symbol": symbol, "timestamp": 1000, "value": 1,
                        "full_accuracy_value": "81287352616498420000000", "window_s": lookback}}


@pytest.mark.parametrize("symbol,lookback", [("btc/usd", 30), ("btc/usd", 60), ("eth/usd", 60)])
def test_subscription_and_exact_e18_parser(symbol, lookback):
    assert chainlink_subscription(symbol, lookback)["subscriptions"] == [
        {"topic": TWAP_TOPICS[lookback], "type": "update", "filters": json.dumps({"symbol": symbol}, separators=(",", ":"))},
    ]
    sample = parse_twap_sample(message(symbol=symbol, lookback=lookback), symbol=symbol,
                               lookback_seconds=lookback, received_at_ms=1200)
    assert sample.price == 81287.35261649842  # full_accuracy_value, not display convenience value.
    assert sample.timestamp_ms == 1000  # Observation time, not publisher/receipt time.


@pytest.mark.parametrize("changes", [
    {"timestamp": 1201}, {"timestamp": True}, {"timestamp": 0}, {"window_s": 30},
    {"window_s": True}, {"full_accuracy_value": None}, {"full_accuracy_value": "-1"},
    {"full_accuracy_value": "0"}, {"full_accuracy_value": "nan"}, {"full_accuracy_value": "1" * 79},
])
def test_invalid_twap_identity_time_and_prices_are_rejected(changes):
    payload = message()
    payload["payload"].update(changes)
    with pytest.raises(ValueError):
        parse_twap_sample(payload, symbol="btc/usd", lookback_seconds=60, received_at_ms=1200)


@pytest.mark.parametrize("symbol,topic", [("eth/usd", TWAP_TOPICS[60]),
                                         ("btc/usd", TWAP_TOPICS[30]), ("btc/usd", "crypto_prices_chainlink")])
async def test_wrong_source_never_reaches_oracle_callback(symbol, topic):
    samples = []
    feed = ChainlinkReadonlyFeed(expected_symbol="btc/usd", source="unused", lookback_seconds=60,
                                 on_sample=lambda sample, generation: samples.append(sample))
    payload = message(symbol=symbol)
    payload["topic"] = topic
    await feed.on_raw(json.dumps(payload), generation=1, received_at_ms=1200)
    assert samples == []


async def test_application_heartbeat_is_sent_even_with_continuous_updates():
    stop = asyncio.Event()

    class BusySocket(FakeSocket):
        async def recv(self):
            await asyncio.sleep(.01)
            return '{"value":1}'

        async def send(self, value):
            await super().send(value)
            if value == "PING":
                stop.set()

    socket = BusySocket([])
    transport = PublicWebSocketTransport(
        endpoint="wss://ws-live-data.polymarket.com", subscription=chainlink_subscription("btc/usd", 60),
        queue_size=100, on_payload=lambda *_: None, on_generation=lambda _: None, on_disconnect=lambda: None,
        reconnect_min_seconds=1, reconnect_max_seconds=2,
        heartbeat_interval_seconds=.05, application_heartbeat="PING", clock_ms=lambda: 1000,
    )
    await asyncio.wait_for(transport._receive_to_queue(socket, 1, stop), 1)
    assert "PING" in socket.sent
