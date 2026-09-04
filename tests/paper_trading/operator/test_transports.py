from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime

import pytest

from bigan.features.binance_ofi import BinanceOFICalculator
from bigan.paper_trading.operator.discovery import DiscoveryFilters
from bigan.paper_trading.operator.feeds import (
    BinanceDepthSynchronizer,
    PolymarketBookSynchronizer,
)
from bigan.paper_trading.operator.transports import (
    BinanceReadonlyFeed,
    ChainlinkReadonlyFeed,
    GammaDiscoveryClient,
    PolymarketReadonlyFeed,
    PublicWebSocketTransport,
    binance_subscription,
    chainlink_subscription,
)


class FakeHTTP:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[tuple[str, Mapping[str, str | int]]] = []

    async def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> object:
        self.calls.append((endpoint, dict(params or {})))
        return self.payload


class FakeSocket:
    def __init__(self, messages: list[str | bytes | Exception]) -> None:
        self.messages = list(messages)
        self.sent: list[str | bytes] = []
        self.pings = 0
        self.message_delivered = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        if not self.messages:
            await asyncio.Future()
        value = self.messages.pop(0)
        if isinstance(value, Exception):
            raise value
        self.message_delivered.set()
        return value

    async def ping(self) -> object:
        self.pings += 1
        return None


class FakeContext(AbstractAsyncContextManager[FakeSocket]):
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> FakeSocket:
        return self.socket

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeConnector:
    def __init__(self, sockets: list[FakeSocket]) -> None:
        self.sockets = list(sockets)
        self.endpoints: list[str] = []

    def __call__(self, endpoint: str) -> FakeContext:
        self.endpoints.append(endpoint)
        return FakeContext(self.sockets.pop(0))


async def test_public_websocket_reconnects_resubscribes_and_fences_generations() -> None:
    stop = asyncio.Event()
    first = FakeSocket([OSError("disconnect")])
    second = FakeSocket([json.dumps({"value": 2})])
    connector = FakeConnector([first, second])
    generations: list[int] = []
    received: list[tuple[int, int]] = []
    sleeps: list[float] = []
    now = iter((1_000, 2_000))

    async def on_payload(payload: Mapping[str, object], generation: int, _received: int) -> None:
        received.append((int(payload["value"]), generation))
        stop.set()

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    transport = PublicWebSocketTransport(
        endpoint="wss://stream.binance.com/ws",
        subscription=binance_subscription("BTCUSDT"),
        queue_size=2,
        reconnect_min_seconds=1,
        reconnect_max_seconds=4,
        heartbeat_interval_seconds=5,
        clock_ms=lambda: next(now),
        on_payload=on_payload,
        on_generation=lambda generation: generations.append(generation),
        on_disconnect=lambda: None,
        connect_factory=connector,
        sleep=fake_sleep,
    )
    await transport.run(stop)

    assert generations == [1, 2]
    assert received == [(2, 2)]
    assert sleeps == [1.0]
    assert transport.health().reconnect_count == 1
    assert len(first.sent) == len(second.sent) == 1


async def test_public_websocket_subscribes_and_buffers_before_bootstrap() -> None:
    stop = asyncio.Event()
    bootstrap_started = asyncio.Event()
    release_bootstrap = asyncio.Event()
    socket = FakeSocket([json.dumps({"value": 7})])

    async def on_generation(_generation: int) -> None:
        assert socket.sent, "subscription must be sent before REST bootstrap starts"
        bootstrap_started.set()
        await release_bootstrap.wait()

    async def on_payload(*_args: object) -> None:
        stop.set()

    transport = PublicWebSocketTransport(
        endpoint="wss://stream.binance.com/ws",
        subscription=binance_subscription("BTCUSDT"),
        queue_size=2,
        reconnect_min_seconds=1,
        reconnect_max_seconds=2,
        heartbeat_interval_seconds=5,
        clock_ms=lambda: 1_000,
        on_payload=on_payload,
        on_generation=on_generation,
        on_disconnect=lambda: None,
        connect_factory=FakeConnector([socket]),
    )
    task = asyncio.create_task(transport.run(stop))
    try:
        await bootstrap_started.wait()

        await asyncio.wait_for(socket.message_delivered.wait(), timeout=1.0)
        assert transport.queue.size == 1
        release_bootstrap.set()
        await task
    finally:
        release_bootstrap.set()
        stop.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert transport.message_count == 1


async def test_public_websocket_supports_application_ping_and_protocol_pong() -> None:
    stop = asyncio.Event()
    socket = FakeSocket([TimeoutError(), json.dumps({"value": 1})])

    async def on_payload(*_args: object) -> None:
        stop.set()

    transport = PublicWebSocketTransport(
        endpoint="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        subscription={"type": "market", "assets_ids": ["yes", "no"]},
        queue_size=2,
        reconnect_min_seconds=1,
        reconnect_max_seconds=2,
        heartbeat_interval_seconds=1,
        application_heartbeat="PING",
        clock_ms=lambda: 1_000,
        on_payload=on_payload,
        on_generation=lambda _generation: None,
        on_disconnect=lambda: None,
        connect_factory=FakeConnector([socket]),
    )
    await transport.run(stop)

    assert socket.sent[1] == "PING"
    assert socket.pings == 1


@pytest.mark.parametrize(
    "endpoint,subscription",
    [
        ("ws://stream.binance.com/ws", {"method": "SUBSCRIBE"}),
        ("wss://example.com/private/order", {"type": "market"}),
        ("wss://example.com/ws", {"operation": "create_order"}),
        ("wss://example.com/ws", {"authorization": "secret"}),
    ],
)
def test_websocket_transport_rejects_unsafe_channels(
    endpoint: str,
    subscription: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PublicWebSocketTransport(
            endpoint=endpoint,
            subscription=subscription,
            queue_size=1,
            reconnect_min_seconds=1,
            reconnect_max_seconds=2,
            heartbeat_interval_seconds=1,
            clock_ms=lambda: 0,
            on_payload=lambda *_args: None,
            on_generation=lambda _generation: None,
            on_disconnect=lambda: None,
        )


async def test_binance_readonly_feed_bootstraps_then_applies_delta() -> None:
    calculator = BinanceOFICalculator(symbol="BTCUSDT", zscore_min_samples=1)
    synchronizer = BinanceDepthSynchronizer(
        calculator=calculator,
        symbol="BTCUSDT",
        max_age_ms=1_000,
        delta_buffer_size=10,
    )
    http = FakeHTTP(
        {"lastUpdateId": 10, "bids": [["100", "2"]], "asks": [["101", "3"]]}
    )
    feed = BinanceReadonlyFeed(
        symbol="BTCUSDT",
        depth_endpoint="https://api.binance.com/api/v3/depth",
        synchronizer=synchronizer,
        http=http,
        clock_ms=lambda: 1_000,
    )
    await feed.on_generation(1)
    feed.on_payload(
        {"s": "BTCUSDT", "E": 1_001, "U": 11, "u": 11, "b": [["100", "4"]], "a": [["101", "2"]]},
        1,
        1_002,
    )

    assert synchronizer.last_update_id == 11
    assert synchronizer.health(now_ms=1_002).fresh
    assert http.calls[0][1] == {"symbol": "BTCUSDT", "limit": 1000}

    with pytest.raises(ConnectionError, match="immediate re-bootstrap"):
        feed.on_payload(
            {
                "s": "BTCUSDT",
                "E": 1_003,
                "U": 13,
                "u": 13,
                "b": [],
                "a": [],
            },
            1,
            1_003,
        )
    assert synchronizer.needs_bootstrap is True


async def test_polymarket_wrapper_emits_only_complete_dual_token_snapshot() -> None:
    sync = PolymarketBookSynchronizer(
        window_id="w1",
        yes_token_id="yes",
        no_token_id="no",
        max_age_ms=100,
    )
    snapshots: list[tuple[int, int]] = []
    feed = PolymarketReadonlyFeed(
        synchronizer=sync,
        on_snapshot=lambda snapshot, generation: snapshots.append(
            (snapshot.timestamp_ms, generation)
        ),
    )
    feed.on_generation(1)
    base = {
        "event_type": "book",
        "sequence": 1,
        "timestamp": 1_000,
        "bids": [{"price": "0.4", "size": "5"}],
        "asks": [{"price": "0.5", "size": "6"}],
    }
    await feed.on_payload({**base, "asset_id": "yes"}, 1, 1_000)
    await feed.on_payload({**base, "asset_id": "no"}, 1, 1_000)
    assert snapshots == [(1_000, 1)]


async def test_chainlink_transport_reuses_strict_parser() -> None:
    samples = []
    feed = ChainlinkReadonlyFeed(
        expected_symbol="btc/usd",
        source="polymarket_rtds_chainlink:btc/usd",
        on_sample=lambda sample, generation: samples.append((sample, generation)),
    )
    raw = json.dumps(
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": 1_001,
            "payload": {"symbol": "btc/usd", "timestamp": 1_000, "value": 100_000},
        }
    )
    await feed.on_raw(raw, generation=3, received_at_ms=1_002)
    await feed.on_raw("{bad", generation=3, received_at_ms=1_003)

    assert samples[0][0].price == 100_000
    assert samples[0][0].received_at_ms == 1_002
    assert samples[0][1] == 3
    assert feed.parse_error_count == 1


def test_subscription_contracts_are_public_and_identity_bound() -> None:
    assert binance_subscription("BTCUSDT")["params"] == ["btcusdt@depth@100ms"]
    assert chainlink_subscription("BTC/USD")["subscriptions"] == [
        {"topic": "crypto_prices_chainlink", "type": "update", "filters": "btc/usd"}
    ]


@pytest.mark.parametrize("public_shape", [False, True])
async def test_gamma_client_queries_deterministic_exact_current_and_next_slugs(public_shape) -> None:
    class SlugHTTP:
        def __init__(self) -> None:
            self.slugs: list[str] = []

        async def get_json(self, _endpoint: str, *, params=None):
            slug = str(params["slug"])
            self.slugs.append(slug)
            start = int(slug.rsplit("-", 1)[1]) * 1_000
            if start > 2_700_000:
                return []
            rows = [
                {
                    "id": f"market-{start}",
                    "conditionId": f"condition-{start}",
                    "slug": slug,
                    "question": "BTC Up or Down",
                    "start_ts_ms": start,
                    "end_ts_ms": start + 900_000,
                    "active": True,
                    "closed": False,
                    "acceptingOrders": True,
                    "outcomes": ["Up", "Down"],
                    "clobTokenIds": [f"yes-{start}", f"no-{start}"],
                    "resolutionSource": "chainlink",
                    "referencePriceAtStart": 100_000,
                }
            ]
            if public_shape:
                row = rows[0]
                row.pop("start_ts_ms")
                row.pop("end_ts_ms")
                row["startDate"] = "1970-01-01T00:00:00Z"
                row["eventStartTime"] = datetime.fromtimestamp(start / 1000, UTC).isoformat()
                row["endDate"] = datetime.fromtimestamp((start + 900_000) / 1000, UTC).isoformat()
            return rows

    http = SlugHTTP()
    client = GammaDiscoveryClient(
        endpoint="https://gamma-api.polymarket.com/markets",
        http=http,
    )
    selected = await client.discover(
        filters=DiscoveryFilters(
            underlying="BTC",
            market_type="binary_up_down",
            window_duration_ms=900_000,
            slug_pattern=r"btc-updown-15m-\d+",
            max_preopen_ms=1_800_000,
        ),
        now_ms=2_000_000,
    )
    assert selected.current is not None
    assert selected.current.start_ts_ms == 1_800_000
    assert selected.next is not None
    assert selected.next.start_ts_ms == 2_700_000
    assert http.slugs == [
        "btc-updown-15m-1800",
        "btc-updown-15m-2700",
        "btc-updown-15m-3600",
    ]
