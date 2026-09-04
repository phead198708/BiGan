"""PONG waiting must not stop bounded market-message consumption."""

from __future__ import annotations

import asyncio
import json

import pytest

from bigan.paper_trading.operator.transports import PublicWebSocketTransport
from tests.paper_trading.operator.test_transports import FakeConnector, FakeSocket


class PongBehindData(FakeSocket):
    """Model a bounded library queue with PONG behind queued data frames."""

    def __init__(self, *, respond: bool = True):
        super().__init__([])
        self.frames: asyncio.Queue[str] = asyncio.Queue(maxsize=2)
        self.pong: asyncio.Future[None] | None = None
        self.ping_started = asyncio.Event()
        self.pong_consumed = asyncio.Event()
        self.respond = respond

    async def ping(self):
        self.pings += 1
        self.pong = asyncio.get_running_loop().create_future()
        self.frames.put_nowait(json.dumps({"value": 1}))
        self.frames.put_nowait(json.dumps({"value": 2}))
        self.ping_started.set()
        return self.pong

    async def recv(self):
        frame = await self.frames.get()
        if self.respond and self.frames.empty() and self.pong is not None:
            self.pong.set_result(None)
            self.pong_consumed.set()
        return frame


def make_transport(socket, *, stop, on_payload=None, connector=None, on_generation=None):
    async def no_backoff(_seconds):
        await asyncio.sleep(0)

    return PublicWebSocketTransport(
        endpoint="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        subscription={"type": "market", "assets_ids": ["yes", "no"]},
        queue_size=4, reconnect_min_seconds=0.01, reconnect_max_seconds=0.02,
        heartbeat_interval_seconds=0.03, application_heartbeat="PING",
        clock_ms=lambda: 1000, on_payload=on_payload or (lambda *_args: None),
        on_generation=on_generation or (lambda _generation: None),
        on_disconnect=lambda: None, connect_factory=connector or FakeConnector([socket]),
        sleep=no_backoff,
    )


def assert_no_connection_tasks():
    assert not [task for task in asyncio.all_tasks()
                if task.get_name().startswith(("public-feed-messages-", "public-feed-heartbeat-"))]


async def test_pong_behind_data_is_consumed_without_a_reconnect():
    stop = asyncio.Event()
    socket = PongBehindData()
    seen = []

    async def payload(value, generation, received):
        seen.append((value, generation, received))
        if len(seen) == 2:
            assert socket.pong_consumed.is_set()
            stop.set()

    transport = make_transport(socket, stop=stop, on_payload=payload)
    await asyncio.wait_for(transport.run(stop), 2)
    assert [row[0]["value"] for row in seen] == [1, 2]
    assert all(row[1:] == (1, 1000) for row in seen)
    assert socket.pings == 1 and socket.sent[-1] == "PING"
    assert transport.connection_error_count == transport.reconnect_count == 0
    assert transport.queue.dropped_count == 0
    assert_no_connection_tasks()


@pytest.mark.parametrize("cancel", [False, True])
async def test_stop_or_cancel_during_pong_cleans_up_all_connection_tasks(cancel):
    stop = asyncio.Event()
    socket = PongBehindData(respond=False)
    transport = make_transport(socket, stop=stop)
    task = asyncio.create_task(transport.run(stop))
    await asyncio.wait_for(socket.ping_started.wait(), 1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    else:
        stop.set()
        await asyncio.wait_for(task, 1)
    assert socket.pong is not None and socket.pong.cancelled()
    assert transport.connection_error_count == 0
    assert not transport.connected
    assert_no_connection_tasks()


@pytest.mark.parametrize("failure", ["missing_pong", "send_stalled", "reader_invalid"])
async def test_failed_connection_tasks_reconnect_and_fence_generations(failure):
    stop = asyncio.Event()

    class BrokenSocket(PongBehindData):
        async def send(self, value):
            await super().send(value)
            if failure == "send_stalled" and value == "PING":
                await asyncio.Future()

        async def ping(self):
            result = await super().ping()
            if failure == "reader_invalid":
                # Reader fails while the heartbeat owns a pending PONG.
                self.frames.get_nowait()
                self.frames.get_nowait()
                self.frames.put_nowait("not-json")
            return result

    first = BrokenSocket(respond=False)
    second = FakeSocket([json.dumps({"value": "recovered"})])
    seen = []

    async def payload(value, generation, received):
        seen.append((value, generation, received))
        if generation == 2:
            stop.set()

    transport = make_transport(first, stop=stop, on_payload=payload,
                               connector=FakeConnector([first, second]))
    await asyncio.wait_for(transport.run(stop), 2)
    assert seen[-1] == ({"value": "recovered"}, 2, 1000)
    assert transport.connection_error_count == transport.reconnect_count == 1
    assert transport.parse_error_count == (1 if failure == "reader_invalid" else 0)
    if first.pong is not None:
        assert first.pong.cancelled()
    assert_no_connection_tasks()


async def test_known_receiver_failure_wins_over_buffered_payload():
    stop = asyncio.Event()
    transport = make_transport(FakeSocket([]), stop=stop)
    transport.queue.put_nowait(({"old": True}, 1, 1000))

    async def failed():
        raise TimeoutError("PONG deadline")

    receiver = asyncio.create_task(failed())
    await asyncio.sleep(0)
    with pytest.raises(TimeoutError, match="PONG deadline"):
        await transport._next_queue_item(receiver, stop)


async def test_bootstrap_buffer_cannot_hide_failed_heartbeat():
    stop = asyncio.Event()
    first = PongBehindData(respond=False)
    second = FakeSocket([json.dumps({"value": "recovered"})])
    seen = []

    async def bootstrap(generation):
        if generation == 1:
            await first.ping_started.wait()
            assert first.pong is not None
            # Wait for the bounded PONG timeout, as if a REST request is busy.
            await asyncio.gather(first.pong, return_exceptions=True)
            await asyncio.sleep(0.02)

    async def payload(value, generation, _received):
        seen.append((value, generation))
        stop.set()

    transport = make_transport(first, stop=stop, on_payload=payload, on_generation=bootstrap,
                               connector=FakeConnector([first, second]))
    await asyncio.wait_for(transport.run(stop), 2)
    assert seen == [({"value": "recovered"}, 2)]
    assert transport.connection_error_count == transport.reconnect_count == 1
    assert_no_connection_tasks()
