from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from dataclasses import replace

from bigan.paper_trading.operator.read_model import OperatorState, OperatorStatusWriter
from bigan.paper_trading.storage import IDEMPOTENCY_INDEX_FILE, SIGNAL_EVENTS_FILE


async def test_sqlite_exclusive_lock_warns_then_recovers(client, bundle):
    path = bundle.operator.session.store.run_dir / IDEMPOTENCY_INDEX_FILE
    db = sqlite3.connect(path)
    try:
        db.execute("BEGIN EXCLUSIVE")
        response = await client.get("/api/v1/dashboard")
        view = await response.json()
        assert response.status == 200
        assert view["account"] and view["recent"]["decisions"]
        assert view["recent"]["fills"] is None
    finally:
        db.rollback()
        db.close()
    assert (await (await client.get("/api/v1/dashboard")).json())["recent"]["fills"]


async def test_incomplete_jsonl_does_not_break_other_sections_and_next_poll_recovers(client, bundle):
    path = bundle.operator.session.store.run_dir / SIGNAL_EVENTS_FILE
    saved = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    view = await (await client.get("/api/v1/dashboard")).json()
    assert view["recent"]["decisions"] is None
    assert view["recent"]["fills"] and view["account"]
    path.write_bytes(saved)
    assert (await (await client.get("/api/v1/dashboard")).json())["recent"]["decisions"]


async def test_atomic_status_replacement_during_api_reads(client, bundle):
    status = bundle.operator.status()
    writer = OperatorStatusWriter(bundle.reader.status_path)
    stop = threading.Event()

    def publish():
        n = 0
        while not stop.is_set():
            writer.write(replace(status, state_reason=f"atomic-publication-{n}"))
            n += 1
            stop.wait(.002)

    thread = threading.Thread(target=publish)
    thread.start()
    try:
        responses = await asyncio.gather(*(client.get("/api/v1/dashboard") for _ in range(12)))
        for response in responses:
            assert response.status == 200
            view = await response.json()
            assert view["status"]["run_id"] == view["account"]["run_id"]
    finally:
        stop.set()
        thread.join(timeout=2)


async def test_rollover_mid_read_retries_without_mixing_runs(bundle, monkeypatch):
    reader, operator = bundle.reader, bundle.operator
    old_status = operator.status()
    old_checkpoint = operator._checkpoint
    bundle.clock.now_ms = bundle.markets[0].end_ts_ms + 1
    await operator.poll()
    new_checkpoint = operator._checkpoint
    new_status = operator.status()
    original_status, original_load = reader._status, reader.checkpoints.load
    statuses = iter((old_status, new_status, new_status, new_status))
    checkpoints = iter((old_checkpoint, new_checkpoint, new_checkpoint, new_checkpoint))
    monkeypatch.setattr(reader, "_status", lambda: next(statuses, original_status()))
    monkeypatch.setattr(reader.checkpoints, "load", lambda **kwargs: next(checkpoints, original_load(**kwargs)))
    view = reader.read()
    assert view["status"]["run_id"] == new_status.run_id
    assert view["account"]["run_id"] == new_status.run_id
    assert view["active_market"]["market_id"] == bundle.markets[1].market_id


def test_persistent_rollover_mismatch_is_bounded_and_returns_only_status(bundle, monkeypatch):
    status = replace(bundle.operator.status(), state=OperatorState.ROLLING_OVER, run_id="paper-" + "b" * 24)
    calls = []

    def changed():
        calls.append(1)
        return status

    monkeypatch.setattr(bundle.reader, "_status", changed)
    started = time.monotonic()
    view = bundle.reader.read()
    assert time.monotonic() - started < .250
    assert len(calls) == 4  # three attempts, final current-status-only projection
    assert view["account"] is None and view["recent"]["fills"] is None
    assert view["status"]["run_id"] == status.run_id
