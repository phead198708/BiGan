from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from bigan.paper_trading.storage import IDEMPOTENCY_INDEX_FILE, PaperRunStore


def fingerprint(root: Path) -> dict:
    """Check content and write-related metadata, not OS-managed access times."""
    result = {}
    for path in [root, *sorted(root.rglob("*"))]:
        stat = path.stat()
        result[str(path.relative_to(root))] = (
            path.is_dir(), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
            None if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return result


async def test_repeated_api_reads_leave_complete_tree_unchanged(client, bundle, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("dashboard attempted recovery or index rebuilding")
    monkeypatch.setattr(PaperRunStore, "resume_existing", forbidden)
    monkeypatch.setattr(PaperRunStore, "recover_ledger", forbidden)
    monkeypatch.setattr(PaperRunStore, "_rebuild_idempotency_index", forbidden)
    before = fingerprint(bundle.reader.root)
    for _ in range(3):
        for route in ("dashboard", "status", "account", "runs", "decisions", "fills", "settlements"):
            assert (await client.get("/api/v1/" + route)).status == 200
    assert fingerprint(bundle.reader.root) == before


async def test_missing_index_is_never_created_by_requests(client, bundle):
    path = bundle.operator.session.store.run_dir / IDEMPOTENCY_INDEX_FILE
    path.unlink()
    before = fingerprint(bundle.reader.root)
    for _ in range(3):
        assert (await client.get("/api/v1/fills")).status == 503
        assert (await client.get("/api/v1/dashboard")).status == 200
    assert not path.exists()
    assert fingerprint(bundle.reader.root) == before


@pytest.mark.parametrize("method,kwargs", [
    ("append_decision", {"decision": None, "ledger_event": None, "snapshot": None}),
    ("append_settlement", {"settlement": None, "ledger_event": None, "snapshot": None}),
    ("_append_observation", {"ledger_event": None, "snapshot": None}),
    ("_index_decision", {"decision": None}), ("_rebuild_idempotency_index", {"decisions": ()}),
    ("_append_jsonl", {"name": "forbidden.jsonl", "payload": {}}),
    ("_write_atomic_json", {"name": "forbidden.json", "payload": {}}),
    ("recover_ledger", {}), ("create_new", {}), ("resume_existing", {}),
])
def test_read_only_mutations_rejected_before_io(bundle, monkeypatch, method, kwargs):
    store = PaperRunStore.open_read_only(output_dir=bundle.reader.root, run_id=bundle.operator.run_id)
    before = fingerprint(bundle.reader.root)

    def forbidden(*_args, **_kwargs):
        pytest.fail("read-only mutation reached filesystem I/O")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "mkdir", forbidden)
        patch.setattr(Path, "unlink", forbidden)
        patch.setattr(sqlite3, "connect", forbidden)
        with pytest.raises(PermissionError, match="read-only"):
            getattr(store, method)(**kwargs)
    assert fingerprint(bundle.reader.root) == before


def test_sqlite_connection_is_ro_query_only_and_not_immutable(bundle, monkeypatch):
    store = PaperRunStore.open_read_only(output_dir=bundle.reader.root, run_id=bundle.operator.run_id)
    before = fingerprint(bundle.reader.root)
    connect, calls = sqlite3.connect, []

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", observed)
    with store._open_idempotency_index() as db:
        assert db.execute("PRAGMA query_only").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM decision_events")
    assert "mode=ro" in calls[0][0][0]
    assert "immutable" not in calls[0][0][0]
    assert calls[0][1]["uri"] is True
    assert fingerprint(bundle.reader.root) == before


def test_wal_database_is_refused_without_sidecar_creation(bundle):
    path = bundle.operator.session.store.run_dir / IDEMPOTENCY_INDEX_FILE
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
    db.close()
    before = fingerprint(bundle.reader.root)
    assert bundle.reader.read()["recent"]["fills"] is None
    assert fingerprint(bundle.reader.root) == before


async def test_historical_run_enumeration_stays_read_only(bundle, monkeypatch):
    old = bundle.operator.run_id
    bundle.clock.now_ms = bundle.markets[0].end_ts_ms + 1
    await bundle.operator.poll()
    opened = []
    original = PaperRunStore.open_read_only

    def observed(**kwargs):
        store = original(**kwargs)
        assert store.read_only
        opened.append(store.manifest.run_id)
        return store

    monkeypatch.setattr(PaperRunStore, "open_read_only", observed)
    before = fingerprint(bundle.reader.root)
    assert bundle.reader.read()["recent"]["settlements"]
    assert old in opened
    assert fingerprint(bundle.reader.root) == before
