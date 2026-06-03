"""Acceptance tests for the Phase 4.3 low-latency BTC-15M feature path."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bigan.canonical.writer import warehouse_files
from bigan.features.aggregation import aggregate_features_15m_v1
from bigan.features.low_latency import (
    IncrementalBtc15mFeaturePath,
    JsonlRawQueue,
    run_low_latency_feature_queue_batch,
)


def _ts_at(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp() * 1000
    )


def _identity(ts: int, *, source_symbol: str = "btc-up-token") -> dict:
    side = "DOWN" if "down" in source_symbol.lower() else "UP"
    return {
        "ts": ts,
        "message_ts": ts,
        "ingest_ts": ts + 100,
        "source": "polymarket",
        "source_symbol": source_symbol,
        "source_market": "0xmkt",
        "canonical_symbol": _btc15_canonical_symbol(ts, side=side),
        "provenance": "ws",
    }


def _btc15_canonical_symbol(ts: int, *, side: str = "UP") -> str:
    round_start = (int(ts) // (15 * 60_000)) * (15 * 60)
    return f"BTC-15M:btc-updown-15m-{round_start}:{side}"


def _tob(
    ts: int,
    bid: float,
    ask: float,
    *,
    source_symbol: str = "btc-up-token",
    canonical_symbol: str | None = None,
) -> dict:
    return {
        **_identity(ts, source_symbol=source_symbol),
        "canonical_symbol": canonical_symbol or _identity(ts, source_symbol=source_symbol)["canonical_symbol"],
        "bid_price": bid,
        "ask_price": ask,
        "spread": ask - bid,
    }


def _depth(
    ts: int,
    side: str,
    level: int,
    size: float,
    *,
    source_symbol: str = "btc-up-token",
) -> dict:
    price = 0.50 - level * 0.01 if side == "BID" else 0.52 + level * 0.01
    return {
        **_identity(ts, source_symbol=source_symbol),
        "side": side,
        "level": level,
        "price": price,
        "size": size,
        "snapshot_hash": f"h-{ts}",
    }


def _trade(
    ts: int,
    price: float,
    size: float,
    side: str,
    *,
    source_symbol: str = "btc-up-token",
) -> dict:
    return {
        **_identity(ts, source_symbol=source_symbol),
        "price": price,
        "size": size,
        "side": side,
        "fee_rate_bps": 0.0,
        "trade_id": f"trade-{ts}-{side}-{size}",
    }


def _sorted_feature_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["feature_ts"], row["source"], row["source_symbol"]))


def test_jsonl_raw_queue_replays_canonical_rows_from_cursor(tmp_path: Path) -> None:
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")
    t0 = _ts_at(2026, 5, 28, 1, 0)

    queue.append("raw_top_of_book", _tob(t0, 0.49, 0.51), published_at_ms=t0 + 1)
    queue.append("raw_trades", _trade(t0 + 5_000, 0.50, 3, "BUY"), published_at_ms=t0 + 2)

    first_batch, cursor = queue.read_from(0)
    second_batch, next_cursor = queue.read_from(cursor)

    assert [item.table for item in first_batch] == ["raw_top_of_book", "raw_trades"]
    assert first_batch[0].row["canonical_symbol"] == _btc15_canonical_symbol(t0)
    assert cursor == 2
    assert second_batch == []
    assert next_cursor == cursor


def test_incremental_btc15_features_match_batch_recompute_for_ordered_rows(
    tmp_path: Path,
) -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    ingest_ts = t0 + 999
    raw_items = [
        ("raw_top_of_book", _tob(t0 + 5_000, 0.49, 0.51)),
        ("raw_orderbook_snapshot", _depth(t0 + 30_000, "BID", 0, 100)),
        ("raw_orderbook_snapshot", _depth(t0 + 30_000, "ASK", 0, 50)),
        ("raw_trades", _trade(t0 + 50_000, 0.50, 3, "BUY")),
        ("raw_top_of_book", _tob(t0 + 65_000, 0.52, 0.54)),
        ("raw_trades", _trade(t0 + 95_000, 0.53, 2, "SELL")),
        (
            "raw_top_of_book",
            _tob(
                t0 + 70_000,
                0.10,
                0.12,
                source_symbol="eth-up-token",
                canonical_symbol="ETH-15M:eth-updown-15m-1778423700:UP",
            ),
        ),
    ]
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")
    for idx, (table, row) in enumerate(raw_items):
        queue.append(table, row, published_at_ms=t0 + idx)
    queued, cursor = queue.read_from(0)
    path = IncrementalBtc15mFeaturePath(ingest_ts=ingest_ts)
    path.apply_queue_items(queued)

    batch_rows = aggregate_features_15m_v1(
        top_of_book_rows=[row for table, row in raw_items if table == "raw_top_of_book"],
        orderbook_rows=[row for table, row in raw_items if table == "raw_orderbook_snapshot"],
        trade_rows=[row for table, row in raw_items if table == "raw_trades"],
        ingest_ts=ingest_ts,
    )
    batch_btc_rows = [
        row
        for row in batch_rows
        if str(row.get("canonical_symbol") or "").startswith("BTC-15M:")
    ]

    assert _sorted_feature_rows(path.latest_feature_rows()) == _sorted_feature_rows(batch_btc_rows)
    assert cursor == len(raw_items)
    assert {row["source_symbol"] for row in path.latest_feature_rows()} == {"btc-up-token"}


def test_incremental_btc15_features_update_when_late_raw_row_changes_prior_minute() -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    feature_ts = t0 + 60_000
    ingest_ts = t0 + 999
    path = IncrementalBtc15mFeaturePath(ingest_ts=ingest_ts)

    path.apply_table_row("raw_top_of_book", _tob(t0 + 5_000, 0.49, 0.51))
    initial = next(row for row in path.latest_feature_rows() if row["feature_ts"] == feature_ts)
    assert initial["trade_count_1m"] == 0

    updates = path.apply_table_row("raw_trades", _trade(t0 + 50_000, 0.50, 3, "BUY"))
    updated = next(row for row in path.latest_feature_rows() if row["feature_ts"] == feature_ts)

    assert any(row["feature_ts"] == feature_ts for row in updates)
    assert updated["trade_count_1m"] == 1
    assert updated["trade_volume_1m"] == pytest.approx(3.0)

    batch_rows = aggregate_features_15m_v1(
        top_of_book_rows=[_tob(t0 + 5_000, 0.49, 0.51)],
        orderbook_rows=[],
        trade_rows=[_trade(t0 + 50_000, 0.50, 3, "BUY")],
        ingest_ts=ingest_ts,
    )
    assert _sorted_feature_rows(path.latest_feature_rows()) == _sorted_feature_rows(batch_rows)


def test_low_latency_queue_batch_persists_cursor_state_and_writes_changed_rows(
    tmp_path: Path,
) -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    warehouse = tmp_path / "warehouse"
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")
    cursor_path = tmp_path / "queue.cursor"
    state_path = tmp_path / "queue-state.json"

    queue.append("raw_top_of_book", _tob(t0 + 5_000, 0.49, 0.51))
    first = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        ingest_ts=t0 + 120_000,
    )

    assert first.to_dict() == {
        "rows_read": 1,
        "rows_generated": 1,
        "rows_written": 1,
        "start_cursor": 0,
        "next_cursor": 1,
    }

    no_new_rows = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        ingest_ts=t0 + 121_000,
    )

    assert no_new_rows.rows_read == 0
    assert no_new_rows.rows_generated == 0
    assert no_new_rows.rows_written == 0

    queue.append("raw_trades", _trade(t0 + 50_000, 0.50, 3, "BUY"))
    late_update = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        ingest_ts=t0 + 122_000,
    )

    assert late_update.rows_read == 1
    assert late_update.rows_generated == 1
    assert late_update.rows_written == 1
    assert cursor_path.read_text(encoding="utf-8").strip() == "2"

    feature_rows = [
        row
        for file in warehouse_files(warehouse, "features_15m_v1")
        for row in pq.ParquetFile(file).read().to_pylist()
    ]
    latest = max(feature_rows, key=lambda row: int(row["ingest_ts"]))
    assert latest["trade_count_1m"] == 1
    assert latest["trade_volume_1m"] == pytest.approx(3.0)


def test_low_latency_queue_batch_resumes_from_persisted_file_offset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    warehouse = tmp_path / "warehouse"
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")
    cursor_path = tmp_path / "queue.cursor"
    state_path = tmp_path / "queue-state.json"

    queue.append("raw_top_of_book", _tob(t0 + 5_000, 0.49, 0.51))
    first = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        max_records=1,
        ingest_ts=t0 + 60_000,
    )
    assert first.next_cursor == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["raw_queue_progress"]["line_cursor"] == 1
    assert state["raw_queue_progress"]["byte_offset"] > 0

    queue.append("raw_trades", _trade(t0 + 50_000, 0.50, 3, "BUY"))

    def fail_line_scan(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("line-cursor scan should not be used after offset progress exists")

    monkeypatch.setattr(JsonlRawQueue, "read_from_line_cursor", fail_line_scan)
    second = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        max_records=1,
        ingest_ts=t0 + 61_000,
    )

    assert second.rows_read == 1
    assert second.start_cursor == 1
    assert second.next_cursor == 2


def test_low_latency_queue_batch_coalesces_same_symbol_updates_in_one_batch(
    tmp_path: Path,
) -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    warehouse = tmp_path / "warehouse"
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")

    queue.append("raw_top_of_book", _tob(t0 + 5_000, 0.49, 0.51))
    queue.append("raw_top_of_book", _tob(t0 + 10_000, 0.52, 0.54))

    report = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        ingest_ts=t0 + 60_000,
    )

    assert report.rows_read == 2
    assert report.rows_generated == 1
    assert report.rows_written == 1
    feature_rows = [
        row
        for file in warehouse_files(warehouse, "features_15m_v1")
        for row in pq.ParquetFile(file).read().to_pylist()
    ]
    assert feature_rows[0]["feature_ts"] == t0 + 60_000
    assert feature_rows[0]["market_implied_prob"] == pytest.approx(0.54)


def test_low_latency_queue_batch_withholds_future_feature_until_boundary(
    tmp_path: Path,
) -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    warehouse = tmp_path / "warehouse"
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")
    cursor_path = tmp_path / "queue.cursor"
    state_path = tmp_path / "queue-state.json"

    queue.append("raw_top_of_book", _tob(t0 + 5_000, 0.49, 0.51))
    early = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        ingest_ts=t0 + 30_000,
    )

    assert early.rows_read == 1
    assert early.rows_generated == 0
    assert early.rows_written == 0

    mature = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        ingest_ts=t0 + 60_000,
    )

    assert mature.rows_read == 0
    assert mature.rows_generated == 1
    assert mature.rows_written == 1
    feature_rows = [
        row
        for file in warehouse_files(warehouse, "features_15m_v1")
        for row in pq.ParquetFile(file).read().to_pylist()
    ]
    assert feature_rows[0]["feature_ts"] == t0 + 60_000
    assert feature_rows[0]["ingest_ts"] == t0 + 60_000


def test_low_latency_queue_batch_skips_post_expiry_boundary_feature(
    tmp_path: Path,
) -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    round_end = t0 + 15 * 60_000
    warehouse = tmp_path / "warehouse"
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")
    cursor_path = tmp_path / "queue.cursor"
    state_path = tmp_path / "queue-state.json"

    queue.append("raw_top_of_book", _tob(round_end - 1_000, 0.49, 0.51))
    report = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        ingest_ts=round_end,
    )

    assert report.rows_read == 1
    assert report.rows_generated == 0
    assert report.rows_written == 0


def test_low_latency_queue_batch_skips_degenerate_market_price(
    tmp_path: Path,
) -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    warehouse = tmp_path / "warehouse"
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")

    queue.append("raw_top_of_book", _tob(t0 + 5_000, 0.0, 1.0))
    report = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        ingest_ts=t0 + 60_000,
    )

    assert report.rows_read == 1
    assert report.rows_generated == 0
    assert report.rows_written == 0


def test_low_latency_queue_batch_prunes_expired_round_state(
    tmp_path: Path,
) -> None:
    t0 = _ts_at(2026, 5, 28, 1, 0)
    round_end = t0 + 15 * 60_000
    warehouse = tmp_path / "warehouse"
    queue = JsonlRawQueue(tmp_path / "raw-queue.jsonl")
    cursor_path = tmp_path / "queue.cursor"
    state_path = tmp_path / "queue-state.json"

    queue.append("raw_top_of_book", _tob(t0 + 5_000, 0.49, 0.51))
    initial = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        ingest_ts=t0 + 60_000,
    )
    assert initial.rows_written == 1

    pruned = run_low_latency_feature_queue_batch(
        warehouse,
        queue.path,
        cursor_path=cursor_path,
        state_path=state_path,
        ingest_ts=round_end,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert pruned.rows_read == 0
    assert pruned.rows_written == 0
    assert state["top_of_book_rows"] == []
    assert state["orderbook_rows"] == []
    assert state["trade_rows"] == []
    assert state["latest_feature_rows"] == []
