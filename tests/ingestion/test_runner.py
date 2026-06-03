from __future__ import annotations

import asyncio
import logging
from typing import Any

from bigan.canonical.schemas import PROVENANCE_REST_SEED, PROVENANCE_WS
from bigan.features.low_latency import JsonlRawQueue
from bigan.ingestion.clob_rest import RestOrderbook
from bigan.ingestion.config import IngestionSettings
from bigan.ingestion.gamma_client import ActiveMarket, MarketDiscoverySpec
from bigan.ingestion.message_types import BookEvent
from bigan.ingestion.runner import (
    IngestionRunner,
    _gamma_poll_timeout_seconds,
    _log_gamma_poll_failure,
    _raw_record_from_ws_event,
)


class _RecordingSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


def test_ws_raw_record_carries_ordering_metadata() -> None:
    event = BookEvent(
        event_type="book",
        asset_id="tok-1",
        market="0xmkt",
        timestamp=1_700_000_030_000,
        receive_time=1_700_000_030_111,
        bids=[],
        asks=[],
        hash="h0",
    )
    raw = {
        "event_type": "book",
        "asset_id": "tok-1",
        "market": "0xmkt",
        "timestamp": "1700000030000",
        "bids": [],
        "asks": [],
        "hash": "h0",
    }

    record = _raw_record_from_ws_event(event, raw)

    assert record["receive_time"] == 1_700_000_030_111
    assert record["source_timestamp_ms"] == 1_700_000_030_000
    assert record["capture_timestamp_ms"] == 1_700_000_030_111
    assert record["source_channel"] == "clob-ws"
    assert record["provenance"] == PROVENANCE_WS
    assert record["raw"] is raw


def test_initial_snapshot_candidates_skip_seeded_and_inflight(tmp_path) -> None:
    runner = IngestionRunner(
        IngestionSettings(data_dir=tmp_path, metrics_enabled=False, rollup_enabled=False)
    )
    runner._snapshot_seeded_assets.add("seeded")
    runner._snapshot_seed_inflight.add("inflight")

    candidates = runner._initial_snapshot_candidates(
        {"seeded", "inflight", "fresh"}
    )

    assert candidates == {"fresh"}
    assert runner._active_asset_ids == {"seeded", "inflight", "fresh"}


def test_gamma_poll_retryable_failure_logs_warning_without_traceback(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _log_gamma_poll_failure(TimeoutError("gamma timed out"))

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert record.message == "gamma.poll_retryable_failed"
    assert record.err_type == "TimeoutError"
    assert record.exc_info is None


def test_gamma_poll_timeout_wakes_on_market_boundary() -> None:
    specs = [
        MarketDiscoverySpec(
            slug_prefix="btc-updown-15m-",
            underlying="BTC",
            horizon_ms=15 * 60_000,
            symbol_kind="btc_15m_outcome",
        )
    ]

    timeout = _gamma_poll_timeout_seconds(
        specs,
        interval_seconds=60.0,
        now_ms=1_779_461_990_000,
    )

    assert timeout == 10.25


def test_runner_persists_and_applies_gamma_outcome_metadata(tmp_path) -> None:
    runner = IngestionRunner(
        IngestionSettings(data_dir=tmp_path, metrics_enabled=False, rollup_enabled=False)
    )
    sink = _RecordingSink()
    runner._sink = sink  # type: ignore[assignment]
    market = ActiveMarket(
        slug="btc-updown-15m-1778423700",
        condition_id="0xabc",
        asset_id_up="tok-up",
        asset_id_down="tok-down",
        start_ts_ms=1_778_423_400_000,
        end_ts_ms=1_778_424_300_000,
        tick_size="0.01",
    )

    rows = runner._refresh_market_metadata([market])
    asyncio.run(runner._persist_symbol_mapping_rows(rows))
    asyncio.run(runner._persist_symbol_mapping_rows(rows))

    event = BookEvent(
        event_type="book",
        asset_id="tok-up",
        market="0xabc",
        timestamp=1_778_423_410_000,
        receive_time=1_778_423_410_111,
        bids=[],
        asks=[],
        hash="h0",
    )
    record = runner._annotate_raw_record(
        _raw_record_from_ws_event(
            event,
            {
                "event_type": "book",
                "asset_id": "tok-up",
                "market": "0xabc",
                "timestamp": "1778423410000",
                "bids": [],
                "asks": [],
                "hash": "h0",
            },
        )
    )

    assert len(sink.records) == 1
    assert sink.records[0]["raw"]["event_type"] == "symbol_mapping"
    assert len(sink.records[0]["raw"]["mappings"]) == 2
    assert record["raw"]["canonical_symbol"] == "BTC-15M:btc-updown-15m-1778423700:UP"
    assert record["raw"]["outcome_side"] == "UP"


def test_runner_publishes_btc15_canonical_rows_to_low_latency_raw_queue(tmp_path) -> None:
    queue_path = tmp_path / "live" / "btc15-raw-queue.jsonl"
    runner = IngestionRunner(
        IngestionSettings(
            data_dir=tmp_path,
            metrics_enabled=False,
            rollup_enabled=False,
            low_latency_raw_queue_path=queue_path,
        )
    )
    sink = _RecordingSink()
    runner._sink = sink  # type: ignore[assignment]
    runner._refresh_market_metadata(
        [
            ActiveMarket(
                slug="btc-updown-15m-1778423700",
                condition_id="0xabc",
                asset_id_up="tok-up",
                asset_id_down="tok-down",
                start_ts_ms=1_778_423_700_000,
                end_ts_ms=1_778_424_600_000,
                tick_size="0.01",
            )
        ]
    )
    event = BookEvent(
        event_type="book",
        asset_id="tok-up",
        market="0xabc",
        timestamp=1_778_423_710_000,
        receive_time=1_778_423_710_111,
        bids=[],
        asks=[],
        hash="h0",
    )

    asyncio.run(
        runner.make_handler()(
            event,
                {
                    "event_type": "book",
                    "asset_id": "tok-up",
                    "market": "0xabc",
                    "timestamp": "1778423710000",
                "bids": [{"price": "0.49", "size": "100"}],
                "asks": [{"price": "0.51", "size": "50"}],
                "hash": "h0",
            },
        )
    )

    queued, cursor = JsonlRawQueue(queue_path).read_from(0)

    assert len(sink.records) == 1
    assert cursor == 3
    assert [item.table for item in queued] == [
        "raw_orderbook_snapshot",
        "raw_orderbook_snapshot",
        "raw_top_of_book",
    ]
    assert {item.row["canonical_symbol"] for item in queued} == {
        "BTC-15M:btc-updown-15m-1778423700:UP"
    }
    assert queued[-1].row["bid_price"] == 0.49
    assert queued[-1].row["ask_price"] == 0.51


def test_runner_limits_low_latency_orderbook_queue_to_feature_depth_levels(
    tmp_path,
) -> None:
    queue_path = tmp_path / "live" / "btc15-raw-queue.jsonl"
    runner = IngestionRunner(
        IngestionSettings(
            data_dir=tmp_path,
            metrics_enabled=False,
            rollup_enabled=False,
            low_latency_raw_queue_path=queue_path,
        )
    )
    sink = _RecordingSink()
    runner._sink = sink  # type: ignore[assignment]
    runner._refresh_market_metadata(
        [
            ActiveMarket(
                slug="btc-updown-15m-1778423700",
                condition_id="0xabc",
                asset_id_up="tok-up",
                asset_id_down="tok-down",
                start_ts_ms=1_778_423_700_000,
                end_ts_ms=1_778_424_600_000,
                tick_size="0.01",
            )
        ]
    )
    bids = [
        {"price": f"{0.49 - level * 0.001:.3f}", "size": "1"}
        for level in range(12)
    ]
    asks = [
        {"price": f"{0.51 + level * 0.001:.3f}", "size": "1"}
        for level in range(12)
    ]

    asyncio.run(
        runner.make_handler()(
            BookEvent(
                event_type="book",
                asset_id="tok-up",
                market="0xabc",
                timestamp=1_778_423_710_000,
                receive_time=1_778_423_710_111,
                bids=[],
                asks=[],
                hash="h0",
            ),
            {
                "event_type": "book",
                "asset_id": "tok-up",
                "market": "0xabc",
                "timestamp": "1778423710000",
                "bids": bids,
                "asks": asks,
                "hash": "h0",
            },
        )
    )

    queued, cursor = JsonlRawQueue(queue_path).read_from(0)
    depth_rows = [item for item in queued if item.table == "raw_orderbook_snapshot"]

    assert cursor == 21
    assert len(depth_rows) == 20
    assert max(int(item.row["level"]) for item in depth_rows) == 9
    assert [item.table for item in queued][-1] == "raw_top_of_book"


def test_runner_excludes_future_and_expired_round_rows_from_low_latency_raw_queue(
    tmp_path,
) -> None:
    queue_path = tmp_path / "live" / "btc15-raw-queue.jsonl"
    runner = IngestionRunner(
        IngestionSettings(
            data_dir=tmp_path,
            metrics_enabled=False,
            rollup_enabled=False,
            low_latency_raw_queue_path=queue_path,
        )
    )
    sink = _RecordingSink()
    runner._sink = sink  # type: ignore[assignment]
    start_ts = 1_778_423_400_000
    end_ts = 1_778_424_300_000
    runner._refresh_market_metadata(
        [
            ActiveMarket(
                slug="btc-updown-15m-1778423400",
                condition_id="0xabc",
                asset_id_up="tok-up",
                asset_id_down="tok-down",
                start_ts_ms=start_ts,
                end_ts_ms=end_ts,
                tick_size="0.01",
            )
        ]
    )

    for timestamp in (start_ts - 1_000, end_ts):
        event = BookEvent(
            event_type="book",
            asset_id="tok-up",
            market="0xabc",
            timestamp=timestamp,
            receive_time=timestamp + 111,
            bids=[],
            asks=[],
            hash=f"h-{timestamp}",
        )
        asyncio.run(
            runner.make_handler()(
                event,
                {
                    "event_type": "book",
                    "asset_id": "tok-up",
                    "market": "0xabc",
                    "timestamp": str(timestamp),
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "50"}],
                    "hash": f"h-{timestamp}",
                },
            )
        )

    queued, cursor = JsonlRawQueue(queue_path).read_from(0)

    assert len(sink.records) == 2
    assert queued == []
    assert cursor == 0


def test_initial_snapshot_seed_writes_rest_book(monkeypatch, tmp_path) -> None:
    runner = IngestionRunner(
        IngestionSettings(data_dir=tmp_path, metrics_enabled=False, rollup_enabled=False)
    )
    sink = _RecordingSink()
    runner._sink = sink  # type: ignore[assignment]
    gate_calls = 0

    class _FakeRestClient:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def __aenter__(self) -> _FakeRestClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            pass

        async def fetch_orderbook(self, asset_id: str) -> RestOrderbook | None:
            assert asset_id == "tok-1"
            return RestOrderbook(
                asset_id=asset_id,
                market="0xmkt",
                timestamp_ms=1_700_000_030_000,
                hash="h0",
                bids=[(0.49, 100.0)],
                asks=[(0.51, 50.0)],
                raw={},
            )

    async def before_rest_call() -> None:
        nonlocal gate_calls
        gate_calls += 1

    monkeypatch.setattr("bigan.ingestion.runner.PolymarketRestClient", _FakeRestClient)

    outcome = asyncio.run(
        runner._seed_initial_orderbook_snapshot_once(
            "tok-1",
            before_rest_call=before_rest_call,
        )
    )

    assert outcome == "ok"
    assert gate_calls == 1
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record["source_timestamp_ms"] == 1_700_000_030_000
    assert isinstance(record["capture_timestamp_ms"], int)
    assert record["receive_time"] == record["capture_timestamp_ms"]
    assert record["source_channel"] == "clob-rest"
    assert record["raw"]["event_type"] == "book"
    assert record["raw"]["asset_id"] == "tok-1"
    assert record["raw"]["provenance"] == PROVENANCE_REST_SEED
