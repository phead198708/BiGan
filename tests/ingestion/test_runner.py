from __future__ import annotations

import asyncio
from typing import Any

from bigan.canonical.schemas import PROVENANCE_REST_SEED, PROVENANCE_WS
from bigan.ingestion.clob_rest import RestOrderbook
from bigan.ingestion.config import IngestionSettings
from bigan.ingestion.gamma_client import ActiveMarket
from bigan.ingestion.message_types import BookEvent
from bigan.ingestion.runner import IngestionRunner, _raw_record_from_ws_event


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
