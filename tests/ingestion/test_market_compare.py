from __future__ import annotations

import asyncio
import gzip
from pathlib import Path
from typing import Any

import orjson

from bigan.ingestion.clob_rest import RestOrderbook
from bigan.ingestion.gamma_client import ActiveMarket
from bigan.ingestion.market_compare import compare_market_coverage, scan_raw_asset_states


def test_scan_raw_asset_states_tracks_books_and_price_changes(tmp_path: Path) -> None:
    _write_raw(
        tmp_path,
        [
            _book_record("tok-up", "0xmkt", "h0", receive_time=10_000),
            _price_change_record("tok-up", "0xmkt", "h1", receive_time=20_000),
        ],
    )

    scan = scan_raw_asset_states(tmp_path)

    assert scan.records == 2
    assert scan.bad_lines == 0
    assert scan.assets["tok-up"].seen_book is True
    assert scan.assets["tok-up"].latest_event_type == "price_change"
    assert scan.assets["tok-up"].latest_hash == "h1"


def test_compare_market_coverage_passes_complete_assets(tmp_path: Path) -> None:
    market = _market()
    _write_raw(
        tmp_path,
        [
            _book_record("tok-up", "0xmkt", "h-up", receive_time=100_000),
            _book_record("tok-down", "0xmkt", "h-down", receive_time=100_001),
        ],
    )
    rest = _FakeRest(
        {
            "tok-up": _rest_book("tok-up", "0xmkt", "h-up"),
            "tok-down": _rest_book("tok-down", "0xmkt", "h-down"),
        }
    )

    report = asyncio.run(
        compare_market_coverage(
            markets=[market],
            raw_dir=tmp_path,
            rest=rest,  # type: ignore[arg-type]
            max_stale_seconds=120,
            require_hash_match=True,
            now_ms=101_000,
        )
    )

    assert report["passed"] is True
    assert report["coverage"]["missing_book_assets"] == 0
    assert report["hash"]["matches"] == 2


def test_compare_market_coverage_reports_missing_book(tmp_path: Path) -> None:
    market = _market()
    _write_raw(tmp_path, [_book_record("tok-up", "0xmkt", "h-up", receive_time=100_000)])
    rest = _FakeRest(
        {
            "tok-up": _rest_book("tok-up", "0xmkt", "h-up"),
            "tok-down": _rest_book("tok-down", "0xmkt", "h-down"),
        }
    )

    report = asyncio.run(
        compare_market_coverage(
            markets=[market],
            raw_dir=tmp_path,
            rest=rest,  # type: ignore[arg-type]
            max_stale_seconds=None,
        )
    )

    assert report["passed"] is False
    assert report["coverage"]["missing_any_assets"] == 1
    assert report["coverage"]["missing_book_examples"][0]["asset_id"] == "tok-down"


def test_compare_market_coverage_hash_mismatch_is_optional(tmp_path: Path) -> None:
    market = _market()
    _write_raw(
        tmp_path,
        [
            _book_record("tok-up", "0xmkt", "h-up", receive_time=100_000),
            _book_record("tok-down", "0xmkt", "old-hash", receive_time=100_001),
        ],
    )
    rest = _FakeRest(
        {
            "tok-up": _rest_book("tok-up", "0xmkt", "h-up"),
            "tok-down": _rest_book("tok-down", "0xmkt", "new-hash"),
        }
    )

    soft_report = asyncio.run(
        compare_market_coverage(
            markets=[market],
            raw_dir=tmp_path,
            rest=rest,  # type: ignore[arg-type]
            max_stale_seconds=None,
            require_hash_match=False,
        )
    )
    strict_report = asyncio.run(
        compare_market_coverage(
            markets=[market],
            raw_dir=tmp_path,
            rest=rest,  # type: ignore[arg-type]
            max_stale_seconds=None,
            require_hash_match=True,
        )
    )

    assert soft_report["passed"] is True
    assert soft_report["hash"]["mismatches"] == 1
    assert strict_report["passed"] is False


def test_compare_market_coverage_can_ignore_markets_opened_after_raw_end(
    tmp_path: Path,
) -> None:
    current_market = ActiveMarket(
        slug="btc-updown-15m-4102444800",
        condition_id="0xmkt",
        asset_id_up="tok-up",
        asset_id_down="tok-down",
        start_ts_ms=50_000,
        end_ts_ms=4_102_445_700_000,
        tick_size="0.01",
    )
    future_market = ActiveMarket(
        slug="btc-updown-15m-4102445700",
        condition_id="0xfuture",
        asset_id_up="future-up",
        asset_id_down="future-down",
        start_ts_ms=500_000,
        end_ts_ms=4_102_446_600_000,
        tick_size="0.01",
    )
    _write_raw(
        tmp_path,
        [
            _book_record("tok-up", "0xmkt", "h-up", receive_time=100_000),
            _book_record("tok-down", "0xmkt", "h-down", receive_time=100_001),
        ],
    )
    rest = _FakeRest(
        {
            "tok-up": _rest_book("tok-up", "0xmkt", "h-up"),
            "tok-down": _rest_book("tok-down", "0xmkt", "h-down"),
            "future-up": _rest_book("future-up", "0xfuture", "h-fu"),
            "future-down": _rest_book("future-down", "0xfuture", "h-fd"),
        }
    )

    report = asyncio.run(
        compare_market_coverage(
            markets=[current_market, future_market],
            raw_dir=tmp_path,
            rest=rest,  # type: ignore[arg-type]
            max_stale_seconds=None,
            ignore_markets_opened_after_raw_end=True,
            raw_end_grace_seconds=1,
        )
    )

    assert report["passed"] is True
    assert report["gamma"]["source_markets"] == 2
    assert report["gamma"]["ignored_markets_opened_after_raw_end"] == 1


class _FakeRest:
    def __init__(self, books: dict[str, RestOrderbook | None]) -> None:
        self._books = books

    async def fetch_orderbook(self, asset_id: str) -> RestOrderbook | None:
        return self._books.get(asset_id)


def _market() -> ActiveMarket:
    return ActiveMarket(
        slug="btc-updown-15m-4102444800",
        condition_id="0xmkt",
        asset_id_up="tok-up",
        asset_id_down="tok-down",
        start_ts_ms=4_102_444_800_000,
        end_ts_ms=4_102_445_700_000,
        tick_size="0.01",
    )


def _rest_book(asset_id: str, market: str, book_hash: str) -> RestOrderbook:
    return RestOrderbook(
        asset_id=asset_id,
        market=market,
        timestamp_ms=100_000,
        hash=book_hash,
        bids=[],
        asks=[],
        raw={},
    )


def _book_record(
    asset_id: str,
    market: str,
    book_hash: str,
    *,
    receive_time: int,
) -> dict[str, Any]:
    return {
        "receive_time": receive_time,
        "raw": {
            "event_type": "book",
            "asset_id": asset_id,
            "market": market,
            "timestamp": str(receive_time - 10),
            "hash": book_hash,
            "bids": [],
            "asks": [],
        },
    }


def _price_change_record(
    asset_id: str,
    market: str,
    book_hash: str,
    *,
    receive_time: int,
) -> dict[str, Any]:
    return {
        "receive_time": receive_time,
        "raw": {
            "event_type": "price_change",
            "market": market,
            "timestamp": str(receive_time - 10),
            "price_changes": [
                {
                    "asset_id": asset_id,
                    "price": "0.5",
                    "size": "10",
                    "side": "BUY",
                    "hash": book_hash,
                }
            ],
        },
    }


def _write_raw(path: Path, records: list[dict[str, Any]]) -> None:
    with gzip.open(path / "2026-01-01.ndjson.gz", "wb") as fp:
        for record in records:
            fp.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
