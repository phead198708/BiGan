"""Live Polymarket coverage checks against Gamma + CLOB REST.

This module answers a narrower question than the soak summary: did our WS raw
archive see the markets/tokens that Polymarket currently advertises, and do we
have at least one order-book snapshot for each token?

The REST hash comparison is intentionally reported separately. It is a strong
signal while ingestion is running, but a strict equality check can race with
fresh WS deltas or become stale after a completed short soak.
"""

from __future__ import annotations

import asyncio
import gzip
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from .clob_rest import PolymarketRestClient, RestOrderbook
from .gamma_client import ActiveMarket


@dataclass(frozen=True, slots=True)
class ExpectedAsset:
    asset_id: str
    condition_id: str
    slug: str
    side: str
    gamma_start_s: int
    round_start_s: int
    round_end_s: int


@dataclass(slots=True)
class RawAssetState:
    asset_id: str
    market: str | None = None
    seen_any: bool = False
    seen_book: bool = False
    latest_event_type: str | None = None
    latest_receive_time_ms: int | None = None
    latest_event_time_ms: int | None = None
    latest_hash: str | None = None


@dataclass(frozen=True, slots=True)
class RawScanResult:
    files: int
    records: int
    bad_lines: int
    bad_files: int
    incomplete_files: int
    assets: Mapping[str, RawAssetState]


async def compare_market_coverage(
    *,
    markets: Sequence[ActiveMarket],
    raw_dir: Path,
    rest: PolymarketRestClient,
    max_stale_seconds: float | None = 120.0,
    require_hash_match: bool = False,
    ignore_markets_opened_after_raw_end: bool = False,
    raw_end_grace_seconds: float = 120.0,
    max_concurrency: int = 12,
    max_examples: int = 20,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Compare current Gamma assets with local raw coverage and REST books."""

    raw = scan_raw_asset_states(raw_dir)
    raw_end_ms = max(
        (state.latest_receive_time_ms or 0 for state in raw.assets.values()),
        default=0,
    )
    filtered_markets = list(markets)
    ignored_opened_after_raw_end = 0
    ignored_scheduled_after_raw_end = 0
    if ignore_markets_opened_after_raw_end and raw_end_ms:
        grace_ms = int(raw_end_grace_seconds * 1000)
        keep: list[ActiveMarket] = []
        for market in filtered_markets:
            opened_after_raw_end = market.start_ts_ms > raw_end_ms + grace_ms
            scheduled_after_raw_end = (
                _round_start_s(market) * 1000 > raw_end_ms + grace_ms
            )
            if opened_after_raw_end or scheduled_after_raw_end:
                ignored_opened_after_raw_end += int(opened_after_raw_end)
                ignored_scheduled_after_raw_end += int(scheduled_after_raw_end)
                continue
            keep.append(market)
        filtered_markets = keep

    expected = _expected_assets(filtered_markets)
    rest_books = await _fetch_rest_books(
        rest,
        expected.keys(),
        max_concurrency=max_concurrency,
    )
    checked_at_ms = int(time.time() * 1000) if now_ms is None else now_ms

    expected_ids = set(expected)
    raw_any_ids = {
        asset_id
        for asset_id, state in raw.assets.items()
        if state.seen_any and asset_id in expected_ids
    }
    raw_book_ids = {
        asset_id
        for asset_id, state in raw.assets.items()
        if state.seen_book and asset_id in expected_ids
    }
    rest_ok_ids = {
        asset_id for asset_id, book in rest_books.items() if book is not None
    }

    missing_any = sorted(expected_ids - raw_any_ids, key=lambda aid: _asset_sort_key(expected[aid]))
    missing_book = sorted(expected_ids - raw_book_ids, key=lambda aid: _asset_sort_key(expected[aid]))
    rest_missing = sorted(expected_ids - rest_ok_ids, key=lambda aid: _asset_sort_key(expected[aid]))

    stale_assets: list[str] = []
    if max_stale_seconds is not None:
        stale_cutoff_ms = int(max_stale_seconds * 1000)
        for asset_id in sorted(expected_ids & raw_any_ids, key=lambda aid: _asset_sort_key(expected[aid])):
            latest = raw.assets[asset_id].latest_receive_time_ms
            if latest is None or checked_at_ms - latest > stale_cutoff_ms:
                stale_assets.append(asset_id)

    hash_compared = 0
    hash_matches = 0
    hash_mismatches: list[str] = []
    for asset_id in sorted(expected_ids & raw_book_ids & rest_ok_ids, key=lambda aid: _asset_sort_key(expected[aid])):
        local_hash = raw.assets[asset_id].latest_hash
        rest_hash = rest_books[asset_id].hash if rest_books[asset_id] is not None else None
        if not local_hash or not rest_hash:
            continue
        hash_compared += 1
        if local_hash == rest_hash:
            hash_matches += 1
        else:
            hash_mismatches.append(asset_id)

    passed = not missing_any and not missing_book and not rest_missing and not stale_assets
    if require_hash_match:
        passed = passed and not hash_mismatches

    return {
        "passed": passed,
        "checked_at_ms": checked_at_ms,
        "gamma": {
            "markets": len(filtered_markets),
            "assets": len(expected),
            "source_markets": len(markets),
            "ignored_markets_after_raw_end": len(markets) - len(filtered_markets),
            "ignored_markets_opened_after_raw_end": ignored_opened_after_raw_end,
            "ignored_markets_scheduled_after_raw_end": ignored_scheduled_after_raw_end,
        },
        "raw": {
            "dir": str(raw_dir),
            "files": raw.files,
            "records": raw.records,
            "bad_lines": raw.bad_lines,
            "bad_files": raw.bad_files,
            "incomplete_files": raw.incomplete_files,
            "assets_with_any_event": len(raw_any_ids),
            "assets_with_book": len(raw_book_ids),
        },
        "rest": {
            "books_ok": len(rest_ok_ids),
            "books_missing": len(rest_missing),
            "missing_examples": _asset_examples(rest_missing, expected, raw.assets, rest_books, max_examples),
        },
        "coverage": {
            "missing_any_assets": len(missing_any),
            "missing_any_examples": _asset_examples(missing_any, expected, raw.assets, rest_books, max_examples),
            "missing_book_assets": len(missing_book),
            "missing_book_examples": _asset_examples(missing_book, expected, raw.assets, rest_books, max_examples),
        },
        "freshness": {
            "max_stale_seconds": max_stale_seconds,
            "stale_assets": len(stale_assets),
            "stale_examples": _asset_examples(stale_assets, expected, raw.assets, rest_books, max_examples),
        },
        "hash": {
            "required": require_hash_match,
            "compared": hash_compared,
            "matches": hash_matches,
            "mismatches": len(hash_mismatches),
            "mismatch_examples": _asset_examples(hash_mismatches, expected, raw.assets, rest_books, max_examples),
        },
    }


def scan_raw_asset_states(raw_dir: Path) -> RawScanResult:
    """Scan active and rolled raw NDJSON.gz files into per-token state."""

    files = _raw_files(raw_dir)
    states: dict[str, RawAssetState] = {}
    records = 0
    bad_lines = 0
    bad_files = 0
    incomplete_files = 0

    for path in files:
        try:
            with gzip.open(path, mode="rb") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        bad_lines += 1
                        continue
                    records += 1
                    _apply_raw_record(states, record)
        except EOFError:
            incomplete_files += 1
        except OSError:
            bad_files += 1

    return RawScanResult(
        files=len(files),
        records=records,
        bad_lines=bad_lines,
        bad_files=bad_files,
        incomplete_files=incomplete_files,
        assets=states,
    )


def _apply_raw_record(states: dict[str, RawAssetState], record: Mapping[str, Any]) -> None:
    raw = record.get("raw")
    if not isinstance(raw, dict):
        return
    event_type = raw.get("event_type")
    receive_time_ms = _first_int(
        record.get("capture_timestamp_ms"),
        record.get("receive_time"),
        raw.get("capture_timestamp_ms"),
    )
    event_time_ms = _first_int(
        record.get("source_timestamp_ms"),
        raw.get("source_timestamp_ms"),
        raw.get("timestamp"),
    )
    market = _as_str(raw.get("market") or raw.get("condition_id"))

    if event_type == "book":
        asset_id = _as_str(raw.get("asset_id"))
        if asset_id is None:
            return
        state = states.setdefault(asset_id, RawAssetState(asset_id=asset_id))
        _touch_state(
            state,
            event_type=_as_str(event_type),
            market=market,
            receive_time_ms=receive_time_ms,
            event_time_ms=event_time_ms,
            latest_hash=_as_str(raw.get("hash")),
            seen_book=True,
        )
        return

    if event_type == "price_change":
        for change in raw.get("price_changes") or []:
            if not isinstance(change, dict):
                continue
            asset_id = _as_str(change.get("asset_id"))
            if asset_id is None:
                continue
            state = states.setdefault(asset_id, RawAssetState(asset_id=asset_id))
            _touch_state(
                state,
                event_type=_as_str(event_type),
                market=market,
                receive_time_ms=receive_time_ms,
                event_time_ms=event_time_ms,
                latest_hash=_as_str(change.get("hash")),
            )
        return

    if event_type in {"best_bid_ask", "last_trade_price", "tick_size_change"}:
        asset_id = _as_str(raw.get("asset_id"))
        if asset_id is None:
            return
        state = states.setdefault(asset_id, RawAssetState(asset_id=asset_id))
        _touch_state(
            state,
            event_type=_as_str(event_type),
            market=market,
            receive_time_ms=receive_time_ms,
            event_time_ms=event_time_ms,
        )


def _touch_state(
    state: RawAssetState,
    *,
    event_type: str | None,
    market: str | None,
    receive_time_ms: int | None,
    event_time_ms: int | None,
    latest_hash: str | None = None,
    seen_book: bool = False,
) -> None:
    state.seen_any = True
    state.seen_book = state.seen_book or seen_book
    if market is not None:
        state.market = market
    if receive_time_ms is not None and (
        state.latest_receive_time_ms is None
        or receive_time_ms >= state.latest_receive_time_ms
    ):
        state.latest_receive_time_ms = receive_time_ms
        state.latest_event_time_ms = event_time_ms
        state.latest_event_type = event_type
        if latest_hash is not None:
            state.latest_hash = latest_hash


async def _fetch_rest_books(
    rest: PolymarketRestClient,
    asset_ids: Sequence[str],
    *,
    max_concurrency: int,
) -> dict[str, RestOrderbook | None]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    out: dict[str, RestOrderbook | None] = {}

    async def fetch_one(asset_id: str) -> None:
        async with semaphore:
            out[asset_id] = await rest.fetch_orderbook(asset_id)

    await asyncio.gather(*(fetch_one(asset_id) for asset_id in asset_ids))
    return out


def _expected_assets(markets: Sequence[ActiveMarket]) -> dict[str, ExpectedAsset]:
    out: dict[str, ExpectedAsset] = {}
    for market in markets:
        round_start_s = _round_start_s(market)
        out[market.asset_id_up] = ExpectedAsset(
            asset_id=market.asset_id_up,
            condition_id=market.condition_id,
            slug=market.slug,
            side="up",
            gamma_start_s=market.start_ts_ms // 1000,
            round_start_s=round_start_s,
            round_end_s=round_start_s + 900,
        )
        out[market.asset_id_down] = ExpectedAsset(
            asset_id=market.asset_id_down,
            condition_id=market.condition_id,
            slug=market.slug,
            side="down",
            gamma_start_s=market.start_ts_ms // 1000,
            round_start_s=round_start_s,
            round_end_s=round_start_s + 900,
        )
    return out


def _round_start_s(market: ActiveMarket) -> int:
    suffix = market.slug.rsplit("-", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return market.start_ts_ms // 1000


def _asset_examples(
    asset_ids: Sequence[str],
    expected: Mapping[str, ExpectedAsset],
    raw_states: Mapping[str, RawAssetState],
    rest_books: Mapping[str, RestOrderbook | None],
    max_examples: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for asset_id in asset_ids[:max_examples]:
        exp = expected[asset_id]
        raw = raw_states.get(asset_id)
        rest = rest_books.get(asset_id)
        examples.append(
            {
                "asset_id": asset_id,
                "slug": exp.slug,
                "condition_id": exp.condition_id,
                "side": exp.side,
                "gamma_start_s": exp.gamma_start_s,
                "round_start_s": exp.round_start_s,
                "round_end_s": exp.round_end_s,
                "raw_seen_any": bool(raw and raw.seen_any),
                "raw_seen_book": bool(raw and raw.seen_book),
                "raw_latest_event_type": raw.latest_event_type if raw else None,
                "raw_latest_receive_time_ms": raw.latest_receive_time_ms if raw else None,
                "raw_latest_hash": raw.latest_hash if raw else None,
                "rest_book_found": rest is not None,
                "rest_hash": rest.hash if rest is not None else None,
                "rest_timestamp_ms": rest.timestamp_ms if rest is not None else None,
            }
        )
    return examples


def _asset_sort_key(asset: ExpectedAsset) -> tuple[int, str, str]:
    return (asset.round_start_s, asset.slug, asset.side)


def _raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    files = list(raw_dir.glob("*.ndjson.gz"))
    done_dir = raw_dir / "_done"
    if done_dir.exists():
        files.extend(done_dir.glob("*.ndjson.gz"))
    return sorted(files)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
