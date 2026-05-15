"""Tests for issue #25 soak evidence summarisation."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import orjson

from bigan.ingestion.soak import (
    SoakThresholds,
    append_soak_sample,
    finalize_soak_rollup,
    inspect_raw_archive,
    inspect_rollup_outputs,
    read_soak_samples,
    summarize_soak,
)


def _sample(
    ts: float,
    *,
    reconnects: float = 0,
    last_event_ts: float | None = None,
    hash_mismatches: float = 0,
    records: float = 0,
    rollups_ok: float = 0,
    rss_mb: float = 100,
) -> dict[str, Any]:
    return {
        "sample_ts": ts,
        "uptime_seconds": ts,
        "metrics": {
            "ws_reconnects_total": reconnects,
            "last_event_receive_time_seconds": ts - 1
            if last_event_ts is None
            else last_event_ts,
            "ws_hash_mismatch_total": hash_mismatches,
            "sink_records_written_total": records,
            "ws_subscribed_markets": 2,
            "rollup_files_total_ok": rollups_ok,
            "rollup_files_total_error": 0,
        },
        "process": {"max_rss_mb": rss_mb},
    }


def _write_raw_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wb") as fp:
        for row in rows:
            fp.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def test_summarize_soak_passes_when_metrics_and_artifacts_are_healthy(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    rollup_dir = tmp_path / "rollup"
    _write_raw_file(
        raw_dir / "2026-05-13.ndjson.gz",
        [
            {"receive_time": 1778650000000, "raw": {"event_type": "book"}},
            {"receive_time": 1778650001000, "raw": {"event_type": "price_change"}},
        ],
    )
    parquet_path = rollup_dir / "date=2026-05-13" / "event_type=book" / "part.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"placeholder")

    summary = summarize_soak(
        [
            _sample(1_000, records=0, rollups_ok=0, rss_mb=100),
            _sample(1_120, records=2, rollups_ok=1, rss_mb=110),
        ],
        raw_dir=raw_dir,
        rollup_dir=rollup_dir,
        thresholds=SoakThresholds(min_duration_seconds=60),
    )

    assert summary["passed"] is True
    assert summary["raw"]["records"] == 2
    assert summary["rollup"]["parquet_files"] == 1
    assert {check["name"] for check in summary["checks"] if not check["passed"]} == set()


def test_summarize_soak_includes_market_coverage_check(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    rollup_dir = tmp_path / "rollup"
    _write_raw_file(raw_dir / "2026-05-13.ndjson.gz", [{"receive_time": 1, "raw": {}}])
    parquet_path = rollup_dir / "date=2026-05-13" / "event_type=book" / "part.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"placeholder")
    market_coverage = {
        "passed": True,
        "gamma": {"markets": 1, "assets": 2},
        "raw": {"records": 1, "assets_with_any_event": 2, "assets_with_book": 2},
        "rest": {"books_ok": 2, "books_missing": 0},
        "coverage": {"missing_any_assets": 0, "missing_book_assets": 0},
        "hash": {"required": False, "compared": 2, "matches": 2, "mismatches": 0},
    }

    summary = summarize_soak(
        [
            _sample(1_000, records=0, rollups_ok=0, rss_mb=100),
            _sample(1_120, records=1, rollups_ok=1, rss_mb=110),
        ],
        raw_dir=raw_dir,
        rollup_dir=rollup_dir,
        thresholds=SoakThresholds(min_duration_seconds=60),
        market_coverage=market_coverage,
    )

    coverage_check = next(
        check for check in summary["checks"] if check["name"] == "market_coverage"
    )
    assert summary["passed"] is True
    assert coverage_check["passed"] is True
    assert summary["market_coverage"] == market_coverage


def test_summarize_soak_fails_failed_market_coverage(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    rollup_dir = tmp_path / "rollup"
    _write_raw_file(raw_dir / "2026-05-13.ndjson.gz", [{"receive_time": 1, "raw": {}}])
    parquet_path = rollup_dir / "date=2026-05-13" / "event_type=book" / "part.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"placeholder")

    summary = summarize_soak(
        [
            _sample(1_000, records=0, rollups_ok=0, rss_mb=100),
            _sample(1_120, records=1, rollups_ok=1, rss_mb=110),
        ],
        raw_dir=raw_dir,
        rollup_dir=rollup_dir,
        thresholds=SoakThresholds(min_duration_seconds=60),
        market_coverage={"passed": False, "error": "Gamma timeout"},
    )

    failed = {check["name"] for check in summary["checks"] if not check["passed"]}
    assert summary["passed"] is False
    assert "market_coverage" in failed


def test_summarize_soak_fails_stale_liveness_metric(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    rollup_dir = tmp_path / "rollup"
    _write_raw_file(raw_dir / "2026-05-13.ndjson.gz", [{"receive_time": 1, "raw": {}}])
    parquet_path = rollup_dir / "date=2026-05-13" / "event_type=book" / "part.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"placeholder")

    summary = summarize_soak(
        [
            _sample(1_000, last_event_ts=995, records=0, rollups_ok=0),
            _sample(1_100, last_event_ts=1_000, records=1, rollups_ok=1),
        ],
        raw_dir=raw_dir,
        rollup_dir=rollup_dir,
        thresholds=SoakThresholds(
            min_duration_seconds=60,
            max_last_event_lag_seconds=60,
        ),
    )

    failed = {check["name"] for check in summary["checks"] if not check["passed"]}
    assert summary["passed"] is False
    assert "last_event_receive_lag_seconds" in failed


def test_summarize_soak_ignores_startup_samples_before_first_event(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    rollup_dir = tmp_path / "rollup"
    _write_raw_file(raw_dir / "2026-05-13.ndjson.gz", [{"receive_time": 1, "raw": {}}])
    parquet_path = rollup_dir / "date=2026-05-13" / "event_type=book" / "part.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"placeholder")

    summary = summarize_soak(
        [
            _sample(1_000, last_event_ts=0, records=0, rollups_ok=0),
            _sample(1_100, last_event_ts=1_099, records=1, rollups_ok=1),
        ],
        raw_dir=raw_dir,
        rollup_dir=rollup_dir,
        thresholds=SoakThresholds(min_duration_seconds=60),
    )

    assert summary["passed"] is True
    assert summary["metrics"]["max_last_event_lag_seconds"] == 1


def test_raw_archive_counts_bad_lines_and_done_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_file(raw_dir / "2026-05-13.ndjson.gz", [{"ok": True}])
    done_file = raw_dir / "_done" / "2026-05-12.ndjson.gz"
    done_file.parent.mkdir(parents=True)
    with gzip.open(done_file, mode="wb") as fp:
        fp.write(b"{\"ok\": true}\n")
        fp.write(b"{bad json}\n")

    stats = inspect_raw_archive(raw_dir)

    assert stats == {
        "files": 2,
        "records": 2,
        "bad_lines": 1,
        "bad_files": 0,
        "incomplete_files": 0,
    }


def test_raw_archive_counts_unclosed_gzip_as_incomplete(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    active = raw_dir / "active.ndjson.gz"
    fp = gzip.open(active, mode="wb")  # noqa: SIM115 - keep stream open.
    try:
        fp.write(orjson.dumps({"ok": True}, option=orjson.OPT_APPEND_NEWLINE))
        fp.flush()

        stats = inspect_raw_archive(raw_dir)
    finally:
        fp.close()

    assert stats == {
        "files": 1,
        "records": 1,
        "bad_lines": 0,
        "bad_files": 0,
        "incomplete_files": 1,
    }


def test_soak_sample_ndjson_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "soak.ndjson"
    append_soak_sample(path, _sample(1_000, records=1))
    append_soak_sample(path, _sample(1_001, records=2))

    samples = read_soak_samples(path)

    assert [sample["metrics"]["sink_records_written_total"] for sample in samples] == [
        1,
        2,
    ]


def test_finalize_soak_rollup_converts_remaining_raw_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    rollup_dir = tmp_path / "rollup"
    _write_raw_file(
        raw_dir / "2026-05-13.ndjson.gz",
        [{"receive_time": 1778650000000, "raw": {"event_type": "book"}}],
    )

    result = finalize_soak_rollup(raw_dir, rollup_dir)

    assert result == {"files": 1, "records": 1, "errors": []}
    assert inspect_rollup_outputs(rollup_dir)["parquet_files"] == 1
    assert not list(raw_dir.glob("*.ndjson.gz"))
    assert list((raw_dir / "_done").glob("*.ndjson.gz"))
