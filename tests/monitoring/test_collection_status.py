"""Tests for live collection status snapshots."""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from bigan.canonical.writer import WarehouseWriter
from bigan.mlops import connect_mlops_db, initialize_mlops_db
from bigan.monitoring import collection_status as collection_status_module
from bigan.monitoring import record_label_rows_as_outcomes, record_prediction_rows_as_events
from bigan.monitoring.collection_status import (
    build_live_collection_status,
    live_collection_readiness_decision,
    read_live_collection_status,
    write_live_collection_status,
)


def test_build_live_collection_status_counts_segments_manifest_rows_and_families(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    _write_gzip(raw_dir / "2026-05-23T120100Z.ndjson.gz", b"{}\n")
    (raw_dir / ".2026-05-23T120200Z.ndjson.gz.123.tmp").write_bytes(b"")
    manifest = tmp_path / "processed.txt"
    manifest.write_text("a\n\nb\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "scorer.log").write_text("ok\nstep failed status=137\n")

    with WarehouseWriter(live_root / "warehouse", max_rows_per_partition=10) as writer:
        writer.append_rows(
            "features_15m_v1",
            [
                {
                    "ts": 1_000,
                    "message_ts": 1_000,
                    "feature_ts": 1_000,
                    "ingest_ts": 1_100,
                    "source": "polymarket",
                    "source_symbol": "tok-1",
                    "source_market": "mkt-1",
                    "canonical_symbol": "BTC-15M:round:UP",
                    "symbol": "BTC-15M:round:UP",
                    "feature_version": "bigan-mvp-v1.0.0",
                    "completeness_score": 1.0,
                    "data_gap_flag": False,
                    "quality_filter_pass": True,
                    "quote_age_ms": 0,
                    "depth_age_ms": 0,
                    "trade_age_ms": 0,
                    "market_implied_prob": 0.52,
                },
                {
                    "ts": 2_000,
                    "message_ts": 2_000,
                    "feature_ts": 2_000,
                    "ingest_ts": 2_100,
                    "source": "polymarket",
                    "source_symbol": "tok-2",
                    "source_market": "mkt-2",
                    "canonical_symbol": "ETH-5M:round:DOWN",
                    "symbol": "ETH-5M:round:DOWN",
                    "feature_version": "bigan-mvp-v1.0.0",
                    "completeness_score": 1.0,
                    "data_gap_flag": False,
                    "quality_filter_pass": True,
                    "quote_age_ms": 0,
                    "depth_age_ms": 0,
                    "trade_age_ms": 0,
                    "market_implied_prob": 0.48,
                },
            ],
        )

    status = build_live_collection_status(
        live_root=live_root,
        manifest_path=manifest,
        log_dir=log_dir,
        screen_session="collector",
        screen_state="running",
        generated_at="2026-05-23T12:00:00Z",
    )

    assert status["generated_at"] == "2026-05-23T12:00:00Z"
    assert status["status"] == "running_at_evidence_capture"
    assert status["raw_segment_count"] == 2
    assert status["active_tmp_segment_count"] == 1
    assert status["liveness_evidence"]["max_progress_staleness_seconds"] is None
    assert status["liveness_evidence"]["raw_segments_fresh"] is None
    assert status["liveness_evidence"]["latest_raw_segment"] == {
        "segment": "2026-05-23T120100Z.ndjson.gz",
        "segment_ts": "2026-05-23T12:01:00Z",
        "age_seconds": 0.0,
    }
    assert status["liveness_evidence"]["latest_processed_segment"] is None
    assert status["warehouse_freshness_evidence"]["max_progress_staleness_seconds"] is None
    assert status["warehouse_freshness_evidence"]["tables"]["features_15m_v1"]["fresh"] is None
    assert status["warehouse_freshness_evidence"]["tables"]["predictions"]["fresh"] is None
    assert status["raw_segment_integrity"] == {
        "check_limit": 20,
        "checked_count": 2,
        "unchecked_count": 0,
        "invalid_count": 0,
        "invalid_segments": [],
    }
    assert status["raw_segment_quarantine"] == {
        "quarantined_count": 0,
        "total_size_bytes": 0,
        "quarantined_segments": [],
        "latest_quarantined_segment": None,
    }
    assert status["disk_headroom_evidence"]["available"] is True
    assert status["disk_headroom_evidence"]["headroom_ok"] is True
    assert status["disk_headroom_evidence"]["free_bytes"] > 0
    feature_schema = status["warehouse_schema_evidence"]["features_15m_v1"]
    assert feature_schema["feature_file_count"] == 1
    assert feature_schema["sampled_file_count"] == 1
    assert feature_schema["missing_required_columns"] == []
    assert feature_schema["required_columns_present"] is True
    assert status["processed_manifest_rows"] == 2
    assert status["processed_manifest_integrity"] == {
        "line_count": 2,
        "unique_count": 2,
        "duplicate_count": 0,
        "malformed_count": 2,
        "duplicate_entries": [],
        "malformed_entries": [{"entry": "a"}, {"entry": "b"}],
    }
    assert status["family_counts"]["features_15m_v1"] == {"BTC-15M": 1, "ETH-5M": 1}
    assert status["totals"]["features_15m_v1_rows"] == 2
    readiness = status["collection_readiness"]
    assert readiness["target_days"] == 7.0
    assert readiness["features_15m_v1"]["missing_families"] == ["ETH-15M", "BTC-5M"]
    assert readiness["features_15m_v1"]["families"]["BTC-15M"]["span_ms"] == 0
    assert readiness["features_15m_v1"]["target_progress_pct"] == 0.0
    assert readiness["features_15m_v1"]["remaining_target_days"] == 7.0
    assert readiness["features_15m_v1"]["limiting_family"] == "BTC-15M"
    assert readiness["labels_15m_v1"]["missing_families"] == [
        "BTC-15M",
        "ETH-15M",
        "BTC-5M",
        "ETH-5M",
    ]
    assert readiness["ready_for_training"] is False
    assert status["health_evidence"]["error_match_count"] == 1
    assert status["health_evidence"]["unrecovered_error_match_count"] == 1

    decision = live_collection_readiness_decision(status)

    assert decision["ready"] is False
    assert "log health evidence contains fatal-pattern matches" in decision["blockers"]
    assert decision["feature_rows"] == 2
    assert decision["label_rows"] == 0


def test_readiness_blocks_when_feature_schema_is_missing_v4_columns(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    live_root = tmp_path / "live"
    feature_dir = live_root / "warehouse" / "features_15m_v1" / "dt=2026-05-23"
    feature_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "canonical_symbol": ["BTC-15M:round:UP"],
                "feature_ts": [1_000],
            }
        ),
        feature_dir / "part.parquet",
    )

    status = build_live_collection_status(
        live_root=live_root,
        generated_at="2026-05-23T12:00:00Z",
    )

    feature_schema = status["warehouse_schema_evidence"]["features_15m_v1"]
    assert feature_schema["sampled_file_count"] == 1
    assert feature_schema["required_columns_present"] is False
    assert "tick_price_velocity" in feature_schema["missing_required_columns"]

    decision = live_collection_readiness_decision(status)

    assert any(
        "feature warehouse schema missing required v4/tick columns" in blocker
        for blocker in decision["blockers"]
    )


def test_build_live_collection_status_counts_done_raw_segments(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    done_dir = raw_dir / "_done"
    done_dir.mkdir(parents=True)
    _write_gzip(done_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    _write_gzip(raw_dir / "2026-05-23T120100Z.ndjson.gz", b"{}\n")

    status = build_live_collection_status(
        live_root=live_root,
        generated_at="2026-05-23T12:01:00Z",
        gzip_check_limit=5,
    )

    assert status["raw_segments"] == [
        "_done/2026-05-23T120000Z.ndjson.gz",
        "2026-05-23T120100Z.ndjson.gz",
    ]
    assert status["raw_segment_count"] == 2
    assert status["raw_segment_integrity"]["checked_count"] == 2
    assert status["raw_segment_integrity"]["invalid_count"] == 0
    assert status["liveness_evidence"]["latest_raw_segment"] == {
        "segment": "2026-05-23T120100Z.ndjson.gz",
        "segment_ts": "2026-05-23T12:01:00Z",
        "age_seconds": 0.0,
    }


def test_live_collection_status_write_is_atomic_and_leaves_no_temp_file(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "status.json"
    status = {"generated_at": "2026-05-23T12:00:00Z", "raw_segment_count": 2}

    write_live_collection_status(status_path, status)

    assert read_live_collection_status(status_path) == status
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_read_live_collection_status_retries_transient_partial_json(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text("{", encoding="utf-8")
    status = {"generated_at": "2026-05-23T12:00:00Z", "raw_segment_count": 2}

    def finish_write() -> None:
        time.sleep(0.1)
        status_path.write_text(json.dumps(status), encoding="utf-8")

    writer = threading.Thread(target=finish_write)
    writer.start()
    try:
        assert read_live_collection_status(status_path) == status
    finally:
        writer.join()


def test_live_collection_status_blocks_malformed_or_duplicate_manifest_entries(
    tmp_path: Path,
) -> None:
    status = {
        "screen_session": "collector",
        "screen_state": "running",
        "raw_segment_integrity": {"invalid_count": 0},
        "processed_manifest_integrity": {
            "malformed_count": 1,
            "malformed_entries": [{"entry": "/tmp/not-a-segment.ndjson.gz"}],
            "duplicate_count": 1,
            "duplicate_entries": [{"entry": "/tmp/2026-05-23T120000Z.ndjson.gz"}],
        },
        "liveness_evidence": {
            "raw_segments_fresh": True,
            "processed_manifest_fresh": True,
        },
        "warehouse_freshness_evidence": {"tables": {}},
        "collection_readiness": {"ready_for_training": True},
    }

    decision = live_collection_readiness_decision(status)

    assert decision["ready"] is False
    assert (
        "processed manifest has malformed entries: /tmp/not-a-segment.ndjson.gz"
        in decision["blockers"]
    )
    assert (
        "processed manifest has duplicate entries: /tmp/2026-05-23T120000Z.ndjson.gz"
        in decision["blockers"]
    )


def test_build_live_collection_status_reports_live_root_lock_owner(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    lock_dir = live_root / ".run_champion_live.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")

    status = build_live_collection_status(live_root=live_root)

    lock = status["live_root_lock_evidence"]
    assert lock["lock_dir_exists"] is True
    assert lock["pid_file_exists"] is True
    assert lock["pid"] == os.getpid()
    assert lock["owner_running"] is True
    assert lock["pid_parse_error"] is None


def test_live_collection_readiness_blocks_stale_live_root_lock() -> None:
    status = {
        "screen_session": "collector",
        "screen_state": "running",
        "raw_segment_integrity": {"invalid_count": 0},
        "processed_manifest_integrity": {},
        "liveness_evidence": {},
        "warehouse_freshness_evidence": {"tables": {}},
        "collection_readiness": {"ready_for_training": True},
        "live_root_lock_evidence": {
            "lock_dir_exists": True,
            "pid_file_exists": True,
            "pid": 987654321,
            "owner_running": False,
        },
    }

    decision = live_collection_readiness_decision(status)

    assert decision["ready"] is False
    assert "live root lock owner is not running: pid=987654321" in decision["blockers"]


def test_live_collection_readiness_blocks_orphaned_live_root_lock_owner() -> None:
    status = {
        "screen_session": "collector",
        "screen_state": "not_found",
        "raw_segment_integrity": {"invalid_count": 0},
        "processed_manifest_integrity": {},
        "liveness_evidence": {},
        "warehouse_freshness_evidence": {"tables": {}},
        "collection_readiness": {"ready_for_training": True},
        "live_root_lock_evidence": {
            "lock_dir_exists": True,
            "pid_file_exists": True,
            "pid": os.getpid(),
            "owner_running": True,
        },
    }

    decision = live_collection_readiness_decision(status)

    assert decision["ready"] is False
    assert (
        "collector screen is not running but live root lock owner is active: "
        f"pid={os.getpid()}"
    ) in decision["blockers"]


def test_live_collection_status_treats_recovered_log_error_as_non_blocking(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "scorer.log").write_text(
        "\n".join(
            [
                "[2026-05-23T12:00:00Z] scan first started",
                "Traceback (most recent call last)",
                "[2026-05-23T12:00:01Z] predictions-v1 failed status=1",
                "[2026-05-23T12:00:06Z] scan second completed",
            ]
        )
        + "\n"
    )

    status = build_live_collection_status(
        live_root=live_root,
        log_dir=log_dir,
        screen_state="running",
    )

    assert status["health_evidence"]["error_match_count"] == 2
    assert status["health_evidence"]["unrecovered_error_match_count"] == 0
    assert status["health_evidence"]["latest_successful_scan"]["line"] == 4

    decision = live_collection_readiness_decision(
        {
            **status,
            "collection_readiness": {"ready_for_training": True},
            "totals": {"features_15m_v1_rows": 1, "labels_15m_v1_rows": 1},
        }
    )

    assert "log health evidence contains fatal-pattern matches" not in decision["blockers"]


def test_live_collection_status_treats_capture_progress_as_recovery(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "capture.log").write_text(
        "\n".join(
            [
                "2026-05-23 12:00:00,000 ERROR bigan.ingestion.sink sink.flush_failed",
                "Traceback (most recent call last):",
                "OSError: [Errno 28] No space left on device",
                "2026-05-23 12:00:30,000 INFO bigan.ingestion.runner gamma.refreshed",
            ]
        )
        + "\n"
    )

    status = build_live_collection_status(
        live_root=live_root,
        log_dir=log_dir,
        screen_state="running",
    )

    health = status["health_evidence"]
    assert health["error_match_count"] == 2
    assert health["recovered_error_match_count"] == 2
    assert health["unrecovered_error_match_count"] == 0
    assert health["latest_successful_scan"] is None
    assert health["latest_recovery_marker"]["line"] == 4


def test_live_collection_status_recovers_errors_across_log_rollover(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "scorer-20260524T062930Z.log").write_text(
        "\n".join(
            [
                "./scripts/run_champion_live.sh: line 176: 89438 Killed: 9",
                "[2026-05-24T06:29:30Z] predictions-v1 failed status=137",
            ]
        )
        + "\n"
    )
    (log_dir / "capture-20260524T063013Z.log").write_text(
        "Traceback (most recent call last)\n"
    )
    (log_dir / "scorer-20260524T063116Z.log").write_text(
        "[2026-05-24T06:33:27Z] scan 20260524T063327Z completed\n"
    )

    status = build_live_collection_status(
        live_root=live_root,
        log_dir=log_dir,
        screen_state="running",
    )

    health = status["health_evidence"]
    assert health["error_match_count"] == 3
    assert health["recovered_error_match_count"] == 3
    assert health["unrecovered_error_match_count"] == 0
    assert health["latest_recovery_marker"]["path"].endswith("scorer-20260524T063116Z.log")


def test_live_collection_status_reports_span_gate_eta(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    t0 = int(datetime(2026, 5, 23, 12, 0, tzinfo=UTC).timestamp() * 1000)
    with WarehouseWriter(live_root / "warehouse", max_rows_per_partition=10) as writer:
        writer.append_rows(
            "features_15m_v1",
            [
                _feature_row(ts=t0, canonical_symbol="BTC-15M:round:UP"),
                _feature_row(ts=t0 + 12 * 60 * 60 * 1000, canonical_symbol="BTC-15M:round:UP"),
            ],
        )

    status = build_live_collection_status(
        live_root=live_root,
        required_families=("BTC-15M",),
        target_days=1.0,
    )

    readiness = status["collection_readiness"]
    features = readiness["features_15m_v1"]
    assert features["limiting_family"] == "BTC-15M"
    assert features["min_family_span_days"] == 0.5
    assert features["remaining_target_days"] == 0.5
    assert features["estimated_ready_at"] == "2026-05-24T12:00:00Z"
    assert features["families"]["BTC-15M"]["remaining_target_ms"] == 12 * 60 * 60 * 1000
    assert features["families"]["BTC-15M"]["estimated_ready_at"] == "2026-05-24T12:00:00Z"
    assert readiness["estimated_ready_at"] is None


def test_build_live_collection_status_reports_invalid_recent_gzip(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    (raw_dir / "2026-05-23T120100Z.ndjson.gz").write_bytes(b"not-gzip")

    status = build_live_collection_status(
        live_root=live_root,
        gzip_check_limit=2,
    )
    decision = live_collection_readiness_decision(status)

    integrity = status["raw_segment_integrity"]
    assert integrity["checked_count"] == 2
    assert integrity["invalid_count"] == 1
    assert integrity["invalid_segments"][0]["segment"] == "2026-05-23T120100Z.ndjson.gz"
    assert any("invalid gzip" in blocker for blocker in decision["blockers"])


def test_live_collection_status_reports_quarantined_raw_segments(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    quarantine_dir = live_root / "raw_invalid" / "ws_market"
    quarantine_dir.mkdir(parents=True)
    quarantined = quarantine_dir / "2026-05-23T115900Z.ndjson.gz"
    quarantined.write_bytes(b"bad-gzip")

    status = build_live_collection_status(
        live_root=live_root,
        generated_at="2026-05-23T12:00:00Z",
    )
    decision = live_collection_readiness_decision(status)

    quarantine = status["raw_segment_quarantine"]
    assert quarantine["quarantined_count"] == 1
    assert quarantine["total_size_bytes"] == len(b"bad-gzip")
    latest = quarantine["latest_quarantined_segment"]
    assert latest["path"] == "raw_invalid/ws_market/2026-05-23T115900Z.ndjson.gz"
    assert latest["segment"] == "2026-05-23T115900Z.ndjson.gz"
    assert latest["segment_ts"] == "2026-05-23T11:59:00Z"
    assert latest["size_bytes"] == len(b"bad-gzip")
    assert latest["gzip_probe"]["gzip_valid"] is False
    assert "BadGzipFile" in latest["gzip_probe"]["error"]
    assert latest["gzip_probe"]["readable_prefix_bytes"] == 0
    assert latest["gzip_probe"]["readable_prefix_lines"] == 0
    clean_window = status["collection_readiness"]["quarantine_clean_window"]
    assert clean_window["meets_target"] is False
    assert clean_window["estimated_ready_at"] == "2026-05-30T11:59:00Z"
    assert clean_window["remaining_target_days"] == pytest.approx(6.999305555555556)
    assert any("raw segment quarantine clean window not ready" in blocker for blocker in decision["blockers"])


def test_live_collection_status_reports_quarantined_raw_segment_readable_prefix(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    quarantine_dir = live_root / "raw_invalid" / "ws_market"
    quarantine_dir.mkdir(parents=True)
    quarantined = quarantine_dir / "2026-05-23T115900Z.ndjson.gz"
    _write_gzip(quarantined, b'{"ok": true}\n')
    with quarantined.open("ab") as fp:
        fp.write(b"trailing-corruption")

    status = build_live_collection_status(
        live_root=live_root,
        generated_at="2026-05-23T12:00:00Z",
    )

    latest = status["raw_segment_quarantine"]["latest_quarantined_segment"]
    probe = latest["gzip_probe"]
    assert probe["gzip_valid"] is False
    assert "BadGzipFile" in probe["error"]
    assert probe["readable_prefix_bytes"] == len(b'{"ok": true}\n')
    assert probe["readable_prefix_lines"] == 1


def test_live_collection_status_allows_quarantine_after_clean_window(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-30T120000Z.ndjson.gz", b"{}\n")
    quarantine_dir = live_root / "raw_invalid" / "ws_market"
    quarantine_dir.mkdir(parents=True)
    (quarantine_dir / "2026-05-23T115900Z.ndjson.gz").write_bytes(b"bad-gzip")

    status = build_live_collection_status(
        live_root=live_root,
        generated_at="2026-05-30T12:00:00Z",
    )
    decision = live_collection_readiness_decision(
        {
            **status,
            "collection_readiness": {
                **status["collection_readiness"],
                "ready_for_training": True,
                "features_15m_v1": {"meets_target": True, "missing_families": []},
                "labels_15m_v1": {"meets_target": True, "missing_families": []},
            },
            "totals": {"features_15m_v1_rows": 1, "labels_15m_v1_rows": 1},
        }
    )

    clean_window = status["collection_readiness"]["quarantine_clean_window"]
    assert clean_window["meets_target"] is True
    assert clean_window["target_progress_pct"] == 100.0
    assert not any("raw segment quarantine" in blocker for blocker in decision["blockers"])


def test_live_collection_status_blocks_insufficient_disk_headroom(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    _write_gzip(raw_dir / "2026-05-23T130000Z.ndjson.gz", b"{}\n")

    status = build_live_collection_status(
        live_root=live_root,
        min_disk_free_bytes=10**30,
    )
    decision = live_collection_readiness_decision(status)

    disk = status["disk_headroom_evidence"]
    assert disk["available"] is True
    assert disk["headroom_ok"] is False
    assert disk["required_free_bytes"] == 10**30
    assert any("disk headroom blocked" in blocker for blocker in decision["blockers"])


def test_live_collection_status_warns_on_low_disk_headroom_margin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    _write_gzip(raw_dir / "2026-05-24T120000Z.ndjson.gz", b"{}\n")

    class _StatVfs:
        f_blocks = 4_000
        f_bavail = 2_100
        f_frsize = 1

    monkeypatch.setattr(collection_status_module.os, "statvfs", lambda _: _StatVfs())
    monkeypatch.setattr(collection_status_module, "_directory_size_bytes", lambda _: 1_000)
    monkeypatch.setattr(collection_status_module, "_raw_segment_span_days", lambda _: 1.0)

    status = build_live_collection_status(
        live_root=live_root,
        target_days=2.0,
        min_disk_free_bytes=100,
        disk_projection_multiplier=1.0,
    )
    decision = live_collection_readiness_decision(status)

    disk = status["disk_headroom_evidence"]
    assert disk["headroom_ok"] is True
    assert disk["required_free_bytes"] == 2_000
    assert disk["headroom_margin_bytes"] == 100
    assert disk["headroom_margin_pct"] == 5.0
    assert disk["headroom_low_margin"] is True
    assert not any("disk headroom blocked" in blocker for blocker in decision["blockers"])
    assert any("disk headroom low margin" in warning for warning in decision["warnings"])


def test_live_collection_readiness_blocks_current_filesystem_disk_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_root = tmp_path / "live"
    live_root.mkdir()

    class _StatVfs:
        f_blocks = 2_000
        f_bavail = 900
        f_frsize = 1

    monkeypatch.setattr(collection_status_module.os, "statvfs", lambda _: _StatVfs())

    status = {
        "generated_at": "2026-05-30T12:00:00Z",
        "screen_session": "collector",
        "screen_state": "running",
        "live_root": str(live_root),
        "raw_segment_count": 10_080,
        "processed_manifest_rows": 10_079,
        "totals": {
            "features_15m_v1_rows": 1000,
            "labels_15m_v1_rows": 1000,
        },
        "collection_readiness": {
            "ready_for_training": True,
        },
        "health_evidence": {
            "error_match_count": 0,
        },
        "disk_headroom_evidence": {
            "headroom_ok": True,
            "headroom_low_margin": False,
            "free_bytes": 5_000,
            "required_free_bytes": 1_000,
            "projected_remaining_bytes": 1_000,
            "headroom_margin_bytes": 4_000,
            "low_margin_threshold_bytes": 100,
        },
    }

    decision = live_collection_readiness_decision(status)

    assert decision["ready"] is False
    assert any(
        "current filesystem disk headroom blocked" in blocker
        for blocker in decision["blockers"]
    )
    current_disk = decision["current_disk_headroom_evidence"]
    assert current_disk["available"] is True
    assert current_disk["free_bytes"] == 900
    assert current_disk["required_free_bytes"] == 1_000
    assert current_disk["headroom_margin_bytes"] == -100
    assert current_disk["headroom_ok"] is False


def test_live_collection_readiness_decision_blocks_stale_progress(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    raw_segment = raw_dir / "2026-05-23T120000Z.ndjson.gz"
    _write_gzip(raw_segment, b"{}\n")
    manifest = tmp_path / "processed.txt"
    manifest.write_text(str(raw_segment) + "\n")

    status = build_live_collection_status(
        live_root=live_root,
        manifest_path=manifest,
        generated_at="2026-05-23T12:10:00Z",
        max_progress_staleness_seconds=300,
    )
    decision = live_collection_readiness_decision(status)

    liveness = status["liveness_evidence"]
    assert liveness["latest_raw_segment"]["age_seconds"] == 600.0
    assert liveness["latest_processed_segment"]["age_seconds"] == 600.0
    assert liveness["raw_segments_fresh"] is False
    assert liveness["processed_manifest_fresh"] is False
    assert any("raw segments stale" in blocker for blocker in decision["blockers"])
    assert any("processed manifest stale" in blocker for blocker in decision["blockers"])


def test_live_collection_readiness_blocks_stale_raw_missing_from_manifest(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    stale_raw = raw_dir / "2026-05-23T120000Z.ndjson.gz"
    fresh_raw = raw_dir / "2026-05-23T120900Z.ndjson.gz"
    _write_gzip(stale_raw, b"{}\n")
    _write_gzip(fresh_raw, b"{}\n")
    manifest = tmp_path / "processed.txt"
    manifest.write_text(str(fresh_raw) + "\n")

    status = build_live_collection_status(
        live_root=live_root,
        manifest_path=manifest,
        generated_at="2026-05-23T12:10:00Z",
        max_progress_staleness_seconds=300,
    )
    decision = live_collection_readiness_decision(status)

    coverage = status["raw_manifest_coverage_evidence"]
    assert coverage["missing_processed_count"] == 1
    assert coverage["stale_missing_processed_count"] == 1
    assert coverage["stale_missing_processed_segments"][0]["segment"] == stale_raw.name
    assert status["liveness_evidence"]["raw_segments_fresh"] is True
    assert status["liveness_evidence"]["processed_manifest_fresh"] is True
    assert any("raw segments missing from processed manifest" in blocker for blocker in decision["blockers"])


def test_live_collection_readiness_allows_recent_raw_manifest_lag(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    processed_raw = raw_dir / "2026-05-23T120800Z.ndjson.gz"
    newest_raw = raw_dir / "2026-05-23T120900Z.ndjson.gz"
    _write_gzip(processed_raw, b"{}\n")
    _write_gzip(newest_raw, b"{}\n")
    manifest = tmp_path / "processed.txt"
    manifest.write_text(str(processed_raw) + "\n")

    status = build_live_collection_status(
        live_root=live_root,
        manifest_path=manifest,
        generated_at="2026-05-23T12:10:00Z",
        max_progress_staleness_seconds=300,
    )
    decision = live_collection_readiness_decision(
        {
            **status,
            "collection_readiness": {"ready_for_training": True},
            "totals": {"features_15m_v1_rows": 1, "labels_15m_v1_rows": 1},
        }
    )

    coverage = status["raw_manifest_coverage_evidence"]
    assert coverage["missing_processed_count"] == 1
    assert coverage["stale_missing_processed_count"] == 0
    assert not any("raw segments missing from processed manifest" in blocker for blocker in decision["blockers"])


def test_live_collection_readiness_decision_blocks_stale_features_and_predictions(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    raw_segment = raw_dir / "2026-05-23T120900Z.ndjson.gz"
    _write_gzip(raw_segment, b"{}\n")
    manifest = tmp_path / "processed.txt"
    manifest.write_text(str(raw_segment) + "\n")

    with WarehouseWriter(live_root / "warehouse", max_rows_per_partition=10) as writer:
        writer.append_rows(
            "features_15m_v1",
            [_feature_row(ts=1_779_537_600_000, canonical_symbol="BTC-15M:round:UP")],
        )
        writer.append_rows(
            "predictions",
            [_prediction_row(ts=1_779_537_600_000, canonical_symbol="BTC-15M:round:UP")],
        )

    status = build_live_collection_status(
        live_root=live_root,
        manifest_path=manifest,
        generated_at="2026-05-23T12:10:00Z",
        required_families=("BTC-15M",),
        max_progress_staleness_seconds=300,
    )
    decision = live_collection_readiness_decision(status)

    freshness = status["warehouse_freshness_evidence"]["tables"]
    assert freshness["features_15m_v1"]["fresh"] is False
    assert freshness["predictions"]["fresh"] is False
    assert freshness["features_15m_v1"]["stale_families"] == ["BTC-15M"]
    assert freshness["predictions"]["stale_families"] == ["BTC-15M"]
    assert any("features_15m_v1 freshness blocked" in blocker for blocker in decision["blockers"])
    assert any("predictions freshness blocked" in blocker for blocker in decision["blockers"])


def test_live_collection_status_reports_label_freshness_lag_by_family(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T123000Z.ndjson.gz", b"{}\n")
    feature_ts = int(datetime(2026, 5, 23, 12, 30, tzinfo=UTC).timestamp() * 1000)
    label_ts = feature_ts - 10 * 60 * 1000

    with WarehouseWriter(live_root / "warehouse", max_rows_per_partition=10) as writer:
        writer.append_rows(
            "features_15m_v1",
            [_feature_row(ts=feature_ts, canonical_symbol="BTC-15M:round:UP")],
        )
        writer.append_rows(
            "labels_15m_v1",
            [_warehouse_label_row(ts=label_ts, canonical_symbol="BTC-15M:round:UP")],
        )

    fresh_status = build_live_collection_status(
        live_root=live_root,
        generated_at="2026-05-23T12:31:00Z",
        required_families=("BTC-15M",),
        max_label_lag_seconds=900,
    )
    stale_status = build_live_collection_status(
        live_root=live_root,
        generated_at="2026-05-23T12:31:00Z",
        required_families=("BTC-15M",),
        max_label_lag_seconds=300,
    )
    stale_decision = live_collection_readiness_decision(stale_status)

    fresh_evidence = fresh_status["label_freshness_evidence"]
    stale_evidence = stale_status["label_freshness_evidence"]
    assert fresh_evidence["fresh"] is True
    assert fresh_evidence["families"]["BTC-15M"]["lag_seconds"] == 600.0
    assert stale_evidence["fresh"] is False
    assert stale_evidence["stale_families"] == ["BTC-15M"]
    assert any("labels_15m_v1 freshness blocked" in blocker for blocker in stale_decision["blockers"])


def test_live_collection_status_reports_monitoring_outcome_coverage(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    monitoring_db = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(monitoring_db)
    try:
        initialize_mlops_db(conn)
        btc_prediction = _prediction_row(
            ts=1_779_552_000_000,
            canonical_symbol="BTC-15M:round:UP",
        )
        eth_prediction = _prediction_row(
            ts=1_779_552_060_000,
            canonical_symbol="ETH-15M:round:UP",
        )
        eth_prediction["source_symbol"] = "tok-2"
        record_prediction_rows_as_events(
            conn,
            [btc_prediction, eth_prediction],
        )
        assert (
            record_label_rows_as_outcomes(
                conn,
                [_label_row(prediction_row=btc_prediction)],
                model_version="xgboost-v4",
            )
            == 1
        )
    finally:
        conn.close()

    status = build_live_collection_status(
        live_root=live_root,
        monitoring_db_path=monitoring_db,
        monitoring_model_version="xgboost-v4",
        required_families=("BTC-15M", "ETH-15M"),
    )
    decision = live_collection_readiness_decision(status)

    monitoring = status["monitoring_outcome_evidence"]
    assert monitoring["available"] is True
    assert monitoring["event_rows"] == 2
    assert monitoring["outcome_rows"] == 1
    assert monitoring["missing_event_families"] == []
    assert monitoring["missing_outcome_families"] == ["ETH-15M"]
    assert monitoring["families"]["BTC-15M"]["outcome_rows"] == 1
    assert monitoring["families"]["BTC-15M"]["brier_score"] == pytest.approx(0.16)
    assert monitoring["families"]["BTC-15M"]["hit_rate"] == pytest.approx(1.0)
    assert monitoring["families"]["BTC-15M"]["avg_realized_return"] == pytest.approx(0.42)
    assert monitoring["families"]["ETH-15M"]["outcome_rows"] == 0
    assert monitoring["brier_score"] == pytest.approx(0.16)
    assert monitoring["hit_rate"] == pytest.approx(1.0)
    assert monitoring["avg_realized_return"] == pytest.approx(0.42)
    assert any("monitoring outcomes missing families: ETH-15M" in blocker for blocker in decision["blockers"])


def test_live_collection_status_retries_transient_monitoring_db_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_root = tmp_path / "live"
    raw_dir = live_root / "raw" / "ws_market"
    raw_dir.mkdir(parents=True)
    _write_gzip(raw_dir / "2026-05-23T120000Z.ndjson.gz", b"{}\n")
    monitoring_db = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(monitoring_db)
    try:
        initialize_mlops_db(conn)
        btc_prediction = _prediction_row(
            ts=1_779_552_000_000,
            canonical_symbol="BTC-15M:round:UP",
        )
        record_prediction_rows_as_events(conn, [btc_prediction])
        record_label_rows_as_outcomes(
            conn,
            [_label_row(prediction_row=btc_prediction)],
            model_version="xgboost-v4",
        )
    finally:
        conn.close()

    real_connect = duckdb.connect
    read_only_calls: list[dict[str, object]] = []
    monkeypatch.setenv("BIGAN_MLOPS_CONNECT_RETRY_DELAY_SECONDS", "0")

    def flaky_connect(path: str, *args: object, **kwargs: object):
        if path == str(monitoring_db) and kwargs.get("read_only") is True:
            read_only_calls.append(dict(kwargs))
            if len(read_only_calls) == 1:
                raise duckdb.IOException("IO Error: Could not set lock on file mlops.duckdb")
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", flaky_connect)

    status = build_live_collection_status(
        live_root=live_root,
        monitoring_db_path=monitoring_db,
        monitoring_model_version="xgboost-v4",
        required_families=("BTC-15M",),
    )

    monitoring = status["monitoring_outcome_evidence"]
    assert len(read_only_calls) == 2
    assert monitoring["available"] is True
    assert monitoring["event_rows"] == 1
    assert monitoring["outcome_rows"] == 1


def test_live_collection_readiness_decision_passes_when_status_is_ready() -> None:
    status = {
        "generated_at": "2026-05-30T12:00:00Z",
        "screen_session": "collector",
        "screen_state": "running",
        "live_root": "data/live/collector",
        "raw_segment_count": 10_080,
        "processed_manifest_rows": 10_079,
        "totals": {
            "features_15m_v1_rows": 1000,
            "labels_15m_v1_rows": 1000,
        },
        "collection_readiness": {
            "ready_for_training": True,
        },
        "health_evidence": {
            "error_match_count": 0,
        },
    }

    decision = live_collection_readiness_decision(status)

    assert decision == {
        "ready": True,
        "blockers": [],
        "warnings": [],
        "screen_session": "collector",
        "screen_state": "running",
        "generated_at": "2026-05-30T12:00:00Z",
        "live_root": "data/live/collector",
        "raw_segment_count": 10_080,
        "processed_manifest_rows": 10_079,
        "feature_rows": 1000,
        "label_rows": 1000,
        "ready_for_training": True,
        "estimated_ready_at": None,
    }


def _write_gzip(path: Path, payload: bytes) -> None:
    with gzip.open(path, "wb") as fp:
        fp.write(payload)


def _feature_row(*, ts: int, canonical_symbol: str) -> dict[str, object]:
    return {
        "ts": ts,
        "message_ts": ts,
        "feature_ts": ts,
        "ingest_ts": ts + 1000,
        "source": "polymarket",
        "source_symbol": "tok-1",
        "source_market": "mkt-1",
        "canonical_symbol": canonical_symbol,
        "symbol": canonical_symbol,
        "feature_version": "bigan-mvp-v1.0.0",
        "completeness_score": 1.0,
        "data_gap_flag": False,
        "quality_filter_pass": True,
        "quote_age_ms": 0,
        "depth_age_ms": 0,
        "trade_age_ms": 0,
        "market_implied_prob": 0.52,
    }


def _prediction_row(*, ts: int, canonical_symbol: str) -> dict[str, object]:
    return {
        "ts": ts,
        "message_ts": ts,
        "prediction_ts": ts,
        "ingest_ts": ts + 1000,
        "source": "polymarket",
        "source_symbol": "tok-1",
        "source_market": "mkt-1",
        "canonical_symbol": canonical_symbol,
        "symbol": canonical_symbol,
        "feature_version": "bigan-mvp-v1.0.0",
        "model_version": "xgboost-v4",
        "calibration_method": "isotonic",
        "prob_up_15m": 0.60,
        "raw_prob_up_15m": 0.58,
        "market_implied_prob": 0.52,
        "confidence_bucket": "high",
        "top_features_json": "[]",
        "feature_values_json": "{}",
    }


def _label_row(*, prediction_row: dict[str, object]) -> dict[str, object]:
    feature_ts = int(prediction_row["prediction_ts"])
    return {
        "feature_ts": feature_ts,
        "target_ts": feature_ts + 900_000,
        "ingest_ts": feature_ts + 901_000,
        "source": prediction_row["source"],
        "source_symbol": prediction_row["source_symbol"],
        "canonical_symbol": prediction_row["canonical_symbol"],
        "label_kind": "up_token_profitability",
        "label_profit_up_15m": True,
        "label_up_15m": True,
        "realized_return": 0.42,
    }


def _warehouse_label_row(*, ts: int, canonical_symbol: str) -> dict[str, object]:
    return {
        "ts": ts,
        "message_ts": ts,
        "feature_ts": ts,
        "target_ts": ts + 900_000,
        "ingest_ts": ts + 901_000,
        "source": "polymarket",
        "source_symbol": f"label-{ts}",
        "source_market": "mkt-1",
        "canonical_symbol": canonical_symbol,
        "symbol": canonical_symbol,
        "label_version": "bigan-labels-15m-profitability-v1.1.0",
        "label_kind": "up_token_profitability",
        "round_slug": "btc-updown-15m-test",
        "round_start_ts": ts,
        "round_end_ts": ts + 900_000,
        "start_price": 100.0,
        "target_price": 101.0,
        "direction_up_15m": True,
        "entry_ask_price": 0.52,
        "settlement_price": 1.0,
        "entry_fee": 0.0,
        "entry_cost": 0.52,
        "realized_return": 0.48,
        "fee_bps": 0.0,
        "label_profit_up_15m": True,
        "label_profit_down_15m": None,
        "label_up_15m": True,
        "label_down_15m": None,
        "label_source": "test",
    }
