"""Live multi-market collection status snapshots."""

from __future__ import annotations

import gzip
import json
import os
import re
import time
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from bigan.canonical.query import open_warehouse

STATUS_ERROR_PATTERNS = ("Killed", "failed status", "Traceback", "ERROR")
STATUS_TABLE_TS_COLS = {
    "symbol_mapping": "ingest_ts",
    "features_15m_v1": "feature_ts",
    "predictions": "prediction_ts",
    "labels_15m_v1": "feature_ts",
}
FRESHNESS_TABLES = ("features_15m_v1", "predictions")
REQUIRED_COLLECTION_FAMILIES = ("BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M")
REQUIRED_XGBOOST_V4_ADDED_FEATURES = (
    "minute_of_day",
    "day_of_week",
    "ret_30m",
    "rv_30m",
    "aggressor_buy_ratio_1m",
    "avg_trade_size_1m",
)
REQUIRED_XGBOOST_V4_TICK_FEATURES = (
    "tick_spread",
    "tick_obi_l1",
    "tick_obi_l3",
    "tick_mid_price",
    "tick_price_velocity",
    "tick_trade_arrival_rate",
)
MS_PER_DAY = 86_400_000
DEFAULT_MAX_LABEL_LAG_SECONDS = 2 * 60 * 60
DEFAULT_MIN_DISK_FREE_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_DISK_PROJECTION_MULTIPLIER = 1.25
DEFAULT_DISK_LOW_MARGIN_BYTES = 1 * 1024 * 1024 * 1024
DEFAULT_DISK_LOW_MARGIN_RATIO = 0.10
RAW_SEGMENT_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{6}Z)")
LOG_FILE_TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")


def build_live_collection_status(
    *,
    live_root: Path | str,
    manifest_path: Path | str | None = None,
    log_dir: Path | str | None = None,
    monitoring_db_path: Path | str | None = None,
    monitoring_model_version: str | None = None,
    screen_session: str | None = None,
    screen_state: str | None = None,
    generated_at: str | None = None,
    settings: dict[str, Any] | None = None,
    target_days: float = 7.0,
    required_families: tuple[str, ...] = REQUIRED_COLLECTION_FAMILIES,
    gzip_check_limit: int = 20,
    max_progress_staleness_seconds: float | None = None,
    max_label_lag_seconds: float | None = DEFAULT_MAX_LABEL_LAG_SECONDS,
    min_disk_free_bytes: int = DEFAULT_MIN_DISK_FREE_BYTES,
    disk_projection_multiplier: float = DEFAULT_DISK_PROJECTION_MULTIPLIER,
) -> dict[str, Any]:
    """Build a JSON-serializable status snapshot for the 7-day collector."""

    root = Path(live_root)
    generated_at_value = generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    generated_at_dt = _parse_status_ts(generated_at_value)
    raw_segments = _raw_segments(root, pattern="*.ndjson.gz")
    active_tmp_segments = _raw_segments(root, pattern="*.tmp")
    family_counts, spans, totals = _warehouse_family_counts(root / "warehouse")
    log_health = _log_health_evidence(Path(log_dir) if log_dir is not None else None)
    target_ms = int(target_days * MS_PER_DAY)
    raw_quarantine = _raw_segment_quarantine(root)
    status = {
        "generated_at": generated_at_value,
        "status": "running_at_evidence_capture" if screen_state == "running" else "observed",
        "screen_session": screen_session,
        "screen_state": screen_state,
        "live_root_lock_evidence": _live_root_lock_evidence(root),
        "live_root": str(root),
        "warehouse": str(root / "warehouse"),
        "log_dir": None if log_dir is None else str(log_dir),
        "raw_segments": raw_segments,
        "raw_segment_count": len(raw_segments),
        "active_tmp_segments": active_tmp_segments,
        "active_tmp_segment_count": len(active_tmp_segments),
        "raw_segment_integrity": _raw_segment_integrity(
            root,
            raw_segments,
            check_limit=gzip_check_limit,
        ),
        "raw_segment_quarantine": raw_quarantine,
        "processed_manifest_rows": _manifest_rows(manifest_path),
        "processed_manifest_integrity": _manifest_integrity(manifest_path),
        "liveness_evidence": _liveness_evidence(
            raw_segments=raw_segments,
            active_tmp_segments=active_tmp_segments,
            manifest_path=manifest_path,
            generated_at=generated_at_dt,
            max_staleness_seconds=max_progress_staleness_seconds,
        ),
        "raw_manifest_coverage_evidence": _raw_manifest_coverage_evidence(
            raw_segments=raw_segments,
            manifest_path=manifest_path,
            generated_at=generated_at_dt,
            max_staleness_seconds=max_progress_staleness_seconds,
        ),
        "warehouse_freshness_evidence": _warehouse_freshness_evidence(
            spans=spans,
            required_families=required_families,
            generated_at=generated_at_dt,
            max_staleness_seconds=max_progress_staleness_seconds,
        ),
        "label_freshness_evidence": _label_freshness_evidence(
            spans=spans,
            required_families=required_families,
            generated_at=generated_at_dt,
            max_label_lag_seconds=max_label_lag_seconds,
        ),
        "monitoring_outcome_evidence": _monitoring_outcome_evidence(
            monitoring_db_path=monitoring_db_path,
            monitoring_model_version=monitoring_model_version,
            required_families=required_families,
        ),
        "warehouse_schema_evidence": _warehouse_schema_evidence(root / "warehouse"),
        "settings": settings or {},
        "family_counts": family_counts,
        "family_spans": spans,
        "totals": totals,
        "collection_readiness": {
            "target_days": target_days,
            "target_ms": target_ms,
            "required_families": list(required_families),
            "quarantine_clean_window": _quarantine_clean_window_status(
                raw_quarantine,
                target_ms=target_ms,
                generated_at=generated_at_dt,
            ),
            "features_15m_v1": _coverage_status(
                spans.get("features_15m_v1", {}),
                required_families=required_families,
                target_ms=target_ms,
            ),
            "labels_15m_v1": _coverage_status(
                spans.get("labels_15m_v1", {}),
                required_families=required_families,
                target_ms=target_ms,
            ),
        },
        "health_evidence": {
            "error_patterns": list(STATUS_ERROR_PATTERNS),
            **log_health,
        },
    }
    readiness = status["collection_readiness"]
    readiness["estimated_ready_at"] = _combined_estimated_ready_at(
        readiness["quarantine_clean_window"],
        readiness["features_15m_v1"],
        readiness["labels_15m_v1"],
    )
    status["disk_headroom_evidence"] = _disk_headroom_evidence(
        root,
        raw_segments=raw_segments,
        readiness=readiness,
        min_free_bytes=min_disk_free_bytes,
        projection_multiplier=disk_projection_multiplier,
    )
    readiness["ready_for_training"] = bool(
        readiness["features_15m_v1"]["meets_target"]
        and readiness["labels_15m_v1"]["meets_target"]
        and readiness["quarantine_clean_window"]["meets_target"]
        and status["disk_headroom_evidence"]["headroom_ok"]
    )
    return status


def write_live_collection_status(path: Path | str, status: dict[str, Any]) -> None:
    """Write a status snapshot with stable JSON formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(status, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        tmp.replace(output_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def live_collection_readiness_decision(status: dict[str, Any]) -> dict[str, Any]:
    """Return a compact pass/block decision for the 7-day corpus gate."""

    readiness = status.get("collection_readiness") or {}
    health = status.get("health_evidence") or {}
    blockers: list[str] = []
    warnings: list[str] = []
    unrecovered_error_count = health.get("unrecovered_error_match_count")
    if unrecovered_error_count is None:
        unrecovered_error_count = health.get("error_match_count")
    if int(unrecovered_error_count or 0) > 0:
        blockers.append("log health evidence contains fatal-pattern matches")
    raw_integrity = status.get("raw_segment_integrity") or {}
    if int(raw_integrity.get("invalid_count") or 0) > 0:
        invalid = raw_integrity.get("invalid_segments") or []
        names = ", ".join(str(item.get("segment")) for item in invalid[:5])
        blockers.append(f"raw segment integrity has invalid gzip files: {names}")
    disk = status.get("disk_headroom_evidence") or {}
    if disk and disk.get("headroom_ok") is False:
        free_gib = float(disk.get("free_bytes") or 0.0) / (1024**3)
        required_gib = float(disk.get("required_free_bytes") or 0.0) / (1024**3)
        projected_gib = float(disk.get("projected_remaining_bytes") or 0.0) / (1024**3)
        blockers.append(
            "disk headroom blocked: "
            f"free={free_gib:.2f}GiB, required={required_gib:.2f}GiB, "
            f"projected_remaining={projected_gib:.2f}GiB"
        )
    elif disk and disk.get("headroom_low_margin") is True:
        margin_gib = float(disk.get("headroom_margin_bytes") or 0.0) / (1024**3)
        threshold_gib = float(disk.get("low_margin_threshold_bytes") or 0.0) / (1024**3)
        free_gib = float(disk.get("free_bytes") or 0.0) / (1024**3)
        required_gib = float(disk.get("required_free_bytes") or 0.0) / (1024**3)
        warnings.append(
            "disk headroom low margin: "
            f"free={free_gib:.2f}GiB, required={required_gib:.2f}GiB, "
            f"margin={margin_gib:.2f}GiB, low_margin_threshold={threshold_gib:.2f}GiB"
        )
    current_disk = _current_filesystem_headroom_evidence(status)
    if current_disk and current_disk.get("available") is True:
        current_free_gib = float(current_disk.get("free_bytes") or 0.0) / (1024**3)
        current_required_gib = float(current_disk.get("required_free_bytes") or 0.0) / (
            1024**3
        )
        current_margin_gib = float(current_disk.get("headroom_margin_bytes") or 0.0) / (
            1024**3
        )
        if current_disk.get("headroom_ok") is False:
            blockers.append(
                "current filesystem disk headroom blocked: "
                f"free={current_free_gib:.2f}GiB, "
                f"required={current_required_gib:.2f}GiB, "
                f"margin={current_margin_gib:.2f}GiB"
            )
        elif current_disk.get("headroom_low_margin") is True:
            current_threshold_gib = float(
                current_disk.get("low_margin_threshold_bytes") or 0.0
            ) / (1024**3)
            warnings.append(
                "current filesystem disk headroom low margin: "
                f"free={current_free_gib:.2f}GiB, "
                f"required={current_required_gib:.2f}GiB, "
                f"margin={current_margin_gib:.2f}GiB, "
                f"low_margin_threshold={current_threshold_gib:.2f}GiB"
            )
    schema_evidence = status.get("warehouse_schema_evidence") or {}
    feature_schema = schema_evidence.get("features_15m_v1") or {}
    missing_required_columns = [
        str(column) for column in feature_schema.get("missing_required_columns") or []
    ]
    if int(feature_schema.get("sampled_file_count") or 0) > 0 and missing_required_columns:
        blockers.append(
            "feature warehouse schema missing required v4/tick columns: "
            + ", ".join(missing_required_columns)
        )
    schema_read_errors = feature_schema.get("read_errors") or []
    if schema_read_errors:
        warnings.append(
            "feature warehouse schema read errors: "
            + ", ".join(str(item.get("path")) for item in schema_read_errors[:5])
        )
    raw_quarantine = status.get("raw_segment_quarantine") or {}
    quarantine_clean_window = readiness.get("quarantine_clean_window") or {}
    if int(raw_quarantine.get("quarantined_count") or 0) > 0 and not quarantine_clean_window.get(
        "meets_target"
    ):
        quarantined = raw_quarantine.get("quarantined_segments") or []
        names = ", ".join(str(item.get("path")) for item in quarantined[:5])
        blockers.append(
            "raw segment quarantine clean window not ready: "
            f"{names}; progress={float(quarantine_clean_window.get('target_progress_pct') or 0.0):.4f}%, "
            f"remaining={float(quarantine_clean_window.get('remaining_target_days') or 0.0):.4f}d, "
            f"estimated_ready_at={quarantine_clean_window.get('estimated_ready_at')}"
        )
    manifest_integrity = status.get("processed_manifest_integrity") or {}
    if int(manifest_integrity.get("malformed_count") or 0) > 0:
        malformed = manifest_integrity.get("malformed_entries") or []
        names = ", ".join(str(item.get("entry")) for item in malformed[:5])
        blockers.append(f"processed manifest has malformed entries: {names}")
    if int(manifest_integrity.get("duplicate_count") or 0) > 0:
        duplicates = manifest_integrity.get("duplicate_entries") or []
        names = ", ".join(str(item.get("entry")) for item in duplicates[:5])
        blockers.append(f"processed manifest has duplicate entries: {names}")
    if status.get("screen_state") not in {None, "running"}:
        blockers.append(f"collector screen is not running: {status.get('screen_state')}")
    lock_evidence = status.get("live_root_lock_evidence") or {}
    if isinstance(lock_evidence, dict):
        if lock_evidence.get("pid_parse_error"):
            blockers.append(
                "live root lock pid is malformed: "
                f"{lock_evidence.get('pid_parse_error')}"
            )
        elif lock_evidence.get("lock_dir_exists") and not lock_evidence.get("pid_file_exists"):
            blockers.append(
                "live root lock is missing its pid file: "
                f"{lock_evidence.get('pid_file')}"
            )
        elif lock_evidence.get("owner_running") is False:
            blockers.append(
                "live root lock owner is not running: "
                f"pid={lock_evidence.get('pid')}"
            )
        elif (
            status.get("screen_state") not in {None, "running"}
            and lock_evidence.get("owner_running") is True
        ):
            blockers.append(
                "collector screen is not running but live root lock owner is active: "
                f"pid={lock_evidence.get('pid')}"
            )
    liveness = status.get("liveness_evidence") or {}
    if liveness.get("raw_segments_fresh") is False:
        blockers.append(_stale_progress_blocker("raw segments", liveness, "latest_raw_segment"))
    if liveness.get("processed_manifest_fresh") is False:
        blockers.append(
            _stale_progress_blocker("processed manifest", liveness, "latest_processed_segment")
        )
    coverage = status.get("raw_manifest_coverage_evidence") or {}
    if isinstance(coverage, dict):
        stale_missing_count = coverage.get("stale_missing_processed_count")
        if stale_missing_count is not None and int(stale_missing_count or 0) > 0:
            stale_missing = coverage.get("stale_missing_processed_segments") or []
            names = ", ".join(str(item.get("segment")) for item in stale_missing[:5])
            blockers.append(f"raw segments missing from processed manifest: {names}")
        if int(coverage.get("extra_processed_count") or 0) > 0:
            extras = coverage.get("extra_processed_segments") or []
            names = ", ".join(str(item.get("segment")) for item in extras[:5])
            blockers.append(f"processed manifest references missing raw segments: {names}")
    warehouse_freshness = status.get("warehouse_freshness_evidence") or {}
    for table, evidence in (warehouse_freshness.get("tables") or {}).items():
        if evidence.get("fresh") is False:
            blockers.append(_stale_warehouse_blocker(str(table), evidence))
    label_freshness = status.get("label_freshness_evidence") or {}
    if label_freshness.get("fresh") is False:
        blockers.append(_stale_label_blocker(label_freshness))
    monitoring = status.get("monitoring_outcome_evidence")
    if isinstance(monitoring, dict) and monitoring.get("available") is True:
        missing_outcome_families = monitoring.get("missing_outcome_families") or []
        if missing_outcome_families:
            blockers.append(
                "monitoring outcomes missing families: "
                + ", ".join(str(family) for family in missing_outcome_families)
            )
    if not readiness.get("ready_for_training"):
        blockers.extend(_readiness_blockers(readiness))
    decision = {
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "screen_session": status.get("screen_session"),
        "screen_state": status.get("screen_state"),
        "generated_at": status.get("generated_at"),
        "live_root": status.get("live_root"),
        "raw_segment_count": status.get("raw_segment_count"),
        "processed_manifest_rows": status.get("processed_manifest_rows"),
        "feature_rows": (status.get("totals") or {}).get("features_15m_v1_rows"),
        "label_rows": (status.get("totals") or {}).get("labels_15m_v1_rows", 0),
        "ready_for_training": bool(readiness.get("ready_for_training")),
        "estimated_ready_at": readiness.get("estimated_ready_at"),
    }
    if current_disk:
        decision["current_disk_headroom_evidence"] = current_disk
    return decision


def read_live_collection_status(path: Path | str) -> dict[str, Any]:
    """Read a status snapshot from JSON."""

    status_path = Path(path)
    deadline = time.monotonic() + 2.0
    while True:
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _raw_segments(root: Path, *, pattern: str) -> list[str]:
    raw_dir = root / "raw" / "ws_market"
    if not raw_dir.exists():
        return []
    paths = [
        path.relative_to(raw_dir).as_posix()
        for path in raw_dir.glob(pattern)
        if path.is_file()
    ]
    if pattern == "*.ndjson.gz":
        done_dir = raw_dir / "_done"
        if done_dir.exists():
            paths.extend(
                path.relative_to(raw_dir).as_posix()
                for path in done_dir.glob(pattern)
                if path.is_file()
            )
    return sorted(paths, key=_segment_sort_key)


def _raw_segment_integrity(
    root: Path,
    raw_segments: list[str],
    *,
    check_limit: int,
) -> dict[str, Any]:
    checked_segments = raw_segments[-max(0, check_limit) :] if check_limit else []
    invalid: list[dict[str, str]] = []
    raw_dir = root / "raw" / "ws_market"
    for segment in checked_segments:
        path = raw_dir / segment
        try:
            _check_gzip(path)
        except (OSError, EOFError, gzip.BadGzipFile, zlib.error) as exc:
            invalid.append(
                {
                    "segment": segment,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "check_limit": check_limit,
        "checked_count": len(checked_segments),
        "unchecked_count": max(0, len(raw_segments) - len(checked_segments)),
        "invalid_count": len(invalid),
        "invalid_segments": invalid,
    }


def _live_root_lock_evidence(root: Path) -> dict[str, Any]:
    lock_dir = root / ".run_champion_live.lock"
    pid_file = lock_dir / "pid"
    evidence: dict[str, Any] = {
        "lock_dir": str(lock_dir),
        "lock_dir_exists": lock_dir.exists(),
        "pid_file": str(pid_file),
        "pid_file_exists": pid_file.exists(),
        "pid": None,
        "pid_raw": None,
        "pid_parse_error": None,
        "owner_running": None,
    }
    if not evidence["pid_file_exists"]:
        return evidence
    try:
        pid_raw = pid_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        evidence["pid_parse_error"] = f"{type(exc).__name__}: {exc}"
        return evidence
    evidence["pid_raw"] = pid_raw
    try:
        pid = int(pid_raw)
    except ValueError:
        evidence["pid_parse_error"] = f"invalid pid {pid_raw!r}"
        return evidence
    evidence["pid"] = pid
    evidence["owner_running"] = _pid_running(pid)
    return evidence


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _raw_segment_quarantine(root: Path, *, limit: int = 20) -> dict[str, Any]:
    quarantine_root = root / "raw_invalid"
    if not quarantine_root.exists():
        return {
            "quarantined_count": 0,
            "total_size_bytes": 0,
            "quarantined_segments": [],
            "latest_quarantined_segment": None,
        }
    entries: list[dict[str, Any]] = []
    total_size_bytes = 0
    for path in sorted(item for item in quarantine_root.glob("**/*") if item.is_file()):
        size_bytes = path.stat().st_size
        total_size_bytes += size_bytes
        segment_ts = _segment_name_ts(path.name)
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "segment": path.name,
                "segment_ts": None
                if segment_ts is None
                else segment_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "size_bytes": size_bytes,
                "gzip_probe": _gzip_probe(path),
            }
        )
    latest = max(
        entries,
        key=lambda item: (str(item.get("segment_ts") or ""), str(item.get("path") or "")),
        default=None,
    )
    return {
        "quarantined_count": len(entries),
        "total_size_bytes": total_size_bytes,
        "quarantined_segments": entries[-limit:],
        "latest_quarantined_segment": latest,
    }


def _quarantine_clean_window_status(
    quarantine: dict[str, Any],
    *,
    target_ms: int,
    generated_at: datetime | None,
) -> dict[str, Any]:
    count = int(quarantine.get("quarantined_count") or 0)
    latest = quarantine.get("latest_quarantined_segment")
    if count <= 0 or not isinstance(latest, dict):
        return {
            "quarantined_count": count,
            "latest_quarantined_segment": None,
            "target_ms": target_ms,
            "target_days": target_ms / MS_PER_DAY if target_ms > 0 else 0.0,
            "target_progress_pct": 100.0,
            "remaining_target_days": 0.0,
            "estimated_ready_at": None,
            "meets_target": True,
        }
    segment_ts = _parse_status_ts(str(latest.get("segment_ts") or ""))
    if segment_ts is None:
        return {
            "quarantined_count": count,
            "latest_quarantined_segment": latest,
            "target_ms": target_ms,
            "target_days": target_ms / MS_PER_DAY if target_ms > 0 else 0.0,
            "target_progress_pct": 0.0,
            "remaining_target_days": target_ms / MS_PER_DAY if target_ms > 0 else 0.0,
            "estimated_ready_at": None,
            "meets_target": False,
        }
    ready_at = segment_ts + timedelta(milliseconds=target_ms)
    elapsed_ms = 0.0
    if generated_at is not None:
        elapsed_ms = max(0.0, (generated_at - segment_ts).total_seconds() * 1000.0)
    remaining_ms = max(0.0, target_ms - elapsed_ms)
    return {
        "quarantined_count": count,
        "latest_quarantined_segment": latest,
        "target_ms": target_ms,
        "target_days": target_ms / MS_PER_DAY if target_ms > 0 else 0.0,
        "target_progress_pct": (
            min(1.0, elapsed_ms / target_ms) * 100.0 if target_ms > 0 else 100.0
        ),
        "remaining_target_days": remaining_ms / MS_PER_DAY,
        "estimated_ready_at": ready_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meets_target": bool(generated_at is not None and generated_at >= ready_at),
    }


def _disk_headroom_evidence(
    root: Path,
    *,
    raw_segments: list[str],
    readiness: dict[str, Any],
    min_free_bytes: int,
    projection_multiplier: float,
) -> dict[str, Any]:
    try:
        stat = os.statvfs(root if root.exists() else root.parent)
    except OSError as exc:
        return {
            "available": False,
            "path": str(root),
            "error": f"{type(exc).__name__}: {exc}",
            "headroom_ok": False,
        }
    total_bytes = int(stat.f_blocks * stat.f_frsize)
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    used_bytes = max(0, total_bytes - free_bytes)
    live_root_size_bytes = _directory_size_bytes(root)
    observed_span_days = _raw_segment_span_days(raw_segments)
    remaining_days = _readiness_remaining_target_days(readiness)
    projected_remaining_bytes = 0
    if observed_span_days and remaining_days:
        projected_remaining_bytes = int(
            (live_root_size_bytes / observed_span_days)
            * remaining_days
            * max(0.0, projection_multiplier)
        )
    required_free_bytes = max(int(min_free_bytes), projected_remaining_bytes)
    headroom_margin_bytes = free_bytes - required_free_bytes
    headroom_margin_pct = (
        None
        if required_free_bytes <= 0
        else headroom_margin_bytes / required_free_bytes * 100.0
    )
    low_margin_threshold_bytes = max(
        DEFAULT_DISK_LOW_MARGIN_BYTES,
        int(required_free_bytes * DEFAULT_DISK_LOW_MARGIN_RATIO),
    )
    headroom_ok = free_bytes >= required_free_bytes
    return {
        "available": True,
        "path": str(root),
        "total_bytes": total_bytes,
        "free_bytes": free_bytes,
        "used_bytes": used_bytes,
        "used_pct": None if total_bytes <= 0 else used_bytes / total_bytes * 100.0,
        "live_root_size_bytes": live_root_size_bytes,
        "observed_raw_span_days": observed_span_days,
        "remaining_target_days": remaining_days,
        "projection_multiplier": projection_multiplier,
        "projected_remaining_bytes": projected_remaining_bytes,
        "min_free_bytes": int(min_free_bytes),
        "required_free_bytes": required_free_bytes,
        "headroom_margin_bytes": headroom_margin_bytes,
        "headroom_margin_pct": headroom_margin_pct,
        "low_margin_threshold_bytes": low_margin_threshold_bytes,
        "headroom_low_margin": headroom_ok
        and headroom_margin_bytes < low_margin_threshold_bytes,
        "headroom_ok": headroom_ok,
    }


def _current_filesystem_headroom_evidence(status: dict[str, Any]) -> dict[str, Any] | None:
    """Return current filesystem headroom for a status snapshot's live root."""

    live_root = status.get("live_root")
    disk = status.get("disk_headroom_evidence") or {}
    if not live_root or not isinstance(disk, dict):
        return None
    required_free_bytes = _optional_int(disk.get("required_free_bytes"))
    if required_free_bytes is None:
        return None
    root = Path(str(live_root))
    if not root.exists():
        return None
    try:
        stat = os.statvfs(root)
    except OSError as exc:
        return {
            "available": False,
            "path": str(root),
            "error": f"{type(exc).__name__}: {exc}",
            "headroom_ok": False,
        }
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    low_margin_threshold_bytes = _optional_int(disk.get("low_margin_threshold_bytes"))
    if low_margin_threshold_bytes is None:
        low_margin_threshold_bytes = max(
            DEFAULT_DISK_LOW_MARGIN_BYTES,
            int(required_free_bytes * DEFAULT_DISK_LOW_MARGIN_RATIO),
        )
    headroom_margin_bytes = free_bytes - required_free_bytes
    headroom_ok = free_bytes >= required_free_bytes
    return {
        "available": True,
        "path": str(root),
        "free_bytes": free_bytes,
        "status_free_bytes": disk.get("free_bytes"),
        "required_free_bytes": required_free_bytes,
        "projected_remaining_bytes": disk.get("projected_remaining_bytes"),
        "headroom_margin_bytes": headroom_margin_bytes,
        "low_margin_threshold_bytes": low_margin_threshold_bytes,
        "headroom_low_margin": headroom_ok
        and headroom_margin_bytes < low_margin_threshold_bytes,
        "headroom_ok": headroom_ok,
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _directory_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.glob("**/*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _raw_segment_span_days(raw_segments: list[str]) -> float | None:
    parsed = [
        segment_ts
        for name in raw_segments
        if (segment_ts := _segment_name_ts(name)) is not None
    ]
    if len(parsed) < 2:
        return None
    span_seconds = max(0.0, (max(parsed) - min(parsed)).total_seconds())
    if span_seconds <= 0:
        return None
    return span_seconds / 86_400.0


def _segment_sort_key(name: str) -> tuple[datetime, str]:
    return (_segment_name_ts(name) or datetime.min.replace(tzinfo=UTC), name)


def _readiness_remaining_target_days(readiness: dict[str, Any]) -> float:
    remaining: list[float] = []
    quarantine = readiness.get("quarantine_clean_window")
    if isinstance(quarantine, dict):
        remaining.append(float(quarantine.get("remaining_target_days") or 0.0))
    for table in ("features_15m_v1", "labels_15m_v1"):
        coverage = readiness.get(table)
        if isinstance(coverage, dict):
            remaining.append(float(coverage.get("remaining_target_days") or 0.0))
    return max(remaining, default=0.0)


def _check_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as fp:
        while fp.read(1024 * 1024):
            pass


def _gzip_probe(path: Path) -> dict[str, Any]:
    readable_bytes = 0
    readable_lines = 0
    try:
        with gzip.open(path, "rb") as fp:
            while chunk := fp.read1(64 * 1024):
                readable_bytes += len(chunk)
                readable_lines += chunk.count(b"\n")
    except (OSError, EOFError, gzip.BadGzipFile, zlib.error) as exc:
        return {
            "gzip_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "readable_prefix_bytes": readable_bytes,
            "readable_prefix_lines": readable_lines,
        }
    return {
        "gzip_valid": True,
        "error": None,
        "readable_prefix_bytes": readable_bytes,
        "readable_prefix_lines": readable_lines,
    }


def _manifest_rows(manifest_path: Path | str | None) -> int | None:
    if manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return 0
    return len(_manifest_entry_keys(path))


def _manifest_integrity(manifest_path: Path | str | None) -> dict[str, Any]:
    if manifest_path is None:
        return {
            "line_count": None,
            "unique_count": None,
            "duplicate_count": None,
            "malformed_count": None,
            "duplicate_entries": [],
            "malformed_entries": [],
        }
    path = Path(manifest_path)
    if not path.exists():
        return {
            "line_count": 0,
            "unique_count": 0,
            "duplicate_count": 0,
            "malformed_count": 0,
            "duplicate_entries": [],
            "malformed_entries": [],
        }
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    seen: set[str] = set()
    duplicate_entries: list[dict[str, str]] = []
    malformed_entries: list[dict[str, str]] = []
    duplicate_seen: set[str] = set()
    for entry in entries:
        name = Path(entry).name
        if _segment_name_ts(name) is None:
            malformed_entries.append({"entry": entry})
        if name in seen and name not in duplicate_seen:
            duplicate_entries.append({"entry": entry, "segment": name})
            duplicate_seen.add(name)
        seen.add(name)
    return {
        "line_count": len(entries),
        "unique_count": len(seen),
        "duplicate_count": len(entries) - len(seen),
        "malformed_count": len(malformed_entries),
        "duplicate_entries": duplicate_entries,
        "malformed_entries": malformed_entries,
    }


def _liveness_evidence(
    *,
    raw_segments: list[str],
    active_tmp_segments: list[str],
    manifest_path: Path | str | None,
    generated_at: datetime | None,
    max_staleness_seconds: float | None,
) -> dict[str, Any]:
    latest_raw = _latest_segment_evidence(raw_segments, generated_at=generated_at)
    latest_tmp = _latest_segment_evidence(active_tmp_segments, generated_at=generated_at)
    processed_segments = _manifest_segment_names(manifest_path)
    latest_processed = _latest_segment_evidence(processed_segments, generated_at=generated_at)
    evidence = {
        "max_progress_staleness_seconds": max_staleness_seconds,
        "latest_raw_segment": latest_raw,
        "latest_active_tmp_segment": latest_tmp,
        "latest_processed_segment": latest_processed,
    }
    if max_staleness_seconds is None:
        evidence["raw_segments_fresh"] = None
        evidence["processed_manifest_fresh"] = None
        return evidence
    evidence["raw_segments_fresh"] = _fresh_enough(latest_raw, max_staleness_seconds)
    evidence["processed_manifest_fresh"] = _fresh_enough(latest_processed, max_staleness_seconds)
    return evidence


def _raw_manifest_coverage_evidence(
    *,
    raw_segments: list[str],
    manifest_path: Path | str | None,
    generated_at: datetime | None,
    max_staleness_seconds: float | None,
) -> dict[str, Any]:
    raw_names = {Path(segment).name for segment in raw_segments}
    processed_names = set(_manifest_segment_names(manifest_path))
    missing_processed = sorted(raw_names - processed_names, key=_segment_sort_key)
    extra_processed = sorted(processed_names - raw_names, key=_segment_sort_key)
    stale_missing = [
        segment
        for segment in missing_processed
        if _segment_age_seconds(segment, generated_at) is not None
        and max_staleness_seconds is not None
        and float(_segment_age_seconds(segment, generated_at) or 0.0) > max_staleness_seconds
    ]
    return {
        "max_progress_staleness_seconds": max_staleness_seconds,
        "raw_segment_count": len(raw_names),
        "processed_segment_count": len(processed_names),
        "missing_processed_count": len(missing_processed),
        "missing_processed_segments": [
            _segment_age_evidence(segment, generated_at=generated_at)
            for segment in missing_processed[-20:]
        ],
        "stale_missing_processed_count": (
            None if max_staleness_seconds is None else len(stale_missing)
        ),
        "stale_missing_processed_segments": [
            _segment_age_evidence(segment, generated_at=generated_at)
            for segment in stale_missing[-20:]
        ],
        "extra_processed_count": len(extra_processed),
        "extra_processed_segments": [
            _segment_age_evidence(segment, generated_at=generated_at)
            for segment in extra_processed[-20:]
        ],
    }


def _warehouse_freshness_evidence(
    *,
    spans: dict[str, dict[str, dict[str, int | None]]],
    required_families: tuple[str, ...],
    generated_at: datetime | None,
    max_staleness_seconds: float | None,
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for table in FRESHNESS_TABLES:
        table_spans = spans.get(table, {})
        families: dict[str, Any] = {}
        missing_families: list[str] = []
        stale_families: list[str] = []
        for family in required_families:
            span = table_spans.get(family)
            max_ts = None if span is None else span.get("max_ts")
            family_evidence = _warehouse_ts_evidence(max_ts, generated_at=generated_at)
            if family_evidence is None:
                missing_families.append(family)
                families[family] = {
                    "latest_ts": None,
                    "age_seconds": None,
                    "fresh": False if max_staleness_seconds is not None else None,
                }
                continue
            fresh = (
                None
                if max_staleness_seconds is None
                else _fresh_enough(family_evidence, max_staleness_seconds)
            )
            if fresh is False:
                stale_families.append(family)
            families[family] = {
                **family_evidence,
                "fresh": fresh,
            }
        table_fresh = (
            None
            if max_staleness_seconds is None
            else not missing_families and not stale_families
        )
        tables[table] = {
            "fresh": table_fresh,
            "missing_families": missing_families,
            "stale_families": stale_families,
            "families": families,
        }
    return {
        "max_progress_staleness_seconds": max_staleness_seconds,
        "tables": tables,
    }


def _warehouse_schema_evidence(
    warehouse: Path,
    *,
    sample_file_limit: int = 20,
) -> dict[str, Any]:
    required_added = list(REQUIRED_XGBOOST_V4_ADDED_FEATURES)
    required_tick = list(REQUIRED_XGBOOST_V4_TICK_FEATURES)
    required_columns = required_added + required_tick
    features_dir = warehouse / "features_15m_v1"
    feature_files = sorted(path for path in features_dir.glob("**/*.parquet") if path.is_file())
    sampled_files = feature_files[-max(0, sample_file_limit) :] if sample_file_limit else []
    null_summary = {
        column: {"rows": 0, "nulls": 0, "non_nulls": 0, "null_rate": None}
        for column in required_columns
    }
    files_missing_columns: list[dict[str, Any]] = []
    read_errors: list[dict[str, str]] = []
    sampled_rows = 0
    readable_file_count = 0

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        return {
            "features_15m_v1": {
                "available": features_dir.exists(),
                "feature_file_count": len(feature_files),
                "sample_file_limit": sample_file_limit,
                "sampled_file_count": len(sampled_files),
                "readable_file_count": 0,
                "sampled_rows": 0,
                "required_added_features": required_added,
                "required_tick_features": required_tick,
                "missing_required_columns": [],
                "missing_added_features": [],
                "missing_tick_features": [],
                "required_columns_present": None,
                "files_missing_required_columns": [],
                "feature_null_summary": null_summary,
                "latest_feature_file": _warehouse_relative_path(warehouse, feature_files[-1])
                if feature_files
                else None,
                "read_errors": [{"path": None, "error": f"ImportError: {exc}"}],
            }
        }

    for path in sampled_files:
        rel_path = _warehouse_relative_path(warehouse, path)
        try:
            parquet_file = pq.ParquetFile(path)
            schema_names = set(parquet_file.schema_arrow.names)
        except Exception as exc:  # pragma: no cover - exact parquet exceptions vary.
            read_errors.append({"path": rel_path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        readable_file_count += 1
        missing = [column for column in required_columns if column not in schema_names]
        if missing:
            files_missing_columns.append(
                {"path": rel_path, "missing_required_columns": missing}
            )
        columns_to_read = [column for column in required_columns if column in schema_names]
        if not columns_to_read:
            continue
        try:
            table = parquet_file.read(columns=columns_to_read)
        except Exception as exc:  # pragma: no cover - exact parquet exceptions vary.
            read_errors.append({"path": rel_path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        sampled_rows += table.num_rows
        for column in columns_to_read:
            values = table[column]
            rows = len(values)
            nulls = int(values.null_count)
            summary = null_summary[column]
            summary["rows"] += rows
            summary["nulls"] += nulls
            summary["non_nulls"] += rows - nulls

    for summary in null_summary.values():
        rows = int(summary["rows"])
        summary["null_rate"] = None if rows <= 0 else int(summary["nulls"]) / rows
    missing_required = sorted(
        {
            str(column)
            for item in files_missing_columns
            for column in item.get("missing_required_columns", [])
        }
    )
    missing_added = [column for column in required_added if column in missing_required]
    missing_tick = [column for column in required_tick if column in missing_required]
    return {
        "features_15m_v1": {
            "available": features_dir.exists(),
            "feature_file_count": len(feature_files),
            "sample_file_limit": sample_file_limit,
            "sampled_file_count": len(sampled_files),
            "readable_file_count": readable_file_count,
            "sampled_rows": sampled_rows,
            "required_added_features": required_added,
            "required_tick_features": required_tick,
            "missing_required_columns": missing_required,
            "missing_added_features": missing_added,
            "missing_tick_features": missing_tick,
            "required_columns_present": None
            if not sampled_files
            else not missing_required and readable_file_count == len(sampled_files),
            "files_missing_required_columns": files_missing_columns[-sample_file_limit:],
            "feature_null_summary": null_summary,
            "latest_feature_file": _warehouse_relative_path(warehouse, feature_files[-1])
            if feature_files
            else None,
            "read_errors": read_errors[-sample_file_limit:],
        }
    }


def _warehouse_relative_path(warehouse: Path, path: Path) -> str:
    try:
        return path.relative_to(warehouse).as_posix()
    except ValueError:
        return str(path)


def _label_freshness_evidence(
    *,
    spans: dict[str, dict[str, dict[str, int | None]]],
    required_families: tuple[str, ...],
    generated_at: datetime | None,
    max_label_lag_seconds: float | None,
) -> dict[str, Any]:
    feature_spans = spans.get("features_15m_v1", {})
    label_spans = spans.get("labels_15m_v1", {})
    families: dict[str, Any] = {}
    missing_feature_families: list[str] = []
    missing_label_families: list[str] = []
    stale_families: list[str] = []
    for family in required_families:
        feature_max_ts = (feature_spans.get(family) or {}).get("max_ts")
        label_max_ts = (label_spans.get(family) or {}).get("max_ts")
        feature_evidence = _warehouse_ts_evidence(feature_max_ts, generated_at=generated_at)
        label_evidence = _warehouse_ts_evidence(label_max_ts, generated_at=generated_at)
        if feature_evidence is None:
            missing_feature_families.append(family)
        if label_evidence is None:
            missing_label_families.append(family)
        lag_seconds = (
            None
            if feature_max_ts is None or label_max_ts is None
            else max(0.0, (int(feature_max_ts) - int(label_max_ts)) / 1000.0)
        )
        fresh = None
        if max_label_lag_seconds is not None:
            fresh = (
                feature_evidence is not None
                and label_evidence is not None
                and lag_seconds is not None
                and lag_seconds <= max_label_lag_seconds
            )
            if fresh is False and family not in missing_feature_families + missing_label_families:
                stale_families.append(family)
        families[family] = {
            "feature_latest_ts": (
                None if feature_evidence is None else feature_evidence["latest_ts"]
            ),
            "feature_age_seconds": (
                None if feature_evidence is None else feature_evidence["age_seconds"]
            ),
            "label_latest_ts": None if label_evidence is None else label_evidence["latest_ts"],
            "label_age_seconds": None if label_evidence is None else label_evidence["age_seconds"],
            "lag_seconds": lag_seconds,
            "fresh": fresh,
        }
    overall_fresh = (
        None
        if max_label_lag_seconds is None
        else not missing_feature_families and not missing_label_families and not stale_families
    )
    return {
        "max_label_lag_seconds": max_label_lag_seconds,
        "fresh": overall_fresh,
        "missing_feature_families": missing_feature_families,
        "missing_label_families": missing_label_families,
        "stale_families": stale_families,
        "families": families,
    }


def _warehouse_ts_evidence(
    ts_ms: int | None,
    *,
    generated_at: datetime | None,
) -> dict[str, Any] | None:
    if ts_ms is None:
        return None
    latest = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=UTC)
    age_seconds = None
    if generated_at is not None:
        age_seconds = max(0.0, (generated_at - latest).total_seconds())
    return {
        "latest_ts": latest.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "age_seconds": age_seconds,
    }


def _ts_ms_to_iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _monitoring_outcome_evidence(
    *,
    monitoring_db_path: Path | str | None,
    monitoring_model_version: str | None,
    required_families: tuple[str, ...],
) -> dict[str, Any] | None:
    if monitoring_db_path is None or monitoring_model_version is None:
        return None
    db_path = Path(monitoring_db_path)
    if not db_path.exists():
        return {
            "available": False,
            "db_path": str(db_path),
            "model_version": monitoring_model_version,
            "error": "monitoring db does not exist",
        }
    try:
        from bigan.mlops.registry import connect_mlops_db

        conn = connect_mlops_db(db_path, read_only=True)
    except duckdb.Error as exc:
        return {
            "available": False,
            "db_path": str(db_path),
            "model_version": monitoring_model_version,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        rows = conn.execute(
            """
            WITH events AS (
                SELECT
                    json_extract_string(e.feature_snapshot_json, '$.canonical_symbol')
                        AS canonical_symbol,
                    e.ts AS prediction_ts,
                    o.outcome_ts AS outcome_ts,
                    o.event_id IS NOT NULL AS has_outcome,
                    e.prob_up_15m AS prob_up_15m,
                    o.realized_label AS realized_label,
                    o.realized_return AS realized_return,
                    o.brier_component AS brier_component
                FROM prediction_events e
                LEFT JOIN prediction_outcomes o USING (event_id)
                WHERE e.model_version = ?
            ),
            normalized AS (
                SELECT
                    regexp_extract(canonical_symbol, '^([A-Z]+-[0-9]+M)', 1) AS family,
                    prediction_ts,
                    outcome_ts,
                    has_outcome,
                    CASE
                        WHEN upper(canonical_symbol) LIKE '%:DOWN'
                          OR upper(canonical_symbol) LIKE '%-DOWN-15M'
                        THEN 1.0 - prob_up_15m
                        ELSE prob_up_15m
                    END AS token_prob,
                    realized_label,
                    realized_return,
                    brier_component
                FROM events
            )
            SELECT
                family,
                COUNT(*) AS event_rows,
                SUM(CASE WHEN has_outcome THEN 1 ELSE 0 END) AS outcome_rows,
                AVG(CASE WHEN has_outcome THEN brier_component ELSE NULL END) AS brier_score,
                AVG(CASE
                    WHEN has_outcome THEN
                        CASE
                            WHEN (token_prob >= 0.5 AND realized_label = TRUE)
                              OR (token_prob < 0.5 AND realized_label = FALSE)
                            THEN 1.0
                            ELSE 0.0
                        END
                    ELSE NULL
                END) AS hit_rate,
                AVG(CASE WHEN has_outcome THEN realized_return ELSE NULL END)
                    AS avg_realized_return,
                MIN(prediction_ts) AS min_prediction_ts,
                MAX(prediction_ts) AS max_prediction_ts,
                MIN(outcome_ts) AS min_outcome_ts,
                MAX(outcome_ts) AS max_outcome_ts
            FROM normalized
            WHERE family IS NOT NULL AND family <> ''
            GROUP BY family
            ORDER BY family
            """,
            [monitoring_model_version],
        ).fetchall()
    except duckdb.Error as exc:
        return {
            "available": False,
            "db_path": str(db_path),
            "model_version": monitoring_model_version,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        conn.close()

    families: dict[str, dict[str, Any]] = {}
    event_total = 0
    outcome_total = 0
    for (
        family,
        event_rows,
        outcome_rows,
        brier_score,
        hit_rate,
        avg_realized_return,
        min_prediction_ts,
        max_prediction_ts,
        min_outcome_ts,
        max_outcome_ts,
    ) in rows:
        event_count = int(event_rows or 0)
        outcome_count = int(outcome_rows or 0)
        event_total += event_count
        outcome_total += outcome_count
        families[str(family)] = {
            "event_rows": event_count,
            "outcome_rows": outcome_count,
            "outcome_coverage_pct": (
                (outcome_count / event_count) * 100.0 if event_count else 0.0
            ),
            "brier_score": None if brier_score is None else float(brier_score),
            "hit_rate": None if hit_rate is None else float(hit_rate),
            "avg_realized_return": (
                None if avg_realized_return is None else float(avg_realized_return)
            ),
            "min_prediction_ts": None if min_prediction_ts is None else int(min_prediction_ts),
            "max_prediction_ts": None if max_prediction_ts is None else int(max_prediction_ts),
            "min_outcome_ts": None if min_outcome_ts is None else int(min_outcome_ts),
            "max_outcome_ts": None if max_outcome_ts is None else int(max_outcome_ts),
        }
    missing_event_families = [
        family for family in required_families if families.get(family, {}).get("event_rows", 0) == 0
    ]
    missing_outcome_families = [
        family
        for family in required_families
        if families.get(family, {}).get("outcome_rows", 0) == 0
    ]
    return {
        "available": True,
        "db_path": str(db_path),
        "model_version": monitoring_model_version,
        "event_rows": event_total,
        "outcome_rows": outcome_total,
        "outcome_coverage_pct": (outcome_total / event_total) * 100.0 if event_total else 0.0,
        "brier_score": _weighted_family_metric(families, "brier_score"),
        "hit_rate": _weighted_family_metric(families, "hit_rate"),
        "avg_realized_return": _weighted_family_metric(families, "avg_realized_return"),
        "missing_event_families": missing_event_families,
        "missing_outcome_families": missing_outcome_families,
        "families": families,
    }


def _manifest_segment_names(manifest_path: Path | str | None) -> list[str]:
    if manifest_path is None:
        return []
    path = Path(manifest_path)
    if not path.exists():
        return []
    return sorted(_manifest_entry_keys(path), key=_segment_sort_key)


def _manifest_entry_keys(path: Path) -> set[str]:
    return {
        Path(text).name
        for line in path.read_text().splitlines()
        if (text := line.strip())
    }


def _segment_age_evidence(
    segment_name: str,
    *,
    generated_at: datetime | None,
) -> dict[str, Any]:
    segment_ts = _segment_name_ts(segment_name)
    return {
        "segment": segment_name,
        "segment_ts": None
        if segment_ts is None
        else segment_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "age_seconds": _segment_age_seconds(segment_name, generated_at),
    }


def _segment_age_seconds(segment_name: str, generated_at: datetime | None) -> float | None:
    segment_ts = _segment_name_ts(segment_name)
    if segment_ts is None or generated_at is None:
        return None
    return max(0.0, (generated_at - segment_ts).total_seconds())


def _latest_segment_evidence(
    segment_names: list[str],
    *,
    generated_at: datetime | None,
) -> dict[str, Any] | None:
    parsed = [
        (segment_ts, name)
        for name in segment_names
        if (segment_ts := _segment_name_ts(name)) is not None
    ]
    if not parsed:
        return None
    segment_ts, name = max(parsed, key=lambda item: item[0])
    age_seconds = None
    if generated_at is not None:
        age_seconds = max(0.0, (generated_at - segment_ts).total_seconds())
    return {
        "segment": name,
        "segment_ts": segment_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "age_seconds": age_seconds,
    }


def _segment_name_ts(name: str) -> datetime | None:
    match = RAW_SEGMENT_TIMESTAMP_RE.search(name)
    if match is None:
        return None
    return _parse_status_ts(match.group(1))


def _parse_status_ts(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fresh_enough(segment: dict[str, Any] | None, max_staleness_seconds: float) -> bool:
    if segment is None:
        return False
    age_seconds = segment.get("age_seconds")
    if age_seconds is None:
        return False
    return float(age_seconds) <= max_staleness_seconds


def _stale_progress_blocker(
    label: str,
    liveness: dict[str, Any],
    segment_key: str,
) -> str:
    max_age = float(liveness.get("max_progress_staleness_seconds") or 0.0)
    segment = liveness.get(segment_key)
    if not isinstance(segment, dict):
        return f"{label} have no timestamped progress evidence"
    age = float(segment.get("age_seconds") or 0.0)
    name = segment.get("segment")
    return f"{label} stale: latest {name} age {age:.1f}s exceeds {max_age:.1f}s"


def _weighted_family_metric(
    families: dict[str, dict[str, Any]],
    key: str,
) -> float | None:
    weighted_total = 0.0
    weight_sum = 0
    for family in families.values():
        value = family.get(key)
        weight = int(family.get("outcome_rows") or 0)
        if value is None or weight <= 0:
            continue
        weighted_total += float(value) * weight
        weight_sum += weight
    return None if weight_sum == 0 else weighted_total / weight_sum


def _stale_warehouse_blocker(table: str, evidence: dict[str, Any]) -> str:
    missing = evidence.get("missing_families") or []
    stale = evidence.get("stale_families") or []
    parts: list[str] = []
    if missing:
        parts.append(f"missing families: {', '.join(str(item) for item in missing)}")
    if stale:
        family_evidence = evidence.get("families") or {}
        stale_text = ", ".join(
            _family_staleness_text(str(family), family_evidence.get(family) or {})
            for family in stale
        )
        parts.append(f"stale families: {stale_text}")
    detail = "; ".join(parts) if parts else "freshness is false"
    return f"{table} freshness blocked: {detail}"


def _stale_label_blocker(evidence: dict[str, Any]) -> str:
    missing_features = evidence.get("missing_feature_families") or []
    missing_labels = evidence.get("missing_label_families") or []
    stale = evidence.get("stale_families") or []
    parts: list[str] = []
    if missing_features:
        parts.append(f"missing feature families: {', '.join(str(item) for item in missing_features)}")
    if missing_labels:
        parts.append(f"missing label families: {', '.join(str(item) for item in missing_labels)}")
    if stale:
        family_evidence = evidence.get("families") or {}
        stale_text = ", ".join(
            _family_label_lag_text(str(family), family_evidence.get(family) or {})
            for family in stale
        )
        max_lag = float(evidence.get("max_label_lag_seconds") or 0.0)
        parts.append(f"stale families: {stale_text} exceeds {max_lag:.1f}s")
    detail = "; ".join(parts) if parts else "freshness is false"
    return f"labels_15m_v1 freshness blocked: {detail}"


def _family_staleness_text(family: str, evidence: dict[str, Any]) -> str:
    age = evidence.get("age_seconds")
    latest = evidence.get("latest_ts")
    if age is None:
        return f"{family} latest={latest}"
    return f"{family} latest={latest} age={float(age):.1f}s"


def _family_label_lag_text(family: str, evidence: dict[str, Any]) -> str:
    lag = evidence.get("lag_seconds")
    label_latest = evidence.get("label_latest_ts")
    feature_latest = evidence.get("feature_latest_ts")
    if lag is None:
        return f"{family} feature_latest={feature_latest} label_latest={label_latest}"
    return (
        f"{family} feature_latest={feature_latest} "
        f"label_latest={label_latest} lag={float(lag):.1f}s"
    )


def _warehouse_family_counts(
    warehouse: Path,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int | None]]], dict[str, int]]:
    family_counts: dict[str, dict[str, int]] = {}
    spans: dict[str, dict[str, dict[str, int | None]]] = {}
    totals: dict[str, int] = {}
    if not warehouse.exists():
        return family_counts, spans, totals
    with open_warehouse(warehouse) as conn:
        for table, ts_col in STATUS_TABLE_TS_COLS.items():
            try:
                rows = conn.execute(
                    f"""
                    SELECT regexp_extract(canonical_symbol, '^([A-Z]+-[0-9]+M)', 1) AS family,
                           COUNT(*) AS rows,
                           MIN({ts_col}) AS min_ts,
                           MAX({ts_col}) AS max_ts
                    FROM {table}
                    GROUP BY 1
                    ORDER BY 1
                    """
                ).fetchall()
            except (duckdb.CatalogException, duckdb.IOException):
                continue
            family_counts[table] = {str(family): int(row_count) for family, row_count, _, _ in rows}
            spans[table] = {
                str(family): {
                    "rows": int(row_count),
                    "min_ts": None if min_ts is None else int(min_ts),
                    "max_ts": None if max_ts is None else int(max_ts),
                }
                for family, row_count, min_ts, max_ts in rows
            }
            totals[f"{table}_rows"] = sum(family_counts[table].values())
    return family_counts, spans, totals


def _coverage_status(
    family_spans: dict[str, dict[str, int | None]],
    *,
    required_families: tuple[str, ...],
    target_ms: int,
) -> dict[str, Any]:
    per_family: dict[str, dict[str, Any]] = {}
    missing_families: list[str] = []
    span_days: list[float] = []
    ready_at_values: list[int] = []
    limiting_family: str | None = None
    limiting_span_ms: int | None = None
    for family in required_families:
        span = family_spans.get(family)
        if span is None:
            missing_families.append(family)
            per_family[family] = {
                "rows": 0,
                "span_ms": 0,
                "span_days": 0.0,
                "remaining_target_ms": target_ms,
                "estimated_ready_at": None,
                "meets_target": False,
            }
            continue
        min_ts = span.get("min_ts")
        max_ts = span.get("max_ts")
        span_ms = (
            max(0, int(max_ts) - int(min_ts))
            if min_ts is not None and max_ts is not None
            else 0
        )
        days = span_ms / MS_PER_DAY
        span_days.append(days)
        if limiting_span_ms is None or span_ms < limiting_span_ms:
            limiting_family = family
            limiting_span_ms = span_ms
        ready_at_ms = None if min_ts is None else int(min_ts) + target_ms
        if ready_at_ms is not None:
            ready_at_values.append(ready_at_ms)
        per_family[family] = {
            "rows": int(span.get("rows") or 0),
            "min_ts": min_ts,
            "max_ts": max_ts,
            "span_ms": span_ms,
            "span_days": days,
            "remaining_target_ms": max(0, target_ms - span_ms),
            "estimated_ready_at": _ts_ms_to_iso(ready_at_ms),
            "meets_target": span_ms >= target_ms,
        }
    table_estimated_ready_at = (
        None if missing_families or not ready_at_values else _ts_ms_to_iso(max(ready_at_values))
    )
    return {
        "missing_families": missing_families,
        "limiting_family": limiting_family,
        "min_family_span_days": min(span_days) if span_days else 0.0,
        "target_progress_pct": (
            min(1.0, min(span_days) * MS_PER_DAY / target_ms) * 100.0
            if span_days and target_ms > 0
            else 0.0
        ),
        "remaining_target_days": (
            max(0.0, (target_ms - min(span_days) * MS_PER_DAY) / MS_PER_DAY)
            if span_days
            else target_ms / MS_PER_DAY
        ),
        "estimated_ready_at": table_estimated_ready_at,
        "meets_target": not missing_families
        and all(item["meets_target"] for item in per_family.values()),
        "families": per_family,
    }


def _combined_estimated_ready_at(*coverage_blocks: dict[str, Any]) -> str | None:
    ready_at_values: list[datetime] = []
    for coverage in coverage_blocks:
        if coverage.get("missing_families"):
            return None
        ready_at = coverage.get("estimated_ready_at")
        if not ready_at and coverage.get("meets_target") is True:
            continue
        if not ready_at:
            return None
        parsed = _parse_status_ts(str(ready_at))
        if parsed is None:
            return None
        ready_at_values.append(parsed)
    if not ready_at_values:
        return None
    return max(ready_at_values).strftime("%Y-%m-%dT%H:%M:%SZ")


def _readiness_blockers(readiness: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for table in ("features_15m_v1", "labels_15m_v1"):
        coverage = readiness.get(table) or {}
        missing = coverage.get("missing_families") or []
        if missing:
            blockers.append(f"{table} missing families: {', '.join(missing)}")
        if not coverage.get("meets_target"):
            min_days = float(coverage.get("min_family_span_days") or 0.0)
            target_days = float(readiness.get("target_days") or 0.0)
            blockers.append(
                f"{table} minimum family span {min_days:.4f}d below target {target_days:.4f}d"
            )
    if not blockers:
        blockers.append("collection_readiness.ready_for_training is false")
    return blockers


def _log_health_evidence(log_dir: Path | None, *, limit: int = 20) -> dict[str, Any]:
    if log_dir is None or not log_dir.exists():
        return {
            "error_matches": [],
            "error_match_count": 0,
            "recovered_error_match_count": 0,
            "unrecovered_error_matches": [],
            "unrecovered_error_match_count": 0,
            "latest_successful_scan": None,
            "latest_recovery_marker": None,
        }
    matches: list[dict[str, Any]] = []
    latest_recovery_by_path: dict[str, dict[str, Any]] = {}
    latest_success_by_path: dict[str, dict[str, Any]] = {}
    for path in sorted(log_dir.glob("*.log")):
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if _is_successful_scan_line(line):
                latest_success_by_path[str(path)] = {
                    "path": str(path),
                    "line": line_number,
                    "text": line,
                    "ts": _log_entry_ts(path, line),
                }
            if _is_recovery_line(line):
                latest_recovery_by_path[str(path)] = {
                    "path": str(path),
                    "line": line_number,
                    "text": line,
                    "ts": _log_entry_ts(path, line),
                }
            if any(pattern in line for pattern in STATUS_ERROR_PATTERNS):
                matches.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "text": line,
                        "ts": _log_entry_ts(path, line),
                    }
                )
    latest_successes = sorted(
        latest_success_by_path.values(),
        key=lambda item: (str(item.get("ts") or ""), str(item["path"]), int(item["line"])),
    )
    latest_recovery_markers = sorted(
        latest_recovery_by_path.values(),
        key=lambda item: (str(item.get("ts") or ""), str(item["path"]), int(item["line"])),
    )
    latest_recovery_marker = latest_recovery_markers[-1] if latest_recovery_markers else None
    unrecovered = [
        match
        for match in matches
        if not _error_match_recovered(match, latest_recovery_by_path, latest_recovery_marker)
    ]
    return {
        "error_matches": matches[-limit:],
        "error_match_count": len(matches),
        "recovered_error_match_count": len(matches) - len(unrecovered),
        "unrecovered_error_matches": unrecovered[-limit:],
        "unrecovered_error_match_count": len(unrecovered),
        "latest_successful_scan": latest_successes[-1] if latest_successes else None,
        "latest_recovery_marker": latest_recovery_marker,
    }


def _error_match_recovered(
    match: dict[str, Any],
    latest_recovery_by_path: dict[str, dict[str, Any]],
    latest_recovery_marker: dict[str, Any] | None,
) -> bool:
    path_recovery = latest_recovery_by_path.get(str(match["path"])) or {}
    if int(match["line"]) <= int(path_recovery.get("line") or 0):
        return True
    match_ts = match.get("ts")
    recovery_ts = (latest_recovery_marker or {}).get("ts")
    return bool(match_ts and recovery_ts and str(match_ts) <= str(recovery_ts))


def _is_recovery_line(line: str) -> bool:
    return (
        _is_successful_scan_line(line)
        or "INFO bigan.ingestion.runner gamma.refreshed" in line
        or "INFO bigan.ingestion.clob_ws ws.subscribed" in line
    )


def _is_successful_scan_line(line: str) -> bool:
    return "] scan " in line and " completed" in line


def _log_line_ts(line: str) -> str | None:
    if not line.startswith("["):
        return None
    marker_end = line.find("]")
    if marker_end <= 1:
        return None
    return line[1:marker_end]


def _log_entry_ts(path: Path, line: str) -> str | None:
    return _log_line_ts(line) or _log_path_ts(path)


def _log_path_ts(path: Path) -> str | None:
    raw_match = RAW_SEGMENT_TIMESTAMP_RE.search(path.name)
    log_match = LOG_FILE_TIMESTAMP_RE.search(path.name)
    if raw_match is not None:
        raw_value = raw_match.group(1)
        fmt = "%Y-%m-%dT%H%M%SZ"
    elif log_match is not None:
        raw_value = log_match.group(1)
        fmt = "%Y%m%dT%H%M%SZ"
    else:
        return None
    try:
        parsed = datetime.strptime(raw_value, fmt).replace(tzinfo=UTC)
    except ValueError:
        return None
    return parsed.isoformat().replace("+00:00", "Z")
