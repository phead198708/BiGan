"""Operational soak-run evidence helpers for issue #25.

The 24h acceptance test is mostly operational, but the repo should make it
easy to run the service, capture durable evidence, and fail fast when the
observed Prometheus metrics violate the agreed thresholds.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import resource
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson

from .metrics import REGISTRY
from .rollup import rollup_file

_DURATION_CHECK_GRACE_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class SoakThresholds:
    """Thresholds used to judge a soak run."""

    min_duration_seconds: float = 86_400.0
    max_reconnects: float = 24.0
    max_last_event_lag_seconds: float = 60.0
    max_hash_mismatches: float = 0.0
    max_rss_growth_mb: float = 256.0


def build_soak_sample(started_at_seconds: float) -> dict[str, Any]:
    """Snapshot the in-process metrics and lightweight process state."""

    sample_ts = time.time()
    return {
        "sample_ts": sample_ts,
        "uptime_seconds": max(0.0, sample_ts - started_at_seconds),
        "metrics": {
            "ws_reconnects_total": _metric_sum("bigan_ws_reconnects_total"),
            "last_event_receive_time_seconds": _metric_value(
                "bigan_last_event_receive_time_seconds"
            ),
            "ws_hash_mismatch_total": _metric_sum("bigan_ws_hash_mismatch_total"),
            "sink_records_written_total": _metric_sum(
                "bigan_sink_records_written_total"
            ),
            "ws_subscribed_markets": _metric_value("bigan_ws_subscribed_markets"),
            "rollup_files_total_ok": _metric_sum(
                "bigan_rollup_files_total",
                labels={"outcome": "ok"},
            ),
            "rollup_files_total_error": _metric_sum(
                "bigan_rollup_files_total",
                labels={"outcome": "error"},
            ),
        },
        "process": {
            "max_rss_mb": _max_rss_mb(),
        },
    }


async def record_soak_samples(
    path: Path,
    *,
    started_at_seconds: float,
    interval_seconds: float,
    stop_event: asyncio.Event,
) -> None:
    """Append metric samples until ``stop_event`` is set."""

    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        append_soak_sample(path, build_soak_sample(started_at_seconds))
        if stop_event.is_set():
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue
        append_soak_sample(path, build_soak_sample(started_at_seconds))
        return


def append_soak_sample(path: Path, sample: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fp:
        fp.write(orjson.dumps(sample, option=orjson.OPT_APPEND_NEWLINE))


def read_soak_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("rb") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            samples.append(orjson.loads(line))
    return samples


def summarize_soak(
    samples: list[dict[str, Any]],
    *,
    raw_dir: Path,
    rollup_dir: Path,
    thresholds: SoakThresholds,
    fatal_exit: str | None = None,
    market_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate collected samples and raw artifacts against soak thresholds."""

    raw_stats = inspect_raw_archive(raw_dir)
    rollup_stats = inspect_rollup_outputs(rollup_dir)
    checks: list[dict[str, Any]] = []

    if samples:
        first = samples[0]
        last = samples[-1]
        duration_seconds = float(last["sample_ts"]) - float(first["sample_ts"])
        metrics = _summarize_metrics(samples)
        process = _summarize_process(samples)
    else:
        first = last = {}
        duration_seconds = 0.0
        metrics = {
            "ws_reconnects_delta": 0.0,
            "max_last_event_lag_seconds": float("inf"),
            "ws_hash_mismatch_delta": 0.0,
            "sink_records_written_delta": 0.0,
            "rollup_files_total_ok_delta": 0.0,
            "rollup_files_total_error_delta": 0.0,
            "last_ws_subscribed_markets": 0.0,
        }
        process = {"first_max_rss_mb": 0.0, "last_max_rss_mb": 0.0, "rss_growth_mb": 0.0}

    _add_check(
        checks,
        "no_fatal_exit",
        fatal_exit is None,
        observed=fatal_exit or "none",
        threshold="none",
    )
    _add_check(
        checks,
        "duration_seconds",
        duration_seconds + _DURATION_CHECK_GRACE_SECONDS
        >= thresholds.min_duration_seconds,
        observed=round(duration_seconds, 3),
        threshold={
            "min_seconds": thresholds.min_duration_seconds,
            "grace_seconds": _DURATION_CHECK_GRACE_SECONDS,
        },
    )
    _add_check(
        checks,
        "ws_reconnects_total",
        metrics["ws_reconnects_delta"] <= thresholds.max_reconnects,
        observed=metrics["ws_reconnects_delta"],
        threshold=thresholds.max_reconnects,
    )
    _add_check(
        checks,
        "last_event_receive_lag_seconds",
        metrics["max_last_event_lag_seconds"] <= thresholds.max_last_event_lag_seconds,
        observed=metrics["max_last_event_lag_seconds"],
        threshold=thresholds.max_last_event_lag_seconds,
    )
    _add_check(
        checks,
        "ws_hash_mismatch_total",
        metrics["ws_hash_mismatch_delta"] <= thresholds.max_hash_mismatches,
        observed=metrics["ws_hash_mismatch_delta"],
        threshold=thresholds.max_hash_mismatches,
    )
    _add_check(
        checks,
        "sink_records_written",
        metrics["sink_records_written_delta"] > 0 or raw_stats["records"] > 0,
        observed={
            "metric_delta": metrics["sink_records_written_delta"],
            "raw_records": raw_stats["records"],
        },
        threshold="> 0",
    )
    _add_check(
        checks,
        "raw_ndjson_decodable",
        raw_stats["bad_lines"] == 0 and raw_stats["bad_files"] == 0,
        observed=raw_stats,
        threshold={"bad_lines": 0, "bad_files": 0},
    )
    _add_check(
        checks,
        "rollup_completed",
        rollup_stats["parquet_files"] > 0 or metrics["rollup_files_total_ok_delta"] > 0,
        observed={
            "parquet_files": rollup_stats["parquet_files"],
            "metric_delta": metrics["rollup_files_total_ok_delta"],
        },
        threshold="> 0",
    )
    _add_check(
        checks,
        "rss_growth_mb",
        process["rss_growth_mb"] <= thresholds.max_rss_growth_mb,
        observed=round(process["rss_growth_mb"], 3),
        threshold=thresholds.max_rss_growth_mb,
    )

    if market_coverage is not None:
        _add_check(
            checks,
            "market_coverage",
            bool(market_coverage.get("passed")),
            observed=_market_coverage_observed(market_coverage),
            threshold={"passed": True},
        )

    summary = {
        "passed": all(check["passed"] for check in checks),
        "thresholds": asdict(thresholds),
        "started_at": first.get("sample_ts"),
        "ended_at": last.get("sample_ts"),
        "duration_seconds": duration_seconds,
        "checks": checks,
        "metrics": metrics,
        "process": process,
        "raw": raw_stats,
        "rollup": rollup_stats,
    }
    if market_coverage is not None:
        summary["market_coverage"] = market_coverage
    return summary


def write_soak_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def finalize_soak_rollup(raw_dir: Path, rollup_dir: Path) -> dict[str, Any]:
    """Roll up remaining top-level NDJSON files after the soak service stops."""

    if not raw_dir.exists():
        return {"files": 0, "records": 0, "errors": []}

    done_dir = raw_dir / "_done"
    files = sorted(raw_dir.glob("*.ndjson.gz"))
    records = 0
    errors: list[str] = []
    for path in files:
        try:
            records += rollup_file(path, rollup_dir, done_dir=done_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")
    return {"files": len(files), "records": records, "errors": errors}


def inspect_raw_archive(raw_dir: Path) -> dict[str, Any]:
    """Count and decode-check active plus archived NDJSON gzip files."""

    files = _raw_files(raw_dir)
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
                        orjson.loads(line)
                        records += 1
                    except orjson.JSONDecodeError:
                        bad_lines += 1
        except EOFError:
            incomplete_files += 1
        except OSError:
            bad_files += 1
    return {
        "files": len(files),
        "records": records,
        "bad_lines": bad_lines,
        "bad_files": bad_files,
        "incomplete_files": incomplete_files,
    }


def inspect_rollup_outputs(rollup_dir: Path) -> dict[str, Any]:
    if not rollup_dir.exists():
        return {"parquet_files": 0}
    return {"parquet_files": sum(1 for _ in rollup_dir.rglob("*.parquet"))}


def _raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    files = list(raw_dir.glob("*.ndjson.gz"))
    done_dir = raw_dir / "_done"
    if done_dir.exists():
        files.extend(done_dir.glob("*.ndjson.gz"))
    return sorted(files)


def _metric_sum(name: str, *, labels: Mapping[str, str] | None = None) -> float:
    total = 0.0
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name != name:
                continue
            if labels is not None and not all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                continue
            total += float(sample.value)
    return total


def _metric_value(name: str) -> float:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name:
                return float(sample.value)
    return 0.0


def _summarize_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    first = samples[0]["metrics"]
    last = samples[-1]["metrics"]
    lags = []
    for sample in samples:
        sample_ts = float(sample["sample_ts"])
        last_event_ts = float(sample["metrics"].get("last_event_receive_time_seconds") or 0)
        if last_event_ts > 0:
            lags.append(max(0.0, sample_ts - last_event_ts))

    return {
        "ws_reconnects_delta": _delta(first, last, "ws_reconnects_total"),
        "max_last_event_lag_seconds": max(lags) if lags else float("inf"),
        "ws_hash_mismatch_delta": _delta(first, last, "ws_hash_mismatch_total"),
        "sink_records_written_delta": _delta(first, last, "sink_records_written_total"),
        "rollup_files_total_ok_delta": _delta(first, last, "rollup_files_total_ok"),
        "rollup_files_total_error_delta": _delta(first, last, "rollup_files_total_error"),
        "last_ws_subscribed_markets": float(last.get("ws_subscribed_markets") or 0),
    }


def _summarize_process(samples: list[dict[str, Any]]) -> dict[str, float]:
    first = float(samples[0]["process"].get("max_rss_mb") or 0)
    last = float(samples[-1]["process"].get("max_rss_mb") or 0)
    return {
        "first_max_rss_mb": first,
        "last_max_rss_mb": last,
        "rss_growth_mb": max(0.0, last - first),
    }


def _delta(first: Mapping[str, Any], last: Mapping[str, Any], key: str) -> float:
    return max(0.0, float(last.get(key) or 0) - float(first.get(key) or 0))


def _max_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any,
    threshold: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": passed,
            "observed": observed,
            "threshold": threshold,
        }
    )


def _market_coverage_observed(report: Mapping[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {"passed": bool(report.get("passed"))}
    if "error" in report:
        observed["error"] = report["error"]

    for section, keys in {
        "gamma": (
            "markets",
            "assets",
            "source_markets",
            "ignored_markets_after_raw_end",
            "ignored_markets_opened_after_raw_end",
            "ignored_markets_scheduled_after_raw_end",
        ),
        "raw": (
            "records",
            "assets_with_any_event",
            "assets_with_book",
            "bad_lines",
            "bad_files",
            "incomplete_files",
        ),
        "rest": ("books_ok", "books_missing"),
        "coverage": ("missing_any_assets", "missing_book_assets"),
        "freshness": ("stale_assets",),
        "hash": ("required", "compared", "matches", "mismatches"),
    }.items():
        value = report.get(section)
        if not isinstance(value, Mapping):
            continue
        observed[section] = {
            key: value[key]
            for key in keys
            if key in value
        }
    return observed
