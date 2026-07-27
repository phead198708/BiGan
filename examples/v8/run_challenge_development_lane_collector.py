"""Run the persistent BTC-15m outcome-blind challenge development lane."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_development_lane import (  # noqa: E402
    SAFETY,
    append_outcome_blind_batch,
    atomic_write_json,
    build_daily_capture_summary,
    load_jsonl,
    sha256_file,
    validate_development_lane_protocol,
)
from examples.v8.run_polymarket_async_round_collector import (  # noqa: E402
    run_polymarket_async_round_collector_cli,
)

DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/challenge_model_development_lane_15m_protocol.json"
)


def run_service(
    *,
    service_root: Path | str,
    protocol_path: Path | str,
    expected_protocol_sha256: str,
    max_batches: int,
    failure_backoff_seconds: float,
) -> dict:
    root = Path(service_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(protocol_path).resolve()
    if sha256_file(protocol_path) != expected_protocol_sha256.lower():
        raise ValueError("development lane protocol SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validated = validate_development_lane_protocol(protocol, repo_root=ROOT)
    frozen_protocol = root / "development_lane_protocol.json"
    if frozen_protocol.is_file():
        if sha256_file(frozen_protocol) != expected_protocol_sha256.lower():
            raise ValueError("service protocol bytes differ from the launched protocol")
    else:
        frozen_protocol.write_bytes(protocol_path.read_bytes())

    lock_path = root / "development_lane_collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("development lane collector is already running") from exc
        return _run_locked(
            root=root,
            validated=validated,
            protocol_sha256=expected_protocol_sha256.lower(),
            max_batches=max_batches,
            failure_backoff_seconds=failure_backoff_seconds,
        )


def _run_locked(
    *,
    root: Path,
    validated: dict,
    protocol_sha256: str,
    max_batches: int,
    failure_backoff_seconds: float,
) -> dict:
    index_path = root / "outcome_blind_capture_batch_index.jsonl"
    state_path = root / "collector_state.json"
    capture_root = root / "captures"
    daily_root = root / "daily_capture_summaries"
    collector_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    completed_in_process = 0
    consecutive_failures = 0

    while max_batches == 0 or completed_in_process < max_batches:
        index_rows = load_jsonl(index_path)
        attempted = sum(int(row.get("capture_count") or 0) for row in index_rows)
        maximum = int(validated["maximum_capture_attempts_before_additional_permission"])
        remaining = maximum - attempted
        if remaining <= 0:
            state = _state(
                status="paused_before_attempt_120_pending_explicit_permission",
                collector_commit=collector_commit,
                protocol_sha256=protocol_sha256,
                validated=validated,
                attempted=attempted,
                last_batch=None if not index_rows else index_rows[-1],
            )
            atomic_write_json(state_path, state)
            _write_daily(daily_root, index_rows, status=state["status"])
            return state

        batch_sequence = len(index_rows) + 1
        batch_round_count = min(int(validated["batch_round_count"]), remaining)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        batch_id = f"challenge-development-15m-{batch_sequence:06d}-{stamp}"
        collecting_state = _state(
            status="collecting_outcome_blind_15m_development_batch",
            collector_commit=collector_commit,
            protocol_sha256=protocol_sha256,
            validated=validated,
            attempted=attempted,
            last_batch=None if not index_rows else index_rows[-1],
        )
        collecting_state.update(
            {
                "active_batch_id": batch_id,
                "active_batch_round_count": batch_round_count,
                "capture_process_reads_outcomes_labels_or_pnl": False,
            }
        )
        atomic_write_json(state_path, collecting_state)
        try:
            summary = run_polymarket_async_round_collector_cli(
                batch_id=batch_id,
                output_dir=capture_root,
                round_count=batch_round_count,
                market_family=str(validated["market_family"]),
                public_provider_timeout_seconds=930.0,
                public_provider_http_timeout_seconds=5.0,
                orderbook_snapshot_interval_seconds=1.0,
                orderbook_ws_initial_complete_book_timeout_seconds=15.0,
                rest_orderbook_fallback_collection_seconds=930.0,
                settlement_poll_interval_seconds=15.0,
                settlement_grace_seconds=0.0,
                max_round_start_lag_seconds=30.0,
                chainlink_rtds_warmup_seconds=60.0,
                chainlink_rtds_stale_reconnect_seconds=15.0,
                market_identity_cache_path=root / "gamma_market_identity_cache.json",
                gamma_market_identity_prefetch_round_count=8,
                market_identity_cache_max_age_seconds=7_200.0,
                clob_identity_revalidation_max_attempts=3,
                clob_identity_revalidation_retry_seconds=0.25,
                feature_enrichment_max_attempts=40,
                outcome_blind_collection_only=True,
            )
            summary_path = Path(str(summary["batch_summary_path"])).resolve()
            entry = append_outcome_blind_batch(
                index_path=index_path,
                summary_path=summary_path,
                collector_commit=collector_commit,
                protocol_sha256=protocol_sha256,
                diagnostic_freeze_sha256=str(validated["diagnostic_freeze_sha256"]),
            )
            index_rows = load_jsonl(index_path)
            attempted = sum(int(row.get("capture_count") or 0) for row in index_rows)
            completed_in_process += 1
            consecutive_failures = 0
            state = _state(
                status="running_persistent_outcome_blind_15m_development_lane",
                collector_commit=collector_commit,
                protocol_sha256=protocol_sha256,
                validated=validated,
                attempted=attempted,
                last_batch=entry,
            )
            atomic_write_json(state_path, state)
            _write_daily(daily_root, index_rows, status=state["status"])
        except Exception as exc:
            consecutive_failures += 1
            failure_state = _state(
                status="capture_batch_failed_fail_closed",
                collector_commit=collector_commit,
                protocol_sha256=protocol_sha256,
                validated=validated,
                attempted=attempted,
                last_batch=None if not index_rows else index_rows[-1],
            )
            failure_state.update(
                {
                    "failed_batch_id": batch_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "consecutive_failure_count": consecutive_failures,
                }
            )
            atomic_write_json(state_path, failure_state)
            _write_daily(daily_root, index_rows, status=failure_state["status"])
            if consecutive_failures >= 3:
                raise
            time.sleep(failure_backoff_seconds)

    final_rows = load_jsonl(index_path)
    final_state = _state(
        status="bounded_collector_smoke_complete",
        collector_commit=collector_commit,
        protocol_sha256=protocol_sha256,
        validated=validated,
        attempted=sum(int(row.get("capture_count") or 0) for row in final_rows),
        last_batch=None if not final_rows else final_rows[-1],
    )
    atomic_write_json(state_path, final_state)
    _write_daily(daily_root, final_rows, status=final_state["status"])
    return final_state


def _state(
    *,
    status: str,
    collector_commit: str,
    protocol_sha256: str,
    validated: dict,
    attempted: int,
    last_batch: dict | None,
) -> dict:
    return {
        "schema_version": "bigan-challenge-model-development-lane-collector-state-v1",
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "collector_pid": os.getpid(),
        "collector_commit": collector_commit,
        "protocol_sha256": protocol_sha256,
        "diagnostic_freeze_path": validated["diagnostic_freeze_path"],
        "diagnostic_freeze_sha256": validated["diagnostic_freeze_sha256"],
        "market_family": validated["market_family"],
        "attempted_market_count": attempted,
        "authorization_checkpoint": int(
            validated["maximum_capture_attempts_before_additional_permission"]
        ),
        "attempt_120_authorized": False,
        "outcomes_labels_or_pnl_available_to_capture_control": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "last_completed_batch": last_batch,
        "safety": dict(SAFETY),
    }


def _write_daily(root: Path, rows: list[dict], *, status: str) -> None:
    date_utc = datetime.now(UTC).date().isoformat()
    summary = build_daily_capture_summary(
        index_rows=rows,
        date_utc=date_utc,
        collector_pid=os.getpid(),
        service_status=status,
    )
    atomic_write_json(root / f"{date_utc}.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Zero runs persistently; positive values are bounded operational smoke.",
    )
    parser.add_argument("--failure-backoff-seconds", type=float, default=30.0)
    args = parser.parse_args()
    state = run_service(
        service_root=args.service_root,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.protocol_sha256,
        max_batches=args.max_batches,
        failure_backoff_seconds=args.failure_backoff_seconds,
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
