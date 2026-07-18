"""Run durable candidate-agnostic raw collection in bounded resumable batches."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (  # noqa: E402
    PersistentOutcomeBlindBatchIndexConfig,
    index_persistent_outcome_blind_batch,
    validate_persistent_outcome_blind_collector_protocol,
)
from examples.v8.run_polymarket_async_round_collector import (  # noqa: E402
    run_polymarket_async_round_collector_cli,
)

DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
SAFETY = {
    "paper_only": True,
    "capital_at_risk": False,
    "polymarket_write_enabled": False,
    "wallet_signing_enabled": False,
    "source_model_candidate_eligible": False,
    "freeze_ready": False,
    "promotion_evidence_eligible": False,
    "v8_execution_handoff_allowed": False,
    "#134_resume_allowed": False,
    "#146_start_allowed": False,
}


def _single_service_instance(function):
    """Prevent overlapping service processes from collecting the same round window."""

    @wraps(function)
    def wrapped(**kwargs):
        root = Path(kwargs["service_root"]).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / "persistent_outcome_blind_service.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(
                    lock_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as exc:
                raise RuntimeError(
                    "persistent outcome-blind collector service already running"
                ) from exc
            try:
                return function(**kwargs)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    return wrapped


@_single_service_instance
def run_service(
    *,
    service_root: Path | str,
    protocol_path: Path | str,
    protocol_sha256: str,
    batch_round_count: int,
    max_batches: int,
    max_consecutive_failures: int,
    failure_backoff_seconds: float,
) -> dict:
    if batch_round_count <= 0:
        raise ValueError("batch_round_count must be positive")
    if max_batches < 0:
        raise ValueError("max_batches must be non-negative")
    if max_consecutive_failures <= 0:
        raise ValueError("max_consecutive_failures must be positive")
    root = Path(service_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(protocol_path).resolve()
    if _sha256(protocol_path) != protocol_sha256.lower():
        raise ValueError("persistent collector protocol SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_persistent_outcome_blind_collector_protocol(protocol)
    frozen_protocol_path = root / "persistent_outcome_blind_collector_protocol.json"
    if frozen_protocol_path.exists():
        if _sha256(frozen_protocol_path) != protocol_sha256.lower():
            raise ValueError("frozen service protocol SHA-256 mismatch")
    else:
        frozen_protocol_path.write_bytes(protocol_path.read_bytes())
    protocol_path = frozen_protocol_path
    index_path = root / "persistent_outcome_blind_round_index.jsonl"
    service_state_path = root / "persistent_outcome_blind_service_state.json"
    service_state = _load_state(service_state_path)
    batch_sequence = int(service_state.get("last_completed_batch_sequence") or 0) + 1
    completed_batches = 0
    consecutive_failures = 0
    collector_commit = _git_head()
    while max_batches == 0 or completed_batches < max_batches:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        batch_id = f"persistent-outcome-blind-{batch_sequence:06d}-{stamp}"
        try:
            summary = run_polymarket_async_round_collector_cli(
                batch_id=batch_id,
                output_dir=root / "captures",
                round_count=batch_round_count,
                market_family="btc_updown_5m",
                public_provider_timeout_seconds=330.0,
                public_provider_http_timeout_seconds=5.0,
                orderbook_snapshot_interval_seconds=1.0,
                orderbook_ws_initial_complete_book_timeout_seconds=15.0,
                rest_orderbook_fallback_collection_seconds=330.0,
                settlement_poll_interval_seconds=15.0,
                settlement_grace_seconds=0.0,
                max_round_start_lag_seconds=30.0,
                chainlink_rtds_warmup_seconds=60.0,
                chainlink_rtds_stale_reconnect_seconds=15.0,
                market_identity_cache_path=root / "gamma_market_identity_cache.json",
                gamma_market_identity_prefetch_round_count=12,
                market_identity_cache_max_age_seconds=7_200.0,
                clob_identity_revalidation_max_attempts=3,
                clob_identity_revalidation_retry_seconds=0.25,
                feature_enrichment_max_attempts=40,
                outcome_blind_collection_only=True,
            )
            summary_path = Path(str(summary["batch_summary_path"])).resolve()
            index_result = index_persistent_outcome_blind_batch(
                PersistentOutcomeBlindBatchIndexConfig(
                    run_id=f"persistent-outcome-blind-index-{batch_sequence:06d}-{stamp}",
                    output_dir=root / "index_runs",
                    protocol_path=protocol_path,
                    expected_protocol_sha256=protocol_sha256,
                    index_path=index_path,
                    batch_summary_path=summary_path,
                    expected_batch_summary_sha256=_sha256(summary_path),
                    collector_git_commit=collector_commit,
                )
            )
            consecutive_failures = 0
            completed_batches += 1
            _write_state(
                service_state_path,
                {
                    "status": "collecting_persistent_outcome_blind_batches",
                    "last_completed_batch_sequence": batch_sequence,
                    "last_batch_id": batch_id,
                    "last_batch_summary_path": str(summary_path),
                    "last_batch_summary_sha256": _sha256(summary_path),
                    "collector_git_commit": collector_commit,
                    "protocol_path": str(protocol_path),
                    "protocol_sha256": protocol_sha256.lower(),
                    "batch_round_count": batch_round_count,
                    "index_path": str(index_path),
                    "index_sha256": index_result["index_sha256"],
                    "index_entry_count": index_result["report"]["index_entry_count"],
                    "quality_valid_index_entry_count": index_result["report"][
                        "quality_valid_index_entry_count"
                    ],
                    "consecutive_failure_count": 0,
                    "outcome_blind_collection_only": True,
                    "settlement_finalizer_started": False,
                    "resolution_provider_called": False,
                    "training_corpus_export_attempted": False,
                    "labels_outcomes_or_pnl_opened": False,
                    **SAFETY,
                },
            )
            batch_sequence += 1
        except Exception as exc:
            consecutive_failures += 1
            _write_state(
                service_state_path,
                {
                    "status": "collection_batch_failed_fail_closed",
                    "failed_batch_sequence": batch_sequence,
                    "failed_batch_id": batch_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "consecutive_failure_count": consecutive_failures,
                    "collector_git_commit": collector_commit,
                    "protocol_path": str(protocol_path),
                    "protocol_sha256": protocol_sha256.lower(),
                    "batch_round_count": batch_round_count,
                    "outcome_blind_collection_only": True,
                    "settlement_finalizer_started": False,
                    "resolution_provider_called": False,
                    "training_corpus_export_attempted": False,
                    "labels_outcomes_or_pnl_opened": False,
                    **SAFETY,
                },
            )
            if consecutive_failures >= max_consecutive_failures:
                raise
            batch_sequence += 1
            time.sleep(failure_backoff_seconds)
    state = json.loads(service_state_path.read_text(encoding="utf-8"))
    state["status"] = "bounded_collection_smoke_completed"
    state["bounded_completed_batch_count"] = completed_batches
    _write_state(service_state_path, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--batch-round-count", type=int, default=12)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Zero runs continuously; positive values provide bounded smoke mode.",
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--failure-backoff-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    state = run_service(
        service_root=args.service_root,
        protocol_path=args.protocol,
        protocol_sha256=args.protocol_sha256,
        batch_round_count=args.batch_round_count,
        max_batches=args.max_batches,
        max_consecutive_failures=args.max_consecutive_failures,
        failure_backoff_seconds=args.failure_backoff_seconds,
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def _load_state(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
