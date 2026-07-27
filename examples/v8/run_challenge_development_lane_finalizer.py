"""Finalize closed 15m development captures in a target-isolated process."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
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
    append_finalized_development_rows,
    atomic_write_json,
    load_jsonl,
    sha256_file,
    validate_development_lane_protocol,
)
from examples.v8.run_polymarket_async_round_collector import (  # noqa: E402
    run_polymarket_async_finalizer_cli,
)

DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/challenge_model_development_lane_15m_protocol.json"
)


def run_service(
    *,
    service_root: Path | str,
    protocol_path: Path | str,
    expected_protocol_sha256: str,
    poll_seconds: float,
    run_once: bool,
) -> dict:
    root = Path(service_root).expanduser().resolve()
    protocol_path = Path(protocol_path).resolve()
    if sha256_file(protocol_path) != expected_protocol_sha256.lower():
        raise ValueError("development lane protocol SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_development_lane_protocol(protocol, repo_root=ROOT)
    lock_path = root / "development_lane_finalizer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("development lane finalizer is already running") from exc
        return _run_locked(
            root=root,
            protocol_sha256=expected_protocol_sha256.lower(),
            poll_seconds=poll_seconds,
            run_once=run_once,
        )


def _run_locked(
    *,
    root: Path,
    protocol_sha256: str,
    poll_seconds: float,
    run_once: bool,
) -> dict:
    capture_index_path = root / "outcome_blind_capture_batch_index.jsonl"
    finalized_index_path = root / "finalized_development_corpus_index.jsonl"
    state_path = root / "finalizer_state.json"
    while True:
        capture_batches = load_jsonl(capture_index_path)
        finalized = load_jsonl(finalized_index_path)
        attempted_batches = 0
        exported_rows = 0
        pending_rows = 0
        errors = 0
        for batch in capture_batches:
            batch_id = str(batch["batch_id"])
            summary = run_polymarket_async_finalizer_cli(
                batch_id=batch_id,
                output_dir=root / "captures",
                settlement_poll_interval_seconds=15.0,
                settlement_grace_seconds=0.0,
                training_corpus_root=root / "development_corpus",
                public_provider_http_timeout_seconds=5.0,
                overwrite_existing=False,
            )
            attempted_batches += 1
            summary_path = Path(str(summary["finalizer_summary_path"])).resolve()
            appended = append_finalized_development_rows(
                index_path=finalized_index_path,
                finalizer_summary=summary,
                finalizer_summary_path=summary_path,
                protocol_sha256=protocol_sha256,
            )
            exported_rows += len(appended)
            pending_rows += int(summary.get("pending_resolution_count") or 0)
            pending_rows += int(summary.get("pending_feature_enrichment_count") or 0)
            errors += int(summary.get("error_count") or 0)
        finalized = load_jsonl(finalized_index_path)
        state = {
            "schema_version": "bigan-challenge-model-development-lane-finalizer-state-v1",
            "status": "running_post_close_development_finalizer",
            "updated_at": datetime.now(UTC).isoformat(),
            "finalizer_pid": os.getpid(),
            "protocol_sha256": protocol_sha256,
            "capture_batch_count": len(capture_batches),
            "batches_scanned_this_cycle": attempted_batches,
            "new_exported_corpus_count": exported_rows,
            "finalized_development_corpus_count": len(finalized),
            "pending_or_enrichment_count": pending_rows,
            "error_count": errors,
            "outcomes_opened_only_in_separate_post_close_process": True,
            "outcomes_labels_or_pnl_fed_to_capture_control": False,
            "development_only_forever": True,
            "promotion_evidence_eligible": False,
            "safety": dict(SAFETY),
        }
        atomic_write_json(state_path, state)
        _write_daily(root, state)
        if run_once:
            return state
        time.sleep(poll_seconds)


def _write_daily(root: Path, state: dict) -> None:
    date_utc = datetime.now(UTC).date().isoformat()
    payload = {
        "schema_version": "bigan-challenge-model-development-lane-daily-finalization-summary-v1",
        "date_utc": date_utc,
        **state,
    }
    atomic_write_json(root / "daily_finalization_summaries" / f"{date_utc}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()
    state = run_service(
        service_root=args.service_root,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.protocol_sha256,
        poll_seconds=args.poll_seconds,
        run_once=args.run_once,
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
