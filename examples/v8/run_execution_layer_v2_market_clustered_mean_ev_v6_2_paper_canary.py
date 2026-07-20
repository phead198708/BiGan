"""Run the approved v6.2 bounded local-paper canary on read-only public data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket import (  # noqa: E402
    PolymarketPublicHTTPRealCorpusProvider,
    finalize_polymarket_pending_round,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256  # noqa: E402
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_paper_canary import (  # noqa: E402
    HARD_CAPTURE_FAILURE_LIMIT,
    MarketClusteredMeanEVV62PaperCanaryConfig,
    classify_capture_hard_failure,
    run_market_clustered_mean_ev_v6_2_paper_canary,
    validate_v6_2_paper_candidate_unlock,
)
from examples.v8.run_polymarket_async_round_collector import (  # noqa: E402
    run_polymarket_async_round_collector_cli,
)

PINNED_APPROVED_UNLOCK_PATH = ROOT / (
    "examples/v8/polymarket_runs/"
    "market-clustered-mean-ev-v6-2-paper-candidate-approved-20260720T124500Z/"
    "v6_2_paper_candidate_unlock_manifest.json"
)
PINNED_APPROVED_UNLOCK_SHA256 = (
    "0e08825810a7310fc7274dbd51505761696410d776ba0570a6eb0ec05a718b02"
)


def run_v6_2_paper_canary_cli(
    *,
    run_id: str,
    output_dir: Path | str,
    unlock_manifest_path: Path | str = PINNED_APPROVED_UNLOCK_PATH,
    expected_unlock_manifest_sha256: str = PINNED_APPROVED_UNLOCK_SHA256,
    round_count: int = 12,
    snapshot_capture_dirs: tuple[Path | str, ...] = (),
    public_provider_timeout_seconds: float = 330.0,
    public_provider_http_timeout_seconds: float = 15.0,
    orderbook_snapshot_interval_seconds: float = 1.0,
    settlement_poll_interval_seconds: float = 15.0,
    settlement_grace_seconds: float = 1_200.0,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Collect or load captures, score causally, and persist paper-only evidence."""

    if round_count <= 0:
        raise ValueError("round_count must be positive")
    if not snapshot_capture_dirs and round_count != 12:
        raise ValueError("real provider canary must use the approved 12-round bound")
    if settlement_poll_interval_seconds <= 0 or settlement_grace_seconds < 0:
        raise ValueError("invalid settlement polling configuration")

    # The unlock is verified before collector/provider construction.
    validate_v6_2_paper_candidate_unlock(
        unlock_manifest_path,
        expected_unlock_manifest_sha256,
    )
    output_root = Path(output_dir).expanduser().resolve()
    capture_root = output_root / f"{run_id}-captures"
    frozen_capture_root = capture_root / "frozen_decision_inputs"
    settlement_root = output_root / f"{run_id}-settled-corpus"
    capture_dirs: list[Path] = [
        Path(value).expanduser().resolve() for value in snapshot_capture_dirs
    ]
    settlement_source_dirs: list[Path] = []
    collection_rows: list[dict[str, Any]] = []
    settlement_rows: list[dict[str, Any]] = []
    settlement_lock = threading.Lock()
    settlement_stop = threading.Event()

    def settlement_worker() -> None:
        while not settlement_stop.is_set():
            for capture_dir in tuple(settlement_source_dirs):
                previous = next(
                    (
                        row
                        for row in settlement_rows
                        if row["capture_run_id"] == capture_dir.name
                    ),
                    None,
                )
                if previous and previous.get("finalization_status") == "exported":
                    continue
                try:
                    provider = PolymarketPublicHTTPRealCorpusProvider(
                        max_markets=1,
                        timeout_seconds=15.0,
                        http_timeout_seconds=public_provider_http_timeout_seconds,
                    )
                    result = finalize_polymarket_pending_round(
                        capture_dir,
                        public_provider=provider,
                        destination_root=settlement_root,
                        overwrite_existing=True,
                    )
                    status = {
                        "capture_run_id": capture_dir.name,
                        "capture_run_dir": str(capture_dir),
                        "finalization_status": result.report["finalization_status"],
                        "pending_resolution": result.report["pending_resolution"],
                        "raw_resolution_count": result.report["raw_resolution_count"],
                        "resolution_provider_called_after_decision_freeze": True,
                        "settlement_used_for_decision": False,
                    }
                except Exception as exc:  # noqa: BLE001
                    status = {
                        "capture_run_id": capture_dir.name,
                        "capture_run_dir": str(capture_dir),
                        "finalization_status": "provider_error_fail_closed",
                        "pending_resolution": True,
                        "raw_resolution_count": 0,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "resolution_provider_called_after_decision_freeze": True,
                        "settlement_used_for_decision": False,
                    }
                with settlement_lock:
                    _upsert(settlement_rows, status)
            settlement_stop.wait(settlement_poll_interval_seconds)

    worker: threading.Thread | None = None
    if not snapshot_capture_dirs:
        worker = threading.Thread(
            target=settlement_worker,
            name=f"{run_id}-async-settlement",
            daemon=True,
        )
        worker.start()

    consecutive_hard_failures = 0
    provider_fail_fast_stop_triggered = False
    final_result: dict[str, Any] | None = None
    try:
        if snapshot_capture_dirs:
            collection_rows = [
                {
                    "round_index": index,
                    "run_dir": str(path),
                    "hard_failure_reason_codes": classify_capture_hard_failure(path),
                    "public_data_source": "snapshot_fixture",
                }
                for index, path in enumerate(capture_dirs, 1)
            ]
        else:
            for index in range(1, round_count + 1):
                batch_id = f"{run_id}-capture-{index:02d}"
                summary = run_polymarket_async_round_collector_cli(
                    batch_id=batch_id,
                    output_dir=capture_root,
                    round_count=1,
                    market_family="btc_updown_5m",
                    public_provider_timeout_seconds=public_provider_timeout_seconds,
                    public_provider_http_timeout_seconds=(
                        public_provider_http_timeout_seconds
                    ),
                    orderbook_snapshot_interval_seconds=(
                        orderbook_snapshot_interval_seconds
                    ),
                    orderbook_ws_initial_complete_book_timeout_seconds=15.0,
                    rest_orderbook_fallback_collection_seconds=(
                        public_provider_timeout_seconds
                    ),
                    settlement_poll_interval_seconds=settlement_poll_interval_seconds,
                    settlement_grace_seconds=0.0,
                    outcome_blind_collection_only=True,
                    overwrite_existing=overwrite_existing,
                )
                attempted_source_dirs = [
                    Path(str(row["run_dir"])).resolve()
                    for row in summary.get("captures") or []
                ]
                if not attempted_source_dirs:
                    attempted_source_dirs = [
                        Path(str(row["run_dir"])).resolve()
                        for row in summary.get("errors") or []
                        if row.get("run_dir")
                    ]
                attempted_dirs = [
                    _freeze_capture_for_decision(
                        source=path,
                        frozen_root=frozen_capture_root,
                        overwrite_existing=overwrite_existing,
                    )
                    for path in attempted_source_dirs
                    if path.exists()
                ]
                hard_reasons = [
                    reason
                    for path in attempted_dirs
                    for reason in classify_capture_hard_failure(path)
                ] or (["capture_not_persisted"] if not attempted_dirs else [])
                capture_dirs.extend(path for path in attempted_dirs if path not in capture_dirs)
                row = {
                    "round_index": index,
                    "batch_id": batch_id,
                    "source_capture_dirs": [
                        str(path) for path in attempted_source_dirs
                    ],
                    "capture_dirs": [str(path) for path in attempted_dirs],
                    "hard_failure_reason_codes": sorted(set(hard_reasons)),
                    "optional_http_error_count": int(summary.get("error_count") or 0),
                    "public_data_source": "read_only_public_provider",
                }
                collection_rows.append(row)
                if hard_reasons:
                    consecutive_hard_failures += 1
                else:
                    consecutive_hard_failures = 0

                final_result = _run_scoring(
                    run_id=run_id,
                    output_root=output_root,
                    unlock_manifest_path=unlock_manifest_path,
                    expected_unlock_manifest_sha256=expected_unlock_manifest_sha256,
                    capture_dirs=capture_dirs,
                    overwrite_existing=True,
                )
                # Only the mutable source capture enters settlement after its
                # immutable decision-input copy has been scored and frozen.
                settlement_source_dirs.extend(
                    path
                    for path in attempted_source_dirs
                    if path.exists() and path not in settlement_source_dirs
                )
                if consecutive_hard_failures >= HARD_CAPTURE_FAILURE_LIMIT:
                    provider_fail_fast_stop_triggered = True
                    break

        final_result = _run_scoring(
            run_id=run_id,
            output_root=output_root,
            unlock_manifest_path=unlock_manifest_path,
            expected_unlock_manifest_sha256=expected_unlock_manifest_sha256,
            capture_dirs=capture_dirs,
            overwrite_existing=True if final_result else overwrite_existing,
        )
    finally:
        if worker is not None:
            deadline = time.monotonic() + settlement_grace_seconds
            while time.monotonic() < deadline:
                with settlement_lock:
                    unresolved = sum(
                        row.get("pending_resolution") is True for row in settlement_rows
                    )
                    observed = len(settlement_rows)
                if observed >= len(settlement_source_dirs) and unresolved == 0:
                    break
                time.sleep(min(settlement_poll_interval_seconds, deadline - time.monotonic()))
            settlement_stop.set()
            worker.join(timeout=max(1.0, settlement_poll_interval_seconds + 1.0))

    if final_result is None:
        raise RuntimeError("v6.2 paper canary produced no runtime result")
    run_dir = Path(final_result["run_dir"])
    collection_status_path = run_dir / "v6_2_paper_canary_collection_status.json"
    settlement_status_path = run_dir / "v6_2_paper_canary_async_settlement_status.json"
    collection_status = {
        "schema_version": "bigan-v8-v6-2-paper-canary-collection-status-v1",
        "run_id": run_id,
        "public_data_source": (
            "snapshot_fixture" if snapshot_capture_dirs else "read_only_public_provider"
        ),
        "attempted_round_count": len(collection_rows),
        "provider_fail_fast_stop_triggered": provider_fail_fast_stop_triggered,
        "consecutive_hard_failure_count_at_stop": consecutive_hard_failures,
        "hard_failure_limit": HARD_CAPTURE_FAILURE_LIMIT,
        "rows": collection_rows,
        **_safety(),
    }
    _write_json(collection_status_path, collection_status)
    settlement_status = {
        "schema_version": "bigan-v8-v6-2-paper-canary-async-settlement-status-v1",
        "run_id": run_id,
        "settlement_mode": "asynchronous_after_each_decision_freeze",
        "settlement_may_block_next_round_collection": False,
        "finalization_attempted_round_count": len(settlement_rows),
        "resolved_round_count": sum(
            row.get("pending_resolution") is False for row in settlement_rows
        ),
        "unresolved_round_count": sum(
            row.get("pending_resolution") is True for row in settlement_rows
        ),
        "rows": sorted(settlement_rows, key=lambda row: row["capture_run_id"]),
        **_safety(),
    }
    _write_json(settlement_status_path, settlement_status)
    manifest_path = Path(final_result["manifest_path"])
    manifest = _load_json(manifest_path)
    manifest["collection_status"] = _descriptor(collection_status_path)
    manifest["async_settlement_status"] = _descriptor(settlement_status_path)
    manifest["provider_fail_fast_stop_triggered"] = provider_fail_fast_stop_triggered
    manifest["manifest_id"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    )
    _write_json(manifest_path, manifest)
    return {
        **final_result,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "collection_status": collection_status,
        "settlement_status": settlement_status,
    }


def _run_scoring(
    *,
    run_id: str,
    output_root: Path,
    unlock_manifest_path: Path | str,
    expected_unlock_manifest_sha256: str,
    capture_dirs: list[Path],
    overwrite_existing: bool,
) -> dict[str, Any]:
    return run_market_clustered_mean_ev_v6_2_paper_canary(
        MarketClusteredMeanEVV62PaperCanaryConfig(
            run_id=run_id,
            output_dir=output_root,
            unlock_manifest_path=unlock_manifest_path,
            expected_unlock_manifest_sha256=expected_unlock_manifest_sha256,
            captured_round_dirs=tuple(capture_dirs),
            runtime_created_ts=int(time.time() * 1000),
            builder_git_commit=_git_commit(),
            overwrite_existing=overwrite_existing,
        )
    )


def _freeze_capture_for_decision(
    *, source: Path, frozen_root: Path, overwrite_existing: bool
) -> Path:
    """Copy outcome-blind capture inputs before any settlement task may mutate them."""

    raw_resolution = source / "raw" / "raw_polymarket_resolutions.jsonl"
    provider_resolution = source / "provider_raw" / "raw_polymarket_resolutions.jsonl"
    for path in (raw_resolution, provider_resolution):
        if path.is_file() and path.stat().st_size > 0:
            raise ValueError("capture contains resolution before decision-input freeze")
    destination = frozen_root / source.name
    if destination.exists():
        if not overwrite_existing:
            raise FileExistsError(f"frozen decision-input capture already exists: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination.resolve()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _upsert(rows: list[dict[str, Any]], value: dict[str, Any]) -> None:
    for index, row in enumerate(rows):
        if row["capture_run_id"] == value["capture_run_id"]:
            rows[index] = value
            return
    rows.append(value)


def _safety() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "live_trading_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--unlock-manifest", default=str(PINNED_APPROVED_UNLOCK_PATH))
    parser.add_argument(
        "--unlock-manifest-sha256", default=PINNED_APPROVED_UNLOCK_SHA256
    )
    parser.add_argument("--round-count", type=int, default=12)
    parser.add_argument("--snapshot-capture-dir", action="append", default=[])
    parser.add_argument("--public-provider-timeout-seconds", type=float, default=330.0)
    parser.add_argument("--public-provider-http-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--orderbook-snapshot-interval-seconds", type=float, default=1.0)
    parser.add_argument("--settlement-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--settlement-grace-seconds", type=float, default=1_200.0)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    result = run_v6_2_paper_canary_cli(
        run_id=args.run_id,
        output_dir=args.output_dir,
        unlock_manifest_path=args.unlock_manifest,
        expected_unlock_manifest_sha256=args.unlock_manifest_sha256,
        round_count=args.round_count,
        snapshot_capture_dirs=tuple(args.snapshot_capture_dir),
        public_provider_timeout_seconds=args.public_provider_timeout_seconds,
        public_provider_http_timeout_seconds=args.public_provider_http_timeout_seconds,
        orderbook_snapshot_interval_seconds=args.orderbook_snapshot_interval_seconds,
        settlement_poll_interval_seconds=args.settlement_poll_interval_seconds,
        settlement_grace_seconds=args.settlement_grace_seconds,
        overwrite_existing=args.overwrite_existing,
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "report": result["report"],
                "manifest_sha256": result["manifest_sha256"],
                "collection_status": result["collection_status"],
                "settlement_status": result["settlement_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
