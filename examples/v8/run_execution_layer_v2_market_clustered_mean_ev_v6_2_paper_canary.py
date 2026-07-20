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
from collections import Counter
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
    normalize_resolution_for_settlement,
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
BTC_UPDOWN_5M_HORIZON_MS = 300_000
TERMINAL_COMPLETION_FILENAME = "v6_2_paper_canary_terminal_completion.json"
SETTLEMENT_RECONCILIATION_MANIFEST_FILENAME = (
    "v6_2_paper_settlement_reconciliation_manifest.json"
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
    terminal_result = _load_terminal_canary_result(
        run_id=run_id,
        output_root=output_root,
    )
    if terminal_result is not None:
        return terminal_result
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
                    normalization = _capture_resolution_normalization(capture_dir)
                    status = {
                        "capture_run_id": capture_dir.name,
                        "capture_run_dir": str(capture_dir),
                        "finalized_corpus_dir": (
                            None if result.corpus_dir is None else str(result.corpus_dir)
                        ),
                        "finalization_status": result.report["finalization_status"],
                        "pending_resolution": (
                            normalization["normalized_resolution_count"] == 0
                        ),
                        "raw_resolution_count": result.report["raw_resolution_count"],
                        **normalization,
                        "raw_resolution_artifact": (
                            _descriptor(
                                capture_dir
                                / "raw"
                                / "raw_polymarket_resolutions.jsonl"
                            )
                            if (
                                capture_dir
                                / "raw"
                                / "raw_polymarket_resolutions.jsonl"
                            ).is_file()
                            else None
                        ),
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
                        "normalized_resolution_count": 0,
                        "derived_reference_price_resolution_count": 0,
                        "explicit_resolution_count": 0,
                        "resolution_normalization_reason_codes": [
                            "settlement_provider_error_fail_closed"
                        ],
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
    previous_scheduled_round_start_ts: int | None = None
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
            batch_id = f"{run_id}-capture"
            collector_stop = threading.Event()
            collector_done = threading.Event()
            collector_result: dict[str, Any] = {}

            def continuous_collector_worker() -> None:
                try:
                    collector_result["summary"] = (
                        run_polymarket_async_round_collector_cli(
                            batch_id=batch_id,
                            output_dir=capture_root,
                            round_count=round_count,
                            market_family="btc_updown_5m",
                            public_provider_timeout_seconds=(
                                public_provider_timeout_seconds
                            ),
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
                            settlement_poll_interval_seconds=(
                                settlement_poll_interval_seconds
                            ),
                            settlement_grace_seconds=0.0,
                            max_round_start_lag_seconds=30.0,
                            outcome_blind_collection_only=True,
                            overwrite_existing=overwrite_existing,
                            external_stop_event=collector_stop,
                        )
                    )
                except BaseException as exc:  # noqa: BLE001
                    collector_result["error"] = exc
                finally:
                    collector_done.set()

            collector_thread = threading.Thread(
                target=continuous_collector_worker,
                name=f"{run_id}-continuous-round-collector",
                daemon=True,
            )
            collector_thread.start()
            progress_path = capture_root / batch_id / "batch_progress.json"
            processed_capture_ids: set[str] = set()
            processed_error_ids: set[str] = set()
            while True:
                progress = _load_json(progress_path) if progress_path.is_file() else {}
                if collector_done.is_set() and collector_result.get("summary"):
                    progress = dict(collector_result["summary"])
                new_work = False
                for capture_row in sorted(
                    progress.get("captures") or [],
                    key=lambda row: int(row.get("round_index") or 0),
                ):
                    capture_id = str(capture_row.get("run_id") or capture_row.get("run_dir"))
                    if capture_id in processed_capture_ids:
                        continue
                    source = Path(str(capture_row["run_dir"])).resolve()
                    if not (
                        (source / "pending_round_capture_report.json").is_file()
                        and (source / "pending_round_capture_manifest.json").is_file()
                    ):
                        continue
                    frozen = _freeze_capture_for_decision(
                        source=source,
                        frozen_root=frozen_capture_root,
                        overwrite_existing=overwrite_existing,
                    )
                    hard_reasons = classify_capture_hard_failure(frozen)
                    scheduled_round_start_ts = capture_row.get(
                        "scheduled_round_start_ts"
                    )
                    scheduled_round_start_ts = (
                        None
                        if scheduled_round_start_ts is None
                        else int(scheduled_round_start_ts)
                    )
                    if (
                        previous_scheduled_round_start_ts is not None
                        and scheduled_round_start_ts is not None
                        and scheduled_round_start_ts <= previous_scheduled_round_start_ts
                    ):
                        hard_reasons.append("duplicate_or_non_monotonic_round_boundary")
                    if scheduled_round_start_ts is not None:
                        previous_scheduled_round_start_ts = scheduled_round_start_ts
                    capture_dirs.append(frozen)
                    collection_rows.append(
                        {
                            "round_index": int(capture_row.get("round_index") or 0),
                            "batch_id": batch_id,
                            "scheduled_round_start_ts": scheduled_round_start_ts,
                            "source_capture_dirs": [str(source)],
                            "capture_dirs": [str(frozen)],
                            "hard_failure_reason_codes": sorted(set(hard_reasons)),
                            "optional_http_error_count": 0,
                            "public_data_source": "read_only_public_provider",
                        }
                    )
                    processed_capture_ids.add(capture_id)
                    new_work = True
                    consecutive_hard_failures = (
                        consecutive_hard_failures + 1 if hard_reasons else 0
                    )
                    final_result = _run_scoring(
                        run_id=run_id,
                        output_root=output_root,
                        unlock_manifest_path=unlock_manifest_path,
                        expected_unlock_manifest_sha256=(
                            expected_unlock_manifest_sha256
                        ),
                        capture_dirs=capture_dirs,
                        overwrite_existing=True,
                    )
                    # Settlement sees only mutable source rows after the immutable
                    # decision copy is frozen and scored.
                    if source not in settlement_source_dirs:
                        settlement_source_dirs.append(source)
                    if consecutive_hard_failures >= HARD_CAPTURE_FAILURE_LIMIT:
                        provider_fail_fast_stop_triggered = True
                        collector_stop.set()

                for error_row in sorted(
                    progress.get("errors") or [],
                    key=lambda row: int(row.get("round_index") or 0),
                ):
                    error_id = canonical_json_sha256(error_row)
                    if error_id in processed_error_ids:
                        continue
                    processed_error_ids.add(error_id)
                    new_work = True
                    collection_rows.append(
                        {
                            "round_index": int(error_row.get("round_index") or 0),
                            "batch_id": batch_id,
                            "scheduled_round_start_ts": error_row.get(
                                "scheduled_round_start_ts"
                            ),
                            "source_capture_dirs": [],
                            "capture_dirs": [],
                            "hard_failure_reason_codes": ["capture_not_persisted"],
                            "optional_http_error_count": 1,
                            "collector_error": error_row.get("error"),
                            "public_data_source": "read_only_public_provider",
                        }
                    )
                    consecutive_hard_failures += 1
                    if consecutive_hard_failures >= HARD_CAPTURE_FAILURE_LIMIT:
                        provider_fail_fast_stop_triggered = True
                        collector_stop.set()

                if collector_done.is_set() and not new_work:
                    break
                collector_done.wait(0.5)
            collector_thread.join()
            if collector_result.get("error") is not None:
                raise collector_result["error"]

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
    settlement_evaluation = _settlement_evaluation(
        run_id=run_id,
        run_dir=run_dir,
        settlement_source_dirs=settlement_source_dirs,
    )
    settlement_evaluation_rows_path = (
        run_dir / "v6_2_paper_settlement_evaluation_rows.jsonl"
    )
    settlement_evaluation_report_path = (
        run_dir / "v6_2_paper_settlement_evaluation_report.json"
    )
    _write_jsonl(settlement_evaluation_rows_path, settlement_evaluation["rows"])
    _write_json(settlement_evaluation_report_path, settlement_evaluation["report"])
    collection_status_path = run_dir / "v6_2_paper_canary_collection_status.json"
    settlement_status_path = run_dir / "v6_2_paper_canary_async_settlement_status.json"
    collection_status = {
        "schema_version": "bigan-v8-v6-2-paper-canary-collection-status-v1",
        "run_id": run_id,
        "public_data_source": (
            "snapshot_fixture" if snapshot_capture_dirs else "read_only_public_provider"
        ),
        "attempted_round_count": len(collection_rows),
        "continuous_round_collector_enabled": not snapshot_capture_dirs,
        "round_capture_scheduler_decoupled_from_scoring": not snapshot_capture_dirs,
        "round_capture_scheduler_decoupled_from_exit_monitoring": (
            not snapshot_capture_dirs
        ),
        "round_capture_scheduler_decoupled_from_settlement": not snapshot_capture_dirs,
        "round_capture_scheduler_decoupled_from_report_persistence": (
            not snapshot_capture_dirs
        ),
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
            int(row.get("normalized_resolution_count") or 0) > 0
            for row in settlement_rows
        ),
        "unresolved_round_count": sum(
            int(row.get("normalized_resolution_count") or 0) == 0
            for row in settlement_rows
        ),
        "normalized_resolution_count": sum(
            int(row.get("normalized_resolution_count") or 0)
            for row in settlement_rows
        ),
        "derived_reference_price_resolution_count": sum(
            int(row.get("derived_reference_price_resolution_count") or 0)
            for row in settlement_rows
        ),
        "explicit_resolution_count": sum(
            int(row.get("explicit_resolution_count") or 0)
            for row in settlement_rows
        ),
        "resolution_normalization_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in settlement_rows
                    for reason in row.get("resolution_normalization_reason_codes")
                    or []
                ).items()
            )
        ),
        "rows": sorted(settlement_rows, key=lambda row: row["capture_run_id"]),
        **_safety(),
    }
    _write_json(settlement_status_path, settlement_status)
    manifest_path = Path(final_result["manifest_path"])
    manifest = _load_json(manifest_path)
    manifest["collection_status"] = _descriptor(collection_status_path)
    manifest["async_settlement_status"] = _descriptor(settlement_status_path)
    manifest["settlement_evaluation_rows"] = _descriptor(
        settlement_evaluation_rows_path
    )
    manifest["settlement_evaluation_report"] = _descriptor(
        settlement_evaluation_report_path
    )
    manifest.setdefault("artifacts", {}).update(
        {
            "settlement_evaluation_rows": manifest["settlement_evaluation_rows"],
            "settlement_evaluation_report": manifest[
                "settlement_evaluation_report"
            ],
            "collection_status": manifest["collection_status"],
            "async_settlement_status": manifest["async_settlement_status"],
        }
    )
    manifest["settled_position_count"] = settlement_evaluation["report"][
        "settled_position_count"
    ]
    manifest["unresolved_open_position_count"] = settlement_evaluation["report"][
        "unresolved_open_position_count"
    ]
    manifest["sell_before_close_residual_settled_count"] = settlement_evaluation[
        "report"
    ]["sell_before_close_residual_settled_count"]
    manifest["provider_fail_fast_stop_triggered"] = provider_fail_fast_stop_triggered
    manifest["continuous_round_collector_enabled"] = (
        collection_status["continuous_round_collector_enabled"]
    )
    manifest["round_capture_scheduler_decoupled_from_scoring"] = (
        collection_status["round_capture_scheduler_decoupled_from_scoring"]
    )
    manifest["manifest_id"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    )
    _write_json(manifest_path, manifest)
    terminal_completion = _write_terminal_completion(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        collection_status_path=collection_status_path,
        settlement_status_path=settlement_status_path,
        settlement_evaluation_rows_path=settlement_evaluation_rows_path,
        settlement_evaluation_report_path=settlement_evaluation_report_path,
    )
    return {
        **final_result,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "collection_status": collection_status,
        "settlement_status": settlement_status,
        "settlement_evaluation": settlement_evaluation,
        "terminal_completion": terminal_completion,
        "terminal_idempotent_replay": False,
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


def _settlement_evaluation(
    *,
    run_id: str,
    run_dir: Path,
    settlement_source_dirs: list[Path],
    positions_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate frozen paper positions after asynchronous official resolution only."""

    normalized = _load_normalized_settlement_resolutions(settlement_source_dirs)
    resolutions = normalized["resolutions"]
    resolution_descriptors = normalized["resolution_descriptors"]

    resolved_positions_path = positions_path or (
        run_dir / "v6_2_paper_positions.json"
    )
    positions = _load_json(resolved_positions_path)
    rows = []
    for position in positions.get("positions") or []:
        market_id = str(position["market_id"])
        resolution = resolutions.get(market_id)
        status = str(position["status"])
        side = str(position["selected_side"])
        size = float(position["entry_contract_size"])
        entry_price = float(position["entry_price"])
        realized_trade_pnl = position.get("realized_trade_pnl")
        reason_codes = []
        settlement_pnl = None
        total_pnl = (
            None if realized_trade_pnl is None else float(realized_trade_pnl)
        )
        if status == "closed":
            reason_codes.append("position_closed_before_settlement")
        elif resolution is None:
            reason_codes.append("official_settlement_unresolved")
        else:
            payout_value = resolution.get(f"payout_{side.lower()}")
            payout = (
                float(payout_value)
                if payout_value is not None
                else float(str(resolution["resolved_outcome"]).upper() == side)
            )
            settlement_pnl = size * (payout - entry_price)
            total_pnl = settlement_pnl
            reason_codes.append("official_read_only_settlement_applied_to_open_position")
        row = {
            "settlement_evaluation_row_id": canonical_json_sha256(
                {"run_id": run_id, "position_id": position["position_id"]}
            ),
            "run_id": run_id,
            "position_id": position["position_id"],
            "market_id": market_id,
            "selected_side": side,
            "entry_action": position["entry_action"],
            "intended_exit_policy": position["intended_exit_policy"],
            "position_status_before_settlement": status,
            "entry_contract_size": size,
            "entry_price": entry_price,
            "exit_price": position.get("exit_price"),
            "resolved_outcome": (
                None if resolution is None else resolution["resolved_outcome"]
            ),
            "resolved_outcome_source": (
                None if resolution is None else resolution["resolved_outcome_source"]
            ),
            "resolution_normalization_reason_codes": (
                []
                if resolution is None
                else resolution["resolution_normalization_reason_codes"]
            ),
            "settlement_pnl": settlement_pnl,
            "realized_trade_pnl": realized_trade_pnl,
            "total_paper_pnl": total_pnl,
            "sell_before_close_settlement_residual": (
                status == "open"
                and position["intended_exit_policy"] == "sell_before_close"
            ),
            "settlement_resolution_reason_codes": reason_codes,
            "settlement_used_for_decision": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "v8_execution_handoff_allowed": False,
        }
        row["settlement_evaluation_row_sha256"] = canonical_json_sha256(row)
        rows.append(row)

    unresolved_open = [
        row
        for row in rows
        if row["position_status_before_settlement"] == "open"
        and row["resolved_outcome"] is None
    ]
    pnl_reconciled = [row for row in rows if row["total_paper_pnl"] is not None]
    officially_settled_open = [
        row
        for row in rows
        if row["position_status_before_settlement"] == "open"
        and row["resolved_outcome"] is not None
    ]
    report = {
        "schema_version": "bigan-v8-v6-2-paper-canary-settlement-evaluation-v1",
        "run_id": run_id,
        "official_resolution_artifacts": resolution_descriptors,
        "position_count": len(rows),
        "settled_position_count": len(officially_settled_open),
        "pnl_reconciled_position_count": len(pnl_reconciled),
        "unresolved_open_position_count": len(unresolved_open),
        "closed_before_settlement_count": sum(
            row["position_status_before_settlement"] == "closed" for row in rows
        ),
        "sell_before_close_residual_settled_count": sum(
            row["sell_before_close_settlement_residual"]
            and row["resolved_outcome"] is not None
            for row in rows
        ),
        "normalized_resolution_count": len(resolutions),
        "explicit_resolution_count": normalized["explicit_resolution_count"],
        "derived_reference_price_resolution_count": normalized[
            "derived_reference_price_resolution_count"
        ],
        "invalid_resolution_row_count": normalized["invalid_resolution_row_count"],
        "resolution_normalization_reason_distribution": normalized[
            "resolution_normalization_reason_distribution"
        ],
        "total_paper_pnl": sum(
            float(row["total_paper_pnl"]) for row in pnl_reconciled
        ),
        "settlement_used_for_decision": False,
        "settlement_evaluation_happened_after_decision_freeze": True,
        "unresolved_positions_remain_unresolved": True,
        "paper_results_are_promotion_evidence": False,
        **_safety(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return {"rows": rows, "report": report}


def _capture_resolution_normalization(capture_dir: Path) -> dict[str, Any]:
    normalized = _load_normalized_settlement_resolutions([capture_dir])
    return {
        "normalized_resolution_count": len(normalized["resolutions"]),
        "derived_reference_price_resolution_count": normalized[
            "derived_reference_price_resolution_count"
        ],
        "explicit_resolution_count": normalized["explicit_resolution_count"],
        "invalid_resolution_row_count": normalized["invalid_resolution_row_count"],
        "resolution_normalization_reason_codes": sorted(
            normalized["resolution_normalization_reason_distribution"]
        ),
    }


def _load_normalized_settlement_resolutions(
    settlement_source_dirs: list[Path],
) -> dict[str, Any]:
    resolutions: dict[str, dict[str, Any]] = {}
    resolution_descriptors: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    invalid_count = 0
    explicit_count = 0
    derived_count = 0
    for source in sorted(set(settlement_source_dirs)):
        market_path = source / "raw" / "raw_polymarket_markets.jsonl"
        resolution_path = source / "raw" / "raw_polymarket_resolutions.jsonl"
        if not resolution_path.is_file() or resolution_path.stat().st_size == 0:
            reason_counts["official_resolution_artifact_missing"] += 1
            continue
        resolution_descriptors.append(_descriptor(resolution_path))
        markets = {
            str(row.get("market_id") or ""): row
            for row in (_load_jsonl(market_path) if market_path.is_file() else [])
            if str(row.get("market_id") or "")
        }
        for resolution in _load_jsonl(resolution_path):
            market_id = str(resolution.get("market_id") or "")
            market = markets.get(market_id)
            if market is None:
                invalid_count += 1
                reason_counts["resolution_market_metadata_missing"] += 1
                continue
            normalized, reasons = normalize_resolution_for_settlement(
                market=market,
                resolution=resolution,
            )
            if normalized is None:
                invalid_count += 1
                reason_counts.update(reasons or ["resolution_normalization_failed"])
                continue
            if market_id in resolutions:
                invalid_count += 1
                reason_counts["duplicate_normalized_market_resolution"] += 1
                resolutions.pop(market_id, None)
                continue
            source_type = str(normalized["resolved_outcome_source"])
            if source_type == "official_explicit_resolution":
                explicit_count += 1
            else:
                derived_count += 1
            reason_counts.update(normalized["resolution_normalization_reason_codes"])
            resolutions[market_id] = normalized
    return {
        "resolutions": resolutions,
        "resolution_descriptors": resolution_descriptors,
        "explicit_resolution_count": explicit_count,
        "derived_reference_price_resolution_count": derived_count,
        "invalid_resolution_row_count": invalid_count,
        "resolution_normalization_reason_distribution": dict(
            sorted(reason_counts.items())
        ),
    }


def reconcile_v6_2_paper_canary_settlement(
    *,
    source_run_dir: Path | str,
    run_id: str,
    output_dir: Path | str,
    unlock_manifest_path: Path | str = PINNED_APPROVED_UNLOCK_PATH,
    expected_unlock_manifest_sha256: str = PINNED_APPROVED_UNLOCK_SHA256,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Reconcile frozen paper positions without rescoring or mutating source evidence."""

    validate_v6_2_paper_candidate_unlock(
        unlock_manifest_path,
        expected_unlock_manifest_sha256,
    )
    source = Path(source_run_dir).expanduser().resolve()
    source_manifest_path = source / "v6_2_paper_canary_manifest.json"
    source_runtime_path = source / "v6_2_paper_canary_runtime_report.json"
    source_positions_path = source / "v6_2_paper_positions.json"
    source_settlement_status_path = (
        source / "v6_2_paper_canary_async_settlement_status.json"
    )
    for path in (
        source_manifest_path,
        source_runtime_path,
        source_positions_path,
        source_settlement_status_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required reconciliation source missing: {path}")
    source_manifest = _load_json(source_manifest_path)
    source_runtime = _load_json(source_runtime_path)
    _validate_reconciliation_source(source_manifest, source_runtime)
    source_settlement_status = _load_json(source_settlement_status_path)
    source_dirs = [
        Path(str(row["capture_run_dir"])).expanduser().resolve()
        for row in source_settlement_status.get("rows") or []
    ]
    if not source_dirs:
        raise ValueError("reconciliation source has no settlement capture directories")

    output_root = Path(output_dir).expanduser().resolve()
    run_dir = output_root / run_id
    if run_dir.exists():
        if not overwrite_existing:
            raise FileExistsError(f"reconciliation run already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    evaluation = _settlement_evaluation(
        run_id=run_id,
        run_dir=run_dir,
        settlement_source_dirs=source_dirs,
        positions_path=source_positions_path,
    )
    for row in evaluation["rows"]:
        row["source_run_id"] = source.name
        row["source_decision_artifacts_mutated"] = False
        row["settlement_evaluation_row_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in row.items()
                if key != "settlement_evaluation_row_sha256"
            }
        )
    evaluation["report"].update(
        {
            "source_run_id": source.name,
            "source_run_manifest": _descriptor(source_manifest_path),
            "source_runtime_report": _descriptor(source_runtime_path),
            "source_positions": _descriptor(source_positions_path),
            "source_decision_artifacts_mutated": False,
            "reconciliation_mode": "settlement_only_post_freeze",
        }
    )
    evaluation["report"]["report_id"] = canonical_json_sha256(
        {
            key: value
            for key, value in evaluation["report"].items()
            if key != "report_id"
        }
    )
    rows_path = run_dir / "v6_2_paper_settlement_evaluation_rows.jsonl"
    report_path = run_dir / "v6_2_paper_settlement_evaluation_report.json"
    _write_jsonl(rows_path, evaluation["rows"])
    _write_json(report_path, evaluation["report"])
    manifest = {
        "schema_version": "bigan-v8-v6-2-paper-settlement-reconciliation-v1",
        "run_id": run_id,
        "source_run_id": source.name,
        "source_run_manifest": _descriptor(source_manifest_path),
        "source_runtime_report": _descriptor(source_runtime_path),
        "source_positions": _descriptor(source_positions_path),
        "settlement_evaluation_rows": _descriptor(rows_path),
        "settlement_evaluation_report": _descriptor(report_path),
        "source_decision_artifacts_mutated": False,
        "settlement_only_reconciliation": True,
        "unresolved_open_position_count": evaluation["report"][
            "unresolved_open_position_count"
        ],
        "total_paper_pnl": evaluation["report"]["total_paper_pnl"],
        "paper_results_are_promotion_evidence": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        **_safety(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / SETTLEMENT_RECONCILIATION_MANIFEST_FILENAME
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "settlement_evaluation": evaluation,
    }


def _validate_reconciliation_source(
    manifest: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    for payload_name, payload in (("manifest", manifest), ("runtime", runtime)):
        for field, expected in _safety().items():
            if payload.get(field) != expected:
                raise ValueError(
                    f"reconciliation source {payload_name} safety mismatch: {field}"
                )
    if runtime.get("decision_target_outcome_or_pnl_accessed") is not False:
        raise ValueError("reconciliation source used target/outcome/PnL for decisions")
    if runtime.get("source_or_o_score_mutated") is not False:
        raise ValueError("reconciliation source mutated source/O scores")
    if runtime.get("threshold_cost_sizing_or_guard_mutated") is not False:
        raise ValueError("reconciliation source mutated frozen execution policy")


def _wait_until_epoch_ms(
    target_ts: int,
    *,
    now_fn=time.time,
    sleep_fn=time.sleep,
) -> None:
    """Keep one-round collector invocations on strictly increasing 5m boundaries."""

    while True:
        remaining = target_ts / 1000.0 - now_fn()
        if remaining <= 0.0:
            return
        sleep_fn(remaining)


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


def _write_terminal_completion(
    *,
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    collection_status_path: Path,
    settlement_status_path: Path,
    settlement_evaluation_rows_path: Path,
    settlement_evaluation_report_path: Path,
) -> dict[str, Any]:
    runtime_report_path = run_dir / "v6_2_paper_canary_runtime_report.json"
    runtime_safety_path = run_dir / "v6_2_paper_canary_runtime_safety_report.json"
    payload = {
        "schema_version": "bigan-v8-v6-2-paper-canary-terminal-completion-v1",
        "run_id": run_id,
        "terminal_completion": True,
        "terminal_rerun_behavior": "return_verified_artifacts_without_rescoring",
        "manifest": _descriptor(manifest_path),
        "runtime_report": _descriptor(runtime_report_path),
        "runtime_safety_report": _descriptor(runtime_safety_path),
        "collection_status": _descriptor(collection_status_path),
        "async_settlement_status": _descriptor(settlement_status_path),
        "settlement_evaluation_rows": _descriptor(
            settlement_evaluation_rows_path
        ),
        "settlement_evaluation_report": _descriptor(
            settlement_evaluation_report_path
        ),
        **_safety(),
    }
    payload["terminal_completion_id"] = canonical_json_sha256(payload)
    _write_json(run_dir / TERMINAL_COMPLETION_FILENAME, payload)
    return payload


def _load_terminal_canary_result(
    *,
    run_id: str,
    output_root: Path,
) -> dict[str, Any] | None:
    run_dir = output_root / run_id
    completion_path = run_dir / TERMINAL_COMPLETION_FILENAME
    if not completion_path.is_file():
        return None
    completion = _load_json(completion_path)
    if completion.get("run_id") != run_id or completion.get("terminal_completion") is not True:
        raise ValueError("terminal completion marker identity is invalid")
    for field, expected in _safety().items():
        if completion.get(field) != expected:
            raise ValueError(f"terminal completion safety mismatch: {field}")

    required = {
        "manifest",
        "runtime_report",
        "runtime_safety_report",
        "collection_status",
        "async_settlement_status",
        "settlement_evaluation_rows",
        "settlement_evaluation_report",
    }
    verified_paths = {
        name: _verify_terminal_descriptor(
            completion[name],
            expected_parent=run_dir,
            field_name=name,
        )
        for name in sorted(required)
    }
    manifest = _load_json(verified_paths["manifest"])
    runtime_report = _load_json(verified_paths["runtime_report"])
    collection_status = _load_json(verified_paths["collection_status"])
    settlement_status = _load_json(verified_paths["async_settlement_status"])
    settlement_report = _load_json(verified_paths["settlement_evaluation_report"])
    settlement_rows = _load_jsonl(verified_paths["settlement_evaluation_rows"])
    return {
        "run_dir": run_dir,
        "report": runtime_report,
        "report_path": verified_paths["runtime_report"],
        "manifest_path": verified_paths["manifest"],
        "manifest": manifest,
        "manifest_sha256": _sha256(verified_paths["manifest"]),
        "collection_status": collection_status,
        "settlement_status": settlement_status,
        "settlement_evaluation": {
            "rows": settlement_rows,
            "report": settlement_report,
        },
        "terminal_completion": completion,
        "terminal_idempotent_replay": True,
    }


def _verify_terminal_descriptor(
    descriptor: dict[str, Any],
    *,
    expected_parent: Path,
    field_name: str,
) -> Path:
    path = Path(str(descriptor.get("path") or "")).expanduser().resolve()
    try:
        path.relative_to(expected_parent.resolve())
    except ValueError as exc:
        raise ValueError(f"terminal descriptor escapes run dir: {field_name}") from exc
    if not path.is_file() or _sha256(path) != descriptor.get("sha256"):
        raise ValueError(f"terminal descriptor hash mismatch: {field_name}")
    return path


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
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
    parser.add_argument("--reconcile-existing-run-dir")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.reconcile_existing_run_dir:
        result = reconcile_v6_2_paper_canary_settlement(
            source_run_dir=args.reconcile_existing_run_dir,
            run_id=args.run_id,
            output_dir=args.output_dir,
            unlock_manifest_path=args.unlock_manifest,
            expected_unlock_manifest_sha256=args.unlock_manifest_sha256,
            overwrite_existing=args.overwrite_existing,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(result["run_dir"]),
                    "manifest_sha256": result["manifest_sha256"],
                    "settlement_evaluation_report": result[
                        "settlement_evaluation"
                    ]["report"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
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
                "settlement_evaluation_report": result["settlement_evaluation"][
                    "report"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
