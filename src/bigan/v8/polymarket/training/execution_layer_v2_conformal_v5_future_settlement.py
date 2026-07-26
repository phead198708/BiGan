"""Reconcile post-freeze official targets and run the #204 side-only PnL gate."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.canonical_payload import (
    DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
    build_canonical_payload_comparison_report,
    compare_canonical_payloads,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.recorder import (
    PendingRoundFinalizationResult,
    PolymarketPublicHTTPRealCorpusProvider,
    finalize_polymarket_pending_round,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    _blocked_safety_fields,
    _descriptor,
    _is_git_sha,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _write_json,
    _write_text,
    build_conformal_v5_side_only_future_pnl_gate,
    validate_conformal_v5_future_evaluation_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    ALLOWED_RAW_FEATURE_FILES,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    SCHEMA_PREFIX as PREDICTION_FREEZE_SCHEMA_PREFIX,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

SCHEMA_PREFIX = "bigan-v8-conformal-v5-strict-future-settlement"
SETTLED_CORPUS_INDEX_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-corpus-index-v2"
CANONICAL_FEATURE_COMPARISON_REPORT_FILENAME = (
    "canonical_feature_payload_comparison_report.json"
)


@dataclass(frozen=True, slots=True)
class ConformalV5FutureSettlementConfig:
    """Pinned post-freeze inputs for one-shot official outcome reconciliation."""

    run_id: str
    output_dir: Path | str
    prediction_freeze_manifest_path: Path | str
    expected_prediction_freeze_manifest_sha256: str
    settled_corpus_index_path: Path | str
    expected_settled_corpus_index_sha256: str
    builder_git_commit: str
    reconciliation_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_prediction_freeze_manifest_sha256",
            "expected_settled_corpus_index_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        if not _is_git_sha(self.builder_git_commit):
            raise ValueError("builder_git_commit must be a Git SHA-1")
        if self.reconciliation_started_ts <= 0:
            raise ValueError("reconciliation_started_ts must be positive")
        for field in (
            "output_dir",
            "prediction_freeze_manifest_path",
            "settled_corpus_index_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


@dataclass(frozen=True, slots=True)
class ConformalV5FutureSettlementCorpusIndexConfig:
    """Pinned one-shot post-close corpus finalization for the frozen future window."""

    run_id: str
    output_dir: Path | str
    prediction_freeze_manifest_path: Path | str
    expected_prediction_freeze_manifest_sha256: str
    builder_git_commit: str
    target_access_started_ts: int
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_prediction_freeze_manifest_sha256,
            name="expected_prediction_freeze_manifest_sha256",
        )
        if not _is_git_sha(self.builder_git_commit):
            raise ValueError("builder_git_commit must be a Git SHA-1")
        if self.target_access_started_ts <= 0:
            raise ValueError("target_access_started_ts must be positive")
        if self.provider_timeout_seconds <= 0 or self.provider_http_timeout_seconds <= 0:
            raise ValueError("provider timeouts must be positive")
        if self.settlement_max_wait_seconds < 0:
            raise ValueError("settlement_max_wait_seconds must be non-negative")
        if self.settlement_poll_interval_seconds <= 0:
            raise ValueError("settlement_poll_interval_seconds must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        for field in ("output_dir", "prediction_freeze_manifest_path"):
            object.__setattr__(self, field, Path(getattr(self, field)))


def build_conformal_v5_future_settled_corpus_index(
    config: ConformalV5FutureSettlementCorpusIndexConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Finalize copied rounds after freeze; never mutate the outcome-blind source."""

    freeze_manifest_path = config.prediction_freeze_manifest_path.resolve()
    _verify_pin(
        freeze_manifest_path,
        config.expected_prediction_freeze_manifest_sha256,
        "prediction freeze manifest",
    )
    freeze_manifest = _load_json(freeze_manifest_path)
    if (
        freeze_manifest.get("schema_version") != f"{PREDICTION_FREEZE_SCHEMA_PREFIX}-manifest-v1"
        or freeze_manifest.get("decision_freeze_written_before_target_access") is not True
        or freeze_manifest.get("future_labels_outcomes_or_pnl_opened") is not False
        or freeze_manifest.get("resolution_artifact_opened") is not False
    ):
        raise ValueError("prediction freeze is not eligible for post-close finalization")
    decision_freeze_descriptor = _verified_descriptor(
        freeze_manifest["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
    )
    decision_freeze = _load_json(Path(decision_freeze_descriptor["path"]))
    decision_freeze_created_ts = int(decision_freeze["decision_freeze_created_ts"])
    if config.target_access_started_ts <= decision_freeze_created_ts:
        raise ValueError("settlement corpus index attempted before decision freeze")
    selected_descriptor = _verified_descriptor(
        freeze_manifest["selected_window_rows"], "selected window rows"
    )
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    action_descriptor = _verified_descriptor(
        freeze_manifest["target_free_five_action_rows"], "target-free five-action rows"
    )
    action_rows = _load_jsonl(Path(action_descriptor["path"]))
    max_market_close_ts = max(int(row["market_close_ts"]) for row in action_rows)
    if config.target_access_started_ts <= max_market_close_ts:
        raise ValueError("settlement corpus index attempted before all markets closed")
    if len(selected_rows) != 220 or len({str(row["market_id"]) for row in selected_rows}) != 220:
        raise ValueError("settlement corpus builder requires the exact frozen 220-market window")
    safety_mismatches = [
        field
        for field, expected in _blocked_safety_fields().items()
        if freeze_manifest.get(field) != expected
    ]
    if safety_mismatches:
        raise ValueError("prediction freeze safety mismatch: " + ", ".join(safety_mismatches))

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-corpus-finalization-start-marker-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "target_access_started_ts": config.target_access_started_ts,
        "decision_freeze_created_ts": decision_freeze_created_ts,
        "max_market_close_ts": max_market_close_ts,
        "prediction_freeze_manifest": _descriptor(freeze_manifest_path),
        "accepted_bet_decision_freeze": decision_freeze_descriptor,
        "selected_window_rows": selected_descriptor,
        "target_access_started_after_decision_freeze": True,
        "all_markets_closed_before_target_access": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "direct_training_corpus_exported": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "future_settled_corpus_finalization_started.json"
    _write_json(marker_path, marker)

    factory = provider_factory or (
        lambda: PolymarketPublicHTTPRealCorpusProvider(
            max_markets=1,
            timeout_seconds=config.provider_timeout_seconds,
            http_timeout_seconds=config.provider_http_timeout_seconds,
            use_rest_orderbooks=False,
        )
    )
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    success_by_market: dict[str, dict[str, Any]] = {}
    failure_by_market: dict[str, dict[str, Any]] = {}
    selected_by_market = {str(row["market_id"]): row for row in selected_rows}
    pending_rows = list(selected_rows)
    retry_market_ids: set[str] = set()
    settlement_attempt_count = 0
    deadline = monotonic_fn() + config.settlement_max_wait_seconds
    while pending_rows:
        settlement_attempt_count += 1
        attempt_results = _finalize_selected_rounds(
            pending_rows,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=settlement_attempt_count,
        )
        retryable_market_ids: set[str] = set()
        for result in attempt_results:
            market_id = str(result["market_id"])
            if result["settled_corpus_ready"]:
                success_by_market[market_id] = result["index_entry"]
                failure_by_market.pop(market_id, None)
                continue
            failure = result["failure"]
            failure_by_market[market_id] = failure
            if _is_retryable_settlement_failure(failure):
                retryable_market_ids.add(market_id)
        if not retryable_market_ids:
            break
        retry_market_ids.update(retryable_market_ids)
        remaining_seconds = deadline - monotonic_fn()
        if remaining_seconds <= 0:
            for market_id in retryable_market_ids:
                failure = failure_by_market[market_id]
                failure["reason_codes"] = sorted(
                    {*failure["reason_codes"], "settlement_resolution_max_wait_elapsed"}
                )
            break
        sleep_fn(min(config.settlement_poll_interval_seconds, remaining_seconds))
        pending_rows = [selected_by_market[market_id] for market_id in sorted(retryable_market_ids)]
    successes = sorted(success_by_market.values(), key=lambda row: str(row["market_id"]))
    failures = sorted(failure_by_market.values(), key=lambda row: str(row["market_id"]))
    complete = len(successes) == len(selected_rows) and not failures
    index_finalized_ts = int(clock_ms_fn())
    if index_finalized_ts < config.target_access_started_ts:
        raise ValueError("index_finalized_ts precedes target access start")
    index_path = run_dir / "conformal_v5_future_settled_corpus_index.json"
    index_payload: dict[str, Any] | None = None
    if complete:
        index_payload = {
            "schema_version": SETTLED_CORPUS_INDEX_SCHEMA_VERSION,
            "run_id": config.run_id,
            "builder_git_commit": config.builder_git_commit,
            "target_access_started_ts": config.target_access_started_ts,
            "index_finalized_ts": index_finalized_ts,
            "decision_freeze_sha256": decision_freeze_descriptor["sha256"],
            "prediction_freeze_manifest": _descriptor(freeze_manifest_path),
            "selected_window_rows": selected_descriptor,
            "entry_count": len(successes),
            "entries": successes,
            "outcomes_used_for_decision_or_selection": False,
            "outcomes_used_for_threshold_or_model_tuning": False,
            "source_outcome_blind_rounds_mutated": False,
            "direct_training_corpus_exported": False,
            **_blocked_safety_fields(),
        }
        index_payload["settled_corpus_index_id"] = canonical_json_sha256(index_payload)
        _write_json(index_path, index_payload)
    reason_distribution: dict[str, int] = {}
    for failure in failures:
        for reason in failure["reason_codes"]:
            reason_distribution[reason] = reason_distribution.get(reason, 0) + 1
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-corpus-index-build-report-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "selected_market_count": len(selected_rows),
        "settled_corpus_ready_market_count": len(successes),
        "unresolved_or_failed_market_count": len(failures),
        "target_access_started_ts": config.target_access_started_ts,
        "index_finalized_ts": index_finalized_ts,
        "settlement_attempt_count": settlement_attempt_count,
        "settlement_retry_market_count": len(retry_market_ids),
        "settlement_max_wait_seconds": config.settlement_max_wait_seconds,
        "settlement_poll_interval_seconds": config.settlement_poll_interval_seconds,
        "unresolved_or_failed_reason_distribution": dict(sorted(reason_distribution.items())),
        "unresolved_or_failed_markets": failures,
        "settled_corpus_index_ready": complete,
        "settled_corpus_index_path": str(index_path) if complete else None,
        "settled_corpus_index_sha256": _sha256_file(index_path) if complete else None,
        "source_outcome_blind_rounds_mutated": False,
        "direct_training_corpus_exported": False,
        "official_read_only_resolution_only": True,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "blocking_reason_codes": [] if complete else ["settled_corpus_window_incomplete"],
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v5_future_settled_corpus_index_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "conformal_v5_future_settled_corpus_index_report.md",
        _settled_index_markdown(report),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-corpus-index-build-manifest-v1",
        "run_id": config.run_id,
        "prediction_freeze_manifest": _descriptor(freeze_manifest_path),
        "accepted_bet_decision_freeze": decision_freeze_descriptor,
        "selected_window_rows": selected_descriptor,
        "finalization_start_marker": _descriptor(marker_path),
        "report": _descriptor(report_path),
        "settled_corpus_index": _descriptor(index_path) if complete else None,
        "settled_corpus_index_ready": complete,
        "target_access_started_ts": config.target_access_started_ts,
        "index_finalized_ts": index_finalized_ts,
        "settlement_attempt_count": settlement_attempt_count,
        "settlement_retry_market_count": len(retry_market_ids),
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "source_outcome_blind_rounds_mutated": False,
        "direct_training_corpus_exported": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v5_future_settled_corpus_index_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "index": index_payload,
        "index_path": index_path if complete else None,
        "index_sha256": _sha256_file(index_path) if complete else None,
    }


def _finalize_selected_rounds(
    selected_rows: list[dict[str, Any]],
    *,
    run_dir: Path,
    provider_factory: Callable[[], Any],
    max_workers: int,
    settlement_attempt: int,
    evaluation_only_frozen_features_by_market: (
        dict[str, list[dict[str, Any]]] | None
    ) = None,
    require_complete_raw_feature_lineage: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _copy_and_finalize_selected_round,
                selected,
                run_dir=run_dir,
                provider_factory=provider_factory,
                settlement_attempt=settlement_attempt,
                evaluation_only_frozen_feature_rows=(
                    evaluation_only_frozen_features_by_market.get(
                        str(selected.get("market_id") or ""), []
                    )
                    if evaluation_only_frozen_features_by_market is not None
                    else None
                ),
                require_complete_raw_feature_lineage=(
                    require_complete_raw_feature_lineage
                ),
            ): selected
            for selected in selected_rows
        }
        for future in as_completed(futures):
            selected = futures[future]
            market_id = str(selected.get("market_id") or "")
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "market_id": market_id,
                        "settled_corpus_ready": False,
                        "failure": {
                            "market_id": market_id,
                            "run_id": str(selected.get("run_id") or ""),
                            "reason_codes": ["settled_corpus_finalization_exception"],
                            "settlement_attempt_count": settlement_attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    }
                )
    return sorted(results, key=lambda row: str(row["market_id"]))


def _is_retryable_settlement_failure(failure: dict[str, Any]) -> bool:
    if failure.get("pending_resolution") is True:
        return True
    return any(
        any(token in str(reason) for token in ("resolution", "provider", "timeout"))
        for reason in failure.get("reason_codes", [])
    )


def _copy_and_finalize_selected_round(
    selected: dict[str, Any],
    *,
    run_dir: Path,
    provider_factory: Callable[[], Any],
    settlement_attempt: int = 1,
    evaluation_only_frozen_feature_rows: list[dict[str, Any]] | None = None,
    require_complete_raw_feature_lineage: bool = False,
) -> dict[str, Any]:
    market_id = str(selected.get("market_id") or "")
    if not market_id:
        raise ValueError("selected settlement row is missing market_id")
    capture_manifest_descriptor = _verified_descriptor(
        selected["pending_round_capture_manifest"], "pending capture manifest"
    )
    source_run_dir = Path(capture_manifest_descriptor["path"]).parent
    source_run_id = source_run_dir.name
    copied_parent = run_dir / "settled_round_copies"
    copied_run_dir = copied_parent / source_run_id
    if not copied_run_dir.exists():
        shutil.copytree(source_run_dir, copied_run_dir)
    copied_capture_manifest_path = copied_run_dir / "pending_round_capture_manifest.json"
    copied_capture_manifest = _load_json(copied_capture_manifest_path)
    copied_config = dict(copied_capture_manifest.get("config") or {})
    if not copied_config or str(copied_config.get("run_id") or "") != source_run_id:
        raise ValueError("copied pending capture config lineage is invalid")
    if copied_capture_manifest.get("post_freeze_settlement_copy") is True:
        if copied_capture_manifest.get("source_pending_capture_manifest") != (
            capture_manifest_descriptor
        ):
            raise ValueError("settlement copy source lineage changed between attempts")
    else:
        copied_config["output_dir"] = str(copied_parent)
        copied_capture_manifest["config"] = copied_config
        copied_capture_manifest["post_freeze_settlement_copy"] = True
        copied_capture_manifest["source_pending_capture_manifest"] = capture_manifest_descriptor
        copied_capture_manifest["source_outcome_blind_round_mutated"] = False
        _write_json(copied_capture_manifest_path, copied_capture_manifest)
    result: PendingRoundFinalizationResult = finalize_polymarket_pending_round(
        copied_run_dir,
        public_provider=provider_factory(),
        destination_root=run_dir / "settled_corpus_quarantine",
        overwrite_existing=True,
    )
    source_lineage_evidence = _verify_selected_source_unchanged(
        selected,
        capture_manifest_descriptor,
        require_complete_raw_feature_lineage=(
            require_complete_raw_feature_lineage
        ),
    )
    report = dict(result.report)
    evaluation_only_corpus_dir = None
    evaluation_only_reason_codes: list[str] = []
    if evaluation_only_frozen_feature_rows is not None:
        evaluation_only_corpus_dir, evaluation_only_reason_codes = (
            _evaluation_only_settled_corpus_if_safe(
                copied_run_dir=copied_run_dir,
                report=report,
                frozen_feature_rows=evaluation_only_frozen_feature_rows,
                source_lineage_evidence=source_lineage_evidence,
                require_complete_raw_feature_lineage=(
                    require_complete_raw_feature_lineage
                ),
            )
        )
    evaluation_only_fallback = evaluation_only_corpus_dir is not None
    if (
        (
            report.get("finalization_status") != "exported"
            and not evaluation_only_fallback
        )
        or report.get("pending_resolution") is not False
        or report.get("resolution_provider_called") is not True
        or report.get("phase2_corpus_built") is not True
        or (result.corpus_dir is None and not evaluation_only_fallback)
    ):
        reasons = sorted(
            {
                *(str(reason) for reason in dict(report.get("reject_reason_counts") or {})),
                *(
                    ["official_resolution_still_pending"]
                    if report.get("pending_resolution") is True
                    else []
                ),
                *(
                    ["phase2_settled_corpus_not_built"]
                    if report.get("phase2_corpus_built") is not True
                    else []
                ),
                *(
                        ["settled_corpus_finalization_blocked"]
                        if report.get("finalization_status") == "blocked_fail_closed"
                        else []
                    ),
                    *evaluation_only_reason_codes,
                }
            )
        failure = {
            "market_id": market_id,
            "settled_corpus_ready": False,
            "failure": {
                "market_id": market_id,
                "run_id": source_run_id,
                "reason_codes": reasons or ["official_resolution_unavailable"],
                "phase2_error": report.get("phase2_error"),
                "finalization_status": report.get("finalization_status"),
                "pending_resolution": report.get("pending_resolution"),
                "settlement_attempt_count": settlement_attempt,
                "copied_round_dir": str(copied_run_dir),
            },
        }
        canonical_report_path = (
            copied_run_dir
            / CANONICAL_FEATURE_COMPARISON_REPORT_FILENAME
        )
        if canonical_report_path.is_file():
            failure["failure"][
                "canonical_feature_payload_comparison_report"
            ] = _descriptor(canonical_report_path)
        return failure
    corpus_dir = (
        evaluation_only_corpus_dir
        if evaluation_only_fallback
        else result.corpus_dir.resolve()
    )
    corpus_manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    feature_rows_path = corpus_dir / "polymarket_feature_rows.jsonl"
    label_rows_path = corpus_dir / "polymarket_label_rows.jsonl"
    resolution_events_path = corpus_dir / "polymarket_resolution_events.jsonl"
    finalization_manifest_path = copied_run_dir / "pending_round_finalization_manifest.json"
    for path in (
        corpus_manifest_path,
        feature_rows_path,
        label_rows_path,
        resolution_events_path,
        finalization_manifest_path,
    ):
        if not path.is_file():
            raise ValueError(f"settled corpus artifact missing: {path}")
    canonical_report_path: Path | None = None
    if evaluation_only_frozen_feature_rows is not None:
        canonical_report_path = (
            copied_run_dir
            / CANONICAL_FEATURE_COMPARISON_REPORT_FILENAME
        )
        if not evaluation_only_fallback:
            canonical_report = _write_canonical_feature_comparison_report(
                copied_run_dir=copied_run_dir,
                corpus_dir=corpus_dir,
                frozen_feature_rows=evaluation_only_frozen_feature_rows,
                source_lineage_evidence=source_lineage_evidence,
                require_complete_raw_feature_lineage=(
                    require_complete_raw_feature_lineage
                ),
                context="settled_export_feature_payload",
            )
            if canonical_report["canonical_comparison_passed"] is not True:
                return {
                    "market_id": market_id,
                    "settled_corpus_ready": False,
                    "failure": {
                        "market_id": market_id,
                        "run_id": source_run_id,
                        "reason_codes": [
                            "settled_feature_canonical_comparison_failed"
                        ],
                        "pending_resolution": False,
                        "settlement_attempt_count": settlement_attempt,
                        "copied_round_dir": str(copied_run_dir),
                        "canonical_feature_payload_comparison_report": (
                            _descriptor(canonical_report_path)
                        ),
                    },
                }
        if not canonical_report_path.is_file():
            raise ValueError(
                "canonical feature comparison report was not materialized"
            )
    index_entry = {
        "market_id": market_id,
        "run_id": source_run_id,
        "scheduled_round_start_ts": int(selected["scheduled_round_start_ts"]),
        "market_close_ts": int(selected["market_end_ts"]),
        "source_pending_capture_manifest": capture_manifest_descriptor,
        "copied_pending_capture_manifest": _descriptor(copied_capture_manifest_path),
        "pending_round_finalization_manifest": _descriptor(finalization_manifest_path),
        "corpus_manifest": _descriptor(corpus_manifest_path),
        "feature_rows": _descriptor(feature_rows_path),
        "label_rows": _descriptor(label_rows_path),
        "resolution_events": _descriptor(resolution_events_path),
        "official_read_only_resolution": True,
        "corpus_built_after_decision_freeze": True,
        "settled_after_market_close": True,
        "source_outcome_blind_round_mutated": False,
        "direct_training_corpus_exported": False,
        "settlement_attempt_count": settlement_attempt,
        "evaluation_only_settlement_fallback": evaluation_only_fallback,
        "evaluation_only_settlement_fallback_reason_codes": (
            ["frozen_feature_equivalent_chainlink_training_gate_block"]
            if evaluation_only_fallback
            else []
        ),
        "canonical_feature_payload_comparison_required": (
            evaluation_only_frozen_feature_rows is not None
        ),
        "canonical_feature_payload_comparison_report": (
            _descriptor(canonical_report_path)
            if canonical_report_path is not None
            else None
        ),
        "direct_training_eligibility_relaxed": False,
    }
    index_entry["entry_sha256"] = canonical_json_sha256(index_entry)
    return {
        "market_id": market_id,
        "settled_corpus_ready": True,
        "index_entry": index_entry,
    }


def _evaluation_only_settled_corpus_if_safe(
    *,
    copied_run_dir: Path,
    report: dict[str, Any],
    frozen_feature_rows: list[dict[str, Any]],
    source_lineage_evidence: dict[str, Any] | None = None,
    require_complete_raw_feature_lineage: bool = False,
) -> tuple[Path | None, list[str]]:
    """Accept a training-blocked corpus only when frozen evaluation inputs match."""

    if report.get("finalization_status") == "exported":
        return None, []
    reasons = []
    allowed_phase2_error = "Chainlink decision-time feature integration failed:"
    if report.get("finalization_status") != "blocked_fail_closed":
        reasons.append("evaluation_only_finalization_status_invalid")
    if report.get("pending_resolution") is not False:
        reasons.append("evaluation_only_resolution_pending")
    if report.get("resolution_provider_called") is not True:
        reasons.append("evaluation_only_resolution_provider_not_called")
    if report.get("phase2_corpus_built") is not True:
        reasons.append("evaluation_only_phase2_corpus_not_built")
    if not str(report.get("phase2_error") or "").startswith(allowed_phase2_error):
        reasons.append("evaluation_only_blocker_not_chainlink_training_integration")
    if report.get("raw_resolution_count") != 1:
        reasons.append("evaluation_only_official_resolution_count_invalid")
    if dict(report.get("reject_reason_counts") or {}):
        reasons.append("evaluation_only_additional_reject_reasons_present")
    chainlink = dict(report.get("chainlink_corpus_evidence") or {})
    allowed_chainlink_reasons = {
        "chainlink_feature_builder_integration_failed",
        "chainlink_feature_builder_integration_still_required",
    }
    if set(chainlink.get("reason_codes") or []) - allowed_chainlink_reasons:
        reasons.append("evaluation_only_chainlink_blocker_not_allowlisted")

    finalization_manifest_path = copied_run_dir / "pending_round_finalization_manifest.json"
    if not finalization_manifest_path.is_file():
        reasons.append("evaluation_only_finalization_manifest_missing")
        return None, sorted(set(reasons))
    finalization_manifest = _load_json(finalization_manifest_path)
    corpus_dir_text = str(finalization_manifest.get("phase2_corpus_dir") or "")
    corpus_dir = Path(corpus_dir_text).resolve() if corpus_dir_text else Path()
    if not corpus_dir_text or not corpus_dir.is_relative_to(copied_run_dir.resolve()):
        reasons.append("evaluation_only_corpus_path_outside_settlement_copy")
        return None, sorted(set(reasons))
    corpus_manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    feature_rows_path = corpus_dir / "polymarket_feature_rows.jsonl"
    label_rows_path = corpus_dir / "polymarket_label_rows.jsonl"
    resolution_events_path = corpus_dir / "polymarket_resolution_events.jsonl"
    for path, reason in (
        (corpus_manifest_path, "evaluation_only_corpus_manifest_missing"),
        (feature_rows_path, "evaluation_only_feature_rows_missing"),
        (label_rows_path, "evaluation_only_label_rows_missing"),
        (resolution_events_path, "evaluation_only_resolution_events_missing"),
    ):
        if not path.is_file():
            reasons.append(reason)
    if reasons:
        return None, sorted(set(reasons))

    corpus_manifest = _load_json(corpus_manifest_path)
    integration = dict(corpus_manifest.get("chainlink_decision_time_feature_integration") or {})
    if corpus_manifest.get("sell_before_close_label_gate_passed") is not True:
        reasons.append("evaluation_only_sell_before_close_label_gate_failed")
    if int(corpus_manifest.get("label_row_count") or 0) <= 0:
        reasons.append("evaluation_only_label_rows_empty")
    if int(integration.get("timestamp_causality_violation_count") or 0) != 0:
        reasons.append("evaluation_only_chainlink_timestamp_causality_violation")
    settled_features = _load_jsonl(feature_rows_path)
    if not frozen_feature_rows:
        reasons.append("evaluation_only_frozen_feature_rows_missing")
    if any(int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0) for row in settled_features):
        reasons.append("evaluation_only_feature_timestamp_causality_violation")
    comparison = _write_canonical_feature_comparison_report(
        copied_run_dir=copied_run_dir,
        corpus_dir=corpus_dir,
        frozen_feature_rows=frozen_feature_rows,
        source_lineage_evidence=source_lineage_evidence,
        require_complete_raw_feature_lineage=(
            require_complete_raw_feature_lineage
        ),
        context="evaluation_only_settlement_fallback_feature_payload",
    )
    if comparison["canonical_comparison_passed"] is not True:
        reasons.append("evaluation_only_frozen_feature_payload_mismatch")
    if sum(1 for line in label_rows_path.read_text(encoding="utf-8").splitlines() if line) <= 0:
        reasons.append("evaluation_only_label_rows_empty")
    if sum(
        1 for line in resolution_events_path.read_text(encoding="utf-8").splitlines() if line
    ) != 1:
        reasons.append("evaluation_only_resolution_event_count_invalid")
    if reasons:
        return None, sorted(set(reasons))
    return corpus_dir, []


def _write_canonical_feature_comparison_report(
    *,
    copied_run_dir: Path,
    corpus_dir: Path,
    frozen_feature_rows: list[dict[str, Any]],
    source_lineage_evidence: dict[str, Any] | None,
    require_complete_raw_feature_lineage: bool,
    context: str,
) -> dict[str, Any]:
    """Persist validator-derived source lineage and canonical feature equality."""

    finalization_manifest_path = (
        copied_run_dir / "pending_round_finalization_manifest.json"
    )
    corpus_manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    feature_rows_path = corpus_dir / "polymarket_feature_rows.jsonl"
    label_rows_path = corpus_dir / "polymarket_label_rows.jsonl"
    resolution_events_path = (
        corpus_dir / "polymarket_resolution_events.jsonl"
    )
    finalization_manifest = _load_json(finalization_manifest_path)
    corpus_manifest = _load_json(corpus_manifest_path)
    settled_features = _load_jsonl(feature_rows_path)
    normalized = dict(
        corpus_manifest.get("normalized_artifact_hashes") or {}
    )
    source_evidence = dict(source_lineage_evidence or {})
    source_checks = dict(source_evidence.get("checks") or {})
    source_raw = {
        str(name): str(dict(descriptor or {}).get("sha256") or "")
        for name, descriptor in dict(
            source_evidence.get("raw_artifacts") or {}
        ).items()
    }
    corpus_raw = {
        str(name): str(digest or "")
        for name, digest in dict(
            corpus_manifest.get("raw_artifact_hashes") or {}
        ).items()
    }
    expected_raw_names = set(ALLOWED_RAW_FEATURE_FILES)
    raw_lineage_matches = all(
        corpus_raw.get(name) == digest
        for name, digest in source_raw.items()
    )
    if require_complete_raw_feature_lineage:
        raw_lineage_matches = (
            expected_raw_names.issubset(source_raw)
            and all(
                corpus_raw.get(name) == source_raw[name]
                for name in expected_raw_names
            )
        )
    lineage_checks = {
        **{
            f"capture_{name}": passed
            for name, passed in source_checks.items()
            if type(passed) is bool
        },
        "source_lineage_evidence_supplied": bool(source_checks),
        "settled_corpus_within_quarantine_copy": corpus_dir.resolve().is_relative_to(
            copied_run_dir.resolve()
        ),
        "finalization_manifest_corpus_path_matches": (
            str(finalization_manifest.get("phase2_corpus_dir") or "")
            == str(corpus_dir.resolve())
        ),
        "finalization_manifest_corpus_hash_matches": (
            str(
                finalization_manifest.get(
                    "phase2_corpus_manifest_sha256"
                )
                or ""
            )
            == _sha256_file(corpus_manifest_path)
        ),
        "normalized_feature_rows_hash_matches": (
            normalized.get("feature_rows")
            == _sha256_file(feature_rows_path)
        ),
        "normalized_label_rows_hash_matches": (
            normalized.get("label_rows") == _sha256_file(label_rows_path)
        ),
        "normalized_resolution_events_hash_matches": (
            normalized.get("resolution_events")
            == _sha256_file(resolution_events_path)
        ),
        "source_raw_artifact_hashes_match_settled_corpus": (
            raw_lineage_matches
        ),
    }
    frozen_payloads = sorted(
        (_feature_payload(row) for row in frozen_feature_rows),
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"])),
    )
    settled_payloads = sorted(
        (_feature_payload(row) for row in settled_features),
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"])),
    )
    comparison = build_canonical_payload_comparison_report(
        frozen_payloads,
        settled_payloads,
        frozen_payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        settled_payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        source_lineage_checks=lineage_checks,
        source_lineage_evidence={
            "capture": source_evidence,
            "settled_corpus_manifest": _descriptor(corpus_manifest_path),
            "settled_feature_rows": _descriptor(feature_rows_path),
            "settled_label_rows": _descriptor(label_rows_path),
            "settled_resolution_events": _descriptor(
                resolution_events_path
            ),
            "frozen_feature_payload_row_count": len(frozen_payloads),
            "settled_feature_payload_row_count": len(settled_payloads),
            "raw_source_artifacts_preserved": True,
            "historical_manifest_rewritten": False,
        },
        context=context,
    )
    _write_json(
        copied_run_dir / CANONICAL_FEATURE_COMPARISON_REPORT_FILENAME,
        comparison,
    )
    return comparison


def _verify_selected_source_unchanged(
    selected: dict[str, Any],
    capture_manifest_descriptor: dict[str, Any],
    *,
    require_complete_raw_feature_lineage: bool = False,
) -> dict[str, Any]:
    _verify_pin(
        Path(capture_manifest_descriptor["path"]),
        capture_manifest_descriptor["sha256"],
        "source pending capture manifest after settlement-copy finalization",
    )
    raw_artifacts = dict(selected.get("raw_artifacts") or {})
    verified_raw_artifacts: dict[str, dict[str, str]] = {}
    for name, descriptor in sorted(raw_artifacts.items()):
        if isinstance(descriptor, dict) and descriptor.get("path") and descriptor.get("sha256"):
            verified_raw_artifacts[name] = _verified_descriptor(
                descriptor,
                f"source raw artifact {name}",
            )
    required_raw_artifacts_complete = set(
        ALLOWED_RAW_FEATURE_FILES
    ).issubset(verified_raw_artifacts)
    return {
        "schema_version": "bigan-v8-canonical-source-lineage-evidence-v1",
        "checks": {
            "source_pending_capture_manifest_hash_verified": True,
            "declared_raw_artifact_hashes_verified": (
                len(verified_raw_artifacts) == len(raw_artifacts)
            ),
            "required_raw_feature_artifacts_complete": (
                required_raw_artifacts_complete
                if require_complete_raw_feature_lineage
                else True
            ),
        },
        "source_pending_capture_manifest": capture_manifest_descriptor,
        "raw_artifacts": verified_raw_artifacts,
    }


def reconcile_conformal_v5_future_settlement(
    config: ConformalV5FutureSettlementConfig,
) -> dict[str, Any]:
    """Open targets only after freeze, join both policies, and evaluate by side."""

    freeze_manifest_path = config.prediction_freeze_manifest_path.resolve()
    _verify_pin(
        freeze_manifest_path,
        config.expected_prediction_freeze_manifest_sha256,
        "prediction freeze manifest",
    )
    freeze_manifest = _load_json(freeze_manifest_path)
    if (
        freeze_manifest.get("schema_version") != f"{PREDICTION_FREEZE_SCHEMA_PREFIX}-manifest-v1"
        or freeze_manifest.get("decision_freeze_written_before_target_access") is not True
        or freeze_manifest.get("future_labels_outcomes_or_pnl_opened") is not False
        or freeze_manifest.get("resolution_artifact_opened") is not False
    ):
        raise ValueError("prediction freeze is not eligible for settlement reconciliation")
    safety_mismatches = [
        field
        for field, expected in _blocked_safety_fields().items()
        if freeze_manifest.get(field) != expected
    ]
    if safety_mismatches:
        raise ValueError("prediction freeze safety mismatch: " + ", ".join(safety_mismatches))

    decision_freeze_descriptor = _verified_descriptor(
        freeze_manifest["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
    )
    decision_freeze = _load_json(Path(decision_freeze_descriptor["path"]))
    if (
        decision_freeze.get("decision_freeze_written_before_target_access") is not True
        or decision_freeze.get("future_labels_outcomes_or_pnl_opened") is not False
        or decision_freeze.get("target_or_outcome_used_for_decision") is not False
    ):
        raise ValueError("accepted-bet decision freeze contract is invalid")
    action_rows_descriptor = _verified_descriptor(
        freeze_manifest["target_free_five_action_rows"], "target-free five-action rows"
    )
    action_rows = _load_jsonl(Path(action_rows_descriptor["path"]))
    max_market_close_ts = max(int(row["market_close_ts"]) for row in action_rows)
    if config.reconciliation_started_ts <= max_market_close_ts:
        raise ValueError("settlement reconciliation attempted before all markets closed")

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-target-access-marker-v1",
        "run_id": config.run_id,
        "reconciliation_started_ts": config.reconciliation_started_ts,
        "prediction_freeze_manifest": _descriptor(freeze_manifest_path),
        "accepted_bet_decision_freeze": decision_freeze_descriptor,
        "max_market_close_ts": max_market_close_ts,
        "all_markets_closed_before_target_access": True,
        "future_outcomes_opened_before_decision_freeze": False,
        "target_access_started_after_decision_freeze": True,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "future_settlement_target_access_started.json"
    _write_json(marker_path, marker)

    settled_index_path = config.settled_corpus_index_path.resolve()
    _verify_pin(
        settled_index_path,
        config.expected_settled_corpus_index_sha256,
        "settled corpus index",
    )
    settled_index = _load_json(settled_index_path)
    selected_descriptor = _verified_descriptor(
        freeze_manifest["selected_window_rows"], "selected window rows"
    )
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    feature_descriptor = _verified_descriptor(
        freeze_manifest["target_free_feature_rows"], "target-free feature rows"
    )
    frozen_features = _load_jsonl(Path(feature_descriptor["path"]))
    index_entries = _validate_settled_corpus_index(
        settled_index,
        expected_decision_freeze_sha256=decision_freeze_descriptor["sha256"],
        decision_freeze_created_ts=int(decision_freeze["decision_freeze_created_ts"]),
        selected_rows=selected_rows,
        reconciliation_started_ts=config.reconciliation_started_ts,
    )
    targets, source_descriptors = _load_and_validate_targets(
        index_entries,
        selected_rows=selected_rows,
        frozen_features=frozen_features,
    )
    target_path = run_dir / "conformal_v5_future_settled_five_action_targets.jsonl"
    _write_jsonl(target_path, targets)

    candidate_replay_descriptor = _verified_descriptor(
        freeze_manifest["candidate_outcome_blind_guard_replay"], "candidate guard replay"
    )
    baseline_replay_descriptor = _verified_descriptor(
        freeze_manifest["matched_baseline_outcome_blind_guard_replay"],
        "matched baseline guard replay",
    )
    candidate_replay = _load_jsonl(Path(candidate_replay_descriptor["path"]))
    baseline_replay = _load_jsonl(Path(baseline_replay_descriptor["path"]))
    targets_by_decision = {(str(row["market_id"]), int(row["decision_ts"])): row for row in targets}
    candidate_evaluation = _join_frozen_replay_targets(
        candidate_replay,
        targets_by_decision=targets_by_decision,
        policy_name="guard_compatible_conformal_net_return_v5",
        decision_freeze_sha256=decision_freeze_descriptor["sha256"],
    )
    baseline_evaluation = _join_frozen_replay_targets(
        baseline_replay,
        targets_by_decision=targets_by_decision,
        policy_name="guard_compatible_direct_net_return_v4",
        decision_freeze_sha256=decision_freeze_descriptor["sha256"],
    )
    candidate_evaluation_path = run_dir / "conformal_v5_future_settled_evaluation_rows.jsonl"
    baseline_evaluation_path = run_dir / "matched_v4_future_settled_evaluation_rows.jsonl"
    _write_jsonl(candidate_evaluation_path, candidate_evaluation)
    _write_jsonl(baseline_evaluation_path, baseline_evaluation)

    profile_descriptor = _verified_descriptor(
        freeze_manifest["evaluation_profile"], "future evaluation profile"
    )
    profile = _load_json(Path(profile_descriptor["path"]))
    validate_conformal_v5_future_evaluation_profile(profile)
    evaluation_market_ids = [str(row["market_id"]) for row in selected_rows]
    gate = build_conformal_v5_side_only_future_pnl_gate(
        candidate_evaluation,
        matched_baseline_evaluation_rows=baseline_evaluation,
        evaluation_market_ids=evaluation_market_ids,
        profile=profile,
        decision_freeze_sha256=decision_freeze_descriptor["sha256"],
    )
    gate.update(
        {
            "schema_version": f"{SCHEMA_PREFIX}-side-only-gate-report-v1",
            "run_id": config.run_id,
            "builder_git_commit": config.builder_git_commit,
            "settled_corpus_index": _descriptor(settled_index_path),
            "settled_target_rows": _descriptor(target_path),
            "candidate_evaluation_rows": _descriptor(candidate_evaluation_path),
            "matched_baseline_evaluation_rows": _descriptor(baseline_evaluation_path),
            "target_access_marker": _descriptor(marker_path),
            "target_opened_after_decision_freeze": True,
            "future_results_used_for_tuning": False,
            "future_results_used_for_rerun": False,
            "future_results_used_for_automatic_unlock": False,
            **_blocked_safety_fields(),
        }
    )
    gate["report_id"] = canonical_json_sha256(gate)
    gate_path = run_dir / "conformal_v5_future_side_only_pnl_gate_report.json"
    _write_json(gate_path, gate)
    gate_md_path = run_dir / "conformal_v5_future_side_only_pnl_gate_report.md"
    _write_text(gate_md_path, _gate_markdown(gate))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "prediction_freeze_manifest": _descriptor(freeze_manifest_path),
        "accepted_bet_decision_freeze": decision_freeze_descriptor,
        "settled_corpus_index": _descriptor(settled_index_path),
        "source_settled_corpora": source_descriptors,
        "target_access_marker": _descriptor(marker_path),
        "settled_five_action_targets": _descriptor(target_path),
        "candidate_settled_evaluation_rows": _descriptor(candidate_evaluation_path),
        "matched_baseline_settled_evaluation_rows": _descriptor(baseline_evaluation_path),
        "side_only_pnl_gate_report": _descriptor(gate_path),
        "side_only_pnl_gate_report_markdown": _descriptor(gate_md_path),
        "future_gate_passed": gate["future_gate_passed"],
        "future_gate_blocking_reason_codes": gate["future_gate_blocking_reason_codes"],
        "target_opened_after_decision_freeze": True,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "future_results_used_for_automatic_unlock": False,
        **_blocked_safety_fields(),
    }
    manifest["settlement_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v5_future_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "gate": gate,
        "gate_path": gate_path,
        "gate_sha256": _sha256_file(gate_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _validate_settled_corpus_index(
    index: dict[str, Any],
    *,
    expected_decision_freeze_sha256: str,
    decision_freeze_created_ts: int,
    selected_rows: list[dict[str, Any]],
    reconciliation_started_ts: int,
) -> list[dict[str, Any]]:
    entries = list(index.get("entries") or [])
    expected_markets = {str(row["market_id"]) for row in selected_rows}
    entry_markets = {str(row.get("market_id") or "") for row in entries}
    checks = {
        "schema": index.get("schema_version") == SETTLED_CORPUS_INDEX_SCHEMA_VERSION,
        "freeze_hash": index.get("decision_freeze_sha256") == expected_decision_freeze_sha256,
        "complete_market_set": len(entries) == len(expected_markets)
        and entry_markets == expected_markets,
        "official_read_only": all(
            row.get("official_read_only_resolution") is True for row in entries
        ),
        "post_freeze": all(
            row.get("corpus_built_after_decision_freeze") is True for row in entries
        ),
        "post_close": all(row.get("settled_after_market_close") is True for row in entries),
        "target_access_after_decision_freeze": int(index.get("target_access_started_ts") or 0)
        > decision_freeze_created_ts,
        "index_finalized_after_target_access": int(index.get("index_finalized_ts") or 0)
        >= int(index.get("target_access_started_ts") or 0),
        "before_reconciliation": int(index.get("index_finalized_ts") or 0)
        <= reconciliation_started_ts,
        "no_selection_tuning": index.get("outcomes_used_for_decision_or_selection") is False
        and index.get("outcomes_used_for_threshold_or_model_tuning") is False,
        "safety": all(
            index.get(field) == expected for field, expected in _blocked_safety_fields().items()
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("settled corpus index validation failed: " + ", ".join(blockers))
    return sorted(entries, key=lambda row: str(row["market_id"]))


def _load_and_validate_targets(
    entries: list[dict[str, Any]],
    *,
    selected_rows: list[dict[str, Any]],
    frozen_features: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_by_market = {str(row["market_id"]): row for row in selected_rows}
    frozen_feature_by_key = {
        (str(row["market_id"]), int(row["decision_ts"])): row for row in frozen_features
    }
    targets: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for entry in entries:
        market_id = str(entry["market_id"])
        corpus_descriptor = _verified_descriptor(entry["corpus_manifest"], "corpus manifest")
        feature_descriptor = _verified_descriptor(entry["feature_rows"], "settled feature rows")
        label_descriptor = _verified_descriptor(entry["label_rows"], "settled label rows")
        resolution_descriptor = _verified_descriptor(
            entry["resolution_events"], "official resolution events"
        )
        corpus = _load_json(Path(corpus_descriptor["path"]))
        normalized = dict(corpus.get("normalized_artifact_hashes") or {})
        if (
            normalized.get("feature_rows") != feature_descriptor["sha256"]
            or normalized.get("label_rows") != label_descriptor["sha256"]
            or normalized.get("resolution_events") != resolution_descriptor["sha256"]
        ):
            raise ValueError("settled corpus normalized artifact lineage mismatch")
        source_raw = dict(selected_by_market[market_id].get("raw_artifacts") or {})
        corpus_raw = dict(corpus.get("raw_artifact_hashes") or {})
        for filename in ALLOWED_RAW_FEATURE_FILES:
            if corpus_raw.get(filename) != (source_raw.get(filename) or {}).get("sha256"):
                raise ValueError(f"settled corpus changed frozen raw feature input: {filename}")
        feature_rows = _load_jsonl(Path(feature_descriptor["path"]))
        expected_feature_keys = {key for key in frozen_feature_by_key if key[0] == market_id}
        settled_feature_keys = {
            (str(feature["market_id"]), int(feature["decision_ts"])) for feature in feature_rows
        }
        if (
            len(settled_feature_keys) != len(feature_rows)
            or settled_feature_keys != expected_feature_keys
        ):
            raise ValueError("settled corpus feature decision grid differs from freeze")
        for feature in feature_rows:
            key = (str(feature["market_id"]), int(feature["decision_ts"]))
            frozen = frozen_feature_by_key.get(key)
            comparison = (
                compare_canonical_payloads(
                    _feature_payload(frozen),
                    _feature_payload(feature),
                    frozen_payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
                    settled_payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
                    approved_source_lineage=True,
                )
                if frozen is not None
                else None
            )
            if comparison is None or not comparison.settlement_evaluation_eligible:
                raise ValueError("settled corpus feature rows differ from decision freeze")
        resolutions = _load_jsonl(Path(resolution_descriptor["path"]))
        if len(resolutions) != 1 or str(resolutions[0].get("market_id") or "") != market_id:
            raise ValueError("official resolution row is missing or duplicated")
        resolution = resolutions[0]
        if (
            resolution.get("resolution_status") != "normal"
            or resolution.get("resolved_outcome") not in {"UP", "DOWN"}
            or not str(resolution.get("raw_resolution_sha256") or "")
        ):
            raise ValueError("official resolution is not final")
        labels = _load_jsonl(Path(label_descriptor["path"]))
        if len(labels) != len(feature_rows) * len(REQUIRED_ACTIONS):
            raise ValueError("settled five-action target row count is incomplete")
        by_decision: dict[int, dict[str, dict[str, Any]]] = {}
        for label in labels:
            if (
                str(label.get("market_id") or "") != market_id
                or label.get("resolved_outcome") != resolution["resolved_outcome"]
                or label.get("raw_resolution_sha256") != resolution["raw_resolution_sha256"]
            ):
                raise ValueError("label and official resolution provenance mismatch")
            decision_labels = by_decision.setdefault(int(label["decision_ts"]), {})
            action = str(label["action"])
            if action in decision_labels:
                raise ValueError("settled five-action target contains duplicate action")
            decision_labels[action] = label
        for feature in feature_rows:
            decision_ts = int(feature["decision_ts"])
            action_labels = by_decision.get(decision_ts, {})
            if set(action_labels) != set(REQUIRED_ACTIONS):
                raise ValueError("settled five-action target grid is incomplete")
            values = {
                action: float(action_labels[action]["total_net_pnl_per_notional"])
                for action in REQUIRED_ACTIONS
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("settled action target is non-finite")
            target = {
                "schema_version": f"{SCHEMA_PREFIX}-five-action-target-v1",
                "market_id": market_id,
                "decision_ts": decision_ts,
                "resolved_outcome": resolution["resolved_outcome"],
                "target_net_pnl_per_notional_by_action": values,
                "raw_resolution_sha256": resolution["raw_resolution_sha256"],
                "resolution_rule_sha256": resolution["resolution_rule_sha256"],
                "target_available_only_after_market_close": True,
                "target_joined_after_decision_freeze": True,
                "target_used_as_decision_input": False,
                "future_results_used_for_tuning": False,
                **_blocked_safety_fields(),
            }
            target["target_row_sha256"] = canonical_json_sha256(target)
            targets.append(target)
        sources.append(
            {
                "market_id": market_id,
                "corpus_manifest": corpus_descriptor,
                "feature_rows": feature_descriptor,
                "label_rows": label_descriptor,
                "resolution_events": resolution_descriptor,
            }
        )
    targets.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    target_keys = {(str(row["market_id"]), int(row["decision_ts"])) for row in targets}
    if len(targets) != len(frozen_features) or len(target_keys) != len(targets):
        raise ValueError("settled target count does not match frozen feature rows")
    return targets, sources


def _join_frozen_replay_targets(
    replay_rows: list[dict[str, Any]],
    *,
    targets_by_decision: dict[tuple[str, int], dict[str, Any]],
    policy_name: str,
    decision_freeze_sha256: str,
) -> list[dict[str, Any]]:
    output = []
    for replay in replay_rows:
        key = (str(replay["market_id"]), int(replay["decision_ts"]))
        target = targets_by_decision.get(key)
        if target is None:
            raise ValueError("frozen replay decision has no settled target")
        allowed = replay.get("execution_guard_order_allowed") is True
        action = str(replay["executed_action"])
        target_value = float(target["target_net_pnl_per_notional_by_action"][action])
        order_size = float(replay.get("proposed_order_size") or 0.0)
        net_pnl = order_size * target_value if allowed else 0.0
        row = {
            **replay,
            "policy_name": policy_name,
            "decision_freeze_sha256": decision_freeze_sha256,
            "settlement_resolved": True,
            "resolved_outcome": target["resolved_outcome"],
            "target_net_pnl_per_notional": target_value,
            "accepted_bet_net_pnl": net_pnl,
            "target_joined_after_decision_freeze": True,
            "target_used_as_decision_input": False,
            "forbidden_outcome_field_used_for_decision": False,
            "feature_causality_violation": False,
            "provenance_violation": False,
            "runtime_state_violation": False,
            "future_results_used_for_tuning": False,
            **_blocked_safety_fields(),
        }
        row["settled_evaluation_row_sha256"] = canonical_json_sha256(row)
        output.append(row)
    return output


def _feature_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": str(row["market_id"]),
        "condition_id": str(row["condition_id"]),
        "slug": str(row["slug"]),
        "market_family": str(row["market_family"]),
        "horizon_ms": int(row["horizon_ms"]),
        "decision_ts": int(row["decision_ts"]),
        "feature_cutoff_ts": int(row["feature_cutoff_ts"]),
        "max_input_ts": int(row["max_input_ts"]),
        "available_at_ts": int(row["available_at_ts"]),
        "features": row["features"],
        "feature_provenance": row["feature_provenance"],
    }


def _verify_pin(path: Path, expected: str, name: str) -> None:
    _require_sha256(expected, name=f"expected_{name.replace(' ', '_')}_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _gate_markdown(gate: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Conformal v5 strictly-later side-only PnL gate",
            "",
            f"- passed: `{str(gate['future_gate_passed']).lower()}`",
            f"- candidate PnL: `{gate['candidate_post_cost_net_pnl']:.8f}`",
            f"- matched baseline PnL: `{gate['matched_baseline_post_cost_net_pnl']:.8f}`",
            f"- candidate - baseline: `{gate['candidate_minus_matched_baseline_post_cost_net_pnl']:.8f}`",
            f"- accepted bets / markets: `{gate['guard_accepted_bet_count']} / {gate['guard_accepted_unique_market_count']}`",
            "- hard PnL aggregation: `BUY_UP / BUY_DOWN side-only`",
            "- action/family PnL: `diagnostic_only`",
            "- future results used for tuning/rerun/unlock: `false/false/false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _settled_index_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Conformal v5 future settled-corpus index",
            "",
            f"- ready: `{str(report['settled_corpus_index_ready']).lower()}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- settled corpora: `{report['settled_corpus_ready_market_count']}`",
            f"- unresolved/failed: `{report['unresolved_or_failed_market_count']}`",
            f"- settlement attempts: `{report['settlement_attempt_count']}`",
            f"- retried markets: `{report['settlement_retry_market_count']}`",
            f"- bounded resolution wait seconds: `{report['settlement_max_wait_seconds']}`",
            "- source outcome-blind rounds mutated: `false`",
            "- direct training corpus export: `false`",
            "- official read-only outcome access: `true`",
            "- future results used for tuning/rerun: `false/false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


__all__ = [
    "ConformalV5FutureSettlementCorpusIndexConfig",
    "ConformalV5FutureSettlementConfig",
    "SETTLED_CORPUS_INDEX_SCHEMA_VERSION",
    "build_conformal_v5_future_settled_corpus_index",
    "reconcile_conformal_v5_future_settlement",
]
