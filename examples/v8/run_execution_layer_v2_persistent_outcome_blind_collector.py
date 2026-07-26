"""Run durable candidate-agnostic raw collection in bounded resumable batches."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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

from bigan.v8.polymarket.exact_model_runtime_binding import (  # noqa: E402
    ExactModelRuntimeBindingConfig,
    verify_exact_model_runtime_binding,
)
from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9_future_batch_canary import (  # noqa: E402
    CalibrationScaleAlignedV69FutureBatchCanaryConfig,
    build_v6_9_future_cumulative_canary,
    run_v6_9_future_batch_canary,
    validate_v6_9_future_collection_plan,
    write_v6_9_future_cumulative_canary,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_batch_canary import (  # noqa: E402
    MarketClusteredMeanEVV62FutureBatchCanaryConfig,
    build_v6_2_future_cumulative_canary,
    run_market_clustered_mean_ev_v6_2_future_batch_canary,
    write_v6_2_future_cumulative_canary,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (  # noqa: E402
    OutcomeBlindDevelopmentBatchCanaryConfig,
    run_outcome_blind_development_batch_canary,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (  # noqa: E402
    PersistentOutcomeBlindBatchIndexConfig,
    index_persistent_outcome_blind_batch,
    load_and_validate_persistent_outcome_blind_index,
    validate_persistent_outcome_blind_collector_protocol,
)
from examples.v8.run_polymarket_async_round_collector import (  # noqa: E402
    run_polymarket_async_round_collector_cli,
)

DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
DEFAULT_BATCH_CANARY_FEATURE_CONTRACT = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)
DEFAULT_BATCH_CANARY_FEATURE_CONTRACT_SHA256 = (
    "a4819ad6beec8d72612aa25ef2af751c357e807d514dcf1d2c94b37eba07c959"
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


class OutcomeBlindBatchCanaryFailure(RuntimeError):
    """Stop collection immediately when a completed batch violates the canary contract."""


def _single_service_instance(function):
    """Prevent overlapping service processes from collecting the same round window."""

    @wraps(function)
    def wrapped(**kwargs):
        runtime_binding_summary = (
            _verify_challenge_runtime_binding_from_kwargs(kwargs)
        )
        if runtime_binding_summary is not None:
            kwargs["_exact_model_runtime_binding_summary"] = (
                runtime_binding_summary
            )
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
    batch_canary_feature_contract_path: Path | str = DEFAULT_BATCH_CANARY_FEATURE_CONTRACT,
    batch_canary_feature_contract_sha256: str = (DEFAULT_BATCH_CANARY_FEATURE_CONTRACT_SHA256),
    v6_2_candidate_manifest_path: Path | str | None = None,
    v6_2_candidate_manifest_sha256: str | None = None,
    v6_6_point_freeze_manifest_path: Path | str | None = None,
    v6_6_point_freeze_manifest_sha256: str | None = None,
    v6_9_candidate_manifest_path: Path | str | None = None,
    v6_9_candidate_manifest_sha256: str | None = None,
    v6_9_collection_plan_path: Path | str | None = None,
    v6_9_collection_plan_sha256: str | None = None,
    challenge_candidate_contract_path: Path | str | None = None,
    challenge_candidate_contract_sha256: str | None = None,
    challenge_frozen_model_binding_path: Path | str | None = None,
    challenge_frozen_model_binding_sha256: str | None = None,
    challenge_frozen_model_artifact_path: Path | str | None = None,
    challenge_frozen_model_artifact_sha256: str | None = None,
    challenge_candidate_profile_path: Path | str | None = None,
    challenge_candidate_profile_sha256: str | None = None,
    _exact_model_runtime_binding_summary: dict | None = None,
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
    batch_canary_feature_contract_path = Path(batch_canary_feature_contract_path).resolve()
    if _sha256(batch_canary_feature_contract_path) != batch_canary_feature_contract_sha256.lower():
        raise ValueError("batch canary feature contract SHA-256 mismatch")
    if (v6_2_candidate_manifest_path is None) != (v6_2_candidate_manifest_sha256 is None):
        raise ValueError("v6.2 candidate manifest path and SHA-256 must be provided together")
    if v6_2_candidate_manifest_path is not None:
        v6_2_candidate_manifest_path = Path(v6_2_candidate_manifest_path).resolve()
        if _sha256(v6_2_candidate_manifest_path) != str(v6_2_candidate_manifest_sha256).lower():
            raise ValueError("v6.2 candidate manifest SHA-256 mismatch")
    v6_9_enabled = any(
        value is not None
        for value in (
            v6_9_candidate_manifest_path,
            v6_9_candidate_manifest_sha256,
            v6_9_collection_plan_path,
            v6_9_collection_plan_sha256,
        )
    )
    if v6_9_enabled and any(
        value is None
        for value in (
            v6_9_candidate_manifest_path,
            v6_9_candidate_manifest_sha256,
            v6_9_collection_plan_path,
            v6_9_collection_plan_sha256,
        )
    ):
        raise ValueError("v6.9 candidate and collection-plan paths/hashes are required together")
    if v6_9_enabled and v6_2_candidate_manifest_path is None:
        raise ValueError("v6.9 batch canary requires the frozen v6.2 candidate manifest")
    if v6_9_enabled:
        v6_9_candidate_manifest_path = Path(v6_9_candidate_manifest_path).resolve()
        v6_9_collection_plan_path = Path(v6_9_collection_plan_path).resolve()
        if _sha256(v6_9_candidate_manifest_path) != str(v6_9_candidate_manifest_sha256).lower():
            raise ValueError("v6.9 candidate manifest SHA-256 mismatch")
        if _sha256(v6_9_collection_plan_path) != str(v6_9_collection_plan_sha256).lower():
            raise ValueError("v6.9 collection plan SHA-256 mismatch")
        v6_9_candidate_manifest = _load_state(v6_9_candidate_manifest_path)
        v6_9_collection_plan = _load_state(v6_9_collection_plan_path)
        validate_v6_9_future_collection_plan(
            v6_9_collection_plan,
            candidate_manifest=v6_9_candidate_manifest,
            candidate_manifest_sha256=str(v6_9_candidate_manifest_sha256),
        )
    else:
        v6_9_collection_plan = None
    v6_6_plan = _load_v6_6_calibration_plan(
        manifest_path=v6_6_point_freeze_manifest_path,
        expected_sha256=v6_6_point_freeze_manifest_sha256,
    )
    frozen_protocol_path = root / "persistent_outcome_blind_collector_protocol.json"
    if frozen_protocol_path.exists():
        if _sha256(frozen_protocol_path) != protocol_sha256.lower():
            raise ValueError("frozen service protocol SHA-256 mismatch")
    else:
        frozen_protocol_path.write_bytes(protocol_path.read_bytes())
    protocol_path = frozen_protocol_path
    index_path = root / "persistent_outcome_blind_round_index.jsonl"
    service_state_path = root / "persistent_outcome_blind_service_state.json"
    terminal_stop_path = root / "persistent_outcome_blind_canary_terminal_stop.json"
    collection_complete_stop_path = (
        root / "persistent_outcome_blind_v6_2_future_collection_complete_stop.json"
    )
    v6_6_collection_stop_path = (
        root / "persistent_outcome_blind_v6_6_calibration_collection_stop.json"
    )
    v6_9_collection_stop_path = (
        root / "persistent_outcome_blind_v6_9_confirmatory_collection_stop.json"
    )
    if v6_9_enabled and v6_9_collection_stop_path.is_file():
        return _load_state(service_state_path)
    if v6_6_plan is not None and v6_6_collection_stop_path.is_file():
        return _load_state(service_state_path)
    if v6_6_plan is None and collection_complete_stop_path.is_file():
        completed_state = _load_state(service_state_path)
        completed_state["status"] = "v6_2_future_holdout_collection_complete"
        return completed_state
    if terminal_stop_path.is_file():
        terminal_stop = _load_state(terminal_stop_path)
        raise OutcomeBlindBatchCanaryFailure(
            "persistent outcome-blind canary terminal stop is active: "
            + ",".join(terminal_stop.get("blocking_reason_codes") or [])
        )
    service_state = _load_state(service_state_path)
    if v6_6_plan is not None:
        initial_progress = _v6_6_calibration_progress(index_path, v6_6_plan)
        terminal_state = _v6_6_terminal_collection_state(
            progress=initial_progress,
            plan=v6_6_plan,
            point_freeze_manifest_path=Path(str(v6_6_point_freeze_manifest_path)).resolve(),
            point_freeze_manifest_sha256=str(v6_6_point_freeze_manifest_sha256).lower(),
        )
        if terminal_state is not None:
            _write_state(service_state_path, terminal_state)
            _write_state(v6_6_collection_stop_path, terminal_state)
            return terminal_state
    batch_sequence = int(service_state.get("last_completed_batch_sequence") or 0) + 1
    completed_batches = 0
    consecutive_failures = 0
    collector_commit = _git_head()
    v6_2_batch_report_descriptors = list(service_state.get("v6_2_batch_canary_reports") or [])
    v6_9_batch_report_descriptors = list(service_state.get("v6_9_batch_canary_reports") or [])
    while max_batches == 0 or completed_batches < max_batches:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        batch_id = f"persistent-outcome-blind-{batch_sequence:06d}-{stamp}"
        canary_result: dict | None = None
        v6_2_canary_result: dict | None = None
        v6_2_cumulative_result: dict | None = None
        v6_9_canary_result: dict | None = None
        v6_9_cumulative_result: dict | None = None
        try:
            effective_batch_round_count = batch_round_count
            if v6_6_plan is not None:
                progress_before_batch = _v6_6_calibration_progress(index_path, v6_6_plan)
                effective_batch_round_count = min(
                    batch_round_count,
                    v6_6_plan["maximum_attempted_market_count"]
                    - progress_before_batch["attempted_market_count"],
                )
                if effective_batch_round_count <= 0:
                    raise OutcomeBlindBatchCanaryFailure(
                        "v6.6 fresh calibration attempt cap reached before target support"
                    )
            summary = run_polymarket_async_round_collector_cli(
                batch_id=batch_id,
                output_dir=root / "captures",
                round_count=effective_batch_round_count,
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
            try:
                canary_result = run_outcome_blind_development_batch_canary(
                    OutcomeBlindDevelopmentBatchCanaryConfig(
                        run_id=f"outcome-blind-batch-canary-{batch_sequence:06d}-{stamp}",
                        output_dir=root / "batch_canary_runs",
                        collector_index_path=index_path,
                        expected_collector_index_sha256=index_result["index_sha256"],
                        batch_id=batch_id,
                        feature_contract_path=batch_canary_feature_contract_path,
                        expected_feature_contract_sha256=(
                            batch_canary_feature_contract_sha256.lower()
                        ),
                        exact_model_runtime_binding_summary=(
                            _exact_model_runtime_binding_summary
                        ),
                    )
                )
            except Exception as exc:
                raise OutcomeBlindBatchCanaryFailure(
                    f"completed batch canary could not be validated: {type(exc).__name__}: {exc}"
                ) from exc
            _require_batch_canary_passed(canary_result)
            if v6_2_candidate_manifest_path is not None:
                v6_2_canary_result = run_market_clustered_mean_ev_v6_2_future_batch_canary(
                    MarketClusteredMeanEVV62FutureBatchCanaryConfig(
                        run_id=f"v6-2-future-batch-canary-{batch_sequence:06d}-{stamp}",
                        output_dir=root / "v6_2_batch_canary_runs",
                        development_batch_canary_manifest_path=canary_result["manifest_path"],
                        expected_development_batch_canary_manifest_sha256=(
                            canary_result["manifest_sha256"]
                        ),
                        candidate_manifest_path=v6_2_candidate_manifest_path,
                        expected_candidate_manifest_sha256=str(v6_2_candidate_manifest_sha256),
                    )
                )
                if not v6_9_enabled:
                    v6_2_batch_report_descriptors.append(
                        {
                            "path": str(v6_2_canary_result["report_path"]),
                            "sha256": v6_2_canary_result["report_sha256"],
                        }
                    )
                    batch_reports = []
                    batch_report_paths = []
                    for descriptor in v6_2_batch_report_descriptors:
                        report_path = Path(str(descriptor["path"])).resolve()
                        if _sha256(report_path) != str(descriptor["sha256"]):
                            raise OutcomeBlindBatchCanaryFailure(
                                "v6.2 prior batch report SHA-256 mismatch"
                            )
                        batch_report_paths.append(report_path)
                        batch_reports.append(_load_state(report_path))
                    cumulative_report = build_v6_2_future_cumulative_canary(
                        batch_reports,
                        run_id=f"v6-2-future-cumulative-{batch_sequence:06d}-{stamp}",
                    )
                    v6_2_cumulative_result = write_v6_2_future_cumulative_canary(
                        report=cumulative_report,
                        batch_report_paths=batch_report_paths,
                        output_dir=root / "v6_2_cumulative_canary_runs",
                        run_id=f"v6-2-future-cumulative-{batch_sequence:06d}-{stamp}",
                    )
                    if cumulative_report["target_free_terminal_blocked"]:
                        raise OutcomeBlindBatchCanaryFailure(
                            "v6.2 future cumulative canary blocked: "
                            + ",".join(
                                cumulative_report["target_free_terminal_blocking_reason_codes"]
                            )
                        )
            if v6_9_enabled:
                v6_9_canary_result = run_v6_9_future_batch_canary(
                    CalibrationScaleAlignedV69FutureBatchCanaryConfig(
                        run_id=f"v6-9-future-batch-canary-{batch_sequence:06d}-{stamp}",
                        output_dir=root / "v6_9_batch_canary_runs",
                        v6_2_batch_canary_manifest_path=v6_2_canary_result["manifest_path"],
                        expected_v6_2_batch_canary_manifest_sha256=v6_2_canary_result[
                            "manifest_sha256"
                        ],
                        candidate_manifest_path=v6_9_candidate_manifest_path,
                        expected_candidate_manifest_sha256=str(v6_9_candidate_manifest_sha256),
                        collection_plan_path=v6_9_collection_plan_path,
                        expected_collection_plan_sha256=str(v6_9_collection_plan_sha256),
                    )
                )
                v6_9_batch_report_descriptors.append(
                    {
                        "path": str(v6_9_canary_result["report_path"]),
                        "sha256": v6_9_canary_result["report_sha256"],
                    }
                )
                v6_9_reports = []
                v6_9_report_paths = []
                for descriptor in v6_9_batch_report_descriptors:
                    report_path = Path(str(descriptor["path"])).resolve()
                    if _sha256(report_path) != str(descriptor["sha256"]):
                        raise OutcomeBlindBatchCanaryFailure(
                            "v6.9 prior batch report SHA-256 mismatch"
                        )
                    v6_9_report_paths.append(report_path)
                    v6_9_reports.append(_load_state(report_path))
                v6_9_cumulative_report = build_v6_9_future_cumulative_canary(
                    v6_9_reports,
                    run_id=f"v6-9-future-cumulative-{batch_sequence:06d}-{stamp}",
                    collection_plan=v6_9_collection_plan,
                    collection_plan_sha256=str(v6_9_collection_plan_sha256),
                )
                v6_9_cumulative_result = write_v6_9_future_cumulative_canary(
                    report=v6_9_cumulative_report,
                    batch_report_paths=v6_9_report_paths,
                    collection_plan_path=v6_9_collection_plan_path,
                    output_dir=root / "v6_9_cumulative_canary_runs",
                    run_id=f"v6-9-future-cumulative-{batch_sequence:06d}-{stamp}",
                )
                if v6_9_cumulative_report["target_free_terminal_blocked"]:
                    raise OutcomeBlindBatchCanaryFailure(
                        "v6.9 future cumulative canary blocked: "
                        + ",".join(
                            v6_9_cumulative_report["target_free_terminal_blocking_reason_codes"]
                        )
                    )
            consecutive_failures = 0
            completed_batches += 1
            next_state = {
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
                "last_batch_canary_report_path": str(canary_result["report_path"]),
                "last_batch_canary_report_sha256": canary_result["report_sha256"],
                "last_batch_canary_manifest_path": str(canary_result["manifest_path"]),
                "last_batch_canary_manifest_sha256": canary_result["manifest_sha256"],
                "last_batch_canary_passed": True,
                "batch_canary_feature_contract_path": str(batch_canary_feature_contract_path),
                "batch_canary_feature_contract_sha256": (
                    batch_canary_feature_contract_sha256.lower()
                ),
                "consecutive_failure_count": 0,
                "outcome_blind_collection_only": True,
                "settlement_finalizer_started": False,
                "resolution_provider_called": False,
                "training_corpus_export_attempted": False,
                "labels_outcomes_or_pnl_opened": False,
                "exact_model_runtime_binding_required": (
                    _exact_model_runtime_binding_summary is not None
                ),
                "exact_model_runtime_binding_verified": (
                    _exact_model_runtime_binding_summary is not None
                ),
                "exact_model_runtime_binding_summary": (
                    _exact_model_runtime_binding_summary
                ),
                **SAFETY,
            }
            if v6_2_canary_result is not None and v6_2_cumulative_result is not None:
                next_state.update(
                    {
                        "v6_2_candidate_manifest_path": str(v6_2_candidate_manifest_path),
                        "v6_2_candidate_manifest_sha256": str(
                            v6_2_candidate_manifest_sha256
                        ).lower(),
                        "last_v6_2_batch_canary_report_path": str(
                            v6_2_canary_result["report_path"]
                        ),
                        "last_v6_2_batch_canary_report_sha256": v6_2_canary_result["report_sha256"],
                        "last_v6_2_batch_canary_manifest_path": str(
                            v6_2_canary_result["manifest_path"]
                        ),
                        "last_v6_2_batch_canary_manifest_sha256": v6_2_canary_result[
                            "manifest_sha256"
                        ],
                        "v6_2_batch_canary_reports": v6_2_batch_report_descriptors,
                        "v6_2_cumulative_canary_report_path": str(
                            v6_2_cumulative_result["report_path"]
                        ),
                        "v6_2_cumulative_canary_report_sha256": (
                            v6_2_cumulative_result["report_sha256"]
                        ),
                        "v6_2_cumulative_canary_manifest_path": str(
                            v6_2_cumulative_result["manifest_path"]
                        ),
                        "v6_2_cumulative_canary_manifest_sha256": (
                            v6_2_cumulative_result["manifest_sha256"]
                        ),
                        "v6_2_future_quality_valid_market_count": (
                            v6_2_cumulative_result["report"]["quality_valid_market_count"]
                        ),
                        "v6_2_future_guard_accepted_unique_market_count": (
                            v6_2_cumulative_result["report"]["guard_accepted_unique_market_count"]
                        ),
                        "v6_2_future_guard_accepted_unique_market_count_by_side": (
                            v6_2_cumulative_result["report"][
                                "guard_accepted_unique_market_count_by_side"
                            ]
                        ),
                        "v6_2_future_holdout_collection_complete": (
                            v6_2_cumulative_result["report"]["future_holdout_collection_complete"]
                        ),
                    }
                )
            if v6_6_plan is not None:
                v6_6_progress = _v6_6_calibration_progress(index_path, v6_6_plan)
                next_state.update(
                    {
                        "status": "collecting_v6_6_fresh_calibration_batches",
                        "v6_6_point_freeze_manifest_path": str(
                            Path(str(v6_6_point_freeze_manifest_path)).resolve()
                        ),
                        "v6_6_point_freeze_manifest_sha256": str(
                            v6_6_point_freeze_manifest_sha256
                        ).lower(),
                        "v6_6_fresh_calibration_boundary_sequence": v6_6_plan[
                            "collector_index_boundary_sequence"
                        ],
                        "v6_6_fresh_calibration_attempted_market_count": (
                            v6_6_progress["attempted_market_count"]
                        ),
                        "v6_6_fresh_calibration_quality_valid_market_count": (
                            v6_6_progress["quality_valid_market_count"]
                        ),
                        "v6_6_fresh_calibration_target_quality_valid_market_count": (
                            v6_6_plan["target_quality_valid_market_count"]
                        ),
                        "v6_6_fresh_calibration_maximum_attempted_market_count": (
                            v6_6_plan["maximum_attempted_market_count"]
                        ),
                        "v6_6_fresh_calibration_collection_complete": (
                            v6_6_progress["target_reached"]
                        ),
                        "v6_6_candidate_scoring_attempted": False,
                    }
                )
            if v6_9_canary_result is not None and v6_9_cumulative_result is not None:
                next_state.update(
                    {
                        "status": "collecting_v6_9_future_confirmatory_batches",
                        "v6_9_candidate_manifest_path": str(v6_9_candidate_manifest_path),
                        "v6_9_candidate_manifest_sha256": str(
                            v6_9_candidate_manifest_sha256
                        ).lower(),
                        "v6_9_collection_plan_path": str(v6_9_collection_plan_path),
                        "v6_9_collection_plan_sha256": str(v6_9_collection_plan_sha256).lower(),
                        "last_v6_9_batch_canary_report_path": str(
                            v6_9_canary_result["report_path"]
                        ),
                        "last_v6_9_batch_canary_report_sha256": v6_9_canary_result["report_sha256"],
                        "last_v6_9_batch_action_liveness_passed": v6_9_canary_result["report"][
                            "batch_action_liveness_passed"
                        ],
                        "v6_9_batch_canary_reports": v6_9_batch_report_descriptors,
                        "v6_9_cumulative_canary_report_path": str(
                            v6_9_cumulative_result["report_path"]
                        ),
                        "v6_9_cumulative_canary_report_sha256": v6_9_cumulative_result[
                            "report_sha256"
                        ],
                        "v6_9_future_attempted_market_count": v6_9_cumulative_result["report"][
                            "attempted_market_count"
                        ],
                        "v6_9_future_quality_valid_market_count": v6_9_cumulative_result["report"][
                            "quality_valid_market_count"
                        ],
                        "v6_9_future_guard_accepted_unique_market_count": v6_9_cumulative_result[
                            "report"
                        ]["guard_accepted_unique_market_count"],
                        "v6_9_future_confirmatory_collection_complete": v6_9_cumulative_result[
                            "report"
                        ]["future_confirmatory_collection_complete"],
                    }
                )
            _write_state(
                service_state_path,
                next_state,
            )
            if v6_6_plan is not None:
                terminal_state = _v6_6_terminal_collection_state(
                    progress=v6_6_progress,
                    plan=v6_6_plan,
                    point_freeze_manifest_path=Path(str(v6_6_point_freeze_manifest_path)).resolve(),
                    point_freeze_manifest_sha256=str(v6_6_point_freeze_manifest_sha256).lower(),
                    base_state=next_state,
                )
                if terminal_state is not None:
                    _write_state(service_state_path, terminal_state)
                    _write_state(v6_6_collection_stop_path, terminal_state)
                    return terminal_state
            if (
                v6_9_cumulative_result is not None
                and v6_9_cumulative_result["report"]["future_confirmatory_collection_complete"]
            ):
                completed_state = _load_state(service_state_path)
                completed_state["status"] = "v6_9_future_confirmatory_collection_complete"
                _write_state(service_state_path, completed_state)
                _write_state(
                    v6_9_collection_stop_path,
                    {
                        "status": "v6_9_future_confirmatory_collection_complete",
                        "last_completed_batch_sequence": batch_sequence,
                        "cumulative_canary_report_path": str(v6_9_cumulative_result["report_path"]),
                        "cumulative_canary_report_sha256": v6_9_cumulative_result["report_sha256"],
                        "labels_outcomes_or_pnl_opened": False,
                        **SAFETY,
                    },
                )
                return completed_state
            if (
                v6_2_cumulative_result is not None
                and v6_2_cumulative_result["report"]["future_holdout_collection_complete"]
            ):
                completed_state = _load_state(service_state_path)
                completed_state["status"] = "v6_2_future_holdout_collection_complete"
                _write_state(service_state_path, completed_state)
                _write_state(
                    collection_complete_stop_path,
                    {
                        "status": "v6_2_future_holdout_collection_complete",
                        "last_completed_batch_sequence": batch_sequence,
                        "cumulative_canary_report_path": str(v6_2_cumulative_result["report_path"]),
                        "cumulative_canary_report_sha256": v6_2_cumulative_result["report_sha256"],
                        "labels_outcomes_or_pnl_opened": False,
                        **SAFETY,
                    },
                )
                return completed_state
            batch_sequence += 1
        except Exception as exc:
            consecutive_failures += 1
            failure_state = {
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
                "exact_model_runtime_binding_required": (
                    _exact_model_runtime_binding_summary is not None
                ),
                "exact_model_runtime_binding_verified": (
                    _exact_model_runtime_binding_summary is not None
                ),
                "exact_model_runtime_binding_summary": (
                    _exact_model_runtime_binding_summary
                ),
                **SAFETY,
            }
            if canary_result is not None:
                failure_state.update(
                    {
                        "failed_batch_canary_report_path": str(canary_result["report_path"]),
                        "failed_batch_canary_report_sha256": canary_result["report_sha256"],
                        "failed_batch_canary_manifest_path": str(canary_result["manifest_path"]),
                        "failed_batch_canary_manifest_sha256": canary_result["manifest_sha256"],
                        "failed_batch_canary_reason_codes": canary_result["report"][
                            "development_data_canary_blocking_reason_codes"
                        ],
                    }
                )
            if v6_2_canary_result is not None:
                failure_state.update(
                    {
                        "failed_v6_2_batch_canary_report_path": str(
                            v6_2_canary_result["report_path"]
                        ),
                        "failed_v6_2_batch_canary_report_sha256": v6_2_canary_result[
                            "report_sha256"
                        ],
                    }
                )
            if v6_2_cumulative_result is not None:
                failure_state.update(
                    {
                        "failed_v6_2_cumulative_report_path": str(
                            v6_2_cumulative_result["report_path"]
                        ),
                        "failed_v6_2_cumulative_report_sha256": (
                            v6_2_cumulative_result["report_sha256"]
                        ),
                        "failed_v6_2_cumulative_reason_codes": (
                            v6_2_cumulative_result["report"][
                                "target_free_terminal_blocking_reason_codes"
                            ]
                        ),
                    }
                )
            if v6_9_canary_result is not None:
                failure_state.update(
                    {
                        "failed_v6_9_batch_canary_report_path": str(
                            v6_9_canary_result["report_path"]
                        ),
                        "failed_v6_9_batch_canary_report_sha256": v6_9_canary_result[
                            "report_sha256"
                        ],
                    }
                )
            if v6_9_cumulative_result is not None:
                failure_state.update(
                    {
                        "failed_v6_9_cumulative_report_path": str(
                            v6_9_cumulative_result["report_path"]
                        ),
                        "failed_v6_9_cumulative_report_sha256": v6_9_cumulative_result[
                            "report_sha256"
                        ],
                        "failed_v6_9_cumulative_reason_codes": v6_9_cumulative_result["report"][
                            "target_free_terminal_blocking_reason_codes"
                        ],
                    }
                )
            if isinstance(exc, OutcomeBlindBatchCanaryFailure):
                if v6_9_cumulative_result is not None:
                    terminal_reasons = v6_9_cumulative_result["report"][
                        "target_free_terminal_blocking_reason_codes"
                    ]
                elif v6_2_cumulative_result is not None:
                    terminal_reasons = v6_2_cumulative_result["report"][
                        "target_free_terminal_blocking_reason_codes"
                    ]
                elif canary_result is not None:
                    terminal_reasons = canary_result["report"][
                        "development_data_canary_blocking_reason_codes"
                    ]
                else:
                    terminal_reasons = ["outcome_blind_batch_canary_validation_exception"]
                terminal_stop = {
                    "status": "persistent_outcome_blind_canary_terminal_stop",
                    "failed_batch_sequence": batch_sequence,
                    "failed_batch_id": batch_id,
                    "blocking_reason_codes": terminal_reasons,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "automatic_restart_may_not_resume_collection": True,
                    "explicit_operator_review_required": True,
                    "labels_outcomes_or_pnl_opened": False,
                    **SAFETY,
                }
                if canary_result is not None:
                    terminal_stop.update(
                        {
                            "canary_report_path": str(canary_result["report_path"]),
                            "canary_report_sha256": canary_result["report_sha256"],
                            "canary_manifest_path": str(canary_result["manifest_path"]),
                            "canary_manifest_sha256": canary_result["manifest_sha256"],
                        }
                    )
                _write_state(terminal_stop_path, terminal_stop)
            _write_state(
                service_state_path,
                failure_state,
            )
            if consecutive_failures >= max_consecutive_failures:
                raise
            if isinstance(exc, OutcomeBlindBatchCanaryFailure):
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
    parser.add_argument(
        "--batch-canary-feature-contract",
        default=str(DEFAULT_BATCH_CANARY_FEATURE_CONTRACT),
    )
    parser.add_argument(
        "--batch-canary-feature-contract-sha256",
        default=DEFAULT_BATCH_CANARY_FEATURE_CONTRACT_SHA256,
    )
    parser.add_argument("--v6-2-candidate-manifest")
    parser.add_argument("--v6-2-candidate-manifest-sha256")
    parser.add_argument("--v6-6-point-freeze-manifest")
    parser.add_argument("--v6-6-point-freeze-manifest-sha256")
    parser.add_argument("--v6-9-candidate-manifest")
    parser.add_argument("--v6-9-candidate-manifest-sha256")
    parser.add_argument("--v6-9-collection-plan")
    parser.add_argument("--v6-9-collection-plan-sha256")
    parser.add_argument("--challenge-candidate-contract")
    parser.add_argument("--challenge-candidate-contract-sha256")
    parser.add_argument("--challenge-frozen-model-binding")
    parser.add_argument("--challenge-frozen-model-binding-sha256")
    parser.add_argument("--challenge-frozen-model-artifact")
    parser.add_argument("--challenge-frozen-model-artifact-sha256")
    parser.add_argument("--challenge-candidate-profile")
    parser.add_argument("--challenge-candidate-profile-sha256")
    args = parser.parse_args(argv)
    state = run_service(
        service_root=args.service_root,
        protocol_path=args.protocol,
        protocol_sha256=args.protocol_sha256,
        batch_round_count=args.batch_round_count,
        max_batches=args.max_batches,
        max_consecutive_failures=args.max_consecutive_failures,
        failure_backoff_seconds=args.failure_backoff_seconds,
        batch_canary_feature_contract_path=args.batch_canary_feature_contract,
        batch_canary_feature_contract_sha256=args.batch_canary_feature_contract_sha256,
        v6_2_candidate_manifest_path=args.v6_2_candidate_manifest,
        v6_2_candidate_manifest_sha256=args.v6_2_candidate_manifest_sha256,
        v6_6_point_freeze_manifest_path=args.v6_6_point_freeze_manifest,
        v6_6_point_freeze_manifest_sha256=args.v6_6_point_freeze_manifest_sha256,
        v6_9_candidate_manifest_path=args.v6_9_candidate_manifest,
        v6_9_candidate_manifest_sha256=args.v6_9_candidate_manifest_sha256,
        v6_9_collection_plan_path=args.v6_9_collection_plan,
        v6_9_collection_plan_sha256=args.v6_9_collection_plan_sha256,
        challenge_candidate_contract_path=(
            args.challenge_candidate_contract
        ),
        challenge_candidate_contract_sha256=(
            args.challenge_candidate_contract_sha256
        ),
        challenge_frozen_model_binding_path=(
            args.challenge_frozen_model_binding
        ),
        challenge_frozen_model_binding_sha256=(
            args.challenge_frozen_model_binding_sha256
        ),
        challenge_frozen_model_artifact_path=(
            args.challenge_frozen_model_artifact
        ),
        challenge_frozen_model_artifact_sha256=(
            args.challenge_frozen_model_artifact_sha256
        ),
        challenge_candidate_profile_path=args.challenge_candidate_profile,
        challenge_candidate_profile_sha256=(
            args.challenge_candidate_profile_sha256
        ),
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def _verify_challenge_runtime_binding_from_kwargs(
    kwargs: dict,
) -> dict | None:
    fields = (
        "challenge_candidate_contract_path",
        "challenge_candidate_contract_sha256",
        "challenge_frozen_model_binding_path",
        "challenge_frozen_model_binding_sha256",
        "challenge_frozen_model_artifact_path",
        "challenge_frozen_model_artifact_sha256",
        "challenge_candidate_profile_path",
        "challenge_candidate_profile_sha256",
    )
    supplied = [kwargs.get(field) is not None for field in fields]
    if not any(supplied):
        return None
    if not all(supplied):
        missing = [
            field for field in fields if kwargs.get(field) is None
        ]
        raise ValueError(
            "challenge runtime binding arguments must be supplied together: "
            + ",".join(missing)
        )
    return verify_exact_model_runtime_binding(
        ExactModelRuntimeBindingConfig(
            candidate_contract_path=kwargs[
                "challenge_candidate_contract_path"
            ],
            expected_candidate_contract_sha256=kwargs[
                "challenge_candidate_contract_sha256"
            ],
            frozen_model_binding_path=kwargs[
                "challenge_frozen_model_binding_path"
            ],
            expected_frozen_model_binding_sha256=kwargs[
                "challenge_frozen_model_binding_sha256"
            ],
            frozen_model_artifact_path=kwargs[
                "challenge_frozen_model_artifact_path"
            ],
            expected_frozen_model_artifact_sha256=kwargs[
                "challenge_frozen_model_artifact_sha256"
            ],
            candidate_profile_path=kwargs[
                "challenge_candidate_profile_path"
            ],
            expected_candidate_profile_sha256=kwargs[
                "challenge_candidate_profile_sha256"
            ],
        )
    )


def _load_state(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_v6_6_calibration_plan(
    *,
    manifest_path: Path | str | None,
    expected_sha256: str | None,
) -> dict | None:
    if (manifest_path is None) != (expected_sha256 is None):
        raise ValueError("v6.6 point freeze manifest path and SHA-256 must be provided together")
    if manifest_path is None:
        return None
    path = Path(manifest_path).resolve()
    if _sha256(path) != str(expected_sha256).lower():
        raise ValueError("v6.6 point freeze manifest SHA-256 mismatch")
    manifest = _load_state(path)
    if manifest.get("candidate_name") != "policy_selected_runtime_pnl_v6_6":
        raise ValueError("v6.6 point freeze candidate identity mismatch")
    if manifest.get("point_model_frozen") is not True:
        raise ValueError("v6.6 point model is not frozen")
    if manifest.get("fresh_calibration_collection_allowed") is not True:
        raise ValueError("v6.6 fresh calibration collection is not allowed")
    if manifest.get("fresh_calibration_outcomes_opened") is not False:
        raise ValueError("v6.6 fresh calibration outcomes must remain sealed")
    if manifest.get("candidate_scoring_frozen") is not False:
        raise ValueError("v6.6 candidate scoring must not be frozen before calibration")
    safety_mismatches = [
        field for field, expected in SAFETY.items() if manifest.get(field) != expected
    ]
    if safety_mismatches:
        raise ValueError("v6.6 point freeze safety mismatch: " + ",".join(safety_mismatches))
    boundary = dict(manifest.get("fresh_calibration_boundary") or {})
    required_positive = (
        "collector_index_boundary_sequence",
        "minimum_market_start_ts_exclusive",
        "target_quality_valid_market_count",
        "maximum_attempted_market_count",
        "batch_market_count",
    )
    if any(int(boundary.get(field) or 0) <= 0 for field in required_positive):
        raise ValueError("v6.6 fresh calibration boundary is incomplete")
    if boundary["target_quality_valid_market_count"] > boundary["maximum_attempted_market_count"]:
        raise ValueError("v6.6 fresh calibration target exceeds attempt cap")
    for field in ("collector_index_boundary_sha256", "collector_last_entry_sha256"):
        value = str(boundary.get(field) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"v6.6 {field} is not a SHA-256 hex digest")
    return boundary


def _v6_6_calibration_progress(index_path: Path, plan: dict) -> dict:
    rows = load_and_validate_persistent_outcome_blind_index(index_path)
    boundary_sequence = int(plan["collector_index_boundary_sequence"])
    if len(rows) < boundary_sequence:
        raise ValueError("v6.6 collector index precedes frozen calibration boundary")
    lines = index_path.read_bytes().splitlines(keepends=True)
    boundary_bytes = b"".join(lines[:boundary_sequence])
    if hashlib.sha256(boundary_bytes).hexdigest() != plan["collector_index_boundary_sha256"]:
        raise ValueError("v6.6 collector index boundary SHA-256 mismatch")
    if rows[boundary_sequence - 1]["entry_sha256"] != plan["collector_last_entry_sha256"]:
        raise ValueError("v6.6 collector boundary last-entry SHA-256 mismatch")
    post_boundary = rows[boundary_sequence:]
    minimum_start = int(plan["minimum_market_start_ts_exclusive"])
    invalid_time_rows = [
        row
        for row in post_boundary
        if int(row.get("scheduled_round_start_ts") or 0) <= minimum_start
    ]
    if invalid_time_rows:
        raise ValueError("v6.6 calibration row is not strictly later than frozen boundary")
    quality_rows = [row for row in post_boundary if row.get("capture_quality_valid") is True]
    unique_quality_markets = {
        str(row.get("market_id") or "") for row in quality_rows if row.get("market_id")
    }
    if len(unique_quality_markets) != len(quality_rows):
        raise ValueError("v6.6 calibration quality rows contain duplicate market identity")
    target = int(plan["target_quality_valid_market_count"])
    maximum = int(plan["maximum_attempted_market_count"])
    return {
        "attempted_market_count": len(post_boundary),
        "quality_valid_market_count": len(quality_rows),
        "target_reached": len(quality_rows) >= target,
        "attempt_cap_reached": len(post_boundary) >= maximum,
        "first_selected_quality_market_ids": [
            str(row["market_id"]) for row in quality_rows[:target]
        ],
        "labels_outcomes_or_pnl_opened": False,
    }


def _v6_6_terminal_collection_state(
    *,
    progress: dict,
    plan: dict,
    point_freeze_manifest_path: Path,
    point_freeze_manifest_sha256: str,
    base_state: dict | None = None,
) -> dict | None:
    if not progress["target_reached"] and not progress["attempt_cap_reached"]:
        return None
    passed = progress["target_reached"]
    state = dict(base_state or {})
    state.update(
        {
            "status": (
                "v6_6_fresh_calibration_collection_complete"
                if passed
                else "v6_6_fresh_calibration_collection_exhausted_fail_closed"
            ),
            "v6_6_point_freeze_manifest_path": str(point_freeze_manifest_path),
            "v6_6_point_freeze_manifest_sha256": point_freeze_manifest_sha256,
            "v6_6_fresh_calibration_boundary_sequence": plan["collector_index_boundary_sequence"],
            "v6_6_fresh_calibration_attempted_market_count": progress["attempted_market_count"],
            "v6_6_fresh_calibration_quality_valid_market_count": progress[
                "quality_valid_market_count"
            ],
            "v6_6_fresh_calibration_target_quality_valid_market_count": plan[
                "target_quality_valid_market_count"
            ],
            "v6_6_fresh_calibration_maximum_attempted_market_count": plan[
                "maximum_attempted_market_count"
            ],
            "v6_6_fresh_calibration_collection_complete": passed,
            "v6_6_fresh_calibration_first_selected_market_ids": progress[
                "first_selected_quality_market_ids"
            ],
            "v6_6_candidate_scoring_attempted": False,
            "labels_outcomes_or_pnl_opened": False,
            "blocking_reason_codes": (
                [] if passed else ["v6_6_fresh_calibration_support_not_met_before_attempt_cap"]
            ),
            **SAFETY,
        }
    )
    return state


def _write_state(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_batch_canary_passed(canary_result: dict) -> None:
    report = dict(canary_result.get("report") or {})
    if report.get("development_data_canary_passed") is not True:
        raise OutcomeBlindBatchCanaryFailure(
            "completed batch failed outcome-blind canary: "
            + ",".join(report.get("development_data_canary_blocking_reason_codes") or [])
        )


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
