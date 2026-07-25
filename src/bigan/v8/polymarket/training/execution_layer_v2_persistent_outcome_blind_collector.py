"""Append-only raw collection index and immutable outcome-blind window freezes."""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.recorder.contracts import (
    DEFAULT_SAMPLING_POLICY_SECONDS,
)
from bigan.v8.polymarket.recorder.orderbook_state import (
    full_decision_window_orderbook_coverage,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_future_unseen_holdout import (
    COLLECTION_FREEZE_MANIFEST_SCHEMA_VERSION,
    load_and_validate_pairwise_future_unseen_collection_freeze,
)

PROTOCOL_SCHEMA_VERSION = "bigan-v8-persistent-outcome-blind-collector-protocol-v1"
INDEX_ENTRY_SCHEMA_VERSION = "bigan-v8-persistent-outcome-blind-round-index-entry-v1"
BATCH_REPORT_SCHEMA_VERSION = "bigan-v8-persistent-outcome-blind-batch-report-v1"
BATCH_MANIFEST_SCHEMA_VERSION = "bigan-v8-persistent-outcome-blind-batch-manifest-v1"
STATE_SCHEMA_VERSION = "bigan-v8-persistent-outcome-blind-collector-state-v1"
WINDOW_REPORT_SCHEMA_VERSION = "bigan-v8-outcome-blind-window-freeze-report-v2"
WINDOW_MANIFEST_SCHEMA_VERSION = "bigan-v8-outcome-blind-window-freeze-manifest-v2"
SOURCE_BOUNDARY_SCHEMA_VERSION = "bigan-v8-outcome-blind-source-boundary-v1"
ZERO_SHA256 = "0" * 64
FORBIDDEN_RAW_FIELDS = {
    "accepted_bet_net_pnl",
    "evaluation_target_net_pnl_per_contract_by_action",
    "evaluation_target_net_return_after_cost_by_action",
    "final_outcome",
    "future_price",
    "future_return",
    "gross_pnl",
    "label",
    "market_resolved",
    "net_pnl",
    "oracle_action",
    "realized_pnl",
    "resolved_outcome",
    "settlement_outcome",
    "settlement_pnl",
    "settlement_return",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
    "winning_outcome",
    "winning_token_id",
}


@dataclass(frozen=True, slots=True)
class PersistentOutcomeBlindBatchIndexConfig:
    """Pin and append one completed collection-only batch."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    index_path: Path | str
    batch_summary_path: Path | str
    expected_batch_summary_sha256: str
    collector_git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if re.fullmatch(r"[0-9a-fA-F]{40}", self.collector_git_commit) is None:
            raise ValueError("collector_git_commit must be a 40-character hex digest")
        for field in (
            "output_dir",
            "protocol_path",
            "index_path",
            "batch_summary_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        for field in (
            "expected_protocol_sha256",
            "expected_batch_summary_sha256",
        ):
            _require_sha256(getattr(self, field), name=field)
            object.__setattr__(self, field, getattr(self, field).lower())
        object.__setattr__(self, "collector_git_commit", self.collector_git_commit.lower())


@dataclass(frozen=True, slots=True)
class OutcomeBlindWindowFreezeConfig:
    """Freeze the earliest valid rows after a pinned source boundary."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    index_path: Path | str
    expected_index_sha256: str
    source_boundary_manifest_path: Path | str
    expected_source_boundary_manifest_sha256: str
    target_valid_market_count: int
    maximum_scan_count: int
    builder_git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.target_valid_market_count <= 0:
            raise ValueError("target_valid_market_count must be positive")
        if self.maximum_scan_count < self.target_valid_market_count:
            raise ValueError("maximum_scan_count must be at least the target")
        if re.fullmatch(r"[0-9a-fA-F]{40}", self.builder_git_commit) is None:
            raise ValueError("builder_git_commit must be a 40-character hex digest")
        for field in (
            "output_dir",
            "protocol_path",
            "index_path",
            "source_boundary_manifest_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        for field in (
            "expected_protocol_sha256",
            "expected_index_sha256",
            "expected_source_boundary_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), name=field)
            object.__setattr__(self, field, getattr(self, field).lower())
        object.__setattr__(self, "builder_git_commit", self.builder_git_commit.lower())


def _serialize_index_update(function: Any) -> Any:
    """Serialize index updates across restarted or accidentally duplicated services."""

    @wraps(function)
    def wrapped(config: PersistentOutcomeBlindBatchIndexConfig) -> dict[str, Any]:
        index_path = config.index_path.resolve()
        lock_path = index_path.with_name(f"{index_path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                return function(config)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    return wrapped


def validate_persistent_outcome_blind_collector_protocol(
    protocol: dict[str, Any],
) -> None:
    """Reject any collection contract that can observe outcomes or control by results."""

    expected = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "frozen": True,
        "market_family": "btc_updown_5m",
        "collection_mode": "raw_pending_capture_only",
        "outcome_blind_collection_only": True,
        "settlement_finalizer_enabled": False,
        "resolution_provider_enabled": False,
        "training_corpus_export_enabled": False,
        "labels_outcomes_or_pnl_opened": False,
        "resolution_artifact": "raw_polymarket_resolutions.jsonl",
        "resolution_artifact_required_row_count": 0,
        "github_comment_policy": "bounded_batch_summary_only",
        "raw_collection_exported_to_direct_training_corpus": False,
        **_blocked_safety_fields(),
    }
    blockers = [key for key, value in expected.items() if protocol.get(key) != value]
    if int(protocol.get("default_batch_round_count") or 0) <= 0:
        blockers.append("default_batch_round_count")
    if set(protocol.get("required_raw_artifacts") or []) != {
        "raw_polymarket_markets.jsonl",
        "raw_polymarket_orderbooks.jsonl",
        "raw_polymarket_trades.jsonl",
        "raw_binance_btcusdt_klines.jsonl",
        "raw_polymarket_chainlink_prices.jsonl",
    }:
        blockers.append("required_raw_artifacts")
    forbidden_control = set(protocol.get("forbidden_collection_control_inputs") or [])
    if forbidden_control != {
        "model_score",
        "accepted_bet_count",
        "settlement_outcome",
        "realized_pnl",
        "oracle_action",
        "future_return",
    }:
        blockers.append("forbidden_collection_control_inputs")
    if set(protocol.get("forbidden_nonempty_raw_fields") or []) != FORBIDDEN_RAW_FIELDS:
        blockers.append("forbidden_nonempty_raw_fields")
    index_contract = dict(protocol.get("append_only_index") or {})
    if index_contract != {
        "hash_chain_required": True,
        "sequence_starts_at": 1,
        "deduplicate_identity_fields": [
            "scheduled_round_start_ts",
            "market_id",
            "slug",
            "run_id",
            "decision_id",
            "source_row_hash",
        ],
        "failed_captures_retained_with_reason_codes": True,
        "existing_rows_rewritten_allowed": False,
    }:
        blockers.append("append_only_index")
    window = dict(protocol.get("window_freeze") or {})
    if (
        any(
            window.get(field) is not True
            for field in (
                "minimum_collection_decision_ts_required",
                "market_id_disjointness_required",
                "slug_disjointness_required",
                "source_row_hash_disjointness_required",
                "index_hash_pin_required",
                "target_and_maximum_scan_count_frozen_before_selection",
            )
        )
        or window.get("labels_outcomes_or_pnl_opened_for_selection") is not False
    ):
        blockers.append("window_freeze")
    if blockers:
        raise ValueError(
            "persistent outcome-blind collector protocol validation failed: "
            + ", ".join(sorted(set(blockers)))
        )


@_serialize_index_update
def index_persistent_outcome_blind_batch(
    config: PersistentOutcomeBlindBatchIndexConfig,
) -> dict[str, Any]:
    """Append unseen round captures to a hash-chained JSONL index."""

    protocol_path = config.protocol_path.resolve()
    batch_summary_path = config.batch_summary_path.resolve()
    _verify_pin(protocol_path, config.expected_protocol_sha256, name="collector protocol")
    _verify_pin(
        batch_summary_path,
        config.expected_batch_summary_sha256,
        name="outcome-blind batch summary",
    )
    protocol = _load_json(protocol_path)
    validate_persistent_outcome_blind_collector_protocol(protocol)
    summary = _load_json(batch_summary_path)
    _validate_outcome_blind_batch_summary(summary)
    run_dir = config.output_dir / config.run_id
    if run_dir.exists():
        raise ValueError(f"batch index run directory already exists: {run_dir}")
    index_path = config.index_path.resolve()
    existing = load_and_validate_persistent_outcome_blind_index(index_path)
    existing_run_ids = {str(row["run_id"]) for row in existing}
    existing_boundaries = {
        int(row["scheduled_round_start_ts"])
        for row in existing
        if int(row.get("scheduled_round_start_ts") or 0) > 0
    }
    existing_market_ids = {
        str(row.get("market_id") or "") for row in existing if row.get("market_id")
    }
    existing_slugs = {str(row.get("slug") or "") for row in existing if row.get("slug")}
    existing_decision_ids = {
        str(row.get("decision_id") or "") for row in existing if row.get("decision_id")
    }
    existing_source_row_hashes = {
        str(row.get("source_row_hash") or "") for row in existing if row.get("source_row_hash")
    }
    previous_sha = str(existing[-1]["entry_sha256"]) if existing else ZERO_SHA256
    next_sequence = len(existing) + 1
    appended: list[dict[str, Any]] = []
    idempotent_skips: list[str] = []
    duplicate_reasons: Counter[str] = Counter()
    captures = sorted(
        summary.get("captures") or [],
        key=lambda row: (
            int(row.get("scheduled_round_start_ts") or 0),
            int(row.get("round_index") or 0),
            str(row.get("run_id") or ""),
        ),
    )
    for capture in captures:
        run_id = str(capture.get("run_id") or "")
        if not run_id:
            raise ValueError("batch capture run_id is missing")
        if run_id in existing_run_ids:
            idempotent_skips.append(run_id)
            continue
        entry = _capture_index_entry(
            capture,
            protocol=protocol,
            batch_summary_path=batch_summary_path,
            batch_id=str(summary.get("batch_id") or ""),
            collector_git_commit=config.collector_git_commit,
            sequence=next_sequence,
            previous_entry_sha256=previous_sha,
            existing_boundaries=existing_boundaries,
            existing_market_ids=existing_market_ids,
            existing_slugs=existing_slugs,
            existing_decision_ids=existing_decision_ids,
            existing_source_row_hashes=existing_source_row_hashes,
        )
        for reason in entry["duplicate_identity_reason_codes"]:
            duplicate_reasons[reason] += 1
        _append_jsonl(index_path, entry)
        appended.append(entry)
        previous_sha = str(entry["entry_sha256"])
        next_sequence += 1
        existing_run_ids.add(run_id)
        boundary = int(entry.get("scheduled_round_start_ts") or 0)
        if boundary > 0:
            existing_boundaries.add(boundary)
        if entry.get("market_id"):
            existing_market_ids.add(str(entry["market_id"]))
        if entry.get("slug"):
            existing_slugs.add(str(entry["slug"]))
        existing_decision_ids.add(str(entry["decision_id"]))
        existing_source_row_hashes.add(str(entry["source_row_hash"]))

    failed_captures = sorted(
        summary.get("errors") or [],
        key=lambda row: (
            int(row.get("scheduled_round_start_ts") or 0),
            int(row.get("round_index") or 0),
            str(row.get("run_id") or ""),
        ),
    )
    for failure in failed_captures:
        run_id = str(failure.get("run_id") or Path(str(failure.get("run_dir") or "")).name)
        if not run_id:
            raise ValueError("failed batch capture run_id is missing")
        if run_id in existing_run_ids:
            idempotent_skips.append(run_id)
            continue
        entry = _failed_capture_index_entry(
            failure,
            batch_summary_path=batch_summary_path,
            batch_id=str(summary.get("batch_id") or ""),
            collector_git_commit=config.collector_git_commit,
            sequence=next_sequence,
            previous_entry_sha256=previous_sha,
            existing_boundaries=existing_boundaries,
            existing_decision_ids=existing_decision_ids,
            existing_source_row_hashes=existing_source_row_hashes,
        )
        for reason in entry["duplicate_identity_reason_codes"]:
            duplicate_reasons[reason] += 1
        _append_jsonl(index_path, entry)
        appended.append(entry)
        previous_sha = str(entry["entry_sha256"])
        next_sequence += 1
        existing_run_ids.add(run_id)
        boundary = int(entry.get("scheduled_round_start_ts") or 0)
        if boundary > 0:
            existing_boundaries.add(boundary)
        existing_decision_ids.add(str(entry["decision_id"]))
        existing_source_row_hashes.add(str(entry["source_row_hash"]))

    all_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    run_dir.mkdir(parents=True, exist_ok=False)
    batch_window_coverage_rows = [
        dict(row["orderbook_window_coverage"])
        for row in appended
        if isinstance(row.get("orderbook_window_coverage"), dict)
    ]
    batch_window_reason_counts: Counter[str] = Counter()
    for coverage in batch_window_coverage_rows:
        batch_window_reason_counts.update(
            str(reason)
            for reason in coverage.get(
                "orderbook_window_coverage_reason_codes"
            )
            or []
        )
    report = {
        "schema_version": BATCH_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "batch_id": summary.get("batch_id"),
        "collector_git_commit": config.collector_git_commit,
        "batch_capture_count": len(captures),
        "batch_failed_capture_count": len(failed_captures),
        "batch_attempt_count": len(captures) + len(failed_captures),
        "appended_entry_count": len(appended),
        "idempotent_existing_run_count": len(idempotent_skips),
        "idempotent_existing_run_ids": sorted(idempotent_skips),
        "duplicate_identity_reason_distribution": dict(sorted(duplicate_reasons.items())),
        "index_entry_count": len(all_rows),
        "quality_valid_index_entry_count": sum(
            row.get("capture_quality_valid") is True for row in all_rows
        ),
        "batch_orderbook_full_window_coverage_passed_count": sum(
            coverage.get(
                "orderbook_full_decision_window_coverage_passed"
            )
            is True
            for coverage in batch_window_coverage_rows
        ),
        "batch_orderbook_full_window_coverage_failed_count": sum(
            coverage.get(
                "orderbook_full_decision_window_coverage_passed"
            )
            is not True
            for coverage in batch_window_coverage_rows
        ),
        "batch_orderbook_expected_decision_pair_count": sum(
            int(
                coverage.get(
                    "orderbook_expected_decision_pair_count"
                )
                or 0
            )
            for coverage in batch_window_coverage_rows
        ),
        "batch_orderbook_observed_decision_pair_count": sum(
            int(
                coverage.get(
                    "orderbook_observed_decision_pair_count"
                )
                or 0
            )
            for coverage in batch_window_coverage_rows
        ),
        "batch_orderbook_window_coverage_reason_distribution": dict(
            sorted(batch_window_reason_counts.items())
        ),
        "index_chain_validation_passed": True,
        "outcome_blind_collection_only": True,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        "training_corpus_export_attempted": False,
        "labels_outcomes_or_pnl_opened": False,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report_path = run_dir / "persistent_outcome_blind_batch_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "persistent_outcome_blind_batch_report.md",
        _batch_markdown(report),
    )
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "protocol": _descriptor(protocol_path),
        "index": _descriptor(index_path),
        "index_entry_count": len(all_rows),
        "quality_valid_index_entry_count": report["quality_valid_index_entry_count"],
        "last_sequence": len(all_rows),
        "last_entry_sha256": previous_sha,
        "last_batch_summary": _descriptor(batch_summary_path),
        "last_batch_report": _descriptor(report_path),
        "outcome_blind_collection_only": True,
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    state_path = index_path.with_name("persistent_outcome_blind_collector_state.json")
    _write_json(state_path, state)
    manifest = {
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "protocol": _descriptor(protocol_path),
        "batch_summary": _descriptor(batch_summary_path),
        "report": _descriptor(report_path),
        "state": _descriptor(state_path),
        "index": _descriptor(index_path),
        "index_entry_count": len(all_rows),
        "last_entry_sha256": previous_sha,
        "outcome_blind_collection_only": True,
        "labels_outcomes_or_pnl_opened": False,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    manifest["batch_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "persistent_outcome_blind_batch_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "state_path": state_path,
        "state_sha256": _sha256_file(state_path),
        "index_path": index_path,
        "index_sha256": _sha256_file(index_path),
        "report": report,
        "manifest": manifest,
    }


def freeze_outcome_blind_window(
    config: OutcomeBlindWindowFreezeConfig,
) -> dict[str, Any]:
    """Select a pre-sized chronological raw window without opening any target."""

    protocol_path = config.protocol_path.resolve()
    index_path = config.index_path.resolve()
    boundary_path = config.source_boundary_manifest_path.resolve()
    _verify_pin(protocol_path, config.expected_protocol_sha256, name="collector protocol")
    _verify_pin(index_path, config.expected_index_sha256, name="collector index")
    _verify_pin(
        boundary_path,
        config.expected_source_boundary_manifest_sha256,
        name="source boundary manifest",
    )
    protocol = _load_json(protocol_path)
    validate_persistent_outcome_blind_collector_protocol(protocol)
    boundary = _load_json(boundary_path)
    _validate_source_boundary_manifest(
        boundary,
        path=boundary_path,
        expected_sha256=config.expected_source_boundary_manifest_sha256,
    )
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    index_snapshot_path = run_dir / "persistent_outcome_blind_round_index_snapshot.jsonl"
    index_snapshot_path.write_bytes(index_path.read_bytes())
    if _sha256_file(index_snapshot_path) != config.expected_index_sha256:
        raise ValueError("collector index changed while immutable snapshot was created")
    rows = load_and_validate_persistent_outcome_blind_index(index_snapshot_path)
    minimum_ts, prior_market_ids, prior_slugs, prior_row_hashes = _boundary_references(boundary)
    ordered_after_boundary = [
        row for row in rows if int(row.get("scheduled_round_start_ts") or 0) >= minimum_ts
    ]
    scanned = ordered_after_boundary[: config.maximum_scan_count]
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    selected_market_ids: set[str] = set()
    selected_slugs: set[str] = set()
    evaluated_entry_count = 0
    for row in scanned:
        evaluated_entry_count += 1
        reasons: list[str] = []
        market_id = str(row.get("market_id") or "")
        slug = str(row.get("slug") or "")
        source_row_hash = str(row.get("source_row_hash") or "")
        if not market_id:
            reasons.append("market_id_missing")
        if row.get("capture_quality_valid") is not True:
            reasons.extend(row.get("capture_quality_reason_codes") or [])
        if market_id in prior_market_ids:
            reasons.append("market_id_overlaps_source_boundary")
        if slug and slug in prior_slugs:
            reasons.append("slug_overlaps_source_boundary")
        if source_row_hash in prior_row_hashes:
            reasons.append("source_row_hash_overlaps_source_boundary")
        if market_id in selected_market_ids:
            reasons.append("duplicate_market_id_within_window")
        if slug and slug in selected_slugs:
            reasons.append("duplicate_slug_within_window")
        if row.get("duplicate_identity_reason_codes"):
            reasons.extend(row["duplicate_identity_reason_codes"])
        try:
            _verify_index_raw_descriptors(row)
        except ValueError as exc:
            reasons.append(f"raw_artifact_lineage_invalid:{exc}")
        if reasons:
            exclusions.append(
                {
                    "sequence": row["sequence"],
                    "run_id": row["run_id"],
                    "market_id": market_id,
                    "reason_codes": sorted(set(reasons)),
                }
            )
            continue
        selected.append(row)
        selected_market_ids.add(market_id)
        if slug:
            selected_slugs.add(slug)
        if len(selected) >= config.target_valid_market_count:
            break
    ready = len(selected) == config.target_valid_market_count
    blockers = [] if ready else ["insufficient_quality_valid_markets_before_scan_cap"]
    selected_start_ts = (
        min(int(row["scheduled_round_start_ts"]) for row in selected) if selected else None
    )
    selected_end_ts = (
        max(int(row["scheduled_round_start_ts"]) for row in selected) if selected else None
    )
    prior_reference_hash = canonical_json_sha256(
        {
            "minimum_collection_decision_ts": minimum_ts,
            "prior_market_ids": sorted(prior_market_ids),
            "prior_slugs": sorted(prior_slugs),
            "prior_source_row_hashes": sorted(prior_row_hashes),
        }
    )
    selected_path = run_dir / "outcome_blind_window_selected_rows.jsonl"
    _write_jsonl(selected_path, selected)
    report = {
        "schema_version": WINDOW_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "source_boundary_schema_version": boundary["schema_version"],
        "minimum_collection_decision_ts": minimum_ts,
        "target_valid_market_count": config.target_valid_market_count,
        "maximum_scan_count": config.maximum_scan_count,
        "available_after_boundary_count": len(ordered_after_boundary),
        "scan_pool_entry_count": len(scanned),
        "scanned_entry_count": evaluated_entry_count,
        "selected_market_count": len(selected),
        "excluded_entry_count": len(exclusions),
        "selected_market_ids": [str(row["market_id"]) for row in selected],
        "selected_decision_ids": [str(row["decision_id"]) for row in selected],
        "selected_window_start_ts": selected_start_ts,
        "selected_window_end_ts": selected_end_ts,
        "prior_reference_hash": prior_reference_hash,
        "collector_index_snapshot_sha256": _sha256_file(index_snapshot_path),
        "collector_index_snapshot_immutable": True,
        "window_selection_used_immutable_index_snapshot": True,
        "selected_market_ids_sha256": canonical_json_sha256(
            [str(row["market_id"]) for row in selected]
        ),
        "selected_source_row_hashes_sha256": canonical_json_sha256(
            [str(row["source_row_hash"]) for row in selected]
        ),
        "exclusion_reason_distribution": dict(
            sorted(Counter(reason for row in exclusions for reason in row["reason_codes"]).items())
        ),
        "selection_method": "earliest_quality_valid_strictly_later_disjoint_rows",
        "window_freeze_ready": ready,
        "labels_outcomes_or_pnl_opened_for_selection": False,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    report_path = run_dir / "outcome_blind_window_freeze_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "outcome_blind_window_freeze_report.md",
        _window_markdown(report),
    )
    manifest = {
        "schema_version": WINDOW_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "source_boundary_schema_version": boundary["schema_version"],
        "protocol": _descriptor(protocol_path),
        "index": _descriptor(index_snapshot_path),
        "source_index_pin_at_freeze": {
            "path": str(index_path),
            "sha256": config.expected_index_sha256,
        },
        "source_boundary_manifest": _descriptor(boundary_path),
        "selected_rows": _descriptor(selected_path),
        "report": _descriptor(report_path),
        "target_valid_market_count": config.target_valid_market_count,
        "maximum_scan_count": config.maximum_scan_count,
        "selected_market_count": len(selected),
        "selected_window_start_ts": selected_start_ts,
        "selected_window_end_ts": selected_end_ts,
        "prior_reference_hash": prior_reference_hash,
        "collector_index_snapshot_immutable": True,
        "window_selection_used_immutable_index_snapshot": True,
        "window_freeze_ready": ready,
        "labels_outcomes_or_pnl_opened_for_selection": False,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    manifest["window_freeze_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "outcome_blind_window_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "selected_rows_path": selected_path,
        "index_snapshot_path": index_snapshot_path,
        "index_snapshot_sha256": _sha256_file(index_snapshot_path),
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
    }


def load_and_validate_persistent_outcome_blind_index(
    index_path: Path | str,
) -> list[dict[str, Any]]:
    """Validate sequence and hash-chain integrity for every existing entry."""

    path = Path(index_path)
    if not path.exists():
        return []
    rows = _load_jsonl(path)
    previous = ZERO_SHA256
    for expected_sequence, row in enumerate(rows, start=1):
        if row.get("schema_version") != INDEX_ENTRY_SCHEMA_VERSION:
            raise ValueError("index_entry_schema_invalid")
        if int(row.get("sequence") or 0) != expected_sequence:
            raise ValueError("index_sequence_invalid")
        if row.get("previous_entry_sha256") != previous:
            raise ValueError("index_previous_hash_mismatch")
        expected_hash = canonical_json_sha256(
            {key: value for key, value in row.items() if key != "entry_sha256"}
        )
        if row.get("entry_sha256") != expected_hash:
            raise ValueError("index_entry_hash_mismatch")
        if row.get("labels_outcomes_or_pnl_opened") is not False:
            raise ValueError("index_entry_outcome_sealing_invalid")
        for field in ("decision_id", "source_row_hash", "entry_sha256"):
            try:
                _require_sha256(str(row.get(field) or ""), name=field)
            except ValueError as exc:
                raise ValueError(f"index_{field}_invalid") from exc
        safety_mismatches = [
            field
            for field, expected in _blocked_safety_fields().items()
            if row.get(field) != expected
        ]
        if safety_mismatches:
            raise ValueError("index_entry_safety_invalid:" + ",".join(safety_mismatches))
        previous = expected_hash
    return rows


def _capture_index_entry(
    capture: dict[str, Any],
    *,
    protocol: dict[str, Any],
    batch_summary_path: Path,
    batch_id: str,
    collector_git_commit: str,
    sequence: int,
    previous_entry_sha256: str,
    existing_boundaries: set[int],
    existing_market_ids: set[str],
    existing_slugs: set[str],
    existing_decision_ids: set[str],
    existing_source_row_hashes: set[str],
) -> dict[str, Any]:
    run_dir = Path(str(capture.get("run_dir") or "")).resolve()
    manifest_path = run_dir / "pending_round_capture_manifest.json"
    report_path = run_dir / "pending_round_capture_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError(f"pending capture evidence missing: {run_dir}")
    manifest = _load_json(manifest_path)
    _load_json(report_path)
    reasons: list[str] = []
    if manifest.get("resolution_provider_called") is not False:
        reasons.append("resolution_provider_called_during_capture")
    if (
        int(
            (manifest.get("raw_artifact_row_counts") or {}).get("raw_polymarket_resolutions.jsonl")
            or 0
        )
        != 0
    ):
        reasons.append("resolution_rows_present_during_capture")
    if manifest.get("pending_resolution") is not True:
        reasons.append("pending_resolution_contract_invalid")
    if manifest.get("paper_only") is not True or manifest.get("capital_at_risk") is not False:
        reasons.append("capture_safety_contract_invalid")
    raw_dir = run_dir / "raw"
    raw_descriptors: dict[str, dict[str, Any]] = {}
    raw_payloads: dict[str, list[dict[str, Any]]] = {}
    manifest_hashes = dict(manifest.get("raw_artifact_hashes") or {})
    for filename in protocol["required_raw_artifacts"]:
        path = raw_dir / filename
        if not path.is_file():
            reasons.append(f"required_raw_artifact_missing:{filename}")
            continue
        rows = _load_jsonl(path)
        digest = _sha256_file(path)
        expected_digest = (
            str(manifest.get("chainlink_raw_artifact_sha256") or "")
            if filename == "raw_polymarket_chainlink_prices.jsonl"
            else str(manifest_hashes.get(filename) or "")
        )
        if digest != expected_digest:
            reasons.append(f"required_raw_artifact_hash_mismatch:{filename}")
        expected_row_count = (
            int(manifest.get("chainlink_raw_artifact_row_count") or 0)
            if filename == "raw_polymarket_chainlink_prices.jsonl"
            else int((manifest.get("raw_artifact_row_counts") or {}).get(filename) or 0)
        )
        if expected_row_count != len(rows):
            reasons.append(f"required_raw_artifact_row_count_mismatch:{filename}")
        raw_payloads[filename] = rows
        raw_descriptors[filename] = {
            "path": str(path),
            "sha256": digest,
            "row_count": len(rows),
        }
    resolution_filename = str(protocol["resolution_artifact"])
    resolution_path = raw_dir / resolution_filename
    resolution_rows: list[dict[str, Any]] = []
    if not resolution_path.is_file():
        reasons.append("resolution_artifact_missing_for_zero_row_proof")
    else:
        resolution_rows = _load_jsonl(resolution_path)
        resolution_digest = _sha256_file(resolution_path)
        if resolution_digest != str(manifest_hashes.get(resolution_filename) or ""):
            reasons.append("resolution_artifact_hash_mismatch")
        if resolution_rows:
            reasons.append("resolution_rows_present_during_capture")
        raw_payloads[resolution_filename] = resolution_rows
        raw_descriptors[resolution_filename] = {
            "path": str(resolution_path),
            "sha256": resolution_digest,
            "row_count": len(resolution_rows),
        }
    if int((manifest.get("raw_artifact_row_counts") or {}).get(resolution_filename) or 0) != len(
        resolution_rows
    ):
        reasons.append("resolution_artifact_row_count_mismatch")
    markets = raw_payloads.get("raw_polymarket_markets.jsonl") or []
    orderbooks = raw_payloads.get("raw_polymarket_orderbooks.jsonl") or []
    candles = raw_payloads.get("raw_binance_btcusdt_klines.jsonl") or []
    chainlink_rows = raw_payloads.get("raw_polymarket_chainlink_prices.jsonl") or []
    if len(markets) != 1:
        reasons.append("raw_market_identity_count_invalid")
    market = markets[0] if len(markets) == 1 else {}
    market_id = str(market.get("market_id") or capture.get("market_id") or "")
    slug = str(market.get("slug") or "")
    if not market_id:
        reasons.append("market_id_missing")
    boundary = int(capture.get("scheduled_round_start_ts") or 0)
    if boundary <= 0:
        reasons.append("scheduled_round_start_ts_missing")
    if capture.get("capture_start_boundary_validation_passed") is not True:
        reasons.append("capture_start_boundary_failed")
    if int(capture.get("provider_raw_orderbook_snapshot_count") or 0) <= 0:
        reasons.append("provider_orderbook_snapshot_coverage_failed")
    if int(capture.get("training_sampled_orderbook_row_count") or 0) <= 0 or not orderbooks:
        reasons.append("sampled_orderbook_coverage_failed")
    orderbook_outcomes = {
        str(row.get("outcome") or "").upper() for row in orderbooks if row.get("outcome")
    }
    if not {"UP", "DOWN"}.issubset(orderbook_outcomes):
        reasons.append("executable_up_down_orderbook_coverage_failed")
    market_family = str(
        market.get("market_family")
        or protocol.get("market_family")
        or ""
    )
    sample_interval_seconds = DEFAULT_SAMPLING_POLICY_SECONDS.get(
        market_family
    )
    if not market or sample_interval_seconds is None:
        orderbook_window_coverage = {
            "orderbook_full_decision_window_coverage_passed": False,
            "orderbook_window_coverage_reason_codes": [
                "orderbook_window_coverage_contract_unavailable"
            ],
            "orderbook_expected_decision_pair_count": 0,
            "orderbook_observed_decision_pair_count": 0,
            "orderbook_last_required_decision_ts": None,
            "orderbook_latest_covered_decision_ts": None,
            "orderbook_observed_collection_end_ts": None,
        }
    else:
        orderbook_window_coverage = (
            full_decision_window_orderbook_coverage(
                market=market,
                book_rows=orderbooks,
                sample_interval_seconds=sample_interval_seconds,
            )
        )
    if (
        orderbook_window_coverage[
            "orderbook_full_decision_window_coverage_passed"
        ]
        is not True
    ):
        reasons.append("orderbook_full_decision_window_coverage_failed")
        reasons.extend(
            str(reason)
            for reason in orderbook_window_coverage[
                "orderbook_window_coverage_reason_codes"
            ]
        )
    if int(capture.get("raw_btc_candle_row_count") or 0) <= 0 or not candles:
        reasons.append("btc_candle_coverage_failed")
    if int(capture.get("raw_chainlink_price_row_count") or 0) <= 0 or not chainlink_rows:
        reasons.append("chainlink_rtds_coverage_failed")
    if any(
        int(row.get("source_ts") or 0) > int(row.get("available_at_ts") or 0)
        for row in chainlink_rows
    ):
        reasons.append("chainlink_timestamp_causality_violation")
    if int(capture.get("market_identity_cache_provenance_violation_count") or 0) != 0:
        reasons.append("market_identity_provenance_violation")
    forbidden = set(protocol.get("forbidden_nonempty_raw_fields") or [])
    forbidden_hits = sorted(
        {
            field
            for payload in raw_payloads.values()
            for row in payload
            for field in _find_nonempty_fields(row, forbidden)
        }
    )
    if forbidden_hits:
        reasons.extend(f"forbidden_raw_field:{field}" for field in forbidden_hits)
    decision_id = canonical_json_sha256(
        {
            "scheduled_round_start_ts": boundary,
            "market_id": market_id,
            "slug": slug,
        }
    )
    source_row_hash = canonical_json_sha256(
        {
            "scheduled_round_start_ts": boundary,
            "market_id": market_id,
            "slug": slug,
            "raw_artifacts": raw_descriptors,
        }
    )
    duplicate_reasons: list[str] = []
    if boundary in existing_boundaries:
        duplicate_reasons.append("duplicate_scheduled_round_start_ts")
    if market_id and market_id in existing_market_ids:
        duplicate_reasons.append("duplicate_market_id")
    if slug and slug in existing_slugs:
        duplicate_reasons.append("duplicate_slug")
    if decision_id in existing_decision_ids:
        duplicate_reasons.append("duplicate_decision_id")
    if source_row_hash in existing_source_row_hashes:
        duplicate_reasons.append("duplicate_source_row_hash")
    if duplicate_reasons:
        reasons.extend(duplicate_reasons)
    entry = {
        "schema_version": INDEX_ENTRY_SCHEMA_VERSION,
        "sequence": sequence,
        "previous_entry_sha256": previous_entry_sha256,
        "batch_id": batch_id,
        "run_id": str(capture.get("run_id") or ""),
        "collector_git_commit": collector_git_commit,
        "scheduled_round_start_ts": boundary,
        "market_start_ts": int(market.get("market_start_ts") or 0),
        "market_end_ts": int(market.get("market_end_ts") or 0),
        "market_id": market_id,
        "slug": slug,
        "decision_id": decision_id,
        "source_row_hash": source_row_hash,
        "capture_quality_valid": not reasons,
        "capture_quality_reason_codes": sorted(set(reasons)),
        "duplicate_identity_reason_codes": sorted(set(duplicate_reasons)),
        "pending_round_capture_manifest": _descriptor(manifest_path),
        "pending_round_capture_report": _descriptor(report_path),
        "batch_summary": _descriptor(batch_summary_path),
        "raw_artifacts": raw_descriptors,
        "orderbook_window_coverage": orderbook_window_coverage,
        "raw_resolution_row_count": len(resolution_rows),
        "resolution_provider_called": False,
        "settlement_finalizer_started": False,
        "training_corpus_export_attempted": False,
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    entry["entry_sha256"] = canonical_json_sha256(entry)
    return entry


def _failed_capture_index_entry(
    failure: dict[str, Any],
    *,
    batch_summary_path: Path,
    batch_id: str,
    collector_git_commit: str,
    sequence: int,
    previous_entry_sha256: str,
    existing_boundaries: set[int],
    existing_decision_ids: set[str],
    existing_source_row_hashes: set[str],
) -> dict[str, Any]:
    """Retain a failed attempt in the chain without treating it as usable evidence."""

    run_id = str(failure.get("run_id") or Path(str(failure.get("run_dir") or "")).name)
    boundary = int(failure.get("scheduled_round_start_ts") or 0)
    decision_id = canonical_json_sha256(
        {
            "scheduled_round_start_ts": boundary,
            "run_id": run_id,
            "round_index": int(failure.get("round_index") or 0),
        }
    )
    source_row_hash = canonical_json_sha256(
        {
            "batch_id": batch_id,
            "run_id": run_id,
            "scheduled_round_start_ts": boundary,
            "capture_failure_stage": str(failure.get("stage") or "unknown"),
            "capture_failure_type": str(failure.get("error_type") or "unknown"),
        }
    )
    duplicate_reasons: list[str] = []
    if boundary > 0 and boundary in existing_boundaries:
        duplicate_reasons.append("duplicate_scheduled_round_start_ts")
    if decision_id in existing_decision_ids:
        duplicate_reasons.append("duplicate_decision_id")
    if source_row_hash in existing_source_row_hashes:
        duplicate_reasons.append("duplicate_source_row_hash")
    quality_reasons = [
        "round_capture_failed",
        f"round_capture_failed_stage:{str(failure.get('stage') or 'unknown')}",
        f"round_capture_failed_type:{str(failure.get('error_type') or 'unknown')}",
        "raw_evidence_unavailable_for_failed_capture",
    ]
    if boundary <= 0:
        quality_reasons.append("scheduled_round_start_ts_missing")
    quality_reasons.extend(duplicate_reasons)
    entry = {
        "schema_version": INDEX_ENTRY_SCHEMA_VERSION,
        "sequence": sequence,
        "previous_entry_sha256": previous_entry_sha256,
        "batch_id": batch_id,
        "run_id": run_id,
        "collector_git_commit": collector_git_commit,
        "scheduled_round_start_ts": boundary,
        "market_start_ts": 0,
        "market_end_ts": 0,
        "market_id": "",
        "slug": "",
        "decision_id": decision_id,
        "source_row_hash": source_row_hash,
        "capture_quality_valid": False,
        "capture_quality_reason_codes": sorted(set(quality_reasons)),
        "duplicate_identity_reason_codes": sorted(set(duplicate_reasons)),
        "capture_failure": {
            "stage": str(failure.get("stage") or "unknown"),
            "error_type": str(failure.get("error_type") or "unknown"),
            "error": str(failure.get("error") or ""),
            "run_dir": str(failure.get("run_dir") or ""),
        },
        "batch_summary": _descriptor(batch_summary_path),
        "raw_artifacts": {},
        "raw_resolution_row_count": 0,
        "resolution_provider_called": False,
        "settlement_finalizer_started": False,
        "training_corpus_export_attempted": False,
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    entry["entry_sha256"] = canonical_json_sha256(entry)
    return entry


def _validate_outcome_blind_batch_summary(summary: dict[str, Any]) -> None:
    expected = {
        "outcome_blind_collection_only": True,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        "training_corpus_export_attempted": False,
        "labels_or_outcomes_opened_during_collection": False,
        "settlement_pnl_opened_during_collection": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    blockers = [key for key, value in expected.items() if summary.get(key) != value]
    if not str(summary.get("batch_id") or ""):
        blockers.append("batch_id_missing")
    if not (summary.get("captures") or summary.get("errors")):
        blockers.append("batch_contains_no_capture_attempts")
    attempt_run_ids = [
        str(row.get("run_id") or Path(str(row.get("run_dir") or "")).name)
        for row in [*(summary.get("captures") or []), *(summary.get("errors") or [])]
    ]
    if any(not run_id for run_id in attempt_run_ids):
        blockers.append("batch_capture_attempt_run_id_missing")
    if len(attempt_run_ids) != len(set(attempt_run_ids)):
        blockers.append("batch_capture_attempt_run_id_duplicate")
    if int(summary.get("finalization_count") or 0) != 0:
        blockers.append("finalization_count_nonzero")
    if summary.get("finalizations") not in ([], None):
        blockers.append("finalization_rows_present")
    if blockers:
        raise ValueError(
            "outcome-blind batch summary validation failed: " + ", ".join(sorted(set(blockers)))
        )


def _validate_source_boundary_manifest(
    boundary: dict[str, Any],
    *,
    path: Path,
    expected_sha256: str,
) -> None:
    schema_version = boundary.get("schema_version")
    if schema_version == COLLECTION_FREEZE_MANIFEST_SCHEMA_VERSION:
        load_and_validate_pairwise_future_unseen_collection_freeze(
            path,
            expected_sha256,
        )
        return
    blockers: list[str] = []
    if schema_version != SOURCE_BOUNDARY_SCHEMA_VERSION:
        blockers.append("source_boundary_schema_invalid")
    if int(boundary.get("minimum_collection_decision_ts") or 0) <= 0:
        blockers.append("source_boundary_minimum_timestamp_invalid")
    if boundary.get("labels_outcomes_or_pnl_opened") is not False:
        blockers.append("source_boundary_outcome_sealing_invalid")
    for field in ("prior_market_ids", "prior_slugs", "prior_source_row_hashes"):
        if not isinstance(boundary.get(field), list):
            blockers.append(f"source_boundary_{field}_invalid")
    safety_mismatches = [
        field
        for field, expected in _blocked_safety_fields().items()
        if boundary.get(field) != expected
    ]
    if safety_mismatches:
        blockers.append("source_boundary_safety_invalid")
    if blockers:
        raise ValueError(
            "source boundary manifest validation failed: " + ", ".join(sorted(set(blockers)))
        )


def _boundary_references(
    boundary: dict[str, Any],
) -> tuple[int, set[str], set[str], set[str]]:
    minimum_ts = int(boundary.get("minimum_collection_decision_ts") or 0)
    if minimum_ts <= 0:
        raise ValueError("source boundary minimum_collection_decision_ts is invalid")
    market_ids = {str(value) for value in boundary.get("prior_market_ids") or []}
    slugs = {str(value) for value in boundary.get("prior_slugs") or []}
    row_hashes = {str(value) for value in boundary.get("prior_source_row_hashes") or []}
    selected_descriptor = boundary.get("source_selected_rows")
    if isinstance(selected_descriptor, dict):
        selected_path = _verified_descriptor(selected_descriptor, "source selected rows")
        for row in _load_jsonl(Path(selected_path["path"])):
            if row.get("market_id"):
                market_ids.add(str(row["market_id"]))
            if row.get("slug"):
                slugs.add(str(row["slug"]))
            if row.get("source_row_hash"):
                row_hashes.add(str(row["source_row_hash"]))
    prior_descriptor = boundary.get("source_prior_evidence_exclusion_registry")
    if isinstance(prior_descriptor, dict):
        prior_path = _verified_descriptor(prior_descriptor, "prior exclusion registry")
        prior = _load_json(Path(prior_path["path"]))
        market_ids.update(str(value) for value in prior.get("prior_market_ids") or [])
        slugs.update(str(value) for value in prior.get("prior_slugs") or [])
        row_hashes.update(str(value) for value in prior.get("prior_source_row_hashes") or [])
    return minimum_ts, market_ids, slugs, row_hashes


def _verified_descriptor(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor missing")
    path = Path(str(value.get("path") or "")).resolve()
    digest = str(value.get("sha256") or "").lower()
    _verify_pin(path, digest, name=name)
    return {"path": str(path), "sha256": digest}


def _verify_index_raw_descriptors(row: dict[str, Any]) -> None:
    for filename, value in dict(row.get("raw_artifacts") or {}).items():
        if not isinstance(value, dict):
            raise ValueError(f"raw descriptor invalid: {filename}")
        path = Path(str(value.get("path") or "")).resolve()
        digest = str(value.get("sha256") or "").lower()
        _verify_pin(path, digest, name=f"indexed raw artifact {filename}")
        if len(_load_jsonl(path)) != int(value.get("row_count") or 0):
            raise ValueError(f"raw row count mismatch: {filename}")


def _find_nonempty_fields(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden and child not in (None, False, 0, "", [], {}):
                found.add(key)
            found.update(_find_nonempty_fields(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_nonempty_fields(child, forbidden))
    return found


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _batch_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Persistent Outcome-Blind Batch Index",
            "",
            f"- batch: `{report['batch_id']}`",
            f"- appended / captures: `{report['appended_entry_count']}/{report['batch_capture_count']}`",
            f"- total / quality-valid index rows: `{report['index_entry_count']}/{report['quality_valid_index_entry_count']}`",
            "- full-window orderbook pass / fail: "
            f"`{report['batch_orderbook_full_window_coverage_passed_count']}/"
            f"{report['batch_orderbook_full_window_coverage_failed_count']}`",
            "- observed / expected decision pairs: "
            f"`{report['batch_orderbook_observed_decision_pair_count']}/"
            f"{report['batch_orderbook_expected_decision_pair_count']}`",
            "- orderbook window blockers: "
            f"`{report['batch_orderbook_window_coverage_reason_distribution']}`",
            f"- idempotent skips: `{report['idempotent_existing_run_count']}`",
            "- labels/outcomes/PnL opened: `false`",
            "- settlement finalizer / resolution / training export: `false/false/false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _window_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Outcome-Blind Window Freeze",
            "",
            f"- ready: `{str(report['window_freeze_ready']).lower()}`",
            f"- selected / target: `{report['selected_market_count']}/{report['target_valid_market_count']}`",
            f"- scanned / cap: `{report['scanned_entry_count']}/{report['maximum_scan_count']}`",
            f"- minimum decision timestamp: `{report['minimum_collection_decision_ts']}`",
            "- immutable collector-index snapshot: `true`",
            f"- blockers: `{report['blocking_reason_codes']}`",
            "- labels/outcomes/PnL opened for selection: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )
