"""Outcome-blind final prior-lineage quarantine for the hybrid ranker."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    FORBIDDEN_REGISTRY_FIELDS,
)

SCHEMA_PREFIX = "bigan-v8-hybrid-pairwise-final-prior-lineage-quarantine"
TERMINAL_SUPERVISOR_STATUSES = {
    "blocked_fail_closed",
    "issue179_role_assignment_ready",
}
TERMINAL_SUPPORT_GATE_STATUSES = {
    "BLOCKED_FAIL_CLOSED",
    "BLOCKED_INSUFFICIENT_SUPPORT_AT_FROZEN_MAXIMUM",
    "OUTCOME_BLIND_SUPPORT_TARGET_READY",
}


@dataclass(frozen=True, slots=True)
class HybridPairwisePriorLineageQuarantineConfig:
    """Hash-pinned inputs for the final outcome-blind prior boundary."""

    run_id: str
    output_dir: Path | str
    created_at_ts: int
    historical_registry_descriptor_path: Path | str
    expected_historical_registry_descriptor_sha256: str
    historical_registry_rows_path: Path | str
    expected_historical_registry_rows_sha256: str
    terminal_lineage_state_path: Path | str
    expected_terminal_lineage_state_sha256: str
    final_support_gate_manifest_path: Path | str
    expected_final_support_gate_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.created_at_ts <= 0:
            raise ValueError("created_at_ts must be positive")
        for name, value in (
            (
                "historical registry descriptor SHA-256",
                self.expected_historical_registry_descriptor_sha256,
            ),
            (
                "historical registry rows SHA-256",
                self.expected_historical_registry_rows_sha256,
            ),
            (
                "terminal lineage state SHA-256",
                self.expected_terminal_lineage_state_sha256,
            ),
            (
                "final support gate manifest SHA-256",
                self.expected_final_support_gate_manifest_sha256,
            ),
        ):
            _require_sha256(value, name=name)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "historical_registry_descriptor_path",
            Path(self.historical_registry_descriptor_path),
        )
        object.__setattr__(
            self,
            "historical_registry_rows_path",
            Path(self.historical_registry_rows_path),
        )
        object.__setattr__(
            self,
            "terminal_lineage_state_path",
            Path(self.terminal_lineage_state_path),
        )
        object.__setattr__(
            self,
            "final_support_gate_manifest_path",
            Path(self.final_support_gate_manifest_path),
        )


def build_hybrid_pairwise_prior_lineage_quarantine(
    config: HybridPairwisePriorLineageQuarantineConfig,
) -> dict[str, Any]:
    """Freeze every prior market without opening target or outcome evidence."""

    registry_descriptor_path = (
        config.historical_registry_descriptor_path.resolve()
    )
    registry_rows_path = config.historical_registry_rows_path.resolve()
    terminal_state_path = config.terminal_lineage_state_path.resolve()
    support_manifest_path = config.final_support_gate_manifest_path.resolve()
    for path, expected_sha256, name in (
        (
            registry_descriptor_path,
            config.expected_historical_registry_descriptor_sha256,
            "historical registry descriptor",
        ),
        (
            registry_rows_path,
            config.expected_historical_registry_rows_sha256,
            "historical registry rows",
        ),
        (
            terminal_state_path,
            config.expected_terminal_lineage_state_sha256,
            "terminal lineage state",
        ),
        (
            support_manifest_path,
            config.expected_final_support_gate_manifest_sha256,
            "final support gate manifest",
        ),
    ):
        _verify_pin(path, expected_sha256, name=name)

    registry_descriptor = _load_json(registry_descriptor_path)
    registry_rows = _load_jsonl(registry_rows_path)
    terminal_state = _load_json(terminal_state_path)
    support_manifest = _load_json(support_manifest_path)
    _reject_forbidden(registry_rows, name="historical registry rows")
    _reject_forbidden(terminal_state, name="terminal lineage state")
    _reject_forbidden(support_manifest, name="support gate manifest")

    historical_entries, historical_market_ids_hash = _historical_entries(
        descriptor=registry_descriptor,
        rows=registry_rows,
    )
    terminal_status = _validate_terminal_state(
        state=terminal_state,
        support_manifest_path=support_manifest_path,
        support_manifest_sha256=(
            config.expected_final_support_gate_manifest_sha256
        ),
    )
    support_chain = _validate_support_chain(
        terminal_state=terminal_state,
        terminal_status=terminal_status,
        support_manifest=support_manifest,
        support_manifest_path=support_manifest_path,
    )
    prior_registry = support_chain["prior_registry"]
    prior_entries = _prior_registry_entries(prior_registry)
    (
        capture_entries,
        capture_audit,
        raw_market_descriptors,
    ) = _capture_entries(support_chain["batch_progress_descriptors"])

    entries_by_market: dict[str, dict[str, Any]] = {}
    for source_kind, entries in (
        ("historical_development_registry", historical_entries),
        ("prior_exclusion_registry", prior_entries),
        ("terminal_collection_capture", capture_entries),
    ):
        for entry in entries:
            market_id = str(entry["market_id"])
            decision_ts = int(entry["decision_ts"])
            if decision_ts <= 0:
                raise ValueError(
                    f"prior market decision boundary is invalid: {market_id}"
                )
            existing = entries_by_market.setdefault(
                market_id,
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "source_kinds": [],
                    "source_paths": [],
                },
            )
            existing["decision_ts"] = max(
                int(existing["decision_ts"]),
                decision_ts,
            )
            existing["source_kinds"].append(source_kind)
            existing["source_paths"].extend(entry["source_paths"])

    if not entries_by_market:
        raise ValueError("final prior-lineage quarantine has no markets")
    market_entries = []
    for market_id in sorted(entries_by_market):
        entry = entries_by_market[market_id]
        market_entries.append(
            {
                "market_id": market_id,
                "decision_ts": int(entry["decision_ts"]),
                "source_kinds": sorted(set(entry["source_kinds"])),
                "source_paths": sorted(set(entry["source_paths"])),
            }
        )
    prior_market_ids = [entry["market_id"] for entry in market_entries]
    maximum_prior_decision_ts = max(
        entry["decision_ts"] for entry in market_entries
    )
    if config.created_at_ts <= maximum_prior_decision_ts:
        raise ValueError(
            "final quarantine creation timestamp must follow all prior decisions"
        )

    run_dir = (config.output_dir / config.run_id).resolve()
    if run_dir.exists():
        if not config.overwrite_existing:
            raise ValueError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    quarantine = {
        "schema_version": f"{SCHEMA_PREFIX}-v1",
        "run_id": config.run_id,
        "status": "prior_lineage_complete",
        "created_at_ts": config.created_at_ts,
        "final": True,
        "active_prior_lineage_complete": True,
        "includes_issue175_through_issue179": True,
        "terminal_lineage_status": terminal_status,
        "terminal_support_gate_status": support_chain[
            "support_gate_status"
        ],
        "historical_registry_descriptor": _descriptor(
            registry_descriptor_path
        ),
        "historical_registry_rows": _descriptor(registry_rows_path),
        "historical_development_market_count": len(historical_entries),
        "historical_development_market_ids_sha256": (
            historical_market_ids_hash
        ),
        "terminal_lineage_state": _descriptor(terminal_state_path),
        "final_support_gate_manifest": _descriptor(
            support_manifest_path
        ),
        "final_support_gate_report": support_chain[
            "support_gate_report_descriptor"
        ],
        "terminal_precollection_freeze_manifest": support_chain[
            "precollection_freeze_descriptor"
        ],
        "terminal_prior_exclusion_registry": support_chain[
            "prior_registry_descriptor"
        ],
        "batch_progress_inputs": support_chain[
            "batch_progress_descriptors"
        ],
        "raw_market_identity_inputs": raw_market_descriptors,
        "terminal_batch_count": len(
            support_chain["batch_progress_descriptors"]
        ),
        "terminal_capture_count": capture_audit["capture_count"],
        "terminal_capture_market_count": capture_audit[
            "capture_market_count"
        ],
        "terminal_empty_fail_closed_capture_count": capture_audit[
            "empty_fail_closed_capture_count"
        ],
        "terminal_missing_market_identity_count": 0,
        "prior_registry_market_count": len(prior_entries),
        "total_prior_unique_market_count": len(prior_market_ids),
        "prior_market_ids": prior_market_ids,
        "prior_market_ids_sha256": canonical_json_sha256(
            prior_market_ids
        ),
        "maximum_prior_decision_ts": maximum_prior_decision_ts,
        "minimum_future_decision_ts": max(
            config.created_at_ts,
            maximum_prior_decision_ts,
        )
        + 1,
        "market_entries": market_entries,
        "outcome_label_or_pnl_artifacts_opened": False,
        "resolution_artifacts_opened": False,
        "label_rows_semantic_content_parsed": False,
        "resolution_rows_semantic_content_parsed": False,
        "outcome_values_loaded": False,
        "pnl_values_loaded": False,
        "oof_or_validation_metrics_used_for_role_assignment": False,
        "confirmatory_labels_opened": False,
        "source_scores_mutated": False,
        "execution_thresholds_mutated": False,
        "safety": _nested_safety_fields(),
        **_blocked_safety_fields(),
    }
    quarantine["quarantine_id"] = canonical_json_sha256(quarantine)
    quarantine_path = (
        run_dir / "hybrid_pairwise_final_prior_lineage_quarantine.json"
    )
    _write_json(quarantine_path, quarantine)
    markdown_path = (
        run_dir / "hybrid_pairwise_final_prior_lineage_quarantine.md"
    )
    markdown_path.write_text(
        _quarantine_markdown(quarantine),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "quarantine": _descriptor(quarantine_path),
        "quarantine_markdown": _descriptor(markdown_path),
        "terminal_lineage_state": _descriptor(terminal_state_path),
        "final_support_gate_manifest": _descriptor(
            support_manifest_path
        ),
        "historical_registry_descriptor": _descriptor(
            registry_descriptor_path
        ),
        "historical_registry_rows": _descriptor(registry_rows_path),
        "total_prior_unique_market_count": len(prior_market_ids),
        "prior_market_ids_sha256": quarantine[
            "prior_market_ids_sha256"
        ],
        "maximum_prior_decision_ts": maximum_prior_decision_ts,
        "minimum_future_decision_ts": quarantine[
            "minimum_future_decision_ts"
        ],
        "outcome_label_or_pnl_artifacts_opened": False,
        "resolution_artifacts_opened": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = (
        run_dir
        / "hybrid_pairwise_final_prior_lineage_quarantine_manifest.json"
    )
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "quarantine_path": quarantine_path,
        "quarantine_sha256": _sha256_file(quarantine_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "quarantine": quarantine,
        "manifest": manifest,
    }


def _historical_entries(
    *,
    descriptor: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    expected_count = int(descriptor.get("selected_market_count") or 0)
    if expected_count != 90 or len(rows) != expected_count:
        raise ValueError("historical registry must contain exactly 90 rows")
    ordered = sorted(rows, key=lambda row: int(row.get("selection_rank") or 0))
    if [int(row.get("selection_rank") or 0) for row in ordered] != list(
        range(1, expected_count + 1)
    ):
        raise ValueError("historical registry selection ranks are invalid")
    market_ids = [str(row.get("market_id") or "") for row in ordered]
    if any(not market_id for market_id in market_ids):
        raise ValueError("historical registry market identity is missing")
    if len(market_ids) != len(set(market_ids)):
        raise ValueError("historical registry market identities are duplicated")
    market_ids_hash = canonical_json_sha256(market_ids)
    if market_ids_hash != descriptor.get("selected_market_ids_sha256"):
        raise ValueError("historical registry market ID hash mismatch")
    entries = []
    for row in ordered:
        decision_ts = int(row.get("maximum_decision_ts") or 0)
        if decision_ts <= 0:
            raise ValueError("historical registry decision boundary is missing")
        entries.append(
            {
                "market_id": str(row["market_id"]),
                "decision_ts": decision_ts,
                "source_paths": [str(row.get("corpus_dir") or "")],
            }
        )
    return entries, market_ids_hash


def _validate_terminal_state(
    *,
    state: dict[str, Any],
    support_manifest_path: Path,
    support_manifest_sha256: str,
) -> str:
    status = str(state.get("status") or "")
    if status not in TERMINAL_SUPERVISOR_STATUSES:
        raise ValueError("prior collection lineage is not terminal")
    state_manifest_path = Path(
        str(state.get("support_gate_manifest_path") or "")
    ).resolve()
    if state_manifest_path != support_manifest_path:
        raise ValueError("terminal state support manifest path mismatch")
    if (
        str(state.get("support_gate_manifest_sha256") or "")
        != support_manifest_sha256
    ):
        raise ValueError("terminal state support manifest SHA-256 mismatch")
    if int(state.get("completed_batch_count") or 0) <= 0:
        raise ValueError("terminal state completed batch count is missing")
    if int(state.get("capture_attempt_count") or 0) <= 0:
        raise ValueError("terminal state capture attempt count is missing")
    return status


def _validate_support_chain(
    *,
    terminal_state: dict[str, Any],
    terminal_status: str,
    support_manifest: dict[str, Any],
    support_manifest_path: Path,
) -> dict[str, Any]:
    support_report_descriptor = _verified_descriptor(
        support_manifest.get("support_gate_report"),
        name="support gate report",
    )
    support_report = _load_json(
        Path(support_report_descriptor["path"])
    )
    _reject_forbidden(support_report, name="support gate report")
    support_gate_status = str(support_report.get("status") or "")
    if support_gate_status not in TERMINAL_SUPPORT_GATE_STATUSES:
        raise ValueError("support gate status is not terminal")
    if (
        str(terminal_state.get("support_gate_status") or "")
        != support_gate_status
    ):
        raise ValueError("terminal state support gate status mismatch")
    state_report_path = Path(
        str(terminal_state.get("support_gate_report_path") or "")
    ).resolve()
    if state_report_path != Path(support_report_descriptor["path"]):
        raise ValueError("terminal state support report path mismatch")
    if (
        str(terminal_state.get("support_gate_report_sha256") or "")
        != support_report_descriptor["sha256"]
    ):
        raise ValueError("terminal state support report SHA-256 mismatch")
    if terminal_status == "issue179_role_assignment_ready" and (
        support_gate_status != "OUTCOME_BLIND_SUPPORT_TARGET_READY"
    ):
        raise ValueError("ready terminal state has inconsistent support status")
    if terminal_status == "blocked_fail_closed" and (
        support_report.get("continuation_allowed") is not False
        or support_report.get("continuation_required") is not False
    ):
        raise ValueError("blocked terminal support gate is not final")
    if (
        support_report.get("labels_or_outcomes_opened_for_continuation")
        is not False
        or support_report.get("settlement_pnl_opened_for_continuation")
        is not False
        or support_report.get(
            "uses_oof_validation_or_confirmatory_pnl_for_continuation"
        )
        is not False
    ):
        raise ValueError("support gate opened forbidden evidence")
    _require_blocked_safety(support_report, name="support gate report")
    _require_blocked_safety(support_manifest, name="support gate manifest")

    precollection_freeze_descriptor = _verified_descriptor(
        support_manifest.get("precollection_freeze_manifest"),
        name="terminal precollection freeze manifest",
    )
    precollection_freeze = _load_json(
        Path(precollection_freeze_descriptor["path"])
    )
    _reject_forbidden(
        precollection_freeze,
        name="terminal precollection freeze manifest",
    )
    _require_blocked_safety(
        precollection_freeze,
        name="terminal precollection freeze manifest",
    )
    prior_registry_descriptor = _verified_descriptor(
        precollection_freeze.get("prior_evidence_exclusion_registry"),
        name="terminal prior exclusion registry",
    )
    prior_registry = _load_json(
        Path(prior_registry_descriptor["path"])
    )
    _reject_forbidden(
        prior_registry,
        name="terminal prior exclusion registry",
    )
    if (
        prior_registry.get("prior_outcome_or_pnl_values_loaded") is not False
        or prior_registry.get(
            "prior_validation_or_future_evidence_used_for_tuning"
        )
        is not False
    ):
        raise ValueError("prior exclusion registry opened forbidden evidence")
    _require_blocked_safety(
        prior_registry,
        name="terminal prior exclusion registry",
    )

    batch_progress_descriptors = [
        _verified_descriptor(value, name="terminal batch progress")
        for value in support_manifest.get("batch_progress_inputs") or []
    ]
    if not batch_progress_descriptors:
        raise ValueError("support manifest has no batch progress inputs")
    if len(batch_progress_descriptors) != int(
        terminal_state.get("completed_batch_count") or 0
    ):
        raise ValueError("terminal completed batch count mismatch")
    if int(support_report.get("unique_batch_progress_count") or 0) != len(
        batch_progress_descriptors
    ):
        raise ValueError("support report batch count mismatch")
    if _sha256_file(support_manifest_path) != str(
        terminal_state["support_gate_manifest_sha256"]
    ):
        raise ValueError("support manifest changed after terminal state")
    return {
        "support_gate_status": support_gate_status,
        "support_gate_report_descriptor": support_report_descriptor,
        "precollection_freeze_descriptor": (
            precollection_freeze_descriptor
        ),
        "prior_registry_descriptor": prior_registry_descriptor,
        "prior_registry": prior_registry,
        "batch_progress_descriptors": batch_progress_descriptors,
    }


def _prior_registry_entries(
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    market_ids = [str(value) for value in registry.get("prior_market_ids") or []]
    if not market_ids or any(not value for value in market_ids):
        raise ValueError("prior exclusion registry market identities are missing")
    if len(market_ids) != len(set(market_ids)):
        raise ValueError("prior exclusion registry contains duplicate markets")
    if canonical_json_sha256(market_ids) != registry.get(
        "prior_market_ids_sha256"
    ):
        raise ValueError("prior exclusion registry market ID hash mismatch")
    maximum_prior_decision_ts = int(
        registry.get("maximum_prior_decision_ts") or 0
    )
    if maximum_prior_decision_ts <= 0:
        raise ValueError("prior exclusion registry decision boundary is missing")
    entry_map = {
        str(row.get("market_id") or ""): int(row.get("decision_ts") or 0)
        for row in registry.get("market_entries") or []
        if isinstance(row, dict)
    }
    entries = []
    source_path = str(
        registry.get("run_id") or "terminal_prior_exclusion_registry"
    )
    for market_id in market_ids:
        decision_ts = entry_map.get(
            market_id,
            maximum_prior_decision_ts,
        )
        if decision_ts <= 0:
            raise ValueError("prior exclusion registry has invalid timestamp")
        entries.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "source_paths": [source_path],
            }
        )
    return entries


def _capture_entries(
    batch_descriptors: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, str]]]:
    market_timestamps: dict[str, list[int]] = defaultdict(list)
    market_source_paths: dict[str, set[str]] = defaultdict(set)
    raw_market_descriptors: dict[tuple[str, str], dict[str, str]] = {}
    capture_count = 0
    capture_market_count = 0
    empty_fail_closed_capture_count = 0
    seen_run_ids: set[str] = set()
    for descriptor in batch_descriptors:
        batch_path = Path(descriptor["path"])
        batch = _load_json(batch_path)
        _reject_forbidden(batch, name="terminal batch progress")
        if (
            batch.get("paper_only") is not True
            or batch.get("capital_at_risk") is not False
        ):
            raise ValueError("terminal batch safety contract failed")
        if int(batch.get("error_count") or 0) != 0:
            raise ValueError("terminal batch collector errors are nonzero")
        captures = [dict(row) for row in batch.get("captures") or []]
        if len(captures) != int(batch.get("capture_count") or 0):
            raise ValueError("terminal batch capture count mismatch")
        for capture in captures:
            capture_count += 1
            run_id = str(capture.get("run_id") or "")
            if not run_id or run_id in seen_run_ids:
                raise ValueError("capture run identity is missing or duplicated")
            seen_run_ids.add(run_id)
            run_dir = Path(str(capture.get("run_dir") or "")).resolve()
            market_path, market_rows = _capture_market_rows(run_dir)
            if market_path is None:
                if (
                    capture.get("capture_status") == "blocked_fail_closed"
                    and int(capture.get("raw_polymarket_market_count") or 0)
                    == 0
                    and bool(capture.get("reject_reason_counts"))
                ):
                    empty_fail_closed_capture_count += 1
                    continue
                raise ValueError(
                    f"capture market identity is incomplete: {run_id}"
                )
            raw_descriptor = _descriptor(market_path)
            raw_market_descriptors[
                (raw_descriptor["path"], raw_descriptor["sha256"])
            ] = raw_descriptor
            scheduled_ts = int(
                capture.get("scheduled_round_start_ts") or 0
            )
            market_ids_seen: set[str] = set()
            for market_row in market_rows:
                market_id = str(
                    market_row.get("market_id")
                    or market_row.get("condition_id")
                    or ""
                )
                if not market_id:
                    continue
                market_ids_seen.add(market_id)
                decision_ts = max(
                    scheduled_ts,
                    int(market_row.get("market_end_ts") or 0),
                )
                if decision_ts <= 0:
                    raise ValueError(
                        f"capture market boundary is missing: {run_id}"
                    )
                market_timestamps[market_id].append(decision_ts)
                market_source_paths[market_id].add(str(market_path))
            if not market_ids_seen:
                raise ValueError(
                    f"capture raw market rows have no identity: {run_id}"
                )
            capture_market_count += len(market_ids_seen)
    entries = [
        {
            "market_id": market_id,
            "decision_ts": max(market_timestamps[market_id]),
            "source_paths": sorted(market_source_paths[market_id]),
        }
        for market_id in sorted(market_timestamps)
    ]
    return (
        entries,
        {
            "capture_count": capture_count,
            "capture_market_count": capture_market_count,
            "empty_fail_closed_capture_count": (
                empty_fail_closed_capture_count
            ),
        },
        sorted(
            raw_market_descriptors.values(),
            key=lambda row: row["path"],
        ),
    )


def _capture_market_rows(
    run_dir: Path,
) -> tuple[Path | None, list[dict[str, Any]]]:
    for candidate_path in (
        run_dir / "raw" / "raw_polymarket_markets.jsonl",
        run_dir / "provider_raw" / "raw_polymarket_markets.jsonl",
    ):
        if not candidate_path.is_file():
            continue
        rows = _load_jsonl(candidate_path)
        _reject_forbidden(rows, name="raw market identity rows")
        if rows:
            return candidate_path, rows
    return None, []


def _verified_descriptor(payload: Any, *, name: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(payload.get("path") or "")).resolve()
    digest = str(payload.get("sha256") or "")
    _verify_pin(path, digest, name=name)
    return {"path": str(path), "sha256": digest.lower()}


def _reject_forbidden(payload: Any, *, name: str) -> None:
    found = sorted(_find_fields(payload, FORBIDDEN_REGISTRY_FIELDS))
    if found:
        raise ValueError(f"{name} contains forbidden fields: " + ", ".join(found))


def _require_blocked_safety(payload: dict[str, Any], *, name: str) -> None:
    if (
        payload.get("paper_only") is not True
        or payload.get("capital_at_risk") is not False
        or payload.get("polymarket_write_enabled") is not False
        or payload.get("wallet_signing_enabled") is not False
        or payload.get("source_model_candidate_eligible") is not False
        or payload.get("freeze_ready") is not False
        or payload.get("promotion_evidence_eligible") is not False
        or payload.get("v8_execution_handoff_allowed") is not False
    ):
        raise ValueError(f"{name} safety contract failed")


def _blocked_safety_fields() -> dict[str, Any]:
    return {
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


def _nested_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _quarantine_markdown(quarantine: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Pairwise Final Prior-Lineage Quarantine",
            "",
            f"- status: `{quarantine['status']}`",
            (
                "- total prior unique markets: "
                f"`{quarantine['total_prior_unique_market_count']}`"
            ),
            (
                "- historical development markets: "
                f"`{quarantine['historical_development_market_count']}`"
            ),
            (
                "- terminal captures: "
                f"`{quarantine['terminal_capture_count']}`"
            ),
            (
                "- terminal empty fail-closed captures: "
                f"`{quarantine['terminal_empty_fail_closed_capture_count']}`"
            ),
            (
                "- maximum prior decision timestamp: "
                f"`{quarantine['maximum_prior_decision_ts']}`"
            ),
            (
                "- minimum future decision timestamp: "
                f"`{quarantine['minimum_future_decision_ts']}`"
            ),
            "- labels/outcomes/PnL opened: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    _require_sha256(expected_sha256, name=f"{name} SHA-256")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object row: {path}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _find_fields(
    payload: Any,
    forbidden: set[str],
    prefix: str = "",
) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden:
                found.add(path)
            found.update(_find_fields(value, forbidden, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(
                _find_fields(value, forbidden, f"{prefix}[{index}]")
            )
    return found
