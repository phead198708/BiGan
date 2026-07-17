"""Immutable terminal reconciliation for asynchronously finalized batches."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    FORBIDDEN_REGISTRY_FIELDS,
    _blocked_safety_fields,
    _capture_quality_audit,
    _descriptor,
    _finalization_quality_reasons,
    _find_fields,
    _load_json,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_text,
)

TERMINAL_RECONCILIATION_REPORT_SCHEMA_VERSION = (
    "bigan-v8-pairwise-terminal-reconciliation-report-v1"
)
TERMINAL_RECONCILIATION_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-pairwise-terminal-reconciliation-manifest-v1"
)
RECONCILED_BATCH_PROGRESS_SCHEMA_VERSION = (
    "bigan-v8-reconciled-batch-progress-v1"
)


@dataclass(frozen=True, slots=True)
class PairwiseTerminalReconciliationConfig:
    """Hash-pinned inputs for metadata-only terminal reconciliation."""

    run_id: str
    output_dir: Path | str
    precollection_freeze_manifest_path: Path | str
    expected_precollection_freeze_manifest_sha256: str
    batch_progress_pins: tuple[tuple[Path | str, str], ...]
    training_corpus_root: Path | str = Path("/Volumes/PHILIPS/v8")

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_precollection_freeze_manifest_sha256,
            name="precollection freeze manifest SHA-256",
        )
        normalized_pins: list[tuple[Path, str]] = []
        for path, digest in self.batch_progress_pins:
            _require_sha256(digest, name="batch progress SHA-256")
            normalized_pins.append((Path(path), digest.lower()))
        if not normalized_pins:
            raise ValueError("at least one batch progress pin is required")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "precollection_freeze_manifest_path",
            Path(self.precollection_freeze_manifest_path),
        )
        object.__setattr__(
            self,
            "batch_progress_pins",
            tuple(normalized_pins),
        )
        object.__setattr__(
            self,
            "training_corpus_root",
            Path(self.training_corpus_root),
        )


def run_pairwise_terminal_reconciliation(
    config: PairwiseTerminalReconciliationConfig,
) -> dict[str, Any]:
    """Reconcile terminal metadata without opening target values."""

    freeze_path = config.precollection_freeze_manifest_path.resolve()
    _verify_pin(
        freeze_path,
        config.expected_precollection_freeze_manifest_sha256,
        name="precollection freeze manifest",
    )
    freeze = _load_json(freeze_path)
    freeze_forbidden = sorted(
        _find_fields(freeze, FORBIDDEN_REGISTRY_FIELDS)
    )
    collector_contract = dict(freeze.get("collector_contract") or {})
    blocking_reasons: list[str] = []
    if freeze_forbidden:
        blocking_reasons.append("freeze_contains_forbidden_outcome_fields")
    frozen_training_root = Path(
        str(collector_contract.get("training_corpus_root") or "")
    ).expanduser()
    training_root = config.training_corpus_root.expanduser().resolve()
    if (
        not collector_contract
        or not frozen_training_root
        or frozen_training_root.resolve() != training_root
    ):
        blocking_reasons.append("frozen_training_corpus_root_mismatch")

    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    batch_results: list[dict[str, Any]] = []
    applied_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    original_hashes: dict[str, str] = {}
    reconciled_paths: list[Path] = []
    seen_batch_ids: set[str] = set()
    seen_run_ids: set[str] = set()

    for ordinal, (source_path_value, expected_sha) in enumerate(
        config.batch_progress_pins,
        start=1,
    ):
        source_path = source_path_value.resolve()
        _verify_pin(
            source_path,
            expected_sha,
            name="source batch progress",
        )
        original_hashes[str(source_path)] = expected_sha
        source = _load_json(source_path)
        result = _reconcile_batch(
            source=source,
            source_path=source_path,
            source_sha256=expected_sha,
            collector_contract=collector_contract,
            training_root=training_root,
            ordinal=ordinal,
            output_dir=run_dir,
            seen_batch_ids=seen_batch_ids,
            seen_run_ids=seen_run_ids,
        )
        batch_results.append(result["summary"])
        applied_rows.extend(result["applied_rows"])
        rejected_rows.extend(result["rejected_rows"])
        blocking_reasons.extend(result["blocking_reason_codes"])
        reconciled_paths.append(result["path"])

    if any(
        _sha256_file(Path(path)) != digest
        for path, digest in original_hashes.items()
    ):
        blocking_reasons.append("source_batch_progress_mutated")
    blocking_reasons = sorted(set(blocking_reasons))
    status = (
        "TERMINAL_RECONCILIATION_READY"
        if not blocking_reasons
        else "BLOCKED_FAIL_CLOSED"
    )
    report = {
        "schema_version": TERMINAL_RECONCILIATION_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": status,
        "terminal_reconciliation_ready": not blocking_reasons,
        "source_batch_count": len(batch_results),
        "source_capture_count": sum(
            int(row["capture_count"]) for row in batch_results
        ),
        "source_exported_finalization_count": sum(
            int(row["source_exported_finalization_count"])
            for row in batch_results
        ),
        "reconciled_exported_finalization_count": sum(
            int(row["reconciled_exported_finalization_count"])
            for row in batch_results
        ),
        "applied_terminal_finalization_count": len(applied_rows),
        "unchanged_terminal_finalization_count": sum(
            int(row["unchanged_terminal_finalization_count"])
            for row in batch_results
        ),
        "capture_quality_failed_no_reconciliation_required_count": sum(
            int(
                row[
                    "capture_quality_failed_no_reconciliation_required_count"
                ]
            )
            for row in batch_results
        ),
        "missing_terminal_finalization_evidence_count": sum(
            int(row["missing_terminal_finalization_evidence_count"])
            for row in batch_results
        ),
        "rejected_terminal_finalization_evidence_count": len(
            rejected_rows
        ),
        "rejection_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in rejected_rows
                    for reason in row["reason_codes"]
                ).items()
            )
        ),
        "blocking_reason_codes": blocking_reasons,
        "batch_summaries": batch_results,
        "source_batch_progress_byte_identity_preserved": not any(
            _sha256_file(Path(path)) != digest
            for path, digest in original_hashes.items()
        ),
        "resolution_payloads_opened": False,
        "labels_or_outcomes_opened_for_reconciliation": False,
        "settlement_pnl_opened_for_reconciliation": False,
        "oracle_actions_opened_for_reconciliation": False,
        "future_returns_opened_for_reconciliation": False,
        "source_scores_mutated": False,
        "execution_thresholds_mutated": False,
        **_blocked_safety_fields(),
    }
    report["terminal_reconciliation_report_id"] = (
        canonical_json_sha256(report)
    )
    report_path = run_dir / "pairwise_terminal_reconciliation_report.json"
    markdown_path = run_dir / "pairwise_terminal_reconciliation_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _report_markdown(report))

    manifest = {
        "schema_version": TERMINAL_RECONCILIATION_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "terminal_reconciliation_ready": report[
            "terminal_reconciliation_ready"
        ],
        "precollection_freeze_manifest": _descriptor(freeze_path),
        "source_batch_progress_inputs": [
            _descriptor(path.resolve())
            for path, _ in config.batch_progress_pins
        ],
        "reconciled_batch_progress_outputs": [
            _descriptor(path) for path in reconciled_paths
        ],
        "applied_terminal_finalization_evidence": [
            {
                "batch_id": row["batch_id"],
                "run_id": row["run_id"],
                "finalization_report": row["finalization_report"],
                "finalization_manifest": row["finalization_manifest"],
                "capture_manifest": row["capture_manifest"],
                "corpus_manifest": row["corpus_manifest"],
            }
            for row in applied_rows
        ],
        "report": _descriptor(report_path),
        "markdown": _descriptor(markdown_path),
        "blocking_reason_codes": blocking_reasons,
        "resolution_payloads_opened": False,
        "labels_or_outcomes_opened_for_reconciliation": False,
        "settlement_pnl_opened_for_reconciliation": False,
        **_blocked_safety_fields(),
    }
    manifest["terminal_reconciliation_manifest_id"] = (
        canonical_json_sha256(manifest)
    )
    manifest_path = (
        run_dir / "pairwise_terminal_reconciliation_manifest.json"
    )
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "reconciled_batch_progress_paths": tuple(reconciled_paths),
        "reconciled_batch_progress_pins": tuple(
            (path, _sha256_file(path)) for path in reconciled_paths
        ),
        "report": report,
        "manifest": manifest,
    }


def _reconcile_batch(
    *,
    source: dict[str, Any],
    source_path: Path,
    source_sha256: str,
    collector_contract: dict[str, Any],
    training_root: Path,
    ordinal: int,
    output_dir: Path,
    seen_batch_ids: set[str],
    seen_run_ids: set[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    batch_id = str(source.get("batch_id") or "")
    captures = [dict(row) for row in source.get("captures") or []]
    finalizations = [
        dict(row) for row in source.get("finalizations") or []
    ]
    if not batch_id or batch_id in seen_batch_ids:
        blockers.append("duplicate_or_missing_batch_id")
    seen_batch_ids.add(batch_id)
    if int(source.get("capture_count") or 0) != len(captures):
        blockers.append("source_capture_count_mismatch")
    if int(source.get("error_count") or 0) != 0:
        blockers.append("source_collector_error_count_nonzero")
    if _find_fields(source, FORBIDDEN_REGISTRY_FIELDS):
        blockers.append("source_batch_contains_forbidden_outcome_fields")
    if not _safe_payload(source, require_explicit_write_flags=False):
        blockers.append("source_batch_safety_flags_invalid")

    finalization_by_run_id: dict[str, dict[str, Any]] = {}
    for row in finalizations:
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in finalization_by_run_id:
            blockers.append("duplicate_or_missing_finalization_run_id")
            continue
        finalization_by_run_id[run_id] = row

    applied_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    capture_quality_failed_count = 0
    missing_evidence_count = 0
    unchanged_count = 0
    capture_order: dict[str, int] = {}
    collection_root = source_path.parent.parent.resolve()
    for index, capture in enumerate(captures):
        run_id = str(capture.get("run_id") or "")
        capture_order[run_id] = index
        if not run_id or run_id in seen_run_ids:
            blockers.append("duplicate_or_missing_capture_run_id")
            continue
        seen_run_ids.add(run_id)
        current = finalization_by_run_id.get(run_id)
        if not _finalization_quality_reasons(current):
            unchanged_count += 1
            continue
        capture_audit = _capture_quality_audit(
            capture,
            collector_contract=collector_contract,
            finalization=current,
        )
        if capture_audit["reason_codes"]:
            capture_quality_failed_count += 1
            continue
        run_dir_value = str(capture.get("run_dir") or "")
        run_dir = Path(run_dir_value).expanduser().resolve()
        if (
            not run_dir_value
            or run_dir.parent != collection_root
            or run_dir.name != run_id
        ):
            reasons = ["capture_run_directory_identity_mismatch"]
            rejected_rows.append(
                _rejected_row(batch_id, run_id, reasons)
            )
            blockers.append("terminal_finalization_evidence_rejected")
            continue
        report_path = run_dir / "pending_round_finalization_report.json"
        manifest_path = (
            run_dir / "pending_round_finalization_manifest.json"
        )
        capture_manifest_path = run_dir / "pending_round_capture_manifest.json"
        if not (
            report_path.is_file()
            and manifest_path.is_file()
            and capture_manifest_path.is_file()
        ):
            missing_evidence_count += 1
            rejected_rows.append(
                _rejected_row(
                    batch_id,
                    run_id,
                    ["terminal_finalization_evidence_missing"],
                )
            )
            blockers.append("quality_capture_terminal_finalization_missing")
            continue
        report = _load_json(report_path)
        manifest = _load_json(manifest_path)
        capture_manifest = _load_json(capture_manifest_path)
        reasons = _terminal_finalization_reasons(
            capture=capture,
            report=report,
            manifest=manifest,
            capture_manifest=capture_manifest,
            run_dir=run_dir,
            collection_root=collection_root,
            training_root=training_root,
        )
        if reasons:
            rejected_rows.append(
                _rejected_row(batch_id, run_id, reasons)
            )
            blockers.append("terminal_finalization_evidence_rejected")
            continue
        terminal = _terminal_finalization_entry(report, run_dir)
        finalization_by_run_id[run_id] = terminal
        corpus_manifest_path = (
            Path(terminal["exported_training_corpus_dir"])
            / "polymarket_corpus_manifest.json"
        )
        applied_rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "prior_finalization_status": (
                    None
                    if current is None
                    else current.get("finalization_status")
                ),
                "terminal_finalization_status": "exported",
                "finalization_report": _descriptor(report_path),
                "finalization_manifest": _descriptor(manifest_path),
                "capture_manifest": _descriptor(capture_manifest_path),
                "corpus_manifest": _descriptor(corpus_manifest_path),
            }
        )

    reconciled = copy.deepcopy(source)
    reconciled_finalizations = sorted(
        finalization_by_run_id.values(),
        key=lambda row: (
            capture_order.get(str(row.get("run_id") or ""), len(captures)),
            str(row.get("run_id") or ""),
        ),
    )
    reconciled["finalizations"] = reconciled_finalizations
    reconciled["exported_round_count"] = sum(
        not _finalization_quality_reasons(row)
        for row in reconciled_finalizations
    )
    reconciled["pending_resolution_count"] = sum(
        row.get("pending_resolution") is True
        for row in reconciled_finalizations
    )
    reconciled["terminal_reconciliation"] = {
        "schema_version": RECONCILED_BATCH_PROGRESS_SCHEMA_VERSION,
        "source_batch_progress": {
            "path": str(source_path),
            "sha256": source_sha256,
        },
        "applied_terminal_finalization_count": len(applied_rows),
        "capture_quality_failed_no_reconciliation_required_count": (
            capture_quality_failed_count
        ),
        "missing_terminal_finalization_evidence_count": (
            missing_evidence_count
        ),
        "rejected_terminal_finalization_evidence_count": len(
            rejected_rows
        ),
        "labels_or_outcomes_opened_for_reconciliation": False,
        "resolution_payloads_opened": False,
        "settlement_pnl_opened_for_reconciliation": False,
        "source_capture_rows_mutated": False,
        **_blocked_safety_fields(),
    }
    output_path = output_dir / (
        f"reconciled_batch_progress_{ordinal:02d}.json"
    )
    _write_json(output_path, reconciled)
    summary = {
        "batch_id": batch_id,
        "capture_count": len(captures),
        "source_batch_progress": _descriptor(source_path),
        "reconciled_batch_progress": _descriptor(output_path),
        "source_exported_finalization_count": sum(
            not _finalization_quality_reasons(row)
            for row in finalizations
        ),
        "reconciled_exported_finalization_count": reconciled[
            "exported_round_count"
        ],
        "applied_terminal_finalization_count": len(applied_rows),
        "unchanged_terminal_finalization_count": unchanged_count,
        "capture_quality_failed_no_reconciliation_required_count": (
            capture_quality_failed_count
        ),
        "missing_terminal_finalization_evidence_count": (
            missing_evidence_count
        ),
        "rejected_terminal_finalization_evidence_count": len(
            rejected_rows
        ),
        "blocking_reason_codes": sorted(set(blockers)),
    }
    return {
        "path": output_path,
        "summary": summary,
        "applied_rows": applied_rows,
        "rejected_rows": rejected_rows,
        "blocking_reason_codes": blockers,
    }


def _terminal_finalization_reasons(
    *,
    capture: dict[str, Any],
    report: dict[str, Any],
    manifest: dict[str, Any],
    capture_manifest: dict[str, Any],
    run_dir: Path,
    collection_root: Path,
    training_root: Path,
) -> list[str]:
    reasons: list[str] = []
    run_id = str(capture.get("run_id") or "")
    if any(
        str(payload.get("run_id") or "") != run_id
        for payload in (report, manifest, capture_manifest)
    ):
        reasons.append("terminal_finalization_run_id_mismatch")
    if str((manifest.get("config") or {}).get("run_id") or "") != run_id:
        reasons.append("terminal_finalization_config_run_id_mismatch")
    output_dir = str((manifest.get("config") or {}).get("output_dir") or "")
    if not output_dir or Path(output_dir).expanduser().resolve() != collection_root:
        reasons.append("terminal_finalization_collection_root_mismatch")
    if _find_fields(report, FORBIDDEN_REGISTRY_FIELDS) or _find_fields(
        manifest, FORBIDDEN_REGISTRY_FIELDS
    ):
        reasons.append("terminal_finalization_contains_forbidden_outcome_fields")
    if not all(_safe_payload(payload) for payload in (report, manifest, capture_manifest)):
        reasons.append("terminal_finalization_safety_flags_invalid")
    for payload in (report, manifest):
        if payload.get("finalization_status") != "exported":
            reasons.append("terminal_finalization_not_exported")
        if payload.get("pending_resolution") is not False:
            reasons.append("terminal_finalization_still_pending")
        if payload.get("phase2_corpus_built") is not True:
            reasons.append("terminal_finalization_phase2_corpus_missing")
        if int(payload.get("raw_resolution_count") or 0) <= 0 and payload is report:
            reasons.append("terminal_finalization_resolution_metadata_missing")
    if report.get("training_eligible") is not True:
        reasons.append("terminal_finalization_training_ineligible")
    if dict(report.get("reject_reason_counts") or {}):
        reasons.append("terminal_finalization_rejected")
    if report.get("resolution_provider_called") is not True:
        reasons.append("terminal_finalization_provider_not_called")
    if manifest.get("provider_raw_artifacts_preserved") is not True:
        reasons.append("terminal_finalization_raw_artifacts_not_preserved")
    corpus_value = str(report.get("exported_training_corpus_dir") or "")
    if corpus_value != str(manifest.get("exported_training_corpus_dir") or ""):
        reasons.append("terminal_finalization_corpus_path_mismatch")
    corpus_dir = Path(corpus_value).expanduser().resolve() if corpus_value else None
    if (
        corpus_dir is None
        or not corpus_dir.is_relative_to(training_root)
        or not corpus_dir.is_dir()
    ):
        reasons.append("terminal_finalization_corpus_outside_training_root")
    else:
        corpus_manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
        if not corpus_manifest_path.is_file():
            reasons.append("terminal_finalization_corpus_manifest_missing")
        else:
            expected = str(manifest.get("phase2_corpus_manifest_sha256") or "")
            if not expected or _sha256_file(corpus_manifest_path) != expected:
                reasons.append("terminal_finalization_corpus_manifest_hash_mismatch")
            provenance_path = corpus_dir / "training_corpus_provenance.json"
            if not provenance_path.is_file():
                reasons.append("terminal_finalization_corpus_provenance_missing")
            else:
                provenance = _load_json(provenance_path)
                expected_capture_manifest = (
                    run_dir / "pending_round_capture_manifest.json"
                ).resolve()
                if str(provenance.get("run_id") or "") != run_id:
                    reasons.append("terminal_finalization_provenance_run_id_mismatch")
                if Path(
                    str(provenance.get("pending_capture_manifest_path") or "")
                ).expanduser().resolve() != expected_capture_manifest:
                    reasons.append("terminal_finalization_capture_provenance_mismatch")
                if not _safe_payload(provenance):
                    reasons.append("terminal_finalization_provenance_safety_invalid")
    reasons.extend(_raw_artifact_hash_reasons(run_dir, manifest))
    return sorted(set(reasons))


def _raw_artifact_hash_reasons(
    run_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for key, directory in (
        ("raw_artifact_hashes", run_dir / "raw"),
        ("provider_raw_artifact_hashes", run_dir / "provider_raw"),
    ):
        hashes = dict(manifest.get(key) or {})
        if not hashes:
            reasons.append("terminal_finalization_raw_artifact_hashes_missing")
            continue
        for filename, expected in hashes.items():
            path = directory / str(filename)
            if not path.is_file():
                reasons.append("terminal_finalization_raw_artifact_missing")
            elif _sha256_file(path) != str(expected):
                reasons.append("terminal_finalization_raw_artifact_hash_mismatch")
    return reasons


def _terminal_finalization_entry(
    report: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    keys = (
        "exported_training_corpus_dir",
        "feature_enrichment_attempt_count",
        "feature_enrichment_post_market_close_candle_rejected_count",
        "feature_enrichment_reason_codes",
        "feature_enrichment_recovered",
        "feature_enrichment_warning_reason_codes",
        "finalization_status",
        "pending_feature_enrichment",
        "pending_resolution",
        "raw_btc_candle_row_count",
        "raw_resolution_count",
        "reject_reason_counts",
        "run_id",
        "training_eligible",
    )
    return {
        **{key: copy.deepcopy(report.get(key)) for key in keys},
        "run_dir": str(run_dir),
    }


def _safe_payload(
    payload: dict[str, Any],
    *,
    require_explicit_write_flags: bool = True,
) -> bool:
    if (
        payload.get("paper_only") is not True
        or payload.get("capital_at_risk") is not False
    ):
        return False
    write_fields = (
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "live_exchange_write_enabled",
        "broker_exchange_write_enabled",
    )
    if require_explicit_write_flags and any(
        field not in payload for field in write_fields[:2]
    ):
        return False
    return all(payload.get(field, False) is False for field in write_fields)


def _rejected_row(
    batch_id: str,
    run_id: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "run_id": run_id,
        "reason_codes": sorted(set(reasons)),
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pairwise Terminal Reconciliation",
        "",
        f"- status: `{report['status']}`",
        f"- source captures: `{report['source_capture_count']}`",
        f"- source exported finalizations: `{report['source_exported_finalization_count']}`",
        f"- reconciled exported finalizations: `{report['reconciled_exported_finalization_count']}`",
        f"- applied terminal finalizations: `{report['applied_terminal_finalization_count']}`",
        f"- capture-quality failures preserved: `{report['capture_quality_failed_no_reconciliation_required_count']}`",
        f"- missing terminal evidence: `{report['missing_terminal_finalization_evidence_count']}`",
        f"- rejected terminal evidence: `{report['rejected_terminal_finalization_evidence_count']}`",
        f"- blocking reasons: `{json.dumps(report['blocking_reason_codes'])}`",
        "- original batch progress byte identity preserved: "
        f"`{str(report['source_batch_progress_byte_identity_preserved']).lower()}`",
        "- resolution payloads opened: `false`",
        "- labels/outcomes/PnL opened: `false`",
        "- paper/write/wallet/capital/handoff unlock: `false`",
    ]
    return "\n".join(lines) + "\n"
