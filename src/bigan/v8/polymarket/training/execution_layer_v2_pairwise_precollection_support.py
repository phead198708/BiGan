"""Outcome-blind support gate for bounded pairwise precollection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    FORBIDDEN_REGISTRY_FIELDS,
    PairwiseActionAdvantageLCBRoleAssignmentConfig,
    _blocked_safety_fields,
    _descriptor,
    _find_fields,
    _load_json,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_text,
    assign_pairwise_action_advantage_lcb_roles,
)

SUPPORT_GATE_SCHEMA_VERSION = (
    "bigan-v8-pairwise-precollection-support-gate-v1"
)
CONTINUATION_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-pairwise-precollection-continuation-manifest-v1"
)
SUPPORT_ONLY_ROLE_BLOCKERS = frozenset(
    {
        "insufficient_quality_valid_unique_market_support",
        "role_market_count_mismatch",
    }
)


@dataclass(frozen=True, slots=True)
class PairwisePrecollectionSupportGateConfig:
    """Hash-pinned inputs for an outcome-blind bounded continuation decision."""

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
        normalized_pins = []
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


def run_pairwise_precollection_support_gate(
    config: PairwisePrecollectionSupportGateConfig,
) -> dict[str, Any]:
    """Run role assignment when eligible and decide bounded continuation."""

    freeze_path = config.precollection_freeze_manifest_path.resolve()
    _verify_pin(
        freeze_path,
        config.expected_precollection_freeze_manifest_sha256,
        name="precollection freeze manifest",
    )
    freeze = _load_json(freeze_path)
    target_count = int(freeze.get("target_valid_market_count") or 0)
    initial_attempt_count = int(
        freeze.get("initial_capture_attempt_count") or 0
    )
    maximum_attempt_count = int(
        freeze.get("maximum_total_capture_attempt_count") or 0
    )
    if (
        target_count <= 0
        or initial_attempt_count < target_count
        or maximum_attempt_count < initial_attempt_count
    ):
        raise ValueError("frozen precollection attempt contract is invalid")

    (
        unique_batch_pins,
        duplicate_excluded_inputs,
        batch_preflight,
    ) = _batch_preflight(config.batch_progress_pins)
    attempted_capture_count = int(
        batch_preflight["attempted_capture_count"]
    )
    preflight_blockers = list(batch_preflight["blocking_reason_codes"])
    if attempted_capture_count > maximum_attempt_count:
        preflight_blockers.append(
            "frozen_maximum_capture_attempt_count_exceeded"
        )
    preflight_blockers = sorted(set(preflight_blockers))

    role_result: dict[str, Any] | None = None
    role_report: dict[str, Any] | None = None
    role_assignment_attempted = False
    if (
        attempted_capture_count >= initial_attempt_count
        and not preflight_blockers
    ):
        role_assignment_attempted = True
        role_result = assign_pairwise_action_advantage_lcb_roles(
            PairwiseActionAdvantageLCBRoleAssignmentConfig(
                run_id=f"{config.run_id}-role-assignment",
                output_dir=config.output_dir,
                precollection_freeze_manifest_path=freeze_path,
                expected_precollection_freeze_manifest_sha256=(
                    config.expected_precollection_freeze_manifest_sha256
                ),
                batch_progress_pins=unique_batch_pins,
                training_corpus_root=config.training_corpus_root,
            )
        )
        role_report = dict(role_result["report"])

    decision = _continuation_decision(
        attempted_capture_count=attempted_capture_count,
        initial_attempt_count=initial_attempt_count,
        maximum_attempt_count=maximum_attempt_count,
        target_market_count=target_count,
        preflight_blocking_reason_codes=preflight_blockers,
        role_assignment_report=role_report,
    )
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": SUPPORT_GATE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": decision["status"],
        "target_valid_market_count": target_count,
        "initial_capture_attempt_count": initial_attempt_count,
        "maximum_total_capture_attempt_count": maximum_attempt_count,
        "attempted_capture_count": attempted_capture_count,
        "remaining_initial_capture_attempt_count": max(
            0,
            initial_attempt_count - attempted_capture_count,
        ),
        "remaining_frozen_capture_attempt_count": max(
            0,
            maximum_attempt_count - attempted_capture_count,
        ),
        "role_assignment_attempted": role_assignment_attempted,
        "role_assignment_ready": bool(
            role_report and role_report.get("role_assignment_ready")
        ),
        "selected_market_count": int(
            (role_report or {}).get("selected_market_count") or 0
        ),
        "excluded_capture_count": int(
            (role_report or {}).get("excluded_capture_count") or 0
        ),
        "role_assignment_blocking_reason_codes": list(
            (role_report or {}).get("blocking_reason_codes") or []
        ),
        "support_only_role_blockers": sorted(
            SUPPORT_ONLY_ROLE_BLOCKERS
        ),
        "support_only_failure": decision["support_only_failure"],
        "continuation_allowed": decision["continuation_allowed"],
        "continuation_required": decision["continuation_required"],
        "continuation_attempt_count": decision[
            "continuation_attempt_count"
        ],
        "continuation_reason_codes": decision[
            "continuation_reason_codes"
        ],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "frozen_maximum_enforced": True,
        "duplicate_excluded_input_count": len(
            duplicate_excluded_inputs
        ),
        "duplicate_excluded_inputs": duplicate_excluded_inputs,
        "unique_batch_progress_count": len(unique_batch_pins),
        "batch_ids": batch_preflight["batch_ids"],
        "capture_run_id_count": batch_preflight[
            "capture_run_id_count"
        ],
        "forbidden_batch_field_paths": batch_preflight[
            "forbidden_batch_field_paths"
        ],
        "collector_batch_error_count": batch_preflight[
            "collector_batch_error_count"
        ],
        "labels_or_outcomes_opened_for_continuation": False,
        "settlement_pnl_opened_for_continuation": False,
        "oracle_actions_opened_for_continuation": False,
        "future_returns_opened_for_continuation": False,
        "uses_oof_validation_or_confirmatory_pnl_for_continuation": False,
        "source_scores_mutated": False,
        "execution_thresholds_mutated": False,
        **_blocked_safety_fields(),
    }
    report["support_gate_report_id"] = canonical_json_sha256(report)
    report_path = (
        run_dir / "pairwise_precollection_support_gate_report.json"
    )
    markdown_path = (
        run_dir / "pairwise_precollection_support_gate_report.md"
    )
    _write_json(report_path, report)
    _write_text(markdown_path, _report_markdown(report))

    manifest = {
        "schema_version": CONTINUATION_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "precollection_freeze_manifest": _descriptor(freeze_path),
        "batch_progress_inputs": [
            _descriptor(path.resolve())
            for path, _ in unique_batch_pins
        ],
        "support_gate_report": _descriptor(report_path),
        "support_gate_markdown": _descriptor(markdown_path),
        "role_assignment_report": (
            None
            if role_result is None
            else _descriptor(Path(role_result["report_path"]))
        ),
        "role_assignment_manifest": (
            None
            if role_result is None
            else _descriptor(Path(role_result["manifest_path"]))
        ),
        "attempted_capture_count": attempted_capture_count,
        "selected_market_count": report["selected_market_count"],
        "continuation_allowed": report["continuation_allowed"],
        "continuation_attempt_count": report[
            "continuation_attempt_count"
        ],
        "blocking_reason_codes": report["blocking_reason_codes"],
        "duplicate_excluded_input_count": report[
            "duplicate_excluded_input_count"
        ],
        "labels_or_outcomes_opened_for_continuation": False,
        "settlement_pnl_opened_for_continuation": False,
        **_blocked_safety_fields(),
    }
    manifest["continuation_manifest_id"] = canonical_json_sha256(
        manifest
    )
    manifest_path = (
        run_dir / "pairwise_precollection_continuation_manifest.json"
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
        "report": report,
        "manifest": manifest,
        "role_assignment_result": role_result,
    }


def _continuation_decision(
    *,
    attempted_capture_count: int,
    initial_attempt_count: int,
    maximum_attempt_count: int,
    target_market_count: int,
    preflight_blocking_reason_codes: list[str],
    role_assignment_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if preflight_blocking_reason_codes:
        return {
            "status": "BLOCKED_FAIL_CLOSED",
            "support_only_failure": False,
            "continuation_allowed": False,
            "continuation_required": False,
            "continuation_attempt_count": 0,
            "continuation_reason_codes": [],
            "blocking_reason_codes": sorted(
                set(preflight_blocking_reason_codes)
            ),
        }
    if attempted_capture_count < initial_attempt_count:
        remaining = initial_attempt_count - attempted_capture_count
        return {
            "status": "INITIAL_COLLECTION_INCOMPLETE",
            "support_only_failure": True,
            "continuation_allowed": True,
            "continuation_required": True,
            "continuation_attempt_count": remaining,
            "continuation_reason_codes": [
                "complete_frozen_initial_capture_attempts"
            ],
            "blocking_reason_codes": [],
        }
    if role_assignment_report is None:
        return {
            "status": "BLOCKED_FAIL_CLOSED",
            "support_only_failure": False,
            "continuation_allowed": False,
            "continuation_required": False,
            "continuation_attempt_count": 0,
            "continuation_reason_codes": [],
            "blocking_reason_codes": [
                "outcome_blind_role_assignment_missing"
            ],
        }
    if role_assignment_report.get("role_assignment_ready") is True:
        selected_count = int(
            role_assignment_report.get("selected_market_count") or 0
        )
        blockers = list(
            role_assignment_report.get("blocking_reason_codes") or []
        )
        if selected_count != target_market_count or blockers:
            return {
                "status": "BLOCKED_FAIL_CLOSED",
                "support_only_failure": False,
                "continuation_allowed": False,
                "continuation_required": False,
                "continuation_attempt_count": 0,
                "continuation_reason_codes": [],
                "blocking_reason_codes": [
                    "role_assignment_ready_state_inconsistent"
                ],
            }
        return {
            "status": "OUTCOME_BLIND_SUPPORT_TARGET_READY",
            "support_only_failure": False,
            "continuation_allowed": False,
            "continuation_required": False,
            "continuation_attempt_count": 0,
            "continuation_reason_codes": [
                "target_valid_market_support_reached"
            ],
            "blocking_reason_codes": [],
        }
    role_blockers = {
        str(reason)
        for reason in role_assignment_report.get(
            "blocking_reason_codes"
        )
        or []
    }
    support_only = bool(role_blockers) and role_blockers <= (
        SUPPORT_ONLY_ROLE_BLOCKERS
    )
    if not support_only:
        return {
            "status": "BLOCKED_FAIL_CLOSED",
            "support_only_failure": False,
            "continuation_allowed": False,
            "continuation_required": False,
            "continuation_attempt_count": 0,
            "continuation_reason_codes": [],
            "blocking_reason_codes": sorted(
                role_blockers
                or {"role_assignment_failed_without_reason_codes"}
            ),
        }
    remaining = maximum_attempt_count - attempted_capture_count
    if remaining <= 0:
        return {
            "status": "BLOCKED_INSUFFICIENT_SUPPORT_AT_FROZEN_MAXIMUM",
            "support_only_failure": True,
            "continuation_allowed": False,
            "continuation_required": False,
            "continuation_attempt_count": 0,
            "continuation_reason_codes": [],
            "blocking_reason_codes": [
                "insufficient_support_at_frozen_maximum"
            ],
        }
    return {
        "status": "BOUNDED_SUPPORT_CONTINUATION_ALLOWED",
        "support_only_failure": True,
        "continuation_allowed": True,
        "continuation_required": True,
        "continuation_attempt_count": remaining,
        "continuation_reason_codes": [
            "support_only_role_assignment_failure",
            "continue_within_frozen_maximum",
        ],
        "blocking_reason_codes": [],
    }


def _batch_preflight(
    pins: tuple[tuple[Path, str], ...],
) -> tuple[
    tuple[tuple[Path, str], ...],
    list[dict[str, str]],
    dict[str, Any],
]:
    unique: list[tuple[Path, str]] = []
    duplicates: list[dict[str, str]] = []
    seen_exact: set[tuple[Path, str]] = set()
    seen_paths: dict[Path, str] = {}
    blockers: list[str] = []
    batch_ids: list[str] = []
    capture_run_ids: list[str] = []
    attempted_capture_count = 0
    forbidden_batch_field_paths: list[str] = []
    collector_batch_error_count = 0
    for path, expected_sha256 in pins:
        resolved = path.resolve()
        exact = (resolved, expected_sha256.lower())
        if exact in seen_exact:
            duplicates.append(
                {
                    "path": str(resolved),
                    "sha256": expected_sha256.lower(),
                    "reason_code": "duplicate_exact_batch_progress_pin",
                }
            )
            continue
        previous_sha = seen_paths.get(resolved)
        if previous_sha is not None and previous_sha != expected_sha256:
            blockers.append("same_batch_path_with_conflicting_hash")
            continue
        _verify_pin(
            resolved,
            expected_sha256,
            name="batch progress",
        )
        seen_exact.add(exact)
        seen_paths[resolved] = expected_sha256.lower()
        unique.append(exact)
        payload = _load_json(resolved)
        forbidden_batch_field_paths.extend(
            str(value)
            for value in _find_fields(
                payload,
                FORBIDDEN_REGISTRY_FIELDS,
            )
        )
        if (
            payload.get("paper_only") is not True
            or payload.get("capital_at_risk") is not False
        ):
            blockers.append("collector_batch_safety_contract_failed")
        batch_error_count = int(payload.get("error_count") or 0)
        collector_batch_error_count += batch_error_count
        if batch_error_count:
            blockers.append("collector_batch_error_count_nonzero")
        captures = [dict(row) for row in payload.get("captures") or []]
        capture_count = int(payload.get("capture_count") or 0)
        if capture_count != len(captures):
            blockers.append("batch_capture_count_mismatch")
        attempted_capture_count += len(captures)
        batch_ids.append(str(payload.get("batch_id") or ""))
        capture_run_ids.extend(
            str(row.get("run_id") or "") for row in captures
        )
    if any(not value for value in batch_ids):
        blockers.append("batch_id_missing")
    if len(batch_ids) != len(set(batch_ids)):
        blockers.append("duplicate_batch_id")
    if any(not value for value in capture_run_ids):
        blockers.append("capture_run_id_missing")
    if len(capture_run_ids) != len(set(capture_run_ids)):
        blockers.append("duplicate_capture_run_id")
    if forbidden_batch_field_paths:
        blockers.append(
            "batch_progress_forbidden_outcome_fields_present"
        )
    return (
        tuple(unique),
        duplicates,
        {
            "attempted_capture_count": attempted_capture_count,
            "batch_ids": batch_ids,
            "capture_run_id_count": len(capture_run_ids),
            "forbidden_batch_field_paths": sorted(
                set(forbidden_batch_field_paths)
            ),
            "collector_batch_error_count": (
                collector_batch_error_count
            ),
            "blocking_reason_codes": sorted(set(blockers)),
        },
    )


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Pairwise Precollection Support Gate",
            "",
            f"- status: `{report['status']}`",
            f"- attempted captures: `{report['attempted_capture_count']}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- target markets: `{report['target_valid_market_count']}`",
            f"- continuation allowed: "
            f"`{str(report['continuation_allowed']).lower()}`",
            f"- continuation attempts: "
            f"`{report['continuation_attempt_count']}`",
            f"- blocking reasons: "
            f"`{json.dumps(report['blocking_reason_codes'])}`",
            "- role assignment/continuation uses labels or outcomes: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )
