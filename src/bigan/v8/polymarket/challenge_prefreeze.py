"""Fail-closed prerequisites for the v8.5 challenge collection refreeze."""

from __future__ import annotations

import re
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.exact_model_runtime_binding import (
    validate_runtime_binding_summary,
)

EXCLUDED_CAPTURE_LEDGER_SCHEMA_VERSION = (
    "bigan-v8-challenge-excluded-capture-ledger-v1"
)
PREFREEZE_CHECKLIST_SCHEMA_VERSION = (
    "bigan-v8-challenge-prefreeze-checklist-v1"
)
SAFETY_FALSE_FIELDS = (
    "paper_candidate_allowed",
    "live_trading_enabled",
    "capital_at_risk",
    "polymarket_write_enabled",
    "wallet_signing_enabled",
    "v8_execution_handoff_allowed",
    "source_model_candidate_eligible",
    "freeze_ready",
    "promotion_evidence_eligible",
    "#134_resume_allowed",
    "#146_start_allowed",
)


class ChallengePrefreezeError(ValueError):
    """Raised when collection prerequisites are incomplete or mutable."""


def validate_excluded_capture_ledger(ledger: dict[str, Any]) -> None:
    """Require a complete, target-blind ledger of every excluded capture."""

    entries = ledger.get("entries")
    incomplete = ledger.get("unmaterialized_capture_directories")
    blockers: list[str] = []
    if (
        ledger.get("schema_version")
        != EXCLUDED_CAPTURE_LEDGER_SCHEMA_VERSION
    ):
        blockers.append("schema_version")
    if ledger.get("issue") != 254:
        blockers.append("issue")
    if ledger.get("complete") is not True:
        blockers.append("complete")
    if not isinstance(entries, list) or not entries:
        blockers.append("entries")
        entries = []
    if int(ledger.get("excluded_entry_count") or -1) != len(entries):
        blockers.append("excluded_entry_count")
    for index, entry_value in enumerate(entries):
        entry = dict(entry_value or {})
        prefix = f"entry_{index}"
        if entry.get("excluded_from_fresh_attempt") is not True:
            blockers.append(f"{prefix}_excluded")
        if entry.get("labels_outcomes_or_pnl_opened") is not False:
            blockers.append(f"{prefix}_target_access")
        if entry.get("consumes_attempt_or_alpha") is not False:
            blockers.append(f"{prefix}_attempt_accounting")
        if not str(entry.get("service_root") or ""):
            blockers.append(f"{prefix}_service_root")
        if entry.get("entry_type") == "superseded_plan_capture":
            if not _valid_sha256(
                entry.get("source_superseded_plan_sha256")
            ):
                blockers.append(f"{prefix}_source_superseded_plan_sha256")
            for name in (
                "capture_manifest_sha256",
                "capture_report_sha256",
                "raw_resolution_artifact_sha256",
            ):
                if not _valid_sha256(entry.get(name)):
                    blockers.append(f"{prefix}_{name}")
            if entry.get("index_entry_written") is not False:
                blockers.append(f"{prefix}_indexed")
            if int(entry.get("raw_resolution_row_count") or 0) != 0:
                blockers.append(f"{prefix}_resolution_rows")
    if not _valid_sha256(ledger.get("superseded_collection_plan_sha256")):
        blockers.append("superseded_collection_plan_sha256")
    if not str(ledger.get("immediate_superseded_plan_service_root") or ""):
        blockers.append("immediate_superseded_plan_service_root")
    if ledger.get("immediate_superseded_plan_collection_started") is not False:
        blockers.append("immediate_superseded_plan_collection_started")
    if int(ledger.get("immediate_superseded_plan_capture_count", -1)) != 0:
        blockers.append("immediate_superseded_plan_capture_count")
    if any(
        entry.get("current_superseded_plan_capture") is True
        for entry in entries
    ):
        blockers.append("immediate_superseded_plan_capture_entries")
    if not isinstance(incomplete, list):
        blockers.append("unmaterialized_capture_directories")
        incomplete = []
    for index, directory_value in enumerate(incomplete):
        directory = dict(directory_value or {})
        if (
            directory.get("file_count") != 0
            or directory.get("index_entry_written") is not False
            or directory.get("labels_outcomes_or_pnl_opened") is not False
            or directory.get("consumes_attempt_or_alpha") is not False
        ):
            blockers.append(f"unmaterialized_{index}")
    if int(ledger.get("fresh_attempt_rows_included") or 0) != 0:
        blockers.append("fresh_attempt_rows_included")
    if ledger.get("outcomes_resolution_labels_or_pnl_opened") is not False:
        blockers.append("ledger_target_access")
    if ledger.get("attempt_or_alpha_consumed") is not False:
        blockers.append("ledger_attempt_accounting")
    if blockers:
        raise ChallengePrefreezeError(
            "excluded capture ledger invalid: "
            + ", ".join(sorted(set(blockers)))
        )


def validate_prefreeze_checklist(
    checklist: dict[str, Any],
    *,
    candidate_contract: dict[str, Any],
    candidate_contract_sha256: str,
    historical_replay_report: dict[str, Any],
    historical_replay_report_sha256: str,
    excluded_capture_ledger: dict[str, Any],
    excluded_capture_ledger_sha256: str,
    collector_protocol_sha256: str,
    feature_missingness_contract_sha256: str,
    feature_missingness_runtime_schema_sha256: str,
) -> None:
    """Validate all technical prerequisites without authorizing collection."""

    blockers: list[str] = []
    historical = dict(checklist.get("historical_replay") or {})
    runtime = dict(checklist.get("runtime_byte_verification") or {})
    feature_completeness = dict(
        checklist.get("feature_completeness") or {}
    )
    alpha = dict(checklist.get("alpha_spending") or {})
    authorization = dict(checklist.get("operator_authorization") or {})
    safety = dict(checklist.get("safety") or {})
    embedded_binding = historical.get("exact_model_binding_summary")
    report_binding = historical_replay_report.get(
        "exact_model_binding_summary"
    )
    runtime_summary = runtime.get("verification_summary")
    if checklist.get("schema_version") != PREFREEZE_CHECKLIST_SCHEMA_VERSION:
        blockers.append("schema_version")
    if checklist.get("issue") != 254:
        blockers.append("issue")
    if checklist.get("preregistered_before_collection") is not True:
        blockers.append("preregistered")
    if (
        checklist.get("candidate_name")
        != candidate_contract.get("primary_policy")
        or checklist.get("candidate_name_source")
        != "candidate_contract.primary_policy"
    ):
        blockers.append("candidate_name")
    if (
        checklist.get("candidate_contract_sha256")
        != candidate_contract_sha256
        or not _valid_sha256(candidate_contract_sha256)
    ):
        blockers.append("candidate_contract_sha256")
    if (
        historical.get("report_sha256")
        != historical_replay_report_sha256
        or historical_replay_report.get(
            "historical_superiority_gate_passed"
        )
        is not True
        or historical.get("historical_superiority_gate_passed") is not True
    ):
        blockers.append("historical_replay")
    if (
        not isinstance(embedded_binding, dict)
        or embedded_binding != report_binding
        or embedded_binding.get("exact_frozen_model_binding_verified")
        is not True
        or not all((embedded_binding.get("checks") or {}).values())
    ):
        blockers.append("exact_model_binding_summary")
    try:
        validate_runtime_binding_summary(runtime_summary)
    except (TypeError, ValueError):
        blockers.append("runtime_binding_summary")
    if (
        runtime.get("implemented") is not True
        or runtime.get("tested") is not True
        or runtime.get("required_at_collector_startup") is not True
        or runtime.get("required_in_every_bounded_batch_report") is not True
        or runtime.get("mismatch_fails_before_indexable_artifacts") is not True
    ):
        blockers.append("runtime_requirements")
    expected_feature_hashes = {
        "persistent_collector_protocol_sha256": collector_protocol_sha256,
        "feature_missingness_contract_sha256": (
            feature_missingness_contract_sha256
        ),
        "feature_missingness_runtime_schema_sha256": (
            feature_missingness_runtime_schema_sha256
        ),
    }
    if feature_completeness.get("issue") != 257:
        blockers.append("feature_completeness_issue")
    for name, expected in expected_feature_hashes.items():
        if (
            not _valid_sha256(expected)
            or str(feature_completeness.get(name) or "").lower()
            != expected.lower()
        ):
            blockers.append(name)
    for name in (
        "full_round_trade_tape_collection_implemented",
        "full_round_trade_tape_collection_tested",
        "per_round_provider_health_rows_required",
        "batch_provider_health_diagnostics_required",
        "legacy_and_frozen_model_inputs_unchanged",
    ):
        if feature_completeness.get(name) is not True:
            blockers.append(name)
    if (
        feature_completeness.get(
            "outcomes_labels_settlement_returns_or_pnl_opened"
        )
        is not False
    ):
        blockers.append("feature_completeness_target_access")
    validate_excluded_capture_ledger(excluded_capture_ledger)
    if (
        checklist.get("excluded_capture_ledger_sha256")
        != excluded_capture_ledger_sha256
        or not _valid_sha256(excluded_capture_ledger_sha256)
    ):
        blockers.append("excluded_capture_ledger_sha256")
    row_hashes = dict(historical.get("replay_row_sha256s") or {})
    if row_hashes != {
        "candidate": (
            "c26c9164cada8abedcff94c05553cb43bdc9eaa138dbff1293abe09af1f4f59f"
        ),
        "baseline": (
            "82acc593d57c37c77da866f989487273865c3501ab39aa8fe346aa15cedb74df"
        ),
    }:
        blockers.append("historical_replay_rows")
    if (
        float(alpha.get("familywise_window_alpha") or 0.0) != 0.025
        or float(alpha.get("per_candidate_alpha") or 0.0) != 0.0125
        or int(alpha.get("parallel_tested_candidate_count") or 0) != 2
        or alpha.get("attempt_consumed") is not False
    ):
        blockers.append("alpha_spending")
    if (
        authorization.get("required") is not True
        or authorization.get("granted") is not False
        or checklist.get("collection_start_allowed") is not False
    ):
        blockers.append("operator_authorization")
    if safety.get("paper_only") is not True or any(
        safety.get(field) is not False for field in SAFETY_FALSE_FIELDS
    ):
        blockers.append("safety")
    if checklist.get("technical_prerequisites_satisfied") is not True:
        blockers.append("technical_prerequisites")
    checklist_without_id = {
        key: value for key, value in checklist.items() if key != "checklist_id"
    }
    if checklist.get("checklist_id") != canonical_json_sha256(
        checklist_without_id
    ):
        blockers.append("checklist_id")
    if blockers:
        raise ChallengePrefreezeError(
            "challenge prefreeze checklist invalid: "
            + ", ".join(sorted(set(blockers)))
        )


def _valid_sha256(value: Any) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()) is not None


__all__ = [
    "ChallengePrefreezeError",
    "EXCLUDED_CAPTURE_LEDGER_SCHEMA_VERSION",
    "PREFREEZE_CHECKLIST_SCHEMA_VERSION",
    "SAFETY_FALSE_FIELDS",
    "validate_excluded_capture_ledger",
    "validate_prefreeze_checklist",
]
