"""Fail-closed statistical budget control for v8 execution-candidate families."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from bigan.v8.canonical_payload import canonical_payload_sha256

CANDIDATE_IDENTITY_SCHEMA_VERSION = "bigan-v8-candidate-identity-v1"
LEDGER_ENTRY_SCHEMA_VERSION = "bigan-v8-candidate-ledger-entry-v1"
ELIGIBILITY_SCHEMA_VERSION = "bigan-v8-candidate-budget-eligibility-v1"
FAMILY_MANIFEST_SCHEMA_VERSION = "bigan-v8-candidate-family-manifest-v1"
BUDGET_PROTOCOL_SCHEMA_VERSION = "bigan-v8-candidate-budget-protocol-v1"
ERROR_CONTROL_SCHEMA_VERSION = "bigan-v8-family-error-control-v1"
ATTEMPT_LEDGER_SCHEMA_VERSION = "bigan-v8-candidate-attempt-ledger-v1"
EVIDENCE_LEDGER_SCHEMA_VERSION = "bigan-v8-evidence-consumption-ledger-v1"
ZERO_SHA256 = "0" * 64
SAFETY = {
    "paper_candidate_unlocked": False,
    "promotion_unlocked": False,
    "live_unlocked": False,
    "write_enabled": False,
    "wallet_enabled": False,
    "capital_at_risk": False,
    "handoff_enabled": False,
    "source_change_enabled": False,
    "freeze_change_enabled": False,
    "promotion_evidence_eligible": False,
    "#134_resume_allowed": False,
    "#146_start_allowed": False,
}
AttemptCase = Literal[
    "engineering_rerun_identical_decisions",
    "new_decision_candidate",
    "parallel_shared_window_candidate",
    "sequential_new_window_candidate",
    "bug_fix_before_target_access",
    "bug_fix_after_target_access",
    "invalid_pre_target_window",
    "target_opened_window",
]


class CandidateBudgetError(ValueError):
    """Raised when a candidate or evidence attempt violates the frozen budget."""


@dataclass(frozen=True, slots=True)
class AttemptConsumption:
    consumes_attempt: bool
    consumes_alpha: bool
    evidence_permanently_consumed: bool
    terminal_for_candidate_hash: bool
    reason: str


def stable_candidate_identity(
    *,
    source_model_hash: str,
    execution_policy_hash: str,
    candidate_definition_hash: str,
) -> str:
    """Bind identity to behavior hashes so renaming cannot reset history."""

    for name, value in (
        ("source_model_hash", source_model_hash),
        ("execution_policy_hash", execution_policy_hash),
        ("candidate_definition_hash", candidate_definition_hash),
    ):
        _require_sha256(value, name=name)
    return canonical_payload_sha256(
        {
            "source_model_hash": source_model_hash,
            "execution_policy_hash": execution_policy_hash,
            "candidate_definition_hash": candidate_definition_hash,
        },
        payload_schema_version=CANDIDATE_IDENTITY_SCHEMA_VERSION,
    )


def classify_attempt_consumption(
    *,
    attempt_case: AttemptCase,
    target_outcomes_opened: bool,
    decisions_identical_to_prior: bool,
) -> AttemptConsumption:
    """Apply the preregistered consumption rule without reading performance."""

    if attempt_case == "engineering_rerun_identical_decisions":
        if not decisions_identical_to_prior:
            raise CandidateBudgetError("engineering rerun changed decisions")
        return AttemptConsumption(False, False, target_outcomes_opened, False, "reproduction_only")
    if attempt_case in {"bug_fix_before_target_access", "invalid_pre_target_window"}:
        if target_outcomes_opened:
            raise CandidateBudgetError("pre-target case cannot have opened target outcomes")
        return AttemptConsumption(False, False, False, False, "no_target_claim_opened")
    if attempt_case == "bug_fix_after_target_access":
        if not target_outcomes_opened:
            raise CandidateBudgetError("post-outcome bug fix requires opened target outcomes")
        return AttemptConsumption(True, True, True, True, "post_outcome_change_consumes_attempt")
    if attempt_case == "target_opened_window" and not target_outcomes_opened:
        raise CandidateBudgetError("target-opened case requires opened outcomes")
    if attempt_case in {
        "new_decision_candidate",
        "parallel_shared_window_candidate",
        "sequential_new_window_candidate",
        "target_opened_window",
    }:
        return AttemptConsumption(
            target_outcomes_opened,
            target_outcomes_opened,
            target_outcomes_opened,
            target_outcomes_opened,
            "decision_claim_consumed" if target_outcomes_opened else "reserved_until_target_access",
        )
    raise CandidateBudgetError(f"unsupported attempt case: {attempt_case}")


def ledger_entry_sha256(entry: dict[str, Any]) -> str:
    payload = dict(entry)
    payload.pop("entry_sha256", None)
    return canonical_payload_sha256(
        payload,
        payload_schema_version=LEDGER_ENTRY_SCHEMA_VERSION,
    )


def validate_append_only_ledger(
    entries: list[dict[str, Any]],
    *,
    previous_entries: list[dict[str, Any]] | None = None,
) -> None:
    """Validate sequence, hash chain, IDs, and an optional immutable prefix."""

    if previous_entries is not None and entries[: len(previous_entries)] != previous_entries:
        raise CandidateBudgetError("append-only ledger prefix was changed")
    seen_ids: set[str] = set()
    previous_hash = ZERO_SHA256
    for expected_sequence, entry in enumerate(entries, start=1):
        if entry.get("schema_version") != LEDGER_ENTRY_SCHEMA_VERSION:
            raise CandidateBudgetError("ledger entry schema invalid")
        if int(entry.get("sequence") or 0) != expected_sequence:
            raise CandidateBudgetError("ledger sequence is not contiguous")
        entry_id = str(entry.get("entry_id") or "")
        if not entry_id or entry_id in seen_ids:
            raise CandidateBudgetError("ledger entry_id is missing or duplicated")
        seen_ids.add(entry_id)
        if entry.get("previous_entry_sha256") != previous_hash:
            raise CandidateBudgetError("ledger hash chain is broken")
        expected_hash = ledger_entry_sha256(entry)
        if entry.get("entry_sha256") != expected_hash:
            raise CandidateBudgetError("ledger entry hash mismatch")
        previous_hash = expected_hash


def validate_candidate_budget_artifacts(
    *,
    family_manifest: dict[str, Any],
    budget_protocol: dict[str, Any],
    error_control_contract: dict[str, Any],
    attempt_ledger: dict[str, Any],
    evidence_ledger: dict[str, Any],
) -> None:
    """Validate the complete issue #255 preregistration and ledger semantics."""

    _validate_family_manifest(family_manifest)
    _validate_budget_protocol(
        budget_protocol,
        family_id=str(family_manifest["family_id"]),
    )
    _validate_error_control_contract(
        error_control_contract,
        family_id=str(family_manifest["family_id"]),
        maximum_attempts=int(
            budget_protocol["maximum_confirmatory_attempts"]
        ),
    )
    if attempt_ledger.get("schema_version") != ATTEMPT_LEDGER_SCHEMA_VERSION:
        raise CandidateBudgetError("attempt ledger schema invalid")
    if evidence_ledger.get("schema_version") != EVIDENCE_LEDGER_SCHEMA_VERSION:
        raise CandidateBudgetError("evidence ledger schema invalid")
    attempt_entries = list(attempt_ledger.get("entries") or [])
    evidence_entries = list(evidence_ledger.get("entries") or [])
    validate_append_only_ledger(attempt_entries)
    validate_append_only_ledger(evidence_entries)
    _validate_attempt_ledger_semantics(
        attempt_entries,
        family_manifest=family_manifest,
    )
    _validate_evidence_ledger_semantics(evidence_entries)


def validate_eligibility_decision(
    decision: dict[str, Any],
    *,
    recomputed: dict[str, Any],
) -> None:
    """Require the committed machine decision to equal fresh validation."""

    if decision != recomputed:
        raise CandidateBudgetError(
            "committed next-gate eligibility decision is stale or altered"
        )
    if decision.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION:
        raise CandidateBudgetError("eligibility decision schema invalid")
    if any(decision.get(field) is not expected for field, expected in SAFETY.items()):
        raise CandidateBudgetError("eligibility decision safety contract invalid")


def evaluate_next_gate_eligibility(
    *,
    family_manifest: dict[str, Any],
    budget_protocol: dict[str, Any],
    error_control_contract: dict[str, Any],
    attempt_ledger: list[dict[str, Any]],
    evidence_ledger: list[dict[str, Any]],
    proposed_attempt: dict[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable, fail-closed next-gate eligibility decision."""

    _validate_family_manifest(family_manifest)
    _validate_budget_protocol(
        budget_protocol,
        family_id=str(family_manifest["family_id"]),
    )
    _validate_error_control_contract(
        error_control_contract,
        family_id=str(family_manifest["family_id"]),
        maximum_attempts=int(
            budget_protocol["maximum_confirmatory_attempts"]
        ),
    )
    validate_append_only_ledger(attempt_ledger)
    validate_append_only_ledger(evidence_ledger)
    _validate_attempt_ledger_semantics(
        attempt_ledger,
        family_manifest=family_manifest,
    )
    _validate_evidence_ledger_semantics(evidence_ledger)
    blockers: list[str] = []
    family_id = str(proposed_attempt.get("family_id") or "")
    if family_id != family_manifest.get("family_id"):
        blockers.append("candidate_family_id_mismatch")
    if (
        proposed_attempt.get("attempt_case")
        != "parallel_shared_window_candidate"
    ):
        blockers.append("proposed_attempt_case_invalid")
    if proposed_attempt.get("target_outcomes_opened") is not False:
        blockers.append("proposed_attempt_not_target_free")
    if proposed_attempt.get("decision_freeze_complete") is not True:
        blockers.append("candidate_decisions_not_frozen")
    if proposed_attempt.get("shared_window_source_rows_frozen") is not True:
        blockers.append("shared_window_source_rows_not_frozen")
    existing_attempt_ids = {
        str(entry.get("attempt_id") or "") for entry in attempt_ledger
    }
    proposed_attempt_id = str(proposed_attempt.get("attempt_id") or "")
    if not proposed_attempt_id:
        blockers.append("attempt_id_missing")
    elif proposed_attempt_id in existing_attempt_ids:
        blockers.append("duplicate_attempt_id")
    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in family_manifest.get("candidates", [])
    }
    proposed_candidates = list(proposed_attempt.get("candidate_ids") or [])
    if len(proposed_candidates) != len(set(proposed_candidates)):
        blockers.append("duplicate_candidate_id")
    if not proposed_candidates or any(candidate not in candidates for candidate in proposed_candidates):
        blockers.append("unknown_or_empty_candidate_set")
    identities = [
        str(candidates[candidate]["stable_candidate_identity"])
        for candidate in proposed_candidates
        if candidate in candidates
    ]
    if len(identities) != len(set(identities)):
        blockers.append("candidate_alias_identity_collision")
    expected_identity_pins = {
        candidate_id: str(candidates[candidate_id]["stable_candidate_identity"])
        for candidate_id in proposed_candidates
        if candidate_id in candidates
    }
    if proposed_attempt.get("candidate_stable_identities") != expected_identity_pins:
        blockers.append("candidate_identity_pins_mismatch")
    ledger_identity_to_names: dict[str, set[str]] = {}
    for entry in attempt_ledger:
        identity = str(entry.get("stable_candidate_identity") or "")
        candidate_id = str(entry.get("candidate_id") or "")
        if identity:
            ledger_identity_to_names.setdefault(identity, set()).add(candidate_id)
    for candidate_id in proposed_candidates:
        if candidate_id not in candidates:
            continue
        identity = str(candidates[candidate_id]["stable_candidate_identity"])
        prior_names = ledger_identity_to_names.get(identity, set())
        if prior_names and candidate_id not in prior_names:
            blockers.append("candidate_rename_cannot_reset_history")
    for entry in attempt_ledger:
        if entry.get("family_id") != family_id:
            continue
        prior_candidate = str(entry.get("candidate_id") or "")
        prior_identity = str(entry.get("stable_candidate_identity") or "")
        if (
            prior_candidate not in candidates
            or prior_identity
            != str(candidates[prior_candidate]["stable_candidate_identity"])
        ):
            blockers.append("prior_family_candidate_identity_invalid")
    opened_not_consumed = [
        entry["entry_id"]
        for entry in evidence_ledger
        if entry.get("target_outcomes_opened") is True
        and entry.get("evidence_permanently_consumed") is not True
    ]
    if opened_not_consumed:
        blockers.append("opened_evidence_not_marked_consumed")
    consumed_attempts = {
        str(entry.get("attempt_id"))
        for entry in attempt_ledger
        if entry.get("family_id") == family_id and entry.get("consumes_attempt") is True
    }
    maximum_attempts = int(budget_protocol.get("maximum_confirmatory_attempts") or 0)
    next_attempt_number = len(consumed_attempts) + 1
    if maximum_attempts <= 0 or next_attempt_number > maximum_attempts:
        blockers.append("candidate_family_attempt_budget_exhausted")
    spending = list(error_control_contract.get("sequential_alpha_spending") or [])
    window_alpha = (
        float(spending[next_attempt_number - 1])
        if 0 < next_attempt_number <= len(spending)
        else 0.0
    )
    parallel_count = len(proposed_candidates)
    per_candidate_alpha = window_alpha / parallel_count if parallel_count else 0.0
    if (
        error_control_contract.get("parallel_method") != "bonferroni"
        or parallel_count > int(error_control_contract.get("maximum_parallel_candidates") or 0)
        or per_candidate_alpha <= 0.0
    ):
        blockers.append("family_error_control_allocation_invalid")
    eligible = not blockers
    return {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "family_id": family_id,
        "attempt_id": proposed_attempt_id,
        "attempt_case": proposed_attempt.get("attempt_case"),
        "candidate_ids": proposed_candidates,
        "candidate_stable_identities": expected_identity_pins,
        "next_attempt_number": next_attempt_number,
        "maximum_confirmatory_attempts": maximum_attempts,
        "parallel_candidate_count": parallel_count,
        "window_familywise_alpha": window_alpha,
        "per_candidate_alpha": per_candidate_alpha,
        "parallel_method": error_control_contract.get("parallel_method"),
        "next_gate_eligible": eligible,
        "reason_codes": sorted(set(blockers)),
        "outcomes_read_to_make_eligibility_decision": False,
        **SAFETY,
    }


def _validate_family_manifest(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema_version") != FAMILY_MANIFEST_SCHEMA_VERSION
        or manifest.get("issue") != 255
        or not str(manifest.get("family_id") or "")
        or not str(manifest.get("source_model_lineage") or "")
        or manifest.get("history_reset_by_branch_version_or_name_allowed")
        is not False
        or manifest.get("new_family_requires_new_lineage_and_evidence_program")
        is not True
        or manifest.get("safety") != SAFETY
    ):
        raise CandidateBudgetError("candidate family manifest contract invalid")
    source_model_hash = str(manifest.get("source_model_hash") or "")
    _require_sha256(source_model_hash, name="source_model_hash")
    candidates = list(manifest.get("candidates") or [])
    candidate_ids: set[str] = set()
    identities: set[str] = set()
    if not candidates:
        raise CandidateBudgetError("candidate family is empty")
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in candidate_ids:
            raise CandidateBudgetError("candidate_id is missing or duplicated")
        candidate_ids.add(candidate_id)
        execution_policy_hash = str(
            candidate.get("execution_policy_hash") or ""
        )
        candidate_definition_hash = str(
            candidate.get("candidate_definition_hash") or ""
        )
        expected_identity = stable_candidate_identity(
            source_model_hash=source_model_hash,
            execution_policy_hash=execution_policy_hash,
            candidate_definition_hash=candidate_definition_hash,
        )
        identity = str(candidate.get("stable_candidate_identity") or "")
        if identity != expected_identity or identity in identities:
            raise CandidateBudgetError(
                "candidate stable identity is invalid or duplicated"
            )
        identities.add(identity)
        if (
            not str(candidate.get("abstention_semantics") or "")
            or type(candidate.get("fallback_enabled")) is not bool
        ):
            raise CandidateBudgetError("candidate behavior contract invalid")
    matched_baseline = dict(manifest.get("matched_baseline") or {})
    if not str(matched_baseline.get("candidate_id") or ""):
        raise CandidateBudgetError("matched baseline candidate_id missing")
    _require_sha256(
        str(matched_baseline.get("execution_policy_hash") or ""),
        name="matched_baseline_execution_policy_hash",
    )


def _validate_budget_protocol(
    protocol: dict[str, Any],
    *,
    family_id: str,
) -> None:
    expected_cases = {
        "engineering_rerun_identical_decisions": {
            "consumes_attempt": False,
            "consumes_alpha": False,
        },
        "new_candidate_changed_decisions": {
            "consumes_when_target_opened": True,
            "consumes_alpha_when_target_opened": True,
        },
        "parallel_candidates_shared_window": {
            "one_sequential_attempt": True,
            "parallel_multiplicity_correction": "bonferroni",
        },
        "sequential_candidate_new_window": {
            "consumes_attempt": True,
            "uses_next_alpha_spending_increment": True,
        },
        "bug_fix_before_target_access": {
            "consumes_attempt": False,
            "requires_new_hash_and_refreeze": True,
        },
        "bug_fix_after_target_access": {
            "consumes_attempt": True,
            "consumes_alpha": True,
            "old_hash_terminal": True,
        },
        "invalid_incomplete_pre_target_window": {
            "consumes_attempt": False,
            "requires_no_target_claim_opened": True,
        },
        "outcomes_opened": {
            "evidence_permanently_consumed": True,
            "attempt_consumed": True,
        },
    }
    if (
        protocol.get("schema_version") != BUDGET_PROTOCOL_SCHEMA_VERSION
        or protocol.get("family_id") != family_id
        or protocol.get("issue") != 255
        or int(protocol.get("maximum_confirmatory_attempts") or 0) <= 0
        or protocol.get("development_evidence_can_claim_promotion") is not False
        or protocol.get("consumed_validation_evidence_reusable") is not False
        or protocol.get("cases") != expected_cases
        or protocol.get("duplicate_attempt_id_rejected") is not True
        or protocol.get("altered_hash_rejected") is not True
        or protocol.get("append_only_ledgers_required") is not True
        or protocol.get("fail_closed") is not True
        or protocol.get("permissions_granted_by_protocol") != []
        or protocol.get("safety") != SAFETY
    ):
        raise CandidateBudgetError("candidate budget protocol contract invalid")


def _validate_error_control_contract(
    contract: dict[str, Any],
    *,
    family_id: str,
    maximum_attempts: int,
) -> None:
    spending = list(contract.get("sequential_alpha_spending") or [])
    familywise_alpha = float(contract.get("familywise_alpha") or 0.0)
    maximum_parallel = int(contract.get("maximum_parallel_candidates") or 0)
    if (
        contract.get("schema_version") != ERROR_CONTROL_SCHEMA_VERSION
        or contract.get("family_id") != family_id
        or not 0.0 < familywise_alpha < 0.5
        or contract.get("sequential_method") != "preregistered_alpha_spending"
        or len(spending) != maximum_attempts
        or any(
            type(value) not in {int, float} or not 0.0 < float(value) < 0.5
            for value in spending
        )
        or math.fsum(float(value) for value in spending)
        > familywise_alpha
        or contract.get("parallel_method") != "bonferroni"
        or maximum_parallel <= 0
        or not math.isclose(
            float(contract.get("attempt_1_parallel_candidate_alpha") or 0.0),
            float(spending[0]) / maximum_parallel if spending else 0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or contract.get("method_selected_before_candidate_outcomes") is not True
        or contract.get(
            "winner_selection_after_results_without_preregistered_rule_allowed"
        )
        is not False
        or contract.get("confidence_intervals_use_adjusted_alpha") is not True
        or contract.get("permissions_granted_by_error_control") != []
        or contract.get("safety") != SAFETY
    ):
        raise CandidateBudgetError("family error control contract invalid")


def _validate_attempt_ledger_semantics(
    entries: list[dict[str, Any]],
    *,
    family_manifest: dict[str, Any],
) -> None:
    known_cases = {
        "engineering_rerun_identical_decisions",
        "new_decision_candidate",
        "parallel_shared_window_candidate",
        "sequential_new_window_candidate",
        "bug_fix_before_target_access",
        "bug_fix_after_target_access",
        "invalid_pre_target_window",
        "target_opened_window",
    }
    family_id = str(family_manifest["family_id"])
    candidates = {
        str(candidate["candidate_id"]): str(
            candidate["stable_candidate_identity"]
        )
        for candidate in family_manifest["candidates"]
    }
    seen_attempt_candidates: set[tuple[str, str]] = set()
    attempt_contracts: dict[str, tuple[Any, ...]] = {}
    for entry in entries:
        attempt_id = str(entry.get("attempt_id") or "")
        entry_family = str(entry.get("family_id") or "")
        candidate_id = str(entry.get("candidate_id") or "")
        attempt_case = str(entry.get("attempt_case") or "")
        opened = entry.get("target_outcomes_opened")
        consumes_attempt = entry.get("consumes_attempt")
        consumes_alpha = entry.get("consumes_alpha")
        terminal = entry.get("terminal")
        attempt_candidate = (attempt_id, candidate_id)
        if (
            not attempt_id
            or attempt_candidate in seen_attempt_candidates
            or not entry_family
            or not candidate_id
            or attempt_case not in known_cases
            or type(opened) is not bool
            or type(consumes_attempt) is not bool
            or type(consumes_alpha) is not bool
            or type(terminal) is not bool
        ):
            raise CandidateBudgetError("attempt ledger entry semantics invalid")
        seen_attempt_candidates.add(attempt_candidate)
        attempt_contract = (
            entry_family,
            attempt_case,
            opened,
            consumes_attempt,
            consumes_alpha,
            terminal,
        )
        if (
            attempt_id in attempt_contracts
            and attempt_contracts[attempt_id] != attempt_contract
        ):
            raise CandidateBudgetError(
                "parallel attempt entries disagree on consumption"
            )
        attempt_contracts[attempt_id] = attempt_contract
        if opened is True and (
            consumes_attempt is not True
            or consumes_alpha is not True
            or terminal is not True
        ):
            raise CandidateBudgetError(
                "opened attempt is not permanently consumed"
            )
        if opened is False and (
            consumes_attempt is not False
            or consumes_alpha is not False
            or terminal is not False
        ):
            raise CandidateBudgetError(
                "unopened attempt consumed budget or became terminal"
            )
        identity = str(entry.get("stable_candidate_identity") or "")
        if entry_family == family_id:
            if candidate_id not in candidates or identity != candidates[candidate_id]:
                raise CandidateBudgetError(
                    "current-family ledger identity is invalid"
                )
        elif identity:
            _require_sha256(identity, name="legacy_stable_candidate_identity")


def _validate_evidence_ledger_semantics(
    entries: list[dict[str, Any]],
) -> None:
    seen_evidence_ids: set[str] = set()
    for entry in entries:
        evidence_id = str(entry.get("evidence_id") or "")
        opened = entry.get("target_outcomes_opened")
        consumed = entry.get("evidence_permanently_consumed")
        reusable = entry.get("reusable_for_promotion")
        if (
            not evidence_id
            or evidence_id in seen_evidence_ids
            or type(opened) is not bool
            or type(consumed) is not bool
            or type(reusable) is not bool
        ):
            raise CandidateBudgetError("evidence ledger entry semantics invalid")
        seen_evidence_ids.add(evidence_id)
        if opened is True and (
            consumed is not True or reusable is not False
        ):
            raise CandidateBudgetError(
                "opened evidence is not permanently consumed"
            )
        if opened is False and consumed is True:
            raise CandidateBudgetError(
                "unopened evidence cannot be marked consumed"
            )


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CandidateBudgetError(f"{name} must be lowercase SHA-256")


__all__ = [
    "ATTEMPT_LEDGER_SCHEMA_VERSION",
    "BUDGET_PROTOCOL_SCHEMA_VERSION",
    "CANDIDATE_IDENTITY_SCHEMA_VERSION",
    "ELIGIBILITY_SCHEMA_VERSION",
    "ERROR_CONTROL_SCHEMA_VERSION",
    "EVIDENCE_LEDGER_SCHEMA_VERSION",
    "FAMILY_MANIFEST_SCHEMA_VERSION",
    "LEDGER_ENTRY_SCHEMA_VERSION",
    "SAFETY",
    "AttemptConsumption",
    "CandidateBudgetError",
    "classify_attempt_consumption",
    "evaluate_next_gate_eligibility",
    "ledger_entry_sha256",
    "stable_candidate_identity",
    "validate_append_only_ledger",
    "validate_candidate_budget_artifacts",
    "validate_eligibility_decision",
]
