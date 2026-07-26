"""Fail-closed statistical budget control for v8 execution-candidate families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from bigan.v8.canonical_payload import canonical_payload_sha256

CANDIDATE_IDENTITY_SCHEMA_VERSION = "bigan-v8-candidate-identity-v1"
LEDGER_ENTRY_SCHEMA_VERSION = "bigan-v8-candidate-ledger-entry-v1"
ELIGIBILITY_SCHEMA_VERSION = "bigan-v8-candidate-budget-eligibility-v1"
ZERO_SHA256 = "0" * 64
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

    validate_append_only_ledger(attempt_ledger)
    validate_append_only_ledger(evidence_ledger)
    blockers: list[str] = []
    family_id = str(proposed_attempt.get("family_id") or "")
    if family_id != family_manifest.get("family_id"):
        blockers.append("candidate_family_id_mismatch")
    if proposed_attempt.get("target_outcomes_opened") is not False:
        blockers.append("proposed_attempt_not_target_free")
    if proposed_attempt.get("decision_freeze_complete") is not True:
        blockers.append("candidate_decisions_not_frozen")
    if proposed_attempt.get("shared_window_source_rows_frozen") is not True:
        blockers.append("shared_window_source_rows_not_frozen")
    existing_attempt_ids = {
        str(entry.get("attempt_id") or "") for entry in attempt_ledger
    }
    if proposed_attempt.get("attempt_id") in existing_attempt_ids:
        blockers.append("duplicate_attempt_id")
    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in family_manifest.get("candidates", [])
    }
    proposed_candidates = list(proposed_attempt.get("candidate_ids") or [])
    if not proposed_candidates or any(candidate not in candidates for candidate in proposed_candidates):
        blockers.append("unknown_or_empty_candidate_set")
    identities = [
        str(candidates[candidate]["stable_candidate_identity"])
        for candidate in proposed_candidates
        if candidate in candidates
    ]
    if len(identities) != len(set(identities)):
        blockers.append("candidate_alias_identity_collision")
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
        "attempt_id": proposed_attempt.get("attempt_id"),
        "next_attempt_number": next_attempt_number,
        "maximum_confirmatory_attempts": maximum_attempts,
        "parallel_candidate_count": parallel_count,
        "window_familywise_alpha": window_alpha,
        "per_candidate_alpha": per_candidate_alpha,
        "parallel_method": error_control_contract.get("parallel_method"),
        "next_gate_eligible": eligible,
        "reason_codes": sorted(set(blockers)),
        "outcomes_read_to_make_eligibility_decision": False,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CandidateBudgetError(f"{name} must be lowercase SHA-256")


__all__ = [
    "CANDIDATE_IDENTITY_SCHEMA_VERSION",
    "ELIGIBILITY_SCHEMA_VERSION",
    "LEDGER_ENTRY_SCHEMA_VERSION",
    "AttemptConsumption",
    "CandidateBudgetError",
    "classify_attempt_consumption",
    "evaluate_next_gate_eligibility",
    "ledger_entry_sha256",
    "stable_candidate_identity",
    "validate_append_only_ledger",
]
