from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.candidate_budget import (
    CandidateBudgetError,
    classify_attempt_consumption,
    evaluate_next_gate_eligibility,
    stable_candidate_identity,
    validate_append_only_ledger,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/v8/polymarket_configs"


def _json(name: str):
    return json.loads((CONFIG / name).read_text())


def _proposed() -> dict:
    return {
        "family_id": "v8-high-signal-vs-fallback-2026q3",
        "attempt_id": "v8-parallel-future-gate-attempt-001",
        "candidate_ids": [
            "v8_1_primary_no_fallback",
            "v8_3_primary_with_fallback",
        ],
        "target_outcomes_opened": False,
        "decision_freeze_complete": True,
        "shared_window_source_rows_frozen": True,
    }


def _decision(**overrides):
    result = evaluate_next_gate_eligibility(
        family_manifest=_json("candidate_family_manifest.json"),
        budget_protocol=_json("candidate_budget_protocol.json"),
        error_control_contract=_json("family_error_control_contract.json"),
        attempt_ledger=_json("candidate_attempt_ledger.json")["entries"],
        evidence_ledger=_json("evidence_consumption_ledger.json")["entries"],
        proposed_attempt={**_proposed(), **overrides},
    )
    return result


def test_all_required_artifacts_are_hash_pinned() -> None:
    for name in (
        "candidate_family_manifest.json",
        "candidate_budget_protocol.json",
        "family_error_control_contract.json",
        "candidate_attempt_ledger.json",
        "evidence_consumption_ledger.json",
        "next_gate_eligibility_decision.json",
    ):
        path = CONFIG / name
        expected = path.with_suffix(".sha256").read_text().strip()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_candidate_identities_are_stable_and_not_name_based() -> None:
    manifest = _json("candidate_family_manifest.json")
    for candidate in manifest["candidates"]:
        assert candidate["stable_candidate_identity"] == stable_candidate_identity(
            source_model_hash=manifest["source_model_hash"],
            execution_policy_hash=candidate["execution_policy_hash"],
            candidate_definition_hash=candidate["candidate_definition_hash"],
        )


def test_append_only_ledgers_and_first_gate_eligibility() -> None:
    validate_append_only_ledger(_json("candidate_attempt_ledger.json")["entries"])
    validate_append_only_ledger(_json("evidence_consumption_ledger.json")["entries"])
    decision = _decision()
    assert decision == _json("next_gate_eligibility_decision.json")
    assert decision["next_gate_eligible"] is True
    assert decision["per_candidate_alpha"] == 0.0125


def test_duplicate_attempt_and_opened_unconsumed_evidence_fail_closed() -> None:
    duplicate = _decision(attempt_id="issue-238-retained-v6-7-future")
    assert duplicate["next_gate_eligible"] is False
    assert "duplicate_attempt_id" in duplicate["reason_codes"]
    evidence = _json("evidence_consumption_ledger.json")["entries"]
    evidence[-1]["evidence_permanently_consumed"] = False
    family = _json("candidate_family_manifest.json")
    budget = _json("candidate_budget_protocol.json")
    error = _json("family_error_control_contract.json")
    with pytest.raises(CandidateBudgetError, match="hash mismatch"):
        evaluate_next_gate_eligibility(
            family_manifest=family,
            budget_protocol=budget,
            error_control_contract=error,
            attempt_ledger=_json("candidate_attempt_ledger.json")["entries"],
            evidence_ledger=evidence,
            proposed_attempt=_proposed(),
        )


def test_altered_hash_and_append_only_prefix_are_rejected() -> None:
    entries = _json("candidate_attempt_ledger.json")["entries"]
    altered = copy.deepcopy(entries)
    altered[0]["candidate_id"] = "renamed"
    with pytest.raises(CandidateBudgetError, match="hash mismatch"):
        validate_append_only_ledger(altered)
    with pytest.raises(CandidateBudgetError, match="prefix"):
        validate_append_only_ledger(altered, previous_entries=entries)


def test_post_outcome_bug_fix_consumes_attempt_and_pre_target_invalid_does_not() -> None:
    post = classify_attempt_consumption(
        attempt_case="bug_fix_after_target_access",
        target_outcomes_opened=True,
        decisions_identical_to_prior=False,
    )
    assert post.consumes_attempt is True
    assert post.terminal_for_candidate_hash is True
    invalid = classify_attempt_consumption(
        attempt_case="invalid_pre_target_window",
        target_outcomes_opened=False,
        decisions_identical_to_prior=False,
    )
    assert invalid.consumes_attempt is False
    assert invalid.consumes_alpha is False


def test_identical_engineering_rerun_is_not_an_attempt_but_changed_rerun_is_rejected() -> None:
    reproduction = classify_attempt_consumption(
        attempt_case="engineering_rerun_identical_decisions",
        target_outcomes_opened=False,
        decisions_identical_to_prior=True,
    )
    assert reproduction.consumes_attempt is False
    with pytest.raises(CandidateBudgetError, match="changed decisions"):
        classify_attempt_consumption(
            attempt_case="engineering_rerun_identical_decisions",
            target_outcomes_opened=False,
            decisions_identical_to_prior=False,
        )


def test_exhausted_budget_is_rejected() -> None:
    budget = _json("candidate_budget_protocol.json")
    budget["maximum_confirmatory_attempts"] = 0
    decision = evaluate_next_gate_eligibility(
        family_manifest=_json("candidate_family_manifest.json"),
        budget_protocol=budget,
        error_control_contract=_json("family_error_control_contract.json"),
        attempt_ledger=_json("candidate_attempt_ledger.json")["entries"],
        evidence_ledger=_json("evidence_consumption_ledger.json")["entries"],
        proposed_attempt=_proposed(),
    )
    assert decision["next_gate_eligible"] is False
    assert "candidate_family_attempt_budget_exhausted" in decision["reason_codes"]
