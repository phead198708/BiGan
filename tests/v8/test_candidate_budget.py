from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.candidate_budget import (
    SAFETY,
    CandidateBudgetError,
    classify_attempt_consumption,
    evaluate_next_gate_eligibility,
    ledger_entry_sha256,
    stable_candidate_identity,
    validate_append_only_ledger,
    validate_candidate_budget_artifacts,
    validate_eligibility_decision,
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
        "attempt_case": "parallel_shared_window_candidate",
        "candidate_stable_identities": {
            "v8_1_primary_no_fallback": (
                "2f343da4bb8d61265fc597f53094e3e327987dd74f7cff3eecfb89ac067af12c"
            ),
            "v8_3_primary_with_fallback": (
                "6c03c9d6815522ed53655d152eac674df09d6fec3485543641a86e1ba686e766"
            ),
        },
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
    attempts = _json("candidate_attempt_ledger.json")
    evidence = _json("evidence_consumption_ledger.json")
    validate_append_only_ledger(attempts["entries"])
    validate_append_only_ledger(evidence["entries"])
    validate_candidate_budget_artifacts(
        family_manifest=_json("candidate_family_manifest.json"),
        budget_protocol=_json("candidate_budget_protocol.json"),
        error_control_contract=_json("family_error_control_contract.json"),
        attempt_ledger=attempts,
        evidence_ledger=evidence,
    )
    decision = _decision()
    assert decision == _json("next_gate_eligibility_decision.json")
    validate_eligibility_decision(
        _json("next_gate_eligibility_decision.json"),
        recomputed=decision,
    )
    assert decision["next_gate_eligible"] is True
    assert decision["per_candidate_alpha"] == 0.0125
    assert all(decision[field] is expected for field, expected in SAFETY.items())


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
    attempts = _json("candidate_attempt_ledger.json")["entries"]
    for index in range(1, 4):
        attempts.append(
            {
                "schema_version": "bigan-v8-candidate-ledger-entry-v1",
                "sequence": 0,
                "entry_id": f"current-family-attempt-{index}",
                "previous_entry_sha256": "",
                "entry_sha256": "",
                "family_id": "v8-high-signal-vs-fallback-2026q3",
                "attempt_id": f"consumed-current-attempt-{index}",
                "candidate_id": "v8_1_primary_no_fallback",
                "stable_candidate_identity": (
                    "2f343da4bb8d61265fc597f53094e3e327987dd74f7cff3eecfb89ac067af12c"
                ),
                "attempt_case": "target_opened_window",
                "target_outcomes_opened": True,
                "consumes_attempt": True,
                "consumes_alpha": True,
                "terminal": True,
            }
        )
    _rehash(attempts)
    decision = evaluate_next_gate_eligibility(
        family_manifest=_json("candidate_family_manifest.json"),
        budget_protocol=_json("candidate_budget_protocol.json"),
        error_control_contract=_json("family_error_control_contract.json"),
        attempt_ledger=attempts,
        evidence_ledger=_json("evidence_consumption_ledger.json")["entries"],
        proposed_attempt=_proposed(),
    )
    assert decision["next_gate_eligible"] is False
    assert "candidate_family_attempt_budget_exhausted" in decision["reason_codes"]


@pytest.mark.parametrize("field", list(SAFETY))
def test_every_budget_safety_field_is_required_false(field: str) -> None:
    manifest = _json("candidate_family_manifest.json")
    manifest["safety"][field] = True
    with pytest.raises(CandidateBudgetError, match="family manifest"):
        evaluate_next_gate_eligibility(
            family_manifest=manifest,
            budget_protocol=_json("candidate_budget_protocol.json"),
            error_control_contract=_json("family_error_control_contract.json"),
            attempt_ledger=_json("candidate_attempt_ledger.json")["entries"],
            evidence_ledger=_json("evidence_consumption_ledger.json")["entries"],
            proposed_attempt=_proposed(),
        )


def test_rehashed_semantic_ledger_tampering_still_fails_closed() -> None:
    attempts = _json("candidate_attempt_ledger.json")["entries"]
    attempts[-1]["consumes_alpha"] = False
    _rehash(attempts)
    with pytest.raises(CandidateBudgetError, match="permanently consumed"):
        evaluate_next_gate_eligibility(
            family_manifest=_json("candidate_family_manifest.json"),
            budget_protocol=_json("candidate_budget_protocol.json"),
            error_control_contract=_json("family_error_control_contract.json"),
            attempt_ledger=attempts,
            evidence_ledger=_json("evidence_consumption_ledger.json")["entries"],
            proposed_attempt=_proposed(),
        )

    evidence = _json("evidence_consumption_ledger.json")["entries"]
    evidence[-1]["reusable_for_promotion"] = True
    _rehash(evidence)
    with pytest.raises(CandidateBudgetError, match="permanently consumed"):
        evaluate_next_gate_eligibility(
            family_manifest=_json("candidate_family_manifest.json"),
            budget_protocol=_json("candidate_budget_protocol.json"),
            error_control_contract=_json("family_error_control_contract.json"),
            attempt_ledger=_json("candidate_attempt_ledger.json")["entries"],
            evidence_ledger=evidence,
            proposed_attempt=_proposed(),
        )


def test_alpha_spending_and_identity_pins_cannot_drift() -> None:
    error_control = _json("family_error_control_contract.json")
    error_control["sequential_alpha_spending"][0] = 0.03
    with pytest.raises(CandidateBudgetError, match="error control"):
        evaluate_next_gate_eligibility(
            family_manifest=_json("candidate_family_manifest.json"),
            budget_protocol=_json("candidate_budget_protocol.json"),
            error_control_contract=error_control,
            attempt_ledger=_json("candidate_attempt_ledger.json")["entries"],
            evidence_ledger=_json("evidence_consumption_ledger.json")["entries"],
            proposed_attempt=_proposed(),
        )

    decision = _decision(candidate_stable_identities={})
    assert decision["next_gate_eligible"] is False
    assert "candidate_identity_pins_mismatch" in decision["reason_codes"]


def test_duplicate_parallel_candidate_and_wrong_case_are_rejected() -> None:
    duplicate = _decision(
        candidate_ids=[
            "v8_1_primary_no_fallback",
            "v8_1_primary_no_fallback",
        ],
        candidate_stable_identities={
            "v8_1_primary_no_fallback": (
                "2f343da4bb8d61265fc597f53094e3e327987dd74f7cff3eecfb89ac067af12c"
            )
        },
    )
    assert duplicate["next_gate_eligible"] is False
    assert "duplicate_candidate_id" in duplicate["reason_codes"]

    wrong_case = _decision(attempt_case="sequential_new_window_candidate")
    assert wrong_case["next_gate_eligible"] is False
    assert "proposed_attempt_case_invalid" in wrong_case["reason_codes"]


def test_parallel_candidate_rows_consume_one_sequential_attempt() -> None:
    attempts = _json("candidate_attempt_ledger.json")["entries"]
    for index, (candidate_id, identity) in enumerate(
        (
            (
                "v8_1_primary_no_fallback",
                "2f343da4bb8d61265fc597f53094e3e327987dd74f7cff3eecfb89ac067af12c",
            ),
            (
                "v8_3_primary_with_fallback",
                "6c03c9d6815522ed53655d152eac674df09d6fec3485543641a86e1ba686e766",
            ),
        ),
        start=1,
    ):
        attempts.append(
            {
                "schema_version": "bigan-v8-candidate-ledger-entry-v1",
                "sequence": 0,
                "entry_id": f"parallel-attempt-row-{index}",
                "previous_entry_sha256": "",
                "entry_sha256": "",
                "family_id": "v8-high-signal-vs-fallback-2026q3",
                "attempt_id": "consumed-parallel-attempt-001",
                "candidate_id": candidate_id,
                "stable_candidate_identity": identity,
                "attempt_case": "parallel_shared_window_candidate",
                "target_outcomes_opened": True,
                "consumes_attempt": True,
                "consumes_alpha": True,
                "terminal": True,
            }
        )
    _rehash(attempts)

    decision = evaluate_next_gate_eligibility(
        family_manifest=_json("candidate_family_manifest.json"),
        budget_protocol=_json("candidate_budget_protocol.json"),
        error_control_contract=_json("family_error_control_contract.json"),
        attempt_ledger=attempts,
        evidence_ledger=_json("evidence_consumption_ledger.json")["entries"],
        proposed_attempt={
            **_proposed(),
            "attempt_id": "v8-parallel-future-gate-attempt-002",
        },
    )
    assert decision["next_attempt_number"] == 2
    assert decision["window_familywise_alpha"] == 0.015
    assert decision["per_candidate_alpha"] == 0.0075
    assert decision["next_gate_eligible"] is True


def _rehash(entries: list[dict]) -> None:
    previous = "0" * 64
    for sequence, entry in enumerate(entries, start=1):
        entry["sequence"] = sequence
        entry["previous_entry_sha256"] = previous
        entry["entry_sha256"] = ledger_entry_sha256(entry)
        previous = entry["entry_sha256"]
