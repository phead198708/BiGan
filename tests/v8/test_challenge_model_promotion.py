from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_model_promotion import (
    ChallengeModelPromotionError,
    audit_challenge_model_promotion,
    promotion_readiness_markdown,
)

ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, payload: dict) -> dict[str, str]:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _runtime(tmp_path: Path):
    fresh_attempt_id = "challenge-future-attempt-001"
    freeze_sha256 = "f" * 64
    selected_candidate = "v8_1_primary_no_fallback"
    selected_gate = {"all_hard_gates_passed": True}
    parallel = {
        "schema_version": "bigan-v8-parallel-future-evaluation-v1",
        "multiplicity_aware_selected_candidate": selected_candidate,
        "candidate_gates": {selected_candidate: selected_gate},
        "single_use_claim": {
            "single_use": True,
            "target_access_after_decision_freeze": True,
            "result_selected_rerun_allowed": False,
            "freeze_sha256": freeze_sha256,
        },
    }
    parallel_descriptor = _write(tmp_path / "parallel.json", parallel)
    common = {
        "fresh_attempt_id": fresh_attempt_id,
        "selected_candidate_id": selected_candidate,
        "parallel_freeze_sha256": freeze_sha256,
        "source_parallel_evaluation_report_sha256": parallel_descriptor["sha256"],
    }
    policy_common = {
        **common,
        "policy_candidate_count": 3,
        "all_preregistered_policy_candidates_evaluated": True,
        "outcome_selected_policy_used": False,
        "passed": True,
    }
    return {
        "parallel_evaluation_report": parallel_descriptor,
        "regime_stratified_pnl_report": _write(
            tmp_path / "regime.json",
            {
                "schema_version": "bigan-v8-regime-stratified-pnl-report-v1",
                **common,
                "all_dimension_partitions_reconcile": True,
                "diagnostic_only": True,
                "stratified_metrics_are_eligibility_blockers": False,
            },
        ),
        "replay_parity_report": _write(
            tmp_path / "parity.json",
            {
                "schema_version": ("bigan-v8-challenge-execution-policy-replay-parity-v1"),
                **policy_common,
            },
        ),
        "policy_safety_report": _write(
            tmp_path / "safety.json",
            {
                "schema_version": ("bigan-v8-challenge-execution-policy-safety-v1"),
                **policy_common,
            },
        ),
        "policy_reconciliation_report": _write(
            tmp_path / "reconciliation.json",
            {
                "schema_version": ("bigan-v8-challenge-execution-policy-reconciliation-v1"),
                **policy_common,
            },
        ),
        "powered_paper_gate_report": _write(
            tmp_path / "paper.json",
            {
                "schema_version": "bigan-v8-challenge-powered-paper-gate-v1",
                **common,
                "selected_candidate_gate": selected_gate,
                "checks": {
                    "selected_candidate_matches_parallel_winner": True,
                    "selected_candidate_all_hard_gates_passed": True,
                },
                "powered_paper_gate_passed": True,
                "separate_result_selected_retest_performed": False,
                "paper_only": True,
                "capital_at_risk": False,
            },
        ),
        "attempt_consumption_record": _write(
            tmp_path / "consumption.json",
            {
                "schema_version": "bigan-v8-challenge-attempt-consumption-v1",
                "fresh_attempt_id": fresh_attempt_id,
                "parallel_freeze_sha256": freeze_sha256,
                "fresh_attempt_number": 1,
                "familywise_window_alpha": 0.025,
                "per_candidate_alpha": 0.0125,
                "attempt_consumed": True,
                "alpha_consumed": True,
                "consumes_attempt": True,
                "consumes_alpha": True,
                "evidence_permanently_consumed": True,
            },
        ),
    }


def test_static_issue_prerequisites_pass_but_promotion_waits_for_fresh_evidence() -> None:
    report = audit_challenge_model_promotion(repository_root=ROOT)
    assert all(report["static_checks"].values())
    assert report["static_checks"]["historical_replay_strictly_superior_before_collection"] is True
    assert report["fresh_runtime_evidence_supplied"] is False
    assert report["decision"] == "BLOCKED"
    assert report["promotion_unlocked"] is False
    assert report["selected_champion_candidate"] is None
    assert "evidence:fresh_parallel_evaluation_present" in report["blockers"]
    assert "Historical or consumed results are not substituted" in (
        promotion_readiness_markdown(report)
    )


def test_complete_fresh_hash_bound_evidence_can_promote_candidate(tmp_path: Path) -> None:
    report = audit_challenge_model_promotion(
        repository_root=ROOT,
        runtime_evidence=_runtime(tmp_path),
    )
    assert report["decision"] == "PROMOTE_TO_CHAMPION"
    assert report["challenge_model_promotion_eligible"] is True
    assert report["selected_champion_candidate"] == "v8_1_primary_no_fallback"
    assert report["promotion_unlocked"] is True
    assert report["live_unlocked"] is False
    assert report["capital_at_risk"] is False


def test_failed_paper_evidence_and_hash_tamper_fail_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    paper_path = Path(runtime["powered_paper_gate_report"]["path"])
    paper_path.write_text(
        json.dumps(
            {
                "schema_version": "bigan-v8-challenge-powered-paper-gate-v1",
                "powered_paper_gate_passed": False,
                "paper_only": True,
                "capital_at_risk": False,
            }
        )
    )
    runtime["powered_paper_gate_report"]["sha256"] = hashlib.sha256(
        paper_path.read_bytes()
    ).hexdigest()
    report = audit_challenge_model_promotion(
        repository_root=ROOT,
        runtime_evidence=runtime,
    )
    assert report["challenge_model_promotion_eligible"] is False
    assert "evidence:powered_paper_gate_passed" in report["blockers"]
    runtime["parallel_evaluation_report"]["sha256"] = "0" * 64
    with pytest.raises(ChallengeModelPromotionError, match="hash mismatch"):
        audit_challenge_model_promotion(
            repository_root=ROOT,
            runtime_evidence=runtime,
        )
