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
    parallel = {
        "multiplicity_aware_selected_candidate": "v8_1_primary_no_fallback",
        "candidate_gates": {
            "v8_1_primary_no_fallback": {"all_hard_gates_passed": True}
        },
        "single_use_claim": {
            "single_use": True,
            "target_access_after_decision_freeze": True,
            "result_selected_rerun_allowed": False,
        },
    }
    return {
        "parallel_evaluation_report": _write(tmp_path / "parallel.json", parallel),
        "regime_stratified_pnl_report": _write(
            tmp_path / "regime.json",
            {
                "all_dimension_partitions_reconcile": True,
                "diagnostic_only": True,
                "stratified_metrics_are_eligibility_blockers": False,
            },
        ),
        "replay_parity_report": _write(tmp_path / "parity.json", {"passed": True}),
        "policy_safety_report": _write(tmp_path / "safety.json", {"passed": True}),
        "policy_reconciliation_report": _write(
            tmp_path / "reconciliation.json", {"passed": True}
        ),
        "powered_paper_gate_report": _write(
            tmp_path / "paper.json",
            {
                "powered_paper_gate_passed": True,
                "paper_only": True,
                "capital_at_risk": False,
            },
        ),
        "attempt_consumption_record": _write(
            tmp_path / "consumption.json",
            {
                "consumes_attempt": True,
                "consumes_alpha": True,
                "evidence_permanently_consumed": True,
            },
        ),
    }


def test_static_issue_prerequisites_pass_but_promotion_waits_for_fresh_evidence() -> None:
    report = audit_challenge_model_promotion(repository_root=ROOT)
    assert all(report["static_checks"].values())
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
