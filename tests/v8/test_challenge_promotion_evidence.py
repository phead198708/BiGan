from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_promotion_evidence import (
    AGGREGATE_PARITY_SCHEMA_VERSION,
    AGGREGATE_RECONCILIATION_SCHEMA_VERSION,
    AGGREGATE_SAFETY_SCHEMA_VERSION,
    ChallengePromotionEvidenceError,
    _aggregate_policy_report,
    _causal_context,
    _execution_policy_inputs,
    _powered_paper_gate,
    _run_all_execution_policies,
    validate_challenge_promotion_evidence_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/v8/polymarket_configs"


def _json(name: str) -> dict:
    return json.loads((CONFIG / name).read_text())


def _sha(name: str) -> str:
    return hashlib.sha256((CONFIG / name).read_bytes()).hexdigest()


def _validate_protocol(protocol: dict) -> None:
    validate_challenge_promotion_evidence_protocol(
        protocol,
        regime_contract_sha256=_sha("regime_definition_contract.json"),
        execution_policy_contract_sha256=_sha("execution_policy_contract.json"),
        policy_candidate_manifest_sha256=_sha("policy_candidate_manifest.json"),
        compatibility_manifest_sha256=_sha("source_execution_compatibility_manifest.json"),
        post_freeze_protocol_sha256=_sha("challenge_future_post_freeze_protocol.json"),
        feature_missingness_contract_sha256=_sha("feature_missingness_contract.json"),
        canonical_payload_contract_sha256=_sha("canonical_payload_contract.json"),
    )


def test_promotion_evidence_protocol_is_hash_pinned_and_exact() -> None:
    path = CONFIG / "challenge_promotion_evidence_protocol.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        path.with_suffix(".sha256").read_text().strip()
    )
    protocol = _json(path.name)
    _validate_protocol(protocol)
    tampered = copy.deepcopy(protocol)
    tampered["powered_paper_gate"]["separate_result_selected_retest_allowed"] = True
    with pytest.raises(
        ChallengePromotionEvidenceError,
        match="powered_paper_gate",
    ):
        _validate_protocol(tampered)


def test_missing_provider_context_is_explicit_and_policy_fails_closed() -> None:
    context = _causal_context(decision_ts=1_000, feature_row=None)
    assert context["available_at_ts"] == 1_000
    assert context["reference_return"] is None
    assert context["provider_coverage_complete"] == 0

    source_hash = _json("source_execution_compatibility_manifest.json")["source_model_hash"]
    policy_inputs = _execution_policy_inputs(
        source_rows=[
            {
                "market_id": "market-1",
                "decision_ts": 1_000,
                "policy_grid_decision_ts": 900,
                "collector_batch_id": "batch-1",
            }
        ],
        feature_rows=[],
        native_decisions=[
            {
                "market_id": "market-1",
                "baseline_action": "BUY_UP_HOLD_TO_SETTLEMENT",
                "predicted_baseline_return": 0.12,
                "opposite_action": "BUY_DOWN_HOLD_TO_SETTLEMENT",
                "predicted_opposite_return": 0.04,
            }
        ],
        source_model_hash=source_hash,
    )
    assert policy_inputs[0]["source_action_scores"] == {
        "NO_TRADE": 0.0,
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.12,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.04,
    }
    assert policy_inputs[0]["provider_health_score"] == 0.0
    assert policy_inputs[0]["provider_features_complete"] is False


def test_all_preregistered_policies_reconcile_and_powered_gate_is_inherited() -> None:
    compatibility = _json("source_execution_compatibility_manifest.json")
    policy_inputs = [
        {
            "market_id": f"market-{index}",
            "decision_ts": 1_000_000 + index * 1_000,
            "source_model_hash": compatibility["source_model_hash"],
            "source_action_scores": {
                "BUY_UP_HOLD_TO_SETTLEMENT": 0.12,
                "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.04,
                "NO_TRADE": 0.0,
            },
            "uncertainty": 0.08,
            "opportunity_window_id": f"window-{index}",
            "fill_quality_score": 0.95,
            "provider_health_score": 1.0,
            "provider_features_complete": True,
            "kill_switch_active": False,
        }
        for index in range(3)
    ]
    results = _run_all_execution_policies(
        policy_inputs=policy_inputs,
        policy_manifest=_json("policy_candidate_manifest.json"),
        policy_manifest_path=CONFIG / "policy_candidate_manifest.json",
        compatibility=compatibility,
    )
    assert len(results) == 3
    common = {
        "fresh_attempt_id": "challenge-future-attempt-001",
        "selected_candidate_id": "v8_1_primary_no_fallback",
        "parallel_freeze_sha256": "f" * 64,
        "source_parallel_evaluation_report_sha256": "e" * 64,
    }
    for schema_version, report_key in (
        (AGGREGATE_PARITY_SCHEMA_VERSION, "parity"),
        (AGGREGATE_SAFETY_SCHEMA_VERSION, "safety"),
        (AGGREGATE_RECONCILIATION_SCHEMA_VERSION, "reconciliation"),
    ):
        report = _aggregate_policy_report(
            schema_version=schema_version,
            common=common,
            policy_results=results,
            report_key=report_key,
        )
        assert report["policy_candidate_count"] == 3
        assert report["outcome_selected_policy_used"] is False
        assert report["passed"] is True

    selected_gate = {
        "all_hard_gates_passed": True,
        "accepted_bet_count": 40,
        "minimum_total_support": 30,
        "total_after_cost_pnl": 0.4,
        "candidate_minus_baseline_after_cost_pnl": 0.3,
        "candidate_minus_baseline_bootstrap_lcb": 0.1,
        "candidate_largest_winner_removed_after_cost_pnl": 0.2,
        "candidate_minus_baseline_largest_winner_removed_after_cost_pnl": (0.15),
    }
    paper = _powered_paper_gate(
        parallel_report={
            "multiplicity_aware_selected_candidate": ("v8_1_primary_no_fallback"),
            "candidate_gates": {"v8_1_primary_no_fallback": selected_gate},
        },
        common=common,
    )
    assert paper["selected_candidate_gate"] == selected_gate
    assert paper["powered_paper_gate_passed"] is True
    assert paper["separate_result_selected_retest_performed"] is False
