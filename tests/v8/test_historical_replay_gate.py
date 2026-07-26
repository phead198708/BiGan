from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.historical_replay_gate import (
    EXACT_MODEL_BINDING_CHECK_NAMES,
    HistoricalReplayGateError,
    audit_historical_replay_superiority,
    validate_exact_historical_model_binding,
    validate_historical_replay_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/v8/polymarket_configs"


def _contract() -> dict:
    return json.loads(
        (CONFIG / "historical_replay_superiority_contract.json").read_text()
    )


def _candidate_contract() -> dict:
    return json.loads(
        (
            CONFIG
            / "parallel_candidate_v8_1_primary_no_fallback_contract.json"
        ).read_text()
    )


def _rows() -> tuple[list[dict], list[dict]]:
    baseline = []
    candidate = []
    for index in range(120):
        common = {
            "market_id": f"market-{index:03d}",
            "decision_ts": 1_000_000 + index,
            "action": "BUY_UP_SELL_BEFORE_CLOSE",
            "side": "UP",
            "fixed_position_size": 0.2,
            "target_used_as_decision_time_input": False,
        }
        baseline.append(
            {
                **common,
                "after_cost_net_pnl_at_frozen_size": -0.01,
            }
        )
        if index < 45:
            candidate.append(
                {
                    **common,
                    "after_cost_net_pnl_at_frozen_size": 0.01,
                }
            )
    return candidate, baseline


def _inputs() -> dict:
    candidate, baseline = _rows()
    market_hash = canonical_json_sha256(
        sorted(row["market_id"] for row in baseline)
    )
    candidate_total = sum(
        row["after_cost_net_pnl_at_frozen_size"] for row in candidate
    )
    baseline_total = sum(
        row["after_cost_net_pnl_at_frozen_size"] for row in baseline
    )
    source_report = {
        "candidate_name": "adaptive_support_controller_v8_1",
        "historical_hard_gate_passed": True,
        "fit_leakage_audit_passed": True,
        "historical_gate_blocking_reason_codes": [],
        "historical_policy_difference_market_count": 75,
        "historical_prequential_hard_gate": {
            "evaluation_market_count": 120,
            "evaluation_market_ids_hash": market_hash,
            "fixed_position_size": 0.2,
            "same_runtime_aligned_target_and_cost_contract": True,
            "same_position_management_and_guard_contract": True,
            "full_execution_guard_applied_to_candidate_and_baseline": True,
            "common_selected_row_filter_applied": False,
            "no_bet_market_pnl": 0.0,
            "historical_oof_or_validation_pnl_used_for_feature_hyperparameter_or_threshold_tuning": False,
            "historical_pnl_used_for_pre_collection_screening_only": True,
            "candidate": {
                "total_after_cost_net_pnl_at_frozen_size": candidate_total,
                "largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
                    candidate_total - 0.01
                ),
            },
            "v6_7_baseline": {
                "total_after_cost_net_pnl_at_frozen_size": baseline_total,
                "largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
                    baseline_total
                ),
            },
            "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": (
                candidate_total - baseline_total
            ),
            "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
                candidate_total - 0.01 - baseline_total
            ),
        },
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    leakage = {
        "fit_leakage_audit_passed": True,
        "fit_leakage_blocking_reason_codes": [],
        "fit_leakage_checks": {"prequential": True, "causal": True},
        "issue243_pnl_targets_winners_or_losers_used_for_threshold_selection": False,
        "issue244_pnl_targets_winners_or_losers_used_for_controller_design": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    return {
        "gate_contract": _contract(),
        "candidate_contract": _candidate_contract(),
        "source_report": source_report,
        "leakage_audit": leakage,
        "candidate_rows": candidate,
        "baseline_rows": baseline,
        "lineage_sha256s": {"source_report": "a" * 64},
        "evaluation_completed_ts": 2_000_000,
        "exact_model_binding_summary": _binding_summary(),
    }


def _binding_inputs() -> dict:
    booster_base64 = base64.b64encode(b"exact-booster-bytes").decode("ascii")
    booster_sha256 = canonical_json_sha256(booster_base64)
    candidate_name = "candidate-from-contract"
    safety_fields = (
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "paper_candidate_allowed",
        "live_trading_enabled",
        "v8_execution_handoff_allowed",
        "capital_at_risk",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "#134_resume_allowed",
        "#146_start_allowed",
    )
    candidate_contract = {
        "primary_policy": candidate_name,
        "frozen_model_binding_sha256": "a" * 64,
        "frozen_model_artifact_sha256": "b" * 64,
        "frozen_model_artifact_id": "c" * 64,
        "profile_sha256": "d" * 64,
        "initial_controller_state_id": "e" * 64,
    }
    binding = {
        "candidate_name": candidate_name,
        "historical_fit_manifest_sha256": "f" * 64,
        "frozen_model_artifact_sha256": "b" * 64,
        "frozen_model_artifact_id": "c" * 64,
        "frozen_booster_sha256": booster_sha256,
        "frozen_profile_sha256": "d" * 64,
        "initial_controller_state": {
            "rank_state_id": "e" * 64,
            "rank_lineage_hash": "2" * 64,
            "eligible_prediction_scores_hash": "3" * 64,
            "controller_guard_acceptance_history_hash": "4" * 64,
            "controller_state_uses_target_outcome_label_or_pnl": False,
            "future_controller_updates_use_strictly_prior_guard_results_only": True,
        },
    }
    model = {
        "candidate_name": candidate_name,
        "model_artifact_id": "c" * 64,
        "historical_hard_gate_passed": True,
        "historical_gate_blocking_reason_codes": [],
        "final_weighted_model": {
            "booster_json_base64": booster_base64,
            "booster_sha256": booster_sha256,
        },
        "final_rank_state": {
            "rank_state_id": "e" * 64,
            "rank_lineage_hash": "2" * 64,
            "eligible_prediction_scores_hash": "3" * 64,
            "controller_guard_acceptance_history_hash": "4" * 64,
            "controller_target_outcome_label_or_pnl_free": True,
        },
        **dict.fromkeys(safety_fields, False),
    }
    return {
        "candidate_contract": candidate_contract,
        "frozen_model_binding": binding,
        "frozen_model_artifact": model,
        "source_manifest": {
            "model": {"sha256": "b" * 64},
            "profile": {"sha256": "d" * 64},
        },
        "expected_binding_sha256": "a" * 64,
        "expected_model_artifact_sha256": "b" * 64,
        "expected_source_manifest_sha256": "f" * 64,
        "expected_candidate_profile_sha256": "d" * 64,
    }


def _binding_summary() -> dict:
    return validate_exact_historical_model_binding(**_binding_inputs())


def test_contract_is_strict_and_bootstrap_is_diagnostic_only() -> None:
    contract = _contract()
    validate_historical_replay_contract(contract)
    assert (
        contract["gate"][
            "candidate_minus_champion_total_pnl_minimum_exclusive"
        ]
        == 0.0
    )
    assert contract["paired_market_bootstrap"]["eligibility_blocker"] is False


def test_strict_historical_superiority_passes_without_unlocking_promotion() -> None:
    report = audit_historical_replay_superiority(**_inputs())
    assert report["historical_superiority_gate_passed"] is True
    assert report["future_collection_prerequisite_satisfied"] is True
    assert report["metrics"]["candidate_minus_champion_total_after_cost_pnl"] > 0
    assert (
        report["metrics"][
            "candidate_minus_champion_largest_winner_removed_after_cost_pnl"
        ]
        > 0
    )
    assert (
        report["metrics"][
            "paired_delta_largest_winner_removed_after_cost_pnl"
        ]
        > 0
    )
    assert report["historical_replay_is_promotion_evidence"] is False
    assert report["promotion_unlocked"] is False
    assert report["capital_at_risk"] is False


def test_equality_with_champion_fails_the_strict_gate() -> None:
    inputs = _inputs()
    inputs["candidate_rows"] = copy.deepcopy(inputs["baseline_rows"])
    source = inputs["source_report"]["historical_prequential_hard_gate"]
    source["candidate"] = copy.deepcopy(source["v6_7_baseline"])
    source["candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"] = 0.0
    source[
        "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
    ] = 0.0
    report = audit_historical_replay_superiority(**inputs)
    assert report["historical_superiority_gate_passed"] is False
    assert (
        "candidate_total_pnl_strictly_better_than_champion"
        in report["blocking_reason_codes"]
    )
    assert report["future_collection_prerequisite_satisfied"] is False


def test_target_contamination_fails_closed() -> None:
    inputs = _inputs()
    inputs["candidate_rows"][0]["target_used_as_decision_time_input"] = True
    with pytest.raises(HistoricalReplayGateError, match="contaminated"):
        audit_historical_replay_superiority(**inputs)


def test_exact_model_binding_returns_validator_derived_summary() -> None:
    summary = _binding_summary()
    assert summary["check_count"] == 13
    assert tuple(summary["checks"]) == EXACT_MODEL_BINDING_CHECK_NAMES
    assert all(summary["checks"].values())
    assert summary["exact_frozen_model_binding_verified"] is True
    assert len(summary["verified_hashes"]["booster_bytes_sha256"]) == 64


@pytest.mark.parametrize(
    "failed_check",
    EXACT_MODEL_BINDING_CHECK_NAMES,
)
def test_each_exact_model_binding_check_fails_independently(
    failed_check: str,
) -> None:
    inputs = _binding_inputs()
    contract = inputs["candidate_contract"]
    binding = inputs["frozen_model_binding"]
    model = inputs["frozen_model_artifact"]
    state = binding["initial_controller_state"]
    mutations = {
        "binding_sha256": lambda: contract.__setitem__(
            "frozen_model_binding_sha256", "0" * 64
        ),
        "model_artifact_identity": lambda: model.__setitem__(
            "model_artifact_id", "0" * 64
        ),
        "historical_manifest_sha256": lambda: binding.__setitem__(
            "historical_fit_manifest_sha256", "0" * 64
        ),
        "candidate_profile_sha256": lambda: binding.__setitem__(
            "frozen_profile_sha256", "0" * 64
        ),
        "candidate_identity": lambda: binding.__setitem__(
            "candidate_name", "other-candidate"
        ),
        "booster_sha256": lambda: model[
            "final_weighted_model"
        ].__setitem__(
            "booster_json_base64",
            base64.b64encode(b"drifted-booster").decode("ascii"),
        ),
        "rank_state_id": lambda: state.__setitem__(
            "rank_state_id", "0" * 64
        ),
        "rank_lineage_hash": lambda: state.__setitem__(
            "rank_lineage_hash", "0" * 64
        ),
        "eligible_prediction_scores_hash": lambda: state.__setitem__(
            "eligible_prediction_scores_hash", "0" * 64
        ),
        "controller_guard_acceptance_history_hash": lambda: state.__setitem__(
            "controller_guard_acceptance_history_hash", "0" * 64
        ),
        "historical_gate_passed": lambda: model.__setitem__(
            "historical_hard_gate_passed", False
        ),
        "target_free_controller": lambda: state.__setitem__(
            "controller_state_uses_target_outcome_label_or_pnl", True
        ),
        "all_safety_unlocks_remain_false": lambda: model.__setitem__(
            "capital_at_risk", True
        ),
    }
    mutations[failed_check]()
    with pytest.raises(
        HistoricalReplayGateError,
        match=failed_check,
    ):
        validate_exact_historical_model_binding(**inputs)


@pytest.mark.parametrize(
    "field",
    [
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "paper_candidate_allowed",
        "live_trading_enabled",
        "v8_execution_handoff_allowed",
        "capital_at_risk",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "#134_resume_allowed",
        "#146_start_allowed",
    ],
)
@pytest.mark.parametrize("mode", ["true", "missing"])
def test_each_safety_field_true_or_missing_fails_closed(
    field: str,
    mode: str,
) -> None:
    inputs = _binding_inputs()
    model = inputs["frozen_model_artifact"]
    if mode == "true":
        model[field] = True
    else:
        model.pop(field)
    with pytest.raises(
        HistoricalReplayGateError,
        match="all_safety_unlocks_remain_false",
    ):
        validate_exact_historical_model_binding(**inputs)


@pytest.mark.parametrize(
    "field",
    [
        "expected_binding_sha256",
        "expected_model_artifact_sha256",
        "expected_source_manifest_sha256",
        "expected_candidate_profile_sha256",
    ],
)
def test_malformed_expected_sha256_fails_closed(field: str) -> None:
    inputs = _binding_inputs()
    inputs[field] = "not-a-sha256"
    with pytest.raises(HistoricalReplayGateError):
        validate_exact_historical_model_binding(**inputs)


def test_audit_cannot_assert_binding_without_passing_summary() -> None:
    inputs = _inputs()
    summary = copy.deepcopy(inputs["exact_model_binding_summary"])
    summary["checks"]["booster_sha256"] = False
    summary["exact_frozen_model_binding_verified"] = False
    summary_without_id = {
        key: value for key, value in summary.items() if key != "summary_id"
    }
    summary["summary_id"] = canonical_json_sha256(summary_without_id)
    inputs["exact_model_binding_summary"] = summary
    report = audit_historical_replay_superiority(**inputs)
    assert report["checks"]["exact_frozen_model_binding_verified"] is False
    assert report["historical_superiority_gate_passed"] is False
    assert (
        "exact_frozen_model_binding_verified"
        in report["blocking_reason_codes"]
    )
