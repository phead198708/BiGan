from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.historical_replay_gate import (
    HistoricalReplayGateError,
    audit_historical_replay_superiority,
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
    }


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
