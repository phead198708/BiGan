from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout import (
    EXACT_MARKET_COUNT,
    FROZEN_PLAN_SHA256,
    MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
    SCAN_CAP,
    STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE,
    _safety_fields,
    validate_v7_7_future_holdout_plan,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def test_plan_freezes_bounded_strictly_later_outcome_blind_collection() -> None:
    plan = _plan()

    validate_v7_7_future_holdout_plan(plan)
    assert _sha256_file(PLAN_PATH) == FROZEN_PLAN_SHA256

    collection = plan["collection"]
    assert collection["exact_quality_valid_market_count"] == EXACT_MARKET_COUNT == 120
    assert collection["maximum_attempted_market_count"] == SCAN_CAP == 180
    assert (
        collection["strictly_later_minimum_market_start_ts_exclusive"]
        == STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE
        == 1_784_760_900_000
    )
    assert collection["outcomes_resolution_labels_or_pnl_opened"] is False
    assert collection["candidate_model_scoring_during_collection_allowed"] is False


def test_plan_uses_inclusive_noninferiority_without_side_quota() -> None:
    plan = _plan()
    freeze = plan["target_free_decision_freeze"]
    gate = plan["single_use_future_pnl_gate"]

    assert freeze["minimum_v7_7_guard_accepted_unique_market_count"] == (
        MINIMUM_GUARD_ACCEPTED_MARKET_COUNT
    )
    assert freeze["side_quota_enabled"] is False
    assert gate["comparison_operator"] == "greater_than_or_equal"
    assert gate["equality_passes_noninferiority"] is True
    assert gate["candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"] == 0.0
    assert (
        gate[
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive"
        ]
        == 0.0
    )
    assert gate["candidate_total_after_cost_pnl_minimum_exclusive"] == 0.0


def test_plan_rejects_outcome_access_or_gate_drift() -> None:
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["collection"]["outcomes_resolution_labels_or_pnl_opened"] = True
    with pytest.raises(ValueError, match="collection"):
        validate_v7_7_future_holdout_plan(changed)

    changed = copy.deepcopy(plan)
    changed["single_use_future_pnl_gate"][
        "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"
    ] = -0.01
    with pytest.raises(ValueError, match="single_use_gate"):
        validate_v7_7_future_holdout_plan(changed)


def test_plan_safety_remains_fail_closed() -> None:
    assert _plan()["safety"] == _safety_fields()
    assert _safety_fields() == {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_candidate_allowed": False,
        "live_trading_enabled": False,
    }
