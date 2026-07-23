from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout import (
    COMPLETE_CANARY_BATCH_LATEST_MARKET_CLOSE_TS,
    EXACT_MARKET_COUNT,
    FROZEN_PLAN_SHA256,
    MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
    SCAN_CAP,
    STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE,
    validate_adaptive_support_controller_v8_1_future_holdout_plan,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def test_v8_1_future_holdout_plan_is_frozen_before_collection() -> None:
    plan = _plan()
    validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)
    assert hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == FROZEN_PLAN_SHA256
    assert plan["collection"]["exact_quality_valid_market_count"] == EXACT_MARKET_COUNT
    assert plan["collection"]["maximum_attempted_market_count"] == SCAN_CAP
    assert (
        plan["collection"]["strictly_later_minimum_market_start_ts_exclusive"]
        == STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE
    )
    assert (
        plan["collection"]["complete_target_free_canary_batch_latest_market_close_ts"]
        == COMPLETE_CANARY_BATCH_LATEST_MARKET_CLOSE_TS
    )
    assert (
        plan["target_free_decision_freeze"][
            "minimum_candidate_guard_accepted_unique_market_count"
        ]
        == MINIMUM_GUARD_ACCEPTED_MARKET_COUNT
    )
    assert plan["target_free_decision_freeze"]["side_quota_enabled"] is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("collection", "exact_quality_valid_market_count", 119),
        (
            "target_free_decision_freeze",
            "minimum_candidate_guard_accepted_unique_market_count",
            39,
        ),
        (
            "single_use_future_pnl_gate",
            "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive",
            -0.01,
        ),
        ("safety", "capital_at_risk", True),
    ],
)
def test_v8_1_future_holdout_plan_rejects_gate_or_safety_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    plan = copy.deepcopy(_plan())
    plan[section][field] = value
    with pytest.raises(ValueError, match="future holdout plan drifted"):
        validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)


def test_v8_1_future_holdout_lineage_binds_completed_canary() -> None:
    plan = _plan()
    lineage = plan["lineage"]
    assert (
        lineage["target_free_canary_manifest_sha256"]
        == "225f243a73654c97042032ed77493e83dc44dfdf68f5a14bb5d25b47d2aee6c5"
    )
    assert (
        lineage["target_free_canary_batch_index_sha256"]
        == "d042f1b534956dd8bddfaa3a8269f3a00219d2873f558015a56603f6e8d57ebb"
    )
    assert (
        lineage["target_free_canary_batch_last_entry_sha256"]
        == "09ac313199e170e68900343426dafd4c05a7745053c7079c7fcab4465340fdc3"
    )
    assert plan["single_use_future_pnl_gate"]["equality_passes_noninferiority"]
    assert plan["safety"]["paper_only"] is True
    assert plan["safety"]["capital_at_risk"] is False
    assert plan["safety"]["polymarket_write_enabled"] is False
    assert plan["safety"]["wallet_signing_enabled"] is False
    assert plan["safety"]["#134_resume_allowed"] is False
    assert plan["safety"]["#146_start_allowed"] is False
