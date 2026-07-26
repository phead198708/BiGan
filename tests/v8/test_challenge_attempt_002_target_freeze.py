from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_attempt_002 import CANDIDATE_ID
from bigan.v8.polymarket.challenge_attempt_002_target_freeze import (
    ChallengeAttempt002TargetFreezeError,
    build_attempt_002_decision_freeze,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.contracts import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
PROTOCOL = json.loads(
    (CONFIG_DIR / "challenge_attempt_002_preregistration.json").read_text()
)
ENTRY_PROFILE = json.loads(
    (
        CONFIG_DIR / "challenge_v8_1_entry_price_floor_0_30_profile.json"
    ).read_text()
)
SIZING_PROFILE = json.loads(
    (
        CONFIG_DIR
        / "challenge_v8_1_entry_price_floor_0_30_sized_1_0_profile.json"
    ).read_text()
)


def _fixture() -> dict:
    boundary = int(PROTOCOL["preregistration_freeze_created_ts"])
    selected = []
    actions = []
    features = []
    native = []
    candidate_guard = []
    baseline_guard = []
    for offset in range(120):
        index = offset + 1
        market_id = f"future-market-{index:03d}"
        start = boundary + index * 300_000
        end = start + 300_000
        decision_ts = start + 10_000
        selected.append(
            {
                "sequence": index,
                "batch_id": f"future-batch-{offset // 12:03d}",
                "scheduled_round_start_ts": start,
                "market_start_ts": start,
                "market_end_ts": end,
                "market_id": market_id,
                "slug": f"btc-updown-{index:03d}",
                "decision_id": f"{index:064x}",
                "source_row_hash": f"{index + 1000:064x}",
                "entry_sha256": f"{index + 2000:064x}",
                "raw_artifacts": {},
                "capture_quality_valid": True,
                "labels_outcomes_or_pnl_opened": False,
            }
        )
        if offset % 3 == 0:
            candidate_action = "BUY_UP_SELL_BEFORE_CLOSE"
            candidate_side = "UP"
            execution_price = 0.40
        elif offset % 3 == 1:
            candidate_action = "BUY_DOWN_SELL_BEFORE_CLOSE"
            candidate_side = "DOWN"
            execution_price = 0.20
        else:
            candidate_action = "NO_TRADE"
            candidate_side = "NONE"
            execution_price = None
        candidate_guard.append(
            {
                "candidate_name": "adaptive_support_controller_v8_1",
                "market_id": market_id,
                "decision_ts": decision_ts,
                "market_close_ts": end,
                "max_input_ts": decision_ts,
                "selected_action": candidate_action,
                "selected_side": candidate_side,
                "execution_guard_order_allowed": (
                    candidate_action != "NO_TRADE"
                ),
                "selection_reason_codes": (
                    []
                    if candidate_action != "NO_TRADE"
                    else ["controller_abstention"]
                ),
                "target_used_as_decision_time_input": False,
                "outcome_or_pnl_field_used_at_inference": False,
                "labels_outcomes_or_pnl_opened": False,
                "source_score_mutated": False,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "v8_execution_handoff_allowed": False,
                "promotion_evidence_eligible": False,
                "live_trading_enabled": False,
            }
        )
        if candidate_action != "NO_TRADE":
            action = {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "action": candidate_action,
                "max_input_ts": decision_ts,
                "decision_time_features": {
                    "execution_price": execution_price,
                },
                "microstructure_snapshot": {
                    "entry_ask": execution_price,
                },
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
                "paper_only": True,
                "capital_at_risk": False,
            }
            action["action_row_sha256"] = canonical_json_sha256(action)
            actions.append(action)
        features.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "features": {"provider_health_score": 1.0},
            }
        )
        native.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "target_used_as_decision_time_input": False,
            }
        )
        baseline_action = (
            "BUY_DOWN_SELL_BEFORE_CLOSE"
            if offset % 2 == 0
            else "NO_TRADE"
        )
        baseline_guard.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "selected_action": baseline_action,
                "selected_side": (
                    "DOWN" if baseline_action != "NO_TRADE" else "NONE"
                ),
                "execution_guard_order_allowed": (
                    baseline_action != "NO_TRADE"
                ),
                "execution_blocking_reason_codes": (
                    []
                    if baseline_action != "NO_TRADE"
                    else ["policy_abstention"]
                ),
                "target_used_as_decision_time_input": False,
                "outcome_or_pnl_field_used_at_inference": False,
                "labels_outcomes_or_pnl_opened": False,
            }
        )
    return {
        "selected_index_rows": selected,
        "action_rows": actions,
        "feature_rows": features,
        "native_decisions": native,
        "candidate_guard_rows": candidate_guard,
        "baseline_guard_rows": baseline_guard,
        "entry_price_floor_profile": ENTRY_PROFILE,
        "sizing_profile": SIZING_PROFILE,
        "protocol": PROTOCOL,
        "decision_freeze_created_ts": selected[-1]["market_end_ts"] + 1,
    }


def test_builds_exact_120_future_pairs_without_target_access() -> None:
    freeze = build_attempt_002_decision_freeze(**_fixture())

    assert len(freeze["shared_source_rows"]) == 120
    assert len(freeze["candidate_decisions"]) == 120
    assert len(freeze["baseline_decisions"]) == 120
    assert len(freeze["target_free_pairs"]) == 120
    assert freeze["candidate_accepted_market_count"] == 40
    assert freeze["baseline_accepted_market_count"] == 60
    assert freeze["target_access_claim_written"] is False
    assert freeze["outcomes_resolution_labels_or_pnl_opened"] is False
    assert freeze["safety"] == SAFE_FALSES
    assert all(
        row["candidate_id"] == CANDIDATE_ID
        and row["candidate_fixed_position_size"] == 1.0
        and row["baseline_fixed_position_size"] == 0.2
        and row["historical_development_data_used"] is False
        for row in freeze["target_free_pairs"]
    )
    assert all(
        row["promotion_evidence_eligible_before_future_gate"] is False
        and row["outcomes_resolution_labels_or_pnl_opened"] is False
        for row in freeze["candidate_decisions"]
    )


def test_rejects_any_forbidden_target_field() -> None:
    fixture = _fixture()
    fixture["native_decisions"][0]["resolved_outcome"] = "UP"

    with pytest.raises(
        ChallengeAttempt002TargetFreezeError,
        match="forbidden fields",
    ):
        build_attempt_002_decision_freeze(**fixture)


def test_rejects_freeze_timestamp_before_complete_window() -> None:
    fixture = _fixture()
    fixture["decision_freeze_created_ts"] = fixture[
        "selected_index_rows"
    ][-1]["market_end_ts"]

    with pytest.raises(
        ChallengeAttempt002TargetFreezeError,
        match="must follow the complete raw window",
    ):
        build_attempt_002_decision_freeze(**fixture)


def test_rejects_market_order_tamper() -> None:
    fixture = _fixture()
    fixture["candidate_guard_rows"] = copy.deepcopy(
        fixture["candidate_guard_rows"]
    )
    fixture["candidate_guard_rows"][0], fixture[
        "candidate_guard_rows"
    ][1] = (
        fixture["candidate_guard_rows"][1],
        fixture["candidate_guard_rows"][0],
    )

    with pytest.raises(
        ChallengeAttempt002TargetFreezeError,
        match="exact market order",
    ):
        build_attempt_002_decision_freeze(**fixture)
