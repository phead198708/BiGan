from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_high_price_support_recovery import (
    CANDIDATE_ID,
    ChallengeHighPriceSupportRecoveryError,
    build_high_price_support_recovery_comparison,
    materialize_high_price_support_recovery_decisions,
    validate_high_price_support_recovery_profile,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs"
    / "challenge_v8_1_high_price_support_recovery_0_30_profile.json"
)


def _profile() -> dict:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _guard(
    market_id: str,
    *,
    v8_action: str,
    v8_side: str,
    baseline_action: str,
    baseline_side: str,
    decision_ts: int,
) -> dict:
    return {
        "candidate_name": "adaptive_support_controller_v8_1",
        "market_id": market_id,
        "decision_ts": decision_ts,
        "market_close_ts": decision_ts + 240_000,
        "max_input_ts": decision_ts - 1,
        "baseline_action": baseline_action,
        "v6_7_baseline_action": baseline_action,
        "baseline_side": baseline_side,
        "baseline_decision_ts": decision_ts,
        "baseline_max_input_ts": decision_ts - 1,
        "selected_action": v8_action,
        "selected_side": v8_side,
        "execution_guard_order_allowed": v8_action != "NO_TRADE",
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
        "guard_replay_row_id": f"guard-{market_id}",
        "controller_state_before_id": f"before-{market_id}",
        "controller_state_after_id": f"after-{market_id}",
    }


def _action_row(
    market_id: str,
    *,
    action: str,
    decision_ts: int,
    price: float,
) -> dict:
    row = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "action": action,
        "decision_time_features": {"execution_price": price},
        "microstructure_snapshot": {"entry_ask": price},
        "target_used_as_decision_input": False,
        "outcome_fields_used_as_decision_input": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    row["action_row_sha256"] = canonical_json_sha256(row)
    return row


def test_profile_is_hash_pinned_and_semantically_exact() -> None:
    expected = PROFILE_PATH.with_suffix(".sha256").read_text().strip()
    assert hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() == expected
    validate_high_price_support_recovery_profile(_profile())

    tampered = copy.deepcopy(_profile())
    tampered["policy"]["recovery_source_action_mutated"] = True
    with pytest.raises(
        ChallengeHighPriceSupportRecoveryError,
        match="policy",
    ):
        validate_high_price_support_recovery_profile(tampered)


def test_high_price_path_recovers_exact_baseline_action() -> None:
    market_ids = ["primary", "recovery", "veto"]
    guards = [
        _guard(
            "primary",
            v8_action="BUY_DOWN_SELL_BEFORE_CLOSE",
            v8_side="DOWN",
            baseline_action="BUY_DOWN_SELL_BEFORE_CLOSE",
            baseline_side="DOWN",
            decision_ts=1_000,
        ),
        _guard(
            "recovery",
            v8_action="NO_TRADE",
            v8_side="NONE",
            baseline_action="BUY_UP_SELL_BEFORE_CLOSE",
            baseline_side="UP",
            decision_ts=2_000,
        ),
        _guard(
            "veto",
            v8_action="NO_TRADE",
            v8_side="NONE",
            baseline_action="BUY_DOWN_SELL_BEFORE_CLOSE",
            baseline_side="DOWN",
            decision_ts=3_000,
        ),
    ]
    actions = [
        _action_row(
            "primary",
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
            decision_ts=1_000,
            price=0.40,
        ),
        _action_row(
            "recovery",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            decision_ts=2_000,
            price=0.35,
        ),
        _action_row(
            "veto",
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
            decision_ts=3_000,
            price=0.20,
        ),
    ]
    decisions = materialize_high_price_support_recovery_decisions(
        base_guard_rows=guards,
        five_action_rows=actions,
        frozen_market_ids=market_ids,
        profile=_profile(),
    )

    assert [row["selected_action"] for row in decisions] == [
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    ]
    assert [row["selection_source"] for row in decisions] == [
        "v8_1_primary",
        "v6_7_high_price_support_recovery",
        "price_floor_veto",
    ]
    assert all(
        row["base_controller_state_transition_changed"] is False
        for row in decisions
    )
    assert all(row["safety"] == SAFE_FALSES for row in decisions)

    comparison = build_high_price_support_recovery_comparison(
        candidate_decisions=decisions,
        base_comparison_rows=[
            {
                "market_id": "primary",
                "v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
                "v6_7_after_cost_pnl": 0.05,
            },
            {
                "market_id": "recovery",
                "v6_7_action": "BUY_UP_SELL_BEFORE_CLOSE",
                "v6_7_after_cost_pnl": -0.03,
            },
            {
                "market_id": "veto",
                "v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
                "v6_7_after_cost_pnl": 0.04,
            },
        ],
        frozen_market_ids=market_ids,
    )
    assert [row["candidate_after_cost_pnl"] for row in comparison] == [
        0.05,
        -0.03,
        0.0,
    ]
    assert [row["candidate_minus_baseline_pnl"] for row in comparison] == [
        0.0,
        0.0,
        -0.04,
    ]
    assert all(row["candidate_id"] == CANDIDATE_ID for row in comparison)


def test_recovery_rejects_action_side_or_hash_drift() -> None:
    guard = _guard(
        "market",
        v8_action="NO_TRADE",
        v8_side="NONE",
        baseline_action="BUY_UP_SELL_BEFORE_CLOSE",
        baseline_side="DOWN",
        decision_ts=1_000,
    )
    with pytest.raises(
        ChallengeHighPriceSupportRecoveryError,
        match="action/side",
    ):
        materialize_high_price_support_recovery_decisions(
            base_guard_rows=[guard],
            five_action_rows=[
                _action_row(
                    "market",
                    action="BUY_UP_SELL_BEFORE_CLOSE",
                    decision_ts=1_000,
                    price=0.40,
                )
            ],
            frozen_market_ids=["market"],
            profile=_profile(),
        )


def test_frozen_v6_7_no_trade_is_preserved_without_action_lookup() -> None:
    guard = _guard(
        "no-action",
        v8_action="NO_TRADE",
        v8_side="NONE",
        baseline_action="NO_TRADE",
        baseline_side="NONE",
        decision_ts=0,
    )
    guard["target_used_as_decision_time_input"] = None
    guard["selection_reason_codes"] = [
        "v6_7_no_positive_guard_compatible_action"
    ]
    guard["market_close_ts"] = None
    guard["max_input_ts"] = None
    guard["baseline_decision_ts"] = None
    guard["baseline_max_input_ts"] = None

    decisions = materialize_high_price_support_recovery_decisions(
        base_guard_rows=[guard],
        five_action_rows=[],
        frozen_market_ids=["no-action"],
        profile=_profile(),
    )

    assert decisions[0]["selected_action"] == "NO_TRADE"
    assert decisions[0]["selection_source"] == (
        "v6_7_no_positive_guard_compatible_action"
    )
