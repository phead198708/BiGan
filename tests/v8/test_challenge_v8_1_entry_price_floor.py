from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_entry_price_floor import (
    CANDIDATE_ID,
    ChallengeEntryPriceFloorError,
    apply_entry_price_floor,
    build_entry_price_floor_comparison,
    materialize_entry_price_floor_decisions,
    validate_entry_price_floor_profile,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs"
    / "challenge_v8_1_entry_price_floor_0_30_profile.json"
)


def _profile() -> dict:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _guard(
    market_id: str,
    *,
    action: str,
    side: str,
    decision_ts: int,
) -> dict:
    return {
        "candidate_name": "adaptive_support_controller_v8_1",
        "market_id": market_id,
        "decision_ts": decision_ts,
        "market_close_ts": decision_ts + 240_000,
        "max_input_ts": decision_ts - 1,
        "selected_action": action,
        "selected_side": side,
        "execution_guard_order_allowed": action != "NO_TRADE",
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
    entry_price: float,
) -> dict:
    row = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "action": action,
        "decision_time_features": {"execution_price": entry_price},
        "microstructure_snapshot": {"entry_ask": entry_price},
        "target_used_as_decision_input": False,
        "outcome_fields_used_as_decision_input": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    row["action_row_sha256"] = canonical_json_sha256(row)
    return row


def _target(
    market_id: str,
    *,
    action: str,
    side: str,
    pnl: float,
) -> dict:
    return {
        "schema_version": (
            "bigan-v8-runtime-aligned-sbc-net-return-v6-4-target-row-v1"
        ),
        "market_id": market_id,
        "action": action,
        "side": side,
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_used_as_decision_time_input": False,
        "target_available_only_post_exit_or_official_resolution": True,
        "paper_position_size": 0.2,
        "cost_fields_subtracted_exactly_once": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "promotion_evidence_eligible": False,
    }


def test_profile_and_sidecar_are_exactly_pinned() -> None:
    expected = PROFILE_PATH.with_suffix(".sha256").read_text().strip()
    assert hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() == expected
    validate_entry_price_floor_profile(_profile())

    tampered = copy.deepcopy(_profile())
    tampered["policy"]["entry_price_floor_inclusive"] = 0.29
    with pytest.raises(ChallengeEntryPriceFloorError, match="policy"):
        validate_entry_price_floor_profile(tampered)


def test_terminal_floor_keeps_or_vetoes_without_changing_controller_state() -> None:
    below_guard = _guard(
        "market-below",
        action="BUY_DOWN_SELL_BEFORE_CLOSE",
        side="DOWN",
        decision_ts=1_000,
    )
    below = apply_entry_price_floor(
        base_decision=below_guard,
        matched_action_row=_action_row(
            "market-below",
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
            decision_ts=1_000,
            entry_price=0.29,
        ),
        profile=_profile(),
    )
    assert below["selected_action"] == "NO_TRADE"
    assert below["selected_side"] == "NONE"
    assert below["entry_price_filter_passed"] is False
    assert below["base_controller_state_transition_changed"] is False
    assert below["safety"] == SAFE_FALSES

    at_floor_guard = _guard(
        "market-at-floor",
        action="BUY_UP_SELL_BEFORE_CLOSE",
        side="UP",
        decision_ts=2_000,
    )
    at_floor_guard.pop("max_input_ts")
    at_floor_guard["baseline_max_input_ts"] = 1_998
    at_floor_guard["opposite_max_input_ts"] = 1_999
    at_floor = apply_entry_price_floor(
        base_decision=at_floor_guard,
        matched_action_row=_action_row(
            "market-at-floor",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            decision_ts=2_000,
            entry_price=0.30,
        ),
        profile=_profile(),
    )
    assert at_floor["selected_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert at_floor["selected_side"] == "UP"
    assert at_floor["entry_price_filter_passed"] is True
    assert at_floor["max_input_ts"] == 1_999

    no_trade_guard = _guard(
        "market-no-trade",
        action="NO_TRADE",
        side="NONE",
        decision_ts=3_000,
    )
    no_trade_guard["target_used_as_decision_time_input"] = None
    no_trade_guard["selection_reason_codes"] = [
        "v6_7_no_positive_guard_compatible_action"
    ]
    no_trade_guard.pop("max_input_ts")
    no_trade = apply_entry_price_floor(
        base_decision=no_trade_guard,
        matched_action_row=None,
        profile=_profile(),
    )
    assert no_trade["selected_action"] == "NO_TRADE"
    assert no_trade["entry_price_filter_evaluated"] is False
    assert no_trade["entry_price"] is None
    assert no_trade["max_input_ts"] is None


def test_action_row_target_or_hash_tamper_fails_closed() -> None:
    guard = _guard(
        "market-1",
        action="BUY_DOWN_SELL_BEFORE_CLOSE",
        side="DOWN",
        decision_ts=1_000,
    )
    target_tamper = _action_row(
        "market-1",
        action="BUY_DOWN_SELL_BEFORE_CLOSE",
        decision_ts=1_000,
        entry_price=0.40,
    )
    target_tamper["target_used_as_decision_input"] = True
    target_tamper["action_row_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in target_tamper.items()
            if key != "action_row_sha256"
        }
    )
    with pytest.raises(ChallengeEntryPriceFloorError, match="causal"):
        apply_entry_price_floor(
            base_decision=guard,
            matched_action_row=target_tamper,
            profile=_profile(),
        )

    hash_tamper = _action_row(
        "market-1",
        action="BUY_DOWN_SELL_BEFORE_CLOSE",
        decision_ts=1_000,
        entry_price=0.40,
    )
    hash_tamper["decision_time_features"]["execution_price"] = 0.41
    with pytest.raises(ChallengeEntryPriceFloorError, match="hash"):
        apply_entry_price_floor(
            base_decision=guard,
            matched_action_row=hash_tamper,
            profile=_profile(),
        )


def test_materialization_and_outcome_join_are_separate_and_reconcile() -> None:
    market_ids = ["market-low", "market-high", "market-no-trade"]
    guards = [
        _guard(
            "market-low",
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
            side="DOWN",
            decision_ts=1_000,
        ),
        _guard(
            "market-high",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            side="UP",
            decision_ts=2_000,
        ),
        _guard(
            "market-no-trade",
            action="NO_TRADE",
            side="NONE",
            decision_ts=3_000,
        ),
    ]
    actions = [
        _action_row(
            "market-low",
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
            decision_ts=1_000,
            entry_price=0.20,
        ),
        _action_row(
            "market-high",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            decision_ts=2_000,
            entry_price=0.40,
        ),
    ]
    decisions = materialize_entry_price_floor_decisions(
        base_guard_rows=guards,
        five_action_rows=actions,
        frozen_market_ids=market_ids,
        profile=_profile(),
    )
    assert [row["trade_selected"] for row in decisions] == [False, True, False]
    assert all(row["outcome_or_pnl_field_used_at_inference"] is False for row in decisions)

    base_comparison = [
        {
            "market_id": "market-low",
            "challenge_after_cost_pnl": -0.05,
            "v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            "v6_7_after_cost_pnl": -0.05,
        },
        {
            "market_id": "market-high",
            "challenge_after_cost_pnl": 0.08,
            "v6_7_action": "BUY_UP_SELL_BEFORE_CLOSE",
            "v6_7_after_cost_pnl": 0.08,
        },
        {
            "market_id": "market-no-trade",
            "challenge_after_cost_pnl": 0.0,
            "v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            "v6_7_after_cost_pnl": -0.02,
        },
    ]
    comparison = build_entry_price_floor_comparison(
        candidate_decisions=decisions,
        base_comparison_rows=base_comparison,
        base_runtime_targets=[
            _target(
                "market-low",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                side="DOWN",
                pnl=-0.05,
            ),
            _target(
                "market-high",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                side="UP",
                pnl=0.08,
            ),
        ],
        frozen_market_ids=market_ids,
    )
    assert [row["candidate_after_cost_pnl"] for row in comparison] == [
        0.0,
        0.08,
        0.0,
    ]
    assert [row["candidate_minus_baseline_pnl"] for row in comparison] == [
        0.05,
        0.0,
        0.02,
    ]
    assert all(row["candidate_id"] == CANDIDATE_ID for row in comparison)
    assert all(row["promotion_evidence_eligible"] is False for row in comparison)
