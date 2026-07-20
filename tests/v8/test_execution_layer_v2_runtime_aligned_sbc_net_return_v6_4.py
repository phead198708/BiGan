from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 as target_module,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _independent_counterfactual_position_rows,
    compute_runtime_policy_after_cost_target,
    runtime_policy_source_hashes,
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)


def test_v6_4_profile_freezes_development_only_runtime_contract() -> None:
    profile = _profile()
    validate_runtime_aligned_sbc_net_return_v6_4_profile(profile)
    assert profile["development_roles"]["total_market_count"] == 134
    assert profile["access_policy"]["issue_223_oof_opened"] is False
    assert profile["access_policy"]["issue_212_future_outcomes_opened"] is False
    assert profile["access_policy"]["issue_221_paper_outcomes_opened"] is False
    assert profile["safety"]["promotion_evidence_eligible"] is False


def test_v6_4_profile_fails_closed_when_oof_access_is_enabled() -> None:
    profile = copy.deepcopy(_profile())
    profile["access_policy"]["issue_223_oof_opened"] = True
    with pytest.raises(ValueError, match="access"):
        validate_runtime_aligned_sbc_net_return_v6_4_profile(profile)


def test_runtime_policy_source_hashes_match_frozen_profile() -> None:
    assert runtime_policy_source_hashes() == _profile()["runtime_policy_contract"][
        "source_function_sha256"
    ]


def test_closed_target_uses_executable_bid_and_subtracts_cost_once() -> None:
    target = compute_runtime_policy_after_cost_target(
        selected_side="UP",
        entry_price=0.50,
        exit_price=0.60,
        resolved_outcome="DOWN",
        fees=0.001,
        slippage=0.002,
        liquidity_impact=0.003,
        paper_position_size=0.2,
    )
    assert target["position_lifecycle_class"] == "closed_before_settlement"
    assert target["terminal_value_per_contract"] == pytest.approx(0.60)
    assert target["execution_cost_per_contract"] == pytest.approx(0.006)
    assert target["runtime_policy_after_cost_net_pnl_per_contract"] == pytest.approx(
        0.094
    )
    assert target["runtime_policy_after_cost_net_pnl_at_frozen_size"] == pytest.approx(
        0.0188
    )
    assert target["settlement_timestamp_used"] is False


def test_residual_target_uses_official_payout_only_after_decision() -> None:
    winner = compute_runtime_policy_after_cost_target(
        selected_side="DOWN",
        entry_price=0.40,
        exit_price=None,
        resolved_outcome="DOWN",
        fees=0.001,
        slippage=0.002,
        liquidity_impact=0.0,
        paper_position_size=0.2,
    )
    loser = compute_runtime_policy_after_cost_target(
        selected_side="UP",
        entry_price=0.60,
        exit_price=None,
        resolved_outcome="DOWN",
        fees=0.001,
        slippage=0.002,
        liquidity_impact=0.0,
        paper_position_size=0.2,
    )
    assert winner["position_lifecycle_class"] == "settlement_residual"
    assert winner["runtime_policy_after_cost_net_pnl_per_contract"] == pytest.approx(
        0.597
    )
    assert loser["runtime_policy_after_cost_net_pnl_per_contract"] == pytest.approx(
        -0.603
    )
    assert winner["cost_fields_subtracted_exactly_once"] is True


def test_runtime_target_rejects_invalid_side_and_negative_cost() -> None:
    with pytest.raises(ValueError, match="UP/DOWN"):
        compute_runtime_policy_after_cost_target(
            selected_side="NONE",
            entry_price=0.5,
            exit_price=None,
            resolved_outcome="UP",
            fees=0.0,
            slippage=0.0,
            liquidity_impact=0.0,
            paper_position_size=0.2,
        )


def test_counterfactual_actions_replay_in_isolated_position_state(monkeypatch) -> None:
    call_sizes = []

    def fake_lifecycle(*, run_id, feature_rows, entry_fills):
        del run_id, feature_rows
        call_sizes.append(len(entry_fills))
        fill = entry_fills[0]
        return {
            "positions": {
                "positions": [
                    {
                        "position_id": fill["paper_fill_id"],
                        "exit_price": None,
                    }
                ]
            }
        }

    monkeypatch.setattr(target_module, "_paper_position_lifecycle", fake_lifecycle)
    positions = _independent_counterfactual_position_rows(
        run_id="test",
        feature_rows=[],
        entry_fills=[
            {"paper_fill_id": "up"},
            {"paper_fill_id": "down"},
        ],
    )
    assert call_sizes == [1, 1]
    assert sorted(positions) == ["down", "up"]
    with pytest.raises(ValueError, match="non-negative"):
        compute_runtime_policy_after_cost_target(
            selected_side="UP",
            entry_price=0.5,
            exit_price=0.6,
            resolved_outcome="UP",
            fees=-0.001,
            slippage=0.0,
            liquidity_impact=0.0,
            paper_position_size=0.2,
        )


def _profile() -> dict:
    path = Path(
        "examples/v8/polymarket_configs/"
        "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_profile.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))
