from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    FULL_ACTION_GRID,
    build_v7_0_lineage_audit,
    validate_abstention_aware_v7_0_profile,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _runtime_row(*, market: str, role: str, side: str, decision_ts: int) -> dict:
    return {
        "market_id": market,
        "role": role,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "features": {"canonical_v6_2_score": 0.1},
        "target_used_as_decision_time_input": False,
        "target_available_only_post_exit_or_official_resolution": True,
    }


def _full_grid_row(*, market: str, decision_ts: int) -> dict:
    return {
        "market_id": market,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "decision_time_features": {"execution_price": 0.5},
        "evaluation_target_net_pnl_per_contract_by_action": dict.fromkeys(FULL_ACTION_GRID, 0.0),
        "target_outcome_available_only_post_resolution": True,
        "target_provenance": {"outcome_used_as_decision_input": False},
    }


def _future_row(*, market: str, market_start_ts: int) -> dict:
    return {
        "market_id": market,
        "market_start_ts": market_start_ts,
        "market_end_ts": market_start_ts + 300_000,
        "scheduled_round_start_ts": market_start_ts,
        "labels_outcomes_or_pnl_opened": False,
        "resolution_provider_called": False,
    }


def _complete_historical_rows(*, overlap_market: str | None = None) -> tuple[list, list]:
    runtime_rows = [
        _runtime_row(
            market=overlap_market if index == 0 and overlap_market else f"fit-{index}",
            role="development_train",
            side="UP" if index % 2 == 0 else "DOWN",
            decision_ts=1_000 + index,
        )
        for index in range(89)
    ]
    runtime_rows.extend(
        _runtime_row(
            market=f"calibration-{index}",
            role="development_calibration",
            side="UP" if index % 2 == 0 else "DOWN",
            decision_ts=10_000 + index,
        )
        for index in range(45)
    )
    full_grid_rows = [
        _full_grid_row(market=f"full-grid-{index}", decision_ts=5_000 + index)
        for index in range(65)
    ]
    return runtime_rows, full_grid_rows


def test_profile_preserves_abstention_and_rejects_side_quota() -> None:
    validate_abstention_aware_v7_0_profile(_profile())
    drift = _profile()
    drift["selection_policy"]["side_quota_allowed"] = True
    with pytest.raises(ValueError, match="no_side_rule"):
        validate_abstention_aware_v7_0_profile(drift)


def test_lineage_audit_passes_without_opening_future_targets() -> None:
    runtime, full_grid = _complete_historical_rows()
    audit = build_v7_0_lineage_audit(
        profile=_profile(),
        runtime_rows=runtime,
        full_action_grid_rows=full_grid,
        issue229_rows=[_future_row(market="issue229", market_start_ts=20_000)],
        issue231_rows=[_future_row(market="issue231", market_start_ts=30_000)],
        implementation_commit="a" * 40,
        audit_created_ts=40_000,
    )
    assert audit["lineage_audit_passed"] is True
    assert audit["historical_future_market_overlap_count"] == 0
    assert audit["issue229_or_issue231_outcomes_opened"] is False
    assert audit["model_fit_attempted"] is False
    assert audit["paper_candidate_allowed"] is False
    assert audit["capital_at_risk"] is False


def test_lineage_audit_fails_closed_on_excluded_market_overlap() -> None:
    runtime, full_grid = _complete_historical_rows(overlap_market="overlap")
    audit = build_v7_0_lineage_audit(
        profile=_profile(),
        runtime_rows=runtime,
        full_action_grid_rows=full_grid,
        issue229_rows=[_future_row(market="overlap", market_start_ts=20_000)],
        issue231_rows=[_future_row(market="issue231", market_start_ts=30_000)],
        implementation_commit="b" * 40,
        audit_created_ts=40_000,
    )
    assert audit["lineage_audit_passed"] is False
    assert audit["lineage_audit_blocking_reason_codes"] == [
        "historical_source_overlaps_excluded_future_market"
    ]


def test_profile_rejects_future_outcome_tuning() -> None:
    drift = _profile()
    drift["fit_protocol"]["uses_issue229_or_issue231_outcomes_for_fit_or_tuning"] = (
        True
    )
    with pytest.raises(ValueError, match="target_isolation"):
        validate_abstention_aware_v7_0_profile(drift)
