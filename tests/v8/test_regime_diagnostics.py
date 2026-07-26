from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.regime_diagnostics import (
    DIMENSION_BUCKETS,
    RegimeDiagnosticError,
    assign_regime,
    build_regime_stratified_diagnostics,
    regime_diagnostics_markdown,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "examples/v8/polymarket_configs/regime_definition_contract.json"
)
CONTRACT_SHA = CONTRACT_PATH.with_suffix(".sha256")


def _contract():
    return json.loads(CONTRACT_PATH.read_text())


def _decision(index: int, **overrides):
    row = {
        "market_id": f"market-{index}",
        "decision_ts": index * 3_600_000,
        "selected_side": "UP" if index % 2 == 0 else "DOWN",
        "executed_action": "BUY_UP_HOLD_TO_SETTLEMENT",
        "decision_origin": "fallback_v6_7" if index % 3 == 0 else "primary",
        "execution_guard_order_allowed": True,
    }
    row.update(overrides)
    return row


def _context(decision_ts: int, **overrides):
    row = {
        "available_at_ts": decision_ts,
        "max_input_ts": decision_ts,
        "reference_return": 0.001,
        "realized_volatility": 0.0015,
        "combined_spread_bps": 100,
        "liquidity_depth": 500,
        "provider_health_score": 0.9,
        "provider_coverage_complete": 1,
        "trade_tape_provider_timeout": 0,
        "trade_tape_truncated": 0,
        "trade_tape_censored": 0,
        "trade_tape_historical_backfill": 0,
    }
    row.update(overrides)
    return row


def test_regime_contract_is_hash_pinned_and_safety_closed() -> None:
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        CONTRACT_SHA.read_text().strip()
    )
    assert all(value is False for value in _contract()["safety"].values())


def test_boundary_assignments_are_deterministic_and_hash_stable() -> None:
    decision = _decision(7)
    context = _context(decision["decision_ts"])
    first = assign_regime(decision=decision, causal_context=context, contract=_contract())
    second = assign_regime(decision=decision, causal_context=context, contract=_contract())
    assert first == second
    assert first["regime"] == "bullish"
    assert first["realized_volatility_bucket"] == "medium"
    assert first["spread_liquidity_bucket"] == "tight_high"
    assert first["time_of_day_bucket"] == "utc_06_11"
    assert first["provider_health_bucket"] == "healthy"


@pytest.mark.parametrize(
    ("reference_return", "expected"),
    [(-0.001, "bearish"), (-0.000999, "sideways"), (0.000999, "sideways"), (0.001, "bullish")],
)
def test_regime_return_boundaries(reference_return: float, expected: str) -> None:
    decision = _decision(1)
    assignment = assign_regime(
        decision=decision,
        causal_context=_context(decision["decision_ts"], reference_return=reference_return),
        contract=_contract(),
    )
    assert assignment["regime"] == expected


def test_missing_inputs_are_explicit_unknown_strata() -> None:
    decision = _decision(1)
    assignment = assign_regime(
        decision=decision,
        causal_context=_context(
            decision["decision_ts"],
            reference_return=None,
            realized_volatility=None,
            combined_spread_bps=None,
            liquidity_depth=None,
            provider_health_score=None,
        ),
        contract=_contract(),
    )
    assert assignment["regime"] == "unknown"
    assert assignment["realized_volatility_bucket"] == "unknown"
    assert assignment["spread_liquidity_bucket"] == "unknown"
    assert assignment["provider_health_bucket"] == "unknown"


def test_future_or_target_input_fails_closed() -> None:
    decision = _decision(1)
    with pytest.raises(RegimeDiagnosticError, match="decision_ts"):
        assign_regime(
            decision=decision,
            causal_context=_context(
                decision["decision_ts"], available_at_ts=decision["decision_ts"] + 1
            ),
            contract=_contract(),
        )
    with pytest.raises(RegimeDiagnosticError, match="target fields"):
        assign_regime(
            decision=decision,
            causal_context={
                **_context(decision["decision_ts"]),
                "resolved_outcome": "UP",
            },
            contract=_contract(),
        )


def test_all_dimensions_report_empty_low_support_and_reconcile() -> None:
    decisions = [_decision(index) for index in range(12)]
    assignments = [
        assign_regime(
            decision=decision,
            causal_context=_context(
                decision["decision_ts"],
                reference_return=-0.002 if index < 6 else 0.002,
                provider_health_score=1.0 if index != 0 else 0.2,
            ),
            contract=_contract(),
        )
        for index, decision in enumerate(decisions)
    ]
    candidate_rows = [
        {**decision, "after_cost_pnl": 0.02 if index < 8 else -0.01}
        for index, decision in enumerate(decisions)
    ]
    baseline_rows = [{**decision, "after_cost_pnl": 0.0} for decision in decisions]
    artifacts = build_regime_stratified_diagnostics(
        assignments=assignments,
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        contract=_contract(),
    )
    report = artifacts["regime_stratified_pnl_report"]
    assert report["all_dimension_partitions_reconcile"] is True
    assert report["aggregate"]["after_cost_pnl"] == pytest.approx(0.12)
    assert report["dimensions"]["regime"]["sideways"]["status"] == "empty"
    assert report["dimensions"]["provider_health_bucket"]["unhealthy"]["status"] == (
        "insufficient_support"
    )
    for dimension, buckets in DIMENSION_BUCKETS.items():
        assert set(report["dimensions"][dimension]) == set(buckets)
    assert report["diagnostic_only"] is True
    assert report["stratified_metrics_are_eligibility_blockers"] is False
    assert "promotion unlocked: `false`" in regime_diagnostics_markdown(artifacts)


def test_grid_or_assignment_tamper_fails_closed() -> None:
    decision = _decision(1)
    assignment = assign_regime(
        decision=decision,
        causal_context=_context(decision["decision_ts"]),
        contract=_contract(),
    )
    bad_baseline = copy.deepcopy(decision)
    bad_baseline["market_id"] = "other"
    with pytest.raises(RegimeDiagnosticError, match="grids differ"):
        build_regime_stratified_diagnostics(
            assignments=[assignment],
            candidate_rows=[{**decision, "after_cost_pnl": 0.1}],
            baseline_rows=[{**bad_baseline, "after_cost_pnl": 0.0}],
            contract=_contract(),
        )
