"""Deterministic paper replay fixtures."""

from __future__ import annotations

from bigan.v8.phase4 import AdaptiveDecision


def synthetic_phase4_decisions(row_count: int = 12) -> tuple[AdaptiveDecision, ...]:
    """Build a small deterministic Phase 4 decision stream for paper tests."""

    base_returns = (
        0.010,
        0.012,
        0.009,
        0.013,
        0.011,
        0.014,
        0.010,
        0.012,
        0.011,
        0.013,
        0.010,
        0.012,
    )
    rows: list[AdaptiveDecision] = []
    for index in range(row_count):
        net_return = base_returns[index % len(base_returns)]
        filled_action = 0.25 + 0.02 * (index % 4)
        total_cost = 0.001 + 0.0001 * (index % 3)
        rows.append(
            AdaptiveDecision(
                decision_ts=2_200_000 + index * 60_000,
                source="paper_fixture",
                instrument_id="btc-up",
                raw_action=filled_action,
                adapted_action=filled_action,
                filled_action=filled_action,
                confidence=0.82,
                score=0.78,
                regime="trend",
                raw_regime="trend",
                pending_regime_active=False,
                transitioned=False,
                lambda_value=0.30,
                execution_aggressiveness=0.90,
                fill_probability=1.0,
                turnover=0.02,
                shadow_net_return=net_return,
                gross_return=net_return + total_cost,
                spread_cost=total_cost * 0.25,
                fee_cost=total_cost * 0.25,
                slippage_cost=total_cost * 0.25,
                liquidity_impact_cost=total_cost * 0.25,
                total_execution_cost=total_cost,
                risk_penalty=0.0,
                turnover_penalty=0.0,
                net_return=net_return,
                baseline_net_return=net_return,
                drawdown=0.0,
            )
        )
    return tuple(rows)
