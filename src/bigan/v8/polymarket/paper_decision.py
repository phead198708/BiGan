"""Paper-only Polymarket decision mapping from v8 policy signals."""

from __future__ import annotations

from dataclasses import dataclass

from bigan.v8.phase4 import AdaptiveDecision
from bigan.v8.polymarket.contracts import (
    POLYMARKET_SOURCE,
    PolymarketAdapterError,
    PolymarketBinaryDecision,
    PolymarketBinaryMarket,
    PolymarketFeatureRow,
    PolymarketLabelRow,
    PolymarketTokenSnapshot,
)


@dataclass(frozen=True, slots=True)
class PolymarketPolicySignal:
    """Minimal v8 policy signal used by the Polymarket paper adapter."""

    decision_ts: int
    action: float
    confidence: float
    score: float
    estimated_up_probability: float

    def __post_init__(self) -> None:
        if self.decision_ts < 0:
            raise ValueError("decision_ts must be non-negative")
        if not 0.0 <= self.action <= 1.0:
            raise ValueError("action must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.estimated_up_probability <= 1.0:
            raise ValueError("estimated_up_probability must be in [0, 1]")


def build_polymarket_paper_decisions(
    *,
    market: PolymarketBinaryMarket,
    feature_rows: tuple[PolymarketFeatureRow, ...],
    token_snapshots: tuple[PolymarketTokenSnapshot, ...],
    policy_signals: tuple[PolymarketPolicySignal, ...],
    min_confidence: float = 0.60,
    min_edge: float = 0.015,
    max_paper_size: float = 0.20,
) -> tuple[PolymarketBinaryDecision, ...]:
    """Translate v8 policy output into paper-only Polymarket decisions."""

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if min_edge < 0.0:
        raise ValueError("min_edge must be non-negative")
    if max_paper_size < 0.0:
        raise ValueError("max_paper_size must be non-negative")
    signals = {signal.decision_ts: signal for signal in policy_signals}
    snapshots = _snapshots_by_ts(token_snapshots)
    decisions: list[PolymarketBinaryDecision] = []
    for row in feature_rows:
        signal = signals.get(row.decision_ts)
        if signal is None:
            raise PolymarketAdapterError("missing_policy_signal")
        decisions.append(
            _decision_for_signal(
                market=market,
                signal=signal,
                snapshots=snapshots.get(row.decision_ts, {}),
                min_confidence=min_confidence,
                min_edge=min_edge,
                max_paper_size=max_paper_size,
            )
        )
    return tuple(decisions)


def polymarket_decisions_to_phase4(
    *,
    decisions: tuple[PolymarketBinaryDecision, ...],
    labels: tuple[PolymarketLabelRow, ...],
) -> tuple[AdaptiveDecision, ...]:
    """Convert Polymarket paper decisions into existing Phase 4 decisions."""

    label_by_outcome = {label.outcome: label for label in labels}
    phase4: list[AdaptiveDecision] = []
    for index, decision in enumerate(decisions):
        label = label_by_outcome.get(decision.selected_outcome)
        if decision.selected_outcome == "NO_TRADE" or label is None:
            gross_return = 0.0
            spread_cost = 0.0
            fee_cost = 0.0
            slippage_cost = 0.0
            liquidity_impact_cost = 0.0
            net_return = 0.0
            filled_action = 0.0
            instrument_id = f"{decision.slug}:NO_TRADE"
        else:
            gross_return = label.gross_return
            spread_cost = label.spread_cost
            fee_cost = label.fee_cost
            slippage_cost = label.slippage_cost
            liquidity_impact_cost = label.liquidity_impact_cost
            net_return = label.net_return
            filled_action = min(1.0, max(0.0, decision.paper_notional))
            instrument_id = f"{decision.slug}:{decision.selected_outcome}"
        total_cost = spread_cost + fee_cost + slippage_cost + liquidity_impact_cost
        phase4.append(
            AdaptiveDecision(
                decision_ts=decision.decision_ts,
                source=POLYMARKET_SOURCE,
                instrument_id=instrument_id,
                raw_action=decision.v8_action,
                adapted_action=filled_action,
                filled_action=filled_action,
                confidence=decision.v8_confidence,
                score=decision.v8_score,
                regime="trend",
                raw_regime="trend",
                pending_regime_active=False,
                transitioned=False,
                lambda_value=0.30,
                execution_aggressiveness=0.90,
                fill_probability=1.0,
                turnover=filled_action if index == 0 else 0.02,
                shadow_net_return=net_return,
                gross_return=gross_return,
                spread_cost=spread_cost,
                fee_cost=fee_cost,
                slippage_cost=slippage_cost,
                liquidity_impact_cost=liquidity_impact_cost,
                total_execution_cost=total_cost,
                risk_penalty=0.0,
                turnover_penalty=0.0,
                net_return=net_return,
                baseline_net_return=net_return,
                drawdown=0.0,
            )
        )
    if not phase4:
        raise PolymarketAdapterError("no_phase4_decisions")
    return tuple(phase4)


def _decision_for_signal(
    *,
    market: PolymarketBinaryMarket,
    signal: PolymarketPolicySignal,
    snapshots: dict[str, PolymarketTokenSnapshot],
    min_confidence: float,
    min_edge: float,
    max_paper_size: float,
) -> PolymarketBinaryDecision:
    if signal.decision_ts >= market.market_end_ts:
        return _no_trade_decision(market, signal, ("market_closed",))
    if signal.confidence < min_confidence:
        return _no_trade_decision(market, signal, ("low_confidence",))

    up_probability = signal.estimated_up_probability
    selected_outcome = "UP" if up_probability >= 0.5 else "DOWN"
    selected_probability = (
        up_probability if selected_outcome == "UP" else 1.0 - up_probability
    )
    selected_snapshot = snapshots.get(selected_outcome)
    if selected_snapshot is None:
        return _no_trade_decision(market, signal, ("missing_token_price",))
    edge = selected_probability - selected_snapshot.mid_price
    if edge < min_edge:
        return _no_trade_decision(
            market,
            signal,
            ("negative_edge",) if edge < 0.0 else ("insufficient_edge",),
            estimated_probability=selected_probability,
            token_mid_price=selected_snapshot.mid_price,
            edge=edge,
        )
    paper_notional = min(max_paper_size, max(0.0, edge * 2.0))
    return PolymarketBinaryDecision(
        decision_ts=signal.decision_ts,
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        selected_outcome=selected_outcome,  # type: ignore[arg-type]
        selected_token_id=market.token_id_for_outcome(selected_outcome),  # type: ignore[arg-type]
        opposite_token_id=market.opposite_token_id_for_outcome(
            selected_outcome  # type: ignore[arg-type]
        ),
        v8_action=signal.action,
        v8_confidence=signal.confidence,
        v8_score=signal.score,
        estimated_probability=selected_probability,
        token_mid_price=selected_snapshot.mid_price,
        edge=edge,
        max_paper_size=max_paper_size,
        paper_notional=paper_notional,
        reason_codes=("positive_edge", "paper_only_guard"),
        paper_action=f"BUY_{selected_outcome}",  # type: ignore[arg-type]
    )


def _no_trade_decision(
    market: PolymarketBinaryMarket,
    signal: PolymarketPolicySignal,
    reason_codes: tuple[str, ...],
    *,
    estimated_probability: float | None = None,
    token_mid_price: float | None = None,
    edge: float = 0.0,
) -> PolymarketBinaryDecision:
    return PolymarketBinaryDecision(
        decision_ts=signal.decision_ts,
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        selected_outcome="NO_TRADE",
        selected_token_id=None,
        opposite_token_id=None,
        v8_action=signal.action,
        v8_confidence=signal.confidence,
        v8_score=signal.score,
        estimated_probability=estimated_probability,
        token_mid_price=token_mid_price,
        edge=edge,
        max_paper_size=0.0,
        paper_notional=0.0,
        reason_codes=(*reason_codes, "paper_only_guard"),
        paper_action="NO_TRADE",
    )


def _snapshots_by_ts(
    snapshots: tuple[PolymarketTokenSnapshot, ...],
) -> dict[int, dict[str, PolymarketTokenSnapshot]]:
    grouped: dict[int, dict[str, PolymarketTokenSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.ts, {})[snapshot.outcome] = snapshot
    return grouped
