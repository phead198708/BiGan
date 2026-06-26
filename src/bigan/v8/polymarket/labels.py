"""Explicit label construction for Polymarket BTC 15m UP/DOWN markets."""

from __future__ import annotations

from bigan.v8.phase0 import Label
from bigan.v8.polymarket.contracts import (
    POLYMARKET_SOURCE,
    PolymarketAdapterError,
    PolymarketBinaryMarket,
    PolymarketLabelRow,
    PolymarketTokenSnapshot,
    canonical_json_sha256,
)


def build_polymarket_label_rows(
    *,
    market: PolymarketBinaryMarket,
    token_snapshots: tuple[PolymarketTokenSnapshot, ...],
    reference_price_end: float,
) -> tuple[PolymarketLabelRow, ...]:
    """Build cost-aware labels from explicit BTC settlement semantics."""

    if reference_price_end <= 0.0:
        raise PolymarketAdapterError("invalid_reference_price_end")
    entry = _entry_snapshots(market, token_snapshots)
    up_wins = _up_wins(
        settlement_rule=market.settlement_rule,
        reference_price_start=market.reference_price_at_start,
        reference_price_end=reference_price_end,
    )
    settlement_hash = canonical_json_sha256(
        {
            "market_id": market.market_id,
            "condition_id": market.condition_id,
            "settlement_rule": market.settlement_rule,
            "reference_price_start": market.reference_price_at_start,
            "reference_price_end": reference_price_end,
            "market_start_ts": market.market_start_ts,
            "market_end_ts": market.market_end_ts,
        }
    )
    rows = [
        _label_for_outcome(
            market=market,
            snapshot=entry["UP"],
            outcome="UP",
            outcome_wins=up_wins,
            reference_price_end=reference_price_end,
            settlement_hash=settlement_hash,
        ),
        _label_for_outcome(
            market=market,
            snapshot=entry["DOWN"],
            outcome="DOWN",
            outcome_wins=not up_wins,
            reference_price_end=reference_price_end,
            settlement_hash=settlement_hash,
        ),
    ]
    return tuple(rows)


def _label_for_outcome(
    *,
    market: PolymarketBinaryMarket,
    snapshot: PolymarketTokenSnapshot,
    outcome: str,
    outcome_wins: bool,
    reference_price_end: float,
    settlement_hash: str,
) -> PolymarketLabelRow:
    exit_price = 1.0 if outcome_wins else 0.0
    gross_return = exit_price / snapshot.mid_price - 1.0
    spread_cost = snapshot.spread_bps / 100_000.0
    fee_cost = 0.0002
    slippage_cost = 0.0001
    liquidity_impact_cost = 0.00005 if snapshot.liquidity_depth > 0.0 else 0.001
    total_cost = spread_cost + fee_cost + slippage_cost + liquidity_impact_cost
    net_return = gross_return - total_cost
    is_positive = net_return > 0.0
    v8_label = Label(
        decision_ts=market.market_start_ts,
        label_ts=market.market_end_ts,
        horizon_ms=market.horizon_ms,
        source=POLYMARKET_SOURCE,
        instrument_id=f"{market.slug}:{outcome}",
        entry_price=snapshot.mid_price,
        exit_price=max(exit_price, 1e-12),
        side=1,
        gross_return=gross_return,
        spread_cost=spread_cost,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        liquidity_impact_cost=liquidity_impact_cost,
        total_cost=total_cost,
        net_return=net_return,
        is_positive=is_positive,
    )
    return PolymarketLabelRow(
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        outcome=outcome,  # type: ignore[arg-type]
        reference_price_start=market.reference_price_at_start,
        reference_price_end=reference_price_end,
        market_start_ts=market.market_start_ts,
        market_end_ts=market.market_end_ts,
        horizon_ms=market.horizon_ms,
        settlement_rule=market.settlement_rule,
        raw_settlement_metadata_hash=settlement_hash,
        is_up=outcome == "UP",
        is_down=outcome == "DOWN",
        is_positive=is_positive,
        entry_token_price=snapshot.mid_price,
        exit_token_price=exit_price,
        gross_return=gross_return,
        spread_cost=spread_cost,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        liquidity_impact_cost=liquidity_impact_cost,
        total_cost=total_cost,
        net_return=net_return,
        v8_label=v8_label,
    )


def _entry_snapshots(
    market: PolymarketBinaryMarket,
    snapshots: tuple[PolymarketTokenSnapshot, ...],
) -> dict[str, PolymarketTokenSnapshot]:
    by_outcome: dict[str, PolymarketTokenSnapshot] = {}
    for snapshot in sorted(snapshots, key=lambda item: item.ts):
        if snapshot.ts < market.market_start_ts:
            continue
        by_outcome.setdefault(snapshot.outcome, snapshot)
    if "UP" not in by_outcome:
        raise PolymarketAdapterError("missing_up_entry_snapshot")
    if "DOWN" not in by_outcome:
        raise PolymarketAdapterError("missing_down_entry_snapshot")
    return by_outcome


def _up_wins(
    *,
    settlement_rule: str,
    reference_price_start: float,
    reference_price_end: float,
) -> bool:
    if settlement_rule in {
        "btc_reference_price_end_gt_start_up_else_down",
        "btc_reference_price_close_gt_open_unknown_50_50",
    }:
        return reference_price_end > reference_price_start
    if settlement_rule == "btc_reference_price_close_gte_open_up_else_down":
        return reference_price_end >= reference_price_start
    raise PolymarketAdapterError("unknown_settlement_rule")
