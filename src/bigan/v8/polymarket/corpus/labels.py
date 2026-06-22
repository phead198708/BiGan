"""Settlement-aware label construction for Polymarket BTC corpus rows."""

from __future__ import annotations

from bigan.v8.polymarket.corpus.contracts import (
    CorpusLabelAction,
    PolymarketCorpusBookSnapshot,
    PolymarketCorpusBuildConfig,
    PolymarketCorpusFeatureRow,
    PolymarketCorpusLabelRow,
    PolymarketCorpusMarket,
    PolymarketCorpusResolutionEvent,
)
from bigan.v8.polymarket.rules import PolymarketResolutionRule


def build_polymarket_corpus_label_rows(
    *,
    markets: tuple[PolymarketCorpusMarket, ...],
    rules: dict[str, PolymarketResolutionRule],
    book_snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    resolution_events: dict[str, PolymarketCorpusResolutionEvent],
    feature_rows: tuple[PolymarketCorpusFeatureRow, ...],
    config: PolymarketCorpusBuildConfig,
) -> tuple[PolymarketCorpusLabelRow, ...]:
    """Build future-aware labels from feature rows and Phase 1 rule semantics."""

    market_by_id = {market.market_id: market for market in markets}
    snapshots_by_market = _snapshots_by_market(book_snapshots)
    rows: list[PolymarketCorpusLabelRow] = []
    for feature in sorted(feature_rows, key=lambda item: (item.decision_ts, item.market_id)):
        market = market_by_id[feature.market_id]
        rule = rules[market.market_id]
        resolution = resolution_events[market.market_id]
        actions: list[CorpusLabelAction] = ["NO_TRADE"]
        if config.include_settlement_labels:
            actions.extend(
                [
                    "BUY_UP_HOLD_TO_SETTLEMENT",
                    "BUY_DOWN_HOLD_TO_SETTLEMENT",
                ]
            )
        if config.include_trade_labels:
            actions.extend(
                [
                    "BUY_UP_SELL_BEFORE_CLOSE",
                    "BUY_DOWN_SELL_BEFORE_CLOSE",
                ]
            )
        for action in actions:
            rows.append(
                _label_for_action(
                    market=market,
                    rule=rule,
                    resolution=resolution,
                    snapshots=snapshots_by_market[market.market_id],
                    feature=feature,
                    action=action,
                )
            )
    if not rows:
        raise ValueError("no Polymarket corpus labels")
    return tuple(sorted(rows, key=lambda item: (item.decision_ts, item.market_id, item.action)))


def _label_for_action(
    *,
    market: PolymarketCorpusMarket,
    rule: PolymarketResolutionRule,
    resolution: PolymarketCorpusResolutionEvent,
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    feature: PolymarketCorpusFeatureRow,
    action: CorpusLabelAction,
) -> PolymarketCorpusLabelRow:
    if action == "NO_TRADE":
        return PolymarketCorpusLabelRow(
            market_id=market.market_id,
            condition_id=market.condition_id,
            slug=market.slug,
            market_family=market.market_family,
            horizon_ms=market.horizon_ms,
            decision_ts=feature.decision_ts,
            action=action,
            outcome="NONE",
            entry_bid=0.0,
            entry_ask=0.0,
            entry_mid=0.0,
            exit_bid=0.0,
            exit_ask=0.0,
            settlement_payout=0.0,
            realized_trade_return=0.0,
            settlement_return=0.0,
            total_net_return=0.0,
            fees=0.0,
            slippage=0.0,
            liquidity_impact=0.0,
            is_positive=False,
            resolved_outcome=resolution.resolved_outcome,
            resolution_status=resolution.resolution_status,
            comparator=rule.comparator,
            tie_breaker=rule.tie_breaker,
            resolution_rule_sha256=rule.raw_rule_sha256,
            raw_resolution_sha256=resolution.raw_resolution_sha256,
        )
    outcome = "UP" if "_UP_" in action else "DOWN"
    entry = _last_snapshot(snapshots=snapshots, outcome=outcome, decision_ts=feature.decision_ts)
    if entry is None:
        raise ValueError("missing entry snapshot for label")
    exit_snapshot = _last_snapshot(
        snapshots=snapshots,
        outcome=outcome,
        decision_ts=market.market_end_ts - 1,
    )
    if exit_snapshot is None:
        raise ValueError("missing exit snapshot for label")
    fees = 0.0002
    slippage = max(0.0001, (entry.ask_price - entry.bid_price) / 2.0)
    liquidity_impact = 0.00005 if entry.liquidity_depth > 0.0 else 0.001
    if action.endswith("HOLD_TO_SETTLEMENT"):
        payout = resolution.payout_up if outcome == "UP" else resolution.payout_down
        realized_trade_return = 0.0
        settlement_return = payout / entry.ask_price - 1.0
        exit_bid = 0.0
        exit_ask = 0.0
    else:
        payout = 0.0
        exit_bid = exit_snapshot.bid_price
        exit_ask = exit_snapshot.ask_price
        realized_trade_return = exit_bid / entry.ask_price - 1.0
        settlement_return = 0.0
    total_net_return = (
        realized_trade_return
        + settlement_return
        - fees
        - slippage
        - liquidity_impact
    )
    return PolymarketCorpusLabelRow(
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        market_family=market.market_family,
        horizon_ms=market.horizon_ms,
        decision_ts=feature.decision_ts,
        action=action,
        outcome=outcome,
        entry_bid=entry.bid_price,
        entry_ask=entry.ask_price,
        entry_mid=entry.mid_price,
        exit_bid=exit_bid,
        exit_ask=exit_ask,
        settlement_payout=payout,
        realized_trade_return=realized_trade_return,
        settlement_return=settlement_return,
        total_net_return=total_net_return,
        fees=fees,
        slippage=slippage,
        liquidity_impact=liquidity_impact,
        is_positive=total_net_return > 0.0,
        resolved_outcome=resolution.resolved_outcome,
        resolution_status=resolution.resolution_status,
        comparator=rule.comparator,
        tie_breaker=rule.tie_breaker,
        resolution_rule_sha256=rule.raw_rule_sha256,
        raw_resolution_sha256=resolution.raw_resolution_sha256,
    )


def _snapshots_by_market(
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
) -> dict[str, tuple[PolymarketCorpusBookSnapshot, ...]]:
    grouped: dict[str, list[PolymarketCorpusBookSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.market_id, []).append(snapshot)
    return {
        key: tuple(sorted(value, key=lambda item: (item.ts, item.outcome)))
        for key, value in grouped.items()
    }


def _last_snapshot(
    *,
    snapshots: tuple[PolymarketCorpusBookSnapshot, ...],
    outcome: str,
    decision_ts: int,
) -> PolymarketCorpusBookSnapshot | None:
    eligible = [
        snapshot
        for snapshot in snapshots
        if snapshot.outcome == outcome
        and snapshot.ts <= decision_ts
        and snapshot.available_at_ts <= decision_ts
    ]
    return eligible[-1] if eligible else None
