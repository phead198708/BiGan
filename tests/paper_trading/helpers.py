"""Deterministic paper-contract builders used by focused tests."""

from __future__ import annotations

from bigan.paper_trading.contracts import (
    PAPER_SCHEMA_VERSION,
    PaperDecisionEvent,
    PaperRunManifest,
    PaperWindowRegistration,
)
from bigan.pipeline.events import (
    STRATEGY_DECISION_SCHEMA_VERSION,
    DecisionDisposition,
    DecisionReason,
    StrategyDecisionEvent,
)

RUN_ID = "paper-test-run"
WINDOW = PaperWindowRegistration(
    window_id="btc-window",
    market_symbol="BTC",
    start_ts_ms=1_000,
    end_ts_ms=10_000,
)


def manifest(*, config_sha256: str = "a" * 64) -> PaperRunManifest:
    return PaperRunManifest(
        schema_version=PAPER_SCHEMA_VERSION,
        run_id=RUN_ID,
        created_at="2026-09-03T00:00:00+00:00",
        source_commit="deadbeef",
        initial_bankroll=1_000.0,
        fee_bps=100.0,
        market_symbols=("BTC",),
        window_ids=(WINDOW.window_id,),
        windows=(WINDOW,),
        config_sha256=config_sha256,
    )


def paper_decision(
    sequence: int,
    *,
    disposition: DecisionDisposition = DecisionDisposition.FILLED,
    side: str = "YES",
    shares: float = 10.0,
    price: float = 0.40,
    fee: float = 0.04,
    cash_before: float = 1_000.0,
    yes_bid: float = 0.39,
    no_bid: float = 0.59,
    event_id: str | None = None,
) -> PaperDecisionEvent:
    filled = disposition is DecisionDisposition.FILLED
    rejected = disposition is DecisionDisposition.REJECTED
    no_order = disposition is DecisionDisposition.NO_ORDER
    has_signal = filled or rejected or no_order or disposition is DecisionDisposition.HOLD
    direction = (
        "HOLD"
        if disposition is DecisionDisposition.HOLD
        else ("BUY_YES" if side == "YES" else "BUY_NO") if has_signal else None
    )
    reason = {
        DecisionDisposition.FILLED: DecisionReason.OMS_FILLED,
        DecisionDisposition.REJECTED: DecisionReason.OMS_REJECTED,
        DecisionDisposition.NO_ORDER: DecisionReason.OMS_NO_RESULT,
        DecisionDisposition.HOLD: DecisionReason.SIGNAL_HOLD,
        DecisionDisposition.DROPPED: DecisionReason.PRICING_INPUTS_MISSING,
    }[disposition]
    notional_and_fee = shares * price + fee if filled else 0.0
    strategy = StrategyDecisionEvent(
        schema_version=STRATEGY_DECISION_SCHEMA_VERSION,
        timestamp_ms=2_000 + sequence,
        window_id=WINDOW.window_id,
        market_symbol=WINDOW.market_symbol,
        window_start_ts_ms=WINDOW.start_ts_ms,
        window_end_ts_ms=WINDOW.end_ts_ms,
        yes_bid=yes_bid,
        yes_ask=0.40,
        yes_bid_size=100.0,
        yes_ask_size=100.0,
        no_bid=no_bid,
        no_ask=0.60,
        no_bid_size=100.0,
        no_ask_size=100.0,
        last_traded_price=0.40,
        alpha_timestamp_ms=None,
        alpha_age_ms=None,
        alpha_is_fresh=False,
        alpha_reason_code=DecisionReason.ALPHA_MISSING,
        z_ofi=0.0,
        pricing_inputs_timestamp_ms=2_000 + sequence if has_signal else None,
        pricing_inputs_age_ms=0 if has_signal else None,
        pricing_inputs_are_fresh=has_signal,
        spot_price=100_000.0 if has_signal else None,
        oracle_twap_so_far=100_000.0 if has_signal else None,
        twap_weight=0.0 if has_signal else None,
        volatility_annualized=0.60 if has_signal else None,
        model_probability=0.70 if has_signal else None,
        market_price=price if has_signal else None,
        effective_strike=100_000.0 if has_signal else None,
        edge=0.30 if has_signal else None,
        ev=0.75 if has_signal else None,
        direction=direction,
        recommended_size_pct=0.05 if has_signal else None,
        order_id=f"oms-{sequence}" if filled or rejected else None,
        order_status="FILLED" if filled else "REJECTED" if rejected else None,
        order_side=side if filled or rejected else None,
        shares=shares if filled else 0.0 if rejected else None,
        fill_price=price if filled else 0.0 if rejected else None,
        fee_usdc=fee if filled else 0.0 if rejected else None,
        reject_reason="Spread too wide" if rejected else None,
        cash_before=cash_before,
        cash_after=cash_before - notional_and_fee,
        disposition=disposition,
        reason_code=reason,
    )
    return PaperDecisionEvent(
        schema_version=PAPER_SCHEMA_VERSION,
        run_id=RUN_ID,
        event_id=event_id or f"{RUN_ID}:decision:{sequence:020d}",
        event_sequence=sequence,
        source_snapshot_id=strategy.source_snapshot_id,
        decision=strategy,
    )
