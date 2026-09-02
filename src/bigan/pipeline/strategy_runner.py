"""Async event loop that stitches OFI, pricing, OMS, and the CLOB feed."""

from __future__ import annotations

import logging

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.execution.polymarket_oms import OrderResult, PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator
from bigan.strategies.polymarket_pricing import (
    MarketWindow,
    PolymarketPricingEngine,
    SignalDirection,
)

logger = logging.getLogger(__name__)

OFIEngine = BinanceOFICalculator


class StrategyRunner:
    """Feed-driven loop: snapshot → z_ofi → PricingSignal → OMS fill."""

    __slots__ = (
        "ofi_engine",
        "pricing_engine",
        "oms",
        "feed_handler",
        "window",
        "current_bankroll",
        "execution_history",
        "callback_errors",
        "oms_calls",
        "spot_price",
        "volatility_annualized",
        "oracle_twap_so_far",
        "twap_weight",
        "ofi_bid_qty",
        "ofi_ask_qty",
        "_callback_registered",
    )

    def __init__(
        self,
        *,
        ofi_engine: OFIEngine,
        pricing_engine: PolymarketPricingEngine,
        oms: PolymarketOMS,
        feed_handler: PolymarketFeedHandler,
        window: MarketWindow,
        initial_bankroll: float,
        spot_price: float,
        volatility_annualized: float = 0.60,
        oracle_twap_so_far: float | None = None,
        twap_weight: float = 0.0,
        ofi_bid_qty: float = 1.0,
        ofi_ask_qty: float = 1.0,
    ) -> None:
        bankroll = float(initial_bankroll)
        if bankroll <= 0.0:
            raise ValueError("initial_bankroll must be positive")
        self.ofi_engine = ofi_engine
        self.pricing_engine = pricing_engine
        self.oms = oms
        self.feed_handler = feed_handler
        self.window = window
        self.current_bankroll = bankroll
        self.execution_history: list[OrderResult] = []
        self.callback_errors = 0
        self.oms_calls = 0
        self.spot_price = float(spot_price)
        self.volatility_annualized = float(volatility_annualized)
        self.oracle_twap_so_far = (
            float(oracle_twap_so_far)
            if oracle_twap_so_far is not None
            else float(window.strike_price)
        )
        self.twap_weight = float(twap_weight)
        self.ofi_bid_qty = float(ofi_bid_qty)
        self.ofi_ask_qty = float(ofi_ask_qty)
        self._callback_registered = False

    def push_tick(self, snapshot: MarketSnapshot) -> float:
        """Push YES top-of-book into the OFI engine and return ``z_ofi``."""

        return self.ofi_engine.update_and_get_z(
            bid_price=snapshot.yes_bid,
            bid_qty=self.ofi_bid_qty,
            ask_price=snapshot.yes_ask,
            ask_qty=self.ofi_ask_qty,
            ts_ms=snapshot.timestamp_ms,
        )

    async def start(self) -> None:
        """Connect the feed and register the isolated snapshot callback."""

        if not self._callback_registered:
            self.feed_handler.on_snapshot(self._on_snapshot)
            self._callback_registered = True
        await self.feed_handler.connect()

    async def stop(self) -> None:
        """Close the feed handler without interrupting in-flight isolation."""

        await self.feed_handler.close()

    async def process_snapshot(self, snapshot: MarketSnapshot) -> OrderResult | None:
        """Run one snapshot through OFI → signal → OMS.

        ``HOLD`` never reaches the OMS. ``FILLED`` results debit
        ``current_bankroll``; every OMS ``OrderResult`` is appended to history.
        """

        z_ofi = self.push_tick(snapshot)
        signal = self.pricing_engine.evaluate_signal(
            window=self.window,
            current_ts_ms=snapshot.timestamp_ms,
            spot_price=self.spot_price,
            oracle_twap_so_far=self.oracle_twap_so_far,
            twap_weight=self.twap_weight,
            z_ofi=z_ofi,
            volatility_annualized=self.volatility_annualized,
            yes_ask_price=snapshot.yes_ask,
            no_ask_price=snapshot.no_ask,
        )
        if signal.direction is SignalDirection.HOLD:
            return None
        current_bid = (
            snapshot.yes_bid
            if signal.direction is SignalDirection.BUY_YES
            else snapshot.no_bid
        )
        self.oms_calls += 1
        result = self.oms.process_signal(signal, self.current_bankroll, current_bid)
        if result is None:
            return None
        self.execution_history.append(result)
        if result.status == "FILLED":
            self.current_bankroll = self.oms.bankroll
        return result

    async def _on_snapshot(self, snapshot: MarketSnapshot) -> None:
        try:
            await self.process_snapshot(snapshot)
        except Exception:
            self.callback_errors += 1
            logger.exception(
                "strategy.snapshot.failed window_id=%s ts_ms=%s",
                snapshot.window_id,
                snapshot.timestamp_ms,
            )
