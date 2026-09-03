"""Async event loop that stitches OFI, pricing, OMS, and the CLOB feed."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.execution.polymarket_oms import OrderResult, PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator, OFISnapshot, TopOfBook
from bigan.strategies.polymarket_pricing import (
    MarketWindow,
    PolymarketPricingEngine,
    SignalDirection,
)

logger = logging.getLogger(__name__)

OFIEngine = BinanceOFICalculator
DEFAULT_REFERENCE_MAX_AGE_MS = 5_000
DEFAULT_OFI_MAX_AGE_MS = 2_000


@dataclass(frozen=True, slots=True)
class PricingInputs:
    """Point-in-time external inputs used by one pricing decision."""

    timestamp_ms: int
    spot_price: float
    oracle_twap_so_far: float
    twap_weight: float
    volatility_annualized: float


PricingInputsProvider = Callable[[int], PricingInputs]


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
        "pricing_state_ts_ms",
        "pricing_inputs_provider",
        "reference_max_age_ms",
        "ofi_max_age_ms",
        "stale_pricing_inputs",
        "dropped_window_mismatch",
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
        pricing_state_ts_ms: int | None = None,
        pricing_inputs_provider: PricingInputsProvider | None = None,
        reference_max_age_ms: int = DEFAULT_REFERENCE_MAX_AGE_MS,
        ofi_max_age_ms: int = DEFAULT_OFI_MAX_AGE_MS,
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
        self.pricing_state_ts_ms = int(
            window.start_ts_ms if pricing_state_ts_ms is None else pricing_state_ts_ms
        )
        self.pricing_inputs_provider = pricing_inputs_provider
        self.reference_max_age_ms = int(reference_max_age_ms)
        self.ofi_max_age_ms = int(ofi_max_age_ms)
        if self.reference_max_age_ms < 0 or self.ofi_max_age_ms < 0:
            raise ValueError("input freshness bounds must be non-negative")
        self.stale_pricing_inputs = 0
        self.dropped_window_mismatch = 0
        self._callback_registered = False

    def push_alpha_tick(self, book: TopOfBook) -> float:
        """Push one Binance top-of-book event into the alpha engine."""

        return self.ofi_engine.update_and_get_z(
            bid_price=book.bid_price,
            bid_qty=book.bid_qty,
            ask_price=book.ask_price,
            ask_qty=book.ask_qty,
            ts_ms=book.ts_ms,
        )

    def push_tick(self, book: TopOfBook) -> float:
        """Backward-compatible alias for :meth:`push_alpha_tick`."""

        return self.push_alpha_tick(book)

    def ingest_binance_book_ticker(
        self,
        payload: Mapping[str, object],
        *,
        receive_ts_ms: int | None = None,
    ) -> OFISnapshot | None:
        """Ingest a real Binance ``bookTicker`` payload into the alpha engine."""

        return self.ofi_engine.on_book_ticker(payload, ts_ms=receive_ts_ms)

    def update_pricing_inputs(self, inputs: PricingInputs) -> None:
        """Atomically replace spot, oracle, TWAP progress, and volatility."""

        values = (
            inputs.spot_price,
            inputs.oracle_twap_so_far,
            inputs.twap_weight,
            inputs.volatility_annualized,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("pricing inputs must be finite")
        if inputs.spot_price <= 0.0 or inputs.oracle_twap_so_far <= 0.0:
            raise ValueError("spot and oracle TWAP must be positive")
        if not 0.0 <= inputs.twap_weight <= 1.0:
            raise ValueError("twap_weight must be in [0, 1]")
        if inputs.volatility_annualized < 0.0:
            raise ValueError("volatility_annualized must be non-negative")
        self.spot_price = float(inputs.spot_price)
        self.oracle_twap_so_far = float(inputs.oracle_twap_so_far)
        self.twap_weight = float(inputs.twap_weight)
        self.volatility_annualized = float(inputs.volatility_annualized)
        self.pricing_state_ts_ms = int(inputs.timestamp_ms)

    async def start(self) -> None:
        """Connect the feed and register the isolated snapshot callback."""

        if not self._callback_registered:
            self.feed_handler.on_snapshot(self._on_snapshot)
            self._callback_registered = True
        await self.feed_handler.connect()

    async def stop(self) -> None:
        """Close the feed handler without interrupting in-flight isolation."""

        await self.feed_handler.close()

    def process_snapshot_sync(self, snapshot: MarketSnapshot) -> OrderResult | None:
        """Run one snapshot through OFI → signal → OMS.

        ``HOLD`` never reaches the OMS. ``FILLED`` results debit
        ``current_bankroll``; every OMS ``OrderResult`` is appended to history.
        """

        if snapshot.window_id != self.window.window_id:
            self.dropped_window_mismatch += 1
            return None
        inputs = self._pricing_inputs_for(snapshot.timestamp_ms)
        if inputs is None:
            self.stale_pricing_inputs += 1
            return None
        z_ofi = self._current_z_ofi(snapshot.timestamp_ms)
        signal = self.pricing_engine.evaluate_signal(
            window=self.window,
            current_ts_ms=snapshot.timestamp_ms,
            spot_price=inputs.spot_price,
            oracle_twap_so_far=inputs.oracle_twap_so_far,
            twap_weight=inputs.twap_weight,
            z_ofi=z_ofi,
            volatility_annualized=inputs.volatility_annualized,
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
        current_ask_size = (
            snapshot.yes_ask_size
            if signal.direction is SignalDirection.BUY_YES
            else snapshot.no_ask_size
        )
        result = self.oms.process_signal(
            signal,
            self.current_bankroll,
            current_bid,
            current_ask_size,
        )
        if result is None:
            return None
        self.execution_history.append(result)
        if result.status == "FILLED":
            self.current_bankroll = self.oms.bankroll
        return result

    def _pricing_inputs_for(self, decision_ts_ms: int) -> PricingInputs | None:
        if self.pricing_inputs_provider is not None:
            inputs = self.pricing_inputs_provider(decision_ts_ms)
            self.update_pricing_inputs(inputs)
        else:
            inputs = PricingInputs(
                timestamp_ms=self.pricing_state_ts_ms,
                spot_price=self.spot_price,
                oracle_twap_so_far=self.oracle_twap_so_far,
                twap_weight=self.twap_weight,
                volatility_annualized=self.volatility_annualized,
            )
        age_ms = decision_ts_ms - inputs.timestamp_ms
        if age_ms < 0 or age_ms > self.reference_max_age_ms:
            return None
        return inputs

    def _current_z_ofi(self, decision_ts_ms: int) -> float:
        alpha_ts_ms = self.ofi_engine.last_timestamp_ms
        if alpha_ts_ms is None:
            return 0.0
        age_ms = decision_ts_ms - alpha_ts_ms
        if age_ms < 0 or age_ms > self.ofi_max_age_ms:
            return 0.0
        return self.ofi_engine.get_normalized_ofi()

    async def process_snapshot(self, snapshot: MarketSnapshot) -> OrderResult | None:
        """Async wrapper around :meth:`process_snapshot_sync` for the live feed."""

        return self.process_snapshot_sync(snapshot)

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
