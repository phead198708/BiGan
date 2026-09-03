"""Async event loop that stitches OFI, pricing, OMS, and the CLOB feed."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TypeAlias

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.execution.polymarket_oms import (
    OrderResult,
    PolymarketOMS,
    SignalIdentity,
    signal_identity,
)
from bigan.features.binance_ofi import BinanceOFICalculator, OFISnapshot, TopOfBook
from bigan.strategies.polymarket_pricing import (
    MarketWindow,
    PolymarketPricingEngine,
    PricingSignal,
    SignalDirection,
)

from .events import (
    STRATEGY_DECISION_SCHEMA_VERSION,
    DecisionDisposition,
    DecisionReason,
    StrategyDecisionEvent,
)

logger = logging.getLogger(__name__)

OFIEngine: TypeAlias = BinanceOFICalculator
DEFAULT_REFERENCE_MAX_AGE_MS = 5_000
DEFAULT_OFI_MAX_AGE_MS = 2_000
DEFAULT_EXECUTION_HISTORY_LIMIT = 10_000


@dataclass(frozen=True, slots=True)
class PricingInputs:
    """Point-in-time external inputs used by one pricing decision."""

    timestamp_ms: int
    spot_price: float
    oracle_twap_so_far: float
    twap_weight: float
    volatility_annualized: float


PricingInputsProvider = Callable[[int], PricingInputs | None]
DecisionCallback = Callable[[StrategyDecisionEvent], None]
ProcessedSignalChecker = Callable[[SignalIdentity], bool]


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
        "fee_bps",
        "pricing_inputs_provider_identity",
        "execution_history_limit",
        "execution_count",
        "stale_pricing_inputs",
        "dropped_window_mismatch",
        "last_decision",
        "decision_count",
        "decision_callback_errors",
        "_decision_callbacks",
        "_callback_registered",
        "_paper_session_owner_token",
        "_paper_processed_signal_checker",
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
        fee_bps: float = 0.0,
        pricing_inputs_provider_identity: str | None = None,
        execution_history_limit: int = DEFAULT_EXECUTION_HISTORY_LIMIT,
    ) -> None:
        bankroll = float(initial_bankroll)
        if bankroll <= 0.0 or not math.isfinite(bankroll):
            raise ValueError("initial_bankroll must be positive and finite")
        paper_fee_bps = float(fee_bps)
        if not math.isfinite(paper_fee_bps) or not 0.0 <= paper_fee_bps <= 10_000.0:
            raise ValueError("fee_bps must be finite and in [0, 10_000]")
        history_limit = int(execution_history_limit)
        if history_limit < 1:
            raise ValueError("execution_history_limit must be positive")
        provider_identity = (
            None
            if pricing_inputs_provider_identity is None
            else str(pricing_inputs_provider_identity).strip()
        )
        if pricing_inputs_provider_identity is not None and not provider_identity:
            raise ValueError("pricing_inputs_provider_identity must be non-empty")
        self.ofi_engine = ofi_engine
        self.pricing_engine = pricing_engine
        self.oms = oms
        self.feed_handler = feed_handler
        self.window = window
        self.current_bankroll = bankroll
        self.execution_history: deque[OrderResult] = deque(maxlen=history_limit)
        self.execution_history_limit = history_limit
        self.execution_count = 0
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
        self.fee_bps = paper_fee_bps
        self.pricing_inputs_provider_identity = provider_identity
        if self.reference_max_age_ms < 0 or self.ofi_max_age_ms < 0:
            raise ValueError("input freshness bounds must be non-negative")
        self.stale_pricing_inputs = 0
        self.dropped_window_mismatch = 0
        self.last_decision: StrategyDecisionEvent | None = None
        self.decision_count = 0
        self.decision_callback_errors = 0
        self._decision_callbacks: list[DecisionCallback] = []
        self._callback_registered = False
        self._paper_session_owner_token: object | None = None
        self._paper_processed_signal_checker: ProcessedSignalChecker | None = None
        self.oms.bankroll = bankroll

    def on_decision(self, callback: DecisionCallback) -> None:
        """Register a lightweight isolated callback for every decision event."""

        self._decision_callbacks.append(callback)

    @property
    def paper_session_bound(self) -> bool:
        """Return whether this runner is exclusively owned by a paper session."""

        return self._paper_session_owner_token is not None

    def bind_paper_session(
        self,
        *,
        owner_token: object,
        decision_callback: DecisionCallback,
        processed_signal_checker: ProcessedSignalChecker,
    ) -> None:
        """Exclusively bind one durable paper-session boundary to this runner."""

        if owner_token is None:
            raise ValueError("paper session owner_token must not be None")
        if not callable(decision_callback) or not callable(processed_signal_checker):
            raise TypeError("paper session callbacks must be callable")
        if self._paper_session_owner_token is not None:
            raise ValueError("StrategyRunner is already bound to a paper session")
        self._paper_session_owner_token = owner_token
        self._paper_processed_signal_checker = processed_signal_checker
        self._decision_callbacks.append(decision_callback)

    @property
    def last_execution(self) -> OrderResult | None:
        """Return the newest retained execution result, if any."""

        return self.execution_history[-1] if self.execution_history else None

    def config_identity(self) -> dict[str, object]:
        """Return the complete stable configuration that can affect decisions."""

        return {
            "window": {
                "window_id": self.window.window_id,
                "symbol": self.window.symbol,
                "strike_price": self.window.strike_price,
                "start_ts_ms": self.window.start_ts_ms,
                "end_ts_ms": self.window.end_ts_ms,
                "window_type": self.window.window_type,
            },
            "feed": self.feed_handler.config_identity(),
            "ofi": self.ofi_engine.config_identity(),
            "pricing": self.pricing_engine.config_identity(),
            "oms": self.oms.config_identity(),
            "runner": {
                "spot_price": self.spot_price,
                "volatility_annualized": self.volatility_annualized,
                "oracle_twap_so_far": self.oracle_twap_so_far,
                "twap_weight": self.twap_weight,
                "ofi_bid_qty": self.ofi_bid_qty,
                "ofi_ask_qty": self.ofi_ask_qty,
                "pricing_state_ts_ms": self.pricing_state_ts_ms,
                "reference_max_age_ms": self.reference_max_age_ms,
                "ofi_max_age_ms": self.ofi_max_age_ms,
                "fee_bps": self.fee_bps,
                "pricing_inputs_provider_present": self.pricing_inputs_provider is not None,
                "pricing_inputs_provider_identity": self.pricing_inputs_provider_identity,
                "execution_history_limit": self.execution_history_limit,
            },
        }

    def push_alpha_tick(self, book: TopOfBook) -> float:
        """Push one Binance top-of-book event into the alpha engine."""

        return self.ofi_engine.update_and_get_z(
            bid_price=book.bid_price,
            bid_qty=book.bid_qty,
            ask_price=book.ask_price,
            ask_qty=book.ask_qty,
            ts_ms=book.ts_ms,
        )

    def push_tick(self, tick: MarketSnapshot | TopOfBook) -> float:
        """Ingest a legacy Polymarket snapshot or a Binance alpha book.

        ``MarketSnapshot`` retains the pre-alpha-split compatibility contract
        and uses the configured synthetic OFI quantities. New integrations
        should call :meth:`push_alpha_tick` with ``TopOfBook`` explicitly.
        """

        if isinstance(tick, TopOfBook):
            return self.push_alpha_tick(tick)
        return self.ofi_engine.update_and_get_z(
            bid_price=tick.yes_bid,
            bid_qty=self.ofi_bid_qty,
            ask_price=tick.yes_ask,
            ask_qty=self.ofi_ask_qty,
            ts_ms=tick.timestamp_ms,
        )

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

        if self.paper_session_bound:
            raise RuntimeError(
                "paper-owned StrategyRunner must be started via PaperTradingSession"
            )
        if not self._callback_registered:
            self.feed_handler.on_snapshot(self._on_snapshot)
            self._callback_registered = True
        await self.feed_handler.connect()

    async def stop(self) -> None:
        """Close the feed handler without interrupting in-flight isolation."""

        await self.feed_handler.close()

    def process_snapshot_sync(self, snapshot: MarketSnapshot) -> OrderResult | None:
        """Run one snapshot through OFI → signal → OMS.

        A runner owned by ``PaperTradingSession`` rejects this public entry;
        the session must use its tokenized fail-closed path instead.
        """

        return self._process_snapshot_sync(snapshot, owner_token=None)

    def _process_paper_snapshot_sync(
        self,
        snapshot: MarketSnapshot,
        *,
        owner_token: object,
    ) -> OrderResult | None:
        """Process one paper snapshot only for the bound session owner."""

        return self._process_snapshot_sync(snapshot, owner_token=owner_token)

    def _process_snapshot_sync(
        self,
        snapshot: MarketSnapshot,
        *,
        owner_token: object | None,
    ) -> OrderResult | None:
        """Implement the shared snapshot decision path after access checks.

        ``HOLD`` never reaches the OMS. ``FILLED`` results debit
        ``current_bankroll``; every OMS ``OrderResult`` is appended to history.
        """

        self._require_processing_access(owner_token)
        cash_before = self.current_bankroll
        alpha_ts, alpha_age, alpha_fresh, alpha_reason, z_ofi = self._alpha_state(
            snapshot.timestamp_ms
        )
        if snapshot.window_id != self.window.window_id:
            self.dropped_window_mismatch += 1
            self._emit_decision(
                self._decision_event(
                    snapshot,
                    alpha_ts=alpha_ts,
                    alpha_age=alpha_age,
                    alpha_fresh=alpha_fresh,
                    alpha_reason=alpha_reason,
                    z_ofi=z_ofi,
                    inputs=None,
                    inputs_age=None,
                    inputs_fresh=False,
                    signal=None,
                    result=None,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    disposition=DecisionDisposition.DROPPED,
                    reason=DecisionReason.WINDOW_MISMATCH,
                )
            )
            return None
        inputs, inputs_age, inputs_fresh, input_reason = self._pricing_inputs_for(
            snapshot.timestamp_ms
        )
        if not inputs_fresh:
            self.stale_pricing_inputs += 1
            if input_reason is None:
                raise RuntimeError("unavailable pricing inputs require a reason")
            self._emit_decision(
                self._decision_event(
                    snapshot,
                    alpha_ts=alpha_ts,
                    alpha_age=alpha_age,
                    alpha_fresh=alpha_fresh,
                    alpha_reason=alpha_reason,
                    z_ofi=z_ofi,
                    inputs=inputs,
                    inputs_age=inputs_age,
                    inputs_fresh=False,
                    signal=None,
                    result=None,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    disposition=DecisionDisposition.DROPPED,
                    reason=input_reason,
                )
            )
            return None
        if inputs is None:
            raise RuntimeError("fresh pricing inputs cannot be missing")
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
            self._emit_decision(
                self._decision_event(
                    snapshot,
                    alpha_ts=alpha_ts,
                    alpha_age=alpha_age,
                    alpha_fresh=alpha_fresh,
                    alpha_reason=alpha_reason,
                    z_ofi=z_ofi,
                    inputs=inputs,
                    inputs_age=inputs_age,
                    inputs_fresh=True,
                    signal=signal,
                    result=None,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    disposition=DecisionDisposition.HOLD,
                    reason=DecisionReason.SIGNAL_HOLD,
                )
            )
            return None
        if (
            self._paper_processed_signal_checker is not None
            and self._paper_processed_signal_checker(signal_identity(signal))
        ):
            self._emit_decision(
                self._decision_event(
                    snapshot,
                    alpha_ts=alpha_ts,
                    alpha_age=alpha_age,
                    alpha_fresh=alpha_fresh,
                    alpha_reason=alpha_reason,
                    z_ofi=z_ofi,
                    inputs=inputs,
                    inputs_age=inputs_age,
                    inputs_fresh=True,
                    signal=signal,
                    result=None,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    disposition=DecisionDisposition.NO_ORDER,
                    reason=DecisionReason.DUPLICATE_SIGNAL,
                )
            )
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
            fee_bps=self.fee_bps,
        )
        if result is None:
            self._emit_decision(
                self._decision_event(
                    snapshot,
                    alpha_ts=alpha_ts,
                    alpha_age=alpha_age,
                    alpha_fresh=alpha_fresh,
                    alpha_reason=alpha_reason,
                    z_ofi=z_ofi,
                    inputs=inputs,
                    inputs_age=inputs_age,
                    inputs_fresh=True,
                    signal=signal,
                    result=None,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    disposition=DecisionDisposition.NO_ORDER,
                    reason=DecisionReason.OMS_NO_RESULT,
                )
            )
            return None
        if result.status == "FILLED":
            fee = result.shares * result.price * self.fee_bps / 10_000.0
            if not math.isclose(result.fee_usdc, fee, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("OMS fee differs from StrategyRunner fee")
            expected_cash = cash_before - result.shares * result.price - fee
            cash_after = self.oms.bankroll
            if not math.isclose(cash_after, expected_cash, rel_tol=1e-12, abs_tol=1e-9):
                raise RuntimeError("OMS cash differs from StrategyRunner fill accounting")
            self.current_bankroll = cash_after
            result = replace(result, fee_usdc=fee)
            disposition = DecisionDisposition.FILLED
            reason = DecisionReason.OMS_FILLED
        else:
            cash_after = cash_before
            disposition = DecisionDisposition.REJECTED
            reason = DecisionReason.OMS_REJECTED
        self.execution_history.append(result)
        self.execution_count += 1
        self._emit_decision(
            self._decision_event(
                snapshot,
                alpha_ts=alpha_ts,
                alpha_age=alpha_age,
                alpha_fresh=alpha_fresh,
                alpha_reason=alpha_reason,
                z_ofi=z_ofi,
                inputs=inputs,
                inputs_age=inputs_age,
                inputs_fresh=True,
                signal=signal,
                result=result,
                cash_before=cash_before,
                cash_after=cash_after,
                disposition=disposition,
                reason=reason,
            )
        )
        return result

    def _pricing_inputs_for(
        self,
        decision_ts_ms: int,
    ) -> tuple[PricingInputs | None, int | None, bool, DecisionReason | None]:
        if self.pricing_inputs_provider is not None:
            inputs = self.pricing_inputs_provider(decision_ts_ms)
            if inputs is None:
                return None, None, False, DecisionReason.PRICING_INPUTS_MISSING
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
            return inputs, age_ms, False, DecisionReason.PRICING_INPUTS_STALE
        return inputs, age_ms, True, None

    def _current_z_ofi(self, decision_ts_ms: int) -> float:
        return self._alpha_state(decision_ts_ms)[4]

    def _alpha_state(
        self,
        decision_ts_ms: int,
    ) -> tuple[int | None, int | None, bool, DecisionReason | None, float]:
        alpha_ts_ms = self.ofi_engine.last_timestamp_ms
        if alpha_ts_ms is None:
            return None, None, False, DecisionReason.ALPHA_MISSING, 0.0
        age_ms = decision_ts_ms - alpha_ts_ms
        if age_ms < 0 or age_ms > self.ofi_max_age_ms:
            return alpha_ts_ms, age_ms, False, DecisionReason.ALPHA_STALE, 0.0
        return alpha_ts_ms, age_ms, True, None, self.ofi_engine.get_normalized_ofi()

    def _decision_event(
        self,
        snapshot: MarketSnapshot,
        *,
        alpha_ts: int | None,
        alpha_age: int | None,
        alpha_fresh: bool,
        alpha_reason: DecisionReason | None,
        z_ofi: float,
        inputs: PricingInputs | None,
        inputs_age: int | None,
        inputs_fresh: bool,
        signal: PricingSignal | None,
        result: OrderResult | None,
        cash_before: float,
        cash_after: float,
        disposition: DecisionDisposition,
        reason: DecisionReason,
    ) -> StrategyDecisionEvent:
        return StrategyDecisionEvent(
            schema_version=STRATEGY_DECISION_SCHEMA_VERSION,
            timestamp_ms=snapshot.timestamp_ms,
            window_id=snapshot.window_id,
            market_symbol=self.window.symbol,
            window_start_ts_ms=self.window.start_ts_ms,
            window_end_ts_ms=self.window.end_ts_ms,
            yes_bid=snapshot.yes_bid,
            yes_ask=snapshot.yes_ask,
            yes_bid_size=snapshot.yes_bid_size,
            yes_ask_size=snapshot.yes_ask_size,
            no_bid=snapshot.no_bid,
            no_ask=snapshot.no_ask,
            no_bid_size=snapshot.no_bid_size,
            no_ask_size=snapshot.no_ask_size,
            last_traded_price=snapshot.last_traded_price,
            alpha_timestamp_ms=alpha_ts,
            alpha_age_ms=alpha_age,
            alpha_is_fresh=alpha_fresh,
            alpha_reason_code=alpha_reason,
            z_ofi=z_ofi,
            pricing_inputs_timestamp_ms=None if inputs is None else inputs.timestamp_ms,
            pricing_inputs_age_ms=inputs_age,
            pricing_inputs_are_fresh=inputs_fresh,
            spot_price=None if inputs is None else inputs.spot_price,
            oracle_twap_so_far=None if inputs is None else inputs.oracle_twap_so_far,
            twap_weight=None if inputs is None else inputs.twap_weight,
            volatility_annualized=None if inputs is None else inputs.volatility_annualized,
            model_probability=None if signal is None else signal.model_prob,
            market_price=None if signal is None else signal.market_price,
            effective_strike=(
                None
                if signal is None or not math.isfinite(signal.effective_strike)
                else signal.effective_strike
            ),
            edge=None if signal is None else signal.edge,
            ev=None if signal is None else signal.ev,
            direction=None if signal is None else signal.direction.value,
            recommended_size_pct=None if signal is None else signal.recommended_size_pct,
            order_id=None if result is None else result.order_id,
            order_status=None if result is None else result.status,
            order_side=None if result is None else result.side,
            shares=None if result is None else result.shares,
            fill_price=None if result is None else result.price,
            fee_usdc=None if result is None else result.fee_usdc,
            reject_reason=None if result is None else result.reject_reason,
            cash_before=cash_before,
            cash_after=cash_after,
            disposition=disposition,
            reason_code=reason,
        )

    def _emit_decision(self, event: StrategyDecisionEvent) -> None:
        self.last_decision = event
        self.decision_count += 1
        for callback in tuple(self._decision_callbacks):
            try:
                callback(event)
            except Exception:
                self.decision_callback_errors += 1
                logger.exception(
                    "strategy.decision_callback.failed window_id=%s ts_ms=%s",
                    event.window_id,
                    event.timestamp_ms,
                )

    async def process_snapshot(self, snapshot: MarketSnapshot) -> OrderResult | None:
        """Async wrapper around :meth:`process_snapshot_sync` for the live feed."""

        return self.process_snapshot_sync(snapshot)

    async def _process_paper_snapshot(
        self,
        snapshot: MarketSnapshot,
        *,
        owner_token: object,
    ) -> OrderResult | None:
        """Async wrapper for the tokenized paper-session processing path."""

        return self._process_paper_snapshot_sync(
            snapshot,
            owner_token=owner_token,
        )

    def _require_processing_access(self, owner_token: object | None) -> None:
        if self._paper_session_owner_token is None:
            if owner_token is not None:
                raise RuntimeError("unbound StrategyRunner rejects paper owner token")
            return
        if owner_token is None:
            raise RuntimeError(
                "paper-owned StrategyRunner must be called via PaperTradingSession"
            )
        self._require_paper_owner(owner_token)

    def _require_paper_owner(self, owner_token: object) -> None:
        if (
            self._paper_session_owner_token is None
            or owner_token is not self._paper_session_owner_token
        ):
            raise RuntimeError("invalid PaperTradingSession owner token")

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
