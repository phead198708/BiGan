"""Serializable decision events emitted by the live strategy pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum, StrEnum
from typing import Any

STRATEGY_DECISION_SCHEMA_VERSION = "1.0"


class DecisionDisposition(StrEnum):
    """Stable outcome classification for one strategy decision."""

    DROPPED = "DROPPED"
    HOLD = "HOLD"
    NO_ORDER = "NO_ORDER"
    FILLED = "FILLED"
    REJECTED = "REJECTED"


class DecisionReason(StrEnum):
    """Machine-readable reason codes carried by decision events."""

    WINDOW_MISMATCH = "window_mismatch"
    PRICING_INPUTS_MISSING = "pricing_inputs_missing"
    PRICING_INPUTS_STALE = "pricing_inputs_stale"
    ALPHA_MISSING = "alpha_missing"
    ALPHA_STALE = "alpha_stale"
    SIGNAL_HOLD = "signal_hold"
    OMS_NO_RESULT = "oms_no_result"
    OMS_FILLED = "oms_filled"
    OMS_REJECTED = "oms_rejected"


@dataclass(frozen=True, slots=True)
class StrategyDecisionEvent:
    """Complete immutable audit record for one ``StrategyRunner`` snapshot."""

    schema_version: str
    timestamp_ms: int
    window_id: str
    market_symbol: str
    window_start_ts_ms: int
    window_end_ts_ms: int
    yes_bid: float
    yes_ask: float
    yes_bid_size: float
    yes_ask_size: float
    no_bid: float
    no_ask: float
    no_bid_size: float
    no_ask_size: float
    last_traded_price: float
    alpha_timestamp_ms: int | None
    alpha_age_ms: int | None
    alpha_is_fresh: bool
    alpha_reason_code: DecisionReason | None
    z_ofi: float
    pricing_inputs_timestamp_ms: int | None
    pricing_inputs_age_ms: int | None
    pricing_inputs_are_fresh: bool
    spot_price: float | None
    oracle_twap_so_far: float | None
    twap_weight: float | None
    volatility_annualized: float | None
    model_probability: float | None
    market_price: float | None
    effective_strike: float | None
    edge: float | None
    ev: float | None
    direction: str | None
    recommended_size_pct: float | None
    order_id: str | None
    order_status: str | None
    order_side: str | None
    shares: float | None
    fill_price: float | None
    fee_usdc: float | None
    reject_reason: str | None
    cash_before: float
    cash_after: float
    disposition: DecisionDisposition
    reason_code: DecisionReason

    def __post_init__(self) -> None:
        if self.schema_version != STRATEGY_DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported strategy decision schema_version")
        if not self.window_id or not self.market_symbol:
            raise ValueError("window_id and market_symbol must be non-empty")
        if self.window_end_ts_ms <= self.window_start_ts_ms:
            raise ValueError("window end must be after window start")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        for name in ("yes_bid", "yes_ask", "no_bid", "no_ask", "last_traded_price"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("yes_bid_size", "yes_ask_size", "no_bid_size", "no_ask_size"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.alpha_timestamp_ms is None:
            if self.alpha_age_ms is not None or self.alpha_is_fresh:
                raise ValueError("missing alpha cannot have age or be fresh")
            if self.alpha_reason_code is not DecisionReason.ALPHA_MISSING:
                raise ValueError("missing alpha requires alpha_missing reason")
            if self.z_ofi != 0.0:
                raise ValueError("missing alpha requires zero z_ofi")
        elif not self.alpha_is_fresh:
            if self.alpha_reason_code is not DecisionReason.ALPHA_STALE:
                raise ValueError("stale alpha requires alpha_stale reason")
            if self.z_ofi != 0.0:
                raise ValueError("stale alpha requires zero z_ofi")
        elif self.alpha_reason_code is not None:
            raise ValueError("fresh alpha cannot have an alpha reason")
        if not self.pricing_inputs_are_fresh and self.direction is not None:
            raise ValueError("unavailable pricing inputs cannot produce a signal")
        pricing_values = (
            self.pricing_inputs_timestamp_ms,
            self.pricing_inputs_age_ms,
            self.spot_price,
            self.oracle_twap_so_far,
            self.twap_weight,
            self.volatility_annualized,
        )
        if self.pricing_inputs_are_fresh and any(value is None for value in pricing_values):
            raise ValueError("fresh pricing inputs must be complete")
        if self.direction not in {None, "BUY_YES", "BUY_NO", "HOLD"}:
            raise ValueError("unsupported signal direction")
        if self.cash_before < 0.0 or self.cash_after < 0.0:
            raise ValueError("decision cash cannot be negative")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-native mapping with stable schema field names."""

        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            payload[field.name] = value.value if isinstance(value, Enum) else value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategyDecisionEvent:
        """Parse a persisted event and re-run all constructor validation."""

        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError("strategy decision fields do not match schema")
        values = dict(payload)
        values["disposition"] = DecisionDisposition(str(values["disposition"]))
        values["reason_code"] = DecisionReason(str(values["reason_code"]))
        alpha_reason = values["alpha_reason_code"]
        values["alpha_reason_code"] = (
            None if alpha_reason is None else DecisionReason(str(alpha_reason))
        )
        return cls(**values)
