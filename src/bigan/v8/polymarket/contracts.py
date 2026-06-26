"""Polymarket BTC 15m binary-market contracts for v8."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from bigan.v8.phase0 import FeatureVector, Label, MarketData

POLYMARKET_ADAPTER_SCHEMA_VERSION = "bigan-v8-polymarket-adapter-v1"
POLYMARKET_BTC15M_MARKET_FAMILY = "btc_15m_up_down"
POLYMARKET_BTC15M_HORIZON_MS = 15 * 60 * 1000
POLYMARKET_SOURCE = "polymarket"

PolymarketOutcome = Literal["UP", "DOWN"]
PolymarketDecisionOutcome = Literal["UP", "DOWN", "NO_TRADE"]
PolymarketDecisionAction = Literal[
    "BUY_UP",
    "BUY_DOWN",
    "SELL_UP",
    "SELL_DOWN",
    "HOLD",
    "NO_TRADE",
]
PolymarketMarketStatus = Literal["open", "closed", "settled"]


class PolymarketAdapterError(RuntimeError):
    """Raised when Polymarket market mapping cannot fail closed safely."""


@dataclass(frozen=True, slots=True)
class PolymarketBinaryMarket:
    """Normalized Polymarket binary BTC 15m UP/DOWN market metadata."""

    market_id: str
    condition_id: str
    slug: str
    title: str
    market_family: str
    base_asset: str
    quote_asset: str
    outcome_up: str
    outcome_down: str
    up_token_id: str
    down_token_id: str
    market_start_ts: int
    market_end_ts: int
    settlement_ts: int
    horizon_ms: int
    reference_price_source: str
    reference_price_at_start: float
    settlement_rule: str
    status: PolymarketMarketStatus
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
            title=self.title,
            outcome_up=self.outcome_up,
            outcome_down=self.outcome_down,
            up_token_id=self.up_token_id,
            down_token_id=self.down_token_id,
            reference_price_source=self.reference_price_source,
            settlement_rule=self.settlement_rule,
        )
        if self.market_family != POLYMARKET_BTC15M_MARKET_FAMILY:
            raise ValueError("market_family must be btc_15m_up_down")
        if self.base_asset != "BTC" or self.quote_asset != "USD":
            raise ValueError("base_asset/quote_asset must be BTC/USD")
        if self.up_token_id == self.down_token_id:
            raise ValueError("UP and DOWN token ids must differ")
        if self.market_start_ts < 0 or self.market_end_ts <= self.market_start_ts:
            raise ValueError("market window must be positive and ordered")
        if self.horizon_ms != POLYMARKET_BTC15M_HORIZON_MS:
            raise ValueError("horizon_ms must be 15 minutes")
        if self.market_end_ts - self.market_start_ts != self.horizon_ms:
            raise ValueError("market window must match horizon_ms")
        if self.settlement_ts < self.market_end_ts:
            raise ValueError("settlement_ts cannot precede market_end_ts")
        if self.reference_price_at_start <= 0.0 or not math.isfinite(
            self.reference_price_at_start
        ):
            raise ValueError("reference_price_at_start must be positive and finite")
        if self.status not in ("open", "closed", "settled"):
            raise ValueError("status must be open, closed, or settled")
        _validate_safety_boundary(self)

    def token_id_for_outcome(self, outcome: PolymarketOutcome) -> str:
        return self.up_token_id if outcome == "UP" else self.down_token_id

    def opposite_token_id_for_outcome(self, outcome: PolymarketOutcome) -> str:
        return self.down_token_id if outcome == "UP" else self.up_token_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketTokenSnapshot:
    """Read-only Polymarket token price/book snapshot."""

    market_id: str
    token_id: str
    outcome: PolymarketOutcome
    ts: int
    bid_price: float
    ask_price: float
    mid_price: float
    last_price: float
    spread_bps: float
    volume: float
    liquidity_depth: float
    trade_count: int
    source: str = POLYMARKET_SOURCE
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            token_id=self.token_id,
            outcome=self.outcome,
            source=self.source,
        )
        if self.outcome not in ("UP", "DOWN"):
            raise ValueError("outcome must be UP or DOWN")
        for field_name in (
            "bid_price",
            "ask_price",
            "mid_price",
            "last_price",
            "spread_bps",
            "volume",
            "liquidity_depth",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.ts < 0:
            raise ValueError("ts must be non-negative")
        if self.bid_price <= 0.0 or self.ask_price <= 0.0:
            raise ValueError("bid_price and ask_price must be positive")
        if self.ask_price < self.bid_price:
            raise ValueError("ask_price cannot be below bid_price")
        if self.mid_price <= 0.0 or self.last_price <= 0.0:
            raise ValueError("mid_price and last_price must be positive")
        if self.spread_bps < 0.0:
            raise ValueError("spread_bps must be non-negative")
        if self.volume < 0.0 or self.liquidity_depth < 0.0:
            raise ValueError("volume and liquidity_depth must be non-negative")
        if self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        _validate_safety_boundary(self)
        if self.read_only is not True or self.write_capable is not False:
            raise ValueError("token snapshots must be read-only and not write-capable")

    def to_market_data(self, *, instrument_id: str | None = None) -> MarketData:
        return MarketData(
            ts=self.ts,
            available_at_ts=self.ts,
            source=self.source,
            instrument_id=instrument_id or f"{self.market_id}:{self.outcome}",
            bid_price=self.bid_price,
            ask_price=self.ask_price,
            mid_price=self.mid_price,
            last_price=self.last_price,
            volume=self.volume,
            trade_count=self.trade_count,
            liquidity_depth=self.liquidity_depth,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketFeatureRow:
    """Polymarket-specific causal feature row plus v8 FeatureVector evidence."""

    market_id: str
    condition_id: str
    slug: str
    decision_ts: int
    feature_cutoff_ts: int
    max_input_ts: int
    horizon_ms: int
    features: dict[str, float | int | None]
    v8_feature: FeatureVector
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
        )
        if self.horizon_ms != POLYMARKET_BTC15M_HORIZON_MS:
            raise ValueError("horizon_ms must be 15 minutes")
        if self.feature_cutoff_ts > self.decision_ts:
            raise ValueError("feature_cutoff_ts cannot exceed decision_ts")
        if self.max_input_ts > self.decision_ts:
            raise ValueError("max_input_ts cannot exceed decision_ts")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["v8_feature"] = self.v8_feature.model_dump()
        return payload


@dataclass(frozen=True, slots=True)
class PolymarketLabelRow:
    """Explicit BTC 15m UP/DOWN settlement label and cost-aware return."""

    market_id: str
    condition_id: str
    slug: str
    outcome: PolymarketOutcome
    reference_price_start: float
    reference_price_end: float
    market_start_ts: int
    market_end_ts: int
    horizon_ms: int
    settlement_rule: str
    raw_settlement_metadata_hash: str
    is_up: bool
    is_down: bool
    is_positive: bool
    entry_token_price: float
    exit_token_price: float
    gross_return: float
    spread_cost: float
    fee_cost: float
    slippage_cost: float
    liquidity_impact_cost: float
    total_cost: float
    net_return: float
    v8_label: Label
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
            outcome=self.outcome,
            settlement_rule=self.settlement_rule,
            raw_settlement_metadata_hash=self.raw_settlement_metadata_hash,
        )
        if self.outcome not in ("UP", "DOWN"):
            raise ValueError("outcome must be UP or DOWN")
        if self.horizon_ms != POLYMARKET_BTC15M_HORIZON_MS:
            raise ValueError("horizon_ms must be 15 minutes")
        if not looks_like_sha256(self.raw_settlement_metadata_hash):
            raise ValueError("raw_settlement_metadata_hash must be SHA-256")
        if self.market_end_ts < self.market_start_ts + self.horizon_ms:
            raise ValueError("market_end_ts must honor horizon_ms")
        if self.reference_price_start <= 0.0 or self.reference_price_end <= 0.0:
            raise ValueError("reference prices must be positive")
        if self.entry_token_price <= 0.0 or self.exit_token_price < 0.0:
            raise ValueError("entry/exit token prices must be valid")
        if self.is_up != (self.outcome == "UP"):
            raise ValueError("is_up must match outcome")
        if self.is_down != (self.outcome == "DOWN"):
            raise ValueError("is_down must match outcome")
        expected_cost = (
            self.spread_cost
            + self.fee_cost
            + self.slippage_cost
            + self.liquidity_impact_cost
        )
        if abs(self.total_cost - expected_cost) > 1e-12:
            raise ValueError("total_cost must equal component costs")
        if abs(self.net_return - (self.gross_return - self.total_cost)) > 1e-12:
            raise ValueError("net_return must equal gross_return - total_cost")
        if self.is_positive != (self.net_return > 0.0):
            raise ValueError("is_positive must equal net_return > 0")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["v8_label"] = self.v8_label.model_dump()
        return payload


@dataclass(frozen=True, slots=True)
class PolymarketBinaryDecision:
    """Paper-only Polymarket binary decision derived from v8 policy output."""

    decision_ts: int
    market_id: str
    condition_id: str
    slug: str
    selected_outcome: PolymarketDecisionOutcome
    selected_token_id: str | None
    opposite_token_id: str | None
    v8_action: float
    v8_confidence: float
    v8_score: float
    estimated_probability: float | None
    token_mid_price: float | None
    edge: float
    max_paper_size: float
    paper_notional: float
    reason_codes: tuple[str, ...]
    paper_action: PolymarketDecisionAction = "NO_TRADE"
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
        )
        if self.selected_outcome not in ("UP", "DOWN", "NO_TRADE"):
            raise ValueError("selected_outcome must be UP, DOWN, or NO_TRADE")
        if self.paper_action not in (
            "BUY_UP",
            "BUY_DOWN",
            "SELL_UP",
            "SELL_DOWN",
            "HOLD",
            "NO_TRADE",
        ):
            raise ValueError("unsupported paper_action")
        if self.paper_action == "NO_TRADE" and self.selected_outcome != "NO_TRADE":
            raise ValueError("NO_TRADE action must use NO_TRADE selected_outcome")
        if self.paper_action == "HOLD" and self.selected_outcome != "NO_TRADE":
            raise ValueError("HOLD action must use NO_TRADE selected_outcome")
        if self.paper_action in {"BUY_UP", "SELL_UP"} and self.selected_outcome != "UP":
            raise ValueError("UP actions require selected_outcome=UP")
        if (
            self.paper_action in {"BUY_DOWN", "SELL_DOWN"}
            and self.selected_outcome != "DOWN"
        ):
            raise ValueError("DOWN actions require selected_outcome=DOWN")
        if self.selected_outcome != "NO_TRADE" and not self.selected_token_id:
            raise ValueError("selected token is required for trade decisions")
        if self.paper_action.startswith("SELL_") and self.paper_notional <= 0.0:
            raise ValueError("SELL actions require positive paper_notional")
        if not 0.0 <= self.v8_action <= 1.0:
            raise ValueError("v8_action must be in [0, 1]")
        if not 0.0 <= self.v8_confidence <= 1.0:
            raise ValueError("v8_confidence must be in [0, 1]")
        if not math.isfinite(self.v8_score):
            raise ValueError("v8_score must be finite")
        if self.estimated_probability is not None and not (
            0.0 <= self.estimated_probability <= 1.0
        ):
            raise ValueError("estimated_probability must be in [0, 1]")
        if self.token_mid_price is not None and self.token_mid_price <= 0.0:
            raise ValueError("token_mid_price must be positive")
        if not math.isfinite(self.edge):
            raise ValueError("edge must be finite")
        if self.max_paper_size < 0.0 or self.paper_notional < 0.0:
            raise ValueError("paper sizing fields must be non-negative")
        if self.selected_outcome == "NO_TRADE" and not self.reason_codes:
            raise ValueError("NO_TRADE decisions require reason_codes")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def looks_like_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _require_non_empty(**values: str) -> None:
    for field_name, value in values.items():
        if not str(value).strip():
            raise ValueError(f"{field_name} is required")


def _validate_safety_boundary(payload: Any) -> None:
    checks = {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    for field_name, expected in checks.items():
        if getattr(payload, field_name) is not expected:
            raise ValueError(f"{field_name} must be {str(expected).lower()}")
