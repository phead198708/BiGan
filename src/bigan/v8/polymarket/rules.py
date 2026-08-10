"""Polymarket resolution rules for paper-only binary outcome markets."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from bigan.v8.polymarket.contracts import (
    POLYMARKET_BTC15M_HORIZON_MS,
    POLYMARKET_BTC15M_MARKET_FAMILY,
    PolymarketAdapterError,
    PolymarketBinaryMarket,
    canonical_json_sha256,
    looks_like_sha256,
)

PolymarketComparator = Literal["close_gt_open", "close_gte_open"]
PolymarketTieBreaker = Literal["up", "down", "unknown"]
PolymarketResolvedOutcome = Literal["UP", "DOWN", "UNKNOWN_50_50"]
PolymarketResolutionStatus = Literal["normal", "unknown_50_50"]


@dataclass(frozen=True, slots=True)
class PolymarketResolutionRule:
    """Normalized paper-only rule for resolving a binary Polymarket market."""

    market_id: str
    condition_id: str
    slug: str
    market_family: str
    resolution_source: str
    candle_interval_ms: int
    candle_open_ts: int
    candle_close_ts: int
    open_price_field: str
    close_price_field: str
    comparator: PolymarketComparator
    tie_breaker: PolymarketTieBreaker
    unknown_50_50_enabled: bool
    raw_rule_text: str
    raw_rule_sha256: str
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
            market_family=self.market_family,
            resolution_source=self.resolution_source,
            open_price_field=self.open_price_field,
            close_price_field=self.close_price_field,
            raw_rule_text=self.raw_rule_text,
            raw_rule_sha256=self.raw_rule_sha256,
        )
        if self.candle_interval_ms <= 0:
            raise ValueError("candle_interval_ms must be positive")
        if self.candle_close_ts <= self.candle_open_ts:
            raise ValueError("candle_close_ts must be after candle_open_ts")
        if self.candle_close_ts - self.candle_open_ts != self.candle_interval_ms:
            raise ValueError("candle timestamps must match candle_interval_ms")
        if self.comparator not in ("close_gt_open", "close_gte_open"):
            raise ValueError("unsupported comparator")
        if self.tie_breaker not in ("up", "down", "unknown"):
            raise ValueError("unsupported tie_breaker")
        if self.tie_breaker == "unknown" and not self.unknown_50_50_enabled:
            raise ValueError("unknown tie_breaker requires unknown_50_50_enabled")
        if not looks_like_sha256(self.raw_rule_sha256):
            raise ValueError("raw_rule_sha256 must be SHA-256")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketRuleResolution:
    """Resolved outcome and payout vector for one market window."""

    market_id: str
    condition_id: str
    slug: str
    resolved_outcome: PolymarketResolvedOutcome
    payout_up: float
    payout_down: float
    reference_price_start: float
    reference_price_end: float
    resolution_status: PolymarketResolutionStatus
    comparator: PolymarketComparator
    tie_breaker: PolymarketTieBreaker
    raw_resolution_sha256: str
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
            resolution_status=self.resolution_status,
            resolved_outcome=self.resolved_outcome,
            raw_resolution_sha256=self.raw_resolution_sha256,
        )
        if self.resolved_outcome not in ("UP", "DOWN", "UNKNOWN_50_50"):
            raise ValueError("unsupported resolved_outcome")
        if self.resolution_status not in ("normal", "unknown_50_50"):
            raise ValueError("unsupported resolution_status")
        if (
            self.resolved_outcome == "UNKNOWN_50_50"
            and self.resolution_status != "unknown_50_50"
        ):
            raise ValueError("UNKNOWN_50_50 requires explicit resolution status")
        if not 0.0 <= self.payout_up <= 1.0:
            raise ValueError("payout_up must be in [0, 1]")
        if not 0.0 <= self.payout_down <= 1.0:
            raise ValueError("payout_down must be in [0, 1]")
        if self.resolved_outcome == "UP" and (self.payout_up, self.payout_down) != (
            1.0,
            0.0,
        ):
            raise ValueError("UP resolution must pay UP=1 and DOWN=0")
        if self.resolved_outcome == "DOWN" and (self.payout_up, self.payout_down) != (
            0.0,
            1.0,
        ):
            raise ValueError("DOWN resolution must pay UP=0 and DOWN=1")
        if self.resolved_outcome == "UNKNOWN_50_50" and (
            self.payout_up,
            self.payout_down,
        ) != (0.5, 0.5):
            raise ValueError("UNKNOWN_50_50 resolution must pay both outcomes 0.5")
        if self.reference_price_start <= 0.0 or self.reference_price_end <= 0.0:
            raise ValueError("reference prices must be positive")
        if not looks_like_sha256(self.raw_resolution_sha256):
            raise ValueError("raw_resolution_sha256 must be SHA-256")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_btc15m_resolution_rule(
    market: PolymarketBinaryMarket,
    *,
    raw_rule_text: str | None = None,
    comparator: PolymarketComparator | None = None,
    tie_breaker: PolymarketTieBreaker | None = None,
    unknown_50_50_enabled: bool | None = None,
) -> PolymarketResolutionRule:
    """Build the normalized BTC 15m resolution rule for a paper market."""

    return build_btc_updown_resolution_rule(
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        market_family=market.market_family,
        resolution_source=market.reference_price_source,
        candle_open_ts=market.market_start_ts,
        candle_close_ts=market.market_end_ts,
        raw_rule_text=raw_rule_text or market.settlement_rule,
        comparator=comparator,
        tie_breaker=tie_breaker,
        unknown_50_50_enabled=unknown_50_50_enabled,
    )


def build_btc_updown_resolution_rule(
    *,
    market_id: str,
    condition_id: str,
    slug: str,
    market_family: str,
    resolution_source: str,
    candle_open_ts: int,
    candle_close_ts: int,
    raw_rule_text: str,
    comparator: PolymarketComparator | None = None,
    tie_breaker: PolymarketTieBreaker | None = None,
    unknown_50_50_enabled: bool | None = None,
) -> PolymarketResolutionRule:
    """Build a normalized BTC UP/DOWN resolution rule for any supported horizon."""

    if candle_close_ts <= candle_open_ts:
        raise ValueError("candle_close_ts must be after candle_open_ts")
    text = raw_rule_text
    resolved_comparator = comparator or _infer_comparator(text)
    resolved_tie_breaker = tie_breaker or _infer_tie_breaker(
        text=text,
        comparator=resolved_comparator,
    )
    unknown_enabled = (
        resolved_tie_breaker == "unknown"
        if unknown_50_50_enabled is None
        else unknown_50_50_enabled
    )
    return PolymarketResolutionRule(
        market_id=market_id,
        condition_id=condition_id,
        slug=slug,
        market_family=market_family,
        resolution_source=resolution_source,
        candle_interval_ms=candle_close_ts - candle_open_ts,
        candle_open_ts=candle_open_ts,
        candle_close_ts=candle_close_ts,
        open_price_field="reference_price_at_start",
        close_price_field="reference_price_at_end",
        comparator=resolved_comparator,
        tie_breaker=resolved_tie_breaker,
        unknown_50_50_enabled=unknown_enabled,
        raw_rule_text=text,
        raw_rule_sha256=canonical_json_sha256(
            {
                "market_id": market_id,
                "condition_id": condition_id,
                "raw_rule_text": text,
            }
        ),
    )


def resolve_polymarket_rule(
    rule: PolymarketResolutionRule,
    *,
    reference_price_start: float,
    reference_price_end: float,
    resolution_status: PolymarketResolutionStatus = "normal",
) -> PolymarketRuleResolution:
    """Resolve a normalized rule into payout semantics."""

    for field_name, value in (
        ("reference_price_start", reference_price_start),
        ("reference_price_end", reference_price_end),
    ):
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{field_name} must be positive and finite")
    if resolution_status not in ("normal", "unknown_50_50"):
        raise ValueError("unsupported resolution_status")
    resolved_outcome = _resolved_outcome(
        rule=rule,
        reference_price_start=reference_price_start,
        reference_price_end=reference_price_end,
        resolution_status=resolution_status,
    )
    payout_up, payout_down = payout_for_resolved_outcome(resolved_outcome)
    return PolymarketRuleResolution(
        market_id=rule.market_id,
        condition_id=rule.condition_id,
        slug=rule.slug,
        resolved_outcome=resolved_outcome,
        payout_up=payout_up,
        payout_down=payout_down,
        reference_price_start=reference_price_start,
        reference_price_end=reference_price_end,
        resolution_status=resolution_status,
        comparator=rule.comparator,
        tie_breaker=rule.tie_breaker,
        raw_resolution_sha256=canonical_json_sha256(
            {
                "market_id": rule.market_id,
                "condition_id": rule.condition_id,
                "reference_price_start": reference_price_start,
                "reference_price_end": reference_price_end,
                "resolution_status": resolution_status,
                "resolved_outcome": resolved_outcome,
                "raw_rule_sha256": rule.raw_rule_sha256,
            }
        ),
    )


def payout_for_resolved_outcome(
    resolved_outcome: PolymarketResolvedOutcome,
) -> tuple[float, float]:
    if resolved_outcome == "UP":
        return 1.0, 0.0
    if resolved_outcome == "DOWN":
        return 0.0, 1.0
    if resolved_outcome == "UNKNOWN_50_50":
        return 0.5, 0.5
    raise PolymarketAdapterError("unsupported_resolved_outcome")


def _resolved_outcome(
    *,
    rule: PolymarketResolutionRule,
    reference_price_start: float,
    reference_price_end: float,
    resolution_status: PolymarketResolutionStatus,
) -> PolymarketResolvedOutcome:
    if resolution_status == "unknown_50_50":
        if not rule.unknown_50_50_enabled:
            raise PolymarketAdapterError("unknown_50_50_disabled")
        return "UNKNOWN_50_50"
    if reference_price_end > reference_price_start:
        return "UP"
    if reference_price_end < reference_price_start:
        return "DOWN"
    if rule.comparator == "close_gte_open":
        return "UP"
    return "DOWN"


def _infer_comparator(raw_rule_text: str) -> PolymarketComparator:
    normalized = " ".join(raw_rule_text.lower().replace("_", " ").split())
    tokens = set(normalized.split())
    if (
        ">=" in normalized
        or "greater than or equal" in normalized
        or "at least" in normalized
        or "gte" in tokens
    ):
        return "close_gte_open"
    if (
        "greater than" in normalized
        or ">" in normalized
        or "higher" in normalized
        or "gt start" in normalized
        or "gt" in tokens
    ):
        return "close_gt_open"
    raise PolymarketAdapterError("unknown_resolution_comparator")


def _infer_tie_breaker(
    *,
    text: str,
    comparator: PolymarketComparator,
) -> PolymarketTieBreaker:
    normalized = " ".join(text.lower().replace("_", " ").split())
    if (
        "50/50" in normalized
        or "50-50" in normalized
        or "50 50" in normalized
        or "unknown" in normalized
    ):
        return "unknown"
    if "tie up" in normalized or comparator == "close_gte_open":
        return "up"
    return "down"


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


def default_btc15m_rule_text() -> str:
    return (
        "UP wins if the BTC reference close price is greater than the open "
        "price; otherwise DOWN wins."
    )


def btc15m_rule_payload() -> dict[str, Any]:
    return {
        "market_family": POLYMARKET_BTC15M_MARKET_FAMILY,
        "candle_interval_ms": POLYMARKET_BTC15M_HORIZON_MS,
        "default_rule_text": default_btc15m_rule_text(),
    }
