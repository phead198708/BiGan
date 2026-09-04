"""Window-bound opening TWAP from the public Polymarket price display service.

This is the website's read-only JSON endpoint, not a versioned Gamma API or a
signed Chainlink report. Persist its request identity and response digest; never
substitute closePrice, current spot, or an independently reconstructed TWAP.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .discovery import DiscoveredMarket
    from .transports import PublicJSONClient

OPENING_REFERENCE_ENDPOINT = "https://polymarket.com/api/crypto/crypto-price"


@dataclass(frozen=True, slots=True)
class OpeningReferenceProof:
    source_endpoint: str
    market_id: str
    condition_id: str
    symbol: str
    window_start_ts_ms: int
    window_end_ts_ms: int
    lookback_seconds: int
    price: float
    requested_at_ms: int
    received_at_ms: int
    source_ts_ms: int
    payload_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1 or self.source_endpoint != OPENING_REFERENCE_ENDPOINT:
            raise ValueError("unsupported opening reference source")
        for value in (self.window_start_ts_ms, self.window_end_ts_ms, self.lookback_seconds,
                      self.requested_at_ms, self.received_at_ms, self.source_ts_ms):
            if type(value) is not int or value < 0:
                raise ValueError("invalid opening reference timestamp or lookback")
        if (self.symbol not in {"BTC", "ETH"}
                or any(not isinstance(value, str) or not value.strip() for value in (self.market_id, self.condition_id))
                or self.lookback_seconds not in {30, 60}
                or self.window_end_ts_ms - self.window_start_ts_ms not in {300_000, 900_000}
                or not self.window_start_ts_ms <= self.requested_at_ms <= self.received_at_ms < self.window_end_ts_ms
                or not self.window_start_ts_ms <= self.source_ts_ms <= self.received_at_ms
                or isinstance(self.price, bool) or not math.isfinite(self.price) or self.price <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", self.payload_sha256)):
            raise ValueError("invalid opening reference identity or value")


def opening_reference_params(market: DiscoveredMarket) -> dict[str, str]:
    if market.oracle_twap_lookback_seconds not in {30, 60}:
        raise ValueError("opening reference requires explicit Chainlink TWAP identity")
    return {
        "symbol": market.underlying,
        "eventStartTime": _iso(market.start_ts_ms),
        "variant": {300_000: "fiveminute", 900_000: "fifteen"}[market.window_duration_ms],
        "endDate": _iso(market.end_ts_ms),
        "twapEnabled": "true",
        "twapLookbackSeconds": str(market.oracle_twap_lookback_seconds),
    }


def bind_opening_reference(
    market: DiscoveredMarket, payload: object, *, requested_at_ms: int, received_at_ms: int,
) -> DiscoveredMarket:
    opening_reference_params(market)  # Reject ambiguous/legacy source contracts.
    assert market.oracle_twap_lookback_seconds is not None
    if not isinstance(payload, dict) or set(payload) != {
        "openPrice", "closePrice", "timestamp", "completed", "incomplete", "cached",
    }:
        raise ValueError("invalid opening reference response schema")
    if (any(type(payload[key]) is not bool for key in ("completed", "incomplete", "cached"))
            or payload["completed"] or type(payload["openPrice"]) not in (int, float)):
        raise ValueError("opening reference is unavailable or not an active window")
    # incomplete=True is normal for an open window: it has no closing value yet.
    # closePrice is neither a strike nor a settlement signal.
    price = float(payload["openPrice"])
    proof = OpeningReferenceProof(
        source_endpoint=OPENING_REFERENCE_ENDPOINT, market_id=market.market_id,
        condition_id=market.condition_id, symbol=market.underlying,
        window_start_ts_ms=market.start_ts_ms, window_end_ts_ms=market.end_ts_ms,
        lookback_seconds=market.oracle_twap_lookback_seconds,
        price=price, requested_at_ms=requested_at_ms, received_at_ms=received_at_ms,
        source_ts_ms=payload["timestamp"],
        payload_sha256=hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
    )
    if market.reference_price_at_start is not None and market.reference_price_at_start != price:
        raise ValueError("Gamma and opening reference prices conflict")
    return replace(market, reference_price_at_start=price, opening_reference=proof)


async def fetch_opening_reference(
    market: DiscoveredMarket, *, http: PublicJSONClient, endpoint: str = OPENING_REFERENCE_ENDPOINT,
) -> DiscoveredMarket:
    if endpoint != OPENING_REFERENCE_ENDPOINT:
        raise ValueError("unapproved opening reference endpoint")
    requested = time.time_ns() // 1_000_000
    payload = await http.get_json(endpoint, params=opening_reference_params(market))
    return bind_opening_reference(
        market, payload, requested_at_ms=requested, received_at_ms=time.time_ns() // 1_000_000,
    )


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat().replace("+00:00", "Z")
