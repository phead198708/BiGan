"""Strict official RTDS TWAP wire format; never reconstruct a Chainlink TWAP."""

from __future__ import annotations

import math
import re
from decimal import Decimal, localcontext

from .pricing_inputs import ReferencePriceSample

TWAP_TOPICS = {30: "crypto_prices_twap_thirty", 60: "crypto_prices_twap_sixty"}


def oracle_source(symbol: str, lookback_seconds: int | None) -> str:
    if lookback_seconds is None:
        return f"polymarket_rtds_chainlink:{symbol}"
    return f"polymarket_rtds_chainlink_twap:{symbol}:{lookback_seconds}s"


def parse_twap_sample(
    message: object, *, symbol: str, lookback_seconds: int, received_at_ms: int,
) -> ReferencePriceSample | None:
    if lookback_seconds not in TWAP_TOPICS:
        raise ValueError("unsupported Chainlink TWAP lookback")
    if not isinstance(message, dict):
        raise ValueError("invalid Chainlink TWAP event")
    if message.get("topic") != TWAP_TOPICS[lookback_seconds] or message.get("type") != "update":
        return None
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid Chainlink TWAP payload")
    if payload.get("symbol") != symbol:
        return None
    source_ts, published_ts = payload.get("timestamp"), message.get("timestamp")
    if (type(payload.get("window_s")) is not int or payload["window_s"] != lookback_seconds
            or type(source_ts) is not int or type(published_ts) is not int
            or not 0 < source_ts <= published_ts <= received_at_ms):
        raise ValueError("invalid Chainlink TWAP identity or event time")
    exact = payload.get("full_accuracy_value")
    if not isinstance(exact, str) or not re.fullmatch(r"[0-9]{1,78}", exact):
        raise ValueError("Chainlink TWAP requires its exact E18 value")
    with localcontext() as context:
        context.prec = 100
        price = float(Decimal(exact) / Decimal(10**18))
    if not math.isfinite(price) or price <= 0:
        raise ValueError("invalid Chainlink TWAP price")
    return ReferencePriceSample(source_ts, received_at_ms, price, oracle_source(symbol, lookback_seconds))
