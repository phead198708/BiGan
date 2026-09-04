"""Read-only, constant-size observations for inspecting the actual feed inputs."""

from __future__ import annotations

from .pricing_inputs import ReferencePriceSample


def reference_observation(
    sample: ReferencePriceSample | None,
    *,
    source: str,
    symbol: str,
    kind: str,
    currency: str,
    connected: bool,
    now_ms: int,
    max_age_ms: int,
) -> dict[str, object]:
    age = None if sample is None else now_ms - sample.timestamp_ms
    return {
        "value": None if sample is None else sample.price,
        "source": source if sample is None else sample.source,
        "symbol": symbol,
        "kind": kind,
        "quote_currency": currency,
        "timestamp_ms": None if sample is None else sample.timestamp_ms,
        "received_at_ms": None if sample is None else sample.received_at_ms,
        "age_ms": age,
        "max_age_ms": max_age_ms,
        "connected": connected,
        "fresh": bool(connected and age is not None and 0 <= age <= max_age_ms),
    }
