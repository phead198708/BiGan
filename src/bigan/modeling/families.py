"""Market-family helpers for dataset and model evaluation artifacts."""

from __future__ import annotations

from typing import Any

UNKNOWN_MARKET_FAMILY = "UNKNOWN"
OUTCOME_SIDES = frozenset({"UP", "DOWN"})


def market_family_from_symbol(value: Any) -> str:
    """Return a stable family key such as ``BTC-15M`` from a canonical symbol."""

    if value is None:
        return UNKNOWN_MARKET_FAMILY
    text = str(value).strip().upper()
    if not text:
        return UNKNOWN_MARKET_FAMILY
    if ":" in text:
        family = text.split(":", 1)[0].strip()
        return family or UNKNOWN_MARKET_FAMILY

    parts = [part for part in text.split("-") if part]
    if len(parts) >= 3 and parts[1] in OUTCOME_SIDES:
        return f"{parts[0]}-{parts[2]}"
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return UNKNOWN_MARKET_FAMILY
