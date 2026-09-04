"""Bounded, token-local CLOB depth; top-price notices never invent liquidity."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

MAX_MARKET_LEVELS = 10_000


class MarketDepth:
    def __init__(self, bids: dict[float, float], asks: dict[float, float]) -> None:
        self.bids, self.asks = bids, asks
        self.top()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MarketDepth:
        sides = []
        for name in ("bids", "asks"):
            raw = payload.get(name)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) > MAX_MARKET_LEVELS:
                raise ValueError("invalid or oversized CLOB depth")
            levels = {}
            for level in raw:
                if not isinstance(level, Mapping):
                    raise ValueError("invalid CLOB level")
                price, size = _level(level)
                if price in levels:
                    raise ValueError("duplicate CLOB price level")
                if size:
                    levels[price] = size
            sides.append(levels)
        return cls(*sides)

    def updated(self, payload: Mapping[str, object]) -> MarketDepth:
        # Legacy explicit full-depth updates remain supported.
        if "bids" in payload or "asks" in payload:
            return self.from_payload(payload)
        bids, asks = self.bids.copy(), self.asks.copy()
        side = payload.get("side")
        if side not in {"BUY", "SELL"}:
            raise ValueError("CLOB depth change requires BUY/SELL")
        levels = bids if side == "BUY" else asks
        price, size = _level(payload)
        if size:
            levels[price] = size
        else:
            levels.pop(price, None)
        if len(levels) > MAX_MARKET_LEVELS:
            raise ValueError("CLOB depth exceeds memory bound")
        book = MarketDepth(bids, asks)
        if "best_bid" in payload and "best_ask" in payload and not book.matches(payload):
            raise ValueError("CLOB delta top disagrees with reconstructed depth")
        return book

    def matches(self, payload: Mapping[str, object]) -> bool:
        bid, ask = self.top()
        return bid == _price(payload.get("best_bid")) and ask == _price(payload.get("best_ask"))

    def top(self) -> tuple[float, float]:
        if not self.bids or not self.asks:
            raise ValueError("CLOB requires positive liquidity on both sides")
        bid, ask = max(self.bids), min(self.asks)
        if bid > ask:
            raise ValueError("crossed CLOB depth")
        return bid, ask

    def top_payload(self) -> dict[str, object]:
        bid, ask = self.top()
        return {"bids": [{"price": bid, "size": self.bids[bid]}],
                "asks": [{"price": ask, "size": self.asks[ask]}]}


def _price(raw: object) -> float:
    if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
        raise ValueError("CLOB price must be numeric")
    value = float(raw)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("CLOB price must be in (0, 1]")
    return value


def _level(payload: Mapping[str, object]) -> tuple[float, float]:
    price = _price(payload.get("price"))
    raw = payload.get("size")
    if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
        raise ValueError("CLOB size must be numeric")
    size = float(raw)
    if not math.isfinite(size) or size < 0:
        raise ValueError("CLOB size must be finite and non-negative")
    return price, size
