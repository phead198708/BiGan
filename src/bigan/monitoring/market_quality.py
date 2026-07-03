"""Market-price quality guards for live signal consumers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MAX_TRADABLE_SPREAD = 0.95
MIN_PRICE = 0.0
MAX_PRICE = 1.0
MIN_PLAUSIBLE_ROUND_START_TS_MS = 1_500_000_000_000


def tradable_market_implied_probability(
    snapshot: Mapping[str, Any],
    *,
    event_ts: int | None = None,
) -> float | None:
    """Return a market price only when the snapshot still looks tradable.

    Raw post-expiry Polymarket ``price_change`` events can carry degenerate
    quotes such as bid=0/ask=1 with zero liquidity. Those rows are useful for
    audit, but must not participate in edge thresholds, paper PnL, or bridge
    signals.
    """

    market = market_implied_probability(snapshot)
    if market is None or market < MIN_PRICE or market > MAX_PRICE:
        return None
    if event_ts is not None and is_outside_round_window(snapshot, event_ts=event_ts):
        return None
    if is_degenerate_quote(snapshot, market_implied_prob=market):
        return None
    return market


def market_implied_probability(snapshot: Mapping[str, Any]) -> float | None:
    """Extract the first market-implied price from a prediction snapshot."""

    feature_map = _features(snapshot)
    for source in (snapshot, feature_map):
        for key in ("market_implied_prob", "best_ask", "entry_ask_price"):
            value = _optional_float(source.get(key))
            if value is not None:
                return value
    return None


def is_post_round_end(snapshot: Mapping[str, Any], *, event_ts: int) -> bool:
    """Return True when the event timestamp is at or after round expiry."""

    round_end_ts = round_end_ts_from_snapshot(snapshot)
    return round_end_ts is not None and int(event_ts) >= round_end_ts


def is_outside_round_window(snapshot: Mapping[str, Any], *, event_ts: int) -> bool:
    """Return True when an event is before round start or at/after expiry."""

    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    round_start_ts = round_start_ts_from_canonical_symbol(canonical_symbol)
    round_end_ts = round_end_ts_from_canonical_symbol(canonical_symbol)
    if round_start_ts is None or round_end_ts is None:
        return False
    resolved_event_ts = int(event_ts)
    return resolved_event_ts < round_start_ts or resolved_event_ts >= round_end_ts


def round_end_ts_from_snapshot(snapshot: Mapping[str, Any]) -> int | None:
    canonical_symbol = str(snapshot.get("canonical_symbol") or snapshot.get("symbol") or "")
    return round_end_ts_from_canonical_symbol(canonical_symbol)


def round_end_ts_from_canonical_symbol(canonical_symbol: str) -> int | None:
    round_start_ts = round_start_ts_from_canonical_symbol(canonical_symbol)
    if round_start_ts is None:
        return None
    parts = canonical_symbol.split(":")
    family = parts[0].upper()
    round_slug = parts[-2]
    horizon_ms = _horizon_ms_from_family(family)
    if horizon_ms is None:
        horizon_ms = _horizon_ms_from_slug(round_slug)
    return round_start_ts + horizon_ms if horizon_ms is not None else None


def round_start_ts_from_canonical_symbol(canonical_symbol: str) -> int | None:
    parts = canonical_symbol.split(":")
    if len(parts) < 3:
        return None
    round_slug = parts[-2]
    try:
        round_start_ts = int(round_slug.rsplit("-", 1)[-1]) * 1000
    except ValueError:
        return None
    if round_start_ts < MIN_PLAUSIBLE_ROUND_START_TS_MS:
        return None
    return round_start_ts


def is_degenerate_quote(
    snapshot: Mapping[str, Any],
    *,
    market_implied_prob: float | None = None,
) -> bool:
    """Return True for non-executable post-close/zero-liquidity quote shapes."""

    feature_map = _features(snapshot)
    market = market_implied_prob
    if market is None:
        market = market_implied_probability(snapshot)
    spread = _first_float(snapshot, feature_map, keys=("spread", "tick_spread"))
    if spread is not None and spread >= MAX_TRADABLE_SPREAD:
        return True

    bid = _first_float(snapshot, feature_map, keys=("best_bid", "bid", "bid_price"))
    ask = _first_float(
        snapshot,
        feature_map,
        keys=("best_ask", "ask", "ask_price", "market_implied_prob"),
    )
    if bid == 0.0 and ask == 1.0:
        return True

    liquidity_bucket = _first_float(snapshot, feature_map, keys=("liquidity_bucket",))
    if liquidity_bucket == 0.0 and market in {0.0, 1.0}:
        return True

    size = _first_float(
        snapshot,
        feature_map,
        keys=("size", "best_ask_size", "ask_size", "entry_ask_size"),
    )
    if size == 0.0 and market in {0.0, 1.0}:
        return True
    return False


def _features(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    features = snapshot.get("features")
    return features if isinstance(features, Mapping) else {}


def _first_float(
    snapshot: Mapping[str, Any],
    feature_map: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
) -> float | None:
    for source in (snapshot, feature_map):
        for key in keys:
            value = _optional_float(source.get(key))
            if value is not None:
                return value
    return None


def _horizon_ms_from_family(family: str) -> int | None:
    parts = family.split("-")
    if len(parts) < 2:
        return None
    return _horizon_text_to_ms(parts[1])


def _horizon_ms_from_slug(round_slug: str) -> int | None:
    parts = round_slug.upper().split("-")
    for part in parts:
        horizon_ms = _horizon_text_to_ms(part)
        if horizon_ms is not None:
            return horizon_ms
    return None


def _horizon_text_to_ms(text: str) -> int | None:
    upper = text.upper()
    if upper.endswith("M"):
        value = _optional_float(upper[:-1])
        return None if value is None else int(value * 60_000)
    if upper.endswith("H"):
        value = _optional_float(upper[:-1])
        return None if value is None else int(value * 60 * 60_000)
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
