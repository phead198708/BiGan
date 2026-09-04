"""Strict parsing and deterministic selection of Polymarket windows."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class MarketDiscoveryError(RuntimeError):
    """Fail-closed discovery error with a stable observable reason code."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class DiscoveryFilters:
    underlying: str
    market_type: str
    window_duration_ms: int
    slug_pattern: str | None = None
    title_pattern: str | None = None
    max_preopen_ms: int = 1_800_000

    def __post_init__(self) -> None:
        if not self.underlying or self.underlying != self.underlying.upper():
            raise ValueError("underlying must be uppercase and non-empty")
        if not self.market_type:
            raise ValueError("market_type must be non-empty")
        if self.window_duration_ms <= 0 or self.max_preopen_ms <= 0:
            raise ValueError("discovery durations must be positive")
        if self.window_duration_ms not in {300_000, 900_000}:
            raise ValueError("paper discovery supports only 5m and 15m windows")
        for pattern in (self.slug_pattern, self.title_pattern):
            if pattern is not None:
                re.compile(pattern)


@dataclass(frozen=True, slots=True)
class DiscoveredMarket:
    market_id: str
    condition_id: str
    slug: str
    title: str
    underlying: str
    market_type: str
    window_duration_ms: int
    start_ts_ms: int
    end_ts_ms: int
    yes_token_id: str
    no_token_id: str
    active: bool
    closed: bool
    accepting_orders: bool
    source_endpoint: str
    discovered_at_ms: int
    resolution_source: str
    resolution_identity: str
    reference_price_at_start: float | None
    raw_payload_sha256: str

    @property
    def window_id(self) -> str:
        return f"{self.condition_id}:{self.start_ts_ms}:{self.end_ts_ms}"

    def provenance(self) -> dict[str, object]:
        return {
            "source_endpoint": self.source_endpoint,
            "discovered_at_ms": self.discovered_at_ms,
            "market_id": self.market_id,
            "condition_id": self.condition_id,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "slug": self.slug,
            "window_id": self.window_id,
            "window_start_ts_ms": self.start_ts_ms,
            "window_end_ts_ms": self.end_ts_ms,
            "resolution_source": self.resolution_source,
            "resolution_identity": self.resolution_identity,
            "reference_price_at_start": self.reference_price_at_start,
            "raw_payload_sha256": self.raw_payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class DiscoverySelection:
    current: DiscoveredMarket | None
    next: DiscoveredMarket | None
    eligible_count: int


def parse_gamma_markets(
    payload: object,
    *,
    source_endpoint: str,
    discovered_at_ms: int,
) -> tuple[DiscoveredMarket, ...]:
    """Parse external response shape without performing business selection."""

    rows = _market_rows(payload)
    parsed: list[DiscoveredMarket] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MarketDiscoveryError(
                f"invalid market payload at row {index}",
                reason_code="malformed_market_payload",
            )
        try:
            parsed.append(
                _parse_market_row(
                    row,
                    source_endpoint=source_endpoint,
                    discovered_at_ms=discovered_at_ms,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MarketDiscoveryError(
                f"invalid market payload at row {index}: {exc}",
                reason_code="malformed_market_payload",
            ) from exc
    _validate_unique_identities(parsed)
    return tuple(parsed)


def select_market_windows(
    candidates: tuple[DiscoveredMarket, ...],
    *,
    filters: DiscoveryFilters,
    now_ms: int,
) -> DiscoverySelection:
    """Select current and next windows with explicit, deterministic rules."""

    eligible = [
        market
        for market in candidates
        if market.active
        and not market.closed
        and market.accepting_orders
        and market.underlying == filters.underlying
        and market.market_type == filters.market_type
        and market.window_duration_ms == filters.window_duration_ms
        and market.start_ts_ms - filters.max_preopen_ms <= now_ms < market.end_ts_ms
        and _fullmatch(filters.slug_pattern, market.slug)
        and _fullmatch(filters.title_pattern, market.title)
    ]
    current = sorted(
        (market for market in eligible if market.start_ts_ms <= now_ms),
        key=lambda market: (
            market.end_ts_ms,
            -market.start_ts_ms,
            market.market_id,
            market.condition_id,
        ),
    )
    upcoming = sorted(
        (market for market in eligible if market.start_ts_ms > now_ms),
        key=lambda market: (
            market.start_ts_ms,
            market.end_ts_ms,
            market.market_id,
            market.condition_id,
        ),
    )
    _reject_equal_window_ambiguity(current, "ambiguous_current_market")
    _reject_equal_window_ambiguity(upcoming, "ambiguous_next_market")
    selected_current = current[0] if current else None
    selected_next = upcoming[0] if upcoming else None
    if selected_current is None and selected_next is None:
        raise MarketDiscoveryError(
            "no eligible market matches strict discovery filters",
            reason_code="no_eligible_market",
        )
    return DiscoverySelection(
        current=selected_current,
        next=selected_next,
        eligible_count=len(eligible),
    )


def _parse_market_row(
    row: dict[str, Any],
    *,
    source_endpoint: str,
    discovered_at_ms: int,
) -> DiscoveredMarket:
    slug = _text(row.get("slug"), "slug")
    slug_identity = _slug_market_identity(slug)
    outcomes = _two_strings(row.get("outcomes"), "outcomes")
    tokens = _two_strings(row.get("clobTokenIds"), "clobTokenIds")
    normalized_outcomes = [value.upper() for value in outcomes]
    yes_index = _outcome_index(normalized_outcomes, {"YES", "UP"})
    no_index = _outcome_index(normalized_outcomes, {"NO", "DOWN"})
    yes_token = tokens[yes_index].strip()
    no_token = tokens[no_index].strip()
    if not yes_token or not no_token or yes_token == no_token:
        raise ValueError("YES/NO token IDs must be non-empty and distinct")
    start = _window_start_ms(row, slug_identity)
    end = _timestamp_ms(row, "end_ts_ms", "endDate", "endDateIso")
    if end <= start:
        raise ValueError("market end must be after start")
    raw_duration = row.get("windowDurationMs")
    duration = (
        slug_identity[2]
        if raw_duration is None and slug_identity is not None
        else _strict_int(raw_duration, "windowDurationMs")
    )
    if (duration <= 0 or end - start != duration
            or slug_identity is not None and duration != slug_identity[2]):
        raise ValueError("window duration does not match start/end")
    market_id = _text(row.get("id") or row.get("market_id"), "market_id")
    condition_id = _text(row.get("conditionId") or row.get("condition_id"), "condition_id")
    resolution_source = _text(row.get("resolutionSource"), "resolutionSource")
    resolution_identity = _text(
        row.get("resolutionIdentity")
        or f"condition:{condition_id}:source:{resolution_source}",
        "resolutionIdentity",
    )
    reference_price_at_start = _optional_positive_float(
        row.get("referencePriceAtStart")
        or row.get("reference_price_at_start")
        or row.get("priceToBeat"),
        "referencePriceAtStart",
    )
    encoded = json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return DiscoveredMarket(
        market_id=market_id,
        condition_id=condition_id,
        slug=slug,
        title=_text(row.get("question") or row.get("title"), "title"),
        underlying=(
            _text(row.get("underlying"), "underlying").upper()
            if row.get("underlying") is not None
            else _required_slug_identity(slug_identity)[0]
        ),
        market_type=(
            _text(row.get("marketType"), "marketType")
            if row.get("marketType") is not None
            else _required_slug_identity(slug_identity)[1]
        ),
        window_duration_ms=duration,
        start_ts_ms=start,
        end_ts_ms=end,
        yes_token_id=yes_token,
        no_token_id=no_token,
        active=_strict_bool(row.get("active"), "active"),
        closed=_strict_bool(row.get("closed"), "closed"),
        accepting_orders=_strict_bool(row.get("acceptingOrders"), "acceptingOrders"),
        source_endpoint=source_endpoint,
        discovered_at_ms=_strict_int(discovered_at_ms, "discovered_at_ms"),
        resolution_source=resolution_source,
        resolution_identity=resolution_identity,
        reference_price_at_start=reference_price_at_start,
        raw_payload_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _market_rows(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("data", payload.get("markets"))
        if isinstance(rows, list):
            return rows
    raise MarketDiscoveryError(
        "invalid Gamma market response payload",
        reason_code="malformed_discovery_response",
    )


def _two_strings(value: object, name: str) -> list[str]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if (
        not isinstance(decoded, list)
        or len(decoded) != 2
        or any(not isinstance(item, str) for item in decoded)
    ):
        raise ValueError(f"{name} must contain exactly two strings")
    return decoded


def _outcome_index(outcomes: list[str], accepted: set[str]) -> int:
    indexes = [index for index, value in enumerate(outcomes) if value in accepted]
    if len(indexes) != 1:
        raise ValueError("outcomes must identify one YES/UP and one NO/DOWN token")
    return indexes[0]


def _window_start_ms(
    row: dict[str, Any], slug_identity: tuple[str, str, int, int] | None,
) -> int:
    # Gamma startDate/startDateIso can describe listing time, not the period
    # whose opening price resolves the market. Never use them as a strike time.
    candidates = [
        value for name in ("start_ts_ms", "eventStartTime")
        if (value := _timestamp_ms_optional(row, name)) is not None
    ]
    if slug_identity is not None:
        candidates.append(slug_identity[3])
    if not candidates:
        raise ValueError("market window start timestamp is missing")
    if len(set(candidates)) != 1:
        raise ValueError("conflicting market window start timestamps")
    return candidates[0]


def _timestamp_ms(row: dict[str, Any], *names: str) -> int:
    parsed = _timestamp_ms_optional(row, *names)
    if parsed is None:
        raise ValueError(f"{names[0]} must be present")
    return parsed


def _timestamp_ms_optional(row: dict[str, Any], *names: str) -> int | None:
    value = next((row[name] for name in names if row.get(name) is not None), None)
    if value is None:
        return None
    if isinstance(value, str) and not value.isdigit():
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)
    return _strict_int(value, names[0])


def _slug_market_identity(slug: str) -> tuple[str, str, int, int] | None:
    match = re.fullmatch(r"([a-z0-9]+)-updown-(5m|15m|1h)-([1-9][0-9]*)", slug)
    if match is None:
        return None
    duration = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}[match.group(2)]
    return match.group(1).upper(), "binary_up_down", duration, int(match.group(3)) * 1_000


def _required_slug_identity(
    identity: tuple[str, str, int, int] | None,
) -> tuple[str, str, int, int]:
    if identity is None:
        raise ValueError("market classification is absent and slug is not a known exact family")
    return identity


def _strict_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_positive_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    parsed = float(str(value))
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite when present")
    return parsed


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def _fullmatch(pattern: str | None, value: str) -> bool:
    return pattern is None or re.fullmatch(pattern, value) is not None


def _validate_unique_identities(markets: list[DiscoveredMarket]) -> None:
    market_ids: set[str] = set()
    condition_ids: set[str] = set()
    window_ids: set[str] = set()
    for market in markets:
        if market.market_id in market_ids or market.condition_id in condition_ids:
            raise MarketDiscoveryError(
                "invalid market payload: duplicate market identity",
                reason_code="duplicate_market_identity",
            )
        if market.window_id in window_ids:
            raise MarketDiscoveryError(
                "invalid market payload: duplicate window identity",
                reason_code="duplicate_window_identity",
            )
        market_ids.add(market.market_id)
        condition_ids.add(market.condition_id)
        window_ids.add(market.window_id)


def _reject_equal_window_ambiguity(
    markets: list[DiscoveredMarket],
    reason_code: str,
) -> None:
    if (
        len(markets) > 1
        and markets[0].start_ts_ms == markets[1].start_ts_ms
        and markets[0].end_ts_ms == markets[1].end_ts_ms
    ):
        raise MarketDiscoveryError(
            "ambiguous markets share the same deterministic window rank",
            reason_code=reason_code,
        )
