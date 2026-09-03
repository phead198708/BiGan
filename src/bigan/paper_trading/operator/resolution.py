"""Strict final-resolution parsing for paper settlement provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from .discovery import DiscoveredMarket
from .transports import PublicJSONClient


class ResolutionError(RuntimeError):
    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class FinalResolution:
    market_id: str
    condition_id: str
    window_id: str
    yes_payout: float
    settlement_ts_ms: int
    source: str
    source_ts_ms: int
    received_ts_ms: int
    source_reference: str
    resolution_identity: str


def parse_gamma_resolution(
    payload: object,
    *,
    market: DiscoveredMarket,
    source_endpoint: str,
    received_at_ms: int,
) -> FinalResolution | None:
    """Return only a provably final binary payout; never infer from a book."""

    rows = _rows(payload)
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("id") or row.get("market_id") or "") == market.market_id
    ]
    if len(matches) != 1:
        raise ResolutionError(
            "resolution response does not contain exactly one matching market",
            reason_code="resolution_identity_mismatch",
        )
    row = matches[0]
    condition_id = str(row.get("conditionId") or row.get("condition_id") or "")
    if condition_id != market.condition_id:
        raise ResolutionError(
            "resolution condition identity differs from discovery",
            reason_code="resolution_identity_mismatch",
        )
    if row.get("closed") is not True or row.get("resolved") is False:
        return None
    if row.get("outcomePrices") is None:
        return None
    try:
        outcomes = _two_strings(row.get("outcomes"), "outcomes")
        prices = _two_values(row.get("outcomePrices"), "outcomePrices")
        tokens = _two_strings(row.get("clobTokenIds"), "clobTokenIds")
        yes_index = _yes_index(outcomes)
        no_index = 1 - yes_index
        if tokens[yes_index] != market.yes_token_id or tokens[no_index] != market.no_token_id:
            raise ValueError("resolution token identity differs from discovery")
        yes_payout = _number(prices[yes_index], "YES payout")
        no_payout = _number(prices[no_index], "NO payout")
        if (yes_payout, no_payout) not in {(1.0, 0.0), (0.0, 1.0)}:
            if row.get("resolved") is True:
                raise ValueError("final binary payout must be exactly 1/0")
            return None
        resolution_source = str(row.get("resolutionSource") or "")
        resolution_identity = str(
            row.get("resolutionIdentity")
            or f"condition:{condition_id}:source:{resolution_source}"
        )
        if (
            resolution_source != market.resolution_source
            or resolution_identity != market.resolution_identity
        ):
            raise ValueError("resolution provenance differs from discovery")
        source_ts = _timestamp_ms(
            row.get("resolutionTimestamp")
            or row.get("resolvedAt")
            or row.get("end_ts_ms")
            or row.get("endDate"),
            "resolution timestamp",
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResolutionError(
            f"malformed final resolution: {exc}",
            reason_code="malformed_final_resolution",
        ) from exc
    encoded = json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return FinalResolution(
        market_id=market.market_id,
        condition_id=market.condition_id,
        window_id=market.window_id,
        yes_payout=yes_payout,
        settlement_ts_ms=max(market.end_ts_ms, source_ts),
        source=f"gamma:{resolution_source}",
        source_ts_ms=source_ts,
        received_ts_ms=int(received_at_ms),
        source_reference=f"{source_endpoint}#{hashlib.sha256(encoded).hexdigest()}",
        resolution_identity=resolution_identity,
    )


class GammaResolutionClient:
    def __init__(self, *, endpoint: str, http: PublicJSONClient) -> None:
        self.endpoint = endpoint
        self.http = http

    async def resolve(
        self,
        market: DiscoveredMarket,
        *,
        now_ms: int,
    ) -> FinalResolution | None:
        payload = await self.http.get_json(self.endpoint, params={"id": market.market_id})
        return parse_gamma_resolution(
            payload,
            market=market,
            source_endpoint=self.endpoint,
            received_at_ms=now_ms,
        )


def _rows(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("data", payload.get("markets"))
        if isinstance(rows, list):
            return rows
    raise ResolutionError(
        "malformed resolution response",
        reason_code="malformed_resolution_response",
    )


def _two_strings(value: object, name: str) -> list[str]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, list) or len(decoded) != 2 or not all(
        isinstance(item, str) and item.strip() for item in decoded
    ):
        raise ValueError(f"{name} must contain two non-empty strings")
    return [item.strip() for item in decoded]


def _two_values(value: object, name: str) -> list[object]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, list) or len(decoded) != 2:
        raise ValueError(f"{name} must contain two values")
    return decoded


def _yes_index(outcomes: list[str]) -> int:
    indexes = [index for index, value in enumerate(outcomes) if value.upper() in {"YES", "UP"}]
    if len(indexes) != 1:
        raise ValueError("outcomes must identify exactly one YES/UP")
    opposite = outcomes[1 - indexes[0]].upper()
    if opposite not in {"NO", "DOWN"}:
        raise ValueError("outcomes must identify exactly one NO/DOWN")
    return indexes[0]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _timestamp_ms(value: object, name: str) -> int:
    if isinstance(value, str) and not value.isdigit():
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"{name} must be ISO-8601 or epoch milliseconds") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        value = int(parsed.timestamp() * 1_000)
    return _positive_int(value, name)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        return float(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
