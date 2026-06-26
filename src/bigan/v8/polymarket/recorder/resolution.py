"""Resolution capture helpers for the raw corpus recorder."""

from __future__ import annotations

import math
from typing import Any

from bigan.v8.polymarket.corpus.contracts import safety_fields
from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig


def mock_resolution_rows(
    markets: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> list[dict[str, Any]]:
    """Build deterministic official-resolution rows for resolved mock markets."""

    rows: list[dict[str, Any]] = []
    for index, market in enumerate(markets):
        if config.inject_missing_resolution and index == 0:
            continue
        start = float(market["reference_price_start"])
        end = start + 25.0 + index * 3.0
        rows.append(
            {
                "market_id": market["market_id"],
                "reference_price_start": start,
                "reference_price_end": end,
                "reference_price_source": market.get("reference_price_source"),
                "resolution_status": "normal",
                "raw_resolution_text": (
                    "Resolved from the official Polymarket BTC reference source "
                    f"{market.get('reference_price_source')}"
                ),
                **safety_fields(),
            }
        )
    return rows


def validate_resolution_row(
    *,
    market: dict[str, Any],
    resolution_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: set[str] = set()
    candidates = [row for row in resolution_rows if row.get("market_id") == market["market_id"]]
    if not candidates:
        return None, ["missing_resolution"]
    if len(candidates) > 1:
        reasons.add("duplicate_resolution")
    row = candidates[0]
    if not str(market.get("reference_price_source") or "").strip():
        reasons.add("missing_verified_resolution_source")
    if row.get("reference_price_source") != market.get("reference_price_source"):
        reasons.add("resolution_source_mismatch")
    has_reference_prices = (
        row.get("reference_price_start") is not None
        and row.get("reference_price_end") is not None
    )
    if has_reference_prices:
        if _optional_positive_float(row.get("reference_price_start")) is None:
            reasons.add("invalid_reference_price_start")
        if _optional_positive_float(row.get("reference_price_end")) is None:
            reasons.add("invalid_reference_price_end")
    elif not _valid_payout_resolution(row):
        reasons.add("missing_verified_resolution")
    status = str(row.get("resolution_status") or "")
    if status not in {"normal", "unknown_50_50"}:
        reasons.add("invalid_resolution_status")
    return (None, sorted(reasons)) if reasons else (row, [])


def _valid_payout_resolution(row: dict[str, Any]) -> bool:
    resolved_outcome = str(row.get("resolved_outcome") or "").upper()
    if resolved_outcome == "UNKNOWN":
        resolved_outcome = "UNKNOWN_50_50"
    payout_up = _optional_float(row.get("payout_up"))
    payout_down = _optional_float(row.get("payout_down"))
    if payout_up is None or payout_down is None:
        return False
    if resolved_outcome == "UP":
        return (payout_up, payout_down) == (1.0, 0.0)
    if resolved_outcome == "DOWN":
        return (payout_up, payout_down) == (0.0, 1.0)
    if resolved_outcome == "UNKNOWN_50_50":
        return (payout_up, payout_down) == (0.5, 0.5)
    return False


def _optional_positive_float(value: Any) -> float | None:
    numeric = _optional_float(value)
    if numeric is None or numeric <= 0.0 or not math.isfinite(numeric):
        return None
    return numeric


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
