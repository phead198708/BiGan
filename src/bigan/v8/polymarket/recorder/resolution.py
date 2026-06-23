"""Resolution capture helpers for the raw corpus recorder."""

from __future__ import annotations

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
    if float(row.get("reference_price_start") or 0.0) <= 0.0:
        reasons.add("invalid_reference_price_start")
    if float(row.get("reference_price_end") or 0.0) <= 0.0:
        reasons.add("invalid_reference_price_end")
    status = str(row.get("resolution_status") or "")
    if status not in {"normal", "unknown_50_50"}:
        reasons.add("invalid_resolution_status")
    return (None, sorted(reasons)) if reasons else (row, [])
