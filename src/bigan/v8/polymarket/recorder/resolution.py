"""Resolution capture helpers for the raw corpus recorder."""

from __future__ import annotations

import math
from typing import Any

from bigan.v8.polymarket.contracts import PolymarketAdapterError
from bigan.v8.polymarket.corpus.contracts import safety_fields
from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig
from bigan.v8.polymarket.rules import (
    build_btc_updown_resolution_rule,
    payout_for_resolved_outcome,
    resolve_polymarket_rule,
)


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


def normalize_resolution_for_settlement(
    *,
    market: dict[str, Any],
    resolution: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize a verified post-freeze resolution into explicit payout semantics."""

    validated, reasons = validate_resolution_row(
        market=market,
        resolution_rows=[resolution],
    )
    if validated is None:
        return None, reasons

    if validated.get("paper_only") is not True:
        return None, ["resolution_paper_only_flag_missing_or_false"]
    for field in (
        "capital_at_risk",
        "broker_exchange_write_enabled",
        "live_exchange_write_enabled",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
    ):
        if validated.get(field) is not False:
            return None, [f"resolution_safety_flag_invalid:{field}"]

    source_type = str(validated.get("resolution_source_type") or "")
    derived_resolution = None
    has_reference_prices = (
        validated.get("reference_price_start") is not None
        and validated.get("reference_price_end") is not None
    )
    if has_reference_prices:
        try:
            rule = build_btc_updown_resolution_rule(
                market_id=str(market["market_id"]),
                condition_id=str(market["condition_id"]),
                slug=str(market["slug"]),
                market_family=str(market["market_family"]),
                resolution_source=str(market["reference_price_source"]),
                candle_open_ts=int(market["market_start_ts"]),
                candle_close_ts=int(market["market_end_ts"]),
                raw_rule_text=str(market["settlement_rule"]),
            )
            derived_resolution = resolve_polymarket_rule(
                rule,
                reference_price_start=float(validated["reference_price_start"]),
                reference_price_end=float(validated["reference_price_end"]),
                resolution_status=str(validated["resolution_status"]),
            )
        except (KeyError, PolymarketAdapterError, TypeError, ValueError) as exc:
            return None, [f"resolution_rule_derivation_failed:{type(exc).__name__}"]

    resolved_outcome = str(validated.get("resolved_outcome") or "").upper()
    if resolved_outcome == "UNKNOWN":
        resolved_outcome = "UNKNOWN_50_50"
    if resolved_outcome in {"UP", "DOWN", "UNKNOWN_50_50"}:
        if source_type != "polymarket_clob_market_tokens":
            return None, ["explicit_resolution_source_type_not_approved"]
        if (
            derived_resolution is not None
            and derived_resolution.resolved_outcome != resolved_outcome
        ):
            return None, ["explicit_outcome_reference_price_rule_mismatch"]
        payout_up, payout_down = payout_for_resolved_outcome(resolved_outcome)
        provided_up = _optional_float(validated.get("payout_up"))
        provided_down = _optional_float(validated.get("payout_down"))
        if provided_up is not None and provided_up != payout_up:
            return None, ["resolution_payout_up_mismatch"]
        if provided_down is not None and provided_down != payout_down:
            return None, ["resolution_payout_down_mismatch"]
        normalized = {
            **validated,
            "resolved_outcome": resolved_outcome,
            "payout_up": payout_up,
            "payout_down": payout_down,
            "resolved_outcome_source": "official_explicit_resolution",
            "resolution_normalization_reason_codes": [
                "explicit_official_outcome_normalized"
            ],
        }
        return normalized, []

    if source_type != "reference_prices":
        return None, ["reference_price_resolution_source_type_not_approved"]
    if derived_resolution is None:
        return None, ["reference_price_resolution_missing_verified_prices"]

    normalized = {
        **validated,
        "resolved_outcome": derived_resolution.resolved_outcome,
        "payout_up": derived_resolution.payout_up,
        "payout_down": derived_resolution.payout_down,
        "resolved_outcome_source": "official_reference_price_rule_derivation",
        "resolution_rule_sha256": rule.raw_rule_sha256,
        "resolution_normalization_reason_codes": [
            "resolved_outcome_derived_from_official_reference_prices"
        ],
    }
    return normalized, []


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
