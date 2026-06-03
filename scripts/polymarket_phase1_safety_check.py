#!/usr/bin/env python3
"""Read-only safety check before any Polymarket Phase 1 order/cancel test.

The original dry-run runbook used ``best_ask - 2 ticks`` for a passive BUY.
That is not enough for binary markets: a BUY on one outcome can be matchable
against BUY liquidity on the complementary outcome through mint-style matching.

This script never places orders. It checks same-outcome and complementary
outcome liquidity and reports whether a proposed maker BUY price has a safety
buffer before anyone posts a real order.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from bigan.ingestion.gamma_client import GammaClient


@dataclass(frozen=True, slots=True)
class BookSummary:
    token_id: str
    best_bid: float | None
    best_ask: float | None


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    status: str
    reason: str
    side: str
    proposed_price: float | None
    safe_ceiling: float | None
    same_outcome_ask_ceiling: float | None
    complement_buy_ceiling: float | None
    seconds_to_expiry: float
    same_outcome: BookSummary
    complement_outcome: BookSummary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Optional shell-style env file to load")
    parser.add_argument("--side", choices=("BUY_UP", "BUY_DOWN"), default="BUY_UP")
    parser.add_argument("--price", type=float, help="Candidate limit price to evaluate")
    parser.add_argument("--tick-buffer", type=int, default=3)
    parser.add_argument("--complement-buffer", type=int, default=3)
    parser.add_argument("--min-seconds-to-expiry", type=float, default=600.0)
    args = parser.parse_args()

    if args.env_file:
        _load_env_file(args.env_file)

    payload = asyncio.run(
        _run_check(
            side=args.side,
            proposed_price=args.price,
            tick_buffer=args.tick_buffer,
            complement_buffer=args.complement_buffer,
            min_seconds_to_expiry=args.min_seconds_to_expiry,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["decision"]["status"] == "PASS" else 2


async def _run_check(
    *,
    side: str,
    proposed_price: float | None,
    tick_buffer: int,
    complement_buffer: int,
    min_seconds_to_expiry: float,
) -> dict[str, Any]:
    client = _build_clob_client()
    now_ms = int(time.time() * 1000)
    async with GammaClient("https://gamma-api.polymarket.com", "btc-updown-15m-") as gamma:
        markets = await gamma.list_active_markets(page_limit=100, max_pages=5)
    candidates = [market for market in markets if (market.end_ts_ms or 0) > now_ms]
    if not candidates:
        raise RuntimeError("no active BTC 15m market found")
    market = sorted(candidates, key=lambda m: m.end_ts_ms or 0)[0]

    tick_size = client.get_tick_size(market.asset_id_up)
    neg_risk = client.get_neg_risk(market.asset_id_up)
    tick = _to_decimal(tick_size)
    up_book = _book_summary(client, market.asset_id_up)
    down_book = _book_summary(client, market.asset_id_down)
    if side == "BUY_UP":
        same = up_book
        complement = down_book
    else:
        same = down_book
        complement = up_book

    seconds_to_expiry = ((market.end_ts_ms or now_ms) - now_ms) / 1000.0
    decision = evaluate_buy_safety(
        side=side,
        proposed_price=proposed_price,
        same_outcome=same,
        complement_outcome=complement,
        tick=float(tick),
        tick_buffer=tick_buffer,
        complement_buffer=complement_buffer,
        seconds_to_expiry=seconds_to_expiry,
        min_seconds_to_expiry=min_seconds_to_expiry,
    )

    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "market_slug": market.slug,
        "market_end": datetime.fromtimestamp((market.end_ts_ms or 0) / 1000, UTC).isoformat(),
        "tick_size": str(tick_size),
        "neg_risk": bool(neg_risk),
        "decision": _decision_to_dict(decision),
        "note": (
            "This is a read-only guard. It reduces accidental fill risk but does not make "
            "a real resting order safe; any live GTC/GTD maker order can be filled before cancel."
        ),
    }


def evaluate_buy_safety(
    *,
    side: str,
    proposed_price: float | None,
    same_outcome: BookSummary,
    complement_outcome: BookSummary,
    tick: float,
    tick_buffer: int,
    complement_buffer: int,
    seconds_to_expiry: float,
    min_seconds_to_expiry: float,
) -> SafetyDecision:
    if seconds_to_expiry < min_seconds_to_expiry:
        return SafetyDecision(
            status="FAIL",
            reason=(
                f"market expires in {seconds_to_expiry:.1f}s; require at least "
                f"{min_seconds_to_expiry:.1f}s for an order/cancel test"
            ),
            side=side,
            proposed_price=proposed_price,
            safe_ceiling=None,
            same_outcome_ask_ceiling=None,
            complement_buy_ceiling=None,
            seconds_to_expiry=seconds_to_expiry,
            same_outcome=same_outcome,
            complement_outcome=complement_outcome,
        )

    same_ceiling = None
    if same_outcome.best_ask is not None:
        same_ceiling = _round_down_decimal(
            Decimal(str(same_outcome.best_ask)) - (Decimal(str(tick_buffer)) * Decimal(str(tick))),
            tick,
        )

    complement_ceiling = None
    if complement_outcome.best_bid is not None:
        complement_ceiling = _round_down_decimal(
            Decimal("1")
            - Decimal(str(complement_outcome.best_bid))
            - (Decimal(str(complement_buffer)) * Decimal(str(tick))),
            tick,
        )

    ceilings = [value for value in (same_ceiling, complement_ceiling) if value is not None]
    safe_ceiling = min(ceilings) if ceilings else None
    candidate_price = proposed_price if proposed_price is not None else safe_ceiling
    if safe_ceiling is None or candidate_price is None:
        status = "FAIL"
        reason = "missing orderbook levels; cannot compute a safe maker price"
    elif candidate_price <= 0:
        status = "FAIL"
        reason = f"candidate price {candidate_price:.4f} is not positive"
    elif candidate_price > safe_ceiling:
        status = "FAIL"
        reason = (
            f"candidate price {candidate_price:.4f} exceeds safe ceiling {safe_ceiling:.4f}; "
            "it can be marketable against same-outcome or complementary liquidity"
        )
    else:
        status = "PASS"
        reason = (
            f"candidate price {candidate_price:.4f} is at or below safe ceiling "
            f"{safe_ceiling:.4f}; still use real-order tests only with tiny size and immediate cancel"
        )

    return SafetyDecision(
        status=status,
        reason=reason,
        side=side,
        proposed_price=candidate_price,
        safe_ceiling=safe_ceiling,
        same_outcome_ask_ceiling=same_ceiling,
        complement_buy_ceiling=complement_ceiling,
        seconds_to_expiry=seconds_to_expiry,
        same_outcome=same_outcome,
        complement_outcome=complement_outcome,
    )


def _build_clob_client() -> Any:
    from py_clob_client_v2 import ClobClient, SignatureTypeV2  # type: ignore[import-not-found]

    client = ClobClient(
        os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com"),
        key=_require_env("POLYMARKET_PRIVATE_KEY"),
        chain_id=137,
        signature_type=SignatureTypeV2.POLY_PROXY,
        funder=os.getenv("POLYMARKET_FUNDER"),
    )
    client.set_api_creds(client.create_or_derive_api_key())
    return client


def _book_summary(client: Any, token_id: str) -> BookSummary:
    book = client.get_order_book(token_id)
    bids = getattr(book, "bids", None) or book.get("bids", [])
    asks = getattr(book, "asks", None) or book.get("asks", [])
    bid_prices = [_optional_float(_level_price(level)) for level in bids]
    ask_prices = [_optional_float(_level_price(level)) for level in asks]
    clean_bids = [price for price in bid_prices if price is not None]
    clean_asks = [price for price in ask_prices if price is not None]
    return BookSummary(
        token_id=token_id,
        best_bid=max(clean_bids) if clean_bids else None,
        best_ask=min(clean_asks) if clean_asks else None,
    )


def _decision_to_dict(decision: SafetyDecision) -> dict[str, Any]:
    payload = asdict(decision)
    payload["same_outcome"] = asdict(decision.same_outcome)
    payload["complement_outcome"] = asdict(decision.complement_outcome)
    return payload


def _level_price(level: Any) -> Any:
    if isinstance(level, dict):
        return level.get("price")
    return getattr(level, "price", None)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_down(value: float, tick: float) -> float:
    if value <= 0:
        return value
    raw = Decimal(str(value))
    quantum = Decimal(str(tick))
    return float((raw / quantum).to_integral_value(rounding=ROUND_DOWN) * quantum)


def _round_down_decimal(value: Decimal, tick: float) -> float:
    if value <= 0:
        return float(value)
    quantum = Decimal(str(tick))
    return float((value / quantum).to_integral_value(rounding=ROUND_DOWN) * quantum)


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise RuntimeError(f"invalid tick size: {value!r}") from exc


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
