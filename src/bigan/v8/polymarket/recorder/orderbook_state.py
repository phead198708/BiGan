"""Orderbook state and sampling helpers for the raw corpus recorder."""

from __future__ import annotations

import math
from typing import Any

from bigan.v8.polymarket.corpus.contracts import safety_fields
from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig


def sample_times_for_market(
    market: dict[str, Any],
    config: PolymarketRealCorpusRecorderConfig,
) -> tuple[int, ...]:
    policy = config.resolved_sampling_policy_seconds()
    step_ms = policy[str(market["market_family"])] * 1000
    start_ts = int(market["market_start_ts"])
    end_ts = int(market["market_end_ts"])
    times: list[int] = []
    ts = start_ts
    while ts < end_ts:
        times.append(ts)
        ts += step_ms
    return tuple(times)


def mock_orderbook_rows(
    markets: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> list[dict[str, Any]]:
    """Build deterministic complete UP/DOWN executable book rows."""

    rows: list[dict[str, Any]] = []
    for market_index, market in enumerate(markets):
        for sample_index, decision_ts in enumerate(sample_times_for_market(market, config)):
            up_mid = min(0.92, 0.46 + market_index * 0.015 + sample_index * 0.01)
            down_mid = max(0.08, 1.0 - up_mid)
            for outcome, token_id, mid in (
                ("UP", market["up_token_id"], up_mid),
                ("DOWN", market["down_token_id"], down_mid),
            ):
                if (
                    config.inject_missing_down_book
                    and market_index == 0
                    and sample_index == 0
                    and outcome == "DOWN"
                ):
                    continue
                emitted_token_id = token_id
                if (
                    config.inject_unknown_token_book
                    and market_index == 0
                    and sample_index == 0
                    and outcome == "UP"
                ):
                    emitted_token_id = "unknown-token"
                available_at_ts = decision_ts
                if config.inject_stale_book and market_index == 0 and sample_index == 0:
                    available_at_ts = decision_ts + 5_000
                rows.append(
                    {
                        "market_id": market["market_id"],
                        "token_id": emitted_token_id,
                        "outcome": outcome,
                        "ts": decision_ts,
                        "available_at_ts": available_at_ts,
                        "bid_price": round(mid - 0.01, 6),
                        "ask_price": round(mid + 0.01, 6),
                        "mid_price": round(mid, 6),
                        "bid_size": 750.0 + sample_index * 10.0,
                        "ask_size": 720.0 + sample_index * 10.0,
                        "liquidity_depth": 1_470.0 + sample_index * 20.0,
                        **safety_fields(),
                    }
                )
    return rows


def mock_trade_rows(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market_index, market in enumerate(markets):
        for outcome, token_id, price in (
            ("UP", market["up_token_id"], 0.47 + market_index * 0.01),
            ("DOWN", market["down_token_id"], 0.53 - market_index * 0.01),
        ):
            rows.append(
                {
                    "market_id": market["market_id"],
                    "token_id": token_id,
                    "outcome": outcome,
                    "ts": int(market["market_start_ts"]) + 30_000,
                    "available_at_ts": int(market["market_start_ts"]) + 30_000,
                    "price": round(price, 6),
                    "size": 25.0 + market_index,
                    "side": "BUY" if outcome == "UP" else "SELL",
                    **safety_fields(),
                }
            )
    return rows


def validate_market_books(
    *,
    market: dict[str, Any],
    book_rows: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return validated book rows or fail-closed reason codes for one market."""

    reasons: set[str] = set()
    expected_tokens = {
        str(market["up_token_id"]): "UP",
        str(market["down_token_id"]): "DOWN",
    }
    by_outcome: dict[str, list[dict[str, Any]]] = {"UP": [], "DOWN": []}
    for row in book_rows:
        if row.get("market_id") != market["market_id"]:
            continue
        token_id = str(row.get("token_id") or "")
        expected_outcome = expected_tokens.get(token_id)
        if expected_outcome is None:
            reasons.add("unknown_token_id")
            continue
        outcome = str(row.get("outcome") or "").upper()
        if outcome != expected_outcome:
            reasons.add("token_id_outcome_mismatch")
            continue
        book_ts = int(row["ts"])
        available_at_ts = int(row.get("available_at_ts") or book_ts)
        if available_at_ts < book_ts:
            reasons.add("stale_or_future_orderbook")
            continue
        if not _valid_book_prices(row):
            reasons.add("invalid_orderbook_prices")
            continue
        by_outcome[expected_outcome].append(row)
    for rows in by_outcome.values():
        rows.sort(key=lambda item: (int(item["ts"]), int(item.get("available_at_ts") or item["ts"])))
    policy = config.resolved_sampling_policy_seconds()
    max_book_age_ms = policy[str(market["market_family"])] * 1000
    valid_rows: list[dict[str, Any]] = []
    required_sample_times = _required_book_sample_times(
        market=market,
        config=config,
        by_outcome=by_outcome,
    )
    if not required_sample_times:
        reasons.add("missing_complete_up_down_orderbook")
    elif len(required_sample_times) < 2:
        reasons.add("insufficient_decision_timestamps")
    for decision_ts in required_sample_times:
        sampled = {
            outcome: _latest_causal_book_for_sample(
                rows=rows,
                decision_ts=decision_ts,
                max_book_age_ms=max_book_age_ms,
            )
            for outcome, rows in by_outcome.items()
        }
        if sampled["UP"] is None or sampled["DOWN"] is None:
            reasons.add("missing_complete_up_down_orderbook")
            continue
        valid_rows.extend([sampled["UP"], sampled["DOWN"]])
    if reasons:
        return [], sorted(reasons)
    return valid_rows, []


def orderbook_failure_explanation(
    *,
    market: dict[str, Any],
    book_rows: list[dict[str, Any]],
    config: PolymarketRealCorpusRecorderConfig,
) -> dict[str, Any]:
    """Return aggregate book completeness diagnostics without raw book data."""

    expected_tokens = {
        str(market["up_token_id"]): "UP",
        str(market["down_token_id"]): "DOWN",
    }
    by_outcome: dict[str, list[dict[str, Any]]] = {"UP": [], "DOWN": []}
    invalid_counts = {
        "unknown_token_id": 0,
        "token_id_outcome_mismatch": 0,
        "invalid_book_timestamp": 0,
        "stale_or_future_orderbook": 0,
        "invalid_orderbook_prices": 0,
    }
    matched_market_row_count = 0
    for row in book_rows:
        if row.get("market_id") != market["market_id"]:
            continue
        matched_market_row_count += 1
        token_id = str(row.get("token_id") or "")
        expected_outcome = expected_tokens.get(token_id)
        if expected_outcome is None:
            invalid_counts["unknown_token_id"] += 1
            continue
        outcome = str(row.get("outcome") or "").upper()
        if outcome != expected_outcome:
            invalid_counts["token_id_outcome_mismatch"] += 1
            continue
        try:
            book_ts = int(row["ts"])
            available_at_ts = int(row.get("available_at_ts") or book_ts)
        except (KeyError, TypeError, ValueError):
            invalid_counts["invalid_book_timestamp"] += 1
            continue
        if available_at_ts < book_ts:
            invalid_counts["stale_or_future_orderbook"] += 1
            continue
        if not _valid_book_prices(row):
            invalid_counts["invalid_orderbook_prices"] += 1
            continue
        by_outcome[expected_outcome].append(row)
    for rows in by_outcome.values():
        rows.sort(key=lambda item: (int(item["ts"]), int(item.get("available_at_ts") or item["ts"])))

    policy = config.resolved_sampling_policy_seconds()
    max_book_age_ms = policy[str(market["market_family"])] * 1000
    required_sample_times = _required_book_sample_times(
        market=market,
        config=config,
        by_outcome=by_outcome,
    )
    complete_sample_count = 0
    missing_by_outcome = {"UP": 0, "DOWN": 0}
    for decision_ts in required_sample_times:
        sampled = {
            outcome: _latest_causal_book_for_sample(
                rows=rows,
                decision_ts=decision_ts,
                max_book_age_ms=max_book_age_ms,
            )
            for outcome, rows in by_outcome.items()
        }
        if sampled["UP"] is not None and sampled["DOWN"] is not None:
            complete_sample_count += 1
            continue
        if sampled["UP"] is None:
            missing_by_outcome["UP"] += 1
        if sampled["DOWN"] is None:
            missing_by_outcome["DOWN"] += 1

    valid_counts_by_outcome = {
        outcome: len(rows) for outcome, rows in sorted(by_outcome.items())
    }
    min_required = 2
    return {
        "type": "orderbook_completeness",
        "explanation": _orderbook_failure_message(
            valid_counts_by_outcome=valid_counts_by_outcome,
            required_sample_count=len(required_sample_times),
            complete_sample_count=complete_sample_count,
            missing_by_outcome=missing_by_outcome,
            min_required=min_required,
        ),
        "candidate_book_row_count": len(book_rows),
        "matched_market_book_row_count": matched_market_row_count,
        "valid_book_rows_by_outcome": valid_counts_by_outcome,
        "invalid_book_reason_counts": {
            reason: count for reason, count in invalid_counts.items() if count
        },
        "required_decision_timestamp_count": len(required_sample_times),
        "complete_decision_timestamp_count": complete_sample_count,
        "missing_complete_decision_timestamps_by_outcome": {
            outcome: count for outcome, count in missing_by_outcome.items() if count
        },
        "min_required_complete_decision_timestamps": min_required,
        "raw_book_rows_persisted": False,
    }


def validate_trade_rows(
    *,
    market: dict[str, Any],
    trade_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: set[str] = set()
    expected_tokens = {
        str(market["up_token_id"]): "UP",
        str(market["down_token_id"]): "DOWN",
    }
    valid: list[dict[str, Any]] = []
    for row in trade_rows:
        if row.get("market_id") != market["market_id"]:
            continue
        token_id = str(row.get("token_id") or "")
        expected_outcome = expected_tokens.get(token_id)
        if expected_outcome is None:
            reasons.add("unknown_trade_token_id")
            continue
        if str(row.get("outcome") or "").upper() != expected_outcome:
            reasons.add("trade_token_id_outcome_mismatch")
            continue
        if str(row.get("side") or "").upper() not in {"BUY", "SELL"}:
            reasons.add("unknown_trade_side")
            continue
        if not _finite_positive(row.get("price")) or not _finite_non_negative(row.get("size")):
            reasons.add("invalid_trade_price_or_size")
            continue
        valid.append(row)
    return valid, sorted(reasons)


def _valid_book_prices(row: dict[str, Any]) -> bool:
    bid = _to_float(row.get("bid_price"))
    ask = _to_float(row.get("ask_price"))
    mid = _to_float(row.get("mid_price"))
    return (
        bid is not None
        and ask is not None
        and mid is not None
        and 0.0 < bid <= ask <= 1.0
        and 0.0 < mid <= 1.0
        and _finite_non_negative(row.get("bid_size"))
        and _finite_non_negative(row.get("ask_size"))
        and _finite_non_negative(row.get("liquidity_depth"))
    )


def _latest_causal_book_for_sample(
    *,
    rows: list[dict[str, Any]],
    decision_ts: int,
    max_book_age_ms: int,
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if int(row["ts"]) <= decision_ts
        and int(row.get("available_at_ts") or row["ts"]) <= decision_ts
        and decision_ts - int(row["ts"]) < max_book_age_ms
    ]
    return eligible[-1] if eligible else None


def _required_book_sample_times(
    *,
    market: dict[str, Any],
    config: PolymarketRealCorpusRecorderConfig,
    by_outcome: dict[str, list[dict[str, Any]]],
) -> tuple[int, ...]:
    sample_times = sample_times_for_market(market, config)
    if config.mock_public_data:
        return sample_times
    if not by_outcome["UP"] or not by_outcome["DOWN"]:
        return sample_times
    collection_start_ts = max(
        min(int(row.get("available_at_ts") or row["ts"]) for row in by_outcome["UP"]),
        min(int(row.get("available_at_ts") or row["ts"]) for row in by_outcome["DOWN"]),
    )
    collection_end_ts = _collection_end_ts_for_live_rows(
        market=market,
        by_outcome=by_outcome,
    )
    return tuple(
        decision_ts
        for decision_ts in sample_times
        if collection_start_ts <= decision_ts <= collection_end_ts
    )


def _collection_end_ts_for_live_rows(
    *,
    market: dict[str, Any],
    by_outcome: dict[str, list[dict[str, Any]]],
) -> int:
    by_outcome_collection_ends: list[int] = []
    for rows in by_outcome.values():
        row_collection_ends = [
            int(row["collection_end_ts"])
            for row in rows
            if row.get("collection_end_ts") is not None
        ]
        if row_collection_ends:
            by_outcome_collection_ends.append(max(row_collection_ends))
    if by_outcome_collection_ends:
        return min(by_outcome_collection_ends)
    return int(market["market_end_ts"])


def _orderbook_failure_message(
    *,
    valid_counts_by_outcome: dict[str, int],
    required_sample_count: int,
    complete_sample_count: int,
    missing_by_outcome: dict[str, int],
    min_required: int,
) -> str:
    if valid_counts_by_outcome["UP"] == 0 and valid_counts_by_outcome["DOWN"] == 0:
        return "No valid UP or DOWN orderbook rows were available for this market."
    if valid_counts_by_outcome["UP"] == 0:
        return "No valid UP orderbook rows were available for this market."
    if valid_counts_by_outcome["DOWN"] == 0:
        return "No valid DOWN orderbook rows were available for this market."
    if required_sample_count < min_required:
        return (
            "Orderbook rows were available, but they covered fewer than "
            f"{min_required} required decision timestamps."
        )
    if complete_sample_count < min_required:
        missing = ", ".join(
            f"{outcome} missing at {count} timestamp(s)"
            for outcome, count in sorted(missing_by_outcome.items())
            if count
        )
        suffix = f" ({missing})." if missing else "."
        return (
            "Orderbook rows were available, but fewer than "
            f"{min_required} complete UP/DOWN decision timestamps could be formed"
            f"{suffix}"
        )
    return "Orderbook rows failed validation before a complete causal book could be formed."


def _finite_positive(value: Any) -> bool:
    numeric = _to_float(value)
    return numeric is not None and numeric > 0.0


def _finite_non_negative(value: Any) -> bool:
    numeric = _to_float(value)
    return numeric is not None and numeric >= 0.0


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
