"""Shared action-value policy guard helpers for Polymarket execution."""

from __future__ import annotations

from typing import Any

ACTION_FAMILY_HOLD_TO_SETTLEMENT = "HOLD_TO_SETTLEMENT"
ACTION_FAMILY_SELL_BEFORE_CLOSE = "SELL_BEFORE_CLOSE"
ACTION_FAMILY_NO_TRADE = "NO_TRADE"
ACTION_SIDE_BUY_UP = "BUY_UP"
ACTION_SIDE_BUY_DOWN = "BUY_DOWN"
ACTION_SIDE_NO_TRADE = "NO_TRADE"

ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE = "action_family_high_score_unprofitable"
HOLD_TO_SETTLEMENT_HIGH_SCORE_UNPROFITABLE = (
    "hold_to_settlement_high_score_unprofitable"
)
BUY_UP_HOLD_TO_SETTLEMENT_UNPROFITABLE = "buy_up_hold_to_settlement_unprofitable"
BUY_DOWN_HOLD_TO_SETTLEMENT_UNPROFITABLE = "buy_down_hold_to_settlement_unprofitable"
HOLD_TO_SETTLEMENT_LONGSHOT_GUARD = "hold_to_settlement_longshot_guard"
ACTION_FAMILY_INELIGIBLE = "action_family_ineligible"

LONGSHOT_GUARD_PRICE_BUCKETS = ("<0.20", "0.20-0.40")
LONGSHOT_GUARD_TIME_TO_CLOSE_BUCKETS = ("1-3m", "3-5m")
LONGSHOT_GUARD_RAW_SCORE_BUCKETS = (">=0.15", "0.05-0.15")


def action_value_action_family(action: str) -> str:
    """Return the execution family for an action-value label action."""

    if action.endswith("_HOLD_TO_SETTLEMENT"):
        return ACTION_FAMILY_HOLD_TO_SETTLEMENT
    if action.endswith("_SELL_BEFORE_CLOSE"):
        return ACTION_FAMILY_SELL_BEFORE_CLOSE
    return ACTION_FAMILY_NO_TRADE


def action_value_action_side(action: str) -> str:
    """Return BUY_UP / BUY_DOWN side for an action-value label action."""

    if action.startswith("BUY_UP_"):
        return ACTION_SIDE_BUY_UP
    if action.startswith("BUY_DOWN_"):
        return ACTION_SIDE_BUY_DOWN
    return ACTION_SIDE_NO_TRADE


def action_value_intended_exit_policy(action: str) -> str:
    """Map an action-value label action to runtime intended_exit_policy."""

    if action.endswith("_HOLD_TO_SETTLEMENT"):
        return "hold_to_settlement"
    if action.endswith("_SELL_BEFORE_CLOSE"):
        return "sell_before_close"
    return "none"


def action_value_price_bucket(*, action: str, features: dict[str, Any]) -> str:
    """Bucket the executable entry price used by the action."""

    price = None
    if action.startswith("BUY_UP_"):
        price = features.get("up_ask")
    elif action.startswith("BUY_DOWN_"):
        price = features.get("down_ask")
    elif action == "NO_TRADE":
        return "none"
    if price is None:
        return "unknown"
    return _number_bucket(
        float(price),
        thresholds=(0.20, 0.40, 0.60, 0.80),
        labels=("<0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", ">=0.80"),
    )


def action_value_time_to_close_bucket(features: dict[str, Any]) -> str:
    """Bucket causal time-to-close seconds."""

    seconds = float(features.get("time_to_close_seconds", 0.0))
    return _number_bucket(
        seconds,
        thresholds=(30.0, 60.0, 180.0, 300.0, 900.0),
        labels=("0-30s", "30-60s", "1-3m", "3-5m", "5-15m", "15m+"),
    )


def action_value_raw_score_bucket(raw_score: float) -> str:
    """Bucket raw, uncalibrated action expected return."""

    return _number_bucket(
        float(raw_score),
        thresholds=(-0.10, 0.0, 0.05, 0.15),
        labels=("<-0.10", "-0.10-0.00", "0.00-0.05", "0.05-0.15", ">=0.15"),
    )


def hold_to_settlement_longshot_guard_applies(
    *,
    action: str,
    features: dict[str, Any],
    raw_score: float,
) -> bool:
    """Return true when the initial long-shot HOLD_TO_SETTLEMENT guard applies."""

    return (
        action_value_intended_exit_policy(action) == "hold_to_settlement"
        and action_value_price_bucket(action=action, features=features)
        in LONGSHOT_GUARD_PRICE_BUCKETS
        and action_value_time_to_close_bucket(features)
        in LONGSHOT_GUARD_TIME_TO_CLOSE_BUCKETS
        and action_value_raw_score_bucket(raw_score)
        in LONGSHOT_GUARD_RAW_SCORE_BUCKETS
    )


def action_value_bucket_payload(
    *,
    action: str,
    features: dict[str, Any],
    raw_score: float,
) -> dict[str, Any]:
    """Return deterministic action-family bucket metadata."""

    return {
        "action": action,
        "action_family": action_value_action_family(action),
        "side": action_value_action_side(action),
        "intended_exit_policy": action_value_intended_exit_policy(action),
        "price_bucket": action_value_price_bucket(action=action, features=features),
        "time_to_close_bucket": action_value_time_to_close_bucket(features),
        "raw_score_bucket": action_value_raw_score_bucket(raw_score),
        "hold_to_settlement_longshot_guard_applies": (
            hold_to_settlement_longshot_guard_applies(
                action=action,
                features=features,
                raw_score=raw_score,
            )
        ),
    }


def _number_bucket(
    value: float,
    *,
    thresholds: tuple[float, ...],
    labels: tuple[str, ...],
) -> str:
    for threshold, label in zip(thresholds, labels, strict=False):
        if value < threshold:
            return label
    return labels[-1]
