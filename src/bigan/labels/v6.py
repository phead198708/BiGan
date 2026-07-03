"""V6 corpus label helpers for settlement and volatility heads."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

SETTLEMENT_LABEL_UP = "UP"
SETTLEMENT_LABEL_DOWN = "DOWN"
SETTLEMENT_LABEL_NEUTRAL = "NEUTRAL"

VOLATILITY_PATH_VALID = "valid"
VOLATILITY_PATH_MISSING_PRICE_PATH = "missing_price_path"
VOLATILITY_PATH_MISSING_ENTRY_QUOTE = "missing_entry_quote"
VOLATILITY_PATH_NO_EXIT_WINDOW = "no_exit_window"
VOLATILITY_PATH_MISSING_EXIT_PATH = "missing_exit_path"
VOLATILITY_PATH_ENTRY_PRICE_BELOW_MIN = "entry_price_below_min"

DEFAULT_VOLATILITY_THRESHOLD_CANDIDATES: tuple[float, ...] = (
    0.08,
    0.10,
    0.12,
    0.15,
    0.20,
)


@dataclass(frozen=True, slots=True)
class VolatilityLabelConfig:
    """Execution-aware volatility label settings.

    Prices are top-of-book ask for entry and bid for exit. The worst executable
    price applies the same buy/sell slippage policy used by the Phase 4 signal
    opportunity analyzer.
    """

    min_exit_gain: float = 0.15
    buy_slippage: float = 0.02
    sell_slippage: float = 0.02
    max_entry_wait_ms: int = 60_000
    min_exit_seconds_before_expiry: float = 300.0
    min_entry_price: float = 0.35
    fee_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.min_exit_gain < 0.0:
            raise ValueError("min_exit_gain must be non-negative")
        if self.buy_slippage < 0.0 or self.sell_slippage < 0.0:
            raise ValueError("slippage must be non-negative")
        if self.max_entry_wait_ms < 0:
            raise ValueError("max_entry_wait_ms must be non-negative")
        if self.min_exit_seconds_before_expiry < 0.0:
            raise ValueError("min_exit_seconds_before_expiry must be non-negative")
        if self.min_entry_price < 0.0:
            raise ValueError("min_entry_price must be non-negative")
        if self.fee_bps < 0.0:
            raise ValueError("fee_bps must be non-negative")


@dataclass(frozen=True, slots=True)
class VolatilityPathLabel:
    """One side's forward price-path volatility label."""

    label: bool | None
    max_exit_gain: float | None
    max_exit_return_per_usdc: float | None
    time_to_best_exit_seconds: float | None
    best_exit_price: float | None
    best_exit_bid: float | None
    best_exit_ts: int | None
    entry_quote_ts: int | None
    entry_ask: float | None
    entry_worst_price: float | None
    exit_deadline_ts: int
    path_validity_flag: str


def settlement_margin(start_price: float, target_price: float) -> float:
    """Return the underlying settlement move used by v6 3-way labels."""

    return float(target_price) - float(start_price)


def settlement_3way_label(
    start_price: float,
    target_price: float,
    *,
    neutral_margin_abs: float = 0.0,
) -> str:
    """Classify settlement direction with a low-margin abstention band.

    UP/DOWN remain direction labels from the underlying round settlement. The
    NEUTRAL class is an abstention-quality class for small absolute moves; it is
    intentionally independent from token profitability.
    """

    if neutral_margin_abs < 0.0:
        raise ValueError("neutral_margin_abs must be non-negative")
    margin = settlement_margin(start_price, target_price)
    if not math.isfinite(margin) or abs(margin) <= neutral_margin_abs:
        return SETTLEMENT_LABEL_NEUTRAL
    return SETTLEMENT_LABEL_UP if margin > 0.0 else SETTLEMENT_LABEL_DOWN


def compute_volatility_path_label(
    quotes: Iterable[Any],
    *,
    decision_ts: int,
    round_end_ts: int,
    config: VolatilityLabelConfig | None = None,
) -> VolatilityPathLabel:
    """Compute the execution-aware volatility label for one token side.

    The feature snapshot is at ``decision_ts``. Entry may use a quote at or
    after that timestamp within ``max_entry_wait_ms``; all exit-path quotes are
    strictly after ``decision_ts`` and no earlier than the entry quote.
    """

    cfg = config or VolatilityLabelConfig()
    sorted_quotes = sorted(
        (_normalise_quote(quote) for quote in quotes),
        key=lambda quote: quote["ts"],
    )
    exit_deadline_ts = int(round_end_ts - int(cfg.min_exit_seconds_before_expiry * 1000))
    if exit_deadline_ts <= int(decision_ts):
        return _empty_volatility_label(exit_deadline_ts, VOLATILITY_PATH_NO_EXIT_WINDOW)

    entry_quote = _entry_quote(sorted_quotes, int(decision_ts), cfg.max_entry_wait_ms)
    if entry_quote is None:
        return _empty_volatility_label(exit_deadline_ts, VOLATILITY_PATH_MISSING_ENTRY_QUOTE)

    entry_ask = entry_quote["ask"]
    entry_worst = min(0.99, entry_ask + cfg.buy_slippage + _fee(entry_ask, cfg.fee_bps))
    exit_quotes = [
        quote
        for quote in sorted_quotes
        if quote["ts"] > int(decision_ts)
        and quote["ts"] >= entry_quote["ts"]
        and quote["ts"] <= exit_deadline_ts
        and quote["bid"] is not None
        and math.isfinite(quote["bid"])
    ]
    if not exit_quotes:
        return VolatilityPathLabel(
            label=None,
            max_exit_gain=None,
            max_exit_return_per_usdc=None,
            time_to_best_exit_seconds=None,
            best_exit_price=None,
            best_exit_bid=None,
            best_exit_ts=None,
            entry_quote_ts=entry_quote["ts"],
            entry_ask=entry_ask,
            entry_worst_price=entry_worst,
            exit_deadline_ts=exit_deadline_ts,
            path_validity_flag=VOLATILITY_PATH_MISSING_EXIT_PATH,
        )

    best = max(exit_quotes, key=lambda quote: quote["bid"])
    best_exit_price = max(0.01, best["bid"] - cfg.sell_slippage - _fee(best["bid"], cfg.fee_bps))
    max_exit_gain = best_exit_price - entry_worst
    max_exit_return_per_usdc = (
        (best_exit_price / entry_worst) - 1.0
        if entry_worst > 0.0
        else None
    )
    flag = (
        VOLATILITY_PATH_ENTRY_PRICE_BELOW_MIN
        if entry_worst < cfg.min_entry_price
        else VOLATILITY_PATH_VALID
    )
    label = bool(
        flag == VOLATILITY_PATH_VALID
        and max_exit_gain + 1e-12 >= cfg.min_exit_gain
    )
    return VolatilityPathLabel(
        label=label,
        max_exit_gain=max_exit_gain,
        max_exit_return_per_usdc=max_exit_return_per_usdc,
        time_to_best_exit_seconds=(best["ts"] - entry_quote["ts"]) / 1000.0,
        best_exit_price=best_exit_price,
        best_exit_bid=best["bid"],
        best_exit_ts=best["ts"],
        entry_quote_ts=entry_quote["ts"],
        entry_ask=entry_ask,
        entry_worst_price=entry_worst,
        exit_deadline_ts=exit_deadline_ts,
        path_validity_flag=flag,
    )


def two_sided_volatility_fields(
    *,
    quotes_by_side: Mapping[str, Iterable[Any]],
    decision_ts: int,
    round_end_ts: int,
    config: VolatilityLabelConfig | None = None,
) -> dict[str, Any]:
    """Return labels_15m_v1-compatible UP/DOWN volatility fields."""

    fields: dict[str, Any] = {}
    for side in ("up", "down"):
        result = compute_volatility_path_label(
            quotes_by_side.get(side.upper()) or quotes_by_side.get(side) or (),
            decision_ts=decision_ts,
            round_end_ts=round_end_ts,
            config=config,
        )
        fields[f"max_exit_gain_{side}"] = result.max_exit_gain
        fields[f"max_exit_return_per_usdc_{side}"] = result.max_exit_return_per_usdc
        fields[f"time_to_best_exit_{side}"] = result.time_to_best_exit_seconds
        fields[f"best_exit_price_{side}"] = result.best_exit_price
        fields[f"label_volatility_{side}"] = result.label
        fields[f"volatility_path_validity_{side}"] = result.path_validity_flag
    return fields


def empty_volatility_fields(
    *,
    path_validity_flag: str = VOLATILITY_PATH_MISSING_PRICE_PATH,
) -> dict[str, Any]:
    """Return nullable volatility fields for rows without raw book coverage."""

    fields: dict[str, Any] = {}
    for side in ("up", "down"):
        fields[f"max_exit_gain_{side}"] = None
        fields[f"max_exit_return_per_usdc_{side}"] = None
        fields[f"time_to_best_exit_{side}"] = None
        fields[f"best_exit_price_{side}"] = None
        fields[f"label_volatility_{side}"] = None
        fields[f"volatility_path_validity_{side}"] = path_validity_flag
    return fields


def _empty_volatility_label(exit_deadline_ts: int, flag: str) -> VolatilityPathLabel:
    return VolatilityPathLabel(
        label=None,
        max_exit_gain=None,
        max_exit_return_per_usdc=None,
        time_to_best_exit_seconds=None,
        best_exit_price=None,
        best_exit_bid=None,
        best_exit_ts=None,
        entry_quote_ts=None,
        entry_ask=None,
        entry_worst_price=None,
        exit_deadline_ts=exit_deadline_ts,
        path_validity_flag=flag,
    )


def _entry_quote(
    quotes: list[dict[str, float | int | None]],
    decision_ts: int,
    max_entry_wait_ms: int,
) -> dict[str, float | int | None] | None:
    deadline = decision_ts + max_entry_wait_ms
    for quote in quotes:
        ts = int(quote["ts"])
        ask = quote["ask"]
        if ts < decision_ts:
            continue
        if ts > deadline:
            break
        if ask is not None and math.isfinite(ask):
            return quote
    return None


def _normalise_quote(quote: Any) -> dict[str, float | int | None]:
    ts = _quote_value(quote, "ts")
    if ts is None:
        raise ValueError("quote is missing ts")
    return {
        "ts": int(ts),
        "bid": _finite_float_or_none(_quote_value(quote, "bid", "bid_price")),
        "ask": _finite_float_or_none(_quote_value(quote, "ask", "ask_price")),
    }


def _quote_value(quote: Any, *names: str) -> Any:
    if isinstance(quote, Mapping):
        for name in names:
            if name in quote:
                return quote[name]
        return None
    for name in names:
        if hasattr(quote, name):
            return getattr(quote, name)
    return None


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fee(price: float, fee_bps: float) -> float:
    return price * (fee_bps / 10_000.0)
