"""Polymarket binary pricing and positive-EV signal engine.

Combines a Black-Scholes cash-or-nothing probability (with an additive OFI
alpha layer), oracle TWAP effective-strike magnification, and fractional
Kelly sizing. Both YES and NO asks are scored; the larger qualifying edge
wins. Tail seconds of a window hard-block every new entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
DEFAULT_OFI_GAMMA = 0.0015
DEFAULT_MIN_EDGE_5M = 0.08
DEFAULT_MIN_EDGE_15M = 0.05
DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_TAIL_CUTOFF_MS = 30_000
DEFAULT_PROBABILITY_FLOOR = 0.001
_VOL_TIME_FLOOR = 1e-18


class SignalDirection(Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class MarketWindow:
    """预测窗口元数据"""

    window_id: str
    symbol: str
    strike_price: float
    start_ts_ms: int
    end_ts_ms: int
    window_type: str  # "5m" or "15m"


@dataclass(frozen=True, slots=True)
class PricingSignal:
    """定价与信号输出快照。

    ``model_prob`` / ``market_price`` / ``edge`` are the selected side
    (YES on HOLD). ``ev`` is the selected-side expected return rate
    ``(p / ask) - 1``, or ``0.0`` when the engine does not trade.
    """

    ts_ms: int
    window_id: str
    spot_price: float
    effective_strike: float
    model_prob: float
    market_price: float
    edge: float
    ev: float
    direction: SignalDirection
    recommended_size_pct: float


def effective_strike(
    *,
    strike_price: float,
    oracle_twap_so_far: float,
    twap_weight: float,
) -> float:
    """Implied remaining-path strike that still matches the window TWAP.

    ``K_eff = (K - w * P_twap) / (1 - w)`` for ``w ∈ [0, 1)``. A fully
    sampled window is deterministic from the completed oracle TWAP: it returns
    ``0`` when YES has won and ``+inf`` when NO has won.
    """

    strike = _finite_float("strike_price", strike_price)
    twap = _finite_float("oracle_twap_so_far", oracle_twap_so_far)
    weight = _twap_weight(twap_weight)
    if strike <= 0.0 or twap <= 0.0:
        raise ValueError("strike_price and oracle_twap_so_far must be positive")
    if weight == 1.0:
        return 0.0 if twap >= strike else math.inf
    remaining_weight = 1.0 - weight
    if remaining_weight <= 0.0 or not math.isfinite(remaining_weight):
        return strike
    k_eff = (strike - weight * twap) / remaining_weight
    if not math.isfinite(k_eff):
        return strike
    return k_eff


class PolymarketPricingEngine:
    """5m/15m binary win-probability, two-sided edge, and 1/4-Kelly sizer."""

    __slots__ = (
        "ofi_gamma",
        "min_edge_5m",
        "min_edge_15m",
        "kelly_fraction",
        "tail_cutoff_ms",
    )

    def __init__(
        self,
        *,
        ofi_gamma: float = DEFAULT_OFI_GAMMA,
        min_edge_5m: float = DEFAULT_MIN_EDGE_5M,
        min_edge_15m: float = DEFAULT_MIN_EDGE_15M,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        tail_cutoff_ms: int = DEFAULT_TAIL_CUTOFF_MS,
    ) -> None:
        gamma = _finite_float("ofi_gamma", ofi_gamma)
        edge_5m = _finite_float("min_edge_5m", min_edge_5m)
        edge_15m = _finite_float("min_edge_15m", min_edge_15m)
        fraction = _finite_float("kelly_fraction", kelly_fraction)
        cutoff = int(tail_cutoff_ms)
        if gamma < 0.0:
            raise ValueError("ofi_gamma must be non-negative")
        if not 0.0 <= edge_5m <= 1.0:
            raise ValueError("min_edge_5m must be in [0, 1]")
        if not 0.0 <= edge_15m <= 1.0:
            raise ValueError("min_edge_15m must be in [0, 1]")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("kelly_fraction must be in [0, 1]")
        if cutoff < 0:
            raise ValueError("tail_cutoff_ms must be non-negative")
        self.ofi_gamma = gamma
        self.min_edge_5m = edge_5m
        self.min_edge_15m = edge_15m
        self.kelly_fraction = fraction
        self.tail_cutoff_ms = cutoff

    def effective_strike(
        self,
        *,
        strike_price: float,
        oracle_twap_so_far: float,
        twap_weight: float,
    ) -> float:
        return effective_strike(
            strike_price=strike_price,
            oracle_twap_so_far=oracle_twap_so_far,
            twap_weight=twap_weight,
        )

    def calculate_probability(
        self,
        *,
        spot_price: float,
        effective_strike: float,
        time_to_expiry_sec: float,
        volatility_annualized: float,
        z_ofi: float,
    ) -> float:
        spot = float(spot_price)
        strike = float(effective_strike)
        t_sec = float(time_to_expiry_sec)
        vol = float(volatility_annualized)
        z_value = float(z_ofi)
        if not math.isfinite(z_value):
            z_value = 0.0

        if not math.isfinite(t_sec) or t_sec <= 0.0:
            return _expired_probability(spot, strike)

        if not math.isfinite(spot) or spot <= 0.0:
            raise ValueError("spot_price must be finite and positive")
        if math.isnan(strike):
            raise ValueError("effective_strike must not be NaN")
        if not math.isfinite(vol) or vol < 0.0:
            raise ValueError("volatility_annualized must be finite and non-negative")
        if strike <= 0.0:
            return 1.0 - DEFAULT_PROBABILITY_FLOOR
        if math.isinf(strike):
            return DEFAULT_PROBABILITY_FLOOR

        t_years = t_sec / SECONDS_PER_YEAR
        if t_years <= 0.0 or not math.isfinite(t_years):
            return _expired_probability(spot, strike)

        if vol == 0.0:
            base_probability = _expired_probability(spot, strike)
            return _apply_probability_alpha(
                base_probability,
                gamma=self.ofi_gamma,
                z_ofi=z_value,
                t_years=t_years,
            )

        denom = vol * math.sqrt(t_years)
        if denom <= _VOL_TIME_FLOOR or not math.isfinite(denom):
            base_probability = _expired_probability(spot, strike)
            return _apply_probability_alpha(
                base_probability,
                gamma=self.ofi_gamma,
                z_ofi=z_value,
                t_years=t_years,
            )

        log_moneyness = math.log(spot / strike)
        d2 = (log_moneyness - 0.5 * vol * vol * t_years) / denom
        if not math.isfinite(d2):
            return _expired_probability(spot, strike)
        probability = _norm_cdf(d2)
        if not math.isfinite(probability):
            return _expired_probability(spot, strike)
        return _apply_probability_alpha(
            probability,
            gamma=self.ofi_gamma,
            z_ofi=z_value,
            t_years=t_years,
        )

    def evaluate_signal(
        self,
        *,
        window: MarketWindow,
        current_ts_ms: int,
        spot_price: float,
        oracle_twap_so_far: float,
        twap_weight: float,
        z_ofi: float,
        volatility_annualized: float,
        yes_ask_price: float,
        no_ask_price: float,
    ) -> PricingSignal:
        """Score YES and NO asks and emit the better qualifying trade.

        Tail seconds (``remaining_ms <= tail_cutoff_ms``) force ``HOLD``
        even when either side has a large edge. Kelly sizing and ``ev``
        use the selected probability and ask; ``ev`` is the expected
        return rate ``(p / ask) - 1``, or ``0.0`` on HOLD.
        """

        spot = _positive_float("spot_price", spot_price)
        yes_ask = _market_price("yes_ask_price", yes_ask_price)
        no_ask = _market_price("no_ask_price", no_ask_price)
        vol = _finite_float("volatility_annualized", volatility_annualized)
        if vol < 0.0:
            raise ValueError("volatility_annualized must be non-negative")
        ts_ms = int(current_ts_ms)
        weight = _twap_weight(twap_weight)
        k_eff = self.effective_strike(
            strike_price=window.strike_price,
            oracle_twap_so_far=oracle_twap_so_far,
            twap_weight=weight,
        )
        remaining_ms = int(window.end_ts_ms) - ts_ms
        if weight == 1.0:
            p_yes = (
                1.0
                if float(oracle_twap_so_far) >= float(window.strike_price)
                else 0.0
            )
        else:
            p_yes = self.calculate_probability(
                spot_price=spot,
                effective_strike=k_eff,
                time_to_expiry_sec=remaining_ms / 1000.0,
                volatility_annualized=vol,
                z_ofi=z_ofi,
            )
        p_no = 1.0 - p_yes
        yes_edge = p_yes - yes_ask
        no_edge = p_no - no_ask
        in_tail = remaining_ms <= self.tail_cutoff_ms
        min_edge = self._min_edge_for(window.window_type)

        direction = SignalDirection.HOLD
        selected_p = p_yes
        selected_ask = yes_ask
        selected_edge = yes_edge
        if not in_tail:
            if yes_edge >= min_edge and yes_edge >= no_edge:
                direction = SignalDirection.BUY_YES
            elif no_edge >= min_edge and no_edge > yes_edge:
                direction = SignalDirection.BUY_NO
                selected_p = p_no
                selected_ask = no_ask
                selected_edge = no_edge

        size = 0.0
        ev = 0.0
        if direction is not SignalDirection.HOLD:
            size = self.kelly_fraction * _binary_kelly(selected_p, selected_ask)
            if not math.isfinite(size) or size <= 0.0:
                size = 0.0
                direction = SignalDirection.HOLD
                selected_p = p_yes
                selected_ask = yes_ask
                selected_edge = yes_edge
            else:
                ev = _expected_return_rate(selected_p, selected_ask)

        return PricingSignal(
            ts_ms=ts_ms,
            window_id=window.window_id,
            spot_price=spot,
            effective_strike=k_eff,
            model_prob=selected_p,
            market_price=selected_ask,
            edge=selected_edge,
            ev=ev,
            direction=direction,
            recommended_size_pct=size,
        )

    def _min_edge_for(self, window_type: str) -> float:
        if str(window_type).strip().lower() == "15m":
            return self.min_edge_15m
        return self.min_edge_5m


def _finite_float(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _positive_float(name: str, value: float) -> float:
    out = _finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _market_price(name: str, value: float) -> float:
    out = _finite_float(name, value)
    if not 0.0 < out <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return out


def _twap_weight(value: float) -> float:
    weight = _finite_float("twap_weight", value)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("twap_weight must be in [0, 1]")
    return weight


def _expired_probability(spot: float, strike: float) -> float:
    if not math.isfinite(spot) or not math.isfinite(strike):
        return 0.0
    return 1.0 if spot >= strike else 0.0


def _apply_probability_alpha(
    probability: float,
    *,
    gamma: float,
    z_ofi: float,
    t_years: float,
) -> float:
    adjusted = probability + gamma * z_ofi * math.sqrt(t_years)
    if not math.isfinite(adjusted):
        adjusted = probability
    floor = DEFAULT_PROBABILITY_FLOOR
    return min(1.0 - floor, max(floor, adjusted))


def _norm_cdf(x: float) -> float:
    if not math.isfinite(x):
        return 1.0 if x > 0.0 else 0.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_return_rate(probability: float, ask: float) -> float:
    """Expected ROI of a $1 binary bought at ``ask``: ``(p / ask) - 1``."""

    if not math.isfinite(probability) or not math.isfinite(ask) or ask <= 0.0:
        return 0.0
    ev = probability / ask - 1.0
    if not math.isfinite(ev):
        return 0.0
    return ev


def _binary_kelly(probability: float, price: float) -> float:
    """Full Kelly fraction for a $1 binary, clamped to ``[0, 1]``."""

    if not math.isfinite(probability) or not math.isfinite(price):
        return 0.0
    if price <= 0.0 or price >= 1.0:
        return 0.0
    edge = probability - price
    if edge <= 0.0:
        return 0.0
    fraction = edge / (1.0 - price)
    if not math.isfinite(fraction):
        return 0.0
    return min(1.0, max(0.0, fraction))
