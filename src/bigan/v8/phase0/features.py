"""Causal-only feature construction for v8 Phase 0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from bigan.v8.phase0.alignment import AlignedMarketSeries, TimeAlignmentEngine
from bigan.v8.phase0.contracts import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    FeatureProvenance,
    FeatureVector,
    MarketData,
)

MINUTE_MS = 60_000


@dataclass(frozen=True, slots=True)
class CausalFeatureBuilderConfig:
    """Feature-builder knobs that preserve point-in-time causality."""

    windows_ms: tuple[int, ...] = (MINUTE_MS, 5 * MINUTE_MS, 15 * MINUTE_MS)
    decision_frequency_ms: int | None = None
    feature_version: str = FEATURE_VERSION

    def __post_init__(self) -> None:
        if not self.windows_ms:
            raise ValueError("windows_ms cannot be empty")
        if any(window <= 0 for window in self.windows_ms):
            raise ValueError("all windows_ms values must be positive")
        if self.decision_frequency_ms is not None and self.decision_frequency_ms <= 0:
            raise ValueError("decision_frequency_ms must be positive")


class CausalFeatureBuilder:
    """Build deterministic, backward-looking feature vectors."""

    def __init__(
        self,
        config: CausalFeatureBuilderConfig | None = None,
        *,
        alignment_engine: TimeAlignmentEngine | None = None,
    ) -> None:
        self.config = config or CausalFeatureBuilderConfig()
        self.alignment_engine = alignment_engine or TimeAlignmentEngine()

    def build(
        self,
        market_data: list[MarketData] | AlignedMarketSeries,
        *,
        decision_times: dict[tuple[str, str], tuple[int, ...]] | None = None,
    ) -> list[FeatureVector]:
        series = (
            market_data
            if isinstance(market_data, AlignedMarketSeries)
            else self.alignment_engine.align_market_data(market_data)
        )
        vectors: list[FeatureVector] = []
        for source, instrument_id in sorted(series.groups):
            times = (
                decision_times.get((source, instrument_id), ())
                if decision_times is not None
                else series.decision_times(
                    source,
                    instrument_id,
                    frequency_ms=self.config.decision_frequency_ms,
                )
            )
            for decision_ts in times:
                latest = series.latest_at(source, instrument_id, decision_ts)
                if latest is None:
                    continue
                vectors.append(
                    self._build_one(
                        series=series,
                        latest=latest,
                        decision_ts=decision_ts,
                    )
                )
        return sorted(
            vectors,
            key=lambda row: (row.source, row.instrument_id, row.decision_ts),
        )

    def _build_one(
        self,
        *,
        series: AlignedMarketSeries,
        latest: MarketData,
        decision_ts: int,
    ) -> FeatureVector:
        source = latest.source
        instrument_id = latest.instrument_id
        max_window = max(self.config.windows_ms)
        lookback_start = max(0, decision_ts - max_window)
        full_window = series.window(
            source,
            instrument_id,
            start_exclusive=lookback_start,
            end_inclusive=decision_ts,
        )
        if latest not in full_window:
            full_window = (*full_window, latest)
        max_input_ts = max(row.ts for row in full_window)

        spread = _spread(latest)
        mid_price = latest.effective_mid_price
        spread_bps = None if spread is None else spread / mid_price
        features: dict[str, float | int | None] = {
            "mid_price": mid_price,
            "spread": spread,
            "spread_bps": spread_bps,
            "return_1m": _return(series, latest, decision_ts, MINUTE_MS),
            "return_5m": _return(series, latest, decision_ts, 5 * MINUTE_MS),
            "return_15m": _return(series, latest, decision_ts, 15 * MINUTE_MS),
            "volatility_5m": _realized_vol(series, latest, decision_ts, 5 * MINUTE_MS),
            "volatility_15m": _realized_vol(series, latest, decision_ts, 15 * MINUTE_MS),
            "volume_1m": _window_sum(series, latest, decision_ts, MINUTE_MS, "volume"),
            "volume_5m": _window_sum(series, latest, decision_ts, 5 * MINUTE_MS, "volume"),
            "trade_count_1m": int(
                _window_sum(series, latest, decision_ts, MINUTE_MS, "trade_count")
            ),
            "trade_count_5m": int(
                _window_sum(series, latest, decision_ts, 5 * MINUTE_MS, "trade_count")
            ),
            "orderbook_imbalance_l1": _orderbook_imbalance(latest),
            "liquidity_depth": latest.liquidity_depth,
            "minute_of_day": _minute_of_day(decision_ts),
            "day_of_week": _day_of_week(decision_ts),
        }

        provenance = {
            name: _provenance_for(
                name=name,
                series=series,
                latest=latest,
                decision_ts=decision_ts,
                lookback_ms=_feature_lookback_ms(name),
            )
            for name in FEATURE_COLUMNS
        }

        return FeatureVector(
            decision_ts=decision_ts,
            feature_cutoff_ts=decision_ts,
            lookback_start_ts=lookback_start,
            max_input_ts=max_input_ts,
            source=source,
            instrument_id=instrument_id,
            feature_version=self.config.feature_version,
            features=features,
            provenance=provenance,
        )


def _spread(row: MarketData) -> float | None:
    if row.bid_price is None or row.ask_price is None:
        return None
    return max(0.0, row.ask_price - row.bid_price)


def _return(
    series: AlignedMarketSeries,
    latest: MarketData,
    decision_ts: int,
    window_ms: int,
) -> float | None:
    previous = series.latest_at(latest.source, latest.instrument_id, decision_ts - window_ms)
    if previous is None:
        return None
    current_price = latest.effective_mid_price
    previous_price = previous.effective_mid_price
    if current_price <= 0.0 or previous_price <= 0.0:
        return None
    return math.log(current_price / previous_price)


def _realized_vol(
    series: AlignedMarketSeries,
    latest: MarketData,
    decision_ts: int,
    window_ms: int,
) -> float | None:
    start = max(0, decision_ts - window_ms)
    rows = list(
        series.window(
            latest.source,
            latest.instrument_id,
            start_exclusive=start,
            end_inclusive=decision_ts,
        )
    )
    base = series.latest_at(latest.source, latest.instrument_id, start)
    if base is not None:
        rows.insert(0, base)
    rows = sorted({(row.ts, row.sequence): row for row in rows}.values(), key=lambda row: row.ts)
    if len(rows) < 2:
        return None
    sum_sq = 0.0
    for previous, current in zip(rows, rows[1:], strict=False):
        previous_price = previous.effective_mid_price
        current_price = current.effective_mid_price
        if previous_price <= 0.0 or current_price <= 0.0:
            continue
        sum_sq += math.log(current_price / previous_price) ** 2
    return math.sqrt(sum_sq)


def _window_sum(
    series: AlignedMarketSeries,
    latest: MarketData,
    decision_ts: int,
    window_ms: int,
    field_name: str,
) -> float:
    rows = series.window(
        latest.source,
        latest.instrument_id,
        start_exclusive=max(0, decision_ts - window_ms),
        end_inclusive=decision_ts,
    )
    return float(sum(float(getattr(row, field_name) or 0.0) for row in rows))


def _orderbook_imbalance(row: MarketData) -> float | None:
    if row.bid_size is None or row.ask_size is None:
        return None
    denominator = row.bid_size + row.ask_size
    if denominator <= 0:
        return None
    return (row.bid_size - row.ask_size) / denominator


def _minute_of_day(ts: int) -> float:
    dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
    return (dt.hour * 60 + dt.minute) / 1439.0


def _day_of_week(ts: int) -> int:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).weekday()


def _feature_lookback_ms(name: str) -> int:
    if name.endswith("_15m") or name == "return_15m" or name == "volatility_15m":
        return 15 * MINUTE_MS
    if name.endswith("_5m") or name == "return_5m" or name == "volatility_5m":
        return 5 * MINUTE_MS
    if name.endswith("_1m") or name == "return_1m":
        return MINUTE_MS
    return 0


def _provenance_for(
    *,
    name: str,
    series: AlignedMarketSeries,
    latest: MarketData,
    decision_ts: int,
    lookback_ms: int,
) -> FeatureProvenance:
    start = max(0, decision_ts - lookback_ms)
    if lookback_ms:
        window_rows = list(
            series.window(
                latest.source,
                latest.instrument_id,
                start_exclusive=start,
                end_inclusive=decision_ts,
            )
        )
        if name.startswith(("return", "volatility")):
            base = series.latest_at(latest.source, latest.instrument_id, start)
            if base is not None:
                window_rows.insert(0, base)
        rows = tuple(
            sorted(
                {(row.ts, row.sequence): row for row in window_rows}.values(),
                key=lambda row: row.ts,
            )
        )
    else:
        rows = (latest,)
    if not rows:
        rows = (latest,)
    return FeatureProvenance(
        feature_name=name,
        input_start_ts=min(row.ts for row in rows),
        input_end_ts=max(row.ts for row in rows),
        available_at_ts=max(int(row.available_at_ts or row.ts) for row in rows),
        lookback_ms=lookback_ms,
        source_timeframe_ms=_max_timeframe_ms(rows),
    )


def _max_timeframe_ms(rows: tuple[MarketData, ...]) -> int | None:
    values = [row.timeframe_ms for row in rows if row.timeframe_ms is not None]
    return None if not values else max(values)
