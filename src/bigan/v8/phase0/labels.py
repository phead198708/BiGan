"""Cost-aware future label construction for v8 Phase 0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from bigan.v8.phase0.alignment import AlignedMarketSeries, TimeAlignmentEngine
from bigan.v8.phase0.contracts import LABEL_VERSION, FeatureVector, Label, MarketData
from bigan.v8.phase0.costs import TradingCostModel

MINUTE_MS = 60_000


@dataclass(frozen=True, slots=True)
class CostAwareLabelBuilderConfig:
    """Label-builder settings."""

    horizons_ms: tuple[int, ...] = (MINUTE_MS, 5 * MINUTE_MS, 15 * MINUTE_MS)
    order_size: float = 1.0
    side: int = 1
    label_version: str = LABEL_VERSION

    def __post_init__(self) -> None:
        if not self.horizons_ms:
            raise ValueError("horizons_ms cannot be empty")
        if any(horizon <= 0 for horizon in self.horizons_ms):
            raise ValueError("all horizons_ms values must be positive")
        if self.order_size <= 0:
            raise ValueError("order_size must be positive")
        if self.side not in (-1, 1):
            raise ValueError("side must be -1 or 1")


class CostAwareLabelBuilder:
    """Build labels; this is the only Phase 0 stage that reads future rows."""

    def __init__(
        self,
        cost_model: TradingCostModel | None = None,
        config: CostAwareLabelBuilderConfig | None = None,
        *,
        alignment_engine: TimeAlignmentEngine | None = None,
    ) -> None:
        self.cost_model = cost_model or TradingCostModel()
        self.config = config or CostAwareLabelBuilderConfig()
        self.alignment_engine = alignment_engine or TimeAlignmentEngine()

    def build(
        self,
        market_data: list[MarketData] | AlignedMarketSeries,
        features: list[FeatureVector],
        *,
        horizons_ms: tuple[int, ...] | None = None,
        slippage_multiplier: float = 1.0,
    ) -> list[Label]:
        series = (
            market_data
            if isinstance(market_data, AlignedMarketSeries)
            else self.alignment_engine.align_market_data(market_data)
        )
        labels: list[Label] = []
        for feature in sorted(features, key=lambda row: (row.source, row.instrument_id, row.decision_ts)):
            entry = series.latest_at(feature.source, feature.instrument_id, feature.decision_ts)
            if entry is None:
                continue
            for horizon_ms in sorted(horizons_ms or self.config.horizons_ms):
                exit_row = series.first_at_or_after(
                    feature.source,
                    feature.instrument_id,
                    feature.decision_ts + horizon_ms,
                )
                if exit_row is None:
                    continue
                labels.append(
                    self._build_one(
                        feature=feature,
                        entry=entry,
                        exit_row=exit_row,
                        horizon_ms=horizon_ms,
                        slippage_multiplier=slippage_multiplier,
                    )
                )
        return labels

    def _build_one(
        self,
        *,
        feature: FeatureVector,
        entry: MarketData,
        exit_row: MarketData,
        horizon_ms: int,
        slippage_multiplier: float,
    ) -> Label:
        entry_price = entry.effective_mid_price
        exit_price = exit_row.effective_mid_price
        gross_return = self.config.side * ((exit_price / entry_price) - 1.0)
        volatility = _feature_volatility(feature)
        costs = self.cost_model.estimate(
            entry=entry,
            exit=exit_row,
            order_size=self.config.order_size,
            volatility=volatility,
            slippage_multiplier=slippage_multiplier,
        )
        net_return = gross_return - costs.total_cost
        return Label(
            decision_ts=feature.decision_ts,
            label_ts=exit_row.ts,
            horizon_ms=horizon_ms,
            source=feature.source,
            instrument_id=feature.instrument_id,
            entry_price=entry_price,
            exit_price=exit_price,
            side=self.config.side,
            gross_return=gross_return,
            spread_cost=costs.spread_cost,
            fee_cost=costs.fee_cost,
            slippage_cost=costs.slippage_cost,
            liquidity_impact_cost=costs.liquidity_impact_cost,
            total_cost=costs.total_cost,
            net_return=net_return,
            is_positive=net_return > 0.0,
            label_version=self.config.label_version,
        )


def label_id(label: Label) -> str:
    raw = (
        f"{label.source}|{label.instrument_id}|{label.decision_ts}|"
        f"{label.horizon_ms}|{label.label_version}"
    ).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _feature_volatility(feature: FeatureVector) -> float | None:
    value = feature.features.get("volatility_5m")
    if value is None:
        value = feature.features.get("volatility_15m")
    return None if value is None else float(value)

