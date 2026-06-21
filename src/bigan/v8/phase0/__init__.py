"""Phase 0 data-correctness firewall for the v8 trading architecture."""

from bigan.v8.phase0.alignment import AlignedMarketSeries, TimeAlignmentEngine
from bigan.v8.phase0.contracts import (
    FEATURE_VECTOR_SCHEMA,
    LABEL_SCHEMA,
    MARKET_DATA_SCHEMA,
    FeatureProvenance,
    FeatureVector,
    Label,
    MarketData,
)
from bigan.v8.phase0.costs import CostBreakdown, CostModelConfig, TradingCostModel
from bigan.v8.phase0.features import CausalFeatureBuilder, CausalFeatureBuilderConfig
from bigan.v8.phase0.labels import CostAwareLabelBuilder, CostAwareLabelBuilderConfig
from bigan.v8.phase0.loader import MarketDataLoader
from bigan.v8.phase0.pipeline import Phase0Dataset, Phase0Pipeline, Phase0PipelineConfig
from bigan.v8.phase0.validation import IntegrityValidator, ValidationConfig, ValidationReport

__all__ = [
    "AlignedMarketSeries",
    "CausalFeatureBuilder",
    "CausalFeatureBuilderConfig",
    "CostAwareLabelBuilder",
    "CostAwareLabelBuilderConfig",
    "CostBreakdown",
    "CostModelConfig",
    "FEATURE_VECTOR_SCHEMA",
    "FeatureProvenance",
    "FeatureVector",
    "IntegrityValidator",
    "LABEL_SCHEMA",
    "Label",
    "MARKET_DATA_SCHEMA",
    "MarketData",
    "MarketDataLoader",
    "Phase0Dataset",
    "Phase0Pipeline",
    "Phase0PipelineConfig",
    "TimeAlignmentEngine",
    "TradingCostModel",
    "ValidationConfig",
    "ValidationReport",
]

