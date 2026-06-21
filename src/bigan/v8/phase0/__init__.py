"""Phase 0 data-correctness firewall for the v8 trading architecture."""

from bigan.v8.phase0.alignment import AlignedMarketSeries, TimeAlignmentEngine
from bigan.v8.phase0.artifacts import (
    Phase0ArtifactError,
    Phase0ArtifactGate,
    Phase0ArtifactValidationFailure,
    Phase0ArtifactValidationReport,
    assert_phase0_artifact_ready,
)
from bigan.v8.phase0.contracts import (
    COST_COLUMNS,
    FEATURE_VECTOR_SCHEMA,
    LABEL_SCHEMA,
    MARKET_DATA_SCHEMA,
    DatasetContract,
    FeatureProvenance,
    FeatureVector,
    Label,
    MarketData,
)
from bigan.v8.phase0.costs import (
    CostBreakdown,
    CostCalibrationBucketConfig,
    CostCalibrationBucketReport,
    CostCalibrationConfig,
    CostCalibrationReport,
    CostModelConfig,
    ExecutionCostSample,
    TradingCostModel,
)
from bigan.v8.phase0.features import CausalFeatureBuilder, CausalFeatureBuilderConfig
from bigan.v8.phase0.labels import CostAwareLabelBuilder, CostAwareLabelBuilderConfig
from bigan.v8.phase0.loader import MarketDataLoader
from bigan.v8.phase0.pipeline import Phase0Dataset, Phase0Pipeline, Phase0PipelineConfig
from bigan.v8.phase0.validation import IntegrityValidator, ValidationConfig, ValidationReport

__all__ = [
    "AlignedMarketSeries",
    "CausalFeatureBuilder",
    "CausalFeatureBuilderConfig",
    "COST_COLUMNS",
    "CostAwareLabelBuilder",
    "CostAwareLabelBuilderConfig",
    "CostBreakdown",
    "CostCalibrationBucketConfig",
    "CostCalibrationBucketReport",
    "CostCalibrationConfig",
    "CostCalibrationReport",
    "CostModelConfig",
    "DatasetContract",
    "ExecutionCostSample",
    "FEATURE_VECTOR_SCHEMA",
    "FeatureProvenance",
    "FeatureVector",
    "IntegrityValidator",
    "LABEL_SCHEMA",
    "Label",
    "MARKET_DATA_SCHEMA",
    "MarketData",
    "MarketDataLoader",
    "Phase0ArtifactError",
    "Phase0ArtifactGate",
    "Phase0ArtifactValidationFailure",
    "Phase0ArtifactValidationReport",
    "Phase0Dataset",
    "Phase0Pipeline",
    "Phase0PipelineConfig",
    "TimeAlignmentEngine",
    "TradingCostModel",
    "ValidationConfig",
    "ValidationReport",
    "assert_phase0_artifact_ready",
]
