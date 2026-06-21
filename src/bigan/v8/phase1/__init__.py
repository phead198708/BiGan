"""Phase 1 pure policy learning for the v8 trading architecture."""

from bigan.v8.phase1.contracts import (
    PHASE1_POLICY_VERSION,
    SUPPORTED_POLICY_OBJECTIVES,
    SUPPORTED_RANKING_GROUP_STRATEGIES,
    SUPPORTED_TARGET_ENCODINGS,
    XGBOOST_V8_POLICY_MODEL_VERSION,
    PolicyDataset,
    PolicyDatasetConfig,
    PolicyObjective,
    PolicyPrediction,
    PolicyTargetEncoding,
    PolicyTrainingExample,
    PolicyTrainShadowSplit,
    RankingGroupStrategy,
    XGBoostPolicyConfig,
    assert_no_direct_pnl_optimization,
)
from bigan.v8.phase1.dataset import (
    build_policy_dataset,
    build_policy_dataset_from_phase0,
    build_temporal_policy_split,
    policy_dataset_hash,
)
from bigan.v8.phase1.model import XGBoostPolicyModel, train_xgboost_policy
from bigan.v8.phase1.training import (
    PHASE15_TRAINING_PHASE,
    PolicyTrainingRunConfig,
    PolicyTrainingRunResult,
    run_policy_training,
)
from bigan.v8.phase1.validation import (
    PolicyAcceptanceConfig,
    PolicyAcceptanceFailure,
    PolicyAcceptanceReport,
    validate_policy_acceptance,
    validate_policy_shadow_split,
)

__all__ = [
    "PHASE1_POLICY_VERSION",
    "PHASE15_TRAINING_PHASE",
    "SUPPORTED_RANKING_GROUP_STRATEGIES",
    "SUPPORTED_POLICY_OBJECTIVES",
    "SUPPORTED_TARGET_ENCODINGS",
    "XGBOOST_V8_POLICY_MODEL_VERSION",
    "PolicyAcceptanceConfig",
    "PolicyAcceptanceFailure",
    "PolicyAcceptanceReport",
    "PolicyDataset",
    "PolicyDatasetConfig",
    "PolicyObjective",
    "PolicyPrediction",
    "PolicyTargetEncoding",
    "PolicyTrainShadowSplit",
    "PolicyTrainingExample",
    "PolicyTrainingRunConfig",
    "PolicyTrainingRunResult",
    "RankingGroupStrategy",
    "XGBoostPolicyConfig",
    "XGBoostPolicyModel",
    "assert_no_direct_pnl_optimization",
    "build_temporal_policy_split",
    "build_policy_dataset",
    "build_policy_dataset_from_phase0",
    "policy_dataset_hash",
    "run_policy_training",
    "train_xgboost_policy",
    "validate_policy_acceptance",
    "validate_policy_shadow_split",
]
