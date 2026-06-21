"""Phase 1 pure policy learning for the v8 trading architecture."""

from bigan.v8.phase1.contracts import (
    PHASE1_POLICY_VERSION,
    SUPPORTED_POLICY_OBJECTIVES,
    XGBOOST_V8_POLICY_MODEL_VERSION,
    PolicyDataset,
    PolicyDatasetConfig,
    PolicyObjective,
    PolicyPrediction,
    PolicyTrainingExample,
    XGBoostPolicyConfig,
    assert_no_direct_pnl_optimization,
)
from bigan.v8.phase1.dataset import (
    build_policy_dataset,
    build_policy_dataset_from_phase0,
    policy_dataset_hash,
)
from bigan.v8.phase1.model import XGBoostPolicyModel, train_xgboost_policy
from bigan.v8.phase1.validation import (
    PolicyAcceptanceConfig,
    PolicyAcceptanceFailure,
    PolicyAcceptanceReport,
    validate_policy_acceptance,
)

__all__ = [
    "PHASE1_POLICY_VERSION",
    "SUPPORTED_POLICY_OBJECTIVES",
    "XGBOOST_V8_POLICY_MODEL_VERSION",
    "PolicyAcceptanceConfig",
    "PolicyAcceptanceFailure",
    "PolicyAcceptanceReport",
    "PolicyDataset",
    "PolicyDatasetConfig",
    "PolicyObjective",
    "PolicyPrediction",
    "PolicyTrainingExample",
    "XGBoostPolicyConfig",
    "XGBoostPolicyModel",
    "assert_no_direct_pnl_optimization",
    "build_policy_dataset",
    "build_policy_dataset_from_phase0",
    "policy_dataset_hash",
    "train_xgboost_policy",
    "validate_policy_acceptance",
]
