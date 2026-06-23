"""Polymarket BTC Up/Down policy training package."""

from bigan.v8.polymarket.training.calibration import (
    TIME_TO_CLOSE_BUCKETS,
    calibration_report,
    validation_report,
)
from bigan.v8.polymarket.training.contracts import (
    DEFAULT_POLICY_CREATED_AT,
    POLYMARKET_POLICY_SCHEMA_VERSION,
    POLYMARKET_POLICY_SIGNAL_SOURCE_TRAINED_MODEL,
    POLYMARKET_POLICY_TRAINING_PHASE,
    PolymarketPolicyDataset,
    PolymarketPolicyExample,
    PolymarketPolicyModel,
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
    PolymarketPolicyTrainingResult,
)
from bigan.v8.polymarket.training.dataset import (
    TARGET_LABEL_ACTION,
    dataset_profile,
    load_polymarket_policy_dataset,
)
from bigan.v8.polymarket.training.model import (
    predict_polymarket_policy_examples,
    train_polymarket_probability_model,
)


def run_polymarket_policy_training(*args, **kwargs):
    from bigan.v8.polymarket.training.runner import (
        run_polymarket_policy_training as _run_polymarket_policy_training,
    )

    return _run_polymarket_policy_training(*args, **kwargs)

__all__ = [
    "DEFAULT_POLICY_CREATED_AT",
    "POLYMARKET_POLICY_SCHEMA_VERSION",
    "POLYMARKET_POLICY_SIGNAL_SOURCE_TRAINED_MODEL",
    "POLYMARKET_POLICY_TRAINING_PHASE",
    "TARGET_LABEL_ACTION",
    "TIME_TO_CLOSE_BUCKETS",
    "PolymarketPolicyDataset",
    "PolymarketPolicyExample",
    "PolymarketPolicyModel",
    "PolymarketPolicyPrediction",
    "PolymarketPolicyTrainingConfig",
    "PolymarketPolicyTrainingResult",
    "calibration_report",
    "dataset_profile",
    "load_polymarket_policy_dataset",
    "predict_polymarket_policy_examples",
    "run_polymarket_policy_training",
    "train_polymarket_probability_model",
    "validation_report",
]
