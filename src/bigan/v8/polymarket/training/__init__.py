"""Polymarket BTC Up/Down policy training package."""

from bigan.v8.polymarket.training.calibration import (
    TIME_TO_CLOSE_BUCKETS,
    calibration_report,
    split_calibration_report,
    validation_report,
)
from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    AUXILIARY_OUTCOME_TARGET,
    DEFAULT_ACTION_VALUE_MODEL_VERSION,
    DEFAULT_POLICY_CREATED_AT,
    POLYMARKET_POLICY_SCHEMA_VERSION,
    POLYMARKET_POLICY_SIGNAL_SOURCE_TRAINED_MODEL,
    POLYMARKET_POLICY_TRAINING_PHASE,
    PRIMARY_POLICY_TARGET_ACTION_VALUE,
    PolymarketPolicyDataset,
    PolymarketPolicyExample,
    PolymarketPolicyModel,
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
    PolymarketPolicyTrainingResult,
)
from bigan.v8.polymarket.training.dataset import (
    ACTION_VALUE_TARGET_FIELD,
    TARGET_LABEL_ACTION,
    dataset_profile,
    load_polymarket_policy_dataset,
)
from bigan.v8.polymarket.training.model import (
    predict_polymarket_policy_examples,
    train_polymarket_action_value_model,
    train_polymarket_probability_model,
)
from bigan.v8.polymarket.training.real_corpus_gate import (
    DEFAULT_REAL_CORPUS_MODEL_VERSION,
    DEFAULT_REAL_CORPUS_TRAINING_RUN_ID,
    POLYMARKET_REAL_CORPUS_RETRAINING_GATE_PHASE,
    POLYMARKET_REAL_CORPUS_RETRAINING_GATE_SCHEMA_VERSION,
    PolymarketRealCorpusRetrainingGateConfig,
    PolymarketRealCorpusRetrainingGateResult,
    run_polymarket_real_corpus_retraining_gate,
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
    "ACTION_VALUE_LABEL_ACTIONS",
    "AUXILIARY_OUTCOME_TARGET",
    "DEFAULT_ACTION_VALUE_MODEL_VERSION",
    "PRIMARY_POLICY_TARGET_ACTION_VALUE",
    "ACTION_VALUE_TARGET_FIELD",
    "POLYMARKET_REAL_CORPUS_RETRAINING_GATE_PHASE",
    "POLYMARKET_REAL_CORPUS_RETRAINING_GATE_SCHEMA_VERSION",
    "TARGET_LABEL_ACTION",
    "TIME_TO_CLOSE_BUCKETS",
    "DEFAULT_REAL_CORPUS_MODEL_VERSION",
    "DEFAULT_REAL_CORPUS_TRAINING_RUN_ID",
    "PolymarketPolicyDataset",
    "PolymarketPolicyExample",
    "PolymarketPolicyModel",
    "PolymarketPolicyPrediction",
    "PolymarketPolicyTrainingConfig",
    "PolymarketPolicyTrainingResult",
    "PolymarketRealCorpusRetrainingGateConfig",
    "PolymarketRealCorpusRetrainingGateResult",
    "calibration_report",
    "dataset_profile",
    "load_polymarket_policy_dataset",
    "predict_polymarket_policy_examples",
    "run_polymarket_policy_training",
    "run_polymarket_real_corpus_retraining_gate",
    "split_calibration_report",
    "train_polymarket_action_value_model",
    "train_polymarket_probability_model",
    "validation_report",
]
