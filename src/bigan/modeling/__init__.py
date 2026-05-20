"""Model training dataset helpers."""

from .calibration import (
    CalibrationConfig,
    CalibrationReport,
    ProbabilityCalibrator,
    fit_calibration_from_predictions,
    fit_probability_calibration,
    load_probability_calibrator,
)
from .dataset import (
    DATASET_VERSION,
    DatasetAssemblyReport,
    SplitConfig,
    SplitStats,
    assemble_training_dataset,
)
from .logistic import (
    MODEL_VERSION,
    LogisticBaselineConfig,
    LogisticBaselineModel,
    LogisticBaselineReport,
    load_logistic_baseline,
    train_logistic_baseline,
)
from .predictions import (
    PredictionBatchReport,
    confidence_bucket,
    generate_prediction_rows,
    run_prediction_batch,
)
from .promotion import (
    PromotionCheck,
    PromotionReport,
    PromotionRules,
    evaluate_model_promotion,
)
from .xgboost_v1 import (
    XGBOOST_MODEL_VERSION,
    XGBoostV1Config,
    XGBoostV1Model,
    XGBoostV1Report,
    XGBoostV1Stump,
    load_xgboost_v1_model,
    train_xgboost_v1,
)

__all__ = [
    "CalibrationConfig",
    "CalibrationReport",
    "DATASET_VERSION",
    "DatasetAssemblyReport",
    "LogisticBaselineConfig",
    "LogisticBaselineModel",
    "LogisticBaselineReport",
    "MODEL_VERSION",
    "ProbabilityCalibrator",
    "PredictionBatchReport",
    "PromotionCheck",
    "PromotionReport",
    "PromotionRules",
    "SplitConfig",
    "SplitStats",
    "XGBOOST_MODEL_VERSION",
    "XGBoostV1Config",
    "XGBoostV1Model",
    "XGBoostV1Report",
    "XGBoostV1Stump",
    "assemble_training_dataset",
    "confidence_bucket",
    "fit_calibration_from_predictions",
    "fit_probability_calibration",
    "generate_prediction_rows",
    "evaluate_model_promotion",
    "load_logistic_baseline",
    "load_probability_calibrator",
    "load_xgboost_v1_model",
    "train_logistic_baseline",
    "run_prediction_batch",
    "train_xgboost_v1",
]
