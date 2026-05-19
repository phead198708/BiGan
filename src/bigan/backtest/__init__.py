"""Backtest configuration and evaluation helpers."""

from .config import (
    BACKTEST_CONFIG_SCHEMA_VERSION,
    BacktestConfig,
    BacktestCostConfig,
    BacktestDatasetConfig,
    BacktestExecutionConfig,
    BacktestModelConfig,
    BacktestOutputConfig,
    BacktestStrategyConfig,
    generate_run_id,
    load_backtest_config,
)
from .evaluation import (
    CalibrationBin,
    PredictionEvaluationReport,
    ThresholdMetrics,
    evaluate_predictions,
    save_evaluation_report,
)
from .execution import (
    Quote,
    SimulatedTakerTrade,
    TakerExecutionSettings,
    simulate_taker_long_trade,
)
from .strategy import (
    DEFAULT_HOLD_MS,
    DEFAULT_THRESHOLDS,
    PredictionSignal,
    ThresholdStrategyResult,
    ThresholdStrategySummary,
    ThresholdTrade,
    run_threshold_strategy,
    run_threshold_sweep,
    save_threshold_strategy_outputs,
)

__all__ = [
    "BACKTEST_CONFIG_SCHEMA_VERSION",
    "BacktestConfig",
    "BacktestCostConfig",
    "BacktestDatasetConfig",
    "BacktestExecutionConfig",
    "BacktestModelConfig",
    "BacktestOutputConfig",
    "BacktestStrategyConfig",
    "CalibrationBin",
    "DEFAULT_HOLD_MS",
    "DEFAULT_THRESHOLDS",
    "PredictionSignal",
    "PredictionEvaluationReport",
    "Quote",
    "SimulatedTakerTrade",
    "TakerExecutionSettings",
    "ThresholdMetrics",
    "ThresholdStrategyResult",
    "ThresholdStrategySummary",
    "ThresholdTrade",
    "evaluate_predictions",
    "generate_run_id",
    "load_backtest_config",
    "run_threshold_strategy",
    "run_threshold_sweep",
    "save_evaluation_report",
    "save_threshold_strategy_outputs",
    "simulate_taker_long_trade",
]
