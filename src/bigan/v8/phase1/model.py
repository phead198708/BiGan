"""XGBoost v8 pure policy model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.phase1.contracts import (
    PHASE1_POLICY_VERSION,
    PolicyDataset,
    PolicyPrediction,
    PolicyTrainingExample,
    XGBoostPolicyConfig,
    assert_no_direct_pnl_optimization,
)


@dataclass(slots=True)
class XGBoostPolicyModel:
    """Trained Phase 1 policy model plus auditable training manifest."""

    booster: xgb.Booster
    feature_columns: tuple[str, ...]
    config: XGBoostPolicyConfig
    training_manifest: dict[str, Any]

    def predict_examples(
        self,
        examples: Sequence[PolicyTrainingExample],
    ) -> tuple[PolicyPrediction, ...]:
        """Return action, confidence, and regime embedding for examples."""

        if not examples:
            return ()
        matrix = _dmatrix(examples, self.feature_columns)
        raw_scores = self.booster.predict(matrix)
        predictions: list[PolicyPrediction] = []
        for example, raw_score in zip(examples, raw_scores, strict=True):
            score = float(raw_score)
            normalized_score = (
                _clamp(score, 0.0, 1.0)
                if self.config.objective == "binary:logistic"
                else _sigmoid(score)
            )
            action = (
                normalized_score * self.config.max_position_size
                if normalized_score >= self.config.action_activation_threshold
                else 0.0
            )
            predictions.append(
                PolicyPrediction(
                    decision_ts=example.decision_ts,
                    source=example.source,
                    instrument_id=example.instrument_id,
                    action=action,
                    confidence=_clamp(abs(normalized_score - 0.5) * 2.0, 0.0, 1.0),
                    regime_embedding=tuple(
                        _safe_feature_float(example.features.get(feature_name))
                        for feature_name in self.config.regime_feature_names
                    ),
                    score=score,
                )
            )
        return tuple(predictions)

    def predict_dataset(self, dataset: PolicyDataset) -> tuple[PolicyPrediction, ...]:
        """Predict every row in a Phase 1 policy dataset."""

        if dataset.feature_columns != self.feature_columns:
            raise ValueError("dataset feature_columns do not match the trained model")
        return self.predict_examples(dataset.examples)


def train_xgboost_policy(
    dataset: PolicyDataset,
    config: XGBoostPolicyConfig | None = None,
) -> XGBoostPolicyModel:
    """Train an XGBoost policy model without direct PnL optimization."""

    resolved_config = config or XGBoostPolicyConfig()
    assert_no_direct_pnl_optimization(
        objective=resolved_config.objective,
        eval_metric=resolved_config.eval_metric,
        selection_metric=resolved_config.selection_metric,
    )
    training_examples, group_sizes = _training_examples_for_objective(
        dataset.examples,
        resolved_config,
    )
    labels = _training_labels(training_examples, resolved_config)
    _validate_training_labels(labels, resolved_config)

    dtrain = _dmatrix(training_examples, dataset.feature_columns, labels=labels)
    if group_sizes:
        dtrain.set_group(np.asarray(group_sizes, dtype=np.uint32))

    booster = xgb.train(
        params=resolved_config.xgboost_params(),
        dtrain=dtrain,
        num_boost_round=resolved_config.num_boost_round,
    )
    manifest = {
        "phase1_policy_version": PHASE1_POLICY_VERSION,
        "model_version": resolved_config.model_version,
        "model_family": "xgboost",
        "objective": resolved_config.objective,
        "objective_type": (
            "supervised_binary_policy"
            if resolved_config.objective == "binary:logistic"
            else "pairwise_ranking_policy"
        ),
        "target_encoding": (
            "cost_aware_net_return_threshold"
            if resolved_config.objective == "binary:logistic"
            else "dense_relevance_rank_from_cost_aware_net_return"
        ),
        "direct_pnl_optimization": False,
        "pnl_usage": "shadow_acceptance_after_inference_only",
        "selection_metric": resolved_config.selection_metric,
        "policy_dataset_hash": dataset.policy_dataset_hash,
        "phase0_dataset_hash": dataset.phase0_dataset_hash,
        "phase0_dataset_version": dataset.phase0_dataset_version,
        "feature_columns": list(dataset.feature_columns),
        "row_count": len(dataset.examples),
        "training_config": resolved_config.to_dict(),
    }
    return XGBoostPolicyModel(
        booster=booster,
        feature_columns=dataset.feature_columns,
        config=resolved_config,
        training_manifest=manifest,
    )


def _training_examples_for_objective(
    examples: tuple[PolicyTrainingExample, ...],
    config: XGBoostPolicyConfig,
) -> tuple[tuple[PolicyTrainingExample, ...], tuple[int, ...]]:
    if config.objective != "rank:pairwise":
        return examples, ()

    ordered = tuple(
        sorted(
            examples,
            key=lambda row: (row.source, row.instrument_id, row.decision_ts),
        )
    )
    group_sizes: list[int] = []
    current_key: tuple[str, str] | None = None
    current_size = 0
    for example in ordered:
        key = (example.source, example.instrument_id)
        if current_key is None:
            current_key = key
            current_size = 1
            continue
        if key == current_key:
            current_size += 1
            continue
        group_sizes.append(current_size)
        current_key = key
        current_size = 1
    if current_size:
        group_sizes.append(current_size)
    return ordered, tuple(group_sizes)


def _training_labels(
    examples: tuple[PolicyTrainingExample, ...],
    config: XGBoostPolicyConfig,
) -> np.ndarray:
    if config.objective == "binary:logistic":
        return np.asarray(
            [1.0 if example.target_action > 0.0 else 0.0 for example in examples],
            dtype=np.float32,
        )
    return _dense_relevance_labels(examples)


def _validate_training_labels(
    labels: np.ndarray,
    config: XGBoostPolicyConfig,
) -> None:
    if labels.size == 0:
        raise ValueError("policy training requires at least one label")
    if not np.all(np.isfinite(labels)):
        raise ValueError("policy training labels must be finite")
    if config.objective == "binary:logistic" and np.unique(labels).size < 2:
        raise ValueError("binary policy training requires both flat and active targets")
    if config.objective == "rank:pairwise" and np.unique(labels).size < 2:
        raise ValueError("ranking policy training requires at least two target scores")


def _dense_relevance_labels(
    examples: tuple[PolicyTrainingExample, ...],
) -> np.ndarray:
    """Encode cost-aware label ordering as non-negative ranking relevance."""

    unique_scores = sorted({example.target_score for example in examples})
    score_to_rank = {score: rank for rank, score in enumerate(unique_scores)}
    return np.asarray(
        [float(score_to_rank[example.target_score]) for example in examples],
        dtype=np.float32,
    )


def _dmatrix(
    examples: Sequence[PolicyTrainingExample],
    feature_columns: tuple[str, ...],
    *,
    labels: np.ndarray | None = None,
) -> xgb.DMatrix:
    rows = [
        [
            _feature_value_for_matrix(example.features.get(column))
            for column in feature_columns
        ]
        for example in examples
    ]
    return xgb.DMatrix(
        np.asarray(rows, dtype=np.float32),
        label=labels,
        feature_names=list(feature_columns),
    )


def _feature_value_for_matrix(value: float | int | None) -> float:
    if value is None:
        return np.nan
    numeric = float(value)
    return numeric if math.isfinite(numeric) else np.nan


def _safe_feature_float(value: float | int | None) -> float:
    if value is None:
        return 0.0
    numeric = float(value)
    return numeric if math.isfinite(numeric) else 0.0


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
