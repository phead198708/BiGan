"""Lightweight deterministic P(UP) model for Polymarket BTC policy training."""

from __future__ import annotations

import math
from collections import defaultdict

from bigan.v8.polymarket.training.contracts import (
    PolymarketPolicyDataset,
    PolymarketPolicyExample,
    PolymarketPolicyModel,
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
)

EPSILON = 1e-6


def train_polymarket_probability_model(
    dataset: PolymarketPolicyDataset,
    config: PolymarketPolicyTrainingConfig,
) -> PolymarketPolicyModel:
    """Train an auditable probability model for P(UP).

    This first production-safe model is intentionally simple and deterministic:
    it learns market-family target frequencies and a family-specific offset
    between target outcomes and contemporaneous UP mid prices. PnL is never used
    as the training target.
    """

    targets = [example.target_up_probability for example in dataset.train_examples]
    global_probability = _clamp(sum(targets) / len(targets), EPSILON, 1.0 - EPSILON)
    family_targets: dict[str, list[float]] = defaultdict(list)
    family_offsets: dict[str, list[float]] = defaultdict(list)
    for example in dataset.train_examples:
        family_targets[example.market_family].append(example.target_up_probability)
        up_mid = example.features.get("up_mid")
        if up_mid is not None:
            family_offsets[example.market_family].append(
                example.target_up_probability - float(up_mid)
            )
    family_probabilities = {
        family: _clamp(sum(values) / len(values), EPSILON, 1.0 - EPSILON)
        for family, values in family_targets.items()
    }
    family_feature_offsets = {
        family: sum(values) / len(values)
        for family, values in family_offsets.items()
        if values
    }
    return PolymarketPolicyModel(
        model_version=config.model_version,
        feature_columns=dataset.feature_columns,
        global_probability=global_probability,
        market_family_probabilities=family_probabilities,
        family_feature_offsets=family_feature_offsets,
        feature_schema_hash=dataset.feature_schema_hash,
        label_schema_hash=dataset.label_schema_hash,
        training_corpus_hash=dataset.training_corpus_hash,
        dataset_hash=dataset.dataset_hash,
        train_row_count=len(dataset.train_examples),
    )


def predict_polymarket_policy_examples(
    model: PolymarketPolicyModel,
    examples: tuple[PolymarketPolicyExample, ...],
) -> tuple[PolymarketPolicyPrediction, ...]:
    """Score examples with the trained probability model."""

    return tuple(_prediction(model, example) for example in examples)


def _prediction(
    model: PolymarketPolicyModel,
    example: PolymarketPolicyExample,
) -> PolymarketPolicyPrediction:
    up_mid = float(example.features.get("up_mid", model.global_probability))
    if example.market_family in model.market_family_probabilities:
        learned = model.market_family_probabilities[example.market_family]
        market_adjusted = _clamp(
            up_mid + model.family_feature_offsets.get(example.market_family, 0.0),
            EPSILON,
            1.0 - EPSILON,
        )
        probability = _clamp(0.7 * learned + 0.3 * market_adjusted, EPSILON, 1.0 - EPSILON)
    else:
        probability = _clamp(up_mid, EPSILON, 1.0 - EPSILON)
    confidence = _clamp(abs(probability - 0.5) * 2.0, 0.0, 1.0)
    return PolymarketPolicyPrediction(
        market_id=example.market_id,
        condition_id=example.condition_id,
        slug=example.slug,
        market_family=example.market_family,
        horizon_ms=example.horizon_ms,
        decision_ts=example.decision_ts,
        estimated_up_probability=probability,
        confidence=confidence,
        score=_logit(probability),
        calibration_bucket=_calibration_bucket(probability),
        model_version=model.model_version,
        feature_schema_hash=model.feature_schema_hash,
        training_corpus_hash=model.training_corpus_hash,
        features=example.features,
        target_up_probability=example.target_up_probability,
    )


def _calibration_bucket(probability: float) -> str:
    lower = int(min(9, max(0, math.floor(probability * 10)))) / 10
    upper = lower + 0.1
    return f"{lower:.1f}-{upper:.1f}"


def _logit(probability: float) -> float:
    p = _clamp(probability, EPSILON, 1.0 - EPSILON)
    return math.log(p / (1.0 - p))


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))
