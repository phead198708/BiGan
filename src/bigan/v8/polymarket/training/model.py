"""Lightweight deterministic action-value model for Polymarket BTC policy training."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    PRIMARY_POLICY_TARGET_ACTION_VALUE,
    PolymarketPolicyDataset,
    PolymarketPolicyExample,
    PolymarketPolicyModel,
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
)

EPSILON = 1e-6
ACTION_RETURN_RIDGE = 1e-9
ACTION_RETURN_COEFFICIENT_LIMIT = 5.0
ActionValueMissingFeatureMode = Literal["strict", "train_mean_impute"]


def train_polymarket_action_value_model(
    dataset: PolymarketPolicyDataset,
    config: PolymarketPolicyTrainingConfig,
) -> PolymarketPolicyModel:
    """Train an auditable action-value policy baseline.

    The primary target is cost-aware expected net return per action. P(UP) is
    retained as an auxiliary outcome/calibration head for sanity checks.
    """

    targets = [example.target_up_probability for example in dataset.train_examples]
    global_probability = _clamp(sum(targets) / len(targets), EPSILON, 1.0 - EPSILON)
    family_targets: dict[str, list[float]] = defaultdict(list)
    family_offsets: dict[str, list[float]] = defaultdict(list)
    global_action_values: dict[str, list[float]] = {
        action: [] for action in ACTION_VALUE_LABEL_ACTIONS
    }
    family_action_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {action: [] for action in ACTION_VALUE_LABEL_ACTIONS}
    )
    for example in dataset.train_examples:
        family_targets[example.market_family].append(example.target_up_probability)
        up_mid = example.features.get("up_mid")
        if up_mid is not None:
            family_offsets[example.market_family].append(
                example.target_up_probability - float(up_mid)
            )
        action_returns = _required_action_returns(example)
        for action, value in action_returns.items():
            global_action_values[action].append(value)
            family_action_values[example.market_family][action].append(value)
    family_probabilities = {
        family: _clamp(sum(values) / len(values), EPSILON, 1.0 - EPSILON)
        for family, values in family_targets.items()
    }
    family_feature_offsets = {
        family: sum(values) / len(values)
        for family, values in family_offsets.items()
        if values
    }
    global_action_returns = {
        action: _mean(values)
        for action, values in global_action_values.items()
    }
    market_family_action_returns = {
        family: {action: _mean(values) for action, values in action_values.items()}
        for family, action_values in family_action_values.items()
    }
    action_value_feature_columns = _action_value_feature_columns(dataset)
    action_return_feature_means = _feature_means(
        dataset.train_examples,
        action_value_feature_columns,
    )
    action_return_feature_coefficients = _action_return_feature_coefficients(
        examples=dataset.train_examples,
        feature_columns=action_value_feature_columns,
        feature_means=action_return_feature_means,
        market_family_action_returns=market_family_action_returns,
        global_action_returns=global_action_returns,
    )
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
        primary_policy_target=PRIMARY_POLICY_TARGET_ACTION_VALUE,
        outcome_probability_head_enabled=True,
        action_value_head_enabled=True,
        compatibility_probability_fallback_enabled=True,
        action_value_model_family="feature_conditioned_action_return_model",
        fallback_action_value_model_family="market_family_mean_baseline",
        feature_conditioned_action_value_model_enabled=True,
        action_value_feature_columns=action_value_feature_columns,
        action_return_feature_means=action_return_feature_means,
        action_return_feature_coefficients=action_return_feature_coefficients,
        global_action_returns=global_action_returns,
        market_family_action_returns=market_family_action_returns,
        family_action_feature_offsets={},
    )


def train_polymarket_probability_model(
    dataset: PolymarketPolicyDataset,
    config: PolymarketPolicyTrainingConfig,
) -> PolymarketPolicyModel:
    """Compatibility alias for the new action-value policy model."""

    return train_polymarket_action_value_model(dataset, config)


def predict_polymarket_policy_examples(
    model: PolymarketPolicyModel,
    examples: tuple[PolymarketPolicyExample, ...],
    *,
    missing_feature_mode: ActionValueMissingFeatureMode = "strict",
) -> tuple[PolymarketPolicyPrediction, ...]:
    """Score examples with the trained action-value policy model."""

    if missing_feature_mode not in ("strict", "train_mean_impute"):
        raise ValueError("missing_feature_mode must be strict or train_mean_impute")
    return tuple(
        _prediction(model, example, missing_feature_mode=missing_feature_mode)
        for example in examples
    )


def _prediction(
    model: PolymarketPolicyModel,
    example: PolymarketPolicyExample,
    *,
    missing_feature_mode: ActionValueMissingFeatureMode,
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
    expected_return_by_action = _expected_action_returns(
        model,
        example,
        missing_feature_mode=missing_feature_mode,
    )
    best_action, best_return, second_best_return, best_margin = _rank_actions(
        expected_return_by_action
    )
    policy_confidence = _policy_confidence(best_return=best_return, best_margin=best_margin)
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
        p_up_auxiliary=probability,
        expected_return_by_action=expected_return_by_action,
        expected_return_no_trade=expected_return_by_action["NO_TRADE"],
        expected_return_buy_up_hold_to_settlement=expected_return_by_action[
            "BUY_UP_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_down_hold_to_settlement=expected_return_by_action[
            "BUY_DOWN_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_up_sell_before_close=expected_return_by_action[
            "BUY_UP_SELL_BEFORE_CLOSE"
        ],
        expected_return_buy_down_sell_before_close=expected_return_by_action[
            "BUY_DOWN_SELL_BEFORE_CLOSE"
        ],
        best_policy_action=best_action,
        best_action_expected_return=best_return,
        second_best_action_expected_return=second_best_return,
        best_action_margin=best_margin,
        policy_confidence=policy_confidence,
        action_value_head_enabled=bool(model.action_value_head_enabled),
        outcome_probability_head_enabled=bool(model.outcome_probability_head_enabled),
        action_value_model_family=model.action_value_model_family,
        feature_conditioned_action_value_model_enabled=(
            model.feature_conditioned_action_value_model_enabled
        ),
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


def _required_action_returns(example: PolymarketPolicyExample) -> dict[str, float]:
    missing = set(ACTION_VALUE_LABEL_ACTIONS) - set(example.action_return_targets)
    if missing:
        raise ValueError(
            "training example missing action return targets: " + ", ".join(sorted(missing))
        )
    return {
        action: float(example.action_return_targets[action])
        for action in ACTION_VALUE_LABEL_ACTIONS
    }


def _expected_action_returns(
    model: PolymarketPolicyModel,
    example: PolymarketPolicyExample,
    *,
    missing_feature_mode: ActionValueMissingFeatureMode,
) -> dict[str, float]:
    if model.action_value_head_enabled:
        base = model.market_family_action_returns.get(
            example.market_family,
            model.global_action_returns,
        )
        if base:
            _validate_action_value_features(
                model=model,
                example=example,
                missing_feature_mode=missing_feature_mode,
            )
            expected = {action: float(base[action]) for action in ACTION_VALUE_LABEL_ACTIONS}
            if model.feature_conditioned_action_value_model_enabled:
                for action in ACTION_VALUE_LABEL_ACTIONS:
                    expected[action] = _feature_conditioned_return(
                        baseline=expected[action],
                        coefficients=model.action_return_feature_coefficients.get(action, {}),
                        feature_means=model.action_return_feature_means,
                        example=example,
                        missing_feature_mode=missing_feature_mode,
                    )
            return expected
    return dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, 0.0)


def _validate_action_value_features(
    *,
    model: PolymarketPolicyModel,
    example: PolymarketPolicyExample,
    missing_feature_mode: ActionValueMissingFeatureMode,
) -> None:
    if not model.feature_conditioned_action_value_model_enabled:
        return
    if missing_feature_mode == "train_mean_impute":
        return
    missing_features = [
        feature_name
        for feature_name in model.action_value_feature_columns
        if feature_name not in example.features
    ]
    if missing_features:
        raise ValueError(
            "action_value_feature_missing: "
            f"market_id={example.market_id} decision_ts={example.decision_ts} "
            "missing_features="
            + ",".join(sorted(missing_features))
        )


def _feature_conditioned_return(
    *,
    baseline: float,
    coefficients: dict[str, float],
    feature_means: dict[str, float],
    example: PolymarketPolicyExample,
    missing_feature_mode: ActionValueMissingFeatureMode,
) -> float:
    value = float(baseline)
    missing_features = []
    for feature_name, coefficient in coefficients.items():
        if feature_name in example.features:
            feature_value = float(example.features[feature_name])
        elif missing_feature_mode == "train_mean_impute":
            feature_value = float(feature_means.get(feature_name, 0.0))
        else:
            missing_features.append(feature_name)
            continue
        value += float(coefficient) * (
            feature_value - float(feature_means.get(feature_name, 0.0))
        )
    if missing_features:
        raise ValueError(
            "action_value_feature_missing: "
            f"market_id={example.market_id} decision_ts={example.decision_ts} "
            "missing_features="
            + ",".join(sorted(missing_features))
        )
    return _clamp(value, -10.0, 10.0)


def _rank_actions(action_returns: dict[str, float]) -> tuple[str, float, float, float]:
    ranked = sorted(
        ((action, float(action_returns[action])) for action in ACTION_VALUE_LABEL_ACTIONS),
        key=lambda item: (-item[1], item[0]),
    )
    best_action, best_return = ranked[0]
    second_best_return = ranked[1][1] if len(ranked) > 1 else best_return
    return best_action, best_return, second_best_return, best_return - second_best_return


def _policy_confidence(*, best_return: float, best_margin: float) -> float:
    return _clamp(abs(best_return) * 2.0 + max(0.0, best_margin) * 5.0, 0.0, 1.0)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _action_value_feature_columns(dataset: PolymarketPolicyDataset) -> tuple[str, ...]:
    return tuple(
        feature_name
        for feature_name in dataset.feature_columns
        if not feature_name.startswith("family_")
    )


def _feature_means(
    examples: tuple[PolymarketPolicyExample, ...],
    feature_columns: tuple[str, ...],
) -> dict[str, float]:
    return {
        feature_name: _mean(
            [float(example.features.get(feature_name, 0.0)) for example in examples]
        )
        for feature_name in feature_columns
    }


def _action_return_feature_coefficients(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    feature_columns: tuple[str, ...],
    feature_means: dict[str, float],
    market_family_action_returns: dict[str, dict[str, float]],
    global_action_returns: dict[str, float],
) -> dict[str, dict[str, float]]:
    coefficients: dict[str, dict[str, float]] = {}
    for action in ACTION_VALUE_LABEL_ACTIONS:
        action_coefficients: dict[str, float] = {}
        for feature_name in feature_columns:
            numerator = 0.0
            denominator = ACTION_RETURN_RIDGE
            for example in examples:
                base = market_family_action_returns.get(
                    example.market_family,
                    global_action_returns,
                )
                residual = (
                    float(example.action_return_targets[action])
                    - float(base.get(action, global_action_returns[action]))
                )
                centered_feature = (
                    float(example.features.get(feature_name, 0.0))
                    - float(feature_means[feature_name])
                )
                numerator += centered_feature * residual
                denominator += centered_feature * centered_feature
            coefficient = numerator / denominator if denominator > 0.0 else 0.0
            if math.isfinite(coefficient):
                action_coefficients[feature_name] = _clamp(
                    coefficient,
                    -ACTION_RETURN_COEFFICIENT_LIMIT,
                    ACTION_RETURN_COEFFICIENT_LIMIT,
                )
            else:
                action_coefficients[feature_name] = 0.0
        coefficients[action] = action_coefficients
    return coefficients
