"""Action-value calibration for Polymarket policy execution."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    POLYMARKET_POLICY_SCHEMA_VERSION,
    POLYMARKET_POLICY_TRAINING_PHASE,
    PolymarketPolicyExample,
    PolymarketPolicyPrediction,
    compact_safety_fields,
)

ACTION_VALUE_CALIBRATION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-action-value-calibration-v1"
)
ACTION_VALUE_MIN_CALIBRATION_SUPPORT_PER_ACTION = 3
ACTION_VALUE_CORRECTION_LIMIT = 0.50


def build_action_value_calibration_artifact(
    *,
    calibration_examples: tuple[PolymarketPolicyExample, ...],
    calibration_predictions: tuple[PolymarketPolicyPrediction, ...],
    evaluation_examples: tuple[PolymarketPolicyExample, ...],
    evaluation_predictions: tuple[PolymarketPolicyPrediction, ...],
) -> dict[str, Any]:
    """Fit a deterministic action-value bias correction on validation data."""

    _validate_aligned(calibration_examples, calibration_predictions)
    _validate_aligned(evaluation_examples, evaluation_predictions)
    buckets: dict[str, dict[str, Any]] = {}
    corrections: dict[str, float] = {}
    for action in ACTION_VALUE_LABEL_ACTIONS:
        rows = [
            _calibration_row(example=example, prediction=prediction, action=action)
            for example, prediction in zip(
                calibration_examples,
                calibration_predictions,
                strict=True,
            )
        ]
        residuals = [row["residual"] for row in rows]
        correction = _clamp(_mean(residuals), -ACTION_VALUE_CORRECTION_LIMIT, ACTION_VALUE_CORRECTION_LIMIT)
        corrections[action] = correction
        buckets[action] = {
            "action": action,
            "support_count": len(rows),
            "raw_expected_pnl_per_notional_mean": _mean(
                [row["raw_expected_pnl_per_notional"] for row in rows]
            ),
            "target_expected_pnl_per_notional_mean": _mean(
                [row["target_expected_pnl_per_notional"] for row in rows]
            ),
            "residual_mean": _mean(residuals),
            "residual_mae": _mean([abs(value) for value in residuals]),
            "correction": correction,
            "correction_clipped": not math.isclose(correction, _mean(residuals), abs_tol=1e-12),
        }
    support_passed = all(
        bucket["support_count"] >= ACTION_VALUE_MIN_CALIBRATION_SUPPORT_PER_ACTION
        for bucket in buckets.values()
    )
    artifact = {
        "schema_version": ACTION_VALUE_CALIBRATION_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "calibration_method": "validation_action_bias_correction",
        "calibration_fit_split": "validation",
        "calibration_evaluation_split": "shadow",
        "calibration_uses_training_split": False,
        "raw_expected_pnl_per_notional_by_action_field": "expected_return_by_action",
        "calibrated_expected_pnl_per_notional_by_action_field": (
            "calibrated_expected_pnl_per_notional_by_action"
        ),
        "calibration_support_count": len(calibration_examples),
        "calibration_bucket_count": len(buckets),
        "min_calibration_support_per_action": (
            ACTION_VALUE_MIN_CALIBRATION_SUPPORT_PER_ACTION
        ),
        "calibration_support_passed": support_passed,
        "calibration_support_level": "sufficient" if support_passed else "insufficient",
        "action_corrections": corrections,
        "calibration_buckets": buckets,
        "calibration_metrics": _calibration_metrics(
            examples=calibration_examples,
            predictions=calibration_predictions,
            corrections=corrections,
        ),
        "shadow_evaluation_metrics": _calibration_metrics(
            examples=evaluation_examples,
            predictions=evaluation_predictions,
            corrections=corrections,
        ),
        **compact_safety_fields(),
    }
    artifact["action_value_calibration_id"] = canonical_json_sha256(artifact)
    return artifact


def apply_action_value_calibration(
    *,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    calibration_artifact: dict[str, Any],
) -> tuple[PolymarketPolicyPrediction, ...]:
    """Return predictions with calibrated action-value execution fields."""

    corrections = {
        str(action): float(value)
        for action, value in dict(calibration_artifact["action_corrections"]).items()
    }
    missing = set(ACTION_VALUE_LABEL_ACTIONS) - set(corrections)
    if missing:
        raise ValueError(
            "action_value_calibration missing corrections: " + ", ".join(sorted(missing))
        )
    return tuple(
        _calibrated_prediction(
            prediction=prediction,
            corrections=corrections,
            calibration_artifact=calibration_artifact,
        )
        for prediction in predictions
    )


def _calibrated_prediction(
    *,
    prediction: PolymarketPolicyPrediction,
    corrections: dict[str, float],
    calibration_artifact: dict[str, Any],
) -> PolymarketPolicyPrediction:
    if not prediction.action_value_head_enabled:
        return prediction
    calibrated = {
        action: _clamp(
            float(prediction.expected_return_by_action[action]) + corrections[action],
            -10.0,
            10.0,
        )
        for action in ACTION_VALUE_LABEL_ACTIONS
    }
    best_action, best_value, second_value, margin = _rank_actions(calibrated)
    return replace(
        prediction,
        calibrated_expected_pnl_per_notional_by_action=calibrated,
        calibrated_best_policy_action=best_action,
        calibrated_expected_pnl_per_notional=best_value,
        calibrated_second_best_expected_pnl_per_notional=second_value,
        calibrated_action_margin=margin,
        action_value_calibration_applied=True,
        action_value_calibration_id=str(
            calibration_artifact["action_value_calibration_id"]
        ),
        calibration_support_count=int(calibration_artifact["calibration_support_count"]),
        calibration_bucket_count=int(calibration_artifact["calibration_bucket_count"]),
    )


def _calibration_row(
    *,
    example: PolymarketPolicyExample,
    prediction: PolymarketPolicyPrediction,
    action: str,
) -> dict[str, float]:
    raw = float(prediction.expected_return_by_action[action])
    target = float(example.action_return_targets[action])
    return {
        "raw_expected_pnl_per_notional": raw,
        "target_expected_pnl_per_notional": target,
        "residual": target - raw,
    }


def _calibration_metrics(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    corrections: dict[str, float],
) -> dict[str, Any]:
    raw_errors = []
    calibrated_errors = []
    for example, prediction in zip(examples, predictions, strict=True):
        for action in ACTION_VALUE_LABEL_ACTIONS:
            target = float(example.action_return_targets[action])
            raw = float(prediction.expected_return_by_action[action])
            calibrated = raw + corrections[action]
            raw_errors.append(target - raw)
            calibrated_errors.append(target - calibrated)
    return {
        "sample_count": len(examples),
        "action_value_point_count": len(raw_errors),
        "raw_mae": _mean([abs(value) for value in raw_errors]),
        "calibrated_mae": _mean([abs(value) for value in calibrated_errors]),
        "raw_bias": _mean(raw_errors),
        "calibrated_bias": _mean(calibrated_errors),
    }


def _validate_aligned(
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
) -> None:
    if len(examples) != len(predictions):
        raise ValueError("action-value calibration examples/predictions length mismatch")
    for example, prediction in zip(examples, predictions, strict=True):
        if (example.market_id, example.decision_ts) != (
            prediction.market_id,
            prediction.decision_ts,
        ):
            raise ValueError("action-value calibration examples/predictions misaligned")
        missing = set(ACTION_VALUE_LABEL_ACTIONS) - set(example.action_return_targets)
        if missing:
            raise ValueError(
                "action-value calibration example missing targets: "
                + ", ".join(sorted(missing))
            )
        missing = set(ACTION_VALUE_LABEL_ACTIONS) - set(prediction.expected_return_by_action)
        if missing:
            raise ValueError(
                "action-value calibration prediction missing actions: "
                + ", ".join(sorted(missing))
            )


def _rank_actions(action_returns: dict[str, float]) -> tuple[str, float, float, float]:
    ranked = sorted(
        ((action, float(action_returns[action])) for action in ACTION_VALUE_LABEL_ACTIONS),
        key=lambda item: (-item[1], item[0]),
    )
    best_action, best_return = ranked[0]
    second_best_return = ranked[1][1] if len(ranked) > 1 else best_return
    return best_action, best_return, second_best_return, best_return - second_best_return


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))
