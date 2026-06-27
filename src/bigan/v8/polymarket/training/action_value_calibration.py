"""Action-value calibration for Polymarket policy execution."""

from __future__ import annotations

import math
from collections import defaultdict
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
ACTION_VALUE_MIN_BUCKET_SUPPORT = 1
ACTION_VALUE_BUCKET_SHRINKAGE_PRIOR = 10.0
ACTION_VALUE_CORRECTION_LIMIT = 0.50
ACTION_VALUE_QUALITY_MAE_TOLERANCE = 1e-12
ACTION_VALUE_HIGH_SCORE_THRESHOLD = 0.0
ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT = 10
ACTION_VALUE_DEFAULT_EXECUTION_BUFFER = 0.015


def build_action_value_calibration_artifact(
    *,
    calibration_examples: tuple[PolymarketPolicyExample, ...],
    calibration_predictions: tuple[PolymarketPolicyPrediction, ...],
    evaluation_examples: tuple[PolymarketPolicyExample, ...],
    evaluation_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float = ACTION_VALUE_DEFAULT_EXECUTION_BUFFER,
) -> dict[str, Any]:
    """Fit a deterministic action-value bias correction on validation data."""

    _validate_aligned(calibration_examples, calibration_predictions)
    _validate_aligned(evaluation_examples, evaluation_predictions)
    action_buckets: dict[str, dict[str, Any]] = {}
    corrections: dict[str, float] = {}
    calibration_bucket_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for action in ACTION_VALUE_LABEL_ACTIONS:
        rows = [
            _calibration_row(example=example, prediction=prediction, action=action)
            for example, prediction in zip(
                calibration_examples,
                calibration_predictions,
                strict=True,
            )
        ]
        for example, prediction in zip(
            calibration_examples,
            calibration_predictions,
            strict=True,
        ):
            bucket_key = _calibration_bucket_key(
                action=action,
                prediction=prediction,
            )
            calibration_bucket_rows[bucket_key].append(
                _calibration_row(example=example, prediction=prediction, action=action)
            )
        residuals = [row["residual"] for row in rows]
        raw_correction = _mean(residuals)
        correction = _clamp(
            raw_correction,
            -ACTION_VALUE_CORRECTION_LIMIT,
            ACTION_VALUE_CORRECTION_LIMIT,
        )
        corrections[action] = correction
        action_buckets[action] = {
            "action": action,
            "support_count": len(rows),
            "raw_expected_pnl_per_notional_mean": _mean(
                [row["raw_expected_pnl_per_notional"] for row in rows]
            ),
            "target_expected_pnl_per_notional_mean": _mean(
                [row["target_expected_pnl_per_notional"] for row in rows]
            ),
            "residual_mean": raw_correction,
            "residual_mae": _mean([abs(value) for value in residuals]),
            "correction": correction,
            "correction_clipped": not math.isclose(
                correction,
                raw_correction,
                abs_tol=1e-12,
            ),
        }
    calibration_buckets = _bucket_payloads(calibration_bucket_rows)
    support_passed = all(
        bucket["support_count"] >= ACTION_VALUE_MIN_CALIBRATION_SUPPORT_PER_ACTION
        for bucket in action_buckets.values()
    )
    calibration_metrics = _calibration_metrics(
        examples=calibration_examples,
        predictions=calibration_predictions,
        action_corrections=corrections,
        calibration_buckets=calibration_buckets,
    )
    shadow_evaluation_metrics = _calibration_metrics(
        examples=evaluation_examples,
        predictions=evaluation_predictions,
        action_corrections=corrections,
        calibration_buckets=calibration_buckets,
    )
    shadow_mae_comparison = _mae_comparison(
        examples=evaluation_examples,
        predictions=evaluation_predictions,
        action_corrections=corrections,
        calibration_buckets=calibration_buckets,
    )
    shadow_calibrated_mae_not_worse = (
        float(shadow_mae_comparison["bucketed_calibrated_mae"])
        <= float(shadow_mae_comparison["raw_mae"])
        + ACTION_VALUE_QUALITY_MAE_TOLERANCE
    )
    shadow_high_score_bucket = _high_score_bucket_report(
        examples=evaluation_examples,
        predictions=evaluation_predictions,
        action_corrections=corrections,
        calibration_buckets=calibration_buckets,
        execution_buffer=execution_buffer,
    )
    calibration_quality_passed = (
        shadow_calibrated_mae_not_worse
        and bool(shadow_high_score_bucket["support_passed"])
        and bool(shadow_high_score_bucket["realized_return_exceeds_execution_buffer"])
    )
    artifact = {
        "schema_version": ACTION_VALUE_CALIBRATION_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "calibration_method": "validation_bucketed_action_bias_correction_v1",
        "calibration_granularity": "action_price_time_raw_score_bucket_v1",
        "bucketed_calibration_enabled": True,
        "fallback_calibration_method": "validation_action_bias_correction",
        "bucket_shrinkage_enabled": True,
        "bucket_shrinkage_prior": ACTION_VALUE_BUCKET_SHRINKAGE_PRIOR,
        "calibration_fit_split": "validation",
        "calibration_evaluation_split": "shadow",
        "calibration_uses_training_split": False,
        "raw_expected_pnl_per_notional_by_action_field": "expected_return_by_action",
        "calibrated_expected_pnl_per_notional_by_action_field": (
            "calibrated_expected_pnl_per_notional_by_action"
        ),
        "calibration_support_count": len(calibration_examples),
        "calibration_bucket_count": len(calibration_buckets),
        "action_calibration_bucket_count": len(action_buckets),
        "min_calibration_support_per_action": (
            ACTION_VALUE_MIN_CALIBRATION_SUPPORT_PER_ACTION
        ),
        "min_calibration_bucket_support": ACTION_VALUE_MIN_BUCKET_SUPPORT,
        "high_score_min_support": ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT,
        "high_score_execution_buffer": execution_buffer,
        "calibration_support_passed": support_passed,
        "calibration_support_level": "sufficient" if support_passed else "insufficient",
        "calibration_quality_passed": calibration_quality_passed,
        "calibration_quality_gates": {
            "shadow_calibrated_mae_not_worse": shadow_calibrated_mae_not_worse,
            "shadow_raw_mae": shadow_mae_comparison["raw_mae"],
            "shadow_action_level_calibrated_mae": shadow_mae_comparison[
                "action_level_calibrated_mae"
            ],
            "shadow_bucketed_calibrated_mae": shadow_mae_comparison[
                "bucketed_calibrated_mae"
            ],
            "mae_tolerance": ACTION_VALUE_QUALITY_MAE_TOLERANCE,
            "high_score_bucket_min_support_passed": (
                shadow_high_score_bucket["support_passed"]
            ),
            "high_score_bucket_realized_return_exceeds_buffer": (
                shadow_high_score_bucket["realized_return_exceeds_execution_buffer"]
            ),
            "high_score_threshold": ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            "high_score_min_support": ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT,
            "high_score_execution_buffer": execution_buffer,
        },
        "shadow_high_score_bucket": shadow_high_score_bucket,
        "shadow_mae_comparison": shadow_mae_comparison,
        "action_corrections": corrections,
        "action_calibration_buckets": action_buckets,
        "calibration_buckets": calibration_buckets,
        "calibration_metrics": calibration_metrics,
        "shadow_evaluation_metrics": shadow_evaluation_metrics,
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
    calibration_buckets = dict(calibration_artifact.get("calibration_buckets", {}))
    missing = set(ACTION_VALUE_LABEL_ACTIONS) - set(corrections)
    if missing:
        raise ValueError(
            "action_value_calibration missing corrections: " + ", ".join(sorted(missing))
        )
    return tuple(
        _calibrated_prediction(
            prediction=prediction,
            action_corrections=corrections,
            calibration_buckets=calibration_buckets,
            calibration_artifact=calibration_artifact,
        )
        for prediction in predictions
    )


def _calibrated_prediction(
    *,
    prediction: PolymarketPolicyPrediction,
    action_corrections: dict[str, float],
    calibration_buckets: dict[str, Any],
    calibration_artifact: dict[str, Any],
) -> PolymarketPolicyPrediction:
    if not prediction.action_value_head_enabled:
        return prediction
    calibrated = {
        action: _calibrated_action_value(
            action=action,
            prediction=prediction,
            action_corrections=action_corrections,
            calibration_buckets=calibration_buckets,
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


def _calibrated_action_value(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
    action_corrections: dict[str, float],
    calibration_buckets: dict[str, Any],
) -> float:
    raw = float(prediction.expected_return_by_action[action])
    bucket_key = _calibration_bucket_key(action=action, prediction=prediction)
    bucket = calibration_buckets.get(bucket_key, {})
    if int(bucket.get("support_count", 0)) >= ACTION_VALUE_MIN_BUCKET_SUPPORT:
        correction = float(bucket["correction"])
    else:
        correction = float(action_corrections[action])
    return _clamp(raw + correction, -10.0, 10.0)


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
    action_corrections: dict[str, float],
    calibration_buckets: dict[str, Any],
) -> dict[str, Any]:
    raw_errors = []
    calibrated_errors = []
    for example, prediction in zip(examples, predictions, strict=True):
        for action in ACTION_VALUE_LABEL_ACTIONS:
            target = float(example.action_return_targets[action])
            raw = float(prediction.expected_return_by_action[action])
            calibrated = _calibrated_action_value(
                action=action,
                prediction=prediction,
                action_corrections=action_corrections,
                calibration_buckets=calibration_buckets,
            )
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


def _mae_comparison(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    action_corrections: dict[str, float],
    calibration_buckets: dict[str, Any],
) -> dict[str, Any]:
    raw_errors = []
    action_level_errors = []
    bucketed_errors = []
    for example, prediction in zip(examples, predictions, strict=True):
        for action in ACTION_VALUE_LABEL_ACTIONS:
            target = float(example.action_return_targets[action])
            raw = float(prediction.expected_return_by_action[action])
            action_level = _clamp(
                raw + float(action_corrections[action]),
                -10.0,
                10.0,
            )
            bucketed = _calibrated_action_value(
                action=action,
                prediction=prediction,
                action_corrections=action_corrections,
                calibration_buckets=calibration_buckets,
            )
            raw_errors.append(target - raw)
            action_level_errors.append(target - action_level)
            bucketed_errors.append(target - bucketed)
    return {
        "sample_count": len(examples),
        "action_value_point_count": len(raw_errors),
        "raw_mae": _mean([abs(value) for value in raw_errors]),
        "action_level_calibrated_mae": _mean(
            [abs(value) for value in action_level_errors]
        ),
        "bucketed_calibrated_mae": _mean([abs(value) for value in bucketed_errors]),
        "action_level_delta_vs_raw_mae": (
            _mean([abs(value) for value in action_level_errors])
            - _mean([abs(value) for value in raw_errors])
        ),
        "bucketed_delta_vs_raw_mae": (
            _mean([abs(value) for value in bucketed_errors])
            - _mean([abs(value) for value in raw_errors])
        ),
    }


def _high_score_bucket_report(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    action_corrections: dict[str, float],
    calibration_buckets: dict[str, Any],
    execution_buffer: float,
) -> dict[str, Any]:
    realized_returns = []
    calibrated_scores = []
    actions = []
    for example, prediction in zip(examples, predictions, strict=True):
        calibrated_returns = {
            action: _calibrated_action_value(
                action=action,
                prediction=prediction,
                action_corrections=action_corrections,
                calibration_buckets=calibration_buckets,
            )
            for action in ACTION_VALUE_LABEL_ACTIONS
        }
        best_action, best_return, _, _ = _rank_actions(calibrated_returns)
        if best_action != "NO_TRADE" and best_return >= ACTION_VALUE_HIGH_SCORE_THRESHOLD:
            actions.append(best_action)
            calibrated_scores.append(best_return)
            realized_returns.append(float(example.action_return_targets[best_action]))
    realized_return_mean = _mean(realized_returns)
    support_passed = len(realized_returns) >= ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT
    realized_return_exceeds_buffer = (
        support_passed and realized_return_mean > execution_buffer
    )
    return {
        "bucket_name": "shadow_calibrated_best_score_ge_threshold",
        "score_threshold": ACTION_VALUE_HIGH_SCORE_THRESHOLD,
        "min_support": ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT,
        "execution_buffer": execution_buffer,
        "support_count": len(realized_returns),
        "support_passed": support_passed,
        "realized_return_mean": realized_return_mean,
        "calibrated_score_mean": _mean(calibrated_scores),
        "realized_return_positive": len(realized_returns) > 0 and realized_return_mean > 0.0,
        "realized_return_exceeds_execution_buffer": realized_return_exceeds_buffer,
        "best_action_counts": dict(sorted(_counts(actions).items())),
    }


def _bucket_payloads(
    bucket_rows: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, Any]]:
    payloads = {}
    for bucket_key, rows in sorted(bucket_rows.items()):
        residuals = [row["residual"] for row in rows]
        raw_correction = _mean(residuals)
        shrunk_correction = _shrink_correction(
            correction=raw_correction,
            support_count=len(rows),
        )
        correction = _clamp(
            shrunk_correction,
            -ACTION_VALUE_CORRECTION_LIMIT,
            ACTION_VALUE_CORRECTION_LIMIT,
        )
        payloads[bucket_key] = {
            "bucket_key": bucket_key,
            "support_count": len(rows),
            "raw_expected_pnl_per_notional_mean": _mean(
                [row["raw_expected_pnl_per_notional"] for row in rows]
            ),
            "target_expected_pnl_per_notional_mean": _mean(
                [row["target_expected_pnl_per_notional"] for row in rows]
            ),
            "residual_mean": raw_correction,
            "residual_mae": _mean([abs(value) for value in residuals]),
            "unshrunk_correction": raw_correction,
            "shrinkage_prior": ACTION_VALUE_BUCKET_SHRINKAGE_PRIOR,
            "shrinkage_weight": _shrinkage_weight(len(rows)),
            "correction": correction,
            "correction_clipped": not math.isclose(
                correction,
                shrunk_correction,
                abs_tol=1e-12,
            ),
        }
    return payloads


def _calibration_bucket_key(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
) -> str:
    raw_score = float(prediction.expected_return_by_action[action])
    return "|".join(
        (
            f"action={action}",
            f"price={_price_bucket(action=action, prediction=prediction)}",
            f"time={_time_to_close_bucket(prediction)}",
            f"raw={_raw_score_bucket(raw_score)}",
        )
    )


def _price_bucket(*, action: str, prediction: PolymarketPolicyPrediction) -> str:
    features = prediction.features
    price = None
    if action.startswith("BUY_UP_"):
        price = features.get("up_ask")
    elif action.startswith("BUY_DOWN_"):
        price = features.get("down_ask")
    elif action == "NO_TRADE":
        return "none"
    if price is None:
        return "unknown"
    return _number_bucket(
        float(price),
        thresholds=(0.20, 0.40, 0.60, 0.80),
        labels=("<0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", ">=0.80"),
    )


def _time_to_close_bucket(prediction: PolymarketPolicyPrediction) -> str:
    seconds = float(prediction.features.get("time_to_close_seconds", 0.0))
    return _number_bucket(
        seconds,
        thresholds=(30.0, 60.0, 180.0, 300.0, 900.0),
        labels=("0-30s", "30-60s", "1-3m", "3-5m", "5-15m", "15m+"),
    )


def _raw_score_bucket(raw_score: float) -> str:
    return _number_bucket(
        raw_score,
        thresholds=(-0.10, 0.0, 0.05, 0.15),
        labels=("<-0.10", "-0.10-0.00", "0.00-0.05", "0.05-0.15", ">=0.15"),
    )


def _number_bucket(
    value: float,
    *,
    thresholds: tuple[float, ...],
    labels: tuple[str, ...],
) -> str:
    for threshold, label in zip(thresholds, labels, strict=False):
        if value < threshold:
            return label
    return labels[-1]


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _shrink_correction(*, correction: float, support_count: int) -> float:
    return float(correction) * _shrinkage_weight(support_count)


def _shrinkage_weight(support_count: int) -> float:
    return float(support_count) / (
        float(support_count) + ACTION_VALUE_BUCKET_SHRINKAGE_PRIOR
    )


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
