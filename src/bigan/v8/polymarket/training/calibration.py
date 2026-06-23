"""Calibration and validation reports for Polymarket policy predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.training.contracts import (
    PolymarketPolicyExample,
    PolymarketPolicyPrediction,
    compact_safety_fields,
)

TIME_TO_CLOSE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-30s", 0.0, 30.0),
    ("30-60s", 30.0, 60.0),
    ("1-3m", 60.0, 180.0),
    ("3-5m", 180.0, 300.0),
    ("5-15m", 300.0, 900.0),
    ("15m+", 900.0, math.inf),
)
THRESHOLDS = (0.50, 0.55, 0.60, 0.65)
EPSILON = 1e-12
OUT_OF_SAMPLE_SPLITS = ("validation", "shadow")


def calibration_report(
    predictions: tuple[PolymarketPolicyPrediction, ...],
) -> dict[str, Any]:
    """Build calibration buckets for scored examples."""

    bucket_rows: dict[str, list[PolymarketPolicyPrediction]] = defaultdict(list)
    for prediction in predictions:
        bucket_rows[prediction.calibration_bucket].append(prediction)
    buckets = {}
    total_error = 0.0
    for bucket, rows in sorted(bucket_rows.items()):
        avg_prediction = sum(row.estimated_up_probability for row in rows) / len(rows)
        avg_target = sum(_target(row) for row in rows) / len(rows)
        error = abs(avg_prediction - avg_target)
        total_error += error * len(rows)
        buckets[bucket] = {
            "sample_count": len(rows),
            "avg_prediction": avg_prediction,
            "avg_target": avg_target,
            "absolute_error": error,
        }
    return {
        "schema_version": "bigan-v8-polymarket-policy-calibration-v1",
        "sample_count": len(predictions),
        "calibration_error": 0.0 if not predictions else total_error / len(predictions),
        "buckets": buckets,
        **compact_safety_fields(),
    }


def split_calibration_report(
    *,
    train_predictions: tuple[PolymarketPolicyPrediction, ...],
    validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    primary_calibration_split: str = "validation",
) -> dict[str, Any]:
    """Build split-specific calibration evidence with an explicit primary split."""

    if primary_calibration_split not in OUT_OF_SAMPLE_SPLITS:
        raise ValueError("primary_calibration_split must be validation or shadow")
    split_reports = {
        "train": calibration_report(train_predictions),
        "validation": calibration_report(validation_predictions),
        "shadow": calibration_report(shadow_predictions),
    }
    primary = split_reports[primary_calibration_split]
    return {
        "schema_version": "bigan-v8-polymarket-policy-split-calibration-v1",
        "primary_calibration_split": primary_calibration_split,
        "sample_count": primary["sample_count"],
        "calibration_error": primary["calibration_error"],
        "buckets": primary["buckets"],
        "primary_calibration": primary,
        "train_calibration": split_reports["train"],
        "validation_calibration": split_reports["validation"],
        "shadow_calibration": split_reports["shadow"],
        "sample_counts_by_split": {
            split_name: report["sample_count"]
            for split_name, report in split_reports.items()
        },
        **compact_safety_fields(),
    }


def validation_report(
    *,
    validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    train_examples: tuple[PolymarketPolicyExample, ...],
    evaluation_split: str = "validation",
) -> dict[str, Any]:
    """Build validation metrics by family and time-to-close bucket."""

    if evaluation_split not in OUT_OF_SAMPLE_SPLITS:
        raise ValueError("evaluation_split must be validation or shadow")
    train_baseline = sum(example.target_up_probability for example in train_examples) / len(
        train_examples
    )
    validation_metrics = _metrics(validation_predictions)
    naive_brier = _brier_for_constant(validation_predictions, train_baseline)
    by_family = {
        family: _metrics(
            tuple(row for row in validation_predictions if row.market_family == family)
        )
        for family in BTC_UPDOWN_MARKET_HORIZONS_MS
    }
    by_time_to_close = {
        name: _metrics(
            tuple(
                row
                for row in validation_predictions
                if lower <= float(row.features.get("time_to_close_seconds", 0.0)) < upper
            )
        )
        for name, lower, upper in TIME_TO_CLOSE_BUCKETS
    }
    model_brier = float(validation_metrics["brier_score"])
    return {
        "schema_version": "bigan-v8-polymarket-policy-validation-v1",
        "evaluation_split": evaluation_split,
        "out_of_sample_validation": True,
        "validation": validation_metrics,
        "naive_baseline": {
            "probability": train_baseline,
            "brier_score": naive_brier,
        },
        "model_is_calibrated_better_than_naive_baseline": model_brier <= naive_brier + 1e-12,
        "metrics_by_market_family": by_family,
        "metrics_by_time_to_close_bucket": by_time_to_close,
        **compact_safety_fields(),
    }


def _metrics(predictions: tuple[PolymarketPolicyPrediction, ...]) -> dict[str, Any]:
    if not predictions:
        return {
            "logloss": None,
            "brier_score": None,
            "calibration_error": None,
            "auc": None,
            "accuracy_by_threshold": {str(threshold): None for threshold in THRESHOLDS},
            "sample_count": 0,
            "market_count": 0,
        }
    targets = [_target(row) for row in predictions]
    probabilities = [row.estimated_up_probability for row in predictions]
    calibration = calibration_report(predictions)
    return {
        "logloss": _logloss(targets, probabilities),
        "brier_score": _brier(targets, probabilities),
        "calibration_error": calibration["calibration_error"],
        "auc": _auc(targets, probabilities),
        "accuracy_by_threshold": {
            str(threshold): _accuracy(targets, probabilities, threshold)
            for threshold in THRESHOLDS
        },
        "sample_count": len(predictions),
        "market_count": len({row.market_id for row in predictions}),
    }


def _target(prediction: PolymarketPolicyPrediction) -> float:
    if prediction.target_up_probability is None:
        raise ValueError("prediction target is required for validation")
    return prediction.target_up_probability


def _logloss(targets: list[float], probabilities: list[float]) -> float:
    total = 0.0
    for target, probability in zip(targets, probabilities, strict=True):
        p = min(1.0 - EPSILON, max(EPSILON, probability))
        total += -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))
    return total / len(targets)


def _brier(targets: list[float], probabilities: list[float]) -> float:
    return sum(
        (probability - target) ** 2
        for target, probability in zip(targets, probabilities, strict=True)
    ) / len(targets)


def _brier_for_constant(
    predictions: tuple[PolymarketPolicyPrediction, ...],
    probability: float,
) -> float:
    if not predictions:
        return 0.0
    return sum((probability - _target(row)) ** 2 for row in predictions) / len(predictions)


def _auc(targets: list[float], probabilities: list[float]) -> float | None:
    binary_pairs = [
        (target, probability)
        for target, probability in zip(targets, probabilities, strict=True)
        if target in (0.0, 1.0)
    ]
    positives = [score for target, score in binary_pairs if target == 1.0]
    negatives = [score for target, score in binary_pairs if target == 0.0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _accuracy(targets: list[float], probabilities: list[float], threshold: float) -> float | None:
    binary_pairs = [
        (target, probability)
        for target, probability in zip(targets, probabilities, strict=True)
        if target in (0.0, 1.0)
    ]
    if not binary_pairs:
        return None
    correct = sum(
        int((probability >= threshold) == (target == 1.0))
        for target, probability in binary_pairs
    )
    return correct / len(binary_pairs)
