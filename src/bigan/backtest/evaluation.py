"""Pure prediction-quality metrics for backtests (issue #11)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Binary-classification metrics at one probability threshold."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    support: int


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Observed outcome frequency for one prediction-probability bucket."""

    bin_start: float
    bin_end: float
    count: int
    mean_predicted_probability: float | None
    observed_positive_rate: float | None


@dataclass(frozen=True, slots=True)
class PredictionEvaluationReport:
    """JSON-serializable prediction-quality report."""

    sample_count: int
    positive_count: int
    negative_count: int
    brier_score: float
    roc_auc: float | None
    pr_auc: float | None
    thresholds: tuple[ThresholdMetrics, ...]
    calibration_bins: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "brier_score": self.brier_score,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "thresholds": [asdict(row) for row in self.thresholds],
            "calibration_bins": [asdict(row) for row in self.calibration_bins],
        }


def evaluate_predictions(
    *,
    y_true: Sequence[bool | int],
    y_prob: Sequence[float],
    thresholds: Sequence[float] = (0.5,),
    calibration_bin_count: int = 10,
) -> PredictionEvaluationReport:
    """Evaluate probability predictions against binary labels."""

    labels = _coerce_labels(y_true)
    probabilities = _coerce_probabilities(y_prob)
    if len(labels) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same length")
    if not labels:
        raise ValueError("at least one prediction is required")
    if calibration_bin_count <= 0:
        raise ValueError("calibration_bin_count must be positive")
    checked_thresholds = tuple(_coerce_thresholds(thresholds))

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    brier_score = sum((prob - label) ** 2 for label, prob in zip(labels, probabilities, strict=True)) / len(labels)

    return PredictionEvaluationReport(
        sample_count=len(labels),
        positive_count=positive_count,
        negative_count=negative_count,
        brier_score=brier_score,
        roc_auc=_roc_auc(labels, probabilities),
        pr_auc=_average_precision(labels, probabilities),
        thresholds=tuple(
            _threshold_metrics(labels, probabilities, threshold)
            for threshold in checked_thresholds
        ),
        calibration_bins=tuple(
            _calibration_bins(labels, probabilities, calibration_bin_count)
        ),
    )


def save_evaluation_report(
    report: PredictionEvaluationReport,
    path: Path | str,
) -> None:
    """Persist a prediction-quality report as pretty JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _coerce_labels(values: Sequence[bool | int]) -> list[int]:
    labels: list[int] = []
    for value in values:
        if value in (True, 1):
            labels.append(1)
        elif value in (False, 0):
            labels.append(0)
        else:
            raise ValueError(f"labels must be bool/0/1, got {value!r}")
    return labels


def _coerce_probabilities(values: Sequence[float]) -> list[float]:
    probabilities: list[float] = []
    for value in values:
        prob = float(value)
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"probabilities must be in [0, 1], got {value!r}")
        probabilities.append(prob)
    return probabilities


def _coerce_thresholds(values: Sequence[float]) -> list[float]:
    thresholds: list[float] = []
    for value in values:
        threshold = float(value)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"thresholds must be in [0, 1], got {value!r}")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def _threshold_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> ThresholdMetrics:
    tp = fp = tn = fn = 0
    for label, prob in zip(labels, probabilities, strict=True):
        predicted = prob >= threshold
        if label and predicted:
            tp += 1
        elif label and not predicted:
            fn += 1
        elif not label and predicted:
            fp += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return ThresholdMetrics(
        threshold=threshold,
        accuracy=_safe_div(tp + tn, len(labels)),
        precision=precision,
        recall=recall,
        f1=_safe_div(2 * precision * recall, precision + recall),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        support=len(labels),
    )


def _roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(enumerate(probabilities), key=lambda pair: pair[1])
    ranks = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][1] == ranked[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ranked[k][0]] = avg_rank
        i = j

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None

    true_positives = 0
    precision_sum = 0.0
    ranked = sorted(zip(probabilities, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _calibration_bins(
    labels: Sequence[int],
    probabilities: Sequence[float],
    bin_count: int,
) -> list[CalibrationBin]:
    bins: list[CalibrationBin] = []
    for idx in range(bin_count):
        start = idx / bin_count
        end = (idx + 1) / bin_count
        members = [
            (label, prob)
            for label, prob in zip(labels, probabilities, strict=True)
            if (start <= prob < end) or (idx == bin_count - 1 and prob == 1.0)
        ]
        if members:
            count = len(members)
            bins.append(
                CalibrationBin(
                    bin_start=start,
                    bin_end=end,
                    count=count,
                    mean_predicted_probability=sum(prob for _, prob in members) / count,
                    observed_positive_rate=sum(label for label, _ in members) / count,
                )
            )
        else:
            bins.append(
                CalibrationBin(
                    bin_start=start,
                    bin_end=end,
                    count=0,
                    mean_predicted_probability=None,
                    observed_positive_rate=None,
                )
            )
    return bins


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
