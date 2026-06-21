"""Integrity validation and leakage detection for v8 Phase 0."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from bigan.v8.phase0.alignment import TimeAlignmentEngine
from bigan.v8.phase0.contracts import FEATURE_COLUMNS, FeatureVector, Label, MarketData


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Hard-gate thresholds for the Phase 0 firewall."""

    max_abs_feature_future_corr: float = 0.98
    min_correlation_rows: int = 30
    numeric_tolerance: float = 1e-9
    forbidden_feature_tokens: tuple[str, ...] = ("future", "target", "label", "lead", "forward")
    min_drift_rows: int = 60
    drift_bins: int = 10
    max_ks_statistic: float = 0.95
    max_psi: float = 10.0
    max_kl_divergence: float = 10.0
    drift_excluded_columns: tuple[str, ...] = ("minute_of_day", "day_of_week")

    def __post_init__(self) -> None:
        if not 0.0 < self.max_abs_feature_future_corr <= 1.0:
            raise ValueError("max_abs_feature_future_corr must be in (0, 1]")
        if self.min_correlation_rows < 3:
            raise ValueError("min_correlation_rows must be at least 3")
        if self.numeric_tolerance <= 0:
            raise ValueError("numeric_tolerance must be positive")
        if self.min_drift_rows < 4:
            raise ValueError("min_drift_rows must be at least 4")
        if self.drift_bins < 2:
            raise ValueError("drift_bins must be at least 2")
        if not 0.0 < self.max_ks_statistic <= 1.0:
            raise ValueError("max_ks_statistic must be in (0, 1]")
        if self.max_psi < 0:
            raise ValueError("max_psi must be non-negative")
        if self.max_kl_divergence < 0:
            raise ValueError("max_kl_divergence must be non-negative")


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    """One validation failure."""

    code: str
    message: str
    severity: str = "error"
    row_count: int = 0
    column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationReport:
    """Aggregate Phase 0 validation result."""

    failures: list[ValidationFailure] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(failure.severity == "error" for failure in self.failures)

    def extend(self, other: ValidationReport) -> None:
        self.failures.extend(other.failures)
        self.metrics.update(other.metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
            "metrics": self.metrics,
            "acceptance_criteria": {
                "zero_detectable_leakage": self.passed,
                "feature_causality_strictly_enforced": not any(
                    failure.code
                    in {
                        "feature_causality",
                        "rolling_window",
                        "cross_timeframe_leakage",
                        "forbidden_feature_name",
                    }
                    for failure in self.failures
                ),
                "label_correctness_verified": not any(
                    failure.code.startswith("label_") for failure in self.failures
                ),
                "statistical_validity_verified": not any(
                    failure.code in {"feature_distribution_drift"} for failure in self.failures
                ),
                "cost_model_realistic": not any(
                    failure.code in {"label_cost_math", "negative_cost"} for failure in self.failures
                ),
                "dataset_reproducible": bool(self.metrics.get("dataset_hash")),
            },
        }


class IntegrityValidator:
    """Run all Phase 0 hard gates."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()
        self.alignment_engine = TimeAlignmentEngine()

    def validate_all(
        self,
        *,
        features: list[FeatureVector],
        labels: list[Label],
        market_data: list[MarketData] | None = None,
        dataset_hash: str | None = None,
    ) -> ValidationReport:
        report = ValidationReport(metrics={"feature_rows": len(features), "label_rows": len(labels)})
        if dataset_hash is not None:
            report.metrics["dataset_hash"] = dataset_hash
        for check in (
            self.validate_feature_causality(features),
            self.validate_feature_label_causality(features, labels),
            self.validate_rolling_windows(features),
            self.detect_cross_timeframe_leakage(features),
            self.validate_label_consistency(labels, market_data=market_data),
            self.check_feature_future_correlations(features, labels),
            self.validate_statistical_integrity(features),
        ):
            report.extend(check)
        return report

    def validate_feature_causality(
        self,
        features: Iterable[FeatureVector],
    ) -> ValidationReport:
        failures: list[ValidationFailure] = []
        offenders = []
        for feature in features:
            if feature.feature_cutoff_ts > feature.decision_ts or feature.max_input_ts > feature.decision_ts:
                offenders.append(feature)
                continue
            if any(
                provenance.input_end_ts > feature.decision_ts
                or provenance.available_at_ts > feature.decision_ts
                for provenance in feature.provenance.values()
            ):
                offenders.append(feature)
        if offenders:
            failures.append(
                ValidationFailure(
                    code="feature_causality",
                    message="features use inputs unavailable at the decision timestamp",
                    row_count=len(offenders),
                )
            )
        return ValidationReport(failures=failures)

    def validate_feature_label_causality(
        self,
        features: list[FeatureVector],
        labels: list[Label],
    ) -> ValidationReport:
        """Enforce ``max(feature_timestamp) <= label_start_time`` at runtime."""

        feature_by_key = {
            (feature.source, feature.instrument_id, feature.decision_ts): feature
            for feature in features
        }
        missing_feature_count = 0
        causality_offenders = 0
        for label in labels:
            feature = feature_by_key.get((label.source, label.instrument_id, label.decision_ts))
            if feature is None:
                missing_feature_count += 1
                continue
            max_feature_timestamp = max(
                [feature.max_input_ts]
                + [provenance.input_end_ts for provenance in feature.provenance.values()]
                + [provenance.available_at_ts for provenance in feature.provenance.values()]
            )
            if max_feature_timestamp > label.decision_ts:
                causality_offenders += 1

        failures: list[ValidationFailure] = []
        if missing_feature_count:
            failures.append(
                ValidationFailure(
                    code="feature_label_missing_feature",
                    message="labels exist without a matching feature row",
                    row_count=missing_feature_count,
                )
            )
        if causality_offenders:
            failures.append(
                ValidationFailure(
                    code="feature_label_causality",
                    message="max(feature_timestamp) exceeds label_start_time",
                    row_count=causality_offenders,
                )
            )
        return ValidationReport(failures=failures)

    def validate_rolling_windows(
        self,
        features: Iterable[FeatureVector],
    ) -> ValidationReport:
        bad_count = 0
        for feature in features:
            for provenance in feature.provenance.values():
                if provenance.input_start_ts > provenance.input_end_ts or provenance.input_end_ts > feature.feature_cutoff_ts or provenance.available_at_ts > feature.decision_ts:
                    bad_count += 1
        if not bad_count:
            return ValidationReport()
        return ValidationReport(
            failures=[
                ValidationFailure(
                    code="rolling_window",
                    message="feature provenance falls outside its declared rolling window",
                    row_count=bad_count,
                )
            ]
        )

    def detect_cross_timeframe_leakage(
        self,
        features: Iterable[FeatureVector],
    ) -> ValidationReport:
        failures: list[ValidationFailure] = []
        name_offenders: dict[str, int] = defaultdict(int)
        timeframe_count = 0
        for feature in features:
            for name in feature.features:
                lowered = name.lower()
                if any(token in lowered for token in self.config.forbidden_feature_tokens):
                    name_offenders[name] += 1
            for provenance in feature.provenance.values():
                if provenance.input_end_ts > feature.decision_ts or provenance.available_at_ts > feature.decision_ts or (
                    provenance.source_timeframe_ms is not None
                    and provenance.source_timeframe_ms > 0
                    and provenance.input_end_ts > feature.feature_cutoff_ts
                ):
                    timeframe_count += 1

        for name, count in sorted(name_offenders.items()):
            failures.append(
                ValidationFailure(
                    code="forbidden_feature_name",
                    message=f"feature name suggests future or label leakage: {name}",
                    row_count=count,
                    column=name,
                )
            )
        if timeframe_count:
            failures.append(
                ValidationFailure(
                    code="cross_timeframe_leakage",
                    message="feature provenance crosses the decision timestamp",
                    row_count=timeframe_count,
                )
            )
        return ValidationReport(failures=failures)

    def check_feature_future_correlations(
        self,
        features: list[FeatureVector],
        labels: list[Label],
    ) -> ValidationReport:
        label_by_key: dict[tuple[str, str, int], list[Label]] = defaultdict(list)
        for label in labels:
            label_by_key[(label.source, label.instrument_id, label.decision_ts)].append(label)
        correlations: dict[str, float] = {}
        failures: list[ValidationFailure] = []
        for column in _feature_columns_present(features):
            x_values: list[float] = []
            y_values: list[float] = []
            for feature in features:
                value = feature.features.get(column)
                if value is None:
                    continue
                for label in label_by_key.get((feature.source, feature.instrument_id, feature.decision_ts), []):
                    x_values.append(float(value))
                    y_values.append(label.net_return)
            if len(x_values) < self.config.min_correlation_rows:
                continue
            corr = _pearson(x_values, y_values)
            if corr is None:
                continue
            correlations[column] = corr
            if abs(corr) >= self.config.max_abs_feature_future_corr:
                failures.append(
                    ValidationFailure(
                        code="feature_future_correlation",
                        message=(
                            "feature is suspiciously correlated with future net return "
                            f"({corr:.6f})"
                        ),
                        row_count=len(x_values),
                        column=column,
                    )
                )
        return ValidationReport(
            failures=failures,
            metrics={"feature_future_correlations": correlations},
        )

    def validate_statistical_integrity(
        self,
        features: list[FeatureVector],
    ) -> ValidationReport:
        """Check temporal feature-distribution stability with KS, PSI, and KL."""

        ordered = sorted(features, key=lambda feature: feature.decision_ts)
        if len(ordered) < self.config.min_drift_rows:
            return ValidationReport(
                metrics={
                    "statistical_integrity": {
                        "checked": False,
                        "reason": "insufficient_rows",
                        "row_count": len(ordered),
                    }
                }
            )

        split_index = max(1, int(len(ordered) * 0.7))
        reference = ordered[:split_index]
        candidate = ordered[split_index:]
        drift_metrics: dict[str, dict[str, float]] = {}
        failures: list[ValidationFailure] = []
        excluded = set(self.config.drift_excluded_columns)
        for column in _feature_columns_present(ordered):
            if column in excluded:
                continue
            ref_values = _finite_feature_values(reference, column)
            cand_values = _finite_feature_values(candidate, column)
            if len(ref_values) < 2 or len(cand_values) < 2:
                continue
            stats = _distribution_metrics(
                ref_values,
                cand_values,
                bins=self.config.drift_bins,
            )
            if stats is None:
                continue
            drift_metrics[column] = stats
            if (
                stats["ks_statistic"] > self.config.max_ks_statistic
                or stats["psi"] > self.config.max_psi
                or stats["kl_divergence"] > self.config.max_kl_divergence
            ):
                failures.append(
                    ValidationFailure(
                        code="feature_distribution_drift",
                        message=(
                            "feature distribution drift exceeds KS/PSI/KL thresholds"
                        ),
                        row_count=len(ref_values) + len(cand_values),
                        column=column,
                    )
                )
        return ValidationReport(
            failures=failures,
            metrics={
                "statistical_integrity": {
                    "checked": bool(drift_metrics),
                    "metrics": drift_metrics,
                }
            },
        )

    def validate_label_consistency(
        self,
        labels: list[Label],
        *,
        market_data: list[MarketData] | None = None,
    ) -> ValidationReport:
        failures: list[ValidationFailure] = []
        bad_time = [
            label
            for label in labels
            if label.label_ts < label.decision_ts + label.horizon_ms
        ]
        if bad_time:
            failures.append(
                ValidationFailure(
                    code="label_horizon",
                    message="label_ts must be at or after decision_ts + horizon_ms",
                    row_count=len(bad_time),
                )
            )

        cost_bad = [
            label
            for label in labels
            if abs(
                label.total_cost
                - (
                    label.spread_cost
                    + label.fee_cost
                    + label.slippage_cost
                    + label.liquidity_impact_cost
                )
            )
            > self.config.numeric_tolerance
            or abs(label.net_return - (label.gross_return - label.total_cost))
            > self.config.numeric_tolerance
            or label.is_positive != (label.net_return > 0.0)
        ]
        if cost_bad:
            failures.append(
                ValidationFailure(
                    code="label_cost_math",
                    message="label net_return or total_cost is internally inconsistent",
                    row_count=len(cost_bad),
                )
            )

        negative_cost = [label for label in labels if label.total_cost < 0.0]
        if negative_cost:
            failures.append(
                ValidationFailure(
                    code="negative_cost",
                    message="label contains negative trading costs",
                    row_count=len(negative_cost),
                )
            )

        duplicate_count = _duplicate_label_count(labels)
        if duplicate_count:
            failures.append(
                ValidationFailure(
                    code="label_duplicate",
                    message="duplicate label key detected",
                    row_count=duplicate_count,
                )
            )

        horizon_failures = _horizon_order_failures(labels)
        if horizon_failures:
            failures.append(
                ValidationFailure(
                    code="label_horizon_order",
                    message="label target timestamps are not monotonic across horizons",
                    row_count=horizon_failures,
                )
            )

        if market_data is not None and labels:
            failures.extend(self._validate_against_market_data(labels, market_data))

        return ValidationReport(failures=failures)

    def _validate_against_market_data(
        self,
        labels: list[Label],
        market_data: list[MarketData],
    ) -> list[ValidationFailure]:
        series = self.alignment_engine.align_market_data(market_data)
        mismatches = 0
        for label in labels:
            entry = series.latest_at(label.source, label.instrument_id, label.decision_ts)
            exit_row = series.first_at_or_after(
                label.source,
                label.instrument_id,
                label.decision_ts + label.horizon_ms,
            )
            if entry is None or exit_row is None:
                mismatches += 1
                continue
            expected_gross = label.side * (
                (exit_row.effective_mid_price / entry.effective_mid_price) - 1.0
            )
            if exit_row.ts != label.label_ts or abs(label.entry_price - entry.effective_mid_price) > self.config.numeric_tolerance or abs(label.exit_price - exit_row.effective_mid_price) > self.config.numeric_tolerance or abs(label.gross_return - expected_gross) > self.config.numeric_tolerance:
                mismatches += 1
        if not mismatches:
            return []
        return [
            ValidationFailure(
                code="label_market_consistency",
                message="labels do not match market-data entry/exit prices or gross returns",
                row_count=mismatches,
            )
        ]

    def time_reversal_performance_collapse(
        self,
        *,
        features: list[FeatureVector],
        labels: list[Label],
        feature_column: str,
        horizon_ms: int | None = None,
        min_corr_drop: float = 0.25,
    ) -> dict[str, float | bool | int | None]:
        pairs = _feature_label_pairs(features, labels, feature_column, horizon_ms=horizon_ms)
        if len(pairs) < 3:
            return {
                "passed": False,
                "row_count": len(pairs),
                "baseline_corr": None,
                "reversed_corr": None,
                "corr_drop": None,
            }
        x = [pair[0] for pair in pairs]
        y = [pair[1] for pair in pairs]
        baseline_corr = _pearson(x, y) or 0.0
        reversed_corr = _pearson(list(reversed(x)), y) or 0.0
        corr_drop = abs(baseline_corr) - abs(reversed_corr)
        return {
            "passed": corr_drop >= min_corr_drop,
            "row_count": len(pairs),
            "baseline_corr": baseline_corr,
            "reversed_corr": reversed_corr,
            "corr_drop": corr_drop,
        }


def _feature_columns_present(features: list[FeatureVector]) -> tuple[str, ...]:
    columns = set(FEATURE_COLUMNS)
    for feature in features:
        columns.update(feature.features)
    return tuple(sorted(columns))


def _pearson(x_values: list[float], y_values: list[float]) -> float | None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3:
        return None
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _finite_feature_values(features: list[FeatureVector], column: str) -> list[float]:
    values = [
        float(feature.features[column])
        for feature in features
        if column in feature.features and feature.features[column] is not None
    ]
    array = np.asarray(values, dtype=float)
    return [float(value) for value in array[np.isfinite(array)]]


def _distribution_metrics(
    reference_values: list[float],
    candidate_values: list[float],
    *,
    bins: int,
) -> dict[str, float] | None:
    reference = np.asarray(reference_values, dtype=float)
    candidate = np.asarray(candidate_values, dtype=float)
    combined = np.concatenate([reference, candidate])
    if np.min(combined) == np.max(combined):
        return None
    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(float(np.min(combined)), float(np.max(combined)), bins + 1)
    edges[0] = min(edges[0], float(np.min(combined)))
    edges[-1] = max(edges[-1], float(np.max(combined)))
    ref_pct = _histogram_percentages(reference, edges)
    cand_pct = _histogram_percentages(candidate, edges)
    epsilon = 1e-6
    ref_safe = np.clip(ref_pct, epsilon, None)
    cand_safe = np.clip(cand_pct, epsilon, None)
    ref_safe = ref_safe / np.sum(ref_safe)
    cand_safe = cand_safe / np.sum(cand_safe)
    psi = float(np.sum((cand_safe - ref_safe) * np.log(cand_safe / ref_safe)))
    kl_divergence = float(np.sum(cand_safe * np.log(cand_safe / ref_safe)))
    return {
        "ks_statistic": _ks_statistic(reference, candidate),
        "psi": psi,
        "kl_divergence": kl_divergence,
    }


def _histogram_percentages(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = np.sum(counts)
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def _ks_statistic(reference: np.ndarray, candidate: np.ndarray) -> float:
    points = np.sort(np.unique(np.concatenate([reference, candidate])))
    ref_sorted = np.sort(reference)
    cand_sorted = np.sort(candidate)
    ref_cdf = np.searchsorted(ref_sorted, points, side="right") / len(ref_sorted)
    cand_cdf = np.searchsorted(cand_sorted, points, side="right") / len(cand_sorted)
    return float(np.max(np.abs(ref_cdf - cand_cdf)))


def _duplicate_label_count(labels: list[Label]) -> int:
    seen: set[tuple[str, str, int, int]] = set()
    duplicates = 0
    for label in labels:
        key = (label.source, label.instrument_id, label.decision_ts, label.horizon_ms)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _horizon_order_failures(labels: list[Label]) -> int:
    by_decision: dict[tuple[str, str, int], list[Label]] = defaultdict(list)
    for label in labels:
        by_decision[(label.source, label.instrument_id, label.decision_ts)].append(label)
    failures = 0
    for group in by_decision.values():
        sorted_group = sorted(group, key=lambda label: label.horizon_ms)
        target_times = [label.label_ts for label in sorted_group]
        if target_times != sorted(target_times):
            failures += 1
    return failures


def _feature_label_pairs(
    features: list[FeatureVector],
    labels: list[Label],
    feature_column: str,
    *,
    horizon_ms: int | None,
) -> list[tuple[float, float]]:
    label_by_key: dict[tuple[str, str, int], Label] = {}
    if horizon_ms is None:
        by_decision: dict[tuple[str, str, int], list[Label]] = defaultdict(list)
        for label in labels:
            by_decision[(label.source, label.instrument_id, label.decision_ts)].append(label)
        for key, group in by_decision.items():
            label_by_key[key] = sorted(group, key=lambda item: item.horizon_ms)[0]
    else:
        for label in labels:
            if label.horizon_ms == horizon_ms:
                label_by_key[(label.source, label.instrument_id, label.decision_ts)] = label

    pairs: list[tuple[float, float]] = []
    for feature in sorted(features, key=lambda item: (item.source, item.instrument_id, item.decision_ts)):
        value = feature.features.get(feature_column)
        if value is None:
            continue
        label = label_by_key.get((feature.source, feature.instrument_id, feature.decision_ts))
        if label is None:
            continue
        pairs.append((float(value), label.net_return))
    return pairs
