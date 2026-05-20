"""Feature drift detection metrics and persistence (issue #47)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import duckdb

DRIFT_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS drift_metrics (
    metric_id VARCHAR PRIMARY KEY,
    "date" DATE NOT NULL,
    model_version VARCHAR NOT NULL,
    feature_name VARCHAR NOT NULL,
    psi DOUBLE NOT NULL,
    ks_statistic DOUBLE NOT NULL,
    wasserstein_distance DOUBLE NOT NULL,
    reference_count BIGINT NOT NULL,
    current_count BIGINT NOT NULL,
    severity VARCHAR NOT NULL CHECK (severity IN ('ok', 'warning', 'critical')),
    thresholds_json VARCHAR NOT NULL,
    created_at BIGINT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class DriftMetricRow:
    """One feature drift measurement."""

    metric_id: str
    date: str
    model_version: str
    feature_name: str
    psi: float
    ks_statistic: float
    wasserstein_distance: float
    reference_count: int
    current_count: int
    severity: str
    thresholds_json: str
    created_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["created_at"] = self.created_at or _now_ms()
        return row

    def to_dict(self) -> dict[str, Any]:
        return self.to_row()


def compute_feature_drift(
    reference_values: Sequence[float],
    current_values: Sequence[float],
    *,
    feature_name: str,
    model_version: str,
    date: str,
    bins: int = 10,
    psi_warning: float = 0.10,
    psi_critical: float = 0.25,
    ks_critical: float = 0.20,
    wasserstein_critical: float = 0.15,
) -> DriftMetricRow:
    """Compute PSI, KS, and Wasserstein drift metrics for one feature."""

    reference = _clean_values(reference_values, "reference_values")
    current = _clean_values(current_values, "current_values")
    if bins <= 0:
        raise ValueError("bins must be positive")
    thresholds = {
        "psi_warning": psi_warning,
        "psi_critical": psi_critical,
        "ks_critical": ks_critical,
        "wasserstein_critical": wasserstein_critical,
    }
    psi = population_stability_index(reference, current, bins=bins)
    ks_statistic = kolmogorov_smirnov_statistic(reference, current)
    wasserstein = wasserstein_distance(reference, current)
    severity = _severity(
        psi=psi,
        ks_statistic=ks_statistic,
        wasserstein=wasserstein,
        psi_warning=psi_warning,
        psi_critical=psi_critical,
        ks_critical=ks_critical,
        wasserstein_critical=wasserstein_critical,
    )
    return DriftMetricRow(
        metric_id=f"{date}:{model_version}:{feature_name}",
        date=date,
        model_version=model_version,
        feature_name=feature_name,
        psi=psi,
        ks_statistic=ks_statistic,
        wasserstein_distance=wasserstein,
        reference_count=len(reference),
        current_count=len(current),
        severity=severity,
        thresholds_json=json.dumps(thresholds, sort_keys=True),
    )


def write_drift_metrics(
    conn: duckdb.DuckDBPyConnection,
    rows: Sequence[DriftMetricRow],
) -> None:
    """Upsert drift metric rows."""

    conn.execute(DRIFT_METRICS_DDL)
    for metric in rows:
        row = metric.to_row()
        conn.execute("DELETE FROM drift_metrics WHERE metric_id = ?", [metric.metric_id])
        conn.execute(
            """
            INSERT INTO drift_metrics (
                metric_id, "date", model_version, feature_name, psi, ks_statistic,
                wasserstein_distance, reference_count, current_count, severity,
                thresholds_json, created_at
            ) VALUES (?, CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["metric_id"],
                row["date"],
                row["model_version"],
                row["feature_name"],
                row["psi"],
                row["ks_statistic"],
                row["wasserstein_distance"],
                row["reference_count"],
                row["current_count"],
                row["severity"],
                row["thresholds_json"],
                row["created_at"],
            ],
        )


def drift_metrics_json(rows: Sequence[DriftMetricRow]) -> str:
    """Serialize drift rows into the daily monitoring JSON shape."""

    return json.dumps(
        {
            row.feature_name: {
                "psi": row.psi,
                "ks_statistic": row.ks_statistic,
                "wasserstein_distance": row.wasserstein_distance,
                "severity": row.severity,
            }
            for row in rows
        },
        sort_keys=True,
    )


def drift_report_from_rows(
    reference_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    model_version: str,
    date: str,
) -> tuple[DriftMetricRow, ...]:
    """Compute drift rows from mapping-style feature snapshots."""

    rows = []
    for feature_name in feature_names:
        rows.append(
            compute_feature_drift(
                [_as_float(row[feature_name]) for row in reference_rows if feature_name in row],
                [_as_float(row[feature_name]) for row in current_rows if feature_name in row],
                feature_name=feature_name,
                model_version=model_version,
                date=date,
            )
        )
    return tuple(rows)


def population_stability_index(
    reference_values: Sequence[float],
    current_values: Sequence[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI over equal-width bins spanning both samples."""

    reference = _clean_values(reference_values, "reference_values")
    current = _clean_values(current_values, "current_values")
    edges = _bin_edges([*reference, *current], bins=bins)
    ref_dist = _histogram_distribution(reference, edges, epsilon=epsilon)
    cur_dist = _histogram_distribution(current, edges, epsilon=epsilon)
    return sum(
        (cur - ref) * math.log(cur / ref)
        for ref, cur in zip(ref_dist, cur_dist, strict=True)
    )


def kolmogorov_smirnov_statistic(
    reference_values: Sequence[float],
    current_values: Sequence[float],
) -> float:
    """Two-sample KS statistic."""

    reference = sorted(_clean_values(reference_values, "reference_values"))
    current = sorted(_clean_values(current_values, "current_values"))
    points = sorted(set(reference + current))
    ref_idx = 0
    cur_idx = 0
    max_delta = 0.0
    for point in points:
        while ref_idx < len(reference) and reference[ref_idx] <= point:
            ref_idx += 1
        while cur_idx < len(current) and current[cur_idx] <= point:
            cur_idx += 1
        max_delta = max(max_delta, abs(ref_idx / len(reference) - cur_idx / len(current)))
    return max_delta


def wasserstein_distance(
    reference_values: Sequence[float],
    current_values: Sequence[float],
    *,
    quantile_points: int = 100,
) -> float:
    """Approximate 1D Wasserstein distance using quantile grids."""

    reference = sorted(_clean_values(reference_values, "reference_values"))
    current = sorted(_clean_values(current_values, "current_values"))
    steps = max(1, quantile_points)
    total = 0.0
    for idx in range(steps):
        q = 0.0 if steps == 1 else idx / (steps - 1)
        total += abs(_quantile(reference, q) - _quantile(current, q))
    return total / steps


def _severity(
    *,
    psi: float,
    ks_statistic: float,
    wasserstein: float,
    psi_warning: float,
    psi_critical: float,
    ks_critical: float,
    wasserstein_critical: float,
) -> str:
    if psi >= psi_critical or ks_statistic >= ks_critical or wasserstein >= wasserstein_critical:
        return "critical"
    if psi >= psi_warning:
        return "warning"
    return "ok"


def _histogram_distribution(
    values: Sequence[float],
    edges: Sequence[float],
    *,
    epsilon: float,
) -> list[float]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        for idx in range(len(edges) - 1):
            lower = edges[idx]
            upper = edges[idx + 1]
            if value >= lower and (value < upper or idx == len(edges) - 2):
                counts[idx] += 1
                break
    total = sum(counts)
    adjusted = [count / total + epsilon for count in counts]
    norm = sum(adjusted)
    return [value / norm for value in adjusted]


def _bin_edges(values: Sequence[float], *, bins: int) -> list[float]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return [min_value - 0.5, max_value + 0.5]
    width = (max_value - min_value) / bins
    return [min_value + idx * width for idx in range(bins)] + [max_value]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _clean_values(values: Sequence[float], name: str) -> list[float]:
    cleaned = [_as_float(value) for value in values]
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    if any(not math.isfinite(value) for value in cleaned):
        raise ValueError(f"{name} must contain only finite values")
    return cleaned


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid drift features")
    return float(value)


def _now_ms() -> int:
    return int(time.time() * 1000)
