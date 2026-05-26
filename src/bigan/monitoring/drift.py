"""Feature drift detection metrics and persistence (issue #47)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from .incidents import DataQualityIncident, record_data_quality_incident

CHAMPION_BASELINE_DISTRIBUTIONS: dict[str, dict[str, Any]] = {
    "xgboost-v3": {
        "source": "data/model-train-backtest-rerun-20260520T134420Z/artifacts/training/val.parquet",
        "split": "val",
        "probability_distribution": {
            "count": 2602,
            "mean": 0.5784535391099995,
            "std": 0.4047956508396586,
            "min": 0.00028753106750433946,
            "p05": 0.0007738887218141967,
            "p25": 0.02268596099695062,
            "p50": 0.6729405815045788,
            "p75": 0.9831505277976981,
            "p95": 0.9992003106544867,
            "max": 0.9998801526963161,
        },
        "edge_distribution": {
            "count": 2602,
            "mean": 0.06849965748048373,
            "std": 0.4047500175757351,
            "trigger_rate_edge_ge_0_30": 0.3816295157571099,
        },
    },
    "xgboost-v4": {
        "source": "docs/models/xgboost-v4-offline-val-reference.json",
        "split": "val",
        "probability_distribution": {
            "count": 2602,
            "mean": 0.5633657718039641,
            "std": 0.43144887503556245,
            "min": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.7925170068027211,
            "p75": 0.9921875,
            "p95": 1.0,
            "max": 1.0,
        },
        "edge_distribution": {
            "count": 2602,
            "mean": 0.053411890174448404,
            "std": 0.4314065528672354,
            "trigger_rate_edge_ge_0_30": 0.39315910837817064,
        },
    }
}

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


def champion_baseline_distribution(model_version: str) -> dict[str, Any]:
    """Return the offline validation baseline for live champion drift checks."""

    baseline = CHAMPION_BASELINE_DISTRIBUTIONS.get(model_version)
    if baseline is None:
        raise ValueError(f"no champion drift baseline registered for {model_version!r}")
    return dict(baseline["probability_distribution"])


def build_champion_drift_baseline(
    offline_reference_path: str,
    output_path: str | None = None,
    *,
    thresholds: ChampionDriftThresholds | None = None,
) -> dict[str, Any]:
    """Build a champion drift baseline artifact from offline validation evidence."""

    source = str(offline_reference_path)
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("offline reference must be a JSON object")
    for key in ("model_version", "dataset_dir", "dataset_version", "split"):
        if not payload.get(key):
            raise ValueError(f"offline reference missing {key}")
    if payload.get("split") != "val":
        raise ValueError("offline reference split must be val")
    probability_distribution = payload.get("probability_distribution")
    if not isinstance(probability_distribution, dict):
        raise ValueError("offline reference missing probability_distribution")
    for key in ("count", "mean", "std"):
        if probability_distribution.get(key) is None:
            raise ValueError(f"offline reference probability_distribution missing {key}")
    edge_distribution = payload.get("edge_distribution")
    active_thresholds = thresholds or ChampionDriftThresholds()
    baseline = {
        "generated_at_ms": _now_ms(),
        "source_offline_reference_path": source,
        "model_version": payload.get("model_version"),
        "model_path": payload.get("model_path"),
        "dataset_dir": payload.get("dataset_dir"),
        "dataset_version": payload.get("dataset_version"),
        "split": payload.get("split"),
        "probability_distribution": probability_distribution,
        "edge_distribution": edge_distribution if isinstance(edge_distribution, dict) else None,
        "edge_trigger_rate_at_0_30": payload.get("edge_trigger_rate_at_0_30"),
        "thresholds": active_thresholds.to_dict(),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    return baseline


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


@dataclass(frozen=True, slots=True)
class ChampionDriftThresholds:
    """Live champion prediction and label-shift alert thresholds."""

    probability_mean_shift_abs: float = 0.05
    probability_std_relative_change: float = 0.20
    edge_threshold: float = 0.30
    edge_zero_window_ms: int = 2 * 60 * 60 * 1000
    label_positive_rate_min: float = 0.50
    label_consecutive_samples: int = 50

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


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


def evaluate_live_champion_drift(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    reference_distribution: Mapping[str, Any],
    now_ms: int | None = None,
    windows_ms: Sequence[int] = (60 * 60 * 1000, 6 * 60 * 60 * 1000),
    thresholds: ChampionDriftThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate xgboost-v3 live prediction drift and settled label hit-rate drift."""

    active_thresholds = thresholds or ChampionDriftThresholds()
    ts = _now_ms() if now_ms is None else int(now_ms)
    reference_mean = _optional_float(reference_distribution.get("mean"))
    reference_std = _optional_float(reference_distribution.get("std"))
    window_reports = []
    alerts: list[dict[str, Any]] = []

    for window_ms in windows_ms:
        events = _read_prediction_events(
            conn,
            model_version=model_version,
            start_ms=ts - int(window_ms),
            end_ms=ts,
        )
        probabilities = [_as_safe_float(row["prob_up_15m"]) for row in events]
        probabilities = [value for value in probabilities if value is not None]
        edges = _event_edges(events)
        summary = _live_distribution_summary(probabilities)
        edge_trigger_rate = _edge_trigger_rate(edges, active_thresholds.edge_threshold)
        window_report = {
            "window_ms": int(window_ms),
            "sample_count": len(events),
            "probability_distribution": summary,
            "edge_trigger_rate": edge_trigger_rate,
        }
        if reference_mean is not None and summary["mean"] is not None:
            mean_shift = abs(float(summary["mean"]) - reference_mean)
            window_report["mean_shift_abs"] = mean_shift
            if mean_shift > active_thresholds.probability_mean_shift_abs:
                alerts.append(
                    {
                        "alert_type": "probability_mean_shift",
                        "window_ms": int(window_ms),
                        "severity": "warning",
                        "detail": (
                            f"mean_shift_abs={mean_shift:.6f} exceeds "
                            f"{active_thresholds.probability_mean_shift_abs:.6f}"
                        ),
                    }
                )
        if reference_std is not None and summary["std"] is not None:
            std_change = _relative_change(float(summary["std"]), reference_std)
            window_report["std_relative_change"] = std_change
            if std_change > active_thresholds.probability_std_relative_change:
                alerts.append(
                    {
                        "alert_type": "probability_std_shift",
                        "window_ms": int(window_ms),
                        "severity": "warning",
                        "detail": (
                            f"std_relative_change={std_change:.6f} exceeds "
                            f"{active_thresholds.probability_std_relative_change:.6f}"
                        ),
                    }
                )
        window_reports.append(window_report)

    edge_zero_report = _edge_zero_report(
        conn,
        model_version=model_version,
        now_ms=ts,
        thresholds=active_thresholds,
    )
    if edge_zero_report["trigger_rate"] == 0.0 and edge_zero_report["sample_count"] > 0:
        alerts.append(
            {
                "alert_type": "edge_trigger_zero",
                "window_ms": active_thresholds.edge_zero_window_ms,
                "severity": "warning",
                "detail": "edge trigger rate is zero for the configured zero-edge window",
            }
        )

    label_report = evaluate_label_hit_rate_drift(
        conn,
        model_version=model_version,
        thresholds=active_thresholds,
    )
    if label_report["alert"]:
        alerts.append(
            {
                "alert_type": "label_hit_rate_low",
                "window_ms": None,
                "severity": "critical",
                "detail": (
                    f"positive_rate={label_report['positive_rate']:.6f} below "
                    f"{active_thresholds.label_positive_rate_min:.6f} for "
                    f"{label_report['sample_count']} settled samples"
                ),
            }
        )

    return {
        "model_version": model_version,
        "generated_at_ms": ts,
        "reference_distribution": {
            "mean": reference_mean,
            "std": reference_std,
        },
        "thresholds": active_thresholds.to_dict(),
        "windows": window_reports,
        "edge_zero_window": edge_zero_report,
        "label_hit_rate": label_report,
        "alerts": alerts,
        "passed": not alerts,
    }


def run_live_champion_monitoring(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    now_ms: int | None = None,
    windows_ms: Sequence[int] = (60 * 60 * 1000, 6 * 60 * 60 * 1000),
    thresholds: ChampionDriftThresholds | None = None,
    record_incidents: bool = True,
) -> dict[str, Any]:
    """Evaluate and optionally persist live champion monitoring alerts."""

    report = evaluate_live_champion_drift(
        conn,
        model_version=model_version,
        reference_distribution=champion_baseline_distribution(model_version),
        now_ms=now_ms,
        windows_ms=windows_ms,
        thresholds=thresholds,
    )
    incident_ids = (
        record_champion_drift_incidents(conn, report) if record_incidents else ()
    )
    return {
        **report,
        "incident_ids": list(incident_ids),
    }


def evaluate_label_hit_rate_drift(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    thresholds: ChampionDriftThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate settled label positive-rate drift for the latest outcomes."""

    active_thresholds = thresholds or ChampionDriftThresholds()
    rows = _read_recent_outcomes(
        conn,
        model_version=model_version,
        limit=active_thresholds.label_consecutive_samples,
    )
    labels = [bool(row["realized_label"]) for row in rows]
    positive_rate = None if not labels else sum(1 for label in labels if label) / len(labels)
    alert = (
        len(labels) >= active_thresholds.label_consecutive_samples
        and positive_rate is not None
        and positive_rate < active_thresholds.label_positive_rate_min
    )
    return {
        "sample_count": len(labels),
        "positive_rate": positive_rate,
        "alert": alert,
        "threshold": active_thresholds.label_positive_rate_min,
        "consecutive_samples": active_thresholds.label_consecutive_samples,
    }


def record_champion_drift_incidents(
    conn: duckdb.DuckDBPyConnection,
    report: Mapping[str, Any],
    *,
    source: str = "model_monitoring",
    replace: bool = True,
) -> tuple[str, ...]:
    """Write live champion drift alerts into the incident catalog."""

    alert_ids: list[str] = []
    model_version = str(report.get("model_version", "unknown_model"))
    started_at = int(report.get("generated_at_ms") or _now_ms())
    for alert in report.get("alerts", []):
        if not isinstance(alert, Mapping):
            continue
        alert_type = str(alert.get("alert_type", "prediction_drift"))
        incident_type = "label_shift" if alert_type == "label_hit_rate_low" else "prediction_drift"
        window_ms = alert.get("window_ms")
        suffix = "settled" if window_ms is None else f"{int(window_ms)}ms"
        incident_id = f"{model_version}:{alert_type}:{suffix}:{started_at}"
        record_data_quality_incident(
            conn,
            DataQualityIncident(
                incident_id=incident_id,
                source=source,
                incident_type=incident_type,
                severity=str(alert.get("severity", "warning")),
                started_at=started_at,
                affected_symbol=model_version,
                details_json=json.dumps(alert, sort_keys=True),
                alert_id=alert_type,
                owner="ml-oncall",
            ),
            replace=replace,
        )
        alert_ids.append(incident_id)
    return tuple(alert_ids)


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


def _read_prediction_events(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT event_id, ts, prob_up_15m, serving_latency_ms, feature_snapshot_json
            FROM prediction_events
            WHERE model_version = ?
              AND ts >= ?
              AND ts <= ?
            ORDER BY ts, event_id
            """,
            [model_version, start_ms, end_ms],
        ).fetchall()
    except duckdb.CatalogException:
        return []
    columns = [column[0] for column in conn.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _read_recent_outcomes(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT o.realized_label, o.target_ts, o.outcome_ts
            FROM prediction_outcomes o
            JOIN prediction_events e USING (event_id)
            WHERE e.model_version = ?
            ORDER BY o.target_ts DESC, o.outcome_ts DESC, o.event_id DESC
            LIMIT ?
            """,
            [model_version, int(limit)],
        ).fetchall()
    except duckdb.CatalogException:
        return []
    columns = [column[0] for column in conn.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _edge_zero_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    model_version: str,
    now_ms: int,
    thresholds: ChampionDriftThresholds,
) -> dict[str, Any]:
    events = _read_prediction_events(
        conn,
        model_version=model_version,
        start_ms=now_ms - thresholds.edge_zero_window_ms,
        end_ms=now_ms,
    )
    edges = _event_edges(events)
    return {
        "window_ms": thresholds.edge_zero_window_ms,
        "sample_count": len(events),
        "edge_count": len(edges),
        "trigger_rate": _edge_trigger_rate(edges, thresholds.edge_threshold),
    }


def _event_edges(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    edges = []
    for row in rows:
        probability = _as_safe_float(row.get("prob_up_15m"))
        snapshot = _snapshot_payload(row.get("feature_snapshot_json"))
        market = _market_implied_prob_from_snapshot_payload(snapshot)
        outcome_side = _outcome_side_from_snapshot(snapshot)
        if probability is not None and market is not None:
            edges.append(_token_probability(probability, outcome_side) - market)
    return edges


def _market_implied_prob_from_snapshot(snapshot: Any) -> float | None:
    return _market_implied_prob_from_snapshot_payload(_snapshot_payload(snapshot))


def _snapshot_payload(snapshot: Any) -> Mapping[str, Any]:
    if snapshot is None:
        return {}
    try:
        payload = json.loads(str(snapshot))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _market_implied_prob_from_snapshot_payload(payload: Mapping[str, Any]) -> float | None:
    features = payload.get("features")
    feature_payload = features if isinstance(features, Mapping) else {}
    for source in (payload, feature_payload):
        for key in ("market_implied_prob", "best_ask", "entry_ask_price"):
            value = _as_safe_float(source.get(key))
            if value is not None:
                return value
    return None


def _outcome_side_from_snapshot(payload: Mapping[str, Any]) -> str:
    canonical = str(payload.get("canonical_symbol") or payload.get("symbol") or "")
    text = canonical.strip().upper()
    if text.endswith(":DOWN") or text.endswith("-DOWN-15M"):
        return "DOWN"
    if text.endswith(":UP") or text.endswith("-UP-15M"):
        return "UP"
    return "UP"


def _token_probability(prob_up_15m: float, outcome_side: str) -> float:
    return 1.0 - prob_up_15m if outcome_side == "DOWN" else prob_up_15m


def _live_distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    cleaned = sorted(value for value in values if math.isfinite(value))
    if not cleaned:
        return {"count": 0, "mean": None, "std": None}
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / len(cleaned)
    return {"count": len(cleaned), "mean": mean, "std": math.sqrt(variance)}


def _edge_trigger_rate(edges: Sequence[float], threshold: float) -> float | None:
    if not edges:
        return None
    return sum(1 for edge in edges if edge >= threshold) / len(edges)


def _optional_float(value: Any) -> float | None:
    return _as_safe_float(value)


def _as_safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _relative_change(current: float, reference: float) -> float:
    denominator = abs(reference)
    if denominator <= 1e-12:
        return 0.0 if abs(current) <= 1e-12 else math.inf
    return abs(current - reference) / denominator


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
