"""Champion/challenger shadow comparison framework (issue #45)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import duckdb

from bigan.canonical.query import open_warehouse
from bigan.modeling.calibration import ProbabilityCalibrator, load_probability_calibrator
from bigan.modeling.logistic import load_logistic_baseline
from bigan.modeling.xgboost_v1 import load_xgboost_v1_model


class ProbabilityModel(Protocol):
    """Minimal scoring protocol for shadow models."""

    def predict_proba(self, row: Mapping[str, Any]) -> float:
        """Return P(up in 15m)."""


@dataclass(frozen=True, slots=True)
class ShadowPredictionPair:
    """One champion/challenger side-by-side prediction row."""

    ts: int | None
    source_symbol: str | None
    champion_model_version: str
    challenger_model_version: str
    champion_prob_up_15m: float | None
    challenger_prob_up_15m: float | None
    probability_delta: float | None
    champion_latency_ms: float | None
    challenger_latency_ms: float | None
    challenger_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShadowComparisonReport:
    """Aggregate shadow-mode comparison report."""

    champion_model_version: str
    challenger_model_version: str
    sample_count: int
    scored_count: int
    challenger_error_count: int
    mean_abs_probability_delta: float | None
    max_abs_probability_delta: float | None
    kl_divergence: float | None
    wasserstein_distance: float | None
    avg_champion_latency_ms: float | None
    avg_challenger_latency_ms: float | None
    rows: tuple[ShadowPredictionPair, ...]
    window_start_ts: int | None = None
    window_end_ts: int | None = None
    generated_at_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "rows": [row.to_dict() for row in self.rows],
        }


def run_shadow_comparison(
    *,
    champion_model: ProbabilityModel,
    challenger_model: ProbabilityModel,
    feature_rows: Sequence[Mapping[str, Any]],
    champion_model_version: str,
    challenger_model_version: str,
    bins: int = 10,
    window_start_ts: int | None = None,
    window_end_ts: int | None = None,
) -> ShadowComparisonReport:
    """Score challenger in shadow mode without changing champion output."""

    pairs: list[ShadowPredictionPair] = []
    champion_probs: list[float] = []
    matched_champion_probs: list[float] = []
    challenger_probs: list[float] = []
    champion_latencies: list[float] = []
    challenger_latencies: list[float] = []
    challenger_error_count = 0

    for row in feature_rows:
        champion_start = time.perf_counter()
        champion_prob = _validate_probability(champion_model.predict_proba(row), "champion")
        champion_latency = (time.perf_counter() - champion_start) * 1000.0
        champion_probs.append(champion_prob)
        champion_latencies.append(champion_latency)

        challenger_prob: float | None = None
        challenger_latency: float | None = None
        challenger_error: str | None = None
        challenger_start = time.perf_counter()
        try:
            challenger_prob = _validate_probability(
                challenger_model.predict_proba(row),
                "challenger",
            )
            challenger_probs.append(challenger_prob)
            matched_champion_probs.append(champion_prob)
        except Exception as exc:  # noqa: BLE001 - report and continue; shadow must not affect champion.
            challenger_error = f"{type(exc).__name__}: {exc}"
            challenger_error_count += 1
        finally:
            challenger_latency = (time.perf_counter() - challenger_start) * 1000.0
            challenger_latencies.append(challenger_latency)

        pairs.append(
            ShadowPredictionPair(
                ts=_optional_int(row.get("feature_ts") or row.get("ts")),
                source_symbol=_optional_str(row.get("source_symbol")),
                champion_model_version=champion_model_version,
                challenger_model_version=challenger_model_version,
                champion_prob_up_15m=champion_prob,
                challenger_prob_up_15m=challenger_prob,
                probability_delta=(
                    None if challenger_prob is None else challenger_prob - champion_prob
                ),
                champion_latency_ms=champion_latency,
                challenger_latency_ms=challenger_latency,
                challenger_error=challenger_error,
            )
        )

    deltas = [
        abs(float(pair.probability_delta))
        for pair in pairs
        if pair.probability_delta is not None
    ]
    return ShadowComparisonReport(
        champion_model_version=champion_model_version,
        challenger_model_version=challenger_model_version,
        sample_count=len(feature_rows),
        scored_count=len(deltas),
        challenger_error_count=challenger_error_count,
        mean_abs_probability_delta=_mean(deltas),
        max_abs_probability_delta=None if not deltas else max(deltas),
        kl_divergence=(
            None
            if not challenger_probs
            else distribution_kl_divergence(
                matched_champion_probs,
                challenger_probs,
                bins=bins,
            )
        ),
        wasserstein_distance=(
            None
            if not challenger_probs
            else distribution_wasserstein_distance(
                matched_champion_probs,
                challenger_probs,
            )
        ),
        avg_champion_latency_ms=_mean(champion_latencies),
        avg_challenger_latency_ms=_mean(challenger_latencies),
        rows=tuple(pairs),
        window_start_ts=window_start_ts,
        window_end_ts=window_end_ts,
        generated_at_ms=int(time.time() * 1_000),
    )


def run_shadow_warehouse_comparison(
    *,
    warehouse_dir: Path | str,
    champion_model_path: Path | str,
    challenger_model_path: Path | str,
    output_path: Path | str,
    champion_calibration_path: Path | str | None = None,
    challenger_calibration_path: Path | str | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    limit: int | None = None,
    bins: int = 10,
) -> ShadowComparisonReport:
    """Load champion/challenger artifacts, score warehouse features, and save report."""

    champion_model, champion_version = load_shadow_probability_model(
        champion_model_path,
        calibration_path=champion_calibration_path,
    )
    challenger_model, challenger_version = load_shadow_probability_model(
        challenger_model_path,
        calibration_path=challenger_calibration_path,
    )
    feature_rows = read_shadow_feature_rows(
        warehouse_dir,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=limit,
    )
    report = run_shadow_comparison(
        champion_model=champion_model,
        challenger_model=challenger_model,
        feature_rows=feature_rows,
        champion_model_version=champion_version,
        challenger_model_version=challenger_version,
        bins=bins,
        window_start_ts=since_ms,
        window_end_ts=until_ms,
    )
    save_shadow_report(report, output_path)
    return report


def read_shadow_feature_rows(
    warehouse_dir: Path | str,
    *,
    since_ms: int | None = None,
    until_ms: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read feature rows eligible for shadow comparison from the canonical warehouse."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    filters = ["quality_filter_pass", "not data_gap_flag"]
    params: list[int] = []
    if since_ms is not None:
        filters.append("feature_ts >= ?")
        params.append(int(since_ms))
    if until_ms is not None:
        filters.append("feature_ts < ?")
        params.append(int(until_ms))
    limit_clause = "" if limit is None else f"limit {int(limit)}"
    query = f"""
        select *
        from features_15m_v1
        where {' and '.join(filters)}
        order by feature_ts, source, source_symbol
        {limit_clause}
    """
    with open_warehouse(warehouse_dir) as conn:
        try:
            return conn.execute(query, params).to_arrow_table().to_pylist()
        except (duckdb.CatalogException, duckdb.IOException):
            return []


def load_shadow_probability_model(
    model_path: Path | str,
    *,
    calibration_path: Path | str | None = None,
) -> tuple[ProbabilityModel, str]:
    """Load a supported probability model plus optional calibration wrapper."""

    model = _load_supported_probability_model(model_path)
    calibrator = None if calibration_path is None else load_probability_calibrator(calibration_path)
    wrapped = (
        model
        if calibrator is None
        else CalibratedProbabilityModel(model=model, calibrator=calibrator)
    )
    return wrapped, str(getattr(model, "model_version", Path(model_path).stem))


@dataclass(frozen=True, slots=True)
class CalibratedProbabilityModel:
    """Apply a saved probability calibrator to a model's raw output."""

    model: ProbabilityModel
    calibrator: ProbabilityCalibrator

    def predict_proba(self, row: Mapping[str, Any]) -> float:
        return self.calibrator.transform(self.model.predict_proba(dict(row)))


def _load_supported_probability_model(model_path: Path | str) -> ProbabilityModel:
    try:
        return load_xgboost_v1_model(model_path)
    except Exception:  # noqa: BLE001 - fall through to the JSON logistic format.
        return load_logistic_baseline(model_path)


def save_shadow_report(report: ShadowComparisonReport, output_path: Path | str) -> None:
    """Write a shadow comparison report as JSON."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def distribution_kl_divergence(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-9,
) -> float:
    """Histogram KL divergence from reference to candidate probabilities."""

    _validate_distribution_inputs(reference, candidate, bins)
    ref_hist = _probability_histogram(reference, bins=bins)
    cand_hist = _probability_histogram(candidate, bins=bins)
    return sum(
        (ref + epsilon) * math.log((ref + epsilon) / (cand + epsilon))
        for ref, cand in zip(ref_hist, cand_hist, strict=True)
    )


def distribution_wasserstein_distance(
    reference: Sequence[float],
    candidate: Sequence[float],
) -> float:
    """One-dimensional Wasserstein distance for probability samples."""

    if len(reference) != len(candidate):
        raise ValueError("distributions must have equal length")
    if not reference:
        raise ValueError("distributions must be non-empty")
    ref_sorted = sorted(float(value) for value in reference)
    cand_sorted = sorted(float(value) for value in candidate)
    return sum(abs(ref - cand) for ref, cand in zip(ref_sorted, cand_sorted, strict=True)) / len(
        ref_sorted
    )


def _probability_histogram(values: Sequence[float], *, bins: int) -> list[float]:
    counts = [0] * bins
    for value in values:
        prob = _validate_probability(value, "distribution")
        index = min(int(prob * bins), bins - 1)
        counts[index] += 1
    total = sum(counts)
    return [count / total for count in counts]


def _validate_distribution_inputs(
    reference: Sequence[float],
    candidate: Sequence[float],
    bins: int,
) -> None:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not reference or not candidate:
        raise ValueError("distributions must be non-empty")


def _validate_probability(value: float, role: str) -> float:
    prob = float(value)
    if prob < 0.0 or prob > 1.0 or not math.isfinite(prob):
        raise ValueError(f"{role} probability must be in [0, 1]")
    return prob


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
