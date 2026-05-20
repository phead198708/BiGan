"""Champion/challenger shadow comparison framework (issue #45)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


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
    )


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
