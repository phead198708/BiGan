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
from bigan.modeling.calibration import (
    FamilyAwareProbabilityCalibrator,
    ProbabilityCalibrator,
    load_probability_calibrator,
    transform_probability,
)
from bigan.modeling.families import market_family_from_symbol
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
    market_implied_prob: float | None = None
    settlement_price: float | None = None
    realized_return: float | None = None

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


@dataclass(frozen=True, slots=True)
class ShadowEvaluationResult:
    """Promotion-focused shadow evaluation summary."""

    champion_model_version: str
    challenger_model_version: str
    sample_count: int
    scored_count: int
    edge_threshold: float
    overall_passed: bool
    checks: dict[str, dict[str, Any]]
    champion_probability_distribution: dict[str, float | int | None]
    challenger_probability_distribution: dict[str, float | int | None]
    champion_edge_distribution: dict[str, float | int | None] | None
    challenger_edge_distribution: dict[str, float | int | None] | None
    champion_edge_trigger_rate: float | None
    challenger_edge_trigger_rate: float | None
    schema_error_rate: float
    scoring_error_rate: float
    latency_ms: dict[str, dict[str, float | int | None]]
    simulated_pnl: dict[str, float | int | None] | None
    offline_reference_path: str | None
    offline_reference: dict[str, Any] | None
    window_start_ts: int | None
    window_end_ts: int | None
    session_duration_seconds: float | None
    generated_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Shadow Evaluation Report",
            "",
            "## Summary",
            f"- Champion/baseline: `{self.champion_model_version}`",
            f"- Challenger: `{self.challenger_model_version}`",
            f"- Samples: {self.sample_count}",
            f"- Scored samples: {self.scored_count}",
            f"- Edge threshold: {self.edge_threshold:.4f}",
            f"- Window start: {_format_optional_int(self.window_start_ts)}",
            f"- Window end: {_format_optional_int(self.window_end_ts)}",
            f"- Session duration seconds: "
            f"{_format_optional_float(self.session_duration_seconds)}",
            f"- Overall result: {'PASS' if self.overall_passed else 'FAIL'}",
            "",
            "## Pass/Fail Checks",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
        for name, check in self.checks.items():
            status = "PASS" if check.get("passed") else "FAIL"
            lines.append(f"| {name} | {status} | {_escape_table(str(check.get('detail', '')))} |")

        lines.extend(
            [
                "",
                "## Probability Distribution",
                "| Model | Count | Mean | Std | p05 | p25 | p50 | p75 | p95 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                _distribution_row(
                    self.champion_model_version,
                    self.champion_probability_distribution,
                ),
                _distribution_row(
                    self.challenger_model_version,
                    self.challenger_probability_distribution,
                ),
                "",
                "## Edge Distribution",
                "| Model | Count | Mean | Std | p05 | p25 | p50 | p75 | p95 | Trigger Rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                _edge_distribution_row(
                    self.champion_model_version,
                    self.champion_edge_distribution,
                    self.champion_edge_trigger_rate,
                ),
                _edge_distribution_row(
                    self.challenger_model_version,
                    self.challenger_edge_distribution,
                    self.challenger_edge_trigger_rate,
                ),
                "",
                "## Latency",
                "| Model | Count | p50 ms | p95 ms |",
                "|---|---:|---:|---:|",
            ]
        )
        for model_version, latency in self.latency_ms.items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        _escape_table(model_version),
                        str(latency.get("count")),
                        _format_optional_float(latency.get("p50")),
                        _format_optional_float(latency.get("p95")),
                    )
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "## Errors",
                f"- Schema error rate: {_format_optional_float(self.schema_error_rate)}",
                f"- Scoring error rate: {_format_optional_float(self.scoring_error_rate)}",
                "",
                "## Simulated PnL",
            ]
        )
        if self.simulated_pnl is None:
            lines.append("- Settlement data not available in this shadow window.")
        else:
            lines.extend(
                [
                    f"- Baseline/champion trades: {self.simulated_pnl['champion_trade_count']}",
                    f"- Baseline/champion PnL: "
                    f"{_format_optional_float(self.simulated_pnl['champion_net_pnl'])}",
                    f"- Challenger trades: {self.simulated_pnl['challenger_trade_count']}",
                    f"- Challenger PnL: "
                    f"{_format_optional_float(self.simulated_pnl['challenger_net_pnl'])}",
                    f"- Delta vs baseline: "
                    f"{_format_optional_float(self.simulated_pnl['net_pnl_delta'])}",
                ]
            )
        return "\n".join(lines) + "\n"


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
                market_implied_prob=_market_implied_probability(row),
                settlement_price=_optional_float(row.get("settlement_price")),
                realized_return=_optional_float(row.get("realized_return")),
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
    evaluation_output_path: Path | str | None = None,
    evaluation_json_output_path: Path | str | None = None,
    offline_reference_path: Path | str | None = None,
    edge_threshold: float = 0.30,
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
    markdown_path, json_path = shadow_evaluation_output_paths(
        output_path,
        evaluation_output_path=evaluation_output_path,
        evaluation_json_output_path=evaluation_json_output_path,
    )
    shadow_evaluation_report(
        report,
        output_path=markdown_path,
        json_output_path=json_path,
        offline_reference_path=offline_reference_path,
        edge_threshold=edge_threshold,
    )
    return report


def run_per_family_shadow_analysis(
    *,
    warehouse_dir: Path | str,
    champion_model_path: Path | str,
    challenger_model_path: Path | str,
    champion_calibration_path: Path | str | None = None,
    challenger_calibration_path: Path | str | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    limit: int | None = None,
    edge_thresholds: Sequence[float] = (0.02, 0.03, 0.05, 0.08),
) -> dict[str, Any]:
    """Compare champion vs challenger broken down by market family/horizon.

    Unlike :func:`run_shadow_warehouse_comparison`, this keeps the market family
    for every row, so per-family discrimination (Brier/ROC AUC) and per-family
    edge-trigger / simulated-PnL sweeps can be reported. This is the evidence
    used to pick per-family edge thresholds (e.g. trade 15M, skip 5M).
    """

    champion_model, champion_version = load_shadow_probability_model(
        champion_model_path, calibration_path=champion_calibration_path
    )
    challenger_model, challenger_version = load_shadow_probability_model(
        challenger_model_path, calibration_path=challenger_calibration_path
    )
    feature_rows = read_shadow_feature_rows(
        warehouse_dir, since_ms=since_ms, until_ms=until_ms, limit=limit
    )

    scored_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        family = market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))
        try:
            champion_prob = _validate_probability(champion_model.predict_proba(row), "champion")
            challenger_prob = _validate_probability(
                challenger_model.predict_proba(row), "challenger"
            )
        except Exception:  # noqa: BLE001 - skip unscored rows, shadow must not fail hard.
            continue
        scored_by_family.setdefault(family, []).append(
            {
                "champion_prob_up_15m": champion_prob,
                "challenger_prob_up_15m": challenger_prob,
                "market_implied_prob": _market_implied_probability(row),
                "realized_return": _optional_float(row.get("realized_return")),
                "settlement_price": _optional_float(row.get("settlement_price")),
                "label_profit_up_15m": _optional_label(row.get("label_profit_up_15m")),
            }
        )

    families = {
        family: _per_family_summary(rows, edge_thresholds=edge_thresholds)
        for family, rows in sorted(scored_by_family.items())
    }
    all_rows = [row for rows in scored_by_family.values() for row in rows]
    return {
        "champion_model_version": champion_version,
        "challenger_model_version": challenger_version,
        "window_start_ts": since_ms,
        "window_end_ts": until_ms,
        "edge_thresholds": [float(value) for value in edge_thresholds],
        "row_count": len(all_rows),
        "generated_at_ms": int(time.time() * 1000),
        "families": families,
        "all": _per_family_summary(all_rows, edge_thresholds=edge_thresholds),
    }


def _per_family_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    edge_thresholds: Sequence[float],
) -> dict[str, Any]:
    labelled = [row for row in rows if row.get("label_profit_up_15m") is not None]
    labels = [int(row["label_profit_up_15m"]) for row in labelled]
    champion_probs = [float(row["champion_prob_up_15m"]) for row in labelled]
    challenger_probs = [float(row["challenger_prob_up_15m"]) for row in labelled]
    thresholds: dict[str, Any] = {}
    for threshold in edge_thresholds:
        champion_pnl = _side_pnl(
            rows, probability_key="champion_prob_up_15m", edge_threshold=threshold
        )
        challenger_pnl = _side_pnl(
            rows, probability_key="challenger_prob_up_15m", edge_threshold=threshold
        )
        thresholds[f"{threshold:.2f}"] = {
            "champion_trigger_rate": _trigger_rate(
                _edge_values(rows, "champion_prob_up_15m"), threshold
            ),
            "challenger_trigger_rate": _trigger_rate(
                _edge_values(rows, "challenger_prob_up_15m"), threshold
            ),
            "champion_net_pnl": None if champion_pnl is None else champion_pnl[0],
            "champion_trade_count": None if champion_pnl is None else champion_pnl[1],
            "challenger_net_pnl": None if challenger_pnl is None else challenger_pnl[0],
            "challenger_trade_count": None if challenger_pnl is None else challenger_pnl[1],
        }
    return {
        "row_count": len(rows),
        "labelled_count": len(labelled),
        "positive_rate": (sum(labels) / len(labels)) if labels else None,
        "champion_brier": _brier(labels, champion_probs),
        "challenger_brier": _brier(labels, challenger_probs),
        "champion_roc_auc": _roc_auc(labels, champion_probs),
        "challenger_roc_auc": _roc_auc(labels, challenger_probs),
        "edge_thresholds": thresholds,
    }


def _optional_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _brier(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if not labels:
        return None
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(labels)


def _roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positives = sum(1 for y in labels if y == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(order):
        j = i
        while j < len(order) and probabilities[order[j]] == probabilities[order[i]]:
            j += 1
        average_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if labels[order[k]] == 1:
                rank_sum += average_rank
        rank += j - i
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


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
    prefixed_filters = []
    for item in filters:
        if item == "not data_gap_flag":
            prefixed_filters.append("not f.data_gap_flag")
        elif item.startswith(("feature_ts", "quality_")):
            prefixed_filters.append(f"f.{item}")
        else:
            prefixed_filters.append(item)
    predicate = " and ".join(prefixed_filters)
    query_with_labels = f"""
        select
            f.*,
            l.target_ts,
            l.settlement_price,
            l.realized_return,
            l.label_profit_up_15m
        from features_15m_v1 f
        left join labels_15m_v1 l
          on f.source = l.source
         and f.source_symbol = l.source_symbol
         and f.feature_ts = l.feature_ts
        where {predicate}
        order by f.feature_ts, f.source, f.source_symbol
        {limit_clause}
    """
    query_without_labels = f"""
        select *
        from features_15m_v1
        where {' and '.join(filters)}
        order by feature_ts, source, source_symbol
        {limit_clause}
    """
    with open_warehouse(warehouse_dir) as conn:
        try:
            return conn.execute(query_with_labels, params).to_arrow_table().to_pylist()
        except duckdb.CatalogException:
            try:
                return conn.execute(query_without_labels, params).to_arrow_table().to_pylist()
            except (duckdb.CatalogException, duckdb.IOException):
                return []
        except duckdb.IOException:
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
    """Apply a saved probability calibrator to a model's raw output.

    Supports both global and family-aware calibrators. The feature row is
    forwarded so a ``FamilyAwareProbabilityCalibrator`` can select the
    per-family calibrator instead of silently falling back to the global one.
    """

    model: ProbabilityModel
    calibrator: ProbabilityCalibrator | FamilyAwareProbabilityCalibrator

    def predict_proba(self, row: Mapping[str, Any]) -> float:
        feature = dict(row)
        return transform_probability(
            self.calibrator,
            self.model.predict_proba(feature),
            feature=feature,
        )


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


def shadow_evaluation_output_paths(
    shadow_output_path: Path | str,
    *,
    evaluation_output_path: Path | str | None = None,
    evaluation_json_output_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Return default Markdown and JSON paths for a shadow evaluation."""

    base = Path(shadow_output_path)
    markdown_path = (
        Path(evaluation_output_path)
        if evaluation_output_path is not None
        else base.with_suffix(".md")
    )
    json_path = (
        Path(evaluation_json_output_path)
        if evaluation_json_output_path is not None
        else base.with_name(f"{base.stem}-evaluation.json")
    )
    return markdown_path, json_path


def shadow_evaluation_report(
    report: ShadowComparisonReport | Mapping[str, Any] | Path | str,
    *,
    output_path: Path | str | None = None,
    json_output_path: Path | str | None = None,
    offline_reference_path: Path | str | None = None,
    edge_threshold: float = 0.30,
    max_distribution_mean_abs_diff: float = 0.05,
    max_distribution_std_relative_diff: float = 0.20,
    min_edge_trigger_rate: float = 0.0,
    max_edge_trigger_rate: float = 0.50,
    max_schema_error_rate: float = 0.0,
    max_latency_p95_ms: float = 50.0,
) -> str:
    """Generate a promotion-focused Markdown report from shadow output."""

    result = evaluate_shadow_report(
        report,
        offline_reference_path=offline_reference_path,
        edge_threshold=edge_threshold,
        max_distribution_mean_abs_diff=max_distribution_mean_abs_diff,
        max_distribution_std_relative_diff=max_distribution_std_relative_diff,
        min_edge_trigger_rate=min_edge_trigger_rate,
        max_edge_trigger_rate=max_edge_trigger_rate,
        max_schema_error_rate=max_schema_error_rate,
        max_latency_p95_ms=max_latency_p95_ms,
    )
    markdown = result.to_markdown()
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    if json_output_path is not None:
        target = Path(json_output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return markdown


def evaluate_shadow_report(
    report: ShadowComparisonReport | Mapping[str, Any] | Path | str,
    *,
    offline_reference_path: Path | str | None = None,
    edge_threshold: float = 0.30,
    max_distribution_mean_abs_diff: float = 0.05,
    max_distribution_std_relative_diff: float = 0.20,
    min_edge_trigger_rate: float = 0.0,
    max_edge_trigger_rate: float = 0.50,
    max_schema_error_rate: float = 0.0,
    max_latency_p95_ms: float = 50.0,
) -> ShadowEvaluationResult:
    """Compute pass/fail shadow promotion checks from a shadow report."""

    payload = _shadow_payload(report)
    rows = list(payload.get("rows") or [])
    comparable_rows = _settled_rows(rows) or rows
    champion_key = "champion_prob_up_15m"
    challenger_key = "challenger_prob_up_15m"
    champion_probs = _row_floats(comparable_rows, champion_key)
    challenger_probs = _row_floats(comparable_rows, challenger_key)
    champion_edges = _edge_values(comparable_rows, champion_key)
    challenger_edges = _edge_values(comparable_rows, challenger_key)
    champion_trigger_rate = _trigger_rate(champion_edges, edge_threshold)
    challenger_trigger_rate = _trigger_rate(challenger_edges, edge_threshold)
    scoring_error_rate = _rate(
        int(payload.get("challenger_error_count") or 0),
        int(payload.get("sample_count") or len(rows)),
    )
    schema_error_rate = _rate(
        sum(1 for row in rows if _looks_like_schema_error(row.get("challenger_error"))),
        int(payload.get("sample_count") or len(rows)),
    )
    champion_distribution = _distribution_summary(champion_probs)
    challenger_distribution = _distribution_summary(challenger_probs)
    champion_edge_distribution = _distribution_summary(champion_edges) if champion_edges else None
    challenger_edge_distribution = (
        _distribution_summary(challenger_edges) if challenger_edges else None
    )
    latency = {
        str(payload.get("champion_model_version", "champion")): _latency_summary(
            _row_floats(rows, "champion_latency_ms")
        ),
        str(payload.get("challenger_model_version", "challenger")): _latency_summary(
            _row_floats(rows, "challenger_latency_ms")
        ),
    }
    simulated_pnl = _simulated_pnl(rows, edge_threshold=edge_threshold)
    offline_reference = _read_distribution_reference_payload(offline_reference_path)
    window_start_ts = _optional_int(payload.get("window_start_ts"))
    window_end_ts = _optional_int(payload.get("window_end_ts"))
    session_duration_seconds = _session_duration_seconds(window_start_ts, window_end_ts)
    checks = {
        "prediction_distribution_stability": _distribution_stability_check(
            challenger_distribution,
            _distribution_reference(offline_reference),
            max_mean_abs_diff=max_distribution_mean_abs_diff,
            max_std_relative_diff=max_distribution_std_relative_diff,
            basis="settled_rows" if comparable_rows is not rows else "all_rows",
        ),
        "edge_trigger_rate": _edge_trigger_check(
            challenger_trigger_rate,
            min_rate=min_edge_trigger_rate,
            max_rate=max_edge_trigger_rate,
        ),
        "schema_error_rate": _threshold_check(
            schema_error_rate,
            threshold=max_schema_error_rate,
            detail=f"schema_error_rate={schema_error_rate:.6f}",
        ),
        "scoring_error_rate": _threshold_check(
            scoring_error_rate,
            threshold=0.0,
            detail=f"scoring_error_rate={scoring_error_rate:.6f}",
        ),
        "simulated_pnl": _pnl_check(simulated_pnl),
        "prediction_latency": _latency_check(
            latency[str(payload.get("challenger_model_version", "challenger"))],
            max_p95_ms=max_latency_p95_ms,
        ),
    }
    return ShadowEvaluationResult(
        champion_model_version=str(payload.get("champion_model_version", "champion")),
        challenger_model_version=str(payload.get("challenger_model_version", "challenger")),
        sample_count=int(payload.get("sample_count") or len(rows)),
        scored_count=int(payload.get("scored_count") or len(challenger_probs)),
        edge_threshold=edge_threshold,
        overall_passed=all(bool(check.get("passed")) for check in checks.values()),
        checks=checks,
        champion_probability_distribution=champion_distribution,
        challenger_probability_distribution=challenger_distribution,
        champion_edge_distribution=champion_edge_distribution,
        challenger_edge_distribution=challenger_edge_distribution,
        champion_edge_trigger_rate=champion_trigger_rate,
        challenger_edge_trigger_rate=challenger_trigger_rate,
        schema_error_rate=schema_error_rate,
        scoring_error_rate=scoring_error_rate,
        latency_ms=latency,
        simulated_pnl=simulated_pnl,
        offline_reference_path=None if offline_reference_path is None else str(offline_reference_path),
        offline_reference=_offline_reference_metadata(offline_reference),
        window_start_ts=window_start_ts,
        window_end_ts=window_end_ts,
        session_duration_seconds=session_duration_seconds,
        generated_at_ms=int(time.time() * 1_000),
    )


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


def _session_duration_seconds(start_ts: int | None, end_ts: int | None) -> float | None:
    if start_ts is None or end_ts is None or end_ts < start_ts:
        return None
    return (end_ts - start_ts) / 1_000.0


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _market_implied_probability(row: Mapping[str, Any]) -> float | None:
    for key in ("market_implied_prob", "best_ask", "entry_ask_price"):
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _shadow_payload(
    report: ShadowComparisonReport | Mapping[str, Any] | Path | str,
) -> dict[str, Any]:
    if isinstance(report, ShadowComparisonReport):
        return report.to_dict()
    if isinstance(report, Mapping):
        return dict(report)
    return json.loads(Path(report).read_text(encoding="utf-8"))


def _row_floats(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _optional_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _settled_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if _optional_float(row.get("settlement_price")) is not None
        or _optional_float(row.get("realized_return")) is not None
    ]


def _edge_values(rows: Sequence[Mapping[str, Any]], probability_key: str) -> list[float]:
    values = []
    for row in rows:
        probability = _optional_float(row.get(probability_key))
        market = _market_implied_probability(row)
        if probability is not None and market is not None:
            values.append(probability - market)
    return values


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    cleaned = sorted(value for value in (_optional_float(item) for item in values) if value is not None)
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / len(cleaned)
    return {
        "count": len(cleaned),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": cleaned[0],
        "p05": _quantile(cleaned, 0.05),
        "p25": _quantile(cleaned, 0.25),
        "p50": _quantile(cleaned, 0.50),
        "p75": _quantile(cleaned, 0.75),
        "p95": _quantile(cleaned, 0.95),
        "max": cleaned[-1],
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    distribution = _distribution_summary(values)
    return {
        "count": distribution["count"],
        "p50": distribution["p50"],
        "p95": distribution["p95"],
    }


def _trigger_rate(edges: Sequence[float], threshold: float) -> float | None:
    if not edges:
        return None
    return sum(1 for edge in edges if edge >= threshold) / len(edges)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _read_distribution_reference(path: Path | str | None) -> dict[str, Any] | None:
    return _distribution_reference(_read_distribution_reference_payload(path))


def _read_distribution_reference_payload(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _distribution_reference(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    for key in (
        "challenger_probability_distribution",
        "probability_distribution",
        "prob_up_15m",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return dict(payload)


def _offline_reference_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    keys = (
        "model_version",
        "model_path",
        "dataset_dir",
        "dataset_version",
        "split",
        "row_count",
        "window_start_ts",
        "window_end_ts",
        "generated_at_ms",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _distribution_stability_check(
    current: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    *,
    max_mean_abs_diff: float,
    max_std_relative_diff: float,
    basis: str,
) -> dict[str, Any]:
    if reference is None:
        return {
            "passed": False,
            "detail": "offline validation probability distribution reference missing",
        }
    current_mean = _optional_float(current.get("mean"))
    current_std = _optional_float(current.get("std"))
    reference_mean = _optional_float(reference.get("mean"))
    reference_std = _optional_float(reference.get("std"))
    if None in (current_mean, current_std, reference_mean, reference_std):
        return {"passed": False, "detail": "mean/std missing from current or reference"}
    mean_diff = abs(current_mean - reference_mean)
    std_diff = _relative_diff(current_std, reference_std)
    passed = mean_diff < max_mean_abs_diff and std_diff < max_std_relative_diff
    return {
        "passed": passed,
        "detail": (
            f"basis={basis}, mean_abs_diff={mean_diff:.4f}, "
            f"std_relative_diff={std_diff:.4f}, "
            f"mean_threshold={max_mean_abs_diff:.4f}, "
            f"std_threshold={max_std_relative_diff:.4f}"
        ),
    }


def _edge_trigger_check(
    trigger_rate: float | None,
    *,
    min_rate: float,
    max_rate: float,
) -> dict[str, Any]:
    if trigger_rate is None:
        return {"passed": False, "detail": "market_implied_prob missing; edge unavailable"}
    passed = trigger_rate > min_rate and trigger_rate <= max_rate
    return {
        "passed": passed,
        "detail": (
            f"edge_trigger_rate={trigger_rate:.6f}, min_exclusive={min_rate:.6f}, "
            f"max_inclusive={max_rate:.6f}"
        ),
    }


def _threshold_check(value: float, *, threshold: float, detail: str) -> dict[str, Any]:
    return {"passed": value <= threshold, "detail": detail}


def _latency_check(
    latency: Mapping[str, Any],
    *,
    max_p95_ms: float,
) -> dict[str, Any]:
    p95 = _optional_float(latency.get("p95"))
    if p95 is None:
        return {"passed": False, "detail": "challenger p95 latency missing"}
    return {
        "passed": p95 < max_p95_ms,
        "detail": f"challenger_p95_ms={p95:.4f}, threshold={max_p95_ms:.4f}",
    }


def _pnl_check(simulated_pnl: Mapping[str, Any] | None) -> dict[str, Any]:
    if simulated_pnl is None:
        return {"passed": False, "detail": "settlement data missing; simulated PnL unavailable"}
    delta = _optional_float(simulated_pnl.get("net_pnl_delta"))
    if delta is None:
        return {"passed": False, "detail": "simulated PnL delta missing"}
    return {"passed": delta > 0.0, "detail": f"net_pnl_delta={delta:.6f}"}


def _simulated_pnl(
    rows: Sequence[Mapping[str, Any]],
    *,
    edge_threshold: float,
) -> dict[str, float | int | None] | None:
    champion = _side_pnl(rows, probability_key="champion_prob_up_15m", edge_threshold=edge_threshold)
    challenger = _side_pnl(
        rows,
        probability_key="challenger_prob_up_15m",
        edge_threshold=edge_threshold,
    )
    if champion is None or challenger is None:
        return None
    champion_pnl, champion_trades = champion
    challenger_pnl, challenger_trades = challenger
    return {
        "champion_net_pnl": champion_pnl,
        "champion_trade_count": champion_trades,
        "challenger_net_pnl": challenger_pnl,
        "challenger_trade_count": challenger_trades,
        "net_pnl_delta": challenger_pnl - champion_pnl,
    }


def _side_pnl(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    edge_threshold: float,
) -> tuple[float, int] | None:
    has_settlement = False
    pnl = 0.0
    trades = 0
    for row in rows:
        probability = _optional_float(row.get(probability_key))
        market = _market_implied_probability(row)
        if probability is None or market is None:
            continue
        edge = probability - market
        if edge < edge_threshold:
            continue
        realized_return = _optional_float(row.get("realized_return"))
        settlement_price = _optional_float(row.get("settlement_price"))
        if realized_return is None and settlement_price is None:
            continue
        has_settlement = True
        pnl += realized_return if realized_return is not None else settlement_price - market
        trades += 1
    return (pnl, trades) if has_settlement else None


def _looks_like_schema_error(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).lower()
    return any(token in text for token in ("schema", "missing", "feature", "column"))


def _relative_diff(current: float, reference: float) -> float:
    denominator = abs(reference)
    if denominator <= 1e-12:
        return 0.0 if abs(current) <= 1e-12 else math.inf
    return abs(current - reference) / denominator


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile values must be non-empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _format_optional_float(value: Any) -> str:
    parsed = _optional_float(value)
    return "NA" if parsed is None else f"{parsed:.6f}"


def _format_optional_int(value: int | None) -> str:
    return "NA" if value is None else str(value)


def _distribution_row(model_version: str, distribution: Mapping[str, Any]) -> str:
    return (
        "| "
        + " | ".join(
            (
                _escape_table(model_version),
                str(distribution.get("count")),
                _format_optional_float(distribution.get("mean")),
                _format_optional_float(distribution.get("std")),
                _format_optional_float(distribution.get("p05")),
                _format_optional_float(distribution.get("p25")),
                _format_optional_float(distribution.get("p50")),
                _format_optional_float(distribution.get("p75")),
                _format_optional_float(distribution.get("p95")),
            )
        )
        + " |"
    )


def _edge_distribution_row(
    model_version: str,
    distribution: Mapping[str, Any] | None,
    trigger_rate: float | None,
) -> str:
    distribution = distribution or {}
    return (
        "| "
        + " | ".join(
            (
                _escape_table(model_version),
                str(distribution.get("count", 0)),
                _format_optional_float(distribution.get("mean")),
                _format_optional_float(distribution.get("std")),
                _format_optional_float(distribution.get("p05")),
                _format_optional_float(distribution.get("p25")),
                _format_optional_float(distribution.get("p50")),
                _format_optional_float(distribution.get("p75")),
                _format_optional_float(distribution.get("p95")),
                _format_optional_float(trigger_rate),
            )
        )
        + " |"
    )


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")
