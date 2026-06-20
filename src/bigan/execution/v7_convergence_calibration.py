"""V7 convergence calibration gate for paper/live entry decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class V7ConvergenceCalibrationConfig:
    """Runtime thresholds for the historical convergence calibration gate."""

    path: str = ""
    min_hit_5c_rate: float = 0.0
    min_hit_10c_rate: float = 0.0
    max_model_over_error_p80: float | None = None
    min_adjusted_median_edge: float | None = None
    min_adjusted_p80_edge: float | None = None
    min_bucket_sample_count: int = 20

    @property
    def enabled(self) -> bool:
        return bool(self.path)


@dataclass(frozen=True, slots=True)
class V7ConvergenceCalibrationStats:
    key: tuple[str, ...]
    sample_count: int
    hit_5c_rate: float
    hit_10c_rate: float
    close_rate: float | None = None
    median_best_move: float | None = None
    median_close_move: float | None = None
    median_value_error: float | None = None
    model_over_error_p80: float | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = list(self.key)
        return payload


@dataclass(frozen=True, slots=True)
class V7ConvergenceCalibrationEvaluation:
    enabled: bool
    price: float
    execution_price: float
    model_value: float
    edge: float
    raw_p_side: float | None
    adjusted_model_value_median: float | None
    adjusted_edge_median: float | None
    adjusted_model_value_p80: float | None
    adjusted_edge_p80: float | None
    matched_table: str | None
    key: tuple[str, ...] | None
    stats: V7ConvergenceCalibrationStats | None
    thresholds: V7ConvergenceCalibrationConfig
    skip_reason: str | None

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "price": self.price,
            "execution_price": self.execution_price,
            "model_value": self.model_value,
            "edge": self.edge,
            "raw_p_side": self.raw_p_side,
            "adjusted_model_value_median": self.adjusted_model_value_median,
            "adjusted_edge_median": self.adjusted_edge_median,
            "adjusted_model_value_p80": self.adjusted_model_value_p80,
            "adjusted_edge_p80": self.adjusted_edge_p80,
            "matched_table": self.matched_table,
            "key": None if self.key is None else list(self.key),
            "stats": None if self.stats is None else self.stats.to_payload(),
            "thresholds": asdict(self.thresholds),
            "skip_reason": self.skip_reason,
        }


class V7ConvergenceCalibrationGate:
    """Lookup historical convergence hit-rate stats for a candidate v7 entry."""

    def __init__(
        self,
        *,
        tables: dict[str, dict[tuple[str, ...], V7ConvergenceCalibrationStats]],
        global_stats: V7ConvergenceCalibrationStats | None,
        config: V7ConvergenceCalibrationConfig,
    ) -> None:
        self._tables = tables
        self._global_stats = global_stats
        self.config = config

    @classmethod
    def from_json_path(
        cls,
        path: str | Path,
        *,
        config: V7ConvergenceCalibrationConfig,
    ) -> "V7ConvergenceCalibrationGate":
        artifact_path = Path(path)
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        tables_payload = payload.get("calibration_tables") or payload.get("tables") or {}
        tables: dict[str, dict[tuple[str, ...], V7ConvergenceCalibrationStats]] = {}
        for table_name, rows in tables_payload.items():
            if not isinstance(rows, list):
                continue
            table: dict[tuple[str, ...], V7ConvergenceCalibrationStats] = {}
            for row in rows:
                stats = _stats_from_payload(row)
                if stats is None:
                    continue
                if stats.sample_count < config.min_bucket_sample_count:
                    continue
                table[stats.key] = stats
            tables[str(table_name)] = table
        global_stats = _global_stats_from_payload(payload)
        return cls(tables=tables, global_stats=global_stats, config=config)

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "config": asdict(self.config),
            "table_counts": {
                name: len(table) for name, table in sorted(self._tables.items())
            },
            "has_global_stats": self._global_stats is not None,
        }

    def evaluate(
        self,
        *,
        price: float,
        model_value: float,
        edge: float,
        raw_p_side: float | None,
        execution_price: float | None = None,
    ) -> V7ConvergenceCalibrationEvaluation:
        effective_execution_price = price if execution_price is None else execution_price
        for table_name, key in _hierarchy(price, raw_p_side, edge, model_value):
            stats = self._tables.get(table_name, {}).get(key)
            if stats is None:
                continue
            adjusted = _adjusted_metrics(
                stats,
                model_value=model_value,
                execution_price=effective_execution_price,
            )
            return V7ConvergenceCalibrationEvaluation(
                enabled=True,
                price=price,
                execution_price=effective_execution_price,
                model_value=model_value,
                edge=edge,
                raw_p_side=raw_p_side,
                adjusted_model_value_median=adjusted[0],
                adjusted_edge_median=adjusted[1],
                adjusted_model_value_p80=adjusted[2],
                adjusted_edge_p80=adjusted[3],
                matched_table=table_name,
                key=key,
                stats=stats,
                thresholds=self.config,
                skip_reason=_skip_reason(
                    stats,
                    self.config,
                    adjusted_edge_median=adjusted[1],
                    adjusted_edge_p80=adjusted[3],
                ),
            )
        if self._global_stats is None:
            return V7ConvergenceCalibrationEvaluation(
                enabled=True,
                price=price,
                execution_price=effective_execution_price,
                model_value=model_value,
                edge=edge,
                raw_p_side=raw_p_side,
                adjusted_model_value_median=None,
                adjusted_edge_median=None,
                adjusted_model_value_p80=None,
                adjusted_edge_p80=None,
                matched_table=None,
                key=None,
                stats=None,
                thresholds=self.config,
                skip_reason="calibrated_bucket_missing",
            )
        adjusted = _adjusted_metrics(
            self._global_stats,
            model_value=model_value,
            execution_price=effective_execution_price,
        )
        return V7ConvergenceCalibrationEvaluation(
            enabled=True,
            price=price,
            execution_price=effective_execution_price,
            model_value=model_value,
            edge=edge,
            raw_p_side=raw_p_side,
            adjusted_model_value_median=adjusted[0],
            adjusted_edge_median=adjusted[1],
            adjusted_model_value_p80=adjusted[2],
            adjusted_edge_p80=adjusted[3],
            matched_table="global",
            key=self._global_stats.key,
            stats=self._global_stats,
            thresholds=self.config,
            skip_reason=_skip_reason(
                self._global_stats,
                self.config,
                adjusted_edge_median=adjusted[1],
                adjusted_edge_p80=adjusted[3],
            ),
        )


def _skip_reason(
    stats: V7ConvergenceCalibrationStats,
    config: V7ConvergenceCalibrationConfig,
    *,
    adjusted_edge_median: float | None,
    adjusted_edge_p80: float | None,
) -> str | None:
    if stats.sample_count < config.min_bucket_sample_count:
        return "calibrated_sample_count_below_min"
    if stats.hit_5c_rate < config.min_hit_5c_rate:
        return "calibrated_hit_5c_below_min"
    if stats.hit_10c_rate < config.min_hit_10c_rate:
        return "calibrated_hit_10c_below_min"
    if (
        config.max_model_over_error_p80 is not None
        and stats.model_over_error_p80 is not None
        and stats.model_over_error_p80 > config.max_model_over_error_p80
    ):
        return "model_over_error_p80_above_max"
    if config.min_adjusted_median_edge is not None:
        if adjusted_edge_median is None:
            return "calibrated_median_value_error_missing"
        if adjusted_edge_median < config.min_adjusted_median_edge:
            return "calibrated_median_edge_below_min"
    if config.min_adjusted_p80_edge is not None:
        if adjusted_edge_p80 is None:
            return "calibrated_model_over_error_p80_missing"
        if adjusted_edge_p80 < config.min_adjusted_p80_edge:
            return "calibrated_p80_edge_below_min"
    return None


def _adjusted_metrics(
    stats: V7ConvergenceCalibrationStats,
    *,
    model_value: float,
    execution_price: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    median_value = None
    median_edge = None
    if stats.median_value_error is not None:
        median_value = _clip_probability(model_value + stats.median_value_error)
        median_edge = median_value - execution_price

    p80_value = None
    p80_edge = None
    if stats.model_over_error_p80 is not None:
        p80_value = _clip_probability(model_value - stats.model_over_error_p80)
        p80_edge = p80_value - execution_price
    return median_value, median_edge, p80_value, p80_edge


def _clip_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _global_stats_from_payload(
    payload: dict[str, Any],
) -> V7ConvergenceCalibrationStats | None:
    for key in ("global", "global_stats", "calibration_summary"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            stats = _stats_from_payload({**candidate, "key": ["GLOBAL"]})
            if stats is not None:
                return stats
    return _stats_from_payload(payload)


def _stats_from_payload(payload: Any) -> V7ConvergenceCalibrationStats | None:
    if not isinstance(payload, dict):
        return None
    if "sample_count" not in payload:
        return None
    key_payload = payload.get("key", ["GLOBAL"])
    if isinstance(key_payload, (str, int, float)):
        key = (str(key_payload),)
    elif isinstance(key_payload, list):
        key = tuple(str(item) for item in key_payload)
    elif isinstance(key_payload, tuple):
        key = tuple(str(item) for item in key_payload)
    else:
        key = ("GLOBAL",)
    return V7ConvergenceCalibrationStats(
        key=key,
        sample_count=int(payload.get("sample_count", 0)),
        hit_5c_rate=float(payload.get("hit_5c_rate", 0.0)),
        hit_10c_rate=float(payload.get("hit_10c_rate", 0.0)),
        close_rate=_optional_float(payload.get("close_rate")),
        median_best_move=_optional_float(payload.get("median_best_move")),
        median_close_move=_optional_float(payload.get("median_close_move")),
        median_value_error=_optional_float(payload.get("median_value_error")),
        model_over_error_p80=_optional_float(payload.get("model_over_error_p80")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _hierarchy(
    price: float,
    raw_p_side: float | None,
    edge: float,
    model_value: float,
) -> list[tuple[str, tuple[str, ...]]]:
    price_bucket = _price_bucket(price)
    raw_bucket = _raw_bucket(raw_p_side)
    edge_bucket = _edge_bucket(edge)
    model_bucket = _model_bucket(model_value)
    return [
        (
            "price_raw_edge_model",
            (price_bucket, raw_bucket, edge_bucket, model_bucket),
        ),
        ("price_raw_edge", (price_bucket, raw_bucket, edge_bucket)),
        ("price_raw", (price_bucket, raw_bucket)),
        ("price_edge", (price_bucket, edge_bucket)),
        ("price", (price_bucket,)),
    ]


def _price_bucket(price: float) -> str:
    if price < 0.30:
        return "<0.30"
    if price < 0.40:
        return "0.30-0.40"
    if price < 0.50:
        return "0.40-0.50"
    if price < 0.70:
        return "0.50-0.70"
    return ">=0.70"


def _raw_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.55:
        return "<0.55"
    if value < 0.60:
        return "0.55-0.60"
    if value < 0.65:
        return "0.60-0.65"
    return ">=0.65"


def _edge_bucket(edge: float) -> str:
    if edge < 0.30:
        return "<0.30"
    if edge < 0.40:
        return "0.30-0.40"
    if edge < 0.50:
        return "0.40-0.50"
    return ">=0.50"


def _model_bucket(value: float) -> str:
    if value < 0.70:
        return "<0.70"
    if value < 0.80:
        return "0.70-0.80"
    return ">=0.80"
