"""Feature-row completeness scoring and training filters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureQualityConfig:
    """Thresholds for issue #8 feature data-quality flags."""

    quote_max_age_ms: int = 120_000
    depth_max_age_ms: int = 120_000
    trade_max_age_ms: int = 300_000
    min_completeness_score: float = 0.8


DEFAULT_QUALITY_CONFIG = FeatureQualityConfig()


def compute_quality_fields(
    *,
    feature_ts: int,
    quote_ts: int | None,
    depth_ts: int | None,
    trade_ts: int | None,
    config: FeatureQualityConfig = DEFAULT_QUALITY_CONFIG,
) -> dict[str, int | float | bool | None]:
    """Return quality columns for one feature row.

    Quote and depth are required market-state inputs. Trades are naturally
    sparse, so a missing trade does not mark the row as gappy; when a trade has
    been observed, an old trade timestamp still lowers the score.
    """

    quote_age = _age_ms(feature_ts, quote_ts)
    depth_age = _age_ms(feature_ts, depth_ts)
    trade_age = _age_ms(feature_ts, trade_ts)
    quote_score = _freshness_score(
        quote_age,
        max_age_ms=config.quote_max_age_ms,
        required=True,
    )
    depth_score = _freshness_score(
        depth_age,
        max_age_ms=config.depth_max_age_ms,
        required=True,
    )
    trade_score = _freshness_score(
        trade_age,
        max_age_ms=config.trade_max_age_ms,
        required=False,
    )
    completeness_score = 0.4 * quote_score + 0.4 * depth_score + 0.2 * trade_score
    data_gap_flag = (
        _required_gap(quote_age, config.quote_max_age_ms)
        or _required_gap(depth_age, config.depth_max_age_ms)
        or (trade_age is not None and trade_age > config.trade_max_age_ms)
    )
    quality_filter_pass = (
        not data_gap_flag and completeness_score >= config.min_completeness_score
    )
    return {
        "quote_age_ms": quote_age,
        "depth_age_ms": depth_age,
        "trade_age_ms": trade_age,
        "completeness_score": completeness_score,
        "data_gap_flag": data_gap_flag,
        "quality_filter_pass": quality_filter_pass,
    }


def feature_row_passes_quality(
    row: Mapping[str, Any],
    *,
    min_completeness_score: float = DEFAULT_QUALITY_CONFIG.min_completeness_score,
    allow_gaps: bool = False,
) -> bool:
    """Return whether a feature row is acceptable for model training."""

    score = _as_float(row.get("completeness_score"))
    if score is None or score < min_completeness_score:
        return False
    if not allow_gaps and bool(row.get("data_gap_flag")):
        return False
    return bool(row.get("quality_filter_pass", True))


def filter_trainable_feature_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_completeness_score: float = DEFAULT_QUALITY_CONFIG.min_completeness_score,
    allow_gaps: bool = False,
) -> list[dict[str, Any]]:
    """Keep feature rows that pass the issue #8 quality filter."""

    return [
        dict(row)
        for row in rows
        if feature_row_passes_quality(
            row,
            min_completeness_score=min_completeness_score,
            allow_gaps=allow_gaps,
        )
    ]


def _age_ms(feature_ts: int, ts: int | None) -> int | None:
    if ts is None:
        return None
    return max(0, feature_ts - ts)


def _freshness_score(age_ms: int | None, *, max_age_ms: int, required: bool) -> float:
    if age_ms is None:
        return 0.0 if required else 1.0
    if age_ms <= max_age_ms:
        return 1.0
    if max_age_ms <= 0:
        return 0.0
    return max(0.0, 1.0 - ((age_ms - max_age_ms) / max_age_ms))


def _required_gap(age_ms: int | None, max_age_ms: int) -> bool:
    return age_ms is None or age_ms > max_age_ms


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
