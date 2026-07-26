"""Decision-time provider-health and feature-completeness contracts for v8."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

from bigan.v8.polymarket.corpus.contracts import PolymarketCorpusTrade

FEATURE_MISSINGNESS_CONTRACT_VERSION = "bigan-v8-feature-missingness-v1"
TRADE_TAPE_LOOKBACK_MS = 60_000
TradeTapeCollectionMode = Literal["websocket", "rest", "timeout", "backfill", "mock"]
ALLOWED_COLLECTION_MODES = frozenset(
    {"websocket", "rest", "timeout", "backfill", "mock"}
)


class FeatureCompletenessError(ValueError):
    """Raised when provider-dependent features cannot be proved complete."""


@dataclass(frozen=True, slots=True)
class TradeTapeCoverageStatus:
    """Causal coverage evidence for one market and decision-time lookback."""

    market_id: str
    decision_ts: int
    provider_source: str
    collection_mode: TradeTapeCollectionMode
    collection_started_ts: int
    collection_completed_ts: int
    observation_window_start_ts: int
    observation_window_end_ts: int
    max_causal_input_ts: int
    available_at_ts: int
    observed_trade_count: int
    provider_timeout: bool
    truncated: bool
    censored: bool
    coverage_complete: bool
    missingness_reason: str | None
    provider_health_score: float
    historical_backfill: bool = False

    def __post_init__(self) -> None:
        if not self.market_id.strip() or not self.provider_source.strip():
            raise FeatureCompletenessError("market_id and provider_source are required")
        if self.collection_mode not in ALLOWED_COLLECTION_MODES:
            raise FeatureCompletenessError("unsupported trade-tape collection mode")
        timestamps = {
            "decision_ts": self.decision_ts,
            "collection_started_ts": self.collection_started_ts,
            "collection_completed_ts": self.collection_completed_ts,
            "observation_window_start_ts": self.observation_window_start_ts,
            "observation_window_end_ts": self.observation_window_end_ts,
            "max_causal_input_ts": self.max_causal_input_ts,
            "available_at_ts": self.available_at_ts,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in timestamps.values()):
            raise FeatureCompletenessError("coverage timestamps must be integer milliseconds")
        if any(value < 0 for value in timestamps.values()):
            raise FeatureCompletenessError("coverage timestamps must be non-negative")
        if self.collection_started_ts > self.collection_completed_ts:
            raise FeatureCompletenessError("collection timestamps are not ordered")
        if self.observation_window_start_ts > self.observation_window_end_ts:
            raise FeatureCompletenessError("observation window is not ordered")
        if any(
            value > self.decision_ts
            for value in (
                self.collection_completed_ts,
                self.observation_window_end_ts,
                self.max_causal_input_ts,
                self.available_at_ts,
            )
        ):
            raise FeatureCompletenessError(
                "provider-health evidence must be available no later than decision_ts"
            )
        if self.observed_trade_count < 0:
            raise FeatureCompletenessError("observed_trade_count must be non-negative")
        if not math.isfinite(self.provider_health_score) or not (
            0.0 <= self.provider_health_score <= 1.0
        ):
            raise FeatureCompletenessError("provider_health_score must be finite in [0, 1]")
        impairment = (
            self.provider_timeout
            or self.truncated
            or self.censored
            or self.historical_backfill
            or self.collection_mode in {"timeout", "backfill"}
        )
        if self.coverage_complete and impairment:
            raise FeatureCompletenessError(
                "impaired, truncated, censored, timeout, or backfill tape cannot be complete"
            )
        if self.coverage_complete and self.missingness_reason is not None:
            raise FeatureCompletenessError(
                "complete coverage cannot carry a missingness reason"
            )
        if not self.coverage_complete and not str(self.missingness_reason or "").strip():
            raise FeatureCompletenessError(
                "incomplete coverage requires an explicit missingness reason"
            )
        if self.collection_mode == "timeout" and not self.provider_timeout:
            raise FeatureCompletenessError("timeout mode requires provider_timeout=true")
        if self.collection_mode == "backfill" and not self.historical_backfill:
            raise FeatureCompletenessError("backfill mode requires historical_backfill=true")

    @property
    def collection_latency_ms(self) -> int:
        return self.collection_completed_ts - self.collection_started_ts

    @property
    def data_age_ms(self) -> int:
        return self.decision_ts - self.max_causal_input_ts

    def covers(self, *, lookback_ms: int) -> bool:
        return (
            self.observation_window_start_ts <= self.decision_ts - lookback_ms
            and self.observation_window_end_ts >= self.decision_ts
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = FEATURE_MISSINGNESS_CONTRACT_VERSION
        payload["collection_latency_ms"] = self.collection_latency_ms
        payload["data_age_ms"] = self.data_age_ms
        payload["paper_only"] = True
        payload["capital_at_risk"] = False
        return payload


def build_trade_volume_feature_bundle(
    *,
    trades: tuple[PolymarketCorpusTrade, ...],
    status: TradeTapeCoverageStatus,
    lookback_ms: int = TRADE_TAPE_LOOKBACK_MS,
) -> tuple[dict[str, float | int | str | None], dict[str, dict[str, int | str | None]]]:
    """Return volume plus missingness metadata; never encode unknown as zero."""

    if lookback_ms <= 0:
        raise FeatureCompletenessError("lookback_ms must be positive")
    complete = status.coverage_complete and status.covers(lookback_ms=lookback_ms)
    missingness_reason = status.missingness_reason
    if status.coverage_complete and not status.covers(lookback_ms=lookback_ms):
        complete = False
        missingness_reason = "observation_window_incomplete"
    causal_trades = tuple(
        trade
        for trade in trades
        if trade.market_id == status.market_id
        and status.decision_ts - lookback_ms <= trade.ts <= status.decision_ts
        and trade.available_at_ts <= status.decision_ts
    )
    up_volume = (
        sum(trade.size for trade in causal_trades if trade.outcome == "UP")
        if complete
        else None
    )
    down_volume = (
        sum(trade.size for trade in causal_trades if trade.outcome == "DOWN")
        if complete
        else None
    )
    features: dict[str, float | int | str | None] = {
        "recent_up_trade_volume": up_volume,
        "recent_up_trade_volume_missing": int(not complete),
        "recent_up_trade_volume_coverage_complete": int(complete),
        "recent_down_trade_volume": down_volume,
        "recent_down_trade_volume_missing": int(not complete),
        "recent_down_trade_volume_coverage_complete": int(complete),
        "trade_tape_missingness_reason": missingness_reason,
        "trade_tape_collection_mode": status.collection_mode,
        "trade_tape_provider_source": status.provider_source,
        "trade_tape_provider_timeout": int(status.provider_timeout),
        "trade_tape_truncated": int(status.truncated),
        "trade_tape_censored": int(status.censored),
        "trade_tape_historical_backfill": int(status.historical_backfill),
        "trade_tape_observation_window_start_ts": status.observation_window_start_ts,
        "trade_tape_observation_window_end_ts": status.observation_window_end_ts,
        "trade_tape_max_causal_input_ts": status.max_causal_input_ts,
        "trade_tape_available_at_ts": status.available_at_ts,
        "trade_tape_collection_latency_ms": status.collection_latency_ms,
        "trade_tape_data_age_ms": status.data_age_ms,
        "trade_tape_observed_trade_count": status.observed_trade_count,
        "provider_health_score": status.provider_health_score,
    }
    provenance = {
        name: {
            "source": status.provider_source,
            "input_start_ts": status.observation_window_start_ts,
            "input_end_ts": status.max_causal_input_ts,
            "available_at_ts": status.available_at_ts,
            "lookback_ms": lookback_ms,
            "collection_mode": status.collection_mode,
            "missingness_reason": missingness_reason,
        }
        for name in features
    }
    validate_trade_volume_feature_bundle(
        features,
        decision_ts=status.decision_ts,
        require_complete=False,
    )
    return features, provenance


def validate_trade_volume_feature_bundle(
    features: dict[str, Any],
    *,
    decision_ts: int,
    require_complete: bool,
) -> None:
    """Validate model-facing missingness fields for a future candidate."""

    required = {
        "recent_up_trade_volume",
        "recent_up_trade_volume_missing",
        "recent_up_trade_volume_coverage_complete",
        "recent_down_trade_volume",
        "recent_down_trade_volume_missing",
        "recent_down_trade_volume_coverage_complete",
        "trade_tape_missingness_reason",
        "trade_tape_collection_mode",
        "trade_tape_provider_source",
        "trade_tape_provider_timeout",
        "trade_tape_truncated",
        "trade_tape_censored",
        "trade_tape_historical_backfill",
        "trade_tape_observation_window_start_ts",
        "trade_tape_observation_window_end_ts",
        "trade_tape_max_causal_input_ts",
        "trade_tape_available_at_ts",
        "trade_tape_collection_latency_ms",
        "trade_tape_data_age_ms",
        "provider_health_score",
    }
    missing = sorted(required - set(features))
    if missing:
        raise FeatureCompletenessError(
            "required feature-completeness metadata missing: " + ", ".join(missing)
        )
    for side in ("up", "down"):
        value = features[f"recent_{side}_trade_volume"]
        is_missing = features[f"recent_{side}_trade_volume_missing"] == 1
        complete = features[f"recent_{side}_trade_volume_coverage_complete"] == 1
        if is_missing == complete:
            raise FeatureCompletenessError(
                f"recent_{side}_trade_volume missing and completeness flags conflict"
            )
        if is_missing and value is not None:
            raise FeatureCompletenessError(
                f"missing recent_{side}_trade_volume must be null"
            )
        if complete and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise FeatureCompletenessError(
                f"complete recent_{side}_trade_volume must be finite and non-negative"
            )
    causal_timestamps = (
        int(features["trade_tape_observation_window_end_ts"]),
        int(features["trade_tape_max_causal_input_ts"]),
        int(features["trade_tape_available_at_ts"]),
    )
    if any(value > decision_ts for value in causal_timestamps):
        raise FeatureCompletenessError(
            "feature-completeness metadata is not causally available at decision time"
        )
    impaired = any(
        features[field] == 1
        for field in (
            "trade_tape_provider_timeout",
            "trade_tape_truncated",
            "trade_tape_censored",
            "trade_tape_historical_backfill",
        )
    )
    complete = (
        features["recent_up_trade_volume_coverage_complete"] == 1
        and features["recent_down_trade_volume_coverage_complete"] == 1
    )
    if impaired and complete:
        raise FeatureCompletenessError("impaired trade tape cannot emit complete volume")
    if require_complete and not complete:
        raise FeatureCompletenessError(
            "future candidate requires complete provider-dependent features"
        )


def build_provider_health_diagnostics(
    *,
    feature_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build outcome-blind missingness/fallback/abstention diagnostics."""

    by_key = {
        (str(row["market_id"]), int(row["decision_ts"])): dict(row.get("features") or {})
        for row in feature_rows
    }
    health_counts: Counter[str] = Counter()
    zero_counts: Counter[str] = Counter()
    association: dict[str, Counter[str]] = defaultdict(Counter)
    side_counts: dict[str, Counter[str]] = defaultdict(Counter)
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unmatched = 0
    for decision in decision_rows:
        key = (str(decision["market_id"]), int(decision["decision_ts"]))
        features = by_key.get(key)
        if features is None:
            unmatched += 1
            continue
        bucket = _provider_health_bucket(features)
        health_counts[bucket] += 1
        for side in ("up", "down"):
            value = features.get(f"recent_{side}_trade_volume")
            missing = features.get(f"recent_{side}_trade_volume_missing") == 1
            zero_counts[f"{side}_missing" if missing else f"{side}_zero" if value == 0 else f"{side}_positive"] += 1
        origin = str(
            decision.get("decision_origin")
            or decision.get("selection_source")
            or "primary"
        ).lower()
        selected_action = str(
            decision.get("executed_action")
            or decision.get("selected_action")
            or decision.get("action")
            or "NO_TRADE"
        )
        if selected_action == "NO_TRADE":
            association[bucket]["no_trade"] += 1
        elif "fallback" in origin:
            association[bucket]["fallback"] += 1
        else:
            association[bucket]["primary"] += 1
        if decision.get("execution_guard_order_allowed") is False:
            association[bucket]["execution_guard_rejected"] += 1
        side_counts[bucket][str(decision.get("selected_side") or "NONE")] += 1
        action_counts[bucket][selected_action] += 1
    complete_rows = sum(
        1
        for features in by_key.values()
        if features.get("recent_up_trade_volume_coverage_complete") == 1
        and features.get("recent_down_trade_volume_coverage_complete") == 1
    )
    return {
        "schema_version": "bigan-v8-provider-health-diagnostics-v1",
        "feature_completeness_report": {
            "feature_row_count": len(feature_rows),
            "complete_feature_row_count": complete_rows,
            "incomplete_feature_row_count": len(feature_rows) - complete_rows,
            "provider_health_bucket_counts": dict(sorted(health_counts.items())),
        },
        "missing_versus_zero_audit_report": dict(sorted(zero_counts.items())),
        "fallback_provider_health_association_report": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(association.items())
        },
        "side_composition_by_provider_health": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(side_counts.items())
        },
        "action_family_composition_by_provider_health": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(action_counts.items())
        },
        "unmatched_decision_count": unmatched,
        "diagnostic_only": True,
        "outcomes_settlement_pnl_or_future_information_used": False,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "capital_at_risk": False,
    }


def _provider_health_bucket(features: dict[str, Any]) -> str:
    if features.get("trade_tape_provider_timeout") == 1:
        return "timeout"
    if features.get("trade_tape_truncated") == 1:
        return "truncated"
    if features.get("trade_tape_censored") == 1:
        return "censored"
    if features.get("trade_tape_historical_backfill") == 1:
        return "backfill"
    if features.get("recent_up_trade_volume_coverage_complete") != 1:
        return "incomplete"
    score = float(features.get("provider_health_score") or 0.0)
    if score >= 0.9:
        return "healthy"
    if score >= 0.5:
        return "degraded"
    return "unhealthy"


__all__ = [
    "FEATURE_MISSINGNESS_CONTRACT_VERSION",
    "TRADE_TAPE_LOOKBACK_MS",
    "FeatureCompletenessError",
    "TradeTapeCoverageStatus",
    "build_provider_health_diagnostics",
    "build_trade_volume_feature_bundle",
    "validate_trade_volume_feature_bundle",
]
