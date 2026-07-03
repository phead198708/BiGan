"""Phase 1 policy-learning contracts for the v8 trading architecture."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

PHASE1_POLICY_VERSION = "bigan-v8-phase1-policy-v1.0.0"
XGBOOST_V8_POLICY_MODEL_VERSION = "xgboost-v8-policy"

SUPPORTED_POLICY_OBJECTIVES: tuple[str, ...] = (
    "binary:logistic",
    "rank:pairwise",
)

SUPPORTED_TARGET_ENCODINGS: tuple[str, ...] = (
    "binary_positive_net_return_threshold",
    "rank_discrete_net_return_quality_bucket",
)

SUPPORTED_RANKING_GROUP_STRATEGIES: tuple[str, ...] = (
    "source_instrument_regime",
    "source_instrument",
    "source_instrument_day",
)

FORBIDDEN_DIRECT_PNL_TOKENS: tuple[str, ...] = (
    "pnl",
    "profit",
    "sharpe",
    "sortino",
    "drawdown",
    "roi",
    "return",
    "realized",
)

PolicyObjective = Literal["binary:logistic", "rank:pairwise"]
PolicyTargetEncoding = Literal[
    "binary_positive_net_return_threshold",
    "rank_discrete_net_return_quality_bucket",
]
RankingGroupStrategy = Literal[
    "source_instrument_regime",
    "source_instrument",
    "source_instrument_day",
]


@dataclass(frozen=True, slots=True)
class PolicyDatasetConfig:
    """Configuration for converting Phase 0 labels into pure policy targets."""

    horizon_ms: int | None = None
    target_encoding: PolicyTargetEncoding = "binary_positive_net_return_threshold"
    positive_return_threshold: float = 0.0
    rank_quality_bucket_edges: tuple[float, ...] = (-0.005, 0.0, 0.005)
    max_position_size: float = 1.0
    feature_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon_ms is not None and self.horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive when provided")
        if self.target_encoding not in SUPPORTED_TARGET_ENCODINGS:
            raise ValueError(
                "target_encoding must be one of "
                + ", ".join(SUPPORTED_TARGET_ENCODINGS)
            )
        if not 0.0 < self.max_position_size <= 1.0:
            raise ValueError("max_position_size must be in (0, 1]")
        if self.positive_return_threshold < 0.0:
            raise ValueError("positive_return_threshold must be non-negative")
        previous: float | None = None
        for edge in self.rank_quality_bucket_edges:
            if not math.isfinite(edge):
                raise ValueError("rank_quality_bucket_edges must be finite")
            if previous is not None and edge <= previous:
                raise ValueError("rank_quality_bucket_edges must be strictly increasing")
            previous = edge

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_columns"] = list(self.feature_columns)
        payload["rank_quality_bucket_edges"] = list(self.rank_quality_bucket_edges)
        return payload


@dataclass(frozen=True, slots=True)
class PolicyTrainingExample:
    """One Phase 1 supervised/ranking example derived from Phase 0."""

    decision_ts: int
    source: str
    instrument_id: str
    features: Mapping[str, float | int | None]
    target_label: float
    shadow_net_return: float
    horizon_ms: int
    regime_key: str

    def __post_init__(self) -> None:
        if self.decision_ts < 0:
            raise ValueError("decision_ts must be non-negative")
        if not self.source:
            raise ValueError("source is required")
        if not self.instrument_id:
            raise ValueError("instrument_id is required")
        if not self.features:
            raise ValueError("features must not be empty")
        if not math.isfinite(self.target_label):
            raise ValueError("target_label must be finite")
        if self.target_label < 0.0:
            raise ValueError("target_label must be non-negative")
        if not math.isfinite(self.shadow_net_return):
            raise ValueError("shadow_net_return must be finite")
        if self.horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive")
        if not self.regime_key:
            raise ValueError("regime_key is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_ts": self.decision_ts,
            "source": self.source,
            "instrument_id": self.instrument_id,
            "features": dict(self.features),
            "target_label": self.target_label,
            "shadow_net_return": self.shadow_net_return,
            "horizon_ms": self.horizon_ms,
            "regime_key": self.regime_key,
        }


@dataclass(frozen=True, slots=True)
class PolicyDataset:
    """Deterministic Phase 1 training dataset."""

    examples: tuple[PolicyTrainingExample, ...]
    feature_columns: tuple[str, ...]
    policy_dataset_hash: str
    phase0_dataset_hash: str
    phase0_dataset_version: str
    config: PolicyDatasetConfig

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("policy dataset must contain at least one example")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not self.policy_dataset_hash:
            raise ValueError("policy_dataset_hash is required")
        if not self.phase0_dataset_hash:
            raise ValueError("phase0_dataset_hash is required")
        if not self.phase0_dataset_version:
            raise ValueError("phase0_dataset_version is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase1_policy_version": PHASE1_POLICY_VERSION,
            "policy_dataset_hash": self.policy_dataset_hash,
            "phase0_dataset_hash": self.phase0_dataset_hash,
            "phase0_dataset_version": self.phase0_dataset_version,
            "row_count": len(self.examples),
            "feature_columns": list(self.feature_columns),
            "config": self.config.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PolicyTrainShadowSplit:
    """Temporal out-of-sample split for policy training and shadow acceptance."""

    train_examples: tuple[PolicyTrainingExample, ...]
    shadow_examples: tuple[PolicyTrainingExample, ...]
    split_ts: int
    split_hash: str
    train_dataset_hash: str
    shadow_dataset_hash: str

    def __post_init__(self) -> None:
        if not self.train_examples:
            raise ValueError("train_examples must not be empty")
        if not self.shadow_examples:
            raise ValueError("shadow_examples must not be empty")
        if self.split_ts <= 0:
            raise ValueError("split_ts must be positive")
        max_train_ts = max(example.decision_ts for example in self.train_examples)
        min_shadow_ts = min(example.decision_ts for example in self.shadow_examples)
        if max_train_ts >= min_shadow_ts:
            raise ValueError("temporal split must satisfy max(train_ts) < min(shadow_ts)")
        if max_train_ts >= self.split_ts or min_shadow_ts < self.split_ts:
            raise ValueError("split_ts must separate train and shadow examples")
        if not self.split_hash:
            raise ValueError("split_hash is required")
        if not self.train_dataset_hash:
            raise ValueError("train_dataset_hash is required")
        if not self.shadow_dataset_hash:
            raise ValueError("shadow_dataset_hash is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_ts": self.split_ts,
            "split_hash": self.split_hash,
            "train_dataset_hash": self.train_dataset_hash,
            "shadow_dataset_hash": self.shadow_dataset_hash,
            "train_row_count": len(self.train_examples),
            "shadow_row_count": len(self.shadow_examples),
            "max_train_decision_ts": max(
                example.decision_ts for example in self.train_examples
            ),
            "min_shadow_decision_ts": min(
                example.decision_ts for example in self.shadow_examples
            ),
        }


@dataclass(frozen=True, slots=True)
class XGBoostPolicyConfig:
    """Pure policy-learning config.

    Shadow PnL is intentionally absent from objective and model-selection
    fields. It belongs in the acceptance validator after policy inference.
    """

    model_version: str = XGBOOST_V8_POLICY_MODEL_VERSION
    objective: PolicyObjective = "binary:logistic"
    eval_metric: str = "logloss"
    selection_metric: str = "logloss"
    num_boost_round: int = 40
    max_depth: int = 3
    learning_rate: float = 0.05
    min_child_weight: float = 1.0
    subsample: float = 0.90
    colsample_bytree: float = 0.90
    l2_penalty: float = 1.0
    seed: int = 0
    max_position_size: float = 1.0
    action_activation_threshold: float = 0.55
    ranking_group_strategy: RankingGroupStrategy = "source_instrument_regime"
    min_ranking_group_size: int = 2
    min_effective_ranking_groups: int = 1
    regime_feature_names: tuple[str, ...] = (
        "volatility_5m",
        "volatility_15m",
        "spread_bps",
        "liquidity_depth",
    )

    def __post_init__(self) -> None:
        if self.model_version != XGBOOST_V8_POLICY_MODEL_VERSION:
            raise ValueError(f"model_version must be {XGBOOST_V8_POLICY_MODEL_VERSION!r}")
        if self.objective not in SUPPORTED_POLICY_OBJECTIVES:
            raise ValueError(
                "objective must be one of " + ", ".join(SUPPORTED_POLICY_OBJECTIVES)
            )
        if self.num_boost_round <= 0:
            raise ValueError("num_boost_round must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.min_child_weight <= 0.0:
            raise ValueError("min_child_weight must be positive")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if self.l2_penalty < 0.0:
            raise ValueError("l2_penalty must be non-negative")
        if not 0.0 < self.max_position_size <= 1.0:
            raise ValueError("max_position_size must be in (0, 1]")
        if not 0.0 <= self.action_activation_threshold <= 1.0:
            raise ValueError("action_activation_threshold must be in [0, 1]")
        if self.ranking_group_strategy not in SUPPORTED_RANKING_GROUP_STRATEGIES:
            raise ValueError(
                "ranking_group_strategy must be one of "
                + ", ".join(SUPPORTED_RANKING_GROUP_STRATEGIES)
            )
        if self.min_ranking_group_size < 2:
            raise ValueError("min_ranking_group_size must be at least 2")
        if self.min_effective_ranking_groups < 1:
            raise ValueError("min_effective_ranking_groups must be at least 1")
        assert_no_direct_pnl_optimization(
            objective=self.objective,
            eval_metric=self.eval_metric,
            selection_metric=self.selection_metric,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regime_feature_names"] = list(self.regime_feature_names)
        return payload

    def xgboost_params(self) -> dict[str, float | int | str]:
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda": self.l2_penalty,
            "seed": self.seed,
            "nthread": 1,
            "verbosity": 0,
        }


@dataclass(frozen=True, slots=True)
class PolicyPrediction:
    """Policy inference output consumed by execution simulation."""

    decision_ts: int
    source: str
    instrument_id: str
    action: float
    confidence: float
    regime_embedding: tuple[float, ...]
    score: float

    def __post_init__(self) -> None:
        if self.decision_ts < 0:
            raise ValueError("decision_ts must be non-negative")
        if not 0.0 <= self.action <= 1.0:
            raise ValueError("action must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_ts": self.decision_ts,
            "source": self.source,
            "instrument_id": self.instrument_id,
            "action": self.action,
            "confidence": self.confidence,
            "regime_embedding": list(self.regime_embedding),
            "score": self.score,
        }


def assert_no_direct_pnl_optimization(
    *,
    objective: str,
    eval_metric: str,
    selection_metric: str,
) -> None:
    """Reject direct PnL/profit optimization knobs in the trainer."""

    for field_name, value in (
        ("objective", objective),
        ("eval_metric", eval_metric),
        ("selection_metric", selection_metric),
    ):
        lowered = value.lower()
        for token in FORBIDDEN_DIRECT_PNL_TOKENS:
            if token in lowered:
                raise ValueError(
                    f"{field_name} must not directly optimize trading PnL/profit metrics: {value}"
                )
