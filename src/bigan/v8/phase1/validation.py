"""Phase 1 policy acceptance validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from bigan.v8.phase1.contracts import (
    PolicyPrediction,
    PolicyTrainingExample,
    PolicyTrainShadowSplit,
)


@dataclass(frozen=True, slots=True)
class PolicyAcceptanceConfig:
    """Shadow acceptance thresholds for Phase 1 policies."""

    max_position_size: float = 1.0
    min_shadow_sharpe: float = 0.0
    min_active_rate: float = 0.02
    max_active_rate: float = 0.98
    min_action_std: float = 1e-6
    max_dominant_bucket_ratio: float = 0.98
    max_mean_abs_turnover: float = 1.0
    min_active_regime_count: int = 2
    max_active_regime_ratio: float = 0.80
    action_bucket_count: int = 5
    min_non_empty_buckets: int = 2
    monotonic_tolerance: float = 1e-12
    active_action_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.max_position_size <= 1.0:
            raise ValueError("max_position_size must be in (0, 1]")
        if not 0.0 <= self.min_active_rate <= self.max_active_rate <= 1.0:
            raise ValueError("active-rate bounds must be ordered inside [0, 1]")
        if self.min_action_std < 0.0:
            raise ValueError("min_action_std must be non-negative")
        if not 0.0 < self.max_dominant_bucket_ratio <= 1.0:
            raise ValueError("max_dominant_bucket_ratio must be in (0, 1]")
        if self.max_mean_abs_turnover < 0.0:
            raise ValueError("max_mean_abs_turnover must be non-negative")
        if self.min_active_regime_count < 1:
            raise ValueError("min_active_regime_count must be at least 1")
        if not 0.0 < self.max_active_regime_ratio <= 1.0:
            raise ValueError("max_active_regime_ratio must be in (0, 1]")
        if self.action_bucket_count < 2:
            raise ValueError("action_bucket_count must be at least 2")
        if self.min_non_empty_buckets < 2:
            raise ValueError("min_non_empty_buckets must be at least 2")
        if self.monotonic_tolerance < 0.0:
            raise ValueError("monotonic_tolerance must be non-negative")
        if self.active_action_epsilon < 0.0:
            raise ValueError("active_action_epsilon must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolicyAcceptanceFailure:
    """One Phase 1 acceptance failure."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class PolicyAcceptanceReport:
    """Aggregate Phase 1 acceptance report."""

    failures: tuple[PolicyAcceptanceFailure, ...]
    metrics: dict[str, Any]
    acceptance_criteria: dict[str, bool]

    @property
    def passed(self) -> bool:
        return not self.failures and all(self.acceptance_criteria.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
            "metrics": self.metrics,
            "acceptance_criteria": self.acceptance_criteria,
        }


def validate_policy_acceptance(
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    config: PolicyAcceptanceConfig | None = None,
    *,
    direct_pnl_optimization: bool = False,
    evaluation_scope: str = "shadow",
    split_hash: str | None = None,
    training_row_count: int | None = None,
) -> PolicyAcceptanceReport:
    """Validate Phase 1 acceptance criteria on shadow policy outputs."""

    resolved_config = config or PolicyAcceptanceConfig()
    failures: list[PolicyAcceptanceFailure] = []
    if len(examples) != len(predictions):
        failures.append(
            PolicyAcceptanceFailure(
                code="prediction_row_count_mismatch",
                message="examples and predictions must have the same length",
            )
        )
        return _failed_report(failures, resolved_config, direct_pnl_optimization)

    for example, prediction in zip(examples, predictions, strict=True):
        if (
            example.decision_ts != prediction.decision_ts
            or example.source != prediction.source
            or example.instrument_id != prediction.instrument_id
        ):
            failures.append(
                PolicyAcceptanceFailure(
                    code="prediction_alignment_mismatch",
                    message="prediction keys must match policy examples",
                )
            )
            break

    actions = np.asarray([prediction.action for prediction in predictions], dtype=np.float64)
    confidences = np.asarray(
        [prediction.confidence for prediction in predictions],
        dtype=np.float64,
    )
    shadow_net_returns = np.asarray(
        [example.shadow_net_return for example in examples],
        dtype=np.float64,
    )
    contract_valid = _prediction_contract_valid(actions, confidences, predictions, resolved_config)
    if not contract_valid:
        failures.append(
            PolicyAcceptanceFailure(
                code="prediction_contract_invalid",
                message="actions, confidences, and regime embeddings must be finite and bounded",
            )
        )

    shadow_returns = actions * shadow_net_returns
    shadow_sharpe = _sharpe(shadow_returns)
    distribution = _action_distribution(actions, resolved_config)
    regime_exposure = _regime_exposure(examples, actions, resolved_config)
    bucket_summary = _bucket_summary(actions, shadow_returns, resolved_config)
    monotonic = _bucket_means_are_monotonic(bucket_summary, resolved_config)

    criteria = {
        "shadow_sharpe_positive": shadow_sharpe > resolved_config.min_shadow_sharpe,
        "stable_action_distribution": (
            contract_valid
            and distribution["active_rate"] >= resolved_config.min_active_rate
            and distribution["active_rate"] <= resolved_config.max_active_rate
            and distribution["action_std"] >= resolved_config.min_action_std
            and distribution["dominant_bucket_ratio"]
            <= resolved_config.max_dominant_bucket_ratio
            and distribution["mean_abs_turnover"] <= resolved_config.max_mean_abs_turnover
        ),
        "monotonic_pnl_bucket_behavior": monotonic,
        "regime_action_stability": (
            regime_exposure["active_regime_count"]
            >= resolved_config.min_active_regime_count
            and regime_exposure["max_active_regime_ratio"]
            <= resolved_config.max_active_regime_ratio
        ),
        "no_direct_pnl_optimization": not direct_pnl_optimization,
    }

    if not criteria["shadow_sharpe_positive"]:
        failures.append(
            PolicyAcceptanceFailure(
                code="shadow_sharpe_non_positive",
                message="shadow Sharpe must be positive",
            )
        )
    if not criteria["stable_action_distribution"]:
        failures.append(
            PolicyAcceptanceFailure(
                code="unstable_action_distribution",
                message="policy actions are collapsed or excessively unstable",
            )
        )
    if not criteria["monotonic_pnl_bucket_behavior"]:
        failures.append(
            PolicyAcceptanceFailure(
                code="non_monotonic_pnl_buckets",
                message="mean shadow PnL must be non-decreasing by action bucket",
            )
        )
    if not criteria["regime_action_stability"]:
        failures.append(
            PolicyAcceptanceFailure(
                code="regime_exposure_concentration",
                message="active policy exposure is too concentrated in one regime",
            )
        )
    if direct_pnl_optimization:
        failures.append(
            PolicyAcceptanceFailure(
                code="direct_pnl_optimization",
                message="Phase 1 policies must not optimize PnL directly",
            )
        )

    return PolicyAcceptanceReport(
        failures=tuple(failures),
        metrics={
            "config": resolved_config.to_dict(),
            "evaluation_scope": evaluation_scope,
            "split_hash": split_hash,
            "training_row_count": training_row_count,
            "row_count": len(examples),
            "mean_shadow_return": float(np.mean(shadow_returns)) if shadow_returns.size else 0.0,
            "std_shadow_return": float(np.std(shadow_returns, ddof=1))
            if shadow_returns.size > 1
            else 0.0,
            "shadow_sharpe": shadow_sharpe,
            "mean_shadow_net_return": float(np.mean(shadow_net_returns))
            if shadow_net_returns.size
            else 0.0,
            "action_distribution": distribution,
            "regime_exposure": regime_exposure,
            "pnl_bucket_summary": bucket_summary,
        },
        acceptance_criteria=criteria,
    )


def _failed_report(
    failures: list[PolicyAcceptanceFailure],
    config: PolicyAcceptanceConfig,
    direct_pnl_optimization: bool,
) -> PolicyAcceptanceReport:
    return PolicyAcceptanceReport(
        failures=tuple(failures),
        metrics={"config": config.to_dict(), "row_count": 0},
        acceptance_criteria={
            "shadow_sharpe_positive": False,
            "stable_action_distribution": False,
            "monotonic_pnl_bucket_behavior": False,
            "regime_action_stability": False,
            "no_direct_pnl_optimization": not direct_pnl_optimization,
        },
    )


def validate_policy_shadow_split(
    policy_model: Any,
    split: PolicyTrainShadowSplit,
    config: PolicyAcceptanceConfig | None = None,
    *,
    direct_pnl_optimization: bool = False,
) -> PolicyAcceptanceReport:
    """Run acceptance on the shadow side of a temporal split."""

    resolved_config = config or PolicyAcceptanceConfig()
    provenance_failures, provenance_metrics = _validate_split_provenance(
        policy_model,
        split,
    )
    if provenance_failures:
        return _split_provenance_failed_report(
            failures=provenance_failures,
            metrics=provenance_metrics,
            config=resolved_config,
            split=split,
            direct_pnl_optimization=direct_pnl_optimization,
        )

    predictions = policy_model.predict_examples(split.shadow_examples)
    report = validate_policy_acceptance(
        split.shadow_examples,
        predictions,
        resolved_config,
        direct_pnl_optimization=direct_pnl_optimization,
        evaluation_scope="shadow",
        split_hash=split.split_hash,
        training_row_count=len(split.train_examples),
    )
    criteria = {
        **report.acceptance_criteria,
        "split_provenance_verified": provenance_metrics["passed"],
    }
    metrics = {
        **report.metrics,
        "split_provenance": provenance_metrics,
    }
    return PolicyAcceptanceReport(
        failures=(*report.failures, *provenance_failures),
        metrics=metrics,
        acceptance_criteria=criteria,
    )


def _split_provenance_failed_report(
    *,
    failures: list[PolicyAcceptanceFailure],
    metrics: dict[str, Any],
    config: PolicyAcceptanceConfig,
    split: PolicyTrainShadowSplit,
    direct_pnl_optimization: bool,
) -> PolicyAcceptanceReport:
    return PolicyAcceptanceReport(
        failures=tuple(failures),
        metrics={
            "config": config.to_dict(),
            "evaluation_scope": "shadow",
            "split_hash": split.split_hash,
            "training_row_count": len(split.train_examples),
            "row_count": len(split.shadow_examples),
            "prediction_skipped_due_to_split_provenance": True,
            "split_provenance": metrics,
        },
        acceptance_criteria={
            "shadow_sharpe_positive": False,
            "stable_action_distribution": False,
            "monotonic_pnl_bucket_behavior": False,
            "regime_action_stability": False,
            "no_direct_pnl_optimization": not direct_pnl_optimization,
            "split_provenance_verified": False,
        },
    )


def _validate_split_provenance(
    policy_model: Any,
    split: PolicyTrainShadowSplit,
) -> tuple[list[PolicyAcceptanceFailure], dict[str, Any]]:
    failures: list[PolicyAcceptanceFailure] = []
    manifest = getattr(policy_model, "training_manifest", None)
    manifest_present = isinstance(manifest, Mapping)
    split_hash_matches = False
    train_hash_matches = False
    shadow_hash_matches = False

    if not manifest_present:
        failures.append(
            PolicyAcceptanceFailure(
                code="missing_training_manifest",
                message="policy model must expose a training_manifest for shadow split validation",
            )
        )
    else:
        split_block = manifest.get("split")
        split_hash_matches = (
            isinstance(split_block, Mapping)
            and split_block.get("split_hash") == split.split_hash
        )
        train_hash_matches = manifest.get("train_dataset_hash") == split.train_dataset_hash
        shadow_hash_matches = manifest.get("shadow_dataset_hash") == split.shadow_dataset_hash

        if not split_hash_matches:
            failures.append(
                PolicyAcceptanceFailure(
                    code="split_hash_mismatch",
                    message="model training split_hash does not match the supplied split",
                )
            )
        if not train_hash_matches:
            failures.append(
                PolicyAcceptanceFailure(
                    code="train_split_mismatch",
                    message="model train_dataset_hash does not match the supplied split",
                )
            )
        if not shadow_hash_matches:
            failures.append(
                PolicyAcceptanceFailure(
                    code="shadow_split_mismatch",
                    message="model shadow_dataset_hash does not match the supplied split",
                )
            )

    passed = (
        manifest_present
        and split_hash_matches
        and train_hash_matches
        and shadow_hash_matches
    )
    return failures, {
        "passed": passed,
        "training_manifest_present": manifest_present,
        "split_hash_matches": split_hash_matches,
        "train_dataset_hash_matches": train_hash_matches,
        "shadow_dataset_hash_matches": shadow_hash_matches,
        "expected_split_hash": split.split_hash,
        "expected_train_dataset_hash": split.train_dataset_hash,
        "expected_shadow_dataset_hash": split.shadow_dataset_hash,
    }


def _prediction_contract_valid(
    actions: np.ndarray,
    confidences: np.ndarray,
    predictions: tuple[PolicyPrediction, ...],
    config: PolicyAcceptanceConfig,
) -> bool:
    if actions.size == 0:
        return False
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(confidences)):
        return False
    if np.any(actions < 0.0) or np.any(actions > config.max_position_size + 1e-12):
        return False
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        return False
    return all(
        all(math.isfinite(value) for value in prediction.regime_embedding)
        for prediction in predictions
    )


def _sharpe(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    mean = float(np.mean(returns))
    if returns.size < 2:
        return math.inf if mean > 0.0 else 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return math.inf if mean > 0.0 else 0.0
    return mean / std * math.sqrt(float(returns.size))


def _action_distribution(
    actions: np.ndarray,
    config: PolicyAcceptanceConfig,
) -> dict[str, float]:
    if actions.size == 0:
        return {
            "active_rate": 0.0,
            "max_action_rate": 0.0,
            "action_std": 0.0,
            "dominant_bucket_ratio": 1.0,
            "mean_abs_turnover": 0.0,
        }
    active_rate = float(np.mean(actions > config.active_action_epsilon))
    max_action_rate = float(
        np.mean(actions >= config.max_position_size - config.active_action_epsilon)
    )
    bucket_indices = _action_bucket_indices(actions, config)
    counts = np.bincount(bucket_indices, minlength=config.action_bucket_count)
    turnover = np.abs(np.diff(actions))
    return {
        "active_rate": active_rate,
        "max_action_rate": max_action_rate,
        "action_std": float(np.std(actions, ddof=0)),
        "dominant_bucket_ratio": float(np.max(counts) / actions.size),
        "mean_abs_turnover": float(np.mean(turnover)) if turnover.size else 0.0,
    }


def _regime_exposure(
    examples: tuple[PolicyTrainingExample, ...],
    actions: np.ndarray,
    config: PolicyAcceptanceConfig,
) -> dict[str, Any]:
    active_count_by_regime: dict[str, int] = {}
    for example, action in zip(examples, actions, strict=True):
        if action <= config.active_action_epsilon:
            continue
        active_count_by_regime[example.regime_key] = (
            active_count_by_regime.get(example.regime_key, 0) + 1
        )

    active_total_count = sum(active_count_by_regime.values())
    max_active_count = max(active_count_by_regime.values(), default=0)
    return {
        "active_regime_count": len(active_count_by_regime),
        "active_total_count": active_total_count,
        "max_active_regime_ratio": (
            float(max_active_count / active_total_count)
            if active_total_count
            else 1.0
        ),
        "active_count_by_regime": dict(sorted(active_count_by_regime.items())),
    }


def _bucket_summary(
    actions: np.ndarray,
    shadow_returns: np.ndarray,
    config: PolicyAcceptanceConfig,
) -> list[dict[str, float | int]]:
    if actions.size == 0:
        return []
    bucket_indices = _action_bucket_indices(actions, config)
    summary: list[dict[str, float | int]] = []
    for bucket_index in sorted(set(bucket_indices.tolist())):
        mask = bucket_indices == bucket_index
        bucket_actions = actions[mask]
        bucket_returns = shadow_returns[mask]
        summary.append(
            {
                "bucket": int(bucket_index),
                "row_count": int(mask.sum()),
                "mean_action": float(np.mean(bucket_actions)),
                "mean_shadow_return": float(np.mean(bucket_returns)),
            }
        )
    return summary


def _bucket_means_are_monotonic(
    bucket_summary: list[dict[str, float | int]],
    config: PolicyAcceptanceConfig,
) -> bool:
    if len(bucket_summary) < config.min_non_empty_buckets:
        return False
    means = [float(row["mean_shadow_return"]) for row in bucket_summary]
    return all(
        current + config.monotonic_tolerance >= previous
        for previous, current in zip(means, means[1:], strict=False)
    )


def _action_bucket_indices(
    actions: np.ndarray,
    config: PolicyAcceptanceConfig,
) -> np.ndarray:
    scaled = actions / config.max_position_size * config.action_bucket_count
    return np.clip(np.floor(scaled), 0, config.action_bucket_count - 1).astype(int)
