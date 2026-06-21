"""Phase 1 dataset adapter from validated Phase 0 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from bigan.v8.phase0.artifacts import assert_phase0_artifact_ready
from bigan.v8.phase0.contracts import COST_COLUMNS, FEATURE_COLUMNS, FeatureVector, Label
from bigan.v8.phase0.pipeline import Phase0Dataset
from bigan.v8.phase1.contracts import (
    PHASE1_POLICY_VERSION,
    PolicyDataset,
    PolicyDatasetConfig,
    PolicyTrainingExample,
    PolicyTrainShadowSplit,
)

FORBIDDEN_POLICY_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        *COST_COLUMNS,
        "gross_return",
        "is_positive",
        "label_ts",
        "horizon_ms",
        "entry_price",
        "exit_price",
        "side",
    }
)


def build_policy_dataset_from_phase0(
    phase0_dataset: Phase0Dataset,
    config: PolicyDatasetConfig | None = None,
) -> PolicyDataset:
    """Build a deterministic pure-policy dataset from a Phase 0 dataset.

    This adapter is deliberately narrow: it consumes only Phase 0 causal
    feature rows plus cost-aware labels, and it first runs the Phase 0 artifact
    gate so unsafe manifests cannot flow into policy learning.
    """

    resolved_config = config or PolicyDatasetConfig()
    contract = assert_phase0_artifact_ready(phase0_dataset.manifest)
    return build_policy_dataset(
        features=phase0_dataset.features,
        labels=phase0_dataset.labels,
        phase0_dataset_hash=contract.dataset_hash,
        phase0_dataset_version=contract.dataset_version,
        config=resolved_config,
    )


def build_policy_dataset(
    *,
    features: Sequence[FeatureVector],
    labels: Sequence[Label],
    phase0_dataset_hash: str,
    phase0_dataset_version: str,
    config: PolicyDatasetConfig | None = None,
) -> PolicyDataset:
    """Build Phase 1 examples from already validated Phase 0 rows."""

    resolved_config = config or PolicyDatasetConfig()
    if not features:
        raise ValueError("features must not be empty")
    if not labels:
        raise ValueError("labels must not be empty")

    feature_columns = _resolve_feature_columns(features, resolved_config)
    _validate_feature_columns(feature_columns)
    _assert_feature_columns_available(features, feature_columns)
    _assert_feature_causality(features)

    selected_horizon = resolved_config.horizon_ms or min(label.horizon_ms for label in labels)
    label_by_key = _select_labels(labels, selected_horizon)
    examples: list[PolicyTrainingExample] = []
    for feature in sorted(features, key=lambda row: (row.source, row.instrument_id, row.decision_ts)):
        label = label_by_key.get((feature.source, feature.instrument_id, feature.decision_ts))
        if label is None:
            continue
        example_features = {
            column: _coerce_feature_value(feature.features.get(column))
            for column in feature_columns
        }
        examples.append(
            PolicyTrainingExample(
                decision_ts=feature.decision_ts,
                source=feature.source,
                instrument_id=feature.instrument_id,
                features=example_features,
                target_label=_policy_target_label(label.net_return, resolved_config),
                shadow_net_return=label.net_return,
                horizon_ms=label.horizon_ms,
                regime_key=_regime_key(feature),
            )
        )

    if not examples:
        raise ValueError(
            "no Phase 1 examples could be matched between features and labels "
            f"for horizon_ms={selected_horizon}"
        )

    policy_hash = policy_dataset_hash(
        examples=tuple(examples),
        feature_columns=feature_columns,
        phase0_dataset_hash=phase0_dataset_hash,
        phase0_dataset_version=phase0_dataset_version,
        config=resolved_config,
    )
    return PolicyDataset(
        examples=tuple(examples),
        feature_columns=feature_columns,
        policy_dataset_hash=policy_hash,
        phase0_dataset_hash=phase0_dataset_hash,
        phase0_dataset_version=phase0_dataset_version,
        config=resolved_config,
    )


def build_temporal_policy_split(
    dataset: PolicyDataset,
    *,
    split_ts: int | None = None,
    train_fraction: float = 0.70,
) -> PolicyTrainShadowSplit:
    """Build a strict temporal train/shadow split for out-of-sample acceptance."""

    ordered = tuple(sorted(dataset.examples, key=lambda row: row.decision_ts))
    if split_ts is None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        split_index = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
        split_ts = ordered[split_index].decision_ts

    train_examples = tuple(example for example in ordered if example.decision_ts < split_ts)
    shadow_examples = tuple(example for example in ordered if example.decision_ts >= split_ts)
    train_hash = _examples_hash(
        examples=train_examples,
        feature_columns=dataset.feature_columns,
        namespace="train",
    )
    shadow_hash = _examples_hash(
        examples=shadow_examples,
        feature_columns=dataset.feature_columns,
        namespace="shadow",
    )
    split_hash = _split_hash(
        train_dataset_hash=train_hash,
        shadow_dataset_hash=shadow_hash,
        split_ts=split_ts,
        policy_dataset_hash=dataset.policy_dataset_hash,
    )
    return PolicyTrainShadowSplit(
        train_examples=train_examples,
        shadow_examples=shadow_examples,
        split_ts=split_ts,
        split_hash=split_hash,
        train_dataset_hash=train_hash,
        shadow_dataset_hash=shadow_hash,
    )


def policy_dataset_hash(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    feature_columns: tuple[str, ...],
    phase0_dataset_hash: str,
    phase0_dataset_version: str,
    config: PolicyDatasetConfig,
) -> str:
    """Return the deterministic hash for a Phase 1 policy dataset."""

    payload = {
        "phase1_policy_version": PHASE1_POLICY_VERSION,
        "phase0_dataset_hash": phase0_dataset_hash,
        "phase0_dataset_version": phase0_dataset_version,
        "feature_columns": list(feature_columns),
        "config": config.to_dict(),
        "examples": [example.to_dict() for example in examples],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _policy_target_label(net_return: float, config: PolicyDatasetConfig) -> float:
    if config.target_encoding == "binary_positive_net_return_threshold":
        return 1.0 if net_return > config.positive_return_threshold else 0.0
    return float(_rank_quality_bucket(net_return, config.rank_quality_bucket_edges))


def _rank_quality_bucket(net_return: float, edges: tuple[float, ...]) -> int:
    bucket = 0
    for edge in edges:
        if net_return <= edge:
            return bucket
        bucket += 1
    return bucket


def _examples_hash(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    feature_columns: tuple[str, ...],
    namespace: str,
) -> str:
    payload = {
        "phase1_policy_version": PHASE1_POLICY_VERSION,
        "namespace": namespace,
        "feature_columns": list(feature_columns),
        "examples": [example.to_dict() for example in examples],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _split_hash(
    *,
    train_dataset_hash: str,
    shadow_dataset_hash: str,
    split_ts: int,
    policy_dataset_hash: str,
) -> str:
    payload = {
        "phase1_policy_version": PHASE1_POLICY_VERSION,
        "policy_dataset_hash": policy_dataset_hash,
        "split_ts": split_ts,
        "train_dataset_hash": train_dataset_hash,
        "shadow_dataset_hash": shadow_dataset_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _resolve_feature_columns(
    features: Sequence[FeatureVector],
    config: PolicyDatasetConfig,
) -> tuple[str, ...]:
    if config.feature_columns:
        return tuple(config.feature_columns)

    usable_columns = {
        column
        for feature in features
        for column, value in feature.features.items()
        if _coerce_feature_value(value) is not None
    }
    ordered = [column for column in FEATURE_COLUMNS if column in usable_columns]
    extras = sorted(usable_columns - set(FEATURE_COLUMNS))
    columns = tuple(ordered + extras)
    if not columns:
        raise ValueError("no usable numeric feature columns were found")
    return columns


def _validate_feature_columns(feature_columns: tuple[str, ...]) -> None:
    duplicates = sorted(
        {column for column in feature_columns if feature_columns.count(column) > 1}
    )
    if duplicates:
        raise ValueError("duplicate feature columns: " + ", ".join(duplicates))

    forbidden = sorted(set(feature_columns) & FORBIDDEN_POLICY_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(
            "policy features must not include label/cost columns: "
            + ", ".join(forbidden)
        )


def _assert_feature_columns_available(
    features: Sequence[FeatureVector],
    feature_columns: tuple[str, ...],
) -> None:
    empty_columns = [
        column
        for column in feature_columns
        if not any(_coerce_feature_value(feature.features.get(column)) is not None for feature in features)
    ]
    if empty_columns:
        raise ValueError(
            "feature columns have no usable values: " + ", ".join(empty_columns)
        )


def _assert_feature_causality(features: Sequence[FeatureVector]) -> None:
    offenders = [
        feature
        for feature in features
        if feature.max_input_ts > feature.decision_ts
        or feature.feature_cutoff_ts > feature.decision_ts
        or any(
            provenance.input_end_ts > feature.decision_ts
            or provenance.available_at_ts > feature.decision_ts
            for provenance in feature.provenance.values()
        )
    ]
    if offenders:
        raise ValueError(
            "feature causality violation in Phase 1 dataset adapter: "
            f"{len(offenders)} rows use unavailable inputs"
        )


def _select_labels(
    labels: Sequence[Label],
    horizon_ms: int,
) -> dict[tuple[str, str, int], Label]:
    selected: dict[tuple[str, str, int], Label] = {}
    duplicate_count = 0
    for label in labels:
        if label.horizon_ms != horizon_ms:
            continue
        key = (label.source, label.instrument_id, label.decision_ts)
        if key in selected:
            duplicate_count += 1
            continue
        selected[key] = label
    if duplicate_count:
        raise ValueError(
            f"duplicate labels detected for horizon_ms={horizon_ms}: {duplicate_count}"
        )
    if not selected:
        raise ValueError(f"no labels found for horizon_ms={horizon_ms}")
    return selected


def _coerce_feature_value(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature values must be numeric or None, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return value


def _regime_key(feature: FeatureVector) -> str:
    volatility = _feature_float(feature.features.get("volatility_5m"))
    spread_bps = _feature_float(feature.features.get("spread_bps"))
    liquidity_depth = _feature_float(feature.features.get("liquidity_depth"))
    return "|".join(
        (
            feature.source,
            feature.instrument_id,
            f"vol={_bucket(volatility, (0.005, 0.015))}",
            f"spread={_bucket(spread_bps, (2.0, 5.0))}",
            f"liq={_bucket(liquidity_depth, (50.0, 150.0))}",
        )
    )


def _feature_float(value: float | int | None) -> float:
    coerced = _coerce_feature_value(value)
    return 0.0 if coerced is None else float(coerced)


def _bucket(value: float, edges: tuple[float, ...]) -> str:
    bucket = 0
    for edge in edges:
        if value <= edge:
            return str(bucket)
        bucket += 1
    return str(bucket)
