"""Dataset loader for Polymarket policy training from Phase 2 corpus outputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.training.contracts import (
    PolymarketPolicyDataset,
    PolymarketPolicyExample,
    PolymarketPolicyTrainingConfig,
    compact_safety_fields,
    stable_hash,
)

TARGET_LABEL_ACTION = "BUY_UP_HOLD_TO_SETTLEMENT"
LABEL_SCHEMA = {
    "target": "resolved_up_probability",
    "source_action": TARGET_LABEL_ACTION,
    "mapping": {
        "UP": 1.0,
        "DOWN": 0.0,
        "UNKNOWN_50_50": 0.5,
    },
}


def load_polymarket_policy_dataset(
    config: PolymarketPolicyTrainingConfig,
) -> PolymarketPolicyDataset:
    """Load training examples from a deterministic Phase 2 corpus bundle."""

    corpus_dir = config.corpus_dir.expanduser().resolve()
    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    manifest = _read_json(manifest_path)
    _assert_corpus_safety(manifest)
    market_metadata = {
        row["market_id"]: row
        for row in _read_jsonl(corpus_dir / "polymarket_market_metadata.jsonl")
    }
    resolution_events = {
        row["market_id"]: row
        for row in _read_jsonl(corpus_dir / "polymarket_resolution_events.jsonl")
    }
    target_labels = {
        (row["market_id"], row["decision_ts"]): row
        for row in _read_jsonl(corpus_dir / "polymarket_label_rows.jsonl")
        if row["action"] == TARGET_LABEL_ACTION
    }
    examples = []
    for row in _read_jsonl(corpus_dir / "polymarket_feature_rows.jsonl"):
        _assert_corpus_safety(row)
        label = target_labels.get((row["market_id"], row["decision_ts"]))
        if label is None:
            continue
        _assert_corpus_safety(label)
        examples.append(
            PolymarketPolicyExample(
                market_id=row["market_id"],
                condition_id=row["condition_id"],
                slug=row["slug"],
                market_family=row["market_family"],
                horizon_ms=int(row["horizon_ms"]),
                decision_ts=int(row["decision_ts"]),
                feature_cutoff_ts=int(row["feature_cutoff_ts"]),
                max_input_ts=int(row["max_input_ts"]),
                features=_model_features(row["features"], row["market_family"]),
                target_up_probability=_target_up_probability(label["resolved_outcome"]),
                resolved_outcome=label["resolved_outcome"],
                resolution_status=label["resolution_status"],
            )
        )
    if not examples:
        raise ValueError("no policy examples could be loaded from corpus")

    ordered_examples = tuple(sorted(examples, key=lambda item: (item.decision_ts, item.market_id)))
    feature_columns = tuple(sorted({name for example in ordered_examples for name in example.features}))
    feature_schema_hash = stable_hash(
        {
            "feature_columns": list(feature_columns),
            "market_families": sorted(BTC_UPDOWN_MARKET_HORIZONS_MS),
        }
    )
    label_schema_hash = stable_hash(LABEL_SCHEMA)
    training_corpus_hash = _sha256_file(manifest_path)
    dataset_hash = stable_hash(
        {
            "feature_schema_hash": feature_schema_hash,
            "label_schema_hash": label_schema_hash,
            "training_corpus_hash": training_corpus_hash,
            "examples": [example.to_dict() for example in ordered_examples],
        }
    )
    train, validation, shadow = _split_examples(ordered_examples, config)
    return PolymarketPolicyDataset(
        examples=ordered_examples,
        feature_columns=feature_columns,
        feature_schema_hash=feature_schema_hash,
        label_schema_hash=label_schema_hash,
        training_corpus_hash=training_corpus_hash,
        dataset_hash=dataset_hash,
        corpus_manifest=manifest,
        market_metadata=market_metadata,
        resolution_events=resolution_events,
        train_examples=train,
        validation_examples=validation,
        shadow_examples=shadow,
    )


def dataset_profile(dataset: PolymarketPolicyDataset) -> dict[str, Any]:
    """Return a deterministic profile for audit and manifest evidence."""

    return {
        "schema_version": "bigan-v8-polymarket-policy-dataset-profile-v1",
        "row_count": len(dataset.examples),
        "train_row_count": len(dataset.train_examples),
        "validation_row_count": len(dataset.validation_examples),
        "shadow_row_count": len(dataset.shadow_examples),
        "market_count": len(dataset.market_metadata),
        "market_family_counts": _family_counts(dataset.examples),
        "feature_columns": list(dataset.feature_columns),
        "feature_schema_hash": dataset.feature_schema_hash,
        "label_schema_hash": dataset.label_schema_hash,
        "training_corpus_hash": dataset.training_corpus_hash,
        "dataset_hash": dataset.dataset_hash,
        **compact_safety_fields(),
    }


def _model_features(raw_features: dict[str, Any], market_family: str) -> dict[str, float]:
    features: dict[str, float] = {}
    for name, value in raw_features.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            numeric = float(value)
            if math.isfinite(numeric):
                features[name] = numeric
    for family in sorted(BTC_UPDOWN_MARKET_HORIZONS_MS):
        features[f"family_{family}"] = 1.0 if family == market_family else 0.0
    return features


def _target_up_probability(resolved_outcome: str) -> float:
    if resolved_outcome == "UP":
        return 1.0
    if resolved_outcome == "DOWN":
        return 0.0
    if resolved_outcome == "UNKNOWN_50_50":
        return 0.5
    raise ValueError(f"unsupported resolved_outcome: {resolved_outcome}")


def _split_examples(
    examples: tuple[PolymarketPolicyExample, ...],
    config: PolymarketPolicyTrainingConfig,
) -> tuple[
    tuple[PolymarketPolicyExample, ...],
    tuple[PolymarketPolicyExample, ...],
    tuple[PolymarketPolicyExample, ...],
]:
    if len(examples) < 4:
        raise ValueError("policy dataset requires at least four examples")
    train_count = max(1, int(len(examples) * config.train_fraction))
    validation_count = max(1, int(len(examples) * config.validation_fraction))
    if train_count + validation_count >= len(examples):
        validation_count = 1
        train_count = len(examples) - 2
    train = examples[:train_count]
    validation = examples[train_count : train_count + validation_count]
    shadow = examples[train_count + validation_count :]
    if not train or not validation or not shadow:
        raise ValueError("train, validation, and shadow splits must be non-empty")
    if train[-1].decision_ts > validation[0].decision_ts:
        raise ValueError("validation split must follow train split")
    if validation[-1].decision_ts > shadow[0].decision_ts:
        raise ValueError("shadow split must follow validation split")
    return train, validation, shadow


def _family_counts(examples: tuple[PolymarketPolicyExample, ...]) -> dict[str, int]:
    counts = dict.fromkeys(BTC_UPDOWN_MARKET_HORIZONS_MS, 0)
    for example in examples:
        counts[example.market_family] = counts.get(example.market_family, 0) + 1
    return {family: count for family, count in counts.items() if count > 0}


def _assert_corpus_safety(payload: dict[str, Any]) -> None:
    for field_name, expected in compact_safety_fields().items():
        if payload.get(field_name) is not expected:
            raise ValueError(f"corpus payload violates {field_name}={expected}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
