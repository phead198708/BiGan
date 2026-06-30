"""Dataset loader for Polymarket policy training from Phase 2 corpus outputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.corpus import (
    BTC_UPDOWN_MARKET_HORIZONS_MS,
    POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    AUXILIARY_OUTCOME_TARGET,
    PRIMARY_POLICY_TARGET_ACTION_VALUE,
    PolymarketPolicyDataset,
    PolymarketPolicyExample,
    PolymarketPolicyTrainingConfig,
    compact_safety_fields,
    stable_hash,
)

TARGET_LABEL_ACTION = "BUY_UP_HOLD_TO_SETTLEMENT"
ACTION_VALUE_TARGET_FIELD = "total_net_pnl_per_notional"
LABEL_SCHEMA = {
    "primary_target": PRIMARY_POLICY_TARGET_ACTION_VALUE,
    "auxiliary_target": AUXILIARY_OUTCOME_TARGET,
    "action_value_target_field": ACTION_VALUE_TARGET_FIELD,
    "fixed_notional_target_used": True,
    "sell_before_close_label_schema_version": (
        POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION
    ),
    "sell_before_close_fixed_terminal_bid_only_labels_allowed": False,
    "sell_before_close_target_field": (
        "realized_executable_sell_before_close_return"
    ),
    "sell_before_close_theoretical_gap_field": "execution_gap_return",
    "action_labels": list(ACTION_VALUE_LABEL_ACTIONS),
    "source_actions": list(ACTION_VALUE_LABEL_ACTIONS),
    "outcome_probability_mapping": {
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
    labels_by_key = _labels_by_decision_state(corpus_dir / "polymarket_label_rows.jsonl")
    examples = []
    for row in _read_jsonl(corpus_dir / "polymarket_feature_rows.jsonl"):
        _assert_corpus_safety(row)
        labels = labels_by_key.get((row["market_id"], int(row["decision_ts"])))
        if labels is None:
            raise ValueError(
                "feature row is missing action labels for "
                f"{row['market_id']} at decision_ts={row['decision_ts']}"
            )
        missing_actions = set(ACTION_VALUE_LABEL_ACTIONS) - set(labels)
        if missing_actions:
            raise ValueError(
                "policy action labels missing required actions for "
                f"{row['market_id']} at decision_ts={row['decision_ts']}: "
                + ", ".join(sorted(missing_actions))
            )
        label = labels[TARGET_LABEL_ACTION]
        _assert_action_value_target_fields(
            labels=labels,
            market_id=row["market_id"],
            decision_ts=int(row["decision_ts"]),
        )
        action_return_targets = {
            action: float(labels[action][ACTION_VALUE_TARGET_FIELD])
            for action in ACTION_VALUE_LABEL_ACTIONS
        }
        realized_trade_return_targets = {
            action: float(labels[action]["realized_trade_return"])
            for action in ACTION_VALUE_LABEL_ACTIONS
        }
        settlement_return_targets = {
            action: float(labels[action]["settlement_return"])
            for action in ACTION_VALUE_LABEL_ACTIONS
        }
        sell_before_close_execution_class_targets = {
            action: str(labels[action].get("sell_before_close_execution_class"))
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        sell_before_close_theoretical_return_targets = {
            action: float(labels[action].get("theoretical_terminal_bid_return", 0.0))
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        sell_before_close_executable_return_targets = {
            action: float(
                labels[action].get(
                    "realized_executable_sell_before_close_return",
                    0.0,
                )
            )
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        sell_before_close_execution_gap_targets = {
            action: float(labels[action].get("execution_gap_return", 0.0))
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        sell_before_close_queue_fill_probability_targets = {
            action: float(labels[action].get("queue_fill_probability_estimate", 0.0))
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        sell_before_close_exit_bid_targets = {
            action: float(labels[action].get("exit_bid", 0.0))
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        sell_before_close_executable_liquidity_notional_targets = {
            action: float(labels[action].get("executable_liquidity_notional", 0.0))
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        sell_before_close_exit_path_targets = {
            action: dict(labels[action].get("sell_before_close_exit_path") or {})
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        sell_before_close_label_uses_executable_exit_path_targets = {
            action: bool(labels[action].get("label_uses_executable_exit_path", False))
            for action in ACTION_VALUE_LABEL_ACTIONS
            if action.endswith("SELL_BEFORE_CLOSE")
        }
        action_is_positive_targets = {
            action: bool(labels[action]["is_positive"])
            for action in ACTION_VALUE_LABEL_ACTIONS
        }
        best_action, best_return, second_best_return, best_margin = _best_action(
            action_return_targets
        )
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
                action_return_targets=action_return_targets,
                realized_trade_return_targets=realized_trade_return_targets,
                settlement_return_targets=settlement_return_targets,
                action_is_positive_targets=action_is_positive_targets,
                sell_before_close_execution_class_targets=(
                    sell_before_close_execution_class_targets
                ),
                sell_before_close_theoretical_return_targets=(
                    sell_before_close_theoretical_return_targets
                ),
                sell_before_close_executable_return_targets=(
                    sell_before_close_executable_return_targets
                ),
                sell_before_close_execution_gap_targets=(
                    sell_before_close_execution_gap_targets
                ),
                sell_before_close_queue_fill_probability_targets=(
                    sell_before_close_queue_fill_probability_targets
                ),
                sell_before_close_exit_bid_targets=(
                    sell_before_close_exit_bid_targets
                ),
                sell_before_close_executable_liquidity_notional_targets=(
                    sell_before_close_executable_liquidity_notional_targets
                ),
                sell_before_close_exit_path_targets=(
                    sell_before_close_exit_path_targets
                ),
                sell_before_close_label_uses_executable_exit_path_targets=(
                    sell_before_close_label_uses_executable_exit_path_targets
                ),
                best_policy_action=best_action,
                best_action_expected_return=best_return,
                second_best_action_expected_return=second_best_return,
                best_action_margin=best_margin,
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
    train, validation, shadow, split_metadata = _split_examples(ordered_examples, config)
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
        split_metadata=split_metadata,
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
        "primary_policy_target": PRIMARY_POLICY_TARGET_ACTION_VALUE,
        "auxiliary_outcome_target": AUXILIARY_OUTCOME_TARGET,
        "action_value_target_field": ACTION_VALUE_TARGET_FIELD,
        "fixed_notional_target_used": True,
        "action_value_head_enabled": True,
        "outcome_probability_head_enabled": True,
        "action_label_coverage_by_action": _action_label_coverage(dataset.examples),
        "best_policy_action_counts": _best_policy_action_counts(dataset.examples),
        "sell_before_close_label_schema_version": dataset.corpus_manifest.get(
            "sell_before_close_label_schema_version"
        ),
        "sell_before_close_fixed_terminal_bid_only_labels_allowed": False,
        "sell_before_close_label_gate_passed": dataset.corpus_manifest.get(
            "sell_before_close_label_gate_passed"
        ),
        "sell_before_close_execution_class_counts": dataset.corpus_manifest.get(
            "sell_before_close_execution_class_counts",
            {},
        ),
        "feature_columns": list(dataset.feature_columns),
        "feature_schema_hash": dataset.feature_schema_hash,
        "label_schema_hash": dataset.label_schema_hash,
        "training_corpus_hash": dataset.training_corpus_hash,
        "dataset_hash": dataset.dataset_hash,
        **dataset.split_metadata,
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


def _labels_by_decision_state(path: Path) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        _assert_corpus_safety(row)
        action = str(row["action"])
        if action not in ACTION_VALUE_LABEL_ACTIONS:
            continue
        _assert_sell_before_close_label_redesign(row)
        key = (str(row["market_id"]), int(row["decision_ts"]))
        if action in grouped.setdefault(key, {}):
            raise ValueError(
                f"duplicate policy label action for {key[0]} at decision_ts={key[1]}: {action}"
            )
        grouped[key][action] = row
    if not grouped:
        raise ValueError("no action-value policy labels found")
    return grouped


def _assert_action_value_target_fields(
    *,
    labels: dict[str, dict[str, Any]],
    market_id: str,
    decision_ts: int,
) -> None:
    missing = sorted(
        action
        for action in ACTION_VALUE_LABEL_ACTIONS
        if ACTION_VALUE_TARGET_FIELD not in labels[action]
    )
    if missing:
        raise ValueError(
            "policy action labels missing "
            f"{ACTION_VALUE_TARGET_FIELD} for {market_id} "
            f"at decision_ts={decision_ts}: "
            + ", ".join(missing)
        )


def _assert_sell_before_close_label_redesign(row: dict[str, Any]) -> None:
    action = str(row.get("action") or "")
    if not action.endswith("SELL_BEFORE_CLOSE"):
        return
    if row.get("sell_before_close_label_schema_version") != (
        POLYMARKET_SELL_BEFORE_CLOSE_LABEL_SCHEMA_VERSION
    ):
        raise ValueError(
            "sell-before-close label missing executable exit schema version"
        )
    execution_class = str(row.get("sell_before_close_execution_class") or "")
    if execution_class not in {
        "realizable_sell_before_close",
        "theoretical_sell_before_close",
        "sparse_theoretical_sell_before_close",
        "non_executable_sell_before_close",
    }:
        raise ValueError("sell-before-close label missing execution class")
    exit_path = row.get("sell_before_close_exit_path")
    if not isinstance(exit_path, dict):
        raise ValueError("sell-before-close label missing exit path")
    if exit_path.get("label_source") == "fixed_terminal_bid_only":
        raise ValueError("fixed terminal bid-only sell-before-close label rejected")
    if "queue_fill_probability_estimate" not in row:
        raise ValueError("sell-before-close label missing queue/fill probability")
    if (
        execution_class == "realizable_sell_before_close"
        and row.get("label_uses_executable_exit_path") is not True
    ):
        raise ValueError("realizable sell-before-close label must use executable path")
    if (
        float(row.get("theoretical_terminal_bid_return", 0.0)) > 0.0
        and execution_class != "realizable_sell_before_close"
        and row.get("label_uses_executable_exit_path") is True
    ):
        raise ValueError("theoretical sell-before-close label cannot use fake exit")


def _best_action(action_returns: dict[str, float]) -> tuple[str, float, float, float]:
    ranked = sorted(
        ((action, float(value)) for action, value in action_returns.items()),
        key=lambda item: (-item[1], item[0]),
    )
    best_action, best_return = ranked[0]
    second_best_return = ranked[1][1] if len(ranked) > 1 else best_return
    return best_action, best_return, second_best_return, best_return - second_best_return


def _split_examples(
    examples: tuple[PolymarketPolicyExample, ...],
    config: PolymarketPolicyTrainingConfig,
) -> tuple[
    tuple[PolymarketPolicyExample, ...],
    tuple[PolymarketPolicyExample, ...],
    tuple[PolymarketPolicyExample, ...],
    dict[str, Any],
]:
    decision_times = tuple(sorted({example.decision_ts for example in examples}))
    if len(decision_times) < 3:
        raise ValueError("policy dataset requires at least three unique decision_ts values")
    train_time_count = max(1, int(len(decision_times) * config.train_fraction))
    train_time_count = min(train_time_count, len(decision_times) - 2)
    validation_time_count = max(1, int(len(decision_times) * config.validation_fraction))
    validation_time_count = min(
        validation_time_count,
        len(decision_times) - train_time_count - 1,
    )
    train_times = set(decision_times[:train_time_count])
    validation_times = set(
        decision_times[train_time_count : train_time_count + validation_time_count]
    )
    shadow_times = set(decision_times[train_time_count + validation_time_count :])
    train = tuple(example for example in examples if example.decision_ts in train_times)
    validation = tuple(
        example for example in examples if example.decision_ts in validation_times
    )
    shadow = tuple(example for example in examples if example.decision_ts in shadow_times)
    if not train or not validation or not shadow:
        raise ValueError("train, validation, and shadow splits must be non-empty")
    split_metadata = _split_metadata(
        decision_times=decision_times,
        train=train,
        validation=validation,
        shadow=shadow,
    )
    if split_metadata["train_max_ts"] >= split_metadata["validation_min_ts"]:
        raise ValueError("validation split must strictly follow train split")
    if split_metadata["validation_max_ts"] >= split_metadata["shadow_min_ts"]:
        raise ValueError("shadow split must strictly follow validation split")
    return train, validation, shadow, split_metadata


def _split_metadata(
    *,
    decision_times: tuple[int, ...],
    train: tuple[PolymarketPolicyExample, ...],
    validation: tuple[PolymarketPolicyExample, ...],
    shadow: tuple[PolymarketPolicyExample, ...],
) -> dict[str, Any]:
    train_ts = {example.decision_ts for example in train}
    validation_ts = {example.decision_ts for example in validation}
    shadow_ts = {example.decision_ts for example in shadow}
    if train_ts & validation_ts or validation_ts & shadow_ts or train_ts & shadow_ts:
        raise ValueError("decision_ts values must not cross split boundaries")
    return {
        "split_strategy": "unique_decision_ts_temporal",
        "split_ordering_key": "decision_ts",
        "unique_decision_ts_count": len(decision_times),
        "train_decision_ts_count": len(train_ts),
        "validation_decision_ts_count": len(validation_ts),
        "shadow_decision_ts_count": len(shadow_ts),
        "strict_temporal_separation": True,
        "train_min_ts": min(train_ts),
        "train_max_ts": max(train_ts),
        "validation_min_ts": min(validation_ts),
        "validation_max_ts": max(validation_ts),
        "shadow_min_ts": min(shadow_ts),
        "shadow_max_ts": max(shadow_ts),
    }


def _family_counts(examples: tuple[PolymarketPolicyExample, ...]) -> dict[str, int]:
    counts = dict.fromkeys(BTC_UPDOWN_MARKET_HORIZONS_MS, 0)
    for example in examples:
        counts[example.market_family] = counts.get(example.market_family, 0) + 1
    return {family: count for family, count in counts.items() if count > 0}


def _action_label_coverage(examples: tuple[PolymarketPolicyExample, ...]) -> dict[str, int]:
    counts = dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, 0)
    for example in examples:
        for action in ACTION_VALUE_LABEL_ACTIONS:
            if action in example.action_return_targets:
                counts[action] += 1
    return counts


def _best_policy_action_counts(
    examples: tuple[PolymarketPolicyExample, ...],
) -> dict[str, int]:
    counts = dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, 0)
    for example in examples:
        counts[example.best_policy_action] = counts.get(example.best_policy_action, 0) + 1
    return {action: count for action, count in counts.items() if count > 0}


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
