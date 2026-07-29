"""Governed model-layer training for the BTC 15m development lane."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import (
    SAFETY,
    atomic_write_json,
    load_jsonl,
    sha256_file,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

PREREGISTRATION_SCHEMA_VERSION = "bigan-challenge-model-15m-training-slot-v1"
TRAINING_REPORT_SCHEMA_VERSION = "bigan-challenge-model-15m-training-report-v1"
TRAINING_MANIFEST_SCHEMA_VERSION = "bigan-challenge-model-15m-training-manifest-v1"
DATASET_MANIFEST_SCHEMA_VERSION = "bigan-challenge-model-15m-training-dataset-v1"
FINALIZED_INDEX_SCHEMA_VERSION = "bigan-challenge-model-development-lane-finalized-index-v1"
TARGET_FIELD = "total_net_pnl_per_notional"
TARGET_POLICY = "HOLD_TO_SETTLEMENT"
SIDES = ("UP", "DOWN")
EXPECTED_DECISIONS_PER_MARKET = 2
REPO_ROOT = Path(__file__).resolve().parents[4]

SIDE_RAW_SUFFIXES = (
    "ask",
    "bid",
    "mid",
    "ask_size",
    "bid_size",
    "spread_bps",
    "executable_ask_notional",
    "executable_bid_notional",
    "liquidity_depth",
    "book_staleness_ms",
    "book_update_lag_ms",
    "queue_fill_probability_proxy",
    "recent_bid_depth_volatility_1m",
    "recent_book_update_count_1m",
    "recent_spread_stability_1m",
    "recent_trade_volume",
)

BASE_FEATURE_NAMES = (
    *(f"selected_{suffix}" for suffix in SIDE_RAW_SUFFIXES),
    *(f"opposite_{suffix}" for suffix in SIDE_RAW_SUFFIXES),
    "selected_minus_opposite_mid",
    "selected_minus_opposite_spread_bps",
    "selected_minus_opposite_liquidity_depth",
    "selected_minus_opposite_recent_trade_volume",
    "paired_ask_sum",
    "paired_bid_sum",
    "paired_mid_sum",
    "combined_spread_bps",
    "signed_chainlink_reference_distance",
    "signed_btc_mid_to_chainlink_relative_distance",
    "signed_btc_return_10s",
    "signed_btc_return_30s",
    "signed_btc_return_1m",
    "signed_btc_return_5m",
    "signed_btc_return_15m",
    "btc_volatility_1m",
    "btc_volatility_5m",
    "btc_volatility_15m",
    "market_progress_fraction",
    "time_remaining_fraction",
    "provider_health_score",
    "book_snapshot_pair_ts_delta_ms",
)

GLOBAL_RAW_DEPENDENCIES = {
    "paired_ask_sum": ("up_down_ask_sum",),
    "paired_bid_sum": ("up_down_bid_sum",),
    "paired_mid_sum": ("up_down_mid_sum",),
    "combined_spread_bps": ("combined_spread_bps",),
    "signed_chainlink_reference_distance": (
        "chainlink_reference_distance_at_decision",
    ),
    "signed_btc_mid_to_chainlink_relative_distance": (
        "btc_mid_price",
        "chainlink_price_at_decision",
    ),
    "signed_btc_return_10s": ("btc_return_10s",),
    "signed_btc_return_30s": ("btc_return_30s",),
    "signed_btc_return_1m": ("btc_return_1m",),
    "signed_btc_return_5m": ("btc_return_5m",),
    "signed_btc_return_15m": ("btc_return_15m",),
    "btc_volatility_1m": ("btc_volatility_1m",),
    "btc_volatility_5m": ("btc_volatility_5m",),
    "btc_volatility_15m": ("btc_volatility_15m",),
    "market_progress_fraction": ("market_age_seconds", "horizon_ms"),
    "time_remaining_fraction": ("time_to_close_seconds", "horizon_ms"),
    "provider_health_score": ("provider_health_score",),
    "book_snapshot_pair_ts_delta_ms": ("book_snapshot_pair_ts_delta_ms",),
}

NORMALIZED_ARTIFACT_FILES = {
    "feature_rows": "polymarket_feature_rows.jsonl",
    "label_rows": "polymarket_label_rows.jsonl",
    "market_metadata": "polymarket_market_metadata.jsonl",
    "resolution_events": "polymarket_resolution_events.jsonl",
}


def validate_training_slot_preregistration(payload: Mapping[str, Any]) -> None:
    """Validate one fixed, development-only model-layer candidate."""

    blockers: list[str] = []
    if payload.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("role") != "outcome-aware-development-training-only":
        blockers.append("role")
    if payload.get("development_only_forever") is not True:
        blockers.append("development_only_forever")
    if payload.get("promotion_evidence_eligible") is not False:
        blockers.append("promotion_evidence_eligible")
    discipline = dict(payload.get("development_discipline") or {})
    if discipline.get("candidate_count") != 1:
        blockers.append("candidate_count")
    if discipline.get("hyperparameter_search_allowed") is not False:
        blockers.append("hyperparameter_search_allowed")
    if discipline.get("threshold_search_allowed") is not False:
        blockers.append("threshold_search_allowed")
    if discipline.get("old_15m_plus_12_39_used_as_gate") is not False:
        blockers.append("old_15m_plus_12_39_used_as_gate")
    target = dict(payload.get("target") or {})
    if target.get("policy") != TARGET_POLICY:
        blockers.append("target.policy")
    if target.get("field") != TARGET_FIELD:
        blockers.append("target.field")
    if target.get("unit_sizing") is not True:
        blockers.append("target.unit_sizing")
    feature_contract = dict(payload.get("feature_contract") or {})
    if tuple(feature_contract.get("base_feature_names") or ()) != BASE_FEATURE_NAMES:
        blockers.append("feature_contract.base_feature_names")
    if feature_contract.get("shared_side_symmetric_model") is not True:
        blockers.append("feature_contract.shared_side_symmetric_model")
    if feature_contract.get("side_identity_feature_allowed") is not False:
        blockers.append("feature_contract.side_identity_feature_allowed")
    if feature_contract.get("native_missing_value") != "xgboost_nan":
        blockers.append("feature_contract.native_missing_value")
    if feature_contract.get("explicit_missing_indicator_for_every_feature") is not True:
        blockers.append("feature_contract.explicit_missing_indicator_for_every_feature")
    split = dict(payload.get("split") or {})
    if split.get("method") != "chronological_unique_market_groups":
        blockers.append("split.method")
    if (
        float(split.get("train_fraction") or 0.0),
        float(split.get("validation_fraction") or 0.0),
        float(split.get("test_fraction") or 0.0),
    ) != (0.6, 0.2, 0.2):
        blockers.append("split.fractions")
    if split.get("all_rows_for_one_market_must_remain_in_one_split") is not True:
        blockers.append("split.market_grouping")
    model = dict(payload.get("model") or {})
    if model.get("family") != "xgboost_shared_side_symmetric_regressor":
        blockers.append("model.family")
    parameters = dict(model.get("parameters") or {})
    required_parameters = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "nthread": 1,
    }
    for key, expected in required_parameters.items():
        if parameters.get(key) != expected:
            blockers.append(f"model.parameters.{key}")
    if int(model.get("num_boost_round") or 0) <= 0:
        blockers.append("model.num_boost_round")
    if int(model.get("early_stopping_rounds") or 0) <= 0:
        blockers.append("model.early_stopping_rounds")
    policy = dict(payload.get("development_evaluation_policy") or {})
    if policy.get("decision_rule") != (
        "chronological_first_decision_with_positive_prediction_then_highest_side_prediction"
    ):
        blockers.append("development_evaluation_policy.decision_rule")
    if (
        policy.get("fixed_acceptance_threshold") is None
        or float(policy["fixed_acceptance_threshold"]) != 0.0
    ):
        blockers.append("development_evaluation_policy.fixed_acceptance_threshold")
    if policy.get("one_trade_maximum_per_market") is not True:
        blockers.append("development_evaluation_policy.one_trade_maximum_per_market")
    if int(policy.get("bootstrap_resamples") or 0) <= 0:
        blockers.append("development_evaluation_policy.bootstrap_resamples")
    if dict(payload.get("safety") or {}) != SAFETY:
        blockers.append("safety")
    pins = dict(payload.get("input_pins") or {})
    required_pins = (
        "training_protocol",
        "training_readiness",
        "transfer_freeze",
        "finalized_development_corpus_index",
    )
    for name in required_pins:
        descriptor = dict(pins.get(name) or {})
        if not descriptor.get("path") or not _looks_like_sha256(descriptor.get("sha256")):
            blockers.append(f"input_pins.{name}")
    if not _looks_like_git_commit(payload.get("training_implementation_commit")):
        blockers.append("training_implementation_commit")
    if blockers:
        raise ValueError("BTC 15m training slot preregistration invalid: " + ", ".join(blockers))


def run_challenge_model_15m_training(
    *,
    preregistration_path: Path | str,
    expected_preregistration_sha256: str,
    output_dir: Path | str,
    source_commit: str,
    created_at: str | None = None,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Train exactly one preregistered model-layer candidate and write evidence."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    prereg_path = Path(preregistration_path).resolve()
    if sha256_file(prereg_path) != expected_preregistration_sha256.lower():
        raise ValueError("training slot preregistration SHA-256 mismatch")
    preregistration = _load_json(prereg_path)
    validate_training_slot_preregistration(preregistration)
    if not _looks_like_git_commit(source_commit):
        raise ValueError("source_commit must be a full Git SHA")
    pins = dict(preregistration["input_pins"])
    resolved_pins = {
        name: _verify_repo_descriptor(dict(descriptor), repo_root)
        for name, descriptor in pins.items()
    }
    readiness = _load_json(resolved_pins["training_readiness"])
    _validate_readiness(readiness)
    index_path = resolved_pins["finalized_development_corpus_index"]
    index_rows = _verify_finalized_index(index_path=index_path, repo_root=repo_root)
    expected_market_count = int(
        preregistration["dataset"]["quality_valid_outcome_finalized_market_count"]
    )
    if len(index_rows) != expected_market_count:
        raise ValueError(
            "finalized development corpus count changed after preregistration: "
            f"{len(index_rows)} != {expected_market_count}"
        )
    rows, input_corpora = _load_side_symmetric_rows(index_rows, repo_root=repo_root)
    split = _assign_market_grouped_temporal_splits(
        rows,
        train_fraction=float(preregistration["split"]["train_fraction"]),
        validation_fraction=float(preregistration["split"]["validation_fraction"]),
    )
    feature_names = (
        *BASE_FEATURE_NAMES,
        *(f"{name}__missing" for name in BASE_FEATURE_NAMES),
    )
    split_rows = {
        name: [row for row in rows if split[row["market_id"]] == name]
        for name in ("train", "validation", "test")
    }
    matrices = {
        name: _matrix(partition, feature_names)
        for name, partition in split_rows.items()
    }
    model_config = dict(preregistration["model"])
    booster = xgb.train(
        params=dict(model_config["parameters"]),
        dtrain=matrices["train"],
        num_boost_round=int(model_config["num_boost_round"]),
        evals=[
            (matrices["train"], "train"),
            (matrices["validation"], "validation"),
        ],
        early_stopping_rounds=int(model_config["early_stopping_rounds"]),
        verbose_eval=False,
    )
    iteration_end = int(booster.best_iteration) + 1
    predictions: dict[str, np.ndarray] = {
        name: booster.predict(matrix, iteration_range=(0, iteration_end))
        for name, matrix in matrices.items()
    }
    for split_name, values in predictions.items():
        for row, prediction in zip(split_rows[split_name], values, strict=True):
            row["prediction"] = float(prediction)
    generated_at = created_at or datetime.now(UTC).isoformat()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"training output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.ubj"
    booster.save_model(model_path)
    prediction_path = output / "predictions.jsonl"
    _write_jsonl(
        prediction_path,
        (
            {
                "market_id": row["market_id"],
                "market_start_ts": row["market_start_ts"],
                "decision_ts": row["decision_ts"],
                "side": row["side"],
                "split": split[row["market_id"]],
                "prediction": row["prediction"],
                "target": row["target"],
                "resolved_outcome": row["resolved_outcome"],
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
            for row in sorted(
                rows,
                key=lambda item: (
                    item["market_start_ts"],
                    item["market_id"],
                    item["decision_ts"],
                    SIDES.index(item["side"]),
                ),
            )
        ),
    )
    split_market_ids = {
        name: sorted(
            {row["market_id"] for row in partition},
            key=lambda market_id: _market_order_key(rows, market_id),
        )
        for name, partition in split_rows.items()
    }
    report = {
        "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "training_slot_id": preregistration["training_slot_id"],
        "created_at": generated_at,
        "source_commit": source_commit,
        "preregistration": _descriptor(prereg_path, repo_root=repo_root),
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "promotion_claim_made": False,
        "old_15m_plus_12_39_used_as_gate": False,
        "threshold_or_hyperparameter_search_performed": False,
        "target": dict(preregistration["target"]),
        "model": {
            "family": model_config["family"],
            "best_iteration": int(booster.best_iteration),
            "best_score": float(booster.best_score),
            "feature_count": len(feature_names),
            "model_sha256": sha256_file(model_path),
        },
        "dataset": {
            "market_count": len(split),
            "decision_count": len(rows) // len(SIDES),
            "side_row_count": len(rows),
            "resolved_outcome_distribution": dict(
                sorted(
                    Counter(
                        row["resolved_outcome"]
                        for row in rows
                        if row["side"] == "UP"
                        and row["decision_ts"]
                        == min(
                            item["decision_ts"]
                            for item in rows
                            if item["market_id"] == row["market_id"]
                        )
                    ).items()
                )
            ),
            "input_corpus_count": len(input_corpora),
            "input_index_sha256": sha256_file(index_path),
            "feature_names": list(feature_names),
            "missing_value_policy": {
                "native_missing_value": "xgboost_nan",
                "explicit_missing_indicator_for_every_feature": True,
                "missing_encoded_as_numeric_zero": False,
            },
        },
        "split": {
            "method": "chronological_unique_market_groups",
            "market_ids": split_market_ids,
            "market_counts": {
                name: len(market_ids) for name, market_ids in split_market_ids.items()
            },
            "market_ids_sha256": {
                name: canonical_json_sha256(market_ids)
                for name, market_ids in split_market_ids.items()
            },
            "market_overlap_count": _split_overlap_count(split_market_ids),
        },
        "regression_metrics": {
            name: _regression_metrics(split_rows[name])
            for name in ("train", "validation", "test")
        },
        "fixed_policy_metrics": {
            name: _fixed_policy_metrics(
                split_rows[name],
                bootstrap_resamples=int(
                    preregistration["development_evaluation_policy"][
                        "bootstrap_resamples"
                    ]
                ),
                bootstrap_seed=int(
                    preregistration["development_evaluation_policy"]["bootstrap_seed"]
                ),
            )
            for name in ("train", "validation", "test")
        },
        "development_evidence_interpretation": (
            "report_only_no_promotion_or_cross_lineage_claim"
        ),
        "model_training_started": True,
        "model_training_completed": True,
        "collection_restarted": False,
        "new_outcomes_collected": False,
        "safety": dict(SAFETY),
    }
    report_path = output / "training_report.json"
    atomic_write_json(report_path, report)
    dataset_manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "created_at": generated_at,
        "training_slot_id": preregistration["training_slot_id"],
        "finalized_index": _descriptor(index_path, repo_root=repo_root),
        "input_corpus_count": len(input_corpora),
        "input_corpora": input_corpora,
        "market_count": len(split),
        "side_row_count": len(rows),
        "split_market_ids_sha256": report["split"]["market_ids_sha256"],
        "feature_names_sha256": canonical_json_sha256(list(feature_names)),
        "target_field": TARGET_FIELD,
        "target_policy": TARGET_POLICY,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    dataset_manifest_path = output / "dataset_manifest.json"
    atomic_write_json(dataset_manifest_path, dataset_manifest)
    _write_report_markdown(output / "training_report.md", report)
    manifest = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "training_slot_id": preregistration["training_slot_id"],
        "created_at": generated_at,
        "source_commit": source_commit,
        "training_implementation_commit": preregistration[
            "training_implementation_commit"
        ],
        "artifacts": {
            "model": _descriptor(model_path, repo_root=output),
            "predictions": _descriptor(prediction_path, repo_root=output),
            "training_report": _descriptor(report_path, repo_root=output),
            "dataset_manifest": _descriptor(dataset_manifest_path, repo_root=output),
        },
        "training_completed": True,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_path = output / "training_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "training_slot_id": preregistration["training_slot_id"],
        "output_dir": str(output),
        "training_manifest_path": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "training_report_path": str(report_path),
        "training_report_sha256": sha256_file(report_path),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "test_fixed_policy_metrics": report["fixed_policy_metrics"]["test"],
        "safety": dict(SAFETY),
    }


def side_symmetric_features(feature_row: Mapping[str, Any], side: str) -> dict[str, float]:
    """Build the shared UP/DOWN representation without a side identity feature."""

    if side not in SIDES:
        raise ValueError(f"unsupported side: {side}")
    _validate_feature_causality(feature_row)
    raw = dict(feature_row.get("features") or {})
    selected = side.lower()
    opposite = "down" if side == "UP" else "up"
    sign = 1.0 if side == "UP" else -1.0
    transformed: dict[str, float] = {}
    for suffix in SIDE_RAW_SUFFIXES:
        transformed[f"selected_{suffix}"] = _numeric_or_nan(
            raw.get(_side_raw_name(selected, suffix))
        )
    for suffix in SIDE_RAW_SUFFIXES:
        transformed[f"opposite_{suffix}"] = _numeric_or_nan(
            raw.get(_side_raw_name(opposite, suffix))
        )
    transformed["selected_minus_opposite_mid"] = _difference(
        transformed["selected_mid"],
        transformed["opposite_mid"],
    )
    transformed["selected_minus_opposite_spread_bps"] = _difference(
        transformed["selected_spread_bps"],
        transformed["opposite_spread_bps"],
    )
    transformed["selected_minus_opposite_liquidity_depth"] = _difference(
        transformed["selected_liquidity_depth"],
        transformed["opposite_liquidity_depth"],
    )
    transformed["selected_minus_opposite_recent_trade_volume"] = _difference(
        transformed["selected_recent_trade_volume"],
        transformed["opposite_recent_trade_volume"],
    )
    transformed["paired_ask_sum"] = _numeric_or_nan(raw.get("up_down_ask_sum"))
    transformed["paired_bid_sum"] = _numeric_or_nan(raw.get("up_down_bid_sum"))
    transformed["paired_mid_sum"] = _numeric_or_nan(raw.get("up_down_mid_sum"))
    transformed["combined_spread_bps"] = _numeric_or_nan(
        raw.get("combined_spread_bps")
    )
    transformed["signed_chainlink_reference_distance"] = sign * _numeric_or_nan(
        raw.get("chainlink_reference_distance_at_decision")
    )
    btc_mid = _numeric_or_nan(raw.get("btc_mid_price"))
    chainlink = _numeric_or_nan(raw.get("chainlink_price_at_decision"))
    transformed["signed_btc_mid_to_chainlink_relative_distance"] = (
        sign * (btc_mid / chainlink - 1.0)
        if math.isfinite(btc_mid) and math.isfinite(chainlink) and chainlink != 0.0
        else math.nan
    )
    for horizon in ("10s", "30s", "1m", "5m", "15m"):
        transformed[f"signed_btc_return_{horizon}"] = sign * _numeric_or_nan(
            raw.get(f"btc_return_{horizon}")
        )
    for horizon in ("1m", "5m", "15m"):
        transformed[f"btc_volatility_{horizon}"] = _numeric_or_nan(
            raw.get(f"btc_volatility_{horizon}")
        )
    horizon_seconds = _numeric_or_nan(raw.get("horizon_ms")) / 1000.0
    if not math.isfinite(horizon_seconds) or horizon_seconds != 900.0:
        raise ValueError("BTC 15m feature row must use a 900 second market horizon")
    transformed["market_progress_fraction"] = (
        _numeric_or_nan(raw.get("market_age_seconds")) / horizon_seconds
    )
    transformed["time_remaining_fraction"] = (
        _numeric_or_nan(raw.get("time_to_close_seconds")) / horizon_seconds
    )
    transformed["provider_health_score"] = _numeric_or_nan(
        raw.get("provider_health_score")
    )
    transformed["book_snapshot_pair_ts_delta_ms"] = _numeric_or_nan(
        raw.get("book_snapshot_pair_ts_delta_ms")
    )
    if tuple(transformed) != BASE_FEATURE_NAMES:
        raise AssertionError("side-symmetric feature order drifted")
    with_indicators = dict(transformed)
    with_indicators.update(
        {
            f"{name}__missing": float(not math.isfinite(value))
            for name, value in transformed.items()
        }
    )
    return with_indicators


def _load_side_symmetric_rows(
    index_rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    input_corpora: list[dict[str, Any]] = []
    seen_markets: set[str] = set()
    for entry in index_rows:
        manifest_path = Path(str(entry["exported_corpus_manifest_path"])).resolve()
        if not manifest_path.is_relative_to(repo_root):
            raise ValueError("development corpus manifest escaped repository root")
        manifest = _load_json(manifest_path)
        _verify_corpus_manifest(manifest_path, manifest)
        corpus_dir = manifest_path.parent
        metadata_rows = load_jsonl(corpus_dir / "polymarket_market_metadata.jsonl")
        feature_rows = load_jsonl(corpus_dir / "polymarket_feature_rows.jsonl")
        label_rows = load_jsonl(corpus_dir / "polymarket_label_rows.jsonl")
        if len(metadata_rows) != 1:
            raise ValueError("each finalized development corpus must contain one market")
        metadata = metadata_rows[0]
        market_id = str(metadata["market_id"])
        if market_id in seen_markets:
            raise ValueError(f"duplicate finalized market: {market_id}")
        seen_markets.add(market_id)
        if len(feature_rows) != EXPECTED_DECISIONS_PER_MARKET:
            raise ValueError(
                f"{market_id} does not have {EXPECTED_DECISIONS_PER_MARKET} decisions"
            )
        labels_by_key = {
            (int(label["decision_ts"]), str(label["action"])): label
            for label in label_rows
        }
        for feature_row in sorted(feature_rows, key=lambda item: int(item["decision_ts"])):
            if str(feature_row["market_id"]) != market_id:
                raise ValueError("feature row market does not match metadata")
            raw = dict(feature_row.get("features") or {})
            for side in SIDES:
                action = f"BUY_{side}_HOLD_TO_SETTLEMENT"
                label = labels_by_key.get((int(feature_row["decision_ts"]), action))
                if label is None:
                    raise ValueError(f"missing target label for {market_id} {action}")
                target = float(label[TARGET_FIELD])
                if not math.isfinite(target):
                    raise ValueError("training target must be finite")
                if label.get("resolution_status") != "normal":
                    raise ValueError("training target requires normal finalized resolution")
                features = side_symmetric_features(feature_row, side)
                if not (
                    math.isfinite(float(raw[f"{side.lower()}_ask"]))
                    and math.isfinite(float(raw[("down" if side == "UP" else "up") + "_ask"]))
                ):
                    raise ValueError("true paired executable asks are required")
                gross_price_edge = float(label["settlement_payout"]) - float(
                    label["entry_mid"]
                )
                entry_spread_cost = float(label["entry_ask"]) - float(label["entry_mid"])
                total_cost = (
                    entry_spread_cost
                    + float(label["fees"])
                    + float(label["slippage"])
                    + float(label["liquidity_impact"])
                )
                if not math.isclose(
                    gross_price_edge - total_cost,
                    target,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError("after-cost target decomposition mismatch")
                rows.append(
                    {
                        "market_id": market_id,
                        "market_start_ts": int(metadata["market_start_ts"]),
                        "decision_ts": int(feature_row["decision_ts"]),
                        "side": side,
                        "features": features,
                        "target": target,
                        "resolved_outcome": str(label["resolved_outcome"]),
                        "gross_price_edge": gross_price_edge,
                        "entry_spread_cost": entry_spread_cost,
                        "fees": float(label["fees"]),
                        "slippage": float(label["slippage"]),
                        "liquidity_impact": float(label["liquidity_impact"]),
                    }
                )
        input_corpora.append(
            {
                "market_id": market_id,
                "market_start_ts": int(metadata["market_start_ts"]),
                "manifest_path": str(manifest_path.relative_to(repo_root)),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    return rows, input_corpora


def _assign_market_grouped_temporal_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    market_order = sorted(
        {
            (int(row["market_start_ts"]), str(row["market_id"]))
            for row in rows
        }
    )
    if len(market_order) < 5:
        raise ValueError("training requires at least five unique markets")
    train_count = int(len(market_order) * train_fraction)
    validation_count = int(len(market_order) * validation_fraction)
    if min(train_count, validation_count, len(market_order) - train_count - validation_count) <= 0:
        raise ValueError("train, validation, and test market splits must be non-empty")
    split: dict[str, str] = {}
    for index, (_, market_id) in enumerate(market_order):
        if index < train_count:
            split[market_id] = "train"
        elif index < train_count + validation_count:
            split[market_id] = "validation"
        else:
            split[market_id] = "test"
    return split


def _verify_finalized_index(
    *,
    index_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows = load_jsonl(index_path)
    if not rows:
        raise ValueError("finalized development corpus index is empty")
    previous = "0" * 64
    seen_run_ids: set[str] = set()
    for expected_sequence, row in enumerate(rows, start=1):
        if row.get("schema_version") != FINALIZED_INDEX_SCHEMA_VERSION:
            raise ValueError("finalized development index schema mismatch")
        if int(row.get("sequence") or 0) != expected_sequence:
            raise ValueError("finalized development index sequence mismatch")
        if row.get("previous_entry_sha256") != previous:
            raise ValueError("finalized development index hash chain mismatch")
        recorded = str(row.get("entry_sha256") or "")
        unhashed = dict(row)
        unhashed.pop("entry_sha256", None)
        if canonical_json_sha256(unhashed) != recorded:
            raise ValueError("finalized development index entry SHA-256 mismatch")
        if str(row.get("run_id")) in seen_run_ids:
            raise ValueError("duplicate finalized development run_id")
        seen_run_ids.add(str(row.get("run_id")))
        if not (
            row.get("official_post_close_resolution_opened") is True
            and row.get("target_used_by_capture_control") is False
            and row.get("development_only_forever") is True
            and row.get("promotion_evidence_eligible") is False
            and dict(row.get("safety") or {}) == SAFETY
        ):
            raise ValueError("finalized development index governance mismatch")
        manifest_path = Path(str(row["exported_corpus_manifest_path"])).resolve()
        if not manifest_path.is_relative_to(repo_root):
            raise ValueError("finalized manifest path escaped repository root")
        if not manifest_path.is_file():
            raise ValueError(f"finalized manifest missing: {manifest_path}")
        if sha256_file(manifest_path) != row["exported_corpus_manifest_sha256"]:
            raise ValueError("finalized corpus manifest SHA-256 mismatch")
        previous = recorded
    return rows


def _verify_corpus_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if not (
        manifest.get("market_count") == 1
        and manifest.get("market_family_counts") == {"btc_updown_15m": 1}
        and manifest.get("paper_only") is True
        and manifest.get("capital_at_risk") is False
        and manifest.get("polymarket_write_enabled") is False
        and manifest.get("wallet_signing_enabled") is False
    ):
        raise ValueError(f"finalized corpus manifest is not safe BTC 15m data: {path}")
    hashes = dict(manifest.get("normalized_artifact_hashes") or {})
    for key, filename in NORMALIZED_ARTIFACT_FILES.items():
        artifact = path.parent / filename
        if not artifact.is_file() or sha256_file(artifact) != hashes.get(key):
            raise ValueError(f"finalized corpus artifact SHA-256 mismatch: {filename}")


def _validate_feature_causality(feature_row: Mapping[str, Any]) -> None:
    decision_ts = int(feature_row["decision_ts"])
    for field in ("available_at_ts", "feature_cutoff_ts", "max_input_ts"):
        if int(feature_row[field]) > decision_ts:
            raise ValueError(f"feature causality violation: {field}")
    provenance = dict(feature_row.get("feature_provenance") or {})
    raw_dependencies: set[str] = set()
    for suffix in SIDE_RAW_SUFFIXES:
        raw_dependencies.add(_side_raw_name("up", suffix))
        raw_dependencies.add(_side_raw_name("down", suffix))
    for dependencies in GLOBAL_RAW_DEPENDENCIES.values():
        raw_dependencies.update(dependencies)
    for name in raw_dependencies:
        evidence = dict(provenance.get(name) or {})
        if not evidence:
            raise ValueError(f"feature provenance missing: {name}")
        for timestamp_field in ("available_at_ts", "max_input_ts"):
            value = evidence.get(timestamp_field)
            if value is not None and int(value) > decision_ts:
                raise ValueError(
                    f"feature provenance causality violation: {name}.{timestamp_field}"
                )


def _validate_readiness(readiness: Mapping[str, Any]) -> None:
    if not (
        readiness.get("training_start_allowed") is True
        and readiness.get("model_training_started") is False
        and not list(readiness.get("blockers") or [])
        and dict(readiness.get("safety") or {}) == SAFETY
    ):
        raise ValueError("BTC 15m training readiness is not open")


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> xgb.DMatrix:
    values = np.asarray(
        [
            [float(dict(row["features"])[feature]) for feature in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    return xgb.DMatrix(
        values,
        label=targets,
        feature_names=list(feature_names),
        missing=np.nan,
    )


def _regression_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    residuals = np.asarray(
        [float(row["prediction"]) - float(row["target"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "row_count": len(rows),
        "market_count": len({str(row["market_id"]) for row in rows}),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "mae": float(np.mean(np.abs(residuals))),
        "prediction_mean": float(np.mean([float(row["prediction"]) for row in rows])),
        "target_mean": float(np.mean([float(row["target"]) for row in rows])),
    }


def _fixed_policy_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_market[str(row["market_id"])].append(row)
    accepted: list[Mapping[str, Any]] = []
    for market_rows in by_market.values():
        by_decision: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in market_rows:
            by_decision[int(row["decision_ts"])].append(row)
        for decision_ts in sorted(by_decision):
            candidates = sorted(
                by_decision[decision_ts],
                key=lambda row: SIDES.index(str(row["side"])),
            )
            selected = max(
                candidates,
                key=lambda row: (
                    float(row["prediction"]),
                    -SIDES.index(str(row["side"])),
                ),
            )
            if float(selected["prediction"]) > 0.0:
                accepted.append(selected)
                break
    pnl = [float(row["target"]) for row in accepted]
    interval = _bootstrap_mean_interval(
        pnl,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    gross = sum(float(row["gross_price_edge"]) for row in accepted)
    cost = sum(
        float(row["entry_spread_cost"])
        + float(row["fees"])
        + float(row["slippage"])
        + float(row["liquidity_impact"])
        for row in accepted
    )
    return {
        "market_count": len(by_market),
        "accepted_market_count": len(accepted),
        "acceptance_rate": len(accepted) / len(by_market),
        "accepted_up_count": sum(row["side"] == "UP" for row in accepted),
        "accepted_down_count": sum(row["side"] == "DOWN" for row in accepted),
        "total_unit_net_pnl": sum(pnl),
        "mean_unit_net_pnl": sum(pnl) / len(pnl) if pnl else None,
        "mean_unit_net_pnl_bootstrap_interval": interval,
        "gross_price_edge": gross,
        "total_cost": cost,
        "cost_signal_ratio": cost / gross if gross > 0.0 else None,
        "unit_sizing": True,
        "fixed_acceptance_threshold": 0.0,
        "one_trade_maximum_per_market": True,
        "report_only": True,
        "promotion_claim_allowed": False,
    }


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not values:
        return None
    population = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = float(
            np.mean(generator.choice(population, size=len(population), replace=True))
        )
    return {
        "method": "accepted_market_bootstrap_percentile",
        "confidence": 0.95,
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def _split_overlap_count(split_market_ids: Mapping[str, Sequence[str]]) -> int:
    names = tuple(split_market_ids)
    overlap: set[str] = set()
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap.update(set(split_market_ids[left]) & set(split_market_ids[right]))
    return len(overlap)


def _market_order_key(rows: Sequence[Mapping[str, Any]], market_id: str) -> tuple[int, str]:
    row = next(item for item in rows if item["market_id"] == market_id)
    return int(row["market_start_ts"]), market_id


def _verify_repo_descriptor(descriptor: Mapping[str, Any], repo_root: Path) -> Path:
    path = (repo_root / str(descriptor["path"])).resolve()
    if not path.is_relative_to(repo_root):
        raise ValueError("input descriptor escaped repository root")
    if not path.is_file():
        raise ValueError(f"input descriptor path missing: {path}")
    if sha256_file(path) != str(descriptor["sha256"]).lower():
        raise ValueError(f"input descriptor SHA-256 mismatch: {path}")
    return path


def _descriptor(path: Path, *, repo_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        display_path = str(resolved)
    return {"path": display_path, "sha256": sha256_file(resolved)}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]] | Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# BTC 15m model-layer training slot",
        "",
        "Development-only evidence; permanently ineligible for promotion evidence.",
        "",
        f"- slot: `{report['training_slot_id']}`",
        f"- source commit: `{report['source_commit']}`",
        f"- markets: `{report['dataset']['market_count']}`",
        f"- side rows: `{report['dataset']['side_row_count']}`",
        f"- best iteration: `{report['model']['best_iteration']}`",
        f"- promotion claim made: `{str(report['promotion_claim_made']).lower()}`",
        "",
        "## Fixed policy report",
        "",
        "| Split | Markets | Accepted | UP / DOWN | Unit net PnL | Mean unit net PnL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "validation", "test"):
        metrics = report["fixed_policy_metrics"][name]
        mean = metrics["mean_unit_net_pnl"]
        lines.append(
            f"| {name} | {metrics['market_count']} | "
            f"{metrics['accepted_market_count']} | "
            f"{metrics['accepted_up_count']} / {metrics['accepted_down_count']} | "
            f"{metrics['total_unit_net_pnl']:.6f} | "
            f"{mean:.6f} |"
            if mean is not None
            else (
                f"| {name} | {metrics['market_count']} | 0 | 0 / 0 | "
                "0.000000 | n/a |"
            )
        )
    lines.extend(
        [
            "",
            "The zero acceptance threshold and all model hyperparameters were frozen before fit.",
            "No threshold search, promotion claim, paper action, live action, or write occurred.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _numeric_or_nan(value: Any) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    numeric = float(value)
    return numeric if math.isfinite(numeric) else math.nan


def _side_raw_name(side: str, suffix: str) -> str:
    if suffix == "recent_trade_volume":
        return f"recent_{side}_trade_volume"
    return f"{side}_{suffix}"


def _difference(left: float, right: float) -> float:
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def _looks_like_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _looks_like_git_commit(value: Any) -> bool:
    if not isinstance(value, str) or len(value) not in (40, 64):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
