"""Runner for deterministic Polymarket BTC policy training artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.execution_ev import (
    build_polymarket_ev_decisions,
    ev_threshold_report,
    run_polymarket_policy_replay,
)
from bigan.v8.polymarket.training.calibration import (
    split_calibration_report,
    validation_report,
)
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_SCHEMA_VERSION,
    POLYMARKET_POLICY_SIGNAL_SOURCE_TRAINED_MODEL,
    POLYMARKET_POLICY_TRAINING_PHASE,
    PolymarketPolicyTrainingConfig,
    PolymarketPolicyTrainingResult,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.dataset import dataset_profile, load_polymarket_policy_dataset
from bigan.v8.polymarket.training.model import (
    predict_polymarket_policy_examples,
    train_polymarket_probability_model,
)


def run_polymarket_policy_training(
    config: PolymarketPolicyTrainingConfig,
) -> PolymarketPolicyTrainingResult:
    """Run deterministic offline training, EV execution, and paper replay."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"polymarket policy run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    dataset = load_polymarket_policy_dataset(config)
    profile = dataset_profile(dataset)
    model = train_polymarket_probability_model(dataset, config)
    predictions = predict_polymarket_policy_examples(model, dataset.examples)
    predictions_by_key = {
        (prediction.market_id, prediction.decision_ts): prediction
        for prediction in predictions
    }
    train_predictions = _predictions_for_examples(predictions_by_key, dataset.train_examples)
    validation_predictions = _predictions_for_examples(
        predictions_by_key,
        dataset.validation_examples,
    )
    shadow_predictions = _predictions_for_examples(predictions_by_key, dataset.shadow_examples)
    primary_calibration_split = "validation"
    replay_split = "shadow"
    calibration = split_calibration_report(
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
        shadow_predictions=shadow_predictions,
        primary_calibration_split=primary_calibration_split,
    )
    validation = validation_report(
        validation_predictions=validation_predictions,
        train_examples=dataset.train_examples,
        evaluation_split=primary_calibration_split,
    )
    replay_predictions = shadow_predictions
    decisions = build_polymarket_ev_decisions(predictions=replay_predictions, config=config)
    ev_report = ev_threshold_report(decisions, replay_split=replay_split)
    replay_report = run_polymarket_policy_replay(
        dataset=dataset,
        decisions=decisions,
        config=config,
        calibration_error=float(calibration["calibration_error"]),
        calibration_split=primary_calibration_split,
        replay_split=replay_split,
        prediction_count=len(replay_predictions),
    )
    artifact_paths = _write_artifacts(
        run_dir=run_dir,
        config=config,
        profile=profile,
        model=model.to_dict(),
        predictions=[prediction.to_dict() for prediction in predictions],
        train_predictions=[prediction.to_dict() for prediction in train_predictions],
        validation_predictions=[
            prediction.to_dict() for prediction in validation_predictions
        ],
        shadow_predictions=[prediction.to_dict() for prediction in shadow_predictions],
        decisions=[decision.to_dict() for decision in decisions],
        calibration=calibration,
        validation=validation,
        ev_report=ev_report,
        replay_report=replay_report,
    )
    model_sha256 = _sha256_file(artifact_paths["model"])
    model_manifest = _model_manifest(
        config=config,
        dataset_profile=profile,
        model_sha256=model_sha256,
        validation=validation,
        replay_report=replay_report,
    )
    _write_json(artifact_paths["model_manifest"], model_manifest)
    artifact_hashes = {
        name: _sha256_file(path) for name, path in sorted(artifact_paths.items())
    }
    return PolymarketPolicyTrainingResult(
        run_dir=run_dir,
        dataset=dataset,
        model=model,
        predictions=predictions,
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
        shadow_predictions=shadow_predictions,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        model_manifest=model_manifest,
        calibration_report=calibration,
        validation_report=validation,
        ev_threshold_report=ev_report,
        replay_report=replay_report,
    )


def _write_artifacts(
    *,
    run_dir: Path,
    config: PolymarketPolicyTrainingConfig,
    profile: dict[str, Any],
    model: dict[str, Any],
    predictions: list[dict[str, Any]],
    train_predictions: list[dict[str, Any]],
    validation_predictions: list[dict[str, Any]],
    shadow_predictions: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    calibration: dict[str, Any],
    validation: dict[str, Any],
    ev_report: dict[str, Any],
    replay_report: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "training_config": run_dir / "polymarket_policy_training_config.json",
        "dataset_profile": run_dir / "polymarket_policy_dataset_profile.json",
        "model": run_dir / "polymarket_policy_model.json",
        "model_manifest": run_dir / "polymarket_policy_model_manifest.json",
        "calibration_report": run_dir / "polymarket_policy_calibration_report.json",
        "validation_report": run_dir / "polymarket_policy_validation_report.json",
        "ev_threshold_report": run_dir / "polymarket_ev_threshold_report.json",
        "replay_report": run_dir / "polymarket_policy_replay_report.json",
        "all_predictions": run_dir / "polymarket_policy_predictions.jsonl",
        "predictions": run_dir / "polymarket_policy_predictions.jsonl",
        "train_predictions": run_dir / "polymarket_policy_train_predictions.jsonl",
        "validation_predictions": run_dir / "polymarket_policy_validation_predictions.jsonl",
        "shadow_predictions": run_dir / "polymarket_policy_shadow_predictions.jsonl",
        "ev_decisions": run_dir / "polymarket_ev_decisions.jsonl",
        "summary": run_dir / "polymarket_policy_training_summary.md",
    }
    _write_json(paths["training_config"], config.to_manifest_dict())
    _write_json(paths["dataset_profile"], profile)
    _write_json(paths["model"], model)
    _write_json(paths["calibration_report"], calibration)
    _write_json(paths["validation_report"], validation)
    _write_json(paths["ev_threshold_report"], ev_report)
    _write_json(paths["replay_report"], replay_report)
    _write_jsonl(paths["predictions"], predictions)
    _write_jsonl(paths["train_predictions"], train_predictions)
    _write_jsonl(paths["validation_predictions"], validation_predictions)
    _write_jsonl(paths["shadow_predictions"], shadow_predictions)
    _write_jsonl(paths["ev_decisions"], decisions)
    paths["summary"].write_text(
        _summary_markdown(
            profile=profile,
            validation=validation,
            ev_report=ev_report,
            replay_report=replay_report,
        ),
        encoding="utf-8",
    )
    return paths


def _model_manifest(
    *,
    config: PolymarketPolicyTrainingConfig,
    dataset_profile: dict[str, Any],
    model_sha256: str,
    validation: dict[str, Any],
    replay_report: dict[str, Any],
) -> dict[str, Any]:
    split_fields = {
        field_name: dataset_profile[field_name]
        for field_name in (
            "split_strategy",
            "strict_temporal_separation",
            "train_min_ts",
            "train_max_ts",
            "validation_min_ts",
            "validation_max_ts",
            "shadow_min_ts",
            "shadow_max_ts",
        )
    }
    return {
        "schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "model_version": config.model_version,
        "model_family": "deterministic_frequency_probability",
        "target": "resolved_up",
        "model_output": "estimated_up_probability",
        "market_families": sorted(dataset_profile["market_family_counts"]),
        "training_corpus_hash": dataset_profile["training_corpus_hash"],
        "feature_schema_hash": dataset_profile["feature_schema_hash"],
        "label_schema_hash": dataset_profile["label_schema_hash"],
        "dataset_hash": dataset_profile["dataset_hash"],
        "model_sha256": model_sha256,
        "train_row_count": dataset_profile["train_row_count"],
        "validation_row_count": dataset_profile["validation_row_count"],
        "shadow_row_count": dataset_profile["shadow_row_count"],
        **split_fields,
        "calibration_split": replay_report["calibration_split"],
        "replay_split": replay_report["replay_split"],
        "out_of_sample_replay": replay_report["out_of_sample_replay"],
        "created_at": config.created_at,
        "direct_pnl_optimization": False,
        "pnl_usage": "validation_and_ev_replay_only",
        "trained_model_used": True,
        "policy_signal_source": POLYMARKET_POLICY_SIGNAL_SOURCE_TRAINED_MODEL,
        "synthetic_fixture_signal_used": False,
        "validation_brier_score": validation["validation"]["brier_score"],
        "model_is_calibrated_better_than_naive_baseline": validation[
            "model_is_calibrated_better_than_naive_baseline"
        ],
        "paper_replay_used_phase1_settlement_engine": replay_report[
            "phase1_settlement_engine_used"
        ],
        **compact_safety_fields(),
    }


def _summary_markdown(
    *,
    profile: dict[str, Any],
    validation: dict[str, Any],
    ev_report: dict[str, Any],
    replay_report: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Polymarket BTC Policy Training Summary",
            "",
            f"- row_count: {profile['row_count']}",
            f"- train_row_count: {profile['train_row_count']}",
            f"- validation_row_count: {profile['validation_row_count']}",
            f"- shadow_row_count: {profile['shadow_row_count']}",
            f"- calibration_split: {replay_report['calibration_split']}",
            f"- replay_split: {replay_report['replay_split']}",
            f"- out_of_sample_replay: {str(replay_report['out_of_sample_replay']).lower()}",
            f"- validation_brier_score: {validation['validation']['brier_score']}",
            f"- calibration_error: {replay_report['calibration_error']}",
            f"- trade_count: {replay_report['trade_count']}",
            f"- no_trade_count: {replay_report['no_trade_count']}",
            f"- total_polymarket_pnl: {replay_report['total_polymarket_pnl']}",
            f"- ev_action_counts: {json.dumps(ev_report['action_counts'], sort_keys=True)}",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )


def _predictions_for_examples(
    predictions_by_key: dict[tuple[str, int], Any],
    examples: tuple[Any, ...],
) -> tuple[Any, ...]:
    return tuple(
        predictions_by_key[(example.market_id, example.decision_ts)]
        for example in examples
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
