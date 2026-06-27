"""Runner for deterministic Polymarket BTC policy training artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.execution_ev import (
    build_polymarket_ev_decisions,
    ev_threshold_report,
    run_polymarket_policy_replay,
)
from bigan.v8.polymarket.training.action_family_eligibility import (
    action_family_eligibility_markdown,
    action_family_replay_variants_markdown,
    build_action_family_eligibility_report,
    build_action_family_replay_variants_report,
    build_hold_to_settlement_longshot_guard_report,
    hold_to_settlement_longshot_guard_markdown,
)
from bigan.v8.polymarket.training.action_value_calibration import (
    apply_action_value_calibration,
    build_action_value_calibration_artifact,
)
from bigan.v8.polymarket.training.calibration import (
    split_calibration_report,
    validation_report,
)
from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    AUXILIARY_OUTCOME_TARGET,
    POLYMARKET_POLICY_SCHEMA_VERSION,
    POLYMARKET_POLICY_SIGNAL_SOURCE_TRAINED_MODEL,
    POLYMARKET_POLICY_TRAINING_PHASE,
    PRIMARY_POLICY_TARGET_ACTION_VALUE,
    PolymarketPolicyTrainingConfig,
    PolymarketPolicyTrainingResult,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.dataset import (
    ACTION_VALUE_TARGET_FIELD,
    dataset_profile,
    load_polymarket_policy_dataset,
)
from bigan.v8.polymarket.training.model import (
    predict_polymarket_policy_examples,
    train_polymarket_action_value_model,
)

ACTION_VALUE_CONCENTRATION_WARN_THRESHOLD = 0.80
ACTION_VALUE_CONCENTRATION_FAIL_THRESHOLD = 0.95
P_UP_ACTION_DISAGREEMENT_FAIL_THRESHOLD = 0.50
P_UP_MATERIAL_DISAGREEMENT_THRESHOLD = 0.55


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
    model = train_polymarket_action_value_model(dataset, config)
    raw_predictions = predict_polymarket_policy_examples(model, dataset.examples)
    raw_predictions_by_key = {
        (prediction.market_id, prediction.decision_ts): prediction
        for prediction in raw_predictions
    }
    raw_train_predictions = _predictions_for_examples(
        raw_predictions_by_key,
        dataset.train_examples,
    )
    raw_validation_predictions = _predictions_for_examples(
        raw_predictions_by_key,
        dataset.validation_examples,
    )
    raw_shadow_predictions = _predictions_for_examples(
        raw_predictions_by_key,
        dataset.shadow_examples,
    )
    primary_calibration_split = "validation"
    replay_split = "shadow"
    calibration = split_calibration_report(
        train_predictions=raw_train_predictions,
        validation_predictions=raw_validation_predictions,
        shadow_predictions=raw_shadow_predictions,
        primary_calibration_split=primary_calibration_split,
    )
    validation = validation_report(
        validation_predictions=raw_validation_predictions,
        train_examples=dataset.train_examples,
        evaluation_split=primary_calibration_split,
    )
    action_value_calibration = build_action_value_calibration_artifact(
        calibration_examples=dataset.validation_examples,
        calibration_predictions=raw_validation_predictions,
        evaluation_examples=dataset.shadow_examples,
        evaluation_predictions=raw_shadow_predictions,
        execution_buffer=float(config.ev_threshold),
    )
    predictions = apply_action_value_calibration(
        predictions=raw_predictions,
        calibration_artifact=action_value_calibration,
    )
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
    replay_predictions = shadow_predictions
    action_family_eligibility = build_action_family_eligibility_report(
        examples=dataset.shadow_examples,
        predictions=shadow_predictions,
        execution_buffer=float(config.ev_threshold),
    )
    hold_to_settlement_longshot_guard = build_hold_to_settlement_longshot_guard_report(
        examples=dataset.shadow_examples,
        predictions=shadow_predictions,
        execution_buffer=float(config.ev_threshold),
    )
    action_family_replay_variants = build_action_family_replay_variants_report(
        examples=dataset.shadow_examples,
        predictions=shadow_predictions,
        execution_buffer=float(config.ev_threshold),
    )
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
    signal_sanity = _action_value_signal_sanity_report(
        validation_predictions=validation_predictions,
        shadow_predictions=shadow_predictions,
        action_value_calibration=action_value_calibration,
        action_family_eligibility=action_family_eligibility,
        hold_to_settlement_longshot_guard=hold_to_settlement_longshot_guard,
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
        action_value_calibration=action_value_calibration,
        action_family_eligibility=action_family_eligibility,
        hold_to_settlement_longshot_guard=hold_to_settlement_longshot_guard,
        action_family_replay_variants=action_family_replay_variants,
        signal_sanity=signal_sanity,
    )
    model_sha256 = _sha256_file(artifact_paths["model"])
    action_value_calibration_sha256 = _sha256_file(
        artifact_paths["action_value_calibration"]
    )
    model_manifest = _model_manifest(
        config=config,
        dataset_profile=profile,
        model=model,
        model_sha256=model_sha256,
        action_value_calibration=action_value_calibration,
        action_value_calibration_sha256=action_value_calibration_sha256,
        validation=validation,
        replay_report=replay_report,
        signal_sanity=signal_sanity,
        action_family_eligibility=action_family_eligibility,
        hold_to_settlement_longshot_guard=hold_to_settlement_longshot_guard,
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
        action_value_signal_sanity_report=signal_sanity,
        action_family_eligibility_report=action_family_eligibility,
        hold_to_settlement_longshot_guard_report=hold_to_settlement_longshot_guard,
        action_family_replay_variants_report=action_family_replay_variants,
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
    action_value_calibration: dict[str, Any],
    action_family_eligibility: dict[str, Any],
    hold_to_settlement_longshot_guard: dict[str, Any],
    action_family_replay_variants: dict[str, Any],
    signal_sanity: dict[str, Any],
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
        "action_value_calibration": run_dir / "polymarket_action_value_calibration.json",
        "action_value_signal_sanity_report": (
            run_dir / "polymarket_action_value_signal_sanity_report.json"
        ),
        "action_value_signal_sanity_summary": (
            run_dir / "polymarket_action_value_signal_sanity_report.md"
        ),
        "action_family_eligibility_report": (
            run_dir / "action_family_eligibility_report.json"
        ),
        "action_family_eligibility_summary": (
            run_dir / "action_family_eligibility_report.md"
        ),
        "hold_to_settlement_longshot_guard_report": (
            run_dir / "hold_to_settlement_longshot_guard_report.json"
        ),
        "hold_to_settlement_longshot_guard_summary": (
            run_dir / "hold_to_settlement_longshot_guard_report.md"
        ),
        "action_family_replay_variants_report": (
            run_dir / "action_family_replay_variants_report.json"
        ),
        "action_family_replay_variants_summary": (
            run_dir / "action_family_replay_variants_report.md"
        ),
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
    _write_json(paths["action_value_calibration"], action_value_calibration)
    _write_json(paths["action_value_signal_sanity_report"], signal_sanity)
    _write_json(paths["action_family_eligibility_report"], action_family_eligibility)
    _write_json(
        paths["hold_to_settlement_longshot_guard_report"],
        hold_to_settlement_longshot_guard,
    )
    _write_json(paths["action_family_replay_variants_report"], action_family_replay_variants)
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
    paths["action_value_signal_sanity_summary"].write_text(
        _signal_sanity_markdown(signal_sanity),
        encoding="utf-8",
    )
    paths["action_family_eligibility_summary"].write_text(
        action_family_eligibility_markdown(action_family_eligibility),
        encoding="utf-8",
    )
    paths["hold_to_settlement_longshot_guard_summary"].write_text(
        hold_to_settlement_longshot_guard_markdown(hold_to_settlement_longshot_guard),
        encoding="utf-8",
    )
    paths["action_family_replay_variants_summary"].write_text(
        action_family_replay_variants_markdown(action_family_replay_variants),
        encoding="utf-8",
    )
    return paths


def _model_manifest(
    *,
    config: PolymarketPolicyTrainingConfig,
    dataset_profile: dict[str, Any],
    model: Any,
    model_sha256: str,
    action_value_calibration: dict[str, Any],
    action_value_calibration_sha256: str,
    validation: dict[str, Any],
    replay_report: dict[str, Any],
    signal_sanity: dict[str, Any],
    action_family_eligibility: dict[str, Any],
    hold_to_settlement_longshot_guard: dict[str, Any],
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
        "model_family": "deterministic_action_value_probability",
        "target": PRIMARY_POLICY_TARGET_ACTION_VALUE,
        "primary_policy_target": PRIMARY_POLICY_TARGET_ACTION_VALUE,
        "legacy_primary_policy_target": PRIMARY_POLICY_TARGET_ACTION_VALUE,
        "primary_policy_target_unit": "fixed_notional_net_pnl_per_notional",
        "auxiliary_outcome_target": AUXILIARY_OUTCOME_TARGET,
        "model_output": "action_expected_returns_with_p_up_auxiliary",
        "model_outputs": [
            "p_up_auxiliary",
            "estimated_up_probability",
            "expected_return_by_action",
            "best_policy_action",
            "best_action_expected_return",
            "second_best_action_expected_return",
            "best_action_margin",
            "policy_confidence",
        ],
        "outcome_probability_head_enabled": True,
        "action_value_head_enabled": True,
        "compatibility_probability_fallback_enabled": True,
        "action_value_model_family": model.action_value_model_family,
        "fallback_action_value_model_family": model.fallback_action_value_model_family,
        "feature_conditioned_action_value_model_enabled": (
            model.feature_conditioned_action_value_model_enabled
        ),
        "action_value_target_field": ACTION_VALUE_TARGET_FIELD,
        "fixed_notional_target_used": True,
        "action_value_calibration_id": action_value_calibration[
            "action_value_calibration_id"
        ],
        "action_value_calibration_artifact_path": "polymarket_action_value_calibration.json",
        "action_value_calibration_sha256": action_value_calibration_sha256,
        "action_value_calibration_method": action_value_calibration[
            "calibration_method"
        ],
        "action_value_calibration_fit_split": action_value_calibration[
            "calibration_fit_split"
        ],
        "action_value_calibration_evaluation_split": action_value_calibration[
            "calibration_evaluation_split"
        ],
        "action_value_calibration_support_count": action_value_calibration[
            "calibration_support_count"
        ],
        "action_value_calibration_bucket_count": action_value_calibration[
            "calibration_bucket_count"
        ],
        "calibration_quality_passed": signal_sanity[
            "calibration_quality_passed"
        ],
        "calibration_quality_gates": action_value_calibration[
            "calibration_quality_gates"
        ],
        "shadow_high_score_bucket": action_value_calibration[
            "shadow_high_score_bucket"
        ],
        "shadow_mae_comparison": action_value_calibration["shadow_mae_comparison"],
        "bucket_shrinkage_enabled": action_value_calibration[
            "bucket_shrinkage_enabled"
        ],
        "bucket_shrinkage_prior": action_value_calibration["bucket_shrinkage_prior"],
        "high_score_min_support": action_value_calibration["high_score_min_support"],
        "high_score_execution_buffer": action_value_calibration[
            "high_score_execution_buffer"
        ],
        "action_value_calibration_artifact_used": signal_sanity[
            "action_value_calibration_artifact_used"
        ],
        "execution_uses_calibrated_action_value": signal_sanity[
            "execution_uses_calibrated_action_value"
        ],
        "calibration_support_passed": signal_sanity[
            "calibration_support_passed"
        ],
        "best_action_concentration_passed": signal_sanity[
            "best_action_concentration_passed"
        ],
        "p_up_action_disagreement_within_limit": signal_sanity[
            "p_up_action_disagreement_within_limit"
        ],
        "action_value_paper_decision_eligible": signal_sanity[
            "action_value_paper_decision_eligible"
        ],
        "action_value_paper_decision_ineligible_reasons": signal_sanity[
            "action_value_paper_decision_ineligible_reasons"
        ],
        "action_family_eligibility_report_path": "action_family_eligibility_report.json",
        "hold_to_settlement_longshot_guard_report_path": (
            "hold_to_settlement_longshot_guard_report.json"
        ),
        "action_family_replay_variants_report_path": (
            "action_family_replay_variants_report.json"
        ),
        "action_family_paper_decision_eligible": action_family_eligibility[
            "action_family_paper_decision_eligible"
        ],
        "action_family_paper_decision_ineligible_reasons": action_family_eligibility[
            "action_family_paper_decision_ineligible_reasons"
        ],
        "action_family_eligibility_report": action_family_eligibility,
        "hold_to_settlement_longshot_guard_enabled": (
            hold_to_settlement_longshot_guard["guard_enabled"]
        ),
        "hold_to_settlement_longshot_guard_reason_codes": (
            hold_to_settlement_longshot_guard["guard_reason_codes"]
        ),
        "hold_to_settlement_longshot_guard_report": (
            hold_to_settlement_longshot_guard
        ),
        "action_value_signal_sanity_report": signal_sanity,
        "action_value_feature_columns": list(model.action_value_feature_columns),
        "required_action_value_feature_columns": list(model.action_value_feature_columns),
        "action_label_coverage_by_action": dataset_profile[
            "action_label_coverage_by_action"
        ],
        "best_policy_action_counts": dataset_profile["best_policy_action_counts"],
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
        "pnl_usage": "fixed_notional_net_pnl_label_supervision_validation_and_ev_replay",
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
            f"- primary_policy_target: {PRIMARY_POLICY_TARGET_ACTION_VALUE}",
            f"- action_value_target_field: {ACTION_VALUE_TARGET_FIELD}",
            f"- action_value_head_enabled: {str(ev_report['action_value_head_enabled']).lower()}",
            f"- action_value_model_family: {ev_report['action_value_model_family']}",
            f"- validation_brier_score: {validation['validation']['brier_score']}",
            f"- calibration_error: {replay_report['calibration_error']}",
            f"- mean_best_action_expected_return: {replay_report['action_value_policy_metrics']['mean_best_action_expected_return']}",
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


def _action_value_signal_sanity_report(
    *,
    validation_predictions: tuple[Any, ...],
    shadow_predictions: tuple[Any, ...],
    action_value_calibration: dict[str, Any],
    action_family_eligibility: dict[str, Any],
    hold_to_settlement_longshot_guard: dict[str, Any],
) -> dict[str, Any]:
    split_predictions = {
        "validation": validation_predictions,
        "shadow": shadow_predictions,
    }
    all_predictions = tuple(
        prediction
        for predictions in split_predictions.values()
        for prediction in predictions
        if bool(getattr(prediction, "action_value_head_enabled", False))
    )
    action_counts = Counter(_execution_policy_action(prediction) for prediction in all_predictions)
    sample_count = sum(action_counts.values())
    best_action_max_action, best_action_max_count = (
        ("", 0) if sample_count == 0 else action_counts.most_common(1)[0]
    )
    best_action_max_ratio = (
        0.0 if sample_count == 0 else best_action_max_count / sample_count
    )
    disagreement_examples = [
        _p_up_action_disagreement_example(prediction)
        for prediction in all_predictions
        if _p_up_action_disagrees(prediction)
    ]
    disagreement_rate = (
        0.0 if sample_count == 0 else len(disagreement_examples) / sample_count
    )
    best_action_concentration_passed = (
        best_action_max_ratio <= ACTION_VALUE_CONCENTRATION_FAIL_THRESHOLD
    )
    p_up_action_disagreement_within_limit = (
        disagreement_rate <= P_UP_ACTION_DISAGREEMENT_FAIL_THRESHOLD
    )
    calibration_support_passed = bool(
        action_value_calibration["calibration_support_passed"]
    )
    calibration_quality_passed = bool(
        action_value_calibration["calibration_quality_passed"]
    )
    ineligible_reasons = set()
    if not calibration_support_passed:
        ineligible_reasons.add("action_value_calibration_support_insufficient")
    if not calibration_quality_passed:
        ineligible_reasons.add("action_value_calibration_quality_failed")
    if not best_action_concentration_passed:
        ineligible_reasons.add("action_value_policy_collapse")
    if not p_up_action_disagreement_within_limit:
        ineligible_reasons.add("p_up_action_disagreement_excessive")
    if not action_family_eligibility["action_family_paper_decision_eligible"]:
        ineligible_reasons.update(
            action_family_eligibility[
                "action_family_paper_decision_ineligible_reasons"
            ]
        )
    paper_decision_eligible = (
        calibration_support_passed
        and calibration_quality_passed
        and best_action_concentration_passed
        and p_up_action_disagreement_within_limit
        and action_family_eligibility["action_family_paper_decision_eligible"]
    )
    return {
        "schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "signal_sanity_report_version": "action_value_signal_sanity_v1",
        "out_of_sample_split_names": sorted(split_predictions),
        "sample_count": sample_count,
        "split_sample_counts": {
            split_name: len(predictions)
            for split_name, predictions in sorted(split_predictions.items())
        },
        "action_value_calibration_artifact_used": True,
        "action_value_calibration_id": action_value_calibration[
            "action_value_calibration_id"
        ],
        "action_value_calibration_method": action_value_calibration[
            "calibration_method"
        ],
        "action_value_calibration_fit_split": action_value_calibration[
            "calibration_fit_split"
        ],
        "action_value_calibration_evaluation_split": action_value_calibration[
            "calibration_evaluation_split"
        ],
        "execution_uses_calibrated_action_value": True,
        "calibration_support_passed": calibration_support_passed,
        "calibration_quality_passed": calibration_quality_passed,
        "calibration_quality_gates": action_value_calibration[
            "calibration_quality_gates"
        ],
        "shadow_high_score_bucket": action_value_calibration[
            "shadow_high_score_bucket"
        ],
        "shadow_mae_comparison": action_value_calibration["shadow_mae_comparison"],
        "bucket_shrinkage_enabled": action_value_calibration[
            "bucket_shrinkage_enabled"
        ],
        "bucket_shrinkage_prior": action_value_calibration["bucket_shrinkage_prior"],
        "high_score_min_support": action_value_calibration["high_score_min_support"],
        "high_score_execution_buffer": action_value_calibration[
            "high_score_execution_buffer"
        ],
        "calibration_support_count": action_value_calibration[
            "calibration_support_count"
        ],
        "calibration_bucket_count": action_value_calibration[
            "calibration_bucket_count"
        ],
        "best_action_counts": {
            action: action_counts.get(action, 0) for action in ACTION_VALUE_LABEL_ACTIONS
        },
        "best_action_counts_by_split": {
            split_name: _action_counts(predictions)
            for split_name, predictions in sorted(split_predictions.items())
        },
        "best_action_max_action": best_action_max_action or None,
        "best_action_max_count": best_action_max_count,
        "best_action_max_ratio": best_action_max_ratio,
        "best_action_concentration_warn_threshold": (
            ACTION_VALUE_CONCENTRATION_WARN_THRESHOLD
        ),
        "best_action_concentration_fail_threshold": (
            ACTION_VALUE_CONCENTRATION_FAIL_THRESHOLD
        ),
        "best_action_concentration_warning": (
            best_action_max_ratio > ACTION_VALUE_CONCENTRATION_WARN_THRESHOLD
        ),
        "best_action_concentration_passed": best_action_concentration_passed,
        "p_up_action_disagreement_count": len(disagreement_examples),
        "p_up_action_disagreement_rate": disagreement_rate,
        "p_up_action_disagreement_fail_threshold": (
            P_UP_ACTION_DISAGREEMENT_FAIL_THRESHOLD
        ),
        "p_up_material_disagreement_threshold": P_UP_MATERIAL_DISAGREEMENT_THRESHOLD,
        "p_up_action_disagreement_within_limit": (
            p_up_action_disagreement_within_limit
        ),
        "p_up_action_disagreement_examples": disagreement_examples[:20],
        "action_family_paper_decision_eligible": action_family_eligibility[
            "action_family_paper_decision_eligible"
        ],
        "action_family_paper_decision_ineligible_reasons": action_family_eligibility[
            "action_family_paper_decision_ineligible_reasons"
        ],
        "action_family_eligibility_report": action_family_eligibility,
        "hold_to_settlement_longshot_guard_enabled": (
            hold_to_settlement_longshot_guard["guard_enabled"]
        ),
        "hold_to_settlement_longshot_guard_reason_codes": (
            hold_to_settlement_longshot_guard["guard_reason_codes"]
        ),
        "hold_to_settlement_longshot_guard_report": (
            hold_to_settlement_longshot_guard
        ),
        "action_value_paper_decision_eligible": paper_decision_eligible,
        "action_value_paper_decision_ineligible_reasons": sorted(ineligible_reasons),
        **compact_safety_fields(),
    }


def _action_counts(predictions: tuple[Any, ...]) -> dict[str, int]:
    counts = Counter(
        _execution_policy_action(prediction)
        for prediction in predictions
        if bool(getattr(prediction, "action_value_head_enabled", False))
    )
    return {action: counts.get(action, 0) for action in ACTION_VALUE_LABEL_ACTIONS}


def _execution_policy_action(prediction: Any) -> str:
    calibrated_action = getattr(prediction, "calibrated_best_policy_action", None)
    if calibrated_action is not None:
        return str(calibrated_action)
    return str(getattr(prediction, "best_policy_action", ""))


def _p_up_action_disagrees(prediction: Any) -> bool:
    action = _execution_policy_action(prediction)
    p_up = getattr(prediction, "p_up_auxiliary", None)
    if p_up is None:
        p_up = getattr(prediction, "estimated_up_probability", None)
    if p_up is None:
        return False
    p_up = float(p_up)
    if action.startswith("BUY_DOWN_"):
        return p_up >= P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    if action.startswith("BUY_UP_"):
        return p_up <= 1.0 - P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    return False


def _p_up_action_disagreement_example(prediction: Any) -> dict[str, Any]:
    p_up = getattr(prediction, "p_up_auxiliary", None)
    if p_up is None:
        p_up = getattr(prediction, "estimated_up_probability", None)
    return {
        "market_id": getattr(prediction, "market_id", ""),
        "decision_ts": getattr(prediction, "decision_ts", 0),
        "p_up_auxiliary": p_up,
        "estimated_up_probability": getattr(
            prediction,
            "estimated_up_probability",
            None,
        ),
        "best_policy_action": getattr(prediction, "best_policy_action", None),
        "calibrated_best_policy_action": getattr(
            prediction,
            "calibrated_best_policy_action",
            None,
        ),
        "best_action_expected_return": getattr(
            prediction,
            "best_action_expected_return",
            None,
        ),
        "calibrated_expected_pnl_per_notional": getattr(
            prediction,
            "calibrated_expected_pnl_per_notional",
            None,
        ),
        "best_action_margin": getattr(prediction, "best_action_margin", None),
        "calibrated_action_margin": getattr(
            prediction,
            "calibrated_action_margin",
            None,
        ),
    }


def _signal_sanity_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Polymarket Action-Value Signal Sanity Report",
            "",
            f"- sample_count: {report['sample_count']}",
            "- action_value_calibration_artifact_used: "
            f"{str(report['action_value_calibration_artifact_used']).lower()}",
            "- execution_uses_calibrated_action_value: "
            f"{str(report['execution_uses_calibrated_action_value']).lower()}",
            f"- calibration_support_passed: {str(report['calibration_support_passed']).lower()}",
            f"- calibration_quality_passed: {str(report['calibration_quality_passed']).lower()}",
            "- shadow_calibrated_mae_not_worse: "
            f"{str(report['calibration_quality_gates']['shadow_calibrated_mae_not_worse']).lower()}",
            "- shadow_raw_mae: "
            f"{report['shadow_mae_comparison']['raw_mae']}",
            "- shadow_action_level_calibrated_mae: "
            f"{report['shadow_mae_comparison']['action_level_calibrated_mae']}",
            "- shadow_bucketed_calibrated_mae: "
            f"{report['shadow_mae_comparison']['bucketed_calibrated_mae']}",
            "- bucket_shrinkage_enabled: "
            f"{str(report['bucket_shrinkage_enabled']).lower()}",
            f"- bucket_shrinkage_prior: {report['bucket_shrinkage_prior']}",
            "- high_score_bucket_min_support_passed: "
            f"{str(report['calibration_quality_gates']['high_score_bucket_min_support_passed']).lower()}",
            "- high_score_bucket_realized_return_exceeds_buffer: "
            f"{str(report['calibration_quality_gates']['high_score_bucket_realized_return_exceeds_buffer']).lower()}",
            f"- high_score_min_support: {report['high_score_min_support']}",
            f"- high_score_execution_buffer: {report['high_score_execution_buffer']}",
            f"- best_action_counts: {json.dumps(report['best_action_counts'], sort_keys=True)}",
            f"- best_action_max_action: {report['best_action_max_action']}",
            f"- best_action_max_ratio: {report['best_action_max_ratio']}",
            "- best_action_concentration_warning: "
            f"{str(report['best_action_concentration_warning']).lower()}",
            "- best_action_concentration_passed: "
            f"{str(report['best_action_concentration_passed']).lower()}",
            "- p_up_action_disagreement_count: "
            f"{report['p_up_action_disagreement_count']}",
            "- p_up_action_disagreement_rate: "
            f"{report['p_up_action_disagreement_rate']}",
            "- p_up_action_disagreement_within_limit: "
            f"{str(report['p_up_action_disagreement_within_limit']).lower()}",
            "- action_family_paper_decision_eligible: "
            f"{str(report['action_family_paper_decision_eligible']).lower()}",
            "- action_family_paper_decision_ineligible_reasons: "
            f"{json.dumps(report['action_family_paper_decision_ineligible_reasons'])}",
            "- hold_to_settlement_longshot_guard_enabled: "
            f"{str(report['hold_to_settlement_longshot_guard_enabled']).lower()}",
            "- hold_to_settlement_longshot_guard_reason_codes: "
            f"{json.dumps(report['hold_to_settlement_longshot_guard_reason_codes'])}",
            "- action_value_paper_decision_eligible: "
            f"{str(report['action_value_paper_decision_eligible']).lower()}",
            "- action_value_paper_decision_ineligible_reasons: "
            f"{json.dumps(report['action_value_paper_decision_ineligible_reasons'])}",
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
