"""Runner for deterministic Polymarket BTC policy training artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.execution_ev import (
    build_polymarket_ev_decisions,
    ev_threshold_report,
    run_polymarket_policy_replay,
)
from bigan.v8.polymarket.training.action_family_eligibility import (
    action_family_eligibility_markdown,
    action_family_replay_variants_markdown,
    build_action_family_counterfactual_prediction_sets,
    build_action_family_eligibility_report,
    build_action_family_replay_variants_report,
    build_hold_to_settlement_longshot_guard_report,
    build_sell_before_close_support_aware_prediction_set,
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
from bigan.v8.polymarket.training.model_ranking_diagnostics import (
    action_representation_diagnostic_markdown,
    build_action_representation_diagnostic_report,
    build_model_ranking_candidate_comparison,
    build_model_ranking_error_report,
    build_ranking_overlay_zero_entry_diagnostic_report,
    build_source_model_eligibility_report,
    model_ranking_candidate_comparison_markdown,
    model_ranking_error_markdown,
    ranking_overlay_zero_entry_diagnostic_markdown,
    source_model_eligibility_markdown,
)
from bigan.v8.polymarket.training.sell_before_close_diagnostics import (
    build_sell_before_close_p_up_disagreement_diagnostic_report,
    sell_before_close_p_up_disagreement_diagnostic_markdown,
    sell_before_close_p_up_disagreement_summary,
)
from bigan.v8.polymarket.training.sell_before_close_exit_reliability import (
    build_sell_before_close_exit_reliability_guard_decisions,
    build_sell_before_close_exit_reliability_report,
    sell_before_close_exit_reliability_markdown,
    sell_before_close_exit_reliability_summary,
)
from bigan.v8.polymarket.training.sell_before_close_promotion_support import (
    build_sell_before_close_promotion_support_gate_report,
    evaluate_sell_before_close_promotion_support,
    sell_before_close_promotion_support_gate_markdown,
    sell_before_close_promotion_support_gate_summary,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_GUARD_THRESHOLD_SWEEP_GRID,
    SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
)
from bigan.v8.polymarket.training.sell_before_close_support_aware_thresholds import (
    build_sell_before_close_support_aware_threshold_selection_report,
    sell_before_close_support_aware_threshold_failure_attribution_markdown,
    sell_before_close_support_aware_threshold_failure_attribution_summary,
    sell_before_close_support_aware_threshold_selection_markdown,
    sell_before_close_support_aware_threshold_selection_summary,
    sell_before_close_validation_failure_drilldown_markdown,
    sell_before_close_validation_failure_drilldown_summary,
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
    validation_support_aware_prediction_set = (
        build_sell_before_close_support_aware_prediction_set(
            predictions=validation_predictions,
            execution_buffer=float(config.ev_threshold),
            entry_filter_thresholds=dict(
                SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS
            ),
        )
    )
    shadow_support_aware_prediction_set = (
        build_sell_before_close_support_aware_prediction_set(
            predictions=shadow_predictions,
            execution_buffer=float(config.ev_threshold),
            entry_filter_thresholds=dict(
                SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS
            ),
        )
    )
    sell_before_close_support_aware_threshold_selection = (
        build_sell_before_close_support_aware_threshold_selection_report(
            dataset=dataset,
            validation_predictions=tuple(
                validation_support_aware_prediction_set["predictions"]
            ),
            shadow_predictions=tuple(
                shadow_support_aware_prediction_set["predictions"]
            ),
            config=config,
            calibration_error=float(calibration["calibration_error"]),
            calibration_split=primary_calibration_split,
        )
    )
    sell_before_close_support_aware_threshold_failure_attribution = (
        sell_before_close_support_aware_threshold_selection[
            "failure_attribution_report"
        ]
    )
    sell_before_close_validation_failure_drilldown = (
        sell_before_close_support_aware_threshold_selection[
            "validation_failure_drilldown_report"
        ]
    )
    counterfactual_prediction_sets = build_action_family_counterfactual_prediction_sets(
        examples=dataset.shadow_examples,
        predictions=shadow_predictions,
        execution_buffer=float(config.ev_threshold),
        support_aware_thresholds=(
            sell_before_close_support_aware_threshold_selection[
                "selected_thresholds"
            ]
            or None
        ),
        support_aware_threshold_selection_report=(
            sell_before_close_support_aware_threshold_selection
        ),
    )
    action_family_counterfactual_replays = _build_action_family_counterfactual_replays(
        dataset=dataset,
        prediction_sets=counterfactual_prediction_sets,
        config=config,
        calibration_error=float(calibration["calibration_error"]),
        calibration_split=primary_calibration_split,
        replay_split=replay_split,
    )
    sell_before_close_guard_threshold_sweep = (
        _build_sell_before_close_guard_threshold_sweep_report(
            dataset=dataset,
            prediction_sets=counterfactual_prediction_sets,
            config=config,
            calibration_error=float(calibration["calibration_error"]),
            calibration_split=primary_calibration_split,
            replay_split=replay_split,
        )
    )
    signal_sanity = _action_value_signal_sanity_report(
        validation_predictions=validation_predictions,
        shadow_predictions=shadow_predictions,
        action_value_calibration=action_value_calibration,
        action_family_eligibility=action_family_eligibility,
        hold_to_settlement_longshot_guard=hold_to_settlement_longshot_guard,
    )
    model_ranking_error = build_model_ranking_error_report(
        validation_examples=dataset.validation_examples,
        validation_predictions=validation_predictions,
        shadow_examples=dataset.shadow_examples,
        shadow_predictions=shadow_predictions,
    )
    model_ranking_candidate_comparison = build_model_ranking_candidate_comparison(
        validation_examples=dataset.validation_examples,
        raw_validation_predictions=raw_validation_predictions,
        calibrated_validation_predictions=validation_predictions,
        shadow_examples=dataset.shadow_examples,
        raw_shadow_predictions=raw_shadow_predictions,
        calibrated_shadow_predictions=shadow_predictions,
        execution_buffer=float(config.ev_threshold),
    )
    action_representation_diagnostic = build_action_representation_diagnostic_report(
        validation_examples=dataset.validation_examples,
        validation_predictions=validation_predictions,
        shadow_examples=dataset.shadow_examples,
        shadow_predictions=shadow_predictions,
        execution_buffer=float(config.ev_threshold),
    )
    ranking_overlay_zero_entry_diagnostic = (
        build_ranking_overlay_zero_entry_diagnostic_report(
            validation_examples=dataset.validation_examples,
            raw_validation_predictions=raw_validation_predictions,
            calibrated_validation_predictions=validation_predictions,
            shadow_examples=dataset.shadow_examples,
            raw_shadow_predictions=raw_shadow_predictions,
            calibrated_shadow_predictions=shadow_predictions,
            execution_buffer=float(config.ev_threshold),
        )
    )
    sell_before_close_p_up_disagreement_diagnostic = (
        build_sell_before_close_p_up_disagreement_diagnostic_report(
            shadow_examples=dataset.shadow_examples,
            model_ranking_candidate_comparison=model_ranking_candidate_comparison,
            action_family_counterfactual_replays=action_family_counterfactual_replays,
            pnl_notional=float(config.max_paper_notional),
        )
    )
    sell_before_close_exit_reliability = (
        build_sell_before_close_exit_reliability_report(
            dataset=dataset,
            action_family_counterfactual_replays=action_family_counterfactual_replays,
        )
    )
    sell_before_close_promotion_support_gate = (
        build_sell_before_close_promotion_support_gate_report(
            action_family_counterfactual_replays=(
                action_family_counterfactual_replays
            ),
            sell_before_close_exit_reliability=sell_before_close_exit_reliability,
        )
    )
    _apply_exit_reliability_guard_to_candidate_comparison(
        model_ranking_candidate_comparison=model_ranking_candidate_comparison,
        sell_before_close_exit_reliability=sell_before_close_exit_reliability,
        sell_before_close_promotion_support_gate=(
            sell_before_close_promotion_support_gate
        ),
        sell_before_close_support_aware_threshold_selection=(
            sell_before_close_support_aware_threshold_selection
        ),
    )
    source_model_eligibility = build_source_model_eligibility_report(
        signal_sanity=signal_sanity,
        action_value_calibration=action_value_calibration,
        action_family_eligibility=action_family_eligibility,
        model_ranking_candidate_comparison=model_ranking_candidate_comparison,
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
        action_family_counterfactual_replays=action_family_counterfactual_replays,
        signal_sanity=signal_sanity,
        model_ranking_error=model_ranking_error,
        model_ranking_candidate_comparison=model_ranking_candidate_comparison,
        action_representation_diagnostic=action_representation_diagnostic,
        ranking_overlay_zero_entry_diagnostic=ranking_overlay_zero_entry_diagnostic,
        source_model_eligibility=source_model_eligibility,
        sell_before_close_p_up_disagreement_diagnostic=(
            sell_before_close_p_up_disagreement_diagnostic
        ),
        sell_before_close_exit_reliability=sell_before_close_exit_reliability,
        sell_before_close_promotion_support_gate=(
            sell_before_close_promotion_support_gate
        ),
        sell_before_close_support_aware_threshold_selection=(
            sell_before_close_support_aware_threshold_selection
        ),
        sell_before_close_support_aware_threshold_failure_attribution=(
            sell_before_close_support_aware_threshold_failure_attribution
        ),
        sell_before_close_validation_failure_drilldown=(
            sell_before_close_validation_failure_drilldown
        ),
        sell_before_close_guard_threshold_sweep=(
            sell_before_close_guard_threshold_sweep
        ),
    )
    model_sha256 = _sha256_file(artifact_paths["model"])
    action_value_calibration_sha256 = _sha256_file(
        artifact_paths["action_value_calibration"]
    )
    action_family_artifact_hashes = {
        "action_family_eligibility_sha256": _sha256_file(
            artifact_paths["action_family_eligibility_report"]
        ),
        "hold_to_settlement_longshot_guard_sha256": _sha256_file(
            artifact_paths["hold_to_settlement_longshot_guard_report"]
        ),
        "action_family_replay_variants_sha256": _sha256_file(
            artifact_paths["action_family_replay_variants_report"]
        ),
        "action_family_counterfactual_replay_sha256": _sha256_file(
            artifact_paths["action_family_counterfactual_replay_report"]
        ),
        "model_ranking_error_report_sha256": _sha256_file(
            artifact_paths["model_ranking_error_report"]
        ),
        "model_ranking_candidate_comparison_sha256": _sha256_file(
            artifact_paths["model_ranking_candidate_comparison"]
        ),
        "action_representation_diagnostic_sha256": _sha256_file(
            artifact_paths["action_representation_diagnostic_report"]
        ),
        "ranking_overlay_zero_entry_diagnostic_sha256": _sha256_file(
            artifact_paths["ranking_overlay_zero_entry_diagnostic_report"]
        ),
        "source_model_eligibility_report_sha256": _sha256_file(
            artifact_paths["source_model_eligibility_report"]
        ),
        "sell_before_close_p_up_disagreement_diagnostic_sha256": _sha256_file(
            artifact_paths["sell_before_close_p_up_disagreement_diagnostic_report"]
        ),
        "sell_before_close_exit_reliability_report_sha256": _sha256_file(
            artifact_paths["sell_before_close_exit_reliability_report"]
        ),
        "sell_before_close_guard_threshold_sweep_report_sha256": _sha256_file(
            artifact_paths["sell_before_close_guard_threshold_sweep_report"]
        ),
        "sell_before_close_promotion_support_gate_report_sha256": _sha256_file(
            artifact_paths["sell_before_close_promotion_support_gate_report"]
        ),
        "sell_before_close_support_aware_threshold_selection_report_sha256": (
            _sha256_file(
                artifact_paths[
                    "sell_before_close_support_aware_threshold_selection_report"
                ]
            )
        ),
        "sell_before_close_support_aware_threshold_failure_attribution_report_sha256": (
            _sha256_file(
                artifact_paths[
                    "sell_before_close_support_aware_threshold_failure_attribution_report"
                ]
            )
        ),
        "sell_before_close_validation_failure_drilldown_report_sha256": (
            _sha256_file(
                artifact_paths[
                    "sell_before_close_validation_failure_drilldown_report"
                ]
            )
        ),
        "sell_before_close_side_balanced_candidate_report_sha256": _sha256_file(
            artifact_paths["sell_before_close_side_balanced_candidate_report"]
        ),
    }
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
        action_family_artifact_hashes=action_family_artifact_hashes,
        source_model_eligibility=source_model_eligibility,
        sell_before_close_p_up_disagreement_diagnostic=(
            sell_before_close_p_up_disagreement_diagnostic
        ),
        sell_before_close_exit_reliability=sell_before_close_exit_reliability,
        sell_before_close_promotion_support_gate=(
            sell_before_close_promotion_support_gate
        ),
        sell_before_close_support_aware_threshold_selection=(
            sell_before_close_support_aware_threshold_selection
        ),
        sell_before_close_support_aware_threshold_failure_attribution=(
            sell_before_close_support_aware_threshold_failure_attribution
        ),
        sell_before_close_validation_failure_drilldown=(
            sell_before_close_validation_failure_drilldown
        ),
        sell_before_close_guard_threshold_sweep=(
            sell_before_close_guard_threshold_sweep
        ),
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
        action_family_counterfactual_replay_report=(
            _read_json(artifact_paths["action_family_counterfactual_replay_report"])
        ),
        model_ranking_error_report=model_ranking_error,
        model_ranking_candidate_comparison_report=model_ranking_candidate_comparison,
        action_representation_diagnostic_report=action_representation_diagnostic,
        ranking_overlay_zero_entry_diagnostic_report=(
            ranking_overlay_zero_entry_diagnostic
        ),
        source_model_eligibility_report=source_model_eligibility,
        sell_before_close_p_up_disagreement_diagnostic_report=(
            sell_before_close_p_up_disagreement_diagnostic
        ),
        sell_before_close_exit_reliability_report=(
            sell_before_close_exit_reliability
        ),
        sell_before_close_promotion_support_gate_report=(
            sell_before_close_promotion_support_gate
        ),
        sell_before_close_support_aware_threshold_selection_report=(
            sell_before_close_support_aware_threshold_selection
        ),
        sell_before_close_support_aware_threshold_failure_attribution_report=(
            sell_before_close_support_aware_threshold_failure_attribution
        ),
        sell_before_close_validation_failure_drilldown_report=(
            sell_before_close_validation_failure_drilldown
        ),
        sell_before_close_guard_threshold_sweep_report=(
            sell_before_close_guard_threshold_sweep
        ),
    )


def _build_action_family_counterfactual_replays(
    *,
    dataset: Any,
    prediction_sets: tuple[dict[str, Any], ...],
    config: PolymarketPolicyTrainingConfig,
    calibration_error: float,
    calibration_split: str,
    replay_split: str,
) -> tuple[dict[str, Any], ...]:
    replays = []
    for prediction_set in prediction_sets:
        replay_config = replace(
            config,
            ev_threshold=float(prediction_set["ev_threshold"]),
        )
        predictions = tuple(prediction_set["predictions"])
        exit_reliability_guard_summary = None
        if bool(prediction_set.get("exit_reliability_guard_enabled", False)):
            decisions, exit_reliability_guard_summary = (
                build_sell_before_close_exit_reliability_guard_decisions(
                    predictions=predictions,
                    config=replay_config,
                    thresholds=prediction_set.get("entry_filter_thresholds"),
                    exit_policy=str(prediction_set["exit_policy"]),
                    candidate_name=str(prediction_set["variant"]),
                    p_up_side_alignment_filter_enabled=bool(
                        prediction_set.get(
                            "p_up_side_alignment_filter_enabled",
                            False,
                        )
                    ),
                )
            )
        else:
            decisions = build_polymarket_ev_decisions(
                predictions=predictions,
                config=replay_config,
            )
        ev_report = ev_threshold_report(decisions, replay_split=replay_split)
        replay_report = run_polymarket_policy_replay(
            dataset=dataset,
            decisions=decisions,
            config=replay_config,
            calibration_error=calibration_error,
            calibration_split=calibration_split,
            replay_split=replay_split,
            prediction_count=len(predictions),
        )
        prediction_set_for_summary = {
            **prediction_set,
            "exit_reliability_guard_summary": (
                exit_reliability_guard_summary or {}
            ),
        }
        ledger_pnl_report = _counterfactual_ledger_pnl_report(
            prediction_set=prediction_set_for_summary,
            ev_report=ev_report,
            replay_report=replay_report,
        )
        replays.append(
            {
                "variant": prediction_set["variant"],
                "description": prediction_set["description"],
                "counterfactual_replay_mode": prediction_set[
                    "counterfactual_replay_mode"
                ],
                "allowed_mode": prediction_set["allowed_mode"],
                "ev_threshold": prediction_set["ev_threshold"],
                "eligible_action_families": prediction_set[
                    "eligible_action_families"
                ],
                "family_gate_results": prediction_set["family_gate_results"],
                "exit_reliability_guard_enabled": bool(
                    prediction_set.get("exit_reliability_guard_enabled", False)
                ),
                "p_up_side_alignment_filter_enabled": bool(
                    prediction_set.get("p_up_side_alignment_filter_enabled", False)
                ),
                "exit_policy": prediction_set.get("exit_policy"),
                "entry_filter_thresholds": dict(
                    prediction_set.get("entry_filter_thresholds", {})
                ),
                "exit_reliability_guard_summary": (
                    exit_reliability_guard_summary or {}
                ),
                "side_balance_selection_summary": dict(
                    prediction_set.get("side_balance_selection_summary", {})
                ),
                "predictions": [prediction.to_dict() for prediction in predictions],
                "decisions": [decision.to_dict() for decision in decisions],
                "ev_report": ev_report,
                "replay_report": replay_report,
                "ledger_pnl_report": ledger_pnl_report,
                "summary": _counterfactual_variant_summary(
                    prediction_set=prediction_set_for_summary,
                    ev_report=ev_report,
                    replay_report=replay_report,
                    ledger_pnl_report=ledger_pnl_report,
                ),
            }
        )
    return tuple(replays)


def _build_sell_before_close_guard_threshold_sweep_report(
    *,
    dataset: Any,
    prediction_sets: tuple[dict[str, Any], ...],
    config: PolymarketPolicyTrainingConfig,
    calibration_error: float,
    calibration_split: str,
    replay_split: str,
) -> dict[str, Any]:
    prediction_set = next(
        (
            candidate
            for candidate in prediction_sets
            if candidate["variant"]
            == SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME
        ),
        None,
    )
    if prediction_set is None:
        report = {
            "schema_version": (
                "bigan-v8-polymarket-sell-before-close-guard-threshold-sweep-v1"
            ),
            "candidate_name": SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
            "diagnostic_only": True,
            "uses_shadow_for_fit": False,
            "promotion_evidence_eligible": False,
            "paper_run_resume_allowed": False,
            "sweep_grid": SELL_BEFORE_CLOSE_GUARD_THRESHOLD_SWEEP_GRID,
            "row_count": 0,
            "rows": [],
            "best_threshold_sweep_row": None,
            "reason_codes": ["k_candidate_prediction_set_missing"],
            **compact_safety_fields(),
        }
        report["sell_before_close_guard_threshold_sweep_report_id"] = (
            canonical_json_sha256(report)
        )
        return report

    rows = []
    grid_items = list(SELL_BEFORE_CLOSE_GUARD_THRESHOLD_SWEEP_GRID.items())
    for values in product(*(item[1] for item in grid_items)):
        thresholds = dict(SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS)
        thresholds.update(
            {name: float(value) for (name, _), value in zip(grid_items, values, strict=True)}
        )
        replay_config = replace(
            config,
            ev_threshold=float(prediction_set["ev_threshold"]),
        )
        decisions, guard_summary = build_sell_before_close_exit_reliability_guard_decisions(
            predictions=tuple(prediction_set["predictions"]),
            config=replay_config,
            thresholds=thresholds,
            exit_policy=str(prediction_set["exit_policy"]),
            candidate_name=SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
            p_up_side_alignment_filter_enabled=True,
        )
        ev_report = ev_threshold_report(decisions, replay_split=replay_split)
        replay_report = run_polymarket_policy_replay(
            dataset=dataset,
            decisions=decisions,
            config=replay_config,
            calibration_error=calibration_error,
            calibration_split=calibration_split,
            replay_split=replay_split,
            prediction_count=len(prediction_set["predictions"]),
        )
        action_counts = ev_report["action_counts"]
        entry_count = int(action_counts.get("BUY_UP", 0)) + int(
            action_counts.get("BUY_DOWN", 0)
        )
        sell_count = int(action_counts.get("SELL_UP", 0)) + int(
            action_counts.get("SELL_DOWN", 0)
        )
        residual_count = int(replay_report["settled_position_count"])
        residual_drag = min(0.0, float(replay_report["settlement_pnl"]))
        support = evaluate_sell_before_close_promotion_support(
            candidate_name=SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
            decisions=[decision.to_dict() for decision in decisions],
            replay_report=replay_report,
            exit_reliability_summary=guard_summary,
        )
        p_up_within_limit = bool(
            guard_summary[
                "candidate_scoped_p_up_action_disagreement_within_limit"
            ]
        )
        would_be_source_eligible = (
            entry_count > 0
            and float(replay_report["total_polymarket_pnl"]) > 0.0
            and residual_drag >= 0.0
            and residual_count == 0
            and p_up_within_limit
        )
        rows.append(
            {
                "candidate_name": SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
                "thresholds": thresholds,
                "entry_count": entry_count,
                "sell_count": sell_count,
                "residual_count": residual_count,
                "realized_trade_pnl": replay_report["realized_trade_pnl"],
                "settlement_pnl": replay_report["settlement_pnl"],
                "total_pnl": replay_report["total_polymarket_pnl"],
                "max_drawdown": replay_report["max_drawdown"],
                "p_up_disagreement_count": guard_summary[
                    "candidate_scoped_p_up_action_disagreement_count"
                ],
                "p_up_disagreement_denominator": guard_summary[
                    "candidate_scoped_p_up_action_disagreement_denominator"
                ],
                "p_up_disagreement_rate": guard_summary[
                    "candidate_scoped_p_up_action_disagreement_rate"
                ],
                "p_up_disagreement_within_limit": p_up_within_limit,
                "residual_settlement_drag": residual_drag,
                "support_gate_passed": support["support_gate_passed"],
                "support_gate_reason_codes": support["support_gate_reason_codes"],
                "promotion_support_eligible": support[
                    "promotion_support_eligible"
                ],
                "unique_market_count": support["unique_market_count"],
                "side_count": support["side_count"],
                "side_distribution": support["side_distribution"],
                "mean_pnl_per_entry": support["mean_pnl_per_entry"],
                "entry_decision_count_before_guard": guard_summary[
                    "entry_decision_count_before_guard"
                ],
                "entry_decision_count_after_exit_guard": guard_summary[
                    "entry_decision_count_after_exit_guard"
                ],
                "entry_decision_count_after_p_up_alignment": guard_summary[
                    "entry_decision_count_after_p_up_alignment"
                ],
                "entry_filter_blocked_count": guard_summary[
                    "entry_filter_blocked_count"
                ],
                "entry_filter_blocked_by_p_up_alignment_count": guard_summary[
                    "entry_filter_blocked_by_p_up_alignment_count"
                ],
                "entry_filter_blocked_by_quality_count": guard_summary[
                    "entry_filter_blocked_by_quality_count"
                ],
                "reentry_blocked_count": guard_summary[
                    "reentry_blocked_count"
                ],
                "would_be_source_eligible_under_existing_gates": (
                    would_be_source_eligible
                ),
                "promotion_evidence_eligible": False,
                "paper_run_resume_allowed": False,
                **compact_safety_fields(),
            }
        )
    best = (
        sorted(
            rows,
            key=lambda row: (
                not bool(row["would_be_source_eligible_under_existing_gates"]),
                -float(row["total_pnl"]),
                int(row["residual_count"]),
                float(row["p_up_disagreement_rate"]),
                tuple(sorted(row["thresholds"].items())),
            ),
        )[0]
        if rows
        else None
    )
    report = {
        "schema_version": (
            "bigan-v8-polymarket-sell-before-close-guard-threshold-sweep-v1"
        ),
        "candidate_name": SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
        "diagnostic_only": True,
        "uses_shadow_for_fit": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "sweep_grid": SELL_BEFORE_CLOSE_GUARD_THRESHOLD_SWEEP_GRID,
        "row_count": len(rows),
        "rows": rows,
        "best_threshold_sweep_row": best,
        "best_threshold_sweep_support_gate_passed": None
        if best is None
        else best["support_gate_passed"],
        "best_threshold_sweep_support_gate_reason_codes": []
        if best is None
        else best["support_gate_reason_codes"],
        "reason_codes": [
            "diagnostic_only",
            "shadow_sweep_not_used_for_threshold_fit",
            "promotion_replay_gate_required",
        ],
        **compact_safety_fields(),
    }
    report["sell_before_close_guard_threshold_sweep_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _counterfactual_ledger_pnl_report(
    *,
    prediction_set: dict[str, Any],
    ev_report: dict[str, Any],
    replay_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": (
            "bigan-v8-polymarket-action-family-counterfactual-ledger-pnl-v1"
        ),
        "variant": prediction_set["variant"],
        "counterfactual_replay_mode": prediction_set["counterfactual_replay_mode"],
        "allowed_mode": prediction_set["allowed_mode"],
        "ev_threshold": prediction_set["ev_threshold"],
        "exit_reliability_guard_enabled": bool(
            prediction_set.get("exit_reliability_guard_enabled", False)
        ),
        "p_up_side_alignment_filter_enabled": bool(
            prediction_set.get("p_up_side_alignment_filter_enabled", False)
        ),
        "exit_policy": prediction_set.get("exit_policy"),
        "entry_filter_thresholds": dict(
            prediction_set.get("entry_filter_thresholds", {})
        ),
        "threshold_selection_failed": bool(
            prediction_set.get("threshold_selection_failed", False)
        ),
        "threshold_selection_method": prediction_set.get(
            "threshold_selection_method"
        ),
        "threshold_selection_fit_split": prediction_set.get(
            "threshold_selection_fit_split"
        ),
        "threshold_selection_evaluation_split": prediction_set.get(
            "threshold_selection_evaluation_split"
        ),
        "uses_shadow_for_fit": prediction_set.get("uses_shadow_for_fit"),
        "shadow_sweep_not_used_for_threshold_fit": prediction_set.get(
            "shadow_sweep_not_used_for_threshold_fit"
        ),
        "support_aware_threshold_selection_summary": dict(
            prediction_set.get("support_aware_threshold_selection_summary", {})
        ),
        "side_balance_selection_summary": dict(
            prediction_set.get("side_balance_selection_summary", {})
        ),
        "exit_reliability_guard_summary": dict(
            prediction_set.get("exit_reliability_guard_summary", {})
        ),
        "prediction_count": int(prediction_set["prediction_count"]),
        "decision_count": ev_report["decision_count"],
        "action_counts": ev_report["action_counts"],
        "reason_counts": ev_report["reason_counts"],
        "trade_count": replay_report["trade_count"],
        "no_trade_count": replay_report["no_trade_count"],
        "settled_position_count": replay_report["settled_position_count"],
        "realized_trade_pnl": replay_report["realized_trade_pnl"],
        "settlement_pnl": replay_report["settlement_pnl"],
        "complete_set_pnl": replay_report["complete_set_pnl"],
        "fees": replay_report["fees"],
        "slippage": replay_report["slippage"],
        "total_polymarket_pnl": replay_report["total_polymarket_pnl"],
        "max_drawdown": replay_report["max_drawdown"],
        "ledger_event_count": replay_report["ledger_event_count"],
        "settlement_event_count": replay_report["settlement_event_count"],
        "phase1_position_ledger_used": replay_report["phase1_position_ledger_used"],
        "phase1_settlement_engine_used": replay_report["phase1_settlement_engine_used"],
        "promotion_evidence_eligible": False,
        "promotion_evidence_ineligible_reasons": [
            "source_model_paper_decision_ineligible"
        ],
        **compact_safety_fields(),
    }


def _counterfactual_variant_summary(
    *,
    prediction_set: dict[str, Any],
    ev_report: dict[str, Any],
    replay_report: dict[str, Any],
    ledger_pnl_report: dict[str, Any],
) -> dict[str, Any]:
    action_counts = ev_report["action_counts"]
    entry_decision_count = int(action_counts.get("BUY_UP", 0)) + int(
        action_counts.get("BUY_DOWN", 0)
    )
    blocked_reasons = ["source_model_paper_decision_ineligible"]
    if float(replay_report["total_polymarket_pnl"]) <= 0.0:
        blocked_reasons.append("counterfactual_replay_pnl_not_positive")
    if entry_decision_count <= 0:
        blocked_reasons.append("counterfactual_replay_no_entry_decisions")
    return {
        "variant": prediction_set["variant"],
        "description": prediction_set["description"],
        "counterfactual_replay_mode": prediction_set["counterfactual_replay_mode"],
        "allowed_mode": prediction_set["allowed_mode"],
        "ev_threshold": prediction_set["ev_threshold"],
        "eligible_action_families": prediction_set["eligible_action_families"],
        "exit_reliability_guard_enabled": bool(
            prediction_set.get("exit_reliability_guard_enabled", False)
        ),
        "p_up_side_alignment_filter_enabled": bool(
            prediction_set.get("p_up_side_alignment_filter_enabled", False)
        ),
        "exit_policy": prediction_set.get("exit_policy"),
        "entry_filter_thresholds": dict(
            prediction_set.get("entry_filter_thresholds", {})
        ),
        "threshold_selection_failed": bool(
            prediction_set.get("threshold_selection_failed", False)
        ),
        "threshold_selection_method": prediction_set.get(
            "threshold_selection_method"
        ),
        "threshold_selection_fit_split": prediction_set.get(
            "threshold_selection_fit_split"
        ),
        "threshold_selection_evaluation_split": prediction_set.get(
            "threshold_selection_evaluation_split"
        ),
        "uses_shadow_for_fit": prediction_set.get("uses_shadow_for_fit"),
        "shadow_sweep_not_used_for_threshold_fit": prediction_set.get(
            "shadow_sweep_not_used_for_threshold_fit"
        ),
        "support_aware_threshold_selection_summary": dict(
            prediction_set.get("support_aware_threshold_selection_summary", {})
        ),
        "side_balance_selection_summary": dict(
            prediction_set.get("side_balance_selection_summary", {})
        ),
        "exit_reliability_guard_summary": dict(
            prediction_set.get("exit_reliability_guard_summary", {})
        ),
        "prediction_count": int(prediction_set["prediction_count"]),
        "decision_count": ev_report["decision_count"],
        "entry_decision_count": entry_decision_count,
        "trade_count": replay_report["trade_count"],
        "no_trade_count": replay_report["no_trade_count"],
        "action_counts": action_counts,
        "reason_counts": ev_report["reason_counts"],
        "total_polymarket_pnl": replay_report["total_polymarket_pnl"],
        "realized_trade_pnl": replay_report["realized_trade_pnl"],
        "settlement_pnl": replay_report["settlement_pnl"],
        "fees": replay_report["fees"],
        "slippage": replay_report["slippage"],
        "max_drawdown": replay_report["max_drawdown"],
        "ledger_event_count": ledger_pnl_report["ledger_event_count"],
        "settlement_event_count": ledger_pnl_report["settlement_event_count"],
        "promotion_evidence_eligible": False,
        "promotion_evidence_ineligible_reasons": sorted(set(blocked_reasons)),
        "blocked": True,
        "blocked_reasons": sorted(set(blocked_reasons)),
        **compact_safety_fields(),
    }


def _apply_exit_reliability_guard_to_candidate_comparison(
    *,
    model_ranking_candidate_comparison: dict[str, Any],
    sell_before_close_exit_reliability: dict[str, Any],
    sell_before_close_promotion_support_gate: dict[str, Any],
    sell_before_close_support_aware_threshold_selection: dict[str, Any],
) -> None:
    guard_summaries = {
        SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME: (
            sell_before_close_exit_reliability.get(
                "exit_reliability_guard_candidate_summary"
            )
        ),
        SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME: (
            sell_before_close_exit_reliability.get(
                "exit_reliability_p_up_aligned_candidate_summary"
            )
        ),
        SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME: (
            sell_before_close_exit_reliability.get(
                "exit_reliability_support_aware_p_up_aligned_candidate_summary"
            )
        ),
        SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME: (
            sell_before_close_exit_reliability.get(
                "exit_reliability_side_balanced_candidate_summary"
            )
        ),
    }
    guard_summaries = {
        name: summary for name, summary in guard_summaries.items() if summary
    }
    if not guard_summaries:
        _refresh_candidate_comparison_rollups(model_ranking_candidate_comparison)
        return
    comparison_rows = sell_before_close_exit_reliability.get(
        "i_vs_j_vs_k_vs_l_vs_m_replay_comparison",
        sell_before_close_exit_reliability.get(
            "i_vs_j_vs_k_vs_l_replay_comparison",
            [],
        ),
    )
    legacy_comparison_rows = sell_before_close_exit_reliability.get(
        "i_vs_j_vs_k_vs_l_replay_comparison",
        sell_before_close_exit_reliability.get(
            "i_vs_j_vs_k_replay_comparison",
            [],
        ),
    )
    support_rows = {
        row["candidate_name"]: row
        for row in sell_before_close_promotion_support_gate.get(
            "candidate_rows",
            [],
        )
    }
    support_aware_selection_summary = (
        sell_before_close_support_aware_threshold_selection_summary(
            sell_before_close_support_aware_threshold_selection
        )
    )
    support_aware_selection_failed = bool(
        sell_before_close_support_aware_threshold_selection.get(
            "selection_reason_codes",
            [],
        )
    )
    support_aware_selection_reasons = list(
        sell_before_close_support_aware_threshold_selection.get(
            "selection_reason_codes",
            [],
        )
    )
    for candidate in model_ranking_candidate_comparison["candidates"]:
        guard_summary = guard_summaries.get(candidate["candidate_name"])
        if not guard_summary:
            continue
        support = support_rows.get(candidate["candidate_name"], {})
        comparison = next(
            (
                row
                for row in comparison_rows
                if row["candidate_name"] == candidate["candidate_name"]
            ),
            {},
        )
        candidate["exit_reliability_guard_enabled"] = True
        candidate["p_up_side_alignment_filter_enabled"] = bool(
            guard_summary.get("p_up_side_alignment_filter_enabled", False)
        )
        candidate["exit_policy"] = guard_summary["exit_policy"]
        candidate["entry_filter_thresholds"] = guard_summary[
            "entry_filter_thresholds"
        ]
        candidate["entry_decision_count_before_guard"] = guard_summary[
            "entry_decision_count_before_guard"
        ]
        candidate["entry_decision_count_after_exit_guard"] = guard_summary[
            "entry_decision_count_after_exit_guard"
        ]
        candidate["entry_decision_count_after_p_up_alignment"] = guard_summary[
            "entry_decision_count_after_p_up_alignment"
        ]
        candidate["entry_decision_count_after_guard"] = guard_summary[
            "entry_decision_count_after_guard"
        ]
        candidate["entry_filter_blocked_count"] = guard_summary[
            "entry_filter_blocked_count"
        ]
        candidate["entry_filter_blocked_by_p_up_alignment_count"] = (
            guard_summary["entry_filter_blocked_by_p_up_alignment_count"]
        )
        candidate["entry_filter_blocked_by_quality_count"] = guard_summary[
            "entry_filter_blocked_by_quality_count"
        ]
        candidate["reentry_cooldown_seconds"] = guard_summary[
            "reentry_cooldown_seconds"
        ]
        candidate["reentry_blocked_count"] = guard_summary["reentry_blocked_count"]
        candidate["entries_per_market_distribution"] = guard_summary[
            "entries_per_market_distribution"
        ]
        candidate["positions_opened_count"] = guard_summary[
            "positions_opened_count"
        ]
        candidate["positions_closed_before_settlement_count"] = guard_summary[
            "positions_closed_before_settlement_count"
        ]
        candidate["positions_opened_but_not_closed_before_settlement"] = (
            guard_summary["positions_opened_but_not_closed_before_settlement"]
        )
        candidate["replay_realized_trade_pnl"] = guard_summary[
            "replay_realized_trade_pnl"
        ]
        candidate["replay_settlement_pnl"] = guard_summary["replay_settlement_pnl"]
        candidate["replay_total_polymarket_pnl"] = guard_summary[
            "replay_total_polymarket_pnl"
        ]
        candidate["replay_residual_settlement_drag"] = guard_summary[
            "replay_residual_settlement_drag"
        ]
        candidate["candidate_scoped_p_up_action_disagreement_count"] = (
            guard_summary["candidate_scoped_p_up_action_disagreement_count"]
        )
        candidate["candidate_scoped_p_up_action_disagreement_denominator"] = (
            guard_summary["candidate_scoped_p_up_action_disagreement_denominator"]
        )
        candidate["candidate_scoped_p_up_action_disagreement_rate"] = (
            guard_summary["candidate_scoped_p_up_action_disagreement_rate"]
        )
        candidate["candidate_scoped_p_up_action_disagreement_within_limit"] = (
            guard_summary[
                "candidate_scoped_p_up_action_disagreement_within_limit"
            ]
        )
        candidate["p_up_action_disagreement_rate"] = candidate[
            "candidate_scoped_p_up_action_disagreement_rate"
        ]
        candidate["p_up_action_disagreement_within_limit"] = candidate[
            "candidate_scoped_p_up_action_disagreement_within_limit"
        ]
        candidate["replay_total_pnl_improved_vs_i_candidate"] = bool(
            comparison.get("total_pnl_improved_vs_i_candidate", False)
        )
        candidate["promotion_support_eligible"] = bool(
            support.get("promotion_support_eligible", False)
        )
        candidate["promotion_support_gate_passed"] = bool(
            support.get("support_gate_passed", False)
        )
        candidate["promotion_support_reason_codes"] = list(
            support.get("support_gate_reason_codes", [])
        )
        candidate["promotion_support_thresholds"] = dict(
            sell_before_close_promotion_support_gate.get("thresholds", {})
        )
        candidate["promotion_replay_entry_decision_count"] = support.get(
            "entry_decision_count"
        )
        candidate["promotion_replay_sell_decision_count"] = support.get(
            "sell_decision_count"
        )
        candidate["promotion_replay_unique_market_count"] = support.get(
            "unique_market_count"
        )
        candidate["promotion_replay_side_count"] = support.get("side_count")
        candidate["promotion_replay_side_distribution"] = dict(
            support.get("side_distribution", {})
        )
        candidate["promotion_replay_mean_pnl_per_entry"] = support.get(
            "mean_pnl_per_entry"
        )
        guard_side_balance_summary = dict(
            guard_summary.get("side_balance_selection_summary", {})
        )
        candidate["side_balance_required"] = bool(
            guard_summary.get(
                "side_balance_required",
                support.get(
                    "side_balance_required",
                    candidate.get("side_balance_required", False),
                ),
            )
        )
        candidate["side_balance_gate_passed"] = bool(
            support.get(
                "side_balance_gate_passed",
                guard_summary.get(
                    "side_balance_gate_passed",
                    candidate.get("side_balance_gate_passed", False),
                ),
            )
        )
        candidate["side_balance_thresholds"] = dict(
            support.get(
                "side_balance_thresholds",
                guard_summary.get(
                "side_balance_selection_summary",
                {},
                ).get(
                    "side_balance_thresholds",
                    candidate.get("side_balance_thresholds", {}),
                ),
            )
        )
        if guard_side_balance_summary:
            candidate["side_balance_selection_summary"] = guard_side_balance_summary
        candidate["promotion_replay_up_entry_count"] = support.get("up_entry_count")
        candidate["promotion_replay_down_entry_count"] = support.get(
            "down_entry_count"
        )
        candidate["promotion_replay_up_market_count"] = support.get(
            "up_market_count"
        )
        candidate["promotion_replay_down_market_count"] = support.get(
            "down_market_count"
        )
        candidate["promotion_replay_side_entry_ratio"] = support.get(
            "side_entry_ratio"
        )
        if (
            candidate["candidate_name"]
            == SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
        ):
            candidate["threshold_selection_method"] = support_aware_selection_summary[
                "threshold_selection_method"
            ]
            candidate["threshold_selection_fit_split"] = support_aware_selection_summary[
                "threshold_selection_fit_split"
            ]
            candidate["threshold_selection_evaluation_split"] = (
                support_aware_selection_summary[
                    "threshold_selection_evaluation_split"
                ]
            )
            candidate["uses_shadow_for_fit"] = support_aware_selection_summary[
                "uses_shadow_for_fit"
            ]
            candidate["shadow_sweep_not_used_for_threshold_fit"] = (
                support_aware_selection_summary[
                    "shadow_sweep_not_used_for_threshold_fit"
                ]
            )
            candidate["threshold_selection_passed"] = (
                support_aware_selection_summary["threshold_selection_passed"]
            )
            candidate["threshold_selection_failed"] = (
                support_aware_selection_summary["threshold_selection_failed"]
            )
            candidate["threshold_selection_failure_reason_codes"] = list(
                support_aware_selection_summary[
                    "threshold_selection_failure_reason_codes"
                ]
            )
            candidate["support_aware_threshold_selection_failed"] = (
                support_aware_selection_failed
            )
            candidate["support_aware_threshold_selection_reason_codes"] = (
                support_aware_selection_reasons
            )
            candidate["support_aware_threshold_selection_summary"] = (
                support_aware_selection_summary
            )
        reasons = set(candidate["ineligible_reason_codes"])
        reasons.discard("p_up_action_disagreement_excessive")
        if not candidate["candidate_scoped_p_up_action_disagreement_within_limit"]:
            reasons.add("p_up_action_disagreement_excessive")
        if float(candidate["replay_total_polymarket_pnl"]) <= 0.0:
            reasons.add("exit_reliability_guard_replay_pnl_not_positive")
        if float(candidate["replay_residual_settlement_drag"]) < 0.0:
            reasons.add("exit_reliability_guard_residual_settlement_drag_negative")
        if (
            int(candidate["positions_opened_but_not_closed_before_settlement"])
            > 0
        ):
            reasons.add("exit_reliability_guard_residual_positions_remaining")
        if not candidate["promotion_support_eligible"]:
            reasons.update(candidate["promotion_support_reason_codes"])
        if candidate.get("side_balance_required") and not candidate.get(
            "side_balance_gate_passed",
            False,
        ):
            reasons.add("side_balance_required_gate_failed")
        if (
            candidate["candidate_name"]
            == SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
            and support_aware_selection_failed
        ):
            reasons.add("support_aware_threshold_selection_failed")
            reasons.update(support_aware_selection_reasons)
        candidate["ineligible_reason_codes"] = sorted(reasons)
        candidate["source_model_eligible"] = not reasons
        candidate["source_model_candidate_eligible"] = not reasons
        candidate["paper_run_resume_allowed"] = False
        if candidate.get("candidate_manifest"):
            manifest = candidate["candidate_manifest"]
            manifest["exit_reliability_guard_enabled"] = True
            manifest["p_up_side_alignment_filter_enabled"] = candidate[
                "p_up_side_alignment_filter_enabled"
            ]
            manifest["exit_policy"] = candidate["exit_policy"]
            manifest["entry_filter_thresholds"] = candidate[
                "entry_filter_thresholds"
            ]
            manifest["entry_decision_count_before_guard"] = candidate[
                "entry_decision_count_before_guard"
            ]
            manifest["entry_decision_count_after_exit_guard"] = candidate[
                "entry_decision_count_after_exit_guard"
            ]
            manifest["entry_decision_count_after_p_up_alignment"] = candidate[
                "entry_decision_count_after_p_up_alignment"
            ]
            manifest["entry_decision_count_after_guard"] = candidate[
                "entry_decision_count_after_guard"
            ]
            manifest["entry_filter_blocked_count"] = candidate[
                "entry_filter_blocked_count"
            ]
            manifest["entry_filter_blocked_by_p_up_alignment_count"] = (
                candidate["entry_filter_blocked_by_p_up_alignment_count"]
            )
            manifest["entry_filter_blocked_by_quality_count"] = candidate[
                "entry_filter_blocked_by_quality_count"
            ]
            manifest["reentry_cooldown_seconds"] = candidate[
                "reentry_cooldown_seconds"
            ]
            manifest["reentry_blocked_count"] = candidate["reentry_blocked_count"]
            manifest["entries_per_market_distribution"] = candidate[
                "entries_per_market_distribution"
            ]
            manifest["replay_total_polymarket_pnl"] = candidate[
                "replay_total_polymarket_pnl"
            ]
            manifest["replay_residual_settlement_drag"] = candidate[
                "replay_residual_settlement_drag"
            ]
            manifest["promotion_support_eligible"] = candidate[
                "promotion_support_eligible"
            ]
            manifest["promotion_support_gate_passed"] = candidate[
                "promotion_support_gate_passed"
            ]
            manifest["promotion_support_reason_codes"] = candidate[
                "promotion_support_reason_codes"
            ]
            manifest["promotion_support_thresholds"] = candidate[
                "promotion_support_thresholds"
            ]
            manifest["promotion_replay_entry_decision_count"] = candidate[
                "promotion_replay_entry_decision_count"
            ]
            manifest["promotion_replay_sell_decision_count"] = candidate[
                "promotion_replay_sell_decision_count"
            ]
            manifest["promotion_replay_unique_market_count"] = candidate[
                "promotion_replay_unique_market_count"
            ]
            manifest["promotion_replay_side_count"] = candidate[
                "promotion_replay_side_count"
            ]
            manifest["promotion_replay_side_distribution"] = candidate[
                "promotion_replay_side_distribution"
            ]
            manifest["promotion_replay_mean_pnl_per_entry"] = candidate[
                "promotion_replay_mean_pnl_per_entry"
            ]
            manifest["side_balance_required"] = candidate["side_balance_required"]
            manifest["side_balance_gate_passed"] = candidate[
                "side_balance_gate_passed"
            ]
            manifest["side_balance_thresholds"] = candidate[
                "side_balance_thresholds"
            ]
            manifest["side_balance_selection_summary"] = candidate[
                "side_balance_selection_summary"
            ]
            manifest["promotion_replay_up_entry_count"] = candidate[
                "promotion_replay_up_entry_count"
            ]
            manifest["promotion_replay_down_entry_count"] = candidate[
                "promotion_replay_down_entry_count"
            ]
            manifest["promotion_replay_up_market_count"] = candidate[
                "promotion_replay_up_market_count"
            ]
            manifest["promotion_replay_down_market_count"] = candidate[
                "promotion_replay_down_market_count"
            ]
            manifest["promotion_replay_side_entry_ratio"] = candidate[
                "promotion_replay_side_entry_ratio"
            ]
            if (
                candidate["candidate_name"]
                == SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
            ):
                manifest["threshold_selection_method"] = candidate[
                    "threshold_selection_method"
                ]
                manifest["threshold_selection_fit_split"] = candidate[
                    "threshold_selection_fit_split"
                ]
                manifest["threshold_selection_evaluation_split"] = candidate[
                    "threshold_selection_evaluation_split"
                ]
                manifest["uses_shadow_for_fit"] = candidate["uses_shadow_for_fit"]
                manifest["shadow_sweep_not_used_for_threshold_fit"] = candidate[
                    "shadow_sweep_not_used_for_threshold_fit"
                ]
                manifest["threshold_selection_passed"] = candidate[
                    "threshold_selection_passed"
                ]
                manifest["threshold_selection_failed"] = candidate[
                    "threshold_selection_failed"
                ]
                manifest["threshold_selection_failure_reason_codes"] = candidate[
                    "threshold_selection_failure_reason_codes"
                ]
                manifest["support_aware_threshold_selection_failed"] = candidate[
                    "support_aware_threshold_selection_failed"
                ]
                manifest["support_aware_threshold_selection_reason_codes"] = (
                    candidate["support_aware_threshold_selection_reason_codes"]
                )
                manifest["support_aware_threshold_selection_summary"] = candidate[
                    "support_aware_threshold_selection_summary"
                ]
            manifest["candidate_scoped_p_up_action_disagreement_rate"] = (
                candidate["candidate_scoped_p_up_action_disagreement_rate"]
            )
            manifest[
                "candidate_scoped_p_up_action_disagreement_within_limit"
            ] = candidate["candidate_scoped_p_up_action_disagreement_within_limit"]
            manifest["source_model_candidate_eligible"] = candidate[
                "source_model_candidate_eligible"
            ]
            manifest["action_value_paper_decision_eligible"] = candidate[
                "source_model_candidate_eligible"
            ]
            manifest["ineligible_reason_codes"] = candidate[
                "ineligible_reason_codes"
            ]
    for candidate in model_ranking_candidate_comparison["candidates"]:
        if candidate["candidate_name"] in guard_summaries:
            continue
        support = support_rows.get(candidate["candidate_name"])
        if support is None:
            continue
        candidate["promotion_support_eligible"] = bool(
            support.get("promotion_support_eligible", False)
        )
        candidate["promotion_support_gate_passed"] = bool(
            support.get("support_gate_passed", False)
        )
        candidate["promotion_support_reason_codes"] = list(
            support.get("support_gate_reason_codes", [])
        )
        candidate["promotion_support_thresholds"] = dict(
            sell_before_close_promotion_support_gate.get("thresholds", {})
        )
        candidate["promotion_replay_entry_decision_count"] = support.get(
            "entry_decision_count"
        )
        candidate["promotion_replay_sell_decision_count"] = support.get(
            "sell_decision_count"
        )
        candidate["promotion_replay_unique_market_count"] = support.get(
            "unique_market_count"
        )
        candidate["promotion_replay_side_count"] = support.get("side_count")
        candidate["promotion_replay_side_distribution"] = dict(
            support.get("side_distribution", {})
        )
        candidate["promotion_replay_mean_pnl_per_entry"] = support.get(
            "mean_pnl_per_entry"
        )
        if (
            candidate["candidate_name"]
            == SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
        ):
            candidate["threshold_selection_method"] = support_aware_selection_summary[
                "threshold_selection_method"
            ]
            candidate["threshold_selection_fit_split"] = support_aware_selection_summary[
                "threshold_selection_fit_split"
            ]
            candidate["threshold_selection_evaluation_split"] = (
                support_aware_selection_summary[
                    "threshold_selection_evaluation_split"
                ]
            )
            candidate["uses_shadow_for_fit"] = support_aware_selection_summary[
                "uses_shadow_for_fit"
            ]
            candidate["shadow_sweep_not_used_for_threshold_fit"] = (
                support_aware_selection_summary[
                    "shadow_sweep_not_used_for_threshold_fit"
                ]
            )
            candidate["threshold_selection_passed"] = (
                support_aware_selection_summary["threshold_selection_passed"]
            )
            candidate["threshold_selection_failed"] = (
                support_aware_selection_summary["threshold_selection_failed"]
            )
            candidate["threshold_selection_failure_reason_codes"] = list(
                support_aware_selection_summary[
                    "threshold_selection_failure_reason_codes"
                ]
            )
            candidate["support_aware_threshold_selection_failed"] = (
                support_aware_selection_failed
            )
            candidate["support_aware_threshold_selection_reason_codes"] = (
                support_aware_selection_reasons
            )
            candidate["support_aware_threshold_selection_summary"] = (
                support_aware_selection_summary
            )
        reasons = set(candidate["ineligible_reason_codes"])
        if not candidate["promotion_support_eligible"]:
            reasons.update(candidate["promotion_support_reason_codes"])
        if (
            candidate["candidate_name"]
            == SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
            and support_aware_selection_failed
        ):
            reasons.add("support_aware_threshold_selection_failed")
            reasons.update(support_aware_selection_reasons)
        candidate["ineligible_reason_codes"] = sorted(reasons)
        candidate["source_model_eligible"] = not reasons
        candidate["source_model_candidate_eligible"] = not reasons
        candidate["paper_run_resume_allowed"] = False
        if candidate.get("candidate_manifest"):
            manifest = candidate["candidate_manifest"]
            manifest["promotion_support_eligible"] = candidate[
                "promotion_support_eligible"
            ]
            manifest["promotion_support_gate_passed"] = candidate[
                "promotion_support_gate_passed"
            ]
            manifest["promotion_support_reason_codes"] = candidate[
                "promotion_support_reason_codes"
            ]
            manifest["promotion_support_thresholds"] = candidate[
                "promotion_support_thresholds"
            ]
            manifest["promotion_replay_entry_decision_count"] = candidate[
                "promotion_replay_entry_decision_count"
            ]
            manifest["promotion_replay_sell_decision_count"] = candidate[
                "promotion_replay_sell_decision_count"
            ]
            manifest["promotion_replay_unique_market_count"] = candidate[
                "promotion_replay_unique_market_count"
            ]
            manifest["promotion_replay_side_count"] = candidate[
                "promotion_replay_side_count"
            ]
            manifest["promotion_replay_side_distribution"] = candidate[
                "promotion_replay_side_distribution"
            ]
            manifest["promotion_replay_mean_pnl_per_entry"] = candidate[
                "promotion_replay_mean_pnl_per_entry"
            ]
            if (
                candidate["candidate_name"]
                == SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
            ):
                manifest["threshold_selection_method"] = candidate[
                    "threshold_selection_method"
                ]
                manifest["threshold_selection_fit_split"] = candidate[
                    "threshold_selection_fit_split"
                ]
                manifest["threshold_selection_evaluation_split"] = candidate[
                    "threshold_selection_evaluation_split"
                ]
                manifest["uses_shadow_for_fit"] = candidate["uses_shadow_for_fit"]
                manifest["shadow_sweep_not_used_for_threshold_fit"] = candidate[
                    "shadow_sweep_not_used_for_threshold_fit"
                ]
                manifest["threshold_selection_passed"] = candidate[
                    "threshold_selection_passed"
                ]
                manifest["threshold_selection_failed"] = candidate[
                    "threshold_selection_failed"
                ]
                manifest["threshold_selection_failure_reason_codes"] = candidate[
                    "threshold_selection_failure_reason_codes"
                ]
                manifest["support_aware_threshold_selection_failed"] = candidate[
                    "support_aware_threshold_selection_failed"
                ]
                manifest["support_aware_threshold_selection_reason_codes"] = (
                    candidate["support_aware_threshold_selection_reason_codes"]
                )
                manifest["support_aware_threshold_selection_summary"] = candidate[
                    "support_aware_threshold_selection_summary"
                ]
            manifest["source_model_candidate_eligible"] = candidate[
                "source_model_candidate_eligible"
            ]
            manifest["action_value_paper_decision_eligible"] = candidate[
                "source_model_candidate_eligible"
            ]
            manifest["ineligible_reason_codes"] = candidate[
                "ineligible_reason_codes"
            ]
    candidates_by_name = {
        candidate["candidate_name"]: candidate
        for candidate in model_ranking_candidate_comparison["candidates"]
    }
    for rows in (
        comparison_rows,
        legacy_comparison_rows,
        sell_before_close_exit_reliability.get("i_vs_j_replay_comparison", []),
    ):
        for row in rows:
            candidate = candidates_by_name.get(row["candidate_name"])
            if candidate is None:
                continue
            row["source_model_candidate_eligible"] = candidate[
                "source_model_candidate_eligible"
            ]
            row["promotion_support_eligible"] = candidate.get(
                "promotion_support_eligible",
                False,
            )
            row["market_count"] = candidate.get(
                "promotion_replay_unique_market_count"
            )
            row["side_count"] = candidate.get("promotion_replay_side_count")
            row["mean_pnl_per_entry"] = candidate.get(
                "promotion_replay_mean_pnl_per_entry"
            )
            row["ineligible_reason_codes"] = candidate["ineligible_reason_codes"]
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_guard_summary"
    ] = guard_summaries.get(SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME)
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_p_up_aligned_summary"
    ] = guard_summaries.get(SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME)
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_support_aware_p_up_aligned_summary"
    ] = guard_summaries.get(
        SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
    )
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_side_balanced_summary"
    ] = guard_summaries.get(SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME)
    model_ranking_candidate_comparison["i_vs_j_replay_comparison"] = (
        sell_before_close_exit_reliability.get("i_vs_j_replay_comparison", [])
    )
    model_ranking_candidate_comparison["i_vs_j_vs_k_replay_comparison"] = (
        sell_before_close_exit_reliability.get("i_vs_j_vs_k_replay_comparison", [])
    )
    model_ranking_candidate_comparison["i_vs_j_vs_k_vs_l_replay_comparison"] = (
        legacy_comparison_rows
    )
    model_ranking_candidate_comparison["i_vs_j_vs_k_vs_l_vs_m_replay_comparison"] = (
        comparison_rows
    )
    model_ranking_candidate_comparison[
        "sell_before_close_promotion_support_gate_summary"
    ] = sell_before_close_promotion_support_gate_summary(
        sell_before_close_promotion_support_gate
    )
    model_ranking_candidate_comparison[
        "sell_before_close_support_aware_threshold_selection_summary"
    ] = sell_before_close_support_aware_threshold_selection_summary(
        sell_before_close_support_aware_threshold_selection
    )
    model_ranking_candidate_comparison[
        "sell_before_close_i_vs_j_vs_k_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_promotion_support_comparison",
        [],
    )
    model_ranking_candidate_comparison[
        "sell_before_close_i_vs_j_vs_k_vs_l_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_vs_l_promotion_support_comparison",
        [],
    )
    model_ranking_candidate_comparison[
        "sell_before_close_i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison",
        [],
    )
    _refresh_candidate_comparison_rollups(model_ranking_candidate_comparison)


def _refresh_candidate_comparison_rollups(report: dict[str, Any]) -> None:
    candidates = report["candidates"]
    eligible_candidates = [
        candidate for candidate in candidates if candidate["source_model_candidate_eligible"]
    ]
    report["candidate_count"] = len(candidates)
    report["eligible_candidate_count"] = len(eligible_candidates)
    report["candidate_names"] = [candidate["candidate_name"] for candidate in candidates]
    best_candidate = _best_candidate_report(candidates)
    report["best_candidate_name"] = best_candidate["candidate_name"]
    report["best_candidate_source_model_eligible"] = best_candidate[
        "source_model_candidate_eligible"
    ]
    report["source_model_candidate_eligible"] = bool(eligible_candidates)
    report["no_candidate_eligible"] = not eligible_candidates
    report["no_candidate_eligible_reason_codes"] = sorted(
        {
            reason
            for candidate in candidates
            for reason in candidate["ineligible_reason_codes"]
        }
    )


def _best_candidate_report(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda candidate: (
            not bool(candidate["source_model_eligible"]),
            -float(candidate["high_score_realized_return_mean"]),
            float(candidate["shadow_mean_regret"]),
            candidate["candidate_name"],
        ),
    )[0]


def _build_sell_before_close_side_balanced_candidate_report(
    *,
    model_ranking_candidate_comparison: dict[str, Any],
    source_model_eligibility: dict[str, Any],
    sell_before_close_exit_reliability: dict[str, Any],
    sell_before_close_promotion_support_gate: dict[str, Any],
) -> dict[str, Any]:
    candidate = next(
        (
            row
            for row in model_ranking_candidate_comparison.get("candidates", [])
            if row["candidate_name"]
            == SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME
        ),
        {},
    )
    scoped = next(
        (
            row
            for row in source_model_eligibility.get(
                "candidate_scoped_eligibility_summary",
                [],
            )
            if row["candidate_name"]
            == SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME
        ),
        {},
    )
    replay_row = next(
        (
            row
            for row in sell_before_close_exit_reliability.get(
                "i_vs_j_vs_k_vs_l_vs_m_replay_comparison",
                [],
            )
            if row["candidate_name"]
            == SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME
        ),
        {},
    )
    support_row = next(
        (
            row
            for row in sell_before_close_promotion_support_gate.get(
                "candidate_rows",
                [],
            )
            if row["candidate_name"]
            == SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME
        ),
        {},
    )
    side_balance_summary = dict(
        candidate.get("side_balance_selection_summary")
        or scoped.get("side_balance_selection_summary")
        or replay_row.get("side_balance_selection_summary")
        or {}
    )
    ineligible_reasons = sorted(
        set(candidate.get("ineligible_reason_codes", []))
        | set(support_row.get("support_gate_reason_codes", []))
        | set(replay_row.get("ineligible_reason_codes", []))
    )
    replay_positive = (
        bool(replay_row)
        and float(replay_row.get("total_pnl", 0.0)) > 0.0
        and int(replay_row.get("residual_count", 0)) == 0
    )
    promotion_evidence_eligible = (
        bool(candidate.get("source_model_candidate_eligible", False))
        and bool(support_row.get("promotion_support_eligible", False))
        and replay_positive
    )
    if not promotion_evidence_eligible:
        ineligible_reasons.append("promotion_replay_gate_required")
    report = {
        "schema_version": "bigan-v8-polymarket-sell-before-close-side-balanced-candidate-v1",
        "candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "diagnostic_only": False,
        "enabled_action_families": candidate.get(
            "enabled_action_families",
            scoped.get("enabled_action_families", []),
        ),
        "disabled_action_families": candidate.get(
            "disabled_action_families",
            scoped.get("disabled_action_families", []),
        ),
        "enabled_actions": candidate.get("enabled_actions", scoped.get("enabled_actions", [])),
        "disabled_actions": candidate.get(
            "disabled_actions",
            scoped.get("disabled_actions", []),
        ),
        "side_balance_required": bool(
            candidate.get("side_balance_required", scoped.get("side_balance_required", True))
        ),
        "side_balance_gate_passed": bool(
            candidate.get(
                "side_balance_gate_passed",
                scoped.get("side_balance_gate_passed", False),
            )
        ),
        "side_balance_thresholds": dict(
            candidate.get("side_balance_thresholds")
            or scoped.get("side_balance_thresholds")
            or {}
        ),
        "side_balance_selection_summary": side_balance_summary,
        "entry_count": support_row.get("entry_decision_count"),
        "up_entry_count": support_row.get("up_entry_count"),
        "down_entry_count": support_row.get("down_entry_count"),
        "up_market_count": support_row.get("up_market_count"),
        "down_market_count": support_row.get("down_market_count"),
        "side_entry_ratio": support_row.get("side_entry_ratio"),
        "candidate_scoped_p_up_action_disagreement_rate": candidate.get(
            "candidate_scoped_p_up_action_disagreement_rate"
        ),
        "candidate_scoped_p_up_action_disagreement_within_limit": candidate.get(
            "candidate_scoped_p_up_action_disagreement_within_limit"
        ),
        "candidate_scoped_high_score_support_count": candidate.get(
            "candidate_scoped_high_score_support_count"
        ),
        "candidate_scoped_high_score_realized_return_mean": candidate.get(
            "candidate_scoped_high_score_realized_return_mean"
        ),
        "candidate_scoped_high_score_realized_return_sum": candidate.get(
            "candidate_scoped_high_score_realized_return_sum"
        ),
        "replay_total_pnl": replay_row.get("total_pnl"),
        "replay_residual_count": replay_row.get("residual_count"),
        "promotion_support_eligible": support_row.get(
            "promotion_support_eligible",
            False,
        ),
        "promotion_support_reason_codes": support_row.get(
            "support_gate_reason_codes",
            [],
        ),
        "source_model_candidate_eligible": bool(
            candidate.get("source_model_candidate_eligible", False)
        ),
        "promotion_evidence_eligible": promotion_evidence_eligible,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "ineligible_reason_codes": sorted(set(ineligible_reasons)),
        **compact_safety_fields(),
    }
    report["sell_before_close_side_balanced_candidate_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _sell_before_close_side_balanced_candidate_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "schema_version",
        "candidate_name",
        "side_balance_required",
        "side_balance_gate_passed",
        "side_balance_thresholds",
        "side_balance_selection_summary",
        "entry_count",
        "up_entry_count",
        "down_entry_count",
        "up_market_count",
        "down_market_count",
        "side_entry_ratio",
        "candidate_scoped_p_up_action_disagreement_rate",
        "candidate_scoped_p_up_action_disagreement_within_limit",
        "candidate_scoped_high_score_support_count",
        "candidate_scoped_high_score_realized_return_mean",
        "candidate_scoped_high_score_realized_return_sum",
        "replay_total_pnl",
        "replay_residual_count",
        "promotion_support_eligible",
        "promotion_support_reason_codes",
        "source_model_candidate_eligible",
        "promotion_evidence_eligible",
        "paper_run_resume_allowed",
        "#146_start_allowed",
        "#134_resume_allowed",
        "ineligible_reason_codes",
    )
    return {field: report.get(field) for field in fields}


def _sell_before_close_side_balanced_candidate_markdown(
    report: dict[str, Any],
) -> str:
    lines = [
        "# SELL_BEFORE_CLOSE Side-Balanced Candidate",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- side_balance_gate_passed: `{str(report['side_balance_gate_passed']).lower()}`",
        f"- source_model_candidate_eligible: `{str(report['source_model_candidate_eligible']).lower()}`",
        f"- promotion_evidence_eligible: `{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        "- ineligible_reason_codes: "
        f"`{json.dumps(report['ineligible_reason_codes'])}`",
        "",
        "| entries | up | down | up_markets | down_markets | side_ratio | replay_pnl | p_up_disagreement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {entries} | {up} | {down} | {up_markets} | {down_markets} | {ratio} | {pnl} | {p_up} |".format(
            entries=report.get("entry_count"),
            up=report.get("up_entry_count"),
            down=report.get("down_entry_count"),
            up_markets=report.get("up_market_count"),
            down_markets=report.get("down_market_count"),
            ratio=report.get("side_entry_ratio"),
            pnl=report.get("replay_total_pnl"),
            p_up=report.get("candidate_scoped_p_up_action_disagreement_rate"),
        ),
        "",
        "- paper_only: true",
        "- capital_at_risk: false",
        "- polymarket_write_enabled: false",
        "- wallet_signing_enabled: false",
        "",
    ]
    return "\n".join(lines)


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
    action_family_counterfactual_replays: tuple[dict[str, Any], ...],
    signal_sanity: dict[str, Any],
    model_ranking_error: dict[str, Any],
    model_ranking_candidate_comparison: dict[str, Any],
    action_representation_diagnostic: dict[str, Any],
    ranking_overlay_zero_entry_diagnostic: dict[str, Any],
    source_model_eligibility: dict[str, Any],
    sell_before_close_p_up_disagreement_diagnostic: dict[str, Any],
    sell_before_close_exit_reliability: dict[str, Any],
    sell_before_close_promotion_support_gate: dict[str, Any],
    sell_before_close_support_aware_threshold_selection: dict[str, Any],
    sell_before_close_support_aware_threshold_failure_attribution: dict[str, Any],
    sell_before_close_validation_failure_drilldown: dict[str, Any],
    sell_before_close_guard_threshold_sweep: dict[str, Any],
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
        "action_family_counterfactual_replay_report": (
            run_dir / "action_family_counterfactual_replay_report.json"
        ),
        "action_family_counterfactual_replay_summary": (
            run_dir / "action_family_counterfactual_replay_report.md"
        ),
        "model_ranking_error_report": run_dir / "model_ranking_error_report.json",
        "model_ranking_error_summary": run_dir / "model_ranking_error_report.md",
        "model_ranking_candidate_comparison": (
            run_dir / "model_ranking_candidate_comparison.json"
        ),
        "model_ranking_candidate_comparison_summary": (
            run_dir / "model_ranking_candidate_comparison.md"
        ),
        "action_representation_diagnostic_report": (
            run_dir / "action_representation_diagnostic_report.json"
        ),
        "action_representation_diagnostic_summary": (
            run_dir / "action_representation_diagnostic_report.md"
        ),
        "ranking_overlay_zero_entry_diagnostic_report": (
            run_dir / "ranking_overlay_zero_entry_diagnostic_report.json"
        ),
        "ranking_overlay_zero_entry_diagnostic_summary": (
            run_dir / "ranking_overlay_zero_entry_diagnostic_report.md"
        ),
        "source_model_eligibility_report": (
            run_dir / "source_model_eligibility_report.json"
        ),
        "source_model_eligibility_summary": (
            run_dir / "source_model_eligibility_report.md"
        ),
        "sell_before_close_p_up_disagreement_diagnostic_report": (
            run_dir / "sell_before_close_p_up_disagreement_diagnostic_report.json"
        ),
        "sell_before_close_p_up_disagreement_diagnostic_summary": (
            run_dir / "sell_before_close_p_up_disagreement_diagnostic_report.md"
        ),
        "sell_before_close_exit_reliability_report": (
            run_dir / "sell_before_close_exit_reliability_report.json"
        ),
        "sell_before_close_exit_reliability_summary": (
            run_dir / "sell_before_close_exit_reliability_report.md"
        ),
        "sell_before_close_promotion_support_gate_report": (
            run_dir / "sell_before_close_promotion_support_gate_report.json"
        ),
        "sell_before_close_promotion_support_gate_summary": (
            run_dir / "sell_before_close_promotion_support_gate_report.md"
        ),
        "sell_before_close_support_aware_threshold_selection_report": (
            run_dir
            / "sell_before_close_support_aware_threshold_selection_report.json"
        ),
        "sell_before_close_support_aware_threshold_selection_summary": (
            run_dir
            / "sell_before_close_support_aware_threshold_selection_report.md"
        ),
        "sell_before_close_support_aware_threshold_failure_attribution_report": (
            run_dir
            / "sell_before_close_support_aware_threshold_failure_attribution_report.json"
        ),
        "sell_before_close_support_aware_threshold_failure_attribution_summary": (
            run_dir
            / "sell_before_close_support_aware_threshold_failure_attribution_report.md"
        ),
        "sell_before_close_validation_failure_drilldown_report": (
            run_dir / "sell_before_close_validation_failure_drilldown_report.json"
        ),
        "sell_before_close_validation_failure_drilldown_summary": (
            run_dir / "sell_before_close_validation_failure_drilldown_report.md"
        ),
        "sell_before_close_side_balanced_candidate_report": (
            run_dir / "sell_before_close_side_balanced_candidate_report.json"
        ),
        "sell_before_close_side_balanced_candidate_summary": (
            run_dir / "sell_before_close_side_balanced_candidate_report.md"
        ),
        "sell_before_close_guard_threshold_sweep_report": (
            run_dir / "sell_before_close_guard_threshold_sweep_report.json"
        ),
        "sell_before_close_guard_threshold_sweep_summary": (
            run_dir / "sell_before_close_guard_threshold_sweep_report.md"
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
    _write_json(
        paths["sell_before_close_validation_failure_drilldown_report"],
        sell_before_close_validation_failure_drilldown,
    )
    validation_failure_drilldown_report_sha256 = _sha256_file(
        paths["sell_before_close_validation_failure_drilldown_report"]
    )
    validation_failure_drilldown_summary = (
        sell_before_close_validation_failure_drilldown_summary(
            sell_before_close_validation_failure_drilldown
        )
    )
    drilldown_path = "sell_before_close_validation_failure_drilldown_report.json"
    sell_before_close_support_aware_threshold_failure_attribution[
        "validation_failure_drilldown_report_path"
    ] = drilldown_path
    sell_before_close_support_aware_threshold_failure_attribution[
        "validation_failure_drilldown_report_sha256"
    ] = validation_failure_drilldown_report_sha256
    sell_before_close_support_aware_threshold_failure_attribution[
        "sell_before_close_validation_failure_drilldown_summary"
    ] = validation_failure_drilldown_summary
    _refresh_report_id(
        sell_before_close_support_aware_threshold_failure_attribution,
        "sell_before_close_support_aware_threshold_failure_attribution_report_id",
    )
    _write_json(
        paths["sell_before_close_support_aware_threshold_failure_attribution_report"],
        sell_before_close_support_aware_threshold_failure_attribution,
    )
    failure_attribution_report_sha256 = _sha256_file(
        paths["sell_before_close_support_aware_threshold_failure_attribution_report"]
    )
    sell_before_close_support_aware_threshold_selection[
        "failure_attribution_report_path"
    ] = "sell_before_close_support_aware_threshold_failure_attribution_report.json"
    sell_before_close_support_aware_threshold_selection[
        "failure_attribution_report_sha256"
    ] = failure_attribution_report_sha256
    sell_before_close_support_aware_threshold_selection[
        "validation_failure_drilldown_report_path"
    ] = drilldown_path
    sell_before_close_support_aware_threshold_selection[
        "validation_failure_drilldown_report_sha256"
    ] = validation_failure_drilldown_report_sha256
    sell_before_close_support_aware_threshold_selection[
        "sell_before_close_validation_failure_drilldown_summary"
    ] = validation_failure_drilldown_summary
    sell_before_close_support_aware_threshold_selection.pop(
        "failure_attribution_report",
        None,
    )
    sell_before_close_support_aware_threshold_selection.pop(
        "validation_failure_drilldown_report",
        None,
    )
    _refresh_report_id(
        sell_before_close_support_aware_threshold_selection,
        "sell_before_close_support_aware_threshold_selection_report_id",
    )
    support_aware_threshold_selection_summary = (
        sell_before_close_support_aware_threshold_selection_summary(
            sell_before_close_support_aware_threshold_selection
        )
    )
    support_aware_failure_attribution_summary = (
        sell_before_close_support_aware_threshold_failure_attribution_summary(
            sell_before_close_support_aware_threshold_failure_attribution
        )
    )
    sell_before_close_promotion_support_gate["threshold_selection_passed"] = (
        support_aware_threshold_selection_summary["threshold_selection_passed"]
    )
    sell_before_close_promotion_support_gate["threshold_selection_failed"] = (
        support_aware_threshold_selection_summary["threshold_selection_failed"]
    )
    sell_before_close_promotion_support_gate[
        "threshold_selection_failure_reason_codes"
    ] = support_aware_threshold_selection_summary[
        "threshold_selection_failure_reason_codes"
    ]
    sell_before_close_promotion_support_gate[
        "support_aware_threshold_selection_failed"
    ] = support_aware_threshold_selection_summary["threshold_selection_failed"]
    sell_before_close_promotion_support_gate[
        "threshold_selection_failure_interpretation"
    ] = support_aware_threshold_selection_summary[
        "threshold_selection_failure_interpretation"
    ]
    sell_before_close_promotion_support_gate["recommended_next_action"] = (
        support_aware_threshold_selection_summary["recommended_next_action"]
    )
    sell_before_close_promotion_support_gate["failure_attribution_report_path"] = (
        support_aware_threshold_selection_summary["failure_attribution_report_path"]
    )
    sell_before_close_promotion_support_gate["failure_attribution_report_sha256"] = (
        support_aware_threshold_selection_summary["failure_attribution_report_sha256"]
    )
    sell_before_close_promotion_support_gate[
        "validation_failure_drilldown_report_path"
    ] = drilldown_path
    sell_before_close_promotion_support_gate[
        "validation_failure_drilldown_report_sha256"
    ] = validation_failure_drilldown_report_sha256
    sell_before_close_promotion_support_gate[
        "sell_before_close_validation_failure_drilldown_summary"
    ] = validation_failure_drilldown_summary
    _refresh_report_id(
        sell_before_close_promotion_support_gate,
        "sell_before_close_promotion_support_gate_report_id",
    )
    diagnostic_summary = sell_before_close_p_up_disagreement_summary(
        sell_before_close_p_up_disagreement_diagnostic
    )
    exit_reliability_summary = sell_before_close_exit_reliability_summary(
        sell_before_close_exit_reliability
    )
    sell_before_close_p_up_disagreement_diagnostic[
        "sell_before_close_exit_reliability_summary"
    ] = exit_reliability_summary
    _refresh_report_id(
        sell_before_close_p_up_disagreement_diagnostic,
        "sell_before_close_p_up_disagreement_diagnostic_report_id",
    )
    model_ranking_candidate_comparison[
        "sell_before_close_p_up_disagreement_diagnostic_summary"
    ] = diagnostic_summary
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_summary"
    ] = exit_reliability_summary
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_guard_summary"
    ] = sell_before_close_exit_reliability.get(
        "exit_reliability_guard_candidate_summary"
    )
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_p_up_aligned_summary"
    ] = sell_before_close_exit_reliability.get(
        "exit_reliability_p_up_aligned_candidate_summary"
    )
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_support_aware_p_up_aligned_summary"
    ] = sell_before_close_exit_reliability.get(
        "exit_reliability_support_aware_p_up_aligned_candidate_summary"
    )
    model_ranking_candidate_comparison[
        "sell_before_close_exit_reliability_side_balanced_summary"
    ] = sell_before_close_exit_reliability.get(
        "exit_reliability_side_balanced_candidate_summary"
    )
    model_ranking_candidate_comparison["i_vs_j_replay_comparison"] = (
        sell_before_close_exit_reliability.get("i_vs_j_replay_comparison", [])
    )
    model_ranking_candidate_comparison["i_vs_j_vs_k_replay_comparison"] = (
        sell_before_close_exit_reliability.get("i_vs_j_vs_k_replay_comparison", [])
    )
    model_ranking_candidate_comparison["i_vs_j_vs_k_vs_l_replay_comparison"] = (
        sell_before_close_exit_reliability.get(
            "i_vs_j_vs_k_vs_l_replay_comparison",
            [],
        )
    )
    model_ranking_candidate_comparison["i_vs_j_vs_k_vs_l_vs_m_replay_comparison"] = (
        sell_before_close_exit_reliability.get(
            "i_vs_j_vs_k_vs_l_vs_m_replay_comparison",
            [],
        )
    )
    model_ranking_candidate_comparison[
        "sell_before_close_guard_threshold_sweep_summary"
    ] = _sell_before_close_guard_threshold_sweep_summary(
        sell_before_close_guard_threshold_sweep
    )
    model_ranking_candidate_comparison[
        "sell_before_close_promotion_support_gate_summary"
    ] = sell_before_close_promotion_support_gate_summary(
        sell_before_close_promotion_support_gate
    )
    model_ranking_candidate_comparison[
        "sell_before_close_support_aware_threshold_selection_summary"
    ] = support_aware_threshold_selection_summary
    model_ranking_candidate_comparison[
        "sell_before_close_support_aware_threshold_failure_attribution_summary"
    ] = support_aware_failure_attribution_summary
    model_ranking_candidate_comparison[
        "sell_before_close_validation_failure_drilldown_summary"
    ] = validation_failure_drilldown_summary
    model_ranking_candidate_comparison[
        "sell_before_close_i_vs_j_vs_k_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_promotion_support_comparison",
        [],
    )
    model_ranking_candidate_comparison[
        "sell_before_close_i_vs_j_vs_k_vs_l_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_vs_l_promotion_support_comparison",
        sell_before_close_promotion_support_gate.get(
            "i_vs_j_vs_k_promotion_support_comparison",
            [],
        ),
    )
    model_ranking_candidate_comparison[
        "sell_before_close_i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison",
        sell_before_close_promotion_support_gate.get(
            "i_vs_j_vs_k_vs_l_promotion_support_comparison",
            [],
        ),
    )
    source_model_eligibility[
        "sell_before_close_p_up_disagreement_diagnostic_summary"
    ] = diagnostic_summary
    source_model_eligibility[
        "sell_before_close_exit_reliability_summary"
    ] = exit_reliability_summary
    source_model_eligibility[
        "sell_before_close_exit_reliability_guard_summary"
    ] = sell_before_close_exit_reliability.get(
        "exit_reliability_guard_candidate_summary"
    )
    source_model_eligibility[
        "sell_before_close_exit_reliability_p_up_aligned_summary"
    ] = sell_before_close_exit_reliability.get(
        "exit_reliability_p_up_aligned_candidate_summary"
    )
    source_model_eligibility[
        "sell_before_close_exit_reliability_support_aware_p_up_aligned_summary"
    ] = sell_before_close_exit_reliability.get(
        "exit_reliability_support_aware_p_up_aligned_candidate_summary"
    )
    source_model_eligibility[
        "sell_before_close_exit_reliability_side_balanced_summary"
    ] = sell_before_close_exit_reliability.get(
        "exit_reliability_side_balanced_candidate_summary"
    )
    source_model_eligibility["i_vs_j_replay_comparison"] = (
        sell_before_close_exit_reliability.get("i_vs_j_replay_comparison", [])
    )
    source_model_eligibility["i_vs_j_vs_k_replay_comparison"] = (
        sell_before_close_exit_reliability.get("i_vs_j_vs_k_replay_comparison", [])
    )
    source_model_eligibility["i_vs_j_vs_k_vs_l_replay_comparison"] = (
        sell_before_close_exit_reliability.get(
            "i_vs_j_vs_k_vs_l_replay_comparison",
            [],
        )
    )
    source_model_eligibility["i_vs_j_vs_k_vs_l_vs_m_replay_comparison"] = (
        sell_before_close_exit_reliability.get(
            "i_vs_j_vs_k_vs_l_vs_m_replay_comparison",
            [],
        )
    )
    source_model_eligibility[
        "sell_before_close_guard_threshold_sweep_summary"
    ] = _sell_before_close_guard_threshold_sweep_summary(
        sell_before_close_guard_threshold_sweep
    )
    source_model_eligibility[
        "sell_before_close_promotion_support_gate_summary"
    ] = sell_before_close_promotion_support_gate_summary(
        sell_before_close_promotion_support_gate
    )
    source_model_eligibility[
        "sell_before_close_support_aware_threshold_selection_summary"
    ] = support_aware_threshold_selection_summary
    source_model_eligibility[
        "sell_before_close_support_aware_threshold_failure_attribution_summary"
    ] = support_aware_failure_attribution_summary
    source_model_eligibility[
        "sell_before_close_validation_failure_drilldown_summary"
    ] = validation_failure_drilldown_summary
    source_model_eligibility[
        "sell_before_close_i_vs_j_vs_k_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_promotion_support_comparison",
        [],
    )
    source_model_eligibility[
        "sell_before_close_i_vs_j_vs_k_vs_l_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_vs_l_promotion_support_comparison",
        sell_before_close_promotion_support_gate.get(
            "i_vs_j_vs_k_promotion_support_comparison",
            [],
        ),
    )
    source_model_eligibility[
        "sell_before_close_i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison"
    ] = sell_before_close_promotion_support_gate.get(
        "i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison",
        sell_before_close_promotion_support_gate.get(
            "i_vs_j_vs_k_vs_l_promotion_support_comparison",
            [],
        ),
    )
    sell_before_close_side_balanced_candidate = (
        _build_sell_before_close_side_balanced_candidate_report(
            model_ranking_candidate_comparison=model_ranking_candidate_comparison,
            source_model_eligibility=source_model_eligibility,
            sell_before_close_exit_reliability=sell_before_close_exit_reliability,
            sell_before_close_promotion_support_gate=(
                sell_before_close_promotion_support_gate
            ),
        )
    )
    _write_json(
        paths["sell_before_close_side_balanced_candidate_report"],
        sell_before_close_side_balanced_candidate,
    )
    paths["sell_before_close_side_balanced_candidate_summary"].write_text(
        _sell_before_close_side_balanced_candidate_markdown(
            sell_before_close_side_balanced_candidate
        ),
        encoding="utf-8",
    )
    side_balanced_candidate_sha256 = _sha256_file(
        paths["sell_before_close_side_balanced_candidate_report"]
    )
    side_balanced_candidate_summary = (
        _sell_before_close_side_balanced_candidate_summary(
            sell_before_close_side_balanced_candidate
        )
    )
    for report in (
        model_ranking_candidate_comparison,
        source_model_eligibility,
    ):
        report["sell_before_close_side_balanced_candidate_report_path"] = (
            "sell_before_close_side_balanced_candidate_report.json"
        )
        report["sell_before_close_side_balanced_candidate_report_sha256"] = (
            side_balanced_candidate_sha256
        )
        report["sell_before_close_side_balanced_candidate_summary"] = (
            side_balanced_candidate_summary
        )
    _write_candidate_artifacts(
        run_dir=run_dir,
        model_ranking_candidate_comparison=model_ranking_candidate_comparison,
        source_model_eligibility=source_model_eligibility,
    )
    _refresh_report_id(
        model_ranking_candidate_comparison,
        "model_ranking_candidate_comparison_id",
    )
    _refresh_report_id(
        source_model_eligibility,
        "source_model_eligibility_report_id",
    )
    _write_json(paths["model_ranking_error_report"], model_ranking_error)
    _write_json(
        paths["model_ranking_candidate_comparison"],
        model_ranking_candidate_comparison,
    )
    _write_json(
        paths["action_representation_diagnostic_report"],
        action_representation_diagnostic,
    )
    _write_json(
        paths["ranking_overlay_zero_entry_diagnostic_report"],
        ranking_overlay_zero_entry_diagnostic,
    )
    _write_json(paths["source_model_eligibility_report"], source_model_eligibility)
    _write_json(
        paths["sell_before_close_p_up_disagreement_diagnostic_report"],
        sell_before_close_p_up_disagreement_diagnostic,
    )
    _write_json(
        paths["sell_before_close_exit_reliability_report"],
        sell_before_close_exit_reliability,
    )
    _write_json(
        paths["sell_before_close_promotion_support_gate_report"],
        sell_before_close_promotion_support_gate,
    )
    _write_json(
        paths["sell_before_close_support_aware_threshold_selection_report"],
        sell_before_close_support_aware_threshold_selection,
    )
    _write_json(
        paths["sell_before_close_guard_threshold_sweep_report"],
        sell_before_close_guard_threshold_sweep,
    )
    _write_json(paths["action_family_eligibility_report"], action_family_eligibility)
    _write_json(
        paths["hold_to_settlement_longshot_guard_report"],
        hold_to_settlement_longshot_guard,
    )
    _write_json(paths["action_family_replay_variants_report"], action_family_replay_variants)
    action_family_counterfactual_replay = _write_counterfactual_replay_artifacts(
        run_dir=run_dir,
        counterfactual_replays=action_family_counterfactual_replays,
        source_model_eligibility=source_model_eligibility,
        sell_before_close_p_up_disagreement_diagnostic=(
            sell_before_close_p_up_disagreement_diagnostic
        ),
        sell_before_close_exit_reliability=sell_before_close_exit_reliability,
        sell_before_close_promotion_support_gate=(
            sell_before_close_promotion_support_gate
        ),
        sell_before_close_support_aware_threshold_selection=(
            sell_before_close_support_aware_threshold_selection
        ),
        sell_before_close_validation_failure_drilldown=(
            sell_before_close_validation_failure_drilldown
        ),
        sell_before_close_guard_threshold_sweep=(
            sell_before_close_guard_threshold_sweep
        ),
    )
    _write_json(
        paths["action_family_counterfactual_replay_report"],
        action_family_counterfactual_replay,
    )
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
    paths["model_ranking_error_summary"].write_text(
        model_ranking_error_markdown(model_ranking_error),
        encoding="utf-8",
    )
    paths["model_ranking_candidate_comparison_summary"].write_text(
        model_ranking_candidate_comparison_markdown(
            model_ranking_candidate_comparison
        ),
        encoding="utf-8",
    )
    paths["action_representation_diagnostic_summary"].write_text(
        action_representation_diagnostic_markdown(action_representation_diagnostic),
        encoding="utf-8",
    )
    paths["ranking_overlay_zero_entry_diagnostic_summary"].write_text(
        ranking_overlay_zero_entry_diagnostic_markdown(
            ranking_overlay_zero_entry_diagnostic
        ),
        encoding="utf-8",
    )
    paths["source_model_eligibility_summary"].write_text(
        source_model_eligibility_markdown(source_model_eligibility),
        encoding="utf-8",
    )
    paths["sell_before_close_p_up_disagreement_diagnostic_summary"].write_text(
        sell_before_close_p_up_disagreement_diagnostic_markdown(
            sell_before_close_p_up_disagreement_diagnostic
        ),
        encoding="utf-8",
    )
    paths["sell_before_close_exit_reliability_summary"].write_text(
        sell_before_close_exit_reliability_markdown(
            sell_before_close_exit_reliability
        ),
        encoding="utf-8",
    )
    paths["sell_before_close_promotion_support_gate_summary"].write_text(
        sell_before_close_promotion_support_gate_markdown(
            sell_before_close_promotion_support_gate
        ),
        encoding="utf-8",
    )
    paths["sell_before_close_support_aware_threshold_selection_summary"].write_text(
        sell_before_close_support_aware_threshold_selection_markdown(
            sell_before_close_support_aware_threshold_selection
        ),
        encoding="utf-8",
    )
    paths[
        "sell_before_close_support_aware_threshold_failure_attribution_summary"
    ].write_text(
        sell_before_close_support_aware_threshold_failure_attribution_markdown(
            sell_before_close_support_aware_threshold_failure_attribution
        ),
        encoding="utf-8",
    )
    paths["sell_before_close_validation_failure_drilldown_summary"].write_text(
        sell_before_close_validation_failure_drilldown_markdown(
            sell_before_close_validation_failure_drilldown
        ),
        encoding="utf-8",
    )
    paths["sell_before_close_guard_threshold_sweep_summary"].write_text(
        _sell_before_close_guard_threshold_sweep_markdown(
            sell_before_close_guard_threshold_sweep
        ),
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
    paths["action_family_counterfactual_replay_summary"].write_text(
        _counterfactual_replay_markdown(action_family_counterfactual_replay),
        encoding="utf-8",
    )
    return paths


def _write_candidate_artifacts(
    *,
    run_dir: Path,
    model_ranking_candidate_comparison: dict[str, Any],
    source_model_eligibility: dict[str, Any],
) -> None:
    root = run_dir / "policy_candidate_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    exported = []
    best_candidate_name = model_ranking_candidate_comparison.get("best_candidate_name")
    for candidate in model_ranking_candidate_comparison["candidates"]:
        should_export = bool(candidate.get("candidate_artifact_required")) or (
            candidate["candidate_name"] == best_candidate_name
        )
        predictions = candidate.pop("candidate_predictions", [])
        manifest = candidate.pop("candidate_manifest", None)
        overlay = candidate.pop("ranking_overlay", None)
        if not should_export or manifest is None or overlay is None:
            continue
        safe_name = _safe_artifact_name(candidate["candidate_name"])
        files = {
            "predictions": (
                root / f"polymarket_policy_candidate_{safe_name}_predictions.jsonl"
            ),
            "manifest": (
                root / f"polymarket_policy_candidate_{safe_name}_manifest.json"
            ),
            "ranking_overlay": (
                root
                / f"polymarket_policy_candidate_{safe_name}_ranking_overlay.json"
            ),
        }
        _write_jsonl(files["predictions"], predictions)
        _write_json(files["ranking_overlay"], overlay)
        manifest = dict(manifest)
        manifest["candidate_prediction_count"] = len(predictions)
        manifest["candidate_artifact_paths"] = {
            key: _relative_path(path, run_dir)
            for key, path in sorted(files.items())
            if key != "manifest"
        }
        manifest["candidate_artifact_hashes"] = {
            "predictions": _sha256_file(files["predictions"]),
            "ranking_overlay": _sha256_file(files["ranking_overlay"]),
        }
        _write_json(files["manifest"], manifest)
        artifact_paths = {
            key: _relative_path(path, run_dir) for key, path in sorted(files.items())
        }
        artifact_hashes = {
            key: _sha256_file(path) for key, path in sorted(files.items())
        }
        summary = {
            "candidate_name": candidate["candidate_name"],
            "source_model_candidate_eligible": candidate[
                "source_model_candidate_eligible"
            ],
            "candidate_artifact_reason": candidate["candidate_artifact_reason"],
            "artifact_paths": artifact_paths,
            "artifact_hashes": artifact_hashes,
        }
        candidate["candidate_artifacts"] = summary
        exported.append(summary)
    model_ranking_candidate_comparison["candidate_artifact_count"] = len(exported)
    model_ranking_candidate_comparison["candidate_artifacts"] = exported
    source_model_eligibility["candidate_artifact_count"] = len(exported)
    source_model_eligibility["candidate_artifacts"] = exported


def _refresh_report_id(payload: dict[str, Any], id_field: str) -> None:
    payload.pop(id_field, None)
    payload[id_field] = canonical_json_sha256(payload)


def _safe_artifact_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _write_counterfactual_replay_artifacts(
    *,
    run_dir: Path,
    counterfactual_replays: tuple[dict[str, Any], ...],
    source_model_eligibility: dict[str, Any],
    sell_before_close_p_up_disagreement_diagnostic: dict[str, Any],
    sell_before_close_exit_reliability: dict[str, Any],
    sell_before_close_promotion_support_gate: dict[str, Any],
    sell_before_close_support_aware_threshold_selection: dict[str, Any],
    sell_before_close_validation_failure_drilldown: dict[str, Any],
    sell_before_close_guard_threshold_sweep: dict[str, Any],
) -> dict[str, Any]:
    root = run_dir / "action_family_counterfactual_replays"
    root.mkdir(parents=True, exist_ok=True)
    variant_summaries = []
    source_model_candidate_eligible = bool(
        source_model_eligibility["source_model_candidate_eligible"]
    )
    diagnostic_summary = sell_before_close_p_up_disagreement_summary(
        sell_before_close_p_up_disagreement_diagnostic
    )
    exit_reliability_summary = sell_before_close_exit_reliability_summary(
        sell_before_close_exit_reliability
    )
    support_rows = {
        row["candidate_name"]: row
        for row in sell_before_close_promotion_support_gate.get(
            "candidate_rows",
            [],
        )
    }
    threshold_selection_summary = (
        sell_before_close_support_aware_threshold_selection_summary(
            sell_before_close_support_aware_threshold_selection
        )
    )
    threshold_selection_failed = bool(
        threshold_selection_summary["threshold_selection_failed"]
    )
    threshold_selection_reasons = list(
        threshold_selection_summary["threshold_selection_failure_reason_codes"]
    )
    validation_failure_drilldown_summary = (
        sell_before_close_validation_failure_drilldown_summary(
            sell_before_close_validation_failure_drilldown
        )
    )
    for replay in counterfactual_replays:
        variant_dir = root / replay["variant"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        support_row = support_rows.get(replay["variant"])
        summary = _counterfactual_summary_with_source_gate(
            summary=dict(replay["summary"]),
            source_model_candidate_eligible=source_model_candidate_eligible,
            promotion_support_row=support_row,
        )
        if summary["variant"] in {
            SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        }:
            summary[
                "sell_before_close_p_up_disagreement_diagnostic_summary"
            ] = diagnostic_summary
            summary[
                "sell_before_close_exit_reliability_summary"
            ] = exit_reliability_summary
            summary["i_vs_j_replay_comparison"] = sell_before_close_exit_reliability.get(
                "i_vs_j_replay_comparison",
                [],
            )
            summary["i_vs_j_vs_k_replay_comparison"] = (
                sell_before_close_exit_reliability.get(
                    "i_vs_j_vs_k_replay_comparison",
                    [],
                )
            )
            summary["i_vs_j_vs_k_vs_l_replay_comparison"] = (
                sell_before_close_exit_reliability.get(
                    "i_vs_j_vs_k_vs_l_replay_comparison",
                    [],
                )
            )
            summary["i_vs_j_vs_k_vs_l_vs_m_replay_comparison"] = (
                sell_before_close_exit_reliability.get(
                    "i_vs_j_vs_k_vs_l_vs_m_replay_comparison",
                    [],
                )
            )
            summary["sell_before_close_exit_reliability_guard_summary"] = (
                sell_before_close_exit_reliability.get(
                    "exit_reliability_guard_candidate_summary"
                )
            )
            summary["sell_before_close_exit_reliability_p_up_aligned_summary"] = (
                sell_before_close_exit_reliability.get(
                    "exit_reliability_p_up_aligned_candidate_summary"
                )
            )
            summary[
                "sell_before_close_exit_reliability_support_aware_p_up_aligned_summary"
            ] = sell_before_close_exit_reliability.get(
                "exit_reliability_support_aware_p_up_aligned_candidate_summary"
            )
            summary["sell_before_close_exit_reliability_side_balanced_summary"] = (
                sell_before_close_exit_reliability.get(
                    "exit_reliability_side_balanced_candidate_summary"
                )
            )
            summary["sell_before_close_promotion_support_gate_summary"] = (
                sell_before_close_promotion_support_gate_summary(
                    sell_before_close_promotion_support_gate
                )
            )
            summary[
                "sell_before_close_support_aware_threshold_selection_summary"
            ] = threshold_selection_summary
            summary[
                "sell_before_close_validation_failure_drilldown_summary"
            ] = validation_failure_drilldown_summary
            if (
                summary["variant"]
                == SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
            ):
                summary["threshold_selection_passed"] = threshold_selection_summary[
                    "threshold_selection_passed"
                ]
                summary["threshold_selection_failed"] = threshold_selection_failed
                summary["threshold_selection_failure_reason_codes"] = (
                    threshold_selection_reasons
                )
                summary["support_aware_threshold_selection_failed"] = (
                    threshold_selection_failed
                )
                if threshold_selection_failed:
                    reasons = set(summary["promotion_evidence_ineligible_reasons"])
                    reasons.add("support_aware_threshold_selection_failed")
                    reasons.update(threshold_selection_reasons)
                    summary["promotion_evidence_eligible"] = False
                    summary["promotion_evidence_ineligible_reasons"] = sorted(reasons)
                    summary["blocked"] = True
                    summary["blocked_reasons"] = sorted(reasons)
            summary["sell_before_close_guard_threshold_sweep_summary"] = (
                _sell_before_close_guard_threshold_sweep_summary(
                    sell_before_close_guard_threshold_sweep
                )
            )
        ledger_pnl_report = dict(replay["ledger_pnl_report"])
        ledger_pnl_report["source_model_candidate_eligible"] = (
            source_model_candidate_eligible
        )
        ledger_pnl_report["promotion_support_eligible"] = bool(
            (support_row or {}).get("promotion_support_eligible", False)
        )
        ledger_pnl_report["promotion_support_reason_codes"] = list(
            (support_row or {}).get("support_gate_reason_codes", [])
        )
        ledger_pnl_report["promotion_evidence_eligible"] = summary[
            "promotion_evidence_eligible"
        ]
        ledger_pnl_report["promotion_evidence_ineligible_reasons"] = summary[
            "promotion_evidence_ineligible_reasons"
        ]
        files = {
            "predictions": variant_dir / "predictions.jsonl",
            "decisions": variant_dir / "decisions.jsonl",
            "ev_threshold_report": variant_dir / "ev_threshold_report.json",
            "policy_replay_report": variant_dir / "policy_replay_report.json",
            "ledger_pnl_report": variant_dir / "ledger_pnl_report.json",
        }
        _write_jsonl(files["predictions"], replay["predictions"])
        _write_jsonl(files["decisions"], replay["decisions"])
        _write_json(files["ev_threshold_report"], replay["ev_report"])
        _write_json(files["policy_replay_report"], replay["replay_report"])
        _write_json(files["ledger_pnl_report"], ledger_pnl_report)
        summary["artifact_paths"] = {
            name: _relative_path(path, run_dir) for name, path in sorted(files.items())
        }
        summary["artifact_hashes"] = {
            name: _sha256_file(path) for name, path in sorted(files.items())
        }
        variant_summaries.append(summary)
    promotion_evidence_eligible = any(
        bool(summary["promotion_evidence_eligible"]) for summary in variant_summaries
    )
    ineligible_reasons = sorted(
        {
            reason
            for summary in variant_summaries
            for reason in summary["promotion_evidence_ineligible_reasons"]
        }
    )
    return {
        "schema_version": (
            "bigan-v8-polymarket-action-family-counterfactual-replay-v1"
        ),
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "report_mode": "re_ranked_counterfactual_policy_replay",
        "filtered_estimate_report_path": "action_family_replay_variants_report.json",
        "source_model_candidate_eligible": source_model_candidate_eligible,
        "threshold_selection_passed": threshold_selection_summary[
            "threshold_selection_passed"
        ],
        "threshold_selection_failed": threshold_selection_failed,
        "threshold_selection_failure_reason_codes": threshold_selection_reasons,
        "support_aware_threshold_selection_failed": threshold_selection_failed,
        "promotion_evidence_eligible": promotion_evidence_eligible,
        "promotion_evidence_ineligible_reasons": (
            [] if promotion_evidence_eligible else ineligible_reasons
        ),
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "sell_before_close_p_up_disagreement_diagnostic_summary": (
            diagnostic_summary
        ),
        "sell_before_close_exit_reliability_summary": exit_reliability_summary,
        "sell_before_close_exit_reliability_guard_summary": (
            sell_before_close_exit_reliability.get(
                "exit_reliability_guard_candidate_summary"
            )
        ),
        "sell_before_close_exit_reliability_p_up_aligned_summary": (
            sell_before_close_exit_reliability.get(
                "exit_reliability_p_up_aligned_candidate_summary"
            )
        ),
        "sell_before_close_exit_reliability_support_aware_p_up_aligned_summary": (
            sell_before_close_exit_reliability.get(
                "exit_reliability_support_aware_p_up_aligned_candidate_summary"
            )
        ),
        "sell_before_close_exit_reliability_side_balanced_summary": (
            sell_before_close_exit_reliability.get(
                "exit_reliability_side_balanced_candidate_summary"
            )
        ),
        "sell_before_close_promotion_support_gate_summary": (
            sell_before_close_promotion_support_gate_summary(
                sell_before_close_promotion_support_gate
            )
        ),
        "sell_before_close_support_aware_threshold_selection_summary": (
            threshold_selection_summary
        ),
        "sell_before_close_validation_failure_drilldown_summary": (
            validation_failure_drilldown_summary
        ),
        "sell_before_close_i_vs_j_vs_k_promotion_support_comparison": (
            sell_before_close_promotion_support_gate.get(
                "i_vs_j_vs_k_promotion_support_comparison",
                [],
            )
        ),
        "sell_before_close_i_vs_j_vs_k_vs_l_promotion_support_comparison": (
            sell_before_close_promotion_support_gate.get(
                "i_vs_j_vs_k_vs_l_promotion_support_comparison",
                [],
            )
        ),
        "sell_before_close_i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison": (
            sell_before_close_promotion_support_gate.get(
                "i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison",
                [],
            )
        ),
        "i_vs_j_replay_comparison": sell_before_close_exit_reliability.get(
            "i_vs_j_replay_comparison",
            [],
        ),
        "i_vs_j_vs_k_replay_comparison": sell_before_close_exit_reliability.get(
            "i_vs_j_vs_k_replay_comparison",
            [],
        ),
        "i_vs_j_vs_k_vs_l_replay_comparison": sell_before_close_exit_reliability.get(
            "i_vs_j_vs_k_vs_l_replay_comparison",
            [],
        ),
        "i_vs_j_vs_k_vs_l_vs_m_replay_comparison": (
            sell_before_close_exit_reliability.get(
                "i_vs_j_vs_k_vs_l_vs_m_replay_comparison",
                [],
            )
        ),
        "sell_before_close_guard_threshold_sweep_summary": (
            _sell_before_close_guard_threshold_sweep_summary(
                sell_before_close_guard_threshold_sweep
            )
        ),
        "variant_count": len(variant_summaries),
        "variants": variant_summaries,
        **compact_safety_fields(),
    }


def _counterfactual_summary_with_source_gate(
    *,
    summary: dict[str, Any],
    source_model_candidate_eligible: bool,
    promotion_support_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons = set()
    if not source_model_candidate_eligible:
        reasons.add("source_model_paper_decision_ineligible")
    promotion_support_eligible = bool(
        (promotion_support_row or {}).get("promotion_support_eligible", False)
    )
    if promotion_support_row is not None and not promotion_support_eligible:
        reasons.update(promotion_support_row.get("support_gate_reason_codes", []))
    if float(summary["total_polymarket_pnl"]) <= 0.0:
        reasons.add("counterfactual_replay_pnl_not_positive")
    if int(summary["entry_decision_count"]) <= 0:
        reasons.add("counterfactual_replay_no_entry_decisions")
    promotion_evidence_eligible = not reasons
    summary["source_model_candidate_eligible"] = source_model_candidate_eligible
    summary["promotion_support_eligible"] = promotion_support_eligible
    summary["promotion_support_gate_passed"] = bool(
        (promotion_support_row or {}).get("support_gate_passed", False)
    )
    summary["promotion_support_reason_codes"] = list(
        (promotion_support_row or {}).get("support_gate_reason_codes", [])
    )
    summary["promotion_evidence_eligible"] = promotion_evidence_eligible
    summary["promotion_evidence_ineligible_reasons"] = sorted(reasons)
    summary["blocked"] = not promotion_evidence_eligible
    summary["blocked_reasons"] = sorted(reasons)
    summary["paper_run_resume_allowed"] = False
    summary["paper_run_resume_blocked_reason"] = "promotion_replay_gate_required"
    return summary


def _sell_before_close_guard_threshold_sweep_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    best = report.get("best_threshold_sweep_row")
    return {
        "schema_version": report["schema_version"],
        "candidate_name": report["candidate_name"],
        "diagnostic_only": True,
        "uses_shadow_for_fit": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "row_count": report["row_count"],
        "best_threshold_sweep_row": best,
        "best_threshold_sweep_support_gate_passed": report.get(
            "best_threshold_sweep_support_gate_passed"
        ),
        "best_threshold_sweep_support_gate_reason_codes": report.get(
            "best_threshold_sweep_support_gate_reason_codes",
            [],
        ),
    }


def _sell_before_close_guard_threshold_sweep_markdown(
    report: dict[str, Any],
) -> str:
    lines = [
        "# SELL_BEFORE_CLOSE Guard Threshold Sweep",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- uses_shadow_for_fit: `{str(report['uses_shadow_for_fit']).lower()}`",
        "- promotion_evidence_eligible: "
        f"`{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        f"- row_count: `{report['row_count']}`",
        "",
        "## Best Diagnostic Row",
        "",
    ]
    best = report.get("best_threshold_sweep_row")
    if best is None:
        lines.append("- none")
    else:
        lines.extend(
            [
                f"- thresholds: `{json.dumps(best['thresholds'], sort_keys=True)}`",
                f"- entry_count: `{best['entry_count']}`",
                f"- sell_count: `{best['sell_count']}`",
                f"- residual_count: `{best['residual_count']}`",
                f"- realized_trade_pnl: `{best['realized_trade_pnl']}`",
                f"- settlement_pnl: `{best['settlement_pnl']}`",
                f"- total_pnl: `{best['total_pnl']}`",
                f"- max_drawdown: `{best['max_drawdown']}`",
                f"- p_up_disagreement_rate: `{best['p_up_disagreement_rate']}`",
                f"- support_gate_passed: `{str(best['support_gate_passed']).lower()}`",
                "- support_gate_reason_codes: "
                f"`{json.dumps(best['support_gate_reason_codes'])}`",
                "- would_be_source_eligible_under_existing_gates: "
                f"`{str(best['would_be_source_eligible_under_existing_gates']).lower()}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| entry | sell | markets | sides | residual | total_pnl | residual_drag | p_up_disagreement | support | thresholds |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in report.get("rows", [])[:25]:
        lines.append(
            "| {entry} | {sell} | {markets} | {sides} | {residual} | "
            "{total:.6f} | {drag:.6f} | {p_up:.6f} | {support} | "
            "`{thresholds}` |".format(
                entry=row["entry_count"],
                sell=row["sell_count"],
                markets=row["unique_market_count"],
                sides=row["side_count"],
                residual=row["residual_count"],
                total=row["total_pnl"],
                drag=row["residual_settlement_drag"],
                p_up=row["p_up_disagreement_rate"],
                support=str(row["support_gate_passed"]).lower(),
                thresholds=json.dumps(row["thresholds"], sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )
    return "\n".join(lines)


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
    action_family_artifact_hashes: dict[str, str],
    source_model_eligibility: dict[str, Any],
    sell_before_close_p_up_disagreement_diagnostic: dict[str, Any],
    sell_before_close_exit_reliability: dict[str, Any],
    sell_before_close_promotion_support_gate: dict[str, Any],
    sell_before_close_support_aware_threshold_selection: dict[str, Any],
    sell_before_close_support_aware_threshold_failure_attribution: dict[str, Any],
    sell_before_close_validation_failure_drilldown: dict[str, Any],
    sell_before_close_guard_threshold_sweep: dict[str, Any],
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
    support_aware_threshold_selection_summary = (
        sell_before_close_support_aware_threshold_selection_summary(
            sell_before_close_support_aware_threshold_selection
        )
    )
    support_aware_failure_attribution_summary = (
        sell_before_close_support_aware_threshold_failure_attribution_summary(
            sell_before_close_support_aware_threshold_failure_attribution
        )
    )
    validation_failure_drilldown_summary = (
        sell_before_close_validation_failure_drilldown_summary(
            sell_before_close_validation_failure_drilldown
        )
    )
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
        "action_family_eligibility_sha256": action_family_artifact_hashes[
            "action_family_eligibility_sha256"
        ],
        "hold_to_settlement_longshot_guard_report_path": (
            "hold_to_settlement_longshot_guard_report.json"
        ),
        "hold_to_settlement_longshot_guard_sha256": action_family_artifact_hashes[
            "hold_to_settlement_longshot_guard_sha256"
        ],
        "action_family_replay_variants_report_path": (
            "action_family_replay_variants_report.json"
        ),
        "action_family_replay_variants_sha256": action_family_artifact_hashes[
            "action_family_replay_variants_sha256"
        ],
        "action_family_counterfactual_replay_report_path": (
            "action_family_counterfactual_replay_report.json"
        ),
        "action_family_counterfactual_replay_sha256": action_family_artifact_hashes[
            "action_family_counterfactual_replay_sha256"
        ],
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
        "model_ranking_error_report_path": "model_ranking_error_report.json",
        "model_ranking_error_report_sha256": action_family_artifact_hashes[
            "model_ranking_error_report_sha256"
        ],
        "model_ranking_candidate_comparison_path": (
            "model_ranking_candidate_comparison.json"
        ),
        "model_ranking_candidate_comparison_sha256": action_family_artifact_hashes[
            "model_ranking_candidate_comparison_sha256"
        ],
        "action_representation_diagnostic_report_path": (
            "action_representation_diagnostic_report.json"
        ),
        "action_representation_diagnostic_sha256": action_family_artifact_hashes[
            "action_representation_diagnostic_sha256"
        ],
        "ranking_overlay_zero_entry_diagnostic_report_path": (
            "ranking_overlay_zero_entry_diagnostic_report.json"
        ),
        "ranking_overlay_zero_entry_diagnostic_sha256": (
            action_family_artifact_hashes[
                "ranking_overlay_zero_entry_diagnostic_sha256"
            ]
        ),
        "source_model_eligibility_report_path": (
            "source_model_eligibility_report.json"
        ),
        "source_model_eligibility_report_sha256": action_family_artifact_hashes[
            "source_model_eligibility_report_sha256"
        ],
        "source_model_eligibility_report": source_model_eligibility,
        "sell_before_close_p_up_disagreement_diagnostic_report_path": (
            "sell_before_close_p_up_disagreement_diagnostic_report.json"
        ),
        "sell_before_close_p_up_disagreement_diagnostic_sha256": (
            action_family_artifact_hashes[
                "sell_before_close_p_up_disagreement_diagnostic_sha256"
            ]
        ),
        "sell_before_close_p_up_disagreement_interpretation": (
            sell_before_close_p_up_disagreement_diagnostic[
                "p_up_disagreement_interpretation"
            ]
        ),
        "sell_before_close_disagreed_total_pnl_sum": (
            sell_before_close_p_up_disagreement_diagnostic["summary"][
                "sell_before_close_disagreed_total_pnl_sum"
            ]
        ),
        "sell_before_close_agreed_total_pnl_sum": (
            sell_before_close_p_up_disagreement_diagnostic["summary"][
                "sell_before_close_agreed_total_pnl_sum"
            ]
        ),
        "sell_before_close_disagreed_trade_pnl_sum": (
            sell_before_close_p_up_disagreement_diagnostic["summary"][
                "sell_before_close_disagreed_trade_pnl_sum"
            ]
        ),
        "sell_before_close_label_row_settlement_pnl_sum": (
            sell_before_close_p_up_disagreement_diagnostic["summary"][
                "label_row_sell_before_close_settlement_pnl_sum"
            ]
        ),
        "sell_before_close_label_row_residual_settlement_drag": (
            sell_before_close_p_up_disagreement_diagnostic["summary"][
                "label_row_sell_before_close_residual_settlement_drag"
            ]
        ),
        "sell_before_close_settlement_drag_attribution_interpretation": (
            sell_before_close_p_up_disagreement_diagnostic["summary"][
                "settlement_drag_attribution_interpretation"
            ]
        ),
        "sell_before_close_p_up_disagreement_diagnostic_summary": (
            sell_before_close_p_up_disagreement_summary(
                sell_before_close_p_up_disagreement_diagnostic
            )
        ),
        "sell_before_close_exit_reliability_report_path": (
            "sell_before_close_exit_reliability_report.json"
        ),
        "sell_before_close_exit_reliability_report_sha256": (
            action_family_artifact_hashes[
                "sell_before_close_exit_reliability_report_sha256"
            ]
        ),
        "sell_before_close_exit_failure_interpretation": (
            sell_before_close_exit_reliability["summary"][
                "sell_before_close_exit_failure_interpretation"
            ]
        ),
        "sell_before_close_positions_opened_count": (
            sell_before_close_exit_reliability["summary"][
                "positions_opened_count"
            ]
        ),
        "sell_before_close_positions_closed_before_settlement_count": (
            sell_before_close_exit_reliability["summary"][
                "positions_closed_before_settlement_count"
            ]
        ),
        "sell_before_close_positions_opened_but_not_closed_before_settlement": (
            sell_before_close_exit_reliability["summary"][
                "positions_opened_but_not_closed_before_settlement"
            ]
        ),
        "sell_before_close_replay_realized_trade_pnl": (
            sell_before_close_exit_reliability["summary"]["realized_trade_pnl"]
        ),
        "sell_before_close_replay_settlement_pnl": (
            sell_before_close_exit_reliability["summary"]["settlement_pnl"]
        ),
        "sell_before_close_replay_total_polymarket_pnl": (
            sell_before_close_exit_reliability["summary"]["total_polymarket_pnl"]
        ),
        "sell_before_close_replay_residual_settlement_drag": (
            sell_before_close_exit_reliability["summary"][
                "replay_residual_settlement_drag"
            ]
        ),
        "sell_before_close_best_diagnostic_exit_variant": (
            sell_before_close_exit_reliability["summary"][
                "sell_before_close_best_diagnostic_exit_variant"
            ]
        ),
        "sell_before_close_best_diagnostic_exit_variant_total_pnl": (
            sell_before_close_exit_reliability["summary"][
                "sell_before_close_best_diagnostic_exit_variant_total_pnl"
            ]
        ),
        "sell_before_close_exit_reliability_summary": (
            sell_before_close_exit_reliability_summary(
                sell_before_close_exit_reliability
            )
        ),
        "sell_before_close_exit_reliability_guard_summary": (
            sell_before_close_exit_reliability.get(
                "exit_reliability_guard_candidate_summary"
            )
        ),
        "sell_before_close_exit_reliability_p_up_aligned_summary": (
            sell_before_close_exit_reliability.get(
                "exit_reliability_p_up_aligned_candidate_summary"
            )
        ),
        "sell_before_close_i_vs_j_replay_comparison": (
            sell_before_close_exit_reliability.get(
                "i_vs_j_replay_comparison",
                [],
            )
        ),
        "sell_before_close_i_vs_j_vs_k_replay_comparison": (
            sell_before_close_exit_reliability.get(
                "i_vs_j_vs_k_replay_comparison",
                [],
            )
        ),
        "sell_before_close_i_vs_j_vs_k_vs_l_replay_comparison": (
            sell_before_close_exit_reliability.get(
                "i_vs_j_vs_k_vs_l_replay_comparison",
                [],
            )
        ),
        "sell_before_close_i_vs_j_vs_k_vs_l_vs_m_replay_comparison": (
            sell_before_close_exit_reliability.get(
                "i_vs_j_vs_k_vs_l_vs_m_replay_comparison",
                [],
            )
        ),
        "sell_before_close_promotion_support_gate_report_path": (
            "sell_before_close_promotion_support_gate_report.json"
        ),
        "sell_before_close_promotion_support_gate_report_sha256": (
            action_family_artifact_hashes[
                "sell_before_close_promotion_support_gate_report_sha256"
            ]
        ),
        "sell_before_close_promotion_support_gate_summary": (
            sell_before_close_promotion_support_gate_summary(
                sell_before_close_promotion_support_gate
            )
        ),
        "sell_before_close_i_vs_j_vs_k_promotion_support_comparison": (
            sell_before_close_promotion_support_gate.get(
                "i_vs_j_vs_k_promotion_support_comparison",
                [],
            )
        ),
        "sell_before_close_i_vs_j_vs_k_vs_l_promotion_support_comparison": (
            sell_before_close_promotion_support_gate.get(
                "i_vs_j_vs_k_vs_l_promotion_support_comparison",
                sell_before_close_promotion_support_gate.get(
                    "i_vs_j_vs_k_promotion_support_comparison",
                    [],
                ),
            )
        ),
        "sell_before_close_i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison": (
            sell_before_close_promotion_support_gate.get(
                "i_vs_j_vs_k_vs_l_vs_m_promotion_support_comparison",
                sell_before_close_promotion_support_gate.get(
                    "i_vs_j_vs_k_vs_l_promotion_support_comparison",
                    [],
                ),
            )
        ),
        "sell_before_close_support_aware_threshold_selection_report_path": (
            "sell_before_close_support_aware_threshold_selection_report.json"
        ),
        "sell_before_close_support_aware_threshold_selection_report_sha256": (
            action_family_artifact_hashes[
                "sell_before_close_support_aware_threshold_selection_report_sha256"
            ]
        ),
        "sell_before_close_support_aware_threshold_selection_summary": (
            support_aware_threshold_selection_summary
        ),
        "sell_before_close_support_aware_threshold_failure_attribution_report_path": (
            "sell_before_close_support_aware_threshold_failure_attribution_report.json"
        ),
        "sell_before_close_support_aware_threshold_failure_attribution_report_sha256": (
            action_family_artifact_hashes[
                "sell_before_close_support_aware_threshold_failure_attribution_report_sha256"
            ]
        ),
        "sell_before_close_support_aware_threshold_failure_attribution_summary": (
            support_aware_failure_attribution_summary
        ),
        "sell_before_close_validation_failure_drilldown_report_path": (
            "sell_before_close_validation_failure_drilldown_report.json"
        ),
        "sell_before_close_validation_failure_drilldown_report_sha256": (
            action_family_artifact_hashes[
                "sell_before_close_validation_failure_drilldown_report_sha256"
            ]
        ),
        "sell_before_close_validation_failure_drilldown_summary": (
            validation_failure_drilldown_summary
        ),
        "sell_before_close_side_balanced_candidate_report_path": (
            "sell_before_close_side_balanced_candidate_report.json"
        ),
        "sell_before_close_side_balanced_candidate_report_sha256": (
            action_family_artifact_hashes[
                "sell_before_close_side_balanced_candidate_report_sha256"
            ]
        ),
        "sell_before_close_side_balanced_candidate_summary": (
            source_model_eligibility.get(
                "sell_before_close_side_balanced_candidate_summary",
                {},
            )
        ),
        "sell_before_close_validation_failure_primary_interpretation": (
            validation_failure_drilldown_summary["primary_failure_interpretation"]
        ),
        "sell_before_close_validation_failure_interpretations": (
            validation_failure_drilldown_summary["failure_interpretations"]
        ),
        "sell_before_close_validation_failure_recommended_next_actions": (
            validation_failure_drilldown_summary["recommended_next_actions"]
        ),
        "sell_before_close_validation_support_adequate_row_count": (
            validation_failure_drilldown_summary["support_adequate_row_count"]
        ),
        "sell_before_close_validation_positive_pnl_row_count": (
            validation_failure_drilldown_summary["positive_pnl_row_count"]
        ),
        "sell_before_close_validation_support_passed_pnl_failed_count": (
            validation_failure_drilldown_summary[
                "support_passed_pnl_failed_count"
            ]
        ),
        "sell_before_close_validation_support_failed_pnl_passed_count": (
            validation_failure_drilldown_summary[
                "support_failed_pnl_passed_count"
            ]
        ),
        "sell_before_close_validation_side_coverage_failure_rate": (
            validation_failure_drilldown_summary["side_coverage_failure_rate"]
        ),
        "sell_before_close_support_aware_threshold_selection_passed": (
            support_aware_threshold_selection_summary["threshold_selection_passed"]
        ),
        "sell_before_close_support_aware_threshold_selection_failed": (
            support_aware_threshold_selection_summary["threshold_selection_failed"]
        ),
        "sell_before_close_support_aware_threshold_selection_failure_reason_codes": (
            support_aware_threshold_selection_summary[
                "threshold_selection_failure_reason_codes"
            ]
        ),
        "sell_before_close_support_aware_threshold_selection_failure_interpretation": (
            support_aware_threshold_selection_summary[
                "threshold_selection_failure_interpretation"
            ]
        ),
        "sell_before_close_support_aware_recommended_next_action": (
            support_aware_threshold_selection_summary["recommended_next_action"]
        ),
        "sell_before_close_support_aware_validation_row_count": (
            support_aware_threshold_selection_summary["validation_row_count"]
        ),
        "sell_before_close_support_aware_validation_passing_row_count": (
            support_aware_threshold_selection_summary[
                "validation_passing_row_count"
            ]
        ),
        "sell_before_close_support_aware_top_failed_gates": (
            support_aware_threshold_selection_summary["top_failed_gates"]
        ),
        "threshold_selection_passed": (
            support_aware_threshold_selection_summary["threshold_selection_passed"]
        ),
        "threshold_selection_failed": (
            support_aware_threshold_selection_summary["threshold_selection_failed"]
        ),
        "threshold_selection_failure_reason_codes": (
            support_aware_threshold_selection_summary[
                "threshold_selection_failure_reason_codes"
            ]
        ),
        "threshold_selection_failure_interpretation": (
            support_aware_threshold_selection_summary[
                "threshold_selection_failure_interpretation"
            ]
        ),
        "recommended_next_action": (
            support_aware_threshold_selection_summary["recommended_next_action"]
        ),
        "threshold_selection_validation_row_count": (
            support_aware_threshold_selection_summary["validation_row_count"]
        ),
        "threshold_selection_validation_passing_row_count": (
            support_aware_threshold_selection_summary[
                "validation_passing_row_count"
            ]
        ),
        "top_failed_gates": (
            support_aware_threshold_selection_summary["top_failed_gates"]
        ),
        "sell_before_close_promotion_support_gate_passed": (
            sell_before_close_promotion_support_gate["support_gate_passed"]
        ),
        "sell_before_close_promotion_support_reason_codes": (
            sell_before_close_promotion_support_gate["support_gate_reason_codes"]
        ),
        "sell_before_close_promotion_support_entry_decision_count": (
            sell_before_close_promotion_support_gate["entry_decision_count"]
        ),
        "sell_before_close_promotion_support_unique_market_count": (
            sell_before_close_promotion_support_gate["unique_market_count"]
        ),
        "sell_before_close_promotion_support_side_count": (
            sell_before_close_promotion_support_gate["side_count"]
        ),
        "sell_before_close_promotion_support_sell_decision_count": (
            sell_before_close_promotion_support_gate["sell_decision_count"]
        ),
        "sell_before_close_promotion_support_total_pnl": (
            sell_before_close_promotion_support_gate["total_pnl"]
        ),
        "sell_before_close_promotion_support_mean_pnl_per_entry": (
            sell_before_close_promotion_support_gate["mean_pnl_per_entry"]
        ),
        "sell_before_close_guard_threshold_sweep_report_path": (
            "sell_before_close_guard_threshold_sweep_report.json"
        ),
        "sell_before_close_guard_threshold_sweep_report_sha256": (
            action_family_artifact_hashes[
                "sell_before_close_guard_threshold_sweep_report_sha256"
            ]
        ),
        "sell_before_close_guard_threshold_sweep_summary": (
            _sell_before_close_guard_threshold_sweep_summary(
                sell_before_close_guard_threshold_sweep
            )
        ),
        "candidate_scoped_source_model_eligibility_summary": (
            source_model_eligibility.get("candidate_scoped_eligibility_summary", [])
        ),
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
        "sell_before_close_label_schema_version": dataset_profile[
            "sell_before_close_label_schema_version"
        ],
        "sell_before_close_fixed_terminal_bid_only_labels_allowed": (
            dataset_profile[
                "sell_before_close_fixed_terminal_bid_only_labels_allowed"
            ]
        ),
        "sell_before_close_label_gate_passed": dataset_profile[
            "sell_before_close_label_gate_passed"
        ],
        "sell_before_close_execution_class_counts": dataset_profile[
            "sell_before_close_execution_class_counts"
        ],
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


def _counterfactual_replay_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Action-Family Counterfactual Replay Report",
        "",
        f"- report_mode: {report['report_mode']}",
        f"- variant_count: {report['variant_count']}",
        "- promotion_evidence_eligible: "
        f"{str(report['promotion_evidence_eligible']).lower()}",
        "- promotion_evidence_ineligible_reasons: "
        f"{json.dumps(report['promotion_evidence_ineligible_reasons'])}",
        "",
        "## Variants",
        "",
    ]
    for variant in report["variants"]:
        lines.append(
            "- "
            f"{variant['variant']}: "
            f"threshold={variant['ev_threshold']} "
            f"entries={variant['entry_decision_count']} "
            f"trades={variant['trade_count']} "
            f"pnl={variant['total_polymarket_pnl']} "
            f"max_drawdown={variant['max_drawdown']} "
            f"actions={json.dumps(variant['action_counts'], sort_keys=True)} "
            f"blocked={str(variant['blocked']).lower()} "
            f"reasons={json.dumps(variant['blocked_reasons'])}"
        )
    lines.extend(
        [
            "",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )
    return "\n".join(lines)


def _predictions_for_examples(
    predictions_by_key: dict[tuple[str, int], Any],
    examples: tuple[Any, ...],
) -> tuple[Any, ...]:
    return tuple(
        predictions_by_key[(example.market_id, example.decision_ts)]
        for example in examples
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _relative_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
