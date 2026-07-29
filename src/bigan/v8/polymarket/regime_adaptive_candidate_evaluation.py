"""Strict rolling-origin development evaluation for the regime-adaptive lineage."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import (
    atomic_write_json,
    sha256_file,
)
from bigan.v8.polymarket.challenge_model_15m_training import (
    BASE_FEATURE_NAMES,
    SIDES,
    _apply_pair_probability_normalization,
    _load_side_symmetric_rows,
    _matrix,
    _verify_finalized_index,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.regime_adaptive_lineage import (
    LINEAGE_ID,
    REPO_ROOT,
    SAFETY,
    validate_frozen_protocol_graph,
)

REPORT_SCHEMA_VERSION = (
    "bigan-btc-15m-regime-adaptive-development-evaluation-report-v1"
)
MANIFEST_SCHEMA_VERSION = (
    "bigan-btc-15m-regime-adaptive-development-evaluation-manifest-v1"
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *(f"{name}__missing" for name in BASE_FEATURE_NAMES),
)


def run_regime_adaptive_development_evaluation(
    *,
    config_dir: Path | str,
    output_dir: Path | str,
    source_commit: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Evaluate exactly five frozen candidates on development-only rolling OOF."""

    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    configs = validate_frozen_protocol_graph(config_dir)
    family = configs["candidate_family_protocol.json"]
    evaluation = configs["rolling_origin_evaluation_protocol.json"]
    feature_contract = configs["regime_feature_contract.json"]
    budget = configs["candidate_budget_protocol.json"]

    index_descriptor = dict(evaluation["inputs"]["development_corpus_index"])
    index_path = (REPO_ROOT / str(index_descriptor["path"])).resolve()
    if (
        not index_path.is_relative_to(REPO_ROOT)
        or not index_path.is_file()
        or sha256_file(index_path) != index_descriptor["sha256"]
    ):
        raise ValueError("frozen development corpus index is unavailable or changed")
    index_rows = _verify_finalized_index(
        index_path=index_path,
        repo_root=REPO_ROOT,
    )
    rows, input_corpora = _load_side_symmetric_rows(
        index_rows,
        repo_root=REPO_ROOT,
    )
    for row in rows:
        _annotate_regime(row, feature_contract)
    ordered_markets = sorted(
        {
            (int(row["market_start_ts"]), str(row["market_id"]))
            for row in rows
        }
    )
    initial_count = int(
        evaluation["development_rolling_origin"][
            "initial_strictly_prior_training_markets"
        ]
    )
    if len(ordered_markets) != int(index_descriptor["market_count"]):
        raise ValueError("development market population changed after freeze")
    if len(ordered_markets) - initial_count != int(
        evaluation["development_rolling_origin"]["evaluation_market_count"]
    ):
        raise ValueError("rolling-origin evaluation population is inconsistent")
    rows_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)

    candidate_results: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    candidate_definitions = list(family["candidates"])
    for candidate in candidate_definitions:
        candidate_id = str(candidate["candidate_id"])
        oof_rows, folds = _evaluate_candidate(
            candidate_id=candidate_id,
            candidate=candidate,
            family=family,
            rows_by_market=rows_by_market,
            ordered_markets=ordered_markets,
            initial_count=initial_count,
        )
        metrics = _candidate_metrics(
            candidate_id=candidate_id,
            candidate_ordinal=int(candidate["ordinal"]),
            oof_rows=oof_rows,
            ordered_oof_markets=ordered_markets[initial_count:],
            evaluation=evaluation,
        )
        candidate_results.append(metrics)
        prediction_records.extend(
            _prediction_record(candidate_id, row) for row in oof_rows
        )
        fold_records.extend(folds)

    selection = _select_candidate(candidate_results, evaluation)
    generated_at = created_at or datetime.now(tz=UTC).isoformat()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"development evaluation output directory must be empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "development_oof_predictions.jsonl"
    folds_path = output / "development_fold_audits.jsonl"
    _atomic_write_jsonl(predictions_path, prediction_records)
    _atomic_write_jsonl(folds_path, fold_records)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "role": "candidate_development_selection_only",
        "created_at": generated_at,
        "source_commit": source_commit,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "fresh_validation_evidence_eligible": False,
        "parent_oof_reopened_as_validation": False,
        "candidate_count": len(candidate_results),
        "candidate_ids": [
            str(candidate["candidate_id"]) for candidate in candidate_definitions
        ],
        "development_market_count": len(ordered_markets),
        "initial_strictly_prior_training_market_count": initial_count,
        "rolling_origin_evaluation_market_count": len(ordered_markets) - initial_count,
        "target_or_future_label_leakage_count": sum(
            int(candidate["target_or_future_label_leakage_count"])
            for candidate in candidate_results
        ),
        "candidate_results": candidate_results,
        "selection": selection,
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "model_training_started": True,
        "model_training_completed": True,
        "candidate_budget_executions_consumed": len(candidate_results),
        "candidate_budget_executions_maximum": int(
            budget["candidate_budget"][
                "maximum_total_development_rolling_origin_executions"
            ]
        ),
        "artifacts": {
            "predictions": _descriptor(predictions_path, root=output),
            "fold_audits": _descriptor(folds_path, root=output),
        },
        "input_corpus_manifest_count": len(input_corpora),
        "input_corpus_manifest_set_sha256": canonical_json_sha256(input_corpora),
        "interpretation": (
            "development_selection_only_no_validation_promotion_or_execution_claim"
        ),
        "safety": dict(SAFETY),
    }
    report_path = output / "development_evaluation_report.json"
    atomic_write_json(report_path, report)
    markdown_path = output / "development_evaluation_report.md"
    _atomic_write_text(markdown_path, render_development_evaluation_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "created_at": generated_at,
        "source_commit": source_commit,
        "frozen_protocols": {
            filename: _descriptor(
                Path(config_dir).resolve() / filename,
                root=REPO_ROOT,
            )
            for filename in (
                "regime_adaptive_model_protocol.json",
                "lineage_manifest.json",
                "temporal_drift_diagnostic_report.json",
                "regime_feature_contract.json",
                "candidate_family_protocol.json",
                "rolling_origin_evaluation_protocol.json",
                "candidate_budget_protocol.json",
            )
        },
        "development_corpus_index": _descriptor(index_path, root=REPO_ROOT),
        "artifacts": {
            "report": _descriptor(report_path, root=output),
            "report_markdown": _descriptor(markdown_path, root=output),
            "predictions": _descriptor(predictions_path, root=output),
            "fold_audits": _descriptor(folds_path, root=output),
        },
        "selected_candidate_id": selection["selected_candidate_id"],
        "fresh_collection_authorized": False,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_path = output / "development_evaluation_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection": selection,
        "safety": dict(SAFETY),
    }


def render_development_evaluation_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render the development-only candidate comparison."""

    lines = [
        "# BTC 15m regime-adaptive development evaluation",
        "",
        f"- Lineage: `{report['lineage_id']}`",
        "- Evidence role: development selection only; never promotion evidence",
        f"- Rolling-origin markets: {report['rolling_origin_evaluation_market_count']}",
        f"- Candidate executions: {report['candidate_budget_executions_consumed']}/"
        f"{report['candidate_budget_executions_maximum']}",
        "- Fresh collection authorized: no",
        "",
        "## Candidate results",
        "",
        "| Candidate | Accepted | PnL | 95% LCB | First | Second | Eligible |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for candidate in report["candidate_results"]:
        trading = candidate["trading_metrics"]
        interval = trading["mean_unit_net_pnl_bootstrap_interval"]
        lines.append(
            f"| {candidate['candidate_id']} | "
            f"{trading['accepted_market_count']} | "
            f"{trading['total_unit_net_pnl']:.6f} | "
            f"{interval['lower']:.6f} | "
            f"{trading['first_chronological_half_total_unit_net_pnl']:.6f} | "
            f"{trading['second_chronological_half_total_unit_net_pnl']:.6f} | "
            f"{'yes' if candidate['development_selection_eligible'] else 'no'} |"
        )
    selection = report["selection"]
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Status: `{selection['status']}`",
            f"- Selected candidate: `{selection['selected_candidate_id']}`",
            f"- Fresh collection allowed by this result: "
            f"`{str(selection['fresh_collection_allowed']).lower()}`",
            "",
            "The result is outcome-aware development evidence. It cannot be reused "
            "as validation evidence and makes no promotion, paper, live, wallet, "
            "write, or capital-at-risk claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _evaluate_candidate(
    *,
    candidate_id: str,
    candidate: Mapping[str, Any],
    family: Mapping[str, Any],
    rows_by_market: Mapping[str, Sequence[dict[str, Any]]],
    ordered_markets: Sequence[tuple[int, str]],
    initial_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameters = dict(family["shared_base_learner"]["parameters"])
    oof_rows: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    for target_index in range(initial_count, len(ordered_markets)):
        prior_market_ids = [
            market_id for _, market_id in ordered_markets[:target_index]
        ]
        target_market_id = ordered_markets[target_index][1]
        target_rows = [dict(row) for row in rows_by_market[target_market_id]]
        if candidate_id == "drift_aware_rolling_calibration":
            predicted, audit = _predict_drift_calibrated(
                prior_market_ids=prior_market_ids,
                target_rows=target_rows,
                rows_by_market=rows_by_market,
                parameters=parameters,
                family=family,
                candidate=candidate,
            )
        else:
            base_target, calibration_rows, base_audit = _base_fold_prediction(
                prior_market_ids=prior_market_ids,
                target_rows=target_rows,
                rows_by_market=rows_by_market,
                parameters=parameters,
                family=family,
            )
            if candidate_id == "global_baseline":
                predicted = base_target
                audit = base_audit
            elif candidate_id == "regime_conditioned_calibration":
                predicted = _apply_regime_calibration(
                    target_rows=base_target,
                    calibration_rows=calibration_rows,
                    candidate=candidate,
                )
                audit = {
                    **base_audit,
                    "calibration_side_row_count": len(calibration_rows),
                }
            elif candidate_id == "mixture_of_experts":
                predicted, expert_audit = _predict_mixture_of_experts(
                    prior_market_ids=prior_market_ids,
                    target_rows=target_rows,
                    base_target=base_target,
                    rows_by_market=rows_by_market,
                    parameters=parameters,
                    num_boost_round=int(base_audit["best_iteration"]) + 1,
                    candidate=candidate,
                )
                audit = {**base_audit, **expert_audit}
            elif candidate_id == "uncertainty_aware_abstention":
                predicted, ensemble_audit = _predict_uncertainty_ensemble(
                    prior_market_ids=prior_market_ids,
                    target_rows=target_rows,
                    rows_by_market=rows_by_market,
                    parameters=parameters,
                    num_boost_round=int(base_audit["best_iteration"]) + 1,
                    candidate=candidate,
                )
                audit = {**base_audit, **ensemble_audit}
            else:
                raise ValueError(f"unknown frozen candidate: {candidate_id}")
        for row in predicted:
            row["candidate_id"] = candidate_id
            row["oof_target_rank"] = target_index + 1
            row["strictly_prior_training_market_count"] = len(prior_market_ids)
        oof_rows.extend(predicted)
        fold_audits.append(
            {
                "candidate_id": candidate_id,
                "target_market_id": target_market_id,
                "target_market_rank": target_index + 1,
                "strictly_prior_market_count": len(prior_market_ids),
                "strictly_prior_market_ids_sha256": canonical_json_sha256(
                    prior_market_ids
                ),
                "target_market_used_for_fit": False,
                "future_market_used_for_fit": False,
                "target_or_future_label_leakage_count": 0,
                **audit,
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
            }
        )
    return oof_rows, fold_audits


def _base_fold_prediction(
    *,
    prior_market_ids: Sequence[str],
    target_rows: Sequence[dict[str, Any]],
    rows_by_market: Mapping[str, Sequence[dict[str, Any]]],
    parameters: Mapping[str, Any],
    family: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validation_count = min(
        int(family["shared_base_learner"]["validation_tail_market_count"]),
        len(prior_market_ids)
        - int(family["shared_base_learner"]["minimum_core_training_market_count"]),
    )
    if validation_count <= 0:
        raise ValueError("base fold has no strictly prior validation tail")
    train_ids = prior_market_ids[:-validation_count]
    validation_ids = prior_market_ids[-validation_count:]
    train_rows = [
        row for market_id in train_ids for row in rows_by_market[market_id]
    ]
    validation_rows = [
        dict(row)
        for market_id in validation_ids
        for row in rows_by_market[market_id]
    ]
    predicted_target, predicted_validation, booster = _fit_predict(
        train_rows=train_rows,
        validation_rows=validation_rows,
        target_rows=target_rows,
        parameters=parameters,
        num_boost_round=int(family["shared_base_learner"]["num_boost_round"]),
        early_stopping_rounds=int(
            family["shared_base_learner"]["early_stopping_rounds"]
        ),
    )
    return (
        predicted_target,
        predicted_validation,
        {
            "inner_train_market_count": len(train_ids),
            "inner_validation_market_count": len(validation_ids),
            "best_iteration": int(booster.best_iteration),
            "best_score": float(booster.best_score),
        },
    )


def _fit_predict(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
    parameters: Mapping[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], xgb.Booster]:
    train_matrix = _matrix(
        train_rows,
        FEATURE_NAMES,
        label_field="settlement_payout",
    )
    validation_matrix = _matrix(
        validation_rows,
        FEATURE_NAMES,
        label_field="settlement_payout",
    )
    target_copy = [dict(row) for row in target_rows]
    target_matrix = _matrix(
        target_copy,
        FEATURE_NAMES,
        label_field="settlement_payout",
    )
    booster = xgb.train(
        params=dict(parameters),
        dtrain=train_matrix,
        num_boost_round=num_boost_round,
        evals=[(train_matrix, "train"), (validation_matrix, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    iteration_range = (0, int(booster.best_iteration) + 1)
    _attach_probabilities(
        target_copy,
        booster.predict(target_matrix, iteration_range=iteration_range),
    )
    _attach_probabilities(
        validation_rows,
        booster.predict(validation_matrix, iteration_range=iteration_range),
    )
    return target_copy, validation_rows, booster


def _attach_probabilities(
    rows: Sequence[dict[str, Any]],
    probabilities: Sequence[float],
) -> None:
    for row, probability in zip(rows, probabilities, strict=True):
        row["raw_win_probability"] = float(probability)
        row["win_probability"] = float(probability)
        row["selection_score"] = float(probability) - float(row["execution_cost"])
        row["prediction"] = row["selection_score"]
    _apply_pair_probability_normalization(rows)
    for row in rows:
        row["selection_score"] = float(row["prediction"])


def _apply_regime_calibration(
    *,
    target_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    minimum_rows = int(candidate["minimum_regime_calibration_side_rows"])
    global_calibrator = _fit_platt(calibration_rows, candidate["calibrator"])
    calibrators: dict[str, tuple[float, float]] = {}
    for regime in candidate["regime_values"]:
        rows = [
            row
            for row in calibration_rows
            if row["btc_return_regime"] == regime
        ]
        calibrators[str(regime)] = (
            _fit_platt(rows, candidate["calibrator"])
            if len(rows) >= minimum_rows
            else global_calibrator
        )
    predicted = [dict(row) for row in target_rows]
    for row in predicted:
        intercept, slope = calibrators[str(row["btc_return_regime"])]
        probability = _platt_predict(
            float(row["win_probability"]),
            intercept,
            slope,
        )
        row["raw_win_probability"] = probability
        row["win_probability"] = probability
        row["prediction"] = probability - float(row["execution_cost"])
    _apply_pair_probability_normalization(predicted)
    for row in predicted:
        row["selection_score"] = float(row["prediction"])
    return predicted


def _predict_drift_calibrated(
    *,
    prior_market_ids: Sequence[str],
    target_rows: Sequence[dict[str, Any]],
    rows_by_market: Mapping[str, Sequence[dict[str, Any]]],
    parameters: Mapping[str, Any],
    family: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    minimum_core = int(
        family["shared_base_learner"]["minimum_core_training_market_count"]
    )
    window_count = min(
        int(candidate["calibration_window_market_count"]),
        len(prior_market_ids) - minimum_core,
    )
    if window_count < int(candidate["minimum_calibration_market_count"]):
        base, _, audit = _base_fold_prediction(
            prior_market_ids=prior_market_ids,
            target_rows=target_rows,
            rows_by_market=rows_by_market,
            parameters=parameters,
            family=family,
        )
        return base, {**audit, "drift_calibration_applied": False}
    core_ids = prior_market_ids[:-window_count]
    calibration_ids = prior_market_ids[-window_count:]
    core_rows = [row for market_id in core_ids for row in rows_by_market[market_id]]
    calibration_rows = [
        dict(row)
        for market_id in calibration_ids
        for row in rows_by_market[market_id]
    ]
    base_target, predicted_calibration, booster = _fit_predict(
        train_rows=core_rows,
        validation_rows=calibration_rows,
        target_rows=target_rows,
        parameters=parameters,
        num_boost_round=int(family["shared_base_learner"]["num_boost_round"]),
        early_stopping_rounds=int(
            family["shared_base_learner"]["early_stopping_rounds"]
        ),
    )
    calibrator = _fit_platt(predicted_calibration, candidate["calibrator"])
    predicted = [dict(row) for row in base_target]
    for row in predicted:
        probability = _platt_predict(float(row["win_probability"]), *calibrator)
        row["raw_win_probability"] = probability
        row["win_probability"] = probability
        row["prediction"] = probability - float(row["execution_cost"])
    _apply_pair_probability_normalization(predicted)
    for row in predicted:
        row["selection_score"] = float(row["prediction"])
    return (
        predicted,
        {
            "inner_train_market_count": len(core_ids),
            "inner_validation_market_count": len(calibration_ids),
            "best_iteration": int(booster.best_iteration),
            "best_score": float(booster.best_score),
            "drift_calibration_applied": True,
            "drift_calibration_market_count": len(calibration_ids),
        },
    )


def _fit_platt(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[float, float]:
    probabilities = np.clip(
        np.asarray(
            [float(row["win_probability"]) for row in rows],
            dtype=np.float64,
        ),
        1e-9,
        1.0 - 1e-9,
    )
    outcomes = np.asarray(
        [float(row["settlement_payout"]) for row in rows],
        dtype=np.float64,
    )
    logits = np.log(probabilities / (1.0 - probabilities))
    design = np.column_stack((np.ones(len(logits)), logits))
    coefficients = np.zeros(2, dtype=np.float64)
    penalty = float(config["l2_penalty"])
    for _ in range(int(config["maximum_iterations"])):
        fitted = 1.0 / (
            1.0 + np.exp(-np.clip(design @ coefficients, -30.0, 30.0))
        )
        weights = np.clip(fitted * (1.0 - fitted), 1e-9, None)
        hessian = design.T @ (weights[:, None] * design)
        hessian += np.diag((0.0, penalty))
        gradient = design.T @ (outcomes - fitted)
        gradient -= np.asarray((0.0, penalty * coefficients[1]))
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if float(np.max(np.abs(step))) < float(config["convergence_tolerance"]):
            break
    return float(coefficients[0]), float(coefficients[1])


def _platt_predict(
    probability: float,
    intercept: float,
    slope: float,
) -> float:
    clipped = min(max(probability, 1e-9), 1.0 - 1e-9)
    logit = math.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + math.exp(-max(min(intercept + slope * logit, 30.0), -30.0)))


def _predict_mixture_of_experts(
    *,
    prior_market_ids: Sequence[str],
    target_rows: Sequence[dict[str, Any]],
    base_target: Sequence[dict[str, Any]],
    rows_by_market: Mapping[str, Sequence[dict[str, Any]]],
    parameters: Mapping[str, Any],
    num_boost_round: int,
    candidate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predicted = [dict(row) for row in base_target]
    route_audit: dict[str, Any] = {}
    target_routes = {str(row["expert_route"]) for row in target_rows}
    prior_rows = [
        row for market_id in prior_market_ids for row in rows_by_market[market_id]
    ]
    for route in sorted(target_routes):
        expert_rows = [
            row for row in prior_rows if str(row["expert_route"]) == route
        ]
        expert_market_count = len(
            {str(row["market_id"]) for row in expert_rows}
        )
        target_indexes = [
            index
            for index, row in enumerate(target_rows)
            if str(row["expert_route"]) == route
        ]
        if expert_market_count < int(candidate["minimum_expert_training_markets"]):
            route_audit[f"expert_{route}_fallback"] = True
            continue
        matrix = _matrix(
            expert_rows,
            FEATURE_NAMES,
            label_field="settlement_payout",
        )
        booster = xgb.train(
            params=dict(parameters),
            dtrain=matrix,
            num_boost_round=max(1, num_boost_round),
            verbose_eval=False,
        )
        route_target = [dict(target_rows[index]) for index in target_indexes]
        target_matrix = _matrix(
            route_target,
            FEATURE_NAMES,
            label_field="settlement_payout",
        )
        probabilities = booster.predict(target_matrix)
        for index, probability in zip(target_indexes, probabilities, strict=True):
            predicted[index]["raw_win_probability"] = float(probability)
            predicted[index]["win_probability"] = float(probability)
        route_audit[f"expert_{route}_fallback"] = False
        route_audit[f"expert_{route}_training_market_count"] = expert_market_count
    _apply_pair_probability_normalization(predicted)
    for row in predicted:
        row["selection_score"] = float(row["prediction"])
    return predicted, route_audit


def _predict_uncertainty_ensemble(
    *,
    prior_market_ids: Sequence[str],
    target_rows: Sequence[dict[str, Any]],
    rows_by_market: Mapping[str, Sequence[dict[str, Any]]],
    parameters: Mapping[str, Any],
    num_boost_round: int,
    candidate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(set(prior_market_ids)) < int(candidate["minimum_distinct_training_markets"]):
        raise ValueError("uncertainty ensemble lacks required distinct prior markets")
    member_probabilities: list[list[float]] = []
    for seed in candidate["ensemble_seeds"]:
        generator = np.random.default_rng(int(seed))
        sampled_ids = list(
            generator.choice(
                np.asarray(prior_market_ids, dtype=object),
                size=len(prior_market_ids),
                replace=True,
            )
        )
        sampled_rows = [
            row for market_id in sampled_ids for row in rows_by_market[str(market_id)]
        ]
        member_parameters = dict(parameters)
        member_parameters["seed"] = int(seed)
        matrix = _matrix(
            sampled_rows,
            FEATURE_NAMES,
            label_field="settlement_payout",
        )
        booster = xgb.train(
            params=member_parameters,
            dtrain=matrix,
            num_boost_round=max(1, num_boost_round),
            verbose_eval=False,
        )
        target_copy = [dict(row) for row in target_rows]
        target_matrix = _matrix(
            target_copy,
            FEATURE_NAMES,
            label_field="settlement_payout",
        )
        probabilities = booster.predict(target_matrix)
        _attach_probabilities(target_copy, probabilities)
        member_probabilities.append(
            [float(row["win_probability"]) for row in target_copy]
        )
    values = np.asarray(member_probabilities, dtype=np.float64)
    mean_probabilities = np.mean(values, axis=0)
    predicted = [dict(row) for row in target_rows]
    for index, row in enumerate(predicted):
        edges = values[:, index] - float(row["execution_cost"])
        row["raw_win_probability"] = float(mean_probabilities[index])
        row["win_probability"] = float(mean_probabilities[index])
        row["prediction"] = float(mean_probabilities[index]) - float(
            row["execution_cost"]
        )
        row["selection_score"] = float(
            np.quantile(edges, float(candidate["uncertainty_edge_quantile"]))
        )
        row["uncertainty_edge_quantile"] = row["selection_score"]
    _apply_pair_probability_normalization(predicted)
    return (
        predicted,
        {
            "ensemble_member_count": len(member_probabilities),
            "ensemble_num_boost_round": max(1, num_boost_round),
        },
    )


def _candidate_metrics(
    *,
    candidate_id: str,
    candidate_ordinal: int,
    oof_rows: Sequence[Mapping[str, Any]],
    ordered_oof_markets: Sequence[tuple[int, str]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _selected_rows(oof_rows)
    selected_by_market = {str(row["market_id"]): row for row in selected}
    market_pnl = [
        float(selected_by_market[market_id]["target"])
        if market_id in selected_by_market
        else 0.0
        for _, market_id in ordered_oof_markets
    ]
    bootstrap = dict(evaluation["trading_metrics"]["bootstrap"])
    interval = _bootstrap_interval(
        market_pnl,
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    midpoint = len(ordered_oof_markets) // 2
    first_ids = {
        market_id for _, market_id in ordered_oof_markets[:midpoint]
    }
    second_ids = {
        market_id for _, market_id in ordered_oof_markets[midpoint:]
    }
    largest_removed = (
        sum(market_pnl) - max(market_pnl) if market_pnl else 0.0
    )
    probability = _probability_metrics(oof_rows)
    costs = {
        name: sum(float(row[name]) for row in selected)
        for name in (
            "gross_price_edge",
            "entry_spread_cost",
            "fees",
            "slippage",
            "liquidity_impact",
        )
    }
    total_cost = (
        costs["entry_spread_cost"]
        + costs["fees"]
        + costs["slippage"]
        + costs["liquidity_impact"]
    )
    trading = {
        "market_count": len(ordered_oof_markets),
        "accepted_market_count": len(selected),
        "acceptance_rate": len(selected) / len(ordered_oof_markets),
        "accepted_up_count": sum(row["side"] == "UP" for row in selected),
        "accepted_down_count": sum(row["side"] == "DOWN" for row in selected),
        "total_unit_net_pnl": sum(market_pnl),
        "mean_unit_net_pnl": float(np.mean(market_pnl)),
        "mean_unit_net_pnl_bootstrap_interval": interval,
        "largest_winner_removed_total_unit_net_pnl": largest_removed,
        "first_chronological_half_total_unit_net_pnl": sum(
            float(row["target"])
            for row in selected
            if row["market_id"] in first_ids
        ),
        "second_chronological_half_total_unit_net_pnl": sum(
            float(row["target"])
            for row in selected
            if row["market_id"] in second_ids
        ),
        **costs,
        "total_cost": total_cost,
        "cost_signal_ratio": (
            total_cost / costs["gross_price_edge"]
            if costs["gross_price_edge"] > 0.0
            else None
        ),
    }
    stratified = {
        field: _stratified_selected_metrics(
            selected,
            ordered_oof_markets,
            field,
        )
        for field in (
            "btc_return_regime",
            "side",
            "volatility_bucket",
            "spread_bucket",
            "volume_bucket",
            "depth_bucket",
        )
    }
    thresholds = dict(
        evaluation["development_candidate_selection_rule"][
            "eligibility_requirements"
        ]
    )
    gate_results = {
        "target_or_future_label_leakage": True,
        "minimum_accepted_market_count": (
            trading["accepted_market_count"]
            >= int(thresholds["minimum_accepted_market_count"])
        ),
        "minimum_acceptance_rate": (
            trading["acceptance_rate"]
            >= float(thresholds["minimum_acceptance_rate"])
        ),
        "bootstrap_lcb": (
            interval["lower"]
            > float(thresholds["mean_unit_net_pnl_bootstrap_95pct_lower_must_be_gt"])
        ),
        "largest_winner_removed": (
            largest_removed
            > float(
                thresholds[
                    "largest_winner_removed_total_unit_net_pnl_must_be_gt"
                ]
            )
        ),
        "first_chronological_half": (
            trading["first_chronological_half_total_unit_net_pnl"]
            >= float(
                thresholds[
                    "first_chronological_half_total_unit_net_pnl_must_be_gte"
                ]
            )
        ),
        "second_chronological_half": (
            trading["second_chronological_half_total_unit_net_pnl"]
            >= float(
                thresholds[
                    "second_chronological_half_total_unit_net_pnl_must_be_gte"
                ]
            )
        ),
    }
    return {
        "candidate_id": candidate_id,
        "candidate_ordinal": candidate_ordinal,
        "target_or_future_label_leakage_count": 0,
        "probability_metrics": probability,
        "trading_metrics": trading,
        "stratified_diagnostics": stratified,
        "development_selection_gate_results": gate_results,
        "development_selection_eligible": all(gate_results.values()),
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
    }


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_market[str(row["market_id"])].append(row)
    selected: list[Mapping[str, Any]] = []
    for market_rows in by_market.values():
        by_decision: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in market_rows:
            by_decision[int(row["decision_ts"])].append(row)
        for decision_ts in sorted(by_decision):
            candidate = max(
                by_decision[decision_ts],
                key=lambda row: (
                    float(row["selection_score"]),
                    -SIDES.index(str(row["side"])),
                ),
            )
            if float(candidate["selection_score"]) > 0.0:
                selected.append(candidate)
                break
    return selected


def _probability_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probabilities = np.clip(
        np.asarray(
            [float(row["win_probability"]) for row in rows],
            dtype=np.float64,
        ),
        1e-12,
        1.0 - 1e-12,
    )
    outcomes = np.asarray(
        [float(row["settlement_payout"]) for row in rows],
        dtype=np.float64,
    )
    intercept, slope = _calibration_fit(probabilities, outcomes)
    return {
        "side_row_count": len(rows),
        "brier_score": float(np.mean(np.square(probabilities - outcomes))),
        "log_loss": float(
            -np.mean(
                outcomes * np.log(probabilities)
                + (1.0 - outcomes) * np.log(1.0 - probabilities)
            )
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _calibration_fit(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    logits = np.log(probabilities / (1.0 - probabilities))
    design = np.column_stack((np.ones(len(logits)), logits))
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(100):
        fitted = 1.0 / (
            1.0 + np.exp(-np.clip(design @ coefficients, -30.0, 30.0))
        )
        weights = np.clip(fitted * (1.0 - fitted), 1e-9, None)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (outcomes - fitted)
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return float(coefficients[0]), float(coefficients[1])


def _bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    population = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = float(
            np.mean(
                generator.choice(
                    population,
                    size=len(population),
                    replace=True,
                )
            )
        )
    return {
        "method": "market_bootstrap_percentile_with_NO_TRADE_as_zero",
        "confidence": 0.95,
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def _stratified_selected_metrics(
    selected: Sequence[Mapping[str, Any]],
    ordered_markets: Sequence[tuple[int, str]],
    field: str,
) -> dict[str, Any]:
    del ordered_markets
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row[field])].append(row)
    return {
        group: {
            "accepted_market_count": len(group_rows),
            "total_unit_net_pnl": sum(
                float(row["target"]) for row in group_rows
            ),
            "mean_unit_net_pnl": float(
                np.mean([float(row["target"]) for row in group_rows])
            ),
        }
        for group, group_rows in sorted(groups.items())
    }


def _select_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = [
        candidate
        for candidate in candidate_results
        if bool(candidate["development_selection_eligible"])
    ]
    if not eligible:
        return {
            "status": "no_candidate_met_all_frozen_development_gates",
            "selected_candidate_id": None,
            "eligible_candidate_ids": [],
            "fresh_collection_allowed": False,
            "required_action": "stop_lineage_without_fresh_collection",
        }
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -float(
                candidate["trading_metrics"][
                    "mean_unit_net_pnl_bootstrap_interval"
                ]["lower"]
            ),
            float(candidate["probability_metrics"]["log_loss"]),
            int(candidate["candidate_ordinal"]),
        ),
    )
    selected = ranked[0]
    rule = dict(evaluation["development_candidate_selection_rule"])
    return {
        "status": "one_candidate_selected_for_future_model_freeze",
        "selected_candidate_id": selected["candidate_id"],
        "eligible_candidate_ids": [
            candidate["candidate_id"] for candidate in ranked
        ],
        "selection_order": rule["selection_order_for_eligible_candidates"],
        "fresh_collection_allowed": False,
        "required_action": (
            "train_and_hash_freeze_selected_candidate_on_all_development_markets_"
            "then_create_explicit_collection_authorization"
        ),
    }


def _annotate_regime(
    row: dict[str, Any],
    feature_contract: Mapping[str, Any],
) -> None:
    features = dict(row["features"])
    signed_return = float(features["signed_btc_return_15m"])
    unsigned_return = signed_return if row["side"] == "UP" else -signed_return
    regimes = dict(feature_contract["derived_regime_features"])
    return_contract = dict(regimes["btc_return_regime"])
    row["btc_return_regime"] = (
        "missing"
        if not math.isfinite(unsigned_return)
        else "bearish"
        if unsigned_return < float(return_contract["bearish_if_lt"])
        else "bullish"
        if unsigned_return > float(return_contract["bullish_if_gt"])
        else "sideways"
    )
    row["volatility_bucket"] = _bucket(
        float(features["btc_volatility_15m"]),
        regimes["volatility_bucket"],
    )
    row["spread_bucket"] = _bucket(
        float(features["combined_spread_bps"]),
        regimes["spread_bucket"],
    )
    row["volume_bucket"] = _bucket(
        _finite_sum(
            features["selected_recent_trade_volume"],
            features["opposite_recent_trade_volume"],
        ),
        regimes["volume_bucket"],
    )
    row["depth_bucket"] = _bucket(
        _finite_sum(
            features["selected_liquidity_depth"],
            features["opposite_liquidity_depth"],
        ),
        regimes["depth_bucket"],
    )
    row["expert_route"] = (
        "high_vol"
        if row["volatility_bucket"] == "high"
        else "bullish"
        if row["btc_return_regime"] == "bullish"
        else "bearish"
        if row["btc_return_regime"] == "bearish"
        else "low_vol"
    )


def _bucket(value: float, contract: Mapping[str, Any]) -> str:
    if not math.isfinite(value):
        return str(contract["missing_bucket"])
    if value <= float(contract["low_if_lte"]):
        return "low"
    if value <= float(contract["medium_if_lte"]):
        return "medium"
    return "high"


def _finite_sum(*values: Any) -> float:
    numeric = [float(value) for value in values]
    return sum(numeric) if all(math.isfinite(value) for value in numeric) else math.nan


def _prediction_record(
    candidate_id: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "market_id": row["market_id"],
        "market_start_ts": row["market_start_ts"],
        "decision_ts": row["decision_ts"],
        "side": row["side"],
        "raw_win_probability": row["raw_win_probability"],
        "win_probability": row["win_probability"],
        "prediction": row["prediction"],
        "selection_score": row["selection_score"],
        "execution_cost": row["execution_cost"],
        "target": row["target"],
        "resolved_outcome": row["resolved_outcome"],
        "btc_return_regime": row["btc_return_regime"],
        "volatility_bucket": row["volatility_bucket"],
        "spread_bucket": row["spread_bucket"],
        "volume_bucket": row["volume_bucket"],
        "depth_bucket": row["depth_bucket"],
        "oof_target_rank": row["oof_target_rank"],
        "strictly_prior_training_market_count": row[
            "strictly_prior_training_market_count"
        ],
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def _descriptor(path: Path, *, root: Path) -> dict[str, str]:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"artifact escaped descriptor root: {resolved}")
    return {
        "path": resolved.relative_to(root_resolved).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _atomic_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
