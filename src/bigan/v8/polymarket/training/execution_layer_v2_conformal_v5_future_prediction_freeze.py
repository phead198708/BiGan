"""Build target-free #204 features, score v5/v4, and freeze guard decisions."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.builder import (
    _normalize_book_snapshots,
    _normalize_candles,
    _normalize_chainlink_prices,
    _normalize_markets,
    _normalize_trades,
)
from bigan.v8.polymarket.corpus.features import build_polymarket_corpus_feature_rows
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    BINDING_MANIFEST_SCHEMA_VERSION,
    FORBIDDEN_TARGET_FIELDS,
    PREREG_MANIFEST_SCHEMA_VERSION,
    _blocked_safety_fields,
    _descriptor,
    _find_nonempty_fields,
    _is_git_sha,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _selected_window_blockers,
    _sha256_file,
    _verified_descriptor,
    _write_json,
    _write_text,
    validate_conformal_v5_future_evaluation_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
    apply_conformal_scores,
    validate_guard_compatible_conformal_net_return_v5_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    _attach_predictions_and_mask,
    _predict_regressor,
    _row_key,
    _row_sort_key,
    validate_guard_compatible_direct_net_return_v4_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _materialize_outcome_blind_action_rows,
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_aligned_action_value_support import (
    build_execution_compatible_action_universe,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    FORBIDDEN_RAW_FIELDS,
    load_and_validate_persistent_outcome_blind_index,
)

SCHEMA_PREFIX = "bigan-v8-conformal-v5-strict-future-prediction-freeze"
ALLOWED_RAW_FEATURE_FILES = (
    "raw_polymarket_markets.jsonl",
    "raw_polymarket_orderbooks.jsonl",
    "raw_polymarket_trades.jsonl",
    "raw_binance_btcusdt_klines.jsonl",
    "raw_polymarket_chainlink_prices.jsonl",
)
FORBIDDEN_PREDICTION_FIELDS = frozenset(set(FORBIDDEN_TARGET_FIELDS) | FORBIDDEN_RAW_FIELDS)


@dataclass(frozen=True, slots=True)
class ConformalV5FuturePredictionFreezeConfig:
    """Pinned target-free inputs for one and only one future decision freeze."""

    run_id: str
    output_dir: Path | str
    binding_manifest_path: Path | str
    expected_binding_manifest_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    builder_git_commit: str
    decision_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_binding_manifest_sha256",
            "expected_feature_contract_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        if not _is_git_sha(self.builder_git_commit):
            raise ValueError("builder_git_commit must be a Git SHA-1")
        if self.decision_freeze_created_ts <= 0:
            raise ValueError("decision_freeze_created_ts must be positive")
        for field in ("output_dir", "binding_manifest_path", "feature_contract_path"):
            object.__setattr__(self, field, Path(getattr(self, field)))


@dataclass(frozen=True, slots=True)
class _OutcomeBlindFeatureConfig:
    market_families: tuple[str, ...] = ("btc_updown_5m",)
    min_time_to_close_seconds: int = 0
    max_time_to_close_seconds: int | None = None

    def resolved_sample_intervals(self) -> dict[str, int]:
        return {"btc_updown_5m": 60}


def freeze_conformal_v5_future_predictions(
    config: ConformalV5FuturePredictionFreezeConfig,
) -> dict[str, Any]:
    """Score both frozen candidates and stop before any resolution or target access."""

    binding_path = config.binding_manifest_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_exact_pin(
        binding_path,
        config.expected_binding_manifest_sha256,
        "candidate/window binding manifest",
    )
    _verify_exact_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        "decision-time feature contract",
    )
    binding = _load_json(binding_path)
    if (
        binding.get("schema_version") != BINDING_MANIFEST_SCHEMA_VERSION
        or binding.get("candidate_window_binding_passed") is not True
        or binding.get("feature_materialization_attempted") is not False
        or binding.get("prediction_attempted") is not False
        or binding.get("future_labels_outcomes_or_pnl_opened") is not False
    ):
        raise ValueError("binding manifest is not eligible for target-free prediction")

    prereg_descriptor = _verified_descriptor(
        binding["preregistration_manifest"], "pre-registration manifest"
    )
    prereg = _load_json(Path(prereg_descriptor["path"]))
    if (
        prereg.get("schema_version") != PREREG_MANIFEST_SCHEMA_VERSION
        or prereg.get("pre_registration_ready") is not True
        or prereg.get("future_labels_outcomes_or_pnl_opened") is not False
    ):
        raise ValueError("pre-registration lineage is invalid")
    for field in (
        "candidate_manifest",
        "candidate_model",
        "candidate_calibration_artifact",
        "candidate_fit_profile",
        "matched_baseline_manifest",
        "matched_baseline_model",
        "matched_baseline_fit_profile",
    ):
        if binding.get(field) != prereg.get(field):
            raise ValueError(f"binding/preregistration lineage mismatch: {field}")
    profile_descriptor = _verified_descriptor(prereg["evaluation_profile"], "evaluation profile")
    profile = _load_json(Path(profile_descriptor["path"]))
    validate_conformal_v5_future_evaluation_profile(profile)

    feature_contract = _load_json(feature_contract_path)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=str(feature_contract["parent_protocol_sha256"]),
    )
    candidate_fit_descriptor = _verified_descriptor(
        binding["candidate_fit_profile"], "candidate fit profile"
    )
    baseline_fit_descriptor = _verified_descriptor(
        binding["matched_baseline_fit_profile"], "matched baseline fit profile"
    )
    candidate_fit = _load_json(Path(candidate_fit_descriptor["path"]))
    baseline_fit = _load_json(Path(baseline_fit_descriptor["path"]))
    validate_guard_compatible_conformal_net_return_v5_profile(candidate_fit)
    validate_guard_compatible_direct_net_return_v4_profile(baseline_fit)
    _validate_shared_inference_contracts(
        profile=profile,
        feature_contract_sha256=config.expected_feature_contract_sha256,
        candidate_fit=candidate_fit,
        baseline_fit=baseline_fit,
    )

    selected_descriptor = _verified_descriptor(binding["selected_rows"], "selected window rows")
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    if len(selected_rows) != int(binding["selected_market_count"]):
        raise ValueError("selected window row count mismatch")
    if len(selected_rows) != int(
        profile["issue_192_collection"]["target_quality_valid_market_count"]
    ):
        raise ValueError("selected future window does not contain the frozen 220 markets")
    if len({str(row.get("market_id") or "") for row in selected_rows}) != len(selected_rows):
        raise ValueError("selected future window market identities are not unique")
    window_descriptor = _verified_descriptor(binding["window_manifest"], "window manifest")
    window = _load_json(Path(window_descriptor["path"]))
    if window.get("selected_rows") != binding["selected_rows"]:
        raise ValueError("binding/window selected row descriptor mismatch")
    boundary_descriptor = _verified_descriptor(
        binding["source_boundary_manifest"], "source boundary manifest"
    )
    index_descriptor = _verified_descriptor(binding["collector_index"], "collector index")
    window_blockers = _selected_window_blockers(
        selected_rows=selected_rows,
        index_rows=load_and_validate_persistent_outcome_blind_index(Path(index_descriptor["path"])),
        boundary=_load_json(Path(boundary_descriptor["path"])),
        profile=profile,
    )
    if window_blockers:
        raise ValueError(
            "selected future window revalidation failed before prediction: "
            + ", ".join(window_blockers)
        )
    safety_mismatches = [
        field
        for field, expected in _blocked_safety_fields().items()
        if binding.get(field) != expected
    ]
    if safety_mismatches:
        raise ValueError("binding safety contract mismatch: " + ", ".join(safety_mismatches))

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    audit = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-target-access-audit-v1",
        "run_id": config.run_id,
        "binding_manifest": _descriptor(binding_path),
        "selected_window_rows": selected_descriptor,
        "candidate_manifest": binding["candidate_manifest"],
        "matched_baseline_manifest": binding["matched_baseline_manifest"],
        "feature_contract": _descriptor(feature_contract_path),
        "raw_feature_artifacts_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "prediction_attempted": False,
        **_blocked_safety_fields(),
    }
    audit["audit_id"] = canonical_json_sha256(audit)
    audit_path = run_dir / "pre_target_access_lineage_audit.json"
    _write_json(audit_path, audit)

    feature_rows, raw_descriptors = _materialize_selected_window_features(selected_rows)
    feature_rows_path = run_dir / "conformal_v5_future_target_free_feature_rows.jsonl"
    _write_jsonl(feature_rows_path, feature_rows)
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    action_rows = _materialize_future_action_rows(
        feature_rows,
        selected_rows=selected_rows,
        feature_columns=feature_columns,
    )
    action_rows_path = run_dir / "conformal_v5_future_target_free_five_action_rows.jsonl"
    _write_jsonl(action_rows_path, action_rows)

    candidate_model_descriptor = _verified_descriptor(binding["candidate_model"], "candidate model")
    calibration_descriptor = _verified_descriptor(
        binding["candidate_calibration_artifact"], "candidate calibration artifact"
    )
    candidate_predictions = _candidate_predictions(
        action_rows,
        model_descriptor=candidate_model_descriptor,
        calibration_artifact=_load_json(Path(calibration_descriptor["path"])),
        fit_profile=candidate_fit,
        feature_columns=feature_columns,
    )
    candidate_predictions_path = run_dir / "conformal_v5_future_target_free_predictions.jsonl"
    _write_jsonl(candidate_predictions_path, candidate_predictions)
    candidate_replay = _outcome_blind_acceptance_replay(
        candidate_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    candidate_replay_path = run_dir / "conformal_v5_future_outcome_blind_guard_replay.jsonl"
    _write_jsonl(candidate_replay_path, candidate_replay)

    baseline_model_descriptor = _verified_descriptor(
        binding["matched_baseline_model"], "matched baseline model"
    )
    baseline_predictions = _baseline_predictions(
        action_rows,
        model_descriptor=baseline_model_descriptor,
        fit_profile=baseline_fit,
        feature_columns=feature_columns,
    )
    baseline_predictions_path = run_dir / "matched_v4_future_target_free_predictions.jsonl"
    _write_jsonl(baseline_predictions_path, baseline_predictions)
    baseline_replay = _outcome_blind_acceptance_replay(
        baseline_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    baseline_replay_path = run_dir / "matched_v4_future_outcome_blind_guard_replay.jsonl"
    _write_jsonl(baseline_replay_path, baseline_replay)

    freeze = _decision_freeze(
        config=config,
        binding_path=binding_path,
        profile_descriptor=profile_descriptor,
        selected_descriptor=selected_descriptor,
        feature_rows_path=feature_rows_path,
        action_rows_path=action_rows_path,
        candidate_predictions_path=candidate_predictions_path,
        candidate_replay_path=candidate_replay_path,
        baseline_predictions_path=baseline_predictions_path,
        baseline_replay_path=baseline_replay_path,
        candidate_replay=candidate_replay,
        baseline_replay=baseline_replay,
    )
    freeze_path = run_dir / "conformal_v5_future_accepted_bet_decision_freeze.json"
    _write_json(freeze_path, freeze)
    freeze_sha256 = _sha256_file(freeze_path)
    report = _prediction_report(
        config=config,
        selected_rows=selected_rows,
        feature_rows=feature_rows,
        action_rows=action_rows,
        candidate_replay=candidate_replay,
        baseline_replay=baseline_replay,
        decision_freeze_sha256=freeze_sha256,
    )
    report_path = run_dir / "conformal_v5_future_prediction_freeze_report.json"
    _write_json(report_path, report)
    report_md_path = run_dir / "conformal_v5_future_prediction_freeze_report.md"
    _write_text(report_md_path, _report_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "binding_manifest": _descriptor(binding_path),
        "pre_target_access_audit": _descriptor(audit_path),
        "evaluation_profile": profile_descriptor,
        "feature_contract": _descriptor(feature_contract_path),
        "selected_window_rows": selected_descriptor,
        "opened_raw_feature_artifacts": raw_descriptors,
        "target_free_feature_rows": _descriptor(feature_rows_path),
        "target_free_five_action_rows": _descriptor(action_rows_path),
        "candidate_model": candidate_model_descriptor,
        "candidate_calibration_artifact": calibration_descriptor,
        "candidate_target_free_predictions": _descriptor(candidate_predictions_path),
        "candidate_outcome_blind_guard_replay": _descriptor(candidate_replay_path),
        "matched_baseline_model": baseline_model_descriptor,
        "matched_baseline_target_free_predictions": _descriptor(baseline_predictions_path),
        "matched_baseline_outcome_blind_guard_replay": _descriptor(baseline_replay_path),
        "accepted_bet_decision_freeze": _descriptor(freeze_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "decision_freeze_written_before_target_access": True,
        "candidate_and_baseline_same_window_feature_grid_guard_and_runtime": True,
        **_blocked_safety_fields(),
    }
    manifest["prediction_freeze_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v5_future_prediction_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "decision_freeze_path": freeze_path,
        "decision_freeze_sha256": freeze_sha256,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _materialize_selected_window_features(
    selected_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    opened: list[dict[str, Any]] = []
    config = _OutcomeBlindFeatureConfig()
    for selection_rank, selected in enumerate(
        sorted(
            selected_rows,
            key=lambda row: (int(row["scheduled_round_start_ts"]), str(row["market_id"])),
        ),
        start=1,
    ):
        raw = dict(selected.get("raw_artifacts") or {})
        payloads: dict[str, list[dict[str, Any]]] = {}
        descriptors: dict[str, dict[str, Any]] = {}
        for filename in ALLOWED_RAW_FEATURE_FILES:
            descriptor = _raw_descriptor(raw.get(filename), filename)
            rows = _load_jsonl(Path(descriptor["path"]))
            if len(rows) != int(descriptor["row_count"]):
                raise ValueError(f"raw feature row count mismatch: {filename}")
            forbidden = _find_nonempty_fields(rows, FORBIDDEN_PREDICTION_FIELDS)
            if forbidden:
                raise ValueError(
                    f"raw feature artifact contains forbidden targets: {filename}:"
                    + ",".join(forbidden)
                )
            payloads[filename] = rows
            descriptors[filename] = descriptor
        markets = _normalize_markets(
            payloads["raw_polymarket_markets.jsonl"],
            config,  # type: ignore[arg-type]
        )
        market_id = str(selected["market_id"])
        if len(markets) != 1 or markets[0].market_id != market_id:
            raise ValueError("selected raw market identity mismatch")
        feature_values = build_polymarket_corpus_feature_rows(
            markets=markets,
            book_snapshots=_normalize_book_snapshots(
                payloads["raw_polymarket_orderbooks.jsonl"], markets
            ),
            trades=_normalize_trades(payloads["raw_polymarket_trades.jsonl"], markets),
            btc_candles=_normalize_candles(payloads["raw_binance_btcusdt_klines.jsonl"]),
            chainlink_prices=_normalize_chainlink_prices(
                payloads["raw_polymarket_chainlink_prices.jsonl"]
            ),
            config=config,  # type: ignore[arg-type]
        )
        for value in feature_values:
            row = {
                **value.to_dict(),
                "future_window_selection_rank": selection_rank,
                "future_window_source_entry_sha256": selected["entry_sha256"],
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
            }
            if int(row["max_input_ts"]) > int(row["decision_ts"]):
                raise ValueError("future feature causality violation")
            if _find_nonempty_fields(row, FORBIDDEN_PREDICTION_FIELDS):
                raise ValueError("future feature row contains forbidden target fields")
            row["future_feature_row_sha256"] = canonical_json_sha256(row)
            output.append(row)
        opened.append(
            {
                "market_id": market_id,
                "selection_rank": selection_rank,
                "raw_feature_artifacts": descriptors,
                "resolution_artifact_opened": False,
            }
        )
    output.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    return output, opened


def _materialize_future_action_rows(
    feature_rows: list[dict[str, Any]],
    *,
    selected_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rank_by_market = {
        str(row["market_id"]): index
        for index, row in enumerate(
            sorted(
                selected_rows,
                key=lambda value: (
                    int(value["scheduled_round_start_ts"]),
                    str(value["market_id"]),
                ),
            ),
            start=1,
        )
    }
    by_market: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        by_market.setdefault(str(row["market_id"]), []).append(row)
    actions: list[dict[str, Any]] = []
    for market_id in sorted(by_market, key=lambda value: rank_by_market[value]):
        materialized = _materialize_outcome_blind_action_rows(
            by_market[market_id],
            role_row={"selection_rank": rank_by_market[market_id]},
            feature_columns=feature_columns,
        )
        for row in materialized:
            updated = {key: value for key, value in row.items() if key != "action_row_sha256"}
            updated["role"] = "future_unseen_evaluation"
            updated["action_row_sha256"] = canonical_json_sha256(updated)
            actions.append(updated)
    actions.sort(key=_row_sort_key)
    if len(actions) != len(feature_rows) * 5:
        raise ValueError("future five-action grid is incomplete")
    return actions


def _candidate_predictions(
    action_rows: list[dict[str, Any]],
    *,
    model_descriptor: dict[str, str],
    calibration_artifact: dict[str, Any],
    fit_profile: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    raw = _raw_target_stripped_predictions(
        booster,
        action_rows,
        feature_columns=feature_columns,
    )
    scored = apply_conformal_scores(
        raw,
        calibration_artifact=calibration_artifact,
        profile=fit_profile,
    )
    output = []
    for row in scored:
        prediction = float(row["raw_direct_predicted_net_return"])
        output.append(
            {
                **row,
                "raw_pairwise_rank_score": prediction,
                "pairwise_group_normalized_rank_score": prediction,
                "action_advantage_lcb_score_bucket": "not_applicable_split_conformal",
                "action_advantage_lcb_estimate_source": row["ranking_score_source"],
            }
        )
    return output


def _baseline_predictions(
    action_rows: list[dict[str, Any]],
    *,
    model_descriptor: dict[str, str],
    fit_profile: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    compatibility_rows = build_execution_compatible_action_universe(action_rows)
    compatibility = {
        _row_key(row): bool(row["p_up_alignment_passed"] and row["execution_quality_only_passed"])
        for row in compatibility_rows
    }
    predictions = _predict_regressor(booster, action_rows, feature_columns=feature_columns)
    return _attach_predictions_and_mask(
        action_rows,
        predictions,
        compatibility=compatibility,
        profile=fit_profile,
        fold_index=None,
    )


def _validate_shared_inference_contracts(
    *,
    profile: dict[str, Any],
    feature_contract_sha256: str,
    candidate_fit: dict[str, Any],
    baseline_fit: dict[str, Any],
) -> None:
    baseline = dict(profile["frozen_matched_market_baseline"])
    checks = {
        "candidate_feature_contract": candidate_fit.get("feature_contract_sha256")
        == feature_contract_sha256,
        "baseline_feature_contract": baseline_fit.get("feature_contract_sha256")
        == feature_contract_sha256,
        "shared_guard": candidate_fit.get("execution_guard_config_sha256")
        == baseline_fit.get("execution_guard_config_sha256"),
        "profile_shared_grid": baseline.get("same_feature_grid_as_candidate") is True,
        "profile_shared_execution": baseline.get(
            "same_cost_sizing_guard_exposure_and_position_management"
        )
        is True,
        "baseline_not_result_selected": baseline.get("future_outcomes_used_to_select_baseline")
        is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("candidate/baseline inference contract mismatch: " + ", ".join(blockers))


def _decision_freeze(
    *,
    config: ConformalV5FuturePredictionFreezeConfig,
    binding_path: Path,
    profile_descriptor: dict[str, str],
    selected_descriptor: dict[str, str],
    feature_rows_path: Path,
    action_rows_path: Path,
    candidate_predictions_path: Path,
    candidate_replay_path: Path,
    baseline_predictions_path: Path,
    baseline_replay_path: Path,
    candidate_replay: list[dict[str, Any]],
    baseline_replay: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_allowed = [row for row in candidate_replay if row["execution_guard_order_allowed"]]
    baseline_allowed = [row for row in baseline_replay if row["execution_guard_order_allowed"]]
    payload = {
        "schema_version": f"{SCHEMA_PREFIX}-accepted-bet-decision-freeze-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "binding_manifest": _descriptor(binding_path),
        "evaluation_profile": profile_descriptor,
        "selected_window_rows": selected_descriptor,
        "target_free_feature_rows": _descriptor(feature_rows_path),
        "target_free_five_action_rows": _descriptor(action_rows_path),
        "candidate_target_free_predictions": _descriptor(candidate_predictions_path),
        "candidate_outcome_blind_guard_replay": _descriptor(candidate_replay_path),
        "matched_baseline_target_free_predictions": _descriptor(baseline_predictions_path),
        "matched_baseline_outcome_blind_guard_replay": _descriptor(baseline_replay_path),
        "candidate_guard_accepted_bet_count": len(candidate_allowed),
        "candidate_guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in candidate_allowed}
        ),
        "matched_baseline_guard_accepted_bet_count": len(baseline_allowed),
        "matched_baseline_guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in baseline_allowed}
        ),
        "candidate_and_baseline_same_frozen_window": True,
        "candidate_and_baseline_same_cost_sizing_guard_exposure_and_position_management": True,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "target_or_outcome_used_for_decision": False,
        "decision_freeze_written_before_target_access": True,
        **_blocked_safety_fields(),
    }
    payload["decision_freeze_id"] = canonical_json_sha256(payload)
    return payload


def _prediction_report(
    *,
    config: ConformalV5FuturePredictionFreezeConfig,
    selected_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    candidate_replay: list[dict[str, Any]],
    baseline_replay: list[dict[str, Any]],
    decision_freeze_sha256: str,
) -> dict[str, Any]:
    candidate_allowed = [row for row in candidate_replay if row["execution_guard_order_allowed"]]
    baseline_allowed = [row for row in baseline_replay if row["execution_guard_order_allowed"]]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "selected_market_count": len(selected_rows),
        "target_free_feature_row_count": len(feature_rows),
        "complete_five_action_row_count": len(action_rows),
        "feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"]) for row in feature_rows
        ),
        "candidate_decision_count": len(candidate_replay),
        "candidate_guard_accepted_bet_count": len(candidate_allowed),
        "candidate_guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in candidate_allowed}
        ),
        "candidate_guard_accepted_side_distribution": _side_distribution(candidate_allowed),
        "candidate_blocking_reason_distribution": _blocking_distribution(candidate_replay),
        "matched_baseline_decision_count": len(baseline_replay),
        "matched_baseline_guard_accepted_bet_count": len(baseline_allowed),
        "matched_baseline_guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in baseline_allowed}
        ),
        "matched_baseline_guard_accepted_side_distribution": _side_distribution(baseline_allowed),
        "matched_baseline_blocking_reason_distribution": _blocking_distribution(baseline_replay),
        "decision_freeze_sha256": decision_freeze_sha256,
        "candidate_and_baseline_same_frozen_window_feature_grid_guard_and_runtime": True,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "target_or_outcome_used_for_decision": False,
        "prediction_and_decision_freeze_passed": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _raw_descriptor(value: Any, filename: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"raw feature descriptor missing: {filename}")
    path = Path(str(value.get("path") or "")).resolve()
    digest = str(value.get("sha256") or "").lower()
    row_count = int(value.get("row_count") or 0)
    _verify_exact_pin(path, digest, f"raw feature artifact {filename}")
    return {"path": str(path), "sha256": digest, "row_count": row_count}


def _verify_exact_pin(path: Path, expected: str, name: str) -> None:
    _require_sha256(expected, name=f"expected_{name.replace(' ', '_')}_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _side_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["selected_side"]) for row in rows).items()))


def _blocking_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(reason) for row in rows for reason in row["execution_blocking_reason_codes"]
            ).items()
        )
    )


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Conformal v5 strictly-later prediction freeze",
            "",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- target-free feature rows: `{report['target_free_feature_row_count']}`",
            f"- complete 5-action rows: `{report['complete_five_action_row_count']}`",
            f"- v5 accepted bets / markets: `{report['candidate_guard_accepted_bet_count']} / {report['candidate_guard_accepted_unique_market_count']}`",
            f"- v4 baseline accepted bets / markets: `{report['matched_baseline_guard_accepted_bet_count']} / {report['matched_baseline_guard_accepted_unique_market_count']}`",
            "- resolution/outcome/PnL opened: `false`",
            "- decision freeze written before target access: `true`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


__all__ = [
    "ConformalV5FuturePredictionFreezeConfig",
    "freeze_conformal_v5_future_predictions",
]
