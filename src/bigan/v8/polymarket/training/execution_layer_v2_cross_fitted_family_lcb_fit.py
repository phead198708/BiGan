"""Fixed #172 cross-fitted family model, calibration LCB, and confirmatory gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_cross_fitted_family_lcb import (
    validate_cross_fitted_family_lcb_feature_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_action_value import (
    _accepted_bet_metrics,
    _market_robustness,
    _predict_booster,
    _regression_metrics,
    _release_closed_positions,
    _train_family_booster,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_apply_simulated_order_to_state,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

SCHEMA_PREFIX = "bigan-v8-execution-layer-v2-cross-fitted-family-lcb"
TRADE_FAMILIES = ("HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE")
ROLE_NAMES = (
    "development_train",
    "development_calibration",
    "confirmatory_validation",
)
ROLE_MARKET_COUNTS = {
    "development_train": 40,
    "development_calibration": 20,
    "confirmatory_validation": 30,
}
FAMILY_MODEL_FILENAMES = {
    "HOLD_TO_SETTLEMENT": "cross_fitted_family_lcb_hts_model.xgb.json",
    "SELL_BEFORE_CLOSE": "cross_fitted_family_lcb_sbc_model.xgb.json",
}
FORBIDDEN_DECISION_FIELDS = {
    "future_return",
    "oracle_action",
    "realized_pnl",
    "resolved_outcome",
    "settlement_pnl",
    "settlement_return",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
    "total_net_return",
}


@dataclass(frozen=True, slots=True)
class CrossFittedFamilyLCBFitConfig:
    """Immutable inputs for fitting and one untouched confirmatory evaluation."""

    run_id: str
    output_dir: Path | str
    role_assignment_manifest_path: Path | str
    expected_role_assignment_manifest_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_role_assignment_manifest_sha256,
            name="role assignment manifest SHA-256",
        )
        _require_sha256(
            self.expected_feature_contract_sha256,
            name="feature contract SHA-256",
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "role_assignment_manifest_path",
            Path(self.role_assignment_manifest_path),
        )
        object.__setattr__(self, "feature_contract_path", Path(self.feature_contract_path))


def fit_cross_fitted_family_lcb(
    config: CrossFittedFamilyLCBFitConfig,
) -> dict[str, Any]:
    """Fit on 40 markets, calibrate LCB on 20, and evaluate once on 30."""

    role_manifest_path = config.role_assignment_manifest_path.resolve()
    _verify_pin(
        role_manifest_path,
        config.expected_role_assignment_manifest_sha256,
        name="role assignment manifest",
    )
    role_manifest = _load_json(role_manifest_path)
    if role_manifest.get("role_assignment_ready") is not True:
        raise ValueError("role assignment is not ready")
    if role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        raise ValueError("role assignment did not preserve outcome blindness")
    protocol_descriptor = _verified_descriptor(
        role_manifest.get("protocol"), name="cross-fitted family protocol"
    )
    protocol = _load_json(Path(protocol_descriptor["path"]))

    frozen_feature_descriptor = _verified_descriptor(
        role_manifest.get("feature_contract"), name="frozen feature contract"
    )

    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        name="feature contract",
    )
    if str(feature_contract_path) != frozen_feature_descriptor["path"]:
        raise ValueError("feature contract path does not match precollection freeze")
    if (
        config.expected_feature_contract_sha256.lower()
        != frozen_feature_descriptor["sha256"]
    ):
        raise ValueError("feature contract SHA-256 does not match precollection freeze")
    feature_contract = _load_json(feature_contract_path)
    validate_cross_fitted_family_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=protocol_descriptor["sha256"],
    )
    selected_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"), name="role assignment rows"
    )
    role_rows = _load_jsonl(Path(selected_descriptor["path"]))
    _validate_role_rows(role_rows)
    exclusion_descriptor = _verified_descriptor(
        role_manifest.get("prior_evidence_exclusion_registry"),
        name="prior evidence exclusion registry",
    )
    exclusion_registry = _load_json(Path(exclusion_descriptor["path"]))
    prior_market_ids = {
        str(value) for value in exclusion_registry.get("prior_market_ids") or []
    }
    selected_market_ids = {str(row["market_id"]) for row in role_rows}
    prior_market_overlap = selected_market_ids & prior_market_ids
    if prior_market_overlap:
        raise ValueError("role assignment overlaps prior evidence")

    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    feature_columns = tuple(feature_contract["feature_columns"])
    development_roles = ("development_train", "development_calibration")
    action_rows_by_role, corpus_audits = _materialize_role_action_rows(
        role_rows,
        feature_columns=feature_columns,
        roles=development_roles,
    )
    action_row_paths: dict[str, Path] = {}
    for role, rows in action_rows_by_role.items():
        path = run_dir / f"cross_fitted_family_lcb_{role}_action_rows.jsonl"
        _write_jsonl(path, rows)
        action_row_paths[role] = path

    cross_fit = _cross_fit_training_predictions(
        action_rows_by_role["development_train"],
        feature_columns=feature_columns,
        model_protocol=dict(protocol["cross_fit_protocol"]),
    )
    model_paths: dict[str, Path] = {}
    final_boosters: dict[str, xgb.Booster] = {}
    train_rows = action_rows_by_role["development_train"]
    for family in TRADE_FAMILIES:
        rows = [row for row in train_rows if row["action_family"] == family]
        booster = _train_family_booster(
            rows,
            feature_columns=feature_columns,
            model_protocol=_xgb_model_protocol(
                dict(protocol["cross_fit_protocol"])
            ),
        )
        model_path = run_dir / FAMILY_MODEL_FILENAMES[family]
        booster.save_model(model_path)
        model_paths[family] = model_path
        final_boosters[family] = booster

    calibration_predictions = _predict_role_rows(
        action_rows_by_role["development_calibration"],
        boosters=final_boosters,
        feature_columns=feature_columns,
    )
    lcb_artifact = _family_lcb_artifact(
        calibration_predictions,
        protocol=protocol,
        feature_contract_sha256=config.expected_feature_contract_sha256,
    )
    lcb_path = run_dir / "cross_fitted_family_lcb_calibration_artifact.json"
    _write_json(lcb_path, lcb_artifact)
    calibration_report = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "source_split": "development_calibration_only",
        "method": lcb_artifact["method"],
        "families": lcb_artifact["families"],
        "calibration_artifact": _descriptor(lcb_path),
        "confirmatory_labels_opened_before_calibration_freeze": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    calibration_report_path = run_dir / "conformal_lcb_calibration_report.json"
    _write_json(calibration_report_path, calibration_report)
    _write_text(
        run_dir / "conformal_lcb_calibration_report.md",
        _calibration_markdown(calibration_report),
    )

    development_fit_freeze = {
        "schema_version": f"{SCHEMA_PREFIX}-development-fit-freeze-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "protocol": protocol_descriptor,
        "feature_contract": _descriptor(feature_contract_path),
        "role_assignment_manifest": _descriptor(role_manifest_path),
        "development_action_rows": {
            role: _descriptor(action_row_paths[role]) for role in development_roles
        },
        "models": {family: _descriptor(path) for family, path in model_paths.items()},
        "family_lcb_calibration_artifact": _descriptor(lcb_path),
        "conformal_lcb_calibration_report": _descriptor(calibration_report_path),
        "confirmatory_labels_opened_before_this_freeze": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    development_fit_freeze["development_fit_freeze_id"] = canonical_json_sha256(
        development_fit_freeze
    )
    development_fit_freeze_path = (
        run_dir / "cross_fitted_family_lcb_development_fit_freeze_manifest.json"
    )
    _write_json(development_fit_freeze_path, development_fit_freeze)

    confirmatory_rows, confirmatory_audits = _materialize_role_action_rows(
        role_rows,
        feature_columns=feature_columns,
        roles=("confirmatory_validation",),
    )
    action_rows_by_role.update(confirmatory_rows)
    corpus_audits.extend(confirmatory_audits)
    confirmatory_action_path = (
        run_dir / "cross_fitted_family_lcb_confirmatory_validation_action_rows.jsonl"
    )
    _write_jsonl(
        confirmatory_action_path,
        action_rows_by_role["confirmatory_validation"],
    )
    action_row_paths["confirmatory_validation"] = confirmatory_action_path
    split_manifest = _split_manifest(
        run_id=config.run_id,
        role_manifest_path=role_manifest_path,
        protocol_descriptor=protocol_descriptor,
        feature_contract_path=feature_contract_path,
        action_rows_by_role=action_rows_by_role,
        action_row_paths=action_row_paths,
        corpus_audits=corpus_audits,
    )
    split_manifest_path = run_dir / "cross_fitted_family_lcb_split_manifest.json"
    _write_json(split_manifest_path, split_manifest)
    forbidden_inference_field_violation_count = sum(
        int(
            row["target_used_as_decision_input"] is not False
            or row["outcome_fields_used_as_decision_input"] is not False
        )
        for rows in action_rows_by_role.values()
        for row in rows
    )
    leakage_and_role_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-leakage-role-audit-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "role_market_counts": {
            role: split_manifest["roles"][role]["market_count"] for role in ROLE_NAMES
        },
        "role_market_overlap_count": split_manifest["role_market_overlap_count"],
        "prior_market_overlap_count": len(prior_market_overlap),
        "chronology_validation_passed": split_manifest[
            "chronology_validation_passed"
        ],
        "feature_causality_violation_count": split_manifest[
            "feature_causality_violation_count"
        ],
        "forbidden_inference_field_violation_count": (
            forbidden_inference_field_violation_count
        ),
        "role_assignment_completed_before_label_access": True,
        "development_fit_frozen_before_confirmatory_label_access": True,
        "confirmatory_labels_used_for_tuning": False,
        "prior_or_future_evidence_used_for_tuning": False,
        "leakage_and_role_audit_passed": (
            split_manifest["role_market_overlap_count"] == 0
            and not prior_market_overlap
            and split_manifest["chronology_validation_passed"] is True
            and split_manifest["feature_causality_violation_count"] == 0
            and forbidden_inference_field_violation_count == 0
        ),
        **_blocked_safety_fields(),
    }
    leakage_audit_path = run_dir / "leakage_and_role_audit.json"
    _write_json(leakage_audit_path, leakage_and_role_audit)
    _write_text(
        run_dir / "leakage_and_role_audit.md",
        _leakage_markdown(leakage_and_role_audit),
    )

    confirmatory_predictions = _predict_role_rows(
        action_rows_by_role["confirmatory_validation"],
        boosters=final_boosters,
        feature_columns=feature_columns,
    )
    confirmatory_predictions = _apply_lcb_scores(
        confirmatory_predictions,
        lcb_artifact=lcb_artifact,
    )
    prediction_path = run_dir / "cross_fitted_family_lcb_confirmatory_predictions.jsonl"
    _write_jsonl(prediction_path, confirmatory_predictions)
    entry_threshold = float(
        protocol["frozen_execution_contract"]["entry_edge_threshold"]
    )
    candidate_replay = _run_policy_replay(
        confirmatory_predictions,
        score_field="family_lcb_expected_net_return",
        policy_name="market_grouped_cross_fitted_family_lcb_v1",
        entry_threshold=entry_threshold,
    )
    baseline_replay = _run_policy_replay(
        confirmatory_predictions,
        score_field="raw_family_expected_net_return",
        policy_name="uncertainty_unadjusted_family_model_same_threshold_and_guard",
        entry_threshold=entry_threshold,
    )
    candidate_path = run_dir / "cross_fitted_family_lcb_confirmatory_replay.jsonl"
    baseline_path = run_dir / "cross_fitted_family_lcb_confirmatory_baseline_replay.jsonl"
    _write_jsonl(candidate_path, candidate_replay)
    _write_jsonl(baseline_path, baseline_replay)
    candidate_metrics = _accepted_bet_metrics(candidate_replay)
    baseline_metrics = _accepted_bet_metrics(baseline_replay)
    robustness = _market_robustness(candidate_replay, baseline_replay)
    confirmatory_gate = _confirmatory_gate(
        protocol=protocol,
        action_rows=action_rows_by_role["confirmatory_validation"],
        candidate_replay=candidate_replay,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        robustness=robustness,
    )

    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "feature_columns": list(feature_columns),
        "cross_fit": cross_fit,
        "models": {family: _descriptor(path) for family, path in model_paths.items()},
        "training_market_count": ROLE_MARKET_COUNTS["development_train"],
        "calibration_market_count": ROLE_MARKET_COUNTS["development_calibration"],
        "development_fit_freeze_manifest": _descriptor(development_fit_freeze_path),
        "confirmatory_labels_opened_before_model_and_lcb_freeze": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    training_report_path = run_dir / "cross_fit_training_report.json"
    _write_json(training_report_path, training_report)
    _write_text(
        run_dir / "cross_fit_training_report.md",
        _training_markdown(training_report),
    )
    validation_report = {
        "schema_version": f"{SCHEMA_PREFIX}-confirmatory-validation-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "candidate_minus_baseline_net_pnl": candidate_metrics["net_pnl_sum"]
        - baseline_metrics["net_pnl_sum"],
        "market_robustness_diagnostics": robustness,
        "confirmatory_gate_checks": confirmatory_gate["checks"],
        "confirmatory_gate_passed": confirmatory_gate["passed"],
        "confirmatory_gate_blocking_reason_codes": confirmatory_gate["reason_codes"],
        "confirmatory_labels_used_for_report_only": True,
        "confirmatory_labels_used_for_tuning": False,
        "candidate_frozen_for_future_evaluation": confirmatory_gate["passed"],
        "future_collection_allowed": confirmatory_gate["passed"],
        **_blocked_safety_fields(),
    }
    validation_report_path = (
        run_dir / "confirmatory_validation_report.json"
    )
    _write_json(validation_report_path, validation_report)
    _write_text(
        run_dir / "confirmatory_validation_report.md",
        _validation_markdown(validation_report),
    )

    freeze_manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-candidate-freeze-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "protocol": protocol_descriptor,
        "feature_contract": _descriptor(feature_contract_path),
        "role_assignment_manifest": _descriptor(role_manifest_path),
        "split_manifest": _descriptor(split_manifest_path),
        "models": {family: _descriptor(path) for family, path in model_paths.items()},
        "family_lcb_calibration_artifact": _descriptor(lcb_path),
        "conformal_lcb_calibration_report": _descriptor(calibration_report_path),
        "development_fit_freeze_manifest": _descriptor(development_fit_freeze_path),
        "training_report": _descriptor(training_report_path),
        "leakage_and_role_audit": _descriptor(leakage_audit_path),
        "confirmatory_validation_report": _descriptor(validation_report_path),
        "candidate_frozen_for_future_evaluation": confirmatory_gate["passed"],
        "future_collection_allowed": confirmatory_gate["passed"],
        "future_unseen_evaluation_required": True,
        "confirmatory_gate_passed": confirmatory_gate["passed"],
        "confirmatory_gate_blocking_reason_codes": confirmatory_gate["reason_codes"],
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    freeze_manifest["research_candidate_hash"] = canonical_json_sha256(
        {
            "protocol": protocol_descriptor["sha256"],
            "feature_contract": freeze_manifest["feature_contract"]["sha256"],
            "role_assignment": freeze_manifest["role_assignment_manifest"]["sha256"],
            "split": freeze_manifest["split_manifest"]["sha256"],
            "models": {
                family: descriptor["sha256"]
                for family, descriptor in freeze_manifest["models"].items()
            },
            "lcb": freeze_manifest["family_lcb_calibration_artifact"]["sha256"],
        }
    )
    freeze_manifest_path = run_dir / "candidate_freeze_manifest.json"
    _write_json(freeze_manifest_path, freeze_manifest)
    return {
        "run_dir": run_dir,
        "split_manifest_path": split_manifest_path,
        "development_fit_freeze_manifest_path": development_fit_freeze_path,
        "training_report_path": training_report_path,
        "calibration_report_path": calibration_report_path,
        "leakage_audit_path": leakage_audit_path,
        "validation_report_path": validation_report_path,
        "freeze_manifest_path": freeze_manifest_path,
        "freeze_manifest_sha256": _sha256_file(freeze_manifest_path),
        "validation_report": validation_report,
        "freeze_manifest": freeze_manifest,
    }


def _validate_role_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 90:
        raise ValueError("role assignment must contain exactly 90 markets")
    market_ids = [str(row.get("market_id") or "") for row in rows]
    if any(not value for value in market_ids) or len(market_ids) != len(set(market_ids)):
        raise ValueError("role assignment market identities are incomplete")
    if [int(row.get("selection_rank") or 0) for row in rows] != list(range(1, 91)):
        raise ValueError("role assignment selection ranks are incomplete")
    counts = Counter(str(row.get("role") or "") for row in rows)
    if dict(counts) != ROLE_MARKET_COUNTS:
        raise ValueError("role assignment market counts do not match 40/20/30")
    if any(row.get("labels_or_outcomes_opened_for_role_assignment") is not False for row in rows):
        raise ValueError("role assignment opened labels or outcomes")


def _materialize_role_action_rows(
    role_rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    roles: tuple[str, ...],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not roles or any(role not in ROLE_NAMES for role in roles):
        raise ValueError("requested materialization roles are invalid")
    output: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    audits: list[dict[str, Any]] = []
    for role_row in role_rows:
        role = str(role_row["role"])
        if role not in output:
            continue
        corpus_dir = Path(str(role_row["source_corpus_dir"])).resolve()
        rows, audit = _load_corpus_action_rows(
            corpus_dir,
            role_row=role_row,
            feature_columns=feature_columns,
        )
        if audit["blocking_reason_codes"]:
            raise ValueError(
                f"corpus action-row materialization failed for {corpus_dir}: "
                + ", ".join(audit["blocking_reason_codes"])
            )
        output[role].extend(rows)
        audits.append(audit)
    for role in roles:
        output[role].sort(
            key=lambda row: (
                int(row["decision_ts"]),
                str(row["market_id"]),
                str(row["action"]),
            )
        )
        markets = {str(row["market_id"]) for row in output[role]}
        if len(markets) != ROLE_MARKET_COUNTS[role]:
            raise ValueError(f"{role} action rows have incomplete market coverage")
    return output, audits


def _load_corpus_action_rows(
    corpus_dir: Path,
    *,
    role_row: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filenames = {
        "manifest": "polymarket_corpus_manifest.json",
        "features": "polymarket_feature_rows.jsonl",
        "labels": "polymarket_label_rows.jsonl",
        "metadata": "polymarket_market_metadata.jsonl",
        "resolutions": "polymarket_resolution_events.jsonl",
    }
    missing = sorted(
        filename for filename in filenames.values() if not (corpus_dir / filename).is_file()
    )
    if missing:
        raise ValueError(f"required corpus artifacts are missing: {missing}")
    manifest_path = corpus_dir / filenames["manifest"]
    manifest = _load_json(manifest_path)
    if _sha256_file(manifest_path) != role_row["corpus_manifest"]["sha256"]:
        raise ValueError("role-assigned corpus manifest SHA-256 mismatch")
    normalized_hashes = dict(manifest.get("normalized_artifact_hashes") or {})
    hash_keys = {
        "features": "feature_rows",
        "labels": "label_rows",
        "metadata": "market_metadata",
        "resolutions": "resolution_events",
    }
    for name, hash_key in hash_keys.items():
        path = corpus_dir / filenames[name]
        if normalized_hashes.get(hash_key) != _sha256_file(path):
            raise ValueError(f"normalized artifact SHA-256 mismatch: {path.name}")

    features = _load_jsonl(corpus_dir / filenames["features"])
    labels = _load_jsonl(corpus_dir / filenames["labels"])
    metadata_rows = _load_jsonl(corpus_dir / filenames["metadata"])
    resolution_rows = _load_jsonl(corpus_dir / filenames["resolutions"])
    market_id = str(role_row["market_id"])
    blockers: list[str] = []
    if {str(row.get("market_id") or "") for row in features} != {market_id}:
        blockers.append("feature_market_identity_mismatch")
    if {str(row.get("market_id") or "") for row in labels} != {market_id}:
        blockers.append("label_market_identity_mismatch")
    metadata_by_market = {str(row.get("market_id") or ""): row for row in metadata_rows}
    resolution_by_market = {
        str(row.get("market_id") or ""): row for row in resolution_rows
    }
    metadata = metadata_by_market.get(market_id)
    resolution = resolution_by_market.get(market_id)
    if metadata is None:
        blockers.append("market_metadata_missing")
    if resolution is None or resolution.get("resolved_outcome") not in {"UP", "DOWN"}:
        blockers.append("official_resolution_missing")
    labels_by_decision: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        labels_by_decision[int(label.get("decision_ts") or 0)].append(label)
    action_rows: list[dict[str, Any]] = []
    causality_violations = 0
    cost_component_violations = 0
    incomplete_grids = 0
    for feature_row in features:
        decision_ts = int(feature_row.get("decision_ts") or 0)
        max_input_ts = int(feature_row.get("max_input_ts") or 0)
        if max_input_ts > decision_ts or decision_ts <= 0:
            causality_violations += 1
            continue
        if _find_fields(feature_row, FORBIDDEN_DECISION_FIELDS):
            causality_violations += 1
            continue
        decision_labels = labels_by_decision.get(decision_ts, [])
        label_actions = {str(row.get("action") or "") for row in decision_labels}
        if label_actions != set(REQUIRED_ACTIONS):
            incomplete_grids += 1
            continue
        label_by_action = {str(row["action"]): row for row in decision_labels}
        if any(
            not _cost_aware_label_valid(label_by_action[action])
            for action in REQUIRED_ACTIONS
        ):
            cost_component_violations += 1
            continue
        assert metadata is not None
        for action in REQUIRED_ACTIONS:
            action_rows.append(
                _action_row(
                    feature_row=feature_row,
                    label=label_by_action[action],
                    metadata=metadata,
                    resolution=resolution or {},
                    role=str(role_row["role"]),
                    selection_rank=int(role_row["selection_rank"]),
                    source_corpus_dir=corpus_dir,
                    source_manifest_sha256=_sha256_file(manifest_path),
                    feature_columns=feature_columns,
                )
            )
    if causality_violations:
        blockers.append("feature_timestamp_or_field_causality_violation")
    if incomplete_grids:
        blockers.append("incomplete_5_action_label_grid")
    if cost_component_violations:
        blockers.append("cost_aware_label_contract_violation")
    expected_rows = len(features) * len(REQUIRED_ACTIONS)
    if len(action_rows) != expected_rows:
        blockers.append("materialized_action_row_count_mismatch")
    return action_rows, {
        "market_id": market_id,
        "role": role_row["role"],
        "source_corpus_dir": str(corpus_dir),
        "source_corpus_manifest_sha256": _sha256_file(manifest_path),
        "feature_row_count": len(features),
        "label_row_count": len(labels),
        "materialized_action_row_count": len(action_rows),
        "feature_causality_violation_count": causality_violations,
        "incomplete_action_grid_count": incomplete_grids,
        "cost_component_violation_count": cost_component_violations,
        "blocking_reason_codes": sorted(set(blockers)),
        "role_assignment_completed_before_label_access": True,
        "outcomes_used_as_training_or_evaluation_targets_only": True,
        "outcomes_used_as_decision_inputs": False,
    }


def _cost_aware_label_valid(row: dict[str, Any]) -> bool:
    if row.get("paper_only") is not True or row.get("capital_at_risk") is not False:
        return False
    for field in ("fees", "slippage", "liquidity_impact", "total_net_pnl_per_notional"):
        value = row.get(field)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return False
    return all(float(row[field]) >= 0.0 for field in ("fees", "slippage", "liquidity_impact"))


def _action_row(
    *,
    feature_row: dict[str, Any],
    label: dict[str, Any],
    metadata: dict[str, Any],
    resolution: dict[str, Any],
    role: str,
    selection_rank: int,
    source_corpus_dir: Path,
    source_manifest_sha256: str,
    feature_columns: tuple[str, ...],
) -> dict[str, Any]:
    action = str(label["action"])
    side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
    family = (
        "HOLD_TO_SETTLEMENT"
        if "HOLD_TO_SETTLEMENT" in action
        else "SELL_BEFORE_CLOSE"
        if "SELL_BEFORE_CLOSE" in action
        else "NO_TRADE"
    )
    raw = dict(feature_row.get("features") or {})
    decision_features = _decision_features(raw, action=action, side=side, family=family)
    missing = sorted(name for name in feature_columns if name not in decision_features)
    if missing:
        raise ValueError(f"decision-time features are missing: {missing}")
    values = {name: float(decision_features[name]) for name in feature_columns}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("decision-time features must be finite")
    decision_ts = int(feature_row["decision_ts"])
    max_input_ts = int(feature_row["max_input_ts"])
    feature_provenance = dict(feature_row.get("feature_provenance") or {})
    reference_provenance = dict(
        feature_provenance.get("reference_price_to_beat_distance_at_decision") or {}
    )
    reference_max_input_ts = int(reference_provenance.get("max_input_ts") or 0)
    reference_valid = bool(
        reference_provenance.get("provenance_valid") is True
        and reference_max_input_ts <= decision_ts
    )
    if action != "NO_TRADE" and not reference_valid:
        raise ValueError("reference price feature provenance is invalid")
    p_up = _p_up(raw)
    p_down = 1.0 - p_up
    selected_probability = p_up if side == "UP" else p_down if side == "DOWN" else 0.0
    side_prefix = side.lower()
    microstructure = {
        "entry_bid": float(raw.get(f"{side_prefix}_bid") or 0.0),
        "entry_ask": float(raw.get(f"{side_prefix}_ask") or 0.0),
        "spread_bps": float(raw.get(f"{side_prefix}_spread_bps") or 0.0),
        "book_staleness_ms": float(raw.get(f"{side_prefix}_book_staleness_ms") or 0.0),
        "queue_fill_proxy": float(
            raw.get(f"{side_prefix}_queue_fill_probability_proxy") or 0.0
        ),
        "time_to_close_seconds": float(raw.get("time_to_close_seconds") or 0.0),
    }
    row = {
        "market_id": str(feature_row["market_id"]),
        "condition_id": str(metadata.get("condition_id") or feature_row["market_id"]),
        "market_slug": str(metadata.get("slug") or ""),
        "decision_ts": decision_ts,
        "market_close_ts": int(metadata.get("market_end_ts") or 0),
        "max_input_ts": max_input_ts,
        "role": role,
        "market_selection_rank": selection_rank,
        "action": action,
        "side": side,
        "action_family": family,
        "decision_time_features": values,
        "p_up": p_up,
        "p_down": p_down,
        "selected_side_probability": selected_probability,
        "microstructure_snapshot": microstructure,
        "reference_price_feature_provenance": {
            **reference_provenance,
            "provenance_valid": reference_valid,
        },
        "p_up_action_disagreement": bool(
            (side == "UP" and p_up < 0.5) or (side == "DOWN" and p_up > 0.5)
        ),
        "target_net_pnl_per_contract": float(label["total_net_pnl_per_notional"]),
        "target_cost_components": {
            "fees": float(label["fees"]),
            "slippage": float(label["slippage"]),
            "liquidity_impact": float(label["liquidity_impact"]),
        },
        "target_resolved_outcome": resolution.get("resolved_outcome"),
        "target_used_as_decision_input": False,
        "outcome_fields_used_as_decision_input": False,
        "source_corpus_dir": str(source_corpus_dir),
        "source_corpus_manifest_sha256": source_manifest_sha256,
        "paper_only": True,
        "capital_at_risk": False,
    }
    row["action_row_sha256"] = canonical_json_sha256(row)
    return row


def _decision_features(
    raw: dict[str, Any],
    *,
    action: str,
    side: str,
    family: str,
) -> dict[str, float]:
    p_up = _p_up(raw)
    p_down = 1.0 - p_up
    if side == "NONE":
        selected = "up"
        opposite = "down"
        selected_probability = 0.0
        execution_price = 0.0
    else:
        selected = side.lower()
        opposite = "down" if selected == "up" else "up"
        selected_probability = p_up if side == "UP" else p_down
        execution_price = float(raw.get(f"{selected}_ask") or 0.0)
    return {
        "btc_return_10s": float(raw.get("btc_return_10s") or 0.0),
        "btc_return_30s": float(raw.get("btc_return_30s") or 0.0),
        "btc_return_1m": float(raw.get("btc_return_1m") or 0.0),
        "btc_return_5m": float(raw.get("btc_return_5m") or 0.0),
        "btc_return_15m": float(raw.get("btc_return_15m") or 0.0),
        "btc_volatility_1m": float(raw.get("btc_volatility_1m") or 0.0),
        "btc_volatility_5m": float(raw.get("btc_volatility_5m") or 0.0),
        "btc_volatility_15m": float(raw.get("btc_volatility_15m") or 0.0),
        "reference_price_to_beat_distance_at_decision": float(
            raw.get("reference_price_to_beat_distance_at_decision") or 0.0
        ),
        "time_to_close_seconds": float(raw.get("time_to_close_seconds") or 0.0),
        "market_age_seconds": float(raw.get("market_age_seconds") or 0.0),
        "combined_spread_bps": float(raw.get("combined_spread_bps") or 0.0),
        "liquidity_imbalance": float(raw.get("liquidity_imbalance") or 0.0),
        "recent_selected_side_trade_volume": float(
            raw.get(f"recent_{selected}_trade_volume") or 0.0
        ),
        "recent_opposite_side_trade_volume": float(
            raw.get(f"recent_{opposite}_trade_volume") or 0.0
        ),
        "selected_side_probability": selected_probability,
        "execution_price": execution_price,
        "selected_side_probability_minus_execution_price": (
            selected_probability - execution_price
        ),
        "selected_side_spread_bps": float(
            raw.get(f"{selected}_spread_bps") or 0.0
        ),
        "selected_side_queue_fill_probability_proxy": float(
            raw.get(f"{selected}_queue_fill_probability_proxy") or 0.0
        ),
        "selected_side_book_staleness_ms": float(
            raw.get(f"{selected}_book_staleness_ms") or 0.0
        ),
        "selected_side_liquidity_depth": float(
            raw.get(f"{selected}_liquidity_depth") or 0.0
        ),
        "selected_side_executable_ask_notional": float(
            raw.get(f"{selected}_executable_ask_notional") or 0.0
        ),
        "selected_side_executable_bid_notional": float(
            raw.get(f"{selected}_executable_bid_notional") or 0.0
        ),
        "selected_side_recent_book_update_count_1m": float(
            raw.get(f"{selected}_recent_book_update_count_1m") or 0.0
        ),
        "selected_side_recent_spread_stability_1m": float(
            raw.get(f"{selected}_recent_spread_stability_1m") or 0.0
        ),
        "selected_side_recent_bid_depth_volatility_1m": float(
            raw.get(f"{selected}_recent_bid_depth_volatility_1m") or 0.0
        ),
        "action_buy_up": float("BUY_UP" in action),
        "action_buy_down": float("BUY_DOWN" in action),
        "action_hold_to_settlement": float(family == "HOLD_TO_SETTLEMENT"),
        "action_sell_before_close": float(family == "SELL_BEFORE_CLOSE"),
    }


def _p_up(raw: dict[str, Any]) -> float:
    up_mid = float(
        raw.get("up_mid")
        or (float(raw.get("up_bid") or 0.0) + float(raw.get("up_ask") or 0.0))
        / 2.0
    )
    down_mid = float(
        raw.get("down_mid")
        or (
            float(raw.get("down_bid") or 0.0)
            + float(raw.get("down_ask") or 0.0)
        )
        / 2.0
    )
    denominator = up_mid + down_mid
    if denominator <= 0.0:
        raise ValueError("market-implied probability inputs are invalid")
    value = up_mid / denominator
    if not 0.0 <= value <= 1.0:
        raise ValueError("market-implied probability is outside [0, 1]")
    return value


def _split_manifest(
    *,
    run_id: str,
    role_manifest_path: Path,
    protocol_descriptor: dict[str, str],
    feature_contract_path: Path,
    action_rows_by_role: dict[str, list[dict[str, Any]]],
    action_row_paths: dict[str, Path],
    corpus_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    market_sets = {
        role: {str(row["market_id"]) for row in rows}
        for role, rows in action_rows_by_role.items()
    }
    overlap = (
        market_sets[ROLE_NAMES[0]] & market_sets[ROLE_NAMES[1]]
    ) | (
        market_sets[ROLE_NAMES[0]] & market_sets[ROLE_NAMES[2]]
    ) | (
        market_sets[ROLE_NAMES[1]] & market_sets[ROLE_NAMES[2]]
    )
    if overlap:
        raise ValueError("role action-row market overlap detected")
    previous_max: int | None = None
    role_summaries: dict[str, Any] = {}
    for role in ROLE_NAMES:
        rows = action_rows_by_role[role]
        minimum = min(int(row["decision_ts"]) for row in rows)
        maximum = max(int(row["decision_ts"]) for row in rows)
        if previous_max is not None and minimum <= previous_max:
            raise ValueError("role chronology overlaps")
        previous_max = maximum
        role_summaries[role] = {
            "market_count": len(market_sets[role]),
            "market_ids": sorted(market_sets[role]),
            "market_ids_sha256": canonical_json_sha256(sorted(market_sets[role])),
            "decision_count": len(rows) // len(REQUIRED_ACTIONS),
            "action_row_count": len(rows),
            "minimum_decision_ts": minimum,
            "maximum_decision_ts": maximum,
            "support_by_family_side": dict(
                sorted(
                    Counter(
                        f"{row['action_family']}|{row['side']}"
                        for row in rows
                        if row["action_family"] in TRADE_FAMILIES
                    ).items()
                )
            ),
        }
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-split-manifest-v1",
        "run_id": run_id,
        "role_assignment_manifest": _descriptor(role_manifest_path),
        "protocol": protocol_descriptor,
        "feature_contract": _descriptor(feature_contract_path),
        "roles": role_summaries,
        "action_rows": {
            role: _descriptor(path) for role, path in action_row_paths.items()
        },
        "corpus_audits": corpus_audits,
        "role_market_overlap_count": 0,
        "chronology_validation_passed": True,
        "feature_causality_violation_count": sum(
            int(audit["feature_causality_violation_count"])
            for audit in corpus_audits
        ),
        "role_assignment_completed_before_label_access": True,
        "confirmatory_labels_used_for_tuning": False,
        **_blocked_safety_fields(),
    }
    manifest["split_hash"] = canonical_json_sha256(
        {
            "role_assignment": manifest["role_assignment_manifest"]["sha256"],
            "protocol": protocol_descriptor["sha256"],
            "feature_contract": manifest["feature_contract"]["sha256"],
            "roles": role_summaries,
            "action_rows": {
                role: descriptor["sha256"]
                for role, descriptor in manifest["action_rows"].items()
            },
        }
    )
    return manifest


def _cross_fit_training_predictions(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    model_protocol: dict[str, Any],
) -> dict[str, Any]:
    market_first_ts: dict[str, int] = {}
    for row in rows:
        market_id = str(row["market_id"])
        market_first_ts[market_id] = min(
            market_first_ts.get(market_id, int(row["decision_ts"])),
            int(row["decision_ts"]),
        )
    ordered_markets = sorted(
        market_first_ts, key=lambda value: (market_first_ts[value], value)
    )
    fold_count = int(model_protocol["fold_count"])
    if len(ordered_markets) != 40 or fold_count != 5:
        raise ValueError("cross-fit requires exactly 40 markets and five folds")
    fold_market_groups = [
        ordered_markets[index * 8 : (index + 1) * 8] for index in range(fold_count)
    ]
    oof_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    xgb_protocol = _xgb_model_protocol(model_protocol)
    for fold_index, validation_markets in enumerate(fold_market_groups, start=1):
        validation_set = set(validation_markets)
        training_markets = [
            market for market in ordered_markets if market not in validation_set
        ]
        family_metrics: dict[str, Any] = {}
        for family in TRADE_FAMILIES:
            fit_rows = [
                row
                for row in rows
                if row["action_family"] == family
                and str(row["market_id"]) in training_markets
            ]
            validation_rows = [
                row
                for row in rows
                if row["action_family"] == family
                and str(row["market_id"]) in validation_set
            ]
            booster = _train_family_booster(
                fit_rows,
                feature_columns=feature_columns,
                model_protocol=xgb_protocol,
            )
            predictions = _predict_booster(
                booster,
                validation_rows,
                feature_columns=feature_columns,
            )
            targets = [
                float(row["target_net_pnl_per_contract"])
                for row in validation_rows
            ]
            family_metrics[family] = _regression_metrics(targets, predictions)
            for row, prediction in zip(validation_rows, predictions, strict=True):
                oof_rows.append(
                    {
                        "fold_index": fold_index,
                        "market_id": row["market_id"],
                        "decision_ts": row["decision_ts"],
                        "action": row["action"],
                        "action_family": family,
                        "action_row_sha256": row["action_row_sha256"],
                        "oof_raw_prediction": prediction,
                        "target_net_pnl_per_contract": row[
                            "target_net_pnl_per_contract"
                        ],
                    }
                )
        fold_reports.append(
            {
                "fold_index": fold_index,
                "training_market_count": len(training_markets),
                "validation_market_count": len(validation_markets),
                "training_market_ids_sha256": canonical_json_sha256(
                    sorted(training_markets)
                ),
                "validation_market_ids": validation_markets,
                "validation_market_ids_sha256": canonical_json_sha256(
                    validation_markets
                ),
                "market_overlap_count": 0,
                "family_metrics": family_metrics,
            }
        )
    expected_trade_rows = [
        row for row in rows if row["action_family"] in TRADE_FAMILIES
    ]
    if len(oof_rows) != len(expected_trade_rows):
        raise ValueError("cross-fit OOF prediction coverage is incomplete")
    if len({str(row["action_row_sha256"]) for row in oof_rows}) != len(oof_rows):
        raise ValueError("cross-fit OOF prediction identities are duplicated")
    metrics_by_family = {}
    for family in TRADE_FAMILIES:
        family_rows = [row for row in oof_rows if row["action_family"] == family]
        metrics_by_family[family] = _regression_metrics(
            [float(row["target_net_pnl_per_contract"]) for row in family_rows],
            [float(row["oof_raw_prediction"]) for row in family_rows],
        )
    return {
        "method": "five_fold_chronological_contiguous_market_grouped_oof",
        "fold_count": fold_count,
        "market_count": len(ordered_markets),
        "trade_action_row_count": len(expected_trade_rows),
        "oof_prediction_count": len(oof_rows),
        "oof_prediction_coverage_complete": True,
        "fold_reports": fold_reports,
        "metrics_by_family": metrics_by_family,
        "uses_development_calibration_labels": False,
        "uses_confirmatory_validation_labels": False,
    }


def _xgb_model_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "objective",
        "eval_metric",
        "num_boost_round",
        "max_depth",
        "eta",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "lambda",
        "alpha",
        "seed",
        "nthread",
        "verbosity",
    )
    return {field: protocol[field] for field in fields}


def _predict_role_rows(
    rows: list[dict[str, Any]],
    *,
    boosters: dict[str, xgb.Booster],
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    raw_by_identity: dict[str, float] = {}
    for family in TRADE_FAMILIES:
        family_rows = [row for row in rows if row["action_family"] == family]
        predictions = _predict_booster(
            boosters[family], family_rows, feature_columns=feature_columns
        )
        raw_by_identity.update(
            {
                str(row["action_row_sha256"]): prediction
                for row, prediction in zip(family_rows, predictions, strict=True)
            }
        )
    output = []
    for row in rows:
        family = str(row["action_family"])
        raw_prediction = (
            0.0
            if family == "NO_TRADE"
            else raw_by_identity[str(row["action_row_sha256"])]
        )
        prediction = {
            **row,
            "raw_family_expected_net_return": raw_prediction,
            "ranking_score_source": "raw_family_model_expected_net_return",
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        prediction["prediction_sha256"] = canonical_json_sha256(prediction)
        output.append(prediction)
    return output


def _family_lcb_artifact(
    calibration_predictions: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    feature_contract_sha256: str,
) -> dict[str, Any]:
    lcb_protocol = dict(protocol["conformal_lcb_protocol"])
    quantile = float(lcb_protocol["one_sided_quantile"])
    minimum_rows = int(lcb_protocol["minimum_calibration_rows_per_family"])
    family_rows: dict[str, Any] = {}
    for family in TRADE_FAMILIES:
        rows = [
            row
            for row in calibration_predictions
            if row["action_family"] == family
        ]
        if len(rows) < minimum_rows:
            raise ValueError(f"insufficient calibration rows for {family}")
        residuals = np.asarray(
            [
                float(row["raw_family_expected_net_return"])
                - float(row["target_net_pnl_per_contract"])
                for row in rows
            ],
            dtype=np.float64,
        )
        residual_quantile = float(np.quantile(residuals, quantile, method="higher"))
        if not math.isfinite(residual_quantile):
            raise ValueError("family residual quantile must be finite")
        family_rows[family] = {
            "calibration_row_count": len(rows),
            "one_sided_quantile": quantile,
            "raw_prediction_minus_target_quantile": residual_quantile,
            "raw_metrics": _regression_metrics(
                [float(row["target_net_pnl_per_contract"]) for row in rows],
                [float(row["raw_family_expected_net_return"]) for row in rows],
            ),
        }
    artifact = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-artifact-v1",
        "candidate_name": protocol["candidate_name"],
        "source_split": "development_calibration_only",
        "method": "family_one_sided_prediction_error_quantile_lcb",
        "decision_score_formula": (
            "raw_family_expected_net_return - "
            "family_raw_prediction_minus_target_quantile"
        ),
        "families": family_rows,
        "feature_contract_sha256": feature_contract_sha256,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def _apply_lcb_scores(
    predictions: list[dict[str, Any]],
    *,
    lcb_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for row in predictions:
        family = str(row["action_family"])
        penalty = (
            0.0
            if family == "NO_TRADE"
            else float(
                lcb_artifact["families"][family][
                    "raw_prediction_minus_target_quantile"
                ]
            )
        )
        updated = {
            **row,
            "family_lcb_penalty": penalty,
            "family_lcb_expected_net_return": float(
                row["raw_family_expected_net_return"]
            )
            - penalty,
            "ranking_score_source": "calibration_only_family_lcb_expected_net_return",
        }
        updated["prediction_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def _run_policy_replay(
    predictions: list[dict[str, Any]],
    *,
    score_field: str,
    policy_name: str,
    entry_threshold: float,
) -> list[dict[str, Any]]:
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_decision[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    guard_config = _v8_execution_guard_config()
    state = _v8_initial_runtime_state(guard_config)
    closes: dict[str, int] = {}
    replay: list[dict[str, Any]] = []
    for index, ((market_id, decision_ts), action_rows) in enumerate(
        sorted(by_decision.items(), key=lambda item: (item[0][1], item[0][0])),
        start=1,
    ):
        _release_closed_positions(
            state=state,
            market_close_by_open_position=closes,
            decision_ts=decision_ts,
        )
        if {str(row["action"]) for row in action_rows} != set(REQUIRED_ACTIONS):
            raise ValueError("confirmatory action grid is incomplete")
        ranked = sorted(
            action_rows,
            key=lambda row: (-float(row[score_field]), str(row["action"])),
        )
        full_ranking = [
            {
                "rank": rank,
                "selected_action": row["action"],
                "selected_side": row["side"],
                "selected_action_family": row["action_family"],
                "corrected_model_score": float(row[score_field]),
                "raw_model_score": float(row["raw_family_expected_net_return"]),
                "high_score_flag": float(row[score_field]) >= entry_threshold,
                "p_up_action_disagreement": row["p_up_action_disagreement"],
                "microstructure_snapshot": row["microstructure_snapshot"],
            }
            for rank, row in enumerate(ranked, start=1)
        ]
        selected = ranked[0]
        selected_action = str(selected["action"])
        decision_score = float(selected[score_field])
        blockers: list[str] = []
        guard_result: dict[str, Any] | None = None
        if selected_action == "NO_TRADE":
            blockers.append("policy_selected_no_trade")
        elif decision_score < entry_threshold:
            blockers.append("expected_net_return_below_frozen_entry_threshold")
        else:
            guard_context = {
                "decision_group_id": canonical_json_sha256(
                    {"market_id": market_id, "decision_ts": decision_ts}
                ),
                "market_id": market_id,
                "decision_ts": decision_ts,
                "selected_action": selected_action,
                "selected_side": selected["side"],
                "selected_action_family": selected["action_family"],
                "corrected_model_score": decision_score,
                "raw_model_score": selected["raw_family_expected_net_return"],
                "high_score_flag": decision_score >= entry_threshold,
                "p_up": selected["p_up"],
                "p_down": selected["p_down"],
                "p_up_action_disagreement": selected[
                    "p_up_action_disagreement"
                ],
                "microstructure_snapshot": selected["microstructure_snapshot"],
                "reference_price_feature_provenance": selected[
                    "reference_price_feature_provenance"
                ],
                "decision_time_feature_max_input_ts": selected["max_input_ts"],
                "full_5_action_ranking": full_ranking,
            }
            guard_result = _v8_execution_guard_decision(
                guard_context,
                guard_config=guard_config,
                runtime_state=state,
                runtime_mode="simulated_runtime_state",
            )
            blockers.extend(guard_result["execution_blocking_reason_codes"])
        accepted = bool(guard_result and guard_result["order_allowed"])
        executed_action = (
            str(guard_result["execution_guarded_action"])
            if accepted
            else selected_action
        )
        executed = next(
            row for row in action_rows if str(row["action"]) == executed_action
        )
        size = float(guard_result["proposed_order_size"]) if accepted else 0.0
        if accepted:
            order_id = f"{policy_name}-confirmatory-{index:06d}"
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=guard_result,
                simulated_order_id=order_id,
            )
            closes[market_id] = int(executed["market_close_ts"])
        execution_price = float(
            executed["decision_time_features"]["execution_price"]
        )
        target = float(executed["target_net_pnl_per_contract"])
        replay_row = {
            "policy_name": policy_name,
            "market_id": market_id,
            "decision_ts": decision_ts,
            "source_selected_action": selected_action,
            "selected_action": executed_action,
            "selected_side": executed["side"],
            "selected_action_family": executed["action_family"],
            "decision_score": decision_score,
            "score_field": score_field,
            "frozen_entry_threshold": entry_threshold,
            "execution_guard_order_allowed": accepted,
            "guard_action_remapped": accepted and executed_action != selected_action,
            "proposed_order_size": size,
            "accepted_bet_cost_basis": execution_price * size,
            "accepted_bet_net_pnl": target * size if accepted else 0.0,
            "target_cost_components": executed["target_cost_components"],
            "evaluation_target_used_after_selection_for_report_only": True,
            "settlement_resolved_for_report_only": executed[
                "target_resolved_outcome"
            ]
            in {"UP", "DOWN"},
            "execution_blocking_reason_codes": sorted(set(blockers)),
            "required_runtime_fields_present": bool(
                guard_result is None
                or guard_result["required_runtime_fields_present"]
            ),
            "reference_provenance_valid": executed[
                "reference_price_feature_provenance"
            ].get("provenance_valid")
            is True,
            "paper_only": True,
            "capital_at_risk": False,
        }
        replay_row["replay_row_sha256"] = canonical_json_sha256(replay_row)
        replay.append(replay_row)
    return replay


def _confirmatory_gate(
    *,
    protocol: dict[str, Any],
    action_rows: list[dict[str, Any]],
    candidate_replay: list[dict[str, Any]],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    robustness: dict[str, Any],
) -> dict[str, Any]:
    gates = dict(protocol["confirmatory_validation_gates"])
    market_ids = {str(row["market_id"]) for row in action_rows}
    side_counts = dict(candidate_metrics["accepted_bet_count_by_side"])
    family_counts = dict(candidate_metrics["accepted_bet_count_by_family"])
    checks = {
        "confirmatory_unique_market_support": len(market_ids)
        == int(gates["required_unique_market_count"]),
        "accepted_bet_support": candidate_metrics["accepted_bet_count"]
        >= int(gates["minimum_accepted_bet_count"]),
        "accepted_side_support": all(
            int(side_counts.get(side, 0))
            >= int(gates["minimum_accepted_bet_count_per_side"])
            for side in ("UP", "DOWN")
        ),
        "accepted_family_support": all(
            int(family_counts.get(family, 0))
            >= int(gates["minimum_accepted_bet_count_per_family"])
            for family in TRADE_FAMILIES
        ),
        "candidate_net_pnl_positive": candidate_metrics["net_pnl_sum"] > 0.0,
        "candidate_roi_positive": candidate_metrics["roi"] > 0.0,
        "candidate_better_than_frozen_baseline": candidate_metrics["net_pnl_sum"]
        > baseline_metrics["net_pnl_sum"],
        "all_accepted_bets_settled": all(
            row["settlement_resolved_for_report_only"] is True
            for row in candidate_replay
            if row["execution_guard_order_allowed"] is True
        ),
        "zero_missing_runtime_fields": all(
            row["required_runtime_fields_present"] is True
            for row in candidate_replay
        ),
        "zero_provenance_violations": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            and row["reference_price_feature_provenance"].get("provenance_valid")
            is True
            for row in action_rows
            if row["action"] != "NO_TRADE"
        ),
        "zero_forbidden_inference_field_violations": all(
            row["target_used_as_decision_input"] is False
            and row["outcome_fields_used_as_decision_input"] is False
            for row in action_rows
        ),
        "market_bootstrap_reported": robustness[
            "market_bootstrap_interval_95"
        ].get("reported")
        is True,
        "leave_one_market_out_reported": robustness["leave_one_market_out"][
            "reported"
        ]
        is True,
        "largest_winner_removal_reported": robustness["largest_winner_removal"][
            "reported"
        ]
        is True,
    }
    reason_map = {
        "confirmatory_unique_market_support": "insufficient_confirmatory_unique_market_support",
        "accepted_bet_support": "insufficient_confirmatory_accepted_bet_support",
        "accepted_side_support": "insufficient_confirmatory_side_support",
        "accepted_family_support": "insufficient_confirmatory_family_support",
        "candidate_net_pnl_positive": "confirmatory_candidate_net_pnl_not_positive",
        "candidate_roi_positive": "confirmatory_candidate_roi_not_positive",
        "candidate_better_than_frozen_baseline": "confirmatory_candidate_not_better_than_baseline",
        "all_accepted_bets_settled": "confirmatory_accepted_bet_settlement_incomplete",
        "zero_missing_runtime_fields": "confirmatory_runtime_fields_missing",
        "zero_provenance_violations": "confirmatory_provenance_violation",
        "zero_forbidden_inference_field_violations": "confirmatory_forbidden_inference_field_violation",
        "market_bootstrap_reported": "confirmatory_market_bootstrap_missing",
        "leave_one_market_out_reported": "confirmatory_leave_one_market_out_missing",
        "largest_winner_removal_reported": "confirmatory_largest_winner_removal_missing",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    return {"passed": not reasons, "checks": checks, "reason_codes": reasons}


def _training_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #172 Cross-Fitted Family LCB Training",
            "",
            f"- candidate: `{report['candidate_name']}`",
            f"- training markets: `{report['training_market_count']}`",
            f"- calibration markets: `{report['calibration_market_count']}`",
            f"- cross-fit folds: `{report['cross_fit']['fold_count']}`",
            "- confirmatory labels used for tuning: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _calibration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #172 Conformal Family LCB Calibration",
            "",
            f"- source split: `{report['source_split']}`",
            f"- method: `{report['method']}`",
            "- confirmatory labels opened before calibration freeze: `false`",
            "- confirmatory labels used for tuning: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _leakage_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #172 Leakage And Role Audit",
            "",
            f"- audit passed: `{str(report['leakage_and_role_audit_passed']).lower()}`",
            f"- role overlap count: `{report['role_market_overlap_count']}`",
            f"- prior overlap count: `{report['prior_market_overlap_count']}`",
            f"- feature causality violations: `{report['feature_causality_violation_count']}`",
            f"- forbidden inference fields: `{report['forbidden_inference_field_violation_count']}`",
            "- confirmatory labels used for tuning: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _validation_markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate_metrics"]
    baseline = report["baseline_metrics"]
    return "\n".join(
        [
            "# #172 Untouched Confirmatory Validation",
            "",
            f"- gate passed: `{str(report['confirmatory_gate_passed']).lower()}`",
            f"- accepted bets: `{candidate['accepted_bet_count']}`",
            f"- candidate net PnL: `{candidate['net_pnl_sum']:.12f}`",
            f"- baseline net PnL: `{baseline['net_pnl_sum']:.12f}`",
            f"- blockers: `{report['confirmatory_gate_blocking_reason_codes']}`",
            "- confirmatory labels used for tuning: `false`",
            "- future unseen holdout required: `true`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }


def _find_fields(payload: Any, forbidden: set[str], prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden:
                found.add(path)
            found.update(_find_fields(value, forbidden, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(_find_fields(value, forbidden, f"{prefix}[{index}]"))
    return found


def _verified_descriptor(payload: Any, *, name: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(payload.get("path") or "")).resolve()
    expected_sha256 = str(payload.get("sha256") or "")
    _verify_pin(path, expected_sha256, name=name)
    return {"path": str(path), "sha256": expected_sha256.lower()}


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    _require_sha256(expected_sha256, name=f"{name} SHA-256")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
