"""Pre-register direct decision-group action-advantage v2 without label access."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

SCHEMA_PREFIX = "bigan-v8-direct-decision-group-action-advantage-v2"
CANDIDATE_NAME = "direct_decision_group_action_advantage_v2"
FIT_ROLE = "development_train"
QUARANTINED_ROLES = ("development_calibration", "confirmatory_validation")
EXPECTED_ROLE_COUNTS = {
    "development_train": 90,
    "development_calibration": 45,
    "confirmatory_validation": 60,
}
FORBIDDEN_ROLE_ROW_FIELDS = {
    "accepted_bet_net_pnl",
    "evaluation_target_net_pnl_per_contract_by_action",
    "final_outcome",
    "gross_pnl",
    "label",
    "net_pnl",
    "oracle_action",
    "outcome",
    "realized_pnl",
    "resolved_outcome",
    "settlement_outcome",
    "target_net_pnl_per_contract",
    "target_resolved_outcome",
    "winning_outcome",
}


@dataclass(frozen=True, slots=True)
class DirectDecisionGroupAdvantageV2PreRegistrationConfig:
    """SHA-pinned inputs for the issue #197 protocol freeze."""

    run_id: str
    output_dir: Path | str
    freeze_created_at_ts: int
    protocol_path: Path | str
    expected_protocol_sha256: str
    role_assignment_manifest_path: Path | str
    expected_role_assignment_manifest_sha256: str
    power_design_path: Path | str
    expected_power_design_sha256: str
    power_report_path: Path | str
    expected_power_report_sha256: str
    issue190_collection_freeze_path: Path | str
    expected_issue190_collection_freeze_sha256: str
    persistent_collector_protocol_path: Path | str
    expected_persistent_collector_protocol_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.freeze_created_at_ts <= 0:
            raise ValueError("freeze_created_at_ts must be positive")
        for name in (
            "protocol_path",
            "role_assignment_manifest_path",
            "power_design_path",
            "power_report_path",
            "issue190_collection_freeze_path",
            "persistent_collector_protocol_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "expected_protocol_sha256",
            "expected_role_assignment_manifest_sha256",
            "expected_power_design_sha256",
            "expected_power_report_sha256",
            "expected_issue190_collection_freeze_sha256",
            "expected_persistent_collector_protocol_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)


def validate_direct_decision_group_advantage_v2_protocol(
    protocol: dict[str, Any],
) -> None:
    """Fail closed on semantic drift before a new label is opened."""

    estimands = dict(protocol.get("training_estimands") or {})
    decision = dict(protocol.get("decision_rule") or {})
    cross_fit = dict(protocol.get("cross_fit_protocol") or {})
    calibration = dict(protocol.get("calibration_protocol") or {})
    quarantine = dict(protocol.get("quarantined_lineage") or {})
    future = dict(protocol.get("future_evaluation_protocol") or {})
    hashes = dict(protocol.get("frozen_lineage_hashes") or {})
    checks = {
        "schema_version": protocol.get("schema_version")
        == "bigan-v8-direct-decision-group-action-advantage-protocol-v2",
        "candidate_name": protocol.get("candidate_name") == CANDIDATE_NAME,
        "frozen": protocol.get("frozen") is True,
        "complete_action_grid": protocol.get("required_actions") == list(REQUIRED_ACTIONS),
        "decision_group": protocol.get("decision_group_key_fields") == ["market_id", "decision_ts"],
        "separate_estimands": set(estimands)
        >= {
            "absolute_post_cost_net_return",
            "advantage_vs_no_trade",
            "advantage_vs_best_alternative",
        }
        and len(
            {
                estimands.get("absolute_post_cost_net_return"),
                estimands.get("advantage_vs_no_trade"),
                estimands.get("advantage_vs_best_alternative"),
            }
        )
        == 3,
        "targets_sealed": estimands.get("targets_available_only_after_role_assignment") is True
        and estimands.get("targets_never_allowed_as_decision_inputs") is True,
        "joint_lcb_decision": decision.get("trade_must_pass_all_lower_confidence_bounds") is True
        and math.isclose(
            float(decision.get("absolute_post_cost_net_return_lcb_minimum") or 0.0),
            0.02,
        )
        and math.isclose(
            float(decision.get("advantage_vs_no_trade_lcb_minimum") or 0.0),
            0.0,
        )
        and math.isclose(
            float(decision.get("advantage_vs_best_alternative_lcb_minimum") or 0.0),
            0.0,
        ),
        "execution_contract_unchanged": decision.get("execution_guard_mutation_allowed") is False
        and decision.get("cost_model_mutation_allowed") is False
        and decision.get("order_sizing_mutation_allowed") is False,
        "fit_role": cross_fit.get("fit_eligible_role") == FIT_ROLE
        and int(cross_fit.get("required_fit_market_count") or 0) == EXPECTED_ROLE_COUNTS[FIT_ROLE]
        and cross_fit.get("hyperparameter_search_enabled") is False
        and cross_fit.get("group_key") == "market_id"
        and cross_fit.get("decision_group_complete_five_action_grid_required") is True,
        "honest_cross_fit": cross_fit.get("fold_assignment")
        == "chronological_expanding_window_prior_markets_only"
        and isinstance(cross_fit.get("seed"), int)
        and int(cross_fit.get("nthread") or 0) == 1,
        "new_internal_calibration_only": calibration.get("source")
        == "new_internal_cross_fit_predictions_from_development_train_only"
        and calibration.get("current_issue189_oof_files_may_be_opened") is False
        and calibration.get("current_development_calibration_role_may_be_opened") is False
        and calibration.get("current_confirmatory_validation_role_may_be_opened") is False,
        "reachable_adaptive_buckets": int(calibration.get("adaptive_bucket_count_maximum") or 0)
        == 3
        and calibration.get("strictly_increasing_bucket_boundaries_required") is True
        and calibration.get("duplicate_quantile_boundaries_must_merge") is True
        and calibration.get("unreachable_empty_bucket_allowed") is False
        and int(calibration.get("minimum_unique_markets_per_bucket") or 0) >= 10,
        "full_estimator_bootstrap": calibration.get("bootstrap_unit") == "market_id"
        and int(calibration.get("bootstrap_resample_count") or 0) >= 2_000
        and float(calibration.get("confidence_level") or 0.0) >= 0.95
        and calibration.get("bootstrap_complete_shrunken_estimator_required") is True
        and calibration.get("convex_combination_of_separately_estimated_lcbs_allowed") is False,
        "no_validation_tuning": calibration.get(
            "validation_or_holdout_labels_used_for_bucket_or_threshold_selection"
        )
        is False,
        "quarantined_roles": quarantine.get("eligible_fit_role") == FIT_ROLE
        and quarantine.get("prohibited_roles") == list(QUARANTINED_ROLES)
        and quarantine.get("current_oof_validation_or_confirmatory_pnl_used_for_design") is False,
        "future_power": int(future.get("minimum_quality_valid_market_count") or 0) == 220
        and int(future.get("minimum_accepted_unique_market_count") or 0) == 88
        and int(future.get("minimum_accepted_bet_count_per_side") or 0) >= 10
        and int(future.get("minimum_accepted_bet_count_per_family") or 0) >= 10,
        "strict_future_selection": future.get("selection_method")
        == "earliest_quality_valid_strictly_post_freeze_market_disjoint_rows"
        and future.get("fixed_attempt_count_batch_required") is False
        and future.get("result_dependent_extension_allowed") is False
        and future.get("labels_outcomes_or_pnl_opened_for_selection") is False
        and future.get("decision_ts_strictly_after_protocol_freeze_required") is True,
        "future_disjoint": future.get("market_id_disjoint_from_fit_required") is True
        and future.get("slug_disjoint_from_fit_required") is True
        and future.get("source_row_hash_disjoint_from_fit_required") is True,
        "parallel_future_collection": future.get(
            "future_holdout_collection_may_precede_candidate_fit"
        )
        is True
        and future.get("future_labels_must_remain_sealed_until_candidate_freeze") is True
        and future.get("future_holdout_evaluation_requires_candidate_freeze") is True,
        "lineage_hashes": len(hashes) == 5
        and all(_is_sha256(str(value)) for value in hashes.values()),
        "safety": _safety_blocked(protocol),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "invalid direct decision-group action-advantage v2 protocol: " + ", ".join(failed)
        )


def freeze_direct_decision_group_advantage_v2_pre_registration(
    config: DirectDecisionGroupAdvantageV2PreRegistrationConfig,
) -> dict[str, Any]:
    """Freeze data lineage and future boundary without opening model targets."""

    paths = {
        "protocol": config.protocol_path.resolve(),
        "role_assignment_manifest": config.role_assignment_manifest_path.resolve(),
        "power_design": config.power_design_path.resolve(),
        "power_report": config.power_report_path.resolve(),
        "issue190_collection_freeze": config.issue190_collection_freeze_path.resolve(),
        "persistent_collector_protocol": (config.persistent_collector_protocol_path.resolve()),
    }
    expected_hashes = {
        "protocol": config.expected_protocol_sha256,
        "role_assignment_manifest": (config.expected_role_assignment_manifest_sha256),
        "power_design": config.expected_power_design_sha256,
        "power_report": config.expected_power_report_sha256,
        "issue190_collection_freeze": (config.expected_issue190_collection_freeze_sha256),
        "persistent_collector_protocol": (config.expected_persistent_collector_protocol_sha256),
    }
    for name, path in paths.items():
        _verify_pin(path, expected_hashes[name], name=name)

    protocol = _load_json(paths["protocol"])
    validate_direct_decision_group_advantage_v2_protocol(protocol)
    frozen_hashes = dict(protocol["frozen_lineage_hashes"])
    _require_expected_hash(
        frozen_hashes,
        "power_design_sha256",
        expected_hashes["power_design"],
    )
    _require_expected_hash(
        frozen_hashes,
        "power_report_sha256",
        expected_hashes["power_report"],
    )
    _require_expected_hash(
        frozen_hashes,
        "issue190_collection_freeze_sha256",
        expected_hashes["issue190_collection_freeze"],
    )
    _require_expected_hash(
        frozen_hashes,
        "issue192_persistent_collector_protocol_sha256",
        expected_hashes["persistent_collector_protocol"],
    )

    role_manifest = _load_json(paths["role_assignment_manifest"])
    _validate_role_manifest(role_manifest)
    selected_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"),
        name="role assignment selected rows",
    )
    _require_expected_hash(
        dict(protocol["quarantined_lineage"]),
        "selected_rows_sha256",
        selected_rows_descriptor["sha256"],
    )
    _require_expected_hash(
        dict(protocol["quarantined_lineage"]),
        "role_assignment_manifest_sha256",
        expected_hashes["role_assignment_manifest"],
    )
    selected_rows = _load_jsonl(Path(selected_rows_descriptor["path"]))
    forbidden = _find_fields(
        {"rows": selected_rows},
        FORBIDDEN_ROLE_ROW_FIELDS,
    )
    if forbidden:
        raise ValueError("role assignment rows contain forbidden target fields")
    role_counts = dict(sorted(Counter(str(row.get("role")) for row in selected_rows).items()))
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise ValueError("role assignment counts do not match the frozen split")
    fit_market_ids = {str(row["market_id"]) for row in selected_rows if row["role"] == FIT_ROLE}
    quarantined_market_ids = {
        str(row["market_id"]) for row in selected_rows if row["role"] in QUARANTINED_ROLES
    }
    if fit_market_ids & quarantined_market_ids:
        raise ValueError("fit and quarantined role markets overlap")

    feature_contract = _verified_descriptor(
        role_manifest.get("feature_contract"),
        name="role assignment feature contract",
    )
    _require_expected_hash(
        frozen_hashes,
        "feature_contract_sha256",
        feature_contract["sha256"],
    )
    power_design = _load_json(paths["power_design"])
    power_report = _load_json(paths["power_report"])
    _validate_power_evidence(power_design, power_report)
    issue190_freeze = _load_json(paths["issue190_collection_freeze"])
    _validate_issue190_collection_freeze(issue190_freeze)
    persistent_protocol = _load_json(paths["persistent_collector_protocol"])
    _validate_persistent_protocol(persistent_protocol)

    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    descriptors = {name: _descriptor(path) for name, path in paths.items()}
    descriptors["role_assignment_selected_rows"] = selected_rows_descriptor
    descriptors["feature_contract"] = feature_contract
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-registration-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "pre_registration_ready": True,
        "protocol_frozen": True,
        "protocol_freeze_created_at_ts": config.freeze_created_at_ts,
        "minimum_future_decision_ts_exclusive": config.freeze_created_at_ts,
        "fit_eligible_role": FIT_ROLE,
        "fit_eligible_market_count": len(fit_market_ids),
        "fit_eligible_market_ids_sha256": canonical_json_sha256(sorted(fit_market_ids)),
        "quarantined_roles": list(QUARANTINED_ROLES),
        "quarantined_market_count": len(quarantined_market_ids),
        "quarantined_market_ids_sha256": canonical_json_sha256(sorted(quarantined_market_ids)),
        "role_counts": role_counts,
        "role_assignment_rows_opened": True,
        "feature_row_files_opened": False,
        "label_outcome_or_pnl_files_opened": False,
        "current_issue189_oof_files_opened": False,
        "current_oof_validation_or_confirmatory_pnl_used": False,
        "new_label_access_allowed_by_this_issue": False,
        "training_estimands": protocol["training_estimands"],
        "decision_rule": protocol["decision_rule"],
        "calibration_protocol": protocol["calibration_protocol"],
        "future_evaluation_protocol": protocol["future_evaluation_protocol"],
        "future_quality_valid_market_target": 220,
        "future_accepted_unique_market_target": 88,
        "strictly_post_freeze_future_evidence_required": True,
        "future_holdout_collection_may_precede_candidate_fit": True,
        "future_labels_must_remain_sealed_until_candidate_freeze": True,
        "future_holdout_evaluation_requires_candidate_freeze": True,
        "persistent_stream_shortfall_source_armed": True,
        "fixed_attempt_count_batch_required": False,
        "result_dependent_extension_allowed": False,
        "input_descriptors": descriptors,
        "threshold_guard_cost_or_sizing_mutated": False,
        "fitting_or_prediction_attempted": False,
        **_diagnostic_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "direct_decision_group_action_advantage_v2_pre_registration_report.json"
    markdown_path = run_dir / "direct_decision_group_action_advantage_v2_pre_registration_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _report_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-registration-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "protocol_freeze_created_at_ts": config.freeze_created_at_ts,
        "minimum_future_decision_ts_exclusive": config.freeze_created_at_ts,
        "pre_registration_ready": True,
        "input_descriptors": descriptors,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(markdown_path),
        "feature_row_files_opened": False,
        "label_outcome_or_pnl_files_opened": False,
        "current_oof_validation_or_confirmatory_pnl_used": False,
        "fitting_or_prediction_attempted": False,
        **_diagnostic_safety_fields(),
    }
    manifest["pre_registration_id"] = canonical_json_sha256(manifest)
    manifest_path = (
        run_dir / "direct_decision_group_action_advantage_v2_pre_registration_manifest.json"
    )
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _validate_role_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("role_assignment_ready") is not True:
        raise ValueError("role assignment is not ready")
    if manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        raise ValueError("role assignment was not outcome blind")
    if not _safety_blocked(manifest):
        raise ValueError("role assignment safety contract is not blocked")


def _validate_power_evidence(design: dict[str, Any], report: dict[str, Any]) -> None:
    if not (
        report.get("power_analysis_ready") is True
        and report.get("uses_current_oof_validation_or_confirmatory_pnl") is False
        and report.get("uses_realized_candidate_pnl_for_design") is False
        and int(report.get("recommended_quality_valid_market_count") or 0) == 220
        and int(report.get("recommended_required_accepted_unique_market_count") or 0) == 88
        and report.get("result_dependent_extension_allowed") is False
        and design.get("uses_current_oof_validation_or_confirmatory_pnl") is False
        and design.get("uses_realized_candidate_pnl_for_design") is False
    ):
        raise ValueError("prospective power evidence is invalid")


def _validate_issue190_collection_freeze(payload: dict[str, Any]) -> None:
    if not (
        payload.get("collection_control_is_outcome_blind") is True
        and payload.get("labels_or_outcomes_opened_for_collection_freeze") is False
        and payload.get("settlement_finalizer_started_during_collection") is False
        and payload.get("training_corpus_export_during_collection_allowed") is False
        and int(payload.get("target_valid_market_count") or 0) == 220
        and _safety_blocked(payload)
    ):
        raise ValueError("issue #190 collection freeze is invalid")


def _validate_persistent_protocol(payload: dict[str, Any]) -> None:
    if not (
        payload.get("frozen") is True
        and payload.get("outcome_blind_collection_only") is True
        and payload.get("settlement_finalizer_enabled") is False
        and payload.get("resolution_provider_enabled") is False
        and payload.get("training_corpus_export_enabled") is False
        and payload.get("labels_outcomes_or_pnl_opened") is False
        and payload.get("append_only_index", {}).get("hash_chain_required") is True
        and _safety_blocked(payload)
    ):
        raise ValueError("persistent collector protocol is invalid")


def _safety_blocked(payload: dict[str, Any]) -> bool:
    expected = {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    return all(payload.get(key) is value for key, value in expected.items())


def _diagnostic_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "paper_candidate_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _require_expected_hash(
    payload: dict[str, Any],
    key: str,
    expected: str,
) -> None:
    if str(payload.get(key) or "").lower() != expected.lower():
        raise ValueError(f"frozen lineage hash mismatch: {key}")


def _verified_descriptor(payload: Any, *, name: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(payload.get("path") or "")).expanduser().resolve()
    expected = str(payload.get("sha256") or "").lower()
    _verify_pin(path, expected, name=name)
    return {"path": str(path), "sha256": expected}


def _verify_pin(path: Path, expected: str, *, name: str) -> None:
    _require_sha256(expected, name=f"{name} SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if _sha256_file(path) != expected.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _require_sha256(value: str, *, name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSONL object: {path}")
        rows.append(payload)
    return rows


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


def _prepare_run_dir(output_dir: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_dir.expanduser().resolve() / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Direct Decision-Group Action-Advantage v2 Pre-Registration",
            "",
            f"- candidate: `{report['candidate_name']}`",
            f"- protocol freeze timestamp: `{report['protocol_freeze_created_at_ts']}`",
            f"- fit role / markets: `{report['fit_eligible_role']} / {report['fit_eligible_market_count']}`",
            f"- quarantined roles: `{report['quarantined_roles']}`",
            "- current OOF/validation/confirmatory PnL used: `false`",
            "- feature, label, outcome, or PnL files opened: `false`",
            "- future quality-valid / accepted-market target: `220 / 88`",
            "- future evidence must be strictly post-freeze and market-disjoint: `true`",
            "- future collection may run before candidate fit while labels stay sealed: `true`",
            "- fixed attempt-count batch required: `false`",
            "- threshold/guard/cost/sizing mutation: `false`",
            "- fitting or prediction attempted: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )
