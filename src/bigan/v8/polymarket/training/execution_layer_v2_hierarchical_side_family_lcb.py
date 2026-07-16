"""Frozen #174 hierarchical side/family LCB candidate and collection boundary."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_execution_compatible_mean_lcb_fit import (
    TRADE_FAMILIES,
    _blocked_safety_fields,
    _cross_fit_training_predictions,
    _descriptor,
    _load_json,
    _load_jsonl,
    _market_grouped_mean_residual_ci,
    _materialize_role_action_rows,
    _predict_role_rows,
    _require_sha256,
    _run_policy_replay,
    _sha256_file,
    _validate_role_rows,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
    _xgb_model_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_action_value import (
    _accepted_bet_metrics,
    _train_family_booster,
)

SCHEMA_PREFIX = "bigan-v8-execution-layer-v2-hierarchical-side-family-lcb"
PROTOCOL_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-protocol-v1"
FEATURE_CONTRACT_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-feature-contract-v1"
CANDIDATE_NAME = "hierarchical_side_family_execution_regime_expected_mean_lcb_v1"
DEVELOPMENT_ROLE_COUNTS = {
    "development_train": 60,
    "development_calibration": 30,
}
QUARANTINED_ROLE = "confirmatory_validation"
QUARANTINED_MARKET_COUNT = 30
FRESH_CONFIRMATORY_MARKET_COUNT = 60
MODEL_FILENAMES = {
    "HOLD_TO_SETTLEMENT": "hierarchical_side_family_lcb_hts_model.xgb.json",
    "SELL_BEFORE_CLOSE": "hierarchical_side_family_lcb_sbc_model.xgb.json",
}


@dataclass(frozen=True, slots=True)
class HierarchicalSideFamilyLCBFreezeConfig:
    """Hash-pinned development inputs for freezing #174 before collection."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    issue173_role_assignment_manifest_path: Path | str
    expected_issue173_role_assignment_manifest_sha256: str
    issue173_development_fit_freeze_path: Path | str
    expected_issue173_development_fit_freeze_sha256: str
    expected_prior_unique_market_count: int
    git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, value in (
            ("protocol", self.expected_protocol_sha256),
            ("feature contract", self.expected_feature_contract_sha256),
            (
                "issue173 role assignment manifest",
                self.expected_issue173_role_assignment_manifest_sha256,
            ),
            (
                "issue173 development fit freeze",
                self.expected_issue173_development_fit_freeze_sha256,
            ),
        ):
            _require_sha256(value, name=f"{name} SHA-256")
        if self.expected_prior_unique_market_count < 120:
            raise ValueError("expected prior market count must cover all #173 markets")
        if len(self.git_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.git_commit.lower()
        ):
            raise ValueError("git_commit must be a 40-character hex digest")
        for field in (
            "output_dir",
            "protocol_path",
            "feature_contract_path",
            "issue173_role_assignment_manifest_path",
            "issue173_development_fit_freeze_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def validate_hierarchical_side_family_lcb_protocol(
    protocol: dict[str, Any],
) -> None:
    """Reject any drift in the predeclared #174 protocol."""

    development = dict(protocol.get("development_source_contract") or {})
    collector = dict(protocol.get("collector_contract") or {})
    collection = dict(protocol.get("fresh_confirmatory_collection") or {})
    cross_fit = dict(protocol.get("cross_fit_protocol") or {})
    hierarchy = dict(protocol.get("hierarchical_expected_mean_lcb_protocol") or {})
    regime = dict(hierarchy.get("execution_regime_definition") or {})
    development_gates = dict(protocol.get("development_freeze_gates") or {})
    confirmatory_gates = dict(protocol.get("confirmatory_validation_gates") or {})
    execution = dict(protocol.get("frozen_execution_contract") or {})
    safety = dict(protocol.get("safety") or {})
    checks = {
        "schema_version": protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "candidate_name": protocol.get("candidate_name") == CANDIDATE_NAME,
        "frozen": protocol.get("frozen") is True,
        "decision_time_safe": protocol.get("decision_time_safe") is True,
        "issue173_confirmatory_not_used": protocol.get(
            "uses_issue173_confirmatory_labels_for_tuning"
        )
        is False,
        "prior_future_not_used": protocol.get("uses_prior_validation_or_future_labels_for_tuning")
        is False,
        "development_roles": development.get("development_roles_only") is True
        and development.get("confirmatory_artifact_access_forbidden") is True
        and int(development.get("development_train_market_count") or 0) == 60
        and int(development.get("development_calibration_market_count") or 0) == 30
        and int(development.get("quarantined_market_count") or 0) == 30,
        "issue173_role_pin": development.get("role_assignment_manifest_sha256")
        == "ee5e3778ac400f49be7188013b56a881f2872824a6ece873677a3a01a49f2194",
        "collector": collector.get("orderbook_source_priority")
        == "clob_websocket_primary_rest_fallback"
        and float(collector.get("orderbook_snapshot_interval_seconds") or 0.0) == 1.0
        and float(collector.get("maximum_selected_side_book_staleness_ms") or 0.0) == 2_000.0
        and float(collector.get("maximum_opposite_side_book_staleness_ms") or 0.0) == 2_000.0
        and collector.get("complete_up_down_executable_book_required") is True
        and collector.get("execution_compatibility_validated_before_label_access") is True
        and collector.get("training_corpus_root") == "/Volumes/PHILIPS/v8",
        "fresh_collection": int(collection.get("target_valid_unique_market_count") or 0)
        == FRESH_CONFIRMATORY_MARKET_COUNT
        and int(collection.get("maximum_total_capture_attempt_count") or 0) == 90
        and collection.get("outcome_blind_quality_assignment") is True
        and collection.get("strictly_later_than_candidate_freeze") is True
        and collection.get("strictly_later_than_all_issue173_decisions") is True
        and collection.get("market_disjoint_from_prior_registry") is True,
        "cross_fit": int(cross_fit.get("fold_count") or 0) == 5
        and cross_fit.get("group_key") == "market_id"
        and cross_fit.get("fit_split") == "development_train_only"
        and cross_fit.get("objective") == "reg:squarederror"
        and cross_fit.get("nthread") == 1,
        "hierarchy": hierarchy.get("source_split") == "development_calibration_only"
        and hierarchy.get("estimand") == "conditional_expected_cost_aware_net_return"
        and hierarchy.get("bootstrap_unit") == "market_id"
        and list(hierarchy.get("hierarchy") or [])
        == [
            "action_family",
            "action_family_x_side",
            "action_family_x_side_x_execution_regime",
        ]
        and hierarchy.get("individual_outcome_quantile_subtraction_enabled") is False
        and hierarchy.get("affine_calibration_enabled") is False
        and hierarchy.get("forced_action_side_or_family_quota_enabled") is False,
        "regime_boundaries": list(regime.get("execution_price_boundaries") or []) == [0.55, 0.75]
        and list(regime.get("time_to_close_boundaries_seconds") or []) == [90.0, 200.0]
        and regime.get("boundary_source") == "predeclared_protocol_not_outcome_fit",
        "support": int(hierarchy.get("minimum_unique_markets_per_family") or 0) == 20
        and int(hierarchy.get("minimum_unique_markets_per_family_side") or 0) == 15
        and int(hierarchy.get("minimum_unique_markets_per_leaf") or 0) == 8,
        "development_gate": int(development_gates.get("required_train_market_count") or 0) == 60
        and int(development_gates.get("required_calibration_market_count") or 0) == 30
        and development_gates.get("candidate_net_pnl_must_exceed_frozen_baseline") is True,
        "confirmatory_gate": int(confirmatory_gates.get("required_unique_market_count") or 0) == 60
        and confirmatory_gates.get(
            "candidate_minus_baseline_market_bootstrap_lower_bound_must_be_positive"
        )
        is True,
        "execution_frozen": float(execution.get("entry_edge_threshold") or 0.0) == 0.02
        and execution.get("execution_guard_mutation_allowed") is False
        and execution.get("order_sizing_mutation_allowed") is False
        and execution.get("cost_model_mutation_allowed") is False,
        "safety": safety == _blocked_safety_fields(),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("invalid #174 protocol: " + ", ".join(failures))


def validate_hierarchical_side_family_lcb_feature_contract(
    contract: dict[str, Any],
    *,
    expected_parent_protocol_sha256: str,
) -> None:
    """Validate the causal input and evidence-access contract."""

    feature_columns = list(contract.get("feature_columns") or [])
    regime_columns = list(contract.get("execution_regime_input_columns") or [])
    checks = {
        "schema_version": contract.get("schema_version") == FEATURE_CONTRACT_SCHEMA_VERSION,
        "candidate_name": contract.get("candidate_name") == CANDIDATE_NAME,
        "parent_protocol": contract.get("parent_protocol_sha256")
        == expected_parent_protocol_sha256,
        "frozen": contract.get("frozen") is True,
        "decision_time_safe": contract.get("decision_time_safe") is True,
        "feature_source": contract.get("feature_source") == "phase2_polymarket_feature_rows_only",
        "chainlink": contract.get("chainlink_reference_feature_required") is True
        and contract.get("btc_candle_features_may_not_supply_price_to_beat") is True,
        "development_targets": contract.get("uses_development_train_labels_for_model_fitting")
        is True
        and contract.get("uses_development_calibration_labels_for_hierarchical_residual_bounds")
        is True,
        "quarantine": contract.get("uses_issue173_confirmatory_labels_for_tuning") is False
        and contract.get("issue173_confirmatory_artifact_access_forbidden") is True
        and contract.get("uses_future_confirmatory_labels_for_tuning") is False,
        "features": len(feature_columns) >= 20
        and len(feature_columns) == len(set(feature_columns)),
        "regime_inputs": set(regime_columns).issubset(feature_columns) and len(regime_columns) == 5,
        "cost_target": contract.get("target_field") == "total_net_pnl_per_notional"
        and contract.get("target_includes_fees_slippage_and_liquidity_impact") is True,
        "probability_semantics": contract.get(
            "market_implied_probability_used_as_conditioning_feature"
        )
        is True
        and contract.get("market_implied_probability_used_as_direct_fair_value_ev") is False,
        "outcome_inputs": contract.get("settlement_or_outcome_fields_allowed_as_decision_inputs")
        is False,
        "no_quota": contract.get("forced_action_side_or_family_quota_enabled") is False,
        "safety": all(
            contract.get(key) == value for key, value in _blocked_safety_fields().items()
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("invalid #174 feature contract: " + ", ".join(failures))


def freeze_hierarchical_side_family_lcb_candidate(
    config: HierarchicalSideFamilyLCBFreezeConfig,
) -> dict[str, Any]:
    """Fit only on allowed development roles and freeze fresh collection inputs."""

    protocol_path = config.protocol_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    role_manifest_path = config.issue173_role_assignment_manifest_path.resolve()
    development_fit_path = config.issue173_development_fit_freeze_path.resolve()
    for path, expected, name in (
        (protocol_path, config.expected_protocol_sha256, "#174 protocol"),
        (
            feature_contract_path,
            config.expected_feature_contract_sha256,
            "#174 feature contract",
        ),
        (
            role_manifest_path,
            config.expected_issue173_role_assignment_manifest_sha256,
            "#173 role assignment manifest",
        ),
        (
            development_fit_path,
            config.expected_issue173_development_fit_freeze_sha256,
            "#173 development fit freeze",
        ),
    ):
        _verify_pin(path, expected, name=name)

    protocol = _load_json(protocol_path)
    validate_hierarchical_side_family_lcb_protocol(protocol)
    feature_contract = _load_json(feature_contract_path)
    validate_hierarchical_side_family_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=config.expected_protocol_sha256,
    )
    role_manifest = _load_json(role_manifest_path)
    if (
        role_manifest.get("role_assignment_ready") is not True
        or role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False
    ):
        raise ValueError("#173 outcome-blind role assignment is not valid")
    selected_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"), name="#173 selected role rows"
    )
    role_rows = _load_jsonl(Path(selected_descriptor["path"]))
    _validate_role_rows(role_rows)
    development_fit = _load_json(development_fit_path)
    if (
        development_fit.get("confirmatory_labels_opened_before_this_freeze") is not False
        or development_fit.get("uses_confirmatory_validation_labels_for_tuning") is not False
    ):
        raise ValueError("#173 development fit lineage is not confirmatory-blind")

    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    development_rows = [row for row in role_rows if str(row["role"]) in DEVELOPMENT_ROLE_COUNTS]
    quarantined_rows = [row for row in role_rows if str(row["role"]) == QUARANTINED_ROLE]
    _validate_development_and_quarantine_rows(development_rows, quarantined_rows)
    development_market_ids = {str(row["market_id"]) for row in development_rows}
    quarantined_market_ids = {str(row["market_id"]) for row in quarantined_rows}
    if development_market_ids & quarantined_market_ids:
        raise ValueError("development and quarantine market identities overlap")

    allowlist = _development_allowlist(
        run_id=config.run_id,
        role_manifest_path=role_manifest_path,
        rows=development_rows,
    )
    allowlist_path = run_dir / "development_source_allowlist.json"
    _write_json(allowlist_path, allowlist)
    quarantine = _confirmatory_quarantine_registry(
        run_id=config.run_id,
        role_manifest_path=role_manifest_path,
        rows=quarantined_rows,
    )
    quarantine_path = run_dir / "issue173_confirmatory_quarantine_registry.json"
    _write_json(quarantine_path, quarantine)
    prior_registry = _prior_market_registry(
        run_id=config.run_id,
        role_manifest=role_manifest,
        issue173_rows=role_rows,
        expected_count=config.expected_prior_unique_market_count,
    )
    prior_registry_path = run_dir / "prior_market_exclusion_registry.json"
    _write_json(prior_registry_path, prior_registry)

    feature_columns = tuple(feature_contract["feature_columns"])
    action_rows_by_role, corpus_audits = _materialize_role_action_rows(
        role_rows,
        feature_columns=feature_columns,
        roles=("development_train", "development_calibration"),
    )
    opened_market_ids = {str(row["market_id"]) for row in corpus_audits}
    quarantine_access_overlap = opened_market_ids & quarantined_market_ids
    if quarantine_access_overlap:
        raise ValueError("#173 confirmatory corpus was opened during development fit")
    action_row_paths: dict[str, Path] = {}
    for role, rows in action_rows_by_role.items():
        path = run_dir / f"hierarchical_side_family_lcb_{role}_action_rows.jsonl"
        _write_jsonl(path, rows)
        action_row_paths[role] = path

    cross_fit = _cross_fit_training_predictions(
        action_rows_by_role["development_train"],
        feature_columns=feature_columns,
        model_protocol=dict(protocol["cross_fit_protocol"]),
    )
    oof_predictions = list(cross_fit.pop("oof_predictions"))
    oof_path = run_dir / "hierarchical_side_family_lcb_train_oof_predictions.jsonl"
    _write_jsonl(oof_path, oof_predictions)

    boosters: dict[str, Any] = {}
    model_paths: dict[str, Path] = {}
    train_rows = action_rows_by_role["development_train"]
    for family in TRADE_FAMILIES:
        booster = _train_family_booster(
            [row for row in train_rows if row["action_family"] == family],
            feature_columns=feature_columns,
            model_protocol=_xgb_model_protocol(dict(protocol["cross_fit_protocol"])),
        )
        model_path = run_dir / MODEL_FILENAMES[family]
        booster.save_model(model_path)
        boosters[family] = booster
        model_paths[family] = model_path

    calibration_predictions = _predict_role_rows(
        action_rows_by_role["development_calibration"],
        boosters=boosters,
        feature_columns=feature_columns,
    )
    hierarchy_artifact = _hierarchical_lcb_artifact(
        calibration_predictions,
        protocol=protocol,
        feature_contract_sha256=config.expected_feature_contract_sha256,
    )
    hierarchy_path = run_dir / "hierarchical_expected_mean_lcb_artifact.json"
    _write_json(hierarchy_path, hierarchy_artifact)
    scored_calibration = _apply_hierarchical_lcb_scores(
        calibration_predictions,
        artifact=hierarchy_artifact,
    )
    scored_path = run_dir / "development_calibration_scored_predictions.jsonl"
    _write_jsonl(scored_path, scored_calibration)

    threshold = float(protocol["frozen_execution_contract"]["entry_edge_threshold"])
    candidate_replay = _run_policy_replay(
        scored_calibration,
        score_field="hierarchical_expected_mean_lcb_net_return",
        policy_name=CANDIDATE_NAME,
        entry_threshold=threshold,
    )
    baseline_replay = _run_policy_replay(
        scored_calibration,
        score_field="raw_family_expected_net_return",
        policy_name="frozen_uncertainty_unadjusted_family_model_baseline",
        entry_threshold=threshold,
    )
    candidate_replay_path = run_dir / "development_calibration_candidate_replay.jsonl"
    baseline_replay_path = run_dir / "development_calibration_baseline_replay.jsonl"
    _write_jsonl(candidate_replay_path, candidate_replay)
    _write_jsonl(baseline_replay_path, baseline_replay)
    candidate_metrics = _accepted_bet_metrics(candidate_replay)
    baseline_metrics = _accepted_bet_metrics(baseline_replay)
    development_gate = _development_freeze_gate(
        protocol=protocol,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        hierarchy_artifact=hierarchy_artifact,
        corpus_audits=corpus_audits,
        quarantine_access_overlap=quarantine_access_overlap,
    )

    evidence_access_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-evidence-access-audit-v1",
        "run_id": config.run_id,
        "development_allowlist": _descriptor(allowlist_path),
        "issue173_confirmatory_quarantine_registry": _descriptor(quarantine_path),
        "development_market_count": len(development_market_ids),
        "quarantined_market_count": len(quarantined_market_ids),
        "opened_corpus_market_count": len(opened_market_ids),
        "opened_corpus_market_ids_sha256": canonical_json_sha256(sorted(opened_market_ids)),
        "opened_corpus_is_exact_development_allowlist": opened_market_ids == development_market_ids,
        "issue173_confirmatory_artifacts_opened_for_tuning": False,
        "issue173_confirmatory_market_access_overlap_count": 0,
        "uses_issue173_confirmatory_labels_for_tuning": False,
        "uses_future_confirmatory_labels_for_tuning": False,
        "feature_causality_violation_count": sum(
            int(row["feature_causality_violation_count"]) for row in corpus_audits
        ),
        "forbidden_evidence_access_violation_count": 0,
        "evidence_access_audit_passed": opened_market_ids == development_market_ids,
        **_blocked_safety_fields(),
    }
    evidence_audit_path = run_dir / "evidence_access_and_leakage_audit.json"
    _write_json(evidence_audit_path, evidence_access_audit)

    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-development-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "training_market_count": 60,
        "calibration_market_count": 30,
        "feature_columns": list(feature_columns),
        "cross_fit": cross_fit,
        "hierarchical_calibration": _hierarchical_report_summary(hierarchy_artifact),
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "candidate_minus_baseline_net_pnl": float(candidate_metrics["net_pnl_sum"])
        - float(baseline_metrics["net_pnl_sum"]),
        "development_freeze_gate_passed": development_gate["passed"],
        "development_freeze_gate_checks": development_gate["checks"],
        "development_freeze_blocking_reason_codes": development_gate["reason_codes"],
        "issue173_confirmatory_artifacts_opened_for_tuning": False,
        "uses_issue173_confirmatory_labels_for_tuning": False,
        "uses_future_confirmatory_labels_for_tuning": False,
        "forced_action_side_or_family_quota_enabled": False,
        **_blocked_safety_fields(),
    }
    training_report_path = run_dir / "hierarchical_side_family_lcb_training_report.json"
    _write_json(training_report_path, training_report)
    _write_text(
        run_dir / "hierarchical_side_family_lcb_training_report.md",
        _training_markdown(training_report),
    )

    development_freeze = {
        "schema_version": f"{SCHEMA_PREFIX}-development-candidate-freeze-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "git_commit": config.git_commit,
        "protocol": _descriptor(protocol_path),
        "feature_contract": _descriptor(feature_contract_path),
        "issue173_role_assignment_manifest": _descriptor(role_manifest_path),
        "issue173_development_fit_freeze": _descriptor(development_fit_path),
        "development_source_allowlist": _descriptor(allowlist_path),
        "issue173_confirmatory_quarantine_registry": _descriptor(quarantine_path),
        "prior_market_exclusion_registry": _descriptor(prior_registry_path),
        "development_action_rows": {
            role: _descriptor(path) for role, path in action_row_paths.items()
        },
        "train_oof_predictions": _descriptor(oof_path),
        "models": {family: _descriptor(path) for family, path in model_paths.items()},
        "hierarchical_calibration_artifact": _descriptor(hierarchy_path),
        "development_scored_predictions": _descriptor(scored_path),
        "candidate_replay": _descriptor(candidate_replay_path),
        "baseline_replay": _descriptor(baseline_replay_path),
        "training_report": _descriptor(training_report_path),
        "evidence_access_and_leakage_audit": _descriptor(evidence_audit_path),
        "development_freeze_gate_passed": development_gate["passed"],
        "development_freeze_blocking_reason_codes": development_gate["reason_codes"],
        "candidate_configuration_frozen_before_fresh_confirmatory_collection": True,
        "issue173_confirmatory_artifacts_opened_for_tuning": False,
        "uses_issue173_confirmatory_labels_for_tuning": False,
        "uses_future_confirmatory_labels_for_tuning": False,
        **_blocked_safety_fields(),
    }
    development_freeze["development_candidate_freeze_id"] = canonical_json_sha256(
        development_freeze
    )
    development_freeze_path = run_dir / "development_candidate_freeze_manifest.json"
    _write_json(development_freeze_path, development_freeze)

    created_ts = int(time.time() * 1000)
    max_prior_decision_ts = int(prior_registry["maximum_prior_decision_ts"])
    collection_ready = bool(development_gate["passed"])
    precollection = {
        "schema_version": f"{SCHEMA_PREFIX}-precollection-freeze-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "created_ts": created_ts,
        "git_commit": config.git_commit,
        "protocol": _descriptor(protocol_path),
        "feature_contract": _descriptor(feature_contract_path),
        "development_candidate_freeze": _descriptor(development_freeze_path),
        "prior_market_exclusion_registry": _descriptor(prior_registry_path),
        "collector_contract": protocol["collector_contract"],
        "fresh_confirmatory_collection": protocol["fresh_confirmatory_collection"],
        "target_valid_unique_market_count": FRESH_CONFIRMATORY_MARKET_COUNT,
        "maximum_total_capture_attempt_count": int(
            protocol["fresh_confirmatory_collection"]["maximum_total_capture_attempt_count"]
        ),
        "minimum_collection_decision_ts": max(created_ts + 1, max_prior_decision_ts + 1),
        "collection_ready": collection_ready,
        "collection_started": False,
        "fresh_confirmatory_labels_opened": False,
        "issue173_confirmatory_artifacts_opened_for_tuning": False,
        "uses_issue173_confirmatory_labels_for_tuning": False,
        "uses_future_confirmatory_labels_for_tuning": False,
        **_blocked_safety_fields(),
    }
    precollection["precollection_freeze_id"] = canonical_json_sha256(precollection)
    precollection_path = run_dir / "precollection_freeze_manifest.json"
    _write_json(precollection_path, precollection)

    bundle = {
        "schema_version": f"{SCHEMA_PREFIX}-freeze-bundle-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "development_candidate_freeze": _descriptor(development_freeze_path),
        "precollection_freeze": _descriptor(precollection_path),
        "training_report": _descriptor(training_report_path),
        "evidence_access_and_leakage_audit": _descriptor(evidence_audit_path),
        "development_source_allowlist": _descriptor(allowlist_path),
        "issue173_confirmatory_quarantine_registry": _descriptor(quarantine_path),
        "prior_market_exclusion_registry": _descriptor(prior_registry_path),
        "development_freeze_gate_passed": development_gate["passed"],
        "collection_ready": collection_ready,
        "collection_started": False,
        **_blocked_safety_fields(),
    }
    bundle_path = run_dir / "hierarchical_side_family_lcb_freeze_bundle_manifest.json"
    _write_json(bundle_path, bundle)
    return {
        "run_dir": run_dir,
        "development_freeze_manifest_path": development_freeze_path,
        "development_freeze_manifest_sha256": _sha256_file(development_freeze_path),
        "precollection_freeze_manifest_path": precollection_path,
        "precollection_freeze_manifest_sha256": _sha256_file(precollection_path),
        "bundle_manifest_path": bundle_path,
        "bundle_manifest_sha256": _sha256_file(bundle_path),
        "training_report": training_report,
        "precollection_manifest": precollection,
    }


def _validate_development_and_quarantine_rows(
    development_rows: list[dict[str, Any]],
    quarantined_rows: list[dict[str, Any]],
) -> None:
    counts = Counter(str(row.get("role") or "") for row in development_rows)
    if dict(counts) != DEVELOPMENT_ROLE_COUNTS:
        raise ValueError("development allowlist must contain exact 60/30 roles")
    if len(quarantined_rows) != QUARANTINED_MARKET_COUNT or any(
        str(row.get("role") or "") != QUARANTINED_ROLE for row in quarantined_rows
    ):
        raise ValueError("#173 confirmatory quarantine must contain 30 markets")


def _development_allowlist(
    *,
    run_id: str,
    role_manifest_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    records = [
        {
            "market_id": str(row["market_id"]),
            "role": str(row["role"]),
            "selection_rank": int(row["selection_rank"]),
            "source_corpus_dir": str(row["source_corpus_dir"]),
            "corpus_manifest": row["corpus_manifest"],
        }
        for row in rows
    ]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-development-source-allowlist-v1",
        "run_id": run_id,
        "source_role_assignment_manifest": _descriptor(role_manifest_path),
        "allowed_roles": list(DEVELOPMENT_ROLE_COUNTS),
        "market_count": len(records),
        "role_market_counts": dict(Counter(row["role"] for row in records)),
        "market_ids_sha256": canonical_json_sha256(sorted(row["market_id"] for row in records)),
        "records": records,
        "issue173_confirmatory_role_allowed": False,
        "uses_issue173_confirmatory_labels_for_tuning": False,
        **_blocked_safety_fields(),
    }


def _confirmatory_quarantine_registry(
    *,
    run_id: str,
    role_manifest_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    market_ids = sorted(str(row["market_id"]) for row in rows)
    corpus_dirs = sorted(str(row["source_corpus_dir"]) for row in rows)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-issue173-confirmatory-quarantine-v1",
        "run_id": run_id,
        "source_role_assignment_manifest": _descriptor(role_manifest_path),
        "quarantined_role": QUARANTINED_ROLE,
        "quarantined_market_count": len(market_ids),
        "quarantined_market_ids": market_ids,
        "quarantined_market_ids_sha256": canonical_json_sha256(market_ids),
        "quarantined_corpus_dirs": corpus_dirs,
        "forbidden_artifact_categories": [
            "confirmatory_action_rows",
            "confirmatory_predictions",
            "confirmatory_replay",
            "confirmatory_validation_report",
            "candidate_freeze_decision",
        ],
        "artifacts_opened_for_tuning": False,
        "labels_opened_for_tuning": False,
        **_blocked_safety_fields(),
    }


def _prior_market_registry(
    *,
    run_id: str,
    role_manifest: dict[str, Any],
    issue173_rows: list[dict[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    parent_descriptor = _verified_descriptor(
        role_manifest.get("prior_evidence_exclusion_registry"),
        name="#173 prior evidence exclusion registry",
    )
    parent = _load_json(Path(parent_descriptor["path"]))
    parent_ids = {str(value) for value in parent.get("prior_market_ids") or []}
    issue173_ids = {str(row["market_id"]) for row in issue173_rows}
    if not parent_ids or len(issue173_ids) != 120:
        raise ValueError("prior or #173 market registry is incomplete")
    market_ids = sorted(parent_ids | issue173_ids)
    if len(market_ids) != expected_count:
        raise ValueError(
            f"prior market registry count mismatch: {len(market_ids)} != {expected_count}"
        )
    max_decision_ts = max(
        int(parent.get("maximum_prior_decision_ts") or 0),
        max(int(row["maximum_decision_ts"]) for row in issue173_rows),
    )
    return {
        "schema_version": f"{SCHEMA_PREFIX}-prior-market-exclusion-registry-v1",
        "run_id": run_id,
        "parent_registry": parent_descriptor,
        "issue173_role_assignment_manifest": role_manifest.get("selected_rows"),
        "parent_market_count": len(parent_ids),
        "issue173_market_count": len(issue173_ids),
        "prior_unique_market_count": len(market_ids),
        "prior_market_ids": market_ids,
        "prior_market_ids_sha256": canonical_json_sha256(market_ids),
        "maximum_prior_decision_ts": max_decision_ts,
        "outcome_or_pnl_values_loaded": False,
        **_blocked_safety_fields(),
    }


def _hierarchical_lcb_artifact(
    calibration_predictions: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    feature_contract_sha256: str,
) -> dict[str, Any]:
    hierarchy = dict(protocol["hierarchical_expected_mean_lcb_protocol"])
    confidence = float(hierarchy["confidence_level"])
    samples = int(hierarchy["bootstrap_resample_count"])
    seed = int(hierarchy["bootstrap_seed"])
    min_family = int(hierarchy["minimum_unique_markets_per_family"])
    min_side = int(hierarchy["minimum_unique_markets_per_family_side"])
    min_leaf = int(hierarchy["minimum_unique_markets_per_leaf"])
    side_prior = int(hierarchy["family_side_shrinkage_prior_market_count"])
    leaf_prior = int(hierarchy["leaf_shrinkage_prior_market_count"])
    trade_rows = [
        row for row in calibration_predictions if str(row["action_family"]) in TRADE_FAMILIES
    ]
    families: dict[str, Any] = {}
    side_groups: dict[str, Any] = {}
    leaf_groups: dict[str, Any] = {}
    for family_index, family in enumerate(TRADE_FAMILIES):
        family_rows = [row for row in trade_rows if row["action_family"] == family]
        family_stats = _market_grouped_mean_residual_ci(
            family_rows,
            confidence_level=confidence,
            bootstrap_resample_count=samples,
            seed=seed + family_index * 100_000,
        )
        family_supported = family_stats["unique_market_count"] >= min_family
        families[family] = {
            "calibration_row_count": len(family_rows),
            "calibration_unique_market_count": family_stats["unique_market_count"],
            "minimum_required_unique_markets": min_family,
            "support_passed": family_supported,
            "mean_residual_upper_confidence_bound": family_stats[
                "mean_residual_upper_confidence_bound"
            ],
            "market_grouped_bootstrap": family_stats,
        }
        for side_index, side in enumerate(("UP", "DOWN")):
            rows = [row for row in family_rows if row["side"] == side]
            stats = _market_grouped_mean_residual_ci(
                rows,
                confidence_level=confidence,
                bootstrap_resample_count=samples,
                seed=seed + family_index * 100_000 + side_index * 10_000 + 1,
            )
            support = stats["unique_market_count"] >= min_side
            key = f"{family}|{side}"
            if support and family_supported:
                weight = stats["unique_market_count"] / (stats["unique_market_count"] + side_prior)
                penalty = weight * float(stats["mean_residual_upper_confidence_bound"]) + (
                    1.0 - weight
                ) * float(family_stats["mean_residual_upper_confidence_bound"])
                source = "family_side_shrunk_to_family"
            elif family_supported:
                weight = 0.0
                penalty = float(family_stats["mean_residual_upper_confidence_bound"])
                source = "unsupported_family_side_fallback_to_family"
            else:
                weight = 0.0
                penalty = None
                source = "unsupported_family_fail_closed_no_trade"
            side_groups[key] = {
                "action_family": family,
                "side": side,
                "calibration_row_count": len(rows),
                "calibration_unique_market_count": stats["unique_market_count"],
                "minimum_required_unique_markets": min_side,
                "support_passed": support,
                "family_support_passed": family_supported,
                "family_side_weight": weight,
                "mean_residual_upper_confidence_bound": stats[
                    "mean_residual_upper_confidence_bound"
                ],
                "hierarchical_penalty": penalty,
                "penalty_source": source,
                "market_grouped_bootstrap": stats,
            }
            regimes = sorted({_execution_regime(row, protocol) for row in rows})
            for regime_index, regime_name in enumerate(regimes):
                leaf_rows = [row for row in rows if _execution_regime(row, protocol) == regime_name]
                leaf_stats = _market_grouped_mean_residual_ci(
                    leaf_rows,
                    confidence_level=confidence,
                    bootstrap_resample_count=samples,
                    seed=(seed + family_index * 100_000 + side_index * 10_000 + regime_index + 100),
                )
                leaf_support = leaf_stats["unique_market_count"] >= min_leaf
                leaf_key = f"{family}|{side}|{regime_name}"
                if leaf_support and support and penalty is not None:
                    leaf_weight = leaf_stats["unique_market_count"] / (
                        leaf_stats["unique_market_count"] + leaf_prior
                    )
                    leaf_penalty = leaf_weight * float(
                        leaf_stats["mean_residual_upper_confidence_bound"]
                    ) + (1.0 - leaf_weight) * float(penalty)
                    leaf_source = "leaf_shrunk_to_supported_family_side"
                else:
                    leaf_weight = 0.0
                    leaf_penalty = penalty
                    leaf_source = (
                        "unsupported_leaf_fallback_to_family_side_or_family"
                        if penalty is not None
                        else "unsupported_family_fail_closed_no_trade"
                    )
                leaf_groups[leaf_key] = {
                    "action_family": family,
                    "side": side,
                    "execution_regime": regime_name,
                    "calibration_row_count": len(leaf_rows),
                    "calibration_unique_market_count": leaf_stats["unique_market_count"],
                    "minimum_required_unique_markets": min_leaf,
                    "support_passed": leaf_support,
                    "parent_family_side_support_passed": support,
                    "leaf_weight": leaf_weight,
                    "mean_residual_upper_confidence_bound": leaf_stats[
                        "mean_residual_upper_confidence_bound"
                    ],
                    "hierarchical_penalty": leaf_penalty,
                    "penalty_source": leaf_source,
                    "market_grouped_bootstrap": leaf_stats,
                }
    artifact = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-artifact-v1",
        "candidate_name": CANDIDATE_NAME,
        "source_split": "development_calibration_only",
        "estimand": "conditional_expected_cost_aware_net_return",
        "method": "market_grouped_bootstrap_hierarchical_mean_residual_lcb",
        "decision_score_formula": (
            "raw_family_expected_net_return - hierarchical_residual_upper_bound"
        ),
        "hierarchy": list(hierarchy["hierarchy"]),
        "execution_regime_definition": hierarchy["execution_regime_definition"],
        "confidence_level": confidence,
        "bootstrap_unit": "market_id",
        "bootstrap_resample_count": samples,
        "bootstrap_seed": seed,
        "families": families,
        "family_side_groups": side_groups,
        "leaf_groups": leaf_groups,
        "individual_outcome_quantile_subtraction_enabled": False,
        "affine_calibration_enabled": False,
        "forced_action_side_or_family_quota_enabled": False,
        "feature_contract_sha256": feature_contract_sha256,
        "uses_issue173_confirmatory_labels_for_tuning": False,
        "uses_future_confirmatory_labels_for_tuning": False,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def _execution_regime(row: dict[str, Any], protocol: dict[str, Any]) -> str:
    regime = protocol["hierarchical_expected_mean_lcb_protocol"]["execution_regime_definition"]
    features = dict(row["decision_time_features"])
    price = float(features["execution_price"])
    ttc = float(features["time_to_close_seconds"])
    spread = float(features["selected_side_spread_bps"])
    queue = float(features["selected_side_queue_fill_probability_proxy"])
    staleness = float(features["selected_side_book_staleness_ms"])
    price_low, price_high = [float(value) for value in regime["execution_price_boundaries"]]
    ttc_late, ttc_early = [float(value) for value in regime["time_to_close_boundaries_seconds"]]
    price_bucket = (
        "price_low" if price <= price_low else "price_mid" if price <= price_high else "price_high"
    )
    time_bucket = (
        "ttc_late" if ttc <= ttc_late else "ttc_middle" if ttc <= ttc_early else "ttc_early"
    )
    quality = (
        spread <= float(regime["quality_spread_bps_maximum"])
        and queue >= float(regime["quality_queue_fill_minimum"])
        and staleness <= float(regime["quality_book_staleness_ms_maximum"])
    )
    return f"{price_bucket}|{time_bucket}|{'quality_pass' if quality else 'quality_degraded'}"


def _apply_hierarchical_lcb_scores(
    predictions: list[dict[str, Any]],
    *,
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    protocol_stub = {
        "hierarchical_expected_mean_lcb_protocol": {
            "execution_regime_definition": artifact["execution_regime_definition"]
        }
    }
    for row in predictions:
        family = str(row["action_family"])
        raw_score = float(row["raw_family_expected_net_return"])
        if family == "NO_TRADE":
            regime_name = "none"
            penalty = 0.0
            source = "no_trade_zero_score"
            available = True
        else:
            side = str(row["side"])
            regime_name = _execution_regime(row, protocol_stub)
            leaf = artifact["leaf_groups"].get(f"{family}|{side}|{regime_name}")
            parent = artifact["family_side_groups"][f"{family}|{side}"]
            selected_group = leaf if leaf is not None else parent
            penalty_value = selected_group.get("hierarchical_penalty")
            available = penalty_value is not None
            penalty = float(penalty_value) if available else 1_000_000.0
            source = (
                str(selected_group["penalty_source"])
                if leaf is not None
                else "unseen_leaf_fallback_to_family_side_or_family"
            )
        updated = {
            **row,
            "hierarchical_execution_regime": regime_name,
            "hierarchical_residual_upper_confidence_bound": penalty,
            "hierarchical_penalty_source": source,
            "hierarchical_score_available": available,
            "hierarchical_expected_mean_lcb_net_return": raw_score - penalty,
            "ranking_score_source": (
                "development_calibration_hierarchical_expected_mean_residual_lcb"
            ),
            "forced_action_side_or_family_quota_enabled": False,
        }
        updated["prediction_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def _development_freeze_gate(
    *,
    protocol: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    hierarchy_artifact: dict[str, Any],
    corpus_audits: list[dict[str, Any]],
    quarantine_access_overlap: set[str],
) -> dict[str, Any]:
    gates = protocol["development_freeze_gates"]
    checks = {
        "train_market_support": sum(audit["role"] == "development_train" for audit in corpus_audits)
        == int(gates["required_train_market_count"]),
        "calibration_market_support": sum(
            audit["role"] == "development_calibration" for audit in corpus_audits
        )
        == int(gates["required_calibration_market_count"]),
        "accepted_bet_support": int(candidate_metrics["accepted_bet_count"])
        >= int(gates["minimum_accepted_bet_count"]),
        "accepted_unique_market_support": int(candidate_metrics["accepted_unique_market_count"])
        >= int(gates["minimum_accepted_unique_market_count"]),
        "candidate_net_pnl_positive": float(candidate_metrics["net_pnl_sum"]) > 0.0,
        "candidate_better_than_frozen_baseline": float(candidate_metrics["net_pnl_sum"])
        > float(baseline_metrics["net_pnl_sum"]),
        "family_parent_support": all(
            bool(group["support_passed"]) for group in hierarchy_artifact["families"].values()
        ),
        "zero_feature_causality_violations": all(
            int(audit["feature_causality_violation_count"]) == 0 for audit in corpus_audits
        ),
        "zero_forbidden_evidence_access_violations": not quarantine_access_overlap,
    }
    reason_map = {
        "train_market_support": "development_train_market_support_failed",
        "calibration_market_support": "development_calibration_market_support_failed",
        "accepted_bet_support": "development_accepted_bet_support_failed",
        "accepted_unique_market_support": "development_unique_market_support_failed",
        "candidate_net_pnl_positive": "development_candidate_net_pnl_not_positive",
        "candidate_better_than_frozen_baseline": "development_candidate_not_better_than_baseline",
        "family_parent_support": "development_family_parent_support_failed",
        "zero_feature_causality_violations": "development_feature_causality_violation",
        "zero_forbidden_evidence_access_violations": "issue173_confirmatory_evidence_access_violation",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    return {"passed": not reasons, "checks": checks, "reason_codes": reasons}


def _hierarchical_report_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": artifact["method"],
        "hierarchy": artifact["hierarchy"],
        "family_group_count": len(artifact["families"]),
        "family_side_group_count": len(artifact["family_side_groups"]),
        "leaf_group_count": len(artifact["leaf_groups"]),
        "supported_family_count": sum(
            bool(group["support_passed"]) for group in artifact["families"].values()
        ),
        "supported_family_side_count": sum(
            bool(group["support_passed"]) for group in artifact["family_side_groups"].values()
        ),
        "supported_leaf_count": sum(
            bool(group["support_passed"]) for group in artifact["leaf_groups"].values()
        ),
        "forced_action_side_or_family_quota_enabled": False,
    }


def _training_markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate_metrics"]
    baseline = report["baseline_metrics"]
    return (
        "\n".join(
            [
                "# #174 Hierarchical Side/Family LCB Development Freeze",
                "",
                f"- Gate passed: `{str(report['development_freeze_gate_passed']).lower()}`",
                f"- Candidate accepted bets: `{candidate['accepted_bet_count']}`",
                f"- Candidate net PnL: `{candidate['net_pnl_sum']}`",
                f"- Baseline net PnL: `{baseline['net_pnl_sum']}`",
                f"- Candidate minus baseline: `{report['candidate_minus_baseline_net_pnl']}`",
                f"- Blocking reasons: `{json.dumps(report['development_freeze_blocking_reason_codes'])}`",
                "- #173 confirmatory artifacts opened for tuning: `false`",
                "- Future confirmatory labels used for tuning: `false`",
                "- Forced quotas enabled: `false`",
                "- Paper/live/promotion unlock: `false`",
            ]
        )
        + "\n"
    )


def _verified_descriptor(payload: Any, *, name: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(payload.get("path") or "")).expanduser().resolve()
    sha256 = str(payload.get("sha256") or "").lower()
    _require_sha256(sha256, name=f"{name} SHA-256")
    _verify_pin(path, sha256, name=name)
    return {"path": str(path), "sha256": sha256}
