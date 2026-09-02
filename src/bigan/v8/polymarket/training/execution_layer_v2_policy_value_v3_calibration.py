"""Run the pre-registered #200 independent policy-value v3 calibration gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _adaptive_bucket,
    _market_bootstrap_interval,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    FORBIDDEN_OUTCOME_FIELDS,
    _load_outcome_blind_feature_rows,
    _materialize_outcome_blind_action_rows,
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    _descriptor,
    _find_fields,
    _load_json,
    _load_jsonl,
    _materialize_role_action_rows,
    _predict_role_rows,
    _require_sha256,
    _verified_descriptor,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-policy-value-lcb-v3-calibration-gate-profile-v1"
SCHEMA_PREFIX = "bigan-v8-policy-value-lcb-v3-calibration"
EVALUATION_ROLE = "development_calibration"
TARGET_FIELDS = frozenset(
    {
        *FORBIDDEN_OUTCOME_FIELDS,
        "accepted_bet_net_pnl",
        "evaluation_target_net_pnl_per_contract",
        "training_target_absolute_post_cost_net_return",
        "training_target_advantage_vs_no_trade",
        "training_target_advantage_vs_best_alternative",
    }
)


@dataclass(frozen=True, slots=True)
class PolicyValueV3CalibrationConfig:
    """Pinned inputs for the one-shot #200 development calibration gate."""

    run_id: str
    output_dir: Path | str
    gate_profile_path: Path | str
    expected_gate_profile_sha256: str
    issue198_candidate_manifest_path: Path | str
    expected_issue198_candidate_manifest_sha256: str
    issue199_manifest_path: Path | str
    expected_issue199_manifest_sha256: str
    role_assignment_manifest_path: Path | str
    expected_role_assignment_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in (
            "expected_gate_profile_sha256",
            "expected_issue198_candidate_manifest_sha256",
            "expected_issue199_manifest_sha256",
            "expected_role_assignment_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "gate_profile_path",
            "issue198_candidate_manifest_path",
            "issue199_manifest_path",
            "role_assignment_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_policy_value_v3_calibration_profile(profile: dict[str, Any]) -> None:
    """Reject any drift from the frozen #200 selection and evidence contract."""

    selector = dict(profile.get("selector") or {})
    execution = dict(profile.get("execution") or {})
    gate = dict(profile.get("development_gate") or {})
    access = dict(profile.get("access_sequence") or {})
    mutation = dict(profile.get("mutation_contract") or {})
    safety = dict(profile.get("safety") or {})
    hash_fields = (
        "parent_issue_198_candidate_manifest_sha256",
        "parent_issue_199_manifest_sha256",
        "model_sha256",
        "calibration_artifact_sha256",
        "feature_contract_sha256",
        "role_assignment_manifest_sha256",
        "role_assignment_rows_sha256",
    )
    checks = {
        "schema_version": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "candidate_name": profile.get("candidate_name")
        == "policy_value_lcb_v3_two_safety_estimand_selector",
        "frozen": profile.get("frozen") is True,
        "parent_commit": _is_sha1(str(profile.get("parent_issue_199_commit") or "")),
        "hashes": all(_is_sha256(str(profile.get(key) or "")) for key in hash_fields),
        "role": profile.get("evaluation_role") == EVALUATION_ROLE,
        "coverage": profile.get("expected_market_count") == 45
        and profile.get("expected_decision_group_count") == 180
        and profile.get("expected_action_row_count") == 900,
        "two_safety_selector": selector.get("method")
        == "frozen_pairwise_ranker_with_two_safety_estimand_lcb_filter"
        and selector.get("absolute_post_cost_net_return_lcb_minimum") == 0.0
        and selector.get("advantage_vs_no_trade_lcb_minimum_exclusive") == 0.0
        and selector.get("advantage_vs_best_alternative_used_for_selection") is False
        and selector.get("advantage_vs_best_alternative_role") == "diagnostic_only_oracle_regret"
        and selector.get("no_trade_score") == 0.0
        and selector.get("entry_threshold") == 0.0
        and selector.get("runner_up_advantage_threshold") == 0.0
        and selector.get("model_or_score_mutation_allowed") is False,
        "execution_frozen": execution.get("guard_config_mutation_allowed") is False
        and execution.get("exposure_policy_mutation_allowed") is False
        and execution.get("sizing_mutation_allowed") is False,
        "support_gate": gate.get("minimum_guard_accepted_bet_count") == 10
        and gate.get("minimum_guard_accepted_unique_market_count") == 10
        and gate.get("minimum_action_support_for_action_pnl_gate") == 5
        and gate.get("accepted_bet_total_pnl_minimum_exclusive") == 0.0
        and gate.get("all_market_policy_pnl_lcb_minimum_exclusive") == 0.0
        and gate.get("bootstrap_unit") == "market_id"
        and gate.get("bootstrap_resample_count") == 2000
        and gate.get("bootstrap_confidence_level") == 0.95
        and gate.get("support_floor_is_promotion_power_claim") is False
        and gate.get("issue_191_future_required_accepted_unique_market_count") == 88,
        "access_order": access.get("pre_label_audit_required") is True
        and access.get("feature_only_prediction_before_label_access_required") is True
        and access.get("target_stripped_decision_freeze_before_label_access_required") is True
        and access.get("development_calibration_labels_may_be_opened_once_after_freeze") is True
        and all(
            access.get(key) is False
            for key in (
                "confirmatory_validation_labels_may_be_opened",
                "issue_189_validation_files_may_be_opened",
                "issue_190_or_192_future_labels_may_be_opened",
                "future_accepted_bet_pnl_may_be_opened",
            )
        ),
        "no_mutation": all(value is False for value in mutation.values()),
        "safety": safety == _blocked_safety_fields(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid policy-value v3 calibration profile: {', '.join(failed)}")


def run_policy_value_v3_calibration_gate(
    config: PolicyValueV3CalibrationConfig,
) -> dict[str, Any]:
    """Freeze outcome-blind decisions, then evaluate the sealed calibration split once."""

    profile_path = config.gate_profile_path.resolve()
    candidate_path = config.issue198_candidate_manifest_path.resolve()
    issue199_path = config.issue199_manifest_path.resolve()
    role_path = config.role_assignment_manifest_path.resolve()
    pins = {
        "profile": (profile_path, config.expected_gate_profile_sha256),
        "issue198_candidate": (
            candidate_path,
            config.expected_issue198_candidate_manifest_sha256,
        ),
        "issue199_manifest": (issue199_path, config.expected_issue199_manifest_sha256),
        "role_assignment_manifest": (
            role_path,
            config.expected_role_assignment_manifest_sha256,
        ),
    }
    for name, (path, expected) in pins.items():
        _verify_pin(path, expected, name=name)
    profile = _load_json(profile_path)
    validate_policy_value_v3_calibration_profile(profile)
    if profile["parent_issue_198_candidate_manifest_sha256"] != pins["issue198_candidate"][1]:
        raise ValueError("#198 candidate lineage mismatch")
    if profile["parent_issue_199_manifest_sha256"] != pins["issue199_manifest"][1]:
        raise ValueError("#199 audit lineage mismatch")
    if profile["role_assignment_manifest_sha256"] != pins["role_assignment_manifest"][1]:
        raise ValueError("role assignment lineage mismatch")

    candidate = _load_json(candidate_path)
    issue199_manifest = _load_json(issue199_path)
    role_manifest = _load_json(role_path)
    source = _validate_parent_manifests(
        candidate=candidate,
        issue199_manifest=issue199_manifest,
        role_manifest=role_manifest,
        profile=profile,
    )
    role_rows = _load_jsonl(Path(source["role_assignment_rows"]["path"]))
    if _find_fields({"rows": role_rows}, set(TARGET_FIELDS)):
        raise ValueError("role assignment rows contain forbidden target fields")
    calibration_role_rows = [row for row in role_rows if row.get("role") == EVALUATION_ROLE]
    if len(calibration_role_rows) != int(profile["expected_market_count"]):
        raise ValueError("development calibration role coverage mismatch")

    run_dir = Path(config.output_dir) / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    pre_label = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-label-access-audit-v1",
        "run_id": config.run_id,
        "gate_profile": _descriptor(profile_path),
        "issue198_candidate_manifest": _descriptor(candidate_path),
        "issue199_manifest": _descriptor(issue199_path),
        "role_assignment_manifest": _descriptor(role_path),
        "role_assignment_rows": source["role_assignment_rows"],
        "evaluation_role": EVALUATION_ROLE,
        "expected_market_count": 45,
        "target_label_resolution_or_pnl_files_opened_before_audit": False,
        "confirmatory_or_future_files_opened": False,
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
    }
    pre_label["audit_id"] = canonical_json_sha256(pre_label)
    pre_label_path = run_dir / "policy_value_v3_pre_label_access_audit.json"
    _write_json_fsync(pre_label_path, pre_label)
    _write_text_fsync(
        run_dir / "policy_value_v3_pre_label_access_audit.md",
        _pre_label_markdown(pre_label),
    )

    feature_contract = _load_json(Path(source["feature_contract"]["path"]))
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=feature_contract["parent_protocol_sha256"],
    )
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    feature_action_rows: list[dict[str, Any]] = []
    opened_feature_descriptors: list[dict[str, Any]] = []
    for role_row in calibration_role_rows:
        feature_rows, descriptor = _load_outcome_blind_feature_rows(role_row)
        opened_feature_descriptors.append(descriptor)
        feature_action_rows.extend(
            _materialize_outcome_blind_action_rows(
                feature_rows,
                role_row=role_row,
                feature_columns=feature_columns,
            )
        )
    feature_action_rows.sort(
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"]), str(row["action"]))
    )
    _validate_outcome_blind_action_rows(feature_action_rows, profile=profile)
    booster = xgb.Booster()
    booster.load_model(source["model"]["path"])
    predictions = _predict_role_rows(
        feature_action_rows,
        booster=booster,
        feature_columns=feature_columns,
    )
    calibration = _load_json(Path(source["calibration_artifact"]["path"]))
    scored = apply_policy_value_v3_scores(predictions, calibration=calibration, profile=profile)
    stripped = [strip_policy_value_v3_targets(row) for row in scored]
    validate_policy_value_v3_target_stripped_rows(stripped)
    replay_rows = _outcome_blind_acceptance_replay(
        stripped,
        entry_threshold=float(profile["selector"]["entry_threshold"]),
        runner_up_advantage_threshold=float(profile["selector"]["runner_up_advantage_threshold"]),
    )
    predictions_path = run_dir / "policy_value_v3_target_stripped_predictions.jsonl"
    replay_path = run_dir / "policy_value_v3_outcome_blind_guard_replay.jsonl"
    _write_jsonl_fsync(predictions_path, stripped)
    _write_jsonl_fsync(replay_path, replay_rows)
    target_fields_in_decision_artifacts = sorted(
        _find_fields(
            {"predictions": stripped, "guard_replay": replay_rows},
            set(TARGET_FIELDS),
        )
    )
    if target_fields_in_decision_artifacts:
        raise ValueError("decision artifacts contain forbidden target fields")
    decision_freeze = _decision_freeze_manifest(
        run_id=config.run_id,
        profile_path=profile_path,
        source=source,
        pre_label_path=pre_label_path,
        predictions_path=predictions_path,
        replay_path=replay_path,
        opened_feature_descriptors=opened_feature_descriptors,
        scored=scored,
        replay_rows=replay_rows,
    )
    decision_freeze_path = run_dir / "policy_value_v3_decision_freeze_manifest.json"
    _write_json_fsync(decision_freeze_path, decision_freeze)

    # Label-bearing artifacts may only be opened after the fsynced decision freeze above.
    action_rows_by_role, corpus_audits = _materialize_role_action_rows(
        role_rows,
        feature_columns=feature_columns,
        roles=(EVALUATION_ROLE,),
    )
    target_rows = action_rows_by_role[EVALUATION_ROLE]
    evaluation_rows, join_report = build_policy_value_v3_evaluation_rows(
        replay_rows,
        target_rows=target_rows,
    )
    evaluation_path = run_dir / "policy_value_v3_development_calibration_evaluation_rows.jsonl"
    _write_jsonl(evaluation_path, evaluation_rows)
    gate_report = build_policy_value_v3_gate_report(
        run_id=config.run_id,
        profile=profile,
        decision_freeze=decision_freeze,
        decision_freeze_path=decision_freeze_path,
        evaluation_rows=evaluation_rows,
        join_report=join_report,
        corpus_audits=corpus_audits,
        calibration_market_ids=[str(row["market_id"]) for row in calibration_role_rows],
    )
    report_path = run_dir / "policy_value_v3_development_calibration_gate_report.json"
    report_md_path = run_dir / "policy_value_v3_development_calibration_gate_report.md"
    _write_json(report_path, gate_report)
    _write_text(report_md_path, _gate_report_markdown(gate_report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-gate-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": profile["candidate_name"],
        "gate_profile": _descriptor(profile_path),
        "pre_label_access_audit": _descriptor(pre_label_path),
        "target_stripped_predictions": _descriptor(predictions_path),
        "outcome_blind_guard_replay": _descriptor(replay_path),
        "decision_freeze_manifest": _descriptor(decision_freeze_path),
        "development_calibration_evaluation_rows": _descriptor(evaluation_path),
        "development_calibration_gate_report": _descriptor(report_path),
        "development_calibration_gate_report_markdown": _descriptor(report_md_path),
        "development_calibration_labels_opened_after_decision_freeze": True,
        "development_calibration_gate_passed": gate_report["development_calibration_gate_passed"],
        "candidate_specific_confirmatory_evaluation_allowed": gate_report[
            "candidate_specific_confirmatory_evaluation_allowed"
        ],
        "gate_blocking_reason_codes": gate_report["gate_blocking_reason_codes"],
        "confirmatory_validation_labels_opened": False,
        "issue_190_or_192_future_labels_opened": False,
        "current_oof_validation_or_future_pnl_used_for_tuning": False,
        "selector_threshold_bucket_guard_cost_or_sizing_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "policy_value_v3_calibration_gate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
        "gate_report": gate_report,
    }


def apply_policy_value_v3_scores(
    predictions: list[dict[str, Any]],
    *,
    calibration: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the pre-registered two-safety-estimand rule without target access."""

    selector = dict(profile["selector"])
    output: list[dict[str, Any]] = []
    for row in predictions:
        action = str(row["action"])
        boundaries = list(calibration["actions"][action]["adaptive_score_boundaries"])
        bucket = _adaptive_bucket(float(row["pairwise_group_normalized_rank_score"]), boundaries)
        group = calibration["calibration_groups"][f"{action}|{bucket}"]
        estimators = dict(group["estimators"])
        points = {
            name: float(estimators[name]["point_estimate"])
            for name in (
                "absolute_post_cost_net_return",
                "advantage_vs_no_trade",
                "advantage_vs_best_alternative",
            )
        }
        lcbs = {name: float(estimators[name]["lower_confidence_bound"]) for name in points}
        two_safety_passed = bool(
            action != "NO_TRADE"
            and lcbs["absolute_post_cost_net_return"]
            >= float(selector["absolute_post_cost_net_return_lcb_minimum"])
            and lcbs["advantage_vs_no_trade"]
            > float(selector["advantage_vs_no_trade_lcb_minimum_exclusive"])
        )
        if action == "NO_TRADE":
            score = float(selector["no_trade_score"])
            source = "frozen_no_trade_zero_anchor"
        elif two_safety_passed:
            score = lcbs["absolute_post_cost_net_return"]
            source = "two_safety_estimand_lcbs_passed"
        else:
            score = min(
                0.0,
                lcbs["absolute_post_cost_net_return"],
                lcbs["advantage_vs_no_trade"],
            )
            source = "one_or_more_two_safety_estimand_lcbs_failed"
        updated = {
            **row,
            "policy_value_v3_bucket": bucket,
            "policy_value_v3_estimand_point_estimates": points,
            "policy_value_v3_estimand_lower_confidence_bounds": lcbs,
            "policy_value_v3_two_safety_estimands_passed": two_safety_passed,
            "policy_value_v3_oracle_comparator_diagnostic_only": True,
            "policy_value_v3_score": score,
            "policy_value_v3_score_source": source,
            "calibrated_action_expected_net_return": points["absolute_post_cost_net_return"],
            "action_advantage_lcb_net_return": score,
            "action_advantage_lcb_score_bucket": bucket,
            "action_advantage_lcb_estimate_source": source,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        updated["policy_value_v3_scored_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def strip_policy_value_v3_targets(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only inference/guard fields and the v3 diagnostic score contract."""

    forbidden = _find_fields(row, set(TARGET_FIELDS))
    if forbidden:
        raise ValueError("feature-only v3 row unexpectedly contains target fields")
    stripped = dict(row)
    stripped["training_target_fields_stripped"] = True
    stripped["target_or_outcome_fields_used"] = False
    stripped["target_stripped_row_sha256"] = canonical_json_sha256(stripped)
    return stripped


def validate_policy_value_v3_target_stripped_rows(rows: list[dict[str, Any]]) -> None:
    """Assert target-free complete decision grids before guard replay."""

    found = _find_fields({"rows": rows}, set(TARGET_FIELDS))
    if found:
        raise ValueError("target-stripped v3 rows contain forbidden target fields")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    if not groups or any(actions != set(REQUIRED_ACTIONS) for actions in groups.values()):
        raise ValueError("target-stripped v3 action grids are incomplete")
    if any(row.get("target_used_as_decision_input") is not False for row in rows):
        raise ValueError("target influence flag is not fail-closed")


def build_policy_value_v3_evaluation_rows(
    replay_rows: list[dict[str, Any]],
    *,
    target_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join frozen guard decisions to one cost-aware label per executed action."""

    target_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    duplicate_keys: list[tuple[str, int, str]] = []
    for row in target_rows:
        key = (str(row["market_id"]), int(row["decision_ts"]), str(row["action"]))
        if key in target_by_key:
            duplicate_keys.append(key)
        target_by_key[key] = row
    output: list[dict[str, Any]] = []
    missing: list[tuple[str, int, str]] = []
    accepted_join_count = 0
    for row in replay_rows:
        accepted = bool(row["execution_guard_order_allowed"])
        key = (
            str(row["market_id"]),
            int(row["decision_ts"]),
            str(row["executed_action"]),
        )
        target_row = target_by_key.get(key) if accepted else None
        if accepted and target_row is None:
            missing.append(key)
        target = float(target_row["target_net_pnl_per_contract"]) if target_row else None
        size = float(row["proposed_order_size"]) if accepted else 0.0
        accepted_pnl = float(target * size) if target is not None else 0.0
        accepted_join_count += int(target_row is not None)
        evaluation = {
            "decision_index": row["decision_index"],
            "market_id": row["market_id"],
            "decision_ts": row["decision_ts"],
            "source_selected_action": row["source_selected_action"],
            "executed_action": row["executed_action"],
            "selected_side": row["selected_side"],
            "selected_action_family": row["selected_action_family"],
            "execution_guard_order_allowed": accepted,
            "proposed_order_size": size,
            "evaluation_target_net_pnl_per_contract": target,
            "accepted_bet_net_pnl": accepted_pnl,
            "execution_blocking_reason_codes": row["execution_blocking_reason_codes"],
            "decision_freeze_row_sha256": row["viability_row_sha256"],
            "target_used_as_decision_input": False,
            "outcome_used_for_evaluation_only": True,
            "paper_only": True,
            "capital_at_risk": False,
        }
        evaluation["evaluation_row_sha256"] = canonical_json_sha256(evaluation)
        output.append(evaluation)
    return output, {
        "target_row_count": len(target_rows),
        "target_key_count": len(target_by_key),
        "duplicate_target_key_count": len(duplicate_keys),
        "duplicate_target_keys_sha256": canonical_json_sha256(duplicate_keys),
        "guard_accepted_count": sum(
            int(row["execution_guard_order_allowed"]) for row in replay_rows
        ),
        "accepted_target_join_count": accepted_join_count,
        "missing_accepted_target_count": len(missing),
        "missing_accepted_target_keys_sha256": canonical_json_sha256(missing),
        "accepted_target_join_reconciled": not duplicate_keys
        and not missing
        and accepted_join_count
        == sum(int(row["execution_guard_order_allowed"]) for row in replay_rows),
    }


def build_policy_value_v3_gate_report(
    *,
    run_id: str,
    profile: dict[str, Any],
    decision_freeze: dict[str, Any],
    decision_freeze_path: Path,
    evaluation_rows: list[dict[str, Any]],
    join_report: dict[str, Any],
    corpus_audits: list[dict[str, Any]],
    calibration_market_ids: list[str],
) -> dict[str, Any]:
    """Apply the one-shot pre-registered development calibration gate."""

    gate = dict(profile["development_gate"])
    accepted = [row for row in evaluation_rows if row["execution_guard_order_allowed"]]
    accepted_markets = {str(row["market_id"]) for row in accepted}
    pnl_by_market = dict.fromkeys(sorted(set(calibration_market_ids)), 0.0)
    for row in accepted:
        pnl_by_market[str(row["market_id"])] += float(row["accepted_bet_net_pnl"])
    interval = _market_bootstrap_interval(
        list(pnl_by_market.values()),
        resample_count=int(gate["bootstrap_resample_count"]),
        confidence_level=float(gate["bootstrap_confidence_level"]),
        seed=int(gate["bootstrap_seed"]),
    )
    action_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        action_rows[str(row["executed_action"])].append(row)
    action_pnl_gate: dict[str, Any] = {}
    for action, rows in sorted(action_rows.items()):
        unique_markets = len({str(row["market_id"]) for row in rows})
        pnl = float(sum(float(row["accepted_bet_net_pnl"]) for row in rows))
        applicable = unique_markets >= int(gate["minimum_action_support_for_action_pnl_gate"])
        action_pnl_gate[action] = {
            "accepted_bet_count": len(rows),
            "accepted_unique_market_count": unique_markets,
            "accepted_bet_net_pnl_sum": pnl,
            "gate_applicable": applicable,
            "gate_passed": (pnl > 0.0) if applicable else True,
        }
    feature_causality_violations = sum(
        int(audit["feature_causality_violation_count"]) for audit in corpus_audits
    )
    cost_violations = sum(int(audit["cost_component_violation_count"]) for audit in corpus_audits)
    total_pnl = float(sum(float(row["accepted_bet_net_pnl"]) for row in accepted))
    checks = {
        "decision_freeze_written_before_label_access": decision_freeze.get(
            "development_calibration_labels_opened"
        )
        is False
        and decision_freeze_path.is_file(),
        "coverage_complete": len(calibration_market_ids) == 45 and len(evaluation_rows) == 180,
        "feature_causality_passed": feature_causality_violations == 0,
        "cost_contract_passed": cost_violations == 0,
        "target_join_reconciled": join_report["accepted_target_join_reconciled"] is True,
        "minimum_guard_accepted_bet_support": len(accepted)
        >= int(gate["minimum_guard_accepted_bet_count"]),
        "minimum_guard_accepted_market_support": len(accepted_markets)
        >= int(gate["minimum_guard_accepted_unique_market_count"]),
        "accepted_bet_total_pnl_positive": total_pnl
        > float(gate["accepted_bet_total_pnl_minimum_exclusive"]),
        "all_market_policy_pnl_lcb_positive": interval["lower_confidence_bound"]
        > float(gate["all_market_policy_pnl_lcb_minimum_exclusive"]),
        "supported_action_pnl_gates_passed": all(
            row["gate_passed"] for row in action_pnl_gate.values()
        ),
        "safety_flags_blocked": True,
    }
    reason_map = {
        "decision_freeze_written_before_label_access": "decision_freeze_order_invalid",
        "coverage_complete": "development_calibration_coverage_incomplete",
        "feature_causality_passed": "feature_causality_violation",
        "cost_contract_passed": "cost_contract_violation",
        "target_join_reconciled": "accepted_target_join_not_reconciled",
        "minimum_guard_accepted_bet_support": "insufficient_guard_accepted_bet_support",
        "minimum_guard_accepted_market_support": (
            "insufficient_guard_accepted_unique_market_support"
        ),
        "accepted_bet_total_pnl_positive": "accepted_bet_total_pnl_not_positive",
        "all_market_policy_pnl_lcb_positive": "all_market_policy_pnl_lcb_not_positive",
        "supported_action_pnl_gates_passed": "supported_action_pnl_gate_failed",
        "safety_flags_blocked": "safety_contract_failed",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    passed = not blockers
    guard_intersection_reasons: list[str] = []
    candidate_count = int(decision_freeze["two_safety_selector_trade_decision_count"])
    if candidate_count > 0 and not accepted:
        guard_intersection_reasons.append("zero_guard_accepted_bets")
    if (
        candidate_count > 0
        and int(decision_freeze["trade_candidate_p_up_disagreement_count"]) == candidate_count
    ):
        guard_intersection_reasons.append("all_trade_candidates_p_up_side_disagreement")
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-gate-report-v1",
        "run_id": run_id,
        "candidate_name": profile["candidate_name"],
        "development_calibration_market_count": len(set(calibration_market_ids)),
        "development_calibration_decision_count": len(evaluation_rows),
        "two_safety_selector_candidate_count": decision_freeze[
            "two_safety_selector_trade_decision_count"
        ],
        "candidate_action_distribution": decision_freeze["source_selected_action_distribution"],
        "candidate_side_distribution": decision_freeze["source_selected_side_distribution"],
        "candidate_p_up_disagreement_count": decision_freeze[
            "trade_candidate_p_up_disagreement_count"
        ],
        "guard_evaluated_count": decision_freeze["guard_evaluated_count"],
        "guard_accepted_bet_count": len(accepted),
        "guard_accepted_unique_market_count": len(accepted_markets),
        "guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in evaluation_rows
                    if not row["execution_guard_order_allowed"]
                    for reason in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
        "guard_intersection_reason_codes": guard_intersection_reasons,
        "accepted_bet_pnl_evaluation_status": (
            "evaluated_guard_accepted_bets" if accepted else "no_guard_accepted_bets_no_pnl_sample"
        ),
        "accepted_bet_net_pnl_sum": total_pnl,
        "accepted_bet_win_rate": (
            sum(float(row["accepted_bet_net_pnl"]) > 0.0 for row in accepted) / len(accepted)
            if accepted
            else 0.0
        ),
        "all_calibration_market_policy_pnl": interval,
        "accepted_action_distribution": dict(
            sorted(Counter(str(row["executed_action"]) for row in accepted).items())
        ),
        "accepted_side_distribution": dict(
            sorted(Counter(str(row["selected_side"]) for row in accepted).items())
        ),
        "accepted_family_distribution": dict(
            sorted(Counter(str(row["selected_action_family"]) for row in accepted).items())
        ),
        "action_pnl_gate_results": action_pnl_gate,
        "target_join_report": join_report,
        "feature_causality_violation_count": feature_causality_violations,
        "cost_component_violation_count": cost_violations,
        "gate_checks": checks,
        "gate_blocking_reason_codes": blockers,
        "development_calibration_gate_passed": passed,
        "candidate_specific_confirmatory_evaluation_allowed": passed,
        "support_floor_is_promotion_power_claim": False,
        "future_required_accepted_unique_market_count": 88,
        "calibration_labels_used_for_fitting_or_threshold_selection": False,
        "calibration_labels_used_for_one_shot_evaluation_only": True,
        "development_calibration_labels_opened_after_decision_freeze": True,
        "confirmatory_validation_labels_opened": False,
        "issue_190_or_192_future_labels_opened": False,
        "oracle_best_comparator_used_for_selection": False,
        "selector_threshold_bucket_guard_cost_or_sizing_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _decision_freeze_manifest(
    *,
    run_id: str,
    profile_path: Path,
    source: dict[str, Any],
    pre_label_path: Path,
    predictions_path: Path,
    replay_path: Path,
    opened_feature_descriptors: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_trade_decisions = sum(
        row["source_selected_action"] != "NO_TRADE" for row in replay_rows
    )
    trade_candidates = [row for row in replay_rows if row["source_selected_action"] != "NO_TRADE"]
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-decision-freeze-manifest-v1",
        "run_id": run_id,
        "candidate_name": "policy_value_lcb_v3_two_safety_estimand_selector",
        "gate_profile": _descriptor(profile_path),
        "model": source["model"],
        "calibration_artifact": source["calibration_artifact"],
        "feature_contract": source["feature_contract"],
        "role_assignment_rows": source["role_assignment_rows"],
        "pre_label_access_audit": _descriptor(pre_label_path),
        "opened_feature_artifacts": opened_feature_descriptors,
        "target_stripped_predictions": _descriptor(predictions_path),
        "outcome_blind_guard_replay": _descriptor(replay_path),
        "action_row_count": len(scored),
        "decision_group_count": len(replay_rows),
        "two_safety_lcb_passed_action_row_count": sum(
            row["policy_value_v3_two_safety_estimands_passed"] for row in scored
        ),
        "two_safety_selector_trade_decision_count": selected_trade_decisions,
        "source_selected_action_distribution": dict(
            sorted(Counter(str(row["source_selected_action"]) for row in replay_rows).items())
        ),
        "source_selected_side_distribution": dict(
            sorted(Counter(str(row["source_selected_side"]) for row in trade_candidates).items())
        ),
        "trade_candidate_p_up_disagreement_count": sum(
            bool(row["p_up_action_disagreement"]) for row in trade_candidates
        ),
        "guard_evaluated_count": sum(row["execution_guard_evaluated"] for row in replay_rows),
        "guard_allowed_count": sum(row["execution_guard_order_allowed"] for row in replay_rows),
        "target_or_outcome_field_count": 0,
        "development_calibration_labels_opened": False,
        "decision_freeze_fsynced_before_label_access": True,
        "oracle_best_comparator_used_for_selection": False,
        "selector_threshold_bucket_guard_cost_or_sizing_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["decision_freeze_id"] = canonical_json_sha256(manifest)
    return manifest


def _validate_parent_manifests(
    *,
    candidate: dict[str, Any],
    issue199_manifest: dict[str, Any],
    role_manifest: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    if candidate.get("research_candidate_frozen") is not True:
        raise ValueError("#198 research model is not frozen")
    if candidate.get("current_oof_validation_or_confirmatory_pnl_used_for_tuning") is not False:
        raise ValueError("#198 candidate tuning lineage is unsafe")
    if issue199_manifest.get("oracle_best_comparator_hard_gate_recommendation") != (
        "diagnostic_only_for_ranking_regret_not_source_eligibility_hard_gate"
    ):
        raise ValueError("#199 estimand recommendation mismatch")
    if issue199_manifest.get("current_oof_validation_or_future_pnl_used_for_tuning") is not False:
        raise ValueError("#199 audit tuning lineage is unsafe")
    if role_manifest.get("role_assignment_ready") is not True:
        raise ValueError("role assignment is not ready")
    if role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        raise ValueError("role assignment opened outcomes")
    model = _verified_descriptor(candidate.get("model"), name="#198 model")
    calibration = _verified_descriptor(
        candidate.get("direct_action_advantage_calibration_artifact"),
        name="#198 calibration artifact",
    )
    feature_contract = _verified_descriptor(
        candidate.get("feature_contract"),
        name="#198 feature contract",
    )
    role_rows = _verified_descriptor(
        role_manifest.get("selected_rows"),
        name="role assignment rows",
    )
    expected = {
        "model": profile["model_sha256"],
        "calibration_artifact": profile["calibration_artifact_sha256"],
        "feature_contract": profile["feature_contract_sha256"],
        "role_assignment_rows": profile["role_assignment_rows_sha256"],
    }
    actual = {
        "model": model["sha256"],
        "calibration_artifact": calibration["sha256"],
        "feature_contract": feature_contract["sha256"],
        "role_assignment_rows": role_rows["sha256"],
    }
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise ValueError("frozen source descriptor mismatch: " + ", ".join(mismatches))
    return {
        "model": model,
        "calibration_artifact": calibration,
        "feature_contract": feature_contract,
        "role_assignment_rows": role_rows,
    }


def _validate_outcome_blind_action_rows(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> None:
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    checks = {
        "row_count": len(rows) == int(profile["expected_action_row_count"]),
        "group_count": len(groups) == int(profile["expected_decision_group_count"]),
        "market_count": len({str(row["market_id"]) for row in rows})
        == int(profile["expected_market_count"]),
        "complete_grids": all(actions == set(REQUIRED_ACTIONS) for actions in groups.values()),
        "target_free": not _find_fields({"rows": rows}, set(TARGET_FIELDS)),
        "causal": all(int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("outcome-blind action row validation failed: " + ", ".join(failed))


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "paper_candidate_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _is_sha1(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_json_fsync(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl_fsync(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_text_fsync(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content.rstrip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pre_label_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# Policy-value v3 Pre-label Access Audit",
            "",
            f"- audit id: `{report['audit_id']}`",
            "- evaluation role: `development_calibration`",
            "- target/label/PnL files opened before audit: `false`",
            "- confirmatory/future files opened: `false`",
            "- validation passed: `true`",
        )
    )


def _gate_report_markdown(report: dict[str, Any]) -> str:
    interval = report["all_calibration_market_policy_pnl"]
    lines = [
        "# Policy-value v3 Independent Development Calibration Gate",
        "",
        f"- markets: `{report['development_calibration_market_count']}`",
        f"- decisions: `{report['development_calibration_decision_count']}`",
        f"- two-safety candidates: `{report['two_safety_selector_candidate_count']}`",
        f"- guard evaluated: `{report['guard_evaluated_count']}`",
        f"- guard accepted bets: `{report['guard_accepted_bet_count']}`",
        f"- guard accepted unique markets: `{report['guard_accepted_unique_market_count']}`",
        f"- accepted-bet net PnL sum: `{report['accepted_bet_net_pnl_sum']:.9f}`",
        f"- all-market mean PnL: `{interval['point_estimate']:.9f}`",
        f"- all-market 95% LCB: `{interval['lower_confidence_bound']:.9f}`",
        f"- development gate passed: "
        f"`{str(report['development_calibration_gate_passed']).lower()}`",
        f"- confirmatory evaluation allowed: "
        f"`{str(report['candidate_specific_confirmatory_evaluation_allowed']).lower()}`",
        "",
        "## Blocking reasons",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in report["gate_blocking_reason_codes"])
    if not report["gate_blocking_reason_codes"]:
        lines.append("- none")
    lines.extend(
        (
            "",
            "This is a development screening gate, not promotion evidence. Confirmatory and "
            "future labels remain sealed.",
        )
    )
    return "\n".join(lines)
