"""Audit #198 direct-advantage estimands without fitting or tuning a candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-direct-advantage-estimand-audit-profile-v1"
REPORT_SCHEMA_PREFIX = "bigan-v8-direct-advantage-estimand-audit"
OOF_FILENAME = "direct_advantage_v2_internal_train_oof_predictions.jsonl"
CALIBRATION_FILENAME = "direct_action_advantage_v2_calibration_artifact.json"
TRAINING_REPORT_FILENAME = "direct_advantage_v2_training_report.json"
PRE_LABEL_AUDIT_FILENAME = "pre_label_access_lineage_audit.json"
ESTIMANDS = (
    "absolute_post_cost_net_return",
    "advantage_vs_no_trade",
    "advantage_vs_best_alternative",
)
TARGET_FIELDS = {estimand: f"training_target_{estimand}" for estimand in ESTIMANDS}


@dataclass(frozen=True, slots=True)
class DirectAdvantageEstimandAuditConfig:
    """Pinned inputs for the #199 development-only diagnostic."""

    run_id: str
    output_dir: Path | str
    source_run_dir: Path | str
    audit_profile_path: Path | str
    expected_audit_profile_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_audit_profile_sha256,
            name="expected_audit_profile_sha256",
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "source_run_dir", Path(self.source_run_dir))
        object.__setattr__(self, "audit_profile_path", Path(self.audit_profile_path))


def validate_direct_advantage_estimand_audit_profile(profile: dict[str, Any]) -> None:
    """Fail closed when the pre-registered audit scope drifts."""

    evidence = dict(profile.get("evidence_scope") or {})
    mutation = dict(profile.get("mutation_contract") or {})
    bootstrap = dict(profile.get("bootstrap") or {})
    power = dict(profile.get("prospective_power") or {})
    selector = dict(profile.get("selector_contract") or {})
    safety = dict(profile.get("safety") or {})
    hash_fields = (
        "parent_protocol_sha256",
        "parent_fit_profile_sha256",
        "source_calibration_artifact_sha256",
        "source_internal_oof_predictions_sha256",
        "source_training_report_sha256",
        "source_pre_label_audit_sha256",
    )
    checks = {
        "schema_version": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "frozen": profile.get("frozen") is True,
        "parent_commit": _is_sha1(str(profile.get("parent_commit") or "")),
        "parent_hashes": all(_is_sha256(str(profile.get(key) or "")) for key in hash_fields),
        "development_train_only": profile.get("allowed_role") == "development_train",
        "oof_market_count": profile.get("expected_oof_market_count") == 75,
        "decision_group_count": profile.get("expected_decision_group_count") == 300,
        "action_row_count": profile.get("expected_action_row_count") == 1500,
        "required_actions": profile.get("required_actions") == list(REQUIRED_ACTIONS),
        "estimands": profile.get("estimand_gate_order") == list(ESTIMANDS),
        "zero_thresholds": profile.get("estimand_thresholds") == dict.fromkeys(ESTIMANDS, 0.0),
        "strict_comparisons": profile.get("strict_comparison_estimands")
        == ["advantage_vs_no_trade", "advantage_vs_best_alternative"],
        "market_bootstrap": bootstrap.get("unit") == "market_id"
        and int(bootstrap.get("resample_count") or 0) >= 100
        and float(bootstrap.get("confidence_level") or 0.0) == 0.95,
        "power_is_report_only": power.get("used_for_threshold_or_model_tuning") is False,
        "selector_frozen": selector.get("selector_or_score_mutation_allowed") is False,
        "pre_label_audit_required": evidence.get(
            "development_train_targets_may_be_opened_after_pre_label_audit"
        )
        is True,
        "quarantined_evidence_sealed": all(
            evidence.get(key) is False
            for key in (
                "development_calibration_files_may_be_opened",
                "confirmatory_files_may_be_opened",
                "issue_189_oof_or_validation_files_may_be_opened",
                "issue_190_or_192_future_labels_may_be_opened",
                "accepted_bet_pnl_may_be_opened",
                "future_window_may_be_consumed",
            )
        ),
        "no_mutation_or_fit": all(
            mutation.get(key) is False
            for key in (
                "hyperparameter_search_enabled",
                "threshold_mutation_allowed",
                "bucket_mutation_allowed",
                "cost_model_mutation_allowed",
                "execution_guard_mutation_allowed",
                "sizing_mutation_allowed",
                "new_candidate_fit_allowed",
            )
        ),
        "safety": safety == _blocked_safety_fields(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid frozen estimand audit profile: {', '.join(failed)}")


def run_direct_advantage_estimand_audit(
    config: DirectAdvantageEstimandAuditConfig,
) -> dict[str, Any]:
    """Run the frozen #199 audit and return artifact descriptors."""

    profile_sha256 = _sha256_file(config.audit_profile_path)
    if profile_sha256 != config.expected_audit_profile_sha256:
        raise ValueError("audit profile SHA-256 mismatch")
    profile = _load_json(config.audit_profile_path)
    validate_direct_advantage_estimand_audit_profile(profile)

    run_dir = Path(config.output_dir) / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    source_paths = {
        "internal_oof_predictions": Path(config.source_run_dir) / OOF_FILENAME,
        "calibration_artifact": Path(config.source_run_dir) / CALIBRATION_FILENAME,
        "training_report": Path(config.source_run_dir) / TRAINING_REPORT_FILENAME,
        "source_pre_label_audit": Path(config.source_run_dir) / PRE_LABEL_AUDIT_FILENAME,
    }
    pre_label_audit = {
        "schema_version": f"{REPORT_SCHEMA_PREFIX}-pre-label-access-audit-v1",
        "run_id": config.run_id,
        "audit_profile_path": str(Path(config.audit_profile_path).resolve()),
        "audit_profile_sha256": profile_sha256,
        "source_run_dir": str(Path(config.source_run_dir).resolve()),
        "intended_source_paths": {name: str(path.resolve()) for name, path in source_paths.items()},
        "expected_source_sha256": {
            "internal_oof_predictions": profile["source_internal_oof_predictions_sha256"],
            "calibration_artifact": profile["source_calibration_artifact_sha256"],
            "training_report": profile["source_training_report_sha256"],
            "source_pre_label_audit": profile["source_pre_label_audit_sha256"],
        },
        "allowed_role": "development_train",
        "target_bearing_files_opened_before_audit": False,
        "label_resolution_or_pnl_files_opened_before_audit": False,
        "development_calibration_confirmatory_or_future_files_opened": False,
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
    }
    pre_label_audit["audit_id"] = canonical_json_sha256(pre_label_audit)
    pre_label_path = run_dir / "pre_label_access_lineage_audit.json"
    _write_json_fsync(pre_label_path, pre_label_audit)
    _write_text_fsync(
        run_dir / "pre_label_access_lineage_audit.md",
        _pre_label_markdown(pre_label_audit),
    )

    _verify_source_hashes(source_paths, profile)
    source_training_report = _load_json(source_paths["training_report"])
    source_pre_label_audit = _load_json(source_paths["source_pre_label_audit"])
    calibration = _load_json(source_paths["calibration_artifact"])
    oof_rows = _load_jsonl(source_paths["internal_oof_predictions"])
    _validate_source_lineage(
        oof_rows,
        calibration=calibration,
        training_report=source_training_report,
        source_pre_label_audit=source_pre_label_audit,
        profile=profile,
    )

    semantics = build_estimand_semantics_audit(
        oof_rows,
        run_id=config.run_id,
        profile_sha256=profile_sha256,
    )
    attrition = build_gate_attrition_report(
        oof_rows,
        calibration=calibration,
        profile=profile,
        run_id=config.run_id,
    )
    policy = build_selected_policy_value_report(
        oof_rows,
        calibration=calibration,
        profile=profile,
        run_id=config.run_id,
    )
    power = build_support_power_decomposition_report(
        oof_rows,
        calibration=calibration,
        profile=profile,
        attrition=attrition,
        policy=policy,
        semantics=semantics,
        run_id=config.run_id,
    )

    report_specs = (
        (
            "estimand_semantics_audit",
            "direct_advantage_v2_estimand_semantics_audit",
            semantics,
            _semantics_markdown,
        ),
        (
            "gate_attrition_report",
            "direct_advantage_v2_gate_attrition_report",
            attrition,
            _attrition_markdown,
        ),
        (
            "selected_policy_value_report",
            "direct_advantage_v2_selected_policy_value_report",
            policy,
            _policy_markdown,
        ),
        (
            "support_power_decomposition_report",
            "direct_advantage_v2_support_power_decomposition_report",
            power,
            _power_markdown,
        ),
    )
    artifacts: dict[str, Any] = {}
    for key, stem, payload, markdown_builder in report_specs:
        json_path = run_dir / f"{stem}.json"
        md_path = run_dir / f"{stem}.md"
        _write_json(json_path, payload)
        _write_text(md_path, markdown_builder(payload))
        artifacts[key] = {
            "json": _descriptor(json_path),
            "markdown": _descriptor(md_path),
        }

    manifest = {
        "schema_version": f"{REPORT_SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "issue": 199,
        "audit_mode": "development_train_only_diagnostic_fail_closed",
        "audit_profile": _descriptor(config.audit_profile_path),
        "pre_label_access_audit": _descriptor(pre_label_path),
        "source_artifacts": {name: _descriptor(path) for name, path in source_paths.items()},
        "artifacts": artifacts,
        "oracle_best_comparator_hard_gate_recommendation": power[
            "oracle_best_comparator_hard_gate_recommendation"
        ],
        "next_candidate_pre_registration_allowed": power["next_candidate_pre_registration_allowed"],
        "future_window_consumed": False,
        "current_oof_validation_or_future_pnl_used_for_tuning": False,
        "threshold_guard_cost_bucket_or_sizing_mutated": False,
        "new_candidate_fitted": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "direct_advantage_v2_estimand_audit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "pre_label_audit_path": pre_label_path,
        "reports": artifacts,
        "manifest": manifest,
    }


def build_estimand_semantics_audit(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    profile_sha256: str,
) -> dict[str, Any]:
    """Prove the row/group semantics of the ex-post oracle comparator."""

    groups = _decision_groups(rows)
    identity_violation_count = 0
    non_oracle_positive_advantage_violation_count = 0
    positive_absolute_but_nonpositive_oracle_advantage_count = 0
    positive_oracle_advantage_row_count = 0
    oracle_tie_group_count = 0
    group_rows: list[dict[str, Any]] = []
    for (market_id, decision_ts), group in groups.items():
        returns = {str(row["action"]): float(row["target_net_pnl_per_contract"]) for row in group}
        ordered = sorted(returns.items(), key=lambda item: (-item[1], item[0]))
        top_return = ordered[0][1]
        oracle_actions = sorted(action for action, value in returns.items() if value == top_return)
        second_return = max(
            value
            for action, value in returns.items()
            if action not in oracle_actions or len(oracle_actions) > 1
        )
        if len(oracle_actions) > 1:
            oracle_tie_group_count += 1
            second_return = top_return
        for row in group:
            action = str(row["action"])
            absolute = float(row[TARGET_FIELDS["absolute_post_cost_net_return"]])
            no_trade_advantage = float(row[TARGET_FIELDS["advantage_vs_no_trade"]])
            oracle_advantage = float(row[TARGET_FIELDS["advantage_vs_best_alternative"]])
            best_alternative = max(
                value for candidate, value in returns.items() if candidate != action
            )
            identity_violation_count += int(
                not math.isclose(absolute, returns[action], abs_tol=1e-12)
                or not math.isclose(no_trade_advantage, absolute, abs_tol=1e-12)
                or not math.isclose(
                    oracle_advantage,
                    absolute - best_alternative,
                    abs_tol=1e-12,
                )
            )
            if oracle_advantage > 0.0:
                positive_oracle_advantage_row_count += 1
                non_oracle_positive_advantage_violation_count += int(action not in oracle_actions)
            positive_absolute_but_nonpositive_oracle_advantage_count += int(
                absolute > 0.0 and oracle_advantage <= 0.0
            )
        group_rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "oracle_actions": oracle_actions,
                "oracle_best_return": top_return,
                "oracle_second_best_return": second_return,
                "oracle_margin": top_return - second_return,
            }
        )
    if identity_violation_count or non_oracle_positive_advantage_violation_count:
        raise ValueError("oracle comparator estimand identity validation failed")
    report = {
        "schema_version": f"{REPORT_SCHEMA_PREFIX}-semantics-report-v1",
        "run_id": run_id,
        "audit_profile_sha256": profile_sha256,
        "decision_group_count": len(groups),
        "action_row_count": len(rows),
        "estimand_identity_violation_count": identity_violation_count,
        "oracle_tie_group_count": oracle_tie_group_count,
        "positive_oracle_advantage_row_count": positive_oracle_advantage_row_count,
        "non_oracle_positive_advantage_violation_count": (
            non_oracle_positive_advantage_violation_count
        ),
        "positive_absolute_but_nonpositive_oracle_advantage_count": (
            positive_absolute_but_nonpositive_oracle_advantage_count
        ),
        "oracle_best_comparator_uses_ex_post_counterfactual_returns": True,
        "oracle_best_comparator_is_decision_time_available": False,
        "oracle_best_advantage_is_necessary_for_positive_post_cost_value": False,
        "oracle_best_advantage_semantic_role": (
            "ranking_regret_diagnostic_not_standalone_source_eligibility_hard_gate"
        ),
        "mathematical_explanation": (
            "For action a in a decision group, advantage_vs_best_alternative equals "
            "return(a) - max(return(other actions)). A positive value proves that a is the "
            "ex-post oracle winner for that row; it is not necessary for return(a) to be "
            "positive versus NO_TRADE. Requiring its LCB above zero therefore combines "
            "action-value safety with oracle-ranking perfection."
        ),
        "decision_group_oracle_rows": group_rows,
        "development_train_targets_used_for_report_only": True,
        "targets_used_as_decision_inputs": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def build_gate_attrition_report(
    rows: list[dict[str, Any]],
    *,
    calibration: dict[str, Any],
    profile: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Attribute every trade row to its first frozen LCB gate failure."""

    diagnostics: list[dict[str, Any]] = []
    first_failure = Counter()
    pass_counts = Counter()
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["action"] == "NO_TRADE":
            continue
        state = _row_gate_state(row, calibration=calibration, profile=profile)
        first_failure[state["first_failing_estimand"]] += 1
        by_action[str(row["action"])][state["first_failing_estimand"]] += 1
        by_bucket[f"{row['action']}|{state['bucket_name']}"][state["first_failing_estimand"]] += 1
        for estimand, passed in state["estimand_passed"].items():
            pass_counts[f"{estimand}_passed"] += int(passed)
        pass_counts["two_safety_estimands_passed"] += int(
            state["estimand_passed"]["absolute_post_cost_net_return"]
            and state["estimand_passed"]["advantage_vs_no_trade"]
        )
        pass_counts["all_three_estimands_passed"] += int(state["all_estimands_passed"])
        diagnostics.append(
            {
                "market_id": row["market_id"],
                "decision_ts": row["decision_ts"],
                "fold_index": row["fold_index"],
                "action": row["action"],
                "action_family": row["action_family"],
                "side": row["side"],
                "pairwise_group_normalized_rank_score": row["pairwise_group_normalized_rank_score"],
                **state,
            }
        )
    report = {
        "schema_version": f"{REPORT_SCHEMA_PREFIX}-gate-attrition-report-v1",
        "run_id": run_id,
        "trade_action_row_count": len(diagnostics),
        "first_failing_estimand_distribution": dict(sorted(first_failure.items())),
        "gate_pass_counts": dict(sorted(pass_counts.items())),
        "first_failure_by_action": {
            action: dict(sorted(counts.items())) for action, counts in sorted(by_action.items())
        },
        "first_failure_by_action_bucket": {
            bucket: dict(sorted(counts.items())) for bucket, counts in sorted(by_bucket.items())
        },
        "row_attrition": diagnostics,
        "all_trade_rows_reconciled": len(diagnostics)
        == sum(first_failure.values())
        == len(rows) - len(rows) // len(REQUIRED_ACTIONS),
        "thresholds_mutated": False,
        "calibration_buckets_mutated": False,
        "targets_used_for_gate_replay": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def build_selected_policy_value_report(
    rows: list[dict[str, Any]],
    *,
    calibration: dict[str, Any],
    profile: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Evaluate frozen OOF selectors at policy level using report-only targets."""

    groups = _decision_groups(rows)
    variants = {
        "raw_pairwise_selector": lambda row: True,
        "strict_all_three_lcb_selector": lambda row: (
            row["action"] == "NO_TRADE"
            or _row_gate_state(row, calibration=calibration, profile=profile)[
                "all_estimands_passed"
            ]
        ),
        "two_safety_estimand_selector_diagnostic_only": lambda row: (
            row["action"] == "NO_TRADE"
            or _two_safety_estimands_pass(row, calibration=calibration, profile=profile)
        ),
    }
    variant_reports: dict[str, Any] = {}
    bootstrap = dict(profile["bootstrap"])
    for variant_index, (name, allowed) in enumerate(variants.items()):
        selected_rows: list[dict[str, Any]] = []
        for (market_id, decision_ts), group in groups.items():
            candidates = [row for row in group if allowed(row)]
            if not candidates:
                raise ValueError(f"selector {name} has no NO_TRADE fallback")
            selected = _select_highest_frozen_score(candidates)
            oracle_return = max(float(row["target_net_pnl_per_contract"]) for row in group)
            selected_return = float(selected["target_net_pnl_per_contract"])
            selected_rows.append(
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "selected_action": selected["action"],
                    "selected_side": selected["side"],
                    "selected_action_family": selected["action_family"],
                    "selected_frozen_rank_score": selected["pairwise_group_normalized_rank_score"],
                    "selected_post_cost_net_return": selected_return,
                    "oracle_post_cost_net_return": oracle_return,
                    "oracle_regret": oracle_return - selected_return,
                    "oracle_action_hit": math.isclose(
                        selected_return,
                        oracle_return,
                        abs_tol=1e-12,
                    ),
                    "target_used_for_selection": False,
                }
            )
        variant_reports[name] = _summarize_selected_policy(
            selected_rows,
            bootstrap_resample_count=int(bootstrap["resample_count"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=int(bootstrap["seed"]) + variant_index * 10_000,
        )
        variant_reports[name]["calibration_and_evaluation_share_oof_targets"] = name != (
            "raw_pairwise_selector"
        )
        variant_reports[name]["eligible_as_unbiased_candidate_evidence"] = False
    report = {
        "schema_version": f"{REPORT_SCHEMA_PREFIX}-selected-policy-value-report-v1",
        "run_id": run_id,
        "decision_group_count": len(groups),
        "selector_source": "frozen_198_internal_cross_fit_predictions",
        "selector_or_score_mutated": False,
        "policy_variants": variant_reports,
        "targets_used_for_selection": False,
        "development_train_targets_used_for_report_only": True,
        "two_safety_selector_same_sample_calibration_reuse": True,
        "independent_calibration_or_nested_cross_fit_required_before_candidate_claim": True,
        "validation_confirmatory_or_future_labels_used": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def build_support_power_decomposition_report(
    rows: list[dict[str, Any]],
    *,
    calibration: dict[str, Any],
    profile: dict[str, Any],
    attrition: dict[str, Any],
    policy: dict[str, Any],
    semantics: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Separate negative edge, confidence width, ranking, and comparator semantics."""

    group_power: list[dict[str, Any]] = []
    score_target_correlations: dict[str, dict[str, float | None]] = {}
    for action in REQUIRED_ACTIONS:
        action_rows = [row for row in rows if row["action"] == action]
        score_target_correlations[action] = {
            estimand: _finite_correlation(
                [float(row["pairwise_group_normalized_rank_score"]) for row in action_rows],
                [float(row[TARGET_FIELDS[estimand]]) for row in action_rows],
            )
            for estimand in ESTIMANDS
        }
        boundaries = list(calibration["actions"][action]["adaptive_score_boundaries"])
        for bucket_name in calibration["actions"][action]["adaptive_bucket_names"]:
            bucket_rows = [
                row
                for row in action_rows
                if _adaptive_bucket(
                    float(row["pairwise_group_normalized_rank_score"]),
                    boundaries,
                )
                == bucket_name
            ]
            estimand_power = {}
            for estimand in ESTIMANDS:
                values = _market_target_means(bucket_rows, TARGET_FIELDS[estimand])
                estimand_power[estimand] = _prospective_support_estimate(
                    list(values.values()),
                    confidence_level=float(profile["bootstrap"]["confidence_level"]),
                    target_power=float(profile["prospective_power"]["target_power"]),
                    maximum_market_count=int(
                        profile["prospective_power"]["maximum_reported_market_count"]
                    ),
                )
            calibration_group = calibration["calibration_groups"][f"{action}|{bucket_name}"]
            lcb_classification = {
                estimand: _lcb_root_cause(
                    float(calibration_group["estimators"][estimand]["point_estimate"]),
                    float(calibration_group["estimators"][estimand]["lower_confidence_bound"]),
                )
                for estimand in ESTIMANDS
            }
            group_power.append(
                {
                    "action": action,
                    "bucket_name": bucket_name,
                    "row_count": len(bucket_rows),
                    "unique_market_count": len({row["market_id"] for row in bucket_rows}),
                    "lcb_root_cause_by_estimand": lcb_classification,
                    "prospective_support_by_estimand": estimand_power,
                }
            )

    first_failures = dict(attrition["first_failing_estimand_distribution"])
    oracle_failure_count = int(first_failures.get("advantage_vs_best_alternative", 0))
    two_safety_count = int(attrition["gate_pass_counts"].get("two_safety_estimands_passed", 0))
    raw_policy = policy["policy_variants"]["raw_pairwise_selector"]
    raw_policy_positive_point = raw_policy["market_level_post_cost_return"]["point_estimate"] > 0.0
    raw_policy_positive_lcb = (
        raw_policy["market_level_post_cost_return"]["lower_confidence_bound"] > 0.0
    )
    two_safety_policy = policy["policy_variants"]["two_safety_estimand_selector_diagnostic_only"]
    two_safety_policy_positive_lcb = (
        two_safety_policy["market_level_post_cost_return"]["lower_confidence_bound"] > 0.0
        and two_safety_policy["selected_trade_decision_count"] > 0
    )
    recommendation = (
        "diagnostic_only_for_ranking_regret_not_source_eligibility_hard_gate"
        if oracle_failure_count > 0
        and two_safety_count > 0
        and semantics["oracle_best_advantage_is_necessary_for_positive_post_cost_value"] is False
        else "retain_pending_additional_semantic_evidence"
    )
    if two_safety_policy_positive_lcb:
        next_action = (
            "pre_register_nested_cross_fitted_or_independent_calibration_policy_value_v3_"
            "with_oracle_regret_diagnostic_only"
        )
    elif raw_policy_positive_lcb:
        next_action = "pre_register_policy_value_lcb_v3_without_opening_future_labels"
    elif raw_policy_positive_point:
        next_action = "increase_development_support_before_pre_registering_v3_evaluation"
    else:
        next_action = "redesign_decision_time_model_features_or_objective_before_v3"
    report = {
        "schema_version": f"{REPORT_SCHEMA_PREFIX}-support-power-report-v1",
        "run_id": run_id,
        "score_target_correlation_by_action": score_target_correlations,
        "action_bucket_support_power": group_power,
        "first_failing_estimand_distribution": first_failures,
        "two_safety_estimand_passed_action_row_count": two_safety_count,
        "all_three_estimand_passed_action_row_count": attrition["gate_pass_counts"].get(
            "all_three_estimands_passed",
            0,
        ),
        "oracle_comparator_first_failure_count": oracle_failure_count,
        "raw_selector_policy_value_point_estimate_positive": raw_policy_positive_point,
        "raw_selector_policy_value_lcb_positive": raw_policy_positive_lcb,
        "two_safety_selector_policy_value_lcb_positive": two_safety_policy_positive_lcb,
        "two_safety_selector_same_sample_calibration_reuse": True,
        "two_safety_selector_is_unbiased_candidate_evidence": False,
        "independent_calibration_or_nested_cross_fit_required": True,
        "oracle_best_comparator_hard_gate_recommendation": recommendation,
        "recommended_next_research_action": next_action,
        "next_candidate_pre_registration_allowed": False,
        "next_candidate_fit_allowed_in_this_issue": False,
        "future_evaluation_allowed_in_this_issue": False,
        "diagnostic_does_not_relax_existing_198_gate": True,
        "current_oof_validation_or_future_pnl_used_for_tuning": False,
        "development_train_targets_used_for_report_only": True,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _row_gate_state(
    row: dict[str, Any],
    *,
    calibration: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    action = str(row["action"])
    boundaries = list(calibration["actions"][action]["adaptive_score_boundaries"])
    bucket = _adaptive_bucket(float(row["pairwise_group_normalized_rank_score"]), boundaries)
    group = calibration["calibration_groups"][f"{action}|{bucket}"]
    lcbs = {
        estimand: float(group["estimators"][estimand]["lower_confidence_bound"])
        for estimand in ESTIMANDS
    }
    points = {
        estimand: float(group["estimators"][estimand]["point_estimate"]) for estimand in ESTIMANDS
    }
    thresholds = dict(profile["estimand_thresholds"])
    passed = {
        "absolute_post_cost_net_return": lcbs["absolute_post_cost_net_return"]
        >= float(thresholds["absolute_post_cost_net_return"]),
        "advantage_vs_no_trade": lcbs["advantage_vs_no_trade"]
        > float(thresholds["advantage_vs_no_trade"]),
        "advantage_vs_best_alternative": lcbs["advantage_vs_best_alternative"]
        > float(thresholds["advantage_vs_best_alternative"]),
    }
    first_failing = "all_estimands_passed"
    for estimand in ESTIMANDS:
        if not passed[estimand]:
            first_failing = estimand
            break
    return {
        "bucket_name": bucket,
        "estimand_point_estimates": points,
        "estimand_lower_confidence_bounds": lcbs,
        "estimand_passed": passed,
        "all_estimands_passed": all(passed.values()),
        "first_failing_estimand": first_failing,
    }


def _two_safety_estimands_pass(
    row: dict[str, Any],
    *,
    calibration: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    state = _row_gate_state(row, calibration=calibration, profile=profile)
    return bool(
        state["estimand_passed"]["absolute_post_cost_net_return"]
        and state["estimand_passed"]["advantage_vs_no_trade"]
    )


def _select_highest_frozen_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_priority = {action: -index for index, action in enumerate(REQUIRED_ACTIONS)}
    return max(
        rows,
        key=lambda row: (
            float(row["pairwise_group_normalized_rank_score"]),
            action_priority[str(row["action"])],
        ),
    )


def _summarize_selected_policy(
    selected_rows: list[dict[str, Any]],
    *,
    bootstrap_resample_count: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    by_market: dict[str, list[float]] = defaultdict(list)
    for row in selected_rows:
        by_market[str(row["market_id"])].append(float(row["selected_post_cost_net_return"]))
    market_means = {market: float(np.mean(values)) for market, values in by_market.items()}
    interval = _market_bootstrap_interval(
        list(market_means.values()),
        resample_count=bootstrap_resample_count,
        confidence_level=confidence_level,
        seed=seed,
    )
    action_distribution = Counter(str(row["selected_action"]) for row in selected_rows)
    return {
        "selected_decision_count": len(selected_rows),
        "selected_trade_decision_count": sum(
            action != "NO_TRADE" for action in action_distribution.elements()
        ),
        "selected_action_distribution": dict(sorted(action_distribution.items())),
        "decision_level_post_cost_return_sum": float(
            sum(float(row["selected_post_cost_net_return"]) for row in selected_rows)
        ),
        "decision_level_post_cost_return_mean": float(
            np.mean([float(row["selected_post_cost_net_return"]) for row in selected_rows])
        ),
        "market_level_post_cost_return": interval,
        "oracle_action_hit_rate": float(
            np.mean([row["oracle_action_hit"] for row in selected_rows])
        ),
        "mean_oracle_regret": float(np.mean([row["oracle_regret"] for row in selected_rows])),
        "selected_rows": selected_rows,
    }


def _market_bootstrap_interval(
    values: list[float],
    *,
    resample_count: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("market bootstrap values must be finite and non-empty")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resample_count, len(array)), replace=True).mean(axis=1)
    alpha = 1.0 - confidence_level
    return {
        "point_estimate": float(array.mean()),
        "lower_confidence_bound": float(np.quantile(samples, alpha, method="linear")),
        "upper_confidence_bound": float(np.quantile(samples, confidence_level, method="linear")),
        "market_count": len(values),
        "market_mean_standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "bootstrap_unit": "market_id",
        "bootstrap_resample_count": resample_count,
        "confidence_level": confidence_level,
        "bootstrap_seed": seed,
    }


def _prospective_support_estimate(
    values: list[float],
    *,
    confidence_level: float,
    target_power: float,
    maximum_market_count: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean()) if len(array) else 0.0
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if mean <= 0.0:
        status = "not_estimable_from_nonpositive_development_mean"
        lcb_n = None
        power_n = None
    elif std == 0.0:
        status = "positive_zero_variance"
        lcb_n = 1
        power_n = 1
    else:
        z_confidence = NormalDist().inv_cdf(confidence_level)
        z_power = NormalDist().inv_cdf(target_power)
        lcb_n = math.ceil((z_confidence * std / mean) ** 2)
        power_n = math.ceil(((z_confidence + z_power) * std / mean) ** 2)
        status = "estimated" if power_n <= maximum_market_count else "exceeds_reported_maximum"
        if power_n > maximum_market_count:
            power_n = None
        if lcb_n > maximum_market_count:
            lcb_n = None
    return {
        "observed_market_count": len(values),
        "observed_market_mean": mean,
        "observed_market_standard_deviation": std,
        "minimum_market_count_for_expected_positive_lcb": lcb_n,
        "minimum_market_count_for_target_power": power_n,
        "target_power": target_power,
        "confidence_level": confidence_level,
        "status": status,
        "estimate_is_prospective_report_only": True,
    }


def _lcb_root_cause(point_estimate: float, lower_confidence_bound: float) -> str:
    if lower_confidence_bound > 0.0:
        return "positive_supported"
    if point_estimate > 0.0:
        return "positive_point_estimate_insufficient_support_or_variance"
    return "nonpositive_point_estimate"


def _finite_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def _adaptive_bucket(score: float, boundaries: list[float]) -> str:
    for index, boundary in enumerate(boundaries):
        if score <= float(boundary):
            return f"bucket_{index}"
    return f"bucket_{len(boundaries)}"


def _market_target_means(rows: list[dict[str, Any]], target_field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["market_id"])].append(float(row[target_field]))
    return {market: float(np.mean(values)) for market, values in grouped.items()}


def _decision_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    return dict(sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])))


def _validate_source_lineage(
    rows: list[dict[str, Any]],
    *,
    calibration: dict[str, Any],
    training_report: dict[str, Any],
    source_pre_label_audit: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    groups = _decision_groups(rows)
    markets = {str(row["market_id"]) for row in rows}
    checks = {
        "row_count": len(rows) == int(profile["expected_action_row_count"]),
        "market_count": len(markets) == int(profile["expected_oof_market_count"]),
        "group_count": len(groups) == int(profile["expected_decision_group_count"]),
        "complete_groups": all(
            len(group) == len(REQUIRED_ACTIONS)
            and {str(row["action"]) for row in group} == set(REQUIRED_ACTIONS)
            for group in groups.values()
        ),
        "required_fields": all(
            {
                "market_id",
                "decision_ts",
                "fold_index",
                "action",
                "action_family",
                "side",
                "pairwise_group_normalized_rank_score",
                "target_net_pnl_per_contract",
                *TARGET_FIELDS.values(),
            }.issubset(row)
            for row in rows
        ),
        "targets_cost_aware": all(
            row.get("training_targets_include_costs") is True for row in rows
        ),
        "targets_not_inputs": all(
            row.get("training_targets_used_as_decision_inputs") is False for row in rows
        ),
        "calibration_source": calibration.get("source")
        == "new_internal_development_train_oof_predictions_only",
        "training_scope": training_report.get("fit_market_count") == 90
        and training_report.get("new_internal_oof_market_count") == 75
        and training_report.get("current_oof_validation_or_confirmatory_pnl_used_for_tuning")
        is False
        and training_report.get("development_calibration_confirmatory_or_future_files_opened")
        is False,
        "source_pre_label_audit": source_pre_label_audit.get("pre_label_access_validation_passed")
        is True
        and source_pre_label_audit.get("label_or_resolution_files_opened_before_audit") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"source lineage validation failed: {', '.join(failed)}")


def _verify_source_hashes(paths: dict[str, Path], profile: dict[str, Any]) -> None:
    expected = {
        "internal_oof_predictions": profile["source_internal_oof_predictions_sha256"],
        "calibration_artifact": profile["source_calibration_artifact_sha256"],
        "training_report": profile["source_training_report_sha256"],
        "source_pre_label_audit": profile["source_pre_label_audit_sha256"],
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen source artifact: {path}")
        if _sha256_file(path) != expected[name]:
            raise ValueError(f"frozen source artifact SHA-256 mismatch: {name}")


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


def _descriptor(path: Path | str) -> dict[str, Any]:
    file_path = Path(path).resolve()
    return {"path": str(file_path), "sha256": _sha256_file(file_path)}


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_sha1(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: str, *, name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
            "# #199 Pre-label Access Audit",
            "",
            f"- audit id: `{report['audit_id']}`",
            "- allowed role: `development_train`",
            "- target-bearing files opened before audit: `false`",
            "- calibration / confirmatory / future files opened: `false`",
            "- pre-label access validation passed: `true`",
        )
    )


def _semantics_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# Direct Advantage v2 Estimand Semantics Audit",
            "",
            f"- decision groups: `{report['decision_group_count']}`",
            f"- action rows: `{report['action_row_count']}`",
            f"- identity violations: `{report['estimand_identity_violation_count']}`",
            "- oracle comparator decision-time available: `false`",
            f"- positive absolute but nonpositive oracle advantage rows: "
            f"`{report['positive_absolute_but_nonpositive_oracle_advantage_count']}`",
            f"- recommended semantic role: `{report['oracle_best_advantage_semantic_role']}`",
            "",
            report["mathematical_explanation"],
        )
    )


def _attrition_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Direct Advantage v2 Gate Attrition",
        "",
        f"- trade action rows: `{report['trade_action_row_count']}`",
        f"- all rows reconciled: `{str(report['all_trade_rows_reconciled']).lower()}`",
        "",
        "## First failure",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{count}`"
        for name, count in report["first_failing_estimand_distribution"].items()
    )
    return "\n".join(lines)


def _policy_markdown(report: dict[str, Any]) -> str:
    lines = ["# Direct Advantage v2 Selected Policy Value", ""]
    for name, metrics in report["policy_variants"].items():
        interval = metrics["market_level_post_cost_return"]
        lines.extend(
            (
                f"## {name}",
                "",
                f"- selected trades: `{metrics['selected_trade_decision_count']}`",
                f"- market-level mean: `{interval['point_estimate']:.9f}`",
                f"- market-level 95% LCB: `{interval['lower_confidence_bound']:.9f}`",
                f"- oracle hit rate: `{metrics['oracle_action_hit_rate']:.6f}`",
                f"- mean oracle regret: `{metrics['mean_oracle_regret']:.9f}`",
                "",
            )
        )
    return "\n".join(lines)


def _power_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# Direct Advantage v2 Support and Power Decomposition",
            "",
            f"- two safety-estimand passed rows: "
            f"`{report['two_safety_estimand_passed_action_row_count']}`",
            f"- all-three passed rows: `{report['all_three_estimand_passed_action_row_count']}`",
            f"- oracle comparator first failures: "
            f"`{report['oracle_comparator_first_failure_count']}`",
            f"- raw policy point estimate positive: "
            f"`{str(report['raw_selector_policy_value_point_estimate_positive']).lower()}`",
            f"- raw policy LCB positive: "
            f"`{str(report['raw_selector_policy_value_lcb_positive']).lower()}`",
            f"- two-safety diagnostic policy LCB positive: "
            f"`{str(report['two_safety_selector_policy_value_lcb_positive']).lower()}`",
            "- two-safety selector reuses calibration OOF targets: `true`",
            "- two-safety result is unbiased candidate evidence: `false`",
            f"- oracle comparator recommendation: "
            f"`{report['oracle_best_comparator_hard_gate_recommendation']}`",
            f"- next research action: `{report['recommended_next_research_action']}`",
            "- next candidate pre-registration allowed by this report: `false`",
        )
    )
