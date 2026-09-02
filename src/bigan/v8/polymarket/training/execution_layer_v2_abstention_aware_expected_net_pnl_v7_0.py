"""Historical-only lineage and exclusion contract for issue #232 v7.0."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_text,
)

CANDIDATE_NAME = "abstention_aware_expected_net_pnl_v7_0"
PROFILE_SCHEMA_VERSION = "bigan-v8-abstention-aware-expected-net-pnl-v7-0-profile-v1"
AUDIT_SCHEMA_VERSION = "bigan-v8-abstention-aware-v7-0-lineage-audit-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-abstention-aware-v7-0-lineage-manifest-v1"
SIDES = {"UP", "DOWN"}
SBC_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
}
FULL_ACTION_GRID = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "NO_TRADE",
}
DECISION_TIME_FEATURE_GROUPS = {
    "source_action_score": [
        "action_score_available",
        "action_score",
        "action_score_margin",
    ],
    "btc_anchor_direction": ["btc_anchor_direction"],
    "market_price_value": [
        "selected_side_probability",
        "execution_price",
        "selected_side_probability_minus_execution_price",
    ],
    "execution_quality": [
        "spread_bps",
        "queue_fill_probability_proxy",
        "book_staleness_ms",
        "time_to_close_seconds",
    ],
    "exposure_state": [
        "pre_entry_market_exposure",
        "same_side_prior_entry",
        "side_flip_prior_entry",
    ],
    "side": ["side_is_up"],
}
FAMILY_TARGET_SOURCE = {
    "SELL_BEFORE_CLOSE": "runtime_aligned_after_cost_exit_policy_target",
    "HOLD_TO_SETTLEMENT": "historical_full_action_grid_after_cost_target",
    "NO_TRADE": "deterministic_zero_target",
}
FORBIDDEN_INFERENCE_FIELDS = {
    "resolved_outcome",
    "winning_outcome",
    "settlement_pnl",
    "realized_trade_pnl",
    "total_polymarket_pnl",
    "runtime_policy_after_cost_net_pnl_per_contract",
    "runtime_policy_after_cost_net_pnl_at_frozen_size",
    "future_return",
    "label",
    "oracle_action",
}
FROZEN_LINEAGE = {
    "runtime_target_manifest_sha256": (
        "e08872532724ace0f84829342092f7cea721e4b6ae91e2b4d8e2974f55fdaab9"
    ),
    "runtime_target_rows_sha256": (
        "1565116daeb2f5d4d8c33fefa507276f59251edd5ffb5f4f313041bcf9dbb0ec"
    ),
    "full_action_grid_manifest_sha256": (
        "fa09998a8093d0f1eb5ec88672cde167bac339fe32b1626f9e35eedd4c0a3120"
    ),
    "full_action_grid_rows_sha256": (
        "fc20e07801743d7f62640bb3a99942ea43cf19e7f4b16770b80053886ae6043a"
    ),
    "issue229_target_free_freeze_manifest_sha256": (
        "186f82099ac2075e9c7f34411c68ef1655bddc7510e92426fb6b2ff82214dd80"
    ),
    "issue229_selected_window_rows_sha256": (
        "8c1a5b92ccd4657bdd6d064cbc45af39d208d944c2b763419597500a7dde48fa"
    ),
    "issue231_target_free_freeze_manifest_sha256": (
        "af9bb7a49f87c71c39e4708f89ac8a64c7657063d8ac08777df5496c1be721d2"
    ),
    "issue231_selected_window_rows_sha256": (
        "90cd57f9aa557e264d34d14084c4e4d7811ecbc4f81b9e8799cde1e568e01bbb"
    ),
    "issue231_settled_index_forbidden_reference_sha256": (
        "fac02081f288b58b36a871b231ce186bb8d442b879bb6f614b8a31d495acf826"
    ),
    "issue231_execution_pnl_report_forbidden_reference_sha256": (
        "3a1dcf58dbd5ec059cc010baa3b2c595a98d57dddf235415fb7d8fd859bba20a"
    ),
}


@dataclass(frozen=True, slots=True)
class AbstentionAwareV70LineageAuditConfig:
    """Pinned inputs for the outcome-isolated v7.0 lineage audit."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    runtime_target_manifest_path: Path | str
    runtime_target_rows_path: Path | str
    full_action_grid_manifest_path: Path | str
    full_action_grid_rows_path: Path | str
    issue229_target_free_freeze_manifest_path: Path | str
    issue231_target_free_freeze_manifest_path: Path | str
    implementation_commit: str
    audit_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, "expected_profile_sha256")
        _require_git_sha(self.implementation_commit)
        if self.audit_created_ts <= 0:
            raise ValueError("audit_created_ts must be positive")
        for field in (
            "output_dir",
            "profile_path",
            "runtime_target_manifest_path",
            "runtime_target_rows_path",
            "full_action_grid_manifest_path",
            "full_action_grid_rows_path",
            "issue229_target_free_freeze_manifest_path",
            "issue231_target_free_freeze_manifest_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def validate_abstention_aware_v7_0_profile(profile: dict[str, Any]) -> None:
    """Reject any drift that could reintroduce target or side-rule tuning."""

    fit = dict(profile.get("fit_protocol") or {})
    calibration = dict(profile.get("calibration_protocol") or {})
    selection = dict(profile.get("selection_policy") or {})
    target_free = dict(profile.get("target_free_actionability") or {})
    future = dict(profile.get("future_confirmatory") or {})
    feature_groups = dict(fit.get("decision_time_feature_groups") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 232,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "preregistered": profile.get("preregistered") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "fit_roles": fit.get("runtime_target_fit_role") == "development_train"
        and fit.get("runtime_target_calibration_role") == "development_calibration",
        "fit_market_counts": fit.get("runtime_target_fit_market_count") == 89
        and fit.get("runtime_target_calibration_market_count") == 45
        and fit.get("full_action_grid_market_count") == 65,
        "full_grid": set(fit.get("full_action_grid_required_actions") or [])
        == FULL_ACTION_GRID,
        "fixed_model": fit.get("model_family")
        == "market_weighted_family_specific_ridge_with_unpenalized_intercept"
        and fit.get("ridge_alpha") == 100.0
        and fit.get("coefficient_absolute_bound") == 8.0
        and fit.get("hyperparameter_search_enabled") is False,
        "target_isolation": fit.get("uses_current_oof_or_validation_pnl_for_tuning")
        is False
        and fit.get("uses_issue229_or_issue231_outcomes_for_fit_or_tuning") is False
        and fit.get("result_selected_rerun_allowed") is False,
        "family_target_sources": fit.get("runtime_target_action_families")
        == ["SELL_BEFORE_CLOSE"]
        and fit.get("family_target_source") == FAMILY_TARGET_SOURCE,
        "feature_groups": feature_groups == DECISION_TIME_FEATURE_GROUPS,
        "calibration": calibration.get("method")
        == "market_clustered_one_sided_selected_action_residual_quantile"
        and calibration.get("coverage_level") == 0.8
        and calibration.get("selection_threshold") == 0.0
        and calibration.get("threshold_operator") == "strictly_greater_than"
        and calibration.get("calibration_labels_used_for_model_fit") is False
        and calibration.get(
            "calibration_labels_used_for_hyperparameter_or_threshold_selection"
        )
        is False
        and calibration.get(
            "calibration_labels_used_for_fixed_uncertainty_calibration"
        )
        is True,
        "abstention": selection.get("decision_grid") == "full_five_action_grid"
        and selection.get("no_trade_score") == 0.0
        and selection.get(
            "select_highest_positive_calibrated_lower_bound_else_no_trade"
        )
        is True
        and selection.get("unsupported_family_target_behavior")
        == "fail_closed_with_explicit_reason",
        "no_side_rule": selection.get("side_quota_allowed") is False
        and selection.get("side_count_hard_gate_enabled") is False
        and selection.get("side_pnl_hard_gate_enabled") is False
        and selection.get("side_composition_is_regime_emergent") is True,
        "target_free": target_free
        == {
            "minimum_guard_accepted_unique_market_count_total": 40,
            "minimum_unique_market_count_per_side": None,
            "required_sides": [],
            "labels_outcomes_resolution_or_pnl_access_allowed": False,
        },
        "future": future
        == {
            "new_strictly_later_disjoint_outcome_blind_window_required": True,
            "issue229_and_issue231_market_ids_excluded": True,
            "hard_gate_aggregation": (
                "accepted_total_after_cost_pnl_and_matched_legacy_delta"
            ),
            "action_family_and_side_metrics_diagnostic_only": True,
            "result_selected_extension_or_rerun_allowed": False,
            "separate_explicit_paper_candidate_approval_required": True,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#232 v7.0 profile invalid: " + ", ".join(blockers))


def build_v7_0_lineage_audit(
    *,
    profile: dict[str, Any],
    runtime_rows: list[dict[str, Any]],
    full_action_grid_rows: list[dict[str, Any]],
    issue229_rows: list[dict[str, Any]],
    issue231_rows: list[dict[str, Any]],
    implementation_commit: str,
    audit_created_ts: int,
) -> dict[str, Any]:
    """Prove historical roles and future-market exclusion without target access."""

    validate_abstention_aware_v7_0_profile(profile)
    fit = profile["fit_protocol"]
    runtime_summary = _validate_runtime_target_rows(runtime_rows, fit=fit)
    full_grid_summary = _validate_full_action_grid_rows(
        full_action_grid_rows,
        expected_market_count=int(fit["full_action_grid_market_count"]),
    )
    issue229_summary = _validate_target_free_exclusion_rows(issue229_rows, issue=229)
    issue231_summary = _validate_target_free_exclusion_rows(issue231_rows, issue=231)
    historical_ids = runtime_summary["market_ids"].union(
        full_grid_summary["market_ids"]
    )
    excluded_ids = issue229_summary["market_ids"].union(
        issue231_summary["market_ids"]
    )
    overlap = historical_ids.intersection(excluded_ids)
    historical_max_ts = max(
        runtime_summary["maximum_decision_ts"],
        full_grid_summary["maximum_decision_ts"],
    )
    excluded_min_ts = min(
        issue229_summary["minimum_decision_ts"],
        issue231_summary["minimum_decision_ts"],
    )
    checks = {
        "historical_sources_market_disjoint_from_excluded_future": not overlap,
        "historical_sources_strictly_earlier_than_excluded_future": (
            historical_max_ts < excluded_min_ts
        ),
        "runtime_target_roles_complete": runtime_summary["roles_complete"],
        "runtime_target_split_chronological_and_disjoint": runtime_summary[
            "split_chronological_and_disjoint"
        ],
        "runtime_target_feature_causality": runtime_summary[
            "feature_causality_violation_count"
        ]
        == 0,
        "full_action_grid_complete": full_grid_summary[
            "incomplete_action_grid_row_count"
        ]
        == 0,
        "full_action_grid_feature_causality": full_grid_summary[
            "feature_causality_violation_count"
        ]
        == 0,
        "issue229_target_free_rows_sealed": issue229_summary[
            "target_sealing_violation_count"
        ]
        == 0,
        "issue231_target_free_rows_sealed": issue231_summary[
            "target_sealing_violation_count"
        ]
        == 0,
        "forbidden_future_outcome_artifacts_not_opened": True,
        "no_side_quota_or_side_pnl_gate": profile["selection_policy"][
            "side_quota_allowed"
        ]
        is False
        and profile["selection_policy"]["side_count_hard_gate_enabled"] is False
        and profile["selection_policy"]["side_pnl_hard_gate_enabled"] is False,
    }
    reason_map = {
        "historical_sources_market_disjoint_from_excluded_future": (
            "historical_source_overlaps_excluded_future_market"
        ),
        "historical_sources_strictly_earlier_than_excluded_future": (
            "historical_source_not_strictly_earlier_than_excluded_future"
        ),
        "runtime_target_roles_complete": "runtime_target_role_coverage_invalid",
        "runtime_target_split_chronological_and_disjoint": (
            "runtime_target_split_invalid"
        ),
        "runtime_target_feature_causality": "runtime_target_feature_causality_failed",
        "full_action_grid_complete": "historical_full_action_grid_incomplete",
        "full_action_grid_feature_causality": (
            "historical_full_action_grid_feature_causality_failed"
        ),
        "issue229_target_free_rows_sealed": "issue229_target_free_sealing_invalid",
        "issue231_target_free_rows_sealed": "issue231_target_free_sealing_invalid",
        "forbidden_future_outcome_artifacts_not_opened": (
            "forbidden_future_outcome_artifact_opened"
        ),
        "no_side_quota_or_side_pnl_gate": "side_rule_contract_invalid",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "audit_created_ts": audit_created_ts,
        "runtime_target_summary": _public_summary(runtime_summary),
        "full_action_grid_summary": _public_summary(full_grid_summary),
        "issue229_exclusion_summary": _public_summary(issue229_summary),
        "issue231_exclusion_summary": _public_summary(issue231_summary),
        "historical_unique_market_count": len(historical_ids),
        "excluded_future_unique_market_count": len(excluded_ids),
        "historical_future_market_overlap_count": len(overlap),
        "historical_maximum_decision_ts": historical_max_ts,
        "excluded_future_minimum_decision_ts": excluded_min_ts,
        "issue231_settled_index_forbidden_reference_sha256": profile["lineage"][
            "issue231_settled_index_forbidden_reference_sha256"
        ],
        "issue231_execution_pnl_report_forbidden_reference_sha256": profile[
            "lineage"
        ]["issue231_execution_pnl_report_forbidden_reference_sha256"],
        "issue229_or_issue231_outcomes_opened": False,
        "issue229_or_issue231_outcomes_used_for_fit_or_tuning": False,
        "current_oof_or_validation_pnl_used_for_tuning": False,
        "side_quota_applied": False,
        "side_count_hard_gate_enabled": False,
        "side_pnl_hard_gate_enabled": False,
        "lineage_audit_checks": checks,
        "lineage_audit_passed": not reasons,
        "lineage_audit_blocking_reason_codes": reasons,
        "model_fit_attempted": False,
        "future_target_access_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    audit["audit_id"] = canonical_json_sha256(audit)
    return audit


def run_v7_0_lineage_audit(
    config: AbstentionAwareV70LineageAuditConfig,
) -> dict[str, Any]:
    """Run the profile-pinned audit and write hashable diagnostic artifacts."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "runtime_target_manifest": Path(config.runtime_target_manifest_path).resolve(),
        "runtime_target_rows": Path(config.runtime_target_rows_path).resolve(),
        "full_action_grid_manifest": Path(
            config.full_action_grid_manifest_path
        ).resolve(),
        "full_action_grid_rows": Path(config.full_action_grid_rows_path).resolve(),
        "issue229_target_free_freeze_manifest": Path(
            config.issue229_target_free_freeze_manifest_path
        ).resolve(),
        "issue231_target_free_freeze_manifest": Path(
            config.issue231_target_free_freeze_manifest_path
        ).resolve(),
    }
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#232 profile")
    profile = _load_json(paths["profile"])
    validate_abstention_aware_v7_0_profile(profile)
    lineage = profile["lineage"]
    pins = {
        "runtime_target_manifest": "runtime_target_manifest_sha256",
        "runtime_target_rows": "runtime_target_rows_sha256",
        "full_action_grid_manifest": "full_action_grid_manifest_sha256",
        "full_action_grid_rows": "full_action_grid_rows_sha256",
        "issue229_target_free_freeze_manifest": (
            "issue229_target_free_freeze_manifest_sha256"
        ),
        "issue231_target_free_freeze_manifest": (
            "issue231_target_free_freeze_manifest_sha256"
        ),
    }
    for name, pin_name in pins.items():
        _verify_pin(paths[name], lineage[pin_name], f"#232 {name}")

    runtime_manifest = _load_json(paths["runtime_target_manifest"])
    if runtime_manifest.get("runtime_aligned_rows") != _descriptor(
        paths["runtime_target_rows"]
    ):
        raise ValueError("#232 runtime target manifest row descriptor mismatch")
    full_grid_manifest = _load_json(paths["full_action_grid_manifest"])
    if full_grid_manifest.get("development_rows") != _descriptor(
        paths["full_action_grid_rows"]
    ):
        raise ValueError("#232 full action-grid manifest row descriptor mismatch")

    issue229_manifest = _load_json(paths["issue229_target_free_freeze_manifest"])
    issue231_manifest = _load_json(paths["issue231_target_free_freeze_manifest"])
    issue229_rows_path = _verified_selected_window_descriptor(
        issue229_manifest,
        expected_sha256=lineage["issue229_selected_window_rows_sha256"],
        issue=229,
    )
    issue231_rows_path = _verified_selected_window_descriptor(
        issue231_manifest,
        expected_sha256=lineage["issue231_selected_window_rows_sha256"],
        issue=231,
    )
    audit = build_v7_0_lineage_audit(
        profile=profile,
        runtime_rows=_load_jsonl(paths["runtime_target_rows"]),
        full_action_grid_rows=_load_jsonl(paths["full_action_grid_rows"]),
        issue229_rows=_load_jsonl(issue229_rows_path),
        issue231_rows=_load_jsonl(issue231_rows_path),
        implementation_commit=config.implementation_commit,
        audit_created_ts=config.audit_created_ts,
    )

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    audit_path = run_dir / "v7_0_lineage_and_exclusion_audit.json"
    markdown_path = run_dir / "v7_0_lineage_and_exclusion_audit.md"
    _write_json(audit_path, audit)
    _write_text(markdown_path, _lineage_audit_markdown(audit))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "issue_number": 232,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(paths["profile"]),
        "runtime_target_manifest": _descriptor(paths["runtime_target_manifest"]),
        "runtime_target_rows": _descriptor(paths["runtime_target_rows"]),
        "full_action_grid_manifest": _descriptor(paths["full_action_grid_manifest"]),
        "full_action_grid_rows": _descriptor(paths["full_action_grid_rows"]),
        "issue229_target_free_freeze_manifest": _descriptor(
            paths["issue229_target_free_freeze_manifest"]
        ),
        "issue229_selected_window_rows": _descriptor(issue229_rows_path),
        "issue231_target_free_freeze_manifest": _descriptor(
            paths["issue231_target_free_freeze_manifest"]
        ),
        "issue231_selected_window_rows": _descriptor(issue231_rows_path),
        "lineage_audit": _descriptor(audit_path),
        "lineage_audit_markdown": _descriptor(markdown_path),
        "lineage_audit_passed": audit["lineage_audit_passed"],
        "lineage_audit_blocking_reason_codes": audit[
            "lineage_audit_blocking_reason_codes"
        ],
        "forbidden_future_outcome_artifacts_opened": False,
        "model_fit_attempted": False,
        "future_target_access_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_0_lineage_audit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "audit": audit,
        "audit_path": audit_path,
        "audit_sha256": _sha256_file(audit_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _validate_runtime_target_rows(
    rows: list[dict[str, Any]], *, fit: dict[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("#232 runtime target rows are empty")
    fit_role = str(fit["runtime_target_fit_role"])
    calibration_role = str(fit["runtime_target_calibration_role"])
    roles = {fit_role, calibration_role}
    role_counts = Counter(str(row.get("role") or "") for row in rows)
    market_ids_by_role = {
        role: {str(row.get("market_id") or "") for row in rows if row.get("role") == role}
        for role in roles
    }
    if set(role_counts) != roles:
        raise ValueError("#232 runtime target role set invalid")
    if any(str(row.get("side") or "") not in SIDES for row in rows):
        raise ValueError("#232 runtime target side invalid")
    if any(str(row.get("action") or "") not in SBC_ACTIONS for row in rows):
        raise ValueError("#232 runtime target action invalid")
    if any(row.get("target_used_as_decision_time_input") is not False for row in rows):
        raise ValueError("#232 runtime target used as decision input")
    if any(
        row.get("target_available_only_post_exit_or_official_resolution") is not True
        for row in rows
    ):
        raise ValueError("#232 runtime target availability invalid")
    causality = sum(int(row["max_input_ts"]) > int(row["decision_ts"]) for row in rows)
    fit_ids = market_ids_by_role[fit_role]
    calibration_ids = market_ids_by_role[calibration_role]
    roles_complete = len(fit_ids) == int(fit["runtime_target_fit_market_count"]) and len(
        calibration_ids
    ) == int(fit["runtime_target_calibration_market_count"])
    chronological = max(
        int(row["decision_ts"]) for row in rows if row["role"] == fit_role
    ) < min(
        int(row["decision_ts"])
        for row in rows
        if row["role"] == calibration_role
    )
    market_disjoint = not fit_ids.intersection(calibration_ids)
    return {
        "row_count": len(rows),
        "unique_market_count": len(fit_ids.union(calibration_ids)),
        "market_ids": fit_ids.union(calibration_ids),
        "role_row_counts": dict(sorted(role_counts.items())),
        "fit_market_count": len(fit_ids),
        "calibration_market_count": len(calibration_ids),
        "roles_complete": roles_complete,
        "split_chronological_and_disjoint": chronological and market_disjoint,
        "feature_causality_violation_count": causality,
        "minimum_decision_ts": min(int(row["decision_ts"]) for row in rows),
        "maximum_decision_ts": max(int(row["decision_ts"]) for row in rows),
    }


def _validate_full_action_grid_rows(
    rows: list[dict[str, Any]], *, expected_market_count: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("#232 full action-grid rows are empty")
    market_ids = {str(row.get("market_id") or "") for row in rows}
    if "" in market_ids or len(market_ids) != expected_market_count:
        raise ValueError("#232 full action-grid market support invalid")
    incomplete = 0
    causality = 0
    nested_forbidden = 0
    for row in rows:
        target_map = dict(row.get("evaluation_target_net_pnl_per_contract_by_action") or {})
        if set(target_map) != FULL_ACTION_GRID:
            incomplete += 1
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            causality += 1
        features = dict(row.get("decision_time_features") or {})
        if FORBIDDEN_INFERENCE_FIELDS.intersection(features):
            nested_forbidden += 1
        if row.get("target_outcome_available_only_post_resolution") is not True:
            raise ValueError("#232 full action-grid target availability invalid")
        if row.get("target_provenance", {}).get("outcome_used_as_decision_input") is not False:
            raise ValueError("#232 full action-grid outcome used as decision input")
    if nested_forbidden:
        raise ValueError("#232 forbidden target field in decision-time features")
    return {
        "row_count": len(rows),
        "unique_market_count": len(market_ids),
        "market_ids": market_ids,
        "incomplete_action_grid_row_count": incomplete,
        "feature_causality_violation_count": causality,
        "minimum_decision_ts": min(int(row["decision_ts"]) for row in rows),
        "maximum_decision_ts": max(int(row["decision_ts"]) for row in rows),
    }


def _validate_target_free_exclusion_rows(
    rows: list[dict[str, Any]], *, issue: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"#232 issue{issue} target-free rows are empty")
    market_ids = {str(row.get("market_id") or "") for row in rows}
    if "" in market_ids or len(market_ids) != len(rows):
        raise ValueError(f"#232 issue{issue} target-free market identity invalid")
    target_sealing_violations = 0
    causality_violations = 0
    for row in rows:
        if row.get("labels_outcomes_or_pnl_opened") is not False:
            target_sealing_violations += 1
        if row.get("resolution_provider_called") is not False:
            target_sealing_violations += 1
        if int(row["market_start_ts"]) >= int(row["market_end_ts"]):
            raise ValueError(f"#232 issue{issue} market window invalid")
        if int(row.get("scheduled_round_start_ts") or row["market_start_ts"]) > int(
            row["market_start_ts"]
        ):
            causality_violations += 1
    return {
        "row_count": len(rows),
        "unique_market_count": len(market_ids),
        "market_ids": market_ids,
        "target_sealing_violation_count": target_sealing_violations,
        "feature_causality_violation_count": causality_violations,
        "minimum_decision_ts": min(int(row["market_start_ts"]) for row in rows),
        "maximum_decision_ts": max(int(row["market_start_ts"]) for row in rows),
    }


def _verified_selected_window_descriptor(
    manifest: dict[str, Any], *, expected_sha256: str, issue: int
) -> Path:
    if manifest.get("labels_outcomes_resolution_or_pnl_opened") is not False:
        raise ValueError(f"#232 issue{issue} target-free manifest is not sealed")
    descriptor = dict(manifest.get("selected_window_rows") or {})
    if descriptor.get("sha256") != expected_sha256:
        raise ValueError(f"#232 issue{issue} selected-window pin mismatch")
    path = Path(str(descriptor.get("path") or "")).resolve()
    _verify_pin(path, expected_sha256, f"#232 issue{issue} selected window rows")
    return path


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "market_ids"}


def _v7_0_blocked_safety_fields() -> dict[str, Any]:
    return {
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
        "live_trading_enabled": False,
    }


def _lineage_audit_markdown(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v7.0 Historical Lineage and Exclusion Audit",
            "",
            f"- audit passed: `{str(audit['lineage_audit_passed']).lower()}`",
            f"- historical markets: `{audit['historical_unique_market_count']}`",
            f"- excluded future markets: `{audit['excluded_future_unique_market_count']}`",
            f"- historical/future overlap: `{audit['historical_future_market_overlap_count']}`",
            f"- historical max decision ts: `{audit['historical_maximum_decision_ts']}`",
            f"- future min decision ts: `{audit['excluded_future_minimum_decision_ts']}`",
            f"- blockers: `{audit['lineage_audit_blocking_reason_codes']}`",
            "- issue #229/#231 outcomes opened: `false`",
            "- current OOF/validation PnL used for tuning: `false`",
            "- side quota / side hard gate: `false / false`",
            "- model fit attempted: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )
