"""Pre-register the post-#204 policy-selected conformal net-return v6 study."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    FORBIDDEN_TARGET_FIELDS,
    _blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    load_and_validate_persistent_outcome_blind_index,
    validate_persistent_outcome_blind_collector_protocol,
)

CANDIDATE_NAME = "guard_compatible_policy_selected_conformal_net_return_v6"
PROFILE_SCHEMA_VERSION = "bigan-v8-policy-selected-conformal-v6-preregistration-profile-v1"
ATTRITION_REPORT_SCHEMA_VERSION = "bigan-v8-conformal-v5-target-free-no-trade-attrition-v1"
PREREG_REPORT_SCHEMA_VERSION = "bigan-v8-policy-selected-conformal-v6-preregistration-report-v1"
PREREG_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-policy-selected-conformal-v6-preregistration-manifest-v1"
)
SOURCE_BOUNDARY_SCHEMA_VERSION = "bigan-v8-policy-selected-conformal-v6-source-boundary-v1"

REQUIRED_ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "NO_TRADE",
)
TRADE_ACTIONS = frozenset(REQUIRED_ACTIONS[:-1])
SIDES = ("UP", "DOWN")


@dataclass(frozen=True, slots=True)
class PolicySelectedConformalV6PreRegistrationConfig:
    """All paths and pins opened before post-#204 development target access."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    issue204_window_manifest_path: Path | str
    issue204_decision_freeze_path: Path | str
    issue204_prediction_report_path: Path | str
    collector_index_path: Path | str
    expected_collector_index_prefix_sha256: str
    collector_protocol_path: Path | str
    power_report_path: Path | str
    power_manifest_path: Path | str
    builder_git_commit: str
    preregistration_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_profile_sha256",
            "expected_collector_index_prefix_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        _require_git_sha(self.builder_git_commit)
        if self.preregistration_created_ts <= 0:
            raise ValueError("preregistration_created_ts must be positive")
        for field in (
            "output_dir",
            "profile_path",
            "issue204_window_manifest_path",
            "issue204_decision_freeze_path",
            "issue204_prediction_report_path",
            "collector_index_path",
            "collector_protocol_path",
            "power_report_path",
            "power_manifest_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def validate_policy_selected_conformal_v6_profile(profile: dict[str, Any]) -> None:
    """Reject drift from the #207 protocol before any new target is opened."""

    upstream = dict(profile.get("frozen_upstream") or {})
    development = dict(profile.get("development_window") or {})
    roles = dict(profile.get("chronological_roles") or {})
    model = dict(profile.get("point_model") or {})
    calibration = dict(profile.get("policy_selected_conformal_calibration") or {})
    future = dict(profile.get("future_evaluation") or {})
    exclusions = dict(profile.get("prohibited_inputs") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "upstream_hashes": all(
            _is_sha256(upstream.get(field))
            for field in (
                "issue204_window_manifest_sha256",
                "issue204_decision_freeze_sha256",
                "issue204_prediction_report_sha256",
                "issue205_power_report_sha256",
                "issue205_power_manifest_sha256",
                "collector_protocol_sha256",
                "feature_contract_sha256",
                "v5_model_sha256",
                "matched_v4_model_sha256",
            )
        ),
        "terminal_window": upstream.get("issue204_selected_market_count") == 220
        and upstream.get("issue204_outcomes_permitted_for_fit_or_tuning") is False,
        "development_size": development.get("target_quality_valid_market_count") == 260
        and development.get("maximum_index_scan_count") == 400,
        "development_boundary": development.get("selection_method")
        == "earliest_quality_valid_post_issue204_disjoint_rows"
        and int(development.get("minimum_eligible_index_sequence") or 0) == 237
        and int(development.get("minimum_eligible_market_start_ts") or 0) == 1784445600000,
        "development_outcome_blind": development.get(
            "labels_outcomes_or_pnl_opened_for_selection"
        )
        is False
        and development.get("result_dependent_extension_allowed") is False,
        "role_counts": roles
        == {
            "point_model_fit_market_count": 150,
            "conformal_calibration_market_count": 60,
            "calibration_check_market_count": 50,
            "assignment": "chronological_non_overlapping_market_groups",
        },
        "model_contract": model.get("target") == "target_net_pnl_per_contract"
        and model.get("training_target_includes_costs") is True
        and model.get("hyperparameter_search_enabled") is False
        and model.get("decision_time_features_only") is True,
        "calibration_method": calibration.get("method")
        == "sequential_policy_selected_market_grouped_one_sided_split_conformal"
        and calibration.get("one_sided_alpha") == 0.1,
        "causal_selection": calibration.get("decision_schedule_order") == "chronological"
        and calibration.get("later_decision_rows_visible_to_earlier_decision") is False
        and calibration.get("maximum_selected_trade_rows_per_market") == 1,
        "unchanged_execution": calibration.get("execution_compatibility_mask_required") is True
        and calibration.get("one_position_per_market_exposure_required") is True
        and calibration.get("execution_guard_mutation_allowed") is False
        and calibration.get("cost_model_mutation_allowed") is False,
        "calibration_support": calibration.get("minimum_side_calibration_market_count") == 20
        and calibration.get("minimum_global_calibration_market_count") == 50
        and calibration.get("fallback_order") == ["selected_side", "all_trade_sides"],
        "no_trade_anchor": calibration.get("no_trade_score") == 0.0
        and calibration.get("minimum_selected_lower_bound_exclusive") == 0.0,
        "no_calibration_pnl": calibration.get("policy_pnl_computed_on_calibration") is False
        and calibration.get("policy_pnl_computed_on_calibration_check") is False
        and calibration.get("calibration_threshold_search_enabled") is False,
        "future_size": future.get("target_quality_valid_market_count") == 300
        and future.get("maximum_index_scan_count") == 462,
        "future_support": future.get("minimum_guard_accepted_unique_market_count") == 120
        and future.get("minimum_supported_side_market_count") == 17
        and future.get("required_supported_sides") == ["UP", "DOWN"],
        "future_side_only": future.get("pnl_hard_gate_aggregation")
        == "selected_side_buy_up_buy_down_only"
        and future.get("action_and_action_family_pnl_diagnostic_only") is True,
        "future_single_use": future.get("single_use_holdout") is True
        and future.get("future_result_driven_rerun_or_tuning_allowed") is False,
        "exclusions": exclusions
        == {
            "uses_204_outcomes_for_fitting": False,
            "uses_204_pnl_for_tuning": False,
            "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
            "uses_future_holdout_targets_before_decision_freeze": False,
        },
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("v6 preregistration profile invalid: " + ", ".join(blockers))


def pre_register_policy_selected_conformal_v6(
    config: PolicySelectedConformalV6PreRegistrationConfig,
) -> dict[str, Any]:
    """Freeze #207 inputs and the post-#204 source boundary without target access."""

    profile_path = config.profile_path.resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "v6 profile")
    profile = _load_json(profile_path)
    validate_policy_selected_conformal_v6_profile(profile)
    upstream = dict(profile["frozen_upstream"])
    pinned_paths = {
        "issue204_window_manifest": config.issue204_window_manifest_path.resolve(),
        "issue204_decision_freeze": config.issue204_decision_freeze_path.resolve(),
        "issue204_prediction_report": config.issue204_prediction_report_path.resolve(),
        "collector_protocol": config.collector_protocol_path.resolve(),
        "power_report": config.power_report_path.resolve(),
        "power_manifest": config.power_manifest_path.resolve(),
    }
    pinned_hashes = {
        "issue204_window_manifest": upstream["issue204_window_manifest_sha256"],
        "issue204_decision_freeze": upstream["issue204_decision_freeze_sha256"],
        "issue204_prediction_report": upstream["issue204_prediction_report_sha256"],
        "collector_protocol": upstream["collector_protocol_sha256"],
        "power_report": upstream["issue205_power_report_sha256"],
        "power_manifest": upstream["issue205_power_manifest_sha256"],
    }
    for name, path in pinned_paths.items():
        _verify_pin(path, str(pinned_hashes[name]), name)
    validate_persistent_outcome_blind_collector_protocol(
        _load_json(pinned_paths["collector_protocol"])
    )

    window = _load_json(pinned_paths["issue204_window_manifest"])
    prior_selected_path = _verified_descriptor(window.get("selected_rows"), "#204 selected rows")
    prior_rows = _load_jsonl(Path(prior_selected_path["path"]))
    _validate_prior_window(window, prior_rows, profile=profile)
    exclusion = _prior_exclusion_summary(prior_rows)

    decision_freeze = _load_json(pinned_paths["issue204_decision_freeze"])
    prediction_report = _load_json(pinned_paths["issue204_prediction_report"])
    attrition = build_target_free_v5_no_trade_attrition_report(
        decision_freeze,
        prediction_report=prediction_report,
        expected_decision_freeze_sha256=upstream["issue204_decision_freeze_sha256"],
    )

    index_path = config.collector_index_path.resolve()
    _verify_pin(
        index_path,
        config.expected_collector_index_prefix_sha256,
        "collector index prefix",
    )
    index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    prefix_summary = _validate_and_summarize_index_prefix(
        index_rows,
        profile=profile,
        exclusion=exclusion,
    )

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    index_snapshot_path = run_dir / "persistent_outcome_blind_round_index_prefix.jsonl"
    shutil.copyfile(index_path, index_snapshot_path)
    if _sha256_file(index_snapshot_path) != config.expected_collector_index_prefix_sha256:
        raise ValueError("collector index prefix changed during snapshot")
    if _sha256_file(index_path) != config.expected_collector_index_prefix_sha256:
        raise ValueError("collector index source changed during snapshot")

    attrition_path = run_dir / "conformal_v5_target_free_no_trade_attrition_report.json"
    _write_json(attrition_path, attrition)
    _write_text(
        run_dir / "conformal_v5_target_free_no_trade_attrition_report.md",
        _attrition_markdown(attrition),
    )

    source_boundary = {
        "schema_version": SOURCE_BOUNDARY_SCHEMA_VERSION,
        "run_id": config.run_id,
        "issue204_window_manifest": _descriptor(pinned_paths["issue204_window_manifest"]),
        "issue204_selected_rows": prior_selected_path,
        "excluded_issue204_market_count": exclusion["market_count"],
        "excluded_issue204_slug_count": exclusion["slug_count"],
        "excluded_issue204_source_row_hash_count": exclusion["source_row_hash_count"],
        "issue204_exclusion_identity_hash": exclusion["identity_hash"],
        "issue204_max_selected_index_sequence": exclusion["max_sequence"],
        "issue204_max_market_end_ts": exclusion["max_market_end_ts"],
        "minimum_eligible_index_sequence": profile["development_window"][
            "minimum_eligible_index_sequence"
        ],
        "minimum_eligible_market_start_ts": profile["development_window"][
            "minimum_eligible_market_start_ts"
        ],
        "collector_index_prefix": _descriptor(index_snapshot_path),
        "collector_index_prefix_row_count": len(index_rows),
        "collector_index_prefix_last_entry_sha256": (
            str(index_rows[-1]["entry_sha256"]) if index_rows else None
        ),
        "eligible_quality_valid_rows_already_indexed": prefix_summary[
            "eligible_quality_valid_row_count"
        ],
        "development_target_quality_valid_market_count": profile["development_window"][
            "target_quality_valid_market_count"
        ],
        "development_markets_remaining": prefix_summary["development_markets_remaining"],
        "labels_outcomes_or_pnl_opened": False,
        "raw_artifact_payloads_opened": False,
        "future_prediction_attempted": False,
        **_blocked_safety_fields(),
    }
    source_boundary["source_boundary_id"] = canonical_json_sha256(source_boundary)
    source_boundary_path = run_dir / "conformal_v6_development_source_boundary.json"
    _write_json(source_boundary_path, source_boundary)

    report = {
        "schema_version": PREREG_REPORT_SCHEMA_VERSION,
        "report_id": None,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "builder_git_commit": config.builder_git_commit,
        "preregistration_created_ts": config.preregistration_created_ts,
        "profile": _descriptor(profile_path),
        "target_free_v5_no_trade_attrition": _descriptor(attrition_path),
        "source_boundary": _descriptor(source_boundary_path),
        "issue204_target_free_diagnostic_only": True,
        "issue204_outcome_settlement_target_or_pnl_files_opened": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_204_pnl_for_tuning": False,
        "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
        "new_development_target_accessed": False,
        "development_window_frozen": False,
        "development_target_quality_valid_market_count": profile["development_window"][
            "target_quality_valid_market_count"
        ],
        "development_index_scan_cap": profile["development_window"][
            "maximum_index_scan_count"
        ],
        "chronological_role_market_counts": profile["chronological_roles"],
        "collector_index_prefix_summary": prefix_summary,
        "v6_policy_selected_calibration_contract": profile[
            "policy_selected_conformal_calibration"
        ],
        "future_evaluation_contract": profile["future_evaluation"],
        "preregistration_passed": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v6_preregistration_report.json"
    _write_json(report_path, report)
    _write_text(run_dir / "conformal_v6_preregistration_report.md", _report_markdown(report))

    manifest = {
        "schema_version": PREREG_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "builder_git_commit": config.builder_git_commit,
        "preregistration_created_ts": config.preregistration_created_ts,
        "profile": _descriptor(profile_path),
        "issue204_window_manifest": _descriptor(pinned_paths["issue204_window_manifest"]),
        "issue204_decision_freeze": _descriptor(pinned_paths["issue204_decision_freeze"]),
        "issue204_prediction_report": _descriptor(pinned_paths["issue204_prediction_report"]),
        "collector_protocol": _descriptor(pinned_paths["collector_protocol"]),
        "issue205_power_report": _descriptor(pinned_paths["power_report"]),
        "issue205_power_manifest": _descriptor(pinned_paths["power_manifest"]),
        "collector_index_prefix": _descriptor(index_snapshot_path),
        "target_free_v5_no_trade_attrition": _descriptor(attrition_path),
        "development_source_boundary": _descriptor(source_boundary_path),
        "preregistration_report": _descriptor(report_path),
        "preregistration_passed": True,
        "development_window_frozen": False,
        "new_development_target_accessed": False,
        "future_evaluation_attempted": False,
        "result_dependent_rerun_or_tuning_allowed": False,
        **_blocked_safety_fields(),
    }
    manifest["preregistration_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v6_preregistration_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "attrition_report": attrition,
        "attrition_report_path": attrition_path,
        "attrition_report_sha256": _sha256_file(attrition_path),
        "source_boundary": source_boundary,
        "source_boundary_path": source_boundary_path,
        "source_boundary_sha256": _sha256_file(source_boundary_path),
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def build_target_free_v5_no_trade_attrition_report(
    decision_freeze: dict[str, Any],
    *,
    prediction_report: dict[str, Any],
    expected_decision_freeze_sha256: str,
) -> dict[str, Any]:
    """Explain the terminal v5 no-trade path without opening any target artifact."""

    _require_sha256(expected_decision_freeze_sha256, name="decision freeze sha256")
    blockers = []
    if decision_freeze.get("future_labels_outcomes_or_pnl_opened") is not False:
        blockers.append("decision_freeze_target_sealing_invalid")
    if decision_freeze.get("target_or_outcome_used_for_decision") is not False:
        blockers.append("decision_freeze_target_usage_invalid")
    if decision_freeze.get("candidate_guard_accepted_bet_count") != 0:
        blockers.append("terminal_candidate_accepted_count_not_zero")
    if prediction_report.get("future_labels_outcomes_or_pnl_opened") is not False:
        blockers.append("prediction_report_target_sealing_invalid")
    if prediction_report.get("target_or_outcome_used_for_decision") is not False:
        blockers.append("prediction_report_target_usage_invalid")
    descriptor = _verified_descriptor(
        decision_freeze.get("candidate_target_free_predictions"),
        "candidate target-free predictions",
    )
    rows = _load_jsonl(Path(descriptor["path"]))
    forbidden = _find_nonempty_fields(rows, FORBIDDEN_TARGET_FIELDS)
    if forbidden:
        blockers.append("target_free_predictions_contain_forbidden_fields")
    if any(row.get("target_used_as_decision_input") is not False for row in rows):
        blockers.append("prediction_target_usage_invalid")
    if any(row.get("target_or_outcome_fields_used") is not False for row in rows):
        blockers.append("prediction_outcome_usage_invalid")
    if blockers:
        raise ValueError("target-free v5 attrition input invalid: " + ", ".join(blockers))

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        action = str(row.get("action") or "")
        if action not in REQUIRED_ACTIONS:
            raise ValueError(f"unexpected v5 action: {action}")
        key = (str(row.get("market_id") or ""), int(row.get("decision_ts") or 0))
        if not key[0] or key[1] <= 0:
            raise ValueError("target-free prediction identity invalid")
        groups[key].append(row)
        by_action[action].append(row)
    if any({str(row["action"]) for row in group} != set(REQUIRED_ACTIONS) for group in groups.values()):
        raise ValueError("target-free decision group is not a complete five-action grid")

    action_diagnostics = {}
    for action in REQUIRED_ACTIONS:
        action_rows = by_action[action]
        raw = [float(row["raw_direct_predicted_net_return"]) for row in action_rows]
        penalties = [float(row["conformal_calibration_penalty"]) for row in action_rows]
        bounds = [float(row["conformal_net_return_lower_bound"]) for row in action_rows]
        compatible = [row for row in action_rows if row["guard_compatible_before_ranking"]]
        action_diagnostics[action] = {
            "row_count": len(action_rows),
            "guard_compatible_row_count": len(compatible),
            "raw_positive_row_count": sum(value > 0.0 for value in raw),
            "guard_compatible_raw_positive_row_count": sum(
                float(row["raw_direct_predicted_net_return"]) > 0.0 for row in compatible
            ),
            "guard_compatible_positive_lcb_row_count": sum(
                float(row["conformal_net_return_lower_bound"]) > 0.0 for row in compatible
            ),
            "raw_prediction_summary": _numeric_summary(raw),
            "calibration_penalty_summary": _numeric_summary(penalties),
            "conformal_lcb_summary": _numeric_summary(bounds),
            "calibration_source_distribution": dict(
                sorted(Counter(str(row["conformal_calibration_source"]) for row in action_rows).items())
            ),
        }

    selected = Counter()
    raw_selected = Counter()
    group_stage = Counter()
    groups_with_guard_compatible_raw_positive = 0
    groups_with_positive_lcb = 0
    raw_positive_rows_blocked_by_penalty = 0
    for group in groups.values():
        compatible = [row for row in group if row["guard_compatible_before_ranking"]]
        if not compatible:
            raise ValueError("NO_TRADE must remain guard compatible")
        selected_row = max(
            compatible,
            key=lambda row: (float(row["action_selection_score"]), str(row["action"])),
        )
        raw_row = max(
            compatible,
            key=lambda row: (
                float(row["raw_direct_predicted_net_return"]),
                str(row["action"]),
            ),
        )
        selected[str(selected_row["action"])] += 1
        raw_selected[str(raw_row["action"])] += 1
        compatible_trades = [row for row in compatible if row["action"] in TRADE_ACTIONS]
        raw_positive = [
            row
            for row in compatible_trades
            if float(row["raw_direct_predicted_net_return"]) > 0.0
        ]
        positive_lcb = [
            row for row in compatible_trades if float(row["conformal_net_return_lower_bound"]) > 0.0
        ]
        groups_with_guard_compatible_raw_positive += bool(raw_positive)
        groups_with_positive_lcb += bool(positive_lcb)
        raw_positive_rows_blocked_by_penalty += sum(
            float(row["conformal_net_return_lower_bound"]) <= 0.0 for row in raw_positive
        )
        if not compatible_trades:
            group_stage["no_guard_compatible_trade"] += 1
        elif not raw_positive:
            group_stage["no_positive_raw_trade_prediction"] += 1
        elif not positive_lcb:
            group_stage["positive_raw_trade_blocked_by_conformal_penalty"] += 1
        else:
            group_stage["positive_trade_lcb_available"] += 1

    report = {
        "schema_version": ATTRITION_REPORT_SCHEMA_VERSION,
        "report_id": None,
        "issue204_decision_freeze_sha256": expected_decision_freeze_sha256,
        "candidate_predictions": descriptor,
        "diagnostic_scope": "target_free_prediction_and_selection_only",
        "decision_group_count": len(groups),
        "prediction_row_count": len(rows),
        "complete_five_action_group_count": len(groups),
        "selected_action_distribution": dict(sorted(selected.items())),
        "raw_argmax_action_distribution": dict(sorted(raw_selected.items())),
        "decision_groups_with_guard_compatible_raw_positive_trade": (
            groups_with_guard_compatible_raw_positive
        ),
        "decision_groups_with_positive_conformal_trade_lcb": groups_with_positive_lcb,
        "raw_positive_trade_rows_blocked_by_conformal_penalty": (
            raw_positive_rows_blocked_by_penalty
        ),
        "attrition_stage_distribution": dict(sorted(group_stage.items())),
        "action_diagnostics": action_diagnostics,
        "all_selected_actions_no_trade": selected == Counter({"NO_TRADE": len(groups)}),
        "all_guard_compatible_trade_lcbs_nonpositive": groups_with_positive_lcb == 0,
        "root_cause_classification": (
            "market_simultaneous_all_decision_action_penalty_dominates_raw_trade_scores"
        ),
        "code_bug_indicated": False,
        "calibration_policy_alignment_redesign_required": True,
        "outcomes_labels_settlement_or_pnl_opened": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_204_pnl_for_tuning": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _validate_prior_window(
    window: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> None:
    blockers = []
    if window.get("window_freeze_ready") is not True:
        blockers.append("issue204_window_not_ready")
    if window.get("selected_market_count") != 220 or len(rows) != 220:
        blockers.append("issue204_selected_market_count_invalid")
    if window.get("labels_outcomes_or_pnl_opened_for_selection") is not False:
        blockers.append("issue204_window_target_sealing_invalid")
    if _find_nonempty_fields(rows, FORBIDDEN_TARGET_FIELDS):
        blockers.append("issue204_selected_rows_contain_target_fields")
    if len({str(row.get("market_id") or "") for row in rows}) != 220:
        blockers.append("issue204_market_identity_not_unique")
    if len({str(row.get("slug") or row.get("market_slug") or "") for row in rows}) != 220:
        blockers.append("issue204_slug_identity_not_unique")
    max_sequence = max(int(row.get("sequence") or 0) for row in rows)
    max_market_end = max(int(row.get("market_end_ts") or 0) for row in rows)
    development = profile["development_window"]
    if int(development["minimum_eligible_index_sequence"]) != max_sequence + 1:
        blockers.append("development_minimum_sequence_not_after_issue204")
    if int(development["minimum_eligible_market_start_ts"]) != max_market_end:
        blockers.append("development_time_boundary_not_after_issue204")
    if blockers:
        raise ValueError("#204 prior-window validation failed: " + ", ".join(blockers))


def _prior_exclusion_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    market_ids = sorted(str(row["market_id"]) for row in rows)
    slugs = sorted(str(row.get("slug") or row.get("market_slug")) for row in rows)
    source_hashes = sorted(str(row["source_row_hash"]) for row in rows)
    return {
        "market_ids": set(market_ids),
        "slugs": set(slugs),
        "source_row_hashes": set(source_hashes),
        "market_count": len(set(market_ids)),
        "slug_count": len(set(slugs)),
        "source_row_hash_count": len(set(source_hashes)),
        "max_sequence": max(int(row["sequence"]) for row in rows),
        "max_market_end_ts": max(int(row["market_end_ts"]) for row in rows),
        "identity_hash": canonical_json_sha256(
            {"market_ids": market_ids, "slugs": slugs, "source_row_hashes": source_hashes}
        ),
    }


def _validate_and_summarize_index_prefix(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    exclusion: dict[str, Any],
) -> dict[str, Any]:
    development = profile["development_window"]
    minimum_sequence = int(development["minimum_eligible_index_sequence"])
    minimum_market_start = int(development["minimum_eligible_market_start_ts"])
    eligible = []
    overlap_reasons = Counter()
    for row in rows:
        if int(row["sequence"]) < minimum_sequence or not row.get("capture_quality_valid"):
            continue
        reasons = []
        if int(row.get("market_start_ts") or 0) < minimum_market_start:
            reasons.append("market_start_before_development_boundary")
        if str(row.get("market_id") or "") in exclusion["market_ids"]:
            reasons.append("issue204_market_overlap")
        if str(row.get("slug") or "") in exclusion["slugs"]:
            reasons.append("issue204_slug_overlap")
        if str(row.get("source_row_hash") or "") in exclusion["source_row_hashes"]:
            reasons.append("issue204_source_row_hash_overlap")
        if reasons:
            overlap_reasons.update(reasons)
        else:
            eligible.append(row)
    if overlap_reasons:
        raise ValueError(
            "post-#204 index prefix overlap: "
            + ", ".join(f"{key}={value}" for key, value in sorted(overlap_reasons.items()))
        )
    target = int(development["target_quality_valid_market_count"])
    unique_markets = {str(row["market_id"]) for row in eligible}
    unique_slugs = {str(row["slug"]) for row in eligible}
    if len(unique_markets) != len(eligible) or len(unique_slugs) != len(eligible):
        raise ValueError("post-#204 eligible index prefix identity duplicate")
    return {
        "index_entry_count": len(rows),
        "quality_valid_index_entry_count": sum(
            bool(row.get("capture_quality_valid")) for row in rows
        ),
        "minimum_eligible_index_sequence": minimum_sequence,
        "minimum_eligible_market_start_ts": minimum_market_start,
        "eligible_quality_valid_row_count": len(eligible),
        "eligible_sequence_start": int(eligible[0]["sequence"]) if eligible else None,
        "eligible_sequence_end": int(eligible[-1]["sequence"]) if eligible else None,
        "eligible_unique_market_count": len(unique_markets),
        "eligible_unique_slug_count": len(unique_slugs),
        "development_target_quality_valid_market_count": target,
        "development_markets_remaining": max(0, target - len(eligible)),
        "development_window_ready": len(eligible) >= target,
        "issue204_overlap_reason_distribution": {},
        "labels_outcomes_or_pnl_opened": False,
    }


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("finite numeric values are required")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "maximum": ordered[-1],
    }


def _attrition_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Conformal v5 target-free NO_TRADE attrition",
        "",
        f"- Decision groups: `{report['decision_group_count']}`",
        f"- Selected actions: `{json.dumps(report['selected_action_distribution'], sort_keys=True)}`",
        "- Guard-compatible groups with a positive raw trade: "
        f"`{report['decision_groups_with_guard_compatible_raw_positive_trade']}`",
        "- Groups with a positive conformal trade LCB: "
        f"`{report['decision_groups_with_positive_conformal_trade_lcb']}`",
        "- Outcome/label/PnL access: `false`",
        "",
        "## Action diagnostics",
        "",
        "| action | compatible | raw positive | positive LCB | penalty min | penalty max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for action in REQUIRED_ACTIONS:
        row = report["action_diagnostics"][action]
        penalty = row["calibration_penalty_summary"]
        lines.append(
            f"| {action} | {row['guard_compatible_row_count']} | "
            f"{row['guard_compatible_raw_positive_row_count']} | "
            f"{row['guard_compatible_positive_lcb_row_count']} | "
            f"{penalty['minimum']:.6f} | {penalty['maximum']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The report is target-free. It identifies policy/calibration alignment as the next "
            "research task and does not reinterpret the terminal #204 PnL result.",
            "",
        ]
    )
    return "\n".join(lines)


def _report_markdown(report: dict[str, Any]) -> str:
    prefix = report["collector_index_prefix_summary"]
    return "\n".join(
        [
            "# Policy-selected conformal net-return v6 preregistration",
            "",
            f"- Candidate: `{report['candidate_name']}`",
            f"- Preregistration passed: `{str(report['preregistration_passed']).lower()}`",
            f"- Post-#204 eligible markets already indexed: `{prefix['eligible_quality_valid_row_count']}`",
            f"- Development target: `{report['development_target_quality_valid_market_count']}`",
            f"- Remaining: `{prefix['development_markets_remaining']}`",
            "- #204 outcome/PnL used for fitting or tuning: `false`",
            "- New development targets opened: `false`",
            "- Paper/live/promotion/handoff unlock: `false`",
            "",
            "The development roles, model contract, sequential policy-selected conformal rule, "
            "and future side-only support/PnL gate are frozen before new target access.",
            "",
        ]
    )


def _verified_descriptor(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor missing")
    path = Path(str(value.get("path") or "")).resolve()
    digest = str(value.get("sha256") or "").lower()
    _verify_pin(path, digest, name)
    return {"path": str(path), "sha256": digest}


def _find_nonempty_fields(value: Any, fields: frozenset[str]) -> list[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in fields and nested not in (None, "", [], {}):
                found.add(key)
            found.update(_find_nonempty_fields(nested, fields))
    elif isinstance(value, list):
        for nested in value:
            found.update(_find_nonempty_fields(nested, fields))
    return sorted(found)


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected: str, name: str) -> None:
    _require_sha256(expected, name=name)
    if not path.is_file():
        raise ValueError(f"{name} missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")


def _require_sha256(value: str, *, name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _require_git_sha(value: str) -> None:
    text = str(value or "").lower()
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError("builder_git_commit must be a Git SHA-1")


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
