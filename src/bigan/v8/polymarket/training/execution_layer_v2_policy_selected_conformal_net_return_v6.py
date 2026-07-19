"""Pre-register the post-#204 policy-selected conformal net-return v6 study."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    FORBIDDEN_TARGET_FIELDS,
    _blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
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
DEVELOPMENT_WINDOW_REPORT_SCHEMA_VERSION = (
    "bigan-v8-policy-selected-conformal-v6-development-window-report-v1"
)
DEVELOPMENT_WINDOW_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-policy-selected-conformal-v6-development-window-manifest-v1"
)
DEVELOPMENT_SETTLED_INDEX_SCHEMA_VERSION = (
    "bigan-v8-policy-selected-conformal-v6-development-settled-index-v1"
)
DEVELOPMENT_SETTLEMENT_REPORT_SCHEMA_VERSION = (
    "bigan-v8-policy-selected-conformal-v6-development-settlement-report-v1"
)
DEVELOPMENT_SETTLEMENT_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-policy-selected-conformal-v6-development-settlement-manifest-v1"
)

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


@dataclass(frozen=True, slots=True)
class PolicySelectedConformalV6DevelopmentWindowConfig:
    """Pinned inputs for freezing the development roles before target access."""

    run_id: str
    output_dir: Path | str
    preregistration_manifest_path: Path | str
    expected_preregistration_manifest_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    builder_git_commit: str
    freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_preregistration_manifest_sha256",
            "expected_collector_index_sha256",
            "expected_feature_contract_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        _require_git_sha(self.builder_git_commit)
        if self.freeze_created_ts <= 0:
            raise ValueError("freeze_created_ts must be positive")
        for field in (
            "output_dir",
            "preregistration_manifest_path",
            "collector_index_path",
            "feature_contract_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


@dataclass(frozen=True, slots=True)
class PolicySelectedConformalV6DevelopmentSettlementConfig:
    """Pinned post-freeze read-only target finalization for v6 development roles."""

    run_id: str
    output_dir: Path | str
    development_window_manifest_path: Path | str
    expected_development_window_manifest_sha256: str
    builder_git_commit: str
    target_access_started_ts: int
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_development_window_manifest_sha256,
            name="expected_development_window_manifest_sha256",
        )
        _require_git_sha(self.builder_git_commit)
        if self.target_access_started_ts <= 0:
            raise ValueError("target_access_started_ts must be positive")
        if self.provider_timeout_seconds <= 0 or self.provider_http_timeout_seconds <= 0:
            raise ValueError("provider timeouts must be positive")
        if self.settlement_max_wait_seconds < 0:
            raise ValueError("settlement_max_wait_seconds must be non-negative")
        if self.settlement_poll_interval_seconds <= 0:
            raise ValueError("settlement_poll_interval_seconds must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        for field in ("output_dir", "development_window_manifest_path"):
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


def freeze_policy_selected_conformal_v6_development_window(
    config: PolicySelectedConformalV6DevelopmentWindowConfig,
    *,
    feature_materializer: Callable[
        [list[dict[str, Any]]],
        tuple[list[dict[str, Any]], list[dict[str, Any]]],
    ] = _materialize_selected_window_features,
    action_materializer: Callable[..., list[dict[str, Any]]] = _materialize_future_action_rows,
) -> dict[str, Any]:
    """Freeze the earliest post-#204 150/60/50 roles without opening targets."""

    prereg_path = config.preregistration_manifest_path.resolve()
    _verify_pin(
        prereg_path,
        config.expected_preregistration_manifest_sha256,
        "v6 preregistration manifest",
    )
    prereg = _load_json(prereg_path)
    _validate_preregistration_manifest_for_development_freeze(prereg)
    profile_descriptor = _verified_descriptor(prereg.get("profile"), "v6 profile")
    profile = _load_json(Path(profile_descriptor["path"]))
    validate_policy_selected_conformal_v6_profile(profile)
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        "feature contract",
    )
    if (
        config.expected_feature_contract_sha256
        != profile["frozen_upstream"]["feature_contract_sha256"]
    ):
        raise ValueError("feature contract pin does not match v6 profile")
    feature_contract = _load_json(feature_contract_path)
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    boundary_descriptor = _verified_descriptor(
        prereg.get("development_source_boundary"),
        "development source boundary",
    )
    prefix_descriptor = _verified_descriptor(
        prereg.get("collector_index_prefix"),
        "collector index prefix",
    )
    prefix_rows = load_and_validate_persistent_outcome_blind_index(
        Path(prefix_descriptor["path"])
    )

    index_path = config.collector_index_path.resolve()
    _verify_pin(index_path, config.expected_collector_index_sha256, "collector index")
    current_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    if len(current_rows) < len(prefix_rows) or current_rows[: len(prefix_rows)] != prefix_rows:
        raise ValueError("collector index prefix does not match preregistration snapshot")
    if _sha256_file(index_path) != config.expected_collector_index_sha256:
        raise ValueError("collector index changed during development freeze")

    source_boundary = _load_json(Path(boundary_descriptor["path"]))
    issue204_selected = _load_jsonl(
        Path(
            _verified_descriptor(
                _load_json(Path(prereg["issue204_window_manifest"]["path"])).get(
                    "selected_rows"
                ),
                "#204 selected rows",
            )["path"]
        )
    )
    exclusion = _prior_exclusion_summary(issue204_selected)
    if exclusion["identity_hash"] != source_boundary["issue204_exclusion_identity_hash"]:
        raise ValueError("#204 exclusion identity drift")

    development = profile["development_window"]
    minimum_sequence = int(development["minimum_eligible_index_sequence"])
    scan_cap = int(development["maximum_index_scan_count"])
    target_count = int(development["target_quality_valid_market_count"])
    scan_rows = [row for row in current_rows if int(row["sequence"]) >= minimum_sequence]
    considered_rows = scan_rows[:scan_cap]
    selected = []
    rejected_reasons = Counter()
    seen_markets: set[str] = set()
    seen_slugs: set[str] = set()
    seen_source_hashes: set[str] = set()
    scanned_row_count = 0
    for row in considered_rows:
        scanned_row_count += 1
        reasons = _development_row_rejection_reasons(
            row,
            profile=profile,
            exclusion=exclusion,
            seen_markets=seen_markets,
            seen_slugs=seen_slugs,
            seen_source_hashes=seen_source_hashes,
        )
        if reasons:
            rejected_reasons.update(reasons)
            continue
        selected.append(row)
        seen_markets.add(str(row["market_id"]))
        seen_slugs.add(str(row["slug"]))
        seen_source_hashes.add(str(row["source_row_hash"]))
        if len(selected) == target_count:
            break

    blockers = []
    if len(selected) != target_count:
        blockers.append("development_target_quality_valid_market_count_not_met")
    if selected and config.freeze_created_ts <= max(int(row["market_end_ts"]) for row in selected):
        blockers.append("development_markets_not_closed_before_freeze")
    if _find_nonempty_fields(selected, FORBIDDEN_TARGET_FIELDS):
        blockers.append("development_selected_rows_contain_target_fields")
    if any(row.get("labels_outcomes_or_pnl_opened") is not False for row in selected):
        blockers.append("development_selected_row_target_sealing_invalid")

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    index_snapshot_path = run_dir / "persistent_outcome_blind_round_index_snapshot.jsonl"
    shutil.copyfile(index_path, index_snapshot_path)
    if _sha256_file(index_snapshot_path) != config.expected_collector_index_sha256:
        raise ValueError("collector index snapshot hash mismatch")

    selected_path = run_dir / "conformal_v6_development_window_selected_rows.jsonl"
    role_rows_path = run_dir / "conformal_v6_development_role_assignment_rows.jsonl"
    feature_rows_path = run_dir / "conformal_v6_development_target_free_feature_rows.jsonl"
    action_rows_path = run_dir / "conformal_v6_development_target_free_five_action_rows.jsonl"
    role_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    opened_raw_descriptors: list[dict[str, Any]] = []
    if not blockers:
        _write_jsonl(selected_path, selected)
        role_rows = _assign_chronological_development_roles(selected, profile=profile)
        _write_jsonl(role_rows_path, role_rows)
        role_by_market = {
            str(row["market_id"]): str(row["development_role"]) for row in role_rows
        }
        feature_rows, opened_raw_descriptors = feature_materializer(selected)
        feature_rows = [
            {
                **row,
                "role": role_by_market[str(row["market_id"])],
                "development_role": role_by_market[str(row["market_id"])],
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
            }
            for row in feature_rows
        ]
        action_rows = action_materializer(
            feature_rows,
            selected_rows=selected,
            feature_columns=feature_columns,
        )
        action_rows = [
            _development_target_free_action_row(
                row,
                role=role_by_market[str(row["market_id"])],
            )
            for row in action_rows
        ]
        _validate_development_target_free_materialization(
            feature_rows,
            action_rows,
            selected_market_count=target_count,
            feature_columns=feature_columns,
        )
        _write_jsonl(feature_rows_path, feature_rows)
        _write_jsonl(action_rows_path, action_rows)

    role_counts = dict(sorted(Counter(row["development_role"] for row in role_rows).items()))
    report = {
        "schema_version": DEVELOPMENT_WINDOW_REPORT_SCHEMA_VERSION,
        "report_id": None,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "builder_git_commit": config.builder_git_commit,
        "freeze_created_ts": config.freeze_created_ts,
        "preregistration_manifest": _descriptor(prereg_path),
        "collector_index": _descriptor(index_snapshot_path),
        "collector_index_entry_count": len(current_rows),
        "collector_index_prefix_unchanged": True,
        "minimum_eligible_index_sequence": minimum_sequence,
        "maximum_index_scan_count": scan_cap,
        "scanned_post_boundary_row_count": scanned_row_count,
        "selected_market_count": len(selected),
        "target_market_count": target_count,
        "role_market_counts": role_counts,
        "selected_sequence_start": int(selected[0]["sequence"]) if selected else None,
        "selected_sequence_end": int(selected[-1]["sequence"]) if selected else None,
        "selected_window_start_ts": int(selected[0]["market_start_ts"]) if selected else None,
        "selected_window_end_ts": int(selected[-1]["market_end_ts"]) if selected else None,
        "issue204_market_slug_source_hash_overlap_count": 0,
        "rejected_reason_distribution": dict(sorted(rejected_reasons.items())),
        "target_free_feature_row_count": len(feature_rows),
        "target_free_five_action_row_count": len(action_rows),
        "opened_raw_feature_artifact_market_count": len(opened_raw_descriptors),
        "timestamp_causality_violation_count": 0,
        "target_free_feature_materialization_passed": not blockers,
        "labels_outcomes_or_pnl_opened": False,
        "raw_artifact_payloads_opened": bool(opened_raw_descriptors),
        "resolution_artifact_opened": False,
        "development_target_access_allowed_after_freeze": not blockers,
        "development_window_freeze_ready": not blockers,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v6_development_window_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "conformal_v6_development_window_report.md",
        _development_window_markdown(report),
    )
    manifest = {
        "schema_version": DEVELOPMENT_WINDOW_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "builder_git_commit": config.builder_git_commit,
        "freeze_created_ts": config.freeze_created_ts,
        "preregistration_manifest": _descriptor(prereg_path),
        "profile": profile_descriptor,
        "feature_contract": _descriptor(feature_contract_path),
        "development_source_boundary": boundary_descriptor,
        "collector_index_snapshot": _descriptor(index_snapshot_path),
        "selected_rows": _descriptor(selected_path) if selected_path.is_file() else None,
        "role_assignment_rows": _descriptor(role_rows_path) if role_rows_path.is_file() else None,
        "target_free_feature_rows": (
            _descriptor(feature_rows_path) if feature_rows_path.is_file() else None
        ),
        "target_free_five_action_rows": (
            _descriptor(action_rows_path) if action_rows_path.is_file() else None
        ),
        "report": _descriptor(report_path),
        "selected_market_count": len(selected),
        "role_market_counts": role_counts,
        "development_window_freeze_ready": not blockers,
        "development_target_accessed": False,
        "future_evaluation_attempted": False,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
    }
    manifest["development_window_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v6_development_window_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "selected_rows": selected,
        "role_rows": role_rows,
        "feature_rows": feature_rows,
        "action_rows": action_rows,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def build_policy_selected_conformal_v6_development_settled_corpus_index(
    config: PolicySelectedConformalV6DevelopmentSettlementConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    round_finalizer: Callable[..., list[dict[str, Any]]] = _finalize_selected_rounds,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Finalize quarantine copies only after the immutable development freeze."""

    window_path = config.development_window_manifest_path.resolve()
    _verify_pin(
        window_path,
        config.expected_development_window_manifest_sha256,
        "v6 development window manifest",
    )
    window = _load_json(window_path)
    _validate_development_window_manifest_for_settlement(window)
    selected_descriptor = _verified_descriptor(window.get("selected_rows"), "selected rows")
    roles_descriptor = _verified_descriptor(
        window.get("role_assignment_rows"),
        "development role assignment",
    )
    feature_descriptor = _verified_descriptor(
        window.get("target_free_feature_rows"),
        "target-free development features",
    )
    action_descriptor = _verified_descriptor(
        window.get("target_free_five_action_rows"),
        "target-free development actions",
    )
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    role_rows = _load_jsonl(Path(roles_descriptor["path"]))
    feature_rows = _load_jsonl(Path(feature_descriptor["path"]))
    action_rows = _load_jsonl(Path(action_descriptor["path"]))
    _validate_frozen_development_inputs_before_target_access(
        selected_rows,
        role_rows=role_rows,
        feature_rows=feature_rows,
        action_rows=action_rows,
    )
    freeze_created_ts = int(window["freeze_created_ts"])
    max_market_end_ts = max(int(row["market_end_ts"]) for row in selected_rows)
    if config.target_access_started_ts <= freeze_created_ts:
        raise ValueError("development target access attempted before window freeze")
    if config.target_access_started_ts <= max_market_end_ts:
        raise ValueError("development target access attempted before all markets closed")

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    marker = {
        "schema_version": (
            "bigan-v8-policy-selected-conformal-v6-development-target-access-marker-v1"
        ),
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "target_access_started_ts": config.target_access_started_ts,
        "development_window_freeze_created_ts": freeze_created_ts,
        "max_market_end_ts": max_market_end_ts,
        "development_window_manifest": _descriptor(window_path),
        "selected_rows": selected_descriptor,
        "role_assignment_rows": roles_descriptor,
        "target_free_feature_rows": feature_descriptor,
        "target_free_five_action_rows": action_descriptor,
        "target_access_started_after_freeze": True,
        "all_markets_closed_before_target_access": True,
        "source_outcome_blind_rounds_mutated": False,
        "policy_pnl_computed": False,
        "threshold_or_model_tuning_performed": False,
        "direct_training_corpus_exported": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "conformal_v6_development_target_access_started.json"
    _write_json(marker_path, marker)
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()

    factory = provider_factory or (
        lambda: PolymarketPublicHTTPRealCorpusProvider(
            max_markets=1,
            timeout_seconds=config.provider_timeout_seconds,
            http_timeout_seconds=config.provider_http_timeout_seconds,
            use_rest_orderbooks=False,
        )
    )
    success_by_market: dict[str, dict[str, Any]] = {}
    failure_by_market: dict[str, dict[str, Any]] = {}
    selected_by_market = {str(row["market_id"]): row for row in selected_rows}
    pending_rows = list(selected_rows)
    retry_market_ids: set[str] = set()
    settlement_attempt_count = 0
    deadline = monotonic_fn() + config.settlement_max_wait_seconds
    while pending_rows:
        settlement_attempt_count += 1
        attempt_results = round_finalizer(
            pending_rows,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=settlement_attempt_count,
        )
        retryable: set[str] = set()
        for result in attempt_results:
            market_id = str(result["market_id"])
            if result["settled_corpus_ready"]:
                success_by_market[market_id] = dict(result["index_entry"])
                failure_by_market.pop(market_id, None)
                continue
            failure = dict(result["failure"])
            failure_by_market[market_id] = failure
            if _is_retryable_settlement_failure(failure):
                retryable.add(market_id)
        if not retryable or monotonic_fn() >= deadline:
            break
        retry_market_ids.update(retryable)
        sleep_fn(config.settlement_poll_interval_seconds)
        pending_rows = [selected_by_market[market_id] for market_id in sorted(retryable)]

    role_by_market = {str(row["market_id"]): row for row in role_rows}
    settled_role_rows = []
    for market_id in sorted(success_by_market, key=lambda value: role_by_market[value]["sequence"]):
        entry = success_by_market[market_id]
        role_row = role_by_market[market_id]
        corpus_manifest = _verified_descriptor(entry.get("corpus_manifest"), "corpus manifest")
        settled_role_rows.append(
            {
                **entry,
                "market_id": market_id,
                "role": str(role_row["development_role"]),
                "development_role": str(role_row["development_role"]),
                "selection_rank": int(role_row["development_window_index"]),
                "development_role_index": int(role_row["development_role_index"]),
                "source_corpus_dir": str(Path(corpus_manifest["path"]).parent),
                "corpus_manifest": corpus_manifest,
                "outcomes_used_as_training_targets_only": True,
                "outcomes_used_as_decision_inputs": False,
                "policy_pnl_computed": False,
                "source_outcome_blind_round_mutated": False,
            }
        )
    settled_role_rows.sort(key=lambda row: int(row["selection_rank"]))
    unresolved = [
        failure_by_market[market_id]
        for market_id in sorted(set(selected_by_market) - set(success_by_market))
    ]
    ready = len(settled_role_rows) == len(selected_rows) == 260 and not unresolved
    blockers = [] if ready else ["development_settled_corpus_incomplete"]

    role_rows_path = run_dir / "conformal_v6_settled_development_role_rows.jsonl"
    _write_jsonl(role_rows_path, settled_role_rows)
    settled_index = {
        "schema_version": DEVELOPMENT_SETTLED_INDEX_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "target_access_started_ts": config.target_access_started_ts,
        "development_window_manifest": _descriptor(window_path),
        "target_access_marker": _descriptor(marker_path),
        "settled_market_count": len(settled_role_rows),
        "unresolved_market_count": len(unresolved),
        "role_market_counts": dict(
            sorted(Counter(row["development_role"] for row in settled_role_rows).items())
        ),
        "settled_role_rows": _descriptor(role_rows_path),
        "settled_corpus_ready": ready,
        "source_outcome_blind_rounds_mutated": False,
        "direct_training_corpus_exported": False,
        "policy_pnl_computed": False,
        "threshold_or_model_tuning_performed": False,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
    }
    settled_index["settled_index_id"] = canonical_json_sha256(settled_index)
    index_path = run_dir / "conformal_v6_development_settled_corpus_index.json"
    _write_json(index_path, settled_index)
    reason_distribution = Counter(
        str(reason)
        for row in unresolved
        for reason in row.get("reason_codes", ["development_settlement_unresolved"])
    )
    report = {
        "schema_version": DEVELOPMENT_SETTLEMENT_REPORT_SCHEMA_VERSION,
        "report_id": None,
        "run_id": config.run_id,
        "selected_market_count": len(selected_rows),
        "settled_market_count": len(settled_role_rows),
        "unresolved_market_count": len(unresolved),
        "unresolved_reason_distribution": dict(sorted(reason_distribution.items())),
        "settlement_attempt_count": settlement_attempt_count,
        "settlement_retry_market_count": len(retry_market_ids),
        "role_market_counts": settled_index["role_market_counts"],
        "all_markets_closed_before_target_access": True,
        "source_outcome_blind_rounds_mutated": False,
        "targets_available_for_fixed_training_roles_only": ready,
        "policy_pnl_computed": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_204_pnl_for_tuning": False,
        "development_settled_corpus_ready": ready,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v6_development_settlement_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "conformal_v6_development_settlement_report.md",
        _development_settlement_markdown(report),
    )
    manifest = {
        "schema_version": DEVELOPMENT_SETTLEMENT_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "development_window_manifest": _descriptor(window_path),
        "target_access_marker": _descriptor(marker_path),
        "settled_role_rows": _descriptor(role_rows_path),
        "settled_corpus_index": _descriptor(index_path),
        "report": _descriptor(report_path),
        "development_settled_corpus_ready": ready,
        "policy_pnl_computed": False,
        "source_outcome_blind_rounds_mutated": False,
        "direct_training_corpus_exported": False,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
    }
    manifest["development_settlement_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v6_development_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "settled_role_rows": settled_role_rows,
        "unresolved_rows": unresolved,
        "settled_index": settled_index,
        "settled_index_path": index_path,
        "settled_index_sha256": _sha256_file(index_path),
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


def _validate_development_window_manifest_for_settlement(
    manifest: dict[str, Any],
) -> None:
    checks = {
        "schema": manifest.get("schema_version") == DEVELOPMENT_WINDOW_MANIFEST_SCHEMA_VERSION,
        "candidate": manifest.get("candidate_name") == CANDIDATE_NAME,
        "ready": manifest.get("development_window_freeze_ready") is True,
        "market_count": manifest.get("selected_market_count") == 260,
        "roles": manifest.get("role_market_counts")
        == {
            "calibration_check": 50,
            "conformal_calibration": 60,
            "point_model_fit": 150,
        },
        "target_not_accessed": manifest.get("development_target_accessed") is False,
        "future_not_attempted": manifest.get("future_evaluation_attempted") is False,
        "no_blockers": manifest.get("blocking_reason_codes") == [],
        "safety": all(
            manifest.get(field) == expected
            for field, expected in _blocked_safety_fields().items()
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("development window is not eligible for settlement: " + ", ".join(blockers))


def _validate_frozen_development_inputs_before_target_access(
    selected_rows: list[dict[str, Any]],
    *,
    role_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> None:
    if len(selected_rows) != 260 or len({str(row["market_id"]) for row in selected_rows}) != 260:
        raise ValueError("development selected rows are not the frozen 260-market window")
    if len(role_rows) != 260:
        raise ValueError("development role assignment row count mismatch")
    role_counts = Counter(str(row["development_role"]) for row in role_rows)
    if role_counts != Counter(
        {"point_model_fit": 150, "conformal_calibration": 60, "calibration_check": 50}
    ):
        raise ValueError("development role assignment counts mismatch")
    selected_markets = {str(row["market_id"]) for row in selected_rows}
    if {str(row["market_id"]) for row in role_rows} != selected_markets:
        raise ValueError("development role market identity mismatch")
    if len(feature_rows) != 1040 or {str(row["market_id"]) for row in feature_rows} != selected_markets:
        raise ValueError("development target-free feature coverage mismatch")
    if len(action_rows) != 5200 or {str(row["market_id"]) for row in action_rows} != selected_markets:
        raise ValueError("development target-free action coverage mismatch")
    if _find_nonempty_fields(selected_rows, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("development selected rows contain target fields")
    if _find_nonempty_fields(feature_rows, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("development feature rows contain target fields")
    if _find_nonempty_fields(action_rows, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("development action rows contain target fields")
    if any(int(row["max_input_ts"]) > int(row["decision_ts"]) for row in feature_rows):
        raise ValueError("development feature causality violation before target access")
    if any(row.get("target_used_as_decision_input") is not False for row in action_rows):
        raise ValueError("development action target usage contract invalid")


def _development_settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Policy-selected conformal v6 development settlement",
            "",
            f"- Settled corpus ready: `{str(report['development_settled_corpus_ready']).lower()}`",
            f"- Settled / selected markets: `{report['settled_market_count']} / {report['selected_market_count']}`",
            f"- Unresolved markets: `{report['unresolved_market_count']}`",
            f"- Role counts: `{json.dumps(report['role_market_counts'], sort_keys=True)}`",
            f"- Blocking reasons: `{json.dumps(report['blocking_reason_codes'])}`",
            "- Source outcome-blind rounds mutated: `false`",
            "- Policy PnL computed: `false`",
            "- Paper/live/promotion/handoff unlock: `false`",
            "",
        ]
    )


def _validate_preregistration_manifest_for_development_freeze(
    manifest: dict[str, Any],
) -> None:
    checks = {
        "schema": manifest.get("schema_version") == PREREG_MANIFEST_SCHEMA_VERSION,
        "candidate": manifest.get("candidate_name") == CANDIDATE_NAME,
        "passed": manifest.get("preregistration_passed") is True,
        "window_not_previously_frozen": manifest.get("development_window_frozen") is False,
        "target_not_accessed": manifest.get("new_development_target_accessed") is False,
        "future_not_attempted": manifest.get("future_evaluation_attempted") is False,
        "rerun_not_allowed": manifest.get("result_dependent_rerun_or_tuning_allowed") is False,
        "safety": all(
            manifest.get(field) == expected
            for field, expected in _blocked_safety_fields().items()
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("v6 preregistration manifest invalid: " + ", ".join(blockers))


def _development_row_rejection_reasons(
    row: dict[str, Any],
    *,
    profile: dict[str, Any],
    exclusion: dict[str, Any],
    seen_markets: set[str],
    seen_slugs: set[str],
    seen_source_hashes: set[str],
) -> list[str]:
    reasons = []
    if row.get("capture_quality_valid") is not True:
        reasons.append("capture_quality_invalid")
    if int(row.get("market_start_ts") or 0) < int(
        profile["development_window"]["minimum_eligible_market_start_ts"]
    ):
        reasons.append("market_start_before_development_boundary")
    market_id = str(row.get("market_id") or "")
    slug = str(row.get("slug") or "")
    source_hash = str(row.get("source_row_hash") or "")
    if market_id in exclusion["market_ids"]:
        reasons.append("issue204_market_overlap")
    if slug in exclusion["slugs"]:
        reasons.append("issue204_slug_overlap")
    if source_hash in exclusion["source_row_hashes"]:
        reasons.append("issue204_source_row_hash_overlap")
    if market_id in seen_markets:
        reasons.append("development_market_duplicate")
    if slug in seen_slugs:
        reasons.append("development_slug_duplicate")
    if source_hash in seen_source_hashes:
        reasons.append("development_source_row_hash_duplicate")
    if row.get("labels_outcomes_or_pnl_opened") is not False:
        reasons.append("development_row_target_sealing_invalid")
    if _find_nonempty_fields(row, FORBIDDEN_TARGET_FIELDS):
        reasons.append("development_row_forbidden_target_field")
    source_safety_invalid = any(
        (
            row.get(field) not in (None, False)
            if field == "paper_candidate_allowed"
            else row.get(field) != expected
        )
        for field, expected in _blocked_safety_fields().items()
    )
    if source_safety_invalid:
        reasons.append("development_row_safety_invalid")
    return sorted(set(reasons))


def _assign_chronological_development_roles(
    selected: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    role_config = profile["chronological_roles"]
    fit_count = int(role_config["point_model_fit_market_count"])
    conformal_count = int(role_config["conformal_calibration_market_count"])
    check_count = int(role_config["calibration_check_market_count"])
    if len(selected) != fit_count + conformal_count + check_count:
        raise ValueError("development role counts do not cover selected markets")
    role_rows = []
    for index, row in enumerate(selected):
        if index < fit_count:
            role = "point_model_fit"
            role_index = index + 1
        elif index < fit_count + conformal_count:
            role = "conformal_calibration"
            role_index = index - fit_count + 1
        else:
            role = "calibration_check"
            role_index = index - fit_count - conformal_count + 1
        role_rows.append(
            {
                **row,
                "development_role": role,
                "development_role_index": role_index,
                "development_window_index": index + 1,
                "labels_outcomes_or_pnl_opened": False,
            }
        )
    role_order = [row["development_role"] for row in role_rows]
    expected = (
        ["point_model_fit"] * fit_count
        + ["conformal_calibration"] * conformal_count
        + ["calibration_check"] * check_count
    )
    if role_order != expected:
        raise ValueError("development role assignment is not chronological")
    return role_rows


def _development_target_free_action_row(
    row: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    updated = {
        key: value
        for key, value in row.items()
        if key not in {"action_row_sha256", "future_action_row_sha256"}
    }
    updated.update(
        {
            "role": role,
            "development_role": role,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
        }
    )
    updated["action_row_sha256"] = canonical_json_sha256(updated)
    return updated


def _validate_development_target_free_materialization(
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    *,
    selected_market_count: int,
    feature_columns: tuple[str, ...],
) -> None:
    expected_feature_count = selected_market_count * 4
    if len(feature_rows) != expected_feature_count:
        raise ValueError("development target-free feature row count mismatch")
    if len(action_rows) != expected_feature_count * len(REQUIRED_ACTIONS):
        raise ValueError("development target-free five-action row count mismatch")
    if len({str(row["market_id"]) for row in feature_rows}) != selected_market_count:
        raise ValueError("development feature market coverage mismatch")
    if any(int(row["max_input_ts"]) > int(row["decision_ts"]) for row in feature_rows):
        raise ValueError("development target-free feature causality violation")
    if _find_nonempty_fields(feature_rows, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("development target-free features contain target fields")
    if _find_nonempty_fields(action_rows, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("development target-free actions contain target fields")
    if any(column not in row for row in action_rows for column in feature_columns):
        raise ValueError("development action row feature contract mismatch")
    actions_by_decision: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in action_rows:
        actions_by_decision[(str(row["market_id"]), int(row["decision_ts"]))].add(
            str(row["action"])
        )
    if any(actions != set(REQUIRED_ACTIONS) for actions in actions_by_decision.values()):
        raise ValueError("development target-free action grid incomplete")
    if len(actions_by_decision) != expected_feature_count:
        raise ValueError("development target-free decision group count mismatch")


def _development_window_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Policy-selected conformal v6 development window",
            "",
            f"- Freeze ready: `{str(report['development_window_freeze_ready']).lower()}`",
            f"- Selected markets: `{report['selected_market_count']}`",
            f"- Role counts: `{json.dumps(report['role_market_counts'], sort_keys=True)}`",
            f"- Sequence range: `{report['selected_sequence_start']}..{report['selected_sequence_end']}`",
            f"- Blocking reasons: `{json.dumps(report['blocking_reason_codes'])}`",
            "- Labels/outcomes/PnL opened: `false`",
            "- Paper/live/promotion/handoff unlock: `false`",
            "",
        ]
    )


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
