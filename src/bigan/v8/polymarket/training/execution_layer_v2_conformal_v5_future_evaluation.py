"""Pre-register and bind the strictly-later #204 conformal-v5 future evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _market_bootstrap_interval,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    CANDIDATE_NAME,
    validate_guard_compatible_conformal_net_return_v5_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    WINDOW_MANIFEST_SCHEMA_VERSION,
    load_and_validate_persistent_outcome_blind_index,
    validate_persistent_outcome_blind_collector_protocol,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-conformal-v5-strict-future-evaluation-profile-v1"
PRELABEL_AUDIT_SCHEMA_VERSION = "bigan-v8-conformal-v5-future-pre-label-audit-v1"
PREREG_REPORT_SCHEMA_VERSION = "bigan-v8-conformal-v5-future-preregistration-report-v1"
PREREG_MANIFEST_SCHEMA_VERSION = "bigan-v8-conformal-v5-future-preregistration-manifest-v1"
SOURCE_BOUNDARY_SCHEMA_VERSION = "bigan-v8-outcome-blind-source-boundary-v1"
BINDING_REPORT_SCHEMA_VERSION = "bigan-v8-conformal-v5-future-window-binding-report-v1"
BINDING_MANIFEST_SCHEMA_VERSION = "bigan-v8-conformal-v5-future-window-binding-manifest-v1"

EXPECTED_ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "NO_TRADE",
)
FORBIDDEN_TARGET_FIELDS = frozenset(
    {
        "accepted_bet_net_pnl",
        "evaluation_target_net_pnl_per_contract",
        "final_outcome",
        "future_price",
        "future_return",
        "gross_pnl",
        "label",
        "net_pnl",
        "oracle_action",
        "realized_pnl",
        "resolved_outcome",
        "settlement_outcome",
        "settlement_pnl",
        "target_net_pnl_per_contract",
        "total_net_pnl_per_notional",
        "winning_outcome",
    }
)


@dataclass(frozen=True, slots=True)
class ConformalV5FuturePreRegistrationConfig:
    """Pinned inputs opened before any future collection target or prediction."""

    run_id: str
    output_dir: Path | str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    candidate_manifest_path: Path | str
    expected_candidate_manifest_sha256: str
    collector_protocol_path: Path | str
    expected_collector_protocol_sha256: str
    builder_git_commit: str
    preregistration_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_evaluation_profile_sha256",
            "expected_candidate_manifest_sha256",
            "expected_collector_protocol_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        _require_git_sha(self.builder_git_commit)
        if self.preregistration_created_ts <= 0:
            raise ValueError("preregistration_created_ts must be positive")
        for field in (
            "output_dir",
            "evaluation_profile_path",
            "candidate_manifest_path",
            "collector_protocol_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


@dataclass(frozen=True, slots=True)
class ConformalV5FutureWindowBindingConfig:
    """Inputs for fail-closed window binding before feature/prediction access."""

    run_id: str
    output_dir: Path | str
    preregistration_manifest_path: Path | str
    expected_preregistration_manifest_sha256: str
    window_manifest_path: Path | str
    expected_window_manifest_sha256: str
    builder_git_commit: str
    binding_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_preregistration_manifest_sha256",
            "expected_window_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        _require_git_sha(self.builder_git_commit)
        if self.binding_created_ts <= 0:
            raise ValueError("binding_created_ts must be positive")
        for field in (
            "output_dir",
            "preregistration_manifest_path",
            "window_manifest_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def validate_conformal_v5_future_evaluation_profile(profile: dict[str, Any]) -> None:
    """Reject any drift from the pre-registered side-only future gate."""

    candidate = dict(profile.get("issue_203_candidate") or {})
    collection = dict(profile.get("issue_192_collection") or {})
    sequence = dict(profile.get("prediction_and_settlement_sequence") or {})
    execution = dict(profile.get("frozen_execution") or {})
    gates = dict(profile.get("support_and_pnl_gates") or {})
    safety = dict(profile.get("safety") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "frozen": profile.get("frozen") is True,
        "candidate": candidate.get("candidate_name") == CANDIDATE_NAME,
        "candidate_commit": _is_git_sha(candidate.get("implementation_commit")),
        "candidate_freeze": int(candidate.get("candidate_freeze_created_ts") or 0) > 0,
        "candidate_hashes": all(
            _is_sha256(candidate.get(field))
            for field in (
                "candidate_manifest_sha256",
                "model_sha256",
                "policy_dataset_hash",
                "split_hash",
                "fit_profile_sha256",
                "calibration_artifact_sha256",
                "future_evaluation_protocol_sha256",
                "role_assignment_manifest_sha256",
                "role_assignment_rows_sha256",
            )
        ),
        "fit_market_count": candidate.get("fit_market_count") == 135,
        "calibration_market_count": candidate.get("conformal_calibration_market_count") == 60,
        "source_market_count": candidate.get("source_market_count") == 195,
        "collection_commit": _is_git_sha(collection.get("collector_commit")),
        "collection_protocol": _is_sha256(collection.get("collector_protocol_sha256")),
        "eligible_collection": collection.get("eligible_collection")
        == "issue_192_strictly_later_persistent_window_only",
        "issue190_ineligible": collection.get("issue_190_collection_eligible") is False,
        "window_sizing": collection.get("target_quality_valid_market_count") == 220
        and collection.get("maximum_index_scan_count") == 340,
        "window_selection": collection.get("selection_method")
        == "earliest_quality_valid_strictly_later_disjoint_rows",
        "outcome_blind_selection": collection.get("result_dependent_extension_allowed") is False
        and collection.get("labels_outcomes_or_pnl_opened_for_selection") is False,
        "access_sequence": all(
            sequence.get(field) is True
            for field in (
                "candidate_binding_before_feature_materialization",
                "window_binding_before_feature_materialization",
                "target_stripped_prediction_before_outcome_access",
                "accepted_bet_decision_freeze_before_outcome_access",
                "settlement_resolution_after_decision_freeze_only",
                "single_use_holdout",
            )
        )
        and sequence.get("future_result_driven_rerun_or_tuning_allowed") is False,
        "execution_frozen": execution and all(value is False for value in execution.values()),
        "support": gates.get("minimum_guard_accepted_bet_count") == 88
        and gates.get("minimum_guard_accepted_unique_market_count") == 88
        and gates.get("minimum_supported_side_market_count") == 10,
        "sides": gates.get("required_supported_sides") == ["UP", "DOWN"],
        "side_only": gates.get("pnl_hard_gate_aggregation") == "selected_side_buy_up_buy_down_only"
        and gates.get("action_and_action_family_pnl_diagnostic_only") is True,
        "positive_gates": all(
            float(gates.get(field, -1.0)) == 0.0
            for field in (
                "accepted_bet_total_post_cost_pnl_minimum_exclusive",
                "supported_side_post_cost_pnl_minimum_exclusive",
                "candidate_minus_matched_baseline_pnl_minimum_exclusive",
                "candidate_minus_baseline_bootstrap_lcb_minimum_exclusive",
                "largest_winner_removed_pnl_minimum_exclusive",
            )
        ),
        "bootstrap": gates.get("bootstrap_unit") == "market_id"
        and gates.get("bootstrap_resample_count") == 5000
        and gates.get("bootstrap_seed") == 21080724,
        "safety": safety == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("future evaluation profile validation failed: " + ", ".join(blockers))


def pre_register_conformal_v5_future_evaluation(
    config: ConformalV5FuturePreRegistrationConfig,
) -> dict[str, Any]:
    """Freeze candidate and prior identities before future data can be selected."""

    profile_path = config.evaluation_profile_path.resolve()
    candidate_path = config.candidate_manifest_path.resolve()
    collector_path = config.collector_protocol_path.resolve()
    _verify_pin(profile_path, config.expected_evaluation_profile_sha256, "evaluation profile")
    _verify_pin(candidate_path, config.expected_candidate_manifest_sha256, "candidate manifest")
    _verify_pin(collector_path, config.expected_collector_protocol_sha256, "collector protocol")
    profile = _load_json(profile_path)
    validate_conformal_v5_future_evaluation_profile(profile)
    collector = _load_json(collector_path)
    validate_persistent_outcome_blind_collector_protocol(collector)
    candidate = _load_json(candidate_path)
    lineage = _validate_candidate_lineage(candidate, profile=profile)
    if collector.get("labels_outcomes_or_pnl_opened") is not False:
        raise ValueError("collector protocol outcome sealing is invalid")

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    prelabel_audit = {
        "schema_version": PRELABEL_AUDIT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "preregistration_created_ts": config.preregistration_created_ts,
        "candidate_manifest": _descriptor(candidate_path),
        "evaluation_profile": _descriptor(profile_path),
        "collector_protocol": _descriptor(collector_path),
        "candidate_lineage_hashes_verified": True,
        "role_assignment_rows_content_opened": False,
        "future_collector_index_opened": False,
        "future_window_opened": False,
        "future_features_opened": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "prediction_attempted": False,
        **_blocked_safety_fields(),
    }
    prelabel_audit["audit_id"] = canonical_json_sha256(prelabel_audit)
    prelabel_path = run_dir / "pre_label_access_lineage_audit.json"
    _write_json(prelabel_path, prelabel_audit)

    role_rows_path = Path(lineage["role_assignment_rows"]["path"])
    role_rows = _load_jsonl(role_rows_path)
    _validate_prior_role_rows(role_rows, profile=profile)
    prior_market_ids = sorted({str(row["market_id"]) for row in role_rows})
    prior_slugs = sorted(
        {
            Path(str(row["source_corpus_dir"])).name
            for row in role_rows
            if row.get("source_corpus_dir")
        }
    )
    prior_source_row_hashes = sorted(canonical_json_sha256(row) for row in role_rows)
    max_prior_decision_ts = max(int(row["maximum_decision_ts"]) for row in role_rows)
    candidate_freeze_ts = int(profile["issue_203_candidate"]["candidate_freeze_created_ts"])
    minimum_collection_ts = max(max_prior_decision_ts + 1, candidate_freeze_ts + 1)
    source_boundary = {
        "schema_version": SOURCE_BOUNDARY_SCHEMA_VERSION,
        "minimum_collection_decision_ts": minimum_collection_ts,
        "max_prior_decision_ts": max_prior_decision_ts,
        "candidate_freeze_created_ts": candidate_freeze_ts,
        "prior_market_ids": prior_market_ids,
        "prior_slugs": prior_slugs,
        "prior_source_row_hashes": prior_source_row_hashes,
        "prior_reference_hash": canonical_json_sha256(
            {
                "prior_market_ids": prior_market_ids,
                "prior_slugs": prior_slugs,
                "prior_source_row_hashes": prior_source_row_hashes,
            }
        ),
        "candidate_manifest": _descriptor(candidate_path),
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    boundary_path = run_dir / "conformal_v5_future_source_boundary_manifest.json"
    _write_json(boundary_path, source_boundary)
    report = {
        "schema_version": PREREG_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "preregistration_created_ts": config.preregistration_created_ts,
        "candidate_name": CANDIDATE_NAME,
        "candidate_freeze_created_ts": candidate_freeze_ts,
        "minimum_collection_decision_ts": minimum_collection_ts,
        "max_prior_decision_ts": max_prior_decision_ts,
        "prior_market_count": len(prior_market_ids),
        "fit_market_count": profile["issue_203_candidate"]["fit_market_count"],
        "conformal_calibration_market_count": profile["issue_203_candidate"][
            "conformal_calibration_market_count"
        ],
        "target_quality_valid_market_count": profile["issue_192_collection"][
            "target_quality_valid_market_count"
        ],
        "maximum_index_scan_count": profile["issue_192_collection"]["maximum_index_scan_count"],
        "minimum_guard_accepted_unique_market_count": profile["support_and_pnl_gates"][
            "minimum_guard_accepted_unique_market_count"
        ],
        "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
        "action_and_action_family_pnl_diagnostic_only": True,
        "issue_190_collection_eligible": False,
        "eligible_collection": "issue_192_strictly_later_persistent_window_only",
        "future_labels_outcomes_or_pnl_opened": False,
        "prediction_attempted": False,
        "pre_registration_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v5_future_preregistration_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "conformal_v5_future_preregistration_report.md",
        _preregistration_markdown(report),
    )
    manifest = {
        "schema_version": PREREG_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "preregistration_created_ts": config.preregistration_created_ts,
        "candidate_name": CANDIDATE_NAME,
        "evaluation_profile": _descriptor(profile_path),
        "candidate_manifest": _descriptor(candidate_path),
        "candidate_model": lineage["model"],
        "candidate_calibration_artifact": lineage["calibration_artifact"],
        "candidate_future_protocol": lineage["future_protocol"],
        "role_assignment_manifest": lineage["role_assignment_manifest"],
        "role_assignment_rows": lineage["role_assignment_rows"],
        "collector_protocol": _descriptor(collector_path),
        "source_boundary_manifest": _descriptor(boundary_path),
        "pre_label_access_audit": _descriptor(prelabel_path),
        "report": _descriptor(report_path),
        "candidate_freeze_created_ts": candidate_freeze_ts,
        "minimum_collection_decision_ts": minimum_collection_ts,
        "prior_market_count": len(prior_market_ids),
        "target_quality_valid_market_count": 220,
        "maximum_index_scan_count": 340,
        "minimum_guard_accepted_unique_market_count": 88,
        "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
        "action_and_action_family_pnl_diagnostic_only": True,
        "future_labels_outcomes_or_pnl_opened": False,
        "prediction_attempted": False,
        "pre_registration_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    manifest["pre_registration_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v5_future_preregistration_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "source_boundary_path": boundary_path,
        "source_boundary_sha256": _sha256_file(boundary_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def bind_conformal_v5_future_window_before_prediction(
    config: ConformalV5FutureWindowBindingConfig,
) -> dict[str, Any]:
    """Bind one immutable #192 window and stop before feature or target access."""

    prereg_path = config.preregistration_manifest_path.resolve()
    window_path = config.window_manifest_path.resolve()
    _verify_pin(prereg_path, config.expected_preregistration_manifest_sha256, "preregistration")
    _verify_pin(window_path, config.expected_window_manifest_sha256, "window manifest")
    prereg = _load_json(prereg_path)
    if prereg.get("schema_version") != PREREG_MANIFEST_SCHEMA_VERSION:
        raise ValueError("preregistration schema mismatch")
    if prereg.get("pre_registration_ready") is not True:
        raise ValueError("preregistration is not ready")
    profile_path = Path(_verified_descriptor(prereg["evaluation_profile"], "profile")["path"])
    profile = _load_json(profile_path)
    validate_conformal_v5_future_evaluation_profile(profile)
    boundary_path = Path(
        _verified_descriptor(prereg["source_boundary_manifest"], "source boundary")["path"]
    )
    boundary = _load_json(boundary_path)
    window = _load_json(window_path)
    blockers = _window_binding_blockers(
        prereg=prereg,
        profile=profile,
        boundary=boundary,
        window=window,
    )
    if blockers:
        raise ValueError("future window binding failed before prediction: " + ", ".join(blockers))
    selected_descriptor = _verified_descriptor(window["selected_rows"], "selected rows")
    index_descriptor = _verified_descriptor(window["index"], "collector index")
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    index_rows = load_and_validate_persistent_outcome_blind_index(Path(index_descriptor["path"]))
    blockers = _selected_window_blockers(
        selected_rows=selected_rows,
        index_rows=index_rows,
        boundary=boundary,
        profile=profile,
    )
    if blockers:
        raise ValueError("future selected rows failed before prediction: " + ", ".join(blockers))

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    report = {
        "schema_version": BINDING_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "binding_created_ts": config.binding_created_ts,
        "candidate_name": CANDIDATE_NAME,
        "selected_market_count": len(selected_rows),
        "selected_window_start_ts": min(
            int(row["scheduled_round_start_ts"]) for row in selected_rows
        ),
        "selected_window_end_ts": max(
            int(row["scheduled_round_start_ts"]) for row in selected_rows
        ),
        "strictly_later_than_candidate_freeze": True,
        "market_slug_decision_and_source_hash_disjoint": True,
        "collector_commit_verified": True,
        "collector_index_hash_chain_verified": True,
        "raw_artifact_hashes_verified_by_window_freezer": True,
        "feature_materialization_attempted": False,
        "prediction_attempted": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "candidate_window_binding_passed": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v5_future_candidate_binding_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "conformal_v5_future_candidate_binding_report.md",
        _binding_markdown(report),
    )
    manifest = {
        "schema_version": BINDING_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "binding_created_ts": config.binding_created_ts,
        "candidate_name": CANDIDATE_NAME,
        "preregistration_manifest": _descriptor(prereg_path),
        "candidate_manifest": prereg["candidate_manifest"],
        "candidate_model": prereg["candidate_model"],
        "candidate_calibration_artifact": prereg["candidate_calibration_artifact"],
        "window_manifest": _descriptor(window_path),
        "collector_index": index_descriptor,
        "selected_rows": selected_descriptor,
        "source_boundary_manifest": _descriptor(boundary_path),
        "report": _descriptor(report_path),
        "selected_market_count": len(selected_rows),
        "candidate_window_binding_passed": True,
        "feature_materialization_attempted": False,
        "prediction_attempted": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "single_use_holdout": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    manifest["candidate_binding_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v5_future_candidate_binding_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def build_conformal_v5_side_only_future_pnl_gate(
    evaluation_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    decision_freeze_sha256: str,
) -> dict[str, Any]:
    """Evaluate frozen accepted bets by side; action/family remain diagnostics."""

    validate_conformal_v5_future_evaluation_profile(profile)
    _require_sha256(decision_freeze_sha256, name="decision_freeze_sha256")
    gates = dict(profile["support_and_pnl_gates"])
    accepted = [row for row in evaluation_rows if row.get("execution_guard_order_allowed") is True]
    accepted_markets = sorted({str(row.get("market_id") or "") for row in accepted})
    candidate_by_market = dict.fromkeys(accepted_markets, 0.0)
    baseline_by_market = dict.fromkeys(accepted_markets, 0.0)
    for row in accepted:
        market_id = str(row.get("market_id") or "")
        candidate_by_market[market_id] += float(row["accepted_bet_net_pnl"])
        baseline_by_market[market_id] += float(row["matched_baseline_net_pnl"])
    delta_by_market = {
        market_id: candidate_by_market[market_id] - baseline_by_market[market_id]
        for market_id in accepted_markets
    }
    bootstrap = _market_bootstrap_interval(
        list(delta_by_market.values()),
        resample_count=int(gates["bootstrap_resample_count"]),
        confidence_level=float(gates["bootstrap_confidence_level"]),
        seed=int(gates["bootstrap_seed"]),
    )
    by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        side = str(row.get("selected_side") or "")
        action = str(row.get("executed_action") or "")
        by_side[side].append(row)
        by_action[action].append(row)
        by_family[_action_family(action)].append(row)
    side_metrics = {
        side: _accepted_group_metrics(rows, diagnostic_only=False)
        for side, rows in sorted(by_side.items())
        if side in {"UP", "DOWN"}
    }
    action_metrics = {
        action: _accepted_group_metrics(rows, diagnostic_only=True)
        for action, rows in sorted(by_action.items())
    }
    family_metrics = {
        family: _accepted_group_metrics(rows, diagnostic_only=True)
        for family, rows in sorted(by_family.items())
    }
    candidate_pnl = float(sum(candidate_by_market.values()))
    baseline_pnl = float(sum(baseline_by_market.values()))
    delta_pnl = candidate_pnl - baseline_pnl
    largest_winner = max(candidate_by_market.values(), default=0.0)
    largest_winner_removed_pnl = candidate_pnl - max(largest_winner, 0.0)
    required_sides = list(gates["required_supported_sides"])
    side_support_and_pnl_passed = all(
        side in side_metrics
        and side_metrics[side]["accepted_unique_market_count"]
        >= int(gates["minimum_supported_side_market_count"])
        and side_metrics[side]["accepted_bet_net_pnl_sum"]
        > float(gates["supported_side_post_cost_pnl_minimum_exclusive"])
        for side in required_sides
    )
    safety_rows_passed = all(
        row.get("settlement_resolved") is True
        and row.get("target_joined_after_decision_freeze") is True
        and row.get("target_used_as_decision_input") is False
        and row.get("forbidden_outcome_field_used_for_decision") is False
        and row.get("feature_causality_violation") is False
        and row.get("provenance_violation") is False
        and row.get("runtime_state_violation") is False
        for row in accepted
    )
    checks = {
        "minimum_guard_accepted_bet_support": len(accepted)
        >= int(gates["minimum_guard_accepted_bet_count"]),
        "minimum_guard_accepted_unique_market_support": len(accepted_markets)
        >= int(gates["minimum_guard_accepted_unique_market_count"]),
        "supported_side_post_cost_pnl_gate": side_support_and_pnl_passed,
        "accepted_bet_total_post_cost_pnl_positive": candidate_pnl
        > float(gates["accepted_bet_total_post_cost_pnl_minimum_exclusive"]),
        "candidate_exceeds_matched_baseline": delta_pnl
        > float(gates["candidate_minus_matched_baseline_pnl_minimum_exclusive"]),
        "candidate_minus_baseline_bootstrap_lcb_positive": bootstrap["lower_confidence_bound"]
        > float(gates["candidate_minus_baseline_bootstrap_lcb_minimum_exclusive"]),
        "largest_winner_removed_pnl_positive": largest_winner_removed_pnl
        > float(gates["largest_winner_removed_pnl_minimum_exclusive"]),
        "settlement_causality_provenance_and_runtime_safety": safety_rows_passed,
    }
    reason_map = {
        "minimum_guard_accepted_bet_support": "insufficient_guard_accepted_bet_support",
        "minimum_guard_accepted_unique_market_support": (
            "insufficient_guard_accepted_unique_market_support"
        ),
        "supported_side_post_cost_pnl_gate": "supported_side_post_cost_pnl_gate_failed",
        "accepted_bet_total_post_cost_pnl_positive": "accepted_bet_total_post_cost_pnl_not_positive",
        "candidate_exceeds_matched_baseline": "candidate_does_not_exceed_matched_baseline",
        "candidate_minus_baseline_bootstrap_lcb_positive": (
            "candidate_minus_baseline_bootstrap_lcb_not_positive"
        ),
        "largest_winner_removed_pnl_positive": "largest_winner_removed_pnl_not_positive",
        "settlement_causality_provenance_and_runtime_safety": (
            "settlement_causality_provenance_or_runtime_safety_failed"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    return {
        "schema_version": "bigan-v8-conformal-v5-side-only-future-pnl-gate-report-v1",
        "candidate_name": CANDIDATE_NAME,
        "decision_freeze_sha256": decision_freeze_sha256,
        "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
        "action_and_action_family_pnl_diagnostic_only": True,
        "guard_accepted_bet_count": len(accepted),
        "guard_accepted_unique_market_count": len(accepted_markets),
        "accepted_side_distribution": dict(
            sorted(Counter(str(row.get("selected_side") or "") for row in accepted).items())
        ),
        "accepted_action_distribution": dict(
            sorted(Counter(str(row.get("executed_action") or "") for row in accepted).items())
        ),
        "accepted_side_metrics": side_metrics,
        "accepted_action_metrics": action_metrics,
        "accepted_action_family_metrics": family_metrics,
        "candidate_post_cost_net_pnl": candidate_pnl,
        "matched_baseline_post_cost_net_pnl": baseline_pnl,
        "candidate_minus_matched_baseline_post_cost_net_pnl": delta_pnl,
        "candidate_minus_baseline_market_bootstrap": bootstrap,
        "largest_winning_market_pnl": largest_winner,
        "largest_winner_removed_candidate_pnl": largest_winner_removed_pnl,
        "future_gate_checks": checks,
        "future_gate_passed": not blockers,
        "future_gate_blocking_reason_codes": blockers,
        "manual_promotion_review_required": True,
        **_blocked_safety_fields(),
    }


def _accepted_group_metrics(rows: list[dict[str, Any]], *, diagnostic_only: bool) -> dict[str, Any]:
    return {
        "accepted_bet_count": len(rows),
        "accepted_unique_market_count": len({str(row["market_id"]) for row in rows}),
        "accepted_bet_net_pnl_sum": float(sum(float(row["accepted_bet_net_pnl"]) for row in rows)),
        "diagnostic_only": diagnostic_only,
    }


def _action_family(action: str) -> str:
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    if action == "NO_TRADE":
        return "NO_TRADE"
    return "UNKNOWN"


def _validate_candidate_lineage(
    candidate: dict[str, Any], *, profile: dict[str, Any]
) -> dict[str, dict[str, str]]:
    expected = dict(profile["issue_203_candidate"])
    checks = {
        "candidate_name": candidate.get("candidate_name") == CANDIDATE_NAME,
        "implementation_commit": candidate.get("implementation_commit")
        == expected["implementation_commit"],
        "candidate_freeze": candidate.get("candidate_freeze_created_ts")
        == expected["candidate_freeze_created_ts"],
        "candidate_frozen": candidate.get("research_candidate_frozen") is True,
        "calibration_gate": candidate.get("calibration_gate_passed") is True,
        "future_allowed": candidate.get("candidate_specific_future_evaluation_allowed") is True,
        "eligible_collection": candidate.get("eligible_future_collection")
        == "issue_192_strictly_later_persistent_window_only",
        "issue190_ineligible": candidate.get("issue_190_collection_eligible_for_this_candidate")
        is False,
        "issue192_after_freeze": candidate.get(
            "issue_192_collection_must_start_after_candidate_freeze"
        )
        is True,
        "model_sha256": candidate.get("model_sha256") == expected["model_sha256"],
        "dataset_hash": candidate.get("policy_dataset_hash") == expected["policy_dataset_hash"],
        "split_hash": candidate.get("split_hash") == expected["split_hash"],
        "calibration_policy_pnl": candidate.get("calibration_policy_pnl_computed") is False,
        "no_tuning": candidate.get(
            "uses_current_oof_validation_confirmatory_or_future_pnl_for_tuning"
        )
        is False,
        "safety": all(
            candidate.get(field) == value for field, value in _blocked_safety_fields().items()
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("candidate lineage validation failed: " + ", ".join(blockers))
    model = _verified_descriptor(candidate["model"], "candidate model")
    fit_profile = _verified_descriptor(candidate["fit_profile"], "candidate fit profile")
    calibration = _verified_descriptor(
        candidate["calibration_artifact"], "candidate calibration artifact"
    )
    future = _verified_descriptor(
        candidate["future_evaluation_protocol"], "candidate future protocol"
    )
    role_manifest = _verified_descriptor(
        candidate["role_assignment_manifest"], "role assignment manifest"
    )
    descriptors = {
        "model": model,
        "fit_profile": fit_profile,
        "calibration_artifact": calibration,
        "future_protocol": future,
        "role_assignment_manifest": role_manifest,
    }
    descriptor_expectations = {
        "model": "model_sha256",
        "fit_profile": "fit_profile_sha256",
        "calibration_artifact": "calibration_artifact_sha256",
        "future_protocol": "future_evaluation_protocol_sha256",
        "role_assignment_manifest": "role_assignment_manifest_sha256",
    }
    mismatches = [
        name
        for name, expected_field in descriptor_expectations.items()
        if descriptors[name]["sha256"] != expected[expected_field]
    ]
    if mismatches:
        raise ValueError("candidate descriptor hash mismatch: " + ", ".join(mismatches))
    fit_payload = _load_json(Path(fit_profile["path"]))
    validate_guard_compatible_conformal_net_return_v5_profile(fit_payload)
    future_payload = _load_json(Path(future["path"]))
    if (
        future_payload.get("eligible_collection")
        != "issue_192_strictly_later_persistent_window_only"
        or future_payload.get("issue_190_collection_eligible") is not False
        or future_payload.get("required_checks", {}).get("pnl_hard_gate_aggregation")
        != "selected_side_buy_up_buy_down_only"
        or future_payload.get("required_checks", {}).get("action_and_family_pnl_diagnostic_only")
        is not True
    ):
        raise ValueError("candidate future protocol is not side-only")
    role_payload = _load_json(Path(role_manifest["path"]))
    rows = _verified_descriptor(role_payload["selected_rows"], "role assignment rows")
    if rows["sha256"] != expected["role_assignment_rows_sha256"]:
        raise ValueError("role assignment rows hash mismatch")
    descriptors["role_assignment_rows"] = rows
    return descriptors


def _validate_prior_role_rows(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> None:
    expected_count = int(profile["issue_203_candidate"]["source_market_count"])
    market_ids = {str(row.get("market_id") or "") for row in rows}
    blockers: list[str] = []
    if len(rows) != expected_count or len(market_ids) != expected_count or "" in market_ids:
        blockers.append("prior_market_count_or_identity_mismatch")
    if any(int(row.get("maximum_decision_ts") or 0) <= 0 for row in rows):
        blockers.append("prior_decision_timestamp_missing")
    if any(row.get("labels_or_outcomes_opened_for_role_assignment") is not False for row in rows):
        blockers.append("role_assignment_outcome_sealing_invalid")
    forbidden = _find_nonempty_fields(rows, FORBIDDEN_TARGET_FIELDS)
    if forbidden:
        blockers.append("forbidden_target_field_in_role_assignment:" + ",".join(forbidden))
    if blockers:
        raise ValueError("prior role row validation failed: " + ", ".join(blockers))


def _window_binding_blockers(
    *,
    prereg: dict[str, Any],
    profile: dict[str, Any],
    boundary: dict[str, Any],
    window: dict[str, Any],
) -> list[str]:
    collection = profile["issue_192_collection"]
    blockers: list[str] = []
    if window.get("schema_version") != WINDOW_MANIFEST_SCHEMA_VERSION:
        blockers.append("window_manifest_schema_invalid")
    if window.get("window_freeze_ready") is not True:
        blockers.append("window_freeze_not_ready")
    if window.get("labels_outcomes_or_pnl_opened_for_selection") is not False:
        blockers.append("window_selection_outcome_sealing_invalid")
    if window.get("target_valid_market_count") != collection["target_quality_valid_market_count"]:
        blockers.append("window_target_count_mismatch")
    if window.get("maximum_scan_count") != collection["maximum_index_scan_count"]:
        blockers.append("window_scan_cap_mismatch")
    if window.get("selected_market_count") != collection["target_quality_valid_market_count"]:
        blockers.append("window_selected_market_count_mismatch")
    minimum_ts = int(boundary["minimum_collection_decision_ts"])
    if int(window.get("selected_window_start_ts") or 0) < minimum_ts:
        blockers.append("window_not_strictly_later_than_candidate_boundary")
    if window.get("source_boundary_manifest") != prereg["source_boundary_manifest"]:
        blockers.append("window_source_boundary_mismatch")
    protocol = window.get("protocol") or {}
    if protocol.get("sha256") != collection["collector_protocol_sha256"]:
        blockers.append("window_collector_protocol_mismatch")
    if window.get("blocking_reason_codes"):
        blockers.append("window_has_blocking_reason_codes")
    if any(window.get(field) != expected for field, expected in _blocked_safety_fields().items()):
        blockers.append("window_safety_invalid")
    return sorted(set(blockers))


def _selected_window_blockers(
    *,
    selected_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    boundary: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    expected_count = int(profile["issue_192_collection"]["target_quality_valid_market_count"])
    minimum_ts = int(boundary["minimum_collection_decision_ts"])
    prior_markets = set(boundary["prior_market_ids"])
    prior_slugs = set(boundary["prior_slugs"])
    prior_hashes = set(boundary["prior_source_row_hashes"])
    index_hashes = {str(row.get("entry_sha256") or "") for row in index_rows}
    blockers: list[str] = []
    if len(selected_rows) != expected_count:
        blockers.append("selected_row_count_mismatch")
    if len({row.get("market_id") for row in selected_rows}) != len(selected_rows):
        blockers.append("selected_market_identity_not_unique")
    if any(int(row.get("scheduled_round_start_ts") or 0) < minimum_ts for row in selected_rows):
        blockers.append("selected_row_not_strictly_later")
    if any(
        row.get("collector_git_commit") != profile["issue_192_collection"]["collector_commit"]
        for row in selected_rows
    ):
        blockers.append("selected_row_collector_commit_mismatch")
    if any(row.get("capture_quality_valid") is not True for row in selected_rows):
        blockers.append("selected_row_capture_quality_invalid")
    if any(row.get("labels_outcomes_or_pnl_opened") is not False for row in selected_rows):
        blockers.append("selected_row_outcome_sealing_invalid")
    if any(str(row.get("market_id") or "") in prior_markets for row in selected_rows):
        blockers.append("selected_market_overlaps_prior")
    if any(str(row.get("slug") or "") in prior_slugs for row in selected_rows):
        blockers.append("selected_slug_overlaps_prior")
    if any(str(row.get("source_row_hash") or "") in prior_hashes for row in selected_rows):
        blockers.append("selected_source_hash_overlaps_prior")
    if any(str(row.get("entry_sha256") or "") not in index_hashes for row in selected_rows):
        blockers.append("selected_row_not_in_pinned_index")
    if _find_nonempty_fields(selected_rows, FORBIDDEN_TARGET_FIELDS):
        blockers.append("selected_rows_contain_forbidden_target_fields")
    if any(
        row.get(field) != expected
        for row in selected_rows
        for field, expected in _blocked_safety_fields().items()
    ):
        blockers.append("selected_row_safety_invalid")
    return sorted(set(blockers))


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


def _verified_descriptor(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor missing")
    path = Path(str(value.get("path") or "")).resolve()
    digest = str(value.get("sha256") or "").lower()
    _verify_pin(path, digest, name)
    return {"path": str(path), "sha256": digest}


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
    if not _is_git_sha(value):
        raise ValueError("builder_git_commit must be a Git SHA-1")


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _is_git_sha(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _preregistration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Conformal v5 future evaluation pre-registration",
            "",
            f"- ready: `{str(report['pre_registration_ready']).lower()}`",
            f"- fit / conformal markets: `{report['fit_market_count']} / {report['conformal_calibration_market_count']}`",
            f"- excluded prior markets: `{report['prior_market_count']}`",
            f"- future raw target / scan cap: `{report['target_quality_valid_market_count']} / {report['maximum_index_scan_count']}`",
            f"- minimum accepted unique markets: `{report['minimum_guard_accepted_unique_market_count']}`",
            f"- hard PnL aggregation: `{report['pnl_hard_gate_aggregation']}`",
            "- action/family PnL: `diagnostic_only`",
            "- future labels/outcomes/PnL opened: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _binding_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Conformal v5 future candidate/window binding",
            "",
            f"- passed: `{str(report['candidate_window_binding_passed']).lower()}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- strictly later: `{str(report['strictly_later_than_candidate_freeze']).lower()}`",
            "- feature materialization attempted: `false`",
            "- prediction attempted: `false`",
            "- future labels/outcomes/PnL opened: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


__all__ = [
    "ConformalV5FuturePreRegistrationConfig",
    "ConformalV5FutureWindowBindingConfig",
    "bind_conformal_v5_future_window_before_prediction",
    "build_conformal_v5_side_only_future_pnl_gate",
    "pre_register_conformal_v5_future_evaluation",
    "validate_conformal_v5_future_evaluation_profile",
]
