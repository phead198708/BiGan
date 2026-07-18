"""Development-only audit of p_up-aligned execution-compatible action support."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _market_bootstrap_interval,
    _prospective_support_estimate,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    FORBIDDEN_OUTCOME_FIELDS,
    _blocker_category,
    _load_outcome_blind_feature_rows,
    _materialize_outcome_blind_action_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    _descriptor,
    _find_fields,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-p-up-aligned-action-value-support-audit-profile-v1"
SCHEMA_PREFIX = "bigan-v8-p-up-aligned-action-value-support"
ALLOWED_ROLE = "development_train"
TARGET_FIELDS = frozenset(
    {
        *FORBIDDEN_OUTCOME_FIELDS,
        "target_net_pnl_per_contract",
        "target_resolved_outcome",
        "target_cost_components",
    }
)


@dataclass(frozen=True, slots=True)
class PUpAlignedActionValueSupportConfig:
    """Pinned inputs for the #201 development-only support audit."""

    run_id: str
    output_dir: Path | str
    audit_profile_path: Path | str
    expected_audit_profile_sha256: str
    issue198_candidate_manifest_path: Path | str
    expected_issue198_candidate_manifest_sha256: str
    issue200_manifest_path: Path | str
    expected_issue200_manifest_sha256: str
    role_assignment_manifest_path: Path | str
    expected_role_assignment_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in (
            "expected_audit_profile_sha256",
            "expected_issue198_candidate_manifest_sha256",
            "expected_issue200_manifest_sha256",
            "expected_role_assignment_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "audit_profile_path",
            "issue198_candidate_manifest_path",
            "issue200_manifest_path",
            "role_assignment_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_p_up_aligned_action_value_support_profile(profile: dict[str, Any]) -> None:
    """Fail closed on any drift from the pre-registered #201 audit."""

    probe = dict(profile.get("static_guard_probe") or {})
    bootstrap = dict(profile.get("bootstrap") or {})
    access = dict(profile.get("access_sequence") or {})
    mutation = dict(profile.get("mutation_contract") or {})
    hash_fields = (
        "parent_issue_200_manifest_sha256",
        "parent_issue_198_candidate_manifest_sha256",
        "role_assignment_manifest_sha256",
        "role_assignment_rows_sha256",
        "feature_contract_sha256",
        "development_train_target_rows_sha256",
        "execution_guard_config_sha256",
    )
    checks = {
        "schema_version": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "audit_name": profile.get("audit_name")
        == "p_up_aligned_execution_compatible_action_value_support",
        "frozen": profile.get("frozen") is True,
        "parent_commit": _is_sha1(str(profile.get("parent_issue_200_commit") or "")),
        "hashes": all(_is_sha256(str(profile.get(name) or "")) for name in hash_fields),
        "role": profile.get("allowed_role") == ALLOWED_ROLE,
        "coverage": profile.get("expected_market_count") == 90
        and profile.get("expected_decision_group_count") == 360
        and profile.get("expected_action_row_count") == 1800,
        "actions": profile.get("required_actions") == list(REQUIRED_ACTIONS),
        "probe": probe.get("runtime_state") == "fresh_empty_valid_simulated_state_per_action"
        and probe.get("ranking_score_used_as_model_evidence") is False
        and probe.get("guard_config_mutation_allowed") is False
        and probe.get("exposure_policy_mutation_allowed") is False
        and probe.get("sizing_mutation_allowed") is False,
        "bootstrap": bootstrap.get("unit") == "market_id"
        and int(bootstrap.get("resample_count") or 0) >= 100
        and float(bootstrap.get("confidence_level") or 0.0) == 0.95
        and float(bootstrap.get("prospective_target_power") or 0.0) == 0.8,
        "access": access.get("pre_label_audit_required") is True
        and access.get("feature_only_universe_freeze_before_target_access_required") is True
        and access.get("development_train_targets_may_be_opened_after_universe_freeze") is True
        and all(
            access.get(name) is False
            for name in (
                "development_calibration_files_may_be_opened",
                "confirmatory_files_may_be_opened",
                "issue_190_or_192_future_files_may_be_opened",
                "future_accepted_bet_pnl_may_be_opened",
            )
        ),
        "no_mutation": mutation and all(value is False for value in mutation.values()),
        "safety": dict(profile.get("safety") or {}) == _blocked_safety_fields(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid #201 audit profile: " + ", ".join(failed))


def run_p_up_aligned_action_value_support_audit(
    config: PUpAlignedActionValueSupportConfig,
) -> dict[str, Any]:
    """Freeze the feature-only action universe, then open train targets once."""

    paths = {
        "profile": config.audit_profile_path.resolve(),
        "issue198_candidate": config.issue198_candidate_manifest_path.resolve(),
        "issue200_manifest": config.issue200_manifest_path.resolve(),
        "role_assignment_manifest": config.role_assignment_manifest_path.resolve(),
    }
    expected = {
        "profile": config.expected_audit_profile_sha256,
        "issue198_candidate": config.expected_issue198_candidate_manifest_sha256,
        "issue200_manifest": config.expected_issue200_manifest_sha256,
        "role_assignment_manifest": config.expected_role_assignment_manifest_sha256,
    }
    for name, path in paths.items():
        _verify_file_hash(path, expected[name], name=name)
    profile = _load_json(paths["profile"])
    validate_p_up_aligned_action_value_support_profile(profile)
    if profile["parent_issue_198_candidate_manifest_sha256"] != expected["issue198_candidate"]:
        raise ValueError("#198 candidate lineage mismatch")
    if profile["parent_issue_200_manifest_sha256"] != expected["issue200_manifest"]:
        raise ValueError("#200 manifest lineage mismatch")
    if profile["role_assignment_manifest_sha256"] != expected["role_assignment_manifest"]:
        raise ValueError("role assignment manifest lineage mismatch")

    candidate = _load_json(paths["issue198_candidate"])
    issue200_manifest = _load_json(paths["issue200_manifest"])
    role_manifest = _load_json(paths["role_assignment_manifest"])
    lineage = _validate_lineage(
        candidate=candidate,
        issue200_manifest=issue200_manifest,
        role_manifest=role_manifest,
        profile=profile,
    )

    run_dir = Path(config.output_dir) / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    pre_label = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-label-access-audit-v1",
        "run_id": config.run_id,
        "audit_profile": _descriptor(paths["profile"]),
        "issue198_candidate_manifest": _descriptor(paths["issue198_candidate"]),
        "issue200_manifest": _descriptor(paths["issue200_manifest"]),
        "role_assignment_manifest": _descriptor(paths["role_assignment_manifest"]),
        "allowed_role": ALLOWED_ROLE,
        "intended_target_rows": lineage["target_rows"],
        "target_bearing_files_opened_before_audit": False,
        "target_rows_hash_verified_before_audit": False,
        "development_calibration_confirmatory_or_future_files_opened": False,
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
    }
    pre_label["audit_id"] = canonical_json_sha256(pre_label)
    pre_label_path = run_dir / "pre_label_access_lineage_audit.json"
    _write_json_fsync(pre_label_path, pre_label)
    _write_text_fsync(
        run_dir / "pre_label_access_lineage_audit.md",
        _pre_label_markdown(pre_label),
    )

    role_rows = _load_jsonl(Path(lineage["role_assignment_rows"]["path"]))
    if _find_fields({"rows": role_rows}, set(TARGET_FIELDS)):
        raise ValueError("role assignment rows contain target fields")
    train_role_rows = [row for row in role_rows if row.get("role") == ALLOWED_ROLE]
    if len(train_role_rows) != int(profile["expected_market_count"]):
        raise ValueError("development_train role coverage mismatch")
    feature_contract = _load_json(Path(lineage["feature_contract"]["path"]))
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=feature_contract["parent_protocol_sha256"],
    )
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    action_rows: list[dict[str, Any]] = []
    opened_feature_rows: list[dict[str, str]] = []
    feature_row_count = 0
    for role_row in train_role_rows:
        feature_rows, descriptor = _load_outcome_blind_feature_rows(role_row)
        opened_feature_rows.append(descriptor)
        feature_row_count += len(feature_rows)
        materialized = _materialize_outcome_blind_action_rows(
            feature_rows,
            role_row=role_row,
            feature_columns=feature_columns,
        )
        for row in materialized:
            row["role"] = ALLOWED_ROLE
            row["action_row_sha256"] = canonical_json_sha256(
                {key: value for key, value in row.items() if key != "action_row_sha256"}
            )
        action_rows.extend(materialized)
    action_rows.sort(key=_action_sort_key)
    _validate_complete_action_grid(action_rows, profile=profile)
    universe_rows = build_execution_compatible_action_universe(action_rows)
    universe_path = run_dir / "execution_compatible_action_universe.jsonl"
    _write_jsonl_fsync(universe_path, universe_rows)
    universe_report = build_execution_compatible_action_universe_report(
        run_id=config.run_id,
        rows=universe_rows,
        feature_row_count=feature_row_count,
        market_count=len(train_role_rows),
        profile=profile,
    )
    universe_report_path = run_dir / "execution_compatible_action_universe_report.json"
    _write_json_fsync(universe_report_path, universe_report)
    _write_text_fsync(
        run_dir / "execution_compatible_action_universe_report.md",
        _universe_markdown(universe_report),
    )

    outcome_blind_freeze = {
        "schema_version": f"{SCHEMA_PREFIX}-outcome-blind-universe-freeze-v1",
        "run_id": config.run_id,
        "pre_label_access_audit": _descriptor(pre_label_path),
        "execution_compatible_action_universe": _descriptor(universe_path),
        "execution_compatible_action_universe_report": _descriptor(universe_report_path),
        "opened_feature_rows": opened_feature_rows,
        "feature_contract": lineage["feature_contract"],
        "role_assignment_rows": lineage["role_assignment_rows"],
        "execution_guard_config_sha256": canonical_json_sha256(_v8_execution_guard_config()),
        "target_bearing_files_opened_before_universe_freeze": False,
        "target_rows_hash_verified_before_universe_freeze": False,
        "outcome_blind_universe_frozen_before_target_access": True,
        **_blocked_safety_fields(),
    }
    outcome_blind_freeze["freeze_id"] = canonical_json_sha256(outcome_blind_freeze)
    freeze_path = run_dir / "outcome_blind_action_universe_freeze_manifest.json"
    _write_json_fsync(freeze_path, outcome_blind_freeze)
    freeze_sha256_before_target_access = _sha256_file(freeze_path)
    if freeze_sha256_before_target_access != _sha256_file(freeze_path):
        raise ValueError("outcome-blind universe freeze is not stable on disk")

    target_path = Path(lineage["target_rows"]["path"])
    _verify_file_hash(
        target_path,
        profile["development_train_target_rows_sha256"],
        name="development_train target rows after universe freeze",
    )
    target_rows = _load_jsonl(target_path)
    _validate_target_rows(target_rows, universe_rows=universe_rows, profile=profile)
    joined_rows = _join_targets(universe_rows, target_rows)
    support_report = build_p_up_aligned_action_value_support_report(
        run_id=config.run_id,
        rows=joined_rows,
        profile=profile,
        universe_freeze_sha256=freeze_sha256_before_target_access,
    )
    support_path = run_dir / "p_up_aligned_action_value_support_report.json"
    _write_json_fsync(support_path, support_report)
    _write_text_fsync(
        run_dir / "p_up_aligned_action_value_support_report.md",
        _support_markdown(support_report),
    )

    issue200_replay = _load_jsonl(Path(lineage["issue200_guard_replay"]["path"]))
    if _find_fields({"rows": issue200_replay}, set(TARGET_FIELDS)):
        raise ValueError("#200 outcome-blind replay unexpectedly contains target fields")
    attribution = build_source_guard_intersection_attribution_report(
        run_id=config.run_id,
        issue200_replay=issue200_replay,
        support_report=support_report,
    )
    attribution_path = run_dir / "source_guard_intersection_attribution_report.json"
    _write_json_fsync(attribution_path, attribution)
    _write_text_fsync(
        run_dir / "source_guard_intersection_attribution_report.md",
        _attribution_markdown(attribution),
    )

    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "audit_profile": _descriptor(paths["profile"]),
        "pre_label_access_audit": _descriptor(pre_label_path),
        "outcome_blind_action_universe_freeze_manifest": _descriptor(freeze_path),
        "execution_compatible_action_universe": _descriptor(universe_path),
        "execution_compatible_action_universe_report": _descriptor(universe_report_path),
        "p_up_aligned_action_value_support_report": _descriptor(support_path),
        "source_guard_intersection_attribution_report": _descriptor(attribution_path),
        "development_train_targets_opened_after_universe_freeze": True,
        "development_train_target_rows_hash_verified_after_universe_freeze": True,
        "development_calibration_files_opened": False,
        "confirmatory_files_opened": False,
        "issue_190_or_192_future_files_opened": False,
        "current_oof_validation_or_future_pnl_used_for_tuning": False,
        "new_candidate_fit_or_threshold_guard_mutation_performed": False,
        "support_conclusion": support_report["support_conclusion"],
        "root_cause_classification": attribution["root_cause_classification"],
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "p_up_aligned_action_value_support_audit_manifest.json"
    _write_json_fsync(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "universe_report": universe_report,
        "support_report": support_report,
        "attribution_report": attribution,
    }


def build_execution_compatible_action_universe(
    action_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Probe each action independently with the frozen guard and no outcomes."""

    guard_config = _v8_execution_guard_config()
    by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        by_group[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    output: list[dict[str, Any]] = []
    for _, group in sorted(by_group.items(), key=lambda item: (item[0][1], item[0][0])):
        by_action = {str(row["action"]): row for row in group}
        for action in REQUIRED_ACTIONS:
            row = by_action[action]
            if action == "NO_TRADE":
                probe = {
                    "execution_guard_evaluated": False,
                    "full_guard_order_allowed": False,
                    "full_guard_original_action_allowed": False,
                    "execution_guarded_action": "NO_TRADE",
                    "guard_action_remapped": False,
                    "execution_blocking_reason_codes": ["no_trade_is_not_an_order"],
                    "execution_guard_reason_codes": ["execution_no_trade_selected"],
                    "guard_blocker_categories": ["selected_no_trade"],
                    "execution_quality_only_passed": False,
                }
            else:
                ranking = _static_probe_ranking(group, selected_action=action)
                guard_result = _v8_execution_guard_decision(
                    {
                        "decision_group_id": canonical_json_sha256(
                            {"market_id": row["market_id"], "decision_ts": row["decision_ts"]}
                        ),
                        "market_id": row["market_id"],
                        "decision_ts": row["decision_ts"],
                        "selected_action": action,
                        "selected_side": row["side"],
                        "selected_action_family": row["action_family"],
                        "corrected_model_score": 1.0,
                        "raw_model_score": 1.0,
                        "high_score_flag": True,
                        "p_up": row["p_up"],
                        "p_down": row["p_down"],
                        "p_up_action_disagreement": row["p_up_action_disagreement"],
                        "microstructure_snapshot": row["microstructure_snapshot"],
                        "reference_price_feature_provenance": row[
                            "reference_price_feature_provenance"
                        ],
                        "decision_time_feature_max_input_ts": row["max_input_ts"],
                        "full_5_action_ranking": ranking,
                    },
                    guard_config=guard_config,
                    runtime_state=_v8_initial_runtime_state(guard_config),
                    runtime_mode="simulated_runtime_state",
                )
                blockers = list(guard_result["execution_blocking_reason_codes"])
                blocker_categories = sorted({_blocker_category(code) for code in blockers})
                non_alignment_blockers = [
                    code for code in blockers if _blocker_category(code) != "p_up_disagreement"
                ]
                guarded_action = str(guard_result["execution_guarded_action"])
                allowed = bool(guard_result["order_allowed"])
                probe = {
                    "execution_guard_evaluated": True,
                    "full_guard_order_allowed": allowed,
                    "full_guard_original_action_allowed": allowed and guarded_action == action,
                    "execution_guarded_action": guarded_action,
                    "guard_action_remapped": guarded_action != action,
                    "execution_blocking_reason_codes": blockers,
                    "execution_guard_reason_codes": list(
                        guard_result["execution_guard_reason_codes"]
                    ),
                    "guard_blocker_categories": blocker_categories,
                    "execution_quality_only_passed": not non_alignment_blockers,
                    "guard_proposed_order_size": float(guard_result["proposed_order_size"]),
                }
            enriched = {
                **row,
                **probe,
                "p_up_alignment_passed": action != "NO_TRADE"
                and row["p_up_action_disagreement"] is False,
                "static_guard_probe_only": True,
                "static_guard_probe_ranking_score_used_as_model_evidence": False,
                "target_or_outcome_fields_used": False,
            }
            enriched["universe_row_sha256"] = canonical_json_sha256(enriched)
            output.append(enriched)
    output.sort(key=_action_sort_key)
    return output


def build_execution_compatible_action_universe_report(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    feature_row_count: int,
    market_count: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the frozen outcome-blind action opportunity universe."""

    trade = [row for row in rows if row["action"] != "NO_TRADE"]
    aligned = [row for row in trade if row["p_up_alignment_passed"]]
    quality = [row for row in trade if row["execution_quality_only_passed"]]
    aligned_quality = [
        row
        for row in trade
        if row["p_up_alignment_passed"] and row["execution_quality_only_passed"]
    ]
    original_allowed = [row for row in trade if row["full_guard_original_action_allowed"]]
    groups_with_aligned_quality = {
        (row["market_id"], row["decision_ts"]) for row in aligned_quality
    }
    return {
        "schema_version": f"{SCHEMA_PREFIX}-universe-report-v1",
        "run_id": run_id,
        "status": "outcome_blind_universe_frozen",
        "role": ALLOWED_ROLE,
        "market_count": market_count,
        "source_feature_row_count": feature_row_count,
        "decision_group_count": len(rows) // len(REQUIRED_ACTIONS),
        "action_row_count": len(rows),
        "complete_five_action_grid_passed": len(rows) == int(profile["expected_action_row_count"]),
        "trade_action_row_count": len(trade),
        "p_up_aligned_trade_action_count": len(aligned),
        "p_up_disagreeing_trade_action_count": len(trade) - len(aligned),
        "execution_quality_only_passed_count": len(quality),
        "p_up_aligned_execution_quality_passed_count": len(aligned_quality),
        "full_guard_original_action_allowed_count": len(original_allowed),
        "decision_group_with_p_up_aligned_execution_quality_action_count": len(
            groups_with_aligned_quality
        ),
        "action_distribution": dict(sorted(Counter(row["action"] for row in rows).items())),
        "aligned_quality_action_distribution": dict(
            sorted(Counter(row["action"] for row in aligned_quality).items())
        ),
        "guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    code for row in trade for code in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
        "guard_blocker_category_distribution": dict(
            sorted(
                Counter(
                    category for row in trade for category in row["guard_blocker_categories"]
                ).items()
            )
        ),
        "execution_guard_config_sha256": canonical_json_sha256(_v8_execution_guard_config()),
        "static_guard_probe_is_policy_evidence": False,
        "target_or_outcome_files_opened": False,
        **_blocked_safety_fields(),
    }


def build_p_up_aligned_action_value_support_report(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    universe_freeze_sha256: str,
) -> dict[str, Any]:
    """Summarize cost-aware train-label support after the universe is frozen."""

    trade = [row for row in rows if row["action"] != "NO_TRADE"]
    segments = {
        "all_trade_actions": trade,
        "p_up_aligned_trade_actions": [row for row in trade if row["p_up_alignment_passed"]],
        "p_up_disagreeing_trade_actions": [
            row for row in trade if not row["p_up_alignment_passed"]
        ],
        "execution_quality_only_passed_trade_actions": [
            row for row in trade if row["execution_quality_only_passed"]
        ],
        "p_up_aligned_execution_quality_passed_trade_actions": [
            row
            for row in trade
            if row["p_up_alignment_passed"] and row["execution_quality_only_passed"]
        ],
        "full_guard_original_action_allowed_trade_actions": [
            row for row in trade if row["full_guard_original_action_allowed"]
        ],
    }
    segment_metrics = {
        name: _support_metrics(values, profile=profile, seed_offset=index)
        for index, (name, values) in enumerate(segments.items())
    }
    dimension_metrics: dict[str, Any] = {}
    dimensions = {
        "action": lambda row: row["action"],
        "action_family": lambda row: row["action_family"],
        "side": lambda row: row["side"],
        "p_up_alignment": lambda row: "aligned" if row["p_up_alignment_passed"] else "disagreeing",
        "p_up_alignment_by_family": lambda row: (
            ("aligned" if row["p_up_alignment_passed"] else "disagreeing")
            + "|"
            + row["action_family"]
        ),
        "p_up_alignment_by_action": lambda row: (
            ("aligned" if row["p_up_alignment_passed"] else "disagreeing") + "|" + row["action"]
        ),
        "aligned_execution_quality_by_action": lambda row: (
            (
                "aligned_quality_pass"
                if row["p_up_alignment_passed"] and row["execution_quality_only_passed"]
                else "other"
            )
            + "|"
            + row["action"]
        ),
        "aligned_execution_quality_by_family": lambda row: (
            (
                "aligned_quality_pass"
                if row["p_up_alignment_passed"] and row["execution_quality_only_passed"]
                else "other"
            )
            + "|"
            + row["action_family"]
        ),
        "time_to_close_bucket": lambda row: _numeric_bucket(
            float(row["microstructure_snapshot"]["time_to_close_seconds"]),
            (60.0, 120.0, 180.0),
        ),
        "execution_price_bucket": lambda row: _numeric_bucket(
            float(row["microstructure_snapshot"]["entry_ask"]),
            (0.3, 0.5, 0.7, 0.9),
        ),
        "execution_quality_by_side": lambda row: (
            ("quality_pass" if row["execution_quality_only_passed"] else "quality_blocked")
            + "|"
            + row["side"]
        ),
    }
    for dimension, key_fn in dimensions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in trade:
            grouped[str(key_fn(row))].append(row)
        dimension_metrics[dimension] = {
            key: _support_metrics(values, profile=profile, seed_offset=100 + index)
            for index, (key, values) in enumerate(sorted(grouped.items()))
        }
    focal = segment_metrics["p_up_aligned_execution_quality_passed_trade_actions"]
    lcb = (focal.get("market_level_post_cost_return") or {}).get("lower_confidence_bound")
    mean = focal.get("target_post_cost_return_mean")
    aligned_quality_action_metrics = {
        key.removeprefix("aligned_quality_pass|"): value
        for key, value in dimension_metrics["aligned_execution_quality_by_action"].items()
        if key.startswith("aligned_quality_pass|")
    }
    positive_point_actions = sorted(
        action
        for action, metrics in aligned_quality_action_metrics.items()
        if metrics["target_post_cost_return_mean"] is not None
        and float(metrics["target_post_cost_return_mean"]) > 0.0
    )
    positive_lcb_actions = sorted(
        action
        for action, metrics in aligned_quality_action_metrics.items()
        if (metrics.get("market_level_post_cost_return") or {}).get("lower_confidence_bound")
        is not None
        and float(metrics["market_level_post_cost_return"]["lower_confidence_bound"]) > 0.0
    )
    if not focal["row_count"]:
        conclusion = "no_p_up_aligned_execution_quality_action_support"
    elif lcb is not None and float(lcb) > 0.0:
        conclusion = "positive_lcb_support_exists_for_preregistered_v4_research"
    elif positive_lcb_actions:
        conclusion = "positive_lcb_action_specific_support_exists_for_preregistered_v4_research"
    elif positive_point_actions:
        conclusion = "positive_point_action_specific_support_but_market_lcb_nonpositive"
    elif mean is not None and float(mean) > 0.0:
        conclusion = "positive_point_support_but_market_lcb_nonpositive"
    else:
        conclusion = "p_up_aligned_execution_quality_support_not_positive"
    return {
        "schema_version": f"{SCHEMA_PREFIX}-support-report-v1",
        "run_id": run_id,
        "status": "development_train_diagnostic_complete_fail_closed",
        "role": ALLOWED_ROLE,
        "outcome_blind_universe_freeze_sha256": universe_freeze_sha256,
        "outcome_blind_universe_frozen_before_target_access": True,
        "development_train_targets_opened_after_universe_freeze": True,
        "development_train_target_rows_hash_verified_after_universe_freeze": True,
        "development_calibration_confirmatory_or_future_files_opened": False,
        "target_used_as_decision_input": False,
        "cost_aware_target_field": "target_net_pnl_per_contract",
        "opportunity_set_target_sum_is_policy_pnl": False,
        "segment_metrics": segment_metrics,
        "dimension_metrics": dimension_metrics,
        "p_up_aligned_execution_quality_action_metrics": aligned_quality_action_metrics,
        "positive_point_supported_actions": positive_point_actions,
        "positive_lcb_supported_actions": positive_lcb_actions,
        "support_conclusion": conclusion,
        "new_candidate_fit_allowed_from_this_report": False,
        "separate_preregistration_required_for_v4": True,
        **_blocked_safety_fields(),
    }


def build_source_guard_intersection_attribution_report(
    *,
    run_id: str,
    issue200_replay: list[dict[str, Any]],
    support_report: dict[str, Any],
) -> dict[str, Any]:
    """Attribute #200 zero acceptance without using its calibration targets."""

    selected_trade = [row for row in issue200_replay if row["source_selected_action"] != "NO_TRADE"]
    p_up_disagreeing = [row for row in selected_trade if row["p_up_action_disagreement"]]
    accepted = [row for row in selected_trade if row["execution_guard_order_allowed"]]
    focal = support_report["segment_metrics"]["p_up_aligned_execution_quality_passed_trade_actions"]
    conclusion = str(support_report["support_conclusion"])
    if conclusion.startswith("positive_lcb"):
        classification = (
            "source_selection_anti_aligned_with_guard_despite_positive_lcb_alternative_support"
        )
        recommendation = "preregister_v4_guard_compatible_objective_without_opening_new_splits"
    elif support_report["positive_point_supported_actions"]:
        classification = (
            "source_selection_anti_aligned_with_guard_and_action_specific_support_underpowered"
        )
        recommendation = "collect_more_outcome_blind_data_before_any_v4_fit"
    else:
        classification = (
            "source_selection_anti_aligned_with_guard_and_no_positive_supported_alternative"
        )
        recommendation = "do_not_fit_v4_until_positive_executable_support_exists"
    return {
        "schema_version": f"{SCHEMA_PREFIX}-source-guard-attribution-v1",
        "run_id": run_id,
        "issue200_outcome_blind_decision_count": len(issue200_replay),
        "issue200_selected_trade_decision_count": len(selected_trade),
        "issue200_selected_trade_p_up_disagreement_count": len(p_up_disagreeing),
        "issue200_guard_accepted_trade_count": len(accepted),
        "issue200_all_selected_trade_candidates_p_up_disagreeing": bool(selected_trade)
        and len(selected_trade) == len(p_up_disagreeing),
        "development_train_p_up_aligned_execution_quality_support": focal,
        "positive_point_supported_actions": support_report["positive_point_supported_actions"],
        "positive_lcb_supported_actions": support_report["positive_lcb_supported_actions"],
        "root_cause_classification": classification,
        "recommended_next_action": recommendation,
        "issue200_calibration_targets_opened_by_attribution": False,
        "new_candidate_fit_or_guard_relaxation_performed": False,
        **_blocked_safety_fields(),
    }


def _static_probe_ranking(
    group: list[dict[str, Any]],
    *,
    selected_action: str,
) -> list[dict[str, Any]]:
    ordered = sorted(group, key=lambda row: (row["action"] != selected_action, row["action"]))
    return [
        {
            "rank": rank,
            "selected_action": row["action"],
            "selected_side": row["side"],
            "selected_action_family": row["action_family"],
            "corrected_model_score": 1.0 if row["action"] == selected_action else 0.0,
            "raw_model_score": 1.0 if row["action"] == selected_action else 0.0,
            "high_score_flag": row["action"] == selected_action,
            "p_up_action_disagreement": row["p_up_action_disagreement"],
            "microstructure_snapshot": row["microstructure_snapshot"],
        }
        for rank, row in enumerate(ordered, start=1)
    ]


def _support_metrics(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    values = [float(row["target_net_pnl_per_contract"]) for row in rows]
    by_market: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        by_market[str(row["market_id"])].append(value)
    market_means = [float(np.mean(market_values)) for market_values in by_market.values()]
    if market_means:
        bootstrap = dict(profile["bootstrap"])
        interval = _market_bootstrap_interval(
            market_means,
            resample_count=int(bootstrap["resample_count"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=int(bootstrap["seed"]) + seed_offset,
        )
        prospective = _prospective_support_estimate(
            market_means,
            confidence_level=float(bootstrap["confidence_level"]),
            target_power=float(bootstrap["prospective_target_power"]),
            maximum_market_count=int(bootstrap["prospective_maximum_market_count"]),
        )
    else:
        interval = None
        prospective = {
            "status": "not_estimable_from_zero_support",
            "observed_market_count": 0,
        }
    return {
        "row_count": len(rows),
        "decision_group_count": len({(row["market_id"], row["decision_ts"]) for row in rows}),
        "unique_market_count": len(by_market),
        "positive_target_count": sum(value > 0.0 for value in values),
        "negative_target_count": sum(value < 0.0 for value in values),
        "zero_target_count": sum(value == 0.0 for value in values),
        "target_post_cost_return_sum": float(sum(values)),
        "target_post_cost_return_mean": float(np.mean(values)) if values else None,
        "market_level_post_cost_return": interval,
        "prospective_support": prospective,
    }


def _validate_lineage(
    *,
    candidate: dict[str, Any],
    issue200_manifest: dict[str, Any],
    role_manifest: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, dict[str, str]]:
    if candidate.get("candidate_name") != "direct_decision_group_action_advantage_v2":
        raise ValueError("unexpected #198 candidate")
    if (
        issue200_manifest.get("candidate_name")
        != "policy_value_lcb_v3_two_safety_estimand_selector"
    ):
        raise ValueError("unexpected #200 candidate")
    if role_manifest.get("role_assignment_ready") is not True:
        raise ValueError("role assignment is not ready")
    role_rows = _verified_descriptor(role_manifest.get("selected_rows"), name="role rows")
    feature_contract = _verified_descriptor(
        candidate.get("feature_contract"), name="feature contract"
    )
    target_rows = _claimed_descriptor(
        candidate.get("development_train_action_rows"),
        name="development train target rows",
    )
    issue200_replay = _verified_descriptor(
        issue200_manifest.get("outcome_blind_guard_replay"),
        name="#200 outcome-blind replay",
    )
    expected = {
        "role_assignment_rows": profile["role_assignment_rows_sha256"],
        "feature_contract": profile["feature_contract_sha256"],
        "target_rows": profile["development_train_target_rows_sha256"],
    }
    actual = {
        "role_assignment_rows": role_rows["sha256"],
        "feature_contract": feature_contract["sha256"],
        "target_rows": target_rows["sha256"],
    }
    mismatches = sorted(name for name in expected if expected[name] != actual[name])
    if mismatches:
        raise ValueError("source lineage hash mismatch: " + ", ".join(mismatches))
    guard_hash = canonical_json_sha256(_v8_execution_guard_config())
    if guard_hash != profile["execution_guard_config_sha256"]:
        raise ValueError("execution guard config hash mismatch")
    return {
        "role_assignment_rows": role_rows,
        "feature_contract": feature_contract,
        "target_rows": target_rows,
        "issue200_guard_replay": issue200_replay,
    }


def _validate_complete_action_grid(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> None:
    if len(rows) != int(profile["expected_action_row_count"]):
        raise ValueError("action row coverage mismatch")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("role") != ALLOWED_ROLE:
            raise ValueError("forbidden role in action universe")
        if _find_fields(row, set(TARGET_FIELDS)):
            raise ValueError("outcome-blind action universe contains target fields")
        groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    if len(groups) != int(profile["expected_decision_group_count"]):
        raise ValueError("decision group coverage mismatch")
    if any(actions != set(REQUIRED_ACTIONS) for actions in groups.values()):
        raise ValueError("five-action decision grid is incomplete")


def _validate_target_rows(
    target_rows: list[dict[str, Any]],
    *,
    universe_rows: list[dict[str, Any]],
    profile: dict[str, Any],
) -> None:
    if len(target_rows) != int(profile["expected_action_row_count"]):
        raise ValueError("development_train target coverage mismatch")
    keys = set()
    for row in target_rows:
        if row.get("role") != ALLOWED_ROLE:
            raise ValueError("forbidden role in target rows")
        action = str(row.get("action") or "")
        if action not in REQUIRED_ACTIONS:
            raise ValueError("invalid target action")
        target = row.get("target_net_pnl_per_contract")
        if not isinstance(target, (int, float)) or not math.isfinite(float(target)):
            raise ValueError("target net return must be finite")
        if row.get("target_used_as_decision_input") is not False:
            raise ValueError("target was used as a decision input")
        keys.add((str(row["market_id"]), int(row["decision_ts"]), action))
    universe_keys = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"]))
        for row in universe_rows
    }
    if keys != universe_keys:
        raise ValueError("target and universe keys do not reconcile")


def _join_targets(
    universe_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    targets = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): row
        for row in target_rows
    }
    output = []
    for row in universe_rows:
        key = (str(row["market_id"]), int(row["decision_ts"]), str(row["action"]))
        target = targets[key]
        output.append(
            {
                **row,
                "target_net_pnl_per_contract": float(target["target_net_pnl_per_contract"]),
                "target_cost_components": target.get("target_cost_components"),
                "target_resolved_outcome": target.get("target_resolved_outcome"),
                "target_used_as_decision_input": False,
                "target_joined_after_outcome_blind_universe_freeze": True,
            }
        )
    return output


def _action_sort_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(row["decision_ts"]),
        str(row["market_id"]),
        REQUIRED_ACTIONS.index(str(row["action"])),
    )


def _numeric_bucket(value: float, boundaries: tuple[float, ...]) -> str:
    lower = float("-inf")
    for upper in boundaries:
        if value < upper:
            return f"{lower:g}_{upper:g}"
        lower = upper
    return f"{lower:g}_inf"


def _verify_file_hash(path: Path, expected: str, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")


def _claimed_descriptor(value: Any, *, name: str) -> dict[str, str]:
    """Validate a descriptor claim without touching its target-bearing file."""

    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = str(value.get("path") or "")
    sha256 = str(value.get("sha256") or "")
    if not path or not _is_sha256(sha256):
        raise ValueError(f"{name} descriptor is invalid")
    return {"path": path, "sha256": sha256}


def _write_json_fsync(path: Path, payload: dict[str, Any]) -> None:
    _write_text_fsync(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl_fsync(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_fsync(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _write_text_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _is_sha1(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


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


def _pre_label_markdown(report: dict[str, Any]) -> str:
    return (
        "# #201 Pre-label Access Audit\n\n"
        f"- Audit ID: `{report['audit_id']}`\n"
        f"- Allowed role: `{report['allowed_role']}`\n"
        "- Target-bearing files opened before audit: `false`\n"
        "- Calibration, confirmatory, and future evidence opened: `false`\n"
    )


def _universe_markdown(report: dict[str, Any]) -> str:
    return (
        "# Execution-compatible Action Universe\n\n"
        f"- Markets: `{report['market_count']}`\n"
        f"- Decision groups: `{report['decision_group_count']}`\n"
        f"- Action rows: `{report['action_row_count']}`\n"
        f"- p_up-aligned trade actions: `{report['p_up_aligned_trade_action_count']}`\n"
        "- p_up-aligned + execution-quality actions: "
        f"`{report['p_up_aligned_execution_quality_passed_count']}`\n"
        "- Targets/outcomes opened: `false`\n"
        "- Static probes are opportunity diagnostics, not policy evidence.\n"
    )


def _support_markdown(report: dict[str, Any]) -> str:
    focal = report["segment_metrics"]["p_up_aligned_execution_quality_passed_trade_actions"]
    interval = focal.get("market_level_post_cost_return") or {}
    return (
        "# p_up-aligned Action Value Support\n\n"
        f"- Conclusion: `{report['support_conclusion']}`\n"
        f"- Rows: `{focal['row_count']}`\n"
        f"- Markets: `{focal['unique_market_count']}`\n"
        f"- Cost-aware mean: `{focal['target_post_cost_return_mean']}`\n"
        f"- Market-bootstrap LCB: `{interval.get('lower_confidence_bound')}`\n"
        f"- Positive point-estimate actions: `{report['positive_point_supported_actions']}`\n"
        f"- Positive-LCB actions: `{report['positive_lcb_supported_actions']}`\n"
        "- Opportunity-set target sum is policy PnL: `false`\n"
        "- A separately pre-registered candidate is required: `true`\n"
    )


def _attribution_markdown(report: dict[str, Any]) -> str:
    return (
        "# Source / Guard Intersection Attribution\n\n"
        f"- #200 selected trades: `{report['issue200_selected_trade_decision_count']}`\n"
        "- #200 selected trades with p_up disagreement: "
        f"`{report['issue200_selected_trade_p_up_disagreement_count']}`\n"
        f"- #200 guard accepted trades: `{report['issue200_guard_accepted_trade_count']}`\n"
        f"- Root cause: `{report['root_cause_classification']}`\n"
        f"- Recommendation: `{report['recommended_next_action']}`\n"
        "- Guard relaxation or candidate fitting performed: `false`\n"
    )
