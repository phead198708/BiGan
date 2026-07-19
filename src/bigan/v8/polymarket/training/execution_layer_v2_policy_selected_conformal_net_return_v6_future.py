"""Pre-register the strictly-later single-use future evaluation for #207 v6."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    SOURCE_BOUNDARY_SCHEMA_VERSION,
    load_and_validate_persistent_outcome_blind_index,
    validate_persistent_outcome_blind_collector_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    CANDIDATE_NAME,
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_text,
    validate_policy_selected_conformal_v6_profile,
)

SCHEMA_PREFIX = "bigan-v8-policy-selected-conformal-net-return-v6-future"
PREREG_REPORT_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-preregistration-report-v1"
PREREG_MANIFEST_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-preregistration-manifest-v1"
CANDIDATE_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-policy-selected-conformal-net-return-v6-research-candidate-freeze-manifest-v1"
)
MATCHED_BASELINE_NAME = "guard_compatible_direct_net_return_v4"


@dataclass(frozen=True, slots=True)
class PolicySelectedConformalV6FuturePreRegistrationConfig:
    """Pinned candidate and collector prefix before future-row selection."""

    run_id: str
    output_dir: Path | str
    candidate_manifest_path: Path | str
    expected_candidate_manifest_sha256: str
    baseline_manifest_path: Path | str
    expected_baseline_manifest_sha256: str
    collector_protocol_path: Path | str
    expected_collector_protocol_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    builder_git_commit: str
    preregistration_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_candidate_manifest_sha256",
            "expected_baseline_manifest_sha256",
            "expected_collector_protocol_sha256",
            "expected_collector_index_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        _require_git_sha(self.builder_git_commit)
        if self.preregistration_created_ts <= 0:
            raise ValueError("preregistration_created_ts must be positive")
        for field in (
            "output_dir",
            "candidate_manifest_path",
            "baseline_manifest_path",
            "collector_protocol_path",
            "collector_index_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def pre_register_policy_selected_conformal_v6_future_evaluation(
    config: PolicySelectedConformalV6FuturePreRegistrationConfig,
) -> dict[str, Any]:
    """Freeze candidate, baseline, prior identities, and future sizing without targets."""

    candidate_path = config.candidate_manifest_path.resolve()
    baseline_path = config.baseline_manifest_path.resolve()
    protocol_path = config.collector_protocol_path.resolve()
    index_path = config.collector_index_path.resolve()
    _verify_pin(candidate_path, config.expected_candidate_manifest_sha256, "v6 candidate")
    _verify_pin(baseline_path, config.expected_baseline_manifest_sha256, "matched v4 baseline")
    _verify_pin(protocol_path, config.expected_collector_protocol_sha256, "collector protocol")
    _verify_pin(index_path, config.expected_collector_index_sha256, "collector index")

    candidate = _load_json(candidate_path)
    candidate_lineage = _validate_v6_candidate(candidate)
    profile_descriptor = candidate_lineage["profile"]
    profile = _load_json(Path(profile_descriptor["path"]))
    validate_policy_selected_conformal_v6_profile(profile)
    baseline_lineage = _validate_matched_v4_baseline(
        _load_json(baseline_path),
        expected_model_sha256=str(profile["frozen_upstream"]["matched_v4_model_sha256"]),
    )
    protocol = _load_json(protocol_path)
    validate_persistent_outcome_blind_collector_protocol(protocol)
    if protocol.get("labels_outcomes_or_pnl_opened") is not False:
        raise ValueError("collector protocol target sealing invalid")
    index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    if not index_rows:
        raise ValueError("collector index is empty")
    if any(row.get("labels_outcomes_or_pnl_opened") is not False for row in index_rows):
        raise ValueError("collector index target sealing invalid")

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    index_snapshot_path = run_dir / "collector_index_prefix_at_v6_future_preregistration.jsonl"
    index_snapshot_path.write_bytes(index_path.read_bytes())
    if _sha256_file(index_snapshot_path) != config.expected_collector_index_sha256:
        raise ValueError("collector index changed while future prefix snapshot was written")

    candidate_freeze_ts = int(candidate.get("candidate_freeze_created_ts") or 0)
    if candidate_freeze_ts <= 0 or candidate_freeze_ts >= config.preregistration_created_ts:
        raise ValueError("future preregistration must occur after v6 candidate freeze")
    max_prior_market_end_ts = max(int(row.get("market_end_ts") or 0) for row in index_rows)
    minimum_collection_ts = max(
        max_prior_market_end_ts,
        candidate_freeze_ts,
        config.preregistration_created_ts,
    ) + 1
    prior_market_ids = sorted(
        {str(row["market_id"]) for row in index_rows if row.get("market_id")}
    )
    prior_slugs = sorted({str(row["slug"]) for row in index_rows if row.get("slug")})
    prior_source_hashes = sorted(
        {str(row["source_row_hash"]) for row in index_rows if row.get("source_row_hash")}
    )
    source_boundary = {
        "schema_version": SOURCE_BOUNDARY_SCHEMA_VERSION,
        "minimum_collection_decision_ts": minimum_collection_ts,
        "minimum_collection_index_sequence": len(index_rows) + 1,
        "max_prior_market_end_ts": max_prior_market_end_ts,
        "candidate_freeze_created_ts": candidate_freeze_ts,
        "preregistration_created_ts": config.preregistration_created_ts,
        "prior_market_ids": prior_market_ids,
        "prior_slugs": prior_slugs,
        "prior_source_row_hashes": prior_source_hashes,
        "prior_reference_hash": canonical_json_sha256(
            {
                "prior_market_ids": prior_market_ids,
                "prior_slugs": prior_slugs,
                "prior_source_row_hashes": prior_source_hashes,
            }
        ),
        "collector_index_prefix": _descriptor(index_snapshot_path),
        "candidate_manifest": _descriptor(candidate_path),
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    boundary_path = run_dir / "conformal_v6_future_source_boundary_manifest.json"
    _write_json(boundary_path, source_boundary)

    future = dict(profile["future_evaluation"])
    report = {
        "schema_version": PREREG_REPORT_SCHEMA_VERSION,
        "report_id": None,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "matched_baseline_name": MATCHED_BASELINE_NAME,
        "builder_git_commit": config.builder_git_commit,
        "preregistration_created_ts": config.preregistration_created_ts,
        "candidate_freeze_created_ts": candidate_freeze_ts,
        "collector_index_prefix_entry_count": len(index_rows),
        "collector_index_prefix_last_entry_sha256": index_rows[-1]["entry_sha256"],
        "minimum_collection_index_sequence": len(index_rows) + 1,
        "minimum_collection_decision_ts": minimum_collection_ts,
        "target_quality_valid_market_count": future["target_quality_valid_market_count"],
        "maximum_index_scan_count": future["maximum_index_scan_count"],
        "minimum_guard_accepted_unique_market_count": future[
            "minimum_guard_accepted_unique_market_count"
        ],
        "minimum_supported_side_market_count": future[
            "minimum_supported_side_market_count"
        ],
        "required_supported_sides": future["required_supported_sides"],
        "pnl_hard_gate_aggregation": future["pnl_hard_gate_aggregation"],
        "action_and_action_family_pnl_diagnostic_only": True,
        "future_single_use_holdout": future["single_use_holdout"],
        "future_result_driven_rerun_or_tuning_allowed": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "prediction_attempted": False,
        "future_preregistration_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v6_future_preregistration_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _preregistration_markdown(report))
    manifest = {
        "schema_version": PREREG_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "preregistration_created_ts": config.preregistration_created_ts,
        "candidate_name": CANDIDATE_NAME,
        "candidate_manifest": _descriptor(candidate_path),
        "candidate_model": candidate_lineage["model"],
        "candidate_calibration_artifact": candidate_lineage["calibration_artifact"],
        "candidate_profile": profile_descriptor,
        "matched_baseline_manifest": _descriptor(baseline_path),
        "matched_baseline_model": baseline_lineage["model"],
        "matched_baseline_fit_profile": baseline_lineage["fit_profile"],
        "collector_protocol": _descriptor(protocol_path),
        "collector_index_prefix": _descriptor(index_snapshot_path),
        "source_boundary_manifest": _descriptor(boundary_path),
        "report": _descriptor(report_path),
        "candidate_freeze_created_ts": candidate_freeze_ts,
        "collector_index_prefix_entry_count": len(index_rows),
        "collector_index_prefix_last_entry_sha256": index_rows[-1]["entry_sha256"],
        "minimum_collection_index_sequence": len(index_rows) + 1,
        "minimum_collection_decision_ts": minimum_collection_ts,
        "target_quality_valid_market_count": future["target_quality_valid_market_count"],
        "maximum_index_scan_count": future["maximum_index_scan_count"],
        "minimum_guard_accepted_unique_market_count": future[
            "minimum_guard_accepted_unique_market_count"
        ],
        "minimum_supported_side_market_count": future[
            "minimum_supported_side_market_count"
        ],
        "required_supported_sides": future["required_supported_sides"],
        "future_labels_outcomes_or_pnl_opened": False,
        "prediction_attempted": False,
        "future_preregistration_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    manifest["future_preregistration_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v6_future_preregistration_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "source_boundary": source_boundary,
        "source_boundary_path": boundary_path,
        "source_boundary_sha256": _sha256_file(boundary_path),
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _validate_v6_candidate(candidate: dict[str, Any]) -> dict[str, dict[str, str]]:
    blockers = []
    if candidate.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        blockers.append("candidate_schema_invalid")
    if candidate.get("candidate_name") != CANDIDATE_NAME:
        blockers.append("candidate_name_invalid")
    if candidate.get("research_candidate_frozen") is not True:
        blockers.append("candidate_not_frozen")
    if candidate.get("calibration_gate_passed") is not True:
        blockers.append("candidate_calibration_gate_failed")
    if candidate.get("candidate_specific_future_evaluation_allowed") is not True:
        blockers.append("candidate_future_evaluation_not_allowed")
    if candidate.get("candidate_specific_future_evaluation_blocking_reason_codes") != []:
        blockers.append("candidate_future_evaluation_has_blockers")
    for key, expected in _blocked_safety_fields().items():
        if candidate.get(key) != expected:
            blockers.append(f"candidate_safety_invalid:{key}")
    if candidate.get("uses_204_outcomes_for_fitting") is not False:
        blockers.append("candidate_used_issue204_outcomes")
    if candidate.get("uses_204_pnl_for_tuning") is not False:
        blockers.append("candidate_used_issue204_pnl")
    if candidate.get("policy_pnl_computed") is not False:
        blockers.append("candidate_development_policy_pnl_computed")
    if candidate.get("calibration_check_labels_opened_by_fit") is not False:
        blockers.append("candidate_opened_calibration_check_labels")
    if blockers:
        raise ValueError("v6 candidate lineage invalid: " + ", ".join(blockers))
    model = _verified_descriptor(candidate.get("model"), "v6 model")
    calibration = _verified_descriptor(
        candidate.get("calibration_artifact"), "v6 calibration artifact"
    )
    profile = _verified_descriptor(candidate.get("profile"), "v6 profile")
    if candidate.get("model_sha256") != model["sha256"]:
        raise ValueError("v6 model SHA-256 mismatch in candidate manifest")
    return {"model": model, "calibration_artifact": calibration, "profile": profile}


def _validate_matched_v4_baseline(
    baseline: dict[str, Any],
    *,
    expected_model_sha256: str,
) -> dict[str, dict[str, str]]:
    blockers = []
    if baseline.get("candidate_name") != MATCHED_BASELINE_NAME:
        blockers.append("baseline_name_invalid")
    if baseline.get("research_candidate_frozen") is not True:
        blockers.append("baseline_not_frozen")
    if baseline.get("candidate_specific_future_evaluation_allowed") is not True:
        blockers.append("baseline_future_evaluation_not_allowed")
    if baseline.get("current_oof_validation_or_future_pnl_used_for_tuning") is not False:
        blockers.append("baseline_used_current_or_future_pnl_for_tuning")
    for key, expected in _blocked_safety_fields().items():
        if baseline.get(key) != expected:
            blockers.append(f"baseline_safety_invalid:{key}")
    if blockers:
        raise ValueError("matched v4 baseline lineage invalid: " + ", ".join(blockers))
    model = _verified_descriptor(baseline.get("model"), "matched v4 model")
    fit_profile = _verified_descriptor(baseline.get("fit_profile"), "matched v4 profile")
    if model["sha256"] != expected_model_sha256:
        raise ValueError("matched v4 model does not match preregistered upstream hash")
    return {"model": model, "fit_profile": fit_profile}


def _preregistration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #207 v6 future evaluation preregistration",
            "",
            f"- ready: `{report['future_preregistration_ready']}`",
            f"- collector prefix entries: `{report['collector_index_prefix_entry_count']}`",
            f"- first eligible sequence: `{report['minimum_collection_index_sequence']}`",
            f"- target quality-valid markets: `{report['target_quality_valid_market_count']}`",
            f"- scan cap: `{report['maximum_index_scan_count']}`",
            "- labels/outcomes/PnL opened: `false`",
            "- single-use / result-driven tuning: `true / false`",
            "- paper/live/promotion unlock: `false`",
            "",
        ]
    )
