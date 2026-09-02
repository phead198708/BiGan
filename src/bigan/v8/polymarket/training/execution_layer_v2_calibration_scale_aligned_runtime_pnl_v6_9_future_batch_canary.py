"""Target-free batch canary for the frozen v6.9 runtime-PnL mapping."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9 import (
    CANDIDATE_NAME,
    FORBIDDEN_TARGET_FIELDS,
    apply_v6_9_score_to_runtime_pnl_mapping,
)
from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9 import (
    _blocked_safety_fields as _v6_9_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    build_v6_7_target_free_candidate_rows,
    select_v6_7_target_free_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _blocked_safety_fields,
    _descriptor,
    _find_nonempty_fields,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)

SCHEMA_PREFIX = "bigan-v8-calibration-scale-aligned-v6-9-future-batch-canary"


@dataclass(frozen=True, slots=True)
class CalibrationScaleAlignedV69FutureBatchCanaryConfig:
    """Pinned inputs for one outcome-blind v6.9 batch liveness check."""

    run_id: str
    output_dir: Path | str
    v6_2_batch_canary_manifest_path: Path | str
    expected_v6_2_batch_canary_manifest_sha256: str
    candidate_manifest_path: Path | str
    expected_candidate_manifest_sha256: str
    collection_plan_path: Path | str
    expected_collection_plan_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in (
            "expected_v6_2_batch_canary_manifest_sha256",
            "expected_candidate_manifest_sha256",
            "expected_collection_plan_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        for name in (
            "output_dir",
            "v6_2_batch_canary_manifest_path",
            "candidate_manifest_path",
            "collection_plan_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_v6_9_future_collection_plan(
    plan: dict[str, Any],
    *,
    candidate_manifest: dict[str, Any],
    candidate_manifest_sha256: str,
) -> None:
    """Validate the frozen collection-only contract before any capture starts."""

    expected = {
        "schema_version": "bigan-v8-v6-9-future-collection-plan-v1",
        "issue_number": 231,
        "candidate_name": CANDIDATE_NAME,
        "frozen": True,
        "outcome_blind_collection_only": True,
        "target_quality_valid_market_count": 120,
        "maximum_attempted_market_count": 180,
        "batch_round_count": 12,
        "minimum_quality_valid_markets_for_batch_liveness": 6,
        "minimum_guard_accepted_markets_for_batch_liveness": 1,
        "consecutive_zero_action_batch_limit": 1,
        "side_count_hard_gate_enabled": False,
        "side_quota_applied": False,
        "labels_outcomes_or_pnl_opened": False,
    }
    mismatches = [name for name, value in expected.items() if plan.get(name) != value]
    if mismatches:
        raise ValueError("v6.9 collection plan contract mismatch: " + ",".join(mismatches))
    if plan.get("candidate_manifest_sha256") != candidate_manifest_sha256.lower():
        raise ValueError("v6.9 collection plan candidate SHA-256 mismatch")
    if int(plan.get("minimum_market_start_ts_exclusive") or 0) != int(
        candidate_manifest.get("candidate_freeze_created_ts") or -1
    ):
        raise ValueError("v6.9 collection plan future boundary mismatch")
    required_hashes = {
        "profile_sha256": candidate_manifest["profile"]["sha256"],
        "mapping_artifact_sha256": candidate_manifest["mapping_artifact"]["sha256"],
        "target_free_liveness_report_sha256": candidate_manifest["liveness_report"]["sha256"],
    }
    if any(plan.get(name) != value for name, value in required_hashes.items()):
        raise ValueError("v6.9 collection plan frozen lineage mismatch")
    if int(plan.get("collection_plan_created_ts") or 0) <= int(
        candidate_manifest["candidate_freeze_created_ts"]
    ):
        raise ValueError("v6.9 collection plan was not created after candidate freeze")
    if plan.get("issue229_outcomes_must_remain_sealed") is not True:
        raise ValueError("v6.9 collection plan does not seal #229 outcomes")
    for key, value in _blocked_safety_fields().items():
        if plan.get(key) != value:
            raise ValueError(f"v6.9 collection plan safety mismatch: {key}")


def run_v6_9_future_batch_canary(
    config: CalibrationScaleAlignedV69FutureBatchCanaryConfig,
) -> dict[str, Any]:
    """Apply frozen v6.9 scoring to one completed batch without target access."""

    v6_2_manifest_path = config.v6_2_batch_canary_manifest_path.resolve()
    candidate_manifest_path = config.candidate_manifest_path.resolve()
    collection_plan_path = config.collection_plan_path.resolve()
    _verify_pin(
        v6_2_manifest_path,
        config.expected_v6_2_batch_canary_manifest_sha256,
        "v6.2 batch canary manifest",
    )
    _verify_pin(
        candidate_manifest_path,
        config.expected_candidate_manifest_sha256,
        "v6.9 candidate manifest",
    )
    _verify_pin(
        collection_plan_path,
        config.expected_collection_plan_sha256,
        "v6.9 collection plan",
    )
    v6_2_manifest = _load_json(v6_2_manifest_path)
    candidate_manifest = _load_json(candidate_manifest_path)
    collection_plan = _load_json(collection_plan_path)
    _validate_candidate_manifest(candidate_manifest)
    validate_v6_9_future_collection_plan(
        collection_plan,
        candidate_manifest=candidate_manifest,
        candidate_manifest_sha256=config.expected_candidate_manifest_sha256,
    )
    if v6_2_manifest.get("labels_outcomes_or_pnl_opened") is not False:
        raise ValueError("v6.2 batch target sealing invalid")

    v6_2_report_descriptor = _verified_descriptor(v6_2_manifest.get("report"), "v6.2 batch report")
    v6_2_report = _load_json(Path(v6_2_report_descriptor["path"]))
    if (
        v6_2_report.get("target_free_scoring_passed") is not True
        or v6_2_report.get("labels_outcomes_or_pnl_opened") is not False
    ):
        raise ValueError("v6.2 target-free batch scoring not passed")
    development_manifest_descriptor = _verified_descriptor(
        v6_2_manifest.get("development_batch_canary_manifest"),
        "development batch canary manifest",
    )
    development_manifest = _load_json(Path(development_manifest_descriptor["path"]))
    action_descriptor = _verified_descriptor(
        development_manifest.get("five_action_grid"), "five action grid"
    )
    prediction_descriptor = _verified_descriptor(
        v6_2_manifest.get("mean_ev_scored_rows"), "v6.2 scored rows"
    )
    action_rows = _load_jsonl(Path(action_descriptor["path"]))
    v6_2_predictions = _load_jsonl(Path(prediction_descriptor["path"]))
    forbidden = _find_nonempty_fields(action_rows + v6_2_predictions, FORBIDDEN_TARGET_FIELDS)
    if forbidden:
        raise ValueError("v6.9 future batch contains forbidden target fields")

    freeze_ts = int(candidate_manifest["candidate_freeze_created_ts"])
    if any(
        int(row.get("decision_ts") or 0) <= freeze_ts
        or int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
        for row in action_rows
    ):
        raise ValueError("v6.9 future batch is not strictly later and causal")
    future_market_ids = {str(row.get("market_id") or "") for row in action_rows}
    if not future_market_ids or "" in future_market_ids:
        raise ValueError("v6.9 future batch market identity missing")
    prior_ids, prior_reference_hash = _prior_market_reference(candidate_manifest)
    if future_market_ids & prior_ids:
        raise ValueError("v6.9 future batch overlaps frozen candidate lineage")

    issue229_manifest = _load_json(
        Path(
            _verified_descriptor(
                candidate_manifest["issue229_target_free_freeze_manifest"],
                "#229 target-free freeze",
            )["path"]
        )
    )
    v6_7_freeze = _load_json(
        Path(
            _verified_descriptor(
                issue229_manifest["candidate_freeze_manifest"], "v6.7 candidate freeze"
            )["path"]
        )
    )
    v6_7_profile_descriptor = _verified_descriptor(v6_7_freeze["profile"], "v6.7 candidate profile")
    v6_7_profile = _load_json(Path(v6_7_profile_descriptor["path"]))
    candidate_rows, candidate_summary = build_v6_7_target_free_candidate_rows(
        v6_2_predictions,
        action_rows=action_rows,
        profile=v6_7_profile,
    )
    base_selected = [
        {**row, "v6_7_base_score": float(row["v6_7_selection_score"])}
        for row in select_v6_7_target_free_rows(candidate_rows, profile=v6_7_profile)
    ]
    mapping_descriptor = _verified_descriptor(
        candidate_manifest["mapping_artifact"], "v6.9 mapping artifact"
    )
    mapping_artifact = _load_json(Path(mapping_descriptor["path"]))
    mapped_rows = apply_v6_9_score_to_runtime_pnl_mapping(
        base_selected, mapping_artifact=mapping_artifact
    )
    accepted = [
        row
        for row in mapped_rows
        if row.get("microstructure_safety_passed") is True
        and row.get("hard_execution_safety_thresholds_unchanged") is True
        and row.get("exposure_duplicate_position_and_sizing_guards_unchanged") is True
    ]
    accepted_ids = {str(row["market_id"]) for row in accepted}
    quality_count = int(v6_2_report["quality_valid_market_count"])
    minimum_quality = int(collection_plan["minimum_quality_valid_markets_for_batch_liveness"])
    minimum_accepted = int(collection_plan["minimum_guard_accepted_markets_for_batch_liveness"])
    liveness_evaluated = quality_count >= minimum_quality
    liveness_passed = not liveness_evaluated or len(accepted_ids) >= minimum_accepted
    blockers = [] if liveness_passed else ["completed_batch_zero_v6_9_guard_accepted_actions"]

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    candidate_rows_path = run_dir / "v6_9_future_batch_candidate_rows.jsonl"
    base_selected_path = run_dir / "v6_9_future_batch_v6_7_base_selected_rows.jsonl"
    mapped_path = run_dir / "v6_9_future_batch_mapped_rows.jsonl"
    accepted_path = run_dir / "v6_9_future_batch_guard_accepted_rows.jsonl"
    _write_jsonl(candidate_rows_path, candidate_rows)
    _write_jsonl(base_selected_path, base_selected)
    _write_jsonl(mapped_path, mapped_rows)
    _write_jsonl(accepted_path, accepted)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "batch_id": v6_2_report["batch_id"],
        "candidate_name": CANDIDATE_NAME,
        "candidate_manifest_sha256": config.expected_candidate_manifest_sha256.lower(),
        "collection_plan_sha256": config.expected_collection_plan_sha256.lower(),
        "candidate_freeze_created_ts": freeze_ts,
        "future_strictly_later_and_disjoint_passed": True,
        "prior_reference_market_count": len(prior_ids),
        "prior_reference_hash": prior_reference_hash,
        "bounded_batch_complete": v6_2_report["bounded_batch_complete"],
        "source_sequence_start": int(v6_2_report["source_sequence_start"]),
        "source_sequence_end": int(v6_2_report["source_sequence_end"]),
        "indexed_market_count": int(v6_2_report["indexed_market_count"]),
        "quality_valid_market_count": quality_count,
        "future_market_ids": sorted(future_market_ids),
        "five_action_grid_market_count": len(future_market_ids),
        "five_action_grid_row_count": len(action_rows),
        "v6_7_candidate_summary": candidate_summary,
        "v6_7_base_selected_market_count": len({str(row["market_id"]) for row in base_selected}),
        "positive_mapped_score_unique_market_count": len(
            {str(row["market_id"]) for row in mapped_rows}
        ),
        "guard_accepted_unique_market_count": len(accepted_ids),
        "guard_accepted_market_ids": sorted(accepted_ids),
        "guard_accepted_side_distribution_diagnostic": _side_distribution(accepted),
        "guard_accepted_action_distribution": dict(
            sorted(Counter(str(row["action"]) for row in accepted).items())
        ),
        "batch_action_liveness_evaluated": liveness_evaluated,
        "batch_action_liveness_passed": liveness_passed,
        "batch_action_liveness_blocking_reason_codes": blockers,
        "minimum_quality_valid_markets_for_batch_liveness": minimum_quality,
        "minimum_guard_accepted_markets_for_batch_liveness": minimum_accepted,
        "candidate_model_scoring_attempted": True,
        "labels_outcomes_or_pnl_opened": False,
        "settlement_provider_called": False,
        "threshold_or_guard_tuning_performed": False,
        "source_score_mutated": False,
        "side_count_hard_gate_enabled": False,
        "side_quota_applied": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_9_future_batch_action_liveness_report.json"
    report_md_path = report_path.with_suffix(".md")
    _write_json(report_path, report)
    _write_text(report_md_path, _batch_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "batch_id": report["batch_id"],
        "v6_2_batch_canary_manifest": _descriptor(v6_2_manifest_path),
        "candidate_manifest": _descriptor(candidate_manifest_path),
        "collection_plan": _descriptor(collection_plan_path),
        "v6_7_candidate_profile": v6_7_profile_descriptor,
        "mapping_artifact": mapping_descriptor,
        "candidate_rows": _descriptor(candidate_rows_path),
        "v6_7_base_selected_rows": _descriptor(base_selected_path),
        "mapped_rows": _descriptor(mapped_path),
        "guard_accepted_rows": _descriptor(accepted_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "labels_outcomes_or_pnl_opened": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_9_future_batch_action_liveness_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def build_v6_9_future_cumulative_canary(
    batch_reports: list[dict[str, Any]],
    *,
    run_id: str,
    collection_plan: dict[str, Any],
    collection_plan_sha256: str,
) -> dict[str, Any]:
    """Aggregate only target-free support and stop at the frozen collection boundary."""

    if not batch_reports:
        raise ValueError("at least one v6.9 batch report is required")
    seen_batches: set[str] = set()
    seen_markets: set[str] = set()
    accepted_markets: set[str] = set()
    accepted_by_side: Counter[str] = Counter()
    previous_end: int | None = None
    for report in batch_reports:
        if report.get("candidate_name") != CANDIDATE_NAME:
            raise ValueError("v6.9 cumulative candidate identity mismatch")
        if report.get("collection_plan_sha256") != collection_plan_sha256.lower():
            raise ValueError("v6.9 cumulative collection plan mismatch")
        if report.get("labels_outcomes_or_pnl_opened") is not False:
            raise ValueError("v6.9 cumulative target sealing invalid")
        batch_id = str(report.get("batch_id") or "")
        if not batch_id or batch_id in seen_batches:
            raise ValueError("v6.9 cumulative batch identity missing or duplicate")
        seen_batches.add(batch_id)
        start = int(report.get("source_sequence_start") or 0)
        end = int(report.get("source_sequence_end") or 0)
        if start <= 0 or end < start or (previous_end is not None and start != previous_end + 1):
            raise ValueError("v6.9 cumulative sequence is invalid or non-contiguous")
        previous_end = end
        market_ids = set(report.get("future_market_ids") or [])
        if seen_markets & market_ids:
            raise ValueError("v6.9 cumulative market identity repeated")
        seen_markets.update(market_ids)
        accepted_ids = set(report.get("guard_accepted_market_ids") or [])
        accepted_markets.update(accepted_ids)
        accepted_by_side.update(report.get("guard_accepted_side_distribution_diagnostic") or {})
        for key, value in _blocked_safety_fields().items():
            if report.get(key) != value:
                raise ValueError(f"v6.9 cumulative safety mismatch: {key}")

    attempted = sum(int(report["indexed_market_count"]) for report in batch_reports)
    quality = sum(int(report["quality_valid_market_count"]) for report in batch_reports)
    target = int(collection_plan["target_quality_valid_market_count"])
    scan_cap = int(collection_plan["maximum_attempted_market_count"])
    zero_action_reports = [
        report
        for report in batch_reports
        if report["batch_action_liveness_evaluated"] is True
        and report["batch_action_liveness_passed"] is False
    ]
    blockers = []
    if len(zero_action_reports) >= int(collection_plan["consecutive_zero_action_batch_limit"]):
        blockers.append("v6_9_completed_batch_action_liveness_failed")
    if attempted >= scan_cap and quality < target:
        blockers.append("v6_9_collection_scan_cap_reached_before_quality_target")
    collection_complete = quality >= target and not blockers
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-cumulative-report-v1",
        "run_id": run_id,
        "candidate_name": CANDIDATE_NAME,
        "collection_plan_sha256": collection_plan_sha256.lower(),
        "completed_batch_count": len(batch_reports),
        "attempted_market_count": attempted,
        "quality_valid_market_count": quality,
        "target_quality_valid_market_count": target,
        "maximum_attempted_market_count": scan_cap,
        "guard_accepted_unique_market_count": len(accepted_markets),
        "guard_accepted_side_distribution_diagnostic": dict(sorted(accepted_by_side.items())),
        "side_count_hard_gate_enabled": False,
        "side_quota_applied": False,
        "future_confirmatory_collection_complete": collection_complete,
        "target_free_terminal_blocked": bool(blockers),
        "target_free_terminal_blocking_reason_codes": blockers,
        "labels_outcomes_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def write_v6_9_future_cumulative_canary(
    *,
    report: dict[str, Any],
    batch_report_paths: list[Path],
    collection_plan_path: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    run_dir = _prepare_run_dir(output_dir, run_id, overwrite=False)
    report_path = run_dir / "v6_9_future_cumulative_action_liveness_report.json"
    report_md_path = report_path.with_suffix(".md")
    _write_json(report_path, report)
    _write_text(report_md_path, _cumulative_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-cumulative-manifest-v1",
        "run_id": run_id,
        "collection_plan": _descriptor(collection_plan_path),
        "batch_reports": [_descriptor(path) for path in batch_report_paths],
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "labels_outcomes_or_pnl_opened": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_9_future_cumulative_action_liveness_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _validate_candidate_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "candidate_name": CANDIDATE_NAME,
        "candidate_scoring_frozen": True,
        "strictly_later_outcome_blind_collection_allowed": True,
        "mapping_gate_passed": True,
        "target_free_liveness_gate_passed": True,
        "current_issue229_outcomes_opened": False,
        "future_target_access_allowed": False,
    }
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatches:
        raise ValueError("v6.9 candidate manifest invalid: " + ",".join(mismatches))
    for key, value in _v6_9_blocked_safety_fields().items():
        if manifest.get(key) != value:
            raise ValueError(f"v6.9 candidate safety mismatch: {key}")


def _prior_market_reference(manifest: dict[str, Any]) -> tuple[set[str], str]:
    rows = []
    for key in ("issue229_v6_7_base_selected_rows", "runtime_target_rows"):
        descriptor = _verified_descriptor(manifest[key], f"v6.9 prior {key}")
        rows.extend(_load_jsonl(Path(descriptor["path"])))
    market_ids = {str(row.get("market_id") or "") for row in rows}
    market_ids.discard("")
    return market_ids, canonical_json_sha256(sorted(market_ids))


def _side_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(row.get("side") or row.get("selected_side") or "UNKNOWN") for row in rows
            ).items()
        )
    )


def _batch_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.9 Future Batch Action Liveness",
            "",
            f"- batch: `{report['batch_id']}`",
            f"- quality-valid markets: `{report['quality_valid_market_count']}`",
            f"- positive mapped markets: `{report['positive_mapped_score_unique_market_count']}`",
            f"- guard-accepted markets: `{report['guard_accepted_unique_market_count']}`",
            f"- side distribution (diagnostic): `{report['guard_accepted_side_distribution_diagnostic']}`",
            f"- action liveness passed: `{str(report['batch_action_liveness_passed']).lower()}`",
            f"- blockers: `{report['batch_action_liveness_blocking_reason_codes']}`",
            "- labels/outcomes/PnL opened: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _cumulative_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.9 Future Cumulative Action Liveness",
            "",
            f"- batches: `{report['completed_batch_count']}`",
            f"- attempted markets: `{report['attempted_market_count']}`",
            f"- quality-valid markets: `{report['quality_valid_market_count']}`",
            f"- guard-accepted markets: `{report['guard_accepted_unique_market_count']}`",
            f"- collection complete: `{str(report['future_confirmatory_collection_complete']).lower()}`",
            f"- terminal blocked: `{str(report['target_free_terminal_blocked']).lower()}`",
            f"- blockers: `{report['target_free_terminal_blocking_reason_codes']}`",
            "- labels/outcomes/PnL opened: `false`",
            "",
        ]
    )


__all__ = [
    "CalibrationScaleAlignedV69FutureBatchCanaryConfig",
    "build_v6_9_future_cumulative_canary",
    "run_v6_9_future_batch_canary",
    "validate_v6_9_future_collection_plan",
    "write_v6_9_future_cumulative_canary",
]
