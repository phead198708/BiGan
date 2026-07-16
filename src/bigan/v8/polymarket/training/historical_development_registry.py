"""Freeze an outcome-blind historical development-market registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.historical_corpus_compatibility import (
    FORBIDDEN_DECISION_FIELDS,
    HISTORICAL_DEVELOPMENT_COMPATIBLE,
    REQUIRED_FILES,
)

SCHEMA_PREFIX = "bigan-v8-historical-development-registry"
DEFAULT_SELECTED_MARKET_COUNT = 90
FUTURE_FRESH_CALIBRATION_MARKET_COUNT = 45
FUTURE_FRESH_CONFIRMATORY_MARKET_COUNT = 60
FUTURE_QUALITY_BUFFER_ATTEMPT_COUNT = 15


@dataclass(frozen=True, slots=True)
class HistoricalDevelopmentRegistryConfig:
    """Immutable inputs for the historical development registry freeze."""

    run_id: str
    output_dir: Path | str
    compatibility_report_path: Path | str
    expected_compatibility_report_sha256: str
    compatibility_rows_path: Path | str
    expected_compatibility_rows_sha256: str
    compatibility_manifest_path: Path | str
    expected_compatibility_manifest_sha256: str
    boundary_freeze_manifest_path: Path | str
    expected_boundary_freeze_manifest_sha256: str
    selected_market_count: int = DEFAULT_SELECTED_MARKET_COUNT
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.selected_market_count <= 0:
            raise ValueError("selected_market_count must be positive")
        for name, value in (
            (
                "compatibility report SHA-256",
                self.expected_compatibility_report_sha256,
            ),
            (
                "compatibility rows SHA-256",
                self.expected_compatibility_rows_sha256,
            ),
            (
                "compatibility manifest SHA-256",
                self.expected_compatibility_manifest_sha256,
            ),
            (
                "boundary freeze manifest SHA-256",
                self.expected_boundary_freeze_manifest_sha256,
            ),
        ):
            _require_sha256(value, name=name)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "compatibility_report_path", Path(self.compatibility_report_path)
        )
        object.__setattr__(
            self, "compatibility_rows_path", Path(self.compatibility_rows_path)
        )
        object.__setattr__(
            self, "compatibility_manifest_path", Path(self.compatibility_manifest_path)
        )
        object.__setattr__(
            self, "boundary_freeze_manifest_path", Path(self.boundary_freeze_manifest_path)
        )


def freeze_historical_development_registry(
    config: HistoricalDevelopmentRegistryConfig,
) -> dict[str, Any]:
    """Select the earliest compatible pre-boundary markets without labels."""

    report_path = config.compatibility_report_path.resolve()
    rows_path = config.compatibility_rows_path.resolve()
    compatibility_manifest_path = config.compatibility_manifest_path.resolve()
    boundary_path = config.boundary_freeze_manifest_path.resolve()
    _verify_pin(
        report_path,
        config.expected_compatibility_report_sha256,
        name="compatibility report",
    )
    _verify_pin(
        rows_path,
        config.expected_compatibility_rows_sha256,
        name="compatibility rows",
    )
    _verify_pin(
        compatibility_manifest_path,
        config.expected_compatibility_manifest_sha256,
        name="compatibility manifest",
    )
    _verify_pin(
        boundary_path,
        config.expected_boundary_freeze_manifest_sha256,
        name="boundary freeze manifest",
    )
    compatibility_report = _load_json(report_path)
    compatibility_rows = _load_jsonl(rows_path)
    compatibility_manifest = _load_json(compatibility_manifest_path)
    boundary_freeze = _load_json(boundary_path)
    _validate_compatibility_inputs(
        report=compatibility_report,
        rows=compatibility_rows,
        manifest=compatibility_manifest,
        report_path=report_path,
        rows_path=rows_path,
    )
    _validate_boundary_freeze(boundary_freeze)
    minimum_collection_decision_ts = int(
        boundary_freeze.get("minimum_collection_decision_ts") or 0
    )
    if minimum_collection_decision_ts <= 0:
        raise ValueError("boundary freeze minimum_collection_decision_ts is invalid")

    audited_rows = [
        _candidate_audit(
            row=row,
            minimum_collection_decision_ts=minimum_collection_decision_ts,
        )
        for row in compatibility_rows
    ]
    eligible_rows = sorted(
        (
            row
            for row in audited_rows
            if row["historical_development_registry_eligible"] is True
        ),
        key=lambda row: (
            int(row["minimum_decision_ts"]),
            int(row["maximum_decision_ts"]),
            str(row["market_id"]),
            str(row["corpus_dir"]),
        ),
    )
    if len(eligible_rows) < config.selected_market_count:
        raise ValueError(
            "insufficient eligible pre-boundary historical markets: "
            f"{len(eligible_rows)} < {config.selected_market_count}"
        )
    selected = eligible_rows[: config.selected_market_count]
    selected_market_ids = [str(row["market_id"]) for row in selected]
    if len(set(selected_market_ids)) != config.selected_market_count:
        raise ValueError("selected historical registry contains duplicate market identities")
    registry_rows = []
    for rank, row in enumerate(selected, start=1):
        registry_row = {
            "schema_version": f"{SCHEMA_PREFIX}-row-v1",
            "selection_rank": rank,
            "role": "historical_development_train",
            "market_id": row["market_id"],
            "round_slug": row["round_slug"],
            "corpus_id": row["corpus_id"],
            "corpus_dir": row["corpus_dir"],
            "minimum_decision_ts": row["minimum_decision_ts"],
            "maximum_decision_ts": row["maximum_decision_ts"],
            "maximum_feature_input_ts": row["maximum_feature_input_ts"],
            "strictly_before_boundary": row["strictly_before_boundary"],
            "compatibility_row_id": row["compatibility_row_id"],
            "compatibility_classification": row["compatibility_classification"],
            "artifact_pins": row["artifact_pins"],
            "fresh_calibration_eligible": False,
            "fresh_confirmatory_eligible": False,
            "labels_or_outcomes_used_for_selection": False,
            "outcome_values_loaded": False,
            "pnl_values_loaded": False,
            **compact_safety_fields(),
        }
        registry_row["registry_row_id"] = canonical_json_sha256(registry_row)
        registry_rows.append(registry_row)

    run_dir = (config.output_dir / config.run_id).resolve()
    if run_dir.exists() and not config.overwrite_existing:
        raise ValueError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    registry_rows_path = run_dir / "historical_development_registry_rows.jsonl"
    _write_jsonl(registry_rows_path, registry_rows)
    report = _build_report(
        config=config,
        compatibility_report=compatibility_report,
        compatibility_manifest=compatibility_manifest,
        report_path=report_path,
        rows_path=rows_path,
        compatibility_manifest_path=compatibility_manifest_path,
        boundary_path=boundary_path,
        boundary_freeze=boundary_freeze,
        audited_rows=audited_rows,
        eligible_rows=eligible_rows,
        selected=selected,
        registry_rows_path=registry_rows_path,
    )
    report_path_out = run_dir / "historical_development_registry_report.json"
    _write_json(report_path_out, report)
    markdown_path = run_dir / "historical_development_registry_report.md"
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "registry_status": "frozen_development_only",
        "selected_market_count": len(registry_rows),
        "selected_market_ids_sha256": canonical_json_sha256(selected_market_ids),
        "compatibility_report": _descriptor(report_path),
        "compatibility_rows": _descriptor(rows_path),
        "compatibility_manifest": _descriptor(compatibility_manifest_path),
        "boundary_freeze_manifest": _descriptor(boundary_path),
        "registry_rows": _descriptor(registry_rows_path),
        "registry_report": _descriptor(report_path_out),
        "registry_report_markdown": _descriptor(markdown_path),
        "minimum_collection_decision_ts": minimum_collection_decision_ts,
        "maximum_selected_decision_ts": max(
            int(row["maximum_decision_ts"]) for row in selected
        ),
        "labels_or_outcomes_used_for_selection": False,
        "outcome_values_loaded": False,
        "pnl_values_loaded": False,
        "fresh_calibration_eligible": False,
        "fresh_confirmatory_eligible": False,
        "active_issue175_collection_mutated": False,
        **compact_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "historical_development_registry_manifest.json"
    _write_json(manifest_path, manifest)
    descriptor = {
        "schema_version": f"{SCHEMA_PREFIX}-descriptor-v1",
        "run_id": config.run_id,
        "manifest_id": manifest["manifest_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "registry_rows_sha256": _sha256_file(registry_rows_path),
        "registry_report_sha256": _sha256_file(report_path_out),
        "selected_market_count": len(registry_rows),
        "selected_market_ids_sha256": manifest["selected_market_ids_sha256"],
        "development_only": True,
        "fresh_calibration_eligible": False,
        "fresh_confirmatory_eligible": False,
        **compact_safety_fields(),
    }
    descriptor["descriptor_id"] = canonical_json_sha256(descriptor)
    descriptor_path = run_dir / "historical_development_registry_descriptor.json"
    _write_json(descriptor_path, descriptor)
    return {
        "run_dir": run_dir,
        "registry_rows_path": registry_rows_path,
        "report_path": report_path_out,
        "markdown_path": markdown_path,
        "manifest_path": manifest_path,
        "descriptor_path": descriptor_path,
        "report": report,
        "manifest": manifest,
        "descriptor": descriptor,
    }


def _validate_compatibility_inputs(
    *,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    report_path: Path,
    rows_path: Path,
) -> None:
    if report.get("audit_mode") != "outcome_blind_read_only_historical_compatibility":
        raise ValueError("compatibility report is not outcome blind")
    access = dict(report.get("outcome_blind_access_audit") or {})
    if any(
        access.get(field) is not False
        for field in (
            "label_rows_content_parsed",
            "resolution_rows_content_parsed",
            "outcome_values_loaded",
            "pnl_values_loaded",
            "oracle_values_loaded",
            "validation_metrics_loaded",
        )
    ):
        raise ValueError("compatibility report opened forbidden evidence")
    if int(report.get("discovered_corpus_count") or -1) != len(rows):
        raise ValueError("compatibility report row count mismatch")
    if str(report.get("report_id") or "") != canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    ):
        raise ValueError("compatibility report id mismatch")
    report_rows = _verified_descriptor(
        report.get("compatibility_rows"), name="report compatibility rows"
    )
    if report_rows != _descriptor(rows_path):
        raise ValueError("compatibility report rows descriptor mismatch")
    manifest_report = _verified_descriptor(
        manifest.get("report"), name="manifest compatibility report"
    )
    manifest_rows = _verified_descriptor(
        manifest.get("compatibility_rows"), name="manifest compatibility rows"
    )
    if manifest_report != _descriptor(report_path):
        raise ValueError("compatibility manifest report descriptor mismatch")
    if manifest_rows != _descriptor(rows_path):
        raise ValueError("compatibility manifest rows descriptor mismatch")
    if manifest.get("input_inventory_hash") != report.get("input_inventory_hash"):
        raise ValueError("compatibility input inventory hash mismatch")
    if manifest.get("outcome_values_loaded") is not False:
        raise ValueError("compatibility manifest opened outcome values")
    if manifest.get("pnl_values_loaded") is not False:
        raise ValueError("compatibility manifest opened PnL values")
    expected_manifest_id = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    )
    if str(manifest.get("manifest_id") or "") != expected_manifest_id:
        raise ValueError("compatibility manifest id mismatch")
    for row in rows:
        expected_row_id = canonical_json_sha256(
            {key: value for key, value in row.items() if key != "row_id"}
        )
        if str(row.get("row_id") or "") != expected_row_id:
            raise ValueError("compatibility row id mismatch")


def _validate_boundary_freeze(boundary_freeze: dict[str, Any]) -> None:
    if boundary_freeze.get("schema_version") != (
        "bigan-v8-execution-layer-v2-pairwise-action-advantage-lcb-"
        "precollection-role-freeze-v1"
    ):
        raise ValueError("boundary freeze schema version is invalid")
    expected_freeze_id = canonical_json_sha256(
        {
            key: value
            for key, value in boundary_freeze.items()
            if key != "precollection_freeze_id"
        }
    )
    if str(boundary_freeze.get("precollection_freeze_id") or "") != expected_freeze_id:
        raise ValueError("boundary precollection freeze id mismatch")


def _candidate_audit(
    *,
    row: dict[str, Any],
    minimum_collection_decision_ts: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if row.get("classification") != HISTORICAL_DEVELOPMENT_COMPATIBLE:
        reasons.append("compatibility_classification_not_compatible")
    if row.get("deduplication_status") != "selected_unique_market":
        reasons.append("compatibility_row_not_unique_market_selection")
    if row.get("historical_development_fit_eligible") is not True:
        reasons.append("historical_development_fit_not_eligible")
    if row.get("fresh_confirmatory_eligible") is not False:
        reasons.append("compatibility_row_fresh_confirmatory_semantics_invalid")
    if row.get("outcome_values_loaded") is not False:
        reasons.append("compatibility_row_outcome_access_detected")
    if row.get("pnl_values_loaded") is not False:
        reasons.append("compatibility_row_pnl_access_detected")
    market_id = str(row.get("market_id") or "")
    if not market_id:
        reasons.append("market_identity_missing")
    artifact_pins = _verified_artifact_pins(row.get("file_inventory") or {}, reasons)
    feature_descriptor = artifact_pins.get("polymarket_feature_rows.jsonl")
    feature_rows: list[dict[str, Any]] = []
    if feature_descriptor:
        feature_rows = _load_jsonl(Path(feature_descriptor["path"]))
    feature_market_ids = {
        str(feature.get("market_id") or "")
        for feature in feature_rows
        if str(feature.get("market_id") or "")
    }
    if feature_market_ids != {market_id}:
        reasons.append("feature_market_identity_mismatch")
    decision_timestamps = [int(feature.get("decision_ts") or 0) for feature in feature_rows]
    input_timestamps = [int(feature.get("max_input_ts") or 0) for feature in feature_rows]
    if not decision_timestamps or any(value <= 0 for value in decision_timestamps):
        reasons.append("feature_decision_timestamp_missing")
    if any(
        max_input_ts > decision_ts
        for max_input_ts, decision_ts in zip(
            input_timestamps, decision_timestamps, strict=True
        )
    ):
        reasons.append("feature_timestamp_causality_violation")
    forbidden_fields = sorted(
        {
            field
            for feature in feature_rows
            for field in _find_fields(feature, FORBIDDEN_DECISION_FIELDS)
        }
    )
    if forbidden_fields:
        reasons.append("forbidden_decision_fields_present")
    maximum_decision_ts = max(decision_timestamps, default=0)
    strictly_before_boundary = (
        maximum_decision_ts > 0
        and maximum_decision_ts < minimum_collection_decision_ts
    )
    if not strictly_before_boundary:
        reasons.append("historical_market_not_strictly_before_boundary")
    return {
        "market_id": market_id,
        "round_slug": str(row.get("round_slug") or ""),
        "corpus_id": str(row.get("corpus_id") or ""),
        "corpus_dir": str(row.get("corpus_dir") or ""),
        "compatibility_row_id": str(row.get("row_id") or ""),
        "compatibility_classification": row.get("classification"),
        "minimum_decision_ts": min(decision_timestamps, default=0),
        "maximum_decision_ts": maximum_decision_ts,
        "maximum_feature_input_ts": max(input_timestamps, default=0),
        "strictly_before_boundary": strictly_before_boundary,
        "forbidden_decision_fields": forbidden_fields,
        "artifact_pins": artifact_pins,
        "reason_codes": sorted(set(reasons)),
        "historical_development_registry_eligible": not reasons,
    }


def _verified_artifact_pins(
    inventory: dict[str, Any],
    reasons: list[str],
) -> dict[str, dict[str, Any]]:
    pins: dict[str, dict[str, Any]] = {}
    for filename in REQUIRED_FILES:
        value = inventory.get(filename)
        if not isinstance(value, dict):
            reasons.append(f"required_artifact_pin_missing:{filename}")
            continue
        path = Path(str(value.get("path") or "")).resolve()
        expected_sha256 = str(value.get("sha256") or "")
        if not path.is_file():
            reasons.append(f"required_artifact_missing:{filename}")
            continue
        if not _is_sha256(expected_sha256):
            reasons.append(f"required_artifact_hash_invalid:{filename}")
            continue
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            reasons.append(f"required_artifact_hash_mismatch:{filename}")
            continue
        pins[filename] = {
            "path": str(path),
            "sha256": actual_sha256,
            "size_bytes": path.stat().st_size,
            "semantic_content_parsed": filename
            not in {
                "polymarket_label_rows.jsonl",
                "polymarket_resolution_events.jsonl",
            },
        }
    return dict(sorted(pins.items()))


def _build_report(
    *,
    config: HistoricalDevelopmentRegistryConfig,
    compatibility_report: dict[str, Any],
    compatibility_manifest: dict[str, Any],
    report_path: Path,
    rows_path: Path,
    compatibility_manifest_path: Path,
    boundary_path: Path,
    boundary_freeze: dict[str, Any],
    audited_rows: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    registry_rows_path: Path,
) -> dict[str, Any]:
    exclusion_distribution: dict[str, int] = {}
    for row in audited_rows:
        for reason in row["reason_codes"]:
            exclusion_distribution[reason] = exclusion_distribution.get(reason, 0) + 1
    selected_market_ids = [str(row["market_id"]) for row in selected]
    selected_corpus_manifest_hashes = [
        str(
            row["artifact_pins"]["polymarket_corpus_manifest.json"]["sha256"]
        )
        for row in selected
    ]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "registry_status": "frozen_development_only",
        "selection_method": (
            "earliest_compatible_unique_pre_boundary_markets_chronological_v1"
        ),
        "compatibility_report": _descriptor(report_path),
        "compatibility_rows": _descriptor(rows_path),
        "compatibility_manifest": _descriptor(compatibility_manifest_path),
        "compatibility_report_id": compatibility_report["report_id"],
        "compatibility_input_inventory_hash": compatibility_report[
            "input_inventory_hash"
        ],
        "compatibility_manifest_id": compatibility_manifest["manifest_id"],
        "boundary_freeze_manifest": _descriptor(boundary_path),
        "boundary_precollection_freeze_id": boundary_freeze.get(
            "precollection_freeze_id"
        ),
        "boundary_git_commit": boundary_freeze.get("git_commit"),
        "minimum_collection_decision_ts": int(
            boundary_freeze["minimum_collection_decision_ts"]
        ),
        "compatibility_input_row_count": len(audited_rows),
        "eligible_pre_boundary_market_count": len(eligible_rows),
        "selected_market_count": len(selected),
        "selected_market_ids": selected_market_ids,
        "selected_market_ids_sha256": canonical_json_sha256(selected_market_ids),
        "selected_corpus_manifest_hashes_sha256": canonical_json_sha256(
            selected_corpus_manifest_hashes
        ),
        "minimum_selected_decision_ts": min(
            int(row["minimum_decision_ts"]) for row in selected
        ),
        "maximum_selected_decision_ts": max(
            int(row["maximum_decision_ts"]) for row in selected
        ),
        "maximum_selected_feature_input_ts": max(
            int(row["maximum_feature_input_ts"]) for row in selected
        ),
        "all_selected_strictly_before_boundary": all(
            row["strictly_before_boundary"] is True for row in selected
        ),
        "duplicate_selected_market_count": len(selected_market_ids)
        - len(set(selected_market_ids)),
        "eligible_not_selected_market_count": len(eligible_rows) - len(selected),
        "exclusion_reason_distribution": dict(sorted(exclusion_distribution.items())),
        "registry_rows": _descriptor(registry_rows_path),
        "forbidden_evidence_access_audit": {
            "selection_uses_only_compatibility_time_and_identity": True,
            "label_rows_semantic_content_parsed": False,
            "resolution_rows_semantic_content_parsed": False,
            "outcome_values_loaded": False,
            "pnl_values_loaded": False,
            "oracle_values_loaded": False,
            "oof_metrics_loaded": False,
            "validation_metrics_loaded": False,
            "confirmatory_metrics_loaded": False,
        },
        "future_hybrid_role_plan": {
            "planning_only": True,
            "historical_development_train_market_count": len(selected),
            "fresh_calibration_market_count": FUTURE_FRESH_CALIBRATION_MARKET_COUNT,
            "fresh_confirmatory_market_count": FUTURE_FRESH_CONFIRMATORY_MARKET_COUNT,
            "minimum_fresh_valid_market_count": (
                FUTURE_FRESH_CALIBRATION_MARKET_COUNT
                + FUTURE_FRESH_CONFIRMATORY_MARKET_COUNT
            ),
            "quality_buffer_attempt_count": FUTURE_QUALITY_BUFFER_ATTEMPT_COUNT,
            "estimated_initial_capture_attempt_count": (
                FUTURE_FRESH_CALIBRATION_MARKET_COUNT
                + FUTURE_FRESH_CONFIRMATORY_MARKET_COUNT
                + FUTURE_QUALITY_BUFFER_ATTEMPT_COUNT
            ),
            "separate_future_precollection_freeze_required": True,
            "active_issue175_collection_may_be_reused": False,
            "fresh_confirmatory_history_substitution_allowed": False,
        },
        "active_lineage_invariants": {
            "issue175_collection_mutated": False,
            "issue178_collector_mutated": False,
            "issue179_support_gate_mutated": False,
            "new_collection_started": False,
            "model_fit_attempted": False,
            "labels_opened_for_selection": False,
        },
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    future = report["future_hybrid_role_plan"]
    lines = [
        "# Historical Development Registry",
        "",
        f"- run id: `{report['run_id']}`",
        f"- status: `{report['registry_status']}`",
        (
            "- eligible compatible pre-boundary markets: "
            f"`{report['eligible_pre_boundary_market_count']}`"
        ),
        f"- selected development markets: `{report['selected_market_count']}`",
        (
            "- boundary decision timestamp: "
            f"`{report['minimum_collection_decision_ts']}`"
        ),
        (
            "- maximum selected decision timestamp: "
            f"`{report['maximum_selected_decision_ts']}`"
        ),
        (
            "- all selected strictly before boundary: "
            f"`{str(report['all_selected_strictly_before_boundary']).lower()}`"
        ),
        f"- duplicate selected markets: `{report['duplicate_selected_market_count']}`",
        "- label/outcome/PnL values used for selection: `false`",
        "",
        "## Future Hybrid Role Plan",
        "",
        (
            "- historical development train: "
            f"`{future['historical_development_train_market_count']}`"
        ),
        f"- fresh calibration: `{future['fresh_calibration_market_count']}`",
        f"- fresh confirmatory: `{future['fresh_confirmatory_market_count']}`",
        (
            "- estimated initial fresh capture attempts: "
            f"`{future['estimated_initial_capture_attempt_count']}`"
        ),
        "",
        "This registry is development-only. It does not change #175, start collection, "
        "or qualify any market as fresh calibration/confirmatory evidence.",
        "",
        "## Safety",
        "",
        "- paper only: `true`",
        "- capital at risk: `false`",
        "- Polymarket writes: `false`",
        "- wallet signing: `false`",
        "- source/freeze/promotion/handoff: `false`",
    ]
    return "\n".join(lines) + "\n"


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(value.get("path") or "")).resolve()
    sha256 = str(value.get("sha256") or "")
    _require_sha256(sha256, name=f"{name} SHA-256")
    _verify_pin(path, sha256, name=name)
    return {"path": str(path), "sha256": sha256}


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    if not _is_sha256(value.lower()):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object row: {path}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _find_fields(payload: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in forbidden:
                found.add(str(key))
            found.update(_find_fields(value, forbidden))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_find_fields(value, forbidden))
    return found
