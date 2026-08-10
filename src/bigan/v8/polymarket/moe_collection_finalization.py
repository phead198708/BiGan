"""Outcome-blind finalization for the frozen BTC 15m MoE collection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    ATTEMPT_CAP,
    LINEAGE_ID,
    MARKET_FAMILY,
    MARKET_HORIZON_SECONDS,
    REQUIRED_RAW_STREAMS,
    TARGET_MARKET_COUNT,
    AuthorizationExpectation,
    _write_new_frozen_json,
    _write_new_jsonl,
    build_confirmatory_capture_manifest,
    frozen_authorization_artifacts,
    select_exact_authorized_window,
    validate_authorized_attempt_ledger,
)
from bigan.v8.polymarket.moe_collection_observability import (
    CANDIDATE_BUNDLE_HASH,
    _current_feature_rows,
    _load_completed_captures,
    _load_runtime_bundle,
    _predict_market,
    _quality_observations,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

EXPECTED_EMPTY_FEATURE_ERRORS = {
    "no point-in-time Polymarket corpus feature rows",
    "no supported Polymarket corpus markets",
}


def freeze_exact_confirmatory_collection(
    *,
    service_root: Path | str,
    output_dir: Path | str,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Freeze the first 800 quality-valid markets without reading outcomes."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    service = Path(service_root).resolve()
    target = Path(output_dir).resolve()
    if not service.is_relative_to(repo_root) or not target.is_relative_to(repo_root):
        raise ValueError("collection finalization paths must remain repository-local")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite collection freeze: {target}")
    target.mkdir(parents=True, exist_ok=True)

    authorization = _manual_authorization_expectation(repo_root)
    captures = _load_completed_captures(service)
    if not TARGET_MARKET_COUNT <= len(captures) <= ATTEMPT_CAP:
        raise ValueError("completed attempt ledger cannot reach the exact target")
    bundle = _load_runtime_bundle(repo_root)
    observations = [
        _freeze_observation(
            capture,
            service_root=service,
            repository_root=repo_root,
            bundle=bundle,
        )
        for capture in captures
    ]

    raw_index_path = target / "raw_evidence_manifest_index.jsonl"
    raw_records = [row["raw_evidence"] for row in observations]
    _write_new_jsonl(raw_index_path, raw_records)
    raw_index_rel = raw_index_path.relative_to(repo_root).as_posix()
    raw_index_sha = sha256_file(raw_index_path)

    source_ledger_path = target / "authorized_attempt_ledger.jsonl"
    source_rows: list[dict[str, Any]] = []
    previous = "0" * 64
    for observation in observations:
        raw = observation["raw_evidence"]
        causality = raw["causality"]
        row = {
            "attempt_index": observation["attempt_index"],
            "attempt_id": observation["attempt_id"],
            "market_id": observation["market_id"],
            "market_family": MARKET_FAMILY,
            "market_horizon_seconds": MARKET_HORIZON_SECONDS,
            "market_start_ts": observation["market_start_ts"],
            "decision_ts": causality["decision_ts"],
            "available_at_ts": causality["available_at_ts"],
            "max_input_ts": causality["max_input_ts"],
            "feature_cutoff_ts": causality["feature_cutoff_ts"],
            "paired_executable_ask_capture_attempted": True,
            "missing_values_encoded_as_numeric_zero": False,
            "complement_quote_proxy_used": False,
            "raw_evidence_manifest_index_path": raw_index_rel,
            "raw_evidence_manifest_index_sha256": raw_index_sha,
            "raw_evidence_manifest_sha256": raw["record_sha256"],
            "previous_entry_sha256": previous,
        }
        row["entry_sha256"] = canonical_json_sha256(row)
        previous = row["entry_sha256"]
        source_rows.append(row)
    _write_new_jsonl(source_ledger_path, source_rows)

    normalized_path = target / "normalized_attempt_ledger.jsonl"
    normalized = validate_authorized_attempt_ledger(
        authorization=authorization,
        attempt_ledger_path=source_ledger_path,
        expected_attempt_ledger_sha256=sha256_file(source_ledger_path),
        normalized_output_path=normalized_path,
        repository_root=repo_root,
    )
    selection = select_exact_authorized_window(
        normalized_attempt_ledger_path=normalized_path,
        expected_normalized_attempt_ledger_sha256=sha256_file(normalized_path),
        authorization=authorization,
        repository_root=repo_root,
    )
    if selection["window_complete"] is not True:
        raise ValueError("exact 800-market collection window is incomplete")

    by_attempt = {row["attempt_index"]: row for row in observations}
    candidate_rows = []
    baseline_rows = []
    for selected in selection["selected_rows"]:
        observation = by_attempt[int(selected["attempt_index"])]
        if observation["quality_valid"] is not True:
            raise ValueError("selected market is not independently quality-valid")
        candidate_rows.append(_candidate_decision_row(observation))
        baseline_rows.append(_baseline_decision_row(observation, authorization))
    candidate_path = target / "candidate_decision_rows.jsonl"
    baseline_path = target / "baseline_decision_rows.jsonl"
    _write_new_jsonl(candidate_path, candidate_rows)
    _write_new_jsonl(baseline_path, baseline_rows)

    config_root = repo_root / "examples/v8/polymarket_configs" / LINEAGE_ID
    collector_path = config_root / "moe_confirmatory_collector_protocol_r2.json"
    capture = build_confirmatory_capture_manifest(
        authorization=authorization,
        normalized_attempt_ledger_path=normalized_path,
        expected_normalized_attempt_ledger_sha256=sha256_file(normalized_path),
        candidate_decision_rows_path=candidate_path,
        expected_candidate_decision_rows_sha256=sha256_file(candidate_path),
        baseline_decision_rows_path=baseline_path,
        expected_baseline_decision_rows_sha256=sha256_file(baseline_path),
        raw_evidence_manifest_index_path=raw_index_path,
        expected_raw_evidence_manifest_index_sha256=raw_index_sha,
        collector_protocol_path=collector_path,
        expected_collector_protocol_sha256=sha256_file(collector_path),
        output_dir=target,
        repository_root=repo_root,
    )

    quarantine = _quarantine_payload(
        service_root=service,
        observations=observations,
        selection=selection,
        repository_root=repo_root,
    )
    quarantine_frozen = _write_new_frozen_json(
        target / "post_boundary_capture_quarantine.json",
        quarantine,
    )
    report = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-finalization-v1",
        "lineage_id": LINEAGE_ID,
        "role": "outcome_blind_exact_population_freeze",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "attempts_consumed": len(observations),
        "quality_valid_market_count": int(normalized["quality_valid_count"]),
        "exact_market_count": len(candidate_rows),
        "window_stop_attempt_index": selection["window_stop_attempt_index"],
        "audit_only_post_boundary_attempt_indices": selection[
            "audit_only_post_boundary_attempt_indices"
        ],
        "incomplete_capture_directory_count": len(
            quarantine["incomplete_capture_directories"]
        ),
        "artifacts": {
            "source_attempt_ledger": _descriptor(source_ledger_path, repo_root),
            "normalized_attempt_ledger": _descriptor(normalized_path, repo_root),
            "raw_evidence_manifest_index": _descriptor(raw_index_path, repo_root),
            "candidate_decision_rows": _descriptor(candidate_path, repo_root),
            "baseline_decision_rows": _descriptor(baseline_path, repo_root),
            "capture_manifest": _descriptor(
                Path(capture["capture_manifest_path"]), repo_root
            ),
            "post_boundary_quarantine": _descriptor(
                Path(quarantine_frozen["path"]), repo_root
            ),
        },
        "population_reconciliation_passed": (
            len(candidate_rows) == len(baseline_rows) == TARGET_MARKET_COUNT
            and [row["market_id"] for row in candidate_rows]
            == [row["market_id"] for row in baseline_rows]
            == selection["selected_market_ids"]
        ),
        "collector_stopped": True,
        "collection_restart_allowed": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "outcome_access_authorized_by_this_step": False,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    if report["population_reconciliation_passed"] is not True:
        raise ValueError("candidate and baseline populations did not reconcile")
    report_frozen = _write_new_frozen_json(
        target / "collection_finalization_report.json",
        report,
    )
    return {
        **report,
        "finalization_report_path": report_frozen["path"],
        "finalization_report_sha256": report_frozen["sha256"],
        "capture_manifest_sha256": capture["capture_manifest_sha256"],
    }


def _freeze_observation(
    capture: Mapping[str, Any],
    *,
    service_root: Path,
    repository_root: Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = Path(str(capture["run_dir"])).resolve()
    if not run_dir.is_relative_to(service_root):
        raise ValueError("capture directory escaped service root")
    report_path = run_dir / "pending_round_capture_report.json"
    manifest_path = run_dir / "pending_round_capture_manifest.json"
    report = _json_object(report_path)
    manifest = _json_object(manifest_path)
    if (
        report.get("resolution_provider_called") is not False
        or manifest.get("resolution_provider_called") is not False
    ):
        raise ValueError("capture attempted outcome or settlement access")
    try:
        feature_rows = _current_feature_rows(run_dir=run_dir, manifest=manifest)
    except ValueError as error:
        if str(error) not in EXPECTED_EMPTY_FEATURE_ERRORS:
            raise
        feature_rows = []
    quality = _quality_observations(
        capture=capture,
        report=report,
        feature_rows=feature_rows,
    )
    decisions: list[dict[str, Any]] = []
    market_id = _market_id(run_dir, feature_rows, capture)
    if quality["quality_valid"] is True:
        decisions, market = _predict_market(feature_rows=feature_rows, bundle=bundle)
        market_id = str(market["market_id"])
    causality = _causality(feature_rows, capture)
    direct = sorted(
        f"capture_reject_{reason}"
        for reason, count in dict(capture.get("reject_reason_counts") or {}).items()
        if int(count or 0) > 0
    )
    observations = dict(quality["quality_observations"])
    if direct and all(observations.values()):
        observations["provider_capture_complete"] = False
    derived = sorted(
        f"{name}_failed" for name, passed in observations.items() if passed is not True
    )
    raw_record: dict[str, Any] = {
        "schema_version": "bigan-btc-15m-raw-evidence-record-r2",
        "attempt_id": str(capture["run_id"]),
        "market_id": market_id,
        "raw_streams": {
            name: _descriptor(run_dir / "provider_raw" / name, repository_root)
            for name in REQUIRED_RAW_STREAMS
        },
        "paired_executable_ask_capture_attempted": True,
        "missing_values_encoded_as_numeric_zero": False,
        "complement_quote_proxy_used": False,
        "causality": causality,
        "quality_observations": observations,
        "direct_failure_reason_codes": direct,
        "derived_failure_reason_codes": derived,
    }
    raw_record["record_sha256"] = canonical_json_sha256(raw_record)
    return {
        "attempt_index": int(capture["round_index"]),
        "attempt_id": str(capture["run_id"]),
        "market_id": market_id,
        "market_start_ts": int(capture["scheduled_round_start_ts"]),
        "quality_valid": all(observations.values()),
        "decision_rows": decisions,
        "raw_evidence": raw_record,
        "capture_manifest_path": manifest_path,
        "capture_report_path": report_path,
    }


def _candidate_decision_row(observation: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(observation["decision_rows"])
    accepted = next((row for row in rows if row["accepted"] is True), None)
    action = accepted or rows[-1]
    return {
        "schema_version": "bigan-btc-15m-frozen-candidate-decision-v1",
        "lineage_id": LINEAGE_ID,
        "model_role": "candidate",
        "candidate_bundle_hash": CANDIDATE_BUNDLE_HASH,
        "market_id": observation["market_id"],
        "attempt_index": observation["attempt_index"],
        "market_start_ts": observation["market_start_ts"],
        "decision_ts": action["decision_ts"],
        "decision": "TRADE" if accepted else "NO_TRADE",
        "accepted": accepted is not None,
        "selected_side": action["selected_side"] if accepted else None,
        "requested_route": action["requested_route"],
        "actual_model_used": action["actual_model_used"],
        "expert_id": action["expert_id"],
        "expert_training_support": action["expert_training_support"],
        "expert_available": action["expert_available"],
        "fallback_used": action["fallback_used"],
        "decision_trace": [
            {
                key: row[key]
                for key in (
                    "decision_ts",
                    "requested_route",
                    "actual_model_used",
                    "expert_id",
                    "expert_training_support",
                    "expert_available",
                    "fallback_used",
                    "selected_side",
                    "accepted",
                )
            }
            for row in rows
        ],
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
    }


def _baseline_decision_row(
    observation: Mapping[str, Any],
    authorization: AuthorizationExpectation,
) -> dict[str, Any]:
    rows = list(observation["decision_rows"])
    accepted = next((row for row in rows if row["baseline_accepted"] is True), None)
    action = accepted or rows[-1]
    baseline_sha = authorization.expected_frozen_artifact_sha256[
        "matched_baseline_artifact"
    ]
    return {
        "schema_version": "bigan-btc-15m-frozen-baseline-decision-v1",
        "lineage_id": LINEAGE_ID,
        "model_role": "matched_global_baseline",
        "baseline_artifact_sha256": baseline_sha,
        "market_id": observation["market_id"],
        "attempt_index": observation["attempt_index"],
        "market_start_ts": observation["market_start_ts"],
        "decision_ts": action["decision_ts"],
        "decision": "TRADE" if accepted else "NO_TRADE",
        "accepted": accepted is not None,
        "selected_side": action["baseline_selected_side"] if accepted else None,
        "decision_trace": [
            {
                "decision_ts": row["decision_ts"],
                "selected_side": row["baseline_selected_side"],
                "accepted": row["baseline_accepted"],
            }
            for row in rows
        ],
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
    }


def _quarantine_payload(
    *,
    service_root: Path,
    observations: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    completed_paths = {Path(str(row["capture_manifest_path"])).parent for row in observations}
    all_round_dirs = sorted(
        path.resolve()
        for path in (service_root / "captures").glob("*-round*-*")
        if path.is_dir()
    )
    incomplete = [path for path in all_round_dirs if path not in completed_paths]
    audit_indices = list(selection["audit_only_post_boundary_attempt_indices"])
    by_index = {int(row["attempt_index"]): row for row in observations}
    return {
        "schema_version": "bigan-btc-15m-post-boundary-quarantine-v1",
        "lineage_id": LINEAGE_ID,
        "role": "preserved_audit_only_never_evaluation_population",
        "window_stop_attempt_index": selection["window_stop_attempt_index"],
        "audit_only_post_boundary_attempt_indices": audit_indices,
        "audit_only_complete_captures": [
            {
                "attempt_index": index,
                "attempt_id": by_index[index]["attempt_id"],
                "market_id": by_index[index]["market_id"],
                "capture_manifest": _descriptor(
                    Path(by_index[index]["capture_manifest_path"]), repository_root
                ),
                "capture_report": _descriptor(
                    Path(by_index[index]["capture_report_path"]), repository_root
                ),
            }
            for index in audit_indices
        ],
        "incomplete_capture_directories": [
            path.relative_to(repository_root).as_posix() for path in incomplete
        ],
        "incomplete_capture_reason": "not_in_frozen_completed_batch_ledger",
        "raw_captures_deleted": False,
        "evaluation_population_membership": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def _manual_authorization_expectation(repository_root: Path) -> AuthorizationExpectation:
    path = (
        repository_root
        / "examples/v8/polymarket_configs"
        / LINEAGE_ID
        / "manual_collection_authorization_001.json"
    )
    payload = _json_object(path)
    return AuthorizationExpectation(
        artifact_path=path,
        expected_artifact_sha256=sha256_file(path),
        authorization_request_text_sha256=payload[
            "authorization_request_text_sha256"
        ],
        authorization_decision_text_sha256=payload[
            "authorization_decision_text_sha256"
        ],
        approver_identity=payload["approver_identity"],
        authorization_source_id=payload["authorization_source_id"],
        authorization_source_url=payload["authorization_source_url"],
        authorization_timestamp=payload["authorization_timestamp"],
        strictly_later_than_timestamp=payload["strictly_later_than_timestamp"],
        expected_frozen_artifact_sha256={
            key: descriptor["sha256"]
            for key, descriptor in frozen_authorization_artifacts(
                repository_root
            ).items()
        },
    )


def _causality(
    feature_rows: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
) -> dict[str, int]:
    if feature_rows:
        return {
            field: max(int(row[field]) for row in feature_rows)
            for field in (
                "available_at_ts",
                "decision_ts",
                "max_input_ts",
                "feature_cutoff_ts",
            )
        }
    available = int(
        capture.get("capture_thread_started_at_ts")
        or capture["scheduled_round_start_ts"]
    )
    decision = int(capture["scheduled_round_start_ts"]) + 600_000
    return {
        "available_at_ts": min(available, decision),
        "decision_ts": decision,
        "max_input_ts": min(available, decision),
        "feature_cutoff_ts": min(available, decision),
    }


def _market_id(
    run_dir: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
) -> str:
    if feature_rows:
        return str(feature_rows[0]["market_id"])
    market_path = run_dir / "provider_raw" / "raw_polymarket_markets.jsonl"
    for line in market_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            value = str(row.get("market_id") or row.get("condition_id") or "")
            if value:
                return value
    return (
        f"unresolved-capture-attempt-{int(capture['round_index']):04d}-"
        f"{int(capture['scheduled_round_start_ts'])}"
    )


def _descriptor(path: Path, repository_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError("artifact descriptor escaped repository")
    return {
        "path": resolved.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload
