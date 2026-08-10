"""Outcome-blind exact-population freeze for residual promotion v1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_collection_observability import _current_feature_rows
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import (
    assert_outcome_blind,
    validate_collection_authorization,
    verify_attempt_chain,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
)

SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-exact-population-freeze-v1"
FORBIDDEN_CAPTURE_NAME_TOKENS = ("settlement", "realized_pnl", "outcome")
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_finalization.py"
)
FEATURE_RECONSTRUCTION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/moe_collection_observability.py"
)


def freeze_exact_outcome_blind_population(
    *,
    service_root: Path | str,
    repository_root: Path | str,
    authorization_path: Path | str,
    collector_protocol_path: Path | str,
    output_dir: Path | str | None = None,
    created_at: str | None = None,
    target_market_count: int = TARGET_MARKETS,
    validation_fixture_only: bool = False,
) -> dict[str, Any]:
    """Freeze the exact first-N valid population without touching outcomes."""

    if target_market_count <= 0:
        raise ValueError("target market count must be positive")
    if validation_fixture_only is not (target_market_count != TARGET_MARKETS):
        raise ValueError("non-production target requires validation_fixture_only")
    root = Path(service_root).resolve()
    repo = Path(repository_root).resolve()
    destination = Path(output_dir or root / "exact_population_freeze").resolve()
    if destination.exists():
        raise FileExistsError("exact population freeze already exists")
    validate_collection_authorization(
        authorization_path=authorization_path,
        collector_protocol_path=collector_protocol_path,
        repository_root=repo,
    )
    authorization_file = _repo_file(authorization_path, repo)
    collector_protocol_file = _repo_file(collector_protocol_path, repo)
    authorization = _load_json(authorization_file)
    statistical_protocol = _verified_bound_descriptor(
        authorization.get("statistical_protocol"), repository_root=repo
    )
    statistical_payload = _load_json(repo / statistical_protocol["path"])
    candidate_bundle = _verified_bound_descriptor(
        authorization.get("candidate_bundle"), repository_root=repo
    )
    reporting_contract = _verified_bound_descriptor(
        authorization.get("reporting_contract"), repository_root=repo
    )
    if _bound_pair(statistical_payload.get("candidate_bundle")) != candidate_bundle:
        raise ValueError("statistical protocol candidate bundle binding mismatch")
    if _bound_pair(statistical_payload.get("reporting_contract")) != reporting_contract:
        raise ValueError("statistical protocol reporting contract binding mismatch")
    evaluation_bindings = {
        name: _verified_bound_descriptor(
            statistical_payload.get(name), repository_root=repo
        )
        for name in (
            "baseline_artifact",
            "cost_contract",
            "feature_contract",
            "gate_implementation",
            "runtime_implementation",
            "runtime_parity_report",
        )
    }
    finalization_implementation = _repo_descriptor(
        repo / IMPLEMENTATION_REPOSITORY_PATH, repository_root=repo
    )
    feature_reconstruction_implementation = _repo_descriptor(
        repo / FEATURE_RECONSTRUCTION_REPOSITORY_PATH,
        repository_root=repo,
    )
    prospective_boundary_utc = str(authorization.get("created_at") or "")
    prospective_boundary_ts = _iso_to_epoch_ms(prospective_boundary_utc)
    ledger_path = root / "outcome_blind_attempts.jsonl"
    attempts = _load_jsonl(ledger_path)
    verify_attempt_chain(attempts)
    for attempt in attempts:
        assert_outcome_blind(attempt)
    if not attempts:
        raise ValueError("attempt ledger is empty")
    if len(attempts) > MAXIMUM_ATTEMPTS:
        raise ValueError("attempt cap exceeded")
    selection = select_exact_population(
        attempts,
        target_market_count=target_market_count,
    )
    if selection["population_complete"] is not True:
        raise ValueError("exact quality-valid population is incomplete")
    if selection["stop_attempt_index"] != len(attempts):
        raise ValueError("attempts exist after the exact population boundary")
    if any(
        int(attempt["scheduled_round_start_ts"]) <= prospective_boundary_ts
        for attempt in selection["selected_attempts"]
    ):
        raise ValueError("selected population is not strictly prospective")
    if not validation_fixture_only:
        _validate_progress(root / "collection_progress.json", attempts=attempts)

    population_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    capture_index: list[dict[str, Any]] = []
    for position, attempt in enumerate(selection["selected_attempts"], start=1):
        _validate_quality_contract(
            attempt, validation_fixture_only=validation_fixture_only
        )
        capture, feature_rows = _validate_capture(
            root=root,
            attempt=attempt,
            validation_fixture_only=validation_fixture_only,
        )
        market_id = str(attempt["market_id"])
        decisions = list(attempt.get("decision_rows") or [])
        if not decisions:
            raise ValueError("quality-valid attempt has no decision rows")
        _validate_decision_rows(
            decisions,
            market_id=market_id,
            expected_candidate_bundle_sha256=candidate_bundle["sha256"],
        )
        population_rows.append(
            {
                "population_position": position,
                "attempt_index": int(attempt["attempt_index"]),
                "attempt_id": str(attempt["attempt_id"]),
                "attempt_hash": str(attempt["attempt_hash"]),
                "market_id": market_id,
                "scheduled_round_start_ts": int(
                    attempt["scheduled_round_start_ts"]
                ),
                "quality_valid": True,
                "quality_record_sha256": canonical_json_sha256(
                    dict(attempt["quality"])
                ),
                "candidate_accepted": any(
                    row["candidate_accepted_at_this_decision"] is True
                    for row in decisions
                ),
                "baseline_accepted": any(
                    row["baseline_accepted_at_this_decision"] is True
                    for row in decisions
                ),
                "outcomes_accessed": False,
                "settlement_accessed": False,
                "pnl_accessed": False,
                "safety": dict(SAFETY),
            }
        )
        capture_index.append(
            {
                "population_position": position,
                "attempt_index": int(attempt["attempt_index"]),
                "attempt_id": str(attempt["attempt_id"]),
                "market_id": market_id,
                **capture,
                "outcomes_accessed": False,
                "settlement_accessed": False,
                "pnl_accessed": False,
                "safety": dict(SAFETY),
            }
        )
        market_candidate, market_baseline = _market_level_decisions(
            decisions=decisions,
            population_position=position,
            attempt_index=int(attempt["attempt_index"]),
            market_id=market_id,
            candidate_bundle_sha256=candidate_bundle["sha256"],
            baseline_artifact_sha256=evaluation_bindings["baseline_artifact"][
                "sha256"
            ],
            feature_rows=feature_rows,
            validation_fixture_only=validation_fixture_only,
        )
        candidate_rows.append(market_candidate)
        baseline_rows.append(market_baseline)

    destination.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "population_rows": _write_jsonl(
            destination / "exact_population_rows.jsonl", population_rows
        ),
        "candidate_decision_rows": _write_jsonl(
            destination / "candidate_decision_rows.jsonl", candidate_rows
        ),
        "baseline_decision_rows": _write_jsonl(
            destination / "baseline_decision_rows.jsonl", baseline_rows
        ),
        "raw_capture_index": _write_jsonl(
            destination / "raw_capture_index.jsonl", capture_index
        ),
        "attempt_ledger": {
            "path": ledger_path.relative_to(root).as_posix(),
            "sha256": sha256_file(ledger_path),
        },
    }
    ordered_market_ids = [row["market_id"] for row in population_rows]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "validation_fixture_only": validation_fixture_only,
        "authorization": _repo_descriptor(
            authorization_file, repository_root=repo
        ),
        "collector_protocol": _repo_descriptor(
            collector_protocol_file, repository_root=repo
        ),
        "statistical_protocol": statistical_protocol,
        "reporting_contract": reporting_contract,
        "candidate_bundle": candidate_bundle,
        **evaluation_bindings,
        "finalization_implementation": finalization_implementation,
        "feature_reconstruction_implementation": (
            feature_reconstruction_implementation
        ),
        "target_quality_valid_market_count": target_market_count,
        "exact_market_count": len(population_rows),
        "attempts_consumed": len(attempts),
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "stop_attempt_index": selection["stop_attempt_index"],
        "prospective_boundary_utc": prospective_boundary_utc,
        "prospective_boundary_ts": prospective_boundary_ts,
        "first_market_scheduled_round_start_ts": population_rows[0][
            "scheduled_round_start_ts"
        ],
        "last_market_scheduled_round_start_ts": population_rows[-1][
            "scheduled_round_start_ts"
        ],
        "strictly_later_than_prospective_boundary": True,
        "ordered_market_ids_sha256": canonical_json_sha256(ordered_market_ids),
        "ordered_attempt_hashes_sha256": canonical_json_sha256(
            [row["attempt_hash"] for row in population_rows]
        ),
        "candidate_decision_row_count": len(candidate_rows),
        "baseline_decision_row_count": len(baseline_rows),
        "candidate_and_baseline_population_aligned": True,
        "hash_chain_status": "valid",
        "source_capture_mutated": False,
        "artifacts": artifacts,
        "population_frozen": True,
        "outcome_access_authorized": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "promotion_evidence_eligible": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    manifest_descriptor = _write_json(
        destination / "exact_population_manifest.json", manifest
    )
    validation_report = validate_frozen_population(
        freeze_dir=destination,
        service_root=root,
        repository_root=repo,
        expected_manifest_sha256=manifest_descriptor["sha256"],
        target_market_count=target_market_count,
        validation_fixture_only=validation_fixture_only,
    )
    return {
        "manifest": manifest_descriptor,
        "population_validation": validation_report,
        "outcome_access_authorized": False,
        "outcomes_accessed": False,
        "safety": dict(SAFETY),
    }


def select_exact_population(
    attempts: Sequence[Mapping[str, Any]], *, target_market_count: int
) -> dict[str, Any]:
    """Select the chronological first valid unique markets without reordering."""

    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    stop_index: int | None = None
    for expected_index, source in enumerate(attempts, start=1):
        attempt = dict(source)
        if int(attempt.get("attempt_index") or 0) != expected_index:
            raise ValueError("attempt indices are not contiguous")
        quality = dict(attempt.get("quality") or {})
        if quality.get("quality_valid") is not True:
            continue
        market_id = str(attempt.get("market_id") or "")
        if not market_id or market_id in seen:
            continue
        seen.add(market_id)
        if stop_index is None:
            selected.append(attempt)
            if len(selected) == target_market_count:
                stop_index = expected_index
    return {
        "population_complete": stop_index is not None,
        "selected_attempts": selected if stop_index is not None else [],
        "quality_valid_unique_market_count": len(seen),
        "stop_attempt_index": stop_index,
        "post_boundary_attempt_count": (
            len(attempts) - stop_index if stop_index is not None else 0
        ),
    }


def validate_frozen_population(
    *,
    freeze_dir: Path | str,
    service_root: Path | str,
    repository_root: Path | str,
    expected_manifest_sha256: str,
    target_market_count: int = TARGET_MARKETS,
    validation_fixture_only: bool = False,
) -> dict[str, Any]:
    """Independently verify every frozen population output and safety flag."""

    root = Path(service_root).resolve()
    repo = Path(repository_root).resolve()
    directory = Path(freeze_dir).resolve()
    expected_filenames = {
        "exact_population_manifest.json",
        "exact_population_manifest.json.sha256",
        "exact_population_rows.jsonl",
        "exact_population_rows.jsonl.sha256",
        "candidate_decision_rows.jsonl",
        "candidate_decision_rows.jsonl.sha256",
        "baseline_decision_rows.jsonl",
        "baseline_decision_rows.jsonl.sha256",
        "raw_capture_index.jsonl",
        "raw_capture_index.jsonl.sha256",
    }
    actual_filenames = {
        path.name for path in directory.iterdir() if path.is_file()
    }
    if actual_filenames != expected_filenames:
        raise ValueError("exact population freeze directory file set mismatch")
    manifest_path = directory / "exact_population_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("exact population manifest SHA-256 mismatch")
    _verify_sidecar(manifest_path)
    manifest = _load_json(manifest_path)
    if not (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("lineage_id") == LINEAGE_ID
        and manifest.get("candidate_id") == CANDIDATE_ID
        and manifest.get("validation_fixture_only") is validation_fixture_only
        and manifest.get("target_quality_valid_market_count")
        == target_market_count
        and manifest.get("exact_market_count") == target_market_count
        and manifest.get("population_frozen") is True
        and manifest.get("strictly_later_than_prospective_boundary") is True
        and manifest.get("outcome_access_authorized") is False
        and manifest.get("outcomes_accessed") is False
        and manifest.get("settlement_accessed") is False
        and manifest.get("pnl_accessed") is False
        and manifest.get("promotion_evidence_eligible") is False
        and dict(manifest.get("safety") or {}) == SAFETY
    ):
        raise ValueError("exact population manifest governance mismatch")
    for name in (
        "authorization",
        "collector_protocol",
        "statistical_protocol",
        "reporting_contract",
        "candidate_bundle",
        "baseline_artifact",
        "cost_contract",
        "feature_contract",
        "gate_implementation",
        "runtime_implementation",
        "runtime_parity_report",
        "finalization_implementation",
        "feature_reconstruction_implementation",
    ):
        _verified_bound_descriptor(manifest.get(name), repository_root=repo)
    implementation = dict(manifest["finalization_implementation"])
    if implementation.get("path") != IMPLEMENTATION_REPOSITORY_PATH:
        raise ValueError("finalization implementation binding mismatch")
    reconstruction = dict(manifest["feature_reconstruction_implementation"])
    if reconstruction.get("path") != FEATURE_RECONSTRUCTION_REPOSITORY_PATH:
        raise ValueError("feature reconstruction implementation binding mismatch")
    artifacts = dict(manifest.get("artifacts") or {})
    expected_artifacts = {
        "population_rows",
        "candidate_decision_rows",
        "baseline_decision_rows",
        "raw_capture_index",
        "attempt_ledger",
    }
    if set(artifacts) != expected_artifacts:
        raise ValueError("exact population artifact graph mismatch")
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name in expected_artifacts - {"attempt_ledger"}:
        descriptor = dict(artifacts[name])
        path = (directory / str(descriptor["path"])).resolve()
        if not path.is_relative_to(directory) or sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"frozen population artifact drift: {name}")
        _verify_sidecar(path)
        rows_by_name[name] = _load_jsonl(path)
    ledger_descriptor = dict(artifacts["attempt_ledger"])
    ledger_path = (root / str(ledger_descriptor["path"])).resolve()
    if not ledger_path.is_relative_to(root) or sha256_file(ledger_path) != ledger_descriptor["sha256"]:
        raise ValueError("attempt ledger drifted after population freeze")
    population = rows_by_name["population_rows"]
    candidate = rows_by_name["candidate_decision_rows"]
    baseline = rows_by_name["baseline_decision_rows"]
    captures = rows_by_name["raw_capture_index"]
    if len(population) != target_market_count or len(captures) != target_market_count:
        raise ValueError("frozen population count mismatch")
    market_ids = [str(row["market_id"]) for row in population]
    if len(set(market_ids)) != target_market_count:
        raise ValueError("frozen population market identity is not unique")
    if canonical_json_sha256(market_ids) != manifest["ordered_market_ids_sha256"]:
        raise ValueError("frozen population order hash mismatch")
    if len(candidate) != target_market_count or len(baseline) != target_market_count:
        raise ValueError("candidate and baseline market-level counts differ")
    candidate_keys = [
        (row["population_position"], row["market_id"])
        for row in candidate
    ]
    baseline_keys = [
        (row["population_position"], row["market_id"])
        for row in baseline
    ]
    if candidate_keys != baseline_keys:
        raise ValueError("candidate and baseline decision populations differ")
    for capture in captures:
        _revalidate_capture_index(root=root, capture=capture)
    for rows in rows_by_name.values():
        for row in rows:
            assert_outcome_blind(row)
    return {
        "validation_passed": True,
        "exact_market_count": target_market_count,
        "candidate_decision_row_count": len(candidate),
        "baseline_decision_row_count": len(baseline),
        "outcome_access_authorized": False,
        "outcomes_accessed": False,
        "safety": dict(SAFETY),
    }


def _validate_progress(path: Path, *, attempts: Sequence[Mapping[str, Any]]) -> None:
    progress = _load_json(path)
    if not (
        progress.get("collection_complete") is True
        and progress.get("quality_valid_market_count") == TARGET_MARKETS
        and progress.get("target_quality_valid_market_count") == TARGET_MARKETS
        and progress.get("attempts_consumed") == len(attempts)
        and progress.get("fresh_outcomes_opened") is False
        and progress.get("interim_pnl_evaluated") is False
        and progress.get("hash_chain_status") == "valid"
        and dict(progress.get("safety") or {}) == SAFETY
    ):
        raise ValueError("collection progress is not exact-freeze ready")


def _validate_capture(
    *,
    root: Path,
    attempt: Mapping[str, Any],
    validation_fixture_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempt_id = str(attempt["attempt_id"])
    run_dir = (root / "captures" / attempt_id).resolve()
    if not run_dir.is_relative_to(root) or not run_dir.is_dir():
        raise ValueError("capture directory is missing")
    manifest_path = run_dir / "pending_round_capture_manifest.json"
    report_path = run_dir / "pending_round_capture_report.json"
    if sha256_file(manifest_path) != attempt["capture_manifest_sha256"]:
        raise ValueError("capture manifest hash mismatch")
    if sha256_file(report_path) != attempt["capture_report_sha256"]:
        raise ValueError("capture report hash mismatch")
    manifest = _load_json(manifest_path)
    report = _load_json(report_path)
    if manifest.get("resolution_provider_called") is not False or report.get(
        "resolution_provider_called"
    ) is not False:
        raise ValueError("capture accessed resolution")
    file_graph = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(run_dir):
            raise ValueError("capture artifact escaped capture directory")
        lowered = path.name.lower()
        if any(token in lowered for token in FORBIDDEN_CAPTURE_NAME_TOKENS):
            raise ValueError("capture contains forbidden outcome-bearing artifact")
        if "resolution" in lowered and path.stat().st_size > 0:
            raise ValueError("capture contains non-empty resolution artifact")
        file_graph.append(
            {
                "path": resolved.relative_to(root).as_posix(),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    _verify_manifest_raw_bindings(
        run_dir=run_dir,
        manifest=manifest,
        validation_fixture_only=validation_fixture_only,
    )
    feature_rows = (
        []
        if validation_fixture_only
        else _current_feature_rows(run_dir=run_dir, manifest=manifest)
    )
    return {
        "capture_dir": run_dir.relative_to(root).as_posix(),
        "capture_manifest_path": manifest_path.relative_to(root).as_posix(),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "capture_report_path": report_path.relative_to(root).as_posix(),
        "capture_report_sha256": sha256_file(report_path),
        "file_count": len(file_graph),
        "files": file_graph,
    }, feature_rows


def _validate_decision_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    market_id: str,
    expected_candidate_bundle_sha256: str,
) -> None:
    previous_ts: int | None = None
    candidate_accepts = 0
    baseline_accepts = 0
    for row in rows:
        assert_outcome_blind(row)
        if str(row.get("market_id")) != market_id:
            raise ValueError("decision market identity mismatch")
        decision_ts = int(row["decision_ts"])
        if previous_ts is not None and decision_ts < previous_ts:
            raise ValueError("decision rows are not chronological")
        previous_ts = decision_ts
        if row.get("decision_influenced_collection") is not False:
            raise ValueError("model decision influenced collection")
        if row.get("candidate_bundle_sha256") != expected_candidate_bundle_sha256:
            raise ValueError("candidate decision bundle binding mismatch")
        candidate_accepts += int(row["candidate_accepted_at_this_decision"] is True)
        baseline_accepts += int(row["baseline_accepted_at_this_decision"] is True)
    if candidate_accepts > 1 or baseline_accepts > 1:
        raise ValueError("more than one trade was accepted per market")


def _validate_quality_contract(
    attempt: Mapping[str, Any], *, validation_fixture_only: bool
) -> None:
    quality = dict(attempt.get("quality") or {})
    if quality.get("quality_valid") is not True:
        raise ValueError("selected attempt is not quality-valid")
    if validation_fixture_only:
        return
    observations = dict(quality.get("quality_observations") or {})
    missing_feature_count = quality.get("missing_feature_count")
    missing_feature_counts = dict(quality.get("missing_feature_counts") or {})
    missing_counts_valid = bool(
        isinstance(missing_feature_count, int)
        and not isinstance(missing_feature_count, bool)
        and missing_feature_count >= 0
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in missing_feature_counts.values()
        )
        and sum(missing_feature_counts.values()) == missing_feature_count
        and (missing_feature_count > 0 or not missing_feature_counts)
    )
    if not observations or any(value is not True for value in observations.values()):
        raise ValueError("quality-valid attempt has an unsatisfied observation")
    if not (
        quality.get("invalid_reason_codes") == []
        and int(quality.get("observed_decision_count") or 0) > 0
        and int(quality.get("paired_executable_ask_decision_count") or 0) > 0
        and int(quality.get("btc_feature_complete_decision_count") or 0) > 0
        and int(quality.get("causality_violation_count") or 0) == 0
        and missing_counts_valid
        and quality.get("missing_values_encoded_as_zero") is False
    ):
        raise ValueError("quality-valid attempt is internally inconsistent")


def _market_level_decisions(
    *,
    decisions: Sequence[Mapping[str, Any]],
    population_position: int,
    attempt_index: int,
    market_id: str,
    candidate_bundle_sha256: str,
    baseline_artifact_sha256: str,
    feature_rows: Sequence[Mapping[str, Any]],
    validation_fixture_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_accepted = next(
        (row for row in decisions if row["candidate_accepted_at_this_decision"] is True),
        None,
    )
    baseline_accepted = next(
        (row for row in decisions if row["baseline_accepted_at_this_decision"] is True),
        None,
    )
    candidate_action = candidate_accepted or decisions[-1]
    baseline_action = baseline_accepted or decisions[-1]
    candidate_execution = _execution_features(
        feature_rows=feature_rows,
        market_id=market_id,
        decision_ts=int(candidate_action["decision_ts"]),
        validation_fixture_only=validation_fixture_only,
    )
    baseline_execution = _execution_features(
        feature_rows=feature_rows,
        market_id=market_id,
        decision_ts=int(baseline_action["decision_ts"]),
        validation_fixture_only=validation_fixture_only,
    )
    common = {
        "population_position": population_position,
        "attempt_index": attempt_index,
        "market_id": market_id,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "safety": dict(SAFETY),
    }
    candidate = {
        **common,
        "model_role": "candidate",
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "decision_ts": int(candidate_action["decision_ts"]),
        "action_values": dict(candidate_action["candidate_action_values"]),
        "selected_action": str(candidate_action["candidate_selected_action"]),
        "selected_side": _selected_side(
            str(candidate_action["candidate_selected_action"])
        ),
        "accepted": candidate_accepted is not None,
        "selected_action_value": _selected_action_value(
            candidate_action["candidate_action_values"],
            str(candidate_action["candidate_selected_action"]),
        ),
        "execution_features": candidate_execution,
        "execution_features_sha256": canonical_json_sha256(candidate_execution),
        "decision_trace_sha256": canonical_json_sha256(
            [
                {
                    "decision_ts": int(row["decision_ts"]),
                    "action_values": dict(row["candidate_action_values"]),
                    "selected_action": str(row["candidate_selected_action"]),
                    "accepted_at_this_decision": bool(
                        row["candidate_accepted_at_this_decision"]
                    ),
                }
                for row in decisions
            ]
        ),
    }
    baseline = {
        **common,
        "model_role": "matched_global_baseline",
        "baseline_artifact_sha256": baseline_artifact_sha256,
        "decision_ts": int(baseline_action["decision_ts"]),
        "action_values": dict(baseline_action["baseline_action_values"]),
        "selected_action": str(baseline_action["baseline_selected_action"]),
        "selected_side": _selected_side(
            str(baseline_action["baseline_selected_action"])
        ),
        "accepted": baseline_accepted is not None,
        "selected_action_value": _selected_action_value(
            baseline_action["baseline_action_values"],
            str(baseline_action["baseline_selected_action"]),
        ),
        "execution_features": baseline_execution,
        "execution_features_sha256": canonical_json_sha256(baseline_execution),
        "fail_closed": bool(baseline_action.get("baseline_fail_closed", False)),
        "fail_closed_reasons": list(
            baseline_action.get("baseline_fail_closed_reasons", [])
        ),
        "decision_trace_sha256": canonical_json_sha256(
            [
                {
                    "decision_ts": int(row["decision_ts"]),
                    "action_values": dict(row["baseline_action_values"]),
                    "selected_action": str(row["baseline_selected_action"]),
                    "accepted_at_this_decision": bool(
                        row["baseline_accepted_at_this_decision"]
                    ),
                    "fail_closed": bool(row.get("baseline_fail_closed", False)),
                    "fail_closed_reasons": list(
                        row.get("baseline_fail_closed_reasons", [])
                    ),
                }
                for row in decisions
            ]
        ),
    }
    return candidate, baseline


def _execution_features(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    market_id: str,
    decision_ts: int,
    validation_fixture_only: bool,
) -> dict[str, float]:
    if validation_fixture_only:
        source: Mapping[str, Any] = {
            "up_ask": 0.55,
            "up_bid": 0.53,
            "up_liquidity_depth": 10.0,
            "down_ask": 0.47,
            "down_bid": 0.45,
            "down_liquidity_depth": 10.0,
        }
    else:
        matches = [
            row
            for row in feature_rows
            if str(row.get("market_id")) == market_id
            and int(row.get("decision_ts") or 0) == decision_ts
        ]
        if len(matches) != 1:
            raise ValueError("decision execution feature row did not reconcile")
        source = dict(matches[0].get("features") or {})
    required = (
        "up_ask",
        "up_bid",
        "up_liquidity_depth",
        "down_ask",
        "down_bid",
        "down_liquidity_depth",
    )
    if any(name not in source for name in required):
        raise ValueError("decision execution feature envelope is incomplete")
    try:
        result = {name: float(source[name]) for name in required}
    except (TypeError, ValueError) as exc:
        raise ValueError("decision execution features are not numeric") from exc
    for side in ("up", "down"):
        ask = result[f"{side}_ask"]
        bid = result[f"{side}_bid"]
        depth = result[f"{side}_liquidity_depth"]
        if not (
            math.isfinite(ask)
            and math.isfinite(bid)
            and math.isfinite(depth)
            and 0.0 < bid <= ask < 1.0
            and depth >= 0.0
        ):
            raise ValueError("decision execution features are not executable")
    return result


def _selected_side(action: str) -> str | None:
    if action == "NO_TRADE":
        return None
    if action == "BUY_UP_HOLD":
        return "UP"
    if action == "BUY_DOWN_HOLD":
        return "DOWN"
    raise ValueError("frozen action is unknown")


def _selected_action_value(values: Any, action: str) -> float:
    action_values = dict(values or {})
    value = action_values.get(action)
    if value is None or not math.isfinite(float(value)):
        raise ValueError("selected action value is not finite")
    return float(value)


def _verify_manifest_raw_bindings(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    validation_fixture_only: bool,
) -> None:
    bindings = {
        "raw": dict(manifest.get("raw_artifact_hashes") or {}),
        "provider_raw": dict(manifest.get("provider_raw_artifact_hashes") or {}),
    }
    if not validation_fixture_only and not bindings["provider_raw"]:
        raise ValueError("quality-valid capture has no provider raw artifact graph")
    for directory_name, recorded in bindings.items():
        for filename, expected_sha256 in recorded.items():
            if Path(str(filename)).name != filename:
                raise ValueError("capture manifest raw artifact path is invalid")
            path = (run_dir / directory_name / str(filename)).resolve()
            if not path.is_relative_to(run_dir) or not path.is_file():
                raise ValueError("capture manifest raw artifact is missing")
            if sha256_file(path) != expected_sha256:
                raise ValueError("capture manifest raw artifact SHA-256 mismatch")
    chainlink_sha = manifest.get("provider_chainlink_raw_artifact_sha256")
    if chainlink_sha is not None:
        chainlink_path = (
            run_dir / "provider_raw/raw_polymarket_chainlink_prices.jsonl"
        )
        if sha256_file(chainlink_path) != chainlink_sha:
            raise ValueError("capture manifest Chainlink artifact SHA-256 mismatch")


def _revalidate_capture_index(*, root: Path, capture: Mapping[str, Any]) -> None:
    run_dir = (root / str(capture["capture_dir"])).resolve()
    if not run_dir.is_relative_to(root) or not run_dir.is_dir():
        raise ValueError("frozen capture directory is unavailable")
    descriptors = list(capture.get("files") or [])
    if int(capture.get("file_count") or -1) != len(descriptors):
        raise ValueError("frozen capture file graph count mismatch")
    expected_paths: set[str] = set()
    for value in descriptors:
        descriptor = dict(value)
        if set(descriptor) != {"path", "sha256", "size_bytes"}:
            raise ValueError("frozen capture file descriptor is invalid")
        path = (root / str(descriptor["path"])).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise ValueError("frozen capture source artifact is unavailable")
        relative = path.relative_to(root).as_posix()
        if relative in expected_paths:
            raise ValueError("frozen capture file graph contains a duplicate")
        expected_paths.add(relative)
        if (
            sha256_file(path) != descriptor["sha256"]
            or path.stat().st_size != descriptor["size_bytes"]
        ):
            raise ValueError("frozen capture source artifact drift")
    current_paths = set()
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(run_dir):
            raise ValueError("frozen capture source path escaped capture directory")
        current_paths.add(resolved.relative_to(root).as_posix())
    if current_paths != expected_paths:
        raise ValueError("frozen capture source artifact set drift")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _repo_file(path: Path | str, repository_root: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repository_root / resolved
    resolved = resolved.resolve()
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise ValueError("repository artifact path is invalid")
    return resolved


def _repo_descriptor(path: Path | str, *, repository_root: Path) -> dict[str, str]:
    resolved = _repo_file(path, repository_root)
    return {
        "path": resolved.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _verified_bound_descriptor(
    value: Any, *, repository_root: Path
) -> dict[str, str]:
    descriptor = dict(value or {})
    if not {"path", "sha256"}.issubset(descriptor):
        raise ValueError("repository artifact descriptor is incomplete")
    resolved = _repo_file(str(descriptor["path"]), repository_root)
    if sha256_file(resolved) != descriptor["sha256"]:
        raise ValueError("repository artifact SHA-256 mismatch")
    return {"path": str(descriptor["path"]), "sha256": str(descriptor["sha256"])}


def _bound_pair(value: Any) -> dict[str, str]:
    descriptor = dict(value or {})
    return {
        "path": str(descriptor.get("path") or ""),
        "sha256": str(descriptor.get("sha256") or ""),
    }


def _iso_to_epoch_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("prospective boundary timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("prospective boundary timestamp lacks timezone")
    return int(parsed.timestamp() * 1000)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL artifact is missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    raw = b"".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    path.write_bytes(raw)
    _write_sidecar(path, raw)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    _write_sidecar(path, raw)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _write_sidecar(path: Path, raw: bytes) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ValueError("frozen artifact sidecar mismatch")


__all__ = [
    "freeze_exact_outcome_blind_population",
    "select_exact_population",
    "validate_frozen_population",
]
