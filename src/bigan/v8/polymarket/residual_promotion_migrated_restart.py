"""Fail-closed authorization for one migrated promotion-v1 collector restart."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import (
    assert_outcome_blind,
    verify_attempt_chain,
)
from bigan.v8.polymarket.residual_promotion_service_root_migration import (
    verify_service_root_migration,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
)

SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-external-service-root-restart-authorization-v1"
)
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_migrated_restart.py"
)


def validate_migrated_restart_authorization(
    *,
    authorization_path: Path | str,
    repository_root: Path | str,
    service_root: Path | str,
) -> dict[str, Any]:
    """Validate one exact migration report and additive restart authorization."""

    root = Path(repository_root).resolve()
    authorization_file = _repo_file(authorization_path, root)
    sidecar = authorization_file.with_suffix(authorization_file.suffix + ".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip()
        != sha256_file(authorization_file)
    ):
        raise ValueError("migrated restart authorization hash mismatch")
    authorization = _load_json(authorization_file)
    assert_outcome_blind(authorization)
    destination = Path(service_root).resolve()
    false_fields = (
        "model_bytes_changed",
        "router_features_cost_action_baseline_or_population_changed",
        "collector_decisions_changed",
        "fresh_outcomes_opened",
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
        "paper_candidate_allowed",
        "v8_execution_handoff_allowed",
        "live_trading_allowed",
        "wallet_signing_allowed",
        "polymarket_write_allowed",
        "capital_at_risk",
    )
    if (
        authorization.get("schema_version") != SCHEMA_VERSION
        or authorization.get("lineage_id") != LINEAGE_ID
        or authorization.get("candidate_id") != CANDIDATE_ID
        or authorization.get("authorization_scope")
        != "one_byte_verified_external_service_root_restart_only"
        or authorization.get("collector_restart_authorized") is not True
        or authorization.get("collector_restart_completed") is not False
        or authorization.get("source_capture_deletion_authorized") is not False
        or authorization.get("source_service_root_mutation_authorized") is not False
        or any(authorization.get(field) is not False for field in false_fields)
        or int(authorization.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
        or int(authorization.get("target_quality_valid_market_count", -1))
        != TARGET_MARKETS
        or dict(authorization.get("safety") or {}) != SAFETY
        or destination != Path(str(authorization.get("destination_root") or "")).resolve()
    ):
        raise ValueError("migrated restart authorization governance mismatch")
    implementation = dict(authorization.get("implementation") or {})
    if (
        implementation.get("path") != IMPLEMENTATION_REPOSITORY_PATH
        or implementation.get("sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("migrated restart authorization implementation drift")
    for name in (
        "original_manual_collection_authorization",
        "original_collector_protocol",
        "candidate_bundle_manifest",
        "original_collector_cli",
        "restart_cli",
    ):
        descriptor = dict(authorization.get(name) or {})
        bound = _repo_file(str(descriptor.get("path") or ""), root)
        if descriptor.get("sha256") != sha256_file(bound):
            raise ValueError(f"migrated restart repository binding drift: {name}")
    report_descriptor = dict(authorization.get("migration_report") or {})
    report_file = Path(str(report_descriptor.get("path") or "")).resolve()
    if (
        not report_file.is_file()
        or report_descriptor.get("sha256") != sha256_file(report_file)
    ):
        raise ValueError("migrated restart report binding drift")
    report = verify_service_root_migration(report_path=report_file)
    progress = _load_json(destination / "collection_progress.json")
    attempts = _load_jsonl(destination / "outcome_blind_attempts.jsonl")
    verify_attempt_chain(attempts)
    for attempt in attempts:
        assert_outcome_blind(attempt)
    assert_outcome_blind(progress)
    if not (
        report["destination_root"] == str(destination)
        and report["report_sha256"] == report_descriptor["sha256"]
        and report["source_snapshot"]["tree_sha256"]
        == authorization.get("service_root_tree_sha256")
        and report["attempt_count"]
        == int(authorization.get("ledger_closed_attempt_count", -1))
        == len(attempts)
        == int(progress.get("attempts_consumed", -1))
        and report["quality_valid_market_count"]
        == int(authorization.get("quality_valid_market_count", -1))
        == int(progress.get("quality_valid_market_count", -1))
        and progress.get("candidate_bundle_sha256")
        == dict(authorization["candidate_bundle_manifest"])["sha256"]
        and progress.get("authorization_sha256")
        == dict(authorization["original_manual_collection_authorization"])["sha256"]
        and progress.get("collector_protocol_sha256")
        == dict(authorization["original_collector_protocol"])["sha256"]
        and progress.get("fresh_outcomes_opened") is False
        and progress.get("interim_pnl_evaluated") is False
        and dict(progress.get("safety") or {}) == SAFETY
    ):
        raise ValueError("migrated restart service-root reconciliation mismatch")
    return {
        "authorization": authorization,
        "authorization_sha256": sha256_file(authorization_file),
        "migration_report": report,
        "attempts_consumed": len(attempts),
        "validation_passed": True,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }


def verify_existing_coverage_resume_record(
    service_root: Path | str,
    *,
    original_validation: dict[str, Any],
    resumed_attempt_count: int,
    rest_fallback_collection_seconds: float,
) -> None:
    """Verify the immutable v3 resume record without rewriting its original bytes."""

    root = Path(service_root).resolve()
    resume = _load_json(root / "collection_resume_record_v3.json")
    start = _load_json(root / "collection_start_record.json")
    assert_outcome_blind(resume)
    assert_outcome_blind(start)
    if not (
        resume.get("schema_version")
        == "bigan-btc-15m-residual-promotion-resume-v3"
        and resume.get("resumed_attempt_count") == 1
        and resume.get("prior_attempts_preserved") is True
        and resume.get("coverage_instrumentation_corrected") is True
        and resume.get("rest_fallback_collection_seconds")
        == rest_fallback_collection_seconds
        and resume.get("authorization_sha256")
        == original_validation["authorization_sha256"]
        and resume.get("collector_protocol_sha256")
        == original_validation["collector_protocol_sha256"]
        and resume.get("candidate_bundle_sha256")
        == original_validation["bundle"]["sha256"]
        and resume.get("fresh_outcomes_opened") is False
        and resume.get("zero_capital_read_only") is True
        and resume.get("wallet_signing_allowed") is False
        and resume.get("polymarket_write_allowed") is False
        and resume.get("capital_at_risk") is False
        and dict(resume.get("safety") or {}) == SAFETY
        and start.get("fresh_outcomes_opened") is False
        and dict(start.get("safety") or {}) == SAFETY
        and resumed_attempt_count >= int(resume["resumed_attempt_count"])
    ):
        raise ValueError("migrated restart immutable resume record mismatch")


def _repo_file(path: Path | str, root: Path) -> Path:
    candidate = Path(path)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("migrated restart binding escaped repository")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("attempt ledger contains a non-object row")
    return values


__all__ = [
    "validate_migrated_restart_authorization",
    "verify_existing_coverage_resume_record",
]
