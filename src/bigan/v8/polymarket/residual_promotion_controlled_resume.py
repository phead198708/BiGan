"""Fail-closed authorization for one additive promotion collector resume."""

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
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
)

SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-controlled-resume-authorization-v1"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_controlled_resume.py"
)
ORIGINAL_COLLECTOR_REPOSITORY_PATH = (
    "examples/v8/run_residual_promotion_v1_collector.py"
)
RESUME_CLI_REPOSITORY_PATH = (
    "examples/v8/run_residual_promotion_v1_controlled_resume.py"
)


def validate_controlled_resume_authorization(
    *,
    authorization_path: Path | str,
    repository_root: Path | str,
    service_root: Path | str,
) -> dict[str, Any]:
    """Validate one exact ledger/progress boundary for a collector resume."""

    root = Path(repository_root).resolve()
    authorization_file = _repo_file(authorization_path, root)
    sidecar = authorization_file.with_suffix(authorization_file.suffix + ".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip()
        != sha256_file(authorization_file)
    ):
        raise ValueError("controlled resume authorization hash mismatch")
    authorization = _load_json(authorization_file)
    assert_outcome_blind(authorization)
    service = Path(service_root).resolve()
    false_fields = (
        "model_bytes_changed",
        "gates_thresholds_costs_baseline_population_changed",
        "collector_decisions_changed",
        "fresh_outcomes_opened",
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
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
        != "one_exact_outcome_blind_collector_resume"
        or authorization.get("collector_resume_authorized") is not True
        or authorization.get("authorization_consumed_by_ledger_advance") is not True
        or authorization.get("zero_capital_read_only") is not True
        or any(authorization.get(field) is not False for field in false_fields)
        or int(authorization.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
        or int(authorization.get("target_quality_valid_market_count", -1))
        != TARGET_MARKETS
        or dict(authorization.get("safety") or {}) != SAFETY
        or service != Path(str(authorization.get("service_root") or "")).resolve()
    ):
        raise ValueError("controlled resume governance mismatch")
    for name, expected_path in (
        ("implementation", IMPLEMENTATION_REPOSITORY_PATH),
        ("original_collector_cli", ORIGINAL_COLLECTOR_REPOSITORY_PATH),
        ("resume_cli", RESUME_CLI_REPOSITORY_PATH),
    ):
        descriptor = dict(authorization.get(name) or {})
        bound = _repo_file(str(descriptor.get("path") or ""), root)
        if descriptor.get("path") != expected_path or descriptor.get(
            "sha256"
        ) != sha256_file(bound):
            raise ValueError(f"controlled resume repository binding drift: {name}")

    ledger_path = service / "outcome_blind_attempts.jsonl"
    progress_path = service / "collection_progress.json"
    boundary = dict(authorization.get("exact_resume_boundary") or {})
    if (
        boundary.get("ledger_sha256") != sha256_file(ledger_path)
        or boundary.get("progress_sha256") != sha256_file(progress_path)
    ):
        raise ValueError("controlled resume boundary bytes changed")
    attempts = _load_jsonl(ledger_path)
    progress = _load_json(progress_path)
    verify_attempt_chain(attempts)
    for attempt in attempts:
        assert_outcome_blind(attempt)
    assert_outcome_blind(progress)
    if not (
        len(attempts)
        == int(boundary.get("attempts_consumed", -1))
        == int(progress.get("attempts_consumed", -1))
        and int(boundary.get("quality_valid_market_count", -1))
        == int(progress.get("quality_valid_market_count", -1))
        and progress.get("authorization_sha256")
        == authorization.get("frozen_collection_authorization_sha256")
        and progress.get("collector_protocol_sha256")
        == authorization.get("frozen_collector_protocol_sha256")
        and progress.get("candidate_bundle_sha256")
        == authorization.get("frozen_candidate_bundle_sha256")
        and progress.get("collection_complete") is False
        and progress.get("attempt_cap_exhausted") is False
        and progress.get("fresh_outcomes_opened") is False
        and progress.get("interim_pnl_evaluated") is False
        and dict(progress.get("safety") or {}) == SAFETY
    ):
        raise ValueError("controlled resume service-root reconciliation mismatch")
    return {
        "authorization": authorization,
        "authorization_sha256": sha256_file(authorization_file),
        "attempts_consumed": len(attempts),
        "quality_valid_market_count": int(progress["quality_valid_market_count"]),
        "validation_passed": True,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }


def _repo_file(path: Path | str, root: Path) -> Path:
    candidate = Path(path)
    resolved = (
        (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    )
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("controlled resume binding escaped repository")
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


__all__ = ["validate_controlled_resume_authorization"]
