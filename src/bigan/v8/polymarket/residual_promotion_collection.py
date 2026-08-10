"""Outcome-blind, zero-capital collection ledger for residual promotion v1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_collection_observability import (
    _capture_retry_used,
    _current_feature_rows,
    _quality_observations,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
    ResidualPromotionRuntime,
    load_matched_baseline,
    load_residual_promotion_runtime,
    score_matched_baseline,
)

ATTEMPT_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-attempt-v1"
PROGRESS_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-progress-v1"
FORBIDDEN_FIELD_TOKENS = (
    "outcome",
    "settlement",
    "resolution",
    "realized_pnl",
    "unit_pnl",
    "target",
    "label",
)


def validate_collection_authorization(
    *,
    authorization_path: Path | str,
    collector_protocol_path: Path | str,
    repository_root: Path | str,
) -> dict[str, Any]:
    """Validate an exact capture-only authorization before network access."""

    root = Path(repository_root).resolve()
    authorization_file = _repo_file(authorization_path, root)
    collector_file = _repo_file(collector_protocol_path, root)
    authorization = _verified_json(authorization_file)
    collector = _verified_json(collector_file)
    if not (
        authorization.get("lineage_id") == LINEAGE_ID
        and authorization.get("fresh_collection_authorized") is True
        and authorization.get("fresh_collection_started") is False
        and authorization.get("fresh_outcomes_opened") is False
        and authorization.get("authorization_scope")
        == "zero_capital_read_only_outcome_blind_capture_only"
        and authorization.get("target_quality_valid_market_count") == TARGET_MARKETS
        and authorization.get("maximum_attempts") == MAXIMUM_ATTEMPTS
        and authorization.get("wallet_order_write_or_capital_authorized") is False
        and dict(authorization.get("safety") or {}) == SAFETY
    ):
        raise ValueError("manual collection authorization is invalid")
    descriptor = dict(authorization.get("collector_protocol") or {})
    if not (
        descriptor.get("path") == collector_file.relative_to(root).as_posix()
        and descriptor.get("sha256") == sha256_file(collector_file)
    ):
        raise ValueError("manual authorization collector binding mismatch")
    if not (
        collector.get("lineage_id") == LINEAGE_ID
        and collector.get("capture_only") is True
        and collector.get("resolution_provider_enabled") is False
        and collector.get("settlement_finalizer_enabled") is False
        and collector.get("training_export_enabled") is False
        and collector.get("outcome_fields_forbidden") is True
        and collector.get("collection_may_not_use_candidate_or_baseline_decisions")
        is True
        and collector.get("target_quality_valid_market_count") == TARGET_MARKETS
        and collector.get("maximum_attempts") == MAXIMUM_ATTEMPTS
        and dict(collector.get("safety") or {}) == SAFETY
    ):
        raise ValueError("collector protocol is invalid")
    bundle = dict(authorization["candidate_bundle"])
    runtime = load_residual_promotion_runtime(
        manifest_path=bundle["path"],
        expected_manifest_sha256=bundle["sha256"],
        repository_root=root,
    )
    load_matched_baseline(repository_root=root)
    return {
        "authorization_sha256": sha256_file(authorization_file),
        "collector_protocol_sha256": sha256_file(collector_file),
        "runtime": runtime,
        "bundle": bundle,
        "validation_passed": True,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }


def observe_outcome_blind_capture(
    capture: Mapping[str, Any],
    *,
    runtime: ResidualPromotionRuntime,
    baseline: Any,
) -> dict[str, Any]:
    """Build quality and frozen decisions without loading a target stream."""

    run_dir = Path(str(capture["run_dir"])).resolve()
    report_path = run_dir / "pending_round_capture_report.json"
    manifest_path = run_dir / "pending_round_capture_manifest.json"
    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    if report.get("resolution_provider_called") is not False or manifest.get(
        "resolution_provider_called"
    ) is not False:
        raise ValueError("capture attempted to access resolution")
    feature_rows = _current_feature_rows(run_dir=run_dir, manifest=manifest)
    quality = _quality_observations(
        capture=capture,
        report=report,
        feature_rows=feature_rows,
    )
    decisions = _score_decisions(
        feature_rows=feature_rows,
        runtime=runtime,
        baseline=baseline,
    )
    market_ids = {str(row["market_id"]) for row in feature_rows}
    market_id = next(iter(market_ids)) if len(market_ids) == 1 else None
    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "attempt_index": int(capture["round_index"]),
        "attempt_id": str(capture["run_id"]),
        "scheduled_round_start_ts": int(capture["scheduled_round_start_ts"]),
        "market_id": market_id,
        "capture_manifest_sha256": sha256_file(manifest_path),
        "capture_report_sha256": sha256_file(report_path),
        "quality": quality,
        "provider_health": {
            "provider_failed": not quality["quality_observations"][
                "provider_capture_complete"
            ],
            "retry_used": _capture_retry_used(capture),
            "paired_executable_ask_decision_count": quality[
                "paired_executable_ask_decision_count"
            ],
            "causality_violation_count": quality["causality_violation_count"],
            "missing_feature_count": quality["missing_feature_count"],
        },
        "decision_rows": decisions,
        "collection_decision_inputs": "quality_only_model_decisions_excluded",
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    assert_outcome_blind(attempt)
    return attempt


def append_attempt(
    *,
    service_root: Path | str,
    attempt: Mapping[str, Any],
    authorization_sha256: str,
    collector_protocol_sha256: str,
    candidate_bundle_sha256: str,
) -> dict[str, Any]:
    """Append one immutable hash-chained attempt and refresh progress atomically."""

    root = Path(service_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "outcome_blind_attempts.jsonl"
    attempts = _load_jsonl(ledger) if ledger.exists() else []
    expected_index = len(attempts) + 1
    if int(attempt["attempt_index"]) != expected_index:
        raise ValueError("attempt indices must be contiguous")
    if expected_index > MAXIMUM_ATTEMPTS:
        raise ValueError("maximum attempt cap exceeded")
    previous_hash = attempts[-1]["attempt_hash"] if attempts else "0" * 64
    payload = dict(attempt)
    payload["previous_attempt_hash"] = previous_hash
    payload["attempt_hash"] = canonical_attempt_hash(payload)
    assert_outcome_blind(payload)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
    attempts.append(payload)
    progress = build_progress(
        attempts,
        authorization_sha256=authorization_sha256,
        collector_protocol_sha256=collector_protocol_sha256,
        candidate_bundle_sha256=candidate_bundle_sha256,
    )
    _atomic_json(root / "collection_progress.json", progress)
    return progress


def build_progress(
    attempts: Sequence[Mapping[str, Any]],
    *,
    authorization_sha256: str,
    collector_protocol_sha256: str,
    candidate_bundle_sha256: str,
) -> dict[str, Any]:
    """Reconcile append-only progress without reading any outcome."""

    verify_attempt_chain(attempts)
    unique_valid: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for attempt in attempts:
        quality_valid = bool(dict(attempt["quality"])["quality_valid"])
        market_id = attempt.get("market_id")
        if not quality_valid or not isinstance(market_id, str):
            continue
        if market_id in seen:
            duplicates += 1
            continue
        seen.add(market_id)
        unique_valid.append(attempt)
    if len(unique_valid) > TARGET_MARKETS:
        raise ValueError("quality-valid population exceeded exact target")
    invalid_reasons = Counter(
        reason
        for attempt in attempts
        if not bool(dict(attempt["quality"])["quality_valid"])
        for reason in dict(attempt["quality"])["invalid_reason_codes"]
    )
    attempted = len(attempts)
    valid_count = len(unique_valid)
    observed_rate = valid_count / attempted if attempted else None
    remaining = TARGET_MARKETS - valid_count
    eta_days = (
        remaining / (observed_rate * 96.0)
        if observed_rate is not None and observed_rate > 0.0
        else None
    )
    progress = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "updated_at": datetime.now(UTC).isoformat(),
        "authorization_sha256": authorization_sha256,
        "collector_protocol_sha256": collector_protocol_sha256,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "attempts_consumed": attempted,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "quality_valid_market_count": valid_count,
        "target_quality_valid_market_count": TARGET_MARKETS,
        "remaining_quality_valid_markets": remaining,
        "observed_quality_valid_rate": observed_rate,
        "estimated_remaining_days_at_96_markets_per_day": eta_days,
        "invalid_attempt_count": attempted
        - sum(bool(dict(row["quality"])["quality_valid"]) for row in attempts),
        "duplicate_quality_valid_market_count": duplicates,
        "invalid_reason_distribution": dict(sorted(invalid_reasons.items())),
        "provider_failure_count": sum(
            bool(dict(row["provider_health"])["provider_failed"]) for row in attempts
        ),
        "retry_count": sum(
            bool(dict(row["provider_health"])["retry_used"]) for row in attempts
        ),
        "hash_chain_status": "valid",
        "collection_complete": valid_count == TARGET_MARKETS,
        "attempt_cap_exhausted": attempted == MAXIMUM_ATTEMPTS,
        "fresh_collection_started": attempted > 0,
        "fresh_outcomes_opened": False,
        "interim_pnl_evaluated": False,
        "collection_influenced_by_model_decisions": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    assert_outcome_blind(progress)
    return progress


def verify_attempt_chain(attempts: Sequence[Mapping[str, Any]]) -> None:
    previous = "0" * 64
    for index, attempt in enumerate(attempts, start=1):
        if int(attempt["attempt_index"]) != index:
            raise ValueError("attempt chain index mismatch")
        if attempt.get("previous_attempt_hash") != previous:
            raise ValueError("attempt chain previous hash mismatch")
        expected = canonical_attempt_hash(attempt)
        if attempt.get("attempt_hash") != expected:
            raise ValueError("attempt chain content hash mismatch")
        assert_outcome_blind(attempt)
        previous = expected


def canonical_attempt_hash(attempt: Mapping[str, Any]) -> str:
    value = dict(attempt)
    value.pop("attempt_hash", None)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def assert_outcome_blind(value: Any, *, path: str = "root") -> None:
    """Reject any recursively nested outcome, settlement, label, or PnL field."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_FIELD_TOKENS) and key not in {
                "outcomes_accessed",
                "settlement_accessed",
                "pnl_accessed",
                "fresh_outcomes_opened",
                "interim_pnl_evaluated",
            }:
                raise ValueError(f"forbidden outcome-bearing field: {path}.{key}")
            if key in {
                "outcomes_accessed",
                "settlement_accessed",
                "pnl_accessed",
                "fresh_outcomes_opened",
                "interim_pnl_evaluated",
            } and child is not False:
                raise ValueError(f"outcome-blind safety field must be false: {path}.{key}")
            assert_outcome_blind(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            assert_outcome_blind(child, path=f"{path}[{index}]")


def _score_decisions(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    runtime: ResidualPromotionRuntime,
    baseline: Any,
) -> list[dict[str, Any]]:
    candidate_accepted = False
    baseline_accepted = False
    output = []
    for feature_row in sorted(feature_rows, key=lambda row: int(row["decision_ts"])):
        observed_at = int(feature_row["decision_ts"])
        candidate = runtime.score_feature_row(
            feature_row,
            observed_at_ts=observed_at,
        )
        baseline_result = score_matched_baseline(baseline, feature_row)
        candidate_action = str(candidate["selected_action"])
        baseline_action = str(baseline_result["selected_action"])
        candidate_accept_now = not candidate_accepted and candidate_action != "NO_TRADE"
        baseline_accept_now = not baseline_accepted and baseline_action != "NO_TRADE"
        candidate_accepted = candidate_accepted or candidate_accept_now
        baseline_accepted = baseline_accepted or baseline_accept_now
        output.append(
            {
                "market_id": str(feature_row["market_id"]),
                "decision_ts": int(feature_row["decision_ts"]),
                "candidate_action_values": candidate["action_values"],
                "candidate_selected_action": candidate_action,
                "candidate_accepted_at_this_decision": candidate_accept_now,
                "baseline_action_values": baseline_result["action_values"],
                "baseline_selected_action": baseline_action,
                "baseline_accepted_at_this_decision": baseline_accept_now,
                "candidate_bundle_sha256": runtime.manifest_sha256,
                "decision_recorded_after_quality_classification": True,
                "decision_influenced_collection": False,
                "outcomes_accessed": False,
                "settlement_accessed": False,
                "pnl_accessed": False,
                "safety": dict(SAFETY),
            }
        )
    return output


def _repo_file(value: Path | str, root: Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("repository artifact is unavailable")
    return path


def _verified_json(path: Path) -> dict[str, Any]:
    sidecars = [
        candidate
        for candidate in (
            path.with_suffix(".sha256"),
            path.with_suffix(path.suffix + ".sha256"),
        )
        if candidate.is_file()
    ]
    if len(sidecars) != 1 or sidecars[0].read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ValueError(f"frozen JSON sidecar mismatch: {path}")
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "append_attempt",
    "assert_outcome_blind",
    "build_progress",
    "canonical_attempt_hash",
    "observe_outcome_blind_capture",
    "validate_collection_authorization",
    "verify_attempt_chain",
]
