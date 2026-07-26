"""Fail-closed promotion audit for the v8.1 attempt-002 evidence bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_attempt_002 import (
    ATTEMPT_002_RESULT_SCHEMA_VERSION,
    CANDIDATE_ID,
    evaluate_attempt_002_future_rows,
    validate_attempt_002_preregistration,
)
from bigan.v8.polymarket.challenge_attempt_002_pipeline import (
    PIPELINE_MANIFEST_SCHEMA_VERSION,
    PIPELINE_RESULT_SCHEMA_VERSION,
    TARGET_ACCESS_CLAIM_SCHEMA_VERSION,
    build_attempt_002_settled_comparison,
    validate_attempt_002_operator_authorization,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_model_promotion import (
    audit_challenge_model_promotion,
)
from bigan.v8.polymarket.regime_diagnostics import DIMENSION_BUCKETS

ATTEMPT_002_PROMOTION_READINESS_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-promotion-readiness-v1"
)
PROTOCOL_FILENAME = "challenge_attempt_002_preregistration.json"
EXECUTION_MANIFEST_FILENAME = "challenge_attempt_002_execution_manifest.json"
REQUIRED_SUPPLEMENTAL_EVIDENCE = {
    "provider_health_diagnostics_report": (
        "bigan-v8-provider-health-diagnostics-v1"
    ),
    "regime_stratified_pnl_report": (
        "bigan-v8-regime-stratified-pnl-report-v1"
    ),
    "replay_parity_report": (
        "bigan-v8-challenge-execution-policy-replay-parity-v1"
    ),
    "policy_safety_report": (
        "bigan-v8-challenge-execution-policy-safety-v1"
    ),
    "policy_reconciliation_report": (
        "bigan-v8-challenge-execution-policy-reconciliation-v1"
    ),
}


class ChallengeAttempt002PromotionError(ValueError):
    """Raised when supplied promotion evidence is malformed or hash-invalid."""


def audit_attempt_002_promotion(
    *,
    repository_root: Path | str,
    future_evidence_manifest: Mapping[str, Any] | None = None,
    supplemental_runtime_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute attempt-002 evidence and audit every promotion prerequisite."""

    root = Path(repository_root).resolve()
    config_dir = root / "examples/v8/polymarket_configs"
    protocol_path = config_dir / PROTOCOL_FILENAME
    execution_manifest_path = config_dir / EXECUTION_MANIFEST_FILENAME
    protocol_sha256 = _verify_sidecar(protocol_path, label="attempt-002 protocol")
    execution_manifest_sha256 = _verify_sidecar(
        execution_manifest_path,
        label="attempt-002 execution manifest",
    )
    protocol = _load_json(protocol_path)
    validate_attempt_002_preregistration(protocol)
    execution_manifest = _load_json(execution_manifest_path)

    legacy_static = audit_challenge_model_promotion(repository_root=root)
    static_checks = dict(legacy_static["static_checks"])
    static_checks.update(
        {
            "attempt_002_preregistration_hash_verified": True,
            "attempt_002_preregistration_semantics_valid": True,
            "attempt_002_execution_manifest_hash_verified": True,
            "attempt_002_execution_manifest_fail_closed": (
                execution_manifest.get("attempt_id") == protocol["attempt_id"]
                and execution_manifest.get("candidate_id") == CANDIDATE_ID
                and execution_manifest.get("frozen_protocol", {}).get("sha256")
                == protocol_sha256
                and execution_manifest.get("collection_state", {}).get(
                    "operator_authorization_required"
                )
                is True
                and execution_manifest.get("pipeline_contract", {}).get(
                    "exact_target_free_pair_count"
                )
                == 120
                and execution_manifest.get("pipeline_contract", {}).get(
                    "single_use_target_access_claim_required"
                )
                is True
                and execution_manifest.get("pipeline_contract", {}).get(
                    "synthetic_evidence_promotion_eligible"
                )
                is False
                and execution_manifest.get("safety") == SAFE_FALSES
            ),
        }
    )
    evidence_checks = _empty_evidence_checks()
    evidence_hashes: dict[str, str] = {}
    future_manifest: dict[str, Any] = {}
    future_result: dict[str, Any] = {}

    runtime = dict(supplemental_runtime_evidence or {})
    if future_evidence_manifest is not None:
        future_manifest_path, future_manifest = _verified_json_descriptor(
            future_evidence_manifest,
            label="attempt-002 future evidence manifest",
        )
        future_manifest_sha256 = _sha256_file(future_manifest_path)
        evidence_hashes["future_evidence_manifest"] = future_manifest_sha256
        bundle = _validated_future_bundle(
            future_manifest=future_manifest,
            future_manifest_sha256=future_manifest_sha256,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            runtime=runtime,
        )
        evidence_checks.update(bundle["checks"])
        evidence_hashes.update(bundle["hashes"])
        future_result = bundle["result"]
        _validate_supplemental_reports(
            runtime=runtime,
            attempt_id=str(protocol["attempt_id"]),
            future_manifest_sha256=future_manifest_sha256,
            future_result_sha256=bundle["hashes"]["future_result"],
            checks=evidence_checks,
            hashes=evidence_hashes,
        )

    blockers = [
        f"static:{name}"
        for name, passed in static_checks.items()
        if not passed
    ] + [
        f"evidence:{name}"
        for name, passed in evidence_checks.items()
        if not passed
    ]
    eligible = not blockers
    return {
        "schema_version": ATTEMPT_002_PROMOTION_READINESS_SCHEMA_VERSION,
        "objective": "promote challenge model v8.1 to champion model",
        "issue_sequence": [259, 257, 255, 254, 258, 256],
        "attempt_id": protocol["attempt_id"],
        "candidate_id": CANDIDATE_ID,
        "model_version": "v8.1",
        "protocol_sha256": protocol_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
        "static_checks": static_checks,
        "static_check_failure_reasons": legacy_static[
            "static_check_failure_reasons"
        ],
        "fresh_runtime_evidence_supplied": bool(future_evidence_manifest),
        "evidence_checks": evidence_checks,
        "evidence_hashes": evidence_hashes,
        "future_gate_result": (
            future_result.get("gate_result") if future_result else None
        ),
        "decision": "PROMOTE_TO_CHAMPION" if eligible else "BLOCKED",
        "challenge_model_promotion_eligible": eligible,
        "selected_champion_candidate": CANDIDATE_ID if eligible else None,
        "blockers": blockers,
        "historical_or_synthetic_evidence_substituted_for_future_evidence": False,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": eligible,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
        "handoff_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def attempt_002_promotion_readiness_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact human-readable readiness report."""

    blockers = list(report.get("blockers") or [])
    lines = [
        "# Challenge attempt-002 promotion readiness",
        "",
        f"- model: `{report['model_version']}`",
        f"- candidate: `{report['candidate_id']}`",
        f"- decision: `{report['decision']}`",
        (
            "- static prerequisites passed: "
            f"`{all(report['static_checks'].values())}`"
        ),
        (
            "- fresh runtime evidence supplied: "
            f"`{report['fresh_runtime_evidence_supplied']}`"
        ),
        (
            "- promotion eligible: "
            f"`{report['challenge_model_promotion_eligible']}`"
        ),
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- `{blocker}`" for blocker in blockers] or ["- none"])
    lines.extend(
        [
            "",
            "Historical and synthetic evidence never substitute for the "
            "single-use attempt-002 future window.",
            "Paper, live, write, wallet, capital, handoff, #134, and #146 "
            "permissions remain closed.",
            "",
        ]
    )
    return "\n".join(lines)


def _validated_future_bundle(
    *,
    future_manifest: Mapping[str, Any],
    future_manifest_sha256: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    nested: dict[str, tuple[Path, Any]] = {}
    for name, loader in (
        ("protocol", _load_json),
        ("target_free_pairs", _load_jsonl),
        ("target_access_claim", _load_json),
        ("settlement_targets", _load_jsonl),
        ("comparison", _load_jsonl),
        ("result", _load_json),
    ):
        descriptor = future_manifest.get(name)
        if not isinstance(descriptor, Mapping):
            raise ChallengeAttempt002PromotionError(
                f"attempt-002 future manifest missing descriptor: {name}"
            )
        path = _verified_descriptor(descriptor, label=f"attempt-002 {name}")
        nested[name] = (path, loader(path))
    nested_protocol = nested["protocol"][1]
    if (
        nested["protocol"][0].read_bytes()
        != _repository_protocol_bytes(protocol)
        or _sha256_file(nested["protocol"][0]) != protocol_sha256
        or nested_protocol != protocol
    ):
        raise ChallengeAttempt002PromotionError(
            "future evidence does not use the frozen repository protocol"
        )

    pairs = nested["target_free_pairs"][1]
    claim = nested["target_access_claim"][1]
    targets = nested["settlement_targets"][1]
    comparison = nested["comparison"][1]
    result = nested["result"][1]
    authorization_descriptor = runtime.get("operator_authorization")
    if not isinstance(authorization_descriptor, Mapping):
        authorization = {}
        authorization_sha256 = ""
    else:
        authorization_path, authorization = _verified_json_descriptor(
            authorization_descriptor,
            label="attempt-002 operator authorization",
        )
        authorization_sha256 = _sha256_file(authorization_path)
        validate_attempt_002_operator_authorization(
            authorization,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        )

    recomputed_comparison = build_attempt_002_settled_comparison(
        target_free_pairs=pairs,
        settlement_targets=targets,
        target_access_claim=claim,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    if recomputed_comparison != comparison:
        raise ChallengeAttempt002PromotionError(
            "stored attempt-002 comparison does not match recomputation"
        )
    recomputed_gate = evaluate_attempt_002_future_rows(
        comparison,
        protocol=protocol,
    )
    if result.get("gate_result") != recomputed_gate:
        raise ChallengeAttempt002PromotionError(
            "stored attempt-002 gate result does not match recomputation"
        )
    result_sha256 = _sha256_file(nested["result"][0])
    comparison_sha256 = _sha256_file(nested["comparison"][0])
    claim_sha256 = _sha256_file(nested["target_access_claim"][0])
    manifest_checks = {
        "future_manifest_schema_exact": (
            future_manifest.get("schema_version")
            == PIPELINE_MANIFEST_SCHEMA_VERSION
            and future_manifest.get("attempt_id") == protocol["attempt_id"]
        ),
        "future_manifest_real_single_use": (
            future_manifest.get("synthetic_only") is False
            and future_manifest.get("real_future_evidence") is True
            and future_manifest.get("single_use") is True
            and future_manifest.get("historical_development_data_used")
            is False
        ),
        "future_manifest_promotion_evidence_eligible": (
            future_manifest.get("promotion_evidence_eligible") is True
        ),
        "future_manifest_safety_locked": (
            future_manifest.get("safety") == SAFE_FALSES
        ),
        "future_result_schema_exact": (
            result.get("schema_version") == PIPELINE_RESULT_SCHEMA_VERSION
            and result.get("attempt_id") == protocol["attempt_id"]
            and result.get("gate_result", {}).get("schema_version")
            == ATTEMPT_002_RESULT_SCHEMA_VERSION
        ),
        "future_result_real_and_not_historical": (
            result.get("synthetic_only") is False
            and result.get("real_future_evidence") is True
            and result.get("historical_development_data_used") is False
        ),
        "future_result_all_preregistered_gates_passed": (
            result.get("all_future_success_criteria_passed") is True
            and result.get("promotion_evidence_eligible") is True
            and recomputed_gate.get("all_future_success_criteria_passed")
            is True
            and recomputed_gate.get("promotion_evidence_eligible") is True
            and all(recomputed_gate.get("checks", {}).values())
        ),
        "future_result_requires_separate_promotion_audit": (
            result.get("promotion_audit_required") is True
            and result.get("automatic_promotion_allowed") is False
        ),
        "future_result_safety_locked": result.get("safety") == SAFE_FALSES,
        "target_access_claim_schema_exact": (
            claim.get("schema_version") == TARGET_ACCESS_CLAIM_SCHEMA_VERSION
        ),
        "target_access_claim_real_single_use_and_alpha_consumed": (
            claim.get("synthetic_only") is False
            and claim.get("real_future_outcomes_opened") is True
            and claim.get("attempt_and_promotion_alpha_consumed") is True
            and claim.get("single_use") is True
            and claim.get("result_selected_rerun_allowed") is False
            and claim.get("historical_development_data_used") is False
        ),
        "operator_authorization_hash_reconciles": (
            bool(authorization)
            and len(authorization_sha256) == 64
            and claim.get("operator_authorization_sha256")
            == authorization_sha256
        ),
        "exact_120_comparison_recomputed": (
            len(comparison) == 120
            and all(
                row.get("synthetic_only") is False
                and row.get("historical_development_data_used") is False
                and row.get("safety") == SAFE_FALSES
                for row in comparison
            )
        ),
        "future_bundle_hash_lineage_reconciles": (
            future_manifest.get("result", {}).get("sha256") == result_sha256
            and future_manifest.get("comparison", {}).get("sha256")
            == comparison_sha256
            and future_manifest.get("target_access_claim", {}).get("sha256")
            == claim_sha256
        ),
        "historical_or_synthetic_evidence_not_substituted": True,
    }
    return {
        "checks": manifest_checks,
        "hashes": {
            "future_manifest": future_manifest_sha256,
            "future_result": result_sha256,
            "future_comparison": comparison_sha256,
            "target_access_claim": claim_sha256,
            "operator_authorization": authorization_sha256,
        },
        "result": result,
    }


def _validate_supplemental_reports(
    *,
    runtime: Mapping[str, Any],
    attempt_id: str,
    future_manifest_sha256: str,
    future_result_sha256: str,
    checks: dict[str, bool],
    hashes: dict[str, str],
) -> None:
    reports: dict[str, dict[str, Any]] = {}
    for name, schema in REQUIRED_SUPPLEMENTAL_EVIDENCE.items():
        descriptor = runtime.get(name)
        if not isinstance(descriptor, Mapping):
            continue
        path, report = _verified_json_descriptor(
            descriptor,
            label=f"attempt-002 {name}",
        )
        hashes[name] = _sha256_file(path)
        reports[name] = report
        checks[f"{name}_schema_and_lineage_exact"] = (
            report.get("schema_version") == schema
            and report.get("attempt_id") == attempt_id
            and report.get("selected_candidate_id") == CANDIDATE_ID
            and report.get(
                "source_attempt_002_future_manifest_sha256"
            )
            == future_manifest_sha256
            and report.get("source_attempt_002_result_sha256")
            == future_result_sha256
            and report.get("safety") == SAFE_FALSES
        )

    provider = reports.get("provider_health_diagnostics_report")
    checks["issue_257_future_provider_health_complete"] = bool(provider) and (
        provider.get("decision_row_count") == 120
        and provider.get("matched_decision_count") == 120
        and provider.get("unmatched_decision_count") == 0
        and (
            provider.get("feature_completeness_report") or {}
        ).get("incomplete_feature_row_count")
        == 0
        and provider.get("diagnostic_only") is True
        and provider.get(
            "outcomes_settlement_pnl_or_future_information_used"
        )
        is False
    )

    regime = reports.get("regime_stratified_pnl_report")
    checks["issue_258_future_regime_diagnostics_complete"] = bool(regime) and (
        regime.get("all_dimension_partitions_reconcile") is True
        and regime.get("diagnostic_only") is True
        and regime.get("stratified_metrics_are_eligibility_blockers")
        is False
        and set(regime.get("reported_dimensions") or ())
        == set(DIMENSION_BUCKETS)
    )

    for name in (
        "replay_parity_report",
        "policy_safety_report",
        "policy_reconciliation_report",
    ):
        report = reports.get(name)
        checks[f"issue_256_{name}_passed"] = bool(report) and (
            report.get("policy_candidate_count") == 3
            and report.get("all_preregistered_policy_candidates_evaluated")
            is True
            and report.get("outcome_selected_policy_used") is False
            and report.get("passed") is True
        )
    checks["all_required_supplemental_runtime_evidence_present"] = (
        set(reports) == set(REQUIRED_SUPPLEMENTAL_EVIDENCE)
    )


def _empty_evidence_checks() -> dict[str, bool]:
    checks = {
        "future_manifest_schema_exact": False,
        "future_manifest_real_single_use": False,
        "future_manifest_promotion_evidence_eligible": False,
        "future_manifest_safety_locked": False,
        "future_result_schema_exact": False,
        "future_result_real_and_not_historical": False,
        "future_result_all_preregistered_gates_passed": False,
        "future_result_requires_separate_promotion_audit": False,
        "future_result_safety_locked": False,
        "target_access_claim_schema_exact": False,
        "target_access_claim_real_single_use_and_alpha_consumed": False,
        "operator_authorization_hash_reconciles": False,
        "exact_120_comparison_recomputed": False,
        "future_bundle_hash_lineage_reconciles": False,
        "historical_or_synthetic_evidence_not_substituted": False,
        "issue_257_future_provider_health_complete": False,
        "issue_258_future_regime_diagnostics_complete": False,
        "issue_256_replay_parity_report_passed": False,
        "issue_256_policy_safety_report_passed": False,
        "issue_256_policy_reconciliation_report_passed": False,
        "all_required_supplemental_runtime_evidence_present": False,
    }
    for name in REQUIRED_SUPPLEMENTAL_EVIDENCE:
        checks[f"{name}_schema_and_lineage_exact"] = False
    return checks


def _verified_json_descriptor(
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path = _verified_descriptor(descriptor, label=label)
    return path, _load_json(path)


def _verified_descriptor(
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path_value = descriptor.get("path")
    expected = str(descriptor.get("sha256") or "").lower()
    if not isinstance(path_value, str) or not path_value or not _is_sha256(
        expected
    ):
        raise ChallengeAttempt002PromotionError(
            f"{label} descriptor is malformed"
        )
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ChallengeAttempt002PromotionError(f"{label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ChallengeAttempt002PromotionError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    size = descriptor.get("size_bytes")
    if size is not None and size != path.stat().st_size:
        raise ChallengeAttempt002PromotionError(
            f"{label} size mismatch"
        )
    return path


def _verify_sidecar(path: Path, *, label: str) -> str:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ChallengeAttempt002PromotionError(
            f"{label} or SHA-256 sidecar is missing"
        )
    expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    if not _is_sha256(expected):
        raise ChallengeAttempt002PromotionError(
            f"{label} sidecar is malformed"
        )
    actual = _sha256_file(path)
    if actual != expected:
        raise ChallengeAttempt002PromotionError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def _repository_protocol_bytes(protocol: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ChallengeAttempt002PromotionError(
            f"JSON object required: {path}"
        )
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ChallengeAttempt002PromotionError(
            f"JSONL objects required: {path}"
        )
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ATTEMPT_002_PROMOTION_READINESS_SCHEMA_VERSION",
    "ChallengeAttempt002PromotionError",
    "REQUIRED_SUPPLEMENTAL_EVIDENCE",
    "attempt_002_promotion_readiness_markdown",
    "audit_attempt_002_promotion",
]
