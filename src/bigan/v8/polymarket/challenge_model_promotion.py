"""Final fail-closed promotion readiness gate for the v8 challenge model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROMOTION_READINESS_SCHEMA_VERSION = "bigan-v8-challenge-model-promotion-readiness-v1"
REQUIRED_HASH_PINNED_ARTIFACTS = (
    "canonical_payload_contract.json",
    "feature_missingness_contract.json",
    "candidate_family_manifest.json",
    "candidate_budget_protocol.json",
    "family_error_control_contract.json",
    "parallel_candidate_protocol.json",
    "parallel_candidate_v8_1_primary_no_fallback_contract.json",
    "parallel_candidate_v8_3_primary_with_fallback_contract.json",
    "parallel_candidate_matched_frozen_v6_7_contract.json",
    "parallel_frozen_v8_1_model_binding.json",
    "regime_definition_contract.json",
    "execution_policy_contract.json",
    "policy_candidate_manifest.json",
    "source_execution_compatibility_manifest.json",
    "historical_replay_superiority_contract.json",
    "historical_replay_superiority_report.json",
    "challenge_future_post_freeze_protocol.json",
    "challenge_promotion_evidence_protocol.json",
)


class ChallengeModelPromotionError(ValueError):
    """Raised when a supplied runtime evidence manifest is malformed."""


def audit_challenge_model_promotion(
    *,
    repository_root: Path | str,
    runtime_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit static prerequisites and optional fresh runtime evidence."""

    root = Path(repository_root).resolve()
    config_dir = root / "examples/v8/polymarket_configs"
    artifact_checks: dict[str, bool] = {}
    artifact_hashes: dict[str, str | None] = {}
    for filename in REQUIRED_HASH_PINNED_ARTIFACTS:
        path = config_dir / filename
        sha_path = path.with_suffix(".sha256")
        expected = sha_path.read_text(encoding="ascii").strip() if sha_path.is_file() else None
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        artifact_checks[filename] = bool(expected and actual == expected)
        artifact_hashes[filename] = actual
    next_gate_path = config_dir / "next_gate_eligibility_decision.json"
    next_gate = _read_json(next_gate_path) if next_gate_path.is_file() else {}
    historical_replay_path = (
        config_dir / "historical_replay_superiority_report.json"
    )
    historical_replay = (
        _read_json(historical_replay_path)
        if historical_replay_path.is_file()
        else {}
    )
    static_checks = {
        "all_issue_contract_artifacts_hash_verified": all(artifact_checks.values()),
        "issue_255_next_fresh_gate_statistically_eligible": (
            next_gate.get("next_gate_eligible") is True
            and next_gate.get("outcomes_read_to_make_eligibility_decision") is False
        ),
        "issue_254_parallel_protocol_preregistered": artifact_checks.get(
            "parallel_candidate_protocol.json"
        )
        is True,
        "issue_256_policy_framework_preregistered": artifact_checks.get(
            "execution_policy_contract.json"
        )
        is True,
        "historical_replay_strictly_superior_before_collection": (
            artifact_checks.get("historical_replay_superiority_contract.json")
            is True
            and artifact_checks.get("historical_replay_superiority_report.json")
            is True
            and historical_replay.get("historical_superiority_gate_passed")
            is True
            and historical_replay.get(
                "future_collection_prerequisite_satisfied"
            )
            is True
            and historical_replay.get("historical_replay_is_promotion_evidence")
            is False
            and all(
                (historical_replay.get("checks") or {}).values()
            )
        ),
    }
    runtime = dict(runtime_evidence or {})
    evidence_checks = {
        "fresh_parallel_evaluation_present": False,
        "fresh_parallel_evaluation_single_use": False,
        "multiplicity_aware_candidate_selected": False,
        "selected_candidate_all_hard_gates_passed": False,
        "regime_partitions_reconcile": False,
        "regime_metrics_diagnostic_only": False,
        "offline_paper_replay_parity_passed": False,
        "execution_policy_safety_passed": False,
        "position_intent_fill_ledger_reconciled": False,
        "powered_paper_gate_passed": False,
        "evidence_attempt_and_alpha_consumed": False,
        "fresh_attempt_identity_reconciles": False,
        "parallel_freeze_lineage_reconciles": False,
        "selected_candidate_lineage_reconciles": False,
        "parallel_report_hash_lineage_reconciles": False,
        "runtime_evidence_schemas_exact": False,
        "all_preregistered_execution_policies_validated": False,
        "powered_paper_gate_is_exact_parallel_gate": False,
    }
    evidence_descriptors: dict[str, Any] = {}
    if runtime:
        evidence_descriptors = _verified_runtime_evidence(runtime)
        parallel = evidence_descriptors.get("parallel_evaluation_report", {})
        selected = parallel.get("multiplicity_aware_selected_candidate")
        selected_gate = dict(parallel.get("candidate_gates", {}).get(selected, {}))
        claim = dict(parallel.get("single_use_claim") or {})
        regime = evidence_descriptors.get("regime_stratified_pnl_report", {})
        parity = evidence_descriptors.get("replay_parity_report", {})
        safety = evidence_descriptors.get("policy_safety_report", {})
        reconciliation = evidence_descriptors.get("policy_reconciliation_report", {})
        paper = evidence_descriptors.get("powered_paper_gate_report", {})
        consumption = evidence_descriptors.get("attempt_consumption_record", {})
        parallel_report_sha256 = str(
            (runtime.get("parallel_evaluation_report") or {}).get("sha256")
            or ""
        )
        fresh_attempt_id = str(consumption.get("fresh_attempt_id") or "")
        parallel_freeze_sha256 = str(claim.get("freeze_sha256") or "")
        downstream_reports = (
            regime,
            parity,
            safety,
            reconciliation,
            paper,
        )
        exact_schemas = {
            "parallel": "bigan-v8-parallel-future-evaluation-v1",
            "regime": "bigan-v8-regime-stratified-pnl-report-v1",
            "parity": "bigan-v8-challenge-execution-policy-replay-parity-v1",
            "safety": "bigan-v8-challenge-execution-policy-safety-v1",
            "reconciliation": (
                "bigan-v8-challenge-execution-policy-reconciliation-v1"
            ),
            "paper": "bigan-v8-challenge-powered-paper-gate-v1",
            "consumption": "bigan-v8-challenge-attempt-consumption-v1",
        }
        lineage_common = (
            bool(fresh_attempt_id)
            and all(
                report.get("fresh_attempt_id") == fresh_attempt_id
                for report in downstream_reports
            )
        )
        freeze_common = (
            len(parallel_freeze_sha256) == 64
            and consumption.get("parallel_freeze_sha256")
            == parallel_freeze_sha256
            and all(
                report.get("parallel_freeze_sha256")
                == parallel_freeze_sha256
                for report in downstream_reports
            )
        )
        selected_common = (
            selected in {"v8_1_primary_no_fallback", "v8_3_primary_with_fallback"}
            and all(
                report.get("selected_candidate_id") == selected
                for report in downstream_reports
            )
        )
        report_hash_common = (
            len(parallel_report_sha256) == 64
            and all(
                report.get("source_parallel_evaluation_report_sha256")
                == parallel_report_sha256
                for report in downstream_reports
            )
        )
        evidence_checks.update(
            {
                "fresh_parallel_evaluation_present": (
                    parallel.get("schema_version")
                    == exact_schemas["parallel"]
                ),
                "fresh_parallel_evaluation_single_use": (
                    claim.get("single_use") is True
                    and claim.get("target_access_after_decision_freeze") is True
                    and claim.get("result_selected_rerun_allowed") is False
                ),
                "multiplicity_aware_candidate_selected": selected
                in {"v8_1_primary_no_fallback", "v8_3_primary_with_fallback"},
                "selected_candidate_all_hard_gates_passed": (
                    selected_gate.get("all_hard_gates_passed") is True
                ),
                "regime_partitions_reconcile": (
                    regime.get("all_dimension_partitions_reconcile") is True
                ),
                "regime_metrics_diagnostic_only": (
                    regime.get("diagnostic_only") is True
                    and regime.get("stratified_metrics_are_eligibility_blockers")
                    is False
                ),
                "offline_paper_replay_parity_passed": parity.get("passed") is True,
                "execution_policy_safety_passed": safety.get("passed") is True,
                "position_intent_fill_ledger_reconciled": (
                    reconciliation.get("passed") is True
                ),
                "powered_paper_gate_passed": (
                    paper.get("powered_paper_gate_passed") is True
                    and paper.get("paper_only") is True
                    and paper.get("capital_at_risk") is False
                    and paper.get("separate_result_selected_retest_performed")
                    is False
                ),
                "evidence_attempt_and_alpha_consumed": (
                    consumption.get("attempt_consumed") is True
                    and consumption.get("alpha_consumed") is True
                    and consumption.get("consumes_attempt") is True
                    and consumption.get("consumes_alpha") is True
                    and consumption.get("evidence_permanently_consumed") is True
                    and int(consumption.get("fresh_attempt_number") or 0) == 1
                    and float(consumption.get("familywise_window_alpha") or 0.0)
                    == 0.025
                    and float(consumption.get("per_candidate_alpha") or 0.0)
                    == 0.0125
                ),
                "fresh_attempt_identity_reconciles": lineage_common,
                "parallel_freeze_lineage_reconciles": freeze_common,
                "selected_candidate_lineage_reconciles": selected_common,
                "parallel_report_hash_lineage_reconciles": report_hash_common,
                "runtime_evidence_schemas_exact": (
                    parallel.get("schema_version") == exact_schemas["parallel"]
                    and regime.get("schema_version") == exact_schemas["regime"]
                    and parity.get("schema_version") == exact_schemas["parity"]
                    and safety.get("schema_version") == exact_schemas["safety"]
                    and reconciliation.get("schema_version")
                    == exact_schemas["reconciliation"]
                    and paper.get("schema_version") == exact_schemas["paper"]
                    and consumption.get("schema_version")
                    == exact_schemas["consumption"]
                ),
                "all_preregistered_execution_policies_validated": (
                    all(
                        report.get("policy_candidate_count") == 3
                        and report.get(
                            "all_preregistered_policy_candidates_evaluated"
                        )
                        is True
                        and report.get("outcome_selected_policy_used") is False
                        for report in (parity, safety, reconciliation)
                    )
                ),
                "powered_paper_gate_is_exact_parallel_gate": (
                    paper.get("selected_candidate_gate") == selected_gate
                    and paper.get("selected_candidate_id") == selected
                    and paper.get(
                        "source_parallel_evaluation_report_sha256"
                    )
                    == parallel_report_sha256
                    and all((paper.get("checks") or {}).values())
                ),
            }
        )
    blockers = [
        f"static:{name}" for name, passed in static_checks.items() if not passed
    ] + [
        f"evidence:{name}" for name, passed in evidence_checks.items() if not passed
    ]
    eligible = not blockers
    selected_candidate = (
        evidence_descriptors.get("parallel_evaluation_report", {}).get(
            "multiplicity_aware_selected_candidate"
        )
        if eligible
        else None
    )
    return {
        "schema_version": PROMOTION_READINESS_SCHEMA_VERSION,
        "objective": "promote challenge model to champion model",
        "issue_sequence": [259, 257, 255, 254, 258, 256],
        "artifact_checks": artifact_checks,
        "artifact_hashes": artifact_hashes,
        "static_checks": static_checks,
        "fresh_runtime_evidence_supplied": bool(runtime),
        "evidence_checks": evidence_checks,
        "decision": "PROMOTE_TO_CHAMPION" if eligible else "BLOCKED",
        "challenge_model_promotion_eligible": eligible,
        "selected_champion_candidate": selected_candidate,
        "blockers": blockers,
        "historical_or_consumed_evidence_substituted_for_fresh_evidence": False,
        "paper_candidate_unlocked": eligible,
        "promotion_unlocked": eligible,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def promotion_readiness_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Challenge Model Promotion Readiness",
        "",
        f"- decision: `{report['decision']}`",
        (
            "- static issue prerequisites: "
            f"`{all(report['static_checks'].values())}`"
        ),
        (
            "- fresh runtime evidence supplied: "
            f"`{report['fresh_runtime_evidence_supplied']}`"
        ),
        (
            "- challenge model promotion eligible: "
            f"`{report['challenge_model_promotion_eligible']}`"
        ),
        "",
        "## Blockers",
        "",
    ]
    lines.extend(
        [f"- `{blocker}`" for blocker in report["blockers"]]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "Historical or consumed results are not substituted for fresh evidence.",
            "Live, write, wallet, capital, #134, and #146 permissions remain closed.",
            "",
        ]
    )
    return "\n".join(lines)


def _verified_runtime_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    required = {
        "parallel_evaluation_report",
        "regime_stratified_pnl_report",
        "replay_parity_report",
        "policy_safety_report",
        "policy_reconciliation_report",
        "powered_paper_gate_report",
        "attempt_consumption_record",
    }
    if set(manifest) != required:
        raise ChallengeModelPromotionError(
            "runtime evidence manifest must contain exactly: "
            + ", ".join(sorted(required))
        )
    output: dict[str, Any] = {}
    for name, descriptor in manifest.items():
        if not isinstance(descriptor, dict):
            raise ChallengeModelPromotionError(f"{name} descriptor is invalid")
        path = Path(str(descriptor.get("path") or "")).resolve()
        expected = str(descriptor.get("sha256") or "")
        if (
            len(expected) != 64
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ChallengeModelPromotionError(f"{name} evidence hash mismatch")
        output[name] = _read_json(path)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ChallengeModelPromotionError(f"JSON object required: {path}")
    return payload


__all__ = [
    "PROMOTION_READINESS_SCHEMA_VERSION",
    "REQUIRED_HASH_PINNED_ARTIFACTS",
    "ChallengeModelPromotionError",
    "audit_challenge_model_promotion",
    "promotion_readiness_markdown",
]
