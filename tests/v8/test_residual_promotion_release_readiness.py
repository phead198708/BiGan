"""Fail-closed release-readiness tests for residual promotion v1."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.phase5 import compute_safe_parameters_sha256
from bigan.v8.phase6 import (
    CICDPipelineConfig,
    CICDStageEvidence,
    RollbackPlan,
    run_phase6_cicd_pipeline,
)
from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    REQUIRED_GATE_NAMES,
)
from bigan.v8.polymarket.residual_promotion_release_readiness import (
    ASSESSMENT_SCHEMA_VERSION,
    AUTHORIZATION_TEMPLATE_SCHEMA_VERSION,
    CONTRACT_SCHEMA_VERSION,
    IMPLEMENTATION_REPOSITORY_PATH,
    OPERATIONAL_ROLLBACK_SCHEMA_VERSION,
    PHASE6_AUTHORIZATION_SCHEMA_VERSION,
    PHASE6_AUTHORIZATION_TEMPLATE_SCHEMA_VERSION,
    PREFLIGHT_SCHEMA_VERSION,
    SECURITY_REVIEW_SCHEMA_VERSION,
    SHADOW_SCHEMA_VERSION,
    assess_micro_live_preapproval,
    run_micro_live_preapproval_assessment,
    validate_release_readiness_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
CONTRACT = CONFIG / "micro_live_preapproval_contract_v5.json"
PREFLIGHT = CONFIG / "micro_live_preapproval_preflight_report_v5.json"
AUTHORIZATION_TEMPLATE = CONFIG / "micro_live_authorization_template_v5.json"
PHASE6_AUTHORIZATION_TEMPLATE = CONFIG / "phase6_zero_capital_authorization_template.json"
HISTORICAL_ARTIFACTS = (
    CONFIG / "micro_live_preapproval_contract.json",
    CONFIG / "micro_live_preapproval_preflight_report.json",
    CONFIG / "micro_live_authorization_template.json",
    CONFIG / "micro_live_preapproval_contract_v2.json",
    CONFIG / "micro_live_preapproval_preflight_report_v2.json",
    CONFIG / "micro_live_authorization_template_v2.json",
    CONFIG / "micro_live_preapproval_contract_v3.json",
    CONFIG / "micro_live_preapproval_preflight_report_v3.json",
    CONFIG / "micro_live_authorization_template_v3.json",
    CONFIG / "micro_live_preapproval_contract_v4.json",
    CONFIG / "micro_live_preapproval_preflight_report_v4.json",
    CONFIG / "micro_live_authorization_template_v4.json",
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _closed(payload: dict[str, object]) -> dict[str, object]:
    return {
        **payload,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def _complete_evidence(contract: dict[str, object]) -> dict[str, dict[str, object]]:
    bundle_sha = dict(contract["candidate_bundle"])["sha256"]
    functional_sha = dict(contract["functional_rollback_drill"])["sha256"]
    population_sha = "a" * 64
    safe_parameters = {"action": "NO_TRADE", "capital_fraction": 0.0}
    evidence = {
        "evaluation_manifest": _closed(
            {
                "lineage_id": contract["lineage_id"],
                "candidate_id": contract["candidate_id"],
                "evaluation_executed_exactly_once": True,
                "rerun_allowed": False,
                "fresh_population_reuse_allowed": False,
                "all_fresh_confirmation_gates_passed": True,
                "lineage_terminalized": False,
                "automatic_promotion_or_live_unlock": False,
                "micro_live_approval_granted": False,
                "population_manifest_sha256": population_sha,
                "settlement_ingestion_manifest": {
                    "path": "settlement_ingestion_manifest.json",
                    "sha256": "c" * 64,
                },
                "evaluation_report": {
                    "path": "promotion_evaluation_report.json",
                    "sha256": "b" * 64,
                },
            }
        ),
        "evaluation_report": _closed(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "production_evaluation": True,
                "population": {"passed": True, "paired_market_count": 2_500},
                "gate_results": dict.fromkeys(REQUIRED_GATE_NAMES, True),
                "all_gates_passed": True,
                "failed_gates": [],
                "lineage_terminalized": False,
                "failed_population_reuse_allowed": False,
                "phase6_required": True,
                "rollback_drill_required": True,
                "micro_live_go_no_go": ("NO_GO_PENDING_PHASE6_AND_ROLLBACK_DRILL"),
                "automatic_promotion_or_live_unlock": False,
            }
        ),
        "shadow_stability": _closed(
            {
                "schema_version": SHADOW_SCHEMA_VERSION,
                "lineage_id": contract["lineage_id"],
                "candidate_id": contract["candidate_id"],
                "implementation": dict(contract["shadow_evidence_implementation"]),
                "cli": dict(contract["shadow_evidence_cli"]),
                "candidate_bundle_sha256": bundle_sha,
                "population_manifest_sha256": population_sha,
                "candidate_row_count": 2_500,
                "baseline_row_count": 2_500,
                "paired_row_count": 2_500,
                "zero_capital_read_only": True,
                "runtime_decision_parity_passed": True,
                "shadow_stability_passed": True,
                "monitoring_enabled": True,
                "kill_switch_wired": True,
                "collection_population_changed": False,
                "outcomes_accessed_during_collection": False,
            }
        ),
        "operational_rollback": _closed(
            {
                "schema_version": OPERATIONAL_ROLLBACK_SCHEMA_VERSION,
                "lineage_id": contract["lineage_id"],
                "candidate_id": contract["candidate_id"],
                "implementation": dict(contract["operational_rollback_evidence_implementation"]),
                "cli": dict(contract["operational_rollback_evidence_cli"]),
                "candidate_bundle_sha256": bundle_sha,
                "functional_rollback_report_sha256": functional_sha,
                "rollback_target": "NO_TRADE",
                "safe_parameters": safe_parameters,
                "safe_parameters_sha256": canonical_json_sha256(safe_parameters),
                "latency_measurements_ms": [75, 92, 88],
                "maximum_observed_latency_ms": 92.0,
                "rollback_drill_passed": True,
                "micro_live_authorized": False,
            }
        ),
        "security_review": _closed(
            {
                "schema_version": SECURITY_REVIEW_SCHEMA_VERSION,
                "lineage_id": contract["lineage_id"],
                "candidate_id": contract["candidate_id"],
                "candidate_bundle_sha256": bundle_sha,
                "security_review_passed": True,
                "btc_15m_only_allowlist_verified": True,
                "idempotent_order_identity_verified": True,
                "order_fill_position_cash_settlement_reconciliation_verified": True,
                "kill_switch_verified": True,
                "maximum_initial_capital_fraction": 0.01,
                "explicit_human_approval_recorded": False,
                "micro_live_authorized": False,
            }
        ),
    }
    phase6_authorization = _closed(
        {
            "schema_version": PHASE6_AUTHORIZATION_SCHEMA_VERSION,
            "lineage_id": contract["lineage_id"],
            "candidate_id": contract["candidate_id"],
            "authorization_scope": "post_confirmation_phase6_zero_capital_only",
            "candidate_bundle_sha256": bundle_sha,
            "supersedes_template": dict(contract["phase6_zero_capital_authorization_template"]),
            "fresh_evaluation_manifest_payload_sha256": canonical_json_sha256(
                evidence["evaluation_manifest"]
            ),
            "phase6_zero_capital_authorized": True,
            "requested_capital_fraction": 0.0,
            "rollout_step_index": 0,
            "explicit_human_zero_capital_approval_recorded": True,
            "authorization_record_executable": True,
            "collection_authorization_reused": False,
            "micro_live_authorized": False,
        }
    )
    evidence["phase6_authorization"] = phase6_authorization
    evidence["phase6_report"] = _phase6_report(contract, phase6_authorization)
    return evidence


def _phase6_report(
    contract: dict[str, object], phase6_authorization: dict[str, object]
) -> dict[str, object]:
    identity = dict(contract["phase6_candidate_identity"])
    candidate_id = str(contract["candidate_id"])
    bundle_sha = str(identity["model_sha256"])
    authorization_sha = canonical_json_sha256(phase6_authorization)

    def stage(
        name: str,
        *,
        artifact_sha: str,
        metadata: dict[str, object],
    ) -> CICDStageEvidence:
        return CICDStageEvidence(
            stage=name,  # type: ignore[arg-type]
            passed=True,
            artifact_sha256=artifact_sha,
            report_sha256="f" * 64,
            run_id=f"{name}-001",
            metadata={
                "candidate_run_id": candidate_id,
                "model_sha256": identity["model_sha256"],
                "policy_dataset_hash": identity["policy_dataset_hash"],
                "split_hash": identity["split_hash"],
                **metadata,
            },
        )

    evidence = (
        stage(
            "training",
            artifact_sha=bundle_sha,
            metadata={
                "accepted_candidate_model": True,
                "deterministic_training": True,
            },
        ),
        stage(
            "validation",
            artifact_sha="1" * 64,
            metadata={
                "oos_backtest_passed": True,
                "cost_stress_passed": True,
                "cost_stress_multipliers": [1.2, 1.5, 2.0],
            },
        ),
        stage(
            "shadow_deployment",
            artifact_sha="2" * 64,
            metadata={
                "shadow_mode": True,
                "simulate_live_execution": True,
                "capital_at_risk": False,
            },
        ),
        stage(
            "live_deployment",
            artifact_sha="3" * 64,
            metadata={
                "staged_capital_rollout": True,
                "manual_approval_recorded": True,
                "zero_capital_authorization_sha256": authorization_sha,
                "rollout_capital_fractions": [0.0, 0.01, 0.05, 0.10],
                "rollout_step_index": 0,
                "requested_capital_fraction": 0.0,
                "capital_at_risk": False,
                "wallet_signing_allowed": False,
                "polymarket_write_allowed": False,
                "one_percent_micro_live_authorized": False,
            },
        ),
        stage(
            "monitoring",
            artifact_sha="4" * 64,
            metadata={
                "performance_tracking_enabled": True,
                "risk_tracking_enabled": True,
                "kill_switch_wired": True,
                "feed_health_passed": True,
            },
        ),
    )
    safe_parameters = {"action": "NO_TRADE", "capital_fraction": 0.0}
    rollback = RollbackPlan(
        stable_model_id=candidate_id,
        stable_model_sha256=bundle_sha,
        safe_parameter_sha256=compute_safe_parameters_sha256(safe_parameters),
        safe_parameters=safe_parameters,
        rollback_artifact_sha256=dict(contract["functional_rollback_drill"])["sha256"],
        latency_measurements_ms=(75, 92, 88),
    )
    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_id,
        stage_evidence=evidence,
        rollback_plan=rollback,
        config=CICDPipelineConfig(created_at="2026-09-10T00:00:00+00:00"),
    )
    return result.report.to_dict()


def test_frozen_contract_and_sidecars_reconcile() -> None:
    contract = _json(CONTRACT)
    assert contract["schema_version"] == CONTRACT_SCHEMA_VERSION
    assert dict(contract["supersedes_preapproval_contract"])["path"].endswith(
        "/micro_live_preapproval_contract_v4.json"
    )
    assert dict(contract["finalization_feature_envelope_correction"])["path"].endswith(
        "/finalization_feature_envelope_correction.json"
    )
    assert dict(contract["promotion_settlement_ingestion_contract"])["path"].endswith(
        "/promotion_settlement_ingestion_contract.json"
    )
    assert dict(contract["promotion_outcome_authorization_template"])["path"].endswith(
        "/promotion_outcome_evaluation_authorization_template_v4.json"
    )
    assert dict(contract["phase6_zero_capital_authorization_template"])["path"].endswith(
        "/phase6_zero_capital_authorization_template.json"
    )
    assert dict(contract["collection_authorization_not_valid_for_phase6"])["path"].endswith(
        "/manual_collection_authorization_v3.json"
    )
    validate_release_readiness_contract(
        contract,
        repository_root=REPO_ROOT,
        expected_implementation_sha256=sha256_file(REPO_ROOT / IMPLEMENTATION_REPOSITORY_PATH),
    )
    for path in (
        CONTRACT,
        PREFLIGHT,
        AUTHORIZATION_TEMPLATE,
        PHASE6_AUTHORIZATION_TEMPLATE,
    ):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        assert sidecar.read_text(encoding="utf-8").strip() == sha256_file(path)
    for path in HISTORICAL_ARTIFACTS:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        assert sidecar.read_text(encoding="utf-8").strip() == sha256_file(path)


def test_current_preflight_is_explicitly_blocked() -> None:
    report = _json(PREFLIGHT)
    assert report["schema_version"] == PREFLIGHT_SCHEMA_VERSION
    assert report["technical_checks"] == {
        "fresh_confirmation": False,
        "functional_rollback": True,
        "operational_rollback": False,
        "runtime_parity": True,
        "security_review": False,
        "shadow_stability_and_monitoring": False,
        "phase6_zero_capital_pipeline": False,
    }
    assert report["status"] == "NO_GO_PREREQUISITES_INCOMPLETE"
    assert report["ready_to_request_micro_live_approval"] is False
    assert report["phase6_zero_capital_pipeline_passed"] is False
    assert report["phase6_one_percent_live_stage_executed"] is False
    assert report["micro_live_authorized"] is False
    assert report["wallet_signing_allowed"] is False
    assert report["polymarket_write_allowed"] is False
    assert report["capital_at_risk"] is False
    assert report["safety"] == SAFETY


def test_complete_technical_evidence_can_only_request_human_go_no_go() -> None:
    contract = _json(CONTRACT)
    report = assess_micro_live_preapproval(
        contract=contract,
        evidence=_complete_evidence(contract),
        created_at="2026-09-10T00:00:00+00:00",
    )
    assert report["schema_version"] == ASSESSMENT_SCHEMA_VERSION
    assert all(report["technical_checks"].values())
    assert report["phase6_zero_capital_pipeline_passed"] is True
    assert report["status"] == ("READY_TO_REQUEST_HUMAN_1_PERCENT_MICRO_LIVE_GO_NO_GO")
    assert report["ready_to_request_micro_live_approval"] is True
    assert report["requested_initial_capital_fraction"] == 0.01
    assert report["explicit_human_approval_recorded"] is False
    assert report["phase6_one_percent_live_stage_executed"] is False
    assert report["micro_live_authorized"] is False
    assert report["micro_live_started"] is False
    assert report["automatic_live_unlock"] is False
    assert report["wallet_signing_allowed"] is False
    assert report["polymarket_write_allowed"] is False
    assert report["capital_at_risk"] is False
    assert report["safety"] == SAFETY


@pytest.mark.parametrize(
    ("evidence_name", "field", "value", "failed_check"),
    (
        ("evaluation_report", "all_gates_passed", False, "fresh_confirmation"),
        (
            "shadow_stability",
            "candidate_row_count",
            2_499,
            "shadow_stability_and_monitoring",
        ),
        (
            "operational_rollback",
            "maximum_observed_latency_ms",
            251.0,
            "operational_rollback",
        ),
        (
            "security_review",
            "explicit_human_approval_recorded",
            True,
            "security_review",
        ),
        (
            "phase6_authorization",
            "collection_authorization_reused",
            True,
            "phase6_zero_capital_pipeline",
        ),
    ),
)
def test_any_failed_or_premature_evidence_stays_no_go(
    evidence_name: str,
    field: str,
    value: object,
    failed_check: str,
) -> None:
    contract = _json(CONTRACT)
    evidence = _complete_evidence(contract)
    evidence[evidence_name][field] = value
    if evidence_name == "operational_rollback":
        evidence[evidence_name]["latency_measurements_ms"] = [75, 251]
    report = assess_micro_live_preapproval(
        contract=contract,
        evidence=evidence,
        created_at="2026-09-10T00:00:00+00:00",
    )
    assert failed_check in report["failed_or_missing_checks"]
    assert report["ready_to_request_micro_live_approval"] is False
    assert report["micro_live_authorized"] is False


def test_unexpected_post_hoc_evidence_dimension_fails_closed() -> None:
    contract = _json(CONTRACT)
    evidence = _complete_evidence(contract)
    evidence["post_hoc_override"] = {"passed": True}
    with pytest.raises(ValueError, match="unexpected preapproval evidence"):
        assess_micro_live_preapproval(
            contract=contract,
            evidence=evidence,
            created_at="2026-09-10T00:00:00+00:00",
        )


def test_phase6_one_percent_step_cannot_be_smuggled_into_preapproval() -> None:
    contract = _json(CONTRACT)
    evidence = _complete_evidence(contract)
    phase6_report = evidence["phase6_report"]
    stage_evidence = phase6_report["release_manifest"]["stage_evidence"]
    live = next(row for row in stage_evidence if row["stage"] == "live_deployment")
    live["metadata"]["requested_capital_fraction"] = 0.01
    live["metadata"]["rollout_step_index"] = 1
    report = assess_micro_live_preapproval(
        contract=contract,
        evidence=evidence,
        created_at="2026-09-10T00:00:00+00:00",
    )
    assert report["technical_checks"]["phase6_zero_capital_pipeline"] is False
    assert report["ready_to_request_micro_live_approval"] is False
    assert report["phase6_one_percent_live_stage_executed"] is False
    assert report["micro_live_authorized"] is False


def test_static_byte_drift_fails_closed() -> None:
    contract = _json(CONTRACT)
    changed = copy.deepcopy(contract)
    changed["candidate_bundle"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor SHA-256 mismatch"):
        validate_release_readiness_contract(
            changed,
            repository_root=REPO_ROOT,
            expected_implementation_sha256=sha256_file(REPO_ROOT / IMPLEMENTATION_REPOSITORY_PATH),
        )


def test_authorization_template_is_non_executable_and_empty() -> None:
    template = _json(AUTHORIZATION_TEMPLATE)
    assert template["schema_version"] == AUTHORIZATION_TEMPLATE_SCHEMA_VERSION
    assert dict(template["supersedes_authorization_template"])["path"].endswith(
        "/micro_live_authorization_template_v4.json"
    )
    assert template["requested_initial_capital_fraction"] == 0.01
    assert template["explicit_human_approval_recorded"] is False
    assert template["micro_live_authorized"] is False
    assert template["micro_live_started"] is False
    assert template["executable"] is False
    assert set(template["required_evidence_hashes"].values()) == {None}
    assert template["wallet_signing_allowed"] is False
    assert template["polymarket_write_allowed"] is False
    assert template["capital_at_risk"] is False
    assert template["safety"] == SAFETY


def test_phase6_authorization_template_is_separate_and_non_executable() -> None:
    template = _json(PHASE6_AUTHORIZATION_TEMPLATE)
    assert template["schema_version"] == PHASE6_AUTHORIZATION_TEMPLATE_SCHEMA_VERSION
    assert template["authorization_scope"] == ("post_confirmation_phase6_zero_capital_only")
    assert template["phase6_zero_capital_authorized"] is False
    assert template["fresh_evaluation_manifest_payload_sha256"] is None
    assert template["explicit_human_zero_capital_approval_recorded"] is False
    assert template["authorization_record_executable"] is False
    assert template["collection_authorization_reused"] is False
    assert template["micro_live_authorized"] is False
    assert template["wallet_signing_allowed"] is False
    assert template["polymarket_write_allowed"] is False
    assert template["capital_at_risk"] is False
    assert template["safety"] == SAFETY


def test_exact_future_evidence_hashes_and_one_shot_output(tmp_path: Path) -> None:
    contract = _json(CONTRACT)
    evidence = _complete_evidence(contract)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    descriptors = {}
    report_path = evidence_root / "evaluation_report.json"
    report_path.write_text(
        json.dumps(evidence["evaluation_report"], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    descriptors["evaluation_report"] = {
        "path": report_path.name,
        "sha256": sha256_file(report_path),
    }
    evidence["evaluation_manifest"]["evaluation_report"] = {
        "path": report_path.name,
        "sha256": sha256_file(report_path),
    }
    evidence["phase6_authorization"]["fresh_evaluation_manifest_payload_sha256"] = (
        canonical_json_sha256(evidence["evaluation_manifest"])
    )
    evidence["phase6_report"] = _phase6_report(contract, evidence["phase6_authorization"])
    for name, payload in evidence.items():
        if name == "evaluation_report":
            continue
        path = evidence_root / f"{name}.json"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        descriptors[name] = {"path": path.name, "sha256": sha256_file(path)}
    output = tmp_path / "assessment/report.json"
    report = run_micro_live_preapproval_assessment(
        repository_root=REPO_ROOT,
        contract_path=CONTRACT,
        expected_contract_sha256=sha256_file(CONTRACT),
        evidence_root=evidence_root,
        evidence_descriptors=descriptors,
        output_path=output,
        created_at="2026-09-10T00:00:00+00:00",
    )
    assert report["ready_to_request_micro_live_approval"] is True
    assert report["micro_live_authorized"] is False
    assert output.with_suffix(".json.sha256").is_file()
    with pytest.raises(FileExistsError, match="rerun forbidden"):
        run_micro_live_preapproval_assessment(
            repository_root=REPO_ROOT,
            contract_path=CONTRACT,
            expected_contract_sha256=sha256_file(CONTRACT),
            evidence_root=evidence_root,
            evidence_descriptors=descriptors,
            output_path=output,
            created_at="2026-09-10T00:00:00+00:00",
        )
    changed = dict(descriptors)
    changed["security_review"] = {
        **changed["security_review"],
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="future evidence SHA-256 mismatch"):
        run_micro_live_preapproval_assessment(
            repository_root=REPO_ROOT,
            contract_path=CONTRACT,
            expected_contract_sha256=sha256_file(CONTRACT),
            evidence_root=evidence_root,
            evidence_descriptors=changed,
            output_path=tmp_path / "blocked.json",
            created_at="2026-09-10T00:00:00+00:00",
        )
