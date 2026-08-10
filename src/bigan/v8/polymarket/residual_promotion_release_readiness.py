"""Fail-closed preapproval readiness for residual promotion micro-live."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.phase6 import PHASE6_CICD_PHASE, REQUIRED_STAGE_ORDER
from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    REQUIRED_GATE_NAMES,
)
from bigan.v8.polymarket.residual_promotion_rollback import (
    ROLLBACK_DRILL_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    TARGET_MARKETS,
)

CONTRACT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-micro-live-preapproval-contract-v3"
)
PREFLIGHT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-micro-live-preapproval-preflight-v3"
)
ASSESSMENT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-micro-live-preapproval-assessment-v3"
)
AUTHORIZATION_TEMPLATE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-micro-live-authorization-template-v3"
)
SHADOW_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-shadow-stability-report-v1"
)
OPERATIONAL_ROLLBACK_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-operational-rollback-report-v1"
)
SECURITY_REVIEW_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-security-review-report-v1"
)
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/"
    "BTC-15M-cost-aware-market-residual-promotion-v1"
)
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_release_readiness.py"
)
BUNDLE_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/bundle_manifest.json"
)
PARITY_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/offline_live_parity_report.json"
)
FUNCTIONAL_ROLLBACK_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/zero_capital_rollback_drill_report.json"
)
EVALUATION_CONTRACT_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/promotion_evaluation_execution_contract_v3.json"
)
FINALIZATION_NATIVE_MISSINGNESS_CORRECTION_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/finalization_native_missingness_correction.json"
)
FINALIZATION_FEATURE_ENVELOPE_CORRECTION_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/finalization_feature_envelope_correction.json"
)
PHASE6_PIPELINE_REPOSITORY_PATH = "src/bigan/v8/phase6/pipeline.py"
PHASE6_CONTRACTS_REPOSITORY_PATH = "src/bigan/v8/phase6/contracts.py"
ZERO_CAPITAL_AUTHORIZATION_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/manual_collection_authorization_v3.json"
)
HISTORICAL_PREAPPROVAL_CONTRACT_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/micro_live_preapproval_contract_v2.json"
)
HISTORICAL_MICRO_LIVE_TEMPLATE_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/micro_live_authorization_template_v2.json"
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_INITIAL_CAPITAL_FRACTION = 0.01
MAX_ROLLBACK_LATENCY_MS = 250
REQUIRED_COST_STRESS_MULTIPLIERS = (1.2, 1.5, 2.0)


def freeze_release_readiness_contract(
    *,
    repository_root: Path | str,
    created_at: str,
) -> dict[str, Any]:
    """Freeze the preapproval order without granting any live permission."""

    root = Path(repository_root).resolve()
    config = root / CONFIG_REPOSITORY_PATH
    contract_path = config / "micro_live_preapproval_contract_v3.json"
    preflight_path = config / "micro_live_preapproval_preflight_report_v3.json"
    template_path = config / "micro_live_authorization_template_v3.json"
    for path in (contract_path, preflight_path, template_path):
        if path.exists():
            raise FileExistsError(f"release readiness artifact already exists: {path.name}")

    bundle = _verified_repository_json(root, BUNDLE_REPOSITORY_PATH)
    parity = _verified_repository_json(root, PARITY_REPOSITORY_PATH)
    functional_rollback = _verified_repository_json(
        root, FUNCTIONAL_ROLLBACK_REPOSITORY_PATH
    )
    candidate_freeze_descriptor = dict(bundle.get("candidate_freeze") or {})
    candidate_freeze = _verified_repository_json(
        root, str(candidate_freeze_descriptor.get("path") or "")
    )
    source_freeze_descriptor = dict(bundle.get("candidate_source_freeze") or {})
    source_freeze = _verified_repository_json(
        root, str(source_freeze_descriptor.get("path") or "")
    )
    source_descriptors = dict(source_freeze.get("sources") or {})
    if not (
        bundle.get("lineage_id") == LINEAGE_ID
        and bundle.get("candidate_id") == CANDIDATE_ID
        and parity.get("prediction_and_decision_parity") is True
        and parity.get("fresh_outcomes_accessed") is False
        and functional_rollback.get("schema_version")
        == ROLLBACK_DRILL_SCHEMA_VERSION
        and functional_rollback.get("technical_rollback_drill_passed") is True
        and functional_rollback.get("micro_live_authorized") is False
        and functional_rollback.get("safety") == SAFETY
    ):
        raise ValueError("static release-readiness evidence is invalid")

    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "candidate_bundle": _repository_descriptor(root, BUNDLE_REPOSITORY_PATH),
        "runtime_parity": _repository_descriptor(root, PARITY_REPOSITORY_PATH),
        "functional_rollback_drill": _repository_descriptor(
            root, FUNCTIONAL_ROLLBACK_REPOSITORY_PATH
        ),
        "promotion_evaluation_contract": _repository_descriptor(
            root, EVALUATION_CONTRACT_REPOSITORY_PATH
        ),
        "finalization_native_missingness_correction": _repository_descriptor(
            root, FINALIZATION_NATIVE_MISSINGNESS_CORRECTION_REPOSITORY_PATH
        ),
        "finalization_feature_envelope_correction": _repository_descriptor(
            root, FINALIZATION_FEATURE_ENVELOPE_CORRECTION_REPOSITORY_PATH
        ),
        "phase6_pipeline": _repository_descriptor(
            root, PHASE6_PIPELINE_REPOSITORY_PATH
        ),
        "phase6_contracts": _repository_descriptor(
            root, PHASE6_CONTRACTS_REPOSITORY_PATH
        ),
        "zero_capital_authorization": _repository_descriptor(
            root, ZERO_CAPITAL_AUTHORIZATION_REPOSITORY_PATH
        ),
        "readiness_implementation": _repository_descriptor(
            root, IMPLEMENTATION_REPOSITORY_PATH
        ),
        "supersedes_preapproval_contract": _repository_descriptor(
            root, HISTORICAL_PREAPPROVAL_CONTRACT_REPOSITORY_PATH
        ),
        "phase6_candidate_identity": {
            "candidate_run_id": CANDIDATE_ID,
            "model_sha256": sha256_file(root / BUNDLE_REPOSITORY_PATH),
            "policy_dataset_hash": dict(
                source_descriptors["development_final_fit_rows"]
            )["sha256"],
            "split_hash": dict(source_descriptors["v4_challenger_oof_manifest"])[
                "sha256"
            ],
            "development_population_sha256": candidate_freeze[
                "development_population_sha256"
            ],
        },
        "required_future_evidence": {
            "fresh_confirmation": {
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "exact_market_count": TARGET_MARKETS,
                "all_frozen_gates_must_pass": True,
                "evaluation_exactly_once": True,
                "failed_population_reuse_allowed": False,
            },
            "shadow_stability": {
                "schema_version": SHADOW_SCHEMA_VERSION,
                "exact_candidate_rows": TARGET_MARKETS,
                "exact_baseline_rows": TARGET_MARKETS,
                "exact_paired_rows": TARGET_MARKETS,
                "zero_capital_read_only": True,
                "runtime_decision_parity_required": True,
            },
            "operational_rollback": {
                "schema_version": OPERATIONAL_ROLLBACK_SCHEMA_VERSION,
                "rollback_target": "NO_TRADE",
                "maximum_latency_ms": MAX_ROLLBACK_LATENCY_MS,
                "safe_parameter_hash_required": True,
            },
            "security_review": {
                "schema_version": SECURITY_REVIEW_SCHEMA_VERSION,
                "btc_15m_only_allowlist_required": True,
                "idempotent_order_identity_required": True,
                "reconciliation_required": True,
                "kill_switch_required": True,
            },
            "phase6_zero_capital": {
                "phase": PHASE6_CICD_PHASE,
                "required_stage_order": list(REQUIRED_STAGE_ORDER),
                "all_stage_gates_must_pass": True,
                "rollout_step_index": 0,
                "requested_capital_fraction": 0.0,
                "existing_zero_capital_authorization_required": True,
            },
        },
        "phase6_preapproval_semantics": {
            "training_identity_bound_to_candidate_bundle": True,
            "validation_bound_to_fresh_confirmation": True,
            "shadow_and_monitoring_required": True,
            "cost_stress_multipliers": list(REQUIRED_COST_STRESS_MULTIPLIERS),
            "functional_and_operational_rollback_required": True,
            "full_phase6_pipeline_runs_at_zero_capital_before_1pct_request": True,
            "zero_capital_phase6_pass_can_only_request_human_go_no_go": True,
        },
        "approval_order": [
            "fresh_confirmation_passes_exactly_once",
            "runtime_parity_reconciles",
            "shadow_stability_and_monitoring_pass",
            "functional_and_operational_rollback_pass",
            "security_review_passes",
            "full_phase6_pipeline_passes_at_zero_capital",
            "request_explicit_human_1_percent_micro_live_go_no_go",
            "only_after_explicit_1pct_approval_create_executable_authorization",
        ],
        "micro_live": {
            "maximum_initial_capital_fraction": MAX_INITIAL_CAPITAL_FRACTION,
            "automatic_authorization_allowed": False,
            "automatic_launch_allowed": False,
            "explicit_human_go_no_go_required": True,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(contract_path, contract)
    validate_release_readiness_contract(
        contract,
        repository_root=root,
        expected_implementation_sha256=sha256_file(
            root / IMPLEMENTATION_REPOSITORY_PATH
        ),
    )
    preflight = assess_micro_live_preapproval(
        contract=contract,
        evidence={},
        created_at=created_at,
    )
    preflight["schema_version"] = PREFLIGHT_SCHEMA_VERSION
    _write_frozen_json(preflight_path, preflight)
    template = {
        "schema_version": AUTHORIZATION_TEMPLATE_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "preapproval_contract": _repository_descriptor(
            root, contract_path.relative_to(root).as_posix()
        ),
        "supersedes_authorization_template": _repository_descriptor(
            root, HISTORICAL_MICRO_LIVE_TEMPLATE_REPOSITORY_PATH
        ),
        "required_evidence_hashes": {
            "preapproval_assessment_sha256": None,
            "fresh_evaluation_manifest_sha256": None,
            "phase6_release_manifest_sha256": None,
            "operational_rollback_report_sha256": None,
            "security_review_report_sha256": None,
        },
        "requested_initial_capital_fraction": MAX_INITIAL_CAPITAL_FRACTION,
        "explicit_human_approval_recorded": False,
        "micro_live_authorized": False,
        "micro_live_started": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "executable": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(template_path, template)
    return {
        "contract": _repository_descriptor(
            root, contract_path.relative_to(root).as_posix()
        ),
        "preflight": _repository_descriptor(
            root, preflight_path.relative_to(root).as_posix()
        ),
        "authorization_template": _repository_descriptor(
            root, template_path.relative_to(root).as_posix()
        ),
        "ready_to_request_micro_live_approval": False,
        "micro_live_authorized": False,
        "safety": dict(SAFETY),
    }


def validate_release_readiness_contract(
    contract: Mapping[str, Any],
    *,
    repository_root: Path | str,
    expected_implementation_sha256: str,
) -> None:
    """Validate all static bytes and the fail-closed approval ordering."""

    root = Path(repository_root).resolve()
    for field in (
        "candidate_bundle",
        "runtime_parity",
        "functional_rollback_drill",
        "promotion_evaluation_contract",
        "finalization_native_missingness_correction",
        "finalization_feature_envelope_correction",
        "phase6_pipeline",
        "phase6_contracts",
        "zero_capital_authorization",
        "readiness_implementation",
        "supersedes_preapproval_contract",
    ):
        _verify_repository_descriptor(root, contract.get(field))
    implementation = dict(contract.get("readiness_implementation") or {})
    order = list(contract.get("approval_order") or [])
    micro_live = dict(contract.get("micro_live") or {})
    semantics = dict(contract.get("phase6_preapproval_semantics") or {})
    future = dict(contract.get("required_future_evidence") or {})
    fresh = dict(future.get("fresh_confirmation") or {})
    shadow = dict(future.get("shadow_stability") or {})
    rollback = dict(future.get("operational_rollback") or {})
    security = dict(future.get("security_review") or {})
    phase6_zero = dict(future.get("phase6_zero_capital") or {})
    identity = dict(contract.get("phase6_candidate_identity") or {})
    historical = dict(contract.get("supersedes_preapproval_contract") or {})
    if not (
        contract.get("schema_version") == CONTRACT_SCHEMA_VERSION
        and contract.get("lineage_id") == LINEAGE_ID
        and contract.get("candidate_id") == CANDIDATE_ID
        and implementation.get("path") == IMPLEMENTATION_REPOSITORY_PATH
        and implementation.get("sha256") == expected_implementation_sha256
        and order
        == [
            "fresh_confirmation_passes_exactly_once",
            "runtime_parity_reconciles",
            "shadow_stability_and_monitoring_pass",
            "functional_and_operational_rollback_pass",
            "security_review_passes",
            "full_phase6_pipeline_passes_at_zero_capital",
            "request_explicit_human_1_percent_micro_live_go_no_go",
            "only_after_explicit_1pct_approval_create_executable_authorization",
        ]
        and semantics.get(
            "full_phase6_pipeline_runs_at_zero_capital_before_1pct_request"
        )
        is True
        and semantics.get("zero_capital_phase6_pass_can_only_request_human_go_no_go")
        is True
        and fresh.get("evaluation_schema_version") == EVALUATION_SCHEMA_VERSION
        and fresh.get("exact_market_count") == TARGET_MARKETS
        and fresh.get("all_frozen_gates_must_pass") is True
        and fresh.get("evaluation_exactly_once") is True
        and fresh.get("failed_population_reuse_allowed") is False
        and shadow.get("schema_version") == SHADOW_SCHEMA_VERSION
        and shadow.get("exact_candidate_rows") == TARGET_MARKETS
        and shadow.get("exact_baseline_rows") == TARGET_MARKETS
        and shadow.get("exact_paired_rows") == TARGET_MARKETS
        and shadow.get("zero_capital_read_only") is True
        and shadow.get("runtime_decision_parity_required") is True
        and rollback.get("schema_version") == OPERATIONAL_ROLLBACK_SCHEMA_VERSION
        and rollback.get("rollback_target") == "NO_TRADE"
        and rollback.get("maximum_latency_ms") == MAX_ROLLBACK_LATENCY_MS
        and rollback.get("safe_parameter_hash_required") is True
        and security.get("schema_version") == SECURITY_REVIEW_SCHEMA_VERSION
        and security.get("btc_15m_only_allowlist_required") is True
        and security.get("idempotent_order_identity_required") is True
        and security.get("reconciliation_required") is True
        and security.get("kill_switch_required") is True
        and phase6_zero.get("phase") == PHASE6_CICD_PHASE
        and phase6_zero.get("required_stage_order") == list(REQUIRED_STAGE_ORDER)
        and phase6_zero.get("all_stage_gates_must_pass") is True
        and phase6_zero.get("rollout_step_index") == 0
        and phase6_zero.get("requested_capital_fraction") == 0.0
        and phase6_zero.get("existing_zero_capital_authorization_required") is True
        and identity.get("candidate_run_id") == CANDIDATE_ID
        and historical.get("path")
        == HISTORICAL_PREAPPROVAL_CONTRACT_REPOSITORY_PATH
        and _looks_like_sha256(identity.get("model_sha256"))
        and _looks_like_sha256(identity.get("policy_dataset_hash"))
        and _looks_like_sha256(identity.get("split_hash"))
        and _looks_like_sha256(identity.get("development_population_sha256"))
        and micro_live.get("maximum_initial_capital_fraction")
        == MAX_INITIAL_CAPITAL_FRACTION
        and micro_live.get("automatic_authorization_allowed") is False
        and micro_live.get("automatic_launch_allowed") is False
        and micro_live.get("explicit_human_go_no_go_required") is True
        and dict(contract.get("safety") or {}) == SAFETY
    ):
        raise ValueError("release readiness contract is invalid")


def assess_micro_live_preapproval(
    *,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    """Assess technical readiness; never grant or execute micro-live."""

    expected = {
        "evaluation_manifest",
        "evaluation_report",
        "shadow_stability",
        "operational_rollback",
        "security_review",
        "phase6_report",
    }
    extra = sorted(set(evidence) - expected)
    if extra:
        raise ValueError(f"unexpected preapproval evidence: {extra}")
    checks = {
        "fresh_confirmation": _fresh_confirmation_passes(evidence),
        "runtime_parity": _static_runtime_parity_passes(contract),
        "shadow_stability_and_monitoring": _shadow_passes(evidence, contract),
        "functional_rollback": _functional_rollback_passes(contract),
        "operational_rollback": _operational_rollback_passes(evidence, contract),
        "security_review": _security_review_passes(evidence, contract),
        "phase6_zero_capital_pipeline": _phase6_zero_capital_passes(
            evidence, contract
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    technical_passed = not failed
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "evidence_sha256": {
            name: canonical_json_sha256(dict(value))
            for name, value in sorted(evidence.items())
        },
        "technical_checks": checks,
        "failed_or_missing_checks": failed,
        "phase6_zero_capital_pipeline_passed": checks[
            "phase6_zero_capital_pipeline"
        ],
        "phase6_one_percent_live_stage_executed": False,
        "ready_to_request_micro_live_approval": technical_passed,
        "status": (
            "READY_TO_REQUEST_HUMAN_1_PERCENT_MICRO_LIVE_GO_NO_GO"
            if technical_passed
            else "NO_GO_PREREQUISITES_INCOMPLETE"
        ),
        "requested_initial_capital_fraction": MAX_INITIAL_CAPITAL_FRACTION,
        "explicit_human_approval_recorded": False,
        "micro_live_authorized": False,
        "micro_live_started": False,
        "automatic_live_unlock": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def run_micro_live_preapproval_assessment(
    *,
    repository_root: Path | str,
    contract_path: Path | str,
    expected_contract_sha256: str,
    evidence_root: Path | str,
    evidence_descriptors: Mapping[str, Mapping[str, str]],
    output_path: Path | str,
    created_at: str,
) -> dict[str, Any]:
    """Load exact future evidence and write a non-authorizing assessment."""

    root = Path(repository_root).resolve()
    contract_file = _repository_file(root, contract_path)
    if sha256_file(contract_file) != expected_contract_sha256:
        raise ValueError("release readiness contract SHA-256 mismatch")
    contract = _verified_json_with_sidecar(contract_file)
    validate_release_readiness_contract(
        contract,
        repository_root=root,
        expected_implementation_sha256=sha256_file(
            root / IMPLEMENTATION_REPOSITORY_PATH
        ),
    )
    evidence_base = Path(evidence_root).resolve()
    loaded: dict[str, Mapping[str, Any]] = {}
    for name, descriptor in evidence_descriptors.items():
        if set(descriptor) != {"path", "sha256"}:
            raise ValueError("future evidence descriptor is invalid")
        path = (evidence_base / descriptor["path"]).resolve()
        if (
            not path.is_relative_to(evidence_base)
            or not path.is_file()
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError("future evidence SHA-256 mismatch or path escape")
        loaded[name] = _load_json(path)
    manifest = dict(loaded.get("evaluation_manifest") or {})
    report_descriptor = dict(manifest.get("evaluation_report") or {})
    supplied_report_descriptor = dict(
        evidence_descriptors.get("evaluation_report") or {}
    )
    if report_descriptor.get("sha256") != supplied_report_descriptor.get("sha256"):
        raise ValueError("evaluation manifest/report SHA-256 binding mismatch")
    report = assess_micro_live_preapproval(
        contract=contract,
        evidence=loaded,
        created_at=created_at,
    )
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("preapproval assessment already exists; rerun forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_frozen_json(output, report)
    return report


def _fresh_confirmation_passes(
    evidence: Mapping[str, Mapping[str, Any]],
) -> bool:
    manifest = dict(evidence.get("evaluation_manifest") or {})
    report = dict(evidence.get("evaluation_report") or {})
    population = dict(report.get("population") or {})
    gate_results = dict(report.get("gate_results") or {})
    report_descriptor = dict(manifest.get("evaluation_report") or {})
    return bool(
        manifest.get("lineage_id") == LINEAGE_ID
        and manifest.get("candidate_id") == CANDIDATE_ID
        and manifest.get("evaluation_executed_exactly_once") is True
        and manifest.get("rerun_allowed") is False
        and manifest.get("fresh_population_reuse_allowed") is False
        and manifest.get("all_fresh_confirmation_gates_passed") is True
        and manifest.get("lineage_terminalized") is False
        and manifest.get("automatic_promotion_or_live_unlock") is False
        and manifest.get("micro_live_approval_granted") is False
        and _looks_like_sha256(manifest.get("population_manifest_sha256"))
        and set(report_descriptor) == {"path", "sha256"}
        and _looks_like_sha256(report_descriptor.get("sha256"))
        and report.get("schema_version") == EVALUATION_SCHEMA_VERSION
        and report.get("production_evaluation") is True
        and report.get("all_gates_passed") is True
        and report.get("failed_gates") == []
        and set(gate_results) == REQUIRED_GATE_NAMES
        and report.get("lineage_terminalized") is False
        and report.get("failed_population_reuse_allowed") is False
        and report.get("phase6_required") is True
        and report.get("rollback_drill_required") is True
        and report.get("micro_live_go_no_go")
        == "NO_GO_PENDING_PHASE6_AND_ROLLBACK_DRILL"
        and report.get("automatic_promotion_or_live_unlock") is False
        and population.get("passed") is True
        and population.get("paired_market_count") == TARGET_MARKETS
        and gate_results
        and all(value is True for value in gate_results.values())
        and _safety_is_closed(manifest)
        and _safety_is_closed(report)
    )


def _static_runtime_parity_passes(contract: Mapping[str, Any]) -> bool:
    return _descriptor_matches(
        contract.get("runtime_parity"), PARITY_REPOSITORY_PATH
    )


def _functional_rollback_passes(contract: Mapping[str, Any]) -> bool:
    return _descriptor_matches(
        contract.get("functional_rollback_drill"),
        FUNCTIONAL_ROLLBACK_REPOSITORY_PATH,
    )


def _shadow_passes(
    evidence: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]
) -> bool:
    report = dict(evidence.get("shadow_stability") or {})
    evaluation_manifest = dict(evidence.get("evaluation_manifest") or {})
    bundle_sha = dict(contract.get("candidate_bundle") or {}).get("sha256")
    return bool(
        report.get("schema_version") == SHADOW_SCHEMA_VERSION
        and report.get("lineage_id") == LINEAGE_ID
        and report.get("candidate_id") == CANDIDATE_ID
        and report.get("candidate_bundle_sha256") == bundle_sha
        and report.get("population_manifest_sha256")
        == evaluation_manifest.get("population_manifest_sha256")
        and report.get("candidate_row_count") == TARGET_MARKETS
        and report.get("baseline_row_count") == TARGET_MARKETS
        and report.get("paired_row_count") == TARGET_MARKETS
        and report.get("zero_capital_read_only") is True
        and report.get("runtime_decision_parity_passed") is True
        and report.get("shadow_stability_passed") is True
        and report.get("monitoring_enabled") is True
        and report.get("kill_switch_wired") is True
        and report.get("collection_population_changed") is False
        and report.get("outcomes_accessed_during_collection") is False
        and report.get("live_trading_allowed") is False
        and report.get("wallet_signing_allowed") is False
        and report.get("polymarket_write_allowed") is False
        and report.get("capital_at_risk") is False
        and _safety_is_closed(report)
    )


def _operational_rollback_passes(
    evidence: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]
) -> bool:
    report = dict(evidence.get("operational_rollback") or {})
    bundle_sha = dict(contract.get("candidate_bundle") or {}).get("sha256")
    functional_sha = dict(contract.get("functional_rollback_drill") or {}).get(
        "sha256"
    )
    measurements = report.get("latency_measurements_ms")
    safe_parameters = dict(report.get("safe_parameters") or {})
    valid_measurements = bool(
        isinstance(measurements, list)
        and measurements
        and all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in measurements
        )
    )
    maximum = max(map(float, measurements)) if valid_measurements else math.inf
    return bool(
        report.get("schema_version") == OPERATIONAL_ROLLBACK_SCHEMA_VERSION
        and report.get("lineage_id") == LINEAGE_ID
        and report.get("candidate_id") == CANDIDATE_ID
        and report.get("candidate_bundle_sha256") == bundle_sha
        and report.get("functional_rollback_report_sha256") == functional_sha
        and report.get("rollback_target") == "NO_TRADE"
        and safe_parameters
        and report.get("safe_parameters_sha256")
        == canonical_json_sha256(safe_parameters)
        and valid_measurements
        and maximum <= MAX_ROLLBACK_LATENCY_MS
        and report.get("maximum_observed_latency_ms") == maximum
        and report.get("rollback_drill_passed") is True
        and report.get("micro_live_authorized") is False
        and report.get("live_trading_allowed") is False
        and report.get("wallet_signing_allowed") is False
        and report.get("polymarket_write_allowed") is False
        and report.get("capital_at_risk") is False
        and _safety_is_closed(report)
    )


def _security_review_passes(
    evidence: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]
) -> bool:
    report = dict(evidence.get("security_review") or {})
    bundle_sha = dict(contract.get("candidate_bundle") or {}).get("sha256")
    return bool(
        report.get("schema_version") == SECURITY_REVIEW_SCHEMA_VERSION
        and report.get("lineage_id") == LINEAGE_ID
        and report.get("candidate_id") == CANDIDATE_ID
        and report.get("candidate_bundle_sha256") == bundle_sha
        and report.get("security_review_passed") is True
        and report.get("btc_15m_only_allowlist_verified") is True
        and report.get("idempotent_order_identity_verified") is True
        and report.get("order_fill_position_cash_settlement_reconciliation_verified")
        is True
        and report.get("kill_switch_verified") is True
        and report.get("maximum_initial_capital_fraction")
        == MAX_INITIAL_CAPITAL_FRACTION
        and report.get("explicit_human_approval_recorded") is False
        and report.get("micro_live_authorized") is False
        and report.get("live_trading_allowed") is False
        and report.get("wallet_signing_allowed") is False
        and report.get("polymarket_write_allowed") is False
        and report.get("capital_at_risk") is False
        and _safety_is_closed(report)
    )


def _phase6_zero_capital_passes(
    evidence: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]
) -> bool:
    report = dict(evidence.get("phase6_report") or {})
    identity = dict(contract.get("phase6_candidate_identity") or {})
    expected_authorization_sha = dict(
        contract.get("zero_capital_authorization") or {}
    ).get("sha256")
    report_identity = dict(report.get("candidate_identity") or {})
    stage_gates = list(report.get("stage_gates") or [])
    rollback_gate = dict(report.get("rollback_gate") or {})
    release = dict(report.get("release_manifest") or {})
    stage_evidence = list(release.get("stage_evidence") or [])
    live_entries = [
        dict(item)
        for item in stage_evidence
        if isinstance(item, Mapping) and item.get("stage") == "live_deployment"
    ]
    live_metadata = (
        dict(live_entries[0].get("metadata") or {})
        if len(live_entries) == 1
        else {}
    )
    acceptance = dict(report.get("acceptance_criteria") or {})
    return bool(
        report.get("phase") == PHASE6_CICD_PHASE
        and report.get("passed") is True
        and report.get("deployment_status") == "approved_for_staged_live"
        and report.get("candidate_run_id") == CANDIDATE_ID
        and report.get("candidate_identity_verified") is True
        and report_identity == {
            "candidate_run_id": identity.get("candidate_run_id"),
            "model_sha256": identity.get("model_sha256"),
            "policy_dataset_hash": identity.get("policy_dataset_hash"),
            "split_hash": identity.get("split_hash"),
        }
        and [gate.get("stage") for gate in stage_gates]
        == list(REQUIRED_STAGE_ORDER)
        and all(gate.get("allowed") is True for gate in stage_gates)
        and acceptance
        and all(value is True for value in acceptance.values())
        and rollback_gate.get("available") is True
        and rollback_gate.get("latency_within_threshold") is True
        and float(rollback_gate.get("max_observed_latency_ms", math.inf))
        <= MAX_ROLLBACK_LATENCY_MS
        and release.get("manual_approval_required") is True
        and live_metadata.get("manual_approval_recorded") is True
        and live_metadata.get("zero_capital_authorization_sha256")
        == expected_authorization_sha
        and live_metadata.get("rollout_step_index") == 0
        and live_metadata.get("requested_capital_fraction") == 0.0
        and live_metadata.get("capital_at_risk") is False
        and live_metadata.get("wallet_signing_allowed") is False
        and live_metadata.get("polymarket_write_allowed") is False
        and live_metadata.get("one_percent_micro_live_authorized") is False
    )


def _safety_is_closed(payload: Mapping[str, Any]) -> bool:
    safety = dict(payload.get("safety") or {})
    return bool(
        safety == SAFETY
        and payload.get("live_trading_allowed", False) is False
        and payload.get("wallet_signing_allowed", False) is False
        and payload.get("polymarket_write_allowed", False) is False
        and payload.get("capital_at_risk", False) is False
    )


def _descriptor_matches(value: Any, expected_path: str) -> bool:
    descriptor = dict(value or {})
    return bool(
        descriptor.get("path") == expected_path
        and _looks_like_sha256(descriptor.get("sha256"))
    )


def _repository_descriptor(root: Path, relative_path: str) -> dict[str, str]:
    path = _repository_file(root, relative_path)
    return {"path": relative_path, "sha256": sha256_file(path)}


def _verify_repository_descriptor(root: Path, value: Any) -> Path:
    descriptor = dict(value or {})
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError("repository descriptor is invalid")
    path = _repository_file(root, str(descriptor["path"]))
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError("repository descriptor SHA-256 mismatch")
    return path


def _repository_file(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("repository artifact is missing or escaped root")
    return resolved


def _verified_repository_json(root: Path, relative_path: str) -> dict[str, Any]:
    return _verified_json_with_sidecar(_repository_file(root, relative_path))


def _verified_json_with_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise ValueError("frozen JSON sidecar mismatch")
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


def _looks_like_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "AUTHORIZATION_TEMPLATE_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "OPERATIONAL_ROLLBACK_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "SECURITY_REVIEW_SCHEMA_VERSION",
    "SHADOW_SCHEMA_VERSION",
    "assess_micro_live_preapproval",
    "freeze_release_readiness_contract",
    "run_micro_live_preapproval_assessment",
    "validate_release_readiness_contract",
]
