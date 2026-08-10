"""Strict v7 preapproval bound to the executable-path security review.

The frozen v6 contract remains historical evidence.  V7 changes only the
security-review evidence requirement and keeps every non-security readiness
gate, zero-capital boundary, and safety lock byte-for-byte equivalent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_release_readiness_v6 import (
    CONTRACT_REPOSITORY_PATH as V6_CONTRACT_REPOSITORY_PATH,
)
from bigan.v8.polymarket.residual_promotion_release_readiness_v6 import (
    IMPLEMENTATION_REPOSITORY_PATH as V6_IMPLEMENTATION_REPOSITORY_PATH,
)
from bigan.v8.polymarket.residual_promotion_release_readiness_v6 import (
    assess_micro_live_preapproval_v6,
    validate_release_readiness_contract_v6,
)
from bigan.v8.polymarket.residual_promotion_security_review_v2 import (
    IMPLEMENTATION_REPOSITORY_PATH as SECURITY_IMPLEMENTATION_REPOSITORY_PATH,
)
from bigan.v8.polymarket.residual_promotion_security_review_v2 import (
    PROTOCOL_REPOSITORY_PATH,
    validate_independent_security_review_report_v3,
    validate_security_review_protocol_v2,
)
from bigan.v8.polymarket.residual_promotion_security_review_v2 import (
    REPORT_SCHEMA_VERSION as SECURITY_REPORT_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_security_review_v2 import (
    TEMPLATE_REPOSITORY_PATH as SECURITY_TEMPLATE_REPOSITORY_PATH,
)
from bigan.v8.polymarket.residual_promotion_v1 import CANDIDATE_ID, LINEAGE_ID

CONTRACT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-micro-live-preapproval-contract-v7"
)
PREFLIGHT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-micro-live-preapproval-preflight-v7"
)
ASSESSMENT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-micro-live-preapproval-assessment-v7"
)
AUTHORIZATION_TEMPLATE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-micro-live-authorization-template-v7"
)
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
V6_PREFLIGHT_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/micro_live_preapproval_preflight_report_v6.json"
)
V6_AUTHORIZATION_TEMPLATE_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/micro_live_authorization_template_v6.json"
)
OPERATIONAL_ROLLBACK_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/operational_rollback_drill_report.json"
)
CONTRACT_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/micro_live_preapproval_contract_v7.json"
PREFLIGHT_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/micro_live_preapproval_preflight_report_v7.json"
)
AUTHORIZATION_TEMPLATE_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/micro_live_authorization_template_v7.json"
)
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_release_readiness_v7.py"
)
MAX_INITIAL_CAPITAL_FRACTION = 0.01
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_EVIDENCE_NAMES = {
    "evaluation_manifest",
    "evaluation_report",
    "shadow_stability",
    "operational_rollback",
    "security_review",
    "phase6_authorization",
    "phase6_report",
}


class ReleaseReadinessV7Error(ValueError):
    """Raised when the strict v7 release boundary cannot be trusted."""


def freeze_release_readiness_v7(
    *,
    repository_root: Path | str,
    created_at: str,
) -> dict[str, Any]:
    """Freeze v7 without granting outcome, Phase 6, or live authority."""

    root = Path(repository_root).resolve()
    contract_path = root / CONTRACT_REPOSITORY_PATH
    preflight_path = root / PREFLIGHT_REPOSITORY_PATH
    authorization_path = root / AUTHORIZATION_TEMPLATE_REPOSITORY_PATH
    for path in (contract_path, preflight_path, authorization_path):
        if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
            raise FileExistsError(f"v7 release readiness artifact exists: {path.name}")

    v6 = _verified_repository_json(root, V6_CONTRACT_REPOSITORY_PATH)
    validate_release_readiness_contract_v6(v6, repository_root=root)
    protocol = _verified_repository_json(root, PROTOCOL_REPOSITORY_PATH)
    validate_security_review_protocol_v2(
        protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(
            root / SECURITY_IMPLEMENTATION_REPOSITORY_PATH
        ),
    )
    operational = _verified_repository_json(root, OPERATIONAL_ROLLBACK_REPOSITORY_PATH)

    future = dict(v6["required_future_evidence"])
    future["security_review"] = _expected_security_gate(root)
    contract = {
        **v6,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "created_at": created_at,
        "supersedes_preapproval_contract": _repository_descriptor(
            root, V6_CONTRACT_REPOSITORY_PATH
        ),
        "v6_preapproval_contract": _repository_descriptor(
            root, V6_CONTRACT_REPOSITORY_PATH
        ),
        "historical_v6_preflight": _repository_descriptor(
            root, V6_PREFLIGHT_REPOSITORY_PATH
        ),
        "security_review_protocol": _repository_descriptor(
            root, PROTOCOL_REPOSITORY_PATH
        ),
        "security_review_template": _repository_descriptor(
            root, SECURITY_TEMPLATE_REPOSITORY_PATH
        ),
        "v7_readiness_implementation": _repository_descriptor(
            root, IMPLEMENTATION_REPOSITORY_PATH
        ),
        "required_future_evidence": future,
        "security_review_execution_path_hardening": _expected_hardening(),
    }
    _write_frozen_json(contract_path, contract)
    validate_release_readiness_contract_v7(contract, repository_root=root)

    preflight = assess_micro_live_preapproval_v7(
        contract=contract,
        evidence={"operational_rollback": operational},
        repository_root=root,
        created_at=created_at,
    )
    preflight.update(
        {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "preflight_only": True,
            "evidence_file_descriptors": {
                "operational_rollback": _repository_descriptor(
                    root, OPERATIONAL_ROLLBACK_REPOSITORY_PATH
                )
            },
            "fresh_outcomes_accessed": False,
            "settlement_accessed": False,
            "pnl_accessed": False,
        }
    )
    _write_frozen_json(preflight_path, preflight)

    authorization = {
        "schema_version": AUTHORIZATION_TEMPLATE_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "preapproval_contract": _repository_descriptor(root, CONTRACT_REPOSITORY_PATH),
        "security_review_protocol": _repository_descriptor(
            root, PROTOCOL_REPOSITORY_PATH
        ),
        "security_review_template": _repository_descriptor(
            root, SECURITY_TEMPLATE_REPOSITORY_PATH
        ),
        "supersedes_authorization_template": _repository_descriptor(
            root, V6_AUTHORIZATION_TEMPLATE_REPOSITORY_PATH
        ),
        "required_evidence_hashes": {
            "preapproval_assessment_sha256": None,
            "fresh_evaluation_manifest_sha256": None,
            "phase6_release_manifest_sha256": None,
            "phase6_zero_capital_authorization_sha256": None,
            "operational_rollback_report_sha256": None,
            "independent_security_review_report_sha256": None,
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
    _write_frozen_json(authorization_path, authorization)
    return {
        "contract": _repository_descriptor(root, CONTRACT_REPOSITORY_PATH),
        "preflight": _repository_descriptor(root, PREFLIGHT_REPOSITORY_PATH),
        "authorization_template": _repository_descriptor(
            root, AUTHORIZATION_TEMPLATE_REPOSITORY_PATH
        ),
        "security_review_passed": False,
        "ready_to_request_micro_live_approval": False,
        "micro_live_authorized": False,
        "safety": dict(SAFETY),
    }


def validate_release_readiness_contract_v7(
    contract: Mapping[str, Any],
    *,
    repository_root: Path | str,
) -> None:
    """Prove v7 changes only the security evidence boundary from v6."""

    root = Path(repository_root).resolve()
    v6 = _verified_repository_json(root, V6_CONTRACT_REPOSITORY_PATH)
    validate_release_readiness_contract_v6(v6, repository_root=root)
    protocol = _verified_repository_json(root, PROTOCOL_REPOSITORY_PATH)
    validate_security_review_protocol_v2(
        protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(
            root / SECURITY_IMPLEMENTATION_REPOSITORY_PATH
        ),
    )
    exceptions = {
        "schema_version",
        "created_at",
        "supersedes_preapproval_contract",
        "required_future_evidence",
        "security_review_protocol",
        "security_review_template",
    }
    for key, value in v6.items():
        if key not in exceptions and contract.get(key) != value:
            raise ReleaseReadinessV7Error(f"v7 relaxed or changed frozen v6 field: {key}")
    v6_future = dict(v6.get("required_future_evidence") or {})
    v7_future = dict(contract.get("required_future_evidence") or {})
    if set(v6_future) != set(v7_future):
        raise ReleaseReadinessV7Error("v7 future evidence dimensions changed")
    for key, value in v6_future.items():
        if key != "security_review" and v7_future.get(key) != value:
            raise ReleaseReadinessV7Error(f"v7 changed non-security future gate: {key}")
    if not (
        contract.get("schema_version") == CONTRACT_SCHEMA_VERSION
        and v7_future.get("security_review") == _expected_security_gate(root)
        and contract.get("security_review_execution_path_hardening")
        == _expected_hardening()
        and contract.get("safety") == SAFETY
    ):
        raise ReleaseReadinessV7Error("v7 execution-path hardening is invalid")
    for field, path in (
        ("v6_preapproval_contract", V6_CONTRACT_REPOSITORY_PATH),
        ("supersedes_preapproval_contract", V6_CONTRACT_REPOSITORY_PATH),
        ("historical_v6_preflight", V6_PREFLIGHT_REPOSITORY_PATH),
        ("security_review_protocol", PROTOCOL_REPOSITORY_PATH),
        ("security_review_template", SECURITY_TEMPLATE_REPOSITORY_PATH),
        ("v7_readiness_implementation", IMPLEMENTATION_REPOSITORY_PATH),
    ):
        _verify_descriptor(root, dict(contract.get(field) or {}), expected_path=path)
    _verify_descriptor(
        root,
        dict(v6.get("v6_readiness_implementation") or {}),
        expected_path=V6_IMPLEMENTATION_REPOSITORY_PATH,
    )


def assess_micro_live_preapproval_v7(
    *,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    repository_root: Path | str,
    created_at: str,
) -> dict[str, Any]:
    """Assess strict technical readiness without authorizing execution."""

    root = Path(repository_root).resolve()
    validate_release_readiness_contract_v7(contract, repository_root=root)
    extra = sorted(set(evidence) - _EXPECTED_EVIDENCE_NAMES)
    if extra:
        raise ReleaseReadinessV7Error(f"unexpected v7 preapproval evidence: {extra}")
    v6 = _verified_repository_json(root, V6_CONTRACT_REPOSITORY_PATH)
    base_evidence = {key: value for key, value in evidence.items() if key != "security_review"}
    base = assess_micro_live_preapproval_v6(
        contract=v6,
        evidence=base_evidence,
        repository_root=root,
        created_at=created_at,
    )
    security_passed = False
    if "security_review" in evidence:
        protocol = _verified_repository_json(root, PROTOCOL_REPOSITORY_PATH)
        validate_independent_security_review_report_v3(
            evidence["security_review"],
            protocol=protocol,
            repository_root=root,
        )
        security_passed = True
    checks = dict(base["technical_checks"])
    checks["security_review"] = security_passed
    failed = sorted(name for name, passed in checks.items() if not passed)
    technical_passed = not failed
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "preapproval_contract_sha256": canonical_json_sha256(dict(contract)),
        "security_review_protocol": dict(contract["security_review_protocol"]),
        "evidence_sha256": {
            name: canonical_json_sha256(dict(value))
            for name, value in sorted(evidence.items())
        },
        "technical_checks": checks,
        "failed_or_missing_checks": failed,
        "security_review_independent_exact_head_and_executor_bound": security_passed,
        "security_review_independent_and_exact_head": security_passed,
        "phase6_zero_capital_pipeline_passed": checks["phase6_zero_capital_pipeline"],
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


def run_micro_live_preapproval_assessment_v7(
    *,
    repository_root: Path | str,
    contract_path: Path | str,
    expected_contract_sha256: str,
    evidence_root: Path | str,
    evidence_descriptors: Mapping[str, Mapping[str, str]],
    output_path: Path | str,
    created_at: str,
) -> dict[str, Any]:
    """Load exact evidence and write one non-authorizing v7 assessment."""

    root = Path(repository_root).resolve()
    contract_file = _repository_file(root, contract_path)
    if sha256_file(contract_file) != expected_contract_sha256:
        raise ReleaseReadinessV7Error("v7 preapproval contract SHA-256 mismatch")
    contract = _verified_json_with_sidecar(contract_file)
    validate_release_readiness_contract_v7(contract, repository_root=root)
    if sorted(set(evidence_descriptors) - _EXPECTED_EVIDENCE_NAMES):
        raise ReleaseReadinessV7Error("unexpected v7 evidence descriptor")
    evidence_base = Path(evidence_root).resolve()
    loaded: dict[str, Mapping[str, Any]] = {}
    normalized: dict[str, dict[str, str]] = {}
    for name, descriptor_value in sorted(evidence_descriptors.items()):
        descriptor = dict(descriptor_value)
        if set(descriptor) != {"path", "sha256"} or not _is_sha256(
            descriptor.get("sha256")
        ):
            raise ReleaseReadinessV7Error("v7 future evidence descriptor is invalid")
        path_value = descriptor.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ReleaseReadinessV7Error("v7 future evidence path is invalid")
        path = (evidence_base / path_value).resolve()
        if (
            not path.is_relative_to(evidence_base)
            or not path.is_file()
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ReleaseReadinessV7Error("v7 future evidence SHA-256 mismatch or path escape")
        loaded[name] = _load_json(path)
        normalized[name] = descriptor
    if "evaluation_manifest" in loaded or "evaluation_report" in loaded:
        manifest = dict(loaded.get("evaluation_manifest") or {})
        expected_report = dict(manifest.get("evaluation_report") or {})
        supplied_report = dict(evidence_descriptors.get("evaluation_report") or {})
        if expected_report.get("sha256") != supplied_report.get("sha256"):
            raise ReleaseReadinessV7Error(
                "evaluation manifest/report SHA-256 binding mismatch"
            )
    report = assess_micro_live_preapproval_v7(
        contract=contract,
        evidence=loaded,
        repository_root=root,
        created_at=created_at,
    )
    report["evidence_file_descriptors"] = normalized
    output = Path(output_path).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError("v7 preapproval assessment already exists; rerun forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_frozen_json(output, report)
    return report


def _expected_security_gate(root: Path) -> dict[str, Any]:
    return {
        "schema_version": SECURITY_REPORT_SCHEMA_VERSION,
        "protocol": _repository_descriptor(root, PROTOCOL_REPOSITORY_PATH),
        "independent_github_approval_required": True,
        "exact_reviewed_commit_required": True,
        "hash_bound_scope_and_control_evidence_required": True,
        "actual_executor_control_evidence_required": True,
        "actual_authorization_verifier_evidence_required": True,
        "open_p0_allowed": 0,
        "open_p1_allowed": 0,
        "legacy_v1_or_v2_report_accepted": False,
    }


def _expected_hardening() -> dict[str, bool]:
    return {
        "v6_contract_bytes_preserved": True,
        "all_non_security_gates_preserved": True,
        "actual_executor_control_coverage_required": True,
        "actual_authorization_verifier_coverage_required": True,
        "legacy_security_report_rejected": True,
        "candidate_model_or_gate_change": False,
    }


def _verified_repository_json(root: Path, repository_path: str) -> dict[str, Any]:
    return _verified_json_with_sidecar(_repository_file(root, repository_path))


def _verified_json_with_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise ReleaseReadinessV7Error(f"frozen artifact sidecar mismatch: {path.name}")
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseReadinessV7Error("frozen JSON evidence is invalid") from exc
    if not isinstance(value, dict):
        raise ReleaseReadinessV7Error("frozen JSON root must be an object")
    return value


def _repository_file(root: Path, value: Path | str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ReleaseReadinessV7Error("repository path escapes or is missing")
    return resolved


def _repository_descriptor(root: Path, repository_path: str) -> dict[str, str]:
    path = _repository_file(root, repository_path)
    return {"path": repository_path, "sha256": sha256_file(path)}


def _verify_descriptor(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_path: str,
) -> None:
    if set(descriptor) != {"path", "sha256"} or descriptor.get("path") != expected_path:
        raise ReleaseReadinessV7Error("v7 descriptor schema or path mismatch")
    if not _is_sha256(descriptor.get("sha256")):
        raise ReleaseReadinessV7Error("v7 descriptor SHA-256 is invalid")
    if sha256_file(_repository_file(root, expected_path)) != descriptor["sha256"]:
        raise ReleaseReadinessV7Error("v7 descriptor SHA-256 mismatch")


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "AUTHORIZATION_TEMPLATE_REPOSITORY_PATH",
    "AUTHORIZATION_TEMPLATE_SCHEMA_VERSION",
    "CONTRACT_REPOSITORY_PATH",
    "CONTRACT_SCHEMA_VERSION",
    "IMPLEMENTATION_REPOSITORY_PATH",
    "PREFLIGHT_REPOSITORY_PATH",
    "PREFLIGHT_SCHEMA_VERSION",
    "ReleaseReadinessV7Error",
    "assess_micro_live_preapproval_v7",
    "freeze_release_readiness_v7",
    "run_micro_live_preapproval_assessment_v7",
    "validate_release_readiness_contract_v7",
]
