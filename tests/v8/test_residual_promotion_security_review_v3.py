"""Production-gateway successor security scope tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_security_review_v3 import (
    GATEWAY_MANIFEST_REPOSITORY_PATH,
    GATEWAY_REPOSITORY_PATH,
    GATEWAY_TEST_REPOSITORY_PATH,
    IMPLEMENTATION_REPOSITORY_PATH,
    PROTOCOL_REPOSITORY_PATH,
    REQUIRED_CONTROL_EVIDENCE_PATHS,
    TEMPLATE_REPOSITORY_PATH,
    V2_PROTOCOL_REPOSITORY_PATH,
    SecurityReviewV3Error,
    validate_security_review_protocol_v3,
    validate_security_review_template_v4,
)

ROOT = Path(__file__).resolve().parents[2]


def _json(path: str) -> dict:
    value = json.loads((ROOT / path).read_bytes())
    assert isinstance(value, dict)
    return value


def _assert_sidecar(path: str) -> None:
    selected = ROOT / path
    sidecar = selected.with_suffix(selected.suffix + ".sha256")
    parts = sidecar.read_text(encoding="ascii").strip().split()
    expected = parts[0]
    if len(parts) > 1:
        assert parts[1] == selected.name
    assert expected == hashlib.sha256(selected.read_bytes()).hexdigest()


def test_v3_protocol_and_v4_template_validate_without_unlocking_safety() -> None:
    protocol = _json(PROTOCOL_REPOSITORY_PATH)
    template = _json(TEMPLATE_REPOSITORY_PATH)
    validate_security_review_protocol_v3(
        protocol,
        repository_root=ROOT,
        expected_implementation_sha256=sha256_file(
            ROOT / IMPLEMENTATION_REPOSITORY_PATH
        ),
    )
    validate_security_review_template_v4(
        template,
        protocol=protocol,
        repository_root=ROOT,
    )
    assert template["safety"] == SAFETY
    assert all(value is None for value in template["required_evidence"].values())
    assert template["security_review_complete"] is False
    assert template["micro_live_authorized"] is False
    for path in (
        PROTOCOL_REPOSITORY_PATH,
        TEMPLATE_REPOSITORY_PATH,
        GATEWAY_MANIFEST_REPOSITORY_PATH,
    ):
        _assert_sidecar(path)


def test_v2_bytes_are_preserved_and_gateway_scope_is_additive() -> None:
    _assert_sidecar(V2_PROTOCOL_REPOSITORY_PATH)
    for control_id in (
        "authorization_separation_and_exact_payload_binding",
        "credential_and_wallet_isolation",
        "write_surface_deny_by_default",
        "idempotent_order_and_business_identity",
        "audit_log_integrity_and_restart_recovery",
    ):
        evidence = set(REQUIRED_CONTROL_EVIDENCE_PATHS[control_id])
        assert GATEWAY_REPOSITORY_PATH in evidence
        assert GATEWAY_TEST_REPOSITORY_PATH in evidence
        assert GATEWAY_MANIFEST_REPOSITORY_PATH in evidence


def test_gateway_manifest_hashes_exact_production_graph() -> None:
    manifest = _json(GATEWAY_MANIFEST_REPOSITORY_PATH)
    assert manifest["security_state"] == "NOT_INDEPENDENTLY_REVIEWED_NO_GO"
    assert manifest["safety"] == SAFETY
    assert manifest["model_or_gate_bytes_changed"] is False
    assert manifest["production_entrypoint"].endswith(
        ".run_production_execution_gateway"
    )
    for descriptor in (
        manifest["gateway_implementation"],
        manifest["gateway_process_integration_test"],
        manifest["execution_gateway_runtime_lock"],
        manifest["frozen_model_runtime_lock"],
    ):
        assert sha256_file(ROOT / descriptor["path"]) == descriptor["sha256"]
    assert manifest["credential_ownership_contract"] == {
        "api_credentials_owner": "gateway_process_only",
        "credential_file_mode": "0600",
        "credential_material_over_rpc": False,
        "private_key_owner": "gateway_process_only",
        "receipt_signing_key_owner": "gateway_process_only",
    }


def test_gateway_scope_or_descriptor_drift_fails_closed() -> None:
    protocol = _json(PROTOCOL_REPOSITORY_PATH)
    changed = copy.deepcopy(protocol)
    changed["required_control_evidence_paths"][
        "credential_and_wallet_isolation"
    ].remove(GATEWAY_REPOSITORY_PATH)
    with pytest.raises(SecurityReviewV3Error, match="semantics are invalid"):
        validate_security_review_protocol_v3(
            changed,
            repository_root=ROOT,
            expected_implementation_sha256=sha256_file(
                ROOT / IMPLEMENTATION_REPOSITORY_PATH
            ),
        )

    changed = copy.deepcopy(protocol)
    changed["execution_gateway_service_manifest"]["sha256"] = "0" * 64
    with pytest.raises(SecurityReviewV3Error, match="descriptor SHA-256 mismatch"):
        validate_security_review_protocol_v3(
            changed,
            repository_root=ROOT,
            expected_implementation_sha256=sha256_file(
                ROOT / IMPLEMENTATION_REPOSITORY_PATH
            ),
        )
