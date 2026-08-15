"""Additive security-review scope for the production execution gateway.

The frozen v2 protocol and validator remain immutable historical evidence.
This successor requires independent review of the concrete server, backend,
venue/signing boundary, durable registry, and exact process integration test.
It authorizes nothing and cannot make a micro-live template executable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket import residual_promotion_security_review_v2 as v2
from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_v1 import CANDIDATE_ID, LINEAGE_ID

PROTOCOL_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-protocol-v3"
)
TEMPLATE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-template-v4"
)
SCOPE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-scope-v3"
)
CONFIG_REPOSITORY_PATH = v2.CONFIG_REPOSITORY_PATH
V2_PROTOCOL_REPOSITORY_PATH = v2.PROTOCOL_REPOSITORY_PATH
PROTOCOL_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/security_review_protocol_v3.json"
TEMPLATE_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/security_review_template_v4.json"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_security_review_v3.py"
)
GATEWAY_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_execution_gateway.py"
)
GATEWAY_TEST_REPOSITORY_PATH = (
    "tests/v8/test_residual_promotion_micro_live_executor.py"
)
GATEWAY_MANIFEST_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/execution_gateway_service_manifest_v1.json"
)

REQUIRED_SCOPE_COMPONENT_PATHS = {
    **v2.REQUIRED_SCOPE_COMPONENT_PATHS,
    "security_review_protocol_v3": PROTOCOL_REPOSITORY_PATH,
    "security_review_validator_v3": IMPLEMENTATION_REPOSITORY_PATH,
    "production_execution_gateway": GATEWAY_REPOSITORY_PATH,
    "production_execution_gateway_manifest": GATEWAY_MANIFEST_REPOSITORY_PATH,
}

_GATEWAY_CONTROL_IDS = {
    "authorization_separation_and_exact_payload_binding",
    "credential_and_wallet_isolation",
    "write_surface_deny_by_default",
    "idempotent_order_and_business_identity",
    "conflicting_duplicate_fail_closed",
    "order_fill_position_cash_settlement_reconciliation",
    "kill_switch_and_operator_heartbeat",
    "audit_log_integrity_and_restart_recovery",
    "one_percent_cap_and_no_automatic_launch",
    v2.CAPABILITY_CONTROL_ID,
}


def _paths(control_id: str) -> tuple[str, ...]:
    old = list(v2.REQUIRED_CONTROL_EVIDENCE_PATHS[control_id])
    if control_id in _GATEWAY_CONTROL_IDS:
        old.extend(
            (
                GATEWAY_REPOSITORY_PATH,
                GATEWAY_TEST_REPOSITORY_PATH,
                GATEWAY_MANIFEST_REPOSITORY_PATH,
            )
        )
    return tuple(dict.fromkeys(old))


REQUIRED_CONTROL_IDS = tuple(v2.REQUIRED_CONTROL_IDS)
REQUIRED_CONTROL_EVIDENCE_PATHS = {
    control_id: _paths(control_id) for control_id in REQUIRED_CONTROL_IDS
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SecurityReviewV3Error(v2.SecurityReviewV2Error):
    """Raised when the production-gateway review scope is incomplete."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise SecurityReviewV3Error(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise SecurityReviewV3Error(f"JSON artifact is not an object: {path}")
    return value


def _descriptor(
    root: Path,
    value: Any,
    *,
    expected_path: str,
    expected_sha256: str | None = None,
) -> None:
    if not (
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256"}
        and value.get("path") == expected_path
        and isinstance(value.get("sha256"), str)
        and _SHA256.fullmatch(str(value["sha256"]))
    ):
        raise SecurityReviewV3Error(f"descriptor is invalid: {expected_path}")
    actual = sha256_file(root / expected_path)
    if value["sha256"] != actual or (
        expected_sha256 is not None and actual != expected_sha256
    ):
        raise SecurityReviewV3Error(f"descriptor SHA-256 mismatch: {expected_path}")


def validate_security_review_protocol_v3(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path | str,
    expected_implementation_sha256: str,
) -> None:
    """Prove exact v2 preservation and strict production-gateway addition."""

    root = Path(repository_root).resolve()
    old_protocol = _json(root / V2_PROTOCOL_REPOSITORY_PATH)
    v2.validate_security_review_protocol_v2(
        old_protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(
            root / v2.IMPLEMENTATION_REPOSITORY_PATH
        ),
    )
    expected_successor = {
        "v2_protocol_bytes_preserved": True,
        "v2_scope_and_controls_preserved": True,
        "concrete_gateway_server_required": True,
        "concrete_gateway_backend_required": True,
        "concrete_polymarket_venue_boundary_required": True,
        "credential_owned_receipt_signer_required": True,
        "authenticated_session_registry_required": True,
        "strict_raw_json_parser_required": True,
        "durable_restart_state_required": True,
        "exact_gateway_implementation_config_image_evidence_required": True,
        "process_test_mocks_only_outer_venue_boundary": True,
        "legacy_v2_report_sufficient": False,
        "candidate_model_or_gate_change": False,
    }
    if not (
        protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION
        and protocol.get("scope_schema_version") == SCOPE_SCHEMA_VERSION
        and protocol.get("lineage_id") == LINEAGE_ID
        and protocol.get("candidate_id") == CANDIDATE_ID
        and protocol.get("required_scope_component_paths")
        == REQUIRED_SCOPE_COMPONENT_PATHS
        and protocol.get("required_control_ids") == list(REQUIRED_CONTROL_IDS)
        and protocol.get("required_control_evidence_paths")
        == {
            key: list(value)
            for key, value in REQUIRED_CONTROL_EVIDENCE_PATHS.items()
        }
        and protocol.get("review_independence_contract")
        == old_protocol["review_independence_contract"]
        and protocol.get("findings_contract")
        == old_protocol["findings_contract"]
        and protocol.get("ci_contract") == old_protocol["ci_contract"]
        and protocol.get("execution_safety_contract")
        == old_protocol["execution_safety_contract"]
        and protocol.get("strict_successor_contract") == expected_successor
        and protocol.get("security_review_is_independent_evidence_only") is True
        and protocol.get("automatic_authorization_or_launch") is False
        and protocol.get("safety") == SAFETY
    ):
        raise SecurityReviewV3Error("security review v3 protocol semantics are invalid")
    _descriptor(
        root,
        protocol.get("supersedes_security_review_protocol"),
        expected_path=V2_PROTOCOL_REPOSITORY_PATH,
    )
    _descriptor(
        root,
        protocol.get("validator_implementation"),
        expected_path=IMPLEMENTATION_REPOSITORY_PATH,
        expected_sha256=expected_implementation_sha256,
    )
    _descriptor(
        root,
        protocol.get("execution_gateway_service_manifest"),
        expected_path=GATEWAY_MANIFEST_REPOSITORY_PATH,
    )
    if tuple(REQUIRED_CONTROL_IDS) != tuple(v2.REQUIRED_CONTROL_IDS):
        raise SecurityReviewV3Error("v2 control order was not preserved")
    for component, path in v2.REQUIRED_SCOPE_COMPONENT_PATHS.items():
        if REQUIRED_SCOPE_COMPONENT_PATHS.get(component) != path:
            raise SecurityReviewV3Error("v2 scope was not preserved")
    for control_id, paths in v2.REQUIRED_CONTROL_EVIDENCE_PATHS.items():
        if not set(paths).issubset(REQUIRED_CONTROL_EVIDENCE_PATHS[control_id]):
            raise SecurityReviewV3Error("v2 control evidence was not preserved")
    for control_id in _GATEWAY_CONTROL_IDS:
        evidence = set(REQUIRED_CONTROL_EVIDENCE_PATHS[control_id])
        if not {
            GATEWAY_REPOSITORY_PATH,
            GATEWAY_TEST_REPOSITORY_PATH,
            GATEWAY_MANIFEST_REPOSITORY_PATH,
        }.issubset(evidence):
            raise SecurityReviewV3Error(
                f"production gateway evidence is absent: {control_id}"
            )


def validate_security_review_template_v4(
    template: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    repository_root: Path | str,
) -> None:
    """Require unfilled independent evidence and preserve every safety lock."""

    root = Path(repository_root).resolve()
    expected_evidence = {
        "exact_reviewed_commit_sha": None,
        "exact_head_ci_run_url": None,
        "independent_reviewer_github_login": None,
        "independent_review_payload_sha256": None,
        "gateway_service_configuration_sha256": None,
        "gateway_deployment_image_manifest_digest": None,
        "gateway_credential_ownership_attestation_sha256": None,
        "gateway_process_integration_test_sha256": None,
    }
    if not (
        template.get("schema_version") == TEMPLATE_SCHEMA_VERSION
        and template.get("lineage_id") == LINEAGE_ID
        and template.get("candidate_id") == CANDIDATE_ID
        and template.get("required_evidence") == expected_evidence
        and template.get("review_state") == "NOT_REVIEWED"
        and template.get("security_review_complete") is False
        and template.get("micro_live_authorized") is False
        and template.get("polymarket_write_allowed") is False
        and template.get("wallet_signing_allowed") is False
        and template.get("capital_at_risk") is False
        and template.get("automatic_authorization_or_launch") is False
        and template.get("safety") == SAFETY
    ):
        raise SecurityReviewV3Error("security review v4 template semantics are invalid")
    _descriptor(
        root,
        template.get("protocol"),
        expected_path=PROTOCOL_REPOSITORY_PATH,
        expected_sha256=sha256_file(root / PROTOCOL_REPOSITORY_PATH),
    )
    validate_security_review_protocol_v3(
        protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(
            root / IMPLEMENTATION_REPOSITORY_PATH
        ),
    )
