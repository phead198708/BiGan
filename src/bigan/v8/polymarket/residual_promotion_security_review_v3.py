"""Additive security-review scope for the production execution gateway.

The frozen v2 protocol and validator remain immutable historical evidence.
This successor requires independent review of the concrete server, backend,
venue/signing boundary, durable registry, and exact process integration test.
It authorizes nothing and cannot make a micro-live template executable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
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
CLOB_CONTRACT_TEST_REPOSITORY_PATH = (
    "tests/v8/test_residual_promotion_execution_gateway.py"
)
GATEWAY_MANIFEST_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/execution_gateway_service_manifest_v1.json"
)
CLOB_WHEEL_SBOM_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/py_clob_client_v2_wheel_sbom_v1.json"
)

REQUIRED_SCOPE_COMPONENT_PATHS = {
    **v2.REQUIRED_SCOPE_COMPONENT_PATHS,
    "security_review_protocol_v3": PROTOCOL_REPOSITORY_PATH,
    "security_review_validator_v3": IMPLEMENTATION_REPOSITORY_PATH,
    "production_execution_gateway": GATEWAY_REPOSITORY_PATH,
    "production_execution_gateway_manifest": GATEWAY_MANIFEST_REPOSITORY_PATH,
    "py_clob_client_v2_wheel_sbom": CLOB_WHEEL_SBOM_REPOSITORY_PATH,
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
                CLOB_CONTRACT_TEST_REPOSITORY_PATH,
                GATEWAY_MANIFEST_REPOSITORY_PATH,
                CLOB_WHEEL_SBOM_REPOSITORY_PATH,
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


def validate_py_clob_client_v2_wheel_sbom(
    sbom: Mapping[str, Any],
    *,
    repository_root: Path | str,
) -> None:
    """Bind the installed gateway dependency to the exact reviewed wheel bytes."""

    root = Path(repository_root).resolve()
    files = sbom.get("files")
    expected_semantics = {
        "actual_deployed_wheel_bytes_are_authoritative": True,
        "all_wheel_python_sources_are_enumerated": True,
        "installed_byte_drift_fails_closed": True,
        "installer_mutable_record_excluded_from_installed_byte_equality": True,
        "missing_or_extra_python_source_fails_closed": True,
        "reproducible_source_to_wheel_equivalence_claimed": False,
        "upstream_source_commit_is_informational_only": True,
    }
    if not (
        sbom.get("schema_version") == "bigan-py-clob-client-v2-wheel-sbom-v1"
        and sbom.get("distribution_name") == "py-clob-client-v2"
        and sbom.get("distribution_version") == "1.1.0"
        and sbom.get("wheel_filename")
        == "py_clob_client_v2-1.1.0-py3-none-any.whl"
        and sbom.get("wheel_tag") == "py3-none-any"
        and sbom.get("wheel_sha256")
        == "421dc851ffad5028850dc332a7388863fe854f125b30bf3aa9d3333fb8cd1eeb"
        and isinstance(sbom.get("wheel_size_bytes"), int)
        and sbom["wheel_size_bytes"] > 0
        and sbom.get("review_scope_semantics") == expected_semantics
        and isinstance(files, list)
        and sbom.get("wheel_file_count") == len(files)
        and sbom.get("automatic_authorization_or_launch") is False
        and sbom.get("wallet_signing_allowed") is False
        and sbom.get("polymarket_write_allowed") is False
        and sbom.get("capital_at_risk") is False
    ):
        raise SecurityReviewV3Error("py-clob-client-v2 wheel SBOM is invalid")

    normalized: list[dict[str, Any]] = []
    source_paths: set[str] = set()
    seen: set[str] = set()
    for item in files:
        if not (
            isinstance(item, Mapping)
            and set(item) == {"path", "review_scope", "sha256", "size_bytes"}
            and isinstance(item.get("path"), str)
            and PurePosixPath(str(item["path"])).as_posix() == item["path"]
            and not PurePosixPath(str(item["path"])).is_absolute()
            and ".." not in PurePosixPath(str(item["path"])).parts
            and item["path"] not in seen
            and item.get("review_scope")
            in {"reviewed_python_source", "distribution_metadata", "license"}
            and isinstance(item.get("sha256"), str)
            and _SHA256.fullmatch(str(item["sha256"]))
            and isinstance(item.get("size_bytes"), int)
            and not isinstance(item.get("size_bytes"), bool)
            and item["size_bytes"] >= 0
        ):
            raise SecurityReviewV3Error("py-clob-client-v2 wheel file is invalid")
        path = str(item["path"])
        seen.add(path)
        if path.endswith(".py"):
            if item["review_scope"] != "reviewed_python_source":
                raise SecurityReviewV3Error("wheel Python source is outside review")
            source_paths.add(path)
        normalized.append(dict(item))
    if [item["path"] for item in normalized] != sorted(seen):
        raise SecurityReviewV3Error("wheel SBOM paths are not canonical")
    manifest_sha256 = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not (
        sbom.get("wheel_files_manifest_sha256") == manifest_sha256
        and sbom.get("wheel_python_source_count") == len(source_paths)
    ):
        raise SecurityReviewV3Error("wheel SBOM manifest digest is invalid")

    lock = sbom.get("deployment_lock")
    _descriptor(
        root,
        lock,
        expected_path="examples/v8/residual_promotion_execution_gateway-linux-x86_64.lock.txt",
    )
    lock_text = (root / str(lock["path"])).read_text(encoding="ascii")
    if not (
        "py-clob-client-v2==1.1.0 \\\n" in lock_text
        and "--hash=sha256:" + str(sbom["wheel_sha256"]) in lock_text
    ):
        raise SecurityReviewV3Error("wheel hash is absent from deployment lock")

    try:
        distribution = importlib.metadata.distribution("py-clob-client-v2")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SecurityReviewV3Error("py-clob-client-v2 is not installed") from exc
    if distribution.version != sbom["distribution_version"]:
        raise SecurityReviewV3Error("installed py-clob-client-v2 version drifted")
    installed_sources = {
        PurePosixPath(str(path)).as_posix()
        for path in distribution.files or ()
        if str(path).endswith(".py")
        and (
            str(path).startswith("py_clob_client_v2/")
            or str(path).startswith("examples/")
        )
    }
    if installed_sources != source_paths:
        raise SecurityReviewV3Error("installed wheel Python source scope drifted")
    for item in normalized:
        # pip is permitted to rewrite RECORD to add installer-created files;
        # every executable source and all other immutable wheel bytes remain exact.
        if str(item["path"]).endswith(".dist-info/RECORD"):
            continue
        selected = Path(distribution.locate_file(str(item["path"])))
        try:
            raw = selected.read_bytes()
        except OSError as exc:
            raise SecurityReviewV3Error(
                f"installed wheel file is absent: {item['path']}"
            ) from exc
        if not (
            len(raw) == item["size_bytes"]
            and hashlib.sha256(raw).hexdigest() == item["sha256"]
        ):
            raise SecurityReviewV3Error(
                f"installed wheel byte drift: {item['path']}"
            )


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
        "concrete_clob_wire_and_lifecycle_contract_required": True,
        "strict_raw_json_parser_required": True,
        "durable_restart_state_required": True,
        "runtime_identity_derived_from_gateway_venue_api_account_and_signer": True,
        "service_enforced_deadlines_and_independent_safety_lane_required": True,
        "exact_deployed_wheel_source_sbom_required": True,
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
    _descriptor(
        root,
        protocol.get("py_clob_client_v2_wheel_sbom"),
        expected_path=CLOB_WHEEL_SBOM_REPOSITORY_PATH,
    )
    validate_py_clob_client_v2_wheel_sbom(
        _json(root / CLOB_WHEEL_SBOM_REPOSITORY_PATH),
        repository_root=root,
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
            CLOB_CONTRACT_TEST_REPOSITORY_PATH,
            GATEWAY_MANIFEST_REPOSITORY_PATH,
            CLOB_WHEEL_SBOM_REPOSITORY_PATH,
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
