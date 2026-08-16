"""Strict successor security review bound to the executable micro-live path.

Version 1 remains immutable historical governance.  This module preserves all
v1 reviewer, exact-head CI, findings, and safety requirements while requiring
the independent reviewer to inspect the actual capability verifier and
micro-live executor for every execution-critical control.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket import residual_promotion_security_review as v1
from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_v1 import CANDIDATE_ID, LINEAGE_ID

PROTOCOL_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-protocol-v2"
)
REPORT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-report-v3"
)
TEMPLATE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-template-v3"
)
SCOPE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-scope-v2"
)
CONFIG_REPOSITORY_PATH = v1.CONFIG_REPOSITORY_PATH
CANDIDATE_BUNDLE_REPOSITORY_PATH = v1.CANDIDATE_BUNDLE_REPOSITORY_PATH
V1_PROTOCOL_REPOSITORY_PATH = v1.PROTOCOL_REPOSITORY_PATH
PROTOCOL_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/security_review_protocol_v2.json"
TEMPLATE_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/security_review_template_v3.json"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_security_review_v2.py"
)
RELEASE_READINESS_V7_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_release_readiness_v7.py"
)
EXECUTOR_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py"
)
AUTHORIZATION_VERIFIER_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_micro_live_authorization.py"
)
EXECUTOR_TEST_REPOSITORY_PATH = (
    "tests/v8/test_residual_promotion_micro_live_executor.py"
)
SECURITY_V2_TEST_REPOSITORY_PATH = (
    "tests/v8/test_residual_promotion_security_review_v2.py"
)

REQUIRED_SCOPE_COMPONENT_PATHS = {
    **v1.REQUIRED_SCOPE_COMPONENT_PATHS,
    "release_readiness_v7": RELEASE_READINESS_V7_REPOSITORY_PATH,
    "security_review_protocol_v2": PROTOCOL_REPOSITORY_PATH,
    "security_review_validator_v2": IMPLEMENTATION_REPOSITORY_PATH,
}

CAPABILITY_CONTROL_ID = "capability_gate_and_frozen_runtime_signal_binding"
REQUIRED_CONTROL_IDS = (*v1.REQUIRED_CONTROL_IDS, CAPABILITY_CONTROL_ID)


def _additive_paths(control_id: str, *paths: str) -> tuple[str, ...]:
    ordered = [*v1.REQUIRED_CONTROL_EVIDENCE_PATHS[control_id], *paths]
    return tuple(dict.fromkeys(ordered))


REQUIRED_CONTROL_EVIDENCE_PATHS = {
    **v1.REQUIRED_CONTROL_EVIDENCE_PATHS,
    "btc_15m_market_allowlist": _additive_paths(
        "btc_15m_market_allowlist",
        EXECUTOR_REPOSITORY_PATH,
        EXECUTOR_TEST_REPOSITORY_PATH,
    ),
    "authorization_separation_and_exact_payload_binding": _additive_paths(
        "authorization_separation_and_exact_payload_binding",
        RELEASE_READINESS_V7_REPOSITORY_PATH,
        AUTHORIZATION_VERIFIER_REPOSITORY_PATH,
        EXECUTOR_REPOSITORY_PATH,
        EXECUTOR_TEST_REPOSITORY_PATH,
        SECURITY_V2_TEST_REPOSITORY_PATH,
    ),
    "idempotent_order_and_business_identity": _additive_paths(
        "idempotent_order_and_business_identity",
        EXECUTOR_REPOSITORY_PATH,
        EXECUTOR_TEST_REPOSITORY_PATH,
    ),
    "conflicting_duplicate_fail_closed": _additive_paths(
        "conflicting_duplicate_fail_closed",
        EXECUTOR_REPOSITORY_PATH,
        EXECUTOR_TEST_REPOSITORY_PATH,
    ),
    "order_fill_position_cash_settlement_reconciliation": _additive_paths(
        "order_fill_position_cash_settlement_reconciliation",
        EXECUTOR_REPOSITORY_PATH,
        EXECUTOR_TEST_REPOSITORY_PATH,
    ),
    "kill_switch_and_operator_heartbeat": _additive_paths(
        "kill_switch_and_operator_heartbeat",
        EXECUTOR_REPOSITORY_PATH,
        EXECUTOR_TEST_REPOSITORY_PATH,
    ),
    "audit_log_integrity_and_restart_recovery": _additive_paths(
        "audit_log_integrity_and_restart_recovery",
        EXECUTOR_REPOSITORY_PATH,
        EXECUTOR_TEST_REPOSITORY_PATH,
    ),
    "one_percent_cap_and_no_automatic_launch": _additive_paths(
        "one_percent_cap_and_no_automatic_launch",
        RELEASE_READINESS_V7_REPOSITORY_PATH,
        AUTHORIZATION_VERIFIER_REPOSITORY_PATH,
        EXECUTOR_REPOSITORY_PATH,
        EXECUTOR_TEST_REPOSITORY_PATH,
        SECURITY_V2_TEST_REPOSITORY_PATH,
    ),
    CAPABILITY_CONTROL_ID: (
        CANDIDATE_BUNDLE_REPOSITORY_PATH,
        AUTHORIZATION_VERIFIER_REPOSITORY_PATH,
        EXECUTOR_REPOSITORY_PATH,
        "tests/v8/test_residual_promotion_runtime.py",
        EXECUTOR_TEST_REPOSITORY_PATH,
    ),
}

_EXECUTION_CRITICAL_CONTROL_IDS = (
    "btc_15m_market_allowlist",
    "authorization_separation_and_exact_payload_binding",
    "credential_and_wallet_isolation",
    "write_surface_deny_by_default",
    "idempotent_order_and_business_identity",
    "conflicting_duplicate_fail_closed",
    "order_fill_position_cash_settlement_reconciliation",
    "kill_switch_and_operator_heartbeat",
    "audit_log_integrity_and_restart_recovery",
    "one_percent_cap_and_no_automatic_launch",
    CAPABILITY_CONTROL_ID,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SecurityReviewV2Error(v1.SecurityReviewError):
    """Raised when successor security evidence is incomplete or ambiguous."""


def validate_security_review_protocol_v2(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path | str,
    expected_implementation_sha256: str,
) -> None:
    """Validate v2 and prove it is a strict additive successor to v1."""

    root = Path(repository_root).resolve()
    old_protocol = _verified_repository_json(root, V1_PROTOCOL_REPOSITORY_PATH)
    v1.validate_security_review_protocol(
        old_protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(root / v1.IMPLEMENTATION_REPOSITORY_PATH),
    )
    expected_independence = dict(old_protocol["review_independence_contract"])
    expected_findings = dict(old_protocol["findings_contract"])
    expected_ci = dict(old_protocol["ci_contract"])
    expected_safety = dict(old_protocol["execution_safety_contract"])
    expected_hardening = {
        "v1_protocol_bytes_preserved": True,
        "v1_scope_and_controls_preserved": True,
        "actual_executor_required_for_execution_controls": True,
        "actual_authorization_verifier_required": True,
        "frozen_runtime_signal_binding_review_required": True,
        "legacy_v1_report_sufficient": False,
        "candidate_model_or_gate_change": False,
    }
    if not (
        protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION
        and protocol.get("lineage_id") == LINEAGE_ID
        and protocol.get("candidate_id") == CANDIDATE_ID
        and protocol.get("report_schema_version") == REPORT_SCHEMA_VERSION
        and protocol.get("scope_schema_version") == SCOPE_SCHEMA_VERSION
        and protocol.get("required_scope_component_paths")
        == REQUIRED_SCOPE_COMPONENT_PATHS
        and protocol.get("required_control_ids") == list(REQUIRED_CONTROL_IDS)
        and protocol.get("required_control_evidence_paths")
        == {key: list(value) for key, value in REQUIRED_CONTROL_EVIDENCE_PATHS.items()}
        and protocol.get("review_independence_contract") == expected_independence
        and protocol.get("findings_contract") == expected_findings
        and protocol.get("ci_contract") == expected_ci
        and protocol.get("execution_safety_contract") == expected_safety
        and protocol.get("strict_successor_contract") == expected_hardening
        and protocol.get("security_review_is_independent_evidence_only") is True
        and protocol.get("automatic_authorization_or_launch") is False
        and protocol.get("safety") == SAFETY
    ):
        raise SecurityReviewV2Error("security review v2 protocol semantics are invalid")
    _verify_descriptor(
        root,
        dict(protocol.get("supersedes_security_review_protocol") or {}),
        expected_path=V1_PROTOCOL_REPOSITORY_PATH,
    )
    _verify_descriptor(
        root,
        dict(protocol.get("validator_implementation") or {}),
        expected_path=IMPLEMENTATION_REPOSITORY_PATH,
        expected_sha256=expected_implementation_sha256,
    )
    _verify_descriptor(
        root,
        dict(protocol.get("candidate_bundle") or {}),
        expected_path=CANDIDATE_BUNDLE_REPOSITORY_PATH,
    )
    if tuple(REQUIRED_CONTROL_IDS[: len(v1.REQUIRED_CONTROL_IDS)]) != tuple(
        v1.REQUIRED_CONTROL_IDS
    ):
        raise SecurityReviewV2Error("security review v1 control order was not preserved")
    for component_id, path in v1.REQUIRED_SCOPE_COMPONENT_PATHS.items():
        if REQUIRED_SCOPE_COMPONENT_PATHS.get(component_id) != path:
            raise SecurityReviewV2Error("security review v1 scope was not preserved")
    for control_id, paths in v1.REQUIRED_CONTROL_EVIDENCE_PATHS.items():
        if not set(paths).issubset(REQUIRED_CONTROL_EVIDENCE_PATHS[control_id]):
            raise SecurityReviewV2Error("security review v1 evidence was not preserved")
    for control_id in _EXECUTION_CRITICAL_CONTROL_IDS:
        evidence = set(REQUIRED_CONTROL_EVIDENCE_PATHS[control_id])
        if EXECUTOR_REPOSITORY_PATH not in evidence or EXECUTOR_TEST_REPOSITORY_PATH not in evidence:
            raise SecurityReviewV2Error(
                f"actual executor evidence is absent for control: {control_id}"
            )


def validate_security_review_template_v3(
    template: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    repository_root: Path | str,
) -> None:
    """Validate the successor template while proving it remains inert."""

    root = Path(repository_root).resolve()
    validate_security_review_protocol_v2(
        protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(root / IMPLEMENTATION_REPOSITORY_PATH),
    )
    _verify_descriptor(
        root,
        dict(template.get("security_review_protocol") or {}),
        expected_path=PROTOCOL_REPOSITORY_PATH,
    )
    controls = dict(template.get("controls") or {})
    if not (
        template.get("schema_version") == TEMPLATE_SCHEMA_VERSION
        and template.get("authorized_report_schema_version") == REPORT_SCHEMA_VERSION
        and template.get("lineage_id") == LINEAGE_ID
        and template.get("candidate_id") == CANDIDATE_ID
        and template.get("candidate_bundle") == protocol.get("candidate_bundle")
        and template.get("reviewed_commit_sha") is None
        and template.get("reviewer") is None
        and template.get("implementation_author_logins") == []
        and template.get("scope_manifest") is None
        and set(controls) == set(REQUIRED_CONTROL_IDS)
        and all(
            value == {"evidence": [], "notes": None, "status": "NOT_REVIEWED"}
            for value in controls.values()
        )
        and template.get("findings") is None
        and template.get("ci") is None
        and template.get("security_review_passed") is False
        and template.get("explicit_human_approval_recorded") is False
        and template.get("phase6_zero_capital_authorized") is False
        and template.get("micro_live_authorized") is False
        and template.get("live_trading_allowed") is False
        and template.get("wallet_signing_allowed") is False
        and template.get("polymarket_write_allowed") is False
        and template.get("capital_at_risk") is False
        and template.get("template_is_executable") is False
        and template.get("safety") == SAFETY
    ):
        raise SecurityReviewV2Error("security review v3 template is not inert")


def validate_independent_security_review_report_v3(
    report: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    repository_root: Path | str,
) -> None:
    """Validate v3, then independently re-run every inherited v1 check."""

    root = Path(repository_root).resolve()
    validate_security_review_protocol_v2(
        protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(root / IMPLEMENTATION_REPOSITORY_PATH),
    )
    expected_keys = {
        "schema_version",
        "lineage_id",
        "candidate_id",
        "created_at",
        "security_review_protocol",
        "candidate_bundle_sha256",
        "reviewed_commit_sha",
        "reviewer",
        "implementation_author_logins",
        "scope_manifest",
        "controls",
        "findings",
        "ci",
        "security_review_passed",
        "maximum_initial_capital_fraction",
        "fresh_outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
        "explicit_human_approval_recorded",
        "phase6_zero_capital_authorized",
        "micro_live_authorized",
        "live_trading_allowed",
        "wallet_signing_allowed",
        "polymarket_write_allowed",
        "capital_at_risk",
        "safety",
    }
    if set(report) != expected_keys:
        raise SecurityReviewV2Error("security review v3 report schema is not exact")
    _verify_descriptor(
        root,
        dict(report.get("security_review_protocol") or {}),
        expected_path=PROTOCOL_REPOSITORY_PATH,
    )
    if not (
        report.get("schema_version") == REPORT_SCHEMA_VERSION
        and report.get("lineage_id") == LINEAGE_ID
        and report.get("candidate_id") == CANDIDATE_ID
        and report.get("candidate_bundle_sha256")
        == dict(protocol.get("candidate_bundle") or {}).get("sha256")
    ):
        raise SecurityReviewV2Error("security review v3 identity binding is invalid")
    _validate_v2_scope_and_controls(report, root)

    # Reuse the immutable v1 validator for all reviewer, GitHub provenance,
    # exact-head CI, findings, and safety semantics.  Only protocol/scope/control
    # descriptors are projected to the exact v1 subset; no outcome is involved.
    inherited = copy.deepcopy(dict(report))
    inherited["schema_version"] = v1.REPORT_SCHEMA_VERSION
    inherited["security_review_protocol"] = _descriptor(root, V1_PROTOCOL_REPOSITORY_PATH)
    scope = dict(inherited["scope_manifest"])
    scope["schema_version"] = v1.SCOPE_SCHEMA_VERSION
    scope["components"] = [
        component
        for component in scope["components"]
        if component.get("component_id") in v1.REQUIRED_SCOPE_COMPONENT_PATHS
    ]
    inherited["scope_manifest"] = scope
    inherited_controls: dict[str, Any] = {}
    for control_id in v1.REQUIRED_CONTROL_IDS:
        control = dict(inherited["controls"][control_id])
        expected_paths = set(v1.REQUIRED_CONTROL_EVIDENCE_PATHS[control_id])
        control["evidence"] = [
            item for item in control["evidence"] if item.get("path") in expected_paths
        ]
        inherited_controls[control_id] = control
    inherited["controls"] = inherited_controls
    old_protocol = _verified_repository_json(root, V1_PROTOCOL_REPOSITORY_PATH)
    v1.validate_independent_security_review_report(
        inherited,
        protocol=old_protocol,
        repository_root=root,
    )


def _validate_v2_scope_and_controls(report: Mapping[str, Any], root: Path) -> None:
    scope = dict(report.get("scope_manifest") or {})
    components = scope.get("components")
    if not (
        set(scope) == {"schema_version", "reviewed_commit_sha", "components"}
        and scope.get("schema_version") == SCOPE_SCHEMA_VERSION
        and scope.get("reviewed_commit_sha") == report.get("reviewed_commit_sha")
        and isinstance(components, list)
    ):
        raise SecurityReviewV2Error("security review v2 scope binding is invalid")
    by_id: dict[str, Mapping[str, Any]] = {}
    for value in components:
        if not isinstance(value, Mapping) or set(value) != {"component_id", "path", "sha256"}:
            raise SecurityReviewV2Error("security review v2 scope component is invalid")
        component_id = value.get("component_id")
        if not isinstance(component_id, str) or component_id in by_id:
            raise SecurityReviewV2Error("security review v2 scope component is duplicated")
        by_id[component_id] = value
    if set(by_id) != set(REQUIRED_SCOPE_COMPONENT_PATHS):
        raise SecurityReviewV2Error("security review v2 scope is incomplete or expanded")
    for component_id, path in REQUIRED_SCOPE_COMPONENT_PATHS.items():
        _verify_descriptor(root, by_id[component_id], expected_path=path)

    controls = dict(report.get("controls") or {})
    if set(controls) != set(REQUIRED_CONTROL_IDS):
        raise SecurityReviewV2Error("security review v2 control set is incomplete or expanded")
    for control_id in REQUIRED_CONTROL_IDS:
        control = dict(controls[control_id] or {})
        evidence = control.get("evidence")
        if not (
            set(control) == {"status", "evidence", "notes"}
            and control.get("status") == "PASS"
            and isinstance(control.get("notes"), str)
            and control["notes"].strip()
            and isinstance(evidence, list)
        ):
            raise SecurityReviewV2Error(f"security review v2 control is not proven: {control_id}")
        expected_paths = set(REQUIRED_CONTROL_EVIDENCE_PATHS[control_id])
        actual_paths = {
            item.get("path") for item in evidence if isinstance(item, Mapping)
        }
        if actual_paths != expected_paths or len(evidence) != len(expected_paths):
            raise SecurityReviewV2Error(
                f"security review v2 control evidence scope mismatch: {control_id}"
            )
        for descriptor in evidence:
            if not isinstance(descriptor, Mapping):
                raise SecurityReviewV2Error("security review v2 evidence is invalid")
            _verify_descriptor(root, descriptor)


def _verified_repository_json(root: Path, repository_path: str) -> dict[str, Any]:
    path = (root / repository_path).resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise SecurityReviewV2Error("frozen security artifact sidecar mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SecurityReviewV2Error("frozen security artifact must be an object")
    return value


def _descriptor(root: Path, repository_path: str) -> dict[str, str]:
    path = (root / repository_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SecurityReviewV2Error("security review repository path is missing")
    return {"path": repository_path, "sha256": sha256_file(path)}


def _verify_descriptor(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_path: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    if set(descriptor) not in ({"path", "sha256"}, {"component_id", "path", "sha256"}):
        raise SecurityReviewV2Error("security review v2 descriptor schema is invalid")
    path_value = descriptor.get("path")
    digest = descriptor.get("sha256")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or (expected_path is not None and path_value != expected_path)
    ):
        raise SecurityReviewV2Error("security review v2 descriptor value is invalid")
    path = (root / path_value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SecurityReviewV2Error("security review v2 descriptor escapes or is missing")
    actual = sha256_file(path)
    if actual != digest or (expected_sha256 is not None and actual != expected_sha256):
        raise SecurityReviewV2Error("security review v2 descriptor SHA-256 mismatch")


__all__ = [
    "CANDIDATE_BUNDLE_REPOSITORY_PATH",
    "CAPABILITY_CONTROL_ID",
    "CONFIG_REPOSITORY_PATH",
    "IMPLEMENTATION_REPOSITORY_PATH",
    "PROTOCOL_REPOSITORY_PATH",
    "PROTOCOL_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "REQUIRED_CONTROL_EVIDENCE_PATHS",
    "REQUIRED_CONTROL_IDS",
    "REQUIRED_SCOPE_COMPONENT_PATHS",
    "SCOPE_SCHEMA_VERSION",
    "SecurityReviewV2Error",
    "TEMPLATE_REPOSITORY_PATH",
    "TEMPLATE_SCHEMA_VERSION",
    "V1_PROTOCOL_REPOSITORY_PATH",
    "validate_independent_security_review_report_v3",
    "validate_security_review_protocol_v2",
    "validate_security_review_template_v3",
]
