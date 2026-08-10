"""Fail-closed independent-security-review contract for promotion v1.

The security review is deliberately non-authorizing.  A passing report only
becomes one input to the later, separately authorized Phase 6 preapproval
assessment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_v1 import CANDIDATE_ID, LINEAGE_ID

PROTOCOL_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-protocol-v1"
)
REPORT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-report-v2"
)
TEMPLATE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-template-v2"
)
SCOPE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-security-review-scope-v1"
)
ATTESTATION_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-independent-reviewer-attestation-v1"
)
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
CANDIDATE_BUNDLE_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/bundle_manifest.json"
)
PROTOCOL_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/security_review_protocol_v1.json"
TEMPLATE_REPOSITORY_PATH = f"{CONFIG_REPOSITORY_PATH}/security_review_template_v2.json"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_security_review.py"
)

REQUIRED_SCOPE_COMPONENT_PATHS = {
    "candidate_bundle": CANDIDATE_BUNDLE_REPOSITORY_PATH,
    "residual_runtime": "src/bigan/v8/polymarket/pooled_residual_runtime.py",
    "execution_readiness": (
        "src/bigan/v8/polymarket/residual_promotion_execution_readiness.py"
    ),
    "release_readiness_v6": (
        "src/bigan/v8/polymarket/residual_promotion_release_readiness_v6.py"
    ),
    "security_review_validator": IMPLEMENTATION_REPOSITORY_PATH,
    "security_review_protocol": PROTOCOL_REPOSITORY_PATH,
    "phase6_pipeline": "src/bigan/v8/phase6/pipeline.py",
    "phase6_contracts": "src/bigan/v8/phase6/contracts.py",
    "rollback": "src/bigan/v8/polymarket/residual_promotion_rollback.py",
    "separate_micro_live_executor": (
        "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py"
    ),
    "micro_live_authorization_verifier": (
        "src/bigan/v8/polymarket/residual_promotion_micro_live_authorization.py"
    ),
}
REQUIRED_CONTROL_IDS = (
    "artifact_hash_and_candidate_identity",
    "btc_15m_market_allowlist",
    "authorization_separation_and_exact_payload_binding",
    "credential_and_wallet_isolation",
    "write_surface_deny_by_default",
    "idempotent_order_and_business_identity",
    "conflicting_duplicate_fail_closed",
    "order_fill_position_cash_settlement_reconciliation",
    "kill_switch_and_operator_heartbeat",
    "audit_log_integrity_and_restart_recovery",
    "rollback_to_no_trade",
    "one_percent_cap_and_no_automatic_launch",
)
REQUIRED_CONTROL_EVIDENCE_PATHS = {
    "artifact_hash_and_candidate_identity": (
        CANDIDATE_BUNDLE_REPOSITORY_PATH,
        "tests/v8/test_residual_promotion_runtime.py",
    ),
    "btc_15m_market_allowlist": (
        "src/bigan/v8/polymarket/residual_promotion_execution_readiness.py",
        "tests/v8/test_residual_promotion_execution_readiness.py",
    ),
    "authorization_separation_and_exact_payload_binding": (
        "src/bigan/v8/polymarket/residual_promotion_release_readiness_v6.py",
        "tests/v8/test_residual_promotion_security_review.py",
    ),
    "credential_and_wallet_isolation": (
        "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py",
        "tests/v8/test_residual_promotion_micro_live_executor.py",
    ),
    "write_surface_deny_by_default": (
        "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py",
        "tests/v8/test_residual_promotion_micro_live_executor.py",
    ),
    "idempotent_order_and_business_identity": (
        "src/bigan/v8/polymarket/residual_promotion_execution_readiness.py",
        "tests/v8/test_residual_promotion_execution_readiness.py",
    ),
    "conflicting_duplicate_fail_closed": (
        "src/bigan/v8/polymarket/residual_promotion_execution_readiness.py",
        "tests/v8/test_residual_promotion_execution_readiness.py",
    ),
    "order_fill_position_cash_settlement_reconciliation": (
        "src/bigan/v8/polymarket/residual_promotion_execution_readiness.py",
        "tests/v8/test_residual_promotion_execution_readiness.py",
    ),
    "kill_switch_and_operator_heartbeat": (
        "src/bigan/v8/polymarket/residual_promotion_execution_readiness.py",
        "tests/v8/test_residual_promotion_execution_readiness.py",
    ),
    "audit_log_integrity_and_restart_recovery": (
        "src/bigan/v8/polymarket/residual_promotion_execution_readiness.py",
        "tests/v8/test_residual_promotion_execution_readiness.py",
    ),
    "rollback_to_no_trade": (
        "src/bigan/v8/polymarket/residual_promotion_rollback.py",
        "tests/v8/test_residual_promotion_rollback.py",
    ),
    "one_percent_cap_and_no_automatic_launch": (
        "src/bigan/v8/polymarket/residual_promotion_release_readiness_v6.py",
        "tests/v8/test_residual_promotion_security_review.py",
    ),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REVIEW_URL = re.compile(
    r"^https://github\.com/phead198708/BiGan/pull/[1-9][0-9]*"
    r"#pullrequestreview-[1-9][0-9]*$"
)
_RUN_URL = re.compile(
    r"^https://github\.com/phead198708/BiGan/actions/runs/[1-9][0-9]*$"
)


class SecurityReviewError(ValueError):
    """Raised when security-review evidence cannot be trusted."""


def validate_security_review_protocol(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path | str,
    expected_implementation_sha256: str,
) -> None:
    """Validate the immutable security-review protocol and its static graph."""

    root = Path(repository_root).resolve()
    expected_independence = {
        "github_pull_request_review_required": True,
        "github_review_payload_artifact_required": True,
        "reviewer_github_login_required": True,
        "reviewer_must_not_be_an_implementation_author": True,
        "reviewer_attestation_artifact_required": True,
        "reviewer_attestation_schema_version": ATTESTATION_SCHEMA_VERSION,
        "review_state_required": "APPROVED",
        "exact_reviewed_commit_required": True,
        "self_review_forbidden": True,
    }
    expected_findings = {
        "open_p0_allowed": 0,
        "open_p1_allowed": 0,
        "resolved_p0_p1_requires_resolution_commit": True,
        "p2_p3_must_be_reported": True,
    }
    expected_ci = {
        "exact_head_ci_required": True,
        "required_conclusion": "SUCCESS",
        "github_actions_run_url_required": True,
    }
    expected_safety = {
        "review_authorizes_phase6": False,
        "review_authorizes_micro_live": False,
        "review_authorizes_wallet_signing": False,
        "review_authorizes_polymarket_writes": False,
        "review_authorizes_capital": False,
        "maximum_future_initial_capital_fraction": 0.01,
    }
    implementation = dict(protocol.get("validator_implementation") or {})
    candidate = dict(protocol.get("candidate_bundle") or {})
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
        and protocol.get("security_review_is_independent_evidence_only") is True
        and protocol.get("automatic_authorization_or_launch") is False
        and protocol.get("safety") == SAFETY
    ):
        raise SecurityReviewError("security review protocol semantics are invalid")
    _verify_descriptor(
        root,
        implementation,
        expected_path=IMPLEMENTATION_REPOSITORY_PATH,
        expected_sha256=expected_implementation_sha256,
    )
    _verify_descriptor(
        root,
        candidate,
        expected_path=CANDIDATE_BUNDLE_REPOSITORY_PATH,
    )


def validate_security_review_template(
    template: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    repository_root: Path | str,
) -> None:
    """Validate that the committed review template is inert and incomplete."""

    root = Path(repository_root).resolve()
    validate_security_review_protocol(
        protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(root / IMPLEMENTATION_REPOSITORY_PATH),
    )
    protocol_descriptor = dict(template.get("security_review_protocol") or {})
    _verify_descriptor(
        root,
        protocol_descriptor,
        expected_path=PROTOCOL_REPOSITORY_PATH,
    )
    controls = dict(template.get("controls") or {})
    if not (
        template.get("schema_version") == TEMPLATE_SCHEMA_VERSION
        and template.get("authorized_report_schema_version") == REPORT_SCHEMA_VERSION
        and template.get("lineage_id") == LINEAGE_ID
        and template.get("candidate_id") == CANDIDATE_ID
        and template.get("reviewed_commit_sha") is None
        and template.get("reviewer") is None
        and template.get("implementation_author_logins") == []
        and template.get("scope_manifest") is None
        and set(controls) == set(REQUIRED_CONTROL_IDS)
        and all(value == {"evidence": [], "notes": None, "status": "NOT_REVIEWED"}
                for value in controls.values())
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
        raise SecurityReviewError("security review template is not inert")
    candidate = dict(template.get("candidate_bundle") or {})
    if candidate != protocol.get("candidate_bundle"):
        raise SecurityReviewError("security review template candidate binding mismatch")


def validate_independent_security_review_report(
    report: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    repository_root: Path | str,
) -> None:
    """Validate a future independent report; any ambiguity fails closed."""

    root = Path(repository_root).resolve()
    validate_security_review_protocol(
        protocol,
        repository_root=root,
        expected_implementation_sha256=sha256_file(root / IMPLEMENTATION_REPOSITORY_PATH),
    )
    expected_top_level = {
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
    if set(report) != expected_top_level:
        raise SecurityReviewError("security review report schema is not exact")
    reviewed_commit = report.get("reviewed_commit_sha")
    if not isinstance(reviewed_commit, str) or _COMMIT.fullmatch(reviewed_commit) is None:
        raise SecurityReviewError("security review exact commit is invalid")
    _verify_descriptor(
        root,
        dict(report.get("security_review_protocol") or {}),
        expected_path=PROTOCOL_REPOSITORY_PATH,
    )
    candidate_sha = dict(protocol.get("candidate_bundle") or {}).get("sha256")
    if not (
        report.get("schema_version") == REPORT_SCHEMA_VERSION
        and report.get("lineage_id") == LINEAGE_ID
        and report.get("candidate_id") == CANDIDATE_ID
        and report.get("candidate_bundle_sha256") == candidate_sha
    ):
        raise SecurityReviewError("security review identity binding is invalid")

    _validate_reviewer(report, root, reviewed_commit)
    _validate_scope(report, root, reviewed_commit, candidate_sha)
    _validate_controls(report, root)
    _validate_findings(report, reviewed_commit)
    _validate_ci(report, reviewed_commit)
    if not (
        report.get("security_review_passed") is True
        and report.get("maximum_initial_capital_fraction") == 0.01
        and report.get("fresh_outcomes_accessed") is False
        and report.get("settlement_accessed") is False
        and report.get("pnl_accessed") is False
        and report.get("explicit_human_approval_recorded") is False
        and report.get("phase6_zero_capital_authorized") is False
        and report.get("micro_live_authorized") is False
        and report.get("live_trading_allowed") is False
        and report.get("wallet_signing_allowed") is False
        and report.get("polymarket_write_allowed") is False
        and report.get("capital_at_risk") is False
        and report.get("safety") == SAFETY
    ):
        raise SecurityReviewError("security review safety or authorization state is invalid")


def _validate_reviewer(report: Mapping[str, Any], root: Path, reviewed_commit: str) -> None:
    reviewer = dict(report.get("reviewer") or {})
    expected_keys = {
        "github_login",
        "review_id",
        "review_url",
        "review_state",
        "reviewed_commit_sha",
        "independent_from_implementation",
        "authored_reviewed_bytes",
        "attestation",
        "github_review_payload",
    }
    authors = report.get("implementation_author_logins")
    if (
        set(reviewer) != expected_keys
        or not isinstance(authors, list)
        or not authors
        or any(not isinstance(value, str) or not value for value in authors)
        or len(set(authors)) != len(authors)
    ):
        raise SecurityReviewError("security review independence metadata is invalid")
    login = reviewer.get("github_login")
    if not (
        isinstance(login, str)
        and login
        and login not in set(authors)
        and isinstance(reviewer.get("review_id"), int)
        and not isinstance(reviewer.get("review_id"), bool)
        and reviewer["review_id"] > 0
        and isinstance(reviewer.get("review_url"), str)
        and _REVIEW_URL.fullmatch(reviewer["review_url"]) is not None
        and reviewer.get("review_state") == "APPROVED"
        and reviewer.get("reviewed_commit_sha") == reviewed_commit
        and reviewer.get("independent_from_implementation") is True
        and reviewer.get("authored_reviewed_bytes") is False
    ):
        raise SecurityReviewError("independent reviewer provenance is invalid")
    attestation_descriptor = dict(reviewer.get("attestation") or {})
    _verify_descriptor(root, attestation_descriptor)
    attestation_path = str(attestation_descriptor.get("path") or "")
    if not (
        attestation_path.startswith(f"{CONFIG_REPOSITORY_PATH}/security_review_attestation_")
        and attestation_path.endswith(".json")
    ):
        raise SecurityReviewError("independent reviewer attestation path is invalid")
    attestation = _load_json(root / attestation_path)
    if not (
        set(attestation)
        == {
            "schema_version",
            "reviewer_github_login",
            "review_id",
            "review_url",
            "reviewed_commit_sha",
            "independent_from_implementation",
            "authored_reviewed_bytes",
            "attestation_statement",
        }
        and attestation.get("schema_version") == ATTESTATION_SCHEMA_VERSION
        and attestation.get("reviewer_github_login") == login
        and attestation.get("review_id") == reviewer.get("review_id")
        and attestation.get("review_url") == reviewer.get("review_url")
        and attestation.get("reviewed_commit_sha") == reviewed_commit
        and attestation.get("independent_from_implementation") is True
        and attestation.get("authored_reviewed_bytes") is False
        and isinstance(attestation.get("attestation_statement"), str)
        and attestation["attestation_statement"].strip()
    ):
        raise SecurityReviewError("independent reviewer attestation content is invalid")
    github_descriptor = dict(reviewer.get("github_review_payload") or {})
    _verify_descriptor(root, github_descriptor)
    github_path = str(github_descriptor.get("path") or "")
    if not (
        github_path.startswith(f"{CONFIG_REPOSITORY_PATH}/security_review_github_payload_")
        and github_path.endswith(".json")
    ):
        raise SecurityReviewError("GitHub review payload path is invalid")
    github_payload = _load_json(root / github_path)
    github_user = dict(github_payload.get("user") or {})
    if not (
        github_payload.get("id") == reviewer.get("review_id")
        and github_payload.get("html_url") == reviewer.get("review_url")
        and github_payload.get("state") == "APPROVED"
        and github_payload.get("commit_id") == reviewed_commit
        and github_user.get("login") == login
        and isinstance(github_payload.get("submitted_at"), str)
        and github_payload["submitted_at"].strip()
    ):
        raise SecurityReviewError("GitHub review payload provenance is invalid")


def _validate_scope(
    report: Mapping[str, Any],
    root: Path,
    reviewed_commit: str,
    candidate_sha: Any,
) -> None:
    scope = dict(report.get("scope_manifest") or {})
    if set(scope) != {"schema_version", "reviewed_commit_sha", "components"}:
        raise SecurityReviewError("security review scope schema is invalid")
    components = scope.get("components")
    if (
        scope.get("schema_version") != SCOPE_SCHEMA_VERSION
        or scope.get("reviewed_commit_sha") != reviewed_commit
        or not isinstance(components, list)
    ):
        raise SecurityReviewError("security review scope binding is invalid")
    by_id: dict[str, Mapping[str, Any]] = {}
    for value in components:
        if not isinstance(value, Mapping) or set(value) != {"component_id", "path", "sha256"}:
            raise SecurityReviewError("security review scope component is invalid")
        component_id = value.get("component_id")
        if not isinstance(component_id, str) or component_id in by_id:
            raise SecurityReviewError("security review scope component is duplicated")
        by_id[component_id] = value
    if set(by_id) != set(REQUIRED_SCOPE_COMPONENT_PATHS):
        raise SecurityReviewError("security review scope is incomplete or expanded")
    for component_id, expected_path in REQUIRED_SCOPE_COMPONENT_PATHS.items():
        descriptor = dict(by_id[component_id])
        _verify_descriptor(root, descriptor, expected_path=expected_path)
    if by_id["candidate_bundle"].get("sha256") != candidate_sha:
        raise SecurityReviewError("security review candidate scope SHA-256 mismatch")


def _validate_controls(report: Mapping[str, Any], root: Path) -> None:
    controls = dict(report.get("controls") or {})
    if set(controls) != set(REQUIRED_CONTROL_IDS):
        raise SecurityReviewError("security review control set is incomplete or expanded")
    for control_id in REQUIRED_CONTROL_IDS:
        control = dict(controls[control_id] or {})
        evidence = control.get("evidence")
        if (
            set(control) != {"status", "evidence", "notes"}
            or control.get("status") != "PASS"
            or not isinstance(control.get("notes"), str)
            or not control["notes"].strip()
            or not isinstance(evidence, list)
            or not evidence
        ):
            raise SecurityReviewError(f"security review control is not proven: {control_id}")
        expected_paths = set(REQUIRED_CONTROL_EVIDENCE_PATHS[control_id])
        actual_paths = {
            descriptor.get("path")
            for descriptor in evidence
            if isinstance(descriptor, Mapping)
        }
        if actual_paths != expected_paths or len(evidence) != len(expected_paths):
            raise SecurityReviewError(
                f"security review control evidence scope mismatch: {control_id}"
            )
        for descriptor in evidence:
            if not isinstance(descriptor, Mapping):
                raise SecurityReviewError("security control evidence descriptor is invalid")
            _verify_descriptor(root, dict(descriptor))


def _validate_findings(report: Mapping[str, Any], reviewed_commit: str) -> None:
    findings = dict(report.get("findings") or {})
    if set(findings) != {"open_counts", "items"}:
        raise SecurityReviewError("security findings schema is invalid")
    counts = findings.get("open_counts")
    items = findings.get("items")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != {"P0", "P1", "P2", "P3"}
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values())
        or counts["P0"] != 0
        or counts["P1"] != 0
        or not isinstance(items, list)
    ):
        raise SecurityReviewError("security findings counts are invalid")
    ids: set[str] = set()
    observed_open = dict.fromkeys(("P0", "P1", "P2", "P3"), 0)
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {
            "finding_id",
            "severity",
            "status",
            "title",
            "resolution_commit_sha",
        }:
            raise SecurityReviewError("security finding item schema is invalid")
        finding_id = item.get("finding_id")
        severity = item.get("severity")
        status = item.get("status")
        if (
            not isinstance(finding_id, str)
            or not finding_id
            or finding_id in ids
            or severity not in observed_open
            or status not in {"OPEN", "RESOLVED"}
            or not isinstance(item.get("title"), str)
            or not item["title"].strip()
        ):
            raise SecurityReviewError("security finding item is invalid")
        ids.add(finding_id)
        if status == "OPEN":
            observed_open[str(severity)] += 1
            if item.get("resolution_commit_sha") is not None:
                raise SecurityReviewError("open security finding cannot have a resolution commit")
        elif severity in {"P0", "P1"}:
            resolution = item.get("resolution_commit_sha")
            if not isinstance(resolution, str) or _COMMIT.fullmatch(resolution) is None:
                raise SecurityReviewError("resolved P0/P1 lacks an exact resolution commit")
        elif item.get("resolution_commit_sha") not in {None, reviewed_commit}:
            resolution = item.get("resolution_commit_sha")
            if not isinstance(resolution, str) or _COMMIT.fullmatch(resolution) is None:
                raise SecurityReviewError("security finding resolution commit is invalid")
    if dict(counts) != observed_open:
        raise SecurityReviewError("security findings counts do not reconcile")


def _validate_ci(report: Mapping[str, Any], reviewed_commit: str) -> None:
    ci = dict(report.get("ci") or {})
    if not (
        set(ci) == {"run_url", "conclusion", "exact_head_sha"}
        and isinstance(ci.get("run_url"), str)
        and _RUN_URL.fullmatch(ci["run_url"]) is not None
        and ci.get("conclusion") == "SUCCESS"
        and ci.get("exact_head_sha") == reviewed_commit
    ):
        raise SecurityReviewError("security review exact-head CI evidence is invalid")


def _verify_descriptor(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_path: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    if set(descriptor) not in ({"path", "sha256"}, {"component_id", "path", "sha256"}):
        raise SecurityReviewError("security review descriptor schema is invalid")
    path_value = descriptor.get("path")
    digest = descriptor.get("sha256")
    if not isinstance(path_value, str) or not path_value or not _is_sha256(digest):
        raise SecurityReviewError("security review descriptor value is invalid")
    if expected_path is not None and path_value != expected_path:
        raise SecurityReviewError("security review descriptor path mismatch")
    path = (root / path_value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SecurityReviewError("security review descriptor escapes or is missing")
    actual = sha256_file(path)
    if actual != digest or (expected_sha256 is not None and actual != expected_sha256):
        raise SecurityReviewError("security review descriptor SHA-256 mismatch")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SecurityReviewError("security review JSON evidence is invalid") from exc
    if not isinstance(value, dict):
        raise SecurityReviewError("security review JSON evidence root must be an object")
    return value


__all__ = [
    "CANDIDATE_BUNDLE_REPOSITORY_PATH",
    "ATTESTATION_SCHEMA_VERSION",
    "CONFIG_REPOSITORY_PATH",
    "IMPLEMENTATION_REPOSITORY_PATH",
    "PROTOCOL_REPOSITORY_PATH",
    "PROTOCOL_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "REQUIRED_CONTROL_EVIDENCE_PATHS",
    "REQUIRED_CONTROL_IDS",
    "REQUIRED_SCOPE_COMPONENT_PATHS",
    "SCOPE_SCHEMA_VERSION",
    "SecurityReviewError",
    "TEMPLATE_REPOSITORY_PATH",
    "TEMPLATE_SCHEMA_VERSION",
    "validate_independent_security_review_report",
    "validate_security_review_protocol",
    "validate_security_review_template",
]
