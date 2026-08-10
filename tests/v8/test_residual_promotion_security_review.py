"""Strict independent-security-review and v6 preapproval regression tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_release_readiness_v6 import (
    AUTHORIZATION_TEMPLATE_REPOSITORY_PATH,
    AUTHORIZATION_TEMPLATE_SCHEMA_VERSION,
    CONTRACT_REPOSITORY_PATH,
    CONTRACT_SCHEMA_VERSION,
    PREFLIGHT_REPOSITORY_PATH,
    PREFLIGHT_SCHEMA_VERSION,
    ReleaseReadinessV6Error,
    assess_micro_live_preapproval_v6,
    run_micro_live_preapproval_assessment_v6,
    validate_release_readiness_contract_v6,
)
from bigan.v8.polymarket.residual_promotion_security_review import (
    ATTESTATION_SCHEMA_VERSION,
    CANDIDATE_BUNDLE_REPOSITORY_PATH,
    IMPLEMENTATION_REPOSITORY_PATH,
    PROTOCOL_REPOSITORY_PATH,
    PROTOCOL_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    REQUIRED_CONTROL_EVIDENCE_PATHS,
    REQUIRED_CONTROL_IDS,
    REQUIRED_SCOPE_COMPONENT_PATHS,
    SCOPE_SCHEMA_VERSION,
    TEMPLATE_REPOSITORY_PATH,
    TEMPLATE_SCHEMA_VERSION,
    SecurityReviewError,
    validate_independent_security_review_report,
    validate_security_review_protocol,
    validate_security_review_template,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / CONTRACT_REPOSITORY_PATH
PREFLIGHT = REPO_ROOT / PREFLIGHT_REPOSITORY_PATH
AUTHORIZATION_TEMPLATE = REPO_ROOT / AUTHORIZATION_TEMPLATE_REPOSITORY_PATH
PROTOCOL = REPO_ROOT / PROTOCOL_REPOSITORY_PATH
TEMPLATE = REPO_ROOT / TEMPLATE_REPOSITORY_PATH
V5_CONTRACT = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
    / "micro_live_preapproval_contract_v5.json"
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor(root: Path, repository_path: str) -> dict[str, str]:
    path = root / repository_path
    return {"path": repository_path, "sha256": sha256_file(path)}


def _review_root(tmp_path: Path) -> Path:
    paths = {
        PROTOCOL_REPOSITORY_PATH,
        CANDIDATE_BUNDLE_REPOSITORY_PATH,
        IMPLEMENTATION_REPOSITORY_PATH,
        *REQUIRED_SCOPE_COMPONENT_PATHS.values(),
        *(path for paths in REQUIRED_CONTROL_EVIDENCE_PATHS.values() for path in paths),
    }
    for repository_path in sorted(paths):
        source = REPO_ROOT / repository_path
        destination = tmp_path / repository_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, destination)
        else:
            destination.write_text(
                f'"""Synthetic security-review fixture for {repository_path}."""\n',
                encoding="utf-8",
            )
    return tmp_path


def _valid_report(root: Path) -> dict[str, object]:
    protocol = _json(root / PROTOCOL_REPOSITORY_PATH)
    reviewed_commit = "a" * 40
    components = [
        {
            "component_id": component_id,
            **_descriptor(root, repository_path),
        }
        for component_id, repository_path in REQUIRED_SCOPE_COMPONENT_PATHS.items()
    ]
    controls = {
        control_id: {
            "status": "PASS",
            "evidence": [
                _descriptor(root, repository_path)
                for repository_path in REQUIRED_CONTROL_EVIDENCE_PATHS[control_id]
            ],
            "notes": f"Independent evidence reviewed for {control_id}.",
        }
        for control_id in REQUIRED_CONTROL_IDS
    }
    review_url = "https://github.com/phead198708/BiGan/pull/999#pullrequestreview-123"
    attestation_path = (
        "examples/v8/polymarket_configs/"
        "BTC-15M-cost-aware-market-residual-promotion-v1/"
        "security_review_attestation_123.json"
    )
    attestation_file = root / attestation_path
    attestation_file.parent.mkdir(parents=True, exist_ok=True)
    attestation_file.write_text(
        json.dumps(
            {
                "schema_version": ATTESTATION_SCHEMA_VERSION,
                "reviewer_github_login": "independent-reviewer",
                "review_id": 123,
                "review_url": review_url,
                "reviewed_commit_sha": reviewed_commit,
                "independent_from_implementation": True,
                "authored_reviewed_bytes": False,
                "attestation_statement": (
                    "I independently reviewed the exact scope and control evidence."
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    github_payload_path = (
        "examples/v8/polymarket_configs/"
        "BTC-15M-cost-aware-market-residual-promotion-v1/"
        "security_review_github_payload_123.json"
    )
    github_payload_file = root / github_payload_path
    github_payload_file.write_text(
        json.dumps(
            {
                "id": 123,
                "html_url": review_url,
                "state": "APPROVED",
                "commit_id": reviewed_commit,
                "submitted_at": "2026-09-20T00:00:00Z",
                "user": {"login": "independent-reviewer"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "lineage_id": protocol["lineage_id"],
        "candidate_id": protocol["candidate_id"],
        "created_at": "2026-09-20T00:00:00Z",
        "security_review_protocol": _descriptor(root, PROTOCOL_REPOSITORY_PATH),
        "candidate_bundle_sha256": _descriptor(
            root, CANDIDATE_BUNDLE_REPOSITORY_PATH
        )["sha256"],
        "reviewed_commit_sha": reviewed_commit,
        "reviewer": {
            "github_login": "independent-reviewer",
            "review_id": 123,
            "review_url": review_url,
            "review_state": "APPROVED",
            "reviewed_commit_sha": reviewed_commit,
            "independent_from_implementation": True,
            "authored_reviewed_bytes": False,
            "attestation": _descriptor(root, attestation_path),
            "github_review_payload": _descriptor(root, github_payload_path),
        },
        "implementation_author_logins": ["implementation-author"],
        "scope_manifest": {
            "schema_version": SCOPE_SCHEMA_VERSION,
            "reviewed_commit_sha": reviewed_commit,
            "components": components,
        },
        "controls": controls,
        "findings": {
            "open_counts": {"P0": 0, "P1": 0, "P2": 0, "P3": 0},
            "items": [],
        },
        "ci": {
            "run_url": "https://github.com/phead198708/BiGan/actions/runs/123",
            "conclusion": "SUCCESS",
            "exact_head_sha": reviewed_commit,
        },
        "security_review_passed": True,
        "maximum_initial_capital_fraction": 0.01,
        "fresh_outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "explicit_human_approval_recorded": False,
        "phase6_zero_capital_authorized": False,
        "micro_live_authorized": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def test_frozen_security_protocol_template_and_v6_sidecars_reconcile() -> None:
    protocol = _json(PROTOCOL)
    template = _json(TEMPLATE)
    contract = _json(CONTRACT)
    assert protocol["schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert template["schema_version"] == TEMPLATE_SCHEMA_VERSION
    assert contract["schema_version"] == CONTRACT_SCHEMA_VERSION
    validate_security_review_protocol(
        protocol,
        repository_root=REPO_ROOT,
        expected_implementation_sha256=sha256_file(REPO_ROOT / IMPLEMENTATION_REPOSITORY_PATH),
    )
    validate_security_review_template(
        template,
        protocol=protocol,
        repository_root=REPO_ROOT,
    )
    validate_release_readiness_contract_v6(contract, repository_root=REPO_ROOT)
    for path in (PROTOCOL, TEMPLATE, CONTRACT, PREFLIGHT, AUTHORIZATION_TEMPLATE):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        assert sidecar.read_text(encoding="utf-8").strip() == sha256_file(path)


def test_v6_is_strictly_additive_and_static_preflight_remains_no_go() -> None:
    v5 = _json(V5_CONTRACT)
    v6 = _json(CONTRACT)
    preflight = _json(PREFLIGHT)
    authorization = _json(AUTHORIZATION_TEMPLATE)
    for key, value in v5.items():
        if key not in {
            "schema_version",
            "created_at",
            "supersedes_preapproval_contract",
            "required_future_evidence",
        }:
            assert v6[key] == value
    for key, value in dict(v5["required_future_evidence"]).items():
        if key != "security_review":
            assert dict(v6["required_future_evidence"])[key] == value
    security_gate = dict(v6["required_future_evidence"])["security_review"]
    assert security_gate["legacy_boolean_only_report_accepted"] is False
    assert preflight["schema_version"] == PREFLIGHT_SCHEMA_VERSION
    assert preflight["technical_checks"] == {
        "fresh_confirmation": False,
        "functional_rollback": True,
        "operational_rollback": True,
        "phase6_zero_capital_pipeline": False,
        "runtime_parity": True,
        "security_review": False,
        "shadow_stability_and_monitoring": False,
    }
    assert preflight["ready_to_request_micro_live_approval"] is False
    assert preflight["fresh_outcomes_accessed"] is False
    assert preflight["settlement_accessed"] is False
    assert preflight["pnl_accessed"] is False
    assert authorization["schema_version"] == AUTHORIZATION_TEMPLATE_SCHEMA_VERSION
    assert set(authorization["required_evidence_hashes"].values()) == {None}
    assert authorization["explicit_human_approval_recorded"] is False
    assert authorization["micro_live_authorized"] is False
    assert authorization["executable"] is False
    assert authorization["safety"] == SAFETY


def test_legacy_boolean_only_security_report_is_rejected_by_v6() -> None:
    contract = _json(CONTRACT)
    legacy = {
        "schema_version": "bigan-btc-15m-residual-promotion-security-review-report-v1",
        "lineage_id": contract["lineage_id"],
        "candidate_id": contract["candidate_id"],
        "candidate_bundle_sha256": dict(contract["candidate_bundle"])["sha256"],
        "security_review_passed": True,
        "btc_15m_only_allowlist_verified": True,
        "idempotent_order_identity_verified": True,
        "order_fill_position_cash_settlement_reconciliation_verified": True,
        "kill_switch_verified": True,
        "maximum_initial_capital_fraction": 0.01,
        "explicit_human_approval_recorded": False,
        "micro_live_authorized": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    with pytest.raises(SecurityReviewError, match="schema is not exact"):
        assess_micro_live_preapproval_v6(
            contract=contract,
            evidence={"security_review": legacy},
            repository_root=REPO_ROOT,
            created_at="2026-09-20T00:00:00Z",
        )


def test_complete_independent_security_report_reconciles_in_fresh_fixture(
    tmp_path: Path,
) -> None:
    root = _review_root(tmp_path)
    protocol = _json(root / PROTOCOL_REPOSITORY_PATH)
    report = _valid_report(root)
    validate_independent_security_review_report(
        report,
        protocol=protocol,
        repository_root=root,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "self_review",
        "review_commit_mismatch",
        "github_payload_commit_mismatch",
        "attestation_identity_mismatch",
        "missing_scope_component",
        "control_evidence_scope_mismatch",
        "duplicate_control_evidence",
        "open_p1",
        "ci_not_success",
        "safety_unlock",
    ),
)
def test_independent_security_review_ambiguity_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _review_root(tmp_path)
    protocol = _json(root / PROTOCOL_REPOSITORY_PATH)
    report = _valid_report(root)
    if mutation == "self_review":
        report["reviewer"]["github_login"] = "implementation-author"
    elif mutation == "review_commit_mismatch":
        report["reviewer"]["reviewed_commit_sha"] = "c" * 40
    elif mutation == "github_payload_commit_mismatch":
        descriptor = report["reviewer"]["github_review_payload"]
        path = root / descriptor["path"]
        payload = _json(path)
        payload["commit_id"] = "c" * 40
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        descriptor["sha256"] = sha256_file(path)
    elif mutation == "attestation_identity_mismatch":
        descriptor = report["reviewer"]["attestation"]
        path = root / descriptor["path"]
        payload = _json(path)
        payload["reviewer_github_login"] = "someone-else"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        descriptor["sha256"] = sha256_file(path)
    elif mutation == "missing_scope_component":
        report["scope_manifest"]["components"].pop()
    elif mutation == "control_evidence_scope_mismatch":
        control = report["controls"][REQUIRED_CONTROL_IDS[0]]
        control["evidence"] = control["evidence"][:1]
    elif mutation == "duplicate_control_evidence":
        control = report["controls"][REQUIRED_CONTROL_IDS[0]]
        control["evidence"].append(dict(control["evidence"][0]))
    elif mutation == "open_p1":
        report["findings"]["open_counts"]["P1"] = 1
    elif mutation == "ci_not_success":
        report["ci"]["conclusion"] = "FAILURE"
    elif mutation == "safety_unlock":
        report["wallet_signing_allowed"] = True
    with pytest.raises(SecurityReviewError):
        validate_independent_security_review_report(
            report,
            protocol=protocol,
            repository_root=root,
        )


def test_missing_security_report_remains_explicitly_blocked() -> None:
    report = assess_micro_live_preapproval_v6(
        contract=_json(CONTRACT),
        evidence={},
        repository_root=REPO_ROOT,
        created_at="2026-09-20T00:00:00Z",
    )
    assert report["technical_checks"]["security_review"] is False
    assert report["security_review_independent_and_exact_head"] is False
    assert report["ready_to_request_micro_live_approval"] is False
    assert report["micro_live_authorized"] is False
    assert report["automatic_live_unlock"] is False
    assert report["wallet_signing_allowed"] is False
    assert report["polymarket_write_allowed"] is False
    assert report["capital_at_risk"] is False
    assert report["safety"] == SAFETY


def test_v6_assessment_runner_is_hash_bound_one_shot_and_non_authorizing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "assessment.json"
    report = run_micro_live_preapproval_assessment_v6(
        repository_root=REPO_ROOT,
        contract_path=CONTRACT,
        expected_contract_sha256=sha256_file(CONTRACT),
        evidence_root=tmp_path,
        evidence_descriptors={},
        output_path=output,
        created_at="2026-09-20T00:00:00Z",
    )
    assert report["ready_to_request_micro_live_approval"] is False
    assert report["micro_live_authorized"] is False
    assert output.with_suffix(".json.sha256").read_text(encoding="utf-8").strip() == (
        sha256_file(output)
    )
    with pytest.raises(FileExistsError, match="rerun forbidden"):
        run_micro_live_preapproval_assessment_v6(
            repository_root=REPO_ROOT,
            contract_path=CONTRACT,
            expected_contract_sha256=sha256_file(CONTRACT),
            evidence_root=tmp_path,
            evidence_descriptors={},
            output_path=output,
            created_at="2026-09-20T00:00:00Z",
        )


def test_v6_assessment_runner_contract_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ReleaseReadinessV6Error, match="contract SHA-256 mismatch"):
        run_micro_live_preapproval_assessment_v6(
            repository_root=REPO_ROOT,
            contract_path=CONTRACT,
            expected_contract_sha256="0" * 64,
            evidence_root=tmp_path,
            evidence_descriptors={},
            output_path=tmp_path / "assessment.json",
            created_at="2026-09-20T00:00:00Z",
        )
