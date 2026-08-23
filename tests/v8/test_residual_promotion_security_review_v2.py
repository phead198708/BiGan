"""Executable-path security review v2 and preapproval v7 tests."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_release_readiness_v6 import (
    CONTRACT_REPOSITORY_PATH as V6_CONTRACT_REPOSITORY_PATH,
)
from bigan.v8.polymarket.residual_promotion_release_readiness_v7 import (
    AUTHORIZATION_TEMPLATE_REPOSITORY_PATH,
    AUTHORIZATION_TEMPLATE_SCHEMA_VERSION,
    CONTRACT_REPOSITORY_PATH,
    CONTRACT_SCHEMA_VERSION,
    PREFLIGHT_REPOSITORY_PATH,
    PREFLIGHT_SCHEMA_VERSION,
    ReleaseReadinessV7Error,
    assess_micro_live_preapproval_v7,
    run_micro_live_preapproval_assessment_v7,
    validate_release_readiness_contract_v7,
)
from bigan.v8.polymarket.residual_promotion_security_review import (
    ATTESTATION_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_security_review import (
    PROTOCOL_REPOSITORY_PATH as V1_PROTOCOL_REPOSITORY_PATH,
)
from bigan.v8.polymarket.residual_promotion_security_review import (
    REQUIRED_CONTROL_EVIDENCE_PATHS as V1_CONTROL_EVIDENCE,
)
from bigan.v8.polymarket.residual_promotion_security_review import (
    REQUIRED_SCOPE_COMPONENT_PATHS as V1_SCOPE,
)
from bigan.v8.polymarket.residual_promotion_security_review_v2 import (
    CANDIDATE_BUNDLE_REPOSITORY_PATH,
    CAPABILITY_CONTROL_ID,
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
    SecurityReviewV2Error,
    validate_independent_security_review_report_v3,
    validate_security_review_protocol_v2,
    validate_security_review_template_v3,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / CONTRACT_REPOSITORY_PATH
PREFLIGHT = REPO_ROOT / PREFLIGHT_REPOSITORY_PATH
AUTHORIZATION_TEMPLATE = REPO_ROOT / AUTHORIZATION_TEMPLATE_REPOSITORY_PATH
PROTOCOL = REPO_ROOT / PROTOCOL_REPOSITORY_PATH
TEMPLATE = REPO_ROOT / TEMPLATE_REPOSITORY_PATH
V6_CONTRACT = REPO_ROOT / V6_CONTRACT_REPOSITORY_PATH
EXECUTOR_PATH = "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py"
EXECUTOR_TEST_PATH = "tests/v8/test_residual_promotion_micro_live_executor.py"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _descriptor(root: Path, repository_path: str) -> dict[str, str]:
    return {
        "path": repository_path,
        "sha256": sha256_file(root / repository_path),
    }


def _review_root(tmp_path: Path) -> Path:
    paths = {
        PROTOCOL_REPOSITORY_PATH,
        V1_PROTOCOL_REPOSITORY_PATH,
        CANDIDATE_BUNDLE_REPOSITORY_PATH,
        IMPLEMENTATION_REPOSITORY_PATH,
        *REQUIRED_SCOPE_COMPONENT_PATHS.values(),
        *(path for paths in REQUIRED_CONTROL_EVIDENCE_PATHS.values() for path in paths),
    }
    # The inherited v1 validator requires the old protocol sidecar and old
    # implementation even when a fixture path was otherwise synthesized.
    paths.add("src/bigan/v8/polymarket/residual_promotion_security_review.py")
    for repository_path in sorted(paths):
        source = REPO_ROOT / repository_path
        destination = tmp_path / repository_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, destination)
            source_sidecar = source.with_suffix(source.suffix + ".sha256")
            if source_sidecar.is_file():
                shutil.copy2(
                    source_sidecar,
                    destination.with_suffix(destination.suffix + ".sha256"),
                )
        else:
            destination.write_text(
                f'"""Synthetic executable-path review fixture: {repository_path}."""\n',
                encoding="utf-8",
            )
    return tmp_path


def _valid_report(root: Path) -> dict[str, Any]:
    protocol = _json(root / PROTOCOL_REPOSITORY_PATH)
    reviewed_commit = "a" * 40
    review_url = "https://github.com/phead198708/BiGan/pull/999#pullrequestreview-123"
    config = root / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
    attestation_path = config / "security_review_attestation_123.json"
    attestation_path.write_text(
        json.dumps(
            {
                "schema_version": ATTESTATION_SCHEMA_VERSION,
                "reviewer_github_login": "independent-reviewer",
                "review_id": 123,
                "review_url": review_url,
                "reviewed_commit_sha": reviewed_commit,
                "independent_from_implementation": True,
                "authored_reviewed_bytes": False,
                "attestation_statement": "Independent executable-path review completed.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    github_path = config / "security_review_github_payload_123.json"
    github_path.write_text(
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
                _descriptor(root, path)
                for path in REQUIRED_CONTROL_EVIDENCE_PATHS[control_id]
            ],
            "notes": f"Independent evidence reviewed for {control_id}.",
        }
        for control_id in REQUIRED_CONTROL_IDS
    }
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
            "attestation": _descriptor(
                root, attestation_path.relative_to(root).as_posix()
            ),
            "github_review_payload": _descriptor(
                root, github_path.relative_to(root).as_posix()
            ),
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


def test_frozen_v2_v7_artifacts_validate_and_sidecars_reconcile() -> None:
    protocol = _json(PROTOCOL)
    template = _json(TEMPLATE)
    contract = _json(CONTRACT)
    assert protocol["schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert template["schema_version"] == TEMPLATE_SCHEMA_VERSION
    assert contract["schema_version"] == CONTRACT_SCHEMA_VERSION
    validate_security_review_protocol_v2(
        protocol,
        repository_root=REPO_ROOT,
        expected_implementation_sha256=sha256_file(
            REPO_ROOT / IMPLEMENTATION_REPOSITORY_PATH
        ),
    )
    validate_security_review_template_v3(
        template,
        protocol=protocol,
        repository_root=REPO_ROOT,
    )
    validate_release_readiness_contract_v7(contract, repository_root=REPO_ROOT)
    for path in (PROTOCOL, TEMPLATE, CONTRACT, PREFLIGHT, AUTHORIZATION_TEMPLATE):
        assert path.with_suffix(path.suffix + ".sha256").read_text(
            encoding="utf-8"
        ).strip() == sha256_file(path)


def test_v2_control_graph_is_strictly_additive_and_executor_bound() -> None:
    for component_id, path in V1_SCOPE.items():
        assert REQUIRED_SCOPE_COMPONENT_PATHS[component_id] == path
    for control_id, old_paths in V1_CONTROL_EVIDENCE.items():
        assert set(old_paths).issubset(REQUIRED_CONTROL_EVIDENCE_PATHS[control_id])
    execution_controls = {
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
    }
    for control_id in execution_controls:
        assert EXECUTOR_PATH in REQUIRED_CONTROL_EVIDENCE_PATHS[control_id]
        assert EXECUTOR_TEST_PATH in REQUIRED_CONTROL_EVIDENCE_PATHS[control_id]


def test_v7_preserves_every_non_security_v6_gate_and_stays_no_go() -> None:
    v6 = _json(V6_CONTRACT)
    v7 = _json(CONTRACT)
    for key, value in v6.items():
        if key not in {
            "schema_version",
            "created_at",
            "supersedes_preapproval_contract",
            "required_future_evidence",
            "security_review_protocol",
            "security_review_template",
        }:
            assert v7[key] == value
    for key, value in dict(v6["required_future_evidence"]).items():
        if key != "security_review":
            assert dict(v7["required_future_evidence"])[key] == value
    gate = dict(v7["required_future_evidence"])["security_review"]
    assert gate["actual_executor_control_evidence_required"] is True
    assert gate["actual_authorization_verifier_evidence_required"] is True
    assert gate["legacy_v1_or_v2_report_accepted"] is False
    preflight = _json(PREFLIGHT)
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
    authorization = _json(AUTHORIZATION_TEMPLATE)
    assert authorization["schema_version"] == AUTHORIZATION_TEMPLATE_SCHEMA_VERSION
    assert set(authorization["required_evidence_hashes"].values()) == {None}
    assert authorization["explicit_human_approval_recorded"] is False
    assert authorization["micro_live_authorized"] is False
    assert authorization["executable"] is False
    assert authorization["safety"] == SAFETY


def test_complete_v3_review_reconciles_through_inherited_v1_checks(
    tmp_path: Path,
) -> None:
    root = _review_root(tmp_path)
    validate_independent_security_review_report_v3(
        _valid_report(root),
        protocol=_json(root / PROTOCOL_REPOSITORY_PATH),
        repository_root=root,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "omit_executor",
        "omit_executor_test",
        "omit_capability_control",
        "old_report_schema",
        "self_review",
        "ci_failure",
        "safety_unlock",
    ),
)
def test_v3_ambiguity_or_legacy_evidence_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _review_root(tmp_path)
    report = _valid_report(root)
    if mutation == "omit_executor":
        evidence = report["controls"]["idempotent_order_and_business_identity"][
            "evidence"
        ]
        report["controls"]["idempotent_order_and_business_identity"]["evidence"] = [
            value for value in evidence if value["path"] != EXECUTOR_PATH
        ]
    elif mutation == "omit_executor_test":
        evidence = report["controls"]["kill_switch_and_operator_heartbeat"][
            "evidence"
        ]
        report["controls"]["kill_switch_and_operator_heartbeat"]["evidence"] = [
            value for value in evidence if value["path"] != EXECUTOR_TEST_PATH
        ]
    elif mutation == "omit_capability_control":
        report["controls"].pop(CAPABILITY_CONTROL_ID)
    elif mutation == "old_report_schema":
        report["schema_version"] = (
            "bigan-btc-15m-residual-promotion-independent-security-review-report-v2"
        )
    elif mutation == "self_review":
        report["reviewer"]["github_login"] = "implementation-author"
    elif mutation == "ci_failure":
        report["ci"]["conclusion"] = "FAILURE"
    elif mutation == "safety_unlock":
        report["wallet_signing_allowed"] = True
    with pytest.raises(ValueError):
        validate_independent_security_review_report_v3(
            report,
            protocol=_json(root / PROTOCOL_REPOSITORY_PATH),
            repository_root=root,
        )


def test_v7_missing_security_review_remains_blocked() -> None:
    report = assess_micro_live_preapproval_v7(
        contract=_json(CONTRACT),
        evidence={},
        repository_root=REPO_ROOT,
        created_at="2026-09-20T00:00:00Z",
    )
    assert report["technical_checks"]["security_review"] is False
    assert report["security_review_independent_exact_head_and_executor_bound"] is False
    assert report["ready_to_request_micro_live_approval"] is False
    assert report["micro_live_authorized"] is False
    assert report["automatic_live_unlock"] is False
    assert report["wallet_signing_allowed"] is False
    assert report["polymarket_write_allowed"] is False
    assert report["capital_at_risk"] is False
    assert report["safety"] == SAFETY


def test_v7_assessment_is_hash_bound_one_shot_and_non_authorizing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "assessment.json"
    report = run_micro_live_preapproval_assessment_v7(
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
        run_micro_live_preapproval_assessment_v7(
            repository_root=REPO_ROOT,
            contract_path=CONTRACT,
            expected_contract_sha256=sha256_file(CONTRACT),
            evidence_root=tmp_path,
            evidence_descriptors={},
            output_path=output,
            created_at="2026-09-20T00:00:00Z",
        )


def test_v7_contract_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ReleaseReadinessV7Error, match="contract SHA-256 mismatch"):
        run_micro_live_preapproval_assessment_v7(
            repository_root=REPO_ROOT,
            contract_path=CONTRACT,
            expected_contract_sha256="0" * 64,
            evidence_root=tmp_path,
            evidence_descriptors={},
            output_path=tmp_path / "assessment.json",
            created_at="2026-09-20T00:00:00Z",
        )


def test_v2_protocol_child_sha_drift_fails_closed(tmp_path: Path) -> None:
    root = _review_root(tmp_path)
    protocol = copy.deepcopy(_json(root / PROTOCOL_REPOSITORY_PATH))
    protocol["validator_implementation"]["sha256"] = "0" * 64
    with pytest.raises(SecurityReviewV2Error, match="SHA-256 mismatch"):
        validate_security_review_protocol_v2(
            protocol,
            repository_root=root,
            expected_implementation_sha256=sha256_file(
                root / IMPLEMENTATION_REPOSITORY_PATH
            ),
        )
