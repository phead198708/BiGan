"""Capability-gated residual-promotion micro-live executor tests."""

from __future__ import annotations

import ast
import copy
import json
import shutil
from pathlib import Path
from typing import Any

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
from bigan.v8.polymarket.residual_promotion_micro_live_authorization import (
    AUTHORIZATION_SCHEMA_VERSION,
    HUMAN_ATTESTATION_SCHEMA_VERSION,
    MicroLiveAuthorizationError,
    verify_micro_live_authorization,
)
from bigan.v8.polymarket.residual_promotion_micro_live_executor import (
    SIGNAL_SCHEMA_VERSION,
    MicroLiveExecutionError,
    MicroLiveExecutor,
    create_micro_live_executor,
)
from bigan.v8.polymarket.residual_promotion_release_readiness import (
    OPERATIONAL_ROLLBACK_SCHEMA_VERSION,
    PHASE6_AUTHORIZATION_SCHEMA_VERSION,
    SHADOW_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_release_readiness_v6 import (
    CONTRACT_REPOSITORY_PATH,
    run_micro_live_preapproval_assessment_v6,
)
from bigan.v8.polymarket.residual_promotion_security_review import (
    ATTESTATION_SCHEMA_VERSION,
    CANDIDATE_BUNDLE_REPOSITORY_PATH,
    PROTOCOL_REPOSITORY_PATH,
    REPORT_SCHEMA_VERSION,
    REQUIRED_CONTROL_EVIDENCE_PATHS,
    REQUIRED_CONTROL_IDS,
    REQUIRED_SCOPE_COMPONENT_PATHS,
    SCOPE_SCHEMA_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
AUTHORIZATION_TEMPLATE_PATH = f"{CONFIG_PATH}/micro_live_authorization_template_v6.json"
AUTHORIZED_AT_TS_MS = 1_789_948_800_000
NOW_TS_MS = AUTHORIZED_AT_TS_MS + 301_000


class FakeTransport:
    def __init__(self, *, fail_submit: bool = False, fail_cancel: bool = False) -> None:
        self.fail_submit = fail_submit
        self.fail_cancel = fail_cancel
        self.submit_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []

    def submit_order(self, request: dict[str, Any]) -> dict[str, Any]:
        self.submit_calls.append(copy.deepcopy(request))
        if self.fail_submit:
            raise RuntimeError("synthetic transport timeout")
        return {
            "client_order_id": request["client_order_id"],
            "exchange_order_id": f"exchange-{request['client_order_id'][:12]}",
            "status": "ACCEPTED",
            "market_id": request["market_id"],
            "token_id": request["token_id"],
            "accepted_quantity": request["quantity"],
            "limit_price": request["limit_price"],
        }

    def cancel_order(self, request: dict[str, Any]) -> dict[str, Any]:
        self.cancel_calls.append(copy.deepcopy(request))
        if self.fail_cancel:
            raise RuntimeError("synthetic cancel timeout")
        return {
            "client_order_id": request["client_order_id"],
            "exchange_order_id": request["exchange_order_id"],
            "status": "CANCELED",
        }


@pytest.fixture(scope="module")
def authorized_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("micro-live-repository") / "repo"
    shutil.copytree(
        REPO_ROOT,
        root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    evidence_root = root / "future_evidence"
    evidence_root.mkdir()
    contract = _json(root / CONTRACT_REPOSITORY_PATH)
    evidence = _complete_evidence(root, contract)
    descriptors = _write_evidence(evidence_root, evidence)
    assessment_path = evidence_root / "preapproval_assessment.json"
    assessment = run_micro_live_preapproval_assessment_v6(
        repository_root=root,
        contract_path=root / CONTRACT_REPOSITORY_PATH,
        expected_contract_sha256=sha256_file(root / CONTRACT_REPOSITORY_PATH),
        evidence_root=evidence_root,
        evidence_descriptors=descriptors,
        output_path=assessment_path,
        created_at="2026-09-20T00:00:00Z",
    )
    assert assessment["ready_to_request_micro_live_approval"] is True
    required = {
        "preapproval_assessment": _descriptor(evidence_root, assessment_path),
        "fresh_evaluation_manifest": descriptors["evaluation_manifest"],
        "phase6_release_manifest": descriptors["phase6_report"],
        "phase6_zero_capital_authorization": descriptors["phase6_authorization"],
        "operational_rollback_report": descriptors["operational_rollback"],
        "independent_security_review_report": descriptors["security_review"],
    }
    authorization = _authorization(root, evidence_root, required)
    return {
        "root": root,
        "evidence_root": evidence_root,
        "authorization": authorization,
        "now_ts_ms": NOW_TS_MS,
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _descriptor(base: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(base).as_posix(), "sha256": sha256_file(path)}


def _repository_descriptor(root: Path, repository_path: str) -> dict[str, str]:
    return {"path": repository_path, "sha256": sha256_file(root / repository_path)}


def _closed(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def _complete_evidence(
    root: Path,
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    bundle_sha = dict(contract["candidate_bundle"])["sha256"]
    functional_sha = dict(contract["functional_rollback_drill"])["sha256"]
    population_sha = "a" * 64
    safe_parameters = {"action": "NO_TRADE", "capital_fraction": 0.0}
    evaluation_report = _closed(
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
            "micro_live_go_no_go": "NO_GO_PENDING_PHASE6_AND_ROLLBACK_DRILL",
            "automatic_promotion_or_live_unlock": False,
        }
    )
    evaluation_manifest = _closed(
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
            "evaluation_report": {},
        }
    )
    shadow = _closed(
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
    )
    operational = _closed(
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
    )
    security = _security_review(root, bundle_sha)
    evidence = {
        "evaluation_manifest": evaluation_manifest,
        "evaluation_report": evaluation_report,
        "shadow_stability": shadow,
        "operational_rollback": operational,
        "security_review": security,
    }
    phase6_authorization = _closed(
        {
            "schema_version": PHASE6_AUTHORIZATION_SCHEMA_VERSION,
            "lineage_id": contract["lineage_id"],
            "candidate_id": contract["candidate_id"],
            "authorization_scope": "post_confirmation_phase6_zero_capital_only",
            "candidate_bundle_sha256": bundle_sha,
            "supersedes_template": dict(contract["phase6_zero_capital_authorization_template"]),
            "fresh_evaluation_manifest_payload_sha256": "pending",
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
    evidence["phase6_report"] = {}
    return evidence


def _write_evidence(
    evidence_root: Path,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    report_path = evidence_root / "evaluation_report.json"
    _write_json(report_path, evidence["evaluation_report"])
    evidence["evaluation_manifest"]["evaluation_report"] = _descriptor(
        evidence_root, report_path
    )
    evidence["phase6_authorization"]["fresh_evaluation_manifest_payload_sha256"] = (
        canonical_json_sha256(evidence["evaluation_manifest"])
    )
    contract = _json(
        evidence_root.parent / "repo" / CONTRACT_REPOSITORY_PATH
        if evidence_root.parent.name != "repo"
        else evidence_root.parent / CONTRACT_REPOSITORY_PATH
    )
    evidence["phase6_report"] = _phase6_report(
        contract,
        evidence["phase6_authorization"],
    )
    descriptors: dict[str, dict[str, str]] = {}
    for name, payload in evidence.items():
        path = evidence_root / f"{name}.json"
        _write_json(path, payload)
        descriptors[name] = _descriptor(evidence_root, path)
    return descriptors


def _phase6_report(
    contract: dict[str, Any],
    phase6_authorization: dict[str, Any],
) -> dict[str, Any]:
    identity = dict(contract["phase6_candidate_identity"])
    candidate_id = str(contract["candidate_id"])
    bundle_sha = str(identity["model_sha256"])
    authorization_sha = canonical_json_sha256(phase6_authorization)

    def stage(name: str, artifact_sha: str, metadata: dict[str, Any]) -> CICDStageEvidence:
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

    stages = (
        stage("training", bundle_sha, {"accepted_candidate_model": True, "deterministic_training": True}),
        stage(
            "validation",
            "1" * 64,
            {
                "oos_backtest_passed": True,
                "cost_stress_passed": True,
                "cost_stress_multipliers": [1.2, 1.5, 2.0],
            },
        ),
        stage(
            "shadow_deployment",
            "2" * 64,
            {"shadow_mode": True, "simulate_live_execution": True, "capital_at_risk": False},
        ),
        stage(
            "live_deployment",
            "3" * 64,
            {
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
            "4" * 64,
            {
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
    return run_phase6_cicd_pipeline(
        candidate_run_id=candidate_id,
        stage_evidence=stages,
        rollback_plan=rollback,
        config=CICDPipelineConfig(created_at="2026-09-20T00:00:00Z"),
    ).report.to_dict()


def _security_review(root: Path, bundle_sha: str) -> dict[str, Any]:
    reviewed_commit = "d" * 40
    config = root / CONFIG_PATH
    review_url = "https://github.com/phead198708/BiGan/pull/999#pullrequestreview-9001"
    attestation_path = config / "security_review_attestation_9001.json"
    _write_json(
        attestation_path,
        {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "reviewer_github_login": "independent-reviewer",
            "review_id": 9001,
            "review_url": review_url,
            "reviewed_commit_sha": reviewed_commit,
            "independent_from_implementation": True,
            "authored_reviewed_bytes": False,
            "attestation_statement": "Independent exact-scope security review completed.",
        },
    )
    github_path = config / "security_review_github_payload_9001.json"
    _write_json(
        github_path,
        {
            "id": 9001,
            "html_url": review_url,
            "state": "APPROVED",
            "commit_id": reviewed_commit,
            "submitted_at": "2026-09-20T00:00:00Z",
            "user": {"login": "independent-reviewer"},
        },
    )
    components = [
        {
            "component_id": component_id,
            **_repository_descriptor(root, repository_path),
        }
        for component_id, repository_path in REQUIRED_SCOPE_COMPONENT_PATHS.items()
    ]
    controls = {
        control_id: {
            "status": "PASS",
            "evidence": [
                _repository_descriptor(root, repository_path)
                for repository_path in REQUIRED_CONTROL_EVIDENCE_PATHS[control_id]
            ],
            "notes": f"Independent evidence reviewed for {control_id}.",
        }
        for control_id in REQUIRED_CONTROL_IDS
    }
    return _closed(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "lineage_id": "BTC-15M-cost-aware-market-residual-promotion-v1",
            "candidate_id": "residual-v4-challenger-carry-forward-final-fit-001",
            "created_at": "2026-09-20T00:00:00Z",
            "security_review_protocol": _repository_descriptor(root, PROTOCOL_REPOSITORY_PATH),
            "candidate_bundle_sha256": bundle_sha,
            "reviewed_commit_sha": reviewed_commit,
            "reviewer": {
                "github_login": "independent-reviewer",
                "review_id": 9001,
                "review_url": review_url,
                "review_state": "APPROVED",
                "reviewed_commit_sha": reviewed_commit,
                "independent_from_implementation": True,
                "authored_reviewed_bytes": False,
                "attestation": _repository_descriptor(
                    root, attestation_path.relative_to(root).as_posix()
                ),
                "github_review_payload": _repository_descriptor(
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
                "run_url": "https://github.com/phead198708/BiGan/actions/runs/9001",
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
        }
    )


def _authorization(
    root: Path,
    evidence_root: Path,
    required: dict[str, dict[str, str]],
) -> dict[str, Any]:
    evidence_payload_sha = canonical_json_sha256(
        {name: value["sha256"] for name, value in sorted(required.items())}
    )
    candidate_sha = sha256_file(root / CANDIDATE_BUNDLE_REPOSITORY_PATH)
    identity = {
        "lineage_id": "BTC-15M-cost-aware-market-residual-promotion-v1",
        "candidate_id": "residual-v4-challenger-carry-forward-final-fit-001",
        "candidate_bundle_sha256": candidate_sha,
        "evidence_payload_sha256": evidence_payload_sha,
        "capital_base_usd": "1000",
        "requested_initial_capital_fraction": "0.01",
        "maximum_notional_usd": "10.00",
        "maximum_open_orders": 2,
        "market_allowlist": ["BTC-15M"],
        "allowed_actions": ["BUY_UP_HOLD", "BUY_DOWN_HOLD"],
        "authorized_at_ts_ms": AUTHORIZED_AT_TS_MS,
        "expires_at_ts_ms": AUTHORIZED_AT_TS_MS + 10_000_000,
        "maximum_signal_age_ms": 5_000,
        "maximum_operator_heartbeat_age_ms": 5_000,
        "approval_issue_number": 264,
    }
    authorization_id = canonical_json_sha256(identity)
    command = (
        "APPROVE BTC-15M-cost-aware-market-residual-promotion-v1 MICRO-LIVE "
        f"authorization_id={authorization_id} capital_base_usd=1000 "
        "maximum_notional_usd=10.00 maximum_open_orders=2 "
        f"capital_fraction=0.01 expires_at_ts_ms={identity['expires_at_ts_ms']}"
    )
    comment_url = "https://github.com/phead198708/BiGan/issues/264#issuecomment-99001"
    github_path = evidence_root / "human_approval_github_payload.json"
    _write_json(
        github_path,
        {
            "id": 99001,
            "html_url": comment_url,
            "created_at": "2026-09-21T00:00:00Z",
            "body": command,
            "user": {"login": "phead198708"},
        },
    )
    attestation_path = evidence_root / "human_approval_attestation.json"
    _write_json(
        attestation_path,
        {
            "schema_version": HUMAN_ATTESTATION_SCHEMA_VERSION,
            "github_login": "phead198708",
            "issue_number": 264,
            "comment_id": 99001,
            "comment_url": comment_url,
            "authorization_id": authorization_id,
            "approved_at_ts_ms": AUTHORIZED_AT_TS_MS,
            "attestation_statement": command,
        },
    )
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "lineage_id": identity["lineage_id"],
        "candidate_id": identity["candidate_id"],
        "created_at": "2026-09-21T00:00:00Z",
        "authorization_id": authorization_id,
        "supersedes_template": _repository_descriptor(root, AUTHORIZATION_TEMPLATE_PATH),
        "candidate_bundle": _repository_descriptor(root, CANDIDATE_BUNDLE_REPOSITORY_PATH),
        "preapproval_contract": _repository_descriptor(root, CONTRACT_REPOSITORY_PATH),
        "required_evidence": required,
        "evidence_payload_sha256": evidence_payload_sha,
        "human_approval": {
            "github_login": "phead198708",
            "issue_number": 264,
            "comment_id": 99001,
            "comment_url": comment_url,
            "approved_at_ts_ms": AUTHORIZED_AT_TS_MS,
            "github_comment_payload": _descriptor(evidence_root, github_path),
            "attestation": _descriptor(evidence_root, attestation_path),
        },
        "capital_base_usd": identity["capital_base_usd"],
        "requested_initial_capital_fraction": identity[
            "requested_initial_capital_fraction"
        ],
        "maximum_notional_usd": identity["maximum_notional_usd"],
        "maximum_open_orders": identity["maximum_open_orders"],
        "market_allowlist": identity["market_allowlist"],
        "allowed_actions": identity["allowed_actions"],
        "one_trade_maximum_per_market": True,
        "authorized_at_ts_ms": identity["authorized_at_ts_ms"],
        "expires_at_ts_ms": identity["expires_at_ts_ms"],
        "maximum_signal_age_ms": identity["maximum_signal_age_ms"],
        "maximum_operator_heartbeat_age_ms": identity[
            "maximum_operator_heartbeat_age_ms"
        ],
        "explicit_human_approval_recorded": True,
        "micro_live_authorized": True,
        "micro_live_started": False,
        "live_trading_allowed": True,
        "wallet_signing_allowed": True,
        "polymarket_write_allowed": True,
        "capital_at_risk": True,
        "automatic_launch_allowed": False,
        "capital_increase_allowed": False,
        "executable": True,
    }


def _verified(fixture: dict[str, Any]):
    return verify_micro_live_authorization(
        fixture["authorization"],
        repository_root=fixture["root"],
        evidence_root=fixture["evidence_root"],
        now_ts_ms=fixture["now_ts_ms"],
    )


def _signal(**overrides: Any) -> dict[str, Any]:
    signal_payload = {
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "lineage_id": "BTC-15M-cost-aware-market-residual-promotion-v1",
        "candidate_id": "residual-v4-challenger-carry-forward-final-fit-001",
        "candidate_bundle_sha256": "placeholder",
        "market_id": "0x" + "1" * 64,
        "slug": "btc-updown-15m-1789948800",
        "market_family": "BTC-15M",
        "decision_ts_ms": AUTHORIZED_AT_TS_MS + 300_000,
        "observed_at_ts_ms": AUTHORIZED_AT_TS_MS + 300_000,
        "action_values": {
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD": -0.05,
            "BUY_DOWN_HOLD": 0.05,
        },
        "executable_asks": {"UP": "0.55", "DOWN": "0.40"},
        "up_token_id": "67890",
        "down_token_id": "12345",
        "selected_action": "BUY_DOWN_HOLD",
        "model_scored": True,
        "fail_closed": False,
        "fail_closed_reasons": [],
        "decision_influenced_collection": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "safety": dict(SAFETY),
    }
    payload = {
        "signal_payload": signal_payload,
        "now_ts_ms": NOW_TS_MS,
        "operator_heartbeat_ts_ms": NOW_TS_MS - 50,
    }
    for key, value in overrides.items():
        if key in signal_payload:
            signal_payload[key] = value
        else:
            payload[key] = value
    return payload


def test_current_template_cannot_create_executor(
    authorized_fixture: dict[str, Any],
) -> None:
    template = _json(authorized_fixture["root"] / AUTHORIZATION_TEMPLATE_PATH)
    transport = FakeTransport()
    with pytest.raises(MicroLiveAuthorizationError, match="schema is not exact"):
        create_micro_live_executor(
            authorization=template,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
            transport=transport,
        )
    assert transport.submit_calls == []
    assert transport.cancel_calls == []


def test_executor_bundles_no_network_wallet_or_credential_adapter() -> None:
    paths = (
        REPO_ROOT
        / "src/bigan/v8/polymarket/residual_promotion_micro_live_authorization.py",
        REPO_ROOT / "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py",
    )
    forbidden_modules = {
        "eth_account",
        "httpx",
        "py_clob_client",
        "requests",
        "socket",
        "urllib",
        "web3",
    }
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (
                node.names
                if isinstance(node, ast.Import)
                else [ast.alias(name=str(node.module or ""))]
            )
        }
        assert imports.isdisjoint(forbidden_modules)
        assert "os.environ" not in source
        assert "getenv(" not in source


def test_valid_graph_creates_capability_but_does_not_auto_launch(
    authorized_fixture: dict[str, Any],
) -> None:
    transport = FakeTransport()
    executor = create_micro_live_executor(
        authorization=authorized_fixture["authorization"],
        repository_root=authorized_fixture["root"],
        evidence_root=authorized_fixture["evidence_root"],
        now_ts_ms=NOW_TS_MS,
        transport=transport,
    )
    assert executor.events == ()
    assert executor.reconciliation_snapshot()["cash_usd"] == "10.00"
    assert transport.submit_calls == []


def test_submit_is_idempotent_and_one_market_has_one_intent(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    signal = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    first = executor.submit_signal(**signal)
    replay = executor.submit_signal(**signal)
    assert first["status"] == "ORDER_ACKNOWLEDGED"
    assert replay["status"] == "IDEMPOTENT_REPLAY"
    assert replay["client_order_id"] == first["client_order_id"]
    assert len(transport.submit_calls) == 1


def test_conflicting_duplicate_engages_kill_switch_and_cancels_open_order(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    signal = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    executor.submit_signal(**signal)
    conflict = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        selected_action="BUY_UP_HOLD",
        action_values={
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD": 0.05,
            "BUY_DOWN_HOLD": -0.05,
        },
    )
    with pytest.raises(MicroLiveExecutionError, match="conflicting duplicate"):
        executor.submit_signal(**conflict)
    snapshot = executor.reconciliation_snapshot()
    assert snapshot["kill_switch_active"] is True
    assert snapshot["kill_switch_reason"] == "conflicting_duplicate_intent"
    assert len(transport.cancel_calls) == 1
    assert snapshot["open_order_count"] == 0


def test_allowlist_no_trade_candidate_and_token_contract_block_without_transport(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="identity or safety"):
        executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_family="ETH-15M",
            )
        )
    no_trade = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        selected_action="NO_TRADE",
        action_values={
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD": -0.02,
            "BUY_DOWN_HOLD": -0.01,
        },
    )
    assert executor.submit_signal(**no_trade)["reason"] == "signal_selected_no_trade"
    with pytest.raises(MicroLiveExecutionError, match="identity or safety"):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256="0" * 64)
        )
    duplicate_tokens = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        up_token_id="12345",
        down_token_id="12345",
    )
    with pytest.raises(MicroLiveExecutionError, match="identity or safety"):
        executor.submit_signal(**duplicate_tokens)
    assert transport.submit_calls == []


def test_signal_envelope_tampering_and_outcome_fields_fail_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)

    mismatched = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    mismatched["signal_payload"]["selected_action"] = "BUY_UP_HOLD"
    with pytest.raises(MicroLiveExecutionError, match="zero-threshold decision"):
        executor.submit_signal(**mismatched)

    outcome_bearing = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256
    )
    outcome_bearing["signal_payload"]["outcome"] = "UP"
    with pytest.raises(MicroLiveExecutionError, match="schema is not exact"):
        executor.submit_signal(**outcome_bearing)

    outcome_opened = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        outcomes_accessed=True,
    )
    with pytest.raises(MicroLiveExecutionError, match="identity or safety"):
        executor.submit_signal(**outcome_opened)

    off_schedule = _signal(
        candidate_bundle_sha256=verified.candidate_bundle_sha256,
        decision_ts_ms=AUTHORIZED_AT_TS_MS + 300_001,
        observed_at_ts_ms=AUTHORIZED_AT_TS_MS + 300_001,
    )
    with pytest.raises(MicroLiveExecutionError, match="frozen schedule"):
        executor.submit_signal(**off_schedule)

    assert transport.submit_calls == []
    assert transport.cancel_calls == []


def test_open_order_and_authorization_lifetime_notional_caps(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    for index in (1, 2):
        result = executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_id=f"0x{index:064x}",
                up_token_id=str(10_000 + index * 2),
                down_token_id=str(10_001 + index * 2),
            )
        )
        assert result["status"] == "ORDER_ACKNOWLEDGED"
    blocked = executor.submit_signal(
        **_signal(
            candidate_bundle_sha256=verified.candidate_bundle_sha256,
            market_id=f"0x{3:064x}",
            up_token_id="10006",
            down_token_id="10007",
        )
    )
    assert blocked["reason"] == "maximum_open_orders_reached"
    assert len(transport.submit_calls) == 2

    lifetime_transport = FakeTransport()
    lifetime = MicroLiveExecutor(verified, transport=lifetime_transport)
    for index in range(1, 11):
        result = lifetime.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                market_id=f"0x{index:064x}",
                executable_asks={"UP": "0.98", "DOWN": "0.99"},
                up_token_id=str(20_000 + index * 2),
                down_token_id=str(20_001 + index * 2),
            )
        )
        assert result["status"] == "ORDER_ACKNOWLEDGED"
        lifetime.record_order_closed(
            client_order_id=result["client_order_id"],
            status="CANCELED",
            now_ts_ms=NOW_TS_MS,
            transport_event_sha256=f"{index:064x}",
        )
    capped = lifetime.submit_signal(
        **_signal(
            candidate_bundle_sha256=verified.candidate_bundle_sha256,
            market_id=f"0x{11:064x}",
            executable_asks={"UP": "0.98", "DOWN": "0.99"},
            up_token_id="20022",
            down_token_id="20023",
        )
    )
    assert capped["reason"] == "authorization_notional_cap_exceeded"
    assert len(lifetime_transport.submit_calls) == 10


def test_stale_heartbeat_kills_and_cancels_existing_order(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    base = _signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    executor.submit_signal(**base)
    with pytest.raises(MicroLiveExecutionError, match="heartbeat is stale"):
        executor.submit_signal(
            **_signal(
                candidate_bundle_sha256=verified.candidate_bundle_sha256,
                operator_heartbeat_ts_ms=NOW_TS_MS - 6_000,
            )
        )
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True
    assert len(transport.cancel_calls) == 1


def test_fill_cash_position_settlement_and_restart_reconcile(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    fill = executor.record_fill(
        client_order_id=order["client_order_id"],
        fill_id="fill-001",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.39",
        fee_usd="0.01",
        transport_event_sha256="f" * 64,
    )
    assert fill["snapshot"]["cash_usd"] == "9.60"
    assert fill["snapshot"]["positions"]["DOWN"] == "1"
    assert executor.record_fill(
        client_order_id=order["client_order_id"],
        fill_id="fill-001",
        now_ts_ms=NOW_TS_MS,
        quantity="1",
        price="0.39",
        fee_usd="0.01",
        transport_event_sha256="f" * 64,
    )["status"] == "IDEMPOTENT_FILL_REPLAY"
    settled = executor.record_settlement(
        client_order_id=order["client_order_id"],
        settlement_id="settlement-001",
        now_ts_ms=NOW_TS_MS,
        payout_per_token="1",
        official_settlement_sha256="1" * 64,
    )
    assert settled["snapshot"]["cash_usd"] == "10.60"
    assert settled["snapshot"]["positions"]["DOWN"] == "0"
    state = executor.export_state()
    restored = MicroLiveExecutor.restore(
        authorization=verified,
        transport=transport,
        state=state,
    )
    assert restored.export_state() == state
    assert restored.reconciliation_snapshot() == executor.reconciliation_snapshot()


def test_conflicting_fill_and_partial_open_settlement_engage_kill_switch(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport()
    executor = MicroLiveExecutor(verified, transport=transport)
    order = executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    executor.record_fill(
        client_order_id=order["client_order_id"],
        fill_id="fill-partial",
        now_ts_ms=NOW_TS_MS,
        quantity="0.5",
        price="0.39",
        fee_usd="0.01",
        transport_event_sha256="a" * 64,
    )
    with pytest.raises(MicroLiveExecutionError, match="conflicting duplicate fill"):
        executor.record_fill(
            client_order_id=order["client_order_id"],
            fill_id="fill-partial",
            now_ts_ms=NOW_TS_MS,
            quantity="0.5",
            price="0.38",
            fee_usd="0.01",
            transport_event_sha256="a" * 64,
        )
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True
    assert executor.reconciliation_snapshot()["open_order_count"] == 0
    assert len(transport.cancel_calls) == 1

    second_transport = FakeTransport()
    second = MicroLiveExecutor(verified, transport=second_transport)
    second_order = second.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    second.record_fill(
        client_order_id=second_order["client_order_id"],
        fill_id="fill-partial-2",
        now_ts_ms=NOW_TS_MS,
        quantity="0.5",
        price="0.39",
        fee_usd="0.01",
        transport_event_sha256="b" * 64,
    )
    with pytest.raises(MicroLiveExecutionError, match="open order cannot settle"):
        second.record_settlement(
            client_order_id=second_order["client_order_id"],
            settlement_id="settlement-too-early",
            now_ts_ms=NOW_TS_MS,
            payout_per_token="1",
            official_settlement_sha256="c" * 64,
        )
    assert second.reconciliation_snapshot()["kill_switch_active"] is True
    assert len(second_transport.cancel_calls) == 1


def test_unknown_submission_engages_kill_switch_without_retry(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    transport = FakeTransport(fail_submit=True)
    executor = MicroLiveExecutor(verified, transport=transport)
    with pytest.raises(MicroLiveExecutionError, match="submission became unknown"):
        executor.submit_signal(
            **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
        )
    assert len(transport.submit_calls) == 1
    assert executor.reconciliation_snapshot()["kill_switch_active"] is True


def test_rehashed_tampered_state_still_fails_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    executor = MicroLiveExecutor(verified, transport=FakeTransport())
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    state = executor.export_state()
    state["events"][0]["payload"]["selected_action"] = "BUY_UP_HOLD"
    previous = "GENESIS"
    for event in state["events"]:
        event["previous_event_sha256"] = previous
        core = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = canonical_json_sha256(core)
        previous = event["event_sha256"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(MicroLiveExecutionError, match="prepared order identity"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            state=state,
        )


def test_rehashed_event_timestamp_regression_still_fails_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    verified = _verified(authorized_fixture)
    executor = MicroLiveExecutor(verified, transport=FakeTransport())
    executor.submit_signal(
        **_signal(candidate_bundle_sha256=verified.candidate_bundle_sha256)
    )
    state = executor.export_state()
    state["events"][1]["event_ts_ms"] = state["events"][0]["event_ts_ms"] - 1
    previous = "GENESIS"
    for event in state["events"]:
        event["previous_event_sha256"] = previous
        core = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = canonical_json_sha256(core)
        previous = event["event_sha256"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(MicroLiveExecutionError, match="event chain"):
        MicroLiveExecutor.restore(
            authorization=verified,
            transport=FakeTransport(),
            state=state,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("explicit_human_approval_recorded", False, "state is not explicit"),
        ("micro_live_authorized", False, "state is not explicit"),
        ("automatic_launch_allowed", True, "state is not explicit"),
        ("requested_initial_capital_fraction", "0.02", "limits or validity"),
    ),
)
def test_authorization_tampering_fails_closed(
    authorized_fixture: dict[str, Any],
    field: str,
    value: Any,
    message: str,
) -> None:
    changed = copy.deepcopy(authorized_fixture["authorization"])
    changed[field] = value
    with pytest.raises(MicroLiveAuthorizationError, match=message):
        verify_micro_live_authorization(
            changed,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )


def test_expired_authorization_and_evidence_sha_drift_fail_closed(
    authorized_fixture: dict[str, Any],
) -> None:
    authorization = authorized_fixture["authorization"]
    with pytest.raises(MicroLiveAuthorizationError, match="validity window"):
        verify_micro_live_authorization(
            authorization,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=authorization["expires_at_ts_ms"],
        )
    changed = copy.deepcopy(authorization)
    changed["required_evidence"]["fresh_evaluation_manifest"]["sha256"] = "0" * 64
    with pytest.raises(MicroLiveAuthorizationError, match="path or SHA-256 mismatch"):
        verify_micro_live_authorization(
            changed,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )


def test_human_approval_owner_and_timestamp_are_exact(
    authorized_fixture: dict[str, Any],
) -> None:
    changed = copy.deepcopy(authorized_fixture["authorization"])
    changed["human_approval"]["github_login"] = "untrusted-user"
    with pytest.raises(MicroLiveAuthorizationError, match="provenance"):
        verify_micro_live_authorization(
            changed,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )

    approval_descriptor = authorized_fixture["authorization"]["human_approval"][
        "github_comment_payload"
    ]
    github = _json(authorized_fixture["evidence_root"] / approval_descriptor["path"])
    assert "capital_base_usd=1000" in github["body"]
    assert "maximum_notional_usd=10.00" in github["body"]
    assert "maximum_open_orders=2" in github["body"]

    changed = copy.deepcopy(authorized_fixture["authorization"])
    changed["created_at"] = "2026-09-21T00:00:01Z"
    with pytest.raises(MicroLiveAuthorizationError, match="limits or validity"):
        verify_micro_live_authorization(
            changed,
            repository_root=authorized_fixture["root"],
            evidence_root=authorized_fixture["evidence_root"],
            now_ts_ms=NOW_TS_MS,
        )
