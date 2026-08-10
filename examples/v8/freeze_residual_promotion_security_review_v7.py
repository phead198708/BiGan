#!/usr/bin/env python3
"""Freeze executable-path security review v2 and preapproval v7 artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_development_lane import sha256_file  # noqa: E402
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY  # noqa: E402
from bigan.v8.polymarket.residual_promotion_release_readiness_v7 import (  # noqa: E402
    freeze_release_readiness_v7,
)
from bigan.v8.polymarket.residual_promotion_security_review import (  # noqa: E402
    ATTESTATION_SCHEMA_VERSION,
)
from bigan.v8.polymarket.residual_promotion_security_review_v2 import (  # noqa: E402
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
    V1_PROTOCOL_REPOSITORY_PATH,
    validate_security_review_protocol_v2,
    validate_security_review_template_v3,
)
from bigan.v8.polymarket.residual_promotion_v1 import (  # noqa: E402
    CANDIDATE_ID,
    LINEAGE_ID,
)

PROTOCOL = ROOT / PROTOCOL_REPOSITORY_PATH
TEMPLATE = ROOT / TEMPLATE_REPOSITORY_PATH
CREATED_AT = "2026-08-11T00:00:00Z"


def main() -> int:
    for path in (PROTOCOL, TEMPLATE):
        if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
            raise FileExistsError(f"security review artifact exists: {path.name}")
    candidate = ROOT / CANDIDATE_BUNDLE_REPOSITORY_PATH
    _verified_json(candidate)
    v1_protocol = ROOT / V1_PROTOCOL_REPOSITORY_PATH
    _verified_json(v1_protocol)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": CREATED_AT,
        "candidate_bundle": _descriptor(candidate),
        "supersedes_security_review_protocol": _descriptor(v1_protocol),
        "validator_implementation": _descriptor(ROOT / IMPLEMENTATION_REPOSITORY_PATH),
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "required_scope_component_paths": dict(REQUIRED_SCOPE_COMPONENT_PATHS),
        "required_control_ids": list(REQUIRED_CONTROL_IDS),
        "required_control_evidence_paths": {
            key: list(value) for key, value in REQUIRED_CONTROL_EVIDENCE_PATHS.items()
        },
        "review_independence_contract": {
            "github_pull_request_review_required": True,
            "github_review_payload_artifact_required": True,
            "reviewer_github_login_required": True,
            "reviewer_must_not_be_an_implementation_author": True,
            "reviewer_attestation_artifact_required": True,
            "reviewer_attestation_schema_version": ATTESTATION_SCHEMA_VERSION,
            "review_state_required": "APPROVED",
            "exact_reviewed_commit_required": True,
            "self_review_forbidden": True,
        },
        "findings_contract": {
            "open_p0_allowed": 0,
            "open_p1_allowed": 0,
            "resolved_p0_p1_requires_resolution_commit": True,
            "p2_p3_must_be_reported": True,
        },
        "ci_contract": {
            "exact_head_ci_required": True,
            "required_conclusion": "SUCCESS",
            "github_actions_run_url_required": True,
        },
        "execution_safety_contract": {
            "review_authorizes_phase6": False,
            "review_authorizes_micro_live": False,
            "review_authorizes_wallet_signing": False,
            "review_authorizes_polymarket_writes": False,
            "review_authorizes_capital": False,
            "maximum_future_initial_capital_fraction": 0.01,
        },
        "strict_successor_contract": {
            "v1_protocol_bytes_preserved": True,
            "v1_scope_and_controls_preserved": True,
            "actual_executor_required_for_execution_controls": True,
            "actual_authorization_verifier_required": True,
            "frozen_runtime_signal_binding_review_required": True,
            "legacy_v1_report_sufficient": False,
            "candidate_model_or_gate_change": False,
        },
        "security_review_is_independent_evidence_only": True,
        "automatic_authorization_or_launch": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(PROTOCOL, protocol)
    validate_security_review_protocol_v2(
        protocol,
        repository_root=ROOT,
        expected_implementation_sha256=sha256_file(ROOT / IMPLEMENTATION_REPOSITORY_PATH),
    )

    template = {
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "authorized_report_schema_version": REPORT_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": CREATED_AT,
        "security_review_protocol": _descriptor(PROTOCOL),
        "candidate_bundle": _descriptor(candidate),
        "reviewed_commit_sha": None,
        "reviewer": None,
        "implementation_author_logins": [],
        "scope_manifest": None,
        "controls": {
            control_id: {"status": "NOT_REVIEWED", "evidence": [], "notes": None}
            for control_id in REQUIRED_CONTROL_IDS
        },
        "findings": None,
        "ci": None,
        "security_review_passed": False,
        "explicit_human_approval_recorded": False,
        "phase6_zero_capital_authorized": False,
        "micro_live_authorized": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "template_is_executable": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(TEMPLATE, template)
    validate_security_review_template_v3(
        template,
        protocol=protocol,
        repository_root=ROOT,
    )
    release = freeze_release_readiness_v7(
        repository_root=ROOT,
        created_at=CREATED_AT,
    )
    print(
        json.dumps(
            {
                "security_review_protocol": _descriptor(PROTOCOL),
                "security_review_template": _descriptor(TEMPLATE),
                "release_readiness_v7": release,
                "fresh_outcomes_accessed": False,
                "settlement_accessed": False,
                "pnl_accessed": False,
                "security_review_passed": False,
                "micro_live_authorized": False,
                "safety": dict(SAFETY),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _verified_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise ValueError(f"frozen artifact sidecar mismatch: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen JSON root must be an object")
    return value


def _descriptor(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError("frozen artifact path escapes or is missing")
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _write_frozen_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
