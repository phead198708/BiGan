"""End-to-end golden-path smoke test for the v8 lifecycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from examples.v8.run_golden_path import run_golden_path


def test_v8_golden_path_dry_run_approves_staged_live_release(tmp_path: Path) -> None:
    first = run_golden_path(tmp_path / "first")
    second = run_golden_path(tmp_path / "second")

    manifest = _read_json(first.bundle_manifest_path)
    identity = manifest["identity"]

    assert first.phase6_result.report.deployment_status == "approved_for_staged_live"
    assert first.phase6_result.passed
    assert first.phase4_result.report.input_provenance_verified is True
    assert first.phase6_result.report.candidate_identity_verified is True
    assert first.phase6_result.report.candidate_identity == identity
    assert manifest["phase6_deployment_status"] == "approved_for_staged_live"
    assert manifest["phase4_input_provenance_verified"] is True
    assert manifest["phase6_candidate_identity_verified"] is True
    assert manifest["live_exchange_calls"] is False
    assert manifest["real_trading"] is False
    assert manifest["profitability_claim"] is False
    assert manifest["phase0_artifact_ready"] is True
    assert manifest["phase_statuses"]["phase0_artifact_ready"] is True
    assert manifest["phase0_dataset_hash"] == first.phase0_dataset.manifest["dataset_hash"]
    assert manifest["phase0_dataset_contract"] == first.phase0_contract.to_dict()

    assert _downstream_identity(first.phase2_result.report) == identity
    assert _downstream_identity(first.phase3_result.report) == identity
    assert _phase4_identity(first.phase4_result.report) == identity
    for stage in first.phase6_result.report.release_manifest["stage_evidence"]:
        for field_name, expected in identity.items():
            assert stage["metadata"][field_name] == expected

    assert first.phase2_result.report.passed
    assert first.phase3_result.report.phase2_execution_config_verified is True
    assert first.phase3_result.report.acceptance_criteria["cost_perturbation_robust"] is True
    assert set(first.phase3_result.report.cost_stress_metrics) == {"1.2", "1.5", "2"}
    assert first.phase5_result.report.safety_action["kill_switch_triggered"] is False
    assert "phase1_5_dataset_profile" in manifest["artifacts"]

    for artifact_name, artifact in manifest["artifacts"].items():
        artifact_path = first.bundle_dir / artifact["path"]
        assert artifact_path.exists(), artifact_name
        assert _sha256_file(artifact_path) == artifact["sha256"]

    release_manifest = _read_json(
        first.bundle_dir / manifest["artifacts"]["phase6_release_manifest"]["path"]
    )
    assert release_manifest["deployment_status"] == "approved_for_staged_live"
    assert first.phase6_result.report.release_manifest_sha256 == (
        manifest["phase6_release_manifest_sha256"]
    )
    assert second.phase6_result.report.release_manifest_sha256 == (
        first.phase6_result.report.release_manifest_sha256
    )


def _downstream_identity(report: Any) -> dict[str, str]:
    hashes = report.phase1_5_hashes
    return {
        "candidate_run_id": report.candidate_run_id,
        "model_sha256": hashes["model_sha256"],
        "policy_dataset_hash": hashes["policy_dataset_hash"],
        "split_hash": hashes["split_hash"],
    }


def _phase4_identity(report: Any) -> dict[str, str]:
    return {
        "candidate_run_id": report.candidate_run_id,
        "model_sha256": report.model_sha256,
        "policy_dataset_hash": report.policy_dataset_hash,
        "split_hash": report.split_hash,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
