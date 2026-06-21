"""Replay paper-soak tests for v8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from examples.v8.run_paper_soak import run_paper_soak


def test_paper_soak_healthy_run_writes_summary_and_approves_phase6(
    tmp_path: Path,
) -> None:
    first = run_paper_soak(
        tmp_path / "first",
        run_id="paper-soak",
        row_count=128,
    )
    second = run_paper_soak(
        tmp_path / "second",
        run_id="paper-soak",
        row_count=128,
    )

    summary = _read_json(first.paper_run_summary_path)

    assert len(first.decisions) == 128
    assert summary["row_count"] == 128
    assert summary["order_count"] == summary["fill_count"] == 128
    assert summary["ledger_entry_count"] == 128
    assert summary["paper_only"] is True
    assert summary["capital_at_risk"] is False
    assert summary["broker_exchange_write_enabled"] is False
    assert summary["real_orders"] is False
    assert summary["real_capital"] is False
    assert summary["phase5_passed"] is True
    assert summary["phase5_kill_switch_triggered"] is False
    assert summary["phase5_reason_codes"] == []
    assert summary["phase6_passed"] is True
    assert summary["phase6_candidate_identity_verified"] is True
    assert summary["phase6_deployment_status"] == "approved_for_staged_live"
    assert first.harness_result.phase5_result.passed
    assert first.harness_result.phase6_result.report.deployment_status == (
        "approved_for_staged_live"
    )

    _assert_required_artifacts(first.artifact_paths)
    _assert_artifact_hashes(first)
    _assert_paper_only_artifacts(first.output_dir)

    assert first.harness_result.paper_report.paper_order_stream_sha256 == (
        second.harness_result.paper_report.paper_order_stream_sha256
    )
    assert first.harness_result.paper_report.paper_fill_stream_sha256 == (
        second.harness_result.paper_report.paper_fill_stream_sha256
    )
    assert first.harness_result.paper_report.paper_ledger_sha256 == (
        second.harness_result.paper_report.paper_ledger_sha256
    )
    assert first.harness_result.paper_report.paper_positions_sha256 == (
        second.harness_result.paper_report.paper_positions_sha256
    )


def test_paper_soak_degraded_run_triggers_kill_switch_and_blocks_phase6(
    tmp_path: Path,
) -> None:
    result = run_paper_soak(
        tmp_path / "degraded",
        run_id="paper-soak-degraded",
        row_count=128,
        inject_degradation=True,
    )
    summary = _read_json(result.paper_run_summary_path)

    assert summary["paper_only"] is True
    assert summary["capital_at_risk"] is False
    assert summary["phase5_passed"] is True
    assert summary["phase5_kill_switch_triggered"] is True
    assert summary["phase5_reason_codes"]
    assert summary["rollback_model_id"]
    assert summary["phase6_deployment_status"] == "blocked_fail_closed"
    assert result.harness_result.phase5_result.report.acceptance_criteria[
        "rollback_executes_reliably"
    ] is True
    assert any(
        not gate["allowed"]
        for gate in result.harness_result.phase6_result.report.stage_gates
        if gate["stage"] in {"shadow_deployment", "live_deployment"}
    )

    _assert_required_artifacts(result.artifact_paths)
    _assert_artifact_hashes(result)
    _assert_paper_only_artifacts(result.output_dir)


def _assert_required_artifacts(artifact_paths: dict[str, Path]) -> None:
    expected = {
        "paper_orders": "paper_orders.jsonl",
        "paper_fills": "paper_fills.jsonl",
        "paper_ledger": "paper_ledger.jsonl",
        "paper_positions": "paper_positions.json",
        "paper_pnl_report": "paper_pnl_report.json",
        "phase5_report": "phase5_safety_layer_report.json",
        "paper_bundle_manifest": "paper_bundle_manifest.json",
        "paper_run_summary": "paper_run_summary.json",
    }
    for key, filename in expected.items():
        assert key in artifact_paths
        assert artifact_paths[key].name == filename
        assert artifact_paths[key].exists()
    assert artifact_paths["phase6_report"].name.startswith(
        "phase6_cicd_pipeline_report_"
    )
    assert artifact_paths["phase6_report"].exists()


def _assert_artifact_hashes(result: Any) -> None:
    bundle = _read_json(result.artifact_paths["paper_bundle_manifest"])
    summary = _read_json(result.artifact_paths["paper_run_summary"])

    assert bundle["paper_only"] is True
    assert bundle["capital_at_risk"] is False
    assert bundle["broker_exchange_write_enabled"] is False
    assert "paper_run_summary" in bundle["artifacts"]
    assert bundle["paper_run_summary_sha256"] == _sha256_file(
        result.artifact_paths["paper_run_summary"]
    )
    assert result.paper_bundle_manifest_sha256 == _sha256_file(
        result.artifact_paths["paper_bundle_manifest"]
    )

    for artifact_name, artifact in bundle["artifacts"].items():
        artifact_path = result.output_dir / artifact["path"]
        assert artifact_path.exists(), artifact_name
        assert _sha256_file(artifact_path) == artifact["sha256"]

    for artifact_name, digest in summary["artifact_hashes"].items():
        artifact_path = result.artifact_paths[artifact_name]
        assert _sha256_file(artifact_path) == digest


def _assert_paper_only_artifacts(output_dir: Path) -> None:
    for filename in ("paper_orders.jsonl", "paper_fills.jsonl", "paper_ledger.jsonl"):
        for row in _read_jsonl(output_dir / filename):
            assert row["paper_only"] is True
            assert row["capital_at_risk"] is False

    positions = _read_json(output_dir / "paper_positions.json")
    assert positions["paper_only"] is True
    assert positions["capital_at_risk"] is False
    for position in positions["positions"]:
        assert position["paper_only"] is True
        assert position["capital_at_risk"] is False

    pnl_report = _read_json(output_dir / "paper_pnl_report.json")
    assert pnl_report["paper_only"] is True
    assert pnl_report["capital_at_risk"] is False

    summary = _read_json(output_dir / "paper_run_summary.json")
    assert summary["paper_only"] is True
    assert summary["capital_at_risk"] is False
    assert summary["broker_exchange_write_enabled"] is False

    phase5 = _read_json(output_dir / "phase5_safety_layer_report.json")
    assert phase5["shadow_mode_metrics"]["shadow_capital_risk_free"] is True
    assert phase5["shadow_mode_metrics"]["live_capital_at_risk"] is False

    phase6_path = next(output_dir.glob("phase6_cicd_pipeline_report_*.json"))
    phase6 = _read_json(phase6_path)
    for stage in phase6["release_manifest"]["stage_evidence"]:
        assert stage["metadata"]["paper_only"] is True
        assert stage["metadata"]["capital_at_risk"] is False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
