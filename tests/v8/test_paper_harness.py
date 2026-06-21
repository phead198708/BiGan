"""Paper trading harness tests for v8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.paper import (
    PaperDegradationConfig,
    PaperHarnessConfig,
    PaperTradingError,
    run_paper_trading_harness,
    stream_sha256,
    synthetic_phase4_decisions,
)


def test_paper_harness_writes_ledger_phase5_and_phase6_evidence(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", run_id="paper-smoke")
    second = _run(tmp_path / "second", run_id="paper-smoke")

    assert len(first.orders) == len(first.fills) == len(first.ledger_entries) == 12
    assert first.positions
    assert any(abs(position.position_size) > 0.0 for position in first.positions)
    assert all(not observation.capital_at_risk for observation in first.observations)
    assert first.phase5_result.passed
    assert first.phase5_result.report.safety_action["kill_switch_triggered"] is False
    assert first.phase6_result.report.deployment_status == "approved_for_staged_live"
    assert first.bundle_manifest["phase6_deployment_status"] == "approved_for_staged_live"
    assert first.phase6_result.report.candidate_identity_verified is True
    assert first.paper_report.passed
    assert first.paper_report.paper_only is True
    assert first.paper_report.capital_at_risk is False

    _assert_required_artifacts(first.artifact_paths)
    _assert_paper_artifact_safety_flags(first.output_dir)
    _assert_bundle_hashes(first.bundle_manifest, first.output_dir)

    assert first.paper_report.paper_order_stream_sha256 == stream_sha256(first.orders)
    assert first.paper_report.paper_fill_stream_sha256 == stream_sha256(first.fills)
    assert first.paper_report.paper_ledger_sha256 == stream_sha256(first.ledger_entries)
    assert first.paper_report.paper_positions_sha256 == stream_sha256(first.positions)
    assert first.bundle_manifest["paper_order_stream_sha256"] == (
        first.paper_report.paper_order_stream_sha256
    )
    assert first.bundle_manifest["paper_fill_stream_sha256"] == (
        first.paper_report.paper_fill_stream_sha256
    )
    assert first.bundle_manifest["paper_ledger_sha256"] == first.paper_report.paper_ledger_sha256
    assert first.bundle_manifest["paper_positions_sha256"] == (
        first.paper_report.paper_positions_sha256
    )

    assert second.paper_report.paper_order_stream_sha256 == (
        first.paper_report.paper_order_stream_sha256
    )
    assert second.paper_report.paper_fill_stream_sha256 == (
        first.paper_report.paper_fill_stream_sha256
    )
    assert second.paper_report.paper_ledger_sha256 == first.paper_report.paper_ledger_sha256
    assert second.paper_report.paper_positions_sha256 == (
        first.paper_report.paper_positions_sha256
    )

    for stage in first.phase6_result.report.release_manifest["stage_evidence"]:
        assert stage["metadata"]["paper_only"] is True
        assert stage["metadata"]["capital_at_risk"] is False


def test_paper_harness_degradation_triggers_phase5_kill_switch(tmp_path: Path) -> None:
    result = _run(
        tmp_path / "degraded",
        run_id="paper-degraded",
        degradation=PaperDegradationConfig(
            start_index=4,
            net_return_shift=0.035,
            cost_multiplier=5.0,
            live_regime="high_volatility",
        ),
    )

    assert result.phase5_result.passed
    safety_action = result.phase5_result.report.safety_action
    assert safety_action["kill_switch_triggered"] is True
    assert safety_action["reason_codes"]
    assert result.phase5_result.report.acceptance_criteria["rollback_executes_reliably"] is True
    assert result.phase6_result.report.deployment_status == "blocked_fail_closed"
    assert result.bundle_manifest["phase6_deployment_status"] == "blocked_fail_closed"
    assert any(
        not gate["allowed"]
        for gate in result.phase6_result.report.stage_gates
        if gate["stage"] == "shadow_deployment"
    )


def test_paper_harness_refuses_broker_write_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="broker/exchange writes"):
        _config(tmp_path / "unsafe", broker_write_enabled=True)

    with pytest.raises(PaperTradingError, match="requires at least one"):
        run_paper_trading_harness(decisions=(), config=_config(tmp_path / "empty"))


def _run(
    output_dir: Path,
    *,
    run_id: str,
    degradation: PaperDegradationConfig | None = None,
):
    return run_paper_trading_harness(
        decisions=synthetic_phase4_decisions(),
        config=_config(output_dir, run_id=run_id, degradation=degradation),
    )


def _config(
    output_dir: Path,
    *,
    run_id: str = "paper-smoke",
    degradation: PaperDegradationConfig | None = None,
    broker_write_enabled: bool = False,
) -> PaperHarnessConfig:
    return PaperHarnessConfig(
        run_id=run_id,
        candidate_run_id="paper-candidate-001",
        model_sha256=_sha256_text("paper-model"),
        policy_dataset_hash=_sha256_text("paper-policy-dataset"),
        split_hash=_sha256_text("paper-split"),
        upstream_training_report_sha256=_sha256_text("paper-training-report"),
        upstream_validation_report_sha256=_sha256_text("paper-validation-report"),
        output_dir=output_dir,
        created_at="2026-06-22T01:00:00Z",
        degradation=degradation,
        broker_write_enabled=broker_write_enabled,
    )


def _assert_required_artifacts(artifact_paths: dict[str, Path]) -> None:
    expected = {
        "paper_orders": "paper_orders.jsonl",
        "paper_fills": "paper_fills.jsonl",
        "paper_ledger": "paper_ledger.jsonl",
        "paper_positions": "paper_positions.json",
        "paper_pnl_report": "paper_pnl_report.json",
        "paper_bundle_manifest": "paper_bundle_manifest.json",
        "phase5_report": "phase5_safety_layer_report.json",
    }
    for key, filename in expected.items():
        assert key in artifact_paths
        assert artifact_paths[key].name == filename
        assert artifact_paths[key].exists()
    assert "phase6_report" in artifact_paths
    assert artifact_paths["phase6_report"].name.startswith("phase6_cicd_pipeline_report_")
    assert artifact_paths["phase6_report"].exists()


def _assert_paper_artifact_safety_flags(output_dir: Path) -> None:
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
    report = _read_json(output_dir / "paper_pnl_report.json")
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    bundle = _read_json(output_dir / "paper_bundle_manifest.json")
    assert bundle["paper_only"] is True
    assert bundle["capital_at_risk"] is False


def _assert_bundle_hashes(bundle: dict[str, Any], output_dir: Path) -> None:
    for artifact_name, artifact in bundle["artifacts"].items():
        path = output_dir / artifact["path"]
        assert path.exists(), artifact_name
        assert _sha256_file(path) == artifact["sha256"]
    for field_name in (
        "paper_order_stream_sha256",
        "paper_fill_stream_sha256",
        "paper_ledger_sha256",
        "paper_positions_sha256",
        "phase5_report_sha256",
        "phase6_report_sha256",
    ):
        assert len(bundle[field_name]) == 64


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
