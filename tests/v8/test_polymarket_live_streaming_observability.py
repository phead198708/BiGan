"""Streaming observability tests for Polymarket live paper runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bigan.v8.polymarket import PolymarketLivePaperConfig, run_polymarket_live_paper


def test_streaming_status_and_event_files_are_written(tmp_path: Path) -> None:
    result = _run_streaming(tmp_path, "streaming-healthy")

    live_status = _read_json(result.artifact_paths["live_status"])
    assert live_status["operator_status"] == "completed"
    assert live_status["stage"] == "final"
    assert live_status["prediction_count"] == 9
    assert live_status["decision_count"] == 9
    assert live_status["trade_count"] > 0
    assert live_status["paper_only"] is True
    assert live_status["capital_at_risk"] is False
    assert live_status["polymarket_write_enabled"] is False
    assert live_status["wallet_signing_enabled"] is False
    assert result.artifact_paths["live_status_md"].read_text(encoding="utf-8")

    heartbeats = _read_jsonl(result.artifact_paths["operator_heartbeat"])
    signals = _read_jsonl(result.artifact_paths["signal_events"])
    executions = _read_jsonl(result.artifact_paths["execution_events"])
    positions = _read_jsonl(result.artifact_paths["position_snapshots"])
    pnl = _read_jsonl(result.artifact_paths["pnl_snapshots"])

    assert heartbeats
    assert heartbeats[-1]["operator_status"] == "completed"
    assert len(signals) == result.operator_manifest["prediction_count"]
    assert len(executions) == result.operator_manifest["decision_count"]
    assert positions
    assert pnl

    first_signal = signals[0]
    for field in (
        "expected_return_by_action",
        "best_policy_action",
        "best_action_expected_return",
        "second_best_action_expected_return",
        "best_action_margin",
        "policy_confidence",
        "action_value_model_family",
        "feature_conditioned_action_value_model_enabled",
        "model_manifest_sha256",
    ):
        assert field in first_signal
    first_execution = executions[0]
    for field in (
        "entry_policy_action",
        "intended_exit_policy",
        "planned_exit_before_ts",
        "policy_exit_reason",
        "action_value_head_used",
        "probability_ev_fallback_used",
    ):
        assert field in first_execution
    for row in (*heartbeats, *signals, *executions, *positions, *pnl):
        _assert_safe(row)


def test_streaming_fail_closed_writes_blocked_status(tmp_path: Path) -> None:
    result = _run_streaming(
        tmp_path,
        "streaming-model-mismatch",
        inject_model_manifest_mismatch=True,
    )

    live_status = _read_json(result.artifact_paths["live_status"])
    heartbeats = _read_jsonl(result.artifact_paths["operator_heartbeat"])

    assert result.operator_manifest["operator_status"] == "blocked_fail_closed"
    assert "model_manifest_mismatch" in result.operator_manifest["critical_reason_codes"]
    assert live_status["operator_status"] == "blocked_fail_closed"
    assert "model_manifest_mismatch" in live_status["critical_reason_codes"]
    assert heartbeats[-1]["operator_status"] == "blocked_fail_closed"
    assert "model_manifest_mismatch" in heartbeats[-1]["critical_reason_codes"]
    _assert_safe(live_status)


def test_streaming_files_do_not_enter_training_raw(tmp_path: Path) -> None:
    result = _run_streaming(tmp_path, "streaming-training-boundary")
    streaming_names = {
        "live_status.json",
        "live_status.md",
        "operator_heartbeat.jsonl",
        "signal_events.jsonl",
        "execution_events.jsonl",
        "position_snapshots.jsonl",
        "pnl_snapshots.jsonl",
    }

    for row in _read_jsonl(result.artifact_paths["training_raw_index"]):
        training_raw_dir = result.run_dir / row["training_raw_dir"]
        assert not (streaming_names & {path.name for path in training_raw_dir.rglob("*")})
        _assert_training_raw_is_model_output_free(training_raw_dir)


def test_streaming_preserves_final_audit_artifacts(tmp_path: Path) -> None:
    baseline = run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id="baseline",
            output_dir=tmp_path / "baseline",
            overwrite_existing=True,
        )
    )
    streaming = _run_streaming(tmp_path / "streaming", "baseline")

    for artifact_name in (
        "polymarket_model_predictions",
        "polymarket_ev_decisions",
        "polymarket_pnl_breakdown",
        "paper_observability_report",
        "rounds_index",
        "training_raw_index",
        "paper_audit_index",
        "paper_run_summary_latest",
    ):
        assert _sha256(baseline.artifact_paths[artifact_name]) == _sha256(
            streaming.artifact_paths[artifact_name]
        )
    for streaming_name in (
        "live_status",
        "operator_heartbeat",
        "signal_events",
        "execution_events",
        "position_snapshots",
        "pnl_snapshots",
    ):
        assert streaming_name not in streaming.operator_manifest["artifact_hashes"]


def _run_streaming(tmp_path: Path, run_id: str, **overrides):
    return run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id=run_id,
            output_dir=tmp_path,
            stream_observability=True,
            status_interval_seconds=1,
            heartbeat_interval_seconds=1,
            flush_event_files=True,
            overwrite_existing=True,
            **overrides,
        )
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_training_raw_is_model_output_free(training_raw_dir: Path) -> None:
    forbidden_fields = {
        "estimated_up_probability",
        "p_up_auxiliary",
        "expected_return_by_action",
        "best_policy_action",
        "best_action_expected_return",
        "second_best_action_expected_return",
        "best_action_margin",
        "policy_confidence",
        "action_value_model_family",
        "feature_conditioned_action_value_model_enabled",
        "paper_action",
        "paper_pnl",
        "edge",
        "selected_side",
        "entry_policy_action",
        "intended_exit_policy",
        "planned_exit_before_ts",
        "policy_exit_reason",
    }
    for path in training_raw_dir.glob("raw_*.jsonl"):
        for row in _read_jsonl(path):
            assert not (forbidden_fields & set(row)), path.name


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["broker_exchange_write_enabled"] is False
    assert payload["live_exchange_write_enabled"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
