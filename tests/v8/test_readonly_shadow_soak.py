"""Read-only paper shadow soak tests for v8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.paper import (
    DeterministicReplayFeed,
    ReadOnlyFeedError,
    ReadOnlyFeedEvent,
    ReadOnlyShadowSoakConfig,
    run_readonly_shadow_soak,
    synthetic_readonly_feed_events,
)


def test_readonly_shadow_soak_healthy_short_run_approves_phase6(
    tmp_path: Path,
) -> None:
    result = run_readonly_shadow_soak(
        config=_config(tmp_path, run_id="readonly-healthy", duration_seconds=300),
    )
    summary = result.final_summary

    assert summary["feed_event_count"] == 6
    assert summary["heartbeat_count"] >= 6
    assert summary["periodic_summary_count"] >= 2
    assert summary["feed_gap_count"] == 0
    assert summary["feed_late_event_count"] == 0
    assert summary["feed_out_of_order_count"] == 0
    assert summary["paper_only"] is True
    assert summary["capital_at_risk"] is False
    assert summary["broker_exchange_write_enabled"] is False
    assert summary["live_exchange_write_enabled"] is False
    assert summary["phase5_kill_switch_triggered"] is False
    assert summary["phase6_candidate_identity_verified"] is True
    assert summary["phase6_deployment_status"] == "approved_for_staged_live"

    _assert_required_artifacts(result.artifact_paths)
    _assert_bundle_hashes(result)
    _assert_summary_hashes(result)
    _assert_paper_only_artifacts(result.output_dir)


def test_readonly_shadow_soak_degradation_blocks_phase6(tmp_path: Path) -> None:
    result = run_readonly_shadow_soak(
        config=_config(
            tmp_path,
            run_id="readonly-degraded",
            duration_seconds=900,
            inject_degradation=True,
        ),
    )
    summary = result.final_summary

    assert summary["phase5_kill_switch_triggered"] is True
    assert summary["phase5_reason_codes"]
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
    _assert_bundle_hashes(result)
    _assert_paper_only_artifacts(result.output_dir)


def test_readonly_shadow_soak_operator_stop_file_clean_shutdown(
    tmp_path: Path,
) -> None:
    result = run_readonly_shadow_soak(
        config=_config(
            tmp_path,
            run_id="readonly-stop",
            duration_seconds=1_200,
            stop_after_events=5,
        ),
    )
    summary = result.final_summary

    assert summary["stop_reason"] == "operator_stop"
    assert summary["feed_event_count"] == 5
    assert summary["phase5_kill_switch_triggered"] is False
    assert summary["phase6_deployment_status"] == "approved_for_staged_live"
    assert _read_json(result.artifact_paths["feed_health_report"])[
        "stop_file_seen"
    ] is True


def test_readonly_shadow_soak_fails_closed_for_write_capable_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReadOnlyFeedError, match="write-capable"):
        DeterministicReplayFeed(
            events=synthetic_readonly_feed_events(row_count=4),
            write_capable=True,
        )

    with pytest.raises(ReadOnlyFeedError, match="write-capable"):
        ReadOnlyFeedEvent(
            event_ts=1,
            received_ts=1,
            source="unsafe",
            instrument_id="btc-up",
            bid_price=99.0,
            ask_price=101.0,
            mid_price=100.0,
            volume=1.0,
            trade_count=1,
            spread_bps=10.0,
            feed_sequence=0,
            write_capable=True,
        )

    with pytest.raises(ReadOnlyFeedError, match="broker/exchange writes"):
        _config(
            tmp_path,
            run_id="unsafe-broker",
            broker_exchange_write_enabled=True,
        )

    with pytest.raises(ReadOnlyFeedError, match="live exchange writes"):
        _config(
            tmp_path,
            run_id="unsafe-live",
            live_exchange_write_enabled=True,
        )


def _config(
    output_dir: Path,
    *,
    run_id: str,
    duration_seconds: int = 300,
    inject_degradation: bool = False,
    stop_after_events: int | None = None,
    broker_exchange_write_enabled: bool = False,
    live_exchange_write_enabled: bool = False,
) -> ReadOnlyShadowSoakConfig:
    return ReadOnlyShadowSoakConfig(
        run_id=run_id,
        output_dir=output_dir,
        duration_seconds=duration_seconds,
        feed_event_interval_seconds=60,
        heartbeat_interval_seconds=30,
        summary_interval_seconds=120,
        inject_degradation=inject_degradation,
        stop_after_events=stop_after_events,
        broker_exchange_write_enabled=broker_exchange_write_enabled,
        live_exchange_write_enabled=live_exchange_write_enabled,
        overwrite_existing=False,
    )


def _assert_required_artifacts(artifact_paths: dict[str, Path]) -> None:
    expected = {
        "readonly_feed_events": "readonly_feed_events.jsonl",
        "paper_orders": "paper_orders.jsonl",
        "paper_fills": "paper_fills.jsonl",
        "paper_ledger": "paper_ledger.jsonl",
        "paper_positions": "paper_positions.json",
        "paper_pnl_report": "paper_pnl_report.json",
        "paper_run_summary": "paper_run_summary.json",
        "paper_soak_heartbeat": "paper_soak_heartbeat.jsonl",
        "paper_soak_periodic_summaries": "paper_soak_periodic_summaries.jsonl",
        "feed_health_report": "feed_health_report.json",
        "phase5_report": "phase5_safety_layer_report.json",
        "paper_bundle_manifest": "paper_bundle_manifest.json",
    }
    for key, filename in expected.items():
        assert key in artifact_paths
        assert artifact_paths[key].name == filename
        assert artifact_paths[key].exists()
    assert artifact_paths["phase6_report"].name.startswith(
        "phase6_cicd_pipeline_report_"
    )
    assert artifact_paths["phase6_report"].exists()


def _assert_bundle_hashes(result: Any) -> None:
    bundle = _read_json(result.artifact_paths["paper_bundle_manifest"])
    assert bundle["paper_only"] is True
    assert bundle["capital_at_risk"] is False
    assert bundle["broker_exchange_write_enabled"] is False
    assert bundle["live_exchange_write_enabled"] is False
    for artifact_name, artifact in bundle["artifacts"].items():
        artifact_path = result.output_dir / artifact["path"]
        assert artifact_path.exists(), artifact_name
        assert _sha256_file(artifact_path) == artifact["sha256"]


def _assert_summary_hashes(result: Any) -> None:
    summary = _read_json(result.artifact_paths["paper_run_summary"])
    for artifact_name, digest in summary["artifact_hashes"].items():
        artifact_path = result.artifact_paths[artifact_name]
        assert artifact_path.exists(), artifact_name
        assert _sha256_file(artifact_path) == digest


def _assert_paper_only_artifacts(output_dir: Path) -> None:
    for filename in (
        "readonly_feed_events.jsonl",
        "paper_orders.jsonl",
        "paper_fills.jsonl",
        "paper_ledger.jsonl",
        "paper_soak_heartbeat.jsonl",
        "paper_soak_periodic_summaries.jsonl",
    ):
        rows = _read_jsonl(output_dir / filename)
        assert rows, filename
        for row in rows:
            assert row["paper_only"] is True
            assert row["capital_at_risk"] is False

    for filename in (
        "paper_positions.json",
        "paper_pnl_report.json",
        "paper_run_summary.json",
        "feed_health_report.json",
        "phase5_safety_layer_report.json",
        "paper_bundle_manifest.json",
    ):
        payload = _read_json(output_dir / filename)
        assert payload["paper_only"] is True
        assert payload["capital_at_risk"] is False
        if "broker_exchange_write_enabled" in payload:
            assert payload["broker_exchange_write_enabled"] is False
        if "live_exchange_write_enabled" in payload:
            assert payload["live_exchange_write_enabled"] is False

    phase6_path = next(output_dir.glob("phase6_cicd_pipeline_report_*.json"))
    phase6 = _read_json(phase6_path)
    assert phase6["paper_only"] is True
    assert phase6["capital_at_risk"] is False
    assert phase6["broker_exchange_write_enabled"] is False
    assert phase6["live_exchange_write_enabled"] is False


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
