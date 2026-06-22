"""Paper observability tests for v8."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.paper import (
    DeterministicReplayFeed,
    PaperObservabilityError,
    ReadOnlyShadowSoakConfig,
    run_readonly_shadow_soak,
    summarize_paper_run,
    synthetic_readonly_feed_events,
)
from examples.v8.summarize_paper_run import summarize_paper_run_cli


def test_paper_observability_healthy_run_has_no_critical_alerts(
    tmp_path: Path,
) -> None:
    run_dir = _healthy_run(tmp_path)

    result = summarize_paper_run(
        run_dir=run_dir,
        output_dir=tmp_path / "observability",
    )
    report = result.report

    assert report.run_id == "paper-observability-healthy"
    assert report.alert_severity_counts["critical"] == 0
    assert report.alert_count == 0
    assert report.operator_recommendation == "continue_paper_run"
    assert report.phase6_status == "approved_for_staged_live"
    assert report.feed_health_status == "passed"
    assert report.paper_only is True
    assert report.capital_at_risk is False

    _assert_required_outputs(result.artifact_paths)
    _assert_source_hashes(report, run_dir)
    markdown = result.artifact_paths["operator_summary"].read_text(encoding="utf-8")
    assert "paper-observability-healthy" in markdown
    assert "approved_for_staged_live" in markdown
    assert "continue_paper_run" in markdown


def test_paper_observability_degraded_run_alerts_on_safety_and_phase6(
    tmp_path: Path,
) -> None:
    run_dir = _degraded_run(tmp_path)

    result = summarize_paper_run(
        run_dir=run_dir,
        output_dir=tmp_path / "observability-degraded",
    )
    codes = _alert_codes(result.report)

    assert result.report.operator_recommendation == "blocked_fail_closed"
    assert result.report.phase6_status == "blocked_fail_closed"
    assert result.report.alert_severity_counts["critical"] >= 2
    assert "kill_switch_triggered" in codes
    assert "safety_reason_codes_present" in codes
    assert "phase6_blocked" in codes


def test_paper_observability_feed_gap_run_alerts_and_blocks(
    tmp_path: Path,
) -> None:
    run_dir = _feed_gap_run(tmp_path)

    result = summarize_paper_run(
        run_dir=run_dir,
        output_dir=tmp_path / "observability-gap",
    )
    codes = _alert_codes(result.report)

    assert result.report.operator_recommendation == "blocked_fail_closed"
    assert result.report.feed_health_status == "failed"
    assert result.report.phase6_status == "blocked_fail_closed"
    assert "feed_gap_breach" in codes
    assert "phase6_blocked" in codes


def test_paper_observability_paper_boundary_violation_is_critical(
    tmp_path: Path,
) -> None:
    run_dir = _healthy_run(tmp_path)
    summary_path = run_dir / "paper_run_summary.json"
    summary = _read_json(summary_path)
    summary["capital_at_risk"] = True
    _write_json(summary_path, summary)

    result = summarize_paper_run(
        run_dir=run_dir,
        output_dir=tmp_path / "observability-boundary",
    )
    codes = _alert_codes(result.report)

    assert result.report.operator_recommendation == "stop_paper_run"
    assert result.report.capital_at_risk is True
    assert "capital_risk_violation" in codes
    assert "artifact_hash_mismatch" in codes


def test_paper_observability_missing_required_artifact_fails_closed(
    tmp_path: Path,
) -> None:
    run_dir = _healthy_run(tmp_path)
    (run_dir / "feed_health_report.json").unlink()

    with pytest.raises(PaperObservabilityError, match="missing required"):
        summarize_paper_run(
            run_dir=run_dir,
            output_dir=tmp_path / "observability-missing",
        )


def test_paper_observability_outputs_are_deterministic(tmp_path: Path) -> None:
    run_dir = _healthy_run(tmp_path)

    first = summarize_paper_run(
        run_dir=run_dir,
        output_dir=tmp_path / "observability-first",
    )
    second = summarize_paper_run(
        run_dir=run_dir,
        output_dir=tmp_path / "observability-second",
    )

    for key in (
        "observability_report",
        "operator_summary",
        "alerts",
        "dashboard_summary",
        "periodic_metrics_csv",
    ):
        assert _sha256_file(first.artifact_paths[key]) == _sha256_file(
            second.artifact_paths[key]
        )


def test_paper_observability_cli_and_comparison_outputs(tmp_path: Path) -> None:
    healthy_dir = _healthy_run(tmp_path)
    degraded_dir = _degraded_run(tmp_path)

    console = summarize_paper_run_cli(
        run_dir=healthy_dir,
        output_dir=tmp_path / "observability-cli",
        compare_run_dir=degraded_dir,
    )
    comparison = _read_json(tmp_path / "observability-cli" / "paper_run_comparison.json")

    assert console["run_id"] == "paper-observability-healthy"
    assert console["operator_recommendation"] == "continue_paper_run"
    assert Path(str(console["operator_summary_path"])).exists()
    assert Path(str(console["observability_report_path"])).exists()
    assert comparison["left_run_id"] == "paper-observability-healthy"
    assert comparison["right_run_id"] == "paper-observability-degraded"
    assert comparison["phase6_status_change"] == (
        "approved_for_staged_live->blocked_fail_closed"
    )
    assert comparison["recommendation"] == "right_run_risk_increased"


def _healthy_run(tmp_path: Path) -> Path:
    result = run_readonly_shadow_soak(
        config=_config(
            tmp_path,
            run_id="paper-observability-healthy",
            duration_seconds=300,
        )
    )
    return result.output_dir


def _degraded_run(tmp_path: Path) -> Path:
    result = run_readonly_shadow_soak(
        config=_config(
            tmp_path,
            run_id="paper-observability-degraded",
            duration_seconds=900,
            inject_degradation=True,
        )
    )
    return result.output_dir


def _feed_gap_run(tmp_path: Path) -> Path:
    events = list(synthetic_readonly_feed_events(row_count=5))
    events[2] = replace(
        events[2],
        event_ts=events[1].event_ts + 180_000,
        received_ts=events[1].event_ts + 180_250,
    )
    events[3] = replace(
        events[3],
        event_ts=events[2].event_ts + 60_000,
        received_ts=events[2].event_ts + 60_250,
    )
    events[4] = replace(
        events[4],
        event_ts=events[3].event_ts + 60_000,
        received_ts=events[3].event_ts + 60_250,
    )
    result = run_readonly_shadow_soak(
        config=_config(
            tmp_path,
            run_id="paper-observability-gap",
            duration_seconds=600,
        ),
        feed=DeterministicReplayFeed(events=tuple(events)),
    )
    return result.output_dir


def _config(
    tmp_path: Path,
    *,
    run_id: str,
    duration_seconds: int,
    inject_degradation: bool = False,
) -> ReadOnlyShadowSoakConfig:
    return ReadOnlyShadowSoakConfig(
        run_id=run_id,
        output_dir=tmp_path / "runs",
        duration_seconds=duration_seconds,
        feed_event_interval_seconds=60,
        heartbeat_interval_seconds=30,
        summary_interval_seconds=120,
        inject_degradation=inject_degradation,
        overwrite_existing=True,
    )


def _assert_required_outputs(artifact_paths: dict[str, Path]) -> None:
    expected = {
        "observability_report": "paper_observability_report.json",
        "operator_summary": "paper_operator_summary.md",
        "alerts": "paper_alerts.jsonl",
        "dashboard_summary": "paper_dashboard_summary.json",
        "periodic_metrics_csv": "paper_periodic_metrics.csv",
    }
    for key, filename in expected.items():
        assert artifact_paths[key].name == filename
        assert artifact_paths[key].exists()


def _assert_source_hashes(report: Any, run_dir: Path) -> None:
    assert report.summary_sha256 == _sha256_file(run_dir / "paper_run_summary.json")
    assert report.bundle_sha256 == _sha256_file(run_dir / "paper_bundle_manifest.json")
    assert report.phase5_report_sha256 == _sha256_file(
        run_dir / "phase5_safety_layer_report.json"
    )
    phase6_path = next(run_dir.glob("phase6_cicd_pipeline_report_*.json"))
    assert report.phase6_report_sha256 == _sha256_file(phase6_path)
    for artifact_name, digest in report.source_artifact_hashes.items():
        if artifact_name == "phase6_report":
            path = phase6_path
        else:
            path = run_dir / {
                "paper_run_summary": "paper_run_summary.json",
                "paper_bundle_manifest": "paper_bundle_manifest.json",
                "feed_health_report": "feed_health_report.json",
                "phase5_report": "phase5_safety_layer_report.json",
                "paper_orders": "paper_orders.jsonl",
                "paper_fills": "paper_fills.jsonl",
                "paper_ledger": "paper_ledger.jsonl",
                "paper_positions": "paper_positions.json",
                "paper_pnl_report": "paper_pnl_report.json",
                "paper_soak_heartbeat": "paper_soak_heartbeat.jsonl",
                "paper_soak_periodic_summaries": (
                    "paper_soak_periodic_summaries.jsonl"
                ),
            }[artifact_name]
        assert digest == _sha256_file(path)


def _alert_codes(report: Any) -> set[str]:
    return {alert["code"] for alert in report.alerts}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
