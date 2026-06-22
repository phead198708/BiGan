"""24h-capable paper operator CLI tests for v8."""

from __future__ import annotations

import hashlib
import inspect
import json
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import bigan.v8.paper.operator_cli as operator_cli
from bigan.v8.paper import (
    DeterministicReplayFeed,
    PaperOperatorCLIError,
    PaperOperatorRunConfig,
    run_24h_paper_operator,
    synthetic_readonly_feed_events,
)
from examples.v8.run_24h_paper_operator import run_24h_paper_operator_cli


def test_24h_paper_operator_healthy_short_run_completes_end_to_end(
    tmp_path: Path,
) -> None:
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="operator-healthy", post_mode="gh_command"),
    )
    manifest = result.manifest

    assert result.operator_run_dir == tmp_path / "operator-healthy"
    assert result.paper_run_dir == result.operator_run_dir / "paper_run"
    assert result.observability_dir == result.operator_run_dir / "observability"
    assert result.github_comment_dir == result.operator_run_dir / "github_comment"
    assert manifest["status"] == "completed_continue_paper"
    assert manifest["paper_only"] is True
    assert manifest["capital_at_risk"] is False
    assert manifest["broker_exchange_write_enabled"] is False
    assert manifest["live_exchange_write_enabled"] is False
    assert manifest["operator_recommendation"] == "continue_paper_run"
    assert manifest["phase5_status"] == "passed"
    assert manifest["phase6_deployment_status"] == "approved_for_staged_live"
    assert manifest["feed_health_status"] == "passed"
    assert manifest["critical_alert_count"] == 0
    assert manifest["capital_deployment_allowed"] is False
    assert manifest["live_deployment_allowed"] is False
    _assert_operator_outputs(result.operator_run_dir)


def test_24h_paper_operator_degraded_run_blocks_fail_closed(
    tmp_path: Path,
) -> None:
    result = run_24h_paper_operator(
        config=_config(
            tmp_path,
            run_id="operator-degraded",
            duration_seconds=900,
            inject_degradation=True,
        )
    )
    manifest = result.manifest

    assert manifest["status"] == "completed_blocked_fail_closed"
    assert manifest["operator_recommendation"] == "blocked_fail_closed"
    assert manifest["phase5_kill_switch_triggered"] is True
    assert manifest["phase6_deployment_status"] == "blocked_fail_closed"
    assert manifest["critical_alert_count"] > 0
    assert "kill_switch_triggered" in manifest["reason_codes"]
    assert manifest["capital_deployment_allowed"] is False


def test_24h_paper_operator_feed_anomaly_produces_critical_feed_alert(
    tmp_path: Path,
) -> None:
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="operator-feed-gap", duration_seconds=600),
        feed=DeterministicReplayFeed(events=tuple(_feed_gap_events())),
    )
    manifest = result.manifest
    comment_body = result.comment_result.artifact_paths["comment_body"].read_text(
        encoding="utf-8"
    )

    assert manifest["status"] == "completed_blocked_fail_closed"
    assert manifest["feed_health_status"] == "failed"
    assert manifest["operator_recommendation"] == "blocked_fail_closed"
    assert "feed_gap_breach" in manifest["reason_codes"]
    assert "`feed_gap_breach`" in comment_body


def test_24h_paper_operator_stop_path_writes_outputs(
    tmp_path: Path,
) -> None:
    result = run_24h_paper_operator(
        config=_config(
            tmp_path,
            run_id="operator-stop",
            duration_seconds=1_200,
            stop_after_events=5,
        )
    )
    manifest = result.manifest

    assert manifest["status"] == "operator_stopped"
    assert manifest["stop_reason"] == "operator_stop"
    assert "operator_stop" in manifest["reason_codes"]
    assert manifest["paper_only"] is True
    assert manifest["capital_at_risk"] is False
    _assert_operator_outputs(result.operator_run_dir)


def test_24h_paper_operator_missing_intermediate_artifact_fails_closed(
    tmp_path: Path,
) -> None:
    def delete_feed_health(run_dir: Path) -> None:
        (run_dir / "feed_health_report.json").unlink()

    config = _config(tmp_path, run_id="operator-missing-artifact")

    with pytest.raises(PaperOperatorCLIError, match="observability"):
        operator_cli._run_24h_paper_operator_with_fault_injection_for_tests(
            config=config,
            _after_paper_run_fault_injection_hook_for_tests=delete_feed_health,
        )

    manifest = _read_json(config.manifest_path)
    assert manifest["status"] == "failed_fail_closed"
    assert manifest["reason_codes"] == ["observability_failed"]
    assert manifest["capital_deployment_allowed"] is False
    assert not config.observability_dir.exists()


def test_24h_paper_operator_public_api_has_no_artifact_mutation_hook() -> None:
    signature = inspect.signature(run_24h_paper_operator)

    assert "after_paper_run_hook" not in signature.parameters
    assert "_after_paper_run_fault_injection_hook_for_tests" not in signature.parameters


def test_24h_paper_operator_refuses_to_overwrite_existing_run(
    tmp_path: Path,
) -> None:
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="operator-no-overwrite")
    )
    summary_path = result.paper_run_dir / "paper_run_summary.json"
    original_hash = _sha256_file(summary_path)

    with pytest.raises(FileExistsError, match="operator run_dir already exists"):
        run_24h_paper_operator(
            config=_config(tmp_path, run_id="operator-no-overwrite")
        )

    assert summary_path.exists()
    assert _sha256_file(summary_path) == original_hash


def test_24h_paper_operator_gh_command_uses_absolute_body_file(
    tmp_path: Path,
) -> None:
    result = run_24h_paper_operator(
        config=_config(tmp_path, run_id="operator-gh", post_mode="gh_command")
    )
    command_path = result.comment_result.artifact_paths["gh_command"]
    command = command_path.read_text(encoding="utf-8").strip()
    parts = shlex.split(command)
    body_file = Path(parts[parts.index("--body-file") + 1])

    assert parts[:3] == ["gh", "issue", "comment"]
    assert "128" in parts
    assert "--repo" in parts
    assert "phead198708/BiGan" in parts
    assert body_file.is_absolute()
    assert body_file == result.comment_result.artifact_paths["comment_body"]
    assert body_file.exists()


def test_24h_paper_operator_outputs_are_deterministic(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        run_id="operator-deterministic",
        post_mode="gh_command",
        overwrite_existing=True,
    )
    first = run_24h_paper_operator(config=config)
    first_hashes = {
        "manifest": _sha256_file(first.manifest_path),
        "comment": _sha256_file(first.comment_result.artifact_paths["comment_body"]),
        "payload": _sha256_file(first.comment_result.artifact_paths["payload"]),
    }

    second = run_24h_paper_operator(config=config)

    assert first_hashes == {
        "manifest": _sha256_file(second.manifest_path),
        "comment": _sha256_file(second.comment_result.artifact_paths["comment_body"]),
        "payload": _sha256_file(second.comment_result.artifact_paths["payload"]),
    }


def test_24h_paper_operator_rejects_write_enabled_config(tmp_path: Path) -> None:
    with pytest.raises(PaperOperatorCLIError, match="broker/exchange writes"):
        _config(
            tmp_path,
            run_id="operator-unsafe-broker",
            broker_exchange_write_enabled=True,
        )

    with pytest.raises(PaperOperatorCLIError, match="live exchange writes"):
        _config(
            tmp_path,
            run_id="operator-unsafe-live",
            live_exchange_write_enabled=True,
        )


def test_24h_paper_operator_example_cli_returns_console_summary(
    tmp_path: Path,
) -> None:
    console = run_24h_paper_operator_cli(
        run_id="operator-cli",
        output_dir=tmp_path,
        repo="phead198708/BiGan",
        issue_number=128,
        mode="gh-command",
        duration_seconds=300,
        heartbeat_interval_seconds=30,
        summary_interval_seconds=120,
    )

    assert console["run_id"] == "operator-cli"
    assert console["operator_recommendation"] == "continue_paper_run"
    assert console["critical_alert_count"] == 0
    assert console["phase6_deployment_status"] == "approved_for_staged_live"
    assert console["paper_only"] is True
    assert console["capital_at_risk"] is False
    assert Path(str(console["paper_summary_path"])).exists()
    assert Path(str(console["comment_body_path"])).exists()
    assert Path(str(console["gh_command_path"])).exists()


def _config(
    output_dir: Path,
    *,
    run_id: str,
    post_mode: str = "dry_run",
    duration_seconds: int = 300,
    overwrite_existing: bool = False,
    stop_after_events: int | None = None,
    inject_degradation: bool = False,
    broker_exchange_write_enabled: bool = False,
    live_exchange_write_enabled: bool = False,
) -> PaperOperatorRunConfig:
    return PaperOperatorRunConfig(
        run_id=run_id,
        output_dir=output_dir,
        repo_full_name="phead198708/BiGan",
        issue_number=128,
        post_mode=post_mode,  # type: ignore[arg-type]
        duration_seconds=duration_seconds,
        feed_event_interval_seconds=60,
        heartbeat_interval_seconds=30,
        summary_interval_seconds=120,
        overwrite_existing=overwrite_existing,
        stop_after_events=stop_after_events,
        inject_degradation=inject_degradation,
        broker_exchange_write_enabled=broker_exchange_write_enabled,
        live_exchange_write_enabled=live_exchange_write_enabled,
    )


def _feed_gap_events() -> list[Any]:
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
    return events


def _assert_operator_outputs(operator_run_dir: Path) -> None:
    for path in (
        operator_run_dir / "paper_run" / "paper_run_summary.json",
        operator_run_dir / "paper_run" / "paper_bundle_manifest.json",
        operator_run_dir / "paper_run" / "phase5_safety_layer_report.json",
        operator_run_dir / "paper_run" / "feed_health_report.json",
        operator_run_dir / "observability" / "paper_observability_report.json",
        operator_run_dir / "observability" / "paper_operator_summary.md",
        operator_run_dir / "observability" / "paper_alerts.jsonl",
        operator_run_dir / "observability" / "paper_dashboard_summary.json",
        operator_run_dir / "observability" / "paper_periodic_metrics.csv",
        operator_run_dir / "github_comment" / "github_paper_comment_payload.json",
        operator_run_dir / "github_comment" / "github_paper_comment.md",
        operator_run_dir / "operator_run_manifest.json",
    ):
        assert path.exists(), path
    assert list((operator_run_dir / "paper_run").glob("phase6_cicd_pipeline_report_*.json"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
