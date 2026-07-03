"""GitHub paper alert delivery tests for v8."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import bigan.v8.paper.alert_delivery as alert_delivery
from bigan.v8.paper import (
    DeterministicReplayFeed,
    GitHubCommentDeliveryConfig,
    PaperAlertDeliveryError,
    ReadOnlyShadowSoakConfig,
    deliver_github_paper_comment,
    run_readonly_shadow_soak,
    summarize_paper_run,
    synthetic_readonly_feed_events,
)
from examples.v8.post_paper_observability_comment import (
    post_paper_observability_comment_cli,
)


def test_paper_alert_delivery_healthy_comment_continue_paper_run(
    tmp_path: Path,
) -> None:
    observability_dir = _healthy_observability(tmp_path)

    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(tmp_path / "comment-healthy", post_mode="dry_run"),
    )
    body = result.artifact_paths["comment_body"].read_text(encoding="utf-8")

    assert result.payload.operator_recommendation == "continue_paper_run"
    assert result.payload.critical_alert_count == 0
    assert result.payload.phase6_deployment_status == "approved_for_staged_live"
    assert result.payload.paper_only is True
    assert result.payload.capital_at_risk is False
    assert "Recommendation: continue_paper_run" in body
    assert "critical_alert_count | `0`" in body
    assert "phase6_deployment_status | `approved_for_staged_live`" in body
    assert "paper_only | `true`" in body
    assert "capital_at_risk | `false`" in body
    assert "safety_status" in body
    assert "Source Artifact Hashes" in body
    assert "github_paper_comment_gh_command.sh" not in {
        path.name for path in result.artifact_paths.values()
    }


def test_paper_alert_delivery_degraded_comment_blocks_promotion(
    tmp_path: Path,
) -> None:
    observability_dir = _degraded_observability(tmp_path)

    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(tmp_path / "comment-degraded", post_mode="dry_run"),
    )
    body = result.artifact_paths["comment_body"].read_text(encoding="utf-8")

    assert result.payload.operator_recommendation == "blocked_fail_closed"
    assert result.payload.phase6_deployment_status == "blocked_fail_closed"
    assert result.payload.critical_alert_count > 0
    assert "Do not promote to live trading" in body
    assert "phase6_deployment_status | `blocked_fail_closed`" in body
    assert "`kill_switch_triggered`" in body
    assert "`phase6_blocked`" in body


def test_paper_alert_delivery_feed_anomaly_includes_feed_alerts(
    tmp_path: Path,
) -> None:
    observability_dir = _feed_gap_observability(tmp_path)

    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(tmp_path / "comment-feed-gap", post_mode="dry_run"),
    )
    body = result.artifact_paths["comment_body"].read_text(encoding="utf-8")

    assert result.payload.feed_health_status == "failed"
    assert result.payload.operator_recommendation == "blocked_fail_closed"
    assert "`feed_gap_breach`" in body
    assert "feed_health_status | `failed`" in body


def test_paper_alert_delivery_never_omits_critical_alerts(
    tmp_path: Path,
) -> None:
    observability_dir = _healthy_observability(tmp_path)
    _overwrite_observability_alerts(
        observability_dir,
        [
            _alert("warning", "first_warning", "first warning"),
            _alert("warning", "second_warning", "second warning"),
            _alert("critical", "late_critical_stop", "late critical stop"),
        ],
    )

    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(
            tmp_path / "comment-critical-after-warning",
            post_mode="dry_run",
            max_alerts_to_inline=1,
        ),
    )
    body = result.artifact_paths["comment_body"].read_text(encoding="utf-8")

    assert "`late_critical_stop`" in body
    assert "late critical stop" in body
    assert "`first_warning`" in body
    assert "`second_warning`" not in body
    assert "`1` non-critical alerts omitted from inline summary" in body


def test_paper_alert_delivery_zero_inline_limit_still_includes_critical_alerts(
    tmp_path: Path,
) -> None:
    observability_dir = _healthy_observability(tmp_path)
    _overwrite_observability_alerts(
        observability_dir,
        [
            _alert("warning", "warning_before_critical", "warning before critical"),
            _alert("critical", "critical_survives_zero_limit", "critical survives"),
        ],
    )

    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(
            tmp_path / "comment-zero-inline",
            post_mode="dry_run",
            max_alerts_to_inline=0,
        ),
    )
    body = result.artifact_paths["comment_body"].read_text(encoding="utf-8")

    assert "`critical_survives_zero_limit`" in body
    assert "`warning_before_critical`" not in body
    assert "`1` non-critical alerts omitted from inline summary" in body


def test_paper_alert_delivery_boundary_violation_warns_do_not_promote(
    tmp_path: Path,
) -> None:
    observability_dir = _boundary_violation_observability(tmp_path)

    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(tmp_path / "comment-boundary", post_mode="dry_run"),
    )
    body = result.artifact_paths["comment_body"].read_text(encoding="utf-8")

    assert result.payload.operator_recommendation == "stop_paper_run"
    assert result.payload.paper_only is False
    assert result.payload.capital_at_risk is True
    assert "Do not promote to live trading" in body
    assert "`paper_only_missing`" in body
    assert "paper_only | `false`" in body
    assert "capital_at_risk | `true`" in body
    assert "- paper_only: `false` expected `true`" in body
    assert "- capital_at_risk: `true` expected `false`" in body
    assert "- real capital: `false`" not in body
    assert "- broker/exchange write path: `false`" not in body


def test_paper_alert_delivery_missing_observability_report_fails_closed(
    tmp_path: Path,
) -> None:
    observability_dir = _healthy_observability(tmp_path)
    (observability_dir / "paper_observability_report.json").unlink()
    output_dir = tmp_path / "comment-missing-report"

    with pytest.raises(PaperAlertDeliveryError, match="missing required"):
        deliver_github_paper_comment(
            observability_dir=observability_dir,
            config=_config(output_dir, post_mode="dry_run"),
        )
    assert not output_dir.exists()


def test_paper_alert_delivery_missing_operator_summary_fails_closed(
    tmp_path: Path,
) -> None:
    observability_dir = _healthy_observability(tmp_path)
    (observability_dir / "paper_operator_summary.md").unlink()
    output_dir = tmp_path / "comment-missing-summary"

    with pytest.raises(PaperAlertDeliveryError, match="missing required"):
        deliver_github_paper_comment(
            observability_dir=observability_dir,
            config=_config(output_dir, post_mode="dry_run"),
        )
    assert not output_dir.exists()


def test_paper_alert_delivery_gh_command_mode_writes_command(
    tmp_path: Path,
) -> None:
    observability_dir = _healthy_observability(tmp_path)

    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(tmp_path / "comment-command", post_mode="gh_command"),
    )
    command = result.artifact_paths["gh_command"].read_text(encoding="utf-8")
    payload = _read_json(result.artifact_paths["payload"])

    assert "gh issue comment 126" in command
    assert "--repo phead198708/BiGan" in command
    assert "--body-file" in command
    assert "github_paper_comment.md" in command
    assert payload["gh_command"] == command.strip()
    body_file = Path(shlex.split(command)[shlex.split(command).index("--body-file") + 1])
    assert body_file.is_absolute()
    assert body_file == result.artifact_paths["comment_body"]
    assert body_file.exists()
    assert payload["gh_command"] is not None
    assert payload["issue_number"] == 126
    assert payload["repo_full_name"] == "phead198708/BiGan"


def test_paper_alert_delivery_non_direct_modes_do_not_call_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observability_dir = _healthy_observability(tmp_path)

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(alert_delivery.subprocess, "run", fail_run)

    deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(tmp_path / "comment-dry-no-subprocess", post_mode="dry_run"),
    )
    deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(tmp_path / "comment-gh-no-subprocess", post_mode="gh_command"),
    )


def test_paper_alert_delivery_direct_comment_calls_gh_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observability_dir = _healthy_observability(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> SimpleNamespace:
        calls.append(
            {
                "command": command,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return SimpleNamespace(
            stdout="https://github.com/phead198708/BiGan/issues/126#issuecomment-1\n"
        )

    monkeypatch.setattr(alert_delivery.subprocess, "run", fake_run)

    result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(tmp_path / "comment-direct", post_mode="direct_comment"),
    )

    assert calls == [
        {
            "command": [
                "gh",
                "issue",
                "comment",
                "126",
                "--repo",
                "phead198708/BiGan",
                "--body-file",
                str(result.artifact_paths["comment_body"]),
            ],
            "check": True,
            "capture_output": True,
            "text": True,
        }
    ]
    receipt = _read_json(result.artifact_paths["delivery_receipt"])
    assert receipt == result.delivery_receipt
    assert receipt["comment_url"].endswith("#issuecomment-1")


def test_paper_alert_delivery_cli_outputs_console_summary(
    tmp_path: Path,
) -> None:
    observability_dir = _healthy_observability(tmp_path)

    console = post_paper_observability_comment_cli(
        observability_dir=observability_dir,
        repo="phead198708/BiGan",
        issue_number=126,
        output_dir=tmp_path / "comment-cli",
        mode="gh-command",
    )

    assert console["run_id"] == "paper-alert-delivery-healthy"
    assert console["issue_number"] == 126
    assert console["operator_recommendation"] == "continue_paper_run"
    assert console["critical_alert_count"] == 0
    assert Path(str(console["comment_body_path"])).exists()
    assert Path(str(console["gh_command_path"])).exists()


def test_paper_alert_delivery_outputs_are_deterministic(tmp_path: Path) -> None:
    observability_dir = _healthy_observability(tmp_path)
    output_dir = tmp_path / "comment-repeat"

    first = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(output_dir, post_mode="gh_command"),
    )
    first_hashes = {
        key: _sha256_file(first.artifact_paths[key])
        for key in ("payload", "comment_body", "gh_command")
    }
    second = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=_config(output_dir, post_mode="gh_command"),
    )

    for key in ("payload", "comment_body", "gh_command"):
        assert first_hashes[key] == _sha256_file(second.artifact_paths[key])


def _healthy_observability(tmp_path: Path) -> Path:
    return _observability_for_run(
        tmp_path,
        run_id="paper-alert-delivery-healthy",
        duration_seconds=300,
    )


def _degraded_observability(tmp_path: Path) -> Path:
    return _observability_for_run(
        tmp_path,
        run_id="paper-alert-delivery-degraded",
        duration_seconds=900,
        inject_degradation=True,
    )


def _feed_gap_observability(tmp_path: Path) -> Path:
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
    run = run_readonly_shadow_soak(
        config=_soak_config(
            tmp_path,
            run_id="paper-alert-delivery-feed-gap",
            duration_seconds=600,
        ),
        feed=DeterministicReplayFeed(events=tuple(events)),
    )
    return _summarize(tmp_path, run.output_dir, "paper-alert-delivery-feed-gap")


def _boundary_violation_observability(tmp_path: Path) -> Path:
    run = run_readonly_shadow_soak(
        config=_soak_config(
            tmp_path,
            run_id="paper-alert-delivery-boundary",
            duration_seconds=300,
        )
    )
    summary_path = run.output_dir / "paper_run_summary.json"
    summary = _read_json(summary_path)
    summary.pop("paper_only")
    summary["capital_at_risk"] = True
    _write_json(summary_path, summary)
    return _summarize(tmp_path, run.output_dir, "paper-alert-delivery-boundary")


def _observability_for_run(
    tmp_path: Path,
    *,
    run_id: str,
    duration_seconds: int,
    inject_degradation: bool = False,
) -> Path:
    run = run_readonly_shadow_soak(
        config=_soak_config(
            tmp_path,
            run_id=run_id,
            duration_seconds=duration_seconds,
            inject_degradation=inject_degradation,
        )
    )
    return _summarize(tmp_path, run.output_dir, run_id)


def _summarize(tmp_path: Path, run_dir: Path, run_id: str) -> Path:
    result = summarize_paper_run(
        run_dir=run_dir,
        output_dir=tmp_path / "observability" / run_id,
        overwrite_existing=True,
    )
    return result.output_dir


def _soak_config(
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


def _config(
    output_dir: Path,
    *,
    post_mode: str,
    max_alerts_to_inline: int = 20,
) -> GitHubCommentDeliveryConfig:
    return GitHubCommentDeliveryConfig(
        repo_full_name="phead198708/BiGan",
        issue_number=126,
        output_dir=output_dir,
        post_mode=post_mode,  # type: ignore[arg-type]
        max_alerts_to_inline=max_alerts_to_inline,
        overwrite_existing=True,
    )


def _overwrite_observability_alerts(
    observability_dir: Path,
    alerts: list[dict[str, str]],
) -> None:
    report_path = observability_dir / "paper_observability_report.json"
    report = _read_json(report_path)
    report["alert_count"] = len(alerts)
    report["alert_severity_counts"] = {
        "critical": sum(1 for alert in alerts if alert["severity"] == "critical"),
        "warning": sum(1 for alert in alerts if alert["severity"] == "warning"),
        "info": sum(1 for alert in alerts if alert["severity"] == "info"),
    }
    if report["alert_severity_counts"]["critical"] > 0:
        report["operator_recommendation"] = "blocked_fail_closed"
        report["phase6_status"] = "blocked_fail_closed"
    _write_json(report_path, report)
    _write_jsonl(observability_dir / "paper_alerts.jsonl", alerts)


def _alert(severity: str, code: str, message: str) -> dict[str, str]:
    return {
        "severity": severity,
        "category": "regression_fixture",
        "code": code,
        "message": message,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
