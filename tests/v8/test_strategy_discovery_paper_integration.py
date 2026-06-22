"""Strategy Discovery to paper pipeline integration tests for v8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bigan.v8.strategy_discovery import (
    StrategyCandidate,
    StrategyCandidateReplayConfig,
    build_strategy_candidate_manifest,
    run_strategy_candidate_replay_batch,
)
from examples.v8.run_strategy_candidate_replay import (
    run_strategy_candidate_replay_cli,
)


def test_strategy_candidate_manifest_is_deterministic() -> None:
    candidate = _candidate("candidate_alpha")
    manifest = build_strategy_candidate_manifest(
        candidate=candidate,
        paper_pipeline_config={"duration_seconds": 300},
    )
    second = build_strategy_candidate_manifest(
        candidate=candidate,
        paper_pipeline_config={"duration_seconds": 300},
    )

    assert manifest.candidate_id == "candidate_alpha"
    assert manifest.candidate_sha256 == candidate.candidate_sha256
    assert manifest.manifest_sha256 == second.manifest_sha256
    assert manifest.to_dict()["manifest_sha256"] == manifest.manifest_sha256
    assert manifest.input_artifact_hashes == {
        "dataset_contract_sha256": _sha("candidate_alpha-dataset"),
        "feature_contract_sha256": _sha("candidate_alpha-features"),
        "source_artifact_sha256": _sha("candidate_alpha-source"),
    }


def test_strategy_candidate_replay_healthy_ready_for_manual_review(
    tmp_path: Path,
) -> None:
    result = run_strategy_candidate_replay_batch(
        candidates=[_candidate("candidate_healthy")],
        config=_config(tmp_path / "healthy", post_mode="gh_command"),
    )
    summary = result.candidate_summaries[0]
    candidate_dir = result.output_dir / "candidate_healthy"

    assert result.console_summary["candidate_count"] == 1
    assert result.console_summary["ready_for_manual_review_count"] == 1
    assert result.batch_manifest["paper_only"] is True
    assert result.batch_manifest["capital_at_risk"] is False
    assert summary["status"] == "ready_for_manual_review"
    assert summary["operator_recommendation"] == "continue_paper_run"
    assert summary["phase5_passed"] is True
    assert summary["phase5_kill_switch_triggered"] is False
    assert summary["phase6_deployment_status"] == "approved_for_staged_live"
    assert summary["critical_alert_count"] == 0
    assert summary["paper_only"] is True
    assert summary["capital_at_risk"] is False
    _assert_candidate_outputs(candidate_dir, expect_gh_command=True)
    _assert_required_hashes(summary)


def test_strategy_candidate_degraded_blocks_and_ranking_demotes(
    tmp_path: Path,
) -> None:
    healthy = _candidate("candidate_rank_healthy")
    degraded = _candidate(
        "candidate_rank_degraded",
        policy_config={"inject_degradation": True, "duration_seconds": 900},
    )

    result = run_strategy_candidate_replay_batch(
        candidates=[degraded, healthy],
        config=_config(tmp_path / "ranking", post_mode="dry_run"),
    )
    summaries = {summary["candidate_id"]: summary for summary in result.candidate_summaries}
    ranked_ids = [row["candidate_id"] for row in result.ranking["candidates"]]

    assert summaries["candidate_rank_degraded"]["status"] in {
        "phase5_blocked",
        "phase6_blocked_fail_closed",
        "observability_critical",
    }
    assert summaries["candidate_rank_degraded"]["critical_alert_count"] > 0
    assert result.batch_manifest["blocked_count"] == 1
    assert result.batch_manifest["critical_alert_candidate_count"] == 1
    assert ranked_ids[0] == "candidate_rank_healthy"
    assert ranked_ids[-1] == "candidate_rank_degraded"


def test_strategy_candidate_invalid_retained_in_batch_summary(
    tmp_path: Path,
) -> None:
    invalid = _candidate_payload("candidate_invalid")
    invalid.pop("strategy_name")

    result = run_strategy_candidate_replay_batch(
        candidates=[invalid, _candidate("candidate_valid_after_invalid")],
        config=_config(tmp_path / "invalid"),
    )
    summaries = {summary["candidate_id"]: summary for summary in result.candidate_summaries}

    assert result.batch_manifest["candidate_count"] == 2
    assert result.batch_manifest["invalid_count"] == 1
    assert summaries["candidate_invalid"]["status"] == "candidate_invalid"
    assert summaries["candidate_invalid"]["critical_alert_count"] == 1
    assert "candidate_invalid" in summaries["candidate_invalid"]["reason_codes"]
    assert (result.output_dir / "candidate_invalid" / "candidate_replay_summary.json").exists()
    assert summaries["candidate_valid_after_invalid"]["status"] == (
        "ready_for_manual_review"
    )


def test_strategy_candidate_unsafe_write_flags_are_rejected_and_retained(
    tmp_path: Path,
) -> None:
    unsafe = _candidate_payload("candidate_unsafe")
    unsafe["broker_exchange_write_enabled"] = True

    result = run_strategy_candidate_replay_batch(
        candidates=[unsafe],
        config=_config(tmp_path / "unsafe"),
    )
    summary = result.candidate_summaries[0]

    assert summary["candidate_id"] == "candidate_unsafe"
    assert summary["status"] == "candidate_invalid"
    assert summary["broker_exchange_write_enabled"] is True
    assert "broker/exchange write flag is forbidden" in summary["error_message"]
    assert result.batch_manifest["blocked_count"] == 1


def test_strategy_candidate_replay_outputs_are_deterministic(tmp_path: Path) -> None:
    candidates = [_candidate("candidate_deterministic")]
    config = _config(
        tmp_path / "deterministic",
        post_mode="gh_command",
        overwrite_existing=True,
    )

    first = run_strategy_candidate_replay_batch(candidates=candidates, config=config)
    first_hashes = _batch_hashes(first)
    second = run_strategy_candidate_replay_batch(candidates=candidates, config=config)

    assert first_hashes == _batch_hashes(second)


def test_strategy_candidate_replay_cli_loads_jsonl_and_prints_summary(
    tmp_path: Path,
) -> None:
    candidate_file = tmp_path / "candidates.jsonl"
    _write_jsonl(candidate_file, [_candidate_payload("candidate_cli")])

    console = run_strategy_candidate_replay_cli(
        candidate_file=candidate_file,
        output_dir=tmp_path / "cli-batch",
        repo="phead198708/BiGan",
        issue_number=127,
        mode="gh-command",
        duration_seconds=300,
    )

    assert console["batch_id"] == "strategy_candidate_batch_001"
    assert console["candidate_count"] == 1
    assert console["ready_for_manual_review_count"] == 1
    assert console["blocked_count"] == 0
    assert Path(str(console["ranking_path"])).exists()
    assert Path(str(console["batch_summary_path"])).exists()


def _config(
    output_dir: Path,
    *,
    post_mode: str = "dry_run",
    overwrite_existing: bool = False,
) -> StrategyCandidateReplayConfig:
    return StrategyCandidateReplayConfig(
        batch_id="strategy-candidate-test-batch",
        output_dir=output_dir,
        repo_full_name="phead198708/BiGan",
        issue_number=127,
        post_mode=post_mode,  # type: ignore[arg-type]
        duration_seconds=300,
        overwrite_existing=overwrite_existing,
    )


def _candidate(
    candidate_id: str,
    *,
    policy_config: dict[str, Any] | None = None,
    execution_config: dict[str, Any] | None = None,
    risk_config: dict[str, Any] | None = None,
) -> StrategyCandidate:
    return StrategyCandidate(**_candidate_payload(
        candidate_id,
        policy_config=policy_config,
        execution_config=execution_config,
        risk_config=risk_config,
    ))


def _candidate_payload(
    candidate_id: str,
    *,
    policy_config: dict[str, Any] | None = None,
    execution_config: dict[str, Any] | None = None,
    risk_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "candidate_family": "fixture_family",
        "strategy_name": f"{candidate_id}_strategy",
        "created_at": "2026-06-22T07:00:00Z",
        "source": "deterministic_fixture",
        "source_commit_sha": "abc1234",
        "source_artifact_sha256": _sha(f"{candidate_id}-source"),
        "feature_contract_sha256": _sha(f"{candidate_id}-features"),
        "dataset_contract_sha256": _sha(f"{candidate_id}-dataset"),
        "policy_config": dict(policy_config or {}),
        "execution_config": dict(execution_config or {}),
        "risk_config": dict(risk_config or {}),
        "expected_instruments": ["btc-up"],
        "expected_regime_keys": ["trend", "range", "high_volatility"],
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
    }


def _assert_candidate_outputs(candidate_dir: Path, *, expect_gh_command: bool) -> None:
    expected_paths = (
        candidate_dir / "candidate_manifest.json",
        candidate_dir / "candidate_replay_summary.json",
        candidate_dir / "paper_run" / "paper_run_summary.json",
        candidate_dir / "paper_run" / "paper_bundle_manifest.json",
        candidate_dir / "paper_run" / "phase5_safety_layer_report.json",
        candidate_dir / "paper_run" / "feed_health_report.json",
        candidate_dir / "observability" / "paper_observability_report.json",
        candidate_dir / "observability" / "paper_operator_summary.md",
        candidate_dir / "observability" / "paper_alerts.jsonl",
        candidate_dir / "observability" / "paper_dashboard_summary.json",
        candidate_dir / "github_comment" / "github_paper_comment_payload.json",
        candidate_dir / "github_comment" / "github_paper_comment.md",
    )
    for path in expected_paths:
        assert path.exists(), path
    assert list((candidate_dir / "paper_run").glob("phase6_cicd_pipeline_report_*.json"))
    if expect_gh_command:
        assert (
            candidate_dir / "github_comment" / "github_paper_comment_gh_command.sh"
        ).exists()


def _assert_required_hashes(summary: dict[str, Any]) -> None:
    artifact_hashes = summary["artifact_hashes"]
    for key in (
        "candidate_manifest",
        "paper_run_summary",
        "paper_bundle_manifest",
        "phase5_report",
        "phase6_report",
        "observability_report",
        "operator_summary",
        "github_comment_payload",
    ):
        assert key in artifact_hashes
        assert len(artifact_hashes[key]) == 64


def _batch_hashes(result: Any) -> dict[str, str]:
    candidate_dir = result.output_dir / "candidate_deterministic"
    return {
        "batch_manifest": _sha256_file(result.artifact_paths["batch_manifest"]),
        "ranking": _sha256_file(result.artifact_paths["ranking_json"]),
        "candidate_summary": _sha256_file(
            candidate_dir / "candidate_replay_summary.json"
        ),
        "candidate_manifest": _sha256_file(candidate_dir / "candidate_manifest.json"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
