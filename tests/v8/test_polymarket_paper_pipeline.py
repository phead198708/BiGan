"""End-to-end mocked Polymarket BTC 15m paper pipeline tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from bigan.v8.paper import summarize_paper_run
from bigan.v8.polymarket import (
    PolymarketAdapterRunConfig,
    run_polymarket_btc15m_paper_pipeline,
)
from examples.v8.run_polymarket_btc15m_paper import (
    run_polymarket_btc15m_paper_cli,
)


def test_polymarket_mocked_pipeline_produces_observability_and_comment(
    tmp_path: Path,
) -> None:
    result = run_polymarket_btc15m_paper_pipeline(
        config=PolymarketAdapterRunConfig(
            run_id="polymarket-btc15m-pipeline",
            output_dir=tmp_path,
            comment_post_mode="gh_command",
        )
    )
    summary = result.adapter_summary
    paper_summary = _read_json(result.paper_run_dir / "paper_run_summary.json")
    observability = _read_json(
        result.observability_dir / "paper_observability_report.json"
    )
    operator_summary = (
        result.observability_dir / "paper_operator_summary.md"
    ).read_text(encoding="utf-8")
    comment_payload = _read_json(
        result.github_comment_dir / "github_paper_comment_payload.json"
    )
    comment_body = (
        result.github_comment_dir / "github_paper_comment.md"
    ).read_text(encoding="utf-8")
    decisions = _read_jsonl(
        result.adapter_dir / "polymarket_paper_decisions.jsonl"
    )

    assert summary["market_family"] == "btc_15m_up_down"
    assert summary["feature_row_count"] > 0
    assert summary["label_row_count"] == 2
    assert summary["trade_decision_count"] > 0
    assert summary["paper_only"] is True
    assert summary["capital_at_risk"] is False
    assert summary["polymarket_write_enabled"] is False
    assert summary["wallet_signing_enabled"] is False
    assert summary["policy_signal_source"] == "synthetic_fixture"
    assert summary["trained_model_used"] is False
    assert _sha256_file(result.adapter_dir / "polymarket_feature_rows.jsonl") == (
        summary["artifact_hashes"]["feature_rows"]
    )

    assert paper_summary["duration_seconds"] == 900
    assert paper_summary["configured_duration_seconds"] == 900
    assert paper_summary["started_at"] == _ts_to_utc(result.market.market_start_ts)
    assert paper_summary["ended_at"] == _ts_to_utc(result.market.market_end_ts)
    assert paper_summary["phase5_passed"] is True
    assert paper_summary["phase6_deployment_status"] == "approved_for_staged_live"
    assert paper_summary["feed_mode"] == "polymarket-mocked-btc15m"
    assert paper_summary["paper_only"] is True
    assert paper_summary["capital_at_risk"] is False
    assert paper_summary["broker_exchange_write_enabled"] is False
    assert paper_summary["live_exchange_write_enabled"] is False
    assert paper_summary["polymarket_write_enabled"] is False
    assert paper_summary["wallet_signing_enabled"] is False
    assert paper_summary["policy_signal_source"] == "synthetic_fixture"
    assert paper_summary["trained_model_used"] is False

    assert observability["operator_recommendation"] == "continue_paper_run"
    assert observability["phase6_status"] == "approved_for_staged_live"
    assert observability["paper_only"] is True
    assert observability["capital_at_risk"] is False
    assert observability["polymarket_write_enabled"] is False
    assert observability["wallet_signing_enabled"] is False
    assert "polymarket_write_enabled: `false`" in operator_summary
    assert "wallet_signing_enabled: `false`" in operator_summary

    assert comment_payload["operator_recommendation"] == "continue_paper_run"
    assert comment_payload["paper_only"] is True
    assert comment_payload["capital_at_risk"] is False
    assert comment_payload["polymarket_write_enabled"] is False
    assert comment_payload["wallet_signing_enabled"] is False
    assert "| polymarket_write_enabled | `false` |" in comment_body
    assert "| wallet_signing_enabled | `false` |" in comment_body
    assert (result.github_comment_dir / "github_paper_comment_gh_command.sh").exists()

    forbidden_fields = {"order_id", "private_key", "wallet_signature"}
    for decision in decisions:
        assert decision["paper_only"] is True
        assert decision["capital_at_risk"] is False
        assert decision["polymarket_write_enabled"] is False
        assert decision["wallet_signing_enabled"] is False
        assert forbidden_fields.isdisjoint(decision)


def test_polymarket_missing_or_enabled_safety_flags_are_critical(
    tmp_path: Path,
) -> None:
    result = run_polymarket_btc15m_paper_pipeline(
        config=PolymarketAdapterRunConfig(
            run_id="polymarket-boundary-alerts",
            output_dir=tmp_path,
        )
    )
    summary_path = result.paper_run_dir / "paper_run_summary.json"
    summary = _read_json(summary_path)
    summary.pop("polymarket_write_enabled")
    summary["wallet_signing_enabled"] = True
    _write_json(summary_path, summary)

    report = summarize_paper_run(
        run_dir=result.paper_run_dir,
        output_dir=tmp_path / "observability-mutated",
    ).report
    codes = {alert["code"] for alert in report.alerts}

    assert report.operator_recommendation == "stop_paper_run"
    assert report.polymarket_write_enabled is True
    assert report.wallet_signing_enabled is True
    assert "polymarket_write_flag_missing" in codes
    assert "wallet_signing_enabled" in codes


def test_polymarket_example_cli_is_deterministic(tmp_path: Path) -> None:
    first = run_polymarket_btc15m_paper_cli(
        run_id="polymarket-deterministic",
        output_dir=tmp_path,
        repo="phead198708/BiGan",
        issue_number=130,
        mode="dry-run",
        overwrite_existing=True,
    )
    first_summary = _sha256_file(Path(first["adapter_summary_path"]))
    first_paper = _sha256_file(Path(first["paper_run_summary_path"]))

    second = run_polymarket_btc15m_paper_cli(
        run_id="polymarket-deterministic",
        output_dir=tmp_path,
        repo="phead198708/BiGan",
        issue_number=130,
        mode="dry-run",
        overwrite_existing=True,
    )

    assert first_summary == _sha256_file(Path(second["adapter_summary_path"]))
    assert first_paper == _sha256_file(Path(second["paper_run_summary_path"]))


def test_polymarket_entrypoint_help_imports_from_script_path() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "examples/v8/run_polymarket_btc15m_paper.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--run-id" in completed.stdout
    assert "--mode" in completed.stdout


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ts_to_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
