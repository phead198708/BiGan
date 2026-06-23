"""Polymarket live paper operator tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bigan.v8.polymarket import PolymarketLivePaperConfig, run_polymarket_live_paper


def test_live_paper_operator_writes_required_artifacts_and_comment(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "healthy")

    expected = {
        "live_market_metadata",
        "live_token_orderbooks",
        "live_token_trades",
        "live_btc_reference_ticks",
        "live_btc_reference_candles",
        "polymarket_model_predictions",
        "polymarket_ev_decisions",
        "polymarket_position_ledger",
        "polymarket_settlement_events",
        "polymarket_pnl_breakdown",
        "polymarket_live_operator_manifest",
        "paper_observability_report",
        "paper_operator_summary",
        "github_paper_comment_payload",
        "github_paper_comment_md",
    }
    assert set(result.artifact_paths) == expected
    for name, path in result.artifact_paths.items():
        assert path.exists(), name

    manifest = result.operator_manifest
    assert manifest["operator_status"] == "completed"
    assert manifest["operator_recommendation"] == "continue_paper_run"
    assert manifest["critical_alert_count"] == 0
    assert manifest["live_polymarket_data"] is False
    assert manifest["live_binance_reference_data"] is False
    assert manifest["deterministic_replay"] is True
    assert manifest["prediction_count"] == 9
    assert manifest["decision_count"] == 9
    assert manifest["trade_count"] > 0
    assert manifest["resolved_market_count"] == 3
    assert manifest["unresolved_market_count"] == 0
    assert manifest["capital_deployment_allowed"] is False
    assert manifest["live_deployment_allowed"] is False
    _assert_safe(manifest)

    predictions = _read_jsonl(result.artifact_paths["polymarket_model_predictions"])
    decisions = _read_jsonl(result.artifact_paths["polymarket_ev_decisions"])
    ledger = _read_jsonl(result.artifact_paths["polymarket_position_ledger"])
    comment = result.artifact_paths["github_paper_comment_md"].read_text(encoding="utf-8")
    assert len(predictions) == 9
    assert len(decisions) == 9
    assert ledger
    assert "realized_trade_pnl" in comment
    assert "settlement_pnl" in comment
    assert "total_polymarket_pnl" in comment
    for row in (*predictions, *decisions, *ledger):
        _assert_safe(row)


def test_missing_and_stale_inputs_fail_closed(tmp_path: Path) -> None:
    scenarios = {
        "missing-rule": {"inject_missing_market_rule": True, "code": "missing_market_rule"},
        "missing-book": {"inject_missing_token_book": True, "code": "missing_token_book"},
        "stale-book": {"inject_stale_orderbook": True, "code": "stale_orderbook"},
        "stale-reference": {
            "inject_stale_reference": True,
            "code": "stale_reference_price",
        },
        "model-mismatch": {
            "inject_model_manifest_mismatch": True,
            "code": "model_manifest_mismatch",
        },
    }

    for name, params in scenarios.items():
        result = _run(tmp_path, name, **{k: v for k, v in params.items() if k != "code"})
        manifest = result.operator_manifest
        assert manifest["operator_status"] == "blocked_fail_closed", name
        assert manifest["operator_recommendation"] == "blocked_fail_closed", name
        assert params["code"] in manifest["critical_reason_codes"]
        assert manifest["critical_alert_count"] > 0
        assert manifest["capital_deployment_allowed"] is False
        assert manifest["live_deployment_allowed"] is False
        _assert_safe(manifest)


def test_stop_path_writes_manifest_and_artifacts(tmp_path: Path) -> None:
    result = _run(tmp_path, "operator-stop", stop_requested=True)
    manifest = result.operator_manifest

    assert manifest["operator_status"] == "operator_stopped"
    assert manifest["operator_recommendation"] == "stop_paper_run"
    assert result.artifact_paths["polymarket_live_operator_manifest"].exists()
    assert result.artifact_paths["github_paper_comment_payload"].exists()
    assert manifest["capital_deployment_allowed"] is False
    assert manifest["live_deployment_allowed"] is False


def test_mock_live_operator_is_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", "deterministic")
    second = _run(tmp_path / "second", "deterministic")

    for artifact_name in (
        "polymarket_model_predictions",
        "polymarket_ev_decisions",
        "polymarket_pnl_breakdown",
        "paper_observability_report",
        "github_paper_comment_payload",
    ):
        assert _sha256(first.artifact_paths[artifact_name]) == _sha256(
            second.artifact_paths[artifact_name]
        )


def _run(tmp_path: Path, run_id: str, **overrides):
    return run_polymarket_live_paper(
        PolymarketLivePaperConfig(
            run_id=run_id,
            output_dir=tmp_path,
            overwrite_existing=True,
            **overrides,
        )
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["broker_exchange_write_enabled"] is False
    assert payload["live_exchange_write_enabled"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
