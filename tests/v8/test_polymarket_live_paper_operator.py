"""Polymarket live paper operator tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bigan.v8.polymarket import (
    PolymarketCorpusBuildConfig,
    PolymarketLivePaperConfig,
    build_polymarket_btc_corpus,
    run_polymarket_live_paper,
)


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
        "rounds_index",
        "training_raw_index",
        "paper_audit_index",
        "paper_run_summary_latest",
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
    assert manifest["round_artifact_export_mode"] == "round_finalization_lifecycle"
    assert manifest["round_artifacts_written"] == 3
    assert manifest["training_raw_round_count"] == 3
    assert manifest["paper_audit_round_count"] == 3
    assert _looks_like_sha256(manifest["latest_round_summary_sha256"])
    assert _looks_like_sha256(manifest["latest_run_summary_sha256"])
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
    assert "round_artifact_export_mode" in comment
    for row in (*predictions, *decisions, *ledger):
        _assert_safe(row)

    rounds_index = _read_jsonl(result.artifact_paths["rounds_index"])
    training_index = _read_jsonl(result.artifact_paths["training_raw_index"])
    paper_audit_index = _read_jsonl(result.artifact_paths["paper_audit_index"])
    latest_summary = _read_json(result.artifact_paths["paper_run_summary_latest"])
    assert len(rounds_index) == 3
    assert len(training_index) == 3
    assert len(paper_audit_index) == 3
    assert latest_summary["rounds_seen"] == 3
    assert latest_summary["rounds_resolved"] == 3
    assert latest_summary["rounds_failed_closed"] == 0
    assert latest_summary["rounds_pending_resolution"] == 0
    for row in training_index:
        round_dir = result.run_dir / row["round_dir"]
        training_raw_dir = result.run_dir / row["training_raw_dir"]
        paper_audit_dir = result.run_dir / row["paper_audit_dir"]
        assert (round_dir / "round_summary.json").exists()
        assert (round_dir / "round_summary.md").exists()
        assert (round_dir / "run_summary_after_round.json").exists()
        assert (round_dir / "run_summary_after_round.md").exists()
        assert (training_raw_dir / "round_training_manifest.json").exists()
        assert (paper_audit_dir / "paper_audit_manifest.json").exists()
        _assert_round_summary_has_training_context(round_dir / "round_summary.json")
        _assert_training_manifest_has_provenance(
            training_raw_dir / "round_training_manifest.json",
            result.operator_manifest["model_manifest_sha256"],
            source_operator_run_id="healthy",
        )
        _assert_training_raw_is_model_output_free(training_raw_dir)
        assert _read_jsonl(paper_audit_dir / "polymarket_model_predictions.jsonl")
        assert _read_jsonl(paper_audit_dir / "polymarket_ev_decisions.jsonl")


def test_generated_round_training_raw_can_be_consumed_by_phase2_builder(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "phase2-consumption")
    training_index = _read_jsonl(result.artifact_paths["training_raw_index"])
    training_raw_dir = result.run_dir / training_index[0]["training_raw_dir"]
    training_manifest = _read_json(training_raw_dir / "round_training_manifest.json")

    corpus = build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=training_raw_dir,
            output_dir=tmp_path / "round_phase2_corpus",
            market_families=("btc_updown_5m", "btc_updown_15m", "btc_updown_1h"),
            overwrite_existing=True,
        )
    )

    assert corpus.manifest["feature_row_count"] > 0
    assert corpus.manifest["label_row_count"] > 0
    assert corpus.manifest["raw_artifact_hashes"] == training_manifest["artifact_hashes"]


def test_training_eligibility_is_round_local_when_run_fails_closed(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "round-local", inject_missing_token_book=True)
    manifest = result.operator_manifest
    rounds_index = _read_jsonl(result.artifact_paths["rounds_index"])
    training_index = _read_jsonl(result.artifact_paths["training_raw_index"])

    assert manifest["operator_status"] == "blocked_fail_closed"
    assert "missing_token_book" in manifest["critical_reason_codes"]
    assert len(rounds_index) == 3
    assert len(training_index) == 3
    assert all(row["round_training_eligible"] for row in training_index)
    first_round = rounds_index[0]
    assert first_round["expected_sample_count"] == 3
    assert first_round["complete_up_down_book_sample_count"] == 2
    assert first_round["incomplete_book_sample_count"] == 1
    assert first_round["training_eligibility_policy"] == "min_one_complete_book_sample"
    assert first_round["round_reason_codes"] == []


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
        "rounds_index",
        "training_raw_index",
        "paper_audit_index",
        "paper_run_summary_latest",
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _looks_like_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _assert_training_raw_is_model_output_free(training_raw_dir: Path) -> None:
    forbidden_fields = {
        "model_prediction",
        "model_probability",
        "paper_action",
        "paper_pnl",
        "selected_side",
        "edge",
        "ev_buy_up",
        "ev_buy_down",
        "estimated_up_probability",
        "p_up_auxiliary",
        "expected_return_by_action",
        "best_policy_action",
        "best_action_expected_return",
        "second_best_action_expected_return",
        "best_action_margin",
        "policy_confidence",
    }
    for path in training_raw_dir.glob("raw_*.jsonl"):
        for row in _read_jsonl(path):
            assert not (forbidden_fields & set(row)), path.name
            _assert_safe(row)
    manifest = _read_json(training_raw_dir / "round_training_manifest.json")
    assert manifest["phase2_raw_compatible"] is True
    assert manifest["training_eligible"] is True
    assert manifest["round_training_eligible"] is True
    assert manifest["training_eligibility_policy"] == "min_one_complete_book_sample"
    for field in forbidden_fields:
        assert field in manifest["excluded_audit_fields"]


def _assert_round_summary_has_training_context(path: Path) -> None:
    summary = _read_json(path)
    assert summary["round_training_eligible"] is True
    assert summary["round_reason_codes"] == []
    assert summary["round_feed_health"]["reason_codes"] == []
    assert summary["round_resolution_health"]["reason_codes"] == []
    assert summary["training_eligibility_policy"] == "min_one_complete_book_sample"
    assert summary["expected_sample_count"] == 3
    assert summary["complete_up_down_book_sample_count"] == 3
    assert summary["incomplete_book_sample_count"] == 0
    assert summary["book_coverage_ratio"] == 1.0
    assert summary["first_complete_book_ts"] < summary["last_complete_book_ts"]


def _assert_training_manifest_has_provenance(
    path: Path,
    model_manifest_sha256: str,
    *,
    source_operator_run_id: str,
) -> None:
    manifest = _read_json(path)
    for field in (
        "source_operator_run_id",
        "source_round_id",
        "source_market_id",
        "source_model_run_id",
        "source_model_manifest_sha256",
        "source_collection_mode",
        "live_polymarket_data",
        "live_btc_reference_data",
        "deterministic_replay",
    ):
        assert field in manifest
    assert manifest["source_operator_run_id"] == source_operator_run_id
    assert manifest["source_round_id"] == manifest["round_id"]
    assert manifest["source_market_id"] == manifest["market_id"]
    assert manifest["source_model_manifest_sha256"] == model_manifest_sha256
    assert manifest["source_collection_mode"] == "mock_live"
    assert manifest["live_polymarket_data"] is False
    assert manifest["live_btc_reference_data"] is False
    assert manifest["deterministic_replay"] is True
    assert manifest["training_eligibility_policy"] == "min_one_complete_book_sample"
    assert manifest["expected_sample_count"] == 3
    assert manifest["complete_up_down_book_sample_count"] == 3
    assert manifest["incomplete_book_sample_count"] == 0
    assert manifest["book_coverage_ratio"] == 1.0


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["broker_exchange_write_enabled"] is False
    assert payload["live_exchange_write_enabled"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
