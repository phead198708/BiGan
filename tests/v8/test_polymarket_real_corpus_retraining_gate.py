"""Real-corpus retraining gate tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bigan.v8.polymarket import (
    PolymarketRealCorpusRecorderConfig,
    PolymarketRealCorpusRetrainingGateConfig,
    record_polymarket_real_corpus,
    run_polymarket_real_corpus_retraining_gate,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256
from bigan.v8.polymarket.recorder.btc_reference import mock_btc_feature_candle_rows
from bigan.v8.polymarket.recorder.market_discovery import discover_mock_market_rows
from bigan.v8.polymarket.recorder.orderbook_state import mock_orderbook_rows, mock_trade_rows
from bigan.v8.polymarket.recorder.resolution import mock_resolution_rows


def test_real_corpus_retraining_gate_rejects_mock_recorder_bundle(tmp_path: Path) -> None:
    recorder = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="mock-recorder",
            output_dir=tmp_path / "recorder",
        )
    )

    result = run_polymarket_real_corpus_retraining_gate(
        PolymarketRealCorpusRetrainingGateConfig(
            recorder_run_dir=recorder.run_dir,
            output_dir=tmp_path / "gate",
        )
    )

    assert result.training_result is None
    assert result.model_manifest is None
    assert result.report["gate_status"] == "blocked_fail_closed"
    assert result.report["accepted_for_retraining"] is False
    assert result.report["training_completed"] is False
    assert "recorder_mock_public_data_used" in result.report["reason_codes"]
    assert "recorder_synthetic_public_data_used" in result.report["reason_codes"]
    assert "recorder_real_historical_training_not_eligible" in result.report["reason_codes"]
    assert result.report["real_historical_corpus_used"] is False
    assert result.report["manual_live_evidence_eligible"] is False
    assert result.artifact_paths["gate_report"].exists()
    assert result.artifact_paths["gate_manifest"].exists()


def test_real_corpus_retraining_gate_runs_training_for_real_eligible_bundle(
    tmp_path: Path,
) -> None:
    recorder = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="real-recorder",
            output_dir=tmp_path / "recorder",
            mock_public_data=False,
        ),
        public_provider=FakeRealPublicProvider(),
    )

    result = run_polymarket_real_corpus_retraining_gate(
        PolymarketRealCorpusRetrainingGateConfig(
            recorder_run_dir=recorder.run_dir,
            output_dir=tmp_path / "gate",
        )
    )

    assert result.training_result is not None
    assert result.model_manifest is not None
    assert result.report["gate_status"] == "completed"
    assert result.report["accepted_for_retraining"] is True
    assert result.report["training_completed"] is True
    assert result.report["reason_codes"] == []
    assert result.report["real_historical_corpus_used"] is True
    assert result.report["fixture_corpus_used"] is False
    assert result.report["synthetic_corpus_used"] is False
    assert result.report["synthetic_fixture_signal_used"] is False
    assert result.report["fixture_model_used"] is False
    assert result.report["manual_live_evidence_eligible"] is True

    model_manifest_path = result.training_result.artifact_paths["model_manifest"]
    model_manifest = _read_json(model_manifest_path)
    gate_manifest = _read_json(result.artifact_paths["gate_manifest"])

    assert model_manifest["real_historical_corpus_used"] is True
    assert model_manifest["fixture_corpus_used"] is False
    assert model_manifest["synthetic_corpus_used"] is False
    assert model_manifest["synthetic_fixture_signal_used"] is False
    assert model_manifest["fixture_model_used"] is False
    assert model_manifest["manual_live_evidence_eligible"] is True
    assert model_manifest["policy_signal_source"] == "trained_model"
    assert model_manifest["trained_model_used"] is True
    assert model_manifest["primary_policy_target"] == "action_expected_net_return"
    assert model_manifest["action_value_head_enabled"] is True
    assert model_manifest["outcome_probability_head_enabled"] is True
    assert model_manifest["action_value_model_family"] == "feature_conditioned_action_return_model"
    assert model_manifest["feature_conditioned_action_value_model_enabled"] is True
    assert model_manifest["direct_pnl_optimization"] is False
    assert model_manifest["action_value_paper_decision_eligible"] is False
    assert "action_value_calibration_missing" in model_manifest[
        "action_value_paper_decision_ineligible_reasons"
    ]
    assert model_manifest["training_corpus_hash"] == recorder.report[
        "phase2_corpus_manifest_sha256"
    ]
    assert model_manifest["phase2_corpus_manifest_sha256"] == recorder.report[
        "phase2_corpus_manifest_sha256"
    ]
    assert model_manifest["policy_dataset_hash"] == model_manifest["dataset_hash"]
    for field_name in (
        "model_sha256",
        "model_manifest_sha256",
        "policy_dataset_hash",
        "split_hash",
        "train_dataset_hash",
        "shadow_dataset_hash",
        "recorder_manifest_sha256",
        "recorder_report_sha256",
        "phase2_corpus_manifest_sha256",
        "phase2_train_shadow_split_sha256",
    ):
        value = result.report.get(field_name, model_manifest.get(field_name))
        assert looks_like_sha256(str(value)), field_name

    assert gate_manifest["accepted_for_retraining"] is True
    assert gate_manifest["real_historical_corpus_used"] is True
    assert gate_manifest["model_manifest"]["real_historical_corpus_used"] is True
    assert gate_manifest["model_manifest_sha256"] == result.report["model_manifest_sha256"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class FakeRealPublicProvider:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return [_as_real_public_market_row(row) for row in discover_mock_market_rows(config)]

    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return mock_orderbook_rows(markets, config)

    def trade_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        return mock_trade_rows(markets)

    def btc_feature_candle_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return mock_btc_feature_candle_rows(markets, config)

    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return mock_resolution_rows(markets, config)


def _as_real_public_market_row(row: dict[str, Any]) -> dict[str, Any]:
    market = dict(row)
    market["raw_market_sha256"] = canonical_json_sha256(
        {
            "market_id": market["market_id"],
            "family": market["market_family"],
            "source": "fake_real_public_provider",
        }
    )
    market["raw_public_payload"] = {
        "mock_public_data": False,
        "provider": "fake_real_public_provider",
    }
    return market
