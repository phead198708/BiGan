"""Polymarket expected-value execution tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from bigan.v8.polymarket import (
    ACTION_VALUE_LABEL_ACTIONS,
    PolymarketCorpusBuildConfig,
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
    build_polymarket_btc_corpus,
    build_polymarket_ev_decisions,
    decide_polymarket_ev_action,
    run_polymarket_policy_training,
    write_deterministic_polymarket_corpus_fixtures,
)


def test_ev_buy_up_uses_ask_price_not_mid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prediction = _prediction(probability=0.82, confidence=0.64)

    decision = decide_polymarket_ev_action(prediction=prediction, config=config)

    assert decision.action == "BUY_UP"
    assert decision.selected_outcome == "UP"
    assert decision.execution_price == prediction.features["up_ask"]
    assert decision.execution_price != prediction.features["up_mid"]
    assert decision.used_price_side == "ask"
    assert "ask_price_execution" in decision.reason_codes


def test_ev_buy_down_uses_ask_down_not_mid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prediction = _prediction(probability=0.18, confidence=0.64)

    decision = decide_polymarket_ev_action(prediction=prediction, config=config)

    assert decision.action == "BUY_DOWN"
    assert decision.selected_outcome == "DOWN"
    assert decision.execution_price == prediction.features["down_ask"]
    assert decision.execution_price != prediction.features["down_mid"]
    assert decision.used_price_side == "ask"


def test_sell_decision_uses_bid_price(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prediction = _prediction(probability=0.20, confidence=0.60)

    decision = decide_polymarket_ev_action(
        prediction=prediction,
        config=config,
        existing_position_up=1.5,
    )

    assert decision.action == "SELL_UP"
    assert decision.selected_outcome == "UP"
    assert decision.execution_price == prediction.features["up_bid"]
    assert decision.execution_price != prediction.features["up_mid"]
    assert decision.used_price_side == "bid"
    assert "bid_price_execution" in decision.reason_codes


def test_low_confidence_leads_to_no_trade(tmp_path: Path) -> None:
    config = PolymarketPolicyTrainingConfig(
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "policy",
        min_confidence=0.50,
    )
    prediction = _prediction(probability=0.55, confidence=0.10)

    decision = decide_polymarket_ev_action(prediction=prediction, config=config)

    assert decision.action == "NO_TRADE"
    assert decision.selected_outcome == "NO_TRADE"
    assert decision.paper_notional == 0.0
    assert "low_confidence" in decision.reason_codes


def test_action_value_output_is_preferred_over_probability_ev(tmp_path: Path) -> None:
    config = _config(tmp_path)
    action_returns = dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, -0.05)
    action_returns["NO_TRADE"] = 0.0
    action_returns["BUY_DOWN_HOLD_TO_SETTLEMENT"] = 0.08
    prediction = replace(
        _prediction(probability=0.95, confidence=0.90),
        p_up_auxiliary=0.95,
        expected_return_by_action=action_returns,
        expected_return_no_trade=action_returns["NO_TRADE"],
        expected_return_buy_up_hold_to_settlement=action_returns[
            "BUY_UP_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_down_hold_to_settlement=action_returns[
            "BUY_DOWN_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_up_sell_before_close=action_returns[
            "BUY_UP_SELL_BEFORE_CLOSE"
        ],
        expected_return_buy_down_sell_before_close=action_returns[
            "BUY_DOWN_SELL_BEFORE_CLOSE"
        ],
        best_policy_action="BUY_DOWN_HOLD_TO_SETTLEMENT",
        best_action_expected_return=0.08,
        second_best_action_expected_return=0.0,
        best_action_margin=0.08,
        policy_confidence=0.70,
        action_value_head_enabled=True,
        outcome_probability_head_enabled=True,
        action_value_model_family="feature_conditioned_action_return_model",
        feature_conditioned_action_value_model_enabled=True,
    )

    decision = decide_polymarket_ev_action(prediction=prediction, config=config)

    assert decision.action == "BUY_DOWN"
    assert decision.selected_outcome == "DOWN"
    assert decision.execution_price == prediction.features["down_ask"]
    assert decision.used_price_side == "ask"
    assert decision.best_policy_action == "BUY_DOWN_HOLD_TO_SETTLEMENT"
    assert decision.best_action_expected_return == 0.08
    assert decision.action_value_head_used is True
    assert decision.probability_ev_fallback_used is False
    assert "action_value_head_used" in decision.reason_codes


def test_sell_before_close_intent_triggers_planned_exit(tmp_path: Path) -> None:
    config = PolymarketPolicyTrainingConfig(
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "policy",
        ev_threshold=0.01,
        sell_before_close_exit_buffer_seconds=30,
    )
    action_returns = dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, -0.05)
    action_returns["BUY_UP_SELL_BEFORE_CLOSE"] = 0.08
    first = replace(
        _prediction(probability=0.80, confidence=0.80),
        p_up_auxiliary=0.80,
        expected_return_by_action=action_returns,
        expected_return_no_trade=action_returns["NO_TRADE"],
        expected_return_buy_up_hold_to_settlement=action_returns[
            "BUY_UP_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_down_hold_to_settlement=action_returns[
            "BUY_DOWN_HOLD_TO_SETTLEMENT"
        ],
        expected_return_buy_up_sell_before_close=action_returns[
            "BUY_UP_SELL_BEFORE_CLOSE"
        ],
        expected_return_buy_down_sell_before_close=action_returns[
            "BUY_DOWN_SELL_BEFORE_CLOSE"
        ],
        best_policy_action="BUY_UP_SELL_BEFORE_CLOSE",
        best_action_expected_return=0.08,
        second_best_action_expected_return=-0.05,
        best_action_margin=0.13,
        policy_confidence=0.80,
        action_value_head_enabled=True,
        outcome_probability_head_enabled=True,
        action_value_model_family="feature_conditioned_action_return_model",
        feature_conditioned_action_value_model_enabled=True,
    )
    second_features = dict(first.features)
    second_features["time_to_close_seconds"] = 25.0
    second = replace(
        first,
        decision_ts=95_000,
        features=second_features,
        expected_return_by_action=dict.fromkeys(ACTION_VALUE_LABEL_ACTIONS, 0.0),
        best_policy_action="NO_TRADE",
        best_action_expected_return=0.0,
        second_best_action_expected_return=0.0,
        best_action_margin=0.0,
    )

    decisions = build_polymarket_ev_decisions(predictions=(first, second), config=config)

    assert [decision.action for decision in decisions] == ["BUY_UP", "SELL_UP"]
    assert decisions[0].entry_policy_action == "BUY_UP_SELL_BEFORE_CLOSE"
    assert decisions[0].intended_exit_policy == "sell_before_close"
    assert decisions[0].planned_exit_before_ts == 91_000
    assert decisions[1].entry_policy_action == "BUY_UP_SELL_BEFORE_CLOSE"
    assert decisions[1].intended_exit_policy == "sell_before_close"
    assert decisions[1].planned_exit_before_ts == 91_000
    assert decisions[1].used_price_side == "bid"
    assert "planned_sell_before_close_exit" in decisions[1].reason_codes
    assert decisions[1].probability_ev_fallback_used is False


def test_policy_replay_uses_phase1_settlement_and_reports_pnl(
    tmp_path: Path,
) -> None:
    result = _run_training(tmp_path)
    replay = _read_json(result.artifact_paths["replay_report"])
    ev_report = _read_json(result.artifact_paths["ev_threshold_report"])

    assert replay["phase1_position_ledger_used"] is True
    assert replay["phase1_settlement_engine_used"] is True
    assert replay["calibration_split"] == "validation"
    assert replay["replay_split"] == "shadow"
    assert replay["out_of_sample_replay"] is True
    assert replay["replay_prediction_count"] == len(result.dataset.shadow_examples)
    assert replay["replay_decision_count"] == len(result.dataset.shadow_examples)
    assert replay["trade_count"] >= 0
    assert replay["settled_position_count"] >= 0
    assert "realized_trade_pnl" in replay
    assert "settlement_pnl" in replay
    assert "total_polymarket_pnl" in replay
    assert replay["critical_alert_count"] == 0
    assert ev_report["trained_model_used"] is True
    assert ev_report["synthetic_fixture_signal_used"] is False
    assert ev_report["policy_signal_source"] == "trained_model"
    assert ev_report["replay_split"] == "shadow"
    assert ev_report["out_of_sample_replay"] is True
    assert ev_report["primary_policy_target"] == "action_expected_net_return"
    assert ev_report["action_value_model_family"] == "feature_conditioned_action_return_model"
    assert ev_report["feature_conditioned_action_value_model_used"] is True
    assert ev_report["action_value_head_enabled"] is True
    assert ev_report["action_value_decision_count"] > 0
    assert (
        ev_report["action_value_decision_count"]
        + ev_report["probability_ev_fallback_decision_count"]
        == len(result.dataset.shadow_examples)
    )
    assert replay["outcome_calibration_error"] == replay["calibration_error"]
    assert replay["action_value_policy_metrics"]["primary_policy_target"] == (
        "action_expected_net_return"
    )
    assert replay["action_value_policy_metrics"]["action_value_model_family"] == (
        "feature_conditioned_action_return_model"
    )
    assert replay["action_value_policy_metrics"]["sample_count"] == ev_report[
        "action_value_decision_count"
    ]
    decisions = _read_jsonl(result.artifact_paths["ev_decisions"])
    shadow_keys = {
        (example.market_id, example.decision_ts)
        for example in result.dataset.shadow_examples
    }
    train_keys = {
        (example.market_id, example.decision_ts)
        for example in result.dataset.train_examples
    }
    decision_keys = {(row["market_id"], row["decision_ts"]) for row in decisions}
    assert decision_keys == shadow_keys
    assert not decision_keys & train_keys
    assert any(row["action_value_head_used"] is True for row in decisions)
    _assert_safe(replay)
    _assert_safe(ev_report)


def test_policy_replay_accepts_verified_outcome_without_reference_prices(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    corpus_dir = tmp_path / "corpus"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=corpus_dir,
        )
    )
    resolutions_path = corpus_dir / "polymarket_resolution_events.jsonl"
    resolutions = _read_jsonl(resolutions_path)
    for row in resolutions:
        row["reference_price_start"] = None
        row["reference_price_end"] = None
    _write_jsonl(resolutions_path, resolutions)

    result = run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "policy",
        )
    )

    replay = _read_json(result.artifact_paths["replay_report"])
    assert replay["settlement_resolution_source_counts"] == {
        "verified_outcome_payout_vector": replay["settlement_event_count"]
    }
    assert replay["phase1_position_ledger_used"] is True
    assert replay["phase1_settlement_engine_used"] is True
    _assert_safe(replay)


def _run_training(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    corpus_dir = tmp_path / "corpus"
    write_deterministic_polymarket_corpus_fixtures(raw_dir)
    build_polymarket_btc_corpus(
        PolymarketCorpusBuildConfig(
            input_dir=raw_dir,
            output_dir=corpus_dir,
        )
    )
    return run_polymarket_policy_training(
        PolymarketPolicyTrainingConfig(
            corpus_dir=corpus_dir,
            output_dir=tmp_path / "policy",
        )
    )


def _config(tmp_path: Path) -> PolymarketPolicyTrainingConfig:
    return PolymarketPolicyTrainingConfig(
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "policy",
        min_confidence=0.05,
        ev_threshold=0.01,
    )


def _prediction(*, probability: float, confidence: float) -> PolymarketPolicyPrediction:
    features = {
        "up_bid": 0.45,
        "up_ask": 0.61,
        "up_mid": 0.53,
        "down_bid": 0.44,
        "down_ask": 0.62,
        "down_mid": 0.53,
        "up_liquidity_depth": 1000.0,
        "down_liquidity_depth": 1000.0,
        "time_to_close_seconds": 120.0,
    }
    return PolymarketPolicyPrediction(
        market_id="m1",
        condition_id="0xcondition",
        slug="btc-test",
        market_family="btc_updown_15m",
        horizon_ms=900_000,
        decision_ts=1_000,
        estimated_up_probability=probability,
        confidence=confidence,
        score=0.0,
        calibration_bucket="0.8-0.9",
        model_version="test-model",
        feature_schema_hash="a" * 64,
        training_corpus_hash="b" * 64,
        features=features,
        target_up_probability=1.0,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _assert_safe(payload: dict) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
