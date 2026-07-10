from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2 import (
    EXECUTION_LAYER_V2_BASELINE_NAME,
    ExecutionLayerV2BacktestConfig,
    ExecutionLayerV2Config,
    ExecutionLayerV2Engine,
    ExecutionLayerV2Position,
    ExecutionLayerV2Signal,
    binary_kelly_fraction,
    build_execution_layer_v2_report,
    build_execution_layer_v2_report_from_rows,
    decide_execution_layer_v2,
    run_execution_layer_v2_backtest,
    time_decayed_kelly_notional,
)
from bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal import (
    ONE_HOUR_REMAP_PAPER_GOAL_SCHEMA_VERSION,
    ExecutionLayerV2OneHourRemapPaperGoalConfig,
    _missing_bet_round_classifications,
    _settlement_resolution_report,
    run_execution_layer_v2_one_hour_remap_paper_goal,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    ExecutionLayerV2ForwardShadowConfig,
    ExecutionLayerV2HTSRegimeRiskReplayConfig,
    ExecutionLayerV2PolicyReplayConfig,
    ExecutionLayerV2RegimeEntryEdgeReplayConfig,
    run_execution_layer_v2_forward_shadow_policy,
    run_execution_layer_v2_hts_regime_risk_replay,
    run_execution_layer_v2_policy_replay_from_settlement_csv,
    run_execution_layer_v2_regime_entry_edge_replay,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    _fresh_public_row_from_provider_feature_context,
)
from tests.v8.test_polymarket_post_freeze_holdout import (
    _build_issue160_unlock_fixture,
    _paper_fresh_public_row,
)


def _signal(
    *,
    market_id: str = "btc-updown-5m-v2",
    decision_ts: int = 1_000,
    p_up: float = 0.70,
    ask_up: float = 0.50,
    ask_down: float = 0.52,
    bid_up: float = 0.49,
    bid_down: float = 0.51,
    time_to_expiry_seconds: float = 240.0,
) -> ExecutionLayerV2Signal:
    return ExecutionLayerV2Signal(
        market_id=market_id,
        decision_ts=decision_ts,
        p_up=p_up,
        ask_up=ask_up,
        ask_down=ask_down,
        bid_up=bid_up,
        bid_down=bid_down,
        time_to_expiry_seconds=time_to_expiry_seconds,
        source_signal_id=f"{market_id}-{decision_ts}",
    )


def test_execution_layer_v2_enters_positive_ev_with_time_decayed_kelly() -> None:
    decision = decide_execution_layer_v2(signal=_signal())

    assert decision.action == "ENTER_POSITION"
    assert decision.target_side == "UP"
    assert decision.state_transition == "NO_POSITION->ACTIVE"
    assert decision.selected_ev_t == pytest.approx(0.199)
    assert decision.paper_notional > 0.0
    assert decision.paper_notional <= 500.0
    assert decision.kelly_fraction == pytest.approx(binary_kelly_fraction(0.70, 0.50))
    assert "time_decayed_kelly_sizing" in decision.reason_codes
    assert decision.paper_only is True
    assert decision.capital_at_risk is False
    assert decision.v8_execution_handoff_allowed is False


def test_execution_layer_v2_skips_low_ev_entry() -> None:
    decision = decide_execution_layer_v2(
        signal=_signal(p_up=0.51, ask_up=0.52, ask_down=0.51)
    )

    assert decision.action == "NO_ACTION"
    assert decision.target_side == "NONE"
    assert "entry_ev_threshold_not_met" in decision.reason_codes
    assert decision.paper_notional == 0.0


def test_execution_layer_v2_holds_when_ev_t_above_entry_floor() -> None:
    engine = ExecutionLayerV2Engine()
    first = engine.decide(_signal(decision_ts=1_000, p_up=0.70))
    second = engine.decide(_signal(decision_ts=1_060, p_up=0.68))

    assert first.action == "ENTER_POSITION"
    assert second.action == "HOLD_POSITION"
    assert second.state_transition == "ACTIVE->ACTIVE"
    assert second.ev_ratio_to_entry > 0.60
    assert "ev_t_above_hold_floor" in second.reason_codes


def test_execution_layer_v2_exits_when_ev_t_decays_below_floor() -> None:
    engine = ExecutionLayerV2Engine()
    engine.decide(_signal(decision_ts=1_000, p_up=0.70))
    second = engine.decide(_signal(decision_ts=1_060, p_up=0.57))

    assert second.action == "EXIT_POSITION"
    assert second.state_transition == "ACTIVE->EXIT"
    assert second.ev_ratio_to_entry < 0.60
    assert "ev_t_decayed_below_hold_floor" in second.reason_codes
    assert engine.positions == {}


def test_execution_layer_v2_exits_on_time_to_expiry_threshold() -> None:
    engine = ExecutionLayerV2Engine()
    engine.decide(_signal(decision_ts=1_000, p_up=0.70, time_to_expiry_seconds=180.0))
    second = engine.decide(_signal(decision_ts=1_060, p_up=0.80, time_to_expiry_seconds=30.0))

    assert second.action == "EXIT_POSITION"
    assert "time_to_expiry_exit_threshold_crossed" in second.reason_codes


def test_execution_layer_v2_rotates_when_opposite_signal_is_stronger() -> None:
    engine = ExecutionLayerV2Engine()
    engine.decide(_signal(decision_ts=1_000, p_up=0.70))
    second = engine.decide(
        _signal(
            decision_ts=1_060,
            p_up=0.40,
            ask_up=0.58,
            ask_down=0.45,
            bid_up=0.57,
            bid_down=0.44,
        )
    )

    assert second.action == "ROTATE_POSITION"
    assert second.target_side == "DOWN"
    assert second.state_transition == "ACTIVE->ACTIVE"
    assert "opposite_signal_ev_margin_crossed" in second.reason_codes
    assert engine.positions["btc-updown-5m-v2"].side == "DOWN"


def test_time_decayed_kelly_sizing_uses_issue_166_decay_formula() -> None:
    config = ExecutionLayerV2Config(
        nav_usdc=1_000.0,
        max_nav_fraction_per_position=1.0,
        kelly_time_decay_lambda=0.001,
    )
    short = time_decayed_kelly_notional(
        probability=0.60,
        price=0.50,
        time_to_expiry_seconds=60.0,
        config=config,
    )
    long = time_decayed_kelly_notional(
        probability=0.60,
        price=0.50,
        time_to_expiry_seconds=600.0,
        config=config,
    )

    assert short["time_decay_multiplier"] == pytest.approx(0.9417645335842487)
    assert short["paper_notional"] > long["paper_notional"]


def test_execution_layer_v2_report_includes_baseline_and_fail_closed_safety() -> None:
    report = build_execution_layer_v2_report(
        [
            _signal(market_id="m1", decision_ts=1_000, p_up=0.70),
            _signal(market_id="m1", decision_ts=1_060, p_up=0.68),
            _signal(market_id="m2", decision_ts=1_120, p_up=0.51),
        ]
    )

    assert report["execution_layer_v2_status"] == "diagnostic_only_fail_closed"
    assert report["ev_recalculation_loop_enabled"] is True
    assert report["dynamic_exit_engine_enabled"] is True
    assert report["state_machine_executor_enabled"] is True
    assert report["kelly_time_decay_sizing_enabled"] is True
    assert report["v1_baseline_comparison"]["baseline_name"] == EXECUTION_LAYER_V2_BASELINE_NAME
    assert report["uses_realized_pnl_or_settlement_outcomes"] is False
    assert report["source_scores_mutated"] is False
    assert report["paper_live_unlock_changed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert len(report["lambda_threshold_diagnostics"]) == 3


def test_execution_layer_v2_report_from_rows_fails_closed_on_outcome_fields() -> None:
    report = build_execution_layer_v2_report_from_rows(
        [
            {
                "market_id": "forbidden-row",
                "decision_ts": 1_000,
                "p_up": 0.70,
                "ask_up": 0.50,
                "ask_down": 0.52,
                "future_return": 0.10,
            }
        ]
    )

    assert report["execution_layer_v2_status"] == "blocked_fail_closed"
    assert report["decision_count"] == 0
    assert report["forbidden_outcome_fields_present"] is True
    assert report["forbidden_outcome_fields_used"] == ["future_return"]
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_execution_layer_v2_config_rejects_unsafe_flags() -> None:
    with pytest.raises(ValueError, match="capital_at_risk must be"):
        ExecutionLayerV2Config(capital_at_risk=True)


def test_execution_layer_v2_rejects_non_paper_position() -> None:
    with pytest.raises(ValueError, match="paper_only must be true"):
        ExecutionLayerV2Position(
            market_id="bad-position",
            side="UP",
            entry_ts=1_000,
            entry_price=0.50,
            entry_probability=0.70,
            entry_ev=0.19,
            size_usdc=10.0,
            shares=20.0,
            paper_only=False,
        )


def test_execution_layer_v2_backtest_writes_artifact_bundle_from_full_action_grid(
    tmp_path,
) -> None:
    input_path = tmp_path / "holdout_raw.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_only": True,
                "capital_at_risk": False,
                "uses_realized_pnl_or_labels_for_analysis": False,
                "future_outcome_evaluation_generated": False,
                "holdout_decision_rows": [
                    _full_grid_row(
                        market_id="grid-1",
                        decision_ts=1_000,
                        p_up=0.70,
                        up_ask=0.50,
                        down_ask=0.52,
                        time_to_close=240.0,
                    ),
                    _full_grid_row(
                        market_id="grid-1",
                        decision_ts=1_060,
                        p_up=0.57,
                        up_ask=0.50,
                        down_ask=0.52,
                        time_to_close=180.0,
                    ),
                    _full_grid_row(
                        market_id="grid-2",
                        decision_ts=1_120,
                        p_up=0.42,
                        up_ask=0.57,
                        down_ask=0.45,
                        time_to_close=180.0,
                    ),
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = run_execution_layer_v2_backtest(
        ExecutionLayerV2BacktestConfig(
            run_id="v2-backtest-fixture",
            output_dir=tmp_path / "runs",
            input_path=input_path,
        )
    )

    assert result.report["execution_layer_v2_status"] == "diagnostic_only_fail_closed"
    assert result.report["decision_count"] == 3
    assert result.report["entry_decision_count"] == 2
    assert result.report["exit_decision_count"] == 1
    assert result.report["outcome_evaluation_generated"] is False
    assert result.report["pnl_claim_generated"] is False
    assert result.manifest["artifact_hashes"]["execution_layer_v2_backtest_report"]
    assert result.manifest["paper_only"] is True
    assert result.manifest["capital_at_risk"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.artifact_paths["execution_layer_v2_backtest_report"].exists()
    assert result.artifact_paths["execution_layer_v2_backtest_summary"].exists()
    assert result.artifact_paths["execution_layer_v2_backtest_manifest"].exists()


def test_execution_layer_v2_backtest_fails_closed_when_signal_probability_missing(
    tmp_path,
) -> None:
    input_path = tmp_path / "feature_rows.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "market_id": "feature-only",
                "decision_ts": 1_000,
                "features": {
                    "up_ask": 0.50,
                    "down_ask": 0.52,
                    "up_bid": 0.49,
                    "down_bid": 0.51,
                    "time_to_close_seconds": 180.0,
                },
                "paper_only": True,
                "capital_at_risk": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_execution_layer_v2_backtest(
        ExecutionLayerV2BacktestConfig(
            run_id="v2-backtest-missing-probability",
            output_dir=tmp_path / "runs",
            input_path=input_path,
        )
    )

    assert result.report["execution_layer_v2_status"] == "blocked_fail_closed"
    assert result.report["decision_count"] == 0
    assert result.report["accepted_signal_row_count"] == 0
    assert result.report["rejected_signal_row_count"] == 1
    reasons = result.report["rejected_signal_rows"][0]["reason_codes"]
    assert "missing_decision_time_probability" in reasons
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_execution_layer_v2_policy_replay_from_settlement_csv_metrics_and_warning(
    tmp_path,
) -> None:
    csv_path = tmp_path / "current_clob_condition_settlement_pnl_rows.csv"
    _write_settlement_csv(
        csv_path,
        [
            _settlement_row("BUY_DOWN_HOLD_TO_SETTLEMENT", 300_000, 0.80, 10.0, 3.0, 4),
            _settlement_row("BUY_UP_HOLD_TO_SETTLEMENT", 300_000, 0.95, 10.0, -4.0, 3),
            _settlement_row("BUY_UP_SELL_BEFORE_CLOSE", 300_000, 0.75, 10.0, 1.0, 5),
            _settlement_row("BUY_DOWN_SELL_BEFORE_CLOSE", 900_000, 0.65, 10.0, 0.5, 6),
            _settlement_row("BUY_UP_HOLD_TO_SETTLEMENT", 300_000, 0.80, 10.0, -2.0, 2),
            _settlement_row("BUY_DOWN_HOLD_TO_SETTLEMENT", 900_000, 0.85, 10.0, -1.0, 1),
        ],
    )

    result = run_execution_layer_v2_policy_replay_from_settlement_csv(
        ExecutionLayerV2PolicyReplayConfig(
            run_id="policy-replay-fixture",
            input_csv=csv_path,
            output_dir=tmp_path / "runs",
        )
    )
    report = result.report
    variants = report["policy_variants"]

    baseline = variants["all_executed_baseline"]
    assert baseline["row_count"] == 6
    assert baseline["cost_basis"] == pytest.approx(60.0)
    assert baseline["settlement_pnl"] == pytest.approx(-2.5)
    assert baseline["roi"] == pytest.approx(-2.5 / 60.0)
    assert baseline["win_rate"] == pytest.approx(3 / 6)
    assert baseline["max_drawdown"] == pytest.approx(-7.0)
    assert baseline["max_drawdown_ordering"] == "chronological"
    assert baseline["chronological_sort_fields"] == [
        "numeric_iteration",
        "decision_ts_numeric",
        "intent_id",
        "row_index",
    ]
    assert baseline["action_distribution"]["BUY_UP_HOLD_TO_SETTLEMENT"] == 2
    assert baseline["family_distribution"]["HOLD_TO_SETTLEMENT"] == 4
    assert baseline["horizon_distribution"] == {"15m": 2, "5m": 4}

    assert variants["price_070_090_only"]["row_count"] == 4
    assert variants["price_070_090_only"]["settlement_pnl"] == pytest.approx(1.0)
    assert variants["price_070_090_only"]["rejected_reason_counts"] == {
        "price_above_090": 1,
        "price_below_070": 1,
    }
    assert variants["exclude_buy_up_hts"]["settlement_pnl"] == pytest.approx(3.5)
    assert variants["sell_before_close_only"]["settlement_pnl"] == pytest.approx(1.5)
    assert variants["buy_down_hts_only"]["settlement_pnl"] == pytest.approx(2.0)
    assert variants["five_min_only"]["settlement_pnl"] == pytest.approx(-2.0)
    assert variants["fifteen_min_only"]["settlement_pnl"] == pytest.approx(-0.5)
    assert variants["bucket_aware_v1_conservative"]["row_count"] == 3
    assert variants["bucket_aware_v1_conservative"]["settlement_pnl"] == pytest.approx(3.0)
    assert variants["bucket_aware_v1_conservative"]["rejected_reason_counts"][
        "bucket_aware_conservative_price_not_070_090"
    ] == 2
    assert variants["bucket_aware_v1_plus_sbc"]["row_count"] == 4
    assert variants["bucket_aware_v1_plus_sbc"]["settlement_pnl"] == pytest.approx(3.5)
    assert variants["bucket_aware_v1_plus_sbc"]["family_distribution"][
        "SELL_BEFORE_CLOSE"
    ] == 2
    assert variants["bucket_aware_v1_plus_sbc"]["price_bucket_distribution"][
        "0_60_0_70"
    ] == 1
    assert "bucket_aware_v1_conservative" in report["policy_variant_names"]
    assert "bucket_aware_v1_plus_sbc" in report["policy_variant_names"]
    assert report["max_drawdown_ordering"] == "chronological"
    assert report["small_sample_warnings"] == ["sell_before_close_small_sample"]

    ev = report["signal_to_ev_diagnostic"]
    assert ev["ev_mapping_status"] == "blocked_requires_calibrated_model_fair_value"
    assert ev["p_model_fair_value_source_fields_present"] is False
    assert ev["current_p_up_should_not_be_used_as_ev_fair_value_without_provenance"] is True
    assert "market_implied_probability_collapses_ev_to_spread_minus_cost" in ev[
        "ev_mapping_blocking_reason_codes"
    ]

    recommendation = report["recommended_execution_policy_v1"]
    assert recommendation["policy_name"] == "bucket_aware_execution_policy_v1_diagnostic"
    assert recommendation["candidate_variant_name"] == "bucket_aware_v1_plus_sbc"
    assert recommendation["do_not_relax_execution_guard_thresholds"] is True
    assert "sell_before_close_small_sample" in recommendation["small_sample_warnings"]
    assert recommendation["sell_before_close_positive_in_csv"] is True
    assert recommendation["sell_before_close_summary"]["settlement_pnl"] == pytest.approx(1.5)
    assert "Avoid BUY_UP_HOLD_TO_SETTLEMENT unless strong calibrated edge exists." in (
        recommendation["rules"]
    )

    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert result.artifact_paths["execution_layer_v2_policy_replay_report"].exists()
    assert result.artifact_paths["execution_layer_v2_policy_replay_summary"].exists()
    assert result.artifact_paths["execution_layer_v2_policy_replay_manifest"].exists()
    assert result.artifact_hashes["execution_layer_v2_policy_replay_report"]


def test_execution_layer_v2_hts_regime_risk_replay_is_diagnostic_only(
    tmp_path,
) -> None:
    run_dir = tmp_path / "settled-paper-goal"
    run_dir.mkdir()
    intents = [
        _hts_regime_intent("i1", "m1", "BUY_UP_HOLD_TO_SETTLEMENT", 0.85, 0.15, 0.80),
        _hts_regime_intent("i2", "m2", "BUY_UP_HOLD_TO_SETTLEMENT", 0.52, 0.48, 0.70),
        _hts_regime_intent("i3", "m3", "BUY_DOWN_HOLD_TO_SETTLEMENT", 0.20, 0.80, 0.78),
        _hts_regime_intent("i4", "m4", "BUY_UP_SELL_BEFORE_CLOSE", 0.62, 0.38, 0.56),
        _hts_regime_intent("i5", "m5", "BUY_UP_HOLD_TO_SETTLEMENT", 0.20, 0.80, 0.82),
    ]
    fills = [
        _hts_regime_fill("i1", "m1", "BUY_UP_HOLD_TO_SETTLEMENT", "UP", 0.80),
        _hts_regime_fill("i2", "m2", "BUY_UP_HOLD_TO_SETTLEMENT", "UP", 0.70),
        _hts_regime_fill("i3", "m3", "BUY_DOWN_HOLD_TO_SETTLEMENT", "DOWN", 0.78),
        _hts_regime_fill("i4", "m4", "BUY_UP_SELL_BEFORE_CLOSE", "UP", 0.56),
        _hts_regime_fill("i5", "m5", "BUY_UP_HOLD_TO_SETTLEMENT", "UP", 0.82),
    ]
    settlements = [
        _hts_regime_settlement("i1", "m1", "BUY_UP_HOLD_TO_SETTLEMENT", "UP", "DOWN", -0.16),
        _hts_regime_settlement("i2", "m2", "BUY_UP_HOLD_TO_SETTLEMENT", "UP", "UP", 0.08),
        _hts_regime_settlement("i3", "m3", "BUY_DOWN_HOLD_TO_SETTLEMENT", "DOWN", "DOWN", 0.06),
        _hts_regime_settlement("i4", "m4", "BUY_UP_SELL_BEFORE_CLOSE", "UP", "UP", 0.04),
        _hts_regime_settlement("i5", "m5", "BUY_UP_HOLD_TO_SETTLEMENT", "UP", "DOWN", -0.12),
    ]
    _write_jsonl(run_dir / "one_hour_paper_intent_log.jsonl", intents)
    _write_jsonl(run_dir / "one_hour_paper_fill_log.jsonl", fills)
    _write_jsonl(run_dir / "settlement_pnl_rows.jsonl", settlements)

    result = run_execution_layer_v2_hts_regime_risk_replay(
        ExecutionLayerV2HTSRegimeRiskReplayConfig(
            run_id="hts-regime-risk-fixture",
            input_path=run_dir,
            output_dir=tmp_path / "runs",
        )
    )
    report = result.report

    assert report["fill_count"] == 5
    assert report["hts_fill_count"] == 4
    assert report["global_up_hts_disable_recommended"] is False
    assert report["uses_outcome_for_policy_selection"] is False
    assert report["uses_outcome_for_offline_evaluation"] is True
    assert report["policy_variants"]["baseline_all"]["settled_pnl"] == pytest.approx(-0.10)
    assert report["policy_variants"]["side_blind_hts"]["fill_count"] == 4
    assert report["policy_variants"]["up_hts_only_when_up_regime_confirmed"][
        "fill_count"
    ] == 1
    assert report["policy_variants"]["down_hts_only_when_down_regime_confirmed"][
        "settled_pnl"
    ] == pytest.approx(0.06)
    assert report["false_positive_up_hts_examples"][0]["market_id"] == "m1"
    assert report["missed_opportunity_up_hts_examples"][0]["market_id"] == "m2"
    assert "resolved_outcome" in report["evaluation_only_fields"]
    assert "Do not disable BUY_UP_HOLD_TO_SETTLEMENT globally" in (
        report["recommended_decision_time_guard_signals"][0]
    )
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert result.artifact_paths[
        "execution_layer_v2_hts_regime_risk_replay_report"
    ].exists()
    assert result.artifact_paths[
        "execution_layer_v2_hts_regime_risk_replay_manifest"
    ].exists()
    assert result.artifact_hashes[
        "execution_layer_v2_hts_regime_risk_replay_report"
    ]


def test_execution_layer_v2_hts_regime_risk_replay_uses_decision_time_features(
    tmp_path,
) -> None:
    run_dir = tmp_path / "feature-rich-paper-goal"
    run_dir.mkdir()
    up_intent = _hts_regime_intent(
        "i-feature-up",
        "m-feature-up",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        0.51,
        0.49,
        0.76,
    )
    up_intent.update(
        {
            "btc_momentum": 0.004,
            "reference_price_to_beat_distance_at_decision": 0.003,
            "time_since_market_start_seconds": 120.0,
            "action_score_margin": 0.06,
            "side_specific_action_score_margin": 0.08,
            "decision_time_regime_feature_provenance": {
                "provenance_valid": True,
                "decision_ts": up_intent["decision_ts"],
                "max_input_ts": up_intent["decision_ts"],
                "source_fields_used": [
                    "raw_btc_feature_candles.open_price",
                    "raw_polymarket_markets.reference_price_start",
                    "full_5_action_ranking.corrected_model_score",
                ],
            },
            "decision_time_regime_feature_max_input_ts": up_intent["decision_ts"],
        }
    )
    down_alias_intent = _hts_regime_intent(
        "i-alias-down",
        "m-alias-down",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        0.20,
        0.80,
        0.72,
    )
    _write_jsonl(run_dir / "one_hour_paper_intent_log.jsonl", [up_intent, down_alias_intent])
    trace_dir = run_dir / "incremental_fresh_loop"
    trace_dir.mkdir()
    (trace_dir / "o_v8_paper_fresh_signal_trace.json").write_text(
        json.dumps(
            {
                "trace_rows": [
                    {
                        "paper_intent_id": "i-alias-down",
                        "market_id": "m-alias-down",
                        "decision_ts": down_alias_intent["decision_ts"],
                        "btc_momentum": -0.006,
                        "reference_price_to_beat_distance_at_decision": -0.002,
                        "elapsed_since_market_start_seconds": 180.0,
                        "score_margin": 0.04,
                        "side_specific_action_score_margin": 0.05,
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    down_fill = _hts_regime_fill(
        "i-alias-down",
        "m-alias-down",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "DOWN",
        0.72,
    )
    down_fill.update(
        {
            "btc_momentum": None,
            "reference_price_to_beat_distance_at_decision": None,
            "time_since_market_start_seconds": None,
            "action_score_margin": None,
            "side_specific_action_score_margin": None,
        }
    )
    _write_jsonl(
        run_dir / "one_hour_paper_fill_log.jsonl",
        [
            _hts_regime_fill(
                "i-feature-up",
                "m-feature-up",
                "BUY_UP_HOLD_TO_SETTLEMENT",
                "UP",
                0.76,
            ),
            down_fill,
        ],
    )
    down_settlement = _hts_regime_settlement(
        "i-alias-down",
        "m-alias-down",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "DOWN",
        "DOWN",
        0.03,
    )
    down_settlement.update(
        {
            "btc_momentum": None,
            "reference_price_to_beat_distance_at_decision": None,
            "time_since_market_start_seconds": None,
            "action_score_margin": None,
            "side_specific_action_score_margin": None,
        }
    )
    _write_jsonl(
        run_dir / "settlement_pnl_rows.jsonl",
        [
            _hts_regime_settlement(
                "i-feature-up",
                "m-feature-up",
                "BUY_UP_HOLD_TO_SETTLEMENT",
                "UP",
                "UP",
                0.04,
            ),
            down_settlement,
        ],
    )

    result = run_execution_layer_v2_hts_regime_risk_replay(
        ExecutionLayerV2HTSRegimeRiskReplayConfig(
            run_id="hts-regime-feature-fixture",
            input_path=run_dir,
            output_dir=tmp_path / "runs",
        )
    )
    report = result.report

    assert report["feature_coverage_before"]["btc_momentum"]["available_count"] == 1
    assert report["feature_coverage_after"]["btc_momentum"]["available_count"] == 2
    assert report["feature_coverage_before"][
        "reference_price_to_beat_distance_at_decision"
    ]["available_count"] == 1
    assert report["feature_coverage_after"][
        "reference_price_to_beat_distance_at_decision"
    ]["available_count"] == 2
    assert report["feature_coverage_before"]["time_since_market_start_seconds"][
        "available_count"
    ] == 1
    assert report["feature_coverage_after"]["time_since_market_start_seconds"][
        "available_count"
    ] == 2
    assert report["feature_coverage_after"]["action_score_margin"][
        "available_count"
    ] == 2
    assert report["feature_coverage_after"]["side_specific_action_score_margin"][
        "available_count"
    ] == 2
    assert report["policy_variants"]["up_hts_only_when_up_regime_confirmed"][
        "fill_count"
    ] == 1
    assert report["policy_variants"]["down_hts_only_when_down_regime_confirmed"][
        "fill_count"
    ] == 1
    assert report["up_hts_win_examples"][0]["market_id"] == "m-feature-up"
    assert report["up_hts_win_examples"][0]["btc_momentum"] == pytest.approx(0.004)
    assert report["up_hts_win_examples"][0]["regime_feature_vote_summary"][
        "up_vote_count"
    ] >= 2
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["v8_execution_handoff_allowed"] is False


def test_regime_entry_edge_replay_groups_correlated_signals_and_exposure(
    tmp_path,
) -> None:
    run_dir = tmp_path / "regime-entry-edge-input"
    run_dir.mkdir()
    specs = [
        (
            "i1",
            "m1",
            "BUY_UP_HOLD_TO_SETTLEMENT",
            0.70,
            0.30,
            0.60,
            1.0,
            0.01,
            0.01,
            "UP",
            0.10,
        ),
        (
            "i2",
            "m1",
            "BUY_UP_HOLD_TO_SETTLEMENT",
            0.70,
            0.30,
            0.60,
            2.0,
            0.01,
            0.01,
            "DOWN",
            -0.08,
        ),
        (
            "i3",
            "m1",
            "BUY_UP_HOLD_TO_SETTLEMENT",
            0.95,
            0.05,
            0.95,
            1.5,
            0.02,
            0.02,
            "UP",
            0.04,
        ),
        (
            "i4",
            "m1",
            "BUY_DOWN_HOLD_TO_SETTLEMENT",
            0.20,
            0.80,
            0.65,
            2.5,
            -0.01,
            -0.01,
            "DOWN",
            0.03,
        ),
        (
            "i5",
            "m2",
            "BUY_UP_HOLD_TO_SETTLEMENT",
            0.70,
            0.30,
            0.65,
            1.2,
            -0.01,
            -0.01,
            "DOWN",
            -0.06,
        ),
    ]
    intents = []
    fills = []
    settlements = []
    for index, spec in enumerate(specs, start=1):
        (
            intent_id,
            market_id,
            action,
            p_up,
            p_down,
            price,
            score,
            momentum,
            reference_distance,
            outcome,
            pnl,
        ) = spec
        side = "UP" if "BUY_UP" in action else "DOWN"
        decision_ts = index * 1_000
        intent = _hts_regime_intent(
            intent_id,
            market_id,
            action,
            p_up,
            p_down,
            price,
        )
        intent.update(
            {
                "decision_ts": decision_ts,
                "source_model_score": score,
                "canonical_corrected_score": score,
                "canonical_raw_score": score * 10.0,
                "action_score_margin": 0.03,
                "btc_momentum": momentum,
                "reference_price_to_beat_distance_at_decision": (
                    reference_distance
                ),
            }
        )
        fill = _hts_regime_fill(
            intent_id,
            market_id,
            action,
            side,
            price,
        )
        fill["decision_ts"] = decision_ts
        settlement = _hts_regime_settlement(
            intent_id,
            market_id,
            action,
            side,
            outcome,
            pnl,
        )
        settlement["decision_ts"] = decision_ts
        intents.append(intent)
        fills.append(fill)
        settlements.append(settlement)
    _write_jsonl(run_dir / "one_hour_paper_intent_log.jsonl", intents)
    _write_jsonl(run_dir / "one_hour_paper_fill_log.jsonl", fills)
    _write_jsonl(run_dir / "settlement_pnl_rows.jsonl", settlements)

    calibration_path = tmp_path / "frozen-ev.json"
    calibration_path.write_text(
        json.dumps(
            {
                "frozen": True,
                "decision_time_safe": True,
                "uses_validation_labels_for_tuning": False,
                "market_implied_probability_used_for_ev": False,
                "score_to_expected_net_return": {
                    "intercept": 0.0,
                    "canonical_o_action_score_weight": 0.10,
                },
                "subtract_execution_cost": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = run_execution_layer_v2_regime_entry_edge_replay(
        ExecutionLayerV2RegimeEntryEdgeReplayConfig(
            run_id="regime-entry-edge-fixture",
            input_path=run_dir,
            output_dir=tmp_path / "runs",
            frozen_ev_calibration_artifact=calibration_path,
        )
    )
    report = result.report

    assert report["row_count"] == 5
    assert report["unique_market_count"] == 2
    assert report["decision_time_provenance_violation_count"] == 0
    assert report[
        "correlated_momentum_reference_counted_as_independent_votes"
    ] is False
    correlated = report["independent_signal_group_contract"][
        "correlated_field_groups"
    ][0]
    assert correlated["maximum_vote_weight"] == 1
    assert correlated["counted_as_independent_votes"] is False
    m1_rows = [row for row in report["entry_rows"] if row["market_id"] == "m1"]
    assert [row["entry_index_within_market"] for row in m1_rows] == [1, 2, 3, 4]
    assert m1_rows[0]["cumulative_market_exposure_before_entry"] == 0.0
    assert m1_rows[1]["same_side_reentry"] is True
    assert m1_rows[3]["side_flip"] is True
    assert m1_rows[0]["independent_signal_groups"][
        "direction_regime_evidence"
    ]["correlated_anchor_vote_weight"] == 1
    assert report["policy_variants"]["first_entry_only"]["fill_count"] == 2
    assert report["policy_variants"]["first_entry_only"][
        "settled_pnl"
    ] == pytest.approx(0.04)
    assert report["policy_variants"]["require_incremental_edge_for_reentry"][
        "fill_count"
    ] == 4
    assert report["policy_variants"]["cap_same_market_exposure"][
        "fill_count"
    ] == 3
    assert report["pnl_by_market"]["m1"]["fill_count"] == 4
    assert report["repeated_entry_marginal_pnl"]["same_side_reentries"][
        "fill_count"
    ] == 2
    assert report["repeated_entry_marginal_pnl"]["side_flip_entries"][
        "fill_count"
    ] == 1
    first_entry_tradeoff = report["up_rule_tradeoff_summary"]["first_entry_only"]
    assert first_entry_tradeoff["incorrectly_removed_up_win_count"] == 1
    assert first_entry_tradeoff["avoided_up_loss_count"] == 1
    assert report["bounded_per_condition_clob_settlement_fallback_proposal"][
        "mutate_original_run_manifest"
    ] is False
    assert report["future_unseen_forward_shadow_required"] is True
    assert report["uses_outcome_for_policy_selection"] is False
    assert report["production_gate_implemented"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert result.artifact_paths[
        "execution_layer_v2_regime_entry_edge_replay_report"
    ].exists()
    assert result.artifact_hashes[
        "execution_layer_v2_regime_entry_edge_replay_manifest"
    ]

    mutated_settlements = []
    for row in settlements:
        mutated = dict(row)
        mutated["settlement_pnl"] = -float(mutated["settlement_pnl"])
        mutated["resolved_outcome"] = (
            "DOWN" if mutated["resolved_outcome"] == "UP" else "UP"
        )
        mutated_settlements.append(mutated)
    _write_jsonl(
        run_dir / "settlement_pnl_rows.jsonl",
        mutated_settlements,
    )
    mutated_report = run_execution_layer_v2_regime_entry_edge_replay(
        ExecutionLayerV2RegimeEntryEdgeReplayConfig(
            run_id="regime-entry-edge-mutated-outcomes",
            input_path=run_dir,
            output_dir=tmp_path / "runs",
            frozen_ev_calibration_artifact=calibration_path,
        )
    ).report
    assert [row["variant_decisions"] for row in report["entry_rows"]] == [
        row["variant_decisions"] for row in mutated_report["entry_rows"]
    ]


def test_regime_entry_edge_replay_missing_calibration_fails_closed(
    tmp_path,
) -> None:
    run_dir = tmp_path / "regime-entry-edge-missing-ev"
    run_dir.mkdir()
    intent = _hts_regime_intent(
        "i1",
        "m1",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        0.70,
        0.30,
        0.60,
    )
    intent.update(
        {
            "btc_momentum": 0.01,
            "reference_price_to_beat_distance_at_decision": 0.01,
        }
    )
    _write_jsonl(run_dir / "one_hour_paper_intent_log.jsonl", [intent])
    _write_jsonl(
        run_dir / "one_hour_paper_fill_log.jsonl",
        [
            _hts_regime_fill(
                "i1",
                "m1",
                "BUY_UP_HOLD_TO_SETTLEMENT",
                "UP",
                0.60,
            )
        ],
    )
    _write_jsonl(
        run_dir / "settlement_pnl_rows.jsonl",
        [
            _hts_regime_settlement(
                "i1",
                "m1",
                "BUY_UP_HOLD_TO_SETTLEMENT",
                "UP",
                "UP",
                0.10,
            )
        ],
    )
    report = run_execution_layer_v2_regime_entry_edge_replay(
        ExecutionLayerV2RegimeEntryEdgeReplayConfig(
            run_id="regime-entry-edge-missing-ev-fixture",
            input_path=run_dir,
            output_dir=tmp_path / "runs",
        )
    ).report

    calibrated_variant = report["policy_variants"][
        "calibrated_ev_conditioned_on_regime"
    ]
    assert calibrated_variant["fill_count"] == 0
    assert calibrated_variant["rejected_reason_counts"] == {
        "calibrated_ev_missing": 1
    }
    assert report["frozen_ev_calibration_artifact"]["valid"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_fresh_provider_row_adds_decision_time_regime_features() -> None:
    decision_ts = 1_120_000
    row = _fresh_public_row_from_provider_feature_context(
        run_id="provider-regime-fixture",
        row_index=1,
        market={
            "market_id": "provider-market",
            "condition_id": "provider-condition",
            "slug": "btc-updown-5m-provider",
            "market_family": "btc_updown_5m",
            "up_token_id": "up-token",
            "down_token_id": "down-token",
            "market_start_ts": 1_000_000,
            "market_end_ts": 1_300_000,
            "reference_price_source": "polymarket_official_btc_usd_reference",
            "reference_price_start": 100_000.0,
        },
        up={
            "token_id": "up-token",
            "bid_price": 0.58,
            "ask_price": 0.60,
            "mid_price": 0.59,
            "bid_size": 2.0,
            "ask_size": 2.0,
            "liquidity_depth": 4.0,
            "available_at_ts": 1_119_000,
        },
        down={
            "token_id": "down-token",
            "bid_price": 0.40,
            "ask_price": 0.42,
            "mid_price": 0.41,
            "bid_size": 2.0,
            "ask_size": 2.0,
            "liquidity_depth": 4.0,
            "available_at_ts": 1_119_000,
        },
        candle={
            "ts": 1_000_000,
            "available_at_ts": 1_060_000,
            "open_price": 100_000.0,
            "close_price": 100_200.0,
            "timeframe_ms": 60_000,
            "source": "pytest",
        },
        decision_ts=decision_ts,
    )

    assert row["btc_momentum"] == pytest.approx(0.002)
    assert row["reference_price_to_beat_distance_at_decision"] == pytest.approx(0.002)
    assert row["time_since_market_start_seconds"] == pytest.approx(120.0)
    assert row["action_score_margin"] is not None
    assert "side_specific_action_score_margin" in row
    assert "side_specific_action_score_margin_provenance" in row
    assert row["decision_time_regime_feature_provenance"]["provenance_valid"] is True
    assert row["decision_time_regime_feature_max_input_ts"] <= decision_ts
    assert row["score_components"]["btc_momentum"] == pytest.approx(0.002)
    assert row["paper_only"] is True
    assert row["capital_at_risk"] is False


def test_fresh_provider_row_uses_market_start_btc_proxy_when_price_to_beat_missing() -> None:
    decision_ts = 1_180_000
    row = _fresh_public_row_from_provider_feature_context(
        run_id="provider-regime-reference-proxy-fixture",
        row_index=1,
        market={
            "market_id": "provider-market-no-price-to-beat",
            "condition_id": "provider-condition",
            "slug": "btc-updown-5m-provider",
            "market_family": "btc_updown_5m",
            "up_token_id": "up-token",
            "down_token_id": "down-token",
            "market_start_ts": 1_120_000,
            "market_end_ts": 1_420_000,
            "reference_price_source": "https://data.chain.link/streams/btc-usd",
        },
        up={
            "token_id": "up-token",
            "bid_price": 0.58,
            "ask_price": 0.60,
            "mid_price": 0.59,
            "bid_size": 2.0,
            "ask_size": 2.0,
            "liquidity_depth": 4.0,
            "available_at_ts": 1_179_000,
        },
        down={
            "token_id": "down-token",
            "bid_price": 0.40,
            "ask_price": 0.42,
            "mid_price": 0.41,
            "bid_size": 2.0,
            "ask_size": 2.0,
            "liquidity_depth": 4.0,
            "available_at_ts": 1_179_000,
        },
        candle={
            "ts": 1_120_000,
            "available_at_ts": 1_180_000,
            "open_price": 100_000.0,
            "close_price": 100_500.0,
            "timeframe_ms": 60_000,
            "source": "pytest",
        },
        reference_candle={
            "ts": 1_060_000,
            "close_time": 1_120_000,
            "available_at_ts": 1_120_000,
            "open_price": 99_900.0,
            "close_price": 100_000.0,
            "timeframe_ms": 60_000,
            "source": "pytest",
        },
        decision_ts=decision_ts,
    )

    assert row["reference_price_start"] is None
    assert row["reference_price_to_beat_at_decision"] == pytest.approx(100_000.0)
    assert row["reference_price_to_beat_distance_at_decision"] == pytest.approx(0.005)
    provenance = row["reference_price_to_beat_distance_provenance"]
    assert provenance["provenance_valid"] is True
    assert provenance["max_input_ts"] <= decision_ts
    assert provenance["reference_price_to_beat_source_type"] == (
        "btc_feature_candle_market_start_proxy"
    )
    assert "official_polymarket_price_to_beat_unavailable_btc_feature_candle_proxy_used" in (
        provenance["warning_reason_codes"]
    )
    assert row["score_components"]["reference_price_to_beat"] == pytest.approx(
        100_000.0
    )
    assert row["paper_only"] is True
    assert row["capital_at_risk"] is False


def test_execution_layer_v2_forward_shadow_missing_calibrated_ev_fails_closed(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.json"
    _write_forward_shadow_input(
        input_path,
        [
            _forward_shadow_row(
                market_id="missing-calibrated-ev",
                action="BUY_UP_HOLD_TO_SETTLEMENT",
                selected_side="UP",
                entry_ask=0.50,
                p_up=0.61,
                time_to_close_seconds=240.0,
            )
        ],
    )

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="forward-shadow-missing-ev",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )

    source = result.calibrated_ev_source_report
    assert source["calibrated_ev_source_status"] == "blocked_missing_calibrated_ev_source"
    assert source["calibrated_ev_source"] == "missing_frozen_ev_calibration_artifact"
    assert source["calibration_artifact_path"] is None
    assert source["calibrated_ev_produced_count"] == 0
    assert source["calibrated_ev_missing_count"] == 1
    assert "missing_frozen_ev_calibration_artifact" in source[
        "calibrated_ev_source_blocking_reason_codes"
    ]

    ev = result.ev_mapping_report
    assert ev["ev_mapping_status"] == "blocked_missing_calibrated_ev_source"
    assert ev["calibrated_ev_available"] is False
    assert ev["calibrated_ev_missing_count"] == 1
    assert ev["market_implied_probability_used_for_ev"] is False
    assert "missing_calibrated_model_fair_value_or_action_expected_return" in ev[
        "ev_mapping_blocking_reason_codes"
    ]

    shadow = result.forward_shadow_report
    assert shadow["policy_variants"]["calibrated_ev_v2"]["allowed_decision_count"] == 0
    assert shadow["policy_variants"]["calibrated_ev_v2"]["rejected_reason_counts"][
        "calibrated_ev_source_missing"
    ] == 1
    assert (
        shadow["policy_variants"]["baseline_current_guard"]["rejected_reason_counts"][
            "baseline_current_guard_missing_calibrated_ev_source"
        ]
        == 1
    )
    assert shadow["market_implied_probability_used_as_calibrated_ev_source"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["paper_only"] is True
    assert result.manifest["capital_at_risk"] is False


def test_execution_layer_v2_forward_shadow_frozen_calibration_artifact_produces_ev(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.jsonl"
    calibration_path = tmp_path / "frozen_ev_calibration.json"
    _write_ev_calibration_artifact(
        calibration_path,
        {
            "intercept": 0.0,
            "canonical_o_action_score_weight": 0.10,
        },
    )
    _write_jsonl(
        input_path,
        [
            _forward_shadow_row(
                market_id="artifact-sbc-pass",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                selected_side="DOWN",
                entry_ask=0.55,
                canonical_o_action_score=0.75,
                order_allowed=True,
                execution_guarded_action="BUY_DOWN_SELL_BEFORE_CLOSE",
                execution_guarded_side="DOWN",
                execution_blocking_reason_codes=[],
                time_to_close_seconds=90.0,
            ),
            _forward_shadow_row(
                market_id="artifact-up-hts",
                action="BUY_UP_HOLD_TO_SETTLEMENT",
                selected_side="UP",
                entry_ask=0.50,
                canonical_o_action_score=0.70,
                order_allowed=True,
                execution_guarded_action="BUY_UP_HOLD_TO_SETTLEMENT",
                execution_guarded_side="UP",
                execution_blocking_reason_codes=[],
                time_to_close_seconds=180.0,
            ),
        ],
    )

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="forward-shadow-frozen-ev-artifact",
            input_path=input_path,
            output_dir=tmp_path / "runs",
            frozen_ev_calibration_artifact=calibration_path,
        )
    )

    source = result.calibrated_ev_source_report
    assert source["calibrated_ev_source_status"] == "calibrated_ev_source_available"
    assert source["calibrated_ev_source"] == "frozen_ev_calibration_artifact"
    assert source["calibration_artifact_valid"] is True
    assert source["calibrated_ev_produced_count"] == 2
    assert source["calibrated_ev_missing_count"] == 0
    assert source["source_fields_used"] == [
        "action_family",
        "canonical_o_action_score",
        "execution_cost",
        "intercept",
        "selected_action",
        "selected_side",
    ]
    assert source["source_rows"][0]["calibrated_action_expected_net_return"] == (
        pytest.approx(0.074)
    )
    assert source["calibrated_ev_v2_candidate_count"] == 2
    assert source["calibrated_ev_v2_guard_passed_count"] == 2
    assert source["calibrated_ev_plus_bucket_v2_candidate_count"] == 1
    assert source["calibrated_ev_plus_bucket_v2_guard_passed_count"] == 1

    variants = result.forward_shadow_report["policy_variants"]
    assert variants["calibrated_ev_v2"]["allowed_decision_count"] == 2
    assert variants["calibrated_ev_plus_bucket_v2"]["allowed_decision_count"] == 1
    intersections = result.guard_intersection_report[
        "policy_variant_guard_intersections"
    ]
    assert intersections["calibrated_ev_v2"]["guard_passed_candidate_count"] == 2
    assert intersections["calibrated_ev_plus_bucket_v2"][
        "guard_passed_candidate_count"
    ] == 1
    assert result.ev_mapping_report["ev_mapping_status"] == "calibrated_ev_available"
    assert result.manifest["calibrated_ev_produced_count"] == 2
    assert result.manifest["frozen_ev_calibration_artifact_hash"]
    assert result.manifest["market_implied_probability_used_for_ev"] is False
    assert result.artifact_paths["execution_layer_v2_calibrated_ev_source_report"].exists()


def test_execution_layer_v2_forward_shadow_calibrated_ev_and_bucket_plus_sbc(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            _forward_shadow_row(
                market_id="cal-ev-up-hts",
                action="BUY_UP_HOLD_TO_SETTLEMENT",
                selected_side="UP",
                entry_ask=0.50,
                p_model_fair_value_up=0.58,
                time_to_close_seconds=240.0,
            ),
            _forward_shadow_row(
                market_id="cal-ev-down-sbc",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                selected_side="DOWN",
                entry_ask=0.55,
                calibrated_action_expected_net_return=0.04,
                time_to_close_seconds=90.0,
            ),
            _forward_shadow_row(
                market_id="cal-ev-down-hts",
                action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                selected_side="DOWN",
                entry_ask=0.82,
                p_model_fair_value_down=0.86,
                time_to_close_seconds=180.0,
            ),
        ],
    )

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="forward-shadow-calibrated-ev",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )
    ev = result.ev_mapping_report
    assert ev["ev_mapping_status"] == "calibrated_ev_available"
    assert ev["calibrated_ev_available"] is True
    assert ev["probability_source_summary"]["p_model_fair_value_count"] == 2
    assert ev["probability_source_summary"][
        "calibrated_action_expected_net_return_count"
    ] == 1
    assert {
        row["ev_source"] for row in ev["row_ev_mapping_contracts"]
    } == {
        "p_model_fair_value_minus_execution_price_minus_cost",
        "calibrated_action_expected_net_return",
    }

    variants = result.forward_shadow_report["policy_variants"]
    assert variants["calibrated_ev_v2"]["allowed_decision_count"] == 3
    assert variants["calibrated_ev_v2"]["entry_count"] == 3
    assert variants["bucket_aware_v1_conservative"]["allowed_decision_count"] == 1
    assert variants["bucket_aware_v1_plus_sbc"]["allowed_decision_count"] == 2
    assert variants["bucket_aware_v1_plus_sbc"]["family_distribution"][
        "SELL_BEFORE_CLOSE"
    ] == 1
    assert variants["calibrated_ev_plus_bucket_v2"]["allowed_decision_count"] == 2
    assert variants["calibrated_ev_plus_bucket_v2"]["rejected_reason_counts"][
        "bucket_aware_plus_sbc_excluded_buy_up_hts"
    ] == 1
    assert result.forward_shadow_report["uses_settlement_pnl_or_outcome_labels"] is False
    assert result.forward_shadow_report["uses_oracle_actions_or_future_returns"] is False
    assert result.artifact_paths["execution_layer_v2_calibrated_ev_mapping_report"].exists()
    assert result.artifact_paths["execution_layer_v2_forward_shadow_policy_report"].exists()
    assert result.artifact_paths[
        "execution_layer_v2_forward_shadow_guard_intersection_report"
    ].exists()
    assert result.artifact_paths["execution_layer_v2_forward_shadow_manifest"].exists()
    assert result.artifact_hashes["execution_layer_v2_forward_shadow_manifest"]


def test_execution_layer_v2_forward_shadow_guard_intersection_counts(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            _forward_shadow_row(
                market_id="guard-pass-sbc",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                selected_side="DOWN",
                entry_ask=0.55,
                calibrated_action_expected_net_return=0.05,
                order_allowed=True,
                execution_guarded_action="BUY_DOWN_SELL_BEFORE_CLOSE",
                execution_guarded_side="DOWN",
                execution_blocking_reason_codes=[],
                paper_intent_id="intent-pass",
                paper_fill_id="fill-pass",
                time_to_close_seconds=90.0,
            ),
            _forward_shadow_row(
                market_id="guard-blocked-hts",
                action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                selected_side="DOWN",
                entry_ask=0.80,
                p_model_fair_value_down=0.85,
                order_allowed=False,
                execution_guarded_action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                execution_guarded_side="DOWN",
                execution_blocking_reason_codes=["execution_time_to_close_unsafe"],
                time_to_close_seconds=40.0,
            ),
            _forward_shadow_row(
                market_id="guard-unknown-sbc",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                selected_side="DOWN",
                entry_ask=0.54,
                calibrated_action_expected_net_return=0.04,
                time_to_close_seconds=100.0,
            ),
            _forward_shadow_row(
                market_id="guard-no-trade",
                action="NO_TRADE",
                selected_side="NONE",
                entry_ask=0.0,
                calibrated_action_expected_net_return=0.10,
                order_allowed=True,
                execution_guarded_action="NO_TRADE",
                execution_guarded_side="NONE",
                execution_blocking_reason_codes=[],
            ),
        ],
    )

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="forward-shadow-guard-intersection",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )

    intersections = result.guard_intersection_report[
        "policy_variant_guard_intersections"
    ]
    plus_sbc = intersections["bucket_aware_v1_plus_sbc"]
    assert plus_sbc["policy_candidate_count"] == 3
    assert plus_sbc["guard_passed_candidate_count"] == 1
    assert plus_sbc["guard_blocked_candidate_count"] == 1
    assert plus_sbc["guard_unknown_candidate_count"] == 1
    assert plus_sbc["candidate_but_not_executable_count"] == 2
    assert plus_sbc["executable_shadow_count"] == 1
    assert plus_sbc["executable_shadow_entry_count"] == 1
    assert plus_sbc["executable_shadow_no_trade_count"] == 0
    assert plus_sbc["guard_blocking_reason_distribution"][
        "execution_time_to_close_unsafe"
    ] == 1
    assert plus_sbc["guard_blocking_reason_distribution"][
        "missing_execution_guard_decision_fields"
    ] == 1

    calibrated = intersections["calibrated_ev_v2"]
    assert calibrated["policy_candidate_count"] == 3
    assert calibrated["guard_passed_candidate_count"] == 1
    assert calibrated["guard_blocked_candidate_count"] == 1
    assert calibrated["guard_unknown_candidate_count"] == 1
    assert intersections["calibrated_ev_plus_bucket_v2"]["policy_candidate_count"] == 3
    assert intersections["bucket_aware_v1_conservative"]["policy_candidate_count"] == 1
    assert result.forward_shadow_report["policy_variants"]["baseline_current_guard"][
        "allowed_decision_count"
    ] == 3
    assert result.manifest["guard_intersection_summary"][
        "bucket_aware_v1_plus_sbc"
    ]["executable_shadow_count"] == 1
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False


def test_execution_layer_v2_hts_time_window_remap_reports_guard_passed_sbc(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            _forward_shadow_row(
                market_id="hts-sbc-remap-pass",
                action="BUY_UP_HOLD_TO_SETTLEMENT",
                selected_side="UP",
                entry_ask=0.72,
                calibrated_action_expected_net_return=0.05,
                order_allowed=False,
                execution_guarded_action="BUY_UP_SELL_BEFORE_CLOSE",
                execution_guarded_side="UP",
                execution_blocking_reason_codes=[
                    "execution_hts_downgraded_to_same_side_sbc",
                    "execution_hts_guard_failed",
                    "execution_time_to_close_unsafe",
                ],
                time_to_close_seconds=90.0,
            )
        ],
    )

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="hts-sbc-remap-pass",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )

    remap = result.hts_time_window_remap_report
    assert remap["hts_time_window_blocked_count"] == 1
    assert remap["same_side_sbc_alternative_available_count"] == 1
    assert remap["same_side_sbc_calibrated_ev_available_count"] == 1
    assert remap["remap_candidate_count"] == 1
    assert remap["same_side_sbc_guard_passed_count"] == 1
    assert remap["remap_guard_passed_count"] == 1
    assert remap["remap_rows"][0]["proposed_same_side_sbc_action"] == (
        "BUY_UP_SELL_BEFORE_CLOSE"
    )
    assert remap["remap_rows"][0]["diagnostic_remap_guard_passed"] is True
    assert remap["remap_rows"][0]["remap_reason_codes"] == [
        "diagnostic_remap_guard_passed"
    ]
    assert result.guard_intersection_report["policy_variant_guard_intersections"][
        "calibrated_ev_v2"
    ]["guard_passed_candidate_count"] == 0
    assert result.artifact_paths[
        "execution_layer_v2_hts_time_window_remap_report"
    ].exists()
    assert result.manifest["hts_time_window_remap_summary"][
        "remap_guard_passed_count"
    ] == 1
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert remap["source_scores_mutated"] is False
    assert remap["o_score_mutated"] is False


def test_execution_layer_v2_hts_time_window_remap_missing_sbc_alternative(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            _forward_shadow_row(
                market_id="hts-sbc-remap-missing",
                action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                selected_side="DOWN",
                entry_ask=0.75,
                calibrated_action_expected_net_return=0.06,
                order_allowed=False,
                execution_guarded_action="BUY_DOWN_SELL_BEFORE_CLOSE",
                execution_guarded_side="DOWN",
                execution_blocking_reason_codes=[
                    "execution_hts_downgraded_to_same_side_sbc",
                    "execution_hts_guard_failed",
                    "execution_time_to_close_unsafe",
                ],
                time_to_close_seconds=90.0,
                available_actions=["BUY_DOWN_HOLD_TO_SETTLEMENT"],
            )
        ],
    )

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="hts-sbc-remap-missing",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )

    remap = result.hts_time_window_remap_report
    assert remap["hts_time_window_blocked_count"] == 1
    assert remap["same_side_sbc_alternative_available_count"] == 0
    assert remap["same_side_sbc_calibrated_ev_available_count"] == 1
    assert remap["remap_candidate_count"] == 0
    assert remap["remap_guard_passed_count"] == 0
    assert remap["remap_reason_distribution"][
        "same_side_sbc_alternative_missing"
    ] == 1
    assert remap["remap_rows"][0]["diagnostic_remap_guard_passed"] is False


def test_execution_layer_v2_hts_time_window_remap_guard_blocked_and_ev_negative(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            _forward_shadow_row(
                market_id="hts-sbc-remap-spread-blocked",
                action="BUY_UP_HOLD_TO_SETTLEMENT",
                selected_side="UP",
                entry_ask=0.78,
                calibrated_action_expected_net_return=0.05,
                order_allowed=False,
                execution_guarded_action="BUY_UP_SELL_BEFORE_CLOSE",
                execution_guarded_side="UP",
                execution_blocking_reason_codes=[
                    "execution_hts_downgraded_to_same_side_sbc",
                    "execution_hts_guard_failed",
                    "execution_time_to_close_unsafe",
                    "execution_spread_too_wide",
                ],
                time_to_close_seconds=90.0,
            ),
            _forward_shadow_row(
                market_id="hts-sbc-remap-ev-low",
                action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                selected_side="DOWN",
                entry_ask=0.80,
                calibrated_action_expected_net_return=0.005,
                order_allowed=False,
                execution_guarded_action="BUY_DOWN_SELL_BEFORE_CLOSE",
                execution_guarded_side="DOWN",
                execution_blocking_reason_codes=[
                    "execution_hts_downgraded_to_same_side_sbc",
                    "execution_hts_guard_failed",
                    "execution_time_to_close_unsafe",
                ],
                time_to_close_seconds=90.0,
            ),
        ],
    )

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="hts-sbc-remap-blocked",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )

    remap = result.hts_time_window_remap_report
    assert remap["hts_time_window_blocked_count"] == 2
    assert remap["same_side_sbc_alternative_available_count"] == 2
    assert remap["same_side_sbc_calibrated_ev_available_count"] == 2
    assert remap["remap_candidate_count"] == 1
    assert remap["remap_guard_passed_count"] == 0
    assert remap["remap_reason_distribution"]["execution_spread_too_wide"] == 1
    assert remap["remap_reason_distribution"][
        "same_side_sbc_calibrated_ev_below_threshold"
    ] == 1
    assert remap["remap_rows"][0]["non_remappable_guard_reason_codes"] == [
        "execution_spread_too_wide"
    ]
    assert remap["paper_only"] is True
    assert remap["capital_at_risk"] is False
    assert remap["polymarket_write_enabled"] is False
    assert remap["wallet_signing_enabled"] is False
    assert remap["v8_execution_handoff_allowed"] is False


def test_execution_layer_v2_forward_shadow_forbidden_outcome_fields_fail_closed(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.json"
    _write_forward_shadow_input(
        input_path,
        [
            {
                **_forward_shadow_row(
                    market_id="forbidden-forward",
                    action="BUY_DOWN_HOLD_TO_SETTLEMENT",
                    selected_side="DOWN",
                    entry_ask=0.80,
                    calibrated_action_expected_net_return=0.05,
                ),
                "settlement_pnl": 1.0,
            }
        ],
    )

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="forward-shadow-forbidden",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )

    assert result.ev_mapping_report["ev_mapping_status"] == (
        "blocked_forbidden_outcome_fields_present"
    )
    assert result.calibrated_ev_source_report["calibrated_ev_source_status"] == (
        "blocked_forbidden_outcome_fields_present"
    )
    assert result.forward_shadow_report["forward_shadow_policy_status"] == (
        "blocked_fail_closed"
    )
    assert result.forward_shadow_report["accepted_signal_row_count"] == 0
    assert result.forward_shadow_report["forbidden_outcome_fields_present"] is True
    assert result.manifest["paper_only"] is True
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False


def test_execution_layer_v2_forward_shadow_nested_forbidden_fields_fail_closed(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh_signal_trace.json"
    row = _forward_shadow_row(
        market_id="nested-forbidden-forward",
        action="BUY_DOWN_HOLD_TO_SETTLEMENT",
        selected_side="DOWN",
        entry_ask=0.80,
        calibrated_action_expected_net_return=0.05,
    )
    row["features"] = {"metadata": {"future_return": 0.1}}
    _write_forward_shadow_input(input_path, [row])

    result = run_execution_layer_v2_forward_shadow_policy(
        ExecutionLayerV2ForwardShadowConfig(
            run_id="forward-shadow-nested-forbidden",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )

    assert result.ev_mapping_report["ev_mapping_status"] == (
        "blocked_forbidden_outcome_fields_present"
    )
    assert result.forward_shadow_report["accepted_signal_row_count"] == 0
    forbidden = result.forward_shadow_report["forbidden_outcome_fields_by_row"]
    assert forbidden[0]["forbidden_fields"] == ["features.metadata.future_return"]
    assert result.guard_intersection_report["guard_intersection_status"] == (
        "blocked_fail_closed"
    )


def test_execution_layer_v2_one_hour_goal_remap_success_with_positive_settlement(
    tmp_path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="one-hour-remap-success",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["time_to_close_seconds"] = 90.0
    _set_action_score(row, "BUY_UP_SELL_BEFORE_CLOSE", 1.50)

    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-success",
            output_dir=tmp_path / "runs",
            duration_seconds=3600,
            poll_interval_seconds=0.0,
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
            settlement_evaluation_rows=(
                {"market_id": "one-hour-remap-success", "settlement_pnl": 0.25},
            ),
        )
    )

    report = result.goal_report
    assert report["schema_version"] == ONE_HOUR_REMAP_PAPER_GOAL_SCHEMA_VERSION
    assert report["duration_seconds"] == 3600
    assert report["complete_round_count"] == 1
    assert report["complete_rounds_with_bet_count"] == 1
    assert report["missing_bet_round_count"] == 0
    assert report["normal_policy_bet_count"] == 0
    assert report["remap_paper_bet_count"] == 1
    assert report["forced_coverage_bet_count"] == 0
    assert report["settled_pnl"] == pytest.approx(0.25)
    assert report["unresolved_pnl"] == 0.0
    assert report["settled_fill_count"] == 1
    assert report["winning_fill_count"] == 1
    assert report["losing_fill_count"] == 0
    assert report["pnl_by_side"] == {"UP": pytest.approx(0.25)}
    assert report["final_goal_success"] is True
    assert report["uses_settlement_pnl_or_outcome_labels_in_decision_logic"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False

    intents = _read_jsonl(result.artifact_paths["paper_intent_log"])
    fills = _read_jsonl(result.artifact_paths["paper_fill_log"])
    settlement_rows = _read_jsonl(result.artifact_paths["settlement_pnl_rows"])
    assert len(intents) == 1
    assert len(fills) == 1
    assert len(settlement_rows) == 1
    assert intents[0]["hts_time_window_remap_applied"] is True
    assert intents[0]["original_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert intents[0]["remapped_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert fills[0]["execution_guarded_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    assert settlement_rows[0]["settlement_status"] == "settled"
    assert result.round_coverage_report["missing_bet_round_count"] == 0
    assert result.remap_execution_report["remap_paper_bet_count"] == 1
    assert "one_hour_remap_paper_goal_report" in result.manifest["artifact_hashes"]
    assert result.manifest["final_goal_success"] is True
    assert result.manifest["v8_execution_handoff_allowed"] is False
    per_round_manifest = json.loads(
        Path(result.goal_report["per_round_artifact_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert report["per_round_async_artifact_flush_enabled"] is True
    assert report["per_round_bet_artifact_count"] == 1
    assert report["per_round_outcome_artifact_count"] == 1
    assert per_round_manifest["round_artifact_rows"][0][
        "paper_bet_artifact_exists"
    ] is True
    assert per_round_manifest["round_artifact_rows"][0][
        "round_outcome_artifact_exists"
    ] is True
    round_row = per_round_manifest["round_artifact_rows"][0]
    assert round_row["settled_pnl"] == pytest.approx(0.25)
    assert round_row["winning_fill_count"] == 1
    round_outcome = json.loads(
        Path(round_row["artifact_paths"]["round_outcome"]).read_text(
            encoding="utf-8"
        )
    )
    assert round_outcome["settled_pnl"] == pytest.approx(0.25)
    assert round_outcome["pnl_by_side"] == {"UP": pytest.approx(0.25)}


def test_execution_layer_v2_one_hour_goal_reports_missing_round_bet_fail_closed(
    tmp_path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="one-hour-remap-blocked",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    row["microstructure_snapshot"]["time_to_close_seconds"] = 30.0
    for candidate in row["full_5_action_ranking"]:
        if candidate["selected_action"] == "BUY_UP_SELL_BEFORE_CLOSE":
            candidate["microstructure_snapshot"] = {
                **row["microstructure_snapshot"],
                "spread_bps": 9_999.0,
            }

    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-fail",
            output_dir=tmp_path / "runs",
            duration_seconds=3600,
            poll_interval_seconds=0.0,
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
        )
    )

    report = result.goal_report
    assert report["complete_round_count"] == 1
    assert report["complete_rounds_with_bet_count"] == 0
    assert report["missing_bet_round_count"] == 1
    assert report["guard_justified_no_bet_round_count"] == 1
    assert report["unjustified_missing_bet_round_count"] == 0
    assert report["guard_justified_no_bet_round_ids"] == ["one-hour-remap-blocked"]
    assert report["unjustified_missing_bet_round_ids"] == []
    assert report["remap_paper_bet_count"] == 0
    assert report["forced_coverage_bet_count"] == 0
    assert report["forced_coverage_guard_blocked_count"] == 1
    assert "execution_time_to_close_unsafe" in report[
        "forced_coverage_blocking_reason_distribution"
    ]
    assert report["forced_coverage_blocker_category_distribution"] == {
        "time_to_close": 1
    }
    attempts = result.remap_execution_report["forced_coverage_attempt_rows"]
    assert attempts[0]["forced_coverage_candidate_attempt_count"] >= 1
    assert attempts[0]["forced_coverage_candidate_attempt_count"] == len(
        attempts[0]["forced_coverage_candidate_attempt_rows"]
    )
    assert attempts[0]["forced_coverage_selected_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert attempts[0]["forced_coverage_blocker_categories"] == ["time_to_close"]
    assert attempts[0]["forced_coverage_missing_runtime_field_codes"] == []
    assert attempts[0]["forced_coverage_p_up_action_disagreement"] is False
    assert report["final_goal_success"] is False
    assert "complete_rounds_unjustified_missing_paper_bets" not in report[
        "goal_failure_reason_codes"
    ]
    assert "settled_pnl_not_positive" in report["goal_failure_reason_codes"]
    assert result.remap_execution_report["remap_paper_bet_count"] == 0
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False


def test_execution_layer_v2_one_hour_goal_succeeds_with_guard_justified_no_bet(
    tmp_path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    bet_row = _paper_fresh_public_row(
        index=1,
        market_id="one-hour-justified-success-bet",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    bet_row["microstructure_snapshot"]["time_to_close_seconds"] = 90.0
    _set_action_score(bet_row, "BUY_UP_SELL_BEFORE_CLOSE", 1.50)
    no_bet_row = _paper_fresh_public_row(
        index=2,
        market_id="one-hour-justified-success-no-bet",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    no_bet_row["microstructure_snapshot"]["time_to_close_seconds"] = 30.0

    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-guard-justified-no-bet-success",
            output_dir=tmp_path / "runs",
            duration_seconds=3600,
            poll_interval_seconds=0.0,
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((bet_row, no_bet_row),),
            settlement_evaluation_rows=(
                {
                    "market_id": "one-hour-justified-success-bet",
                    "settlement_pnl": 0.25,
                },
            ),
        )
    )

    report = result.goal_report
    assert report["complete_round_count"] == 2
    assert report["complete_rounds_with_bet_count"] == 1
    assert report["missing_bet_round_count"] == 1
    assert report["guard_justified_no_bet_round_count"] == 1
    assert report["guard_justified_no_bet_round_ids"] == [
        "one-hour-justified-success-no-bet"
    ]
    assert report["unjustified_missing_bet_round_count"] == 0
    assert report["unjustified_missing_bet_round_ids"] == []
    assert report["settled_pnl"] == pytest.approx(0.25)
    assert report["unresolved_settlement_count"] == 0
    assert report["final_goal_success"] is True
    assert report["goal_failure_reason_codes"] == []
    assert result.round_coverage_report["guard_justified_no_bet_round_count"] == 1
    assert result.manifest["guard_justified_no_bet_round_count"] == 1
    assert result.manifest["unjustified_missing_bet_round_count"] == 0
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["#134_resume_allowed"] is False
    assert result.manifest["#146_start_allowed"] is False


def test_execution_layer_v2_one_hour_goal_unjustified_missing_requires_attempt() -> None:
    classification = _missing_bet_round_classifications(
        missing_round_ids=["missing-without-attempt"],
        forced_coverage={"forced_coverage_attempt_rows": []},
    )

    assert classification["guard_justified_no_bet_round_count"] == 0
    assert classification["unjustified_missing_bet_round_count"] == 1
    assert classification["unjustified_missing_bet_round_ids"] == [
        "missing-without-attempt"
    ]
    assert classification["unjustified_missing_bet_reason_distribution"] == {
        "forced_coverage_attempt_missing": 1
    }


def test_execution_layer_v2_one_hour_goal_forced_coverage_creates_guarded_bet(
    tmp_path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="one-hour-forced-coverage",
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    _set_action_score(row, "NO_TRADE", 1.60)
    _set_action_score(row, "BUY_UP_HOLD_TO_SETTLEMENT", 1.30)
    row["microstructure_snapshot"]["time_to_close_seconds"] = 360.0

    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-forced-coverage",
            output_dir=tmp_path / "runs",
            duration_seconds=3600,
            poll_interval_seconds=0.0,
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
            settlement_evaluation_rows=(
                {"market_id": "one-hour-forced-coverage", "settlement_pnl": 0.25},
            ),
        )
    )

    report = result.goal_report
    intents = _read_jsonl(result.artifact_paths["paper_intent_log"])
    fills = _read_jsonl(result.artifact_paths["paper_fill_log"])
    ledger = _read_jsonl(result.artifact_paths["paper_ledger_log"])
    assert report["complete_round_count"] == 1
    assert report["complete_rounds_with_bet_count"] == 1
    assert report["missing_bet_round_count"] == 0
    assert report["normal_policy_bet_count"] == 0
    assert report["remap_paper_bet_count"] == 0
    assert report["forced_coverage_bet_count"] == 1
    assert report["forced_coverage_guard_passed_count"] == 1
    assert report["forced_coverage_guard_blocked_count"] == 0
    assert report["forced_coverage_round_ids"] == ["one-hour-forced-coverage"]
    assert intents[0]["coverage_forced_paper_bet"] is True
    assert intents[0]["order_origin"] == "forced_coverage_full_guard_paper_only"
    assert fills[0]["execution_guarded_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert ledger[0]["outcome_pnl_used"] is False
    assert result.manifest["forced_coverage_bet_count"] == 1
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_execution_layer_v2_one_hour_goal_forced_coverage_searches_guarded_candidates(
    tmp_path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    blocked_late_row = _paper_fresh_public_row(
        index=2,
        market_id="one-hour-forced-coverage-search",
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    _set_action_score(blocked_late_row, "NO_TRADE", 1.90)
    _set_action_score(blocked_late_row, "BUY_UP_HOLD_TO_SETTLEMENT", 1.80)
    blocked_late_row["microstructure_snapshot"]["time_to_close_seconds"] = 60.0
    blocked_late_row["microstructure_snapshot"]["spread_bps"] = 9_999.0
    safe_early_row = _paper_fresh_public_row(
        index=1,
        market_id="one-hour-forced-coverage-search",
        action="NO_TRADE",
        side="NONE",
        p_up=0.82,
    )
    _set_action_score(safe_early_row, "NO_TRADE", 1.60)
    _set_action_score(safe_early_row, "BUY_UP_HOLD_TO_SETTLEMENT", 1.30)
    safe_early_row["microstructure_snapshot"]["time_to_close_seconds"] = 360.0

    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-forced-coverage-search",
            output_dir=tmp_path / "runs",
            duration_seconds=3600,
            poll_interval_seconds=0.0,
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((blocked_late_row, safe_early_row),),
            settlement_evaluation_rows=(
                {
                    "market_id": "one-hour-forced-coverage-search",
                    "settlement_pnl": 0.25,
                },
            ),
        )
    )

    report = result.goal_report
    attempts = result.remap_execution_report["forced_coverage_attempt_rows"]
    candidate_attempts = attempts[0]["forced_coverage_candidate_attempt_rows"]
    intents = _read_jsonl(result.artifact_paths["paper_intent_log"])

    assert report["forced_coverage_bet_count"] == 1
    assert report["complete_rounds_with_bet_count"] == 1
    assert attempts[0]["forced_coverage_candidate_search_found_guard_passed"] is True
    assert attempts[0]["forced_coverage_candidate_attempt_count"] > 1
    assert candidate_attempts[0]["blocker_categories"] == ["spread", "time_to_close"]
    assert any(row["order_allowed"] is True for row in candidate_attempts)
    assert attempts[0]["forced_coverage_selected_action"] in {
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
    }
    assert attempts[0]["forced_coverage_time_to_close_seconds"] == pytest.approx(360.0)
    assert intents[0]["coverage_forced_paper_bet"] is True
    assert intents[0]["order_origin"] == "forced_coverage_full_guard_paper_only"
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_execution_layer_v2_one_hour_goal_polls_read_only_resolution_provider(
    tmp_path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    row = _paper_fresh_public_row(
        index=1,
        market_id="one-hour-settlement-poll",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.82,
    )
    row.update(
        {
            "condition_id": "one-hour-settlement-poll",
            "slug": "btc-updown-5m-3000",
            "market_family": "btc_updown_5m",
            "market_start_ts": 3_000_000,
            "market_end_ts": 3_300_000,
            "settlement_ts": 3_360_000,
            "up_token_id": "up-token",
            "down_token_id": "down-token",
            "reference_price_source": "polymarket_official_btc_usd_reference",
        }
    )
    row["microstructure_snapshot"]["time_to_close_seconds"] = 360.0

    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-resolution-poll",
            output_dir=tmp_path / "runs",
            duration_seconds=3600,
            poll_interval_seconds=0.0,
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_data_cycles=((row,),),
            public_provider=_OneHourResolvedOutcomeProvider(outcome="UP"),
            settlement_poll_max_wait_seconds=0.0,
            settlement_poll_interval_seconds=1.0,
        )
    )

    report = result.goal_report
    settlement_report = result.settlement_resolution_report
    settlement_rows = _read_jsonl(result.artifact_paths["settlement_pnl_rows"])
    assert settlement_report["settlement_poll_attempt_count"] == 1
    assert settlement_report["settlement_evaluation_row_count"] == 1
    assert settlement_report["unresolved_fill_count_after_poll"] == 0
    assert "settlement_resolution_all_fills_resolved" in settlement_report[
        "settlement_resolution_reason_codes"
    ]
    assert settlement_rows[0]["settlement_status"] == "settled"
    assert settlement_rows[0]["resolved_outcome"] == "UP"
    assert report["settled_pnl"] > 0.0
    assert report["unresolved_settlement_count"] == 0
    assert report["final_goal_success"] is True
    assert report["uses_settlement_pnl_or_outcome_labels_in_decision_logic"] is False
    assert result.manifest["settlement_evaluation_row_count"] == 1
    assert "settlement_resolution_report" in result.manifest["artifact_hashes"]
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_execution_layer_v2_one_hour_goal_settlement_resolution_times_out_fail_closed(
    tmp_path,
) -> None:
    report = _settlement_resolution_report(
        config=ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-resolution-timeout",
            output_dir=tmp_path / "runs",
            public_provider=_SlowResolutionProvider(),
            settlement_poll_max_wait_seconds=0.01,
            settlement_poll_interval_seconds=0.01,
        ),
        fills=[
            {
                "paper_fresh_order_intent_id": "intent-timeout",
                "market_id": "one-hour-resolution-timeout",
                "execution_guarded_side": "UP",
                "paper_fill_price": 0.60,
                "filled_size": 0.20,
                "total_execution_cost": 0.0,
                "paper_only": True,
                "capital_at_risk": False,
            }
        ],
        trace_rows=[
            {
                "market_id": "one-hour-resolution-timeout",
                "condition_id": "one-hour-resolution-timeout",
                "slug": "btc-updown-5m-timeout",
                "market_family": "btc_updown_5m",
                "market_start_ts": 4_000_000,
                "market_end_ts": 4_300_000,
                "settlement_ts": 4_360_000,
                "up_token_id": "up-token",
                "down_token_id": "down-token",
                "reference_price_source": "polymarket_official_btc_usd_reference",
                "reference_price_start": 100_000.0,
            }
        ],
        settlement_evaluation_rows=[],
    )

    assert report["settlement_poll_attempt_count"] == 1
    assert "settlement_resolution_provider_timeout" in report[
        "settlement_resolution_reason_codes"
    ]
    assert "settlement_resolution_http_timeout" in report[
        "settlement_resolution_reason_codes"
    ]
    assert report["settlement_evaluation_row_count"] == 0
    assert report["resolved_fill_count"] == 0
    assert report["unresolved_fill_count_after_poll"] == 1
    assert report["unresolved_paper_fresh_order_intent_ids"] == ["intent-timeout"]
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["v8_execution_handoff_allowed"] is False


def test_execution_layer_v2_one_hour_goal_stops_after_consecutive_orderbook_failures(
    tmp_path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    provider = _NoOrderbookProvider()

    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-orderbook-fail-fast",
            output_dir=tmp_path / "runs",
            duration_seconds=3600,
            poll_interval_seconds=3600.0,
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_provider=provider,
            max_consecutive_orderbook_failure_rounds=1,
        )
    )

    report = result.goal_report
    fresh_report = result.manifest
    assert provider.orderbook_calls == 1
    assert report["provider_fail_fast_stop_triggered"] is True
    assert report["max_consecutive_orderbook_failure_rounds"] == 1
    assert report["consecutive_orderbook_failure_count_at_stop"] == 1
    assert "orderbook_collection_failed_consecutive_limit" in report[
        "goal_failure_reason_codes"
    ]
    assert "consecutive_orderbook_collection_failures_exceeded_limit" in report[
        "provider_fail_fast_reason_codes"
    ]
    assert report["complete_round_count"] == 0
    assert report["paper_intent_count"] == 0
    assert report["final_goal_success"] is False
    assert fresh_report["provider_fail_fast_stop_triggered"] is True
    assert result.manifest["v8_execution_handoff_allowed"] is False
    status_rows = _read_jsonl(
        result.output_dir
        / "incremental_fresh_loop"
        / "provider_cycle_status.jsonl"
    )
    assert status_rows[0]["orderbook_failure"] is True
    assert status_rows[0]["public_orderbook_row_count"] == 0


def _write_settlement_csv(path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "market_id",
        "decision_ts",
        "action",
        "horizon_ms",
        "entry_price",
        "cost_basis",
        "settlement_pnl",
        "iteration",
        "intent_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hts_regime_intent(
    intent_id: str,
    market_id: str,
    action: str,
    p_up: float,
    p_down: float,
    entry_price: float,
) -> dict[str, object]:
    side = "UP" if "BUY_UP" in action else "DOWN"
    return {
        "paper_fresh_order_intent_id": intent_id,
        "market_id": market_id,
        "decision_ts": 1_000 + len(intent_id),
        "execution_guarded_action": action,
        "execution_guarded_family": (
            "SELL_BEFORE_CLOSE"
            if "SELL_BEFORE_CLOSE" in action
            else "HOLD_TO_SETTLEMENT"
        ),
        "execution_guarded_side": side,
        "p_up": p_up,
        "p_down": p_down,
        "paper_limit_price": entry_price,
        "time_to_close_seconds": 180.0,
        "spread_bps": 200.0,
        "book_staleness_ms": 500.0,
        "queue_fill_proxy": 0.80,
        "source_model_score": 0.10,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _hts_regime_fill(
    intent_id: str,
    market_id: str,
    action: str,
    side: str,
    price: float,
) -> dict[str, object]:
    return {
        "paper_fresh_order_intent_id": intent_id,
        "paper_fresh_fill_id": f"fill-{intent_id}",
        "market_id": market_id,
        "decision_ts": 1_000 + len(intent_id),
        "execution_guarded_action": action,
        "execution_guarded_side": side,
        "paper_fill_price": price,
        "filled_size": 0.20,
        "total_execution_cost": 0.001,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _hts_regime_settlement(
    intent_id: str,
    market_id: str,
    action: str,
    side: str,
    outcome: str,
    pnl: float,
) -> dict[str, object]:
    return {
        "paper_fresh_order_intent_id": intent_id,
        "paper_fresh_fill_id": f"fill-{intent_id}",
        "market_id": market_id,
        "decision_ts": 1_000 + len(intent_id),
        "execution_guarded_action": action,
        "execution_guarded_side": side,
        "settlement_status": "settled",
        "resolved_outcome": outcome,
        "settlement_pnl": pnl,
        "paper_only": True,
        "capital_at_risk": False,
        "uses_settlement_pnl_for_decision_time_logic": False,
    }


def _write_forward_shadow_input(path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"trace_rows": rows}, sort_keys=True), encoding="utf-8")


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _set_action_score(row: dict[str, object], action: str, score: float) -> None:
    for candidate in row["full_5_action_ranking"]:
        if candidate["selected_action"] == action:
            candidate["corrected_model_score"] = score
            return
    raise AssertionError(f"missing action in full ranking: {action}")


class _OneHourResolvedOutcomeProvider:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(self, *, outcome: str) -> None:
        self.outcome = outcome
        self.resolution_calls = 0

    def resolution_rows(self, markets, config):
        del config
        self.resolution_calls += 1
        rows = []
        for market in markets:
            rows.append(
                {
                    "market_id": market["market_id"],
                    "reference_price_source": market["reference_price_source"],
                    "resolution_status": "normal",
                    "resolved_outcome": self.outcome,
                    "payout_up": 1.0 if self.outcome == "UP" else 0.0,
                    "payout_down": 1.0 if self.outcome == "DOWN" else 0.0,
                    "resolution_source_type": "pytest_read_only_resolution_provider",
                    "raw_resolution_text": "pytest resolved outcome",
                    "paper_only": True,
                    "capital_at_risk": False,
                    "broker_exchange_write_enabled": False,
                    "live_exchange_write_enabled": False,
                    "polymarket_write_enabled": False,
                    "wallet_signing_enabled": False,
                }
            )
        return rows


class _NoOrderbookProvider:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(self) -> None:
        self.orderbook_calls = 0

    def market_rows(self, config):
        del config
        return [
            {
                "market_id": "no-orderbook-round",
                "condition_id": "no-orderbook-condition",
                "slug": "btc-updown-5m-no-orderbook",
                "market_family": "btc_updown_5m",
                "up_token_id": "up-token",
                "down_token_id": "down-token",
                "market_start_ts": 1_000_000,
                "market_end_ts": 1_300_000,
                "reference_price_source": "polymarket_official_btc_usd_reference",
                "reference_price_start": 100_000.0,
            }
        ]

    def orderbook_rows(self, markets, config):
        del markets, config
        self.orderbook_calls += 1
        return []

    def trade_rows(self, markets, config):
        del config
        return [
            {
                "market_id": market["market_id"],
                "price": 0.50,
                "size": 1.0,
                "ts": 1_010_000,
            }
            for market in markets
        ]

    def btc_feature_candle_rows(self, markets, config):
        del markets, config
        return [
            {
                "ts": 1_000_000,
                "available_at_ts": 1_000_000,
                "close_price": 100_001.0,
                "source": "pytest",
            }
        ]


class _SlowResolutionProvider:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def resolution_rows(self, markets, config):
        del markets, config
        time.sleep(0.25)
        return []


def _write_ev_calibration_artifact(
    path,
    score_to_expected_net_return: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "test-frozen-ev-calibration-v1",
                "frozen": True,
                "decision_time_safe": True,
                "uses_validation_labels_for_tuning": False,
                "market_implied_probability_used_for_ev": False,
                "subtract_execution_cost": True,
                "score_to_expected_net_return": score_to_expected_net_return,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _settlement_row(
    action: str,
    horizon_ms: int,
    entry_price: float,
    cost_basis: float,
    settlement_pnl: float,
    iteration: int,
) -> dict[str, object]:
    return {
        "market_id": f"{action}-{horizon_ms}-{entry_price}",
        "decision_ts": str(horizon_ms + int(entry_price * 1000)),
        "action": action,
        "horizon_ms": horizon_ms,
        "entry_price": entry_price,
        "cost_basis": cost_basis,
        "settlement_pnl": settlement_pnl,
        "iteration": iteration,
        "intent_id": f"intent-{iteration:03d}",
    }


def _full_grid_row(
    *,
    market_id: str,
    decision_ts: int,
    p_up: float,
    up_ask: float,
    down_ask: float,
    time_to_close: float,
) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "p_up": p_up,
        "p_down": 1.0 - p_up,
        "selected_action": "BUY_UP_HOLD_TO_SETTLEMENT" if p_up >= 0.5 else "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "selected_side": "UP" if p_up >= 0.5 else "DOWN",
        "full_5_action_ranking": [
            _action_row("BUY_UP_HOLD_TO_SETTLEMENT", "UP", up_ask, up_ask - 0.01, time_to_close),
            _action_row("BUY_UP_SELL_BEFORE_CLOSE", "UP", up_ask, up_ask - 0.01, time_to_close),
            _action_row("BUY_DOWN_HOLD_TO_SETTLEMENT", "DOWN", down_ask, down_ask - 0.01, time_to_close),
            _action_row("BUY_DOWN_SELL_BEFORE_CLOSE", "DOWN", down_ask, down_ask - 0.01, time_to_close),
            {"action": "NO_TRADE", "side": "NONE", "microstructure_snapshot": {}},
        ],
        "paper_only": True,
        "capital_at_risk": False,
    }


def _forward_shadow_row(
    *,
    market_id: str,
    action: str,
    selected_side: str,
    entry_ask: float,
    decision_ts: int = 1_000,
    p_up: float | None = None,
    p_model_fair_value_up: float | None = None,
    p_model_fair_value_down: float | None = None,
    calibrated_action_expected_net_return: float | None = None,
    canonical_o_action_score: float = 0.75,
    time_to_close_seconds: float = 180.0,
    order_allowed: bool | None = None,
    execution_guarded_action: str | None = None,
    execution_guarded_side: str | None = None,
    execution_blocking_reason_codes: list[str] | None = None,
    paper_intent_id: str | None = None,
    paper_fill_id: str | None = None,
    available_actions: list[str] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": selected_side,
        "entry_ask": entry_ask,
        "canonical_o_action_score": canonical_o_action_score,
        "time_to_close_seconds": time_to_close_seconds,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    if p_up is not None:
        row["p_up"] = p_up
    if p_model_fair_value_up is not None:
        row["p_model_fair_value_up"] = p_model_fair_value_up
    if p_model_fair_value_down is not None:
        row["p_model_fair_value_down"] = p_model_fair_value_down
    if calibrated_action_expected_net_return is not None:
        row["calibrated_action_expected_net_return"] = (
            calibrated_action_expected_net_return
        )
    if order_allowed is not None:
        row["order_allowed"] = order_allowed
    if execution_guarded_action is not None:
        row["execution_guarded_action"] = execution_guarded_action
    if execution_guarded_side is not None:
        row["execution_guarded_side"] = execution_guarded_side
    if execution_blocking_reason_codes is not None:
        row["execution_blocking_reason_codes"] = execution_blocking_reason_codes
    if paper_intent_id is not None:
        row["paper_intent_id"] = paper_intent_id
    if paper_fill_id is not None:
        row["paper_fill_id"] = paper_fill_id
    if available_actions is not None:
        row["available_actions"] = available_actions
    return row


def _action_row(
    action: str,
    side: str,
    entry_ask: float,
    exit_bid: float,
    time_to_close: float,
) -> dict:
    return {
        "action": action,
        "side": side,
        "microstructure_snapshot": {
            "entry_ask": entry_ask,
            "executable_exit_bid_proxy": exit_bid,
            "time_to_close_seconds": time_to_close,
        },
    }
