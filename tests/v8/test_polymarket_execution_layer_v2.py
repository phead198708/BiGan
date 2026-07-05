from __future__ import annotations

import csv
import json

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
from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    ExecutionLayerV2ForwardShadowConfig,
    ExecutionLayerV2PolicyReplayConfig,
    run_execution_layer_v2_forward_shadow_policy,
    run_execution_layer_v2_policy_replay_from_settlement_csv,
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


def _write_forward_shadow_input(path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"trace_rows": rows}, sort_keys=True), encoding="utf-8")


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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
