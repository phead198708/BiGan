from __future__ import annotations

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
