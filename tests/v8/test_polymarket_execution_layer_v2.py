from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pytest

from bigan.v8.polymarket.recorder.public_provider import (
    PolymarketPublicHTTPRealCorpusProvider,
    RealCorpusPublicProviderError,
)
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
from bigan.v8.polymarket.training.execution_layer_v2_historical_outcome_reconciliation import (
    ExecutionLayerV2HistoricalOutcomeReconciliationConfig,
    run_execution_layer_v2_historical_outcome_reconciliation,
)
from bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal import (
    ONE_HOUR_REMAP_PAPER_GOAL_SCHEMA_VERSION,
    ExecutionLayerV2OneHourRemapPaperGoalConfig,
    _clob_resolution_rows_for_markets,
    _missing_bet_round_classifications,
    _raw_evidence_completeness_report,
    _settlement_resolution_report,
    _write_per_round_artifacts,
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
from bigan.v8.polymarket.training.execution_layer_v2_pre_promotion_readiness import (
    ExecutionLayerV2PrePromotionFinalizationConfig,
    ExecutionLayerV2PrePromotionGoalConfig,
    finalize_pre_promotion_readiness_goal,
    initialize_pre_promotion_readiness_goal,
)
from bigan.v8.polymarket.training.execution_layer_v2_pre_promotion_remediation import (
    ExecutionLayerV2PrePromotionRemediationConfig,
    _candidate_specifications,
    initialize_pre_promotion_remediation_goal,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev import (
    CURRENT_75_ROW_REPLAY_RUN_ID,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID,
    ExecutionLayerV2RegimeConditionedEVForwardShadowConfig,
    run_execution_layer_v2_regime_conditioned_ev_forward_shadow,
    validate_frozen_regime_conditioned_ev_artifact,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev_calibration import (
    ExecutionLayerV2RegimeConditionedEVCalibrationConfig,
    _validation_coverage_gate,
    regime_conditioned_ev_v2_calibration_row_identity,
    run_execution_layer_v2_regime_conditioned_ev_calibration,
    validate_regime_conditioned_ev_v2_calibration_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev_corpus import (
    ExecutionLayerV2RegimeConditionedEVCorpusConfig,
    run_execution_layer_v2_regime_conditioned_ev_corpus_builder,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    _fresh_public_row_from_provider_feature_context,
    _trace_market_schedule,
    _trace_time_to_close_seconds,
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


def test_regime_conditioned_ev_forward_shadow_valid_artifact_and_full_guard(
    tmp_path,
) -> None:
    schema = json.loads(
        Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_frozen_regime_conditioned_ev_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["artifact_name"]["const"] == (
        "execution_layer_v2_frozen_regime_conditioned_ev_v1"
    )
    assert "market_implied_probability_used_for_ev" not in schema["properties"]
    input_path = tmp_path / "fresh-regime-trace.jsonl"
    artifact_path = tmp_path / "frozen-regime-ev.json"
    _write_regime_conditioned_ev_artifact(artifact_path)
    pass_row = _regime_conditioned_forward_row(
        market_id="regime-up-pass",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.64,
        order_allowed=True,
        blocking_reason_codes=[],
    )
    blocked_row = _regime_conditioned_forward_row(
        market_id="regime-down-blocked",
        action="BUY_DOWN_SELL_BEFORE_CLOSE",
        side="DOWN",
        p_up=0.42,
        order_allowed=False,
        blocking_reason_codes=["execution_spread_too_wide"],
    )
    missing_row = _regime_conditioned_forward_row(
        market_id="regime-missing-margin",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.62,
        order_allowed=True,
        blocking_reason_codes=[],
    )
    missing_row.pop("action_score_margin")
    _write_jsonl(input_path, [pass_row, blocked_row, missing_row])

    result = run_execution_layer_v2_regime_conditioned_ev_forward_shadow(
        ExecutionLayerV2RegimeConditionedEVForwardShadowConfig(
            run_id="regime-conditioned-forward-shadow-valid",
            input_path=input_path,
            output_dir=tmp_path / "runs",
            frozen_regime_conditioned_ev_artifact=artifact_path,
        )
    )
    report = result.forward_shadow_report

    assert result.artifact_validation_report["artifact_valid"] is True
    assert report["regime_conditioned_ev_produced_count"] == 2
    assert report["regime_conditioned_ev_missing_count"] == 1
    assert report["candidate_count"] == 2
    assert report["full_guard_passed_count"] == 1
    assert report["executable_shadow_count"] == 1
    assert report["counts_by_stage"]["candidate"]["by_side"] == {
        "DOWN": 1,
        "UP": 1,
    }
    assert report["counts_by_stage"]["full_guard_passed"]["by_action"] == {
        "BUY_UP_HOLD_TO_SETTLEMENT": 1
    }
    assert report["feature_coverage"]["btc_momentum"]["available_count"] == 3
    assert report["feature_coverage"]["action_score_margin"]["missing_count"] == 1
    assert report["provenance_coverage"]["violation_count"] == 0
    assert report["decision_rows"][0]["regime_conditioned_ev_source"] == (
        "execution_layer_v2_frozen_regime_conditioned_ev_v1"
    )
    assert report["decision_rows"][0][
        "market_implied_probability_used_as_direct_fair_value_ev"
    ] is False
    assert report["decision_rows"][0][
        "market_implied_probability_used_as_conditioning_feature"
    ] is True
    assert report["decision_rows"][0][
        "market_implied_probability_used_as_regime_direction_vote"
    ] is False
    assert report["market_implied_probability_used_as_direct_fair_value_ev"] is False
    assert report["market_implied_probability_used_as_conditioning_feature"] is True
    assert report["market_implied_probability_used_as_regime_direction_vote"] is False
    assert report["p_up_p_down_used_only_in_market_price_value_group"] is True
    assert (
        report["correlated_momentum_reference_counted_as_independent_votes"]
        is False
    )
    assert report["uses_settlement_pnl_or_outcome_labels"] is False
    assert report["source_scores_mutated"] is False
    assert report["o_score_mutated"] is False
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
    assert result.artifact_hashes[
        "execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest"
    ]
    assert result.manifest["frozen_regime_conditioned_ev_contract_hash"]
    assert result.manifest[
        "market_implied_probability_used_as_direct_fair_value_ev"
    ] is False
    assert result.manifest[
        "market_implied_probability_used_as_conditioning_feature"
    ] is True
    assert result.manifest[
        "market_implied_probability_used_as_regime_direction_vote"
    ] is False
    assert result.manifest["legacy_ambiguous_probability_flag_present"] is False
    recommendation = report["future_v2_probability_value_contract_recommendation"]
    assert recommendation["fields"] == [
        "selected_side_probability",
        "execution_price",
        "selected_side_probability_minus_execution_price",
    ]
    assert recommendation["real_coefficients_created"] is False
    validation_report = result.artifact_validation_report
    assert validation_report[
        "market_implied_probability_used_as_direct_fair_value_ev"
    ] is False
    assert validation_report[
        "market_implied_probability_used_as_conditioning_feature"
    ] is True
    assert validation_report[
        "market_implied_probability_used_as_regime_direction_vote"
    ] is False
    assert validation_report["legacy_ambiguous_probability_flag_present"] is False


def test_regime_conditioned_ev_artifact_rejects_double_count_and_current_replay(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "invalid-regime-ev.json"
    payload = _regime_conditioned_ev_artifact_payload()
    payload["feature_groups"]["btc_anchor_direction"]["features"].append("p_up")
    payload["fit_provenance"]["fitted_from_run_ids"] = [
        CURRENT_75_ROW_REPLAY_RUN_ID
    ]
    payload["fit_provenance"][
        "coefficients_fitted_from_current_75_row_replay"
    ] = True
    payload["settlement_pnl"] = 0.0
    artifact_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    validation = validate_frozen_regime_conditioned_ev_artifact(artifact_path)

    assert validation["valid"] is False
    assert "current_75_row_replay_present_in_fit_lineage" in validation[
        "blocking_reason_codes"
    ]
    assert (
        "regime_conditioned_ev_fit_provenance_"
        "coefficients_fitted_from_current_75_row_replay_not_false"
        in validation["blocking_reason_codes"]
    )
    assert "regime_conditioned_ev_feature_group_fields_mismatch:btc_anchor_direction" in validation[
        "blocking_reason_codes"
    ]
    assert "regime_conditioned_ev_artifact_forbidden_fields_present" in validation[
        "blocking_reason_codes"
    ]
    assert validation["forbidden_field_paths"] == ["settlement_pnl"]


def test_regime_conditioned_ev_legacy_ambiguous_probability_flag_fails_closed(
    tmp_path,
) -> None:
    input_path = tmp_path / "legacy-probability-trace.jsonl"
    artifact_path = tmp_path / "legacy-regime-ev.json"
    payload = _regime_conditioned_ev_artifact_payload()
    payload.pop("market_implied_probability_used_as_direct_fair_value_ev")
    payload.pop("market_implied_probability_used_as_conditioning_feature")
    payload.pop("market_implied_probability_used_as_regime_direction_vote")
    payload["market_implied_probability_used_for_ev"] = False
    artifact_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    _write_jsonl(
        input_path,
        [
            _regime_conditioned_forward_row(
                market_id="legacy-ambiguous-artifact",
                action="BUY_UP_HOLD_TO_SETTLEMENT",
                side="UP",
                p_up=0.65,
                order_allowed=True,
                blocking_reason_codes=[],
            )
        ],
    )

    result = run_execution_layer_v2_regime_conditioned_ev_forward_shadow(
        ExecutionLayerV2RegimeConditionedEVForwardShadowConfig(
            run_id="legacy-ambiguous-regime-artifact",
            input_path=input_path,
            output_dir=tmp_path / "runs",
            frozen_regime_conditioned_ev_artifact=artifact_path,
        )
    )
    validation = result.artifact_validation_report
    report = result.forward_shadow_report

    assert validation["artifact_valid"] is False
    assert validation["legacy_ambiguous_probability_flag_present"] is True
    assert (
        "legacy_ambiguous_market_implied_probability_used_for_ev_present"
        in validation["artifact_blocking_reason_codes"]
    )
    assert report["forward_shadow_status"] == "blocked_fail_closed"
    assert report["regime_conditioned_ev_produced_count"] == 0
    assert report["candidate_count"] == 0
    assert report["full_guard_passed_count"] == 0
    assert report["executable_shadow_count"] == 0


def test_regime_conditioned_ev_forward_shadow_missing_artifact_fails_closed(
    tmp_path,
) -> None:
    input_path = tmp_path / "fresh-regime-trace.jsonl"
    _write_jsonl(
        input_path,
        [
            _regime_conditioned_forward_row(
                market_id="missing-artifact",
                action="BUY_UP_HOLD_TO_SETTLEMENT",
                side="UP",
                p_up=0.65,
                order_allowed=True,
                blocking_reason_codes=[],
            )
        ],
    )

    result = run_execution_layer_v2_regime_conditioned_ev_forward_shadow(
        ExecutionLayerV2RegimeConditionedEVForwardShadowConfig(
            run_id="regime-conditioned-forward-shadow-missing-artifact",
            input_path=input_path,
            output_dir=tmp_path / "runs",
        )
    )
    report = result.forward_shadow_report

    assert report["forward_shadow_status"] == "blocked_fail_closed"
    assert report["regime_conditioned_ev_produced_count"] == 0
    assert report["regime_conditioned_ev_missing_count"] == 1
    assert report["candidate_count"] == 0
    assert report["full_guard_passed_count"] == 0
    assert report["executable_shadow_count"] == 0
    assert report["rejection_reason_distribution"] == {
        "missing_frozen_regime_conditioned_ev_artifact": 1
    }
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["promotion_evidence_eligible"] is False


def test_regime_conditioned_ev_forward_shadow_forbidden_outcome_fails_closed(
    tmp_path,
) -> None:
    input_path = tmp_path / "forbidden-regime-trace.jsonl"
    artifact_path = tmp_path / "frozen-regime-ev.json"
    _write_regime_conditioned_ev_artifact(artifact_path)
    row = _regime_conditioned_forward_row(
        market_id="forbidden-outcome",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.65,
        order_allowed=True,
        blocking_reason_codes=[],
    )
    row["settlement_pnl"] = 1.0
    _write_jsonl(input_path, [row])

    report = run_execution_layer_v2_regime_conditioned_ev_forward_shadow(
        ExecutionLayerV2RegimeConditionedEVForwardShadowConfig(
            run_id="regime-conditioned-forward-shadow-forbidden",
            input_path=input_path,
            output_dir=tmp_path / "runs",
            frozen_regime_conditioned_ev_artifact=artifact_path,
        )
    ).forward_shadow_report

    assert report["forward_shadow_status"] == "blocked_fail_closed"
    assert report["forbidden_outcome_fields_present"] is True
    assert report["accepted_signal_row_count"] == 0
    assert report["candidate_count"] == 0
    assert report["executable_shadow_count"] == 0
    assert report["v8_execution_handoff_allowed"] is False


def test_regime_conditioned_ev_v2_calibration_protocol_and_future_shadow(
    tmp_path,
) -> None:
    schema = json.loads(
        Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_frozen_regime_conditioned_ev_v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["artifact_name"]["const"] == (
        "execution_layer_v2_frozen_regime_conditioned_ev_v2"
    )
    calibration_path = tmp_path / "calibration.jsonl"
    rows = _regime_conditioned_ev_v2_calibration_rows()
    rows.extend(
        [
            _regime_conditioned_ev_v2_calibration_row(
                source_run_id=CURRENT_75_ROW_REPLAY_RUN_ID,
                market_index=90,
                row_index=0,
            ),
            _regime_conditioned_ev_v2_calibration_row(
                source_run_id=LATEST_ONE_HOUR_RECONCILED_RUN_ID,
                market_index=91,
                row_index=0,
            ),
            _regime_conditioned_ev_v2_calibration_row(
                source_run_id="future-unseen-forward-shadow-fixture",
                market_index=92,
                row_index=0,
            ),
        ]
    )
    _attach_regime_conditioned_ev_v2_target_source(rows, tmp_path)
    _write_jsonl(calibration_path, rows)
    future_path = tmp_path / "future.jsonl"
    future_row = _regime_conditioned_forward_row(
        market_id="future-disjoint-market",
        action="BUY_UP_HOLD_TO_SETTLEMENT",
        side="UP",
        p_up=0.72,
        order_allowed=True,
        blocking_reason_codes=[],
    )
    future_row["decision_ts"] = 30_000
    future_row["decision_time_regime_feature_max_input_ts"] = 30_000
    _write_jsonl(future_path, [future_row])

    result = run_execution_layer_v2_regime_conditioned_ev_calibration(
        ExecutionLayerV2RegimeConditionedEVCalibrationConfig(
            run_id="v2-calibration-fixture",
            input_path=calibration_path,
            output_dir=tmp_path / "runs",
            future_shadow_input_path=future_path,
            validation_fraction=0.25,
            ridge_alpha=0.01,
            entry_ev_threshold=0.0,
            min_fit_rows=16,
            min_validation_rows=8,
            min_fit_markets=4,
            min_validation_markets=2,
            bootstrap_samples=200,
            min_validation_rows_per_side=2,
            min_validation_rows_per_action_family=2,
            min_validation_rows_per_resolved_outcome=2,
        )
    )

    split = result.split_report
    report = result.calibration_report
    assert split["excluded_from_fit_row_count"] == 3
    assert split["chronological_split_passed"] is True
    assert split["market_id_disjointness_passed"] is True
    assert split["feature_max_input_ts_violation_count"] == 0
    assert split["uses_validation_labels_for_fitting"] is False
    assert split["uses_validation_labels_for_threshold_selection"] is False
    assert split["leakage_checks_passed"] is True
    assert split["schema_validation_row_count"] == len(rows)
    assert split["schema_runtime_validation_agreement_passed"] is True
    assert split["invalid_row_reason_distribution"] == {}
    assert report["artifact_created"] is True
    assert report["validation_improved_over_constant_and_legacy"] is True
    assert report["statistical_eligibility_passed"] is True
    assert report["market_level_metrics"]["validation_candidate"][
        "market_count"
    ] == 2
    assert report["relative_baseline_improvements"][
        "row_level_gate_passed"
    ] is True
    assert report["relative_baseline_improvements"][
        "market_level_gate_passed"
    ] is True
    assert report["market_bootstrap_confidence_intervals"][
        "confidence_gate_passed"
    ] is True
    assert report["coefficient_stability_metrics"][
        "stability_gate_passed"
    ] is True
    assert report["validation_coverage"]["coverage_gate_passed"] is True
    assert report["validation_coverage"]["side_unique_market_counts"] == {
        "DOWN": 2,
        "UP": 2,
    }
    assert report["coefficients_finite_and_bounded"] is True
    assert report["threshold_selection_source"] == "fixed_pre_validation_config"
    artifact_path = result.artifact_paths["frozen_artifact"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    validation = validate_frozen_regime_conditioned_ev_artifact(artifact_path)
    assert validation["valid"] is True
    mismatched_schema_artifact = tmp_path / "mismatched-schema-artifact.json"
    mismatched_payload = json.loads(json.dumps(artifact))
    mismatched_payload["fit_provenance"]["calibration_row_schema_sha256"] = "d" * 64
    mismatched_schema_artifact.write_text(
        json.dumps(mismatched_payload, sort_keys=True), encoding="utf-8"
    )
    mismatched_validation = validate_frozen_regime_conditioned_ev_artifact(
        mismatched_schema_artifact
    )
    assert mismatched_validation["valid"] is False
    assert (
        "regime_conditioned_ev_v2_calibration_row_schema_hash_mismatch"
        in mismatched_validation["blocking_reason_codes"]
    )
    assert artifact["fit_provenance"][
        "settled_outcomes_or_pnl_used_as_training_targets"
    ] is True
    assert artifact["fit_provenance"][
        "settled_outcomes_or_pnl_used_as_decision_time_inputs"
    ] is False
    assert artifact["coefficients"]["subtract_execution_cost"] is False
    assert artifact["fit_provenance"]["statistical_eligibility_passed"] is True
    assert len(
        artifact["fit_provenance"]["statistical_eligibility_config_hash"]
    ) == 64
    assert len(
        artifact["fit_provenance"]["statistical_eligibility_summary_hash"]
    ) == 64
    assert artifact["feature_groups"]["market_price_value"]["features"] == [
        "selected_side_probability",
        "execution_price",
        "selected_side_probability_minus_execution_price",
    ]
    fitted_run_ids = artifact["fit_provenance"]["fitted_from_run_ids"]
    assert CURRENT_75_ROW_REPLAY_RUN_ID not in fitted_run_ids
    assert LATEST_ONE_HOUR_RECONCILED_RUN_ID not in fitted_run_ids
    assert not any("forward-shadow" in run_id for run_id in fitted_run_ids)
    shadow = report["future_shadow"]
    assert shadow["regime_conditioned_ev_produced_count"] == 1
    assert shadow["outcome_free"] is True
    assert shadow["outcomes_reconciled"] is False
    assert shadow["refit_performed"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["source_model_candidate_eligible"] is False
    assert result.manifest["freeze_ready"] is False
    assert result.manifest["promotion_evidence_eligible"] is False


def test_regime_conditioned_ev_v2_missing_validation_family_fails_closed() -> None:
    rows = [
        {
            "selected_side": "UP" if index % 2 == 0 else "DOWN",
            "action_family": "HOLD_TO_SETTLEMENT",
            "resolved_outcome": "UP" if index % 2 == 0 else "DOWN",
            "market_id": f"market-{index}",
        }
        for index in range(10)
    ]

    report = _validation_coverage_gate(
        rows,
        min_rows_per_side=2,
        min_rows_per_action_family=2,
        min_rows_per_resolved_outcome=2,
        min_markets_per_category=2,
    )

    assert report["action_family_counts"]["SELL_BEFORE_CLOSE"] == 0
    assert report["action_family_unique_market_counts"]["SELL_BEFORE_CLOSE"] == 0
    assert report["action_family_coverage_passed"] is False
    assert report["coverage_gate_passed"] is False
    assert "validation_action_family_coverage_gate_failed" in report[
        "blocking_reason_codes"
    ]
    assert "validation_action_family_market_coverage_gate_failed" in report[
        "blocking_reason_codes"
    ]


def test_regime_conditioned_ev_v2_validation_labels_do_not_change_fit(
    tmp_path,
) -> None:
    base_rows = _regime_conditioned_ev_v2_calibration_rows()
    _attach_regime_conditioned_ev_v2_target_source(base_rows, tmp_path)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_jsonl(first_path, base_rows)
    changed_rows = json.loads(json.dumps(base_rows))
    for row in changed_rows:
        if int(str(row["market_id"]).rsplit("-", maxsplit=1)[-1]) >= 6:
            row["target_net_return_after_cost"] *= -1.0
    _write_jsonl(second_path, changed_rows)

    def run(input_path, run_id):
        return run_execution_layer_v2_regime_conditioned_ev_calibration(
            ExecutionLayerV2RegimeConditionedEVCalibrationConfig(
                run_id=run_id,
                input_path=input_path,
                output_dir=tmp_path / "runs",
                validation_fraction=0.25,
                ridge_alpha=0.01,
                entry_ev_threshold=0.017,
                min_fit_rows=16,
                min_validation_rows=8,
                min_fit_markets=4,
                min_validation_markets=2,
            )
        )

    first = run(first_path, "first")
    second = run(second_path, "second")
    assert first.calibration_report["fit_coefficients_hash"] == second.calibration_report[
        "fit_coefficients_hash"
    ]
    assert first.calibration_report["entry_ev_threshold"] == 0.017
    assert second.calibration_report["entry_ev_threshold"] == 0.017
    assert first.split_report["uses_validation_labels_for_fitting"] is False
    assert second.split_report["uses_validation_labels_for_threshold_selection"] is False


def test_regime_conditioned_ev_v2_causality_violation_blocks_artifact(
    tmp_path,
) -> None:
    rows = _regime_conditioned_ev_v2_calibration_rows()
    _attach_regime_conditioned_ev_v2_target_source(rows, tmp_path)
    rows[0]["max_input_ts"] = rows[0]["decision_ts"] + 1
    input_path = tmp_path / "causality-invalid.jsonl"
    _write_jsonl(input_path, rows)

    result = run_execution_layer_v2_regime_conditioned_ev_calibration(
        ExecutionLayerV2RegimeConditionedEVCalibrationConfig(
            run_id="causality-invalid",
            input_path=input_path,
            output_dir=tmp_path / "runs",
            min_fit_rows=16,
            min_validation_rows=8,
            min_fit_markets=4,
            min_validation_markets=2,
        )
    )

    assert result.calibration_report["artifact_created"] is False
    assert "invalid_calibration_rows_present" in result.calibration_report[
        "blocking_reason_codes"
    ]
    assert "frozen_artifact" not in result.artifact_paths
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_regime_conditioned_ev_v2_strict_row_validation_fails_closed(
    tmp_path,
) -> None:
    rows = _regime_conditioned_ev_v2_calibration_rows()
    _attach_regime_conditioned_ev_v2_target_source(rows, tmp_path)
    rows[0].pop("selected_side")
    features = rows[1]["decision_time_features"]
    features["selected_side_probability"] = 1.2
    features["execution_price"] = -0.1
    features["queue_fill_proxy"] = 1.1
    features["cumulative_market_exposure_before_entry"] = -0.1
    rows[1]["target_provenance"]["source_type"] = "unapproved_write_provider"
    rows[1]["target_provenance"]["outcome_observed_at_ts"] = (
        rows[1]["market_close_ts"] - 1
    )
    rows[2]["decision_time_features"][
        "selected_side_probability_minus_execution_price"
    ] += 0.01
    input_path = tmp_path / "strict-invalid.jsonl"
    _write_jsonl(input_path, rows)

    result = run_execution_layer_v2_regime_conditioned_ev_calibration(
        ExecutionLayerV2RegimeConditionedEVCalibrationConfig(
            run_id="strict-invalid",
            input_path=input_path,
            output_dir=tmp_path / "runs",
            min_fit_rows=16,
            min_validation_rows=8,
            min_fit_markets=4,
            min_validation_markets=2,
        )
    )

    split = result.split_report
    reasons = split["invalid_row_reason_distribution"]
    assert split["invalid_row_count"] == 3
    assert split["schema_validation_row_count"] == len(rows)
    assert split["schema_runtime_validation_disagreement_count"] >= 1
    assert split["schema_runtime_validation_agreement_passed"] is False
    assert reasons["selected_side_invalid"] == 1
    assert reasons["selected_side_probability_outside_unit_interval"] == 1
    assert reasons["execution_price_outside_unit_interval"] == 1
    assert reasons["queue_fill_proxy_outside_unit_interval"] == 1
    assert reasons[
        "negative_feature_not_allowed:cumulative_market_exposure_before_entry"
    ] == 1
    assert reasons[
        "target_provenance_source_not_approved_read_only_settlement"
    ] == 1
    assert reasons["outcome_observed_before_market_close"] == 1
    assert reasons[
        "selected_side_probability_minus_execution_price_mismatch"
    ] >= 1
    assert result.calibration_report["artifact_created"] is False
    assert result.calibration_report["final_artifact_eligibility_reason_codes"]
    assert result.manifest["v8_execution_handoff_allowed"] is False


def test_regime_conditioned_ev_v2_accepts_resolved_historical_outcome_without_timestamp(
    tmp_path,
) -> None:
    rows = _regime_conditioned_ev_v2_calibration_rows()
    _attach_regime_conditioned_ev_v2_target_source(rows, tmp_path)
    for row in rows:
        provenance = row["target_provenance"]
        provenance.pop("outcome_observed_at_ts")
        provenance["outcome_observation_time_source"] = "not_recorded_historical"

    normalized, invalid, excluded = validate_regime_conditioned_ev_v2_calibration_rows(
        rows,
        source_root=tmp_path,
        probability_price_tolerance=1e-9,
    )

    assert len(normalized) == len(rows)
    assert invalid == []
    assert excluded == []
    assert all(
        row["target_provenance"]["resolution_status"] == "resolved"
        for row in normalized
    )
    input_path = tmp_path / "historical-outcome-without-observation-time.jsonl"
    _write_jsonl(input_path, rows)
    result = run_execution_layer_v2_regime_conditioned_ev_calibration(
        ExecutionLayerV2RegimeConditionedEVCalibrationConfig(
            run_id="historical-outcome-without-observation-time",
            input_path=input_path,
            output_dir=tmp_path / "runs",
            min_fit_rows=16,
            min_validation_rows=8,
            min_fit_markets=4,
            min_validation_markets=2,
        )
    )
    assert result.split_report["target_observation_time_contract"] == {
        "exact_settlement_timestamp_required": False,
        "historical_missing_outcome_observation_timestamp_allowed": True,
        "recorded_outcome_observation_timestamp_must_follow_market_close": True,
        "resolved_official_outcome_required": True,
    }
    assert result.manifest["target_observation_time_contract"] == result.split_report[
        "target_observation_time_contract"
    ]


def test_regime_conditioned_ev_v2_statistical_gates_fail_closed(
    tmp_path,
) -> None:
    rows = _regime_conditioned_ev_v2_calibration_rows()
    _attach_regime_conditioned_ev_v2_target_source(rows, tmp_path)
    input_path = tmp_path / "statistical-gates.jsonl"
    _write_jsonl(input_path, rows)

    result = run_execution_layer_v2_regime_conditioned_ev_calibration(
        ExecutionLayerV2RegimeConditionedEVCalibrationConfig(
            run_id="statistical-gates",
            input_path=input_path,
            output_dir=tmp_path / "runs",
            validation_fraction=0.25,
            ridge_alpha=0.01,
            min_fit_rows=16,
            min_validation_rows=8,
            min_fit_markets=4,
            min_validation_markets=2,
            min_relative_mae_improvement=0.95,
            min_relative_mse_improvement=0.95,
            bootstrap_samples=200,
            min_bootstrap_improvement_lower_bound=0.1,
            max_lomo_coefficient_absolute_deviation=0.0,
            min_validation_rows_per_side=5,
            min_validation_rows_per_action_family=5,
            min_validation_rows_per_resolved_outcome=5,
        )
    )

    report = result.calibration_report
    reasons = report["final_artifact_eligibility_reason_codes"]
    assert report["artifact_created"] is False
    assert report["statistical_eligibility_passed"] is False
    assert "row_level_relative_improvement_gate_failed" in reasons
    assert "market_level_relative_improvement_gate_failed" in reasons
    assert "market_bootstrap_confidence_gate_failed" in reasons
    assert "coefficient_stability_gate_failed" in reasons
    assert "validation_side_coverage_gate_failed" in reasons
    assert "validation_action_family_coverage_gate_failed" in reasons
    assert "validation_resolved_outcome_coverage_gate_failed" in reasons
    assert report["market_level_metrics"]["validation_candidate"][
        "market_count"
    ] == 2
    assert report["market_bootstrap_confidence_intervals"][
        "resampling_unit"
    ] == "market_id"
    assert report["coefficient_stability_metrics"]["method"] == (
        "leave_one_market_out_fixed_fit_transforms"
    )
    assert result.manifest["source_model_candidate_eligible"] is False
    assert result.manifest["freeze_ready"] is False
    assert result.manifest["promotion_evidence_eligible"] is False


def test_regime_conditioned_ev_v2_corpus_builder_ingests_and_excludes(
    tmp_path,
) -> None:
    source_root = tmp_path / "sources"
    _write_regime_ev_corpus_source_run(
        source_root / "eligible",
        run_id="historical-paper-run-eligible",
        row_count=8,
    )
    _write_regime_ev_corpus_source_run(
        source_root / "prohibited",
        run_id=LATEST_ONE_HOUR_RECONCILED_RUN_ID,
        row_count=1,
    )
    _write_regime_ev_corpus_source_run(
        source_root / "future",
        run_id="future-unseen-forward-shadow-run",
        row_count=1,
    )

    result = run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
        ExecutionLayerV2RegimeConditionedEVCorpusConfig(
            run_id="corpus-ingestion",
            source_roots=(source_root,),
            output_dir=tmp_path / "runs",
        )
    )
    report = result.quality_report

    assert report["source_manifest_discovered_count"] == 3
    assert report["source_run_included_count"] == 1
    assert report["source_run_excluded_count"] == 2
    assert report["eligible_row_count"] == 8
    assert report["unique_market_count"] == 8
    assert report["invalid_row_count"] == 0
    assert report["coverage"]["by_side"] == {"DOWN": 4, "UP": 4}
    assert report["coverage"]["by_action_family"] == {
        "HOLD_TO_SETTLEMENT": 4,
        "SELL_BEFORE_CLOSE": 4,
    }
    assert report["provenance_coverage"]["violation_count"] == 0
    assert report["target_observation_time_contract"][
        "exact_settlement_timestamp_required"
    ] is False
    assert result.manifest["target_observation_time_contract"] == report[
        "target_observation_time_contract"
    ]
    assert report["incremental_full_rebuild_hash_match"] is True
    assert report["minimum_protocol_smoke_passed"] is False
    assert report["real_frozen_artifact_created"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False
    assert result.manifest["source_model_candidate_eligible"] is False
    assert result.manifest["freeze_ready"] is False
    assert result.manifest["promotion_evidence_eligible"] is False
    corpus_rows_path = result.artifact_paths["corpus_rows"]
    calibration = run_execution_layer_v2_regime_conditioned_ev_calibration(
        ExecutionLayerV2RegimeConditionedEVCalibrationConfig(
            run_id="consume-corpus",
            input_path=corpus_rows_path,
            output_dir=tmp_path / "calibration-runs",
            validation_fraction=0.25,
            ridge_alpha=0.01,
            min_fit_rows=4,
            min_validation_rows=2,
            min_fit_markets=4,
            min_validation_markets=2,
            bootstrap_samples=100,
            min_validation_rows_per_side=1,
            min_validation_rows_per_action_family=1,
            min_validation_rows_per_resolved_outcome=1,
            min_validation_markets_per_category=1,
        )
    )
    assert calibration.split_report["invalid_row_count"] == 0
    assert calibration.split_report["leakage_checks_passed"] is True


def test_historical_outcome_reconciliation_resolves_rows_and_feeds_corpus(
    tmp_path,
) -> None:
    source_manifest = _write_regime_ev_corpus_source_run(
        tmp_path / "source",
        run_id="historical-unresolved-for-clob-reconciliation",
        row_count=3,
        unresolved=True,
        repeat_market=True,
    )
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    original_settlement_path = Path(
        source_payload["artifact_paths"]["settlement_evaluation_rows"]
    )
    original_settlement_sha256 = hashlib.sha256(
        original_settlement_path.read_bytes()
    ).hexdigest()

    def _fetch_market(condition_id: str, timeout_seconds: float) -> dict[str, object]:
        assert timeout_seconds == 2.0
        market_index = int(condition_id.rsplit("-", 1)[-1])
        up_wins = market_index % 2 == 0
        return {
            "closed": True,
            "condition_id": condition_id,
            "tokens": [
                {"outcome": "Up", "winner": up_wins},
                {"outcome": "Down", "winner": not up_wins},
            ],
        }

    reconciliation = run_execution_layer_v2_historical_outcome_reconciliation(
        ExecutionLayerV2HistoricalOutcomeReconciliationConfig(
            run_id="historical-clob-reconciliation",
            source_manifest_paths=(source_manifest,),
            output_dir=tmp_path / "reconciliation-runs",
            request_timeout_seconds=2.0,
            max_workers=2,
        ),
        fetch_market=_fetch_market,
        outcome_observed_at_ts=1_000_000.0,
    )

    assert reconciliation.report["unresolved_fill_count_before"] == 3
    assert reconciliation.report["resolved_fill_count"] == 3
    assert reconciliation.report["unresolved_fill_count_after"] == 0
    assert reconciliation.report["original_source_artifacts_mutated"] is False
    assert hashlib.sha256(original_settlement_path.read_bytes()).hexdigest() == (
        original_settlement_sha256
    )
    corpus = run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
        ExecutionLayerV2RegimeConditionedEVCorpusConfig(
            run_id="corpus-from-clob-reconciliation",
            source_roots=(source_manifest.parent, reconciliation.output_dir),
            output_dir=tmp_path / "corpus-runs",
        )
    )
    assert corpus.quality_report["source_run_included_count"] == 1
    assert corpus.quality_report["source_exclusion_reason_distribution"][
        "source_manifest_superseded_by_complete_reconciliation_bundle"
    ] == 1
    assert corpus.quality_report["eligible_row_count"] == 3
    assert corpus.quality_report["invalid_row_count"] == 0
    rows = _read_jsonl(corpus.artifact_paths["corpus_rows"])
    assert all(row["target_provenance"]["resolution_status"] == "resolved" for row in rows)
    assert all(
        row["target_provenance"]["outcome_observation_time_source"]
        == "provider_response_clock"
        for row in rows
    )
    assert [
        row["decision_time_features"]["cumulative_market_exposure_before_entry"]
        for row in rows
    ] == pytest.approx([0.0, 0.2, 0.4])


def test_regime_conditioned_ev_v2_corpus_follows_nested_trace_manifest_hash_chain(
    tmp_path,
) -> None:
    source_root = tmp_path / "sources"
    valid_manifest = _write_regime_ev_corpus_source_run(
        source_root / "nested-valid",
        run_id="historical-nested-trace-valid",
        row_count=2,
    )
    valid_trace_manifest = _nest_regime_ev_corpus_signal_trace(valid_manifest)
    invalid_manifest = _write_regime_ev_corpus_source_run(
        source_root / "nested-invalid",
        run_id="historical-nested-trace-invalid",
        row_count=1,
    )
    invalid_trace_manifest = _nest_regime_ev_corpus_signal_trace(invalid_manifest)
    invalid_trace_manifest.write_text(
        invalid_trace_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    result = run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
        ExecutionLayerV2RegimeConditionedEVCorpusConfig(
            run_id="nested-trace-chain",
            source_roots=(source_root,),
            output_dir=tmp_path / "runs",
        )
    )
    report = result.quality_report
    rows = _read_jsonl(result.artifact_paths["corpus_rows"])

    assert report["source_manifest_discovered_count"] == 2
    assert report["source_run_included_count"] == 1
    assert report["source_run_excluded_count"] == 1
    assert report["eligible_row_count"] == 2
    valid_source_report = next(
        source_report
        for source_report in report["source_ingestion_reports"]
        if source_report["source_run_id"] == "historical-nested-trace-valid"
    )
    assert valid_source_report["signal_trace_resolution_mode"] == (
        "nested_manifest_chain"
    )
    assert valid_source_report["signal_trace_manifest_chain_verified"] is True
    assert report["source_exclusion_reason_distribution"][
        "source_artifact_hash_mismatch:paper_fresh_loop_manifest"
    ] == 1
    assert all(
        row["source_lineage"]["source_manifest_path"]
        == str(valid_manifest.resolve())
        for row in rows
    )
    assert all(
        row["source_lineage"]["trace_manifest_path"]
        == str(valid_trace_manifest.resolve())
        for row in rows
    )
    assert all(
        row["source_lineage"]["trace_manifest_sha256"]
        == hashlib.sha256(valid_trace_manifest.read_bytes()).hexdigest()
        for row in rows
    )


def test_regime_conditioned_ev_v2_corpus_incremental_matches_rebuild(
    tmp_path,
) -> None:
    source_root = tmp_path / "sources"
    _write_regime_ev_corpus_source_run(
        source_root / "first",
        run_id="historical-paper-run-first",
        row_count=4,
    )
    initial = run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
        ExecutionLayerV2RegimeConditionedEVCorpusConfig(
            run_id="initial",
            source_roots=(source_root,),
            output_dir=tmp_path / "runs",
        )
    )
    _write_regime_ev_corpus_source_run(
        source_root / "second",
        run_id="historical-paper-run-second",
        row_count=4,
        market_offset=10,
    )
    incremental = run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
        ExecutionLayerV2RegimeConditionedEVCorpusConfig(
            run_id="incremental",
            source_roots=(source_root,),
            output_dir=tmp_path / "runs",
            existing_corpus_manifest=initial.artifact_paths["corpus_manifest"],
        )
    )
    rebuild = run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
        ExecutionLayerV2RegimeConditionedEVCorpusConfig(
            run_id="rebuild",
            source_roots=(source_root,),
            output_dir=tmp_path / "runs",
        )
    )

    assert incremental.quality_report["incremental_build"]["appended_row_count"] == 4
    assert incremental.quality_report["incremental_build"][
        "existing_rows_preserved"
    ] is True
    assert incremental.quality_report["incremental_full_rebuild_hash_match"] is True
    assert incremental.manifest["corpus_sha256"] == rebuild.manifest["corpus_sha256"]


def test_regime_conditioned_ev_v2_corpus_fail_closed_diagnostics(
    tmp_path,
) -> None:
    source_root = tmp_path / "sources"
    _write_regime_ev_corpus_source_run(
        source_root / "hash-mismatch",
        run_id="historical-hash-mismatch",
        row_count=1,
        hash_mismatch=True,
    )
    _write_regime_ev_corpus_source_run(
        source_root / "unresolved",
        run_id="historical-unresolved",
        row_count=1,
        unresolved=True,
    )
    _write_regime_ev_corpus_source_run(
        source_root / "causality",
        run_id="historical-causality",
        row_count=1,
        max_input_ts_violation=True,
    )
    _write_regime_ev_corpus_source_run(
        source_root / "duplicate-a",
        run_id="historical-duplicate-source",
        row_count=1,
        settlement_pnl_offset=0.0,
    )
    _write_regime_ev_corpus_source_run(
        source_root / "duplicate-b",
        run_id="historical-duplicate-source",
        row_count=1,
        settlement_pnl_offset=0.25,
    )
    _write_regime_ev_corpus_source_run(
        source_root / "duplicate-c",
        run_id="historical-duplicate-source",
        row_count=1,
        settlement_pnl_offset=0.0,
    )

    result = run_execution_layer_v2_regime_conditioned_ev_corpus_builder(
        ExecutionLayerV2RegimeConditionedEVCorpusConfig(
            run_id="fail-closed",
            source_roots=(source_root,),
            output_dir=tmp_path / "runs",
        )
    )
    report = result.quality_report

    assert report["source_exclusion_reason_distribution"][
        "source_artifact_hash_mismatch:signal_trace"
    ] == 1
    assert report["row_exclusion_reason_distribution"][
        "ambiguous_or_unresolved_settlement"
    ] == 1
    assert report["row_exclusion_reason_distribution"][
        "feature_max_input_ts_after_decision_ts"
    ] == 1
    assert report["deduplication"]["conflicting_identity_count"] == 1
    assert report["deduplication"]["exact_duplicate_count"] == 1
    assert report["deduplication"]["supplemental_duplicate_fill_count"] >= 1
    assert "conflicting_duplicate_rows_present" in report[
        "readiness_blocking_reason_codes"
    ]
    assert report["corpus_ready"] is False
    assert report["real_frozen_artifact_created"] is False
    assert result.manifest["v8_execution_handoff_allowed"] is False


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
    assert row["market_start_ts"] == 1_000_000
    assert row["market_end_ts"] == 1_300_000
    assert row["horizon_ms"] == 300_000
    assert row["market_schedule_source_type"] == (
        "normalized_public_market_metadata"
    )
    assert row["market_schedule_provenance"]["provenance_valid"] is True
    assert row["paper_only"] is True
    assert row["capital_at_risk"] is False


def test_trace_market_schedule_uses_canonical_slug_before_zero_ttl_fallback() -> None:
    start_ts = 1_700_001_000_000
    decision_ts = start_ts + 100_000
    schedule = _trace_market_schedule(
        provider_row={
            "slug": "btc-updown-5m-1700001000",
            "raw_market_sha256": "a" * 64,
        },
        decision_ts=decision_ts,
        micro={"time_to_close_seconds": 0.0},
    )

    assert schedule["market_start_ts"] == start_ts
    assert schedule["market_end_ts"] == start_ts + 300_000
    assert schedule["source_type"] == "canonical_market_slug_schedule"
    assert schedule["warning_reason_codes"] == [
        "market_schedule_backfilled_from_canonical_slug"
    ]
    assert schedule["provenance"]["provenance_valid"] is True
    assert _trace_time_to_close_seconds(
        decision_ts=decision_ts,
        market_end_ts=schedule["market_end_ts"],
        micro={"time_to_close_seconds": 0.0},
    ) == pytest.approx(200.0)


def test_trace_market_schedule_rejects_drifting_provider_times_in_favor_of_slug() -> None:
    start_ts = 1_700_001_000_000
    schedule = _trace_market_schedule(
        provider_row={
            "slug": "btc-updown-5m-1700001000",
            "market_start_ts": start_ts + 25_000,
            "market_end_ts": start_ts + 325_000,
        },
        decision_ts=start_ts + 50_000,
        micro={"time_to_close_seconds": 0.0},
    )

    assert schedule["market_start_ts"] == start_ts
    assert schedule["market_end_ts"] == start_ts + 300_000
    assert schedule["source_type"] == "canonical_market_slug_schedule"
    assert schedule["warning_reason_codes"] == [
        "provider_market_schedule_mismatch_canonical_slug"
    ]


def test_pre_promotion_goal_initialization_freezes_gates_and_exclusions(
    tmp_path,
) -> None:
    evidence_root = tmp_path / "evidence"
    explicit_run = (
        evidence_root
        / "execution-layer-v2-regime-entry-edge-replay-20260710T123338Z"
    )
    shadow_run = evidence_root / "test-future-shadow-inspected"
    explicit_run.mkdir(parents=True)
    shadow_run.mkdir(parents=True)
    (explicit_run / "manifest.json").write_text(
        '{"run_id":"excluded"}\n', encoding="utf-8"
    )
    (shadow_run / "rows.jsonl").write_text('{"market_id":"m1"}\n', encoding="utf-8")

    result = initialize_pre_promotion_readiness_goal(
        ExecutionLayerV2PrePromotionGoalConfig(
            run_id="pre-promotion-goal-test",
            output_dir=tmp_path / "runs",
            evidence_root=evidence_root,
            created_at="2026-07-11T12:00:00Z",
            starting_commit="1" * 40,
        )
    )
    configuration = json.loads(
        result.goal_configuration_path.read_text(encoding="utf-8")
    )
    exclusions = json.loads(
        result.excluded_evidence_manifest_path.read_text(encoding="utf-8")
    )
    state = json.loads(result.goal_state_path.read_text(encoding="utf-8"))

    assert result.goal_configuration_sha256 == hashlib.sha256(
        result.goal_configuration_path.read_bytes()
    ).hexdigest()
    assert result.goal_configuration_sha256_path.read_text(
        encoding="utf-8"
    ).strip() == result.goal_configuration_sha256
    assert configuration["min_fit_rows"] == 100
    assert configuration["min_validation_rows"] == 30
    assert configuration["min_relative_mae_improvement"] == pytest.approx(0.05)
    assert configuration["min_relative_mse_improvement"] == pytest.approx(0.05)
    assert configuration["bootstrap_samples"] == 1_000
    assert configuration["required_future_shadow_window_count"] == 2
    assert configuration["promotion_evidence_stage_started"] is False
    assert configuration["promotion_evidence_eligible"] is False
    assert configuration["live_evidence_allowed"] is False
    assert configuration["v8_execution_handoff_allowed"] is False
    assert exclusions["excluded_run_count"] == 2
    assert {row["run_id"] for row in exclusions["excluded_runs"]} == {
        explicit_run.name,
        shadow_run.name,
    }
    assert all(row["run_tree_sha256"] for row in exclusions["excluded_runs"])
    assert state["goal_status"] == "IN_PROGRESS"
    assert state["next_phase"] == "phase_1_collect_strict_causal_historical_corpus"
    assert state["source_model_candidate_eligible"] is False
    assert state["freeze_ready"] is False
    assert state["#134_resume_allowed"] is False
    assert state["#146_start_allowed"] is False


def test_pre_promotion_goal_finalizer_seals_blocked_bundle_without_artifact(
    tmp_path,
) -> None:
    result = initialize_pre_promotion_readiness_goal(
        ExecutionLayerV2PrePromotionGoalConfig(
            run_id="pre-promotion-finalizer-test",
            output_dir=tmp_path / "runs",
            evidence_root=tmp_path / "evidence",
            created_at="2026-07-11T12:00:00Z",
            starting_commit="1" * 40,
        )
    )
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "one_hour_remap_paper_goal_report.json").write_text(
        json.dumps(
            {"complete_round_count": 2, "paper_fill_count": 4, "paper_only": True}
        ),
        encoding="utf-8",
    )
    reconciliation = tmp_path / "reconciliation"
    reconciliation.mkdir()
    (reconciliation / "clob_settlement_reconciliation_report.json").write_text(
        json.dumps(
            {
                "unresolved_fill_count_after": 0,
                "original_source_artifacts_mutated": False,
            }
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "execution_layer_v2_regime_conditioned_ev_v2_corpus_manifest.json").write_text(
        json.dumps({"corpus_sha256": "2" * 64}), encoding="utf-8"
    )
    (corpus / "execution_layer_v2_regime_conditioned_ev_v2_corpus_quality_report.json").write_text(
        json.dumps(
            {
                "eligible_row_count": 12,
                "excluded_row_count": 1,
                "unique_market_count": 4,
                "incremental_full_rebuild_hash_match": True,
                "readiness_blocking_reason_codes": ["minimum_protocol_smoke_not_met"],
            }
        ),
        encoding="utf-8",
    )

    finalized = finalize_pre_promotion_readiness_goal(
        ExecutionLayerV2PrePromotionFinalizationConfig(
            goal_dir=result.goal_dir,
            historical_collection_dirs=(collection,),
            outcome_reconciliation_dirs=(reconciliation,),
            calibration_corpus_dir=corpus,
            stop_reason_codes=("configured_data_window_budget_reached",),
            resumable_next_command="collect-one-more-window",
        )
    )
    report = json.loads(finalized.readiness_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(finalized.readiness_manifest_path.read_text(encoding="utf-8"))

    assert finalized.final_state == "PRE_PROMOTION_BLOCKED"
    assert finalized.pre_promotion_readiness_complete is False
    assert report["promotion_evidence_stage_started"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["live_evidence_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert "minimum_goal_calibration_row_support_not_met" in report[
        "blocking_reason_codes"
    ]
    assert "minimum_goal_calibration_market_support_not_met" in report[
        "blocking_reason_codes"
    ]
    assert not (result.goal_dir / "frozen_diagnostic_artifact.json").exists()
    assert manifest["manifest_self_hash_embedded"] is False
    assert hashlib.sha256(finalized.readiness_manifest_path.read_bytes()).hexdigest() == (
        finalized.readiness_manifest_sha256_path.read_text(encoding="utf-8").strip()
    )
    assert all(
        hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
        for row in manifest["artifacts"]
    )


def test_pre_promotion_remediation_initialization_is_immutable_and_uses_150_rows(
    tmp_path,
) -> None:
    prior_bundle = tmp_path / "prior-bundle"
    prior_bundle.mkdir()
    (prior_bundle / "pre_promotion_readiness_manifest.json").write_text(
        json.dumps({"final_state": "PRE_PROMOTION_BLOCKED"}), encoding="utf-8"
    )
    prior_rows = tmp_path / "prior-rows.jsonl"
    prior_rows.write_text(
        json.dumps(
            {
                "source_run_id": "prior-development-run",
                "market_id": "prior-market",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prior_split = tmp_path / "prior-split.json"
    prior_split.write_text("{}\n", encoding="utf-8")
    prior_calibration = tmp_path / "prior-calibration.json"
    prior_calibration.write_text("{}\n", encoding="utf-8")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    result = initialize_pre_promotion_remediation_goal(
        ExecutionLayerV2PrePromotionRemediationConfig(
            run_id="remediation-initialization-test",
            output_dir=tmp_path / "runs",
            repository_root=Path.cwd(),
            created_at="2026-07-12T03:00:00Z",
            starting_branch="codex/v8-pre-promotion-readiness-goal",
            starting_commit=head,
            prior_blocked_bundle_dir=prior_bundle,
            prior_corpus_rows_path=prior_rows,
            prior_split_report_path=prior_split,
            prior_calibration_report_path=prior_calibration,
        )
    )
    config = json.loads(result.configuration_path.read_text(encoding="utf-8"))
    exclusions = json.loads(result.exclusions_path.read_text(encoding="utf-8"))
    state = json.loads(result.state_path.read_text(encoding="utf-8"))

    assert config["minimum_total_calibration_rows"] == 150
    assert config["maximum_candidate_count"] == 6
    assert config["required_future_shadow_window_count"] == 2
    assert exclusions["development_evidence_only"] is True
    assert exclusions["unseen_validation_eligible"] is False
    assert state["starting_commit_verified"] is True
    assert state["promotion_evidence_stage_started"] is False
    assert hashlib.sha256(result.configuration_path.read_bytes()).hexdigest() == (
        result.configuration_sha256_path.read_text(encoding="utf-8").strip()
    )
    with pytest.raises(FileExistsError):
        initialize_pre_promotion_remediation_goal(
            ExecutionLayerV2PrePromotionRemediationConfig(
                run_id="remediation-initialization-test",
                output_dir=tmp_path / "runs",
                repository_root=Path.cwd(),
                created_at="2026-07-12T03:00:00Z",
                starting_branch="codex/v8-pre-promotion-readiness-goal",
                starting_commit=head,
                prior_blocked_bundle_dir=prior_bundle,
                prior_corpus_rows_path=prior_rows,
                prior_split_report_path=prior_split,
                prior_calibration_report_path=prior_calibration,
            )
        )


def test_pre_promotion_candidate_search_is_bounded_and_decision_time_only() -> None:
    candidates = _candidate_specifications()

    assert len(candidates) == 6
    assert len({row["candidate_name"] for row in candidates}) == 6
    assert all(
        not any(
            forbidden in feature
            for forbidden in ("settlement", "outcome", "pnl", "future", "oracle")
        )
        for row in candidates
        for feature in row["features"]
    )


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
    persisted_goal_manifest = json.loads(
        result.artifact_paths["one_hour_remap_paper_goal_manifest"].read_text(
            encoding="utf-8"
        )
    )
    goal_manifest_descriptor = json.loads(
        result.artifact_paths[
            "one_hour_remap_paper_goal_manifest_descriptor"
        ].read_text(encoding="utf-8")
    )
    assert "one_hour_remap_paper_goal_manifest" not in persisted_goal_manifest[
        "artifact_hashes"
    ]
    assert goal_manifest_descriptor["final_manifest_sha256"] == (
        result.artifact_hashes["one_hour_remap_paper_goal_manifest"]
    )
    assert goal_manifest_descriptor["self_hash_embedded_in_manifest"] is False
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


def test_execution_layer_v2_round_artifacts_preserve_raw_book_and_trace(
    tmp_path,
) -> None:
    market_id = "round-raw-evidence"
    market_start_ts = 3_000_000
    decision_ts = 3_060_000
    manifest = _write_per_round_artifacts(
        goal_dir=tmp_path,
        intents=[{"market_id": market_id, "paper_only": True}],
        fills=[{"market_id": market_id, "paper_only": True}],
        ledger_rows=[{"market_id": market_id, "paper_only": True}],
        settlement_rows=[],
        trace_rows=[
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "market_start_ts": market_start_ts,
                "spread_bps": 200.0,
                "book_staleness_ms": 0.0,
                "queue_fill_proxy": 1.0,
                "paper_only": True,
            }
        ],
        raw_market_rows=[
            {
                "market_id": market_id,
                "condition_id": market_id,
                "market_start_ts": market_start_ts,
                "market_end_ts": 3_300_000,
                "paper_only": True,
            }
        ],
        raw_orderbook_rows=[
            {
                "market_id": market_id,
                "outcome": "UP",
                "ts": decision_ts,
                "bid_price": 0.60,
                "ask_price": 0.62,
            },
            {
                "market_id": market_id,
                "outcome": "DOWN",
                "ts": decision_ts,
                "bid_price": 0.38,
                "ask_price": 0.40,
            },
        ],
        raw_trade_rows=[
            {"market_id": market_id, "ts": 3_050_000, "price": 0.61}
        ],
        raw_btc_candle_rows=[
            {
                "ts": market_start_ts,
                "available_at_ts": 3_050_000,
                "close_price": 65_000.0,
            }
        ],
    )

    round_row = manifest["round_artifact_rows"][0]
    assert round_row["signal_trace_row_count"] == 1
    assert round_row["raw_market_row_count"] == 1
    assert round_row["raw_orderbook_row_count"] == 2
    assert round_row["raw_trade_row_count"] == 1
    assert round_row["raw_btc_feature_candle_row_count"] == 1
    assert set(round_row["artifact_hashes"]) >= {
        "signal_trace",
        "raw_polymarket_markets",
        "raw_polymarket_orderbooks",
        "raw_polymarket_trades",
        "raw_btc_feature_candles",
    }
    assert len(
        _read_jsonl(Path(round_row["artifact_paths"]["raw_polymarket_orderbooks"]))
    ) == 2
    assert manifest["per_round_raw_orderbook_artifact_count"] == 1
    assert manifest["per_round_signal_trace_artifact_count"] == 1
    persisted_manifest = json.loads(
        Path(manifest["manifest_path"]).read_text(encoding="utf-8")
    )
    descriptor = json.loads(
        Path(manifest["manifest_descriptor_path"]).read_text(encoding="utf-8")
    )
    assert "manifest_sha256" not in persisted_manifest
    assert "final_manifest_sha256" not in persisted_manifest
    assert descriptor["final_manifest_sha256"] == manifest["manifest_sha256"]
    assert descriptor["self_hash_embedded_in_manifest"] is False


def test_execution_layer_v2_round_orderbook_coverage_requires_rows(tmp_path) -> None:
    manifest = _write_per_round_artifacts(
        goal_dir=tmp_path,
        intents=[],
        fills=[],
        ledger_rows=[],
        settlement_rows=[],
        trace_rows=[
            {
                "market_id": "empty-book-market",
                "decision_ts": 3_060_000,
                "market_start_ts": 3_000_000,
            }
        ],
        raw_market_rows=[
            {
                "market_id": "empty-book-market",
                "market_start_ts": 3_000_000,
            }
        ],
        raw_orderbook_rows=[],
        raw_trade_rows=[],
        raw_btc_candle_rows=[],
    )

    round_row = manifest["round_artifact_rows"][0]
    assert Path(round_row["artifact_paths"]["raw_polymarket_orderbooks"]).exists()
    assert round_row["raw_orderbook_row_count"] == 0
    assert manifest["per_round_raw_orderbook_artifact_count"] == 0
    assert manifest["per_round_raw_orderbook_covered_market_count"] == 0


def test_execution_layer_v2_raw_evidence_audits_no_bet_market_causally(
    tmp_path,
) -> None:
    market_id = "complete-no-bet-market"
    decision_ts = 3_060_000
    trace_rows = [
        {
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_start_ts": 3_000_000,
            "reference_price_to_beat_distance_at_decision": 25.0,
            "reference_price_to_beat_distance_provenance": {
                "max_input_ts": 3_050_000,
                "provenance_valid": True,
            },
            "decision_time_regime_feature_max_input_ts": 3_050_000,
        }
    ]
    market_rows = [
        {
            "market_id": market_id,
            "market_start_ts": 3_000_000,
            "market_end_ts": 3_300_000,
        }
    ]
    book_rows = [
        {
            "market_id": market_id,
            "outcome": outcome,
            "available_at_ts": 3_050_000,
            "ts": 3_050_000,
        }
        for outcome in ("UP", "DOWN")
    ]
    trade_rows = [
        {"market_id": market_id, "available_at_ts": 3_040_000, "ts": 3_040_000}
    ]
    candle_rows = [
        {
            "available_at_ts": 3_050_000,
            "ts": 3_000_000,
            "close_price": 65_000.0,
        }
    ]
    round_artifacts = _write_per_round_artifacts(
        goal_dir=tmp_path,
        intents=[],
        fills=[],
        ledger_rows=[],
        settlement_rows=[],
        trace_rows=trace_rows,
        raw_market_rows=market_rows,
        raw_orderbook_rows=book_rows,
        raw_trade_rows=trade_rows,
        raw_btc_candle_rows=candle_rows,
    )
    aggregate_paths = {
        "raw_polymarket_markets": Path(
            round_artifacts["round_artifact_rows"][0]["artifact_paths"][
                "raw_polymarket_markets"
            ]
        ),
        "raw_polymarket_orderbooks": Path(
            round_artifacts["round_artifact_rows"][0]["artifact_paths"][
                "raw_polymarket_orderbooks"
            ]
        ),
    }

    report = _raw_evidence_completeness_report(
        run_id="raw-evidence-complete",
        trace_rows=trace_rows,
        intents=[],
        raw_market_rows=market_rows,
        raw_orderbook_rows=book_rows,
        raw_trade_rows=trade_rows,
        raw_btc_candle_rows=candle_rows,
        round_artifacts=round_artifacts,
        aggregate_artifact_paths=aggregate_paths,
    )

    assert report["observed_market_count"] == 1
    assert report["bet_market_count"] == 0
    assert report["no_bet_market_count"] == 1
    assert report["markets_with_raw_orderbook_rows"] == 1
    assert report["markets_with_complete_btc_reference_provenance"] == 1
    assert report["evidence_complete_market_count"] == 1
    assert report["markets_missing_required_evidence_count"] == 0
    assert report["causal_source_timestamp_violation_count"] == 0


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
    assert result.manifest["settlement_unique_condition_count_before_poll"] == 1
    assert result.manifest["settlement_resolved_condition_count"] == 1
    assert result.manifest["settlement_terminal_unresolved_condition_count"] == 0
    assert result.manifest["settlement_total_condition_request_count"] == 1
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


def test_one_hour_settlement_accumulates_concurrent_clob_results_by_condition(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def _fake_clob_resolution_rows(*, market_rows, timeout_seconds):
        assert timeout_seconds > 0.0
        market_ids = tuple(sorted(str(row["market_id"]) for row in market_rows))
        calls.append(market_ids)
        resolved_market_id = "market-a" if len(calls) == 1 else "market-b"
        rows = [
            {
                "market_id": resolved_market_id,
                "condition_id": resolved_market_id,
                "resolution_status": "normal",
                "resolved_outcome": "UP" if resolved_market_id == "market-a" else "DOWN",
                "payout_up": 1.0 if resolved_market_id == "market-a" else 0.0,
                "payout_down": 0.0 if resolved_market_id == "market-a" else 1.0,
                "resolution_source_type": "polymarket_clob_read_only_settlement",
                "paper_only": True,
                "capital_at_risk": False,
            }
        ]
        failures = []
        if len(calls) == 1:
            failures.append(
                {
                    "market_id": "market-b",
                    "condition_id": "market-b",
                    "reason_code": "settlement_resolution_market_not_closed",
                }
            )
        return rows, failures

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal."
        "_clob_resolution_rows_for_markets",
        _fake_clob_resolution_rows,
    )
    provider = PolymarketPublicHTTPRealCorpusProvider(fetch_json=lambda _url: {})
    fills = [
        {
            "paper_fresh_order_intent_id": f"intent-{suffix}",
            "market_id": f"market-{suffix}",
            "execution_guarded_side": "UP" if suffix == "a" else "DOWN",
            "paper_fill_price": 0.60,
            "filled_size": 0.20,
            "total_execution_cost": 0.0,
            "paper_only": True,
            "capital_at_risk": False,
        }
        for suffix in ("a", "b")
    ]
    trace_rows = [
        {
            "market_id": f"market-{suffix}",
            "condition_id": f"market-{suffix}",
            "slug": f"btc-updown-5m-{suffix}",
            "market_family": "btc_updown_5m",
            "market_start_ts": 1_000_000,
            "market_end_ts": 1_300_000,
            "up_token_id": f"up-{suffix}",
            "down_token_id": f"down-{suffix}",
            "reference_price_source": "polymarket_official_btc_usd_reference",
        }
        for suffix in ("a", "b")
    ]

    report = _settlement_resolution_report(
        config=ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="concurrent-clob-settlement",
            output_dir=tmp_path / "runs",
            public_provider=provider,
            settlement_poll_max_wait_seconds=0.05,
            settlement_poll_interval_seconds=0.001,
        ),
        fills=fills,
        trace_rows=trace_rows,
        settlement_evaluation_rows=[],
    )

    assert calls == [("market-a", "market-b"), ("market-b",)]
    assert report["settlement_resolution_query_mode"] == "concurrent_clob_condition_id"
    assert report["resolved_market_count"] == 2
    assert report["unresolved_market_count"] == 0
    assert report["unique_condition_count_before_poll"] == 2
    assert report["resolved_condition_count"] == 2
    assert report["terminal_unresolved_condition_count"] == 0
    assert report["resolved_fill_count"] == 2
    assert report["unresolved_fill_count_after_poll"] == 0
    assert report["condition_request_count_by_attempt"] == [
        {
            "attempt": 1,
            "condition_request_count": 2,
            "condition_ids": ["market-a", "market-b"],
        },
        {
            "attempt": 2,
            "condition_request_count": 1,
            "condition_ids": ["market-b"],
        },
    ]
    assert report["total_condition_request_count"] == 3
    assert report["attempt_failure_count"] == 1
    assert report["eventually_resolved_after_failure_count"] == 1
    assert report["terminal_failure_reason_distribution"] == {}
    history_by_condition = {
        row["condition_key"]: row
        for row in report["condition_resolution_histories"]
    }
    assert history_by_condition["market-b"]["resolved"] is True
    assert history_by_condition["market-b"][
        "historical_attempt_reason_codes"
    ] == ["settlement_resolution_market_not_closed"]
    assert history_by_condition["market-b"]["last_attempt_reason_codes"] == []
    assert history_by_condition["market-b"]["terminal_reason_codes"] == []
    assert report["historical_attempt_reason_codes"] == [
        "settlement_resolution_market_not_closed"
    ]
    assert report["terminal_reason_codes"] == []
    assert report["settlement_resolution_failure_reason_distribution"] == {
        "settlement_resolution_market_not_closed": 1
    }
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["v8_execution_handoff_allowed"] is False


def test_one_hour_settlement_queries_one_condition_for_multiple_fills(
    tmp_path,
    monkeypatch,
) -> None:
    requested_condition_ids: list[tuple[str, ...]] = []

    def _resolve_once(*, market_rows, timeout_seconds):
        assert timeout_seconds > 0.0
        requested_condition_ids.append(
            tuple(str(row["condition_id"]) for row in market_rows)
        )
        return [
            {
                "market_id": "market-shared",
                "condition_id": "condition-shared",
                "resolution_status": "normal",
                "resolved_outcome": "UP",
                "payout_up": 1.0,
                "payout_down": 0.0,
                "resolution_source_type": "polymarket_clob_read_only_settlement",
                "paper_only": True,
                "capital_at_risk": False,
            }
        ], []

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal."
        "_clob_resolution_rows_for_markets",
        _resolve_once,
    )
    fills = [
        {
            "paper_fresh_order_intent_id": f"intent-{index}",
            "market_id": "market-shared",
            "execution_guarded_side": "UP",
            "paper_fill_price": 0.60,
            "filled_size": 0.20,
            "total_execution_cost": 0.0,
        }
        for index in range(3)
    ]
    report = _settlement_resolution_report(
        config=ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-condition-three-fills",
            output_dir=tmp_path / "runs",
            public_provider=PolymarketPublicHTTPRealCorpusProvider(
                fetch_json=lambda _url: {}
            ),
            settlement_poll_max_wait_seconds=0.0,
            settlement_poll_interval_seconds=0.001,
        ),
        fills=fills,
        trace_rows=[
            {
                "market_id": "market-shared",
                "condition_id": "condition-shared",
                "slug": "btc-updown-5m-shared",
                "market_family": "btc_updown_5m",
                "market_start_ts": 1_000_000,
                "market_end_ts": 1_300_000,
                "up_token_id": "up-shared",
                "down_token_id": "down-shared",
            }
        ]
        * 2,
        settlement_evaluation_rows=[],
    )

    assert requested_condition_ids == [("condition-shared",)]
    assert report["fill_derived_settlement_market_metadata_count"] == 3
    assert report["unique_condition_count_before_poll"] == 1
    assert report["duplicate_condition_requests_prevented_count"] == 2
    assert report["relevant_trace_row_count"] == 2
    assert report["relevant_market_count"] == 1
    assert report["within_market_duplicate_trace_row_count"] == 1
    assert report["within_market_conflict_count"] == 0
    assert report["cross_market_condition_conflict_count"] == 0
    assert report["queryable_unique_condition_count"] == 1
    assert report["metadata_conflict_count"] == 0
    assert report["total_condition_request_count"] == 1
    assert report["resolved_condition_count"] == 1
    assert report["terminal_unresolved_condition_count"] == 0
    assert report["resolved_fill_count"] == 3
    assert report["unresolved_fill_count_after_poll"] == 0
    assert {
        row["paper_fresh_order_intent_id"]
        for row in report["settlement_evaluation_rows"]
    } == {"intent-0", "intent-1", "intent-2"}


def _within_market_settlement_conflict_report(
    *,
    tmp_path,
    monkeypatch,
    second_trace_overrides,
):
    def _unexpected_request(**_kwargs):
        raise AssertionError("within-market conflict must not be requested")

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal."
        "_clob_resolution_rows_for_markets",
        _unexpected_request,
    )
    trace = {
        "market_id": "market-within-conflict",
        "condition_id": "condition-within-conflict",
        "slug": "btc-updown-5m-within-conflict",
        "market_family": "btc_updown_5m",
        "market_start_ts": 1_000_000,
        "market_end_ts": 1_300_000,
        "up_token_id": "up-token-a",
        "down_token_id": "down-token",
    }
    return _settlement_resolution_report(
        config=ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="within-market-metadata-conflict",
            output_dir=tmp_path / "runs",
            public_provider=PolymarketPublicHTTPRealCorpusProvider(
                fetch_json=lambda _url: {}
            ),
            settlement_poll_max_wait_seconds=0.0,
            settlement_poll_interval_seconds=0.001,
        ),
        fills=[
            {
                "paper_fresh_order_intent_id": f"intent-within-{index}",
                "market_id": "market-within-conflict",
                "execution_guarded_side": "UP",
                "paper_fill_price": 0.60,
                "filled_size": 0.20,
                "total_execution_cost": 0.0,
            }
            for index in range(2)
        ],
        trace_rows=[trace, {**trace, **second_trace_overrides}],
        settlement_evaluation_rows=[],
    )


def test_one_hour_settlement_same_market_token_conflict_fails_closed(
    tmp_path,
    monkeypatch,
) -> None:
    report = _within_market_settlement_conflict_report(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        second_trace_overrides={"up_token_id": "up-token-b"},
    )

    assert report["total_condition_request_count"] == 0
    assert report["resolved_fill_count"] == 0
    assert report["unresolved_fill_count_after_poll"] == 2
    assert report["within_market_conflict_count"] == 1
    assert report["cross_market_condition_conflict_count"] == 0
    conflict = report["metadata_conflicts"][0]
    assert conflict["conflict_scope"] == "within_market"
    assert conflict["differing_fields"] == ["up_token_id"]
    assert conflict["observed_values"]["up_token_id"] == [
        "up-token-a",
        "up-token-b",
    ]
    assert conflict["source_trace_row_indices"] == [0, 1]
    assert len(conflict["source_trace_row_hashes"]) == 2
    assert conflict["reason_code"] == "settlement_metadata_within_market_conflict"
    terminal = report["terminal_unresolved_conditions"][0]
    assert terminal["historical_attempt_reason_codes"] == []
    assert terminal["last_attempt_reason_codes"] == []
    assert terminal["terminal_reason_codes"] == [
        "settlement_metadata_within_market_conflict"
    ]


def test_one_hour_settlement_same_market_timestamp_conflict_fails_closed(
    tmp_path,
    monkeypatch,
) -> None:
    report = _within_market_settlement_conflict_report(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        second_trace_overrides={"market_end_ts": 1_360_000},
    )

    assert report["settlement_poll_attempt_count"] == 0
    assert report["total_condition_request_count"] == 0
    assert report["within_market_conflict_count"] == 1
    conflict = report["metadata_conflicts"][0]
    assert conflict["conflict_scope"] == "within_market"
    assert conflict["differing_fields"] == ["market_end_ts"]
    assert conflict["observed_values"]["market_end_ts"] == [
        1_300_000,
        1_360_000,
    ]
    assert report["terminal_reason_codes"] == [
        "settlement_metadata_within_market_conflict"
    ]


def test_one_hour_settlement_cross_market_condition_conflict_blocks_request(
    tmp_path,
    monkeypatch,
) -> None:
    def _unexpected_request(**_kwargs):
        raise AssertionError("conflicting condition must not be requested")

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal."
        "_clob_resolution_rows_for_markets",
        _unexpected_request,
    )
    trace_base = {
        "condition_id": "condition-conflict",
        "slug": "btc-updown-5m-conflict",
        "market_family": "btc_updown_5m",
        "market_start_ts": 1_000_000,
        "market_end_ts": 1_300_000,
        "down_token_id": "down-token",
    }
    report = _settlement_resolution_report(
        config=ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="conflicting-condition-metadata",
            output_dir=tmp_path / "runs",
            public_provider=PolymarketPublicHTTPRealCorpusProvider(
                fetch_json=lambda _url: {}
            ),
            settlement_poll_max_wait_seconds=0.0,
            settlement_poll_interval_seconds=0.001,
        ),
        fills=[
            {
                "paper_fresh_order_intent_id": f"intent-conflict-{suffix}",
                "market_id": f"market-conflict-{suffix}",
                "execution_guarded_side": "UP",
                "paper_fill_price": 0.60,
                "filled_size": 0.20,
                "total_execution_cost": 0.0,
            }
            for suffix in ("a", "b")
        ],
        trace_rows=[
            {
                **trace_base,
                "market_id": "market-conflict-a",
                "up_token_id": "up-token-a",
            },
            {
                **trace_base,
                "market_id": "market-conflict-b",
                "up_token_id": "up-token-b",
            },
        ],
        settlement_evaluation_rows=[],
    )

    assert report["settlement_poll_attempt_count"] == 0
    assert report["total_condition_request_count"] == 0
    assert report["settlement_metadata_conflict_count"] == 1
    assert report["settlement_metadata_conflicts"][0]["differing_fields"] == [
        "up_token_id",
    ]
    assert report["settlement_metadata_conflicts"][0]["conflict_scope"] == (
        "cross_market_condition"
    )
    assert report["settlement_metadata_conflicts"][0]["reason_code"] == (
        "settlement_metadata_cross_market_condition_conflict"
    )
    assert report["within_market_conflict_count"] == 0
    assert report["cross_market_condition_conflict_count"] == 1
    assert report["resolved_condition_count"] == 0
    assert report["terminal_unresolved_condition_count"] == 1
    assert report["terminal_failure_reason_distribution"] == {
        "settlement_metadata_cross_market_condition_conflict": 1
    }
    assert report["unresolved_fill_count_after_poll"] == 2


def test_one_hour_settlement_preserves_resolved_condition_at_terminal_deadline(
    tmp_path,
    monkeypatch,
) -> None:
    requests: list[tuple[str, ...]] = []

    def _partially_resolve(*, market_rows, timeout_seconds):
        assert timeout_seconds > 0.0
        condition_ids = tuple(str(row["condition_id"]) for row in market_rows)
        requests.append(condition_ids)
        rows = []
        if "condition-a" in condition_ids:
            rows.append(
                {
                    "market_id": "market-a",
                    "condition_id": "condition-a",
                    "resolution_status": "normal",
                    "resolved_outcome": "UP",
                    "payout_up": 1.0,
                    "payout_down": 0.0,
                }
            )
        failures = [
            {
                "market_id": "market-b",
                "condition_id": "condition-b",
                "reason_code": "settlement_resolution_market_not_closed",
            }
        ]
        return rows, failures

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal."
        "_clob_resolution_rows_for_markets",
        _partially_resolve,
    )
    fills = [
        {
            "paper_fresh_order_intent_id": f"intent-{suffix}",
            "market_id": f"market-{suffix}",
            "execution_guarded_side": "UP",
            "paper_fill_price": 0.60,
            "filled_size": 0.20,
            "total_execution_cost": 0.0,
        }
        for suffix in ("a", "b")
    ]
    trace_rows = [
        {
            "market_id": f"market-{suffix}",
            "condition_id": f"condition-{suffix}",
            "slug": f"btc-updown-5m-{suffix}",
            "market_family": "btc_updown_5m",
            "market_start_ts": 1_000_000,
            "market_end_ts": 1_300_000,
            "up_token_id": f"up-{suffix}",
            "down_token_id": f"down-{suffix}",
        }
        for suffix in ("a", "b")
    ]
    report = _settlement_resolution_report(
        config=ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="terminal-unresolved-condition",
            output_dir=tmp_path / "runs",
            public_provider=PolymarketPublicHTTPRealCorpusProvider(
                fetch_json=lambda _url: {}
            ),
            settlement_poll_max_wait_seconds=0.003,
            settlement_poll_interval_seconds=0.001,
        ),
        fills=fills,
        trace_rows=trace_rows,
        settlement_evaluation_rows=[],
    )

    assert requests[0] == ("condition-a", "condition-b")
    assert all(request == ("condition-b",) for request in requests[1:])
    assert report["resolved_condition_count"] == 1
    assert report["terminal_unresolved_condition_count"] == 1
    assert report["resolved_fill_count"] == 1
    assert report["unresolved_fill_count_after_poll"] == 1
    terminal = report["terminal_unresolved_conditions"][0]
    assert terminal["condition_key"] == "condition-b"
    assert terminal["condition_id"] == "condition-b"
    assert terminal["market_id"] == "market-b"
    assert terminal["resolved"] is False
    assert terminal["historical_attempt_reason_codes"] == [
        "settlement_resolution_market_not_closed"
    ]
    assert terminal["last_attempt_reason_codes"] == [
        "settlement_resolution_market_not_closed"
    ]
    assert terminal["terminal_reason_codes"] == [
        "settlement_resolution_market_not_closed"
    ]
    assert report["attempt_failure_count"] == len(requests)
    assert [row["attempt"] for row in report["settlement_resolution_failures"]] == (
        list(range(1, len(requests) + 1))
    )
    assert report["terminal_failure_reason_distribution"] == {
        "settlement_resolution_market_not_closed": 1
    }


def test_clob_settlement_keeps_not_closed_market_fail_closed(monkeypatch) -> None:
    def _fake_market(*, market_row, timeout_seconds):
        assert market_row["market_id"] == "market-not-closed"
        assert timeout_seconds == 1.0
        return None, "settlement_resolution_market_not_closed"

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_one_hour_goal."
        "_clob_resolution_row_for_market",
        _fake_market,
    )
    rows, failures = _clob_resolution_rows_for_markets(
        market_rows=[
            {
                "market_id": "market-not-closed",
                "condition_id": "condition-not-closed",
            }
        ],
        timeout_seconds=1.0,
    )

    assert rows == []
    assert failures == [
        {
            "market_id": "market-not-closed",
            "condition_id": "condition-not-closed",
            "reason_code": "settlement_resolution_market_not_closed",
        }
    ]


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
    assert "decision_critical_provider_failure_consecutive_limit" in report[
        "goal_failure_reason_codes"
    ]
    assert "consecutive_decision_critical_provider_failures_exceeded_limit" in report[
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


def test_execution_layer_v2_trade_http_failure_is_optional_degradation(
    tmp_path,
) -> None:
    unlock_dir, unlock_manifest_sha = _build_issue160_unlock_fixture(tmp_path)
    result = run_execution_layer_v2_one_hour_remap_paper_goal(
        ExecutionLayerV2OneHourRemapPaperGoalConfig(
            run_id="one-hour-goal-optional-trade-timeout",
            output_dir=tmp_path / "runs",
            duration_seconds=1,
            poll_interval_seconds=0.0,
            allow_short_diagnostic_run=True,
            paper_candidate_unlock_dir=unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=unlock_manifest_sha,
            public_provider=_TradeTimeoutProvider(),
            max_consecutive_orderbook_failure_rounds=1,
        )
    )

    status_rows = _read_jsonl(
        result.output_dir
        / "incremental_fresh_loop"
        / "provider_cycle_status.jsonl"
    )
    assert len(status_rows) == 1
    status = status_rows[0]
    assert status["decision_critical_provider_failure"] is False
    assert status["decision_optional_provider_failure"] is True
    assert status["orderbook_failure"] is False
    assert status["public_feature_row_count"] == 1
    assert status["public_trade_row_count"] == 0
    assert status["consecutive_orderbook_failure_count"] == 0
    assert status["public_data_degradation_reason_codes"] == [
        "read_only_public_http_timeout"
    ]
    assert status["provider_stage_statuses"]["trade_collection"][
        "decision_critical"
    ] is False
    assert result.goal_report["provider_fail_fast_stop_triggered"] is False


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


class _TradeTimeoutProvider(_NoOrderbookProvider):
    def orderbook_rows(self, markets, config):
        del config
        common = {
            "market_id": markets[0]["market_id"],
            "ts": 1_010_000,
            "available_at_ts": 1_010_000,
            "bid_size": 2.0,
            "ask_size": 2.0,
            "liquidity_depth": 4.0,
        }
        return [
            {
                **common,
                "token_id": "up-token",
                "outcome": "UP",
                "bid_price": 0.60,
                "ask_price": 0.62,
                "mid_price": 0.61,
            },
            {
                **common,
                "token_id": "down-token",
                "outcome": "DOWN",
                "bid_price": 0.38,
                "ask_price": 0.40,
                "mid_price": 0.39,
            },
        ]

    def trade_rows(self, markets, config):
        del markets, config
        raise RealCorpusPublicProviderError(
            "pytest optional trade HTTP timeout",
            reason_codes=("read_only_public_http_timeout",),
        )

    def resolution_rows(self, markets, config):
        del markets, config
        return []


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


def _write_regime_conditioned_ev_artifact(path) -> None:
    path.write_text(
        json.dumps(_regime_conditioned_ev_artifact_payload(), sort_keys=True),
        encoding="utf-8",
    )


def _regime_conditioned_ev_artifact_payload() -> dict[str, object]:
    feature_groups = {
        "canonical_o_score_and_action_margin": {
            "features": ["canonical_o_action_score", "action_score_margin"]
        },
        "btc_anchor_direction": {
            "features": [
                "btc_momentum",
                "reference_price_to_beat_distance_at_decision",
            ]
        },
        "market_price_value": {
            "features": ["p_up", "p_down", "execution_price"]
        },
        "execution_quality": {
            "features": [
                "spread_bps",
                "book_staleness_ms",
                "queue_fill_proxy",
                "time_to_close_seconds",
            ]
        },
        "pre_entry_exposure_state": {
            "features": [
                "entry_index_within_market",
                "cumulative_market_exposure_before_entry",
                "same_side_reentry",
                "side_flip",
            ]
        },
    }
    scales = {
        "canonical_o_action_score": 1.0,
        "action_score_margin": 0.10,
        "btc_momentum": 0.01,
        "reference_price_to_beat_distance_at_decision": 0.01,
        "p_up": 1.0,
        "p_down": 1.0,
        "execution_price": 1.0,
        "spread_bps": 1_000.0,
        "book_staleness_ms": 1_000.0,
        "queue_fill_proxy": 1.0,
        "time_to_close_seconds": 600.0,
        "entry_index_within_market": 10.0,
        "cumulative_market_exposure_before_entry": 1.0,
        "same_side_reentry": 1.0,
        "side_flip": 1.0,
    }
    coefficient_groups = {}
    for group_name, group in feature_groups.items():
        features = group["features"]
        feature_count = len(features)
        coefficient_groups[group_name] = {
            "group_coefficient": 0.002,
            "maximum_absolute_contribution": 0.002,
            "feature_weights": dict.fromkeys(features, 1.0 / feature_count),
            "feature_transforms": {
                feature: {
                    "center": 0.0,
                    "scale": scales[feature],
                    "clip_min": -1.0,
                    "clip_max": 1.0,
                }
                for feature in features
            },
        }
    return {
        "schema_version": (
            "bigan-v8-execution-layer-v2-frozen-regime-conditioned-ev-v1"
        ),
        "artifact_name": "execution_layer_v2_frozen_regime_conditioned_ev_v1",
        "diagnostic_only": True,
        "frozen": True,
        "decision_time_safe": True,
        "uses_validation_labels_for_tuning": False,
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        "market_implied_probability_used_as_conditioning_feature": True,
        "market_implied_probability_used_as_regime_direction_vote": False,
        "no_outcome_field_usage": True,
        "no_oracle_field_usage": True,
        "no_future_return_field_usage": True,
        "source_score_mutation_enabled": False,
        "o_score_mutation_enabled": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "fit_provenance": {
            "coefficients_source": "separate_calibration_training_split",
            "coefficients_fitted_from_current_75_row_replay": False,
            "uses_settlement_pnl_for_fitting": False,
            "uses_outcomes_for_fitting": False,
            "uses_oracle_actions_for_fitting": False,
            "uses_future_returns_for_fitting": False,
            "fitted_from_run_ids": ["separate-calibration-fixture"],
            "excluded_run_ids": [CURRENT_75_ROW_REPLAY_RUN_ID],
            "fit_dataset_hash": "a" * 64,
            "fit_config_hash": "b" * 64,
        },
        "independence_constraints": {
            "p_up_p_down_single_group": "market_price_value",
            "btc_anchor_fields_single_group": "btc_anchor_direction",
            "btc_anchor_maximum_signal_vote_weight": 1.0,
            "correlated_momentum_reference_counted_as_independent_votes": False,
        },
        "feature_groups": feature_groups,
        "coefficients": {
            "intercept": 0.03,
            "groups": coefficient_groups,
            "side_offsets": {},
            "family_offsets": {},
            "subtract_execution_cost": True,
        },
    }


def _regime_conditioned_forward_row(
    *,
    market_id: str,
    action: str,
    side: str,
    p_up: float,
    order_allowed: bool,
    blocking_reason_codes: list[str],
) -> dict[str, object]:
    row = _forward_shadow_row(
        market_id=market_id,
        action=action,
        selected_side=side,
        entry_ask=0.56,
        decision_ts=2_000,
        p_up=p_up,
        canonical_o_action_score=0.80,
        time_to_close_seconds=240.0,
        order_allowed=order_allowed,
        execution_guarded_action=action,
        execution_guarded_side=side,
        execution_blocking_reason_codes=blocking_reason_codes,
    )
    row.update(
        {
            "p_down": 1.0 - p_up,
            "action_score_margin": 0.08,
            "btc_momentum": 0.002 if side == "UP" else -0.002,
            "reference_price_to_beat_distance_at_decision": (
                0.001 if side == "UP" else -0.001
            ),
            "spread_bps": 50.0,
            "book_staleness_ms": 200.0,
            "queue_fill_proxy": 0.80,
            "entry_index_within_market": 1,
            "cumulative_market_exposure_before_entry": 0.0,
            "same_side_reentry": False,
            "side_flip": False,
            "decision_time_regime_feature_max_input_ts": 2_000,
        }
    )
    return row


def _regime_conditioned_ev_v2_calibration_rows() -> list[dict[str, object]]:
    return [
        _regime_conditioned_ev_v2_calibration_row(
            source_run_id=f"historical-calibration-source-{market_index}",
            market_index=market_index,
            row_index=row_index,
        )
        for market_index in range(8)
        for row_index in range(4)
    ]


def _write_regime_ev_corpus_source_run(
    run_dir: Path,
    *,
    run_id: str,
    row_count: int,
    market_offset: int = 0,
    settlement_pnl_offset: float = 0.0,
    unresolved: bool = False,
    hash_mismatch: bool = False,
    max_input_ts_violation: bool = False,
    repeat_market: bool = False,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "signal_trace.json"
    intent_path = run_dir / "paper_intents.jsonl"
    fill_path = run_dir / "paper_fills.jsonl"
    settlement_path = run_dir / "settlement_rows.jsonl"
    trace_rows = []
    intent_rows = []
    fill_rows = []
    settlement_rows = []
    for index in range(row_count):
        market_index = market_offset if repeat_market else market_offset + index
        decision_ts = 100_000 + market_offset * 1_000 + index * 100
        side = "UP" if market_index % 2 == 0 else "DOWN"
        family = (
            "HOLD_TO_SETTLEMENT"
            if market_index % 2 == 0
            else "SELL_BEFORE_CLOSE"
        )
        action = f"BUY_{side}_{family}"
        identity_suffix = f"{market_index}-{index}" if repeat_market else str(market_index)
        intent_id = f"intent-{identity_suffix}"
        fill_id = f"fill-{identity_suffix}"
        market_id = f"market-{market_index}"
        p_up = 0.65 if side == "UP" else 0.35
        trace_rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "market_end_ts": decision_ts + 300,
                "paper_intent_id": intent_id,
                "paper_fill_id": fill_id,
                "canonical_selected_action": action,
                "canonical_o_action_score": 0.55 + index * 0.01,
                "action_score_margin": 0.08,
                "btc_momentum": 0.002 if side == "UP" else -0.002,
                "reference_price_to_beat_distance_at_decision": (
                    0.003 if side == "UP" else -0.003
                ),
                "decision_time_regime_feature_max_input_ts": (
                    decision_ts + 1 if max_input_ts_violation else decision_ts
                ),
                "o_v8_paper_fresh_signal_trace_row_hash": hashlib.sha256(
                    f"trace-{run_id}-{index}".encode()
                ).hexdigest(),
            }
        )
        intent_rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "paper_fresh_order_intent_id": intent_id,
                "execution_guarded_action": action,
                "execution_guarded_side": side,
                "execution_guarded_family": family,
                "p_up": p_up,
                "p_down": 1.0 - p_up,
                "entry_ask": 0.55,
                "spread_bps": 80.0,
                "book_staleness_ms": 100.0,
                "queue_fill_proxy": 0.8,
                "time_to_close_seconds": 240.0,
                "pre_decision_exposure_state": {
                    "current_market_exposure_by_market_id": {},
                    "runtime_state_validation_passed": True,
                },
                "decision_time_regime_feature_max_input_ts": decision_ts,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        )
        fill_rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "paper_fresh_order_intent_id": intent_id,
                "paper_fresh_fill_id": fill_id,
                "paper_fill_price": 0.55,
                "filled_size": 0.2,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        )
        settlement_rows.append(
            {
                "market_id": market_id,
                "paper_fresh_order_intent_id": intent_id,
                "paper_fresh_fill_id": fill_id,
                "resolution_status": "pending" if unresolved else "resolved",
                "resolution_source_type": "polymarket_clob_read_only_settlement",
                "resolved_outcome": (
                    None if unresolved else ("UP" if index % 2 == 0 else "DOWN")
                ),
                "settlement_pnl": 0.01 * (index + 1) + settlement_pnl_offset,
                "outcome_observed_at_ts": decision_ts + 350,
                "outcome_observation_time_source": "artifact_recorded",
                "settlement_evaluation_row_hash": hashlib.sha256(
                    f"settlement-{run_id}-{index}-{settlement_pnl_offset}".encode()
                ).hexdigest(),
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        )
    trace_path.write_text(
        json.dumps({"trace_rows": trace_rows}, sort_keys=True), encoding="utf-8"
    )
    _write_jsonl(intent_path, intent_rows)
    _write_jsonl(fill_path, fill_rows)
    _write_jsonl(settlement_path, settlement_rows)
    artifact_paths = {
        "signal_trace": str(trace_path.resolve()),
        "paper_intent_log": str(intent_path.resolve()),
        "paper_fill_log": str(fill_path.resolve()),
        "settlement_evaluation_rows": str(settlement_path.resolve()),
    }
    artifact_hashes = {
        key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for key, path in artifact_paths.items()
    }
    if hash_mismatch:
        artifact_hashes["signal_trace"] = "f" * 64
    manifest = {
        "schema_version": "test-historical-calibration-source-v1",
        "run_id": run_id,
        "source_run_id": run_id,
        "completed": True,
        "immutable": True,
        "unresolved_settlement_count": 0,
        "artifact_paths": artifact_paths,
        "artifact_hashes": artifact_hashes,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    manifest_path = (
        run_dir / "execution_layer_v2_historical_calibration_source_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def _nest_regime_ev_corpus_signal_trace(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_path = Path(manifest["artifact_paths"].pop("signal_trace"))
    trace_sha256 = manifest["artifact_hashes"].pop("signal_trace")
    trace_manifest_dir = manifest_path.parent / "nested-fresh-loop"
    trace_manifest_dir.mkdir(parents=True, exist_ok=True)
    trace_manifest_path = trace_manifest_dir / "o_v8_paper_fresh_loop_manifest.json"
    trace_manifest = {
        "schema_version": "test-paper-fresh-loop-manifest-v1",
        "run_id": f"{manifest['run_id']}-fresh-loop",
        "artifact_paths": {
            "signal_trace_report": str(trace_path.resolve()),
        },
        "artifact_hashes": {
            "signal_trace_report": trace_sha256,
        },
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    trace_manifest_path.write_text(
        json.dumps(trace_manifest, sort_keys=True), encoding="utf-8"
    )
    manifest["artifact_paths"]["paper_fresh_loop_manifest"] = str(
        trace_manifest_path.resolve()
    )
    manifest["artifact_hashes"]["paper_fresh_loop_manifest"] = hashlib.sha256(
        trace_manifest_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return trace_manifest_path


def _attach_regime_conditioned_ev_v2_target_source(
    rows: list[dict[str, object]], tmp_path: Path
) -> None:
    source_path = tmp_path / "historical-target-evidence.jsonl"
    source_content = b"immutable historical settled target evidence\n"
    source_path.write_bytes(source_content)
    source_hash = hashlib.sha256(source_content).hexdigest()
    for row in rows:
        source_intent_id = str(row["source_intent_id"])
        source_fill_id = str(row["source_fill_id"])
        row["row_identity"] = regime_conditioned_ev_v2_calibration_row_identity(
            source_run_id=str(row["source_run_id"]),
            market_id=str(row["market_id"]),
            decision_ts=row["decision_ts"],
            selected_action=str(row["selected_action"]),
            source_intent_id=source_intent_id,
            source_fill_id=source_fill_id,
        )
        row["source_lineage"] = {
            "source_manifest_path": source_path.name,
            "source_manifest_sha256": source_hash,
            "trace_manifest_path": source_path.name,
            "trace_manifest_sha256": source_hash,
            "trace_artifact_path": source_path.name,
            "trace_artifact_sha256": source_hash,
            "trace_row_id": f"trace-{source_fill_id}",
            "intent_artifact_path": source_path.name,
            "intent_artifact_sha256": source_hash,
            "fill_artifact_path": source_path.name,
            "fill_artifact_sha256": source_hash,
            "settlement_artifact_path": source_path.name,
            "settlement_artifact_sha256": source_hash,
            "settlement_row_id": f"settlement-{source_fill_id}",
        }
        row["target_provenance"] = {
            "source_type": "polymarket_clob_read_only_settlement",
            "source_artifact_path": source_path.name,
            "source_artifact_sha256": source_hash,
            "resolution_status": "resolved",
            "resolved_outcome": (
                "UP" if (int(row["decision_ts"]) // 10) % 2 == 0 else "DOWN"
            ),
            "outcome_observed_after_market_close": True,
            "outcome_observation_time_source": "artifact_recorded",
            "outcome_observed_at_ts": row["market_close_ts"] + 50,
        }


def _regime_conditioned_ev_v2_calibration_row(
    *,
    source_run_id: str,
    market_index: int,
    row_index: int,
) -> dict[str, object]:
    decision_ts = 10_000 + market_index * 1_000 + row_index * 10
    side = "UP" if (market_index + row_index) % 2 == 0 else "DOWN"
    direction = 1.0 if side == "UP" else -1.0
    quality = (row_index - 1.5) / 1.5
    selected_probability = 0.58 + 0.04 * quality
    execution_price = 0.52 - 0.03 * quality
    canonical_score = 0.45 + 0.03 * ((market_index + row_index) % 3)
    action_margin = 0.03 + 0.01 * row_index
    momentum = direction * (0.001 + 0.001 * quality)
    reference_distance = direction * (0.002 + 0.0015 * quality)
    spread_bps = 140.0 - 60.0 * quality
    staleness_ms = 400.0 - 150.0 * quality
    queue = 0.55 + 0.25 * quality
    time_to_close = 220.0 + 90.0 * quality
    entry_index = 1.0 + row_index
    exposure = 0.1 * row_index
    same_side_reentry = 1.0 if row_index > 0 else 0.0
    side_flip = 0.0
    family = "HOLD_TO_SETTLEMENT" if row_index % 2 == 0 else "SELL_BEFORE_CLOSE"
    target = (
        0.12 * (selected_probability - execution_price)
        + 0.02 * quality
        + 0.01 * direction * momentum
        + 0.00002 * time_to_close
        - 0.00003 * spread_bps
        - 0.000002 * staleness_ms
        + 0.01 * queue
        - 0.003 * exposure
    )
    return {
        "source_run_id": source_run_id,
        "source_intent_id": f"intent-{market_index}-{row_index}",
        "source_fill_id": f"fill-{market_index}-{row_index}",
        "row_identity": "0" * 64,
        "source_lineage": {
            "source_manifest_path": "pending",
            "source_manifest_sha256": "0" * 64,
            "trace_manifest_path": "pending",
            "trace_manifest_sha256": "0" * 64,
            "trace_artifact_path": "pending",
            "trace_artifact_sha256": "0" * 64,
            "trace_row_id": "pending",
            "intent_artifact_path": "pending",
            "intent_artifact_sha256": "0" * 64,
            "fill_artifact_path": "pending",
            "fill_artifact_sha256": "0" * 64,
            "settlement_artifact_path": "pending",
            "settlement_artifact_sha256": "0" * 64,
            "settlement_row_id": "pending",
        },
        "market_id": f"calibration-market-{market_index}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "market_close_ts": decision_ts + 200,
        "selected_side": side,
        "selected_action": f"BUY_{side}_{family}",
        "action_family": family,
        "decision_time_features": {
            "canonical_o_action_score": canonical_score,
            "action_score_margin": action_margin,
            "btc_momentum": momentum,
            "reference_price_to_beat_distance_at_decision": reference_distance,
            "selected_side_probability": selected_probability,
            "execution_price": execution_price,
            "selected_side_probability_minus_execution_price": (
                selected_probability - execution_price
            ),
            "spread_bps": spread_bps,
            "book_staleness_ms": staleness_ms,
            "queue_fill_proxy": queue,
            "time_to_close_seconds": time_to_close,
            "entry_index_within_market": entry_index,
            "cumulative_market_exposure_before_entry": exposure,
            "same_side_reentry": int(same_side_reentry),
            "side_flip": int(side_flip),
        },
        "target_net_return_after_cost": target,
        "target_provenance": {
            "source_type": "polymarket_clob_read_only_settlement",
            "source_artifact_path": f"historical/{source_run_id}.jsonl",
            "source_artifact_sha256": "c" * 64,
            "resolution_status": "resolved",
            "resolved_outcome": "UP" if row_index % 2 == 0 else "DOWN",
            "outcome_observed_after_market_close": True,
            "outcome_observation_time_source": "artifact_recorded",
            "outcome_observed_at_ts": decision_ts + 300,
        },
    }


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
