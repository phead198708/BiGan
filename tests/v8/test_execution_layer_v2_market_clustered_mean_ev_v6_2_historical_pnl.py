from __future__ import annotations

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_historical_pnl import (
    MarketClusteredMeanEVV62HistoricalPnlConfig,
    build_historical_pnl_report,
    historical_policy_metrics,
    join_historical_targets_after_replay,
)


def _replay_row(*, allowed: bool = True, side: str = "UP") -> dict:
    action = f"BUY_{side}_SELL_BEFORE_CLOSE"
    return {
        "market_id": "market-1",
        "decision_ts": 100,
        "source_selected_action": f"BUY_{side}_HOLD_TO_SETTLEMENT",
        "executed_action": action,
        "selected_side": side,
        "execution_guard_order_allowed": allowed,
        "proposed_order_size": 0.2 if allowed else 0.0,
        "microstructure_snapshot": {"entry_ask": 0.5},
        "target_used_as_decision_input": False,
        "outcome_fields_used_as_decision_input": False,
    }


def _config() -> MarketClusteredMeanEVV62HistoricalPnlConfig:
    return MarketClusteredMeanEVV62HistoricalPnlConfig(
        run_id="test",
        output_dir="runs",
        candidate_manifest_path="candidate.json",
        expected_candidate_manifest_sha256="a" * 64,
        v5_freeze_manifest_path="v5.json",
        expected_v5_freeze_manifest_sha256="b" * 64,
        feature_contract_path="features.json",
        expected_feature_contract_sha256="c" * 64,
        implementation_commit="d" * 40,
    )


def test_historical_target_join_uses_executed_action_after_replay() -> None:
    row = _replay_row()
    targets = {
        ("market-1", 100, "BUY_UP_HOLD_TO_SETTLEMENT"): {
            "target_net_pnl_per_contract": -1.0,
            "historical_source_role": "development_train",
            "historical_model_usage_role": "historical_model_fit",
        },
        ("market-1", 100, "BUY_UP_SELL_BEFORE_CLOSE"): {
            "target_net_pnl_per_contract": 0.3,
            "historical_source_role": "development_train",
            "historical_model_usage_role": "historical_model_fit",
        },
    }

    joined = join_historical_targets_after_replay(
        [row], targets=targets, policy_name="candidate"
    )

    assert joined[0]["accepted_bet_net_pnl"] == 0.06
    assert joined[0]["target_joined_after_outcome_blind_guard_replay"] is True
    assert joined[0]["target_used_as_decision_input"] is False


def test_historical_policy_metrics_are_chronological_and_side_first() -> None:
    rows = []
    for decision_ts, side, pnl in ((300, "UP", 0.1), (100, "DOWN", 0.2), (200, "UP", -0.15)):
        row = _replay_row(side=side)
        row.update(
            {
                "market_id": f"market-{decision_ts}",
                "decision_ts": decision_ts,
                "accepted_bet_net_pnl": pnl,
            }
        )
        rows.append(row)

    metrics = historical_policy_metrics(rows)

    assert metrics["guard_accepted_bet_count"] == 3
    assert metrics["accepted_side_distribution"] == {"DOWN": 1, "UP": 2}
    assert abs(metrics["accepted_bet_net_pnl_sum"] - 0.15) < 1e-12
    assert abs(metrics["chronological_max_drawdown"] - 0.15) < 1e-12


def test_historical_report_is_diagnostic_and_keeps_all_unlocks_false() -> None:
    candidate = join_historical_targets_after_replay(
        [_replay_row()],
        targets={
            ("market-1", 100, "BUY_UP_SELL_BEFORE_CLOSE"): {
                "target_net_pnl_per_contract": 0.3,
                "historical_source_role": "development_train",
                "historical_model_usage_role": "historical_model_fit",
            }
        },
        policy_name="candidate",
    )
    baseline_row = _replay_row(allowed=False)
    baseline = join_historical_targets_after_replay(
        [baseline_row],
        targets={
            ("market-1", 100, "BUY_UP_SELL_BEFORE_CLOSE"): {
                "target_net_pnl_per_contract": 0.3,
                "historical_source_role": "development_train",
                "historical_model_usage_role": "historical_model_fit",
            }
        },
        policy_name="baseline",
    )
    labeled = [
        {
            "market_id": "market-1",
            "source_v5_role": "development_train",
        }
    ]

    report = build_historical_pnl_report(
        config=_config(),
        candidate_rows=candidate,
        baseline_rows=baseline,
        labeled_rows=labeled,
    )

    assert report["final_combined_candidate_post_cost_net_pnl"] == 0.06
    assert report["historical_outcome_aware_diagnostic_only"] is True
    assert report["promotion_evidence"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
