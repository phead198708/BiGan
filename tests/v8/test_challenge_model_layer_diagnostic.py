from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import (
    append_outcome_blind_batch,
    build_daily_capture_summary,
    load_jsonl,
    validate_development_lane_protocol,
    validate_outcome_blind_batch_summary,
)
from bigan.v8.polymarket.challenge_model_layer_diagnostic import (
    canonical_json_sha256,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_model_layer_diagnostic_cost_identities_and_direction() -> None:
    report_path = ROOT / (
        "examples/v8/polymarket_configs/challenge_model_layer_market_diagnostic.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload_sha = report.pop("report_payload_sha256")
    assert canonical_json_sha256(report) == payload_sha
    report["report_payload_sha256"] = payload_sha

    assert report["training_started"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["market_selection"]["recommendation"] == "turn_to_15m"
    assert report["market_selection"]["new_persistent_collection_market_families"] == [
        "btc_updown_15m"
    ]

    exact = report["exact_195_5m_cost_edge_decomposition"]
    assert exact["market_count"] == 195
    assert exact["outcome_distribution"] == {"DOWN": 99, "UP": 96}
    assert exact["variants"]["v8_1_23_bet"]["trade_count"] == 23
    assert exact["variants"]["iteration_4"]["trade_count"] == 76
    assert exact["variants"]["iteration_5"]["trade_count"] == 53
    for rows in exact["per_bet_decomposition"].values():
        for row in rows:
            assert math.isclose(
                row["gross_mark_to_mid_edge"]
                - row["spread_cost"]
                - row["fee"]
                - row["slippage_assumption"]
                - row["liquidity_impact"],
                row["unit_net_pnl"],
                abs_tol=1e-12,
            )

    fifteen = report["historical_15m_cost_edge_decomposition"]
    assert fifteen["trade_count"] == 119
    assert fifteen["true_source_token_executable_quote_count"] == 96
    assert fifteen["opposite_side_complement_quote_proxy_count"] == 23
    assert fifteen["conservative_after_cost_pnl"]["sum"] > 0.0
    assert (
        report["down_concentration_attribution"]["up_mid_minus_realized_frequency"][
            "bootstrap_95_lcb"
        ]
        < 0.0
        < report["down_concentration_attribution"]["up_mid_minus_realized_frequency"][
            "bootstrap_95_ucb"
        ]
    )


def test_development_lane_protocol_binds_15m_diagnostic() -> None:
    protocol_path = ROOT / (
        "examples/v8/polymarket_configs/challenge_model_development_lane_15m_protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validated = validate_development_lane_protocol(protocol, repo_root=ROOT)
    assert validated["market_family"] == "btc_updown_15m"
    assert validated["batch_round_count"] == 4
    assert validated["maximum_capture_attempts_before_additional_permission"] == 119


def test_development_lane_start_record_preserves_safety_and_authorization() -> None:
    start_path = ROOT / (
        "examples/v8/polymarket_configs/"
        "challenge_model_development_lane_15m_start_record.json"
    )
    start = json.loads(start_path.read_text(encoding="utf-8"))
    assert start["market_family"] == "btc_updown_15m"
    assert start["collector"]["commit"] == "7d3a4d7ebe072910bca44647c25d9a7710181dce"
    assert start["collector"]["pid"] > 0
    assert start["post_close_development_finalizer"]["pid"] > 0
    assert start["data_governance"]["development_only_forever"] is True
    assert start["data_governance"]["promotion_evidence_eligible"] is False
    assert start["authorization_checkpoint"] == {
        "collector_stops_before_attempt_120": True,
        "explicit_120_round_authorization_recorded": False,
        "maximum_capture_attempts_before_pause": 119,
    }
    assert not any(start["safety"].values())


def test_outcome_blind_batch_index_and_daily_summary(tmp_path: Path) -> None:
    summary = _batch_summary()
    summary_path = tmp_path / "batch_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    index_path = tmp_path / "capture_index.jsonl"
    entry = append_outcome_blind_batch(
        index_path=index_path,
        summary_path=summary_path,
        collector_commit="1" * 40,
        protocol_sha256="2" * 64,
        diagnostic_freeze_sha256="3" * 64,
        collected_at="2026-07-27T12:00:00+00:00",
    )
    assert entry["market_family"] == "btc_updown_15m"
    assert entry["capture_count"] == 1
    assert load_jsonl(index_path) == [entry]
    daily = build_daily_capture_summary(
        index_rows=[entry],
        date_utc="2026-07-27",
        collector_pid=123,
        service_status="running",
    )
    assert daily["attempted_market_count"] == 1
    assert daily["outcomes_labels_or_pnl_read"] is False
    assert daily["provider_health"]["raw_trade_nonempty_capture_count"] == 1


def test_batch_summary_rejects_target_access() -> None:
    summary = _batch_summary()
    summary["labels_or_outcomes_opened_during_collection"] = True
    with pytest.raises(ValueError, match="forbidden state"):
        validate_outcome_blind_batch_summary(summary)


def _batch_summary() -> dict:
    return {
        "batch_id": "challenge-development-15m-test",
        "outcome_blind_collection_only": True,
        "labels_or_outcomes_opened_during_collection": False,
        "settlement_pnl_opened_during_collection": False,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        "training_corpus_export_attempted": False,
        "finalization_attempt_count": 0,
        "finalizations": [],
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_exchange_write_enabled": False,
        "broker_exchange_write_enabled": False,
        "capture_count": 1,
        "error_count": 0,
        "chainlink_covered_capture_count": 1,
        "chainlink_fresh_capture_count": 1,
        "orderbook_full_window_coverage_passed_capture_count": 1,
        "orderbook_full_window_coverage_failed_capture_count": 0,
        "captures": [
            {
                "market_family": "btc_updown_15m",
                "capture_status": "pending_resolution",
                "raw_trade_row_count": 12,
            }
        ],
    }
