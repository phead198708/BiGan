from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    SETTLED_CORPUS_INDEX_SCHEMA_VERSION,
    _feature_payload,
    _join_frozen_replay_targets,
    _validate_settled_corpus_index,
)


def test_join_targets_uses_executed_action_order_size_and_side_only_safety() -> None:
    replay = [
        _replay_row(
            market_id="market-1",
            decision_ts=100,
            action="BUY_UP_SELL_BEFORE_CLOSE",
            side="UP",
            allowed=True,
            order_size=0.2,
        ),
        _replay_row(
            market_id="market-1",
            decision_ts=200,
            action="BUY_DOWN_HOLD_TO_SETTLEMENT",
            side="DOWN",
            allowed=False,
            order_size=0.0,
        ),
    ]
    targets = {
        ("market-1", 100): _target_row("market-1", 100, BUY_UP_SELL_BEFORE_CLOSE=0.5),
        ("market-1", 200): _target_row("market-1", 200, BUY_DOWN_HOLD_TO_SETTLEMENT=-0.4),
    }

    rows = _join_frozen_replay_targets(
        replay,
        targets_by_decision=targets,
        policy_name="candidate",
        decision_freeze_sha256="a" * 64,
    )

    assert rows[0]["accepted_bet_net_pnl"] == pytest.approx(0.1)
    assert rows[1]["accepted_bet_net_pnl"] == 0.0
    assert rows[0]["target_used_as_decision_input"] is False
    assert rows[0]["future_results_used_for_tuning"] is False
    assert rows[0]["source_model_candidate_eligible"] is False
    assert rows[0]["v8_execution_handoff_allowed"] is False
    assert rows[0]["capital_at_risk"] is False


def test_settled_index_requires_exact_frozen_market_set_and_post_freeze_time() -> None:
    selected = [
        {"market_id": "market-1"},
        {"market_id": "market-2"},
    ]
    index = _settled_index(["market-1", "market-2"])

    rows = _validate_settled_corpus_index(
        index,
        expected_decision_freeze_sha256="a" * 64,
        decision_freeze_created_ts=150,
        selected_rows=selected,
        reconciliation_started_ts=300,
    )

    assert [row["market_id"] for row in rows] == ["market-1", "market-2"]

    index["entries"] = index["entries"][:1]
    with pytest.raises(ValueError, match="complete_market_set"):
        _validate_settled_corpus_index(
            index,
            expected_decision_freeze_sha256="a" * 64,
            decision_freeze_created_ts=150,
            selected_rows=selected,
            reconciliation_started_ts=300,
        )


def test_settled_index_before_decision_freeze_fails_closed() -> None:
    index = _settled_index(["market-1"])
    index["index_created_ts"] = 100

    with pytest.raises(ValueError, match="index_after_decision_freeze"):
        _validate_settled_corpus_index(
            index,
            expected_decision_freeze_sha256="a" * 64,
            decision_freeze_created_ts=150,
            selected_rows=[{"market_id": "market-1"}],
            reconciliation_started_ts=300,
        )


def test_frozen_feature_comparison_ignores_only_freeze_metadata() -> None:
    feature = {
        "market_id": "market-1",
        "condition_id": "condition-1",
        "slug": "btc-updown-5m-1",
        "market_family": "btc_updown_5m",
        "horizon_ms": 300_000,
        "decision_ts": 100,
        "feature_cutoff_ts": 100,
        "max_input_ts": 100,
        "available_at_ts": 100,
        "features": {"p_up": 0.6},
        "feature_provenance": {"p_up": {"max_input_ts": 100}},
    }
    frozen = {
        **feature,
        "future_window_selection_rank": 1,
        "future_feature_row_sha256": "b" * 64,
        "target_used_as_decision_input": False,
    }

    assert _feature_payload(feature) == _feature_payload(frozen)
    frozen["features"] = {"p_up": 0.61}
    assert _feature_payload(feature) != _feature_payload(frozen)


def test_settlement_artifacts_are_hashable(tmp_path: Path) -> None:
    path = tmp_path / "settlement.json"
    path.write_text(json.dumps({"paper_only": True}, sort_keys=True) + "\n", encoding="utf-8")
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64


def _replay_row(
    *,
    market_id: str,
    decision_ts: int,
    action: str,
    side: str,
    allowed: bool,
    order_size: float,
) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "executed_action": action,
        "selected_side": side,
        "selected_action_family": (
            "SELL_BEFORE_CLOSE" if action.endswith("SELL_BEFORE_CLOSE") else "HOLD_TO_SETTLEMENT"
        ),
        "execution_guard_order_allowed": allowed,
        "proposed_order_size": order_size,
    }


def _target_row(market_id: str, decision_ts: int, **overrides: float) -> dict:
    values = {
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.0,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.0,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.0,
        "BUY_DOWN_SELL_BEFORE_CLOSE": 0.0,
        "NO_TRADE": 0.0,
        **overrides,
    }
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "resolved_outcome": "UP",
        "target_net_pnl_per_notional_by_action": values,
    }


def _settled_index(market_ids: list[str]) -> dict:
    safety = {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "v8_execution_handoff_allowed": False,
        "paper_candidate_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    return {
        "schema_version": SETTLED_CORPUS_INDEX_SCHEMA_VERSION,
        "decision_freeze_sha256": "a" * 64,
        "index_created_ts": 200,
        "outcomes_used_for_decision_or_selection": False,
        "outcomes_used_for_threshold_or_model_tuning": False,
        "entries": [
            {
                "market_id": market_id,
                "official_read_only_resolution": True,
                "corpus_built_after_decision_freeze": True,
                "settled_after_market_close": True,
            }
            for market_id in market_ids
        ],
        **safety,
    }
