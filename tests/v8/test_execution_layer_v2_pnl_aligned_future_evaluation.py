from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    evaluate_pnl_aligned_future_accepted_bets,
    validate_pnl_aligned_future_evaluation_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pnl_aligned_future_evaluation_v1.json"
)
ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "NO_TRADE",
)


def test_future_evaluation_protocol_is_frozen_and_non_tunable() -> None:
    protocol = _protocol()
    validate_pnl_aligned_future_evaluation_protocol(protocol)
    assert protocol["frozen_entry_edge_threshold"] == 0.02
    assert protocol["market_bootstrap"]["resample_count"] == 2000
    assert protocol["market_bootstrap"]["seed"] == 20260715
    assert protocol["uses_future_outcomes_for_threshold_selection"] is False
    assert protocol["uses_future_outcomes_for_guard_or_sizing_selection"] is False

    drifted = json.loads(json.dumps(protocol))
    drifted["frozen_entry_edge_threshold"] = 0.0
    with pytest.raises(ValueError, match="threshold"):
        validate_pnl_aligned_future_evaluation_protocol(drifted)


def test_accepted_bet_evaluation_reconciles_market_pnl_and_stays_blocked(
    tmp_path: Path,
) -> None:
    historical_path = tmp_path / "historical.jsonl"
    historical_path.write_text(
        json.dumps({"market_id": "historical-market", "decision_ts": 100}) + "\n"
    )
    collection_freeze = {
        "minimum_future_window_start_ts": 1_000,
        "historical_development_rows": {
            "path": str(historical_path),
            "sha256": _sha256(historical_path),
        },
    }
    candidate = []
    baseline = []
    targets = []
    for index in range(30):
        side = "UP" if index % 2 == 0 else "DOWN"
        action = f"BUY_{side}_HOLD_TO_SETTLEMENT"
        identity = f"future-row-{index}"
        market_id = f"future-market-{index}"
        decision_ts = 2_000 + index * 300_000
        common = {
            "source_row_identity": identity,
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_close_ts": decision_ts + 240_000,
            "selected_action": action,
            "selected_side": side,
            "selected_action_family": "HOLD_TO_SETTLEMENT",
            "selected_execution_price": 0.5,
            "execution_guarded_action": action,
            "execution_guarded_side": side,
            "proposed_order_size": 0.2,
            "outcome_fields_used": False,
            "realized_pnl_used": False,
            "source_o_score_mutated": False,
            "source_ranking_mutated": False,
        }
        candidate.append(
            {
                **common,
                "execution_guard_order_allowed": True,
                "simulated_order_id": f"candidate-{index}",
            }
        )
        baseline.append(
            {
                **common,
                "execution_guard_order_allowed": False,
                "execution_guarded_action": None,
                "execution_guarded_side": None,
                "proposed_order_size": 0.0,
                "simulated_order_id": None,
            }
        )
        target_values = dict.fromkeys(ACTIONS, 0.0)
        target_values[action] = 0.1
        components = {
            name: {
                "gross_pnl_per_contract": 0.0,
                "execution_cost_per_contract": 0.0,
                "net_pnl_per_contract": 0.0,
            }
            for name in ACTIONS
        }
        components[action] = {
            "gross_pnl_per_contract": 0.11,
            "execution_cost_per_contract": 0.01,
            "net_pnl_per_contract": 0.1,
        }
        targets.append(
            {
                "row_identity": identity,
                "market_id": market_id,
                "decision_ts": decision_ts,
                "evaluation_target_net_pnl_per_contract_by_action": target_values,
                "evaluation_target_pnl_components_by_action": components,
            }
        )

    report, pnl_rows = evaluate_pnl_aligned_future_accepted_bets(
        evaluation_protocol=_protocol(),
        collection_freeze_manifest=collection_freeze,
        candidate_shadow_rows=candidate,
        baseline_shadow_rows=baseline,
        settled_evaluation_rows=targets,
    )

    assert len(pnl_rows) == 60
    assert report["future_evidence_gate_passed"] is True
    assert report["candidate_policy_metrics"]["accepted_bet_count"] == 30
    assert report["candidate_policy_metrics"]["accepted_unique_market_count"] == 30
    assert report["candidate_policy_metrics"]["accepted_bet_count_by_side"] == {
        "DOWN": 15,
        "UP": 15,
    }
    assert report["candidate_policy_metrics"]["settled_net_pnl_sum"] == pytest.approx(
        0.6
    )
    assert report["baseline_policy_metrics"]["settled_net_pnl_sum"] == 0.0
    assert report["market_bootstrap_interval"]["reported"] is True
    assert report["market_bootstrap_interval"]["market_count"] == 30
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_future_evaluation_rejects_historical_market_overlap(tmp_path: Path) -> None:
    historical_path = tmp_path / "historical.jsonl"
    historical_path.write_text(
        json.dumps({"market_id": "overlap", "decision_ts": 100}) + "\n"
    )
    freeze = {
        "minimum_future_window_start_ts": 1_000,
        "historical_development_rows": {
            "path": str(historical_path),
            "sha256": _sha256(historical_path),
        },
    }
    shadow = {
        "source_row_identity": "row",
        "market_id": "overlap",
        "decision_ts": 2_000,
        "market_close_ts": 3_000,
        "selected_action": "NO_TRADE",
        "selected_execution_price": 0.0,
        "execution_guard_order_allowed": False,
        "execution_guarded_action": None,
        "execution_guarded_side": None,
        "proposed_order_size": 0.0,
        "simulated_order_id": None,
    }
    targets = {
        "row_identity": "row",
        "market_id": "overlap",
        "decision_ts": 2_000,
        "evaluation_target_net_pnl_per_contract_by_action": dict.fromkeys(
            ACTIONS, 0.0
        ),
        "evaluation_target_pnl_components_by_action": {
            action: {
                "gross_pnl_per_contract": 0.0,
                "execution_cost_per_contract": 0.0,
                "net_pnl_per_contract": 0.0,
            }
            for action in ACTIONS
        },
    }
    with pytest.raises(ValueError, match="overlap historical"):
        evaluate_pnl_aligned_future_accepted_bets(
            evaluation_protocol=_protocol(),
            collection_freeze_manifest=freeze,
            candidate_shadow_rows=[shadow],
            baseline_shadow_rows=[shadow],
            settled_evaluation_rows=[targets],
        )


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
