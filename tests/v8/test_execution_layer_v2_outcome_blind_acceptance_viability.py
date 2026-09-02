from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _load_outcome_blind_feature_rows,
    _materialize_outcome_blind_action_rows,
    _outcome_blind_acceptance_replay,
    _validate_complete_action_grid,
    _validate_frozen_calibration,
    _viability_report,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

FEATURE_CONTRACT_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)


def test_issue196_feature_loader_allows_only_pinned_feature_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "direct-training-corpus"
    feature_path = root / "market-1" / "polymarket_feature_rows.jsonl"
    row = _feature_row(market_id="market-1", decision_ts=1_000)
    descriptor = _write_jsonl(feature_path, [row])
    role_row = {"market_id": "market-1", "feature_rows": descriptor}

    rows, verified = _load_outcome_blind_feature_rows(
        role_row,
        allowed_corpus_root=root,
    )

    assert rows == [row]
    assert verified == descriptor

    label_path = root / "market-1" / "polymarket_label_rows.jsonl"
    role_row["feature_rows"] = _write_jsonl(label_path, [row])
    with pytest.raises(ValueError, match="forbidden non-feature artifact"):
        _load_outcome_blind_feature_rows(role_row, allowed_corpus_root=root)

    leaked = {**row, "settlement_outcome": "UP"}
    role_row["feature_rows"] = _write_jsonl(feature_path, [leaked])
    with pytest.raises(ValueError, match="forbidden outcome fields"):
        _load_outcome_blind_feature_rows(role_row, allowed_corpus_root=root)

    outside_path = tmp_path / "outside" / "polymarket_feature_rows.jsonl"
    role_row["feature_rows"] = _write_jsonl(outside_path, [row])
    with pytest.raises(ValueError, match="outside the direct training corpus root"):
        _load_outcome_blind_feature_rows(role_row, allowed_corpus_root=root)


def test_issue196_calibration_contract_rejects_future_or_confirmatory_tuning() -> None:
    safety = {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    calibration = {
        "source_split": "development_calibration_only",
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_issue174_confirmatory_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        "raw_rank_score_cross_model_comparison_allowed": False,
        **safety,
    }

    _validate_frozen_calibration(calibration)

    calibration["uses_confirmatory_validation_labels_for_tuning"] = True
    with pytest.raises(ValueError, match="confirmatory_labels_not_used"):
        _validate_frozen_calibration(calibration)


def test_issue196_materializer_enforces_causality_and_complete_five_action_grid() -> None:
    feature_columns = tuple(_load_json(FEATURE_CONTRACT_PATH)["feature_columns"])
    role_row = {"market_id": "market-1", "selection_rank": 91}
    rows = _materialize_outcome_blind_action_rows(
        [_feature_row(market_id="market-1", decision_ts=1_000)],
        role_row=role_row,
        feature_columns=feature_columns,
    )

    assert len(rows) == 5
    assert {row["action"] for row in rows} == set(REQUIRED_ACTIONS)
    assert all(row["max_input_ts"] <= row["decision_ts"] for row in rows)
    assert all(row["target_used_as_decision_input"] is False for row in rows)
    _validate_complete_action_grid(rows)

    invalid = _feature_row(market_id="market-1", decision_ts=1_000)
    invalid["max_input_ts"] = 1_001
    with pytest.raises(ValueError, match="timestamp causality violation"):
        _materialize_outcome_blind_action_rows(
            [invalid],
            role_row=role_row,
            feature_columns=feature_columns,
        )

    with pytest.raises(ValueError, match="action grid is incomplete"):
        _validate_complete_action_grid(rows[:-1])


def test_issue196_attrition_is_deterministic_and_uses_first_terminal_stage() -> None:
    predictions = _prediction_rows()

    def guard_blocked(decision: dict, **_: object) -> dict:
        return {
            "order_allowed": False,
            "execution_guarded_action": decision["selected_action"],
            "execution_blocking_reason_codes": ["execution_time_to_close_unsafe"],
        }

    first = _outcome_blind_acceptance_replay(
        predictions,
        entry_threshold=0.02,
        runner_up_advantage_threshold=0.0,
        guard_decision_fn=guard_blocked,
    )
    second = _outcome_blind_acceptance_replay(
        predictions,
        entry_threshold=0.02,
        runner_up_advantage_threshold=0.0,
        guard_decision_fn=guard_blocked,
    )

    assert first == second
    assert [row["first_terminal_stage"] for row in first] == [
        "selected_no_trade",
        "time_to_close",
    ]
    assert first[0]["all_trade_action_lcbs_nonpositive"] is True
    assert first[1]["source_selected_action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    assert first[1]["execution_guard_order_allowed"] is False
    assert len(first[1]["full_five_action_ranking"]) == 5
    assert all(row["target_or_outcome_fields_used"] is False for row in first)


def test_issue196_report_reconciles_attrition_and_keeps_all_unlocks_blocked() -> None:
    viability_rows = _outcome_blind_acceptance_replay(
        _prediction_rows(),
        entry_threshold=0.02,
        runner_up_advantage_threshold=0.0,
        guard_decision_fn=lambda decision, **_: {
            "order_allowed": False,
            "execution_guarded_action": decision["selected_action"],
            "execution_blocking_reason_codes": ["execution_time_to_close_unsafe"],
        },
    )
    report = _viability_report(
        run_id="test-run",
        role_rows=[{"market_id": "market-1"}],
        source_feature_row_count=2,
        action_rows=_prediction_rows(),
        viability_rows=viability_rows,
        protocol={
            "frozen_execution_contract": {
                "entry_edge_threshold": 0.02,
                "runner_up_advantage_threshold": 0.0,
            },
            "development_freeze_gates": {"minimum_accepted_bet_count": 30},
        },
        candidate={"candidate_name": "frozen-test-candidate"},
        opened_feature_paths=[],
        input_descriptors={
            "model": {"sha256": "1" * 64},
            "calibration_artifact": {"sha256": "2" * 64},
            "protocol": {"sha256": "3" * 64},
            "feature_contract": {"sha256": "4" * 64},
        },
    )

    assert report["first_terminal_stage_reconciled"] is True
    assert report["target_or_outcome_files_opened"] is False
    assert report["current_oof_or_validation_pnl_used"] is False
    assert report["threshold_sweep_performed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["paper_candidate_allowed"] is False
    assert report["live_trading_enabled"] is False
    assert report["broker_exchange_write_enabled"] is False
    assert report["live_exchange_write_enabled"] is False


def _prediction_rows() -> list[dict]:
    feature_columns = tuple(_load_json(FEATURE_CONTRACT_PATH)["feature_columns"])
    materialized: list[dict] = []
    for decision_ts in (1_000, 2_000):
        materialized.extend(
            _materialize_outcome_blind_action_rows(
                [_feature_row(market_id="market-1", decision_ts=decision_ts)],
                role_row={"market_id": "market-1", "selection_rank": 91},
                feature_columns=feature_columns,
            )
        )
    scores = {
        1_000: {
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD_TO_SETTLEMENT": -0.01,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.02,
            "BUY_UP_SELL_BEFORE_CLOSE": -0.03,
            "BUY_DOWN_SELL_BEFORE_CLOSE": -0.04,
        },
        2_000: {
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.04,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.01,
            "NO_TRADE": 0.0,
            "BUY_UP_SELL_BEFORE_CLOSE": -0.01,
            "BUY_DOWN_SELL_BEFORE_CLOSE": -0.02,
        },
    }
    output = []
    for row in materialized:
        score = scores[row["decision_ts"]][row["action"]]
        output.append(
            {
                **row,
                "raw_pairwise_rank_score": score,
                "pairwise_group_normalized_rank_score": score,
                "calibrated_action_expected_net_return": score,
                "action_advantage_lcb_net_return": score,
                "action_advantage_lcb_score_bucket": "test_bucket",
                "action_advantage_lcb_estimate_source": "test_only",
            }
        )
    return output


def _feature_row(*, market_id: str, decision_ts: int) -> dict:
    source_ts = decision_ts - 1
    return {
        "market_id": market_id,
        "condition_id": market_id,
        "slug": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": source_ts,
        "available_at_ts": decision_ts,
        "feature_cutoff_ts": decision_ts,
        "features": {
            "up_bid": 0.49,
            "up_ask": 0.51,
            "up_mid": 0.50,
            "down_bid": 0.49,
            "down_ask": 0.51,
            "down_mid": 0.50,
            "up_spread_bps": 400.0,
            "down_spread_bps": 400.0,
            "up_book_staleness_ms": 10.0,
            "down_book_staleness_ms": 10.0,
            "up_queue_fill_probability_proxy": 0.8,
            "down_queue_fill_probability_proxy": 0.8,
            "reference_price_to_beat_distance_at_decision": 0.001,
            "time_to_close_seconds": 120.0,
        },
        "feature_provenance": {
            "reference_price_to_beat_distance_at_decision": {
                "provenance_valid": True,
                "max_input_ts": source_ts,
                "source_ts": source_ts,
            }
        },
        "paper_only": True,
        "capital_at_risk": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
