from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    PnLAlignedActionValueFitConfig,
    build_pnl_aligned_action_conditioned_rows,
    fit_frozen_pnl_aligned_action_value_model,
    predict_frozen_pnl_aligned_action_values,
    run_pnl_aligned_action_value_outcome_blind_shadow,
    validate_pnl_aligned_action_value_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pnl_aligned_action_value_v1.json"
)
ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "NO_TRADE",
)


def test_protocol_is_frozen_non_leaky_and_fail_closed() -> None:
    protocol = _protocol()
    validate_pnl_aligned_action_value_protocol(protocol)
    assert protocol["primary_target"] == "total_net_pnl_per_notional"
    assert protocol["uses_current_oof_pnl_for_hyperparameter_selection"] is False
    assert protocol["uses_validation_labels_for_tuning"] is False
    assert protocol["uses_future_holdout_labels_for_fitting"] is False
    assert protocol["market_implied_probability_used_as_direct_fair_value_ev"] is False
    assert protocol["market_implied_probability_used_as_conditioning_feature"] is True
    assert protocol["market_implied_probability_used_as_regime_direction_vote"] is False
    assert protocol["future_evidence_gates"]["minimum_unique_market_count"] == 30
    assert protocol["future_evidence_gates"]["minimum_accepted_bet_count"] == 30

    drifted = json.loads(json.dumps(protocol))
    drifted["uses_current_oof_pnl_for_hyperparameter_selection"] = True
    with pytest.raises(ValueError, match="no_current_oof_pnl_tuning"):
        validate_pnl_aligned_action_value_protocol(drifted)


def test_builds_complete_action_grid_without_target_leakage() -> None:
    protocol = _protocol()
    rows, audit = build_pnl_aligned_action_conditioned_rows(
        [_source_row(0, include_targets=True)],
        protocol=protocol,
        require_targets=True,
    )

    assert audit["passed"] is True
    assert audit["complete_5_action_grid"] is True
    assert audit["feature_max_input_ts_violation_count"] == 0
    assert audit["forbidden_decision_field_violation_count"] == 0
    assert len(rows) == 5
    assert {row["action"] for row in rows} == set(ACTIONS)
    assert all(row["max_input_ts"] <= row["decision_ts"] for row in rows)
    assert all(row["target_used_as_decision_input"] is False for row in rows)
    assert all(
        "resolved_outcome" not in row["decision_time_features"] for row in rows
    )
    up = next(row for row in rows if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT")
    down = next(
        row for row in rows if row["action"] == "BUY_DOWN_HOLD_TO_SETTLEMENT"
    )
    assert up["decision_time_features"]["selected_side_probability"] == 0.62
    assert down["decision_time_features"]["selected_side_probability"] == 0.38
    assert up["decision_time_features"]["btc_anchor_direction_signal"] == pytest.approx(
        -down["decision_time_features"]["btc_anchor_direction_signal"]
    )
    assert up["execution_handoff_context"]["corrected_model_score"] == 0.8
    assert up["execution_handoff_context"]["p_up_action_disagreement"] is False
    assert down["execution_handoff_context"]["p_up_action_disagreement"] is True


def test_outcome_blind_input_rejects_top_level_target_and_bad_timestamp() -> None:
    protocol = _protocol()
    target_leaky = _source_row(0, include_targets=True)
    rows, audit = build_pnl_aligned_action_conditioned_rows(
        [target_leaky],
        protocol=protocol,
        require_targets=False,
    )
    assert rows == []
    assert "forbidden_decision_field_present" in audit["blocking_reason_codes"]

    bad_timestamp = _source_row(0, include_targets=False)
    bad_timestamp["max_input_ts"] = bad_timestamp["decision_ts"] + 1
    rows, audit = build_pnl_aligned_action_conditioned_rows(
        [bad_timestamp],
        protocol=protocol,
        require_targets=False,
    )
    assert rows == []
    assert "decision_time_feature_causality_violation" in audit[
        "blocking_reason_codes"
    ]


def test_fit_freezes_research_artifact_and_predicts_outcome_blind_grid(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "historical_rows.jsonl"
    _write_jsonl(
        rows_path,
        [_source_row(index, include_targets=True) for index in range(6)],
    )
    corpus_manifest_path = tmp_path / "historical_manifest.json"
    _write_json(
        corpus_manifest_path,
        {
            "development_rows": {
                "path": str(rows_path.resolve()),
                "sha256": _sha256(rows_path),
            }
        },
    )
    result = fit_frozen_pnl_aligned_action_value_model(
        PnLAlignedActionValueFitConfig(
            run_id="pnl-aligned-fit",
            output_dir=tmp_path / "runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            historical_corpus_manifest_path=corpus_manifest_path,
        )
    )

    report = result["report"]
    assert report["status"] == "RESEARCH_MODEL_FIT_AND_FROZEN_FOR_FUTURE_DIAGNOSTIC"
    assert report["dataset_contract"]["market_count"] == 6
    assert report["dataset_contract"]["decision_count"] == 6
    assert report["dataset_contract"]["action_row_count"] == 30
    assert report["feature_leakage_audit"]["sha256"] == _sha256(
        Path(report["feature_leakage_audit"]["path"])
    )
    assert report["model_contract"]["model_sha256"] == _sha256(result["model_path"])
    assert report["training_metric_used_for_model_selection"] is False
    assert report["validation_evaluation_attempted"] is False
    assert report["future_unseen_evaluation_attempted"] is False
    assert report["research_artifact_frozen"] is True
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False

    predictions, prediction_report = predict_frozen_pnl_aligned_action_values(
        model_dir=result["run_dir"],
        decision_rows=[_source_row(10, include_targets=False)],
    )
    assert prediction_report["status"] == "OUTCOME_BLIND_PREDICTION_COMPLETE"
    assert prediction_report["prediction_attempted"] is True
    assert prediction_report["complete_5_action_prediction_grid"] is True
    assert len(predictions) == 5
    assert {row["action"] for row in predictions} == set(ACTIONS)
    assert all(row["target_used_as_decision_input"] is False for row in predictions)
    assert all(row["source_o_score_mutated"] is False for row in predictions)
    assert all(row["source_ranking_mutated"] is False for row in predictions)
    assert prediction_report["source_model_candidate_eligible"] is False
    assert prediction_report["promotion_evidence_eligible"] is False
    assert prediction_report["v8_execution_handoff_allowed"] is False

    shadow_rows, shadow_report = run_pnl_aligned_action_value_outcome_blind_shadow(
        model_dir=result["run_dir"],
        decision_rows=[_source_row(10, include_targets=False)],
    )
    assert shadow_report["status"] == "OUTCOME_BLIND_SHADOW_EXECUTION_COMPLETE"
    assert shadow_report["decision_count"] == 1
    assert len(shadow_rows) == 1
    assert len(shadow_rows[0]["full_5_action_model_ranking"]) == 5
    assert shadow_rows[0]["outcome_fields_used"] is False
    assert shadow_rows[0]["realized_pnl_used"] is False
    assert shadow_rows[0]["source_o_score_mutated"] is False
    assert shadow_rows[0]["source_ranking_mutated"] is False
    assert shadow_report["future_unseen_outcome_reconciliation_required"] is True
    assert shadow_report["source_model_candidate_eligible"] is False
    assert shadow_report["promotion_evidence_eligible"] is False
    assert shadow_report["v8_execution_handoff_allowed"] is False


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _source_row(index: int, *, include_targets: bool) -> dict:
    decision_ts = 2_000_000 + index * 300_000
    market_id = f"market-{index}"
    ranking = [
        _ranking("BUY_UP_HOLD_TO_SETTLEMENT", 1, 0.80, 0.62),
        _ranking("BUY_UP_SELL_BEFORE_CLOSE", 2, 0.70, 0.62),
        _ranking("NO_TRADE", 3, 0.50, 0.0),
        _ranking("BUY_DOWN_SELL_BEFORE_CLOSE", 4, 0.30, 0.39),
        _ranking("BUY_DOWN_HOLD_TO_SETTLEMENT", 5, 0.20, 0.39),
    ]
    row = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "market_close_ts": decision_ts + 240_000,
        "max_input_ts": decision_ts - 100,
        "source_run_id": f"run-{index}",
        "row_identity": f"row-{index}",
        "decision_time_features": {
            "reference_price_to_beat_distance_at_decision": 0.001,
            "chainlink_momentum_60s": 0.0005,
            "chainlink_realized_volatility_120s": 0.0002,
            "cumulative_market_exposure_before_entry": 0.0,
            "same_side_reentry": 0.0,
            "side_flip": 0.0,
        },
        "execution_handoff_context": {
            "decision_group_id": f"group-{index}",
            "market_id": market_id,
            "decision_ts": decision_ts,
            "selected_action": "BUY_UP_HOLD_TO_SETTLEMENT",
            "selected_side": "UP",
            "selected_action_family": "HOLD_TO_SETTLEMENT",
            "full_5_action_ranking": ranking,
            "corrected_model_score": 0.8,
            "raw_model_score": 1.2,
            "high_score_flag": True,
            "p_up": 0.62,
            "p_down": 0.38,
            "p_up_action_disagreement": False,
            "microstructure_snapshot": ranking[0]["microstructure_snapshot"],
            "reference_price_feature_provenance": {"provenance_valid": True},
            "decision_time_feature_max_input_ts": decision_ts - 100,
        },
    }
    if include_targets:
        row.update(
            {
                "evaluation_target_net_pnl_per_contract_by_action": {
                    "BUY_UP_HOLD_TO_SETTLEMENT": 0.30 if index % 2 == 0 else -0.60,
                    "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.40 if index % 2 == 0 else 0.20,
                    "BUY_UP_SELL_BEFORE_CLOSE": 0.08,
                    "BUY_DOWN_SELL_BEFORE_CLOSE": -0.04,
                    "NO_TRADE": 0.0,
                },
                "target_provenance": {"resolved_outcome": "UP"},
            }
        )
    return row


def _ranking(action: str, rank: int, score: float, entry_ask: float) -> dict:
    no_trade = action == "NO_TRADE"
    side = "UP" if "_UP_" in action else "DOWN" if "_DOWN_" in action else "NONE"
    family = (
        "HOLD_TO_SETTLEMENT"
        if action.endswith("HOLD_TO_SETTLEMENT")
        else "SELL_BEFORE_CLOSE"
        if action.endswith("SELL_BEFORE_CLOSE")
        else "NO_TRADE"
    )
    return {
        "selected_action": action,
        "selected_side": side,
        "selected_action_family": family,
        "canonical_rank": rank,
        "rank": rank,
        "corrected_model_score": score,
        "raw_model_score": score * 2.0,
        "high_score_flag": rank <= 2,
        "microstructure_snapshot": {
            "entry_ask": 0.0 if no_trade else entry_ask,
            "spread_bps": 0.0 if no_trade else 100.0,
            "book_staleness_ms": 0.0 if no_trade else 100.0,
            "queue_fill_proxy": 0.0 if no_trade else 0.9,
            "time_to_close_seconds": 0.0 if no_trade else 180.0,
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
