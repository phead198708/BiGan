from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_action_value import (
    HierarchicalActionValueFitConfig,
    fit_historical_hierarchical_action_value,
    predict_frozen_hierarchical_action_values,
    validate_hierarchical_action_value_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_hierarchical_action_value_v2.json"
)
SOURCE_PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pnl_aligned_action_value_v1.json"
)


def test_hierarchical_protocol_is_frozen_and_fail_closed() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_hierarchical_action_value_protocol(protocol)
    assert protocol["candidate_name"] == (
        "historical_fit_only_hierarchical_action_value_v2"
    )
    assert protocol["uses_historical_validation_labels_for_tuning"] is False
    assert protocol["uses_excluded_future_evidence_for_tuning"] is False
    assert protocol["frozen_execution_contract"]["entry_edge_threshold"] == 0.02

    drifted = json.loads(json.dumps(protocol))
    drifted["uses_historical_validation_labels_for_tuning"] = True
    with pytest.raises(ValueError, match="no_validation_tuning"):
        validate_hierarchical_action_value_protocol(drifted)


def test_fit_builds_disjoint_splits_family_heads_and_exclusion_registry(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "historical_rows.jsonl"
    _write_jsonl(rows_path, [_source_row(index) for index in range(35)])
    corpus_manifest = tmp_path / "historical_manifest.json"
    _write_json(
        corpus_manifest,
        {"development_rows": {"path": str(rows_path), "sha256": _sha256(rows_path)}},
    )
    future_rows_path = tmp_path / "future_decision_rows.jsonl"
    future_rows = [
        {
            "market_id": f"future-market-{index}",
            "decision_ts": 50_000_000 + index * 300_000,
            "row_identity": f"future-row-{index}",
        }
        for index in range(5)
    ]
    _write_jsonl(future_rows_path, future_rows)
    future_artifact = tmp_path / "future_evaluation_manifest.json"
    _write_json(future_artifact, {"run_id": "future-holdout", "gate_passed": False})

    result = fit_historical_hierarchical_action_value(
        HierarchicalActionValueFitConfig(
            run_id="hierarchical-fit",
            output_dir=tmp_path / "runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            source_action_protocol_path=SOURCE_PROTOCOL_PATH,
            expected_source_action_protocol_sha256=_sha256(SOURCE_PROTOCOL_PATH),
            historical_corpus_manifest_path=corpus_manifest,
            excluded_future_decision_rows_path=future_rows_path,
            expected_excluded_future_decision_rows_sha256=_sha256(future_rows_path),
            excluded_future_artifact_pins=((future_artifact, _sha256(future_artifact)),),
        )
    )

    exclusion = json.loads(result["exclusion_registry_path"].read_text())
    split = json.loads(result["split_manifest_path"].read_text())
    freeze = result["freeze_manifest"]
    validation = result["validation_report"]

    assert exclusion["excluded_future_market_count"] == 5
    assert exclusion["historical_future_market_overlap_count"] == 0
    assert exclusion["excluded_future_outcome_values_loaded"] is False
    assert exclusion["excluded_future_pnl_values_loaded"] is False
    assert exclusion["excluded_future_evidence_used_for_fitting_or_tuning"] is False
    assert split["split_summary"]["splits"]["historical_train"]["market_count"] == 21
    assert split["split_summary"]["splits"]["historical_calibration"]["market_count"] == 7
    assert split["split_summary"]["splits"]["historical_validation"]["market_count"] == 7
    assert split["market_overlap_count"] == 0
    assert split["chronology_validation_passed"] is True
    assert set(freeze["models"]) == {
        "HOLD_TO_SETTLEMENT",
        "SELL_BEFORE_CLOSE",
    }
    assert freeze["uses_historical_validation_labels_for_tuning"] is False
    assert freeze["uses_excluded_future_evidence_for_tuning"] is False
    assert validation["historical_validation_labels_used_for_report_only"] is True
    assert validation["historical_validation_labels_used_for_tuning"] is False
    assert validation["source_model_candidate_eligible"] is False
    assert validation["freeze_ready"] is False
    assert validation["promotion_evidence_eligible"] is False
    assert validation["v8_execution_handoff_allowed"] is False
    assert validation["#134_resume_allowed"] is False
    assert validation["#146_start_allowed"] is False
    assert validation["paper_only"] is True
    assert validation["capital_at_risk"] is False

    if validation["historical_validation_gate_passed"] is False:
        predictions, report = predict_frozen_hierarchical_action_values(
            model_dir=result["run_dir"],
            decision_rows=[_source_row(100, include_targets=False)],
        )
        assert predictions == []
        assert report["status"] == "BLOCKED_FAIL_CLOSED_BEFORE_PREDICTION"
        assert report["prediction_attempted"] is False


def test_excluded_future_registry_source_rejects_outcome_fields(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "historical_rows.jsonl"
    _write_jsonl(rows_path, [_source_row(index) for index in range(35)])
    corpus_manifest = tmp_path / "historical_manifest.json"
    _write_json(
        corpus_manifest,
        {"development_rows": {"path": str(rows_path), "sha256": _sha256(rows_path)}},
    )
    future_rows_path = tmp_path / "future_decision_rows.jsonl"
    _write_jsonl(
        future_rows_path,
        [{"market_id": "future-market", "decision_ts": 50_000_000, "resolved_outcome": "UP"}],
    )
    future_artifact = tmp_path / "future_manifest.json"
    _write_json(future_artifact, {"run_id": "future"})

    with pytest.raises(ValueError, match="contains outcome fields"):
        fit_historical_hierarchical_action_value(
            HierarchicalActionValueFitConfig(
                run_id="leaky-future-registry",
                output_dir=tmp_path / "runs",
                protocol_path=PROTOCOL_PATH,
                expected_protocol_sha256=_sha256(PROTOCOL_PATH),
                source_action_protocol_path=SOURCE_PROTOCOL_PATH,
                expected_source_action_protocol_sha256=_sha256(SOURCE_PROTOCOL_PATH),
                historical_corpus_manifest_path=corpus_manifest,
                excluded_future_decision_rows_path=future_rows_path,
                expected_excluded_future_decision_rows_sha256=_sha256(future_rows_path),
                excluded_future_artifact_pins=(
                    (future_artifact, _sha256(future_artifact)),
                ),
            )
        )


def _source_row(index: int, *, include_targets: bool = True) -> dict:
    decision_ts = 2_000_000 + index * 300_000
    market_id = f"market-{index}"
    up_probability = 0.62 if index % 2 == 0 else 0.38
    down_probability = 1.0 - up_probability
    ranking = [
        _ranking("BUY_UP_HOLD_TO_SETTLEMENT", 1, 0.80, up_probability),
        _ranking("BUY_UP_SELL_BEFORE_CLOSE", 2, 0.70, up_probability),
        _ranking("NO_TRADE", 3, 0.50, 0.0),
        _ranking("BUY_DOWN_SELL_BEFORE_CLOSE", 4, 0.30, down_probability),
        _ranking("BUY_DOWN_HOLD_TO_SETTLEMENT", 5, 0.20, down_probability),
    ]
    row = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "market_close_ts": decision_ts + 240_000,
        "max_input_ts": decision_ts - 100,
        "source_run_id": f"run-{index}",
        "row_identity": f"row-{index}",
        "decision_time_features": {
            "reference_price_to_beat_distance_at_decision": 0.001 if index % 2 == 0 else -0.001,
            "chainlink_momentum_60s": 0.0005 if index % 2 == 0 else -0.0005,
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
            "p_up": up_probability,
            "p_down": down_probability,
            "p_up_action_disagreement": False,
            "microstructure_snapshot": ranking[0]["microstructure_snapshot"],
            "reference_price_feature_provenance": {"provenance_valid": True},
            "decision_time_feature_max_input_ts": decision_ts - 100,
        },
    }
    if include_targets:
        up_win = index % 2 == 0
        row["evaluation_target_net_pnl_per_contract_by_action"] = {
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.25 if up_win else -0.55,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.55 if up_win else 0.25,
            "BUY_UP_SELL_BEFORE_CLOSE": 0.10 if up_win else -0.08,
            "BUY_DOWN_SELL_BEFORE_CLOSE": -0.08 if up_win else 0.10,
            "NO_TRADE": 0.0,
        }
        row["target_provenance"] = {"resolved_outcome": "UP" if up_win else "DOWN"}
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
            "spread_bps": 0.0 if no_trade else 80.0,
            "book_staleness_ms": 0.0 if no_trade else 100.0,
            "queue_fill_proxy": 0.0 if no_trade else 0.95,
            "time_to_close_seconds": 0.0 if no_trade else 180.0,
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
