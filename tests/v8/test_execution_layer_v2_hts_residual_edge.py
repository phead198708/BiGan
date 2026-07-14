from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_estimand_reformulation import (
    safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_edge import (
    HTSResidualEdgePowerConfig,
    fit_residual_offset_contract,
    predict_residual_offset_probability,
    run_hts_residual_edge_power_analysis,
)


def test_residual_edge_analysis_excludes_all_seen_rows_and_stays_fail_closed(
    tmp_path: Path,
) -> None:
    source_goal = _source_goal(tmp_path)
    source_manifest = source_goal / "pre_promotion_readiness_manifest.json"
    source_manifest_hash_before = _sha256(source_manifest)

    result = run_hts_residual_edge_power_analysis(
        _config(tmp_path, source_goal, run_id="residual-edge")
    )

    analysis_dir = Path(result["analysis_dir"])
    manifest = _read_json(analysis_dir / "hts_residual_edge_manifest.json")
    development = _read_json(
        analysis_dir / "hts_incremental_edge_development_manifest.json"
    )
    diagnostic = _read_json(
        analysis_dir / "hts_incremental_edge_diagnostic_report.json"
    )
    power = _read_json(analysis_dir / "hts_market_level_power_report.json")
    protocol = _read_json(
        analysis_dir / "hts_residual_offset_candidate_protocol.json"
    )

    assert development["combined_row_count"] == 60
    assert development["combined_market_count"] == 60
    assert len(development["excluded_row_identities"]) == 60
    assert development["all_previously_inspected_rows_excluded_from_future_confirmatory"]
    assert development["future_confirmatory_validation_eligible"] is False
    assert diagnostic["market_probability_fixed_offset_verified"] is True
    assert diagnostic["selected_candidate_contract"][
        "market_probability_offset_coefficient"
    ] == 1.0
    assert diagnostic["selected_candidate_contract"][
        "market_probability_offset_trainable"
    ] is False
    assert protocol["protocol_frozen_before_any_new_prospective_data"] is True
    assert protocol["future_confirmatory_validation_start_allowed"] is False
    assert power["fresh_confirmatory_validation_start_allowed"] is False
    assert manifest["new_confirmatory_validation_started"] is False
    assert manifest["prospective_collection_started"] is False
    assert manifest["fresh_confirmatory_validation_start_allowed"] is False
    _assert_safety(manifest)
    _assert_safety(diagnostic)
    _assert_safety(power)
    assert _sha256(source_manifest) == source_manifest_hash_before


def test_residual_edge_analysis_fails_closed_on_causality_violation(
    tmp_path: Path,
) -> None:
    source_goal = _source_goal(tmp_path)
    rows_path = source_goal / "round_3" / "round_3_unseen_validation_rows.jsonl"
    rows = _read_jsonl(rows_path)
    rows[0]["max_input_ts"] = rows[0]["decision_ts"] + 1.0
    _write_jsonl(rows_path, rows)
    _write_sha_descriptor(rows_path)

    with pytest.raises(ValueError, match="invalid source rows"):
        run_hts_residual_edge_power_analysis(
            _config(tmp_path, source_goal, run_id="causality-fail")
        )


def test_residual_edge_analysis_requires_exhausted_three_round_source(
    tmp_path: Path,
) -> None:
    source_goal = _source_goal(tmp_path)
    report_path = source_goal / "pre_promotion_readiness_report.json"
    report = _read_json(report_path)
    report["validation_round_history"] = report["validation_round_history"][:2]
    _write_json(report_path, report)
    manifest_path = source_goal / "pre_promotion_readiness_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["artifacts"][0]["sha256"] = _sha256(report_path)
    _write_json(manifest_path, manifest)
    _write_sha_descriptor(manifest_path)

    with pytest.raises(ValueError, match="three failed rounds"):
        run_hts_residual_edge_power_analysis(
            _config(tmp_path, source_goal, run_id="round-history-fail")
        )


def test_zero_residual_contract_falls_back_to_market_probability() -> None:
    rows = [_row(run=0, market=0)]
    rows[0]["selected_side_win_target"] = 1
    spec = {
        "candidate_name": "test_offset",
        "feature_names": ["canonical_o_action_score", "action_score_margin"],
        "regularization": 25.0,
        "maximum_absolute_residual_coefficient": 3.0,
        "probability_bounds": [0.01, 0.99],
    }
    contract = fit_residual_offset_contract(rows, spec)
    contract["residual_parameters"] = [0.0] * len(
        contract["residual_parameters"]
    )

    prediction = predict_residual_offset_probability(rows[0], contract)

    assert prediction == pytest.approx(
        rows[0]["decision_time_features"]["selected_side_probability"]
    )
    assert contract["market_probability_offset_coefficient"] == 1.0
    assert contract["market_probability_offset_trainable"] is False


def _config(
    tmp_path: Path, source_goal: Path, *, run_id: str
) -> HTSResidualEdgePowerConfig:
    return HTSResidualEdgePowerConfig(
        run_id=run_id,
        output_dir=tmp_path / "runs",
        repository_root=Path.cwd(),
        source_estimand_goal_dir=source_goal,
        created_at="2026-07-14T00:00:00Z",
        bootstrap_samples=50,
        minimum_training_runs=2,
        minimum_prospective_market_count=25,
    )


def _source_goal(tmp_path: Path) -> Path:
    goal = tmp_path / "source_goal"
    goal.mkdir()
    base_rows = [_row(run=run, market=market) for run in range(3) for market in range(10)]
    base_path = goal / "immutable_development_rows.jsonl"
    _write_jsonl(base_path, base_rows)
    _write_sha_descriptor(base_path)
    for round_number in range(1, 4):
        run = round_number + 2
        round_dir = goal / f"round_{round_number}"
        round_dir.mkdir()
        rows = [_row(run=run, market=market) for market in range(10)]
        path = round_dir / f"round_{round_number}_unseen_validation_rows.jsonl"
        _write_jsonl(path, rows)
        _write_sha_descriptor(path)
    report = {
        "final_state": "PRE_PROMOTION_BLOCKED",
        "blocking_reason_codes": [
            "all_predeclared_candidates_exhausted",
            "all_three_validation_rounds_failed",
            "irreducible_statistical_relative_improvement_and_bootstrap_blocker",
            "no_validation_round_passed_all_frozen_gates",
        ],
        "validation_round_history": [
            {
                "round_number": round_number,
                "candidate_name": f"candidate-{round_number}",
                "all_confirmatory_gates_passed": False,
                "blocking_reason_codes": ["confirmatory_gate_failed"],
            }
            for round_number in range(1, 4)
        ],
        **safety_fields(),
    }
    _write_json(goal / "pre_promotion_readiness_report.json", report)
    manifest_path = goal / "pre_promotion_readiness_manifest.json"
    _write_json(
        manifest_path,
        {
            "final_state": "PRE_PROMOTION_BLOCKED",
            "artifacts": [
                {
                    "relative_path": "pre_promotion_readiness_report.json",
                    "sha256": _sha256(goal / "pre_promotion_readiness_report.json"),
                }
            ],
        },
    )
    _write_sha_descriptor(manifest_path)
    return goal


def _row(*, run: int, market: int) -> dict[str, object]:
    decision_ts = float(1_000_000 + run * 10_000 + market * 10)
    side = "UP" if market % 2 == 0 else "DOWN"
    wins = (market + run) % 3 != 0
    outcome = side if wins else ("DOWN" if side == "UP" else "UP")
    signal = 1.0 if wins else -1.0
    probability = 0.56 if side == "UP" else 0.54
    execution_price = probability + 0.01
    return {
        "row_identity": f"run-{run}-market-{market}",
        "market_id": f"market-{run}-{market}",
        "condition_id": f"condition-{run}-{market}",
        "market_slug": f"btc-updown-5m-{run}-{market}",
        "source_run_id": f"run-{run}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "selected_side": side,
        "selected_action": f"BUY_{side}_HOLD_TO_SETTLEMENT",
        "action_family": "HOLD_TO_SETTLEMENT",
        "target_provenance": {
            "resolved_outcome": outcome,
            "target_available_ts": decision_ts + 300.0,
        },
        "decision_time_features": {
            "selected_side_probability": probability,
            "execution_price": execution_price,
            "selected_side_probability_minus_execution_price": (
                probability - execution_price
            ),
            "canonical_o_action_score": signal,
            "action_score_margin": signal * 0.1,
            "btc_momentum": signal * 0.001,
            "reference_price_to_beat_distance_at_decision": signal * 0.001,
            "spread_bps": 100.0,
            "queue_fill_proxy": 0.9,
            "book_staleness_ms": 10.0,
            "time_to_close_seconds": 240.0,
            "cumulative_market_exposure_before_entry": 0.0,
            "same_side_reentry": 0.0,
            "side_flip": 0.0,
        },
    }


def _assert_safety(payload: dict[str, object]) -> None:
    assert payload["diagnostic_only"] is True
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
    assert payload["source_model_candidate_eligible"] is False
    assert payload["freeze_ready"] is False
    assert payload["promotion_evidence_eligible"] is False
    assert payload["v8_execution_handoff_allowed"] is False
    assert payload["#134_resume_allowed"] is False
    assert payload["#146_start_allowed"] is False


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_sha_descriptor(path: Path) -> None:
    path.with_suffix(".sha256").write_text(
        f"{_sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
