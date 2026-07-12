from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_estimand_reformulation import (
    EstimandReformulationConfig,
    develop_probability_candidates,
    freeze_and_evaluate_validation_round,
    initialize_estimand_reformulation_goal,
)


def test_estimand_initialization_freezes_full_rows_and_hts_scope(
    tmp_path: Path,
) -> None:
    rows_path = _write_fixture_rows(tmp_path, row_count=120, market_count=40)
    prior_bundle = _prior_bundle(tmp_path)
    result = initialize_estimand_reformulation_goal(
        EstimandReformulationConfig(
            run_id="estimand-test",
            output_dir=tmp_path / "runs",
            repository_root=Path.cwd(),
            prior_blocked_bundle_dir=prior_bundle,
            inspected_rows_path=rows_path,
            created_at="2026-07-12T07:00:00Z",
        )
    )

    goal_dir = result["goal_dir"]
    config = _json(goal_dir / "initial_goal_configuration.json")
    estimand = _json(goal_dir / "estimand_protocol.json")
    quality = _json(goal_dir / "development_corpus_quality_report.json")
    assert result["candidate_scope"] == "HTS_ONLY"
    assert config["sbc_scope_support"]["full_scope_gate_passed"] is False
    assert estimand["model_output_semantics"] == "selected_side_win_probability"
    assert estimand["execution_cost_contract"][
        "execution_cost_subtracted_exactly_once"
    ] is True
    assert quality["source_artifact_hashes_verified"] is True
    immutable = goal_dir / "immutable_development_rows.jsonl"
    expected = immutable.with_suffix(".sha256").read_text().split()[0]
    assert _sha256(immutable) == expected
    rows = _jsonl(immutable)
    assert all(row["target_outcome_available_only_post_resolution"] for row in rows)
    assert all(row["promotion_evidence_eligible"] is False for row in rows)

    original = immutable.read_text(encoding="utf-8")
    immutable.write_text(original.replace('"lineage":"development"', '"lineage":"changed"', 1), encoding="utf-8")
    assert _sha256(immutable) != expected


def test_candidate_development_keeps_required_baselines_non_selectable(
    tmp_path: Path,
) -> None:
    rows_path = _write_fixture_rows(tmp_path, row_count=120, market_count=40)
    result = initialize_estimand_reformulation_goal(
        EstimandReformulationConfig(
            run_id="candidate-test",
            output_dir=tmp_path / "runs",
            repository_root=Path.cwd(),
            prior_blocked_bundle_dir=_prior_bundle(tmp_path),
            inspected_rows_path=rows_path,
            created_at="2026-07-12T07:00:00Z",
        )
    )
    development = develop_probability_candidates(result["goal_dir"])
    report = _json(result["goal_dir"] / "candidate_development_report.json")
    order = report["validation_round_candidate_order"]
    assert len(report["candidate_reports"]) == 8
    assert len(order) == 3
    assert "raw_selected_side_market_probability" not in order
    assert "constant_development_win_rate" not in order
    assert development["selected_candidate_name"] == order[0]
    assert report["fresh_validation_used_for_selection"] is False


def test_validation_overlap_fails_closed_before_prediction(tmp_path: Path) -> None:
    rows_path = _write_fixture_rows(tmp_path, row_count=120, market_count=40)
    result = initialize_estimand_reformulation_goal(
        EstimandReformulationConfig(
            run_id="overlap-test",
            output_dir=tmp_path / "runs",
            repository_root=Path.cwd(),
            prior_blocked_bundle_dir=_prior_bundle(tmp_path),
            inspected_rows_path=rows_path,
            created_at="2026-07-12T07:00:00Z",
        )
    )
    develop_probability_candidates(result["goal_dir"])
    evaluation = freeze_and_evaluate_validation_round(
        result["goal_dir"],
        round_number=1,
        fresh_rows_path=rows_path,
    )
    round_dir = result["goal_dir"] / "round_1"
    leakage = _json(round_dir / "round_1_leakage_report.json")
    assert evaluation["split_gate_passed"] is False
    assert evaluation["evaluated"] is False
    assert leakage["overlap"]["market_ids"]
    assert leakage["leakage_report_passed"] is False
    assert not (round_dir / "round_1_evaluation_started.json").exists()


def test_invalid_sbc_full_scope_fails_until_separate_estimand_exists(
    tmp_path: Path,
) -> None:
    rows_path = _write_fixture_rows(
        tmp_path,
        row_count=120,
        market_count=40,
        sbc_row_count=30,
        sbc_market_count=10,
    )
    with pytest.raises(ValueError, match="SBC estimand"):
        initialize_estimand_reformulation_goal(
            EstimandReformulationConfig(
                run_id="full-scope-test",
                output_dir=tmp_path / "runs",
                repository_root=Path.cwd(),
                prior_blocked_bundle_dir=_prior_bundle(tmp_path),
                inspected_rows_path=rows_path,
                created_at="2026-07-12T07:00:00Z",
            )
        )


def test_disjoint_validation_is_evaluated_exactly_once(tmp_path: Path) -> None:
    development_path = _write_fixture_rows(
        tmp_path / "development", row_count=120, market_count=40
    )
    result = initialize_estimand_reformulation_goal(
        EstimandReformulationConfig(
            run_id="exactly-once-test",
            output_dir=tmp_path / "runs",
            repository_root=Path.cwd(),
            prior_blocked_bundle_dir=_prior_bundle(tmp_path),
            inspected_rows_path=development_path,
            created_at="2026-07-12T07:00:00Z",
            bootstrap_samples=20,
        )
    )
    develop_probability_candidates(result["goal_dir"])
    fresh_path = _write_fixture_rows(
        tmp_path / "fresh",
        row_count=80,
        market_count=25,
        sbc_row_count=0,
        market_prefix="fresh-market",
        condition_prefix="fresh-condition",
        run_prefix="fresh-run",
        decision_start=1_790_000_000_000,
    )
    evaluation = freeze_and_evaluate_validation_round(
        result["goal_dir"],
        round_number=1,
        fresh_rows_path=fresh_path,
    )
    report = _json(
        result["goal_dir"]
        / "round_1"
        / "round_1_fresh_validation_report.json"
    )
    assert evaluation["split_gate_passed"] is True
    assert evaluation["evaluated"] is True
    assert report["evaluation_attempt_number"] == 1
    assert report["uses_validation_labels_for_tuning"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_and_evaluate_validation_round(
            result["goal_dir"],
            round_number=1,
            fresh_rows_path=fresh_path,
        )


def test_additional_excluded_run_fails_closed_before_prediction(
    tmp_path: Path,
) -> None:
    development_path = _write_fixture_rows(
        tmp_path / "development-excluded", row_count=120, market_count=40
    )
    result = initialize_estimand_reformulation_goal(
        EstimandReformulationConfig(
            run_id="excluded-run-test",
            output_dir=tmp_path / "runs",
            repository_root=Path.cwd(),
            prior_blocked_bundle_dir=_prior_bundle(tmp_path),
            inspected_rows_path=development_path,
            created_at="2026-07-12T07:00:00Z",
            additional_excluded_run_ids=("excluded-fresh-run-0",),
            bootstrap_samples=20,
        )
    )
    develop_probability_candidates(result["goal_dir"])
    fresh_path = _write_fixture_rows(
        tmp_path / "fresh-excluded",
        row_count=80,
        market_count=25,
        sbc_row_count=0,
        market_prefix="excluded-fresh-market",
        condition_prefix="excluded-fresh-condition",
        run_prefix="excluded-fresh-run",
        decision_start=1_790_000_000_000,
    )
    evaluation = freeze_and_evaluate_validation_round(
        result["goal_dir"], round_number=1, fresh_rows_path=fresh_path
    )
    leakage = _json(
        result["goal_dir"] / "round_1" / "round_1_leakage_report.json"
    )
    assert evaluation["evaluated"] is False
    assert leakage["overlap"]["excluded_source_run_ids"] == [
        "excluded-fresh-run-0"
    ]
    assert not (
        result["goal_dir"] / "round_1" / "round_1_evaluation_started.json"
    ).exists()


def _write_fixture_rows(
    tmp_path: Path,
    *,
    row_count: int,
    market_count: int,
    sbc_row_count: int = 8,
    sbc_market_count: int = 8,
    market_prefix: str = "market",
    condition_prefix: str = "condition",
    run_prefix: str = "run",
    decision_start: int = 1_780_000_000_000,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jsonl"
    source.write_text('{"source":"read-only-settlement"}\n', encoding="utf-8")
    source_hash = _sha256(source)
    rows = []
    for index in range(row_count):
        is_sbc = index < sbc_row_count
        market_index = (
            index % sbc_market_count if is_sbc else index % market_count
        )
        side = "UP" if index % 2 == 0 else "DOWN"
        outcome = "UP" if market_index % 3 else "DOWN"
        family = "SELL_BEFORE_CLOSE" if is_sbc else "HOLD_TO_SETTLEMENT"
        action = f"BUY_{side}_{family}"
        decision_ts = decision_start + index * 1000
        rows.append(
            {
                "row_identity": hashlib.sha256(
                    f"{run_prefix}-row-{index}".encode()
                ).hexdigest(),
                "market_id": f"{market_prefix}-{market_index:03d}",
                "condition_id": f"{condition_prefix}-{market_index:03d}",
                "source_run_id": f"{run_prefix}-{index // 40}",
                "source_fill_id": f"fill-{index}",
                "source_intent_id": f"intent-{index}",
                "decision_ts": decision_ts,
                "max_input_ts": decision_ts,
                "market_close_ts": decision_ts + 300_000,
                "selected_side": side,
                "selected_action": action,
                "action_family": family,
                "decision_time_features": {
                    "selected_side_probability": 0.58 if side == outcome else 0.42,
                    "execution_price": 0.55,
                    "spread_bps": 100.0,
                    "queue_fill_proxy": 0.9,
                    "book_staleness_ms": 20.0,
                    "time_to_close_seconds": 250.0,
                    "canonical_o_action_score": 0.8,
                    "action_score_margin": 0.12,
                },
                "source_lineage": {
                    "fill_artifact_path": str(source),
                    "fill_artifact_sha256": source_hash,
                },
                "target_provenance": {
                    "resolved_outcome": outcome,
                    "resolution_status": "resolved",
                    "source_type": "polymarket_clob_read_only_settlement",
                    "source_artifact_path": str(source),
                    "source_artifact_sha256": source_hash,
                },
            }
        )
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _prior_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "prior-bundle"
    bundle.mkdir(exist_ok=True)
    manifest = bundle / "pre_promotion_readiness_manifest.json"
    manifest.write_text(
        json.dumps({"final_state": "PRE_PROMOTION_BLOCKED"}) + "\n",
        encoding="utf-8",
    )
    return bundle


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
