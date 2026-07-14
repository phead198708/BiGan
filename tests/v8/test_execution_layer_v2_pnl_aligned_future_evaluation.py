from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    PnLAlignedFutureCollectionHandoffConfig,
    PnLAlignedFutureDecisionInputConfig,
    PnLAlignedFutureSettlementTargetConfig,
    build_pnl_aligned_future_collection_handoff,
    build_pnl_aligned_future_outcome_blind_decision_inputs,
    build_pnl_aligned_future_settled_evaluation_targets,
    evaluate_pnl_aligned_future_accepted_bets,
    load_pnl_aligned_future_collection_handoff_source_dirs,
    materialize_pnl_aligned_future_action_value_predictions,
    validate_pnl_aligned_future_evaluation_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/execution_layer_v2_pnl_aligned_future_evaluation_v1.json"
)
ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "NO_TRADE",
)
UNLOCK_DIR = Path("examples/v8/polymarket_runs/o-v8-paper-candidate-unlock-20260703T073000Z")


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


def test_action_value_prediction_materialization_is_complete_and_deterministic() -> None:
    shadow = _candidate_shadow_row()
    first = materialize_pnl_aligned_future_action_value_predictions([shadow])
    second = materialize_pnl_aligned_future_action_value_predictions([shadow])

    assert first == second
    assert len(first) == 5
    assert {row["action"] for row in first} == set(ACTIONS)
    assert {row["model_rank"] for row in first} == {1, 2, 3, 4, 5}
    assert sum(row["selected_by_frozen_model"] for row in first) == 1
    assert all(row["outcome_fields_used"] is False for row in first)
    assert all(row["model_rescored_for_artifact"] is False for row in first)
    assert all(len(row["prediction_row_sha256"]) == 64 for row in first)


@pytest.mark.parametrize("mutation", ["incomplete", "duplicate_rank", "selected_mismatch"])
def test_action_value_prediction_materialization_rejects_invalid_grid(mutation: str) -> None:
    shadow = _candidate_shadow_row()
    if mutation == "incomplete":
        shadow["full_5_action_model_ranking"].pop()
    elif mutation == "duplicate_rank":
        shadow["full_5_action_model_ranking"][-1]["rank"] = 1
    else:
        shadow["selected_action"] = "BUY_DOWN_HOLD_TO_SETTLEMENT"

    with pytest.raises(ValueError, match="ranking is incomplete or inconsistent"):
        materialize_pnl_aligned_future_action_value_predictions([shadow])


def test_action_value_prediction_materialization_rejects_outcome_fields() -> None:
    shadow = _candidate_shadow_row()
    shadow["full_5_action_model_ranking"][0]["resolved_outcome"] = "UP"
    with pytest.raises(ValueError, match="forbidden outcome fields"):
        materialize_pnl_aligned_future_action_value_predictions([shadow])


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
    assert report["candidate_policy_metrics"]["settled_net_pnl_sum"] == pytest.approx(0.6)
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
    historical_path.write_text(json.dumps({"market_id": "overlap", "decision_ts": 100}) + "\n")
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
        "evaluation_target_net_pnl_per_contract_by_action": dict.fromkeys(ACTIONS, 0.0),
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


def test_future_decision_input_build_is_feature_only_without_outcome_artifacts(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="future-market",
        market_start_ts=2_000_000,
    )
    assert not (corpus_dir / "polymarket_label_rows.jsonl").exists()
    assert not (corpus_dir / "polymarket_resolution_events.jsonl").exists()
    (corpus_dir / "polymarket_label_rows.jsonl").write_text(
        "this outcome-bearing decoy must not be opened\n"
    )
    (corpus_dir / "polymarket_resolution_events.jsonl").write_text(
        "this resolution decoy must not be opened\n"
    )

    result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="future-decision-input",
            output_dir=tmp_path / "runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(corpus_dir,),
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )

    report = result["report"]
    assert report["status"] == "OUTCOME_BLIND_FUTURE_DECISION_INPUT_READY"
    assert report["source_unique_market_count"] == 1
    assert report["outcome_blind_decision_row_count"] == 1
    assert report["complete_5_action_ranking_count"] == 1
    assert report["future_outcome_targets_loaded"] is False
    assert report["outcome_reconciliation_started"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    row = result["decision_rows"][0]
    assert row["max_input_ts"] <= row["decision_ts"]
    assert {
        item["selected_action"]
        for item in row["execution_handoff_context"]["full_5_action_ranking"]
    } == set(ACTIONS)
    assert "evaluation_target_net_pnl_per_contract_by_action" not in row
    access = json.loads(result["access_audit_path"].read_text())
    assert access["prohibited_future_outcome_artifact_read_count"] == 0
    assert access["label_rows_required"] is False
    assert access["resolution_events_required"] is False
    assert access["prohibited_future_outcome_artifacts_present_but_not_opened"] == [
        "polymarket_label_rows.jsonl",
        "polymarket_resolution_events.jsonl",
    ]
    assert access["outcome_blind_input_access_passed"] is True


def test_collection_handoff_pins_complete_outcome_blind_batch(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="handoff-market",
        market_start_ts=2_000_000,
    )
    (corpus_dir / "polymarket_label_rows.jsonl").write_text("invalid-label-decoy\n")
    (corpus_dir / "polymarket_resolution_events.jsonl").write_text("invalid-resolution-decoy\n")
    batch_path = _collector_batch_progress(tmp_path, corpus_dirs=[corpus_dir])

    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="collection-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=batch_path,
            expected_batch_progress_sha256=_sha256(batch_path),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )

    assert handoff["report"]["status"] == "OUTCOME_BLIND_COLLECTION_HANDOFF_READY"
    assert handoff["report"]["source_corpus_count"] == 1
    access = json.loads(handoff["access_audit_path"].read_text())
    assert access["prohibited_future_outcome_artifact_read_count"] == 0
    assert len(access["prohibited_future_outcome_artifacts_present_but_not_opened"]) == 2
    decision_input = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="handoff-decision-input",
            output_dir=tmp_path / "decision-runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(corpus_dir.resolve(),),
            paper_candidate_unlock_dir=UNLOCK_DIR,
            collection_handoff_manifest_path=handoff["manifest_path"],
            expected_collection_handoff_manifest_sha256=handoff["manifest_sha256"],
        )
    )
    assert decision_input["report"]["collection_handoff_verified"] is True
    assert decision_input["report"]["status"] == "OUTCOME_BLIND_FUTURE_DECISION_INPUT_READY"


def test_collection_handoff_rejects_different_decision_input_corpus_set(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    pinned_corpus = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="pinned-handoff-market",
        market_start_ts=2_000_000,
    )
    different_corpus = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="different-input-market",
        market_start_ts=2_300_000,
    )
    batch_path = _collector_batch_progress(tmp_path, corpus_dirs=[pinned_corpus])
    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="pinned-collection-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=batch_path,
            expected_batch_progress_sha256=_sha256(batch_path),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )

    with pytest.raises(ValueError, match="differs from collection handoff"):
        build_pnl_aligned_future_outcome_blind_decision_inputs(
            PnLAlignedFutureDecisionInputConfig(
                run_id="wrong-handoff-decision-input",
                output_dir=tmp_path / "decision-runs",
                collection_freeze_manifest_path=freeze_path,
                expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
                source_corpus_dirs=(different_corpus.resolve(),),
                paper_candidate_unlock_dir=UNLOCK_DIR,
                collection_handoff_manifest_path=handoff["manifest_path"],
                expected_collection_handoff_manifest_sha256=handoff["manifest_sha256"],
            )
        )


def test_collection_handoff_fails_closed_for_pending_batch(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="pending-market",
        market_start_ts=2_000_000,
    )
    batch_path = _collector_batch_progress(
        tmp_path,
        corpus_dirs=[corpus_dir],
        pending_indices={1},
    )

    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="pending-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=batch_path,
            expected_batch_progress_sha256=_sha256(batch_path),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )

    assert handoff["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert "collector_pending_resolution_present" in handoff["report"]["blocking_reason_codes"]
    assert "collector_export_count_incomplete" in handoff["report"]["blocking_reason_codes"]


def test_collection_handoff_rejects_duplicate_export_identity(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=2)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="duplicate-export-market",
        market_start_ts=2_000_000,
    )
    batch_path = _collector_batch_progress(
        tmp_path,
        corpus_dirs=[corpus_dir, corpus_dir],
    )

    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="duplicate-export-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=batch_path,
            expected_batch_progress_sha256=_sha256(batch_path),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )

    assert handoff["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert "collector_duplicate_exported_corpus_path" in handoff["report"]["blocking_reason_codes"]
    assert (
        "collection_handoff_unique_market_count_mismatch"
        in handoff["report"]["blocking_reason_codes"]
    )


def test_collection_handoff_rejects_path_escape_and_hash_tamper(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="path-escape-market",
        market_start_ts=2_000_000,
    )
    batch_path = _collector_batch_progress(tmp_path, corpus_dirs=[corpus_dir])
    training_root = tmp_path / "allowed-training-root"
    training_root.mkdir()
    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="path-escape-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=batch_path,
            expected_batch_progress_sha256=_sha256(batch_path),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=training_root,
        )
    )
    assert (
        "collector_exported_corpus_outside_training_root"
        in handoff["report"]["blocking_reason_codes"]
    )

    with batch_path.open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="batch progress SHA-256 mismatch"):
        build_pnl_aligned_future_collection_handoff(
            PnLAlignedFutureCollectionHandoffConfig(
                run_id="batch-hash-tamper",
                output_dir=tmp_path / "handoff-runs",
                batch_progress_path=batch_path,
                expected_batch_progress_sha256="a" * 64,
                collection_freeze_manifest_path=freeze_path,
                expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
                training_corpus_root=tmp_path,
            )
        )


def test_collection_handoff_rejects_corpus_hash_tamper(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="corpus-tamper-market",
        market_start_ts=2_000_000,
    )
    batch_path = _collector_batch_progress(tmp_path, corpus_dirs=[corpus_dir])
    with (corpus_dir / "polymarket_feature_rows.jsonl").open("a") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="normalized artifact hash mismatch"):
        build_pnl_aligned_future_collection_handoff(
            PnLAlignedFutureCollectionHandoffConfig(
                run_id="corpus-hash-tamper",
                output_dir=tmp_path / "handoff-runs",
                batch_progress_path=batch_path,
                expected_batch_progress_sha256=_sha256(batch_path),
                collection_freeze_manifest_path=freeze_path,
                expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
                training_corpus_root=tmp_path,
            )
        )


def test_collection_handoff_selects_28_valid_plus_2_later_replacements(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=30)
    original_corpora = [
        _outcome_blind_phase2_corpus(
            tmp_path,
            market_id=f"original-market-{index:02d}",
            market_start_ts=2_000_000 + index * 300_000,
        )
        for index in range(1, 31)
    ]
    original_batch = _collector_batch_progress(
        tmp_path,
        corpus_dirs=original_corpora,
        blocked_indices={12, 14},
        omit_finalization_indices={12, 14},
    )
    replacement_corpora = [
        _outcome_blind_phase2_corpus(
            tmp_path,
            market_id=f"replacement-market-{index:02d}",
            market_start_ts=11_500_000 + index * 300_000,
        )
        for index in range(1, 3)
    ]
    for corpus_dir in replacement_corpora:
        (corpus_dir / "polymarket_label_rows.jsonl").write_text("outcome-decoy\n")
    replacement_batch = _collector_batch_progress(tmp_path, corpus_dirs=replacement_corpora)

    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="multi-batch-replacement-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=original_batch,
            expected_batch_progress_sha256=_sha256(original_batch),
            additional_batch_progress_paths=(replacement_batch,),
            additional_expected_batch_progress_sha256=(_sha256(replacement_batch),),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )

    assert handoff["report"]["status"] == "OUTCOME_BLIND_COLLECTION_HANDOFF_READY"
    assert handoff["report"]["batch_count"] == 2
    assert handoff["report"]["capture_count"] == 32
    assert handoff["report"]["capture_quality_eligible_count"] == 30
    assert handoff["report"]["source_unique_market_count"] == 30
    selection = json.loads(handoff["selection_audit_path"].read_text())
    assert selection["selection_uses_outcome_value"] is False
    assert selection["selection_uses_realized_pnl"] is False
    assert selection["selected_market_count"] == 30
    assert selection["excluded_capture_count"] == 2
    assert {
        row["run_id"] for row in selection["excluded_rows"]
    } == {
        "collector-batch00-round12",
        "collector-batch00-round14",
    }
    assert [row["selection_rank"] for row in selection["selected_rows"]] == list(
        range(1, 31)
    )
    assert [row["scheduled_round_start_ts"] for row in selection["selected_rows"]] == sorted(
        row["scheduled_round_start_ts"] for row in selection["selected_rows"]
    )
    assert selection["selected_rows"][-2]["market_id"] == "replacement-market-01"
    assert selection["selected_rows"][-1]["market_id"] == "replacement-market-02"
    access = json.loads(handoff["access_audit_path"].read_text())
    assert access["prohibited_future_outcome_artifact_read_count"] == 0
    assert len(access["prohibited_future_outcome_artifacts_present_but_not_opened"]) == 2


def test_collection_handoff_multi_batch_insufficient_valid_support_fails_closed(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=3)
    original_corpora = [
        _outcome_blind_phase2_corpus(
            tmp_path,
            market_id=f"insufficient-original-{index}",
            market_start_ts=2_000_000 + index * 300_000,
        )
        for index in range(1, 4)
    ]
    original_batch = _collector_batch_progress(
        tmp_path,
        corpus_dirs=original_corpora,
        blocked_indices={3},
        omit_finalization_indices={3},
    )

    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="insufficient-multi-batch-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=original_batch,
            expected_batch_progress_sha256=_sha256(original_batch),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )

    assert handoff["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert handoff["report"]["source_unique_market_count"] == 2
    assert "collection_handoff_unique_market_count_mismatch" in handoff["report"][
        "blocking_reason_codes"
    ]


def test_collection_handoff_rejects_cross_batch_duplicate_market_identity(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=2)
    original_corpus = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="cross-batch-duplicate-market",
        market_start_ts=2_000_000,
    )
    original_batch = _collector_batch_progress(tmp_path, corpus_dirs=[original_corpus])
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    duplicate_corpus = _outcome_blind_phase2_corpus(
        replacement_root,
        market_id="cross-batch-duplicate-market",
        market_start_ts=2_600_000,
    )
    replacement_batch = _collector_batch_progress(tmp_path, corpus_dirs=[duplicate_corpus])

    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="duplicate-market-multi-batch-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=original_batch,
            expected_batch_progress_sha256=_sha256(original_batch),
            additional_batch_progress_paths=(replacement_batch,),
            additional_expected_batch_progress_sha256=(_sha256(replacement_batch),),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )

    assert handoff["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert "collection_handoff_duplicate_market_identity" in handoff["report"][
        "blocking_reason_codes"
    ]
    assert handoff["report"]["source_unique_market_count"] == 1


def test_collection_handoff_rejects_additional_batch_hash_tamper(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=2)
    corpora = [
        _outcome_blind_phase2_corpus(
            tmp_path,
            market_id=f"additional-hash-{index}",
            market_start_ts=2_000_000 + index * 300_000,
        )
        for index in range(1, 3)
    ]
    original_batch = _collector_batch_progress(tmp_path, corpus_dirs=[corpora[0]])
    replacement_batch = _collector_batch_progress(tmp_path, corpus_dirs=[corpora[1]])

    with pytest.raises(ValueError, match="batch progress SHA-256 mismatch"):
        build_pnl_aligned_future_collection_handoff(
            PnLAlignedFutureCollectionHandoffConfig(
                run_id="additional-batch-hash-tamper",
                output_dir=tmp_path / "handoff-runs",
                batch_progress_path=original_batch,
                expected_batch_progress_sha256=_sha256(original_batch),
                additional_batch_progress_paths=(replacement_batch,),
                additional_expected_batch_progress_sha256=("a" * 64,),
                collection_freeze_manifest_path=freeze_path,
                expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
                training_corpus_root=tmp_path,
            )
        )


def test_collection_handoff_consumer_rejects_selection_audit_tamper(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="selection-audit-tamper",
        market_start_ts=2_000_000,
    )
    batch_path = _collector_batch_progress(tmp_path, corpus_dirs=[corpus_dir])
    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="selection-audit-tamper-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=batch_path,
            expected_batch_progress_sha256=_sha256(batch_path),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )
    with handoff["selection_audit_path"].open("a") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="selection_audit descriptor hash mismatch"):
        load_pnl_aligned_future_collection_handoff_source_dirs(
            handoff["manifest_path"],
            expected_sha256=handoff["manifest_sha256"],
        )


def test_collection_handoff_rejects_replacement_not_strictly_later(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=2)
    original_corpus = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="replacement-boundary-original",
        market_start_ts=2_000_000,
    )
    replacement_corpus = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="replacement-boundary-invalid",
        market_start_ts=1_900_000,
    )
    original_batch = _collector_batch_progress(tmp_path, corpus_dirs=[original_corpus])
    replacement_batch = _collector_batch_progress(tmp_path, corpus_dirs=[replacement_corpus])

    handoff = build_pnl_aligned_future_collection_handoff(
        PnLAlignedFutureCollectionHandoffConfig(
            run_id="replacement-boundary-handoff",
            output_dir=tmp_path / "handoff-runs",
            batch_progress_path=original_batch,
            expected_batch_progress_sha256=_sha256(original_batch),
            additional_batch_progress_paths=(replacement_batch,),
            additional_expected_batch_progress_sha256=(_sha256(replacement_batch),),
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            training_corpus_root=tmp_path,
        )
    )

    assert handoff["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert "replacement_capture_not_strictly_later_than_original_batch" in handoff["report"][
        "blocking_reason_codes"
    ]
    assert handoff["report"]["replacement_strictly_later_validation_passed"] is False


def test_collection_handoff_rejects_outcome_field_in_batch_progress(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="batch-outcome-decoy",
        market_start_ts=2_000_000,
    )
    batch_path = _collector_batch_progress(tmp_path, corpus_dirs=[corpus_dir])
    batch = json.loads(batch_path.read_text())
    batch["resolved_outcome"] = "UP"
    batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="contains forbidden outcome fields"):
        build_pnl_aligned_future_collection_handoff(
            PnLAlignedFutureCollectionHandoffConfig(
                run_id="batch-outcome-decoy-handoff",
                output_dir=tmp_path / "handoff-runs",
                batch_progress_path=batch_path,
                expected_batch_progress_sha256=_sha256(batch_path),
                collection_freeze_manifest_path=freeze_path,
                expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
                training_corpus_root=tmp_path,
            )
        )


def test_future_decision_input_rejects_historical_market_overlap(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(
        tmp_path,
        expected_round_count=1,
        prior_market_id="overlap-market",
    )
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="overlap-market",
        market_start_ts=2_000_000,
    )

    result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="future-overlap",
            output_dir=tmp_path / "runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(corpus_dir,),
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )

    assert result["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert "future_source_rows_rejected" in result["report"]["blocking_reason_codes"]
    assert result["report"]["rejected_reason_distribution"] == {
        "future_market_overlaps_historical_fit": 1
    }
    assert result["decision_rows"] == []
    assert result["report"]["source_model_candidate_eligible"] is False
    assert result["report"]["v8_execution_handoff_allowed"] is False


def test_future_decision_input_rejects_time_boundary_and_causality(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=2)
    pre_freeze = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="pre-freeze-market",
        market_start_ts=800_000,
    )
    bad_causality = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="causality-market",
        market_start_ts=2_000_000,
    )
    feature_path = bad_causality / "polymarket_feature_rows.jsonl"
    feature_rows = [json.loads(line) for line in feature_path.read_text().splitlines()]
    feature_rows[0]["max_input_ts"] = feature_rows[0]["decision_ts"] + 1
    _write_jsonl(feature_path, feature_rows)
    manifest_path = bad_causality / "polymarket_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["normalized_artifact_hashes"]["feature_rows"] = _sha256(feature_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="future-time-causality",
            output_dir=tmp_path / "runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(pre_freeze, bad_causality),
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )

    assert result["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert result["report"]["rejected_reason_distribution"] == {
        "decision_before_frozen_future_window": 1,
        "phase2_feature_causality_violation": 1,
    }
    assert result["decision_rows"] == []


def test_future_decision_input_rejects_feature_hash_tamper(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="tampered-market",
        market_start_ts=2_000_000,
    )
    with (corpus_dir / "polymarket_feature_rows.jsonl").open("a") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="normalized artifact hash mismatch"):
        build_pnl_aligned_future_outcome_blind_decision_inputs(
            PnLAlignedFutureDecisionInputConfig(
                run_id="future-tamper",
                output_dir=tmp_path / "runs",
                collection_freeze_manifest_path=freeze_path,
                expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
                source_corpus_dirs=(corpus_dir,),
                paper_candidate_unlock_dir=UNLOCK_DIR,
            )
        )


def test_settlement_targets_load_post_shadow_exactly_once(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="settled-market",
        market_start_ts=2_000_000,
    )
    _add_settlement_artifacts(corpus_dir, resolved_outcome="UP")
    decision_result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="settled-decision-input",
            output_dir=tmp_path / "runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(corpus_dir,),
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )
    shadow_path = _shadow_manifest(tmp_path, decision_result)

    result = build_pnl_aligned_future_settled_evaluation_targets(
        PnLAlignedFutureSettlementTargetConfig(
            run_id="settlement-targets",
            output_dir=tmp_path / "target-runs",
            shadow_manifest_path=shadow_path,
            expected_shadow_manifest_sha256=_sha256(shadow_path),
        )
    )

    assert result["report"]["status"] == "SETTLED_EVALUATION_TARGETS_READY"
    assert result["report"]["identity_reconciliation_passed"] is True
    assert result["report"]["settled_target_count"] == 1
    assert result["report"]["settled_market_count"] == 1
    assert result["report"]["future_results_used_for_tuning"] is False
    assert result["report"]["promotion_evidence_eligible"] is False
    target = result["targets"][0]
    assert set(target["evaluation_target_net_pnl_per_contract_by_action"]) == set(ACTIONS)
    assert set(target["evaluation_target_pnl_components_by_action"]) == set(ACTIONS)
    assert target["resolved_outcome"] == "UP"
    assert target["outcome_used_for_shadow_selection"] is False
    marker = shadow_path.parent / "pnl_aligned_future_outcome_reconciliation_started.json"
    assert marker.exists()

    with pytest.raises(ValueError, match="already started"):
        build_pnl_aligned_future_settled_evaluation_targets(
            PnLAlignedFutureSettlementTargetConfig(
                run_id="settlement-targets-second-attempt",
                output_dir=tmp_path / "target-runs",
                shadow_manifest_path=shadow_path,
                expected_shadow_manifest_sha256=_sha256(shadow_path),
            )
        )


def test_settlement_target_identity_mismatch_fails_before_marker(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="identity-market",
        market_start_ts=2_000_000,
    )
    _add_settlement_artifacts(corpus_dir, resolved_outcome="DOWN")
    decision_result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="identity-decision-input",
            output_dir=tmp_path / "runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(corpus_dir,),
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )
    shadow_path = _shadow_manifest(
        tmp_path,
        decision_result,
        baseline_identity="wrong-source-row-identity",
        shadow_dir_name="identity-shadow",
    )

    with pytest.raises(ValueError, match="identities do not match"):
        build_pnl_aligned_future_settled_evaluation_targets(
            PnLAlignedFutureSettlementTargetConfig(
                run_id="identity-targets",
                output_dir=tmp_path / "target-runs",
                shadow_manifest_path=shadow_path,
                expected_shadow_manifest_sha256=_sha256(shadow_path),
            )
        )
    assert not (
        shadow_path.parent / "pnl_aligned_future_outcome_reconciliation_started.json"
    ).exists()


def test_settlement_target_rejects_prediction_artifact_drift_before_marker(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="prediction-drift-market",
        market_start_ts=2_000_000,
    )
    _add_settlement_artifacts(corpus_dir, resolved_outcome="UP")
    decision_result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="prediction-drift-decision-input",
            output_dir=tmp_path / "runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(corpus_dir,),
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )
    shadow_path = _shadow_manifest(tmp_path, decision_result)
    shadow = json.loads(shadow_path.read_text())
    prediction_path = Path(shadow["candidate_action_value_predictions"]["path"])
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    predictions[0]["predicted_net_pnl_per_contract"] += 1.0
    _write_jsonl(prediction_path, predictions)
    shadow["candidate_action_value_predictions"] = _descriptor(prediction_path)
    shadow_path.write_text(json.dumps(shadow, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="differs from frozen shadow"):
        build_pnl_aligned_future_settled_evaluation_targets(
            PnLAlignedFutureSettlementTargetConfig(
                run_id="prediction-drift-targets",
                output_dir=tmp_path / "target-runs",
                shadow_manifest_path=shadow_path,
                expected_shadow_manifest_sha256=_sha256(shadow_path),
            )
        )
    assert not (
        shadow_path.parent / "pnl_aligned_future_outcome_reconciliation_started.json"
    ).exists()


def test_settlement_targets_fail_closed_for_unresolved_official_outcome(
    tmp_path: Path,
) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="unresolved-market",
        market_start_ts=2_000_000,
    )
    _add_settlement_artifacts(corpus_dir, resolved_outcome="PENDING")
    decision_result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="unresolved-decision-input",
            output_dir=tmp_path / "runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(corpus_dir,),
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )
    shadow_path = _shadow_manifest(tmp_path, decision_result)

    with pytest.raises(ValueError, match="official resolved outcome is unavailable"):
        build_pnl_aligned_future_settled_evaluation_targets(
            PnLAlignedFutureSettlementTargetConfig(
                run_id="unresolved-targets",
                output_dir=tmp_path / "target-runs",
                shadow_manifest_path=shadow_path,
                expected_shadow_manifest_sha256=_sha256(shadow_path),
            )
        )
    assert (shadow_path.parent / "pnl_aligned_future_outcome_reconciliation_started.json").exists()


def test_settlement_targets_fail_closed_for_incomplete_action_grid(tmp_path: Path) -> None:
    freeze_path = _collection_freeze(tmp_path, expected_round_count=1)
    corpus_dir = _outcome_blind_phase2_corpus(
        tmp_path,
        market_id="incomplete-grid-market",
        market_start_ts=2_000_000,
    )
    _add_settlement_artifacts(corpus_dir, resolved_outcome="UP")
    label_path = corpus_dir / "polymarket_label_rows.jsonl"
    labels = [json.loads(line) for line in label_path.read_text().splitlines() if line]
    _write_jsonl(label_path, labels[:-1])
    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["normalized_artifact_hashes"]["label_rows"] = _sha256(label_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    decision_result = build_pnl_aligned_future_outcome_blind_decision_inputs(
        PnLAlignedFutureDecisionInputConfig(
            run_id="incomplete-grid-decision-input",
            output_dir=tmp_path / "runs",
            collection_freeze_manifest_path=freeze_path,
            expected_collection_freeze_manifest_sha256=_sha256(freeze_path),
            source_corpus_dirs=(corpus_dir,),
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )
    shadow_path = _shadow_manifest(tmp_path, decision_result)

    with pytest.raises(ValueError, match="action target grid is incomplete"):
        build_pnl_aligned_future_settled_evaluation_targets(
            PnLAlignedFutureSettlementTargetConfig(
                run_id="incomplete-grid-targets",
                output_dir=tmp_path / "target-runs",
                shadow_manifest_path=shadow_path,
                expected_shadow_manifest_sha256=_sha256(shadow_path),
            )
        )
    assert (shadow_path.parent / "pnl_aligned_future_outcome_reconciliation_started.json").exists()


def _candidate_shadow_row() -> dict:
    ranking = []
    for rank, action in enumerate(ACTIONS, start=1):
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if action.endswith("HOLD_TO_SETTLEMENT")
            else "SELL_BEFORE_CLOSE"
            if action.endswith("SELL_BEFORE_CLOSE")
            else "NO_TRADE"
        )
        ranking.append(
            {
                "rank": rank,
                "action": action,
                "side": side,
                "action_family": family,
                "predicted_net_pnl_per_contract": 0.1 - rank * 0.01,
            }
        )
    return {
        "source_row_identity": "prediction-row",
        "market_id": "prediction-market",
        "decision_ts": 2_000_000,
        "market_close_ts": 2_300_000,
        "selected_action": ACTIONS[0],
        "full_5_action_model_ranking": ranking,
        "outcome_fields_used": False,
        "realized_pnl_used": False,
    }


def _collection_freeze(
    tmp_path: Path,
    *,
    expected_round_count: int,
    prior_market_id: str = "historical-market",
) -> Path:
    historical_path = tmp_path / f"{prior_market_id}-historical.jsonl"
    historical_row = {"market_id": prior_market_id, "decision_ts": 900_000}
    historical_path.write_text(json.dumps(historical_row) + "\n")
    freeze_path = tmp_path / f"{prior_market_id}-collection-freeze.json"
    freeze = {
        "collection_freeze_id": "frozen-collection-id",
        "future_collection_outcome_blind": True,
        "future_window_must_be_strictly_later": True,
        "future_market_ids_must_be_disjoint": True,
        "model_config_or_threshold_mutation_after_freeze_allowed": False,
        "historical_development_rows": {
            "path": str(historical_path.resolve()),
            "sha256": _sha256(historical_path),
        },
        "prior_market_count": 1,
        "prior_market_ids_sha256": canonical_json_sha256([prior_market_id]),
        "max_prior_decision_ts": 900_000,
        "minimum_future_window_start_ts": 1_000_000,
        "expected_round_count": expected_round_count,
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return freeze_path


def _outcome_blind_phase2_corpus(
    root: Path,
    *,
    market_id: str,
    market_start_ts: int,
) -> Path:
    corpus_dir = root / f"corpus-{market_id}"
    corpus_dir.mkdir()
    decision_ts = market_start_ts + 60_000
    market_end_ts = market_start_ts + 300_000
    feature_rows = [
        {
            "market_id": market_id,
            "condition_id": market_id,
            "slug": f"btc-updown-5m-{market_start_ts // 1000}",
            "decision_ts": decision_ts,
            "max_input_ts": decision_ts - 100,
            "features": {
                "btc_mid_price": 65_130.0,
                "btc_return_1m": 0.001,
                "up_bid": 0.64,
                "up_ask": 0.66,
                "up_bid_size": 200.0,
                "up_ask_size": 150.0,
                "up_liquidity_depth": 20_000.0,
                "up_book_staleness_ms": 100,
                "up_recent_book_update_count_1m": 30,
                "up_recent_spread_stability_1m": 0.9,
                "down_bid": 0.34,
                "down_ask": 0.36,
                "down_bid_size": 100.0,
                "down_ask_size": 120.0,
                "down_liquidity_depth": 18_000.0,
                "down_book_staleness_ms": 100,
                "down_recent_book_update_count_1m": 25,
                "down_recent_spread_stability_1m": 0.9,
            },
            "feature_provenance": {"btc_mid_price": {"input_end_ts": decision_ts - 100}},
        }
    ]
    metadata_rows = [
        {
            "market_id": market_id,
            "condition_id": market_id,
            "slug": f"btc-updown-5m-{market_start_ts // 1000}",
            "market_family": "btc_updown_5m",
            "market_start_ts": market_start_ts,
            "market_end_ts": market_end_ts,
            "horizon_ms": 300_000,
            "reference_price_start": None,
        }
    ]
    chainlink_rows = [
        _chainlink_row(market_start_ts - 120_000, 65_000.0),
        _chainlink_row(market_start_ts, 65_020.0),
        _chainlink_row(decision_ts - 30_000, 65_080.0),
        _chainlink_row(decision_ts - 1_000, 65_130.0),
    ]
    feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
    metadata_path = corpus_dir / "polymarket_market_metadata.jsonl"
    chainlink_path = corpus_dir / "polymarket_chainlink_prices.jsonl"
    _write_jsonl(feature_path, feature_rows)
    _write_jsonl(metadata_path, metadata_rows)
    _write_jsonl(chainlink_path, chainlink_rows)
    (corpus_dir / "polymarket_corpus_manifest.json").write_text(
        json.dumps(
            {
                "normalized_artifact_hashes": {
                    "feature_rows": _sha256(feature_path),
                    "market_metadata": _sha256(metadata_path),
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json").write_text(
        json.dumps(
            {
                "evidence_sha256": _sha256(chainlink_path),
                "timestamp_causality_violation_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (corpus_dir / "training_corpus_provenance.json").write_text(
        json.dumps(
            {"corpus_id": corpus_dir.name, "run_id": f"{corpus_dir.name}-run"},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return corpus_dir


def _collector_batch_progress(
    root: Path,
    *,
    corpus_dirs: list[Path],
    pending_indices: set[int] | None = None,
    blocked_indices: set[int] | None = None,
    omit_finalization_indices: set[int] | None = None,
) -> Path:
    pending_indices = pending_indices or set()
    blocked_indices = blocked_indices or set()
    omit_finalization_indices = omit_finalization_indices or set()
    batch_ordinal = len(list(root.glob("batch-progress-*.json")))
    captures = []
    finalizations = []
    for index, corpus_dir in enumerate(corpus_dirs, start=1):
        run_id = f"collector-batch{batch_ordinal:02d}-round{index:02d}"
        metadata = json.loads(
            (corpus_dir / "polymarket_market_metadata.jsonl").read_text().splitlines()[0]
        )
        captures.append(
            {
                "run_id": run_id,
                "round_index": index,
                "capture_status": (
                    "blocked_fail_closed" if index in blocked_indices else "pending_resolution"
                ),
                "scheduled_round_start_ts": int(metadata["market_start_ts"]),
                "capture_start_boundary_validation_passed": True,
                "raw_polymarket_market_count": 0 if index in blocked_indices else 1,
                "provider_raw_orderbook_snapshot_count": (
                    0 if index in blocked_indices else 100
                ),
                "training_sampled_orderbook_row_count": 0 if index in blocked_indices else 8,
                "raw_chainlink_price_row_count": 0 if index in blocked_indices else 100,
                "reject_reason_counts": (
                    {"read_only_public_http_timeout": 1}
                    if index in blocked_indices
                    else {}
                ),
            }
        )
        if index in omit_finalization_indices:
            continue
        pending = index in pending_indices
        finalizations.append(
            {
                "run_id": run_id,
                "finalization_status": "pending_resolution" if pending else "exported",
                "pending_resolution": pending,
                "training_eligible": not pending,
                "raw_resolution_count": 0 if pending else 1,
                "reject_reason_counts": {"missing_resolution": 1} if pending else {},
                "exported_training_corpus_dir": None if pending else str(corpus_dir.resolve()),
            }
        )
    exported_count = sum(row["finalization_status"] == "exported" for row in finalizations)
    batch_path = root / f"batch-progress-{batch_ordinal}.json"
    batch_path.write_text(
        json.dumps(
            {
                "batch_id": batch_path.stem,
                "paper_only": True,
                "capital_at_risk": False,
                "capture_count": len(captures),
                "exported_round_count": exported_count,
                "pending_resolution_count": len(pending_indices),
                "error_count": 0,
                "captures": captures,
                "finalizations": finalizations,
                "errors": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return batch_path


def _add_settlement_artifacts(corpus_dir: Path, *, resolved_outcome: str) -> None:
    feature_row = json.loads(
        (corpus_dir / "polymarket_feature_rows.jsonl").read_text().splitlines()[0]
    )
    market_id = str(feature_row["market_id"])
    decision_ts = int(feature_row["decision_ts"])
    raw_resolution_sha256 = "a" * 64
    labels = []
    for index, action in enumerate(ACTIONS):
        net_pnl = 0.0 if action == "NO_TRADE" else 0.05 + index * 0.01
        labels.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "action": action,
                "total_net_pnl_per_notional": net_pnl,
                "fees": 0.001,
                "slippage": 0.002,
                "liquidity_impact": 0.001,
                "resolved_outcome": resolved_outcome,
                "raw_resolution_sha256": raw_resolution_sha256,
            }
        )
    resolutions = [
        {
            "market_id": market_id,
            "resolved_outcome": resolved_outcome,
            "resolution_status": "normal",
            "raw_resolution_sha256": raw_resolution_sha256,
            "resolution_rule_sha256": "b" * 64,
        }
    ]
    label_path = corpus_dir / "polymarket_label_rows.jsonl"
    resolution_path = corpus_dir / "polymarket_resolution_events.jsonl"
    _write_jsonl(label_path, labels)
    _write_jsonl(resolution_path, resolutions)
    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["normalized_artifact_hashes"]["label_rows"] = _sha256(label_path)
    manifest["normalized_artifact_hashes"]["resolution_events"] = _sha256(resolution_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _shadow_manifest(
    tmp_path: Path,
    decision_result: dict,
    *,
    baseline_identity: str | None = None,
    shadow_dir_name: str = "shadow",
) -> Path:
    shadow_dir = tmp_path / shadow_dir_name
    shadow_dir.mkdir()
    decision = decision_result["decision_rows"][0]
    source_identity = str(decision["row_identity"])
    candidate_path = shadow_dir / "candidate.jsonl"
    baseline_path = shadow_dir / "baseline.jsonl"
    candidate_row = _candidate_shadow_row()
    candidate_row.update(
        {
            "source_row_identity": source_identity,
            "market_id": decision["market_id"],
            "decision_ts": decision["decision_ts"],
            "market_close_ts": decision["market_close_ts"],
        }
    )
    _write_jsonl(candidate_path, [candidate_row])
    _write_jsonl(
        baseline_path,
        [
            {
                "source_row_identity": baseline_identity
                if baseline_identity is not None
                else source_identity
            }
        ],
    )
    prediction_path = shadow_dir / "predictions.jsonl"
    _write_jsonl(
        prediction_path,
        materialize_pnl_aligned_future_action_value_predictions([candidate_row]),
    )
    manifest_path = shadow_dir / "shadow_manifest.json"
    manifest = {
        "decision_input_manifest": _descriptor(decision_result["manifest_path"]),
        "input_decision_rows": _descriptor(decision_result["decision_rows_path"]),
        "candidate_shadow_rows": _descriptor(candidate_path),
        "baseline_shadow_rows": _descriptor(baseline_path),
        "candidate_action_value_predictions": _descriptor(prediction_path),
        "future_outcome_targets_loaded": False,
        "outcome_reconciliation_started": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _chainlink_row(source_ts: int, price: float) -> dict:
    return {
        "source_ts": source_ts,
        "available_at_ts": source_ts + 100,
        "price": price,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
