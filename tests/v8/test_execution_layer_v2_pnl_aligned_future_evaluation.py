from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_future_evaluation import (
    PnLAlignedFutureDecisionInputConfig,
    build_pnl_aligned_future_outcome_blind_decision_inputs,
    evaluate_pnl_aligned_future_accepted_bets,
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


def _chainlink_row(source_ts: int, price: float) -> dict:
    return {
        "source_ts": source_ts,
        "available_at_ts": source_ts + 100,
        "price": price,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
