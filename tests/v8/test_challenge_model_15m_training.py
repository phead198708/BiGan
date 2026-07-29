from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import SAFETY, sha256_file
from bigan.v8.polymarket.challenge_model_15m_training import (
    BASE_FEATURE_NAMES,
    _apply_pair_probability_normalization,
    _assign_market_grouped_temporal_splits,
    run_challenge_model_15m_rolling_origin_oof,
    run_challenge_model_15m_training,
    side_symmetric_features,
    validate_rolling_origin_oof_preregistration,
    validate_training_slot_preregistration,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256


def test_side_symmetric_features_mirror_direction_and_preserve_missing() -> None:
    row = _feature_row("market", 1_000_000, outcome_index=0)
    up = side_symmetric_features(row, "UP")
    down = side_symmetric_features(row, "DOWN")

    assert up["selected_ask"] == down["opposite_ask"]
    assert up["selected_minus_opposite_mid"] == -down["selected_minus_opposite_mid"]
    assert up["signed_btc_return_1m"] == -down["signed_btc_return_1m"]
    assert up["selected_recent_trade_volume"] != up["selected_recent_trade_volume"]
    assert up["selected_recent_trade_volume__missing"] == 1.0
    assert up["selected_recent_trade_volume"] != 0.0
    assert "side_is_up" not in up


def test_market_grouped_temporal_split_has_no_overlap() -> None:
    rows = [
        {
            "market_id": f"market-{index}",
            "market_start_ts": 1_000 + index,
        }
        for index in range(10)
        for _ in range(4)
    ]
    split = _assign_market_grouped_temporal_splits(
        rows,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    assert list(split.values()).count("train") == 6
    assert list(split.values()).count("validation") == 2
    assert list(split.values()).count("test") == 2
    assert len(split) == 10


def test_training_slot_end_to_end_and_sha_mismatch_fail_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    lane = repo / "lane"
    lane.mkdir(parents=True)
    index_path = lane / "finalized_development_corpus_index.jsonl"
    previous = "0" * 64
    index_rows = []
    for market_index in range(10):
        corpus = lane / "development_corpus" / f"market-{market_index}"
        _write_synthetic_corpus(corpus, market_index)
        manifest_path = corpus / "polymarket_corpus_manifest.json"
        entry = {
            "schema_version": (
                "bigan-challenge-model-development-lane-finalized-index-v1"
            ),
            "sequence": market_index + 1,
            "previous_entry_sha256": previous,
            "run_id": f"run-{market_index}",
            "finalized_at": "2026-01-01T00:00:00+00:00",
            "protocol_sha256": "1" * 64,
            "finalizer_summary_path": str(corpus / "summary.json"),
            "finalizer_summary_sha256": "2" * 64,
            "exported_corpus_manifest_path": str(manifest_path),
            "exported_corpus_manifest_sha256": sha256_file(manifest_path),
            "official_post_close_resolution_opened": True,
            "target_used_by_capture_control": False,
            "corpus_role": "development_training_only",
            "development_only_forever": True,
            "promotion_evidence_eligible": False,
            "safety": dict(SAFETY),
        }
        entry["entry_sha256"] = canonical_json_sha256(entry)
        previous = entry["entry_sha256"]
        index_rows.append(entry)
    _write_jsonl(index_path, index_rows)
    protocol_path = repo / "training_protocol.json"
    _write_json(protocol_path, {"frozen": True})
    transfer_path = repo / "transfer_freeze.json"
    _write_json(transfer_path, {"frozen": True})
    readiness_path = repo / "training_readiness.json"
    _write_json(
        readiness_path,
        {
            "training_start_allowed": True,
            "model_training_started": False,
            "blockers": [],
            "safety": dict(SAFETY),
        },
    )
    prereg_path = repo / "slot.json"
    prereg = _preregistration(
        repo=repo,
        index_path=index_path,
        protocol_path=protocol_path,
        transfer_path=transfer_path,
        readiness_path=readiness_path,
        market_count=10,
    )
    _write_json(prereg_path, prereg)
    validate_training_slot_preregistration(prereg)
    result = run_challenge_model_15m_training(
        preregistration_path=prereg_path,
        expected_preregistration_sha256=sha256_file(prereg_path),
        output_dir=repo / "output",
        source_commit="b" * 40,
        created_at="2026-01-02T00:00:00+00:00",
        repository_root=repo,
    )
    report = json.loads(Path(result["training_report_path"]).read_text())
    assert report["dataset"]["market_count"] == 10
    assert report["dataset"]["side_row_count"] == 40
    assert report["split"]["market_counts"] == {
        "test": 2,
        "train": 6,
        "validation": 2,
    }
    assert report["split"]["market_overlap_count"] == 0
    assert report["promotion_claim_made"] is False
    assert report["threshold_or_hyperparameter_search_performed"] is False
    assert report["development_signal_rule"]["promotion_claim_allowed"] is False
    assert report["safety"] == SAFETY

    with pytest.raises(ValueError, match="preregistration SHA-256 mismatch"):
        run_challenge_model_15m_training(
            preregistration_path=prereg_path,
            expected_preregistration_sha256="0" * 64,
            output_dir=repo / "not-created",
            source_commit="b" * 40,
            repository_root=repo,
        )

    candidate = json.loads(json.dumps(prereg))
    candidate["training_slot_id"] = "synthetic-pair-candidate"
    candidate["model"]["family"] = (
        "xgboost_shared_side_symmetric_pair_normalized_win_probability_"
        "with_cost_subtraction"
    )
    candidate["model"]["parameters"]["objective"] = "binary:logistic"
    candidate["model"]["parameters"]["eval_metric"] = "logloss"
    candidate_path = repo / "candidate_pair.json"
    _write_json(candidate_path, candidate)
    prior_result_path = repo / "prior_result.json"
    _write_json(prior_result_path, {"status": "development_only"})
    oof_prereg = {
        "schema_version": (
            "bigan-challenge-model-15m-rolling-origin-oof-preregistration-v1"
        ),
        "diagnostic_id": "synthetic-oof",
        "role": "outcome-aware-development-diagnostic-only",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "candidate_preregistration": _descriptor(repo, candidate_path),
        "prior_result": _descriptor(repo, prior_result_path),
        "rolling_origin": {
            "initial_training_market_count": 5,
            "target_block_size": 1,
            "inner_train_fraction": 0.8,
            "strictly_prior_market_labels_only": True,
            "future_market_labels_or_features_allowed": False,
        },
        "development_evaluation_policy": {
            "bootstrap_resamples": 100,
            "bootstrap_seed": 1,
            "minimum_accepted_oof_markets": 2,
        },
        "development_discipline": {
            "threshold_search_allowed": False,
            "hyperparameter_search_allowed": False,
            "candidate_behavior_changed": False,
        },
        "safety": dict(SAFETY),
    }
    oof_prereg_path = repo / "oof_prereg.json"
    _write_json(oof_prereg_path, oof_prereg)
    oof_result = run_challenge_model_15m_rolling_origin_oof(
        preregistration_path=oof_prereg_path,
        expected_preregistration_sha256=sha256_file(oof_prereg_path),
        output_dir=repo / "oof_output",
        source_commit="c" * 40,
        created_at="2026-01-03T00:00:00+00:00",
        repository_root=repo,
    )
    oof_report = json.loads(Path(oof_result["report_path"]).read_text())
    assert oof_report["oof_market_count"] == 5
    assert oof_report["fold_count"] == 5
    assert oof_report["target_or_future_label_leakage_count"] == 0
    assert oof_report["promotion_claim_made"] is False


def test_readiness_closed_fails_before_training(tmp_path: Path) -> None:
    prereg = _minimal_preregistration()
    validate_training_slot_preregistration(prereg)
    prereg["development_discipline"]["threshold_search_allowed"] = True
    with pytest.raises(ValueError, match="threshold_search_allowed"):
        validate_training_slot_preregistration(prereg)


def test_win_probability_family_is_valid_and_keeps_fixed_cost_score() -> None:
    prereg = _minimal_preregistration()
    prereg["model"]["family"] = (
        "xgboost_shared_side_symmetric_win_probability_with_cost_subtraction"
    )
    prereg["model"]["parameters"]["objective"] = "binary:logistic"
    prereg["model"]["parameters"]["eval_metric"] = "logloss"
    validate_training_slot_preregistration(prereg)


def test_pair_normalization_enforces_complementarity_before_cost() -> None:
    rows = [
        {
            "market_id": "market",
            "decision_ts": 1,
            "side": "UP",
            "raw_win_probability": 0.60,
            "execution_cost": 0.55,
        },
        {
            "market_id": "market",
            "decision_ts": 1,
            "side": "DOWN",
            "raw_win_probability": 0.50,
            "execution_cost": 0.45,
        },
    ]
    _apply_pair_probability_normalization(rows)
    assert rows[0]["win_probability"] + rows[1]["win_probability"] == pytest.approx(1.0)
    assert rows[0]["prediction"] == pytest.approx(0.60 / 1.10 - 0.55)
    assert rows[1]["prediction"] == pytest.approx(0.50 / 1.10 - 0.45)

    prereg = _minimal_preregistration()
    prereg["model"]["family"] = (
        "xgboost_shared_side_symmetric_pair_normalized_win_probability_"
        "with_cost_subtraction"
    )
    prereg["model"]["parameters"]["objective"] = "binary:logistic"
    prereg["model"]["parameters"]["eval_metric"] = "logloss"
    validate_training_slot_preregistration(prereg)


def test_rolling_origin_preregistration_fails_on_non_prior_labels() -> None:
    payload = {
        "schema_version": (
            "bigan-challenge-model-15m-rolling-origin-oof-preregistration-v1"
        ),
        "role": "outcome-aware-development-diagnostic-only",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "candidate_preregistration": {"path": "candidate.json", "sha256": "1" * 64},
        "prior_result": {"path": "result.json", "sha256": "2" * 64},
        "rolling_origin": {
            "initial_training_market_count": 40,
            "target_block_size": 1,
            "inner_train_fraction": 0.8,
            "strictly_prior_market_labels_only": True,
            "future_market_labels_or_features_allowed": False,
        },
        "development_discipline": {
            "threshold_search_allowed": False,
            "hyperparameter_search_allowed": False,
            "candidate_behavior_changed": False,
        },
        "safety": dict(SAFETY),
    }
    validate_rolling_origin_oof_preregistration(payload)
    payload["rolling_origin"]["future_market_labels_or_features_allowed"] = True
    with pytest.raises(ValueError, match="future_market_labels_or_features_allowed"):
        validate_rolling_origin_oof_preregistration(payload)


def _preregistration(
    *,
    repo: Path,
    index_path: Path,
    protocol_path: Path,
    transfer_path: Path,
    readiness_path: Path,
    market_count: int,
) -> dict:
    payload = _minimal_preregistration()
    payload["dataset"] = {
        "quality_valid_outcome_finalized_market_count": market_count,
    }
    payload["input_pins"] = {
        "training_protocol": _descriptor(repo, protocol_path),
        "training_readiness": _descriptor(repo, readiness_path),
        "transfer_freeze": _descriptor(repo, transfer_path),
        "finalized_development_corpus_index": _descriptor(repo, index_path),
    }
    return payload


def _minimal_preregistration() -> dict:
    return {
        "schema_version": "bigan-challenge-model-15m-training-slot-v1",
        "training_slot_id": "synthetic-slot",
        "role": "outcome-aware-development-training-only",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "training_implementation_commit": "a" * 40,
        "dataset": {
            "quality_valid_outcome_finalized_market_count": 10,
        },
        "input_pins": {
            name: {"path": f"{name}.json", "sha256": "1" * 64}
            for name in (
                "training_protocol",
                "training_readiness",
                "transfer_freeze",
                "finalized_development_corpus_index",
            )
        },
        "target": {
            "policy": "HOLD_TO_SETTLEMENT",
            "field": "total_net_pnl_per_notional",
            "unit_sizing": True,
        },
        "feature_contract": {
            "base_feature_names": list(BASE_FEATURE_NAMES),
            "shared_side_symmetric_model": True,
            "side_identity_feature_allowed": False,
            "native_missing_value": "xgboost_nan",
            "explicit_missing_indicator_for_every_feature": True,
        },
        "split": {
            "method": "chronological_unique_market_groups",
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "all_rows_for_one_market_must_remain_in_one_split": True,
        },
        "model": {
            "family": "xgboost_shared_side_symmetric_regressor",
            "parameters": {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "nthread": 1,
                "max_depth": 2,
                "eta": 0.1,
                "seed": 1,
            },
            "num_boost_round": 5,
            "early_stopping_rounds": 2,
        },
        "development_evaluation_policy": {
            "decision_rule": (
                "chronological_first_decision_with_positive_prediction_then_"
                "highest_side_prediction"
            ),
            "fixed_acceptance_threshold": 0.0,
            "one_trade_maximum_per_market": True,
            "bootstrap_resamples": 100,
            "bootstrap_seed": 1,
            "development_signal_rule": {
                "minimum_accepted_test_markets": 2,
                "test_mean_unit_net_pnl_bootstrap_lcb_must_be_gt": 0.0,
                "report_only": True,
                "promotion_claim_allowed": False,
            },
        },
        "development_discipline": {
            "candidate_count": 1,
            "hyperparameter_search_allowed": False,
            "threshold_search_allowed": False,
            "old_15m_plus_12_39_used_as_gate": False,
        },
        "safety": dict(SAFETY),
    }


def _write_synthetic_corpus(corpus: Path, market_index: int) -> None:
    corpus.mkdir(parents=True)
    market_id = f"market-{market_index}"
    start = 1_000_000 + market_index * 10_000
    outcome = "UP" if market_index % 2 == 0 else "DOWN"
    feature_rows = [
        _feature_row(market_id, start + offset, outcome_index=market_index)
        for offset in (300_000, 600_000)
    ]
    labels = []
    for row in feature_rows:
        for side in ("UP", "DOWN"):
            ask = row["features"][f"{side.lower()}_ask"]
            mid = row["features"][f"{side.lower()}_mid"]
            payout = float(side == outcome)
            spread = ask - mid
            fees = 0.001
            slippage = 0.002
            impact = 0.001
            labels.append(
                {
                    "market_id": market_id,
                    "decision_ts": row["decision_ts"],
                    "action": f"BUY_{side}_HOLD_TO_SETTLEMENT",
                    "total_net_pnl_per_notional": (
                        payout - mid - spread - fees - slippage - impact
                    ),
                    "settlement_payout": payout,
                    "entry_ask": ask,
                    "entry_mid": mid,
                    "fees": fees,
                    "slippage": slippage,
                    "liquidity_impact": impact,
                    "resolution_status": "normal",
                    "resolved_outcome": outcome,
                }
            )
        labels.append(
            {
                "market_id": market_id,
                "decision_ts": row["decision_ts"],
                "action": "NO_TRADE",
            }
        )
    metadata = [
        {
            "market_id": market_id,
            "market_start_ts": start,
        }
    ]
    resolutions = [{"market_id": market_id, "resolved_outcome": outcome}]
    _write_jsonl(corpus / "polymarket_feature_rows.jsonl", feature_rows)
    _write_jsonl(corpus / "polymarket_label_rows.jsonl", labels)
    _write_jsonl(corpus / "polymarket_market_metadata.jsonl", metadata)
    _write_jsonl(corpus / "polymarket_resolution_events.jsonl", resolutions)
    hashes = {
        "feature_rows": sha256_file(corpus / "polymarket_feature_rows.jsonl"),
        "label_rows": sha256_file(corpus / "polymarket_label_rows.jsonl"),
        "market_metadata": sha256_file(corpus / "polymarket_market_metadata.jsonl"),
        "resolution_events": sha256_file(corpus / "polymarket_resolution_events.jsonl"),
    }
    _write_json(
        corpus / "polymarket_corpus_manifest.json",
        {
            "market_count": 1,
            "market_family_counts": {"btc_updown_15m": 1},
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "normalized_artifact_hashes": hashes,
        },
    )


def _feature_row(
    market_id: str,
    decision_ts: int,
    *,
    outcome_index: int,
) -> dict:
    raw = {
        "up_ask": 0.51,
        "up_bid": 0.49,
        "up_mid": 0.50,
        "down_ask": 0.51,
        "down_bid": 0.49,
        "down_mid": 0.50,
        "up_down_ask_sum": 1.02,
        "up_down_bid_sum": 0.98,
        "up_down_mid_sum": 1.0,
        "combined_spread_bps": 400.0,
        "chainlink_reference_distance_at_decision": 0.001 * (-1) ** outcome_index,
        "btc_mid_price": 100.1,
        "chainlink_price_at_decision": 100.0,
        "btc_return_10s": 0.001,
        "btc_return_30s": 0.001,
        "btc_return_1m": 0.002,
        "btc_return_5m": 0.003,
        "btc_return_15m": 0.004,
        "btc_volatility_1m": 0.01,
        "btc_volatility_5m": 0.02,
        "btc_volatility_15m": 0.03,
        "market_age_seconds": 300.0,
        "time_to_close_seconds": 600.0,
        "horizon_ms": 900_000,
        "provider_health_score": 1.0,
        "book_snapshot_pair_ts_delta_ms": 0.0,
    }
    for side in ("up", "down"):
        raw.update(
            {
                f"{side}_ask_size": 10.0,
                f"{side}_bid_size": 11.0,
                f"{side}_spread_bps": 200.0,
                f"{side}_executable_ask_notional": 5.0,
                f"{side}_executable_bid_notional": 5.0,
                f"{side}_liquidity_depth": 100.0,
                f"{side}_book_staleness_ms": 10.0,
                f"{side}_book_update_lag_ms": 10.0,
                f"{side}_queue_fill_probability_proxy": 0.9,
                f"{side}_recent_bid_depth_volatility_1m": 0.1,
                f"{side}_recent_book_update_count_1m": 2.0,
                f"{side}_recent_spread_stability_1m": 0.9,
                f"recent_{side}_trade_volume": None,
            }
        )
    provenance = {
        name: {"available_at_ts": decision_ts, "max_input_ts": decision_ts}
        for name in raw
    }
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "available_at_ts": decision_ts,
        "feature_cutoff_ts": decision_ts,
        "max_input_ts": decision_ts,
        "features": raw,
        "feature_provenance": provenance,
    }


def _descriptor(repo: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(repo)),
        "sha256": sha256_file(path),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
