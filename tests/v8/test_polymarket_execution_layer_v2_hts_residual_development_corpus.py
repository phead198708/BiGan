from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_confirmatory import (
    HTSResidualCandidateFreezeConfig,
    HTSResidualConfirmatoryEvaluationConfig,
    HTSResidualConfirmatoryInputConfig,
    evaluate_hts_residual_confirmatory_once,
    freeze_hts_residual_candidate,
    freeze_hts_residual_confirmatory_input,
)
from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_development_corpus import (
    HTSResidualDevelopmentCorpusConfig,
    HTSResidualForwardOOFConfig,
    build_hts_residual_development_corpus,
    run_hts_residual_development_forward_oof,
)
from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_edge import (
    fit_residual_offset_contract,
)

UNLOCK_DIR = Path(
    "examples/v8/polymarket_runs/o-v8-paper-candidate-unlock-20260703T073000Z"
)


def test_builds_causal_post_protocol_development_rows_fail_closed(
    tmp_path: Path,
) -> None:
    protocol = _protocol(collection_not_before_ts=1_000_000, minimum_markets=1)
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    prior_path = tmp_path / "prior.jsonl"
    _write_jsonl(
        prior_path,
        [{"market_id": "prior-market", "decision_ts": 900_000}],
    )
    corpus_dir = _phase2_corpus(tmp_path, market_id="new-market")

    result = build_hts_residual_development_corpus(
        HTSResidualDevelopmentCorpusConfig(
            run_id="development-build",
            output_dir=tmp_path / "runs",
            protocol_path=protocol_path,
            expected_protocol_sha256=_sha256(protocol_path),
            source_corpus_dirs=(corpus_dir,),
            prior_development_rows_path=prior_path,
            paper_candidate_unlock_dir=UNLOCK_DIR,
        )
    )

    report = result["report"]
    assert report["feature_causality_violation_count"] == 0
    assert set(report["source_chainlink_feature_coverage"].values()) == {1}
    assert set(report["residual_chainlink_feature_coverage"].values()) == {1}
    assert report["chainlink_feature_coverage_scope"] == "residual_hts_rows"
    assert report["residual_market_count"] == 1
    assert report["forward_oof_evaluation_ready"] is True
    assert report["candidate_fit_attempted"] is False
    assert report["confirmatory_validation_started"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["capital_at_risk"] is False
    rows = _read_jsonl(
        result["output_dir"] / "hts_residual_new_development_rows.jsonl"
    )
    assert len(rows) == 1
    assert rows[0]["max_input_ts"] <= rows[0]["decision_ts"]
    assert rows[0]["target_provenance"]["outcome_used_as_decision_input"] is False
    assert rows[0]["chainlink_feature_provenance"]["provenance_valid"] is True
    assert rows[0]["decision_time_features"]["action_score_margin"] >= 0.0


def test_excludes_protocol_named_smoke_corpus(tmp_path: Path) -> None:
    protocol = _protocol(collection_not_before_ts=1_000_000, minimum_markets=1)
    protocol["excluded_smoke_corpus_ids"] = ["excluded-smoke"]
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    prior_path = tmp_path / "prior.jsonl"
    _write_jsonl(prior_path, [])
    corpus_dir = _phase2_corpus(
        tmp_path,
        market_id="smoke-market",
        corpus_id="excluded-smoke",
    )

    with pytest.raises(ValueError, match="pre-protocol smoke corpus is excluded"):
        build_hts_residual_development_corpus(
            HTSResidualDevelopmentCorpusConfig(
                run_id="excluded-build",
                output_dir=tmp_path / "runs",
                protocol_path=protocol_path,
                expected_protocol_sha256=_sha256(protocol_path),
                source_corpus_dirs=(corpus_dir,),
                prior_development_rows_path=prior_path,
                paper_candidate_unlock_dir=UNLOCK_DIR,
            )
        )


def test_protocol_hash_mismatch_fails_before_scoring(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, _protocol(1_000_000, 1))
    prior_path = tmp_path / "prior.jsonl"
    _write_jsonl(prior_path, [])
    corpus_dir = _phase2_corpus(tmp_path, market_id="new-market")

    with pytest.raises(ValueError, match="protocol SHA-256 mismatch"):
        build_hts_residual_development_corpus(
            HTSResidualDevelopmentCorpusConfig(
                run_id="hash-mismatch",
                output_dir=tmp_path / "runs",
                protocol_path=protocol_path,
                expected_protocol_sha256="0" * 64,
                source_corpus_dirs=(corpus_dir,),
                prior_development_rows_path=prior_path,
                paper_candidate_unlock_dir=UNLOCK_DIR,
            )
        )


def test_forward_oof_uses_frozen_protocol_and_never_auto_freezes(
    tmp_path: Path,
) -> None:
    protocol = json.loads(
        Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_hts_residual_development_protocol_v2.json"
        ).read_text()
    )
    protocol["development_evaluation_support"]["minimum_market_count"] = 4
    protocol["development_evaluation_support"]["minimum_source_run_count"] = 4
    protocol["development_gate"]["bootstrap_samples"] = 100
    protocol_path = tmp_path / "oof-protocol.json"
    _write_json(protocol_path, protocol)
    protocol_sha256 = _sha256(protocol_path)
    rows = [
        _development_row(index, protocol_sha256=protocol_sha256)
        for index in range(4)
    ]
    rows_path = tmp_path / "development-rows.jsonl"
    _write_jsonl(rows_path, rows)
    corpus_manifest_path = tmp_path / "development-manifest.json"
    _write_json(
        corpus_manifest_path,
        {
            "protocol": {
                "path": str(protocol_path),
                "sha256": protocol_sha256,
            },
            "development_rows": {
                "path": str(rows_path),
                "sha256": _sha256(rows_path),
            },
        },
    )

    result = run_hts_residual_development_forward_oof(
        HTSResidualForwardOOFConfig(
            run_id="forward-oof",
            output_dir=tmp_path / "runs",
            protocol_path=protocol_path,
            expected_protocol_sha256=protocol_sha256,
            development_corpus_manifest_paths=(corpus_manifest_path,),
        )
    )

    report = result["report"]
    assert report["status"] == "DEVELOPMENT_OOF_COMPLETE"
    assert report["support"]["passed"] is True
    assert len(report["candidate_reports"]) == 3
    assert report["candidate_frozen"] is False
    assert report["confirmatory_validation_started"] is False
    assert report["future_confirmatory_validation_start_allowed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False


def test_forward_oof_rejects_row_protocol_lineage_mismatch(tmp_path: Path) -> None:
    protocol = json.loads(
        Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_hts_residual_development_protocol_v2.json"
        ).read_text()
    )
    protocol_path = tmp_path / "oof-protocol.json"
    _write_json(protocol_path, protocol)
    protocol_sha256 = _sha256(protocol_path)
    rows_path = tmp_path / "development-rows.jsonl"
    _write_jsonl(
        rows_path,
        [_development_row(0, protocol_sha256="f" * 64)],
    )
    corpus_manifest_path = tmp_path / "development-manifest.json"
    _write_json(
        corpus_manifest_path,
        {
            "protocol": {"path": str(protocol_path), "sha256": protocol_sha256},
            "development_rows": {
                "path": str(rows_path),
                "sha256": _sha256(rows_path),
            },
        },
    )

    with pytest.raises(ValueError, match="row protocol lineage mismatch"):
        run_hts_residual_development_forward_oof(
            HTSResidualForwardOOFConfig(
                run_id="forward-oof-lineage-mismatch",
                output_dir=tmp_path / "runs",
                protocol_path=protocol_path,
                expected_protocol_sha256=protocol_sha256,
                development_corpus_manifest_paths=(corpus_manifest_path,),
            )
        )


def test_confirmatory_input_freeze_and_evaluation_are_exactly_once(
    tmp_path: Path,
) -> None:
    candidate = _candidate_freeze(tmp_path)
    source_root = tmp_path / "confirmatory-source"
    source_root.mkdir()
    corpus = _phase2_corpus(
        source_root,
        market_id="confirmatory-market",
        corpus_id="confirmatory-corpus",
        market_start_ts=5_500_000,
    )
    frozen_input = freeze_hts_residual_confirmatory_input(
        HTSResidualConfirmatoryInputConfig(
            run_id="confirmatory-input",
            output_dir=tmp_path / "runs",
            candidate_freeze_manifest_path=candidate["manifest_path"],
            source_corpus_dirs=(corpus,),
        )
    )
    assert frozen_input["manifest"]["input_gate_passed"] is True
    assert frozen_input["manifest"]["outcome_values_inspected_during_input_freeze"] is False

    evaluation_config = HTSResidualConfirmatoryEvaluationConfig(
        confirmatory_input_manifest_path=frozen_input["manifest_path"],
        paper_candidate_unlock_dir=UNLOCK_DIR,
    )
    result = evaluate_hts_residual_confirmatory_once(evaluation_config)
    report = result["report"]
    assert report["status"] == "PRE_PROMOTION_BLOCKED"
    assert report["confirmatory_labels_used_for_fitting"] is False
    assert report["confirmatory_labels_used_for_evaluation_only"] is True
    assert report["pre_promotion_ready"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["capital_at_risk"] is False
    with pytest.raises(FileExistsError, match="exactly-once"):
        evaluate_hts_residual_confirmatory_once(evaluation_config)


def test_confirmatory_input_overlap_blocks_before_evaluation_marker(
    tmp_path: Path,
) -> None:
    candidate = _candidate_freeze(tmp_path)
    source_root = tmp_path / "overlap-source"
    source_root.mkdir()
    corpus = _phase2_corpus(
        source_root,
        market_id="market-0",
        corpus_id="overlap-corpus",
        market_start_ts=5_500_000,
    )
    frozen_input = freeze_hts_residual_confirmatory_input(
        HTSResidualConfirmatoryInputConfig(
            run_id="overlap-input",
            output_dir=tmp_path / "runs",
            candidate_freeze_manifest_path=candidate["manifest_path"],
            source_corpus_dirs=(corpus,),
        )
    )
    manifest = frozen_input["manifest"]
    assert manifest["input_gate_passed"] is False
    assert "confirmatory_market_overlap_detected" in manifest["blocking_reason_codes"]
    with pytest.raises(ValueError, match="input gate did not pass"):
        evaluate_hts_residual_confirmatory_once(
            HTSResidualConfirmatoryEvaluationConfig(
                confirmatory_input_manifest_path=frozen_input["manifest_path"],
                paper_candidate_unlock_dir=UNLOCK_DIR,
            )
        )
    assert not (
        frozen_input["output_dir"]
        / "hts_residual_confirmatory_evaluation_started.json"
    ).exists()


def test_confirmatory_source_tamper_fails_before_exactly_once_marker(
    tmp_path: Path,
) -> None:
    candidate = _candidate_freeze(tmp_path)
    source_root = tmp_path / "tamper-source"
    source_root.mkdir()
    corpus = _phase2_corpus(
        source_root,
        market_id="tamper-market",
        corpus_id="tamper-corpus",
        market_start_ts=5_500_000,
    )
    frozen_input = freeze_hts_residual_confirmatory_input(
        HTSResidualConfirmatoryInputConfig(
            run_id="tamper-input",
            output_dir=tmp_path / "runs",
            candidate_freeze_manifest_path=candidate["manifest_path"],
            source_corpus_dirs=(corpus,),
        )
    )
    with (corpus / "polymarket_label_rows.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="descriptor hash mismatch"):
        evaluate_hts_residual_confirmatory_once(
            HTSResidualConfirmatoryEvaluationConfig(
                confirmatory_input_manifest_path=frozen_input["manifest_path"],
                paper_candidate_unlock_dir=UNLOCK_DIR,
            )
        )
    assert not (
        frozen_input["output_dir"]
        / "hts_residual_confirmatory_evaluation_started.json"
    ).exists()


def test_confirmatory_input_duplicate_market_is_fail_closed(tmp_path: Path) -> None:
    candidate = _candidate_freeze(tmp_path)
    source_root = tmp_path / "duplicate-market-source"
    source_root.mkdir()
    corpora = tuple(
        _phase2_corpus(
            source_root,
            market_id="duplicate-market",
            corpus_id=f"duplicate-corpus-{index}",
            market_start_ts=5_500_000 + index * 300_000,
        )
        for index in range(2)
    )
    frozen_input = freeze_hts_residual_confirmatory_input(
        HTSResidualConfirmatoryInputConfig(
            run_id="duplicate-market-input",
            output_dir=tmp_path / "runs",
            candidate_freeze_manifest_path=candidate["manifest_path"],
            source_corpus_dirs=corpora,
        )
    )
    manifest = frozen_input["manifest"]
    assert manifest["input_gate_passed"] is False
    assert manifest["duplicate_source_market_ids"] == ["duplicate-market"]
    assert (
        "duplicate_confirmatory_source_market" in manifest["blocking_reason_codes"]
    )


def _protocol(collection_not_before_ts: int, minimum_markets: int) -> dict:
    return {
        "schema_version": "bigan-v8-hts-residual-development-protocol-v2",
        "protocol_frozen_before_new_development_collection": True,
        "uses_validation_labels_for_tuning": False,
        "future_confirmatory_validation_start_allowed": False,
        "collection_not_before_ts": collection_not_before_ts,
        "excluded_smoke_corpus_ids": [],
        "development_evaluation_support": {
            "minimum_market_count": minimum_markets,
        },
    }


def _development_row(index: int, *, protocol_sha256: str) -> dict:
    side = "UP" if index % 2 == 0 else "DOWN"
    outcome = "UP" if index in {0, 3} else "DOWN"
    probability = 0.62 if side == "UP" else 0.58
    features = {
        "canonical_o_action_score": 0.2 + index * 0.01,
        "action_score_margin": 0.05,
        "btc_momentum": 0.001 * (1 if side == "UP" else -1),
        "reference_price_to_beat_distance_at_decision": 0.001,
        "chainlink_momentum_30s": 0.001,
        "chainlink_momentum_60s": 0.0012,
        "chainlink_momentum_120s": 0.0008,
        "chainlink_realized_volatility_120s": 0.0002,
        "selected_side_probability": probability,
        "execution_price": probability - 0.01,
        "selected_side_probability_minus_execution_price": 0.01,
        "spread_bps": 100.0,
        "queue_fill_proxy": 0.9,
        "book_staleness_ms": 100.0,
        "time_to_close_seconds": 180.0,
        "side_book_depth_imbalance": 0.1,
        "side_book_update_count_1m": 20.0,
        "side_recent_spread_stability_1m": 0.9,
        "cumulative_market_exposure_before_entry": 0.0,
        "same_side_reentry": 0.0,
        "side_flip": 0.0,
    }
    return {
        "market_id": f"market-{index}",
        "condition_id": f"condition-{index}",
        "market_slug": f"btc-updown-5m-{index}",
        "decision_ts": 2_000_000 + index * 300_000,
        "max_input_ts": 1_999_900 + index * 300_000,
        "selected_action": f"BUY_{side}_HOLD_TO_SETTLEMENT",
        "selected_side": side,
        "action_family": "HOLD_TO_SETTLEMENT",
        "decision_time_features": features,
        "selected_side_win_target": int(side == outcome),
        "target_provenance": {"resolved_outcome": outcome},
        "source_run_id": f"run-{index}",
        "source_lineage": {
            "frozen_development_protocol_sha256": protocol_sha256,
        },
        "lineage": "post_protocol_development_only",
        "row_identity": f"row-{index}",
        "row_content_sha256": hashlib.sha256(f"row-{index}".encode()).hexdigest(),
    }


def _candidate_freeze(tmp_path: Path) -> dict:
    protocol_path = tmp_path / "development-protocol.json"
    _write_json(protocol_path, {"frozen": True})
    protocol_sha256 = _sha256(protocol_path)
    rows = [
        _development_row(index, protocol_sha256=protocol_sha256)
        for index in range(4)
    ]
    rows_path = tmp_path / "development-rows.jsonl"
    _write_jsonl(rows_path, rows)
    spec = {
        "candidate_name": "hts_residual_chainlink_rank_anchor_offset",
        "feature_names": [
            "canonical_o_action_score",
            "action_score_margin",
            "chainlink_anchor_alignment",
        ],
        "maximum_absolute_residual_coefficient": 3.0,
        "regularization": 35.0,
        "probability_bounds": [0.01, 0.99],
    }
    contract = fit_residual_offset_contract(rows, spec)
    report_path = tmp_path / "development-oof-report.json"
    _write_json(
        report_path,
        {
            "development_candidate_gate_passed": True,
            "selected_candidate_name": spec["candidate_name"],
            "selected_candidate_contract": contract,
        },
    )
    manifest_path = tmp_path / "development-oof-manifest.json"
    _write_json(
        manifest_path,
        {
            "development_candidate_gate_passed": True,
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "combined_rows": {"path": str(rows_path), "sha256": _sha256(rows_path)},
            "protocol": {
                "path": str(protocol_path),
                "sha256": protocol_sha256,
            },
        },
    )
    return freeze_hts_residual_candidate(
        HTSResidualCandidateFreezeConfig(
            run_id="candidate-freeze",
            output_dir=tmp_path / "runs",
            development_oof_manifest_path=manifest_path,
            freeze_created_at="1970-01-01T01:06:40+00:00",
            freeze_created_ts=4_000_000,
            minimum_confirmatory_market_count=1,
            minimum_confirmatory_source_run_count=1,
            minimum_input_source_market_count=1,
            bootstrap_samples=20,
        )
    )


def _phase2_corpus(
    root: Path,
    *,
    market_id: str,
    corpus_id: str = "post-protocol-corpus",
    market_start_ts: int = 2_000_000,
) -> Path:
    corpus_dir = root / corpus_id
    corpus_dir.mkdir()
    decision_ts = market_start_ts + 60_000
    market_end_ts = market_start_ts + 300_000
    features = {
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
    }
    feature_rows = [
        {
            "market_id": market_id,
            "condition_id": market_id,
            "slug": "btc-updown-5m-2000",
            "decision_ts": decision_ts,
            "max_input_ts": decision_ts - 100,
            "features": features,
            "feature_provenance": {
                "btc_mid_price": {"input_end_ts": decision_ts - 100}
            },
        }
    ]
    label_rows = [
        {
            "market_id": market_id,
            "decision_ts": decision_ts,
            "action": action,
            "total_net_return": 0.4 if action == "BUY_UP_HOLD_TO_SETTLEMENT" else 0.0,
        }
        for action in (
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
            "BUY_UP_HOLD_TO_SETTLEMENT",
            "BUY_DOWN_HOLD_TO_SETTLEMENT",
            "NO_TRADE",
        )
    ]
    metadata = [
        {
            "market_id": market_id,
            "condition_id": market_id,
            "slug": "btc-updown-5m-2000",
            "market_family": "btc_updown_5m",
            "market_start_ts": market_start_ts,
            "market_end_ts": market_end_ts,
            "horizon_ms": 300_000,
            "reference_price_start": None,
        }
    ]
    resolutions = [
        {
            "market_id": market_id,
            "resolved_outcome": "UP",
            "raw_resolution_sha256": "a" * 64,
        }
    ]
    chainlink_rows = [
        _chainlink_row(market_start_ts - 120_000, 65_000.0),
        _chainlink_row(market_start_ts, 65_020.0),
        _chainlink_row(decision_ts - 30_000, 65_080.0),
        _chainlink_row(decision_ts - 1_000, 65_130.0),
    ]
    _write_jsonl(corpus_dir / "polymarket_feature_rows.jsonl", feature_rows)
    _write_jsonl(corpus_dir / "polymarket_label_rows.jsonl", label_rows)
    _write_jsonl(corpus_dir / "polymarket_market_metadata.jsonl", metadata)
    _write_jsonl(corpus_dir / "polymarket_resolution_events.jsonl", resolutions)
    _write_jsonl(corpus_dir / "polymarket_chainlink_prices.jsonl", chainlink_rows)
    normalized_hashes = {
        "feature_rows": _sha256(corpus_dir / "polymarket_feature_rows.jsonl"),
        "label_rows": _sha256(corpus_dir / "polymarket_label_rows.jsonl"),
        "market_metadata": _sha256(corpus_dir / "polymarket_market_metadata.jsonl"),
        "resolution_events": _sha256(
            corpus_dir / "polymarket_resolution_events.jsonl"
        ),
    }
    _write_json(
        corpus_dir / "polymarket_corpus_manifest.json",
        {"normalized_artifact_hashes": normalized_hashes},
    )
    _write_json(
        corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json",
        {
            "evidence_sha256": _sha256(
                corpus_dir / "polymarket_chainlink_prices.jsonl"
            ),
            "timestamp_causality_violation_count": 0,
        },
    )
    _write_json(
        corpus_dir / "training_corpus_provenance.json",
        {"corpus_id": corpus_id, "run_id": f"{corpus_id}-run"},
    )
    return corpus_dir


def _chainlink_row(source_ts: int, price: float) -> dict:
    return {
        "source_ts": source_ts,
        "available_at_ts": source_ts + 100,
        "price": price,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
