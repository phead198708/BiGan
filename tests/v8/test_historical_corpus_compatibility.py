"""Tests for the outcome-blind historical corpus compatibility audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.training.historical_corpus_compatibility import (
    HISTORICAL_DEVELOPMENT_COMPATIBLE,
    HISTORICAL_DEVELOPMENT_CONVERTIBLE,
    HISTORICAL_INCOMPATIBLE,
    HistoricalCorpusCompatibilityAuditConfig,
    run_historical_corpus_compatibility_audit,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)


def test_complete_history_is_development_only_compatible(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _build_corpus(corpus_root, "btc-updown-5m-1000", "market-a")

    result = _run(tmp_path, corpus_root)

    report = result["report"]
    assert report["historical_development_compatible_market_count"] == 1
    assert report["fresh_confirmatory_eligible_market_count"] == 0
    assert len(report["input_inventory_hash"]) == 64
    assert report["input_inventory_entry_count"] == 1
    row = _rows(result)[0]
    assert row["classification"] == HISTORICAL_DEVELOPMENT_COMPATIBLE
    assert row["historical_development_fit_eligible"] is True
    assert row["fresh_calibration_eligible"] is False
    assert row["fresh_confirmatory_eligible"] is False
    assert row["outcome_values_loaded"] is False
    assert row["pnl_values_loaded"] is False


def test_missing_book_stream_fails_closed(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _build_corpus(
        corpus_root,
        "btc-updown-5m-1000",
        "market-a",
        omitted={"polymarket_token_book_snapshots.jsonl"},
    )

    row = _rows(_run(tmp_path, corpus_root))[0]

    assert row["classification"] == HISTORICAL_INCOMPATIBLE
    assert (
        "required_file_missing:polymarket_token_book_snapshots.jsonl"
        in row["reason_codes"]
    )


def test_empty_stream_is_not_counted_as_evidence(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _build_corpus(
        corpus_root,
        "btc-updown-5m-1000",
        "market-a",
        empty={"polymarket_token_trades.jsonl"},
    )

    row = _rows(_run(tmp_path, corpus_root))[0]

    assert row["classification"] == HISTORICAL_INCOMPATIBLE
    assert "required_stream_empty:polymarket_token_trades.jsonl" in row["reason_codes"]


def test_feature_timestamp_leakage_fails_closed(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"

    def leak(row: dict[str, Any]) -> None:
        row["max_input_ts"] = row["decision_ts"] + 1

    _build_corpus(
        corpus_root,
        "btc-updown-5m-1000",
        "market-a",
        feature_mutator=leak,
    )

    row = _rows(_run(tmp_path, corpus_root))[0]

    assert row["classification"] == HISTORICAL_INCOMPATIBLE
    assert "feature_timestamp_causality_violation" in row["reason_codes"]


def test_missing_derived_runtime_field_is_convertible_only(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"

    def remove_queue(row: dict[str, Any]) -> None:
        del row["features"]["up_queue_fill_probability_proxy"]

    _build_corpus(
        corpus_root,
        "btc-updown-5m-1000",
        "market-a",
        feature_mutator=remove_queue,
    )

    row = _rows(_run(tmp_path, corpus_root))[0]

    assert row["classification"] == HISTORICAL_DEVELOPMENT_CONVERTIBLE
    assert row["historical_development_fit_eligible"] is False
    assert row["historical_development_rebuild_candidate"] is True
    assert "current_runtime_feature_fields_missing" in row["reason_codes"]


def test_duplicate_market_does_not_inflate_support(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _build_corpus(corpus_root, "btc-updown-5m-1000", "market-a")
    _build_corpus(corpus_root / "nested", "btc-updown-5m-1000-copy", "market-a")

    result = _run(tmp_path, corpus_root)
    rows = _rows(result)

    assert result["report"]["discovered_corpus_count"] == 2
    assert result["report"]["unique_market_count"] == 1
    assert result["report"]["duplicate_excluded_corpus_count"] == 1
    assert sum(row["deduplication_status"] == "duplicate_excluded" for row in rows) == 1


def test_forbidden_outcome_field_in_feature_fails_closed(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"

    def add_forbidden(row: dict[str, Any]) -> None:
        row["features"]["settlement_pnl"] = 1.0

    _build_corpus(
        corpus_root,
        "btc-updown-5m-1000",
        "market-a",
        feature_mutator=add_forbidden,
    )

    row = _rows(_run(tmp_path, corpus_root))[0]

    assert row["classification"] == HISTORICAL_INCOMPATIBLE
    assert (
        "forbidden_decision_fields_present:polymarket_feature_rows.jsonl"
        in row["reason_codes"]
    )


def test_old_label_schema_remains_incompatible(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _build_corpus(
        corpus_root,
        "btc-updown-5m-1000",
        "market-a",
        current_label_schema=False,
    )

    row = _rows(_run(tmp_path, corpus_root))[0]

    assert row["classification"] == HISTORICAL_INCOMPATIBLE
    assert "current_cost_aware_label_contract_not_identified" in row["reason_codes"]


def test_outputs_are_deterministic_and_safety_remains_blocked(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _build_corpus(corpus_root, "btc-updown-5m-1000", "market-a")
    first = _run(tmp_path, corpus_root)
    first_hash = _sha256(first["manifest_path"])
    second = _run(tmp_path, corpus_root, overwrite=True)

    assert _sha256(second["manifest_path"]) == first_hash
    report = second["report"]
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False
    assert report["polymarket_write_enabled"] is False
    assert report["wallet_signing_enabled"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    access = report["outcome_blind_access_audit"]
    assert access["label_rows_content_parsed"] is False
    assert access["resolution_rows_content_parsed"] is False
    assert access["outcome_values_loaded"] is False
    assert access["pnl_values_loaded"] is False


def _run(
    tmp_path: Path,
    corpus_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    return run_historical_corpus_compatibility_audit(
        HistoricalCorpusCompatibilityAuditConfig(
            run_id="historical-audit",
            corpus_root=corpus_root,
            output_dir=tmp_path / "runs",
            protocol_path=PROTOCOL_PATH,
            feature_contract_path=FEATURE_CONTRACT_PATH,
            overwrite_existing=overwrite,
        )
    )


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in result["rows_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_corpus(
    root: Path,
    slug: str,
    market_id: str,
    *,
    feature_mutator: Callable[[dict[str, Any]], None] | None = None,
    omitted: set[str] | None = None,
    empty: set[str] | None = None,
    current_label_schema: bool = True,
) -> Path:
    omitted = omitted or set()
    empty = empty or set()
    corpus_dir = root / slug
    corpus_dir.mkdir(parents=True)
    decision_ts = 1_000_000
    features = _feature_values()
    feature_row = {
        "market_id": market_id,
        "condition_id": market_id,
        "slug": slug,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "available_at_ts": decision_ts,
        "feature_provenance": {
            "reference_price_to_beat_distance_at_decision": {
                "reference_price_to_beat_source": (
                    "polymarket_rtds_chainlink_market_start"
                ),
                "source_fields_used": (
                    "raw_polymarket_chainlink_prices."
                    "price_at_or_before_market_start+"
                    "raw_polymarket_chainlink_prices.price_at_or_before_decision"
                ),
                "provenance_valid": True,
                "max_input_ts": decision_ts,
                "available_at_ts": decision_ts,
            }
        },
        "features": features,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    if feature_mutator:
        feature_mutator(feature_row)
    payloads: dict[str, list[dict[str, Any]]] = {
        "polymarket_market_metadata.jsonl": [
            {
                "market_id": market_id,
                "condition_id": market_id,
                "slug": slug,
                "market_start_ts": 900_000,
                "market_end_ts": 1_200_000,
            }
        ],
        "polymarket_feature_rows.jsonl": [feature_row],
        "polymarket_token_book_snapshots.jsonl": [
            _book_row(market_id, "UP", decision_ts),
            _book_row(market_id, "DOWN", decision_ts),
        ],
        "polymarket_token_trades.jsonl": [
            {"market_id": market_id, "outcome": "UP", "price": 0.5, "ts": decision_ts}
        ],
        "polymarket_btc_reference_candles.jsonl": [
            {
                "source": "coinbase",
                "ts": decision_ts - 60_000,
                "available_at_ts": decision_ts,
                "open_price": 100.0,
                "high_price": 101.0,
                "low_price": 99.0,
                "close_price": 100.5,
                "volume": 1.0,
            }
        ],
        "polymarket_chainlink_prices.jsonl": [
            {
                "source_type": "polymarket_rtds_chainlink",
                "source_ts": decision_ts - 1,
                "available_at_ts": decision_ts,
                "price": 100.5,
                "read_only": True,
            }
        ],
        "polymarket_label_rows.jsonl": [{"opaque_label_row": index} for index in range(5)],
        "polymarket_resolution_events.jsonl": [{"opaque_resolution_row": True}],
    }
    for filename, rows in payloads.items():
        if filename in omitted:
            continue
        path = corpus_dir / filename
        if filename in empty:
            path.write_text("", encoding="utf-8")
        else:
            _write_jsonl(path, rows)
    chainlink_manifest = {
        "schema_version": "bigan-v8-polymarket-chainlink-decision-time-evidence-v2",
        "source_type": "polymarket_rtds_chainlink",
        "decision_time_only": True,
        "feature_builder_integration_passed": True,
        "feature_builder_integration_required": False,
        "timestamp_causality_violation_count": 0,
        "integrated_feature_row_count": 1,
        "missing_or_invalid_feature_row_count": 0,
        "row_count": 1,
        "evidence_sha256": _sha256(
            corpus_dir / "polymarket_chainlink_prices.jsonl"
        ),
    }
    if "polymarket_chainlink_decision_time_evidence_manifest.json" not in omitted:
        _write_json(
            corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json",
            chainlink_manifest,
        )
    normalized_hashes = {}
    hash_keys = {
        "polymarket_market_metadata.jsonl": "market_metadata",
        "polymarket_feature_rows.jsonl": "feature_rows",
        "polymarket_token_book_snapshots.jsonl": "token_book_snapshots",
        "polymarket_token_trades.jsonl": "token_trades",
        "polymarket_btc_reference_candles.jsonl": "btc_reference_candles",
        "polymarket_chainlink_prices.jsonl": "chainlink_prices",
        "polymarket_chainlink_decision_time_evidence_manifest.json": (
            "chainlink_decision_time_evidence_manifest"
        ),
        "polymarket_label_rows.jsonl": "label_rows",
        "polymarket_resolution_events.jsonl": "resolution_events",
    }
    for filename, key in hash_keys.items():
        path = corpus_dir / filename
        if path.is_file():
            normalized_hashes[key] = _sha256(path)
    manifest = {
        "schema_version": (
            "bigan-v8-polymarket-corpus-v3"
            if current_label_schema
            else "bigan-v8-polymarket-corpus-v2"
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "feature_row_count": 1,
        "label_row_count": 5,
        "normalized_artifact_hashes": normalized_hashes,
        "raw_artifact_hashes": {
            "raw_polymarket_markets.jsonl": "a" * 64,
            "raw_polymarket_orderbooks.jsonl": "b" * 64,
            "raw_polymarket_trades.jsonl": "c" * 64,
            "raw_polymarket_resolutions.jsonl": "d" * 64,
            "raw_polymarket_chainlink_prices.jsonl": "e" * 64,
            "raw_coinbase_btc_candles.jsonl": "f" * 64,
        },
        "chainlink_decision_time_feature_integration": chainlink_manifest,
        "sell_before_close_label_schema_version": (
            "bigan-v8-polymarket-sell-before-close-executable-exit-v1"
            if current_label_schema
            else None
        ),
        "sell_before_close_label_gate_passed": current_label_schema,
    }
    _write_json(corpus_dir / "polymarket_corpus_manifest.json", manifest)
    provenance = {
        "corpus_id": slug,
        "round_slug": slug,
        "real_historical_corpus_used": True,
        "manual_live_evidence_eligible": True,
        "synthetic_corpus_used": False,
        "synthetic_public_data_used": False,
        "mock_public_data_used": False,
        "round_scoped_export": True,
        "phase2_corpus_manifest_sha256": _sha256(
            corpus_dir / "polymarket_corpus_manifest.json"
        ),
        "chainlink_decision_time_evidence": {
            "attached": True,
            "feature_builder_integration_passed": True,
            "feature_builder_integration_required": False,
            "evidence_sha256": _sha256(
                corpus_dir / "polymarket_chainlink_prices.jsonl"
            ),
            "manifest_sha256": _sha256(
                corpus_dir
                / "polymarket_chainlink_decision_time_evidence_manifest.json"
            ),
        },
    }
    if "training_corpus_provenance.json" not in omitted:
        _write_json(corpus_dir / "training_corpus_provenance.json", provenance)
    return corpus_dir


def _feature_values() -> dict[str, float]:
    values = {
        "btc_return_10s": 0.0,
        "btc_return_30s": 0.0,
        "btc_return_1m": 0.0,
        "btc_return_5m": 0.0,
        "btc_return_15m": 0.0,
        "btc_volatility_1m": 0.01,
        "btc_volatility_5m": 0.01,
        "btc_volatility_15m": 0.01,
        "reference_price_to_beat_distance_at_decision": 0.005,
        "time_to_close_seconds": 120.0,
        "market_age_seconds": 180.0,
        "combined_spread_bps": 100.0,
        "liquidity_imbalance": 0.0,
        "recent_up_trade_volume": 1.0,
        "recent_down_trade_volume": 1.0,
    }
    for side in ("up", "down"):
        values.update(
            {
                f"{side}_bid": 0.49,
                f"{side}_ask": 0.51,
                f"{side}_spread_bps": 400.0,
                f"{side}_queue_fill_probability_proxy": 0.75,
                f"{side}_book_staleness_ms": 0.0,
                f"{side}_liquidity_depth": 100.0,
                f"{side}_executable_ask_notional": 50.0,
                f"{side}_executable_bid_notional": 50.0,
                f"{side}_recent_book_update_count_1m": 10.0,
                f"{side}_recent_spread_stability_1m": 1.0,
                f"{side}_recent_bid_depth_volatility_1m": 0.1,
            }
        )
    return values


def _book_row(market_id: str, outcome: str, ts: int) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "outcome": outcome,
        "token_id": f"{market_id}-{outcome}",
        "ts": ts,
        "available_at_ts": ts,
        "bid_price": 0.49,
        "ask_price": 0.51,
        "bid_size": 100.0,
        "ask_size": 100.0,
        "liquidity_depth": 200.0,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
