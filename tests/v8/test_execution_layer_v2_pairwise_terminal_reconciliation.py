from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    FORBIDDEN_REGISTRY_FIELDS,
    _find_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_precollection_support import (
    _batch_preflight,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_terminal_reconciliation import (
    PairwiseTerminalReconciliationConfig,
    run_pairwise_terminal_reconciliation,
)


def test_stale_pending_finalization_reconciles_immutably(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_before = fixture["batch"].read_bytes()

    result = _run(tmp_path, fixture)

    report = result["report"]
    reconciled = json.loads(
        result["reconciled_batch_progress_paths"][0].read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "TERMINAL_RECONCILIATION_READY"
    assert report["applied_terminal_finalization_count"] == 1
    assert report["reconciled_exported_finalization_count"] == 1
    assert report["resolution_payloads_opened"] is False
    assert report["labels_or_outcomes_opened_for_reconciliation"] is False
    assert fixture["batch"].read_bytes() == source_before
    assert reconciled["captures"] == fixture["batch_payload"]["captures"]
    assert reconciled["finalizations"][0]["finalization_status"] == "exported"
    assert reconciled["finalizations"][0]["training_eligible"] is True
    assert reconciled["terminal_reconciliation"][
        "source_capture_rows_mutated"
    ] is False
    assert not _find_fields(reconciled, FORBIDDEN_REGISTRY_FIELDS)
    _, _, preflight = _batch_preflight(
        result["reconciled_batch_progress_pins"]
    )
    assert preflight["blocking_reason_codes"] == []


def test_raw_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["round_dir"] / "raw" / "raw_market.jsonl").write_text(
        "tampered\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, fixture)

    assert result["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert "terminal_finalization_evidence_rejected" in result[
        "report"
    ]["blocking_reason_codes"]
    assert result["report"]["rejection_reason_distribution"] == {
        "terminal_finalization_raw_artifact_hash_mismatch": 1
    }


def test_mismatched_run_identity_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(
        fixture["report"].read_text(encoding="utf-8")
    )
    report["run_id"] = "wrong-run"
    fixture["report"].write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    result = _run(tmp_path, fixture)

    assert result["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert result["report"]["rejection_reason_distribution"] == {
        "terminal_finalization_run_id_mismatch": 1
    }


def test_missing_terminal_artifact_for_quality_capture_blocks(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["report"].unlink()

    result = _run(tmp_path, fixture)

    assert result["report"]["status"] == "BLOCKED_FAIL_CLOSED"
    assert "quality_capture_terminal_finalization_missing" in result[
        "report"
    ]["blocking_reason_codes"]


def test_capture_quality_failure_does_not_fabricate_finalization(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    batch = json.loads(fixture["batch"].read_text(encoding="utf-8"))
    batch["captures"][0]["raw_polymarket_market_count"] = 0
    batch["captures"][0]["provider_raw_orderbook_snapshot_count"] = 0
    batch["captures"][0]["training_sampled_orderbook_row_count"] = 0
    fixture["batch"].write_text(json.dumps(batch), encoding="utf-8")
    fixture["report"].unlink()
    fixture["batch_payload"] = batch

    result = _run(tmp_path, fixture)

    report = result["report"]
    assert report["status"] == "TERMINAL_RECONCILIATION_READY"
    assert report["applied_terminal_finalization_count"] == 0
    assert (
        report[
            "capture_quality_failed_no_reconciliation_required_count"
        ]
        == 1
    )


def test_source_pin_mismatch_raises_before_reconciliation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    config = _config(tmp_path, fixture, batch_sha="0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_pairwise_terminal_reconciliation(config)


def test_safety_and_manifest_hashes_are_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(
        fixture["manifest"].read_text(encoding="utf-8")
    )
    manifest["wallet_signing_enabled"] = True
    fixture["manifest"].write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    result = _run(tmp_path, fixture)

    report = result["report"]
    assert report["status"] == "BLOCKED_FAIL_CLOSED"
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


def _run(tmp_path: Path, fixture: dict) -> dict:
    return run_pairwise_terminal_reconciliation(
        _config(tmp_path, fixture)
    )


def _config(
    tmp_path: Path,
    fixture: dict,
    *,
    batch_sha: str | None = None,
) -> PairwiseTerminalReconciliationConfig:
    return PairwiseTerminalReconciliationConfig(
        run_id="terminal-reconciliation",
        output_dir=tmp_path / "runs",
        precollection_freeze_manifest_path=fixture["freeze"],
        expected_precollection_freeze_manifest_sha256=_sha256(
            fixture["freeze"]
        ),
        batch_progress_pins=(
            (
                fixture["batch"],
                batch_sha or _sha256(fixture["batch"]),
            ),
        ),
        training_corpus_root=fixture["training_root"],
    )


def _fixture(tmp_path: Path) -> dict:
    training_root = tmp_path / "training"
    corpus = training_root / "market-1"
    corpus.mkdir(parents=True)
    corpus_manifest = corpus / "polymarket_corpus_manifest.json"
    corpus_manifest.write_text(
        json.dumps({"paper_only": True, "capital_at_risk": False}),
        encoding="utf-8",
    )
    collection = tmp_path / "collection"
    batch_dir = collection / "batch-1"
    round_dir = collection / "batch-1-round-001"
    batch_dir.mkdir(parents=True)
    (round_dir / "raw").mkdir(parents=True)
    (round_dir / "provider_raw").mkdir(parents=True)
    raw = round_dir / "raw" / "raw_market.jsonl"
    provider_raw = round_dir / "provider_raw" / "raw_market.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    provider_raw.write_text("{}\n", encoding="utf-8")
    capture_manifest = round_dir / "pending_round_capture_manifest.json"
    capture_manifest.write_text(
        json.dumps(
            {
                "run_id": round_dir.name,
                **_safe_fields(),
            }
        ),
        encoding="utf-8",
    )
    (corpus / "training_corpus_provenance.json").write_text(
        json.dumps(
            {
                "run_id": round_dir.name,
                "pending_capture_manifest_path": str(capture_manifest),
                **_safe_fields(),
            }
        ),
        encoding="utf-8",
    )
    report = round_dir / "pending_round_finalization_report.json"
    report.write_text(
        json.dumps(
            {
                "run_id": round_dir.name,
                "finalization_status": "exported",
                "pending_resolution": False,
                "training_eligible": True,
                "raw_resolution_count": 1,
                "reject_reason_counts": {},
                "resolution_provider_called": True,
                "phase2_corpus_built": True,
                "exported_training_corpus_dir": str(corpus),
                "raw_btc_candle_row_count": 1,
                "pending_feature_enrichment": False,
                "feature_enrichment_attempt_count": 0,
                "feature_enrichment_post_market_close_candle_rejected_count": 0,
                "feature_enrichment_reason_codes": [],
                "feature_enrichment_recovered": False,
                "feature_enrichment_warning_reason_codes": [],
                **_safe_fields(),
            }
        ),
        encoding="utf-8",
    )
    manifest = round_dir / "pending_round_finalization_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": round_dir.name,
                "config": {
                    "run_id": round_dir.name,
                    "output_dir": str(collection),
                },
                "finalization_status": "exported",
                "pending_resolution": False,
                "phase2_corpus_built": True,
                "provider_raw_artifacts_preserved": True,
                "exported_training_corpus_dir": str(corpus),
                "phase2_corpus_manifest_sha256": _sha256(corpus_manifest),
                "raw_artifact_hashes": {
                    raw.name: _sha256(raw),
                },
                "provider_raw_artifact_hashes": {
                    provider_raw.name: _sha256(provider_raw),
                },
                **_safe_fields(),
            }
        ),
        encoding="utf-8",
    )
    capture = {
        "run_id": round_dir.name,
        "run_dir": str(round_dir),
        "capture_start_boundary_validation_passed": True,
        "scheduled_round_start_ts": 1,
        "raw_polymarket_market_count": 1,
        "provider_raw_orderbook_snapshot_count": 1,
        "training_sampled_orderbook_row_count": 1,
        "raw_btc_candle_row_count": 1,
        "raw_chainlink_price_row_count": 1,
        "orderbook_snapshot_interval_seconds": 1.0,
        "public_provider_timeout_seconds": 330.0,
        "public_provider_http_timeout_seconds": 5.0,
        "orderbook_ws_initial_complete_book_timeout_seconds": 15.0,
        "rest_orderbook_fallback_collection_seconds": 330.0,
        "rest_orderbook_fallback_stops_at_market_close": True,
        "gamma_market_identity_prefetch_round_count": 12,
        "market_identity_cache_max_age_seconds": 7200.0,
        "market_identity_cache_path": str(collection / "cache.json"),
        "clob_identity_revalidation_max_attempts": 3,
        "clob_identity_revalidation_retry_seconds": 0.25,
        "feature_enrichment_max_attempts": 40,
        "market_identity_cache_provenance_violation_count": 0,
        "market_identity_cache_fallback_market_count": 0,
        "provider_raw_market_identity_source_type_distribution": {
            "gamma_primary": 1,
        },
        "market_identity_cache_report": {
            "cache_enabled": True,
            "cache_payload_sha256": "a" * 64,
        },
    }
    pending = {
        "run_id": round_dir.name,
        "run_dir": str(round_dir),
        "finalization_status": "pending_resolution",
        "pending_resolution": True,
        "training_eligible": False,
        "raw_resolution_count": 0,
        "reject_reason_counts": {"missing_resolution": 1},
        "exported_training_corpus_dir": None,
    }
    batch_payload = {
        "batch_id": "batch-1",
        "capture_count": 1,
        "captures": [capture],
        "finalizations": [pending],
        "exported_round_count": 0,
        "pending_resolution_count": 1,
        "error_count": 0,
        **_safe_fields(),
    }
    batch = batch_dir / "batch_progress.json"
    batch.write_text(json.dumps(batch_payload), encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "collector_contract": {
                    "training_corpus_root": str(training_root),
                    "orderbook_snapshot_interval_seconds": 1.0,
                    "public_provider_timeout_seconds": 330.0,
                    "public_provider_http_timeout_seconds": 5.0,
                    "orderbook_ws_initial_complete_book_timeout_seconds": 15.0,
                    "rest_orderbook_fallback_collection_seconds": 330.0,
                    "gamma_market_identity_prefetch_round_count": 12,
                    "market_identity_cache_max_age_seconds": 7200.0,
                    "market_identity_cache_clob_revalidation_max_attempts": 3,
                    "market_identity_cache_clob_revalidation_retry_seconds": 0.25,
                    "feature_enrichment_max_attempts": 40,
                },
                **_safe_fields(),
            }
        ),
        encoding="utf-8",
    )
    return {
        "training_root": training_root,
        "corpus": corpus,
        "round_dir": round_dir,
        "capture_manifest": capture_manifest,
        "report": report,
        "manifest": manifest,
        "batch": batch,
        "batch_payload": batch_payload,
        "freeze": freeze,
    }


def _safe_fields() -> dict:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_exchange_write_enabled": False,
        "broker_exchange_write_enabled": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
