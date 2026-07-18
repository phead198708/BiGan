from __future__ import annotations

import hashlib
import json
import plistlib
import sys
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    OutcomeBlindWindowFreezeConfig,
    PersistentOutcomeBlindBatchIndexConfig,
    freeze_outcome_blind_window,
    index_persistent_outcome_blind_batch,
    load_and_validate_persistent_outcome_blind_index,
    validate_persistent_outcome_blind_collector_protocol,
)
from examples.v8 import run_execution_layer_v2_persistent_outcome_blind_collector as service_module
from examples.v8.write_execution_layer_v2_persistent_outcome_blind_launchd_plist import (
    write_launchd_plist,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / (
    "examples/v8/polymarket_configs/execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
GIT_COMMIT = "a" * 40
RAW_FILENAMES = (
    "raw_polymarket_markets.jsonl",
    "raw_polymarket_orderbooks.jsonl",
    "raw_polymarket_trades.jsonl",
    "raw_binance_btcusdt_klines.jsonl",
    "raw_polymarket_resolutions.jsonl",
    "raw_polymarket_chainlink_prices.jsonl",
)


def test_protocol_is_outcome_blind_and_raw_is_not_direct_training_corpus() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    validate_persistent_outcome_blind_collector_protocol(protocol)

    assert protocol["settlement_finalizer_enabled"] is False
    assert protocol["resolution_provider_enabled"] is False
    assert protocol["training_corpus_export_enabled"] is False
    assert protocol["labels_outcomes_or_pnl_opened"] is False
    assert protocol["raw_collection_exported_to_direct_training_corpus"] is False
    assert protocol["direct_training_corpus_root"] == "/Volumes/PHILIPS/v8"
    assert protocol["append_only_index"]["deduplicate_identity_fields"] == [
        "scheduled_round_start_ts",
        "market_id",
        "slug",
        "run_id",
        "decision_id",
        "source_row_hash",
    ]
    _assert_safety(protocol)


def test_index_is_hash_chained_idempotent_and_retains_failed_attempts(
    tmp_path: Path,
) -> None:
    first = _capture_fixture(tmp_path, boundary=2_000, market_id="market-2")
    failure = {
        "round_index": 2,
        "run_id": "round-failed",
        "run_dir": str(tmp_path / "round-failed"),
        "scheduled_round_start_ts": 3_000,
        "stage": "round_capture",
        "error_type": "TimeoutError",
        "error": "bounded public provider timeout",
    }
    batch_path = _batch_summary(
        tmp_path,
        batch_id="batch-1",
        captures=[first],
        errors=[failure],
    )
    index_path = tmp_path / "state" / "index.jsonl"

    first_result = _index_batch(
        tmp_path,
        run_id="index-1",
        index_path=index_path,
        batch_path=batch_path,
    )
    rows = load_and_validate_persistent_outcome_blind_index(index_path)

    assert first_result["report"]["batch_attempt_count"] == 2
    assert first_result["report"]["appended_entry_count"] == 2
    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[0]["previous_entry_sha256"] == "0" * 64
    assert rows[1]["previous_entry_sha256"] == rows[0]["entry_sha256"]
    assert rows[0]["capture_quality_valid"] is True
    assert rows[1]["capture_quality_valid"] is False
    assert "round_capture_failed" in rows[1]["capture_quality_reason_codes"]
    assert rows[1]["capture_failure"]["error_type"] == "TimeoutError"
    assert rows[1]["raw_artifacts"] == {}
    assert all(row["labels_outcomes_or_pnl_opened"] is False for row in rows)
    for row in rows:
        _assert_safety(row)

    original_index_bytes = index_path.read_bytes()
    second_result = _index_batch(
        tmp_path,
        run_id="index-2",
        index_path=index_path,
        batch_path=batch_path,
    )

    assert second_result["report"]["appended_entry_count"] == 0
    assert second_result["report"]["idempotent_existing_run_count"] == 2
    assert index_path.read_bytes() == original_index_bytes


def test_index_chain_tamper_fails_closed(tmp_path: Path) -> None:
    capture = _capture_fixture(tmp_path, boundary=2_000, market_id="market-2")
    batch_path = _batch_summary(tmp_path, batch_id="batch", captures=[capture])
    index_path = tmp_path / "index.jsonl"
    _index_batch(
        tmp_path,
        run_id="index",
        index_path=index_path,
        batch_path=batch_path,
    )
    row = json.loads(index_path.read_text(encoding="utf-8"))
    row["market_id"] = "tampered"
    index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="index_entry_hash_mismatch"):
        load_and_validate_persistent_outcome_blind_index(index_path)


def test_outcome_row_is_retained_invalid_and_cannot_enter_window(
    tmp_path: Path,
) -> None:
    capture = _capture_fixture(
        tmp_path,
        boundary=2_000,
        market_id="market-outcome",
        resolution_rows=[
            {
                "market_id": "market-outcome",
                "winning_outcome": "UP",
            }
        ],
    )
    batch_path = _batch_summary(tmp_path, batch_id="batch", captures=[capture])
    index_path = tmp_path / "index.jsonl"
    result = _index_batch(
        tmp_path,
        run_id="index",
        index_path=index_path,
        batch_path=batch_path,
    )
    row = load_and_validate_persistent_outcome_blind_index(index_path)[0]

    assert row["capture_quality_valid"] is False
    assert "resolution_rows_present_during_capture" in row["capture_quality_reason_codes"]
    assert "forbidden_raw_field:winning_outcome" in row["capture_quality_reason_codes"]
    assert result["report"]["quality_valid_index_entry_count"] == 0

    boundary_path = _boundary_manifest(tmp_path, minimum_ts=2_000)
    freeze = _freeze(
        tmp_path,
        run_id="window",
        index_path=index_path,
        boundary_path=boundary_path,
        target=1,
        maximum=1,
    )
    assert freeze["report"]["window_freeze_ready"] is False
    assert freeze["report"]["selected_market_count"] == 0
    _assert_safety(freeze["manifest"])


def test_window_freeze_is_earliest_strictly_later_disjoint_and_deterministic(
    tmp_path: Path,
) -> None:
    captures = [
        _capture_fixture(tmp_path, boundary=1_000, market_id="prior-market"),
        _capture_fixture(tmp_path, boundary=2_000, market_id="market-2"),
        _capture_fixture(tmp_path, boundary=3_000, market_id="market-3"),
        _capture_fixture(tmp_path, boundary=4_000, market_id="market-4"),
    ]
    batch_path = _batch_summary(tmp_path, batch_id="batch", captures=captures)
    index_path = tmp_path / "index.jsonl"
    _index_batch(
        tmp_path,
        run_id="index",
        index_path=index_path,
        batch_path=batch_path,
    )
    boundary_path = _boundary_manifest(
        tmp_path,
        minimum_ts=2_000,
        prior_market_ids=["prior-market"],
    )

    first = _freeze(
        tmp_path,
        run_id="window-1",
        index_path=index_path,
        boundary_path=boundary_path,
        target=2,
        maximum=3,
    )
    second = _freeze(
        tmp_path,
        run_id="window-2",
        index_path=index_path,
        boundary_path=boundary_path,
        target=2,
        maximum=3,
    )

    assert first["report"]["window_freeze_ready"] is True
    assert first["report"]["selected_market_ids"] == ["market-2", "market-3"]
    assert first["report"]["selected_window_start_ts"] == 2_000
    assert first["report"]["selected_window_end_ts"] == 3_000
    assert first["report"]["scanned_entry_count"] == 2
    assert first["report"]["scan_pool_entry_count"] == 3
    assert (
        first["report"]["selected_market_ids_sha256"]
        == second["report"]["selected_market_ids_sha256"]
    )
    assert _sha256(first["selected_rows_path"]) == _sha256(second["selected_rows_path"])
    assert first["report"]["labels_outcomes_or_pnl_opened_for_selection"] is False
    _assert_safety(first["manifest"])


def test_window_overlap_and_raw_hash_tamper_fail_closed(tmp_path: Path) -> None:
    captures = [
        _capture_fixture(tmp_path, boundary=2_000, market_id="market-2"),
        _capture_fixture(tmp_path, boundary=3_000, market_id="market-3"),
    ]
    batch_path = _batch_summary(tmp_path, batch_id="batch", captures=captures)
    index_path = tmp_path / "index.jsonl"
    _index_batch(
        tmp_path,
        run_id="index",
        index_path=index_path,
        batch_path=batch_path,
    )
    overlap_boundary = _boundary_manifest(
        tmp_path,
        name="overlap-boundary.json",
        minimum_ts=2_000,
        prior_market_ids=["market-2"],
    )
    overlap = _freeze(
        tmp_path,
        run_id="overlap-window",
        index_path=index_path,
        boundary_path=overlap_boundary,
        target=2,
        maximum=2,
    )
    assert overlap["report"]["window_freeze_ready"] is False
    assert overlap["report"]["selected_market_ids"] == ["market-3"]
    assert overlap["report"]["exclusion_reason_distribution"] == {
        "market_id_overlaps_source_boundary": 1
    }

    raw_path = Path(captures[1]["run_dir"]) / "raw/raw_polymarket_markets.jsonl"
    raw_path.write_text('{"market_id":"tampered"}\n', encoding="utf-8")
    clean_boundary = _boundary_manifest(
        tmp_path,
        name="clean-boundary.json",
        minimum_ts=3_000,
    )
    tampered = _freeze(
        tmp_path,
        run_id="tampered-window",
        index_path=index_path,
        boundary_path=clean_boundary,
        target=1,
        maximum=1,
    )
    assert tampered["report"]["window_freeze_ready"] is False
    assert any(
        key.startswith("raw_artifact_lineage_invalid")
        for key in tampered["report"]["exclusion_reason_distribution"]
    )


def test_window_boundary_outcome_access_and_hash_mismatch_fail_before_selection(
    tmp_path: Path,
) -> None:
    capture = _capture_fixture(tmp_path, boundary=2_000, market_id="market-2")
    batch_path = _batch_summary(tmp_path, batch_id="batch", captures=[capture])
    index_path = tmp_path / "index.jsonl"
    _index_batch(
        tmp_path,
        run_id="index",
        index_path=index_path,
        batch_path=batch_path,
    )
    boundary_path = _boundary_manifest(tmp_path, minimum_ts=2_000)
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundary["labels_outcomes_or_pnl_opened"] = True
    boundary_path.write_text(json.dumps(boundary), encoding="utf-8")

    with pytest.raises(ValueError, match="source_boundary_outcome_sealing_invalid"):
        _freeze(
            tmp_path,
            run_id="opened-boundary-window",
            index_path=index_path,
            boundary_path=boundary_path,
            target=1,
            maximum=1,
        )

    with pytest.raises(ValueError, match="collector index SHA-256 mismatch"):
        freeze_outcome_blind_window(
            OutcomeBlindWindowFreezeConfig(
                run_id="bad-index-pin",
                output_dir=tmp_path / "window-runs",
                protocol_path=PROTOCOL_PATH,
                expected_protocol_sha256=_sha256(PROTOCOL_PATH),
                index_path=index_path,
                expected_index_sha256="f" * 64,
                source_boundary_manifest_path=boundary_path,
                expected_source_boundary_manifest_sha256=_sha256(boundary_path),
                target_valid_market_count=1,
                maximum_scan_count=1,
                builder_git_commit=GIT_COMMIT,
            )
        )


def test_summary_with_finalization_or_empty_attempts_fails_before_index(
    tmp_path: Path,
) -> None:
    summary_path = _batch_summary(tmp_path, batch_id="invalid", captures=[])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["finalization_count"] = 1
    payload["finalizations"] = [{"winning_outcome": "UP"}]
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="finalization"):
        _index_batch(
            tmp_path,
            run_id="invalid-index",
            index_path=tmp_path / "index.jsonl",
            batch_path=summary_path,
        )


def test_bounded_service_uses_collection_only_mode_and_persists_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}
    batch_summary_path = _batch_summary(
        tmp_path,
        batch_id="service-batch",
        captures=[],
        errors=[
            {
                "run_id": "failed-round",
                "run_dir": str(tmp_path / "failed-round"),
                "scheduled_round_start_ts": 2_000,
                "stage": "round_capture",
                "error_type": "TimeoutError",
                "error": "timeout",
            }
        ],
    )

    def fake_collector(**kwargs):
        observed.update(kwargs)
        return {"batch_summary_path": str(batch_summary_path)}

    def fake_index(config):
        index_path = Path(config.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("{}\n", encoding="utf-8")
        return {
            "index_sha256": _sha256(index_path),
            "report": {
                "index_entry_count": 1,
                "quality_valid_index_entry_count": 0,
            },
        }

    monkeypatch.setattr(
        service_module,
        "run_polymarket_async_round_collector_cli",
        fake_collector,
    )
    monkeypatch.setattr(
        service_module,
        "index_persistent_outcome_blind_batch",
        fake_index,
    )
    monkeypatch.setattr(service_module, "_git_head", lambda: GIT_COMMIT)

    state = service_module.run_service(
        service_root=tmp_path / "service",
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=2,
        max_batches=1,
        max_consecutive_failures=3,
        failure_backoff_seconds=0.0,
    )

    assert observed["outcome_blind_collection_only"] is True
    assert observed["settlement_grace_seconds"] == 0.0
    assert state["status"] == "bounded_collection_smoke_completed"
    assert state["last_completed_batch_sequence"] == 1
    assert state["labels_outcomes_or_pnl_opened"] is False
    assert state["settlement_finalizer_started"] is False
    assert state["resolution_provider_called"] is False
    assert state["training_corpus_export_attempted"] is False
    frozen_protocol_path = tmp_path / "service/persistent_outcome_blind_collector_protocol.json"
    assert _sha256(frozen_protocol_path) == _sha256(PROTOCOL_PATH)
    _assert_safety(state)


def test_launchd_descriptor_keeps_service_alive_without_training_root_export(
    tmp_path: Path,
) -> None:
    result = write_launchd_plist(
        output_path=tmp_path / "collector.plist",
        label="com.bigan.test.persistent-collector",
        service_root=tmp_path / "raw-service",
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=12,
        python_executable=sys.executable,
    )
    with Path(result["launchd_plist_path"]).open("rb") as handle:
        payload = plistlib.load(handle)

    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is True
    assert payload["ProgramArguments"][-2:] == ["--max-batches", "0"]
    assert "--protocol-sha256" in payload["ProgramArguments"]
    assert "/Volumes/PHILIPS/v8" not in " ".join(payload["ProgramArguments"])
    assert result["labels_outcomes_or_pnl_opened"] is False
    _assert_safety(result)


def test_launchd_descriptor_rejects_direct_training_corpus_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="direct training corpus root"):
        write_launchd_plist(
            output_path=tmp_path / "collector.plist",
            label="com.bigan.test.persistent-collector",
            service_root="/Volumes/PHILIPS/v8/raw-collector",
            protocol_path=PROTOCOL_PATH,
            protocol_sha256=_sha256(PROTOCOL_PATH),
            batch_round_count=12,
            python_executable=sys.executable,
        )


def _capture_fixture(
    tmp_path: Path,
    *,
    boundary: int,
    market_id: str,
    resolution_rows: list[dict] | None = None,
) -> dict:
    run_id = f"round-{boundary}-{market_id}"
    run_dir = tmp_path / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    slug = f"btc-updown-5m-{boundary}"
    payloads = {
        "raw_polymarket_markets.jsonl": [
            {
                "market_id": market_id,
                "slug": slug,
                "market_start_ts": boundary,
                "market_end_ts": boundary + 300_000,
            }
        ],
        "raw_polymarket_orderbooks.jsonl": [
            {
                "market_id": market_id,
                "timestamp": boundary,
                "outcome": "UP",
                "best_bid": 0.49,
                "best_ask": 0.51,
            },
            {
                "market_id": market_id,
                "timestamp": boundary,
                "outcome": "DOWN",
                "best_bid": 0.49,
                "best_ask": 0.51,
            },
        ],
        "raw_polymarket_trades.jsonl": [],
        "raw_binance_btcusdt_klines.jsonl": [
            {"open_time": boundary - 60_000, "close_time": boundary - 1}
        ],
        "raw_polymarket_resolutions.jsonl": resolution_rows or [],
        "raw_polymarket_chainlink_prices.jsonl": [
            {
                "source_ts": boundary - 1,
                "available_at_ts": boundary,
                "price": 100_000.0,
            }
        ],
    }
    for filename in RAW_FILENAMES:
        _write_jsonl(raw_dir / filename, payloads[filename])
    raw_hashes = {filename: _sha256(raw_dir / filename) for filename in RAW_FILENAMES}
    row_counts = {filename: len(payloads[filename]) for filename in RAW_FILENAMES}
    manifest = {
        "run_id": run_id,
        "raw_artifact_hashes": raw_hashes,
        "raw_artifact_row_counts": row_counts,
        "chainlink_raw_artifact_sha256": raw_hashes["raw_polymarket_chainlink_prices.jsonl"],
        "chainlink_raw_artifact_row_count": row_counts["raw_polymarket_chainlink_prices.jsonl"],
        "pending_resolution": True,
        "resolution_provider_called": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    (run_dir / "pending_round_capture_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "pending_round_capture_report.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pending_resolution": True,
                "resolution_provider_called": False,
                "paper_only": True,
                "capital_at_risk": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "round_index": boundary // 1_000,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "scheduled_round_start_ts": boundary,
        "capture_start_boundary_validation_passed": True,
        "provider_raw_orderbook_snapshot_count": 2,
        "training_sampled_orderbook_row_count": 2,
        "raw_btc_candle_row_count": 1,
        "raw_chainlink_price_row_count": 1,
        "market_identity_cache_provenance_violation_count": 0,
    }


def _batch_summary(
    tmp_path: Path,
    *,
    batch_id: str,
    captures: list[dict],
    errors: list[dict] | None = None,
) -> Path:
    path = tmp_path / f"{batch_id}-summary.json"
    path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "captures": captures,
                "capture_count": len(captures),
                "errors": errors or [],
                "error_count": len(errors or []),
                "finalization_count": 0,
                "finalizations": [],
                "outcome_blind_collection_only": True,
                "settlement_finalizer_started": False,
                "resolution_provider_called": False,
                "training_corpus_export_attempted": False,
                "labels_or_outcomes_opened_during_collection": False,
                "settlement_pnl_opened_during_collection": False,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _boundary_manifest(
    tmp_path: Path,
    *,
    minimum_ts: int,
    name: str = "boundary.json",
    prior_market_ids: list[str] | None = None,
) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "schema_version": "bigan-v8-outcome-blind-source-boundary-v1",
                "minimum_collection_decision_ts": minimum_ts,
                "prior_market_ids": prior_market_ids or [],
                "prior_slugs": [],
                "prior_source_row_hashes": [],
                "labels_outcomes_or_pnl_opened": False,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "source_model_candidate_eligible": False,
                "freeze_ready": False,
                "promotion_evidence_eligible": False,
                "v8_execution_handoff_allowed": False,
                "#134_resume_allowed": False,
                "#146_start_allowed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _index_batch(
    tmp_path: Path,
    *,
    run_id: str,
    index_path: Path,
    batch_path: Path,
) -> dict:
    return index_persistent_outcome_blind_batch(
        PersistentOutcomeBlindBatchIndexConfig(
            run_id=run_id,
            output_dir=tmp_path / "index-runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            index_path=index_path,
            batch_summary_path=batch_path,
            expected_batch_summary_sha256=_sha256(batch_path),
            collector_git_commit=GIT_COMMIT,
        )
    )


def _freeze(
    tmp_path: Path,
    *,
    run_id: str,
    index_path: Path,
    boundary_path: Path,
    target: int,
    maximum: int,
) -> dict:
    return freeze_outcome_blind_window(
        OutcomeBlindWindowFreezeConfig(
            run_id=run_id,
            output_dir=tmp_path / "window-runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            index_path=index_path,
            expected_index_sha256=_sha256(index_path),
            source_boundary_manifest_path=boundary_path,
            expected_source_boundary_manifest_sha256=_sha256(boundary_path),
            target_valid_market_count=target,
            maximum_scan_count=maximum,
            builder_git_commit=GIT_COMMIT,
        )
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_safety(payload: dict) -> None:
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
