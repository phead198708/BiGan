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
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _blocked_safety_fields,
)
from examples.v8 import run_execution_layer_v2_persistent_outcome_blind_collector as service_module
from examples.v8.run_polymarket_async_round_collector import _summary as _collector_batch_summary
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


def test_real_collector_batch_summary_contract_indexes_without_safety_schema_gap(
    tmp_path: Path,
) -> None:
    capture = _capture_fixture(tmp_path, boundary=2_000, market_id="market-2")
    summary = _collector_batch_summary("batch-real-shape", [capture], [], [])
    summary.update(
        {
            "outcome_blind_collection_only": True,
            "settlement_finalizer_started": False,
            "resolution_provider_called": False,
            "training_corpus_export_attempted": False,
            "labels_or_outcomes_opened_during_collection": False,
            "settlement_pnl_opened_during_collection": False,
        }
    )
    batch_path = tmp_path / "batch-real-shape-summary.json"
    batch_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    result = _index_batch(
        tmp_path,
        run_id="index-real-shape",
        index_path=tmp_path / "state" / "index.jsonl",
        batch_path=batch_path,
    )

    assert result["report"]["batch_capture_count"] == 1
    assert result["report"]["quality_valid_index_entry_count"] == 1


def test_index_rejects_silent_partial_orderbook_decision_window(
    tmp_path: Path,
) -> None:
    capture = _capture_fixture(
        tmp_path,
        boundary=2_000,
        market_id="market-partial-window",
        orderbook_decision_offsets_ms=(60_000, 120_000),
    )
    batch_path = _batch_summary(
        tmp_path,
        batch_id="batch-partial-window",
        captures=[capture],
    )
    index_path = tmp_path / "state" / "index.jsonl"

    result = _index_batch(
        tmp_path,
        run_id="index-partial-window",
        index_path=index_path,
        batch_path=batch_path,
    )
    row = load_and_validate_persistent_outcome_blind_index(index_path)[0]

    assert row["capture_quality_valid"] is False
    assert "orderbook_full_decision_window_coverage_failed" in (
        row["capture_quality_reason_codes"]
    )
    assert "orderbook_collection_ended_before_last_required_decision" in (
        row["capture_quality_reason_codes"]
    )
    assert "orderbook_required_decision_pair_coverage_incomplete" in (
        row["capture_quality_reason_codes"]
    )
    coverage = row["orderbook_window_coverage"]
    assert coverage["orderbook_expected_decision_pair_count"] == 4
    assert coverage["orderbook_observed_decision_pair_count"] == 2
    assert coverage["orderbook_latest_covered_decision_ts"] == 122_000
    assert result["report"][
        "batch_orderbook_full_window_coverage_failed_count"
    ] == 1
    assert result["report"]["quality_valid_index_entry_count"] == 0
    _assert_safety(row)


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
    assert first["report"]["collector_index_snapshot_immutable"] is True
    assert first["report"]["window_selection_used_immutable_index_snapshot"] is True
    assert first["report"]["paper_candidate_allowed"] is False
    assert first["index_snapshot_path"] != index_path
    assert first["manifest"]["index"]["path"] == str(first["index_snapshot_path"].resolve())
    assert first["manifest"]["source_index_pin_at_freeze"] == {
        "path": str(index_path.resolve()),
        "sha256": _sha256(index_path),
    }
    snapshot_sha256 = _sha256(first["index_snapshot_path"])
    index_path.write_bytes(index_path.read_bytes() + b"\n")
    assert _sha256(first["index_snapshot_path"]) == snapshot_sha256
    assert first["manifest"]["index"]["sha256"] == snapshot_sha256
    assert first["manifest"]["paper_candidate_allowed"] is False
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

    def fake_canary(config):
        canary_dir = tmp_path / "canary"
        canary_dir.mkdir()
        report_path = canary_dir / "report.json"
        manifest_path = canary_dir / "manifest.json"
        report = {
            "batch_id": config.batch_id,
            "development_data_canary_passed": True,
            "development_data_canary_blocking_reason_codes": [],
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        manifest_path.write_text(json.dumps({"batch_id": config.batch_id}), encoding="utf-8")
        return {
            "report": report,
            "report_path": report_path,
            "report_sha256": _sha256(report_path),
            "manifest_path": manifest_path,
            "manifest_sha256": _sha256(manifest_path),
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
    monkeypatch.setattr(
        service_module,
        "run_outcome_blind_development_batch_canary",
        fake_canary,
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
    assert state["last_batch_canary_passed"] is True
    assert state["last_batch_canary_report_sha256"] == _sha256(tmp_path / "canary/report.json")
    frozen_protocol_path = tmp_path / "service/persistent_outcome_blind_collector_protocol.json"
    assert _sha256(frozen_protocol_path) == _sha256(PROTOCOL_PATH)
    _assert_safety(state)


def test_bounded_service_runs_and_persists_v6_2_batch_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    candidate_manifest = tmp_path / "candidate.json"
    candidate_manifest.write_text('{"candidate":"v6.2"}\n', encoding="utf-8")

    def fake_collector(**kwargs):
        return {"batch_summary_path": str(batch_summary_path)}

    def fake_index(config):
        index_path = Path(config.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("{}\n", encoding="utf-8")
        return {
            "index_sha256": _sha256(index_path),
            "report": {"index_entry_count": 1, "quality_valid_index_entry_count": 1},
        }

    def fake_development(config):
        path = tmp_path / "development.json"
        manifest = tmp_path / "development_manifest.json"
        _write_json(
            path,
            {
                "development_data_canary_passed": True,
                "development_data_canary_blocking_reason_codes": [],
            },
        )
        _write_json(manifest, {"batch_id": config.batch_id})
        return {
            "report": json.loads(path.read_text()),
            "report_path": path,
            "report_sha256": _sha256(path),
            "manifest_path": manifest,
            "manifest_sha256": _sha256(manifest),
        }

    def fake_v6_2(config):
        path = tmp_path / "v6_2.json"
        manifest = tmp_path / "v6_2_manifest.json"
        _write_json(
            path,
            {
                "batch_id": "service-batch",
                "candidate_name": "market_clustered_mean_ev_v6_2",
                "future_strictly_later_and_disjoint_passed": True,
                "bounded_batch_complete": True,
                "source_sequence_start": 313,
                "source_sequence_end": 313,
                "indexed_market_count": 1,
                "quality_valid_market_count": 1,
                "positive_guard_compatible_trade_lcb_row_count": 1,
                "positive_mean_ev_lcb_unique_market_count": 1,
                "guard_accepted_unique_market_count": 1,
                "guard_accepted_market_ids": ["market"],
                "guard_accepted_market_ids_by_side": {"UP": ["market"], "DOWN": []},
                "labels_outcomes_or_pnl_opened": False,
                **_blocked_safety_fields(),
            },
        )
        _write_json(manifest, {"batch_id": "service-batch"})
        return {
            "report": json.loads(path.read_text()),
            "report_path": path,
            "report_sha256": _sha256(path),
            "manifest_path": manifest,
            "manifest_sha256": _sha256(manifest),
        }

    def fake_cumulative(reports, *, run_id):
        return {
            "run_id": run_id,
            "quality_valid_market_count": 1,
            "guard_accepted_unique_market_count": 1,
            "guard_accepted_unique_market_count_by_side": {"UP": 1, "DOWN": 0},
            "future_holdout_collection_complete": False,
            "target_free_terminal_blocked": False,
            "target_free_terminal_blocking_reason_codes": [],
        }

    def fake_cumulative_write(**kwargs):
        path = tmp_path / "cumulative.json"
        manifest = tmp_path / "cumulative_manifest.json"
        _write_json(path, kwargs["report"])
        _write_json(manifest, {"run_id": kwargs["run_id"]})
        return {
            "report": kwargs["report"],
            "report_path": path,
            "report_sha256": _sha256(path),
            "manifest_path": manifest,
            "manifest_sha256": _sha256(manifest),
        }

    monkeypatch.setattr(service_module, "run_polymarket_async_round_collector_cli", fake_collector)
    monkeypatch.setattr(service_module, "index_persistent_outcome_blind_batch", fake_index)
    monkeypatch.setattr(
        service_module, "run_outcome_blind_development_batch_canary", fake_development
    )
    monkeypatch.setattr(
        service_module,
        "run_market_clustered_mean_ev_v6_2_future_batch_canary",
        fake_v6_2,
    )
    monkeypatch.setattr(service_module, "build_v6_2_future_cumulative_canary", fake_cumulative)
    monkeypatch.setattr(
        service_module, "write_v6_2_future_cumulative_canary", fake_cumulative_write
    )
    monkeypatch.setattr(service_module, "_git_head", lambda: GIT_COMMIT)

    state = service_module.run_service(
        service_root=tmp_path / "service",
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=1,
        max_batches=1,
        max_consecutive_failures=3,
        failure_backoff_seconds=0.0,
        v6_2_candidate_manifest_path=candidate_manifest,
        v6_2_candidate_manifest_sha256=_sha256(candidate_manifest),
    )
    assert state["v6_2_future_quality_valid_market_count"] == 1
    assert state["v6_2_future_guard_accepted_unique_market_count"] == 1
    assert state["v6_2_future_guard_accepted_unique_market_count_by_side"] == {
        "UP": 1,
        "DOWN": 0,
    }
    assert len(state["v6_2_batch_canary_reports"]) == 1
    assert state["labels_outcomes_or_pnl_opened"] is False
    _assert_safety(state)


def test_bounded_service_runs_v6_9_batch_liveness_and_stops_at_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_summary_path = _batch_summary(
        tmp_path,
        batch_id="v6-9-service-batch",
        captures=[],
        errors=[],
    )
    v6_2_candidate = tmp_path / "v6_2_candidate.json"
    v6_9_candidate = tmp_path / "v6_9_candidate.json"
    v6_9_plan = tmp_path / "v6_9_plan.json"
    for path in (v6_2_candidate, v6_9_candidate, v6_9_plan):
        _write_json(path, {"fixture": path.stem})

    def fake_index(config):
        index_path = Path(config.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("{}\n", encoding="utf-8")
        return {
            "index_sha256": _sha256(index_path),
            "report": {"index_entry_count": 12, "quality_valid_index_entry_count": 12},
        }

    def artifact(name: str, report: dict) -> dict:
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        report_path = root / "report.json"
        manifest_path = root / "manifest.json"
        _write_json(report_path, report)
        _write_json(manifest_path, {"name": name})
        return {
            "report": report,
            "report_path": report_path,
            "report_sha256": _sha256(report_path),
            "manifest_path": manifest_path,
            "manifest_sha256": _sha256(manifest_path),
        }

    monkeypatch.setattr(
        service_module,
        "validate_v6_9_future_collection_plan",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        service_module,
        "run_polymarket_async_round_collector_cli",
        lambda **kwargs: {"batch_summary_path": str(batch_summary_path)},
    )
    monkeypatch.setattr(service_module, "index_persistent_outcome_blind_batch", fake_index)
    monkeypatch.setattr(
        service_module,
        "run_outcome_blind_development_batch_canary",
        lambda config: artifact(
            "development",
            {
                "development_data_canary_passed": True,
                "development_data_canary_blocking_reason_codes": [],
            },
        ),
    )
    monkeypatch.setattr(
        service_module,
        "run_market_clustered_mean_ev_v6_2_future_batch_canary",
        lambda config: artifact("v6_2", {"labels_outcomes_or_pnl_opened": False}),
    )
    monkeypatch.setattr(
        service_module,
        "run_v6_9_future_batch_canary",
        lambda config: artifact(
            "v6_9",
            {
                "batch_action_liveness_passed": True,
                "labels_outcomes_or_pnl_opened": False,
            },
        ),
    )
    monkeypatch.setattr(
        service_module,
        "build_v6_9_future_cumulative_canary",
        lambda *args, **kwargs: {
            "attempted_market_count": 120,
            "quality_valid_market_count": 120,
            "guard_accepted_unique_market_count": 80,
            "future_confirmatory_collection_complete": True,
            "target_free_terminal_blocked": False,
            "target_free_terminal_blocking_reason_codes": [],
            "labels_outcomes_or_pnl_opened": False,
        },
    )
    monkeypatch.setattr(
        service_module,
        "write_v6_9_future_cumulative_canary",
        lambda **kwargs: artifact("v6_9_cumulative", kwargs["report"]),
    )
    monkeypatch.setattr(service_module, "_git_head", lambda: GIT_COMMIT)

    state = service_module.run_service(
        service_root=tmp_path / "service",
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=12,
        max_batches=1,
        max_consecutive_failures=3,
        failure_backoff_seconds=0.0,
        v6_2_candidate_manifest_path=v6_2_candidate,
        v6_2_candidate_manifest_sha256=_sha256(v6_2_candidate),
        v6_9_candidate_manifest_path=v6_9_candidate,
        v6_9_candidate_manifest_sha256=_sha256(v6_9_candidate),
        v6_9_collection_plan_path=v6_9_plan,
        v6_9_collection_plan_sha256=_sha256(v6_9_plan),
    )

    assert state["status"] == "v6_9_future_confirmatory_collection_complete"
    assert state["last_v6_9_batch_action_liveness_passed"] is True
    assert state["v6_9_future_quality_valid_market_count"] == 120
    assert state["v6_9_future_guard_accepted_unique_market_count"] == 80
    assert state["labels_outcomes_or_pnl_opened"] is False
    assert (
        tmp_path / "service/persistent_outcome_blind_v6_9_confirmatory_collection_stop.json"
    ).is_file()
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
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ProgramArguments"][-2:] == ["--max-batches", "0"]
    assert "--protocol-sha256" in payload["ProgramArguments"]
    assert "--batch-canary-feature-contract" in payload["ProgramArguments"]
    assert "--batch-canary-feature-contract-sha256" in payload["ProgramArguments"]
    assert "/Volumes/PHILIPS/v8" not in " ".join(payload["ProgramArguments"])
    assert result["labels_outcomes_or_pnl_opened"] is False
    assert result["automatic_outcome_blind_batch_canary_enabled"] is True
    assert result["automatic_v6_2_frozen_batch_canary_enabled"] is False
    _assert_safety(result)


def test_launchd_descriptor_supports_preregistered_bounded_collection(
    tmp_path: Path,
) -> None:
    result = write_launchd_plist(
        output_path=tmp_path / "collector.plist",
        label="com.bigan.test.bounded-persistent-collector",
        service_root=tmp_path / "raw-service",
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=12,
        python_executable=sys.executable,
        max_batches=10,
    )
    with Path(result["launchd_plist_path"]).open("rb") as handle:
        payload = plistlib.load(handle)

    assert payload["ProgramArguments"][-2:] == ["--max-batches", "10"]
    assert result["continuous_collection"] is False
    assert result["maximum_batch_count"] == 10
    _assert_safety(result)


def test_launchd_descriptor_pins_v6_2_candidate_manifest(tmp_path: Path) -> None:
    candidate_manifest = tmp_path / "candidate.json"
    candidate_manifest.write_text('{"candidate":"v6.2"}\n', encoding="utf-8")
    result = write_launchd_plist(
        output_path=tmp_path / "collector.plist",
        label="com.bigan.test.v6-2-collector",
        service_root=tmp_path / "raw-service",
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=12,
        python_executable=sys.executable,
        v6_2_candidate_manifest_path=candidate_manifest,
        v6_2_candidate_manifest_sha256=_sha256(candidate_manifest),
    )
    with Path(result["launchd_plist_path"]).open("rb") as handle:
        payload = plistlib.load(handle)
    arguments = payload["ProgramArguments"]
    assert "--v6-2-candidate-manifest" in arguments
    assert "--v6-2-candidate-manifest-sha256" in arguments
    assert result["automatic_v6_2_frozen_batch_canary_enabled"] is True
    assert result["v6_2_candidate_manifest_sha256"] == _sha256(candidate_manifest)
    _assert_safety(result)


def test_launchd_descriptor_pins_v6_9_candidate_and_collection_plan(
    tmp_path: Path,
) -> None:
    v6_2_candidate = tmp_path / "v6_2_candidate.json"
    v6_9_candidate = tmp_path / "v6_9_candidate.json"
    v6_9_plan = tmp_path / "v6_9_plan.json"
    v6_2_candidate.write_text('{"candidate":"v6.2"}\n', encoding="utf-8")
    v6_9_candidate.write_text('{"candidate":"v6.9"}\n', encoding="utf-8")
    v6_9_plan.write_text('{"plan":"future"}\n', encoding="utf-8")

    result = write_launchd_plist(
        output_path=tmp_path / "collector.plist",
        label="com.bigan.test.v6-9-collector",
        service_root=tmp_path / "raw-service",
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=12,
        python_executable=sys.executable,
        v6_2_candidate_manifest_path=v6_2_candidate,
        v6_2_candidate_manifest_sha256=_sha256(v6_2_candidate),
        v6_9_candidate_manifest_path=v6_9_candidate,
        v6_9_candidate_manifest_sha256=_sha256(v6_9_candidate),
        v6_9_collection_plan_path=v6_9_plan,
        v6_9_collection_plan_sha256=_sha256(v6_9_plan),
    )
    with Path(result["launchd_plist_path"]).open("rb") as handle:
        arguments = plistlib.load(handle)["ProgramArguments"]

    assert "--v6-9-candidate-manifest" in arguments
    assert "--v6-9-candidate-manifest-sha256" in arguments
    assert "--v6-9-collection-plan" in arguments
    assert "--v6-9-collection-plan-sha256" in arguments
    assert result["automatic_v6_9_frozen_batch_canary_enabled"] is True
    assert result["v6_9_candidate_manifest_sha256"] == _sha256(v6_9_candidate)
    assert result["v6_9_collection_plan_sha256"] == _sha256(v6_9_plan)
    _assert_safety(result)


def test_v6_6_calibration_progress_is_strictly_later_and_stops_at_target(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "state/index.jsonl"
    boundary_capture = _capture_fixture(tmp_path, boundary=2_000, market_id="boundary")
    _index_batch(
        tmp_path,
        run_id="boundary-index",
        index_path=index_path,
        batch_path=_batch_summary(
            tmp_path,
            batch_id="boundary-batch",
            captures=[boundary_capture],
        ),
    )
    boundary_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    point_freeze = _v6_6_point_freeze_manifest(
        tmp_path,
        boundary_sequence=1,
        boundary_sha256=_sha256(index_path),
        last_entry_sha256=boundary_rows[-1]["entry_sha256"],
        minimum_market_start_ts_exclusive=2_000,
        target=2,
        maximum=3,
    )
    plan = service_module._load_v6_6_calibration_plan(
        manifest_path=point_freeze,
        expected_sha256=_sha256(point_freeze),
    )

    later_captures = [
        _capture_fixture(tmp_path, boundary=3_000, market_id="later-1"),
        _capture_fixture(tmp_path, boundary=4_000, market_id="later-2"),
    ]
    _index_batch(
        tmp_path,
        run_id="later-index",
        index_path=index_path,
        batch_path=_batch_summary(
            tmp_path,
            batch_id="later-batch",
            captures=later_captures,
        ),
    )

    progress = service_module._v6_6_calibration_progress(index_path, plan)
    assert progress["attempted_market_count"] == 2
    assert progress["quality_valid_market_count"] == 2
    assert progress["target_reached"] is True
    assert progress["labels_outcomes_or_pnl_opened"] is False
    terminal = service_module._v6_6_terminal_collection_state(
        progress=progress,
        plan=plan,
        point_freeze_manifest_path=point_freeze,
        point_freeze_manifest_sha256=_sha256(point_freeze),
    )
    assert terminal["status"] == "v6_6_fresh_calibration_collection_complete"
    assert terminal["v6_6_candidate_scoring_attempted"] is False
    assert terminal["blocking_reason_codes"] == []
    _assert_safety(terminal)


def test_v6_6_calibration_progress_rejects_boundary_hash_mismatch(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "state/index.jsonl"
    capture = _capture_fixture(tmp_path, boundary=2_000, market_id="boundary")
    _index_batch(
        tmp_path,
        run_id="boundary-index",
        index_path=index_path,
        batch_path=_batch_summary(
            tmp_path,
            batch_id="boundary-batch",
            captures=[capture],
        ),
    )
    rows = load_and_validate_persistent_outcome_blind_index(index_path)
    plan = {
        "collector_index_boundary_sequence": 1,
        "collector_index_boundary_sha256": "f" * 64,
        "collector_last_entry_sha256": rows[-1]["entry_sha256"],
        "minimum_market_start_ts_exclusive": 2_000,
        "target_quality_valid_market_count": 2,
        "maximum_attempted_market_count": 3,
    }
    with pytest.raises(ValueError, match="boundary SHA-256 mismatch"):
        service_module._v6_6_calibration_progress(index_path, plan)


def test_v6_6_service_resumes_past_v6_2_stop_and_writes_own_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_root = tmp_path / "service"
    index_path = service_root / "persistent_outcome_blind_round_index.jsonl"
    boundary_capture = _capture_fixture(tmp_path, boundary=2_000, market_id="boundary")
    _index_batch(
        tmp_path,
        run_id="boundary-index",
        index_path=index_path,
        batch_path=_batch_summary(
            tmp_path,
            batch_id="boundary-batch",
            captures=[boundary_capture],
        ),
    )
    boundary_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    point_freeze = _v6_6_point_freeze_manifest(
        tmp_path,
        boundary_sequence=1,
        boundary_sha256=_sha256(index_path),
        last_entry_sha256=boundary_rows[-1]["entry_sha256"],
        minimum_market_start_ts_exclusive=2_000,
        target=1,
        maximum=2,
    )
    _write_json(
        service_root / "persistent_outcome_blind_v6_2_future_collection_complete_stop.json",
        {"status": "v6_2_future_holdout_collection_complete"},
    )
    later_capture = _capture_fixture(tmp_path, boundary=3_000, market_id="later")
    later_summary = _batch_summary(
        tmp_path,
        batch_id="later-batch",
        captures=[later_capture],
    )

    monkeypatch.setattr(
        service_module,
        "run_polymarket_async_round_collector_cli",
        lambda **kwargs: {"batch_summary_path": str(later_summary)},
    )

    def fake_canary(config):
        output = tmp_path / "canary"
        output.mkdir(exist_ok=True)
        report_path = output / "report.json"
        manifest_path = output / "manifest.json"
        _write_json(
            report_path,
            {
                "development_data_canary_passed": True,
                "development_data_canary_blocking_reason_codes": [],
            },
        )
        _write_json(manifest_path, {"batch_id": config.batch_id})
        return {
            "report": json.loads(report_path.read_text()),
            "report_path": report_path,
            "report_sha256": _sha256(report_path),
            "manifest_path": manifest_path,
            "manifest_sha256": _sha256(manifest_path),
        }

    monkeypatch.setattr(
        service_module,
        "run_outcome_blind_development_batch_canary",
        fake_canary,
    )
    monkeypatch.setattr(service_module, "_git_head", lambda: GIT_COMMIT)

    state = service_module.run_service(
        service_root=service_root,
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=12,
        max_batches=1,
        max_consecutive_failures=3,
        failure_backoff_seconds=0.0,
        v6_6_point_freeze_manifest_path=point_freeze,
        v6_6_point_freeze_manifest_sha256=_sha256(point_freeze),
    )

    assert state["status"] == "v6_6_fresh_calibration_collection_complete"
    assert state["v6_6_fresh_calibration_attempted_market_count"] == 1
    assert state["v6_6_fresh_calibration_quality_valid_market_count"] == 1
    assert state["v6_6_candidate_scoring_attempted"] is False
    assert (
        service_root / "persistent_outcome_blind_v6_6_calibration_collection_stop.json"
    ).is_file()
    _assert_safety(state)


def test_launchd_descriptor_pins_v6_6_calibration_manifest(tmp_path: Path) -> None:
    point_freeze = _v6_6_point_freeze_manifest(
        tmp_path,
        boundary_sequence=1,
        boundary_sha256="a" * 64,
        last_entry_sha256="b" * 64,
        minimum_market_start_ts_exclusive=2_000,
        target=2,
        maximum=3,
    )
    result = write_launchd_plist(
        output_path=tmp_path / "collector.plist",
        label="com.bigan.test.v6-6-calibration-collector",
        service_root=tmp_path / "raw-service",
        protocol_path=PROTOCOL_PATH,
        protocol_sha256=_sha256(PROTOCOL_PATH),
        batch_round_count=12,
        python_executable=sys.executable,
        v6_6_point_freeze_manifest_path=point_freeze,
        v6_6_point_freeze_manifest_sha256=_sha256(point_freeze),
    )
    with Path(result["launchd_plist_path"]).open("rb") as handle:
        arguments = plistlib.load(handle)["ProgramArguments"]
    assert "--v6-6-point-freeze-manifest" in arguments
    assert "--v6-6-point-freeze-manifest-sha256" in arguments
    assert result["v6_6_fresh_calibration_collection_mode"] is True
    assert result["v6_6_point_freeze_manifest_sha256"] == _sha256(point_freeze)
    _assert_safety(result)


def test_completed_batch_canary_failure_is_terminal() -> None:
    with pytest.raises(
        service_module.OutcomeBlindBatchCanaryFailure,
        match="feature_timestamp_causality_violation",
    ):
        service_module._require_batch_canary_passed(
            {
                "report": {
                    "development_data_canary_passed": False,
                    "development_data_canary_blocking_reason_codes": [
                        "feature_timestamp_causality_violation"
                    ],
                }
            }
        )


def test_persistent_canary_terminal_stop_prevents_automatic_collection_restart(
    tmp_path: Path,
) -> None:
    service_root = tmp_path / "service"
    service_root.mkdir()
    marker = service_root / "persistent_outcome_blind_canary_terminal_stop.json"
    marker.write_text(
        json.dumps(
            {
                "status": "persistent_outcome_blind_canary_terminal_stop",
                "blocking_reason_codes": ["complete_five_action_grid_failed"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        service_module.OutcomeBlindBatchCanaryFailure,
        match="complete_five_action_grid_failed",
    ):
        service_module.run_service(
            service_root=service_root,
            protocol_path=PROTOCOL_PATH,
            protocol_sha256=_sha256(PROTOCOL_PATH),
            batch_round_count=2,
            max_batches=1,
            max_consecutive_failures=3,
            failure_backoff_seconds=0.0,
        )


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
    orderbook_decision_offsets_ms: tuple[int, ...] = (
        60_000,
        120_000,
        180_000,
        240_000,
    ),
) -> dict:
    run_id = f"round-{boundary}-{market_id}"
    run_dir = tmp_path / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    slug = f"btc-updown-5m-{boundary}"
    orderbook_collection_end_ts = boundary + max(
        orderbook_decision_offsets_ms,
        default=0,
    )
    payloads = {
        "raw_polymarket_markets.jsonl": [
            {
                "market_id": market_id,
                "slug": slug,
                "market_family": "btc_updown_5m",
                "market_start_ts": boundary,
                "market_end_ts": boundary + 300_000,
                "up_token_id": "up-token",
                "down_token_id": "down-token",
            }
        ],
        "raw_polymarket_orderbooks.jsonl": [
            {
                "market_id": market_id,
                "token_id": token_id,
                "ts": decision_ts,
                "available_at_ts": decision_ts,
                "collection_end_ts": orderbook_collection_end_ts,
                "outcome": outcome,
                "bid_price": 0.49,
                "ask_price": 0.51,
            }
            for decision_ts in (
                boundary + offset
                for offset in orderbook_decision_offsets_ms
            )
            for outcome, token_id in (
                ("UP", "up-token"),
                ("DOWN", "down-token"),
            )
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
        "provider_raw_orderbook_snapshot_count": (
            len(orderbook_decision_offsets_ms) * 2
        ),
        "training_sampled_orderbook_row_count": (
            len(orderbook_decision_offsets_ms) * 2
        ),
        "raw_btc_candle_row_count": 1,
        "raw_chainlink_price_row_count": 1,
        "market_identity_cache_provenance_violation_count": 0,
    }


def _v6_6_point_freeze_manifest(
    tmp_path: Path,
    *,
    boundary_sequence: int,
    boundary_sha256: str,
    last_entry_sha256: str,
    minimum_market_start_ts_exclusive: int,
    target: int,
    maximum: int,
) -> Path:
    path = tmp_path / "v6_6_point_freeze_manifest.json"
    _write_json(
        path,
        {
            "candidate_name": "policy_selected_runtime_pnl_v6_6",
            "point_model_frozen": True,
            "fresh_calibration_collection_allowed": True,
            "fresh_calibration_outcomes_opened": False,
            "candidate_scoring_frozen": False,
            "fresh_calibration_boundary": {
                "collector_index_boundary_sequence": boundary_sequence,
                "collector_index_boundary_sha256": boundary_sha256,
                "collector_last_entry_sha256": last_entry_sha256,
                "minimum_market_start_ts_exclusive": (minimum_market_start_ts_exclusive),
                "target_quality_valid_market_count": target,
                "maximum_attempted_market_count": maximum,
                "batch_market_count": 12,
            },
            **_blocked_safety_fields(),
        },
    )
    return path


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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
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
