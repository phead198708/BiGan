"""Async Polymarket round collector CLI tests."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import examples.v8.run_polymarket_async_round_collector as collector_module
from examples.v8.run_polymarket_async_round_collector import (
    _finalize_pending_once,
    _round_start_alignment_sleep_seconds,
    _scheduled_round_start_epoch_seconds,
    _summary,
    _wait_until_scheduled_round_start,
    main,
    run_polymarket_async_finalizer_cli,
)


def test_round_start_alignment_does_not_sleep_inside_start_window() -> None:
    assert (
        _round_start_alignment_sleep_seconds(
            market_family="btc_updown_5m",
            max_round_start_lag_seconds=30.0,
            now_epoch_seconds=600.0 + 12.0,
        )
        == 0.0
    )


def test_round_start_alignment_waits_when_started_late_in_round() -> None:
    assert (
        _round_start_alignment_sleep_seconds(
            market_family="btc_updown_5m",
            max_round_start_lag_seconds=30.0,
            now_epoch_seconds=600.0 + 247.0,
        )
        == 54.0
    )


def test_round_capture_schedule_advances_one_boundary_while_prior_capture_runs() -> None:
    first_start = _scheduled_round_start_epoch_seconds(
        market_family="btc_updown_5m",
        max_round_start_lag_seconds=30.0,
        now_epoch_seconds=600.0 + 12.0,
        previous_round_start_epoch_seconds=None,
    )
    second_start = _scheduled_round_start_epoch_seconds(
        market_family="btc_updown_5m",
        max_round_start_lag_seconds=30.0,
        now_epoch_seconds=600.0 + 13.0,
        previous_round_start_epoch_seconds=first_start,
    )

    assert first_start == 600.0
    assert second_start == 900.0


def test_round_capture_schedule_targets_next_boundary_before_prior_capture_finishes() -> None:
    second_start = _scheduled_round_start_epoch_seconds(
        market_family="btc_updown_5m",
        max_round_start_lag_seconds=30.0,
        now_epoch_seconds=899.5,
        previous_round_start_epoch_seconds=600.0,
    )

    assert second_start == 900.0


def test_batch_summary_reports_chainlink_freshness_watchdog_without_double_counting() -> None:
    summary = _summary(
        "chainlink-watchdog",
        [
            {
                "raw_chainlink_price_row_count": 20,
                "chainlink_rtds_price_stream_fresh": True,
                "chainlink_rtds_stale_reconnect_count": 1,
            },
            {
                "raw_chainlink_price_row_count": 22,
                "chainlink_rtds_price_stream_fresh": True,
                "chainlink_rtds_stale_reconnect_count": 2,
            },
        ],
        [],
        [],
    )

    assert summary["chainlink_covered_capture_count"] == 2
    assert summary["chainlink_fresh_capture_count"] == 2
    assert summary["chainlink_rtds_stale_reconnect_count"] == 2


def test_batch_summary_aggregates_market_identity_cache_evidence() -> None:
    summary = _summary(
        "identity-cache",
        [
            {
                "provider_raw_market_identity_source_type_distribution": {
                    "gamma_primary": 1
                },
                "market_identity_cache_fallback_market_count": 0,
                "market_identity_cache_fallback_reason_distribution": {},
                "market_identity_cache_provenance_violation_count": 0,
                "market_identity_clob_revalidation_passed_count": 0,
            },
            {
                "provider_raw_market_identity_source_type_distribution": {
                    "gamma_prefetch_cache_fallback": 1
                },
                "market_identity_cache_fallback_market_count": 1,
                "market_identity_cache_fallback_reason_distribution": {
                    "read_only_public_http_timeout": 1
                },
                "market_identity_cache_provenance_violation_count": 0,
                "market_identity_clob_revalidation_passed_count": 1,
            },
        ],
        [],
        [],
    )

    assert summary["market_identity_source_type_distribution"] == {
        "gamma_prefetch_cache_fallback": 1,
        "gamma_primary": 1,
    }
    assert summary["market_identity_cache_fallback_market_count"] == 1
    assert summary["market_identity_cache_fallback_reason_distribution"] == {
        "read_only_public_http_timeout": 1
    }
    assert summary["market_identity_cache_provenance_violation_count"] == 0
    assert summary["market_identity_clob_revalidation_passed_count"] == 1


def test_batch_summary_reports_feature_enrichment_recovery() -> None:
    summary = _summary(
        "feature-enrichment",
        [
            {
                "pending_feature_enrichment": False,
                "pending_resolution": True,
                "feature_enrichment_recovered": True,
                "raw_btc_candle_row_count": 20,
            }
        ],
        [
            {
                "finalization_status": "pending_resolution",
                "feature_enrichment_recovered": True,
            }
        ],
        [],
    )

    assert summary["capture_pending_feature_enrichment_count"] == 0
    assert summary["feature_enrichment_recovered_capture_count"] == 1
    assert summary["feature_enrichment_recovered_count"] == 1
    assert summary["pending_feature_enrichment_count"] == 0


def test_capture_worker_waits_again_when_spawned_before_scheduled_boundary() -> None:
    observed_times = iter((899.952, 900.0))
    sleep_calls: list[float] = []

    started_at = _wait_until_scheduled_round_start(
        900.0,
        now_fn=lambda: next(observed_times),
        sleep_fn=sleep_calls.append,
    )

    assert started_at == 900.0
    assert len(sleep_calls) == 1
    assert abs(sleep_calls[0] - 0.048) < 1e-9


def test_finalize_only_cli_accepts_shared_collector_args(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--batch-id",
                "finalize-smoke",
                "--output-dir",
                str(tmp_path),
                "--finalize-only",
                "--max-round-start-lag-seconds",
                "30",
            ]
        )
        == 0
    )

    summary_path = tmp_path / "finalize-smoke" / "finalizer_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["finalize_only"] is True
    assert summary["finalization_attempt_count"] == 0
    assert summary["error_count"] == 0


def test_finalize_only_merges_into_authoritative_batch_progress(
    tmp_path: Path, monkeypatch
) -> None:
    batch_id = "merge-finalizer"
    batch_dir = tmp_path / batch_id
    batch_dir.mkdir()
    progress_path = batch_dir / "batch_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "paper_only": True,
                "capital_at_risk": False,
                "captures": [{"run_id": f"{batch_id}-round01"}],
                "finalizations": [
                    {
                        "run_id": f"{batch_id}-round01",
                        "finalization_status": "pending_resolution",
                    }
                ],
                "errors": [],
            }
        )
    )
    observed_prefixes: list[str | None] = []

    def fake_finalize_pending_once(**kwargs) -> None:
        observed_prefixes.append(kwargs["batch_id_prefix"])
        kwargs["finalizations"][:] = [
            {
                "run_id": f"{batch_id}-round01",
                "finalization_status": "exported",
                "pending_resolution": False,
                "training_eligible": True,
                "exported_training_corpus_dir": str(tmp_path / "corpus"),
                "raw_resolution_count": 1,
                "reject_reason_counts": {},
            }
        ]

    monkeypatch.setattr(
        collector_module,
        "_finalize_pending_once",
        fake_finalize_pending_once,
    )
    summary = run_polymarket_async_finalizer_cli(
        batch_id=batch_id,
        output_dir=tmp_path,
    )

    progress = json.loads(progress_path.read_text())
    assert observed_prefixes == [batch_id]
    assert progress["capture_count"] == 1
    assert progress["captures"] == [{"run_id": f"{batch_id}-round01"}]
    assert progress["exported_round_count"] == 1
    assert progress["pending_resolution_count"] == 0
    assert summary["exported_round_count"] == 1


def test_finalize_only_preserves_pending_without_official_outcome(
    tmp_path: Path, monkeypatch
) -> None:
    batch_id = "pending-finalizer"
    batch_dir = tmp_path / batch_id
    batch_dir.mkdir()
    progress_path = batch_dir / "batch_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "paper_only": True,
                "capital_at_risk": False,
                "captures": [{"run_id": f"{batch_id}-round01"}],
                "finalizations": [
                    {
                        "run_id": f"{batch_id}-round01",
                        "finalization_status": "pending_resolution",
                        "pending_resolution": True,
                    }
                ],
                "errors": [],
            }
        )
    )
    monkeypatch.setattr(
        collector_module,
        "_finalize_pending_once",
        lambda **kwargs: None,
    )

    run_polymarket_async_finalizer_cli(batch_id=batch_id, output_dir=tmp_path)

    progress = json.loads(progress_path.read_text())
    assert progress["exported_round_count"] == 0
    assert progress["pending_resolution_count"] == 1
    assert progress["finalizations"][0]["finalization_status"] == "pending_resolution"


def test_finalize_pending_once_scopes_scan_to_requested_batch(tmp_path: Path, monkeypatch) -> None:
    matching = tmp_path / "scoped-batch-round01"
    unrelated = tmp_path / "other-batch-round01"
    for run_dir in (matching, unrelated):
        run_dir.mkdir()
        (run_dir / "pending_round_capture_manifest.json").write_text(
            json.dumps({"pending_resolution": True})
        )
    finalized_dirs: list[Path] = []

    def fake_finalize(run_dir, **kwargs):
        finalized_dirs.append(Path(run_dir))
        return SimpleNamespace(
            report={
                "finalization_status": "exported",
                "pending_resolution": False,
                "training_eligible": True,
                "exported_training_corpus_dir": str(tmp_path / "corpus"),
                "raw_resolution_count": 1,
                "reject_reason_counts": {},
            }
        )

    monkeypatch.setattr(
        collector_module,
        "finalize_polymarket_pending_round",
        fake_finalize,
    )
    finalizations: list[dict] = []
    errors: list[dict] = []
    _finalize_pending_once(
        output_dir=tmp_path,
        destination_root=tmp_path / "training",
        clob_ws_url="wss://example.invalid",
        overwrite_existing=False,
        batch_id_prefix="scoped-batch",
        finalizations=finalizations,
        errors=errors,
        lock=threading.Lock(),
    )

    assert finalized_dirs == [matching]
    assert [row["run_id"] for row in finalizations] == [matching.name]
    assert errors == []


def test_finalize_pending_once_scans_pending_feature_enrichment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "feature-batch-round01"
    run_dir.mkdir()
    (run_dir / "pending_round_capture_manifest.json").write_text(
        json.dumps(
            {
                "pending_feature_enrichment": True,
                "pending_resolution": False,
            }
        )
    )
    finalized_dirs: list[Path] = []

    def fake_finalize(run_dir_value, **kwargs):
        finalized_dirs.append(Path(run_dir_value))
        return SimpleNamespace(
            report={
                "finalization_status": "pending_resolution",
                "pending_feature_enrichment": False,
                "pending_resolution": True,
                "feature_enrichment_attempt_count": 1,
                "feature_enrichment_recovered": True,
                "feature_enrichment_reason_codes": [],
                "raw_btc_candle_row_count": 20,
                "training_eligible": False,
                "exported_training_corpus_dir": None,
                "raw_resolution_count": 0,
                "reject_reason_counts": {"missing_resolution": 1},
            }
        )

    monkeypatch.setattr(
        collector_module,
        "finalize_polymarket_pending_round",
        fake_finalize,
    )
    captures = [
        {
            "run_id": run_dir.name,
            "capture_status": "pending_feature_enrichment",
            "pending_feature_enrichment": True,
            "pending_resolution": False,
            "feature_enrichment_recovered": False,
            "raw_btc_candle_row_count": 0,
        }
    ]
    finalizations: list[dict] = []
    errors: list[dict] = []

    _finalize_pending_once(
        output_dir=tmp_path,
        destination_root=tmp_path / "training",
        clob_ws_url="wss://example.invalid",
        overwrite_existing=False,
        batch_id_prefix="feature-batch",
        finalizations=finalizations,
        errors=errors,
        lock=threading.Lock(),
        captures=captures,
        public_provider_http_timeout_seconds=3.0,
    )

    assert finalized_dirs == [run_dir]
    assert captures[0]["feature_enrichment_recovered"] is True
    assert captures[0]["raw_btc_candle_row_count"] == 20
    assert captures[0]["capture_status"] == "pending_resolution"
    assert finalizations[0]["feature_enrichment_recovered"] is True
    assert finalizations[0]["pending_resolution"] is True
    assert errors == []


def test_finalize_pending_once_recovers_hash_verified_existing_export(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir, destination_root = _existing_exported_finalization_fixture(
        tmp_path,
        run_id="recovery-batch-round01",
    )
    monkeypatch.setattr(
        collector_module,
        "finalize_polymarket_pending_round",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider retry must not run for an existing export")
        ),
    )
    finalizations = [
        {
            "run_id": run_dir.name,
            "finalization_status": "pending_resolution",
            "pending_resolution": True,
        }
    ]
    errors: list[dict] = []

    _finalize_pending_once(
        output_dir=tmp_path,
        destination_root=destination_root,
        clob_ws_url="wss://example.invalid",
        overwrite_existing=False,
        batch_id_prefix="recovery-batch",
        finalizations=finalizations,
        errors=errors,
        lock=threading.Lock(),
    )

    assert errors == []
    assert len(finalizations) == 1
    assert finalizations[0]["finalization_status"] == "exported"
    assert finalizations[0]["pending_resolution"] is False
    assert finalizations[0]["training_eligible"] is True
    assert finalizations[0]["recovered_from_existing_exported_report"] is True
    assert finalizations[0]["exported_corpus_manifest_sha256"] == _sha256(
        destination_root / "polymarket" / "market-1" / "polymarket_corpus_manifest.json"
    )


def test_finalize_pending_once_rejects_tampered_existing_export(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir, destination_root = _existing_exported_finalization_fixture(
        tmp_path,
        run_id="tamper-batch-round01",
    )
    exported_manifest = (
        destination_root / "polymarket" / "market-1" / "polymarket_corpus_manifest.json"
    )
    exported_manifest.write_text('{"tampered":true}\n')
    monkeypatch.setattr(
        collector_module,
        "finalize_polymarket_pending_round",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider retry must not overwrite a failed recovery")
        ),
    )
    finalizations = [
        {
            "run_id": run_dir.name,
            "finalization_status": "pending_resolution",
            "pending_resolution": True,
        }
    ]
    errors: list[dict] = []

    _finalize_pending_once(
        output_dir=tmp_path,
        destination_root=destination_root,
        clob_ws_url="wss://example.invalid",
        overwrite_existing=False,
        batch_id_prefix="tamper-batch",
        finalizations=finalizations,
        errors=errors,
        lock=threading.Lock(),
    )

    assert finalizations[0]["finalization_status"] == "pending_resolution"
    assert len(errors) == 1
    assert errors[0]["stage"] == "existing_exported_finalization_recovery"
    assert errors[0]["error"] == "existing exported corpus manifest hash mismatch"
    assert json.loads(
        (run_dir / "pending_round_finalization_report.json").read_text()
    )["finalization_status"] == "exported"


def _existing_exported_finalization_fixture(
    root: Path,
    *,
    run_id: str,
) -> tuple[Path, Path]:
    run_dir = root / run_id
    run_dir.mkdir()
    (run_dir / "pending_round_capture_manifest.json").write_text(
        json.dumps({"pending_resolution": True})
    )
    local_corpus = run_dir / "phase2_corpus"
    local_corpus.mkdir()
    destination_root = root / "training"
    exported_corpus = destination_root / "polymarket" / "market-1"
    exported_corpus.mkdir(parents=True)
    manifest_payload = '{"schema_version":"test-corpus"}\n'
    local_manifest = local_corpus / "polymarket_corpus_manifest.json"
    exported_manifest = exported_corpus / "polymarket_corpus_manifest.json"
    local_manifest.write_text(manifest_payload)
    exported_manifest.write_text(manifest_payload)
    manifest_sha256 = _sha256(local_manifest)
    report = {
        "run_id": run_id,
        "finalization_status": "exported",
        "pending_resolution": False,
        "training_eligible": True,
        "raw_resolution_count": 1,
        "reject_reason_counts": {},
        "exported_training_corpus_dir": str(exported_corpus),
        "phase2_corpus_manifest_sha256": manifest_sha256,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    manifest = {
        "run_id": run_id,
        "finalization_status": "exported",
        "pending_resolution": False,
        "exported_training_corpus_dir": str(exported_corpus),
        "phase2_corpus_manifest_sha256": manifest_sha256,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    (run_dir / "pending_round_finalization_report.json").write_text(json.dumps(report))
    (run_dir / "pending_round_finalization_manifest.json").write_text(json.dumps(manifest))
    return run_dir, destination_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
