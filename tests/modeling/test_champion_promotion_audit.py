"""Champion-promotion.md fail-closed gate audit tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

REQUIRED_FAMILIES = ("BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M")
EXPECTED_LIVE_ROOT = "data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z"
EXPECTED_SCREEN_SESSION = "xgbv4_7d_atomic_20260523T125657Z"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _promotion_attachment(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "# Champion Model Promotion Process",
                "## Stage 1: Offline Evaluation",
                "- Train: past 5 days",
                "- Document results in `rerun_report.md`",
                "## Stage 2: Cost-Adjusted Backtest",
                "## Stage 3: Shadow Evaluation",
                "- Run shadow for a minimum of one full trading session before evaluating",
                "## Stage 4: Bootstrap Decision",
                "- `PROMOTE_CHAMPION`",
                "## Stage 5: Champion Cutover",
            )
        ),
        encoding="utf-8",
    )
    return path


def _ready_status(path: Path, *, ready: bool) -> Path:
    span = 7.0 if ready else 0.25
    estimated_ready_at = None if ready else "2026-05-30T12:54:00Z"
    return _write_json(
        path,
        {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "screen_session": EXPECTED_SCREEN_SESSION,
            "screen_state": "running",
            "live_root": EXPECTED_LIVE_ROOT,
            "warehouse": f"{EXPECTED_LIVE_ROOT}/warehouse",
            "live_root_lock_evidence": {
                "lock_dir_exists": True,
                "pid_file_exists": True,
                "pid": 12345,
                "owner_running": True,
                "pid_parse_error": None,
            },
            "collection_readiness": {
                "ready_for_training": ready,
                "estimated_ready_at": estimated_ready_at,
                "target_days": 7.0,
                "features_15m_v1": {
                    "min_family_span_days": span,
                    "limiting_family": "ETH-15M",
                },
                "labels_15m_v1": {
                    "min_family_span_days": span,
                    "limiting_family": "ETH-15M",
                },
            },
            "raw_segment_integrity": {"invalid_count": 0},
            "raw_segment_quarantine": {"quarantined_count": 0},
            "disk_headroom_evidence": {"headroom_ok": True},
            "raw_manifest_coverage_evidence": {
                "stale_missing_processed_count": 0,
                "extra_processed_count": 0,
            },
            "health_evidence": {"unrecovered_error_match_count": 0},
            "liveness_evidence": {
                "raw_segments_fresh": True,
                "processed_manifest_fresh": True,
            },
            "warehouse_freshness_evidence": {
                "tables": {
                    "features_15m_v1": {"fresh": True},
                    "predictions": {"fresh": True},
                }
            },
            "label_freshness_evidence": {
                "fresh": True,
                "missing_label_families": [],
                "stale_families": [],
            },
        },
    )


def _eval_dir(
    path: Path,
    *,
    model_version: str,
    auc: float,
    brier: float,
    ece: float,
    dataset_dir: str,
    calibrated: bool = False,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_version": model_version,
        "model_path": str(path / "model.json"),
        "dataset_dir": dataset_dir,
        "dataset_version": "dataset-v1",
    }
    if calibrated:
        manifest["calibration_path"] = str(path / "calibration.json")
    _write_json(path / "manifest.json", manifest)
    _write_json(
        path / "offline_reference.json",
        {
            "model_version": model_version,
            "dataset_dir": dataset_dir,
            "dataset_version": "dataset-v1",
            "split": "val",
            "row_count": 10,
            "probability_distribution": {"mean": 0.55, "std": 0.10},
        },
    )
    _write_json(
        path / "metrics.json",
        {"test": {"roc_auc": auc, "brier_score": brier, "ece": ece}},
    )
    _write_json(
        path / "family_metrics.json",
        {
            "test": {
                family: {
                    "sample_count": 10,
                    "roc_auc": auc,
                    "brier_score": brier,
                    "ece": ece,
                }
                for family in REQUIRED_FAMILIES
            }
        },
    )
    return path


def _shadow_reference(candidate_eval: Path) -> dict:
    return {
        "sample_count": 10,
        "scored_count": 10,
        "offline_reference_path": str(candidate_eval / "offline_reference.json"),
        "offline_reference": json.loads(
            (candidate_eval / "offline_reference.json").read_text(encoding="utf-8")
        ),
        "challenger_probability_distribution": {"count": 10, "mean": 0.56, "std": 0.105},
    }


def _shadow_window(duration_seconds: int = 86_700) -> dict:
    return {
        "window_start_ts": 1_000,
        "window_end_ts": 1_000 + duration_seconds * 1_000,
        "session_duration_seconds": duration_seconds,
    }


def _shadow_pnl(delta: float = 0.5) -> dict:
    return {
        "champion_net_pnl": 1.0,
        "champion_trade_count": 3,
        "challenger_net_pnl": 1.0 + delta,
        "challenger_trade_count": 4,
        "net_pnl_delta": delta,
    }


def _dataset_manifest(path: Path, *, train_fraction: float = 5.0 / 7.0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(
        path / "manifest.json",
        {
            "dataset_version": "dataset-v1",
            "split_config": {
                "train_fraction": train_fraction,
                "val_fraction": 1.0 / 7.0,
            },
            "splits": {
                "train": {
                    "row_count": 50,
                    "positive_count": 25,
                    "negative_count": 25,
                    "positive_rate": 0.5,
                    "start_ts": 1_000,
                    "end_ts": 5_000,
                },
                "val": {
                    "row_count": 10,
                    "positive_count": 5,
                    "negative_count": 5,
                    "positive_rate": 0.5,
                    "start_ts": 5_000,
                    "end_ts": 6_000,
                },
                "test": {
                    "row_count": 10,
                    "positive_count": 5,
                    "negative_count": 5,
                    "positive_rate": 0.5,
                    "start_ts": 6_000,
                    "end_ts": 7_000,
                },
            },
            "family_splits": {
                family: {
                    "train": {"row_count": 20},
                    "val": {"row_count": 5},
                    "test": {"row_count": 5},
                }
                for family in REQUIRED_FAMILIES
            },
        },
    )
    return path


def _backtest(
    path: Path,
    *,
    net_pnl: float,
    sharpe: float,
    model_version: str,
    model_path: Path | str,
    dataset_dir: str,
    warehouse_dir: Path | str | None = None,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    latency_ms: int = 0,
    thresholds: tuple[float, ...] = (0.00, 0.03, 0.05),
    hold_ms: int = 900_000,
    required_outcome_side: str | None = "UP",
) -> Path:
    summary_rows = [
        {
            "threshold": threshold,
            "edge_threshold": threshold,
            "net_pnl": net_pnl,
            "max_drawdown_pct": 0.08,
            "sharpe_ratio": sharpe,
            "turnover": 0.10,
            "trade_count": 10,
            "hold_ms": hold_ms,
            "settings": {
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "latency_ms": latency_ms,
            },
        }
        for threshold in thresholds
    ]
    summary_path = _write_json(
        path,
        summary_rows,
    )
    _write_json(
        path.with_name("diagnostics.json"),
        {
            "model_version": model_version,
            "summary": summary_rows,
            "required_outcome_side": required_outcome_side,
            "metadata": {
                "backtest_kind": "direct_model",
                "model_path": str(model_path),
                "dataset_dir": dataset_dir,
                "dataset_version": "dataset-v1",
                "warehouse_dir": str(warehouse_dir or Path(dataset_dir) / "warehouse"),
            },
        },
    )
    return summary_path


def _serving_readiness(
    path: Path,
    *,
    model_path: Path | str,
    dataset_dir: str,
    rollback_runbook_path: Path | str,
    fallback_model_path: Path | str | None = None,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "serving_readiness_v1",
            "model_version": "xgboost-v4",
            "model_path": str(model_path),
            "feature_schema_path": str(path.with_name("feature_schema.json")),
            "dataset_dir": dataset_dir,
            "split": "test",
            "ready": True,
            "serving_ready": True,
            "error_rate": 0.0,
            "p95_latency_ms": 3.0,
            "latency_p95_ms": 3.0,
            "schema_validation": {
                "valid_input_accepted": True,
                "invalid_input_rejected": True,
                "silent_failure": False,
            },
            "fallback": {
                "fallback_model_path": str(fallback_model_path or path.with_name("fallback-model.json")),
                "fallback_model_available": True,
                "rollback_runbook_path": str(rollback_runbook_path),
                "rollback_runbook_available": True,
            },
        },
    )


def _github_issue_closures() -> list[dict[str, object]]:
    return [
        {
            "issue": 52,
            "repo": "phead198708/BiGan",
            "state": "closed",
            "comment": "Shadow PASS. Bootstrap decision: PROMOTE_CHAMPION.",
        },
        {
            "issue": 53,
            "repo": "phead198708/BiGan",
            "state": "closed",
            "comment": "Cutover complete. New champion: xgboost-v4.",
        },
    ]


def test_champion_promotion_audit_blocks_when_readiness_and_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=False),
    )

    assert report.passed is False
    assert report.decision == "BLOCKED"
    assert report.stages[0].passed is False
    ready_check = next(check for check in report.stages[0].checks if check.name == "ready_for_training")
    assert "estimated_ready_at 2026-05-30T12:54:00Z" in ready_check.detail
    assert "feature limiting_family ETH-15M" in ready_check.detail
    assert report.stages[1].passed is False
    rerun_check = next(check for check in report.stages[1].checks if check.name == "rerun_report_exists")
    assert rerun_check.passed is False
    written = json.loads(
        (tmp_path / "audit" / "champion_promotion_audit.json").read_text(encoding="utf-8")
    )
    assert written["decision"] == "BLOCKED"
    assert written["earliest_failed_stage"] == "Stage 0: 7-day Data Readiness"
    audit_markdown = (tmp_path / "audit" / "champion_promotion_audit.md").read_text(
        encoding="utf-8"
    )
    assert "Earliest failed stage: **Stage 0: 7-day Data Readiness**" in audit_markdown


def test_champion_promotion_audit_records_attachment_source_evidence(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    attachment = _promotion_attachment(tmp_path / "champion-promotion.md")
    repo_runbook = tmp_path / "docs" / "champion_promotion.md"
    repo_runbook.parent.mkdir(parents=True)
    repo_runbook.write_text(
        f"Local repo copy of the user-provided attachment at `{attachment}`.\n",
        encoding="utf-8",
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        promotion_process_path=attachment,
        repo_promotion_runbook_path=repo_runbook,
        live_status_path=_ready_status(tmp_path / "status.json", ready=False),
    )

    process = report.to_dict()["promotion_process"]
    assert process["source_path"] == str(attachment)
    assert process["source_exists"] is True
    assert process["source_sha256"] == sha256(attachment.read_bytes()).hexdigest()
    assert process["missing_required_markers"] == []
    assert process["repo_mirror_declares_source"] is True
    source_check = next(
        check for check in report.stages[0].checks if check.name == "promotion_process_source"
    )
    assert source_check.passed is True
    assert source_check.artifact_path == str(attachment)


def test_champion_promotion_audit_blocks_when_attachment_source_is_missing(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    missing_attachment = tmp_path / "missing-champion-promotion.md"
    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        promotion_process_path=missing_attachment,
        repo_promotion_runbook_path=tmp_path / "docs" / "champion_promotion.md",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
    )

    assert report.passed is False
    assert report.earliest_failed_stage == "Stage 0: 7-day Data Readiness"
    source_check = next(
        check for check in report.stages[0].checks if check.name == "promotion_process_source"
    )
    assert source_check.passed is False
    assert "source_exists=False" in source_check.detail


def test_audit_rejects_malformed_status_metrics_without_crashing(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _write_json(
        tmp_path / "status.json",
        {
            "screen_session": EXPECTED_SCREEN_SESSION,
            "live_root": EXPECTED_LIVE_ROOT,
            "warehouse": f"{EXPECTED_LIVE_ROOT}/warehouse",
            "collection_readiness": {
                "ready_for_training": True,
                "target_days": 7.0,
                "features_15m_v1": {"min_family_span_days": 7.0},
                "labels_15m_v1": {"min_family_span_days": 7.0},
            },
            "raw_segment_integrity": {"invalid_count": "none"},
            "raw_segment_quarantine": {"quarantined_count": 0},
            "disk_headroom_evidence": {"headroom_ok": True},
            "health_evidence": {"unrecovered_error_match_count": 0},
            "liveness_evidence": {
                "raw_segments_fresh": True,
                "processed_manifest_fresh": True,
            },
            "warehouse_freshness_evidence": {
                "tables": {
                    "features_15m_v1": {"fresh": True},
                    "predictions": {"fresh": True},
                }
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    raw_check = next(
        check for check in report.stages[0].checks if check.name == "raw_segment_integrity"
    )
    assert raw_check.passed is False
    assert "invalid gzip segments=missing" in raw_check.detail


def test_audit_blocks_quarantined_raw_segments(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    latest_quarantined = {
        "path": "raw_invalid/ws_market/2026-05-24T030000Z.ndjson.gz",
        "segment_ts": "2026-05-24T03:00:00Z",
        "gzip_probe": {
            "gzip_valid": False,
            "error": "error: Error -3 while decompressing data: invalid block type",
            "readable_prefix_bytes": 34454412,
            "readable_prefix_lines": 33900,
        },
    }
    status["raw_segment_quarantine"] = {
        "quarantined_count": 1,
        "latest_quarantined_segment": latest_quarantined,
    }
    status["collection_readiness"]["quarantine_clean_window"] = {
        "meets_target": False,
        "estimated_ready_at": "2026-05-31T03:00:00Z",
        "latest_quarantined_segment": latest_quarantined,
    }
    _write_json(status_path, status)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    quarantine_check = next(
        check for check in report.stages[0].checks if check.name == "raw_segment_quarantine"
    )
    assert quarantine_check.passed is False
    assert "quarantined raw segments=1.0000" in quarantine_check.detail
    assert "latest_path=raw_invalid/ws_market/2026-05-24T030000Z.ndjson.gz" in quarantine_check.detail
    assert "gzip_valid=False" in quarantine_check.detail
    assert "readable_prefix_lines=33900" in quarantine_check.detail
    assert "invalid block type" in quarantine_check.detail


def test_audit_blocks_insufficient_disk_headroom(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["disk_headroom_evidence"] = {
        "headroom_ok": False,
        "free_bytes": 1,
        "required_free_bytes": 2,
        "projected_remaining_bytes": 2,
        "headroom_margin_bytes": -1,
        "headroom_low_margin": False,
    }
    _write_json(status_path, status)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    disk_check = next(check for check in report.stages[0].checks if check.name == "disk_headroom")
    assert disk_check.passed is False
    assert "headroom_ok=False" in disk_check.detail
    assert "headroom_margin_bytes=-1" in disk_check.detail


def test_audit_blocks_current_filesystem_disk_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bigan.modeling import audit_champion_promotion_process
    from bigan.modeling import promotion as promotion_module

    live_root = tmp_path / "live"
    live_root.mkdir()
    status_path = _ready_status(tmp_path / "status.json", ready=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["live_root"] = str(live_root)
    status["warehouse"] = str(live_root / "warehouse")
    status["disk_headroom_evidence"] = {
        "headroom_ok": True,
        "headroom_low_margin": False,
        "free_bytes": 5_000,
        "required_free_bytes": 1_000,
        "projected_remaining_bytes": 1_000,
        "headroom_margin_bytes": 4_000,
        "low_margin_threshold_bytes": 100,
    }
    _write_json(status_path, status)

    class _StatVfs:
        f_blocks = 2_000
        f_bavail = 900
        f_frsize = 1

    monkeypatch.setattr(promotion_module.os, "statvfs", lambda _: _StatVfs())

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    disk_check = next(check for check in report.stages[0].checks if check.name == "disk_headroom")
    assert disk_check.passed is False
    assert "headroom_ok=True" in disk_check.detail
    assert "current_filesystem=available=True" in disk_check.detail
    assert "headroom_ok=False" in disk_check.detail
    assert "headroom_margin_bytes=-100" in disk_check.detail


def test_audit_blocks_stale_raw_manifest_coverage_gap(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["raw_manifest_coverage_evidence"] = {
        "stale_missing_processed_count": 2,
        "extra_processed_count": 0,
    }
    _write_json(status_path, status)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    coverage_check = next(
        check for check in report.stages[0].checks if check.name == "raw_manifest_coverage"
    )
    assert coverage_check.passed is False
    assert "stale_missing_processed_count=2.0000" in coverage_check.detail


@pytest.mark.parametrize(
    ("patch", "expected_detail"),
    [
        (
            {"live_root_lock_evidence": {"lock_dir_exists": True}},
            "pid_file_exists=False",
        ),
        (
            {
                "live_root_lock_evidence": {
                    "lock_dir_exists": True,
                    "pid_file_exists": True,
                    "pid": 987654321,
                    "owner_running": False,
                    "pid_parse_error": None,
                }
            },
            "owner_running=False",
        ),
        (
            {"screen_state": "not_found"},
            "screen_state=not_found",
        ),
    ],
)
def test_audit_blocks_unhealthy_collector_process_liveness(
    tmp_path: Path,
    patch: dict,
    expected_detail: str,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(patch)
    _write_json(status_path, status)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    liveness_check = next(
        check for check in report.stages[0].checks if check.name == "collector_process_liveness"
    )
    assert liveness_check.passed is False
    assert expected_detail in liveness_check.detail


def test_audit_blocks_pre_atomic_status_root(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["live_root"] = "data/live/xgboost-v4-multimarket-7d-segmented-debug"
    status["warehouse"] = "data/live/xgboost-v4-multimarket-7d-segmented-debug/warehouse"
    _write_json(status_path, status)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    checks = {check.name: check for check in report.stages[0].checks}
    assert report.stages[0].passed is False
    assert checks["clean_atomic_live_root"].passed is False
    assert checks["ready_for_training"].passed is True
    assert "segmented-debug" in checks["clean_atomic_live_root"].detail


def test_audit_blocks_stale_status_artifact_even_when_embedded_freshness_is_true(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["generated_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace(
        "+00:00",
        "Z",
    )
    _write_json(status_path, status)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    checks = {check.name: check for check in report.stages[0].checks}
    assert report.stages[0].passed is False
    assert checks["status_artifact_fresh"].passed is False
    assert checks["raw_and_manifest_fresh"].passed is True
    assert checks["warehouse_fresh"].passed is True
    assert checks["label_freshness"].passed is True
    assert "max_age_seconds=1800" in checks["status_artifact_fresh"].detail


def test_audit_requires_fresh_label_evidence(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["label_freshness_evidence"] = {
        "fresh": False,
        "missing_label_families": [],
        "stale_families": ["ETH-15M"],
    }
    _write_json(status_path, status)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    label_check = next(
        check for check in report.stages[0].checks if check.name == "label_freshness"
    )
    assert label_check.passed is False
    assert "stale_families=['ETH-15M']" in label_check.detail


def test_audit_rejects_string_boolean_readiness_evidence(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    status_path = _write_json(
        tmp_path / "status.json",
        {
            "screen_session": EXPECTED_SCREEN_SESSION,
            "live_root": EXPECTED_LIVE_ROOT,
            "warehouse": f"{EXPECTED_LIVE_ROOT}/warehouse",
            "collection_readiness": {
                "ready_for_training": "true",
                "target_days": 7.0,
                "features_15m_v1": {"min_family_span_days": 7.0},
                "labels_15m_v1": {"min_family_span_days": 7.0},
            },
            "raw_segment_integrity": {"invalid_count": 0},
            "raw_segment_quarantine": {"quarantined_count": 0},
            "disk_headroom_evidence": {"headroom_ok": True},
            "health_evidence": {"unrecovered_error_match_count": 0},
            "liveness_evidence": {
                "raw_segments_fresh": "true",
                "processed_manifest_fresh": "true",
            },
            "warehouse_freshness_evidence": {
                "tables": {
                    "features_15m_v1": {"fresh": "true"},
                    "predictions": {"fresh": "true"},
                }
            },
            "label_freshness_evidence": {
                "fresh": "true",
                "missing_label_families": [],
                "stale_families": [],
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
    )

    checks = {check.name: check for check in report.stages[0].checks}
    assert checks["ready_for_training"].passed is False
    assert checks["raw_and_manifest_fresh"].passed is False
    assert "raw_fresh=False" in checks["raw_and_manifest_fresh"].detail
    assert checks["warehouse_fresh"].passed is False
    assert checks["label_freshness"].passed is False


def test_champion_promotion_audit_passes_only_with_fresh_complete_gate_evidence(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process, generate_offline_rerun_report

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rerun_report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )
    assert rerun_report.passed is True
    assert "Decision: **PASS**" in (tmp_path / "rerun_report.md").read_text(encoding="utf-8")
    baseline_backtest = _backtest(
        tmp_path / "baseline" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": True,
            **_shadow_window(86_700),
            "champion_model_version": "xgboost-v3",
            "challenger_model_version": "xgboost-v4",
            **_shadow_reference(candidate_eval),
            "challenger_edge_trigger_rate": 0.12,
            "schema_error_rate": 0.0,
            "scoring_error_rate": 0.0,
            "latency_ms": {"xgboost-v4": {"p95": 4.0}},
            "simulated_pnl": _shadow_pnl(),
            "checks": {
                "prediction_distribution_stability": {"passed": True},
                "edge_trigger_rate": {"passed": True},
                "simulated_pnl": {"passed": True},
                "prediction_latency": {"passed": True},
                "schema_error_rate": {"passed": True},
                "scoring_error_rate": {"passed": True},
            },
        },
    )
    rollback_runbook = tmp_path / "rollback.md"
    rollback_runbook.write_text("# Rollback\n", encoding="utf-8")
    serving_path = _serving_readiness(
        tmp_path / "serving.json",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
        rollback_runbook_path=rollback_runbook,
    )
    bootstrap_path = _write_json(
        tmp_path / "bootstrap.json",
        {
            "recommended_action": "PROMOTE_CHAMPION",
            "candidate_model_version": "xgboost-v4",
            "missing_or_weak_evidence": [],
            "hard_gate_results": [{"model_version": "xgboost-v4", "passed": True}],
            "artifact_paths": {
                "baseline_eval_dir": str(baseline_eval),
                "baseline_backtest_summary_path": str(baseline_backtest),
                "candidate_eval_dir": str(candidate_eval),
                "candidate_backtest_summary_path": str(candidate_backtest),
                "serving_readiness_path": str(serving_path),
                "shadow_evaluation_path": str(shadow_path),
                "rollback_runbook_path": str(rollback_runbook),
            },
            "bootstrap_promotion_checklist": {
                "beats_baseline": True,
                "calibration_acceptable": True,
                "backtest_acceptable": True,
                "serving_readiness_acceptable": True,
                "rollback_fallback_available": True,
                "schema_stable": True,
                "simple_enough": True,
            },
        },
    )
    drift_baseline = tmp_path / "drift-baseline.json"
    _write_json(
        drift_baseline,
        {
            "model_version": "xgboost-v4",
            "source_offline_reference_path": str(candidate_eval / "offline_reference.json"),
            "dataset_dir": dataset_dir,
            "dataset_version": "dataset-v1",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10, "count": 10},
            "thresholds": {"probability_mean_shift_abs": 0.05},
        },
    )
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "current_champion": {
                "model_version": "xgboost-v4",
                "artifact_uri": str(candidate_eval / "model.json"),
                "calibration_artifact_uri": str(candidate_eval / "calibration.json"),
                "metrics_json": json.dumps(
                    {
                        "promotion_metrics": {
                            "auc": 0.78,
                            "brier": 0.16,
                            "delta_vs_baseline": 1.0,
                            "edge_trigger_rate": 0.12,
                            "shadow_p95_ms": 4.0,
                            "schema_error_rate": 0.0,
                        }
                    }
                ),
                "backtest_json": json.dumps(
                    {"net_pnl": 2.0, "max_drawdown": 0.10, "sharpe": 0.60}
                ),
            },
            "current_online_model": {
                "model_version": "xgboost-v4",
                "deployment_status": "succeeded",
                "traffic_percent": 100.0,
                "rollback_to_version": "xgboost-v3",
            },
            "fallback_registry_model": {
                "model_version": "xgboost-v3",
                "artifact_uri": str(tmp_path / "fallback-model.json"),
                "status": "retired",
            },
            "smoke": {
                "passed": True,
                "model_version": "xgboost-v4",
                "model_path": str(candidate_eval / "model.json"),
                "calibration_path": str(candidate_eval / "calibration.json"),
                "error_rate": 0.0,
                "serving_latency_ms": 2.5,
            },
            "drift_baseline_path": str(drift_baseline),
            "github_issue_closures": _github_issue_closures(),
            "evidence": {
                "smoke": str(
                    _write_json(
                        tmp_path / "smoke.json",
                        {
                            "passed": True,
                            "model_version": "xgboost-v4",
                            "model_path": str(candidate_eval / "model.json"),
                            "calibration_path": str(candidate_eval / "calibration.json"),
                            "error_rate": 0.0,
                            "serving_latency_ms": 2.5,
                        },
                    )
                ),
                "bootstrap": str(bootstrap_path),
                "shadow": str(shadow_path),
                "serving_readiness": str(serving_path),
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
        offline_rerun_report_path=tmp_path / "rerun_report.md",
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
        shadow_evaluation_path=shadow_path,
        serving_readiness_path=serving_path,
        bootstrap_decision_path=bootstrap_path,
        cutover_report_path=cutover_path,
        rollback_runbook_path=rollback_runbook,
    )

    assert report.passed is True
    assert report.decision == "PROMOTION_COMPLETE"
    assert report.earliest_failed_stage is None
    assert [stage.passed for stage in report.stages] == [True] * 6
    written = json.loads(
        (tmp_path / "audit" / "champion_promotion_audit.json").read_text(encoding="utf-8")
    )
    assert written["earliest_failed_stage"] is None


def test_audit_rejects_cutover_without_github_issue_closure_evidence(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "current_champion": {"model_version": "xgboost-v4"},
            "current_online_model": {
                "model_version": "xgboost-v4",
                "deployment_status": "succeeded",
                "traffic_percent": 100.0,
                "rollback_to_version": "xgboost-v3",
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        cutover_report_path=cutover_path,
    )

    checks = {check.name: check for check in report.stages[5].checks}
    assert checks["github_issue_closures_recorded"].passed is False
    assert "missing=#52, #53" in checks["github_issue_closures_recorded"].detail


def test_audit_rejects_cutover_with_incomplete_github_issue_closure_evidence(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "github_issue_closures": [
                {
                    "issue": 52,
                    "repo": "phead198708/BiGan",
                    "state": "open",
                    "comment": "Shadow PASS. Bootstrap decision: PROMOTE_CHAMPION.",
                },
                {
                    "issue": 53,
                    "repo": "phead198708/BiGan",
                    "state": "closed",
                    "comment": "Cutover started, not complete.",
                },
            ],
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        cutover_report_path=cutover_path,
    )

    checks = {check.name: check for check in report.stages[5].checks}
    assert checks["github_issue_closures_recorded"].passed is False
    assert "#52: state=open" in checks["github_issue_closures_recorded"].detail
    assert "#53: state=closed" in checks["github_issue_closures_recorded"].detail


def test_audit_rejects_cutover_with_stale_drift_baseline(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    candidate_eval = tmp_path / "candidate-eval"
    _write_json(
        candidate_eval / "manifest.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_eval / "model.json"),
            "calibration_path": str(candidate_eval / "calibration.json"),
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
        },
    )
    shadow_reference = candidate_eval / "offline_reference.json"
    _write_json(
        shadow_reference,
        {
            "model_version": "xgboost-v4",
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10},
        },
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "offline_reference_path": str(shadow_reference),
            "offline_reference": json.loads(shadow_reference.read_text(encoding="utf-8")),
        },
    )
    drift_baseline = _write_json(
        tmp_path / "drift-baseline.json",
        {
            "model_version": "xgboost-v4",
            "source_offline_reference_path": str(tmp_path / "old-reference.json"),
            "dataset_dir": "old-training",
            "dataset_version": "old-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10, "count": 10},
            "thresholds": {"probability_mean_shift_abs": 0.05},
        },
    )
    bootstrap_path = _write_json(tmp_path / "bootstrap.json", {})
    serving_path = _write_json(tmp_path / "serving.json", {})
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "current_champion": {
                "model_version": "xgboost-v4",
                "artifact_uri": str(candidate_eval / "model.json"),
                "calibration_artifact_uri": str(candidate_eval / "calibration.json"),
            },
            "current_online_model": {
                "model_version": "xgboost-v4",
                "deployment_status": "succeeded",
                "traffic_percent": 100.0,
                "rollback_to_version": "xgboost-v3",
            },
            "fallback_registry_model": {
                "model_version": "xgboost-v3",
                "artifact_uri": str(tmp_path / "fallback-model.json"),
                "status": "retired",
            },
            "smoke": {
                "passed": True,
                "model_version": "xgboost-v4",
                "error_rate": 0.0,
                "serving_latency_ms": 2.5,
            },
            "drift_baseline_path": str(drift_baseline),
            "evidence": {
                "smoke": str(
                    _write_json(
                        tmp_path / "smoke.json",
                        {
                            "passed": True,
                            "model_version": "xgboost-v4",
                            "error_rate": 0.0,
                            "serving_latency_ms": 2.5,
                        },
                    )
                ),
                "bootstrap": str(bootstrap_path),
                "shadow": str(shadow_path),
                "serving_readiness": str(serving_path),
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        candidate_eval_dir=candidate_eval,
        cutover_report_path=cutover_path,
        bootstrap_decision_path=bootstrap_path,
        shadow_evaluation_path=shadow_path,
        serving_readiness_path=serving_path,
    )

    drift_check = next(
        check
        for check in report.stages[5].checks
        if check.name == "drift_baseline_matches_shadow_reference"
    )
    assert drift_check.passed is False
    assert "old-training" in drift_check.detail
    assert "fresh-training" in drift_check.detail


def test_audit_rejects_cutover_with_stale_registry_artifact(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    candidate_eval = tmp_path / "candidate-eval"
    _write_json(
        candidate_eval / "manifest.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_eval / "model.json"),
            "calibration_path": str(candidate_eval / "calibration.json"),
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
        },
    )
    offline_reference = _write_json(
        candidate_eval / "offline_reference.json",
        {
            "model_version": "xgboost-v4",
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10},
        },
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "offline_reference_path": str(offline_reference),
            "offline_reference": json.loads(offline_reference.read_text(encoding="utf-8")),
        },
    )
    drift_baseline = _write_json(
        tmp_path / "drift-baseline.json",
        {
            "model_version": "xgboost-v4",
            "source_offline_reference_path": str(offline_reference),
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10, "count": 10},
            "thresholds": {"probability_mean_shift_abs": 0.05},
        },
    )
    bootstrap_path = _write_json(tmp_path / "bootstrap.json", {})
    serving_path = _write_json(tmp_path / "serving.json", {})
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "current_champion": {
                "model_version": "xgboost-v4",
                "artifact_uri": str(tmp_path / "old-model.json"),
                "calibration_artifact_uri": str(tmp_path / "old-calibration.json"),
            },
            "current_online_model": {
                "model_version": "xgboost-v4",
                "deployment_status": "succeeded",
                "traffic_percent": 100.0,
                "rollback_to_version": "xgboost-v3",
            },
            "fallback_registry_model": {
                "model_version": "xgboost-v3",
                "artifact_uri": str(tmp_path / "fallback-model.json"),
                "status": "retired",
            },
            "smoke": {
                "passed": True,
                "model_version": "xgboost-v4",
                "error_rate": 0.0,
                "serving_latency_ms": 2.5,
            },
            "drift_baseline_path": str(drift_baseline),
            "evidence": {
                "smoke": str(
                    _write_json(
                        tmp_path / "smoke.json",
                        {
                            "passed": True,
                            "model_version": "xgboost-v4",
                            "error_rate": 0.0,
                            "serving_latency_ms": 2.5,
                        },
                    )
                ),
                "bootstrap": str(bootstrap_path),
                "shadow": str(shadow_path),
                "serving_readiness": str(serving_path),
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        candidate_eval_dir=candidate_eval,
        cutover_report_path=cutover_path,
        bootstrap_decision_path=bootstrap_path,
        shadow_evaluation_path=shadow_path,
        serving_readiness_path=serving_path,
    )

    artifact_check = next(
        check
        for check in report.stages[5].checks
        if check.name == "registry_champion_artifacts_match_candidate"
    )
    assert artifact_check.passed is False
    assert "old-model" in artifact_check.detail
    assert "candidate-eval" in artifact_check.detail


def test_audit_rejects_cutover_without_registry_promotion_metrics(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "current_champion": {
                "model_version": "xgboost-v4",
                "metrics_json": json.dumps({"test": {"roc_auc": 0.78}}),
                "backtest_json": json.dumps({"net_pnl": 2.0}),
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        cutover_report_path=cutover_path,
    )

    metric_check = next(
        check
        for check in report.stages[5].checks
        if check.name == "registry_champion_promotion_metrics_recorded"
    )
    assert metric_check.passed is False
    assert "auc=0.7800" in metric_check.detail
    assert "missing=brier" in metric_check.detail
    assert "shadow_p95_ms" in metric_check.detail


def test_audit_rejects_cutover_without_fallback_registry_model(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    candidate_eval = tmp_path / "candidate-eval"
    _write_json(
        candidate_eval / "manifest.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_eval / "model.json"),
            "calibration_path": str(candidate_eval / "calibration.json"),
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
        },
    )
    offline_reference = _write_json(
        candidate_eval / "offline_reference.json",
        {
            "model_version": "xgboost-v4",
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10},
        },
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "offline_reference_path": str(offline_reference),
            "offline_reference": json.loads(offline_reference.read_text(encoding="utf-8")),
        },
    )
    drift_baseline = _write_json(
        tmp_path / "drift-baseline.json",
        {
            "model_version": "xgboost-v4",
            "source_offline_reference_path": str(offline_reference),
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10, "count": 10},
            "thresholds": {"probability_mean_shift_abs": 0.05},
        },
    )
    bootstrap_path = _write_json(tmp_path / "bootstrap.json", {})
    serving_path = _write_json(tmp_path / "serving.json", {})
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "current_champion": {
                "model_version": "xgboost-v4",
                "artifact_uri": str(candidate_eval / "model.json"),
                "calibration_artifact_uri": str(candidate_eval / "calibration.json"),
            },
            "current_online_model": {
                "model_version": "xgboost-v4",
                "deployment_status": "succeeded",
                "traffic_percent": 100.0,
                "rollback_to_version": "xgboost-v3",
            },
            "smoke": {
                "passed": True,
                "model_version": "xgboost-v4",
                "error_rate": 0.0,
                "serving_latency_ms": 2.5,
            },
            "drift_baseline_path": str(drift_baseline),
            "evidence": {
                "smoke": str(
                    _write_json(
                        tmp_path / "smoke.json",
                        {
                            "passed": True,
                            "model_version": "xgboost-v4",
                            "error_rate": 0.0,
                            "serving_latency_ms": 2.5,
                        },
                    )
                ),
                "bootstrap": str(bootstrap_path),
                "shadow": str(shadow_path),
                "serving_readiness": str(serving_path),
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        candidate_eval_dir=candidate_eval,
        cutover_report_path=cutover_path,
        bootstrap_decision_path=bootstrap_path,
        shadow_evaluation_path=shadow_path,
        serving_readiness_path=serving_path,
    )

    fallback_check = next(
        check
        for check in report.stages[5].checks
        if check.name == "fallback_registry_model_available"
    )
    assert fallback_check.passed is False
    assert "expected=xgboost-v3" in fallback_check.detail


def test_audit_rejects_cutover_with_mismatched_smoke_artifact(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    candidate_eval = tmp_path / "candidate-eval"
    _write_json(
        candidate_eval / "manifest.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_eval / "model.json"),
            "calibration_path": str(candidate_eval / "calibration.json"),
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
        },
    )
    offline_reference = _write_json(
        candidate_eval / "offline_reference.json",
        {
            "model_version": "xgboost-v4",
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10},
        },
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "offline_reference_path": str(offline_reference),
            "offline_reference": json.loads(offline_reference.read_text(encoding="utf-8")),
        },
    )
    drift_baseline = _write_json(
        tmp_path / "drift-baseline.json",
        {
            "model_version": "xgboost-v4",
            "source_offline_reference_path": str(offline_reference),
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10, "count": 10},
            "thresholds": {"probability_mean_shift_abs": 0.05},
        },
    )
    bootstrap_path = _write_json(tmp_path / "bootstrap.json", {})
    serving_path = _write_json(tmp_path / "serving.json", {})
    stale_smoke_path = _write_json(
        tmp_path / "smoke.json",
        {
            "passed": True,
            "model_version": "xgboost-v3",
            "error_rate": 0.0,
            "serving_latency_ms": 2.5,
        },
    )
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "current_champion": {
                "model_version": "xgboost-v4",
                "artifact_uri": str(candidate_eval / "model.json"),
                "calibration_artifact_uri": str(candidate_eval / "calibration.json"),
            },
            "current_online_model": {
                "model_version": "xgboost-v4",
                "deployment_status": "succeeded",
                "traffic_percent": 100.0,
                "rollback_to_version": "xgboost-v3",
            },
            "fallback_registry_model": {
                "model_version": "xgboost-v3",
                "artifact_uri": str(tmp_path / "fallback-model.json"),
                "status": "retired",
            },
            "smoke": {
                "passed": True,
                "model_version": "xgboost-v4",
                "error_rate": 0.0,
                "serving_latency_ms": 2.5,
            },
            "drift_baseline_path": str(drift_baseline),
            "evidence": {
                "smoke": str(stale_smoke_path),
                "bootstrap": str(bootstrap_path),
                "shadow": str(shadow_path),
                "serving_readiness": str(serving_path),
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        candidate_eval_dir=candidate_eval,
        cutover_report_path=cutover_path,
        bootstrap_decision_path=bootstrap_path,
        shadow_evaluation_path=shadow_path,
        serving_readiness_path=serving_path,
    )

    smoke_check = next(
        check
        for check in report.stages[5].checks
        if check.name == "cutover_uses_current_smoke"
    )
    assert smoke_check.passed is False
    assert "artifact_model=xgboost-v3" in smoke_check.detail


def test_audit_accepts_cutover_relative_evidence_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    monkeypatch.chdir(tmp_path)
    bootstrap_path = _write_json(tmp_path / "bootstrap.json", {})
    shadow_path = _write_json(tmp_path / "shadow.json", {})
    serving_path = _write_json(tmp_path / "serving.json", {})
    smoke_path = _write_json(
        tmp_path / "smoke.json",
        {
            "passed": True,
            "model_version": "xgboost-v4",
            "error_rate": 0.0,
            "serving_latency_ms": 2.5,
        },
    )
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "smoke": {
                "passed": True,
                "model_version": "xgboost-v4",
                "error_rate": 0.0,
                "serving_latency_ms": 2.5,
            },
            "evidence": {
                "smoke": str(smoke_path.relative_to(tmp_path)),
                "bootstrap": str(bootstrap_path.relative_to(tmp_path)),
                "shadow": str(shadow_path.relative_to(tmp_path)),
                "serving_readiness": str(serving_path.relative_to(tmp_path)),
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        cutover_report_path=cutover_path,
        bootstrap_decision_path=bootstrap_path,
        shadow_evaluation_path=shadow_path,
        serving_readiness_path=serving_path,
    )

    checks = {check.name: check for check in report.stages[5].checks}
    assert checks["cutover_uses_current_smoke"].passed is True
    assert checks["cutover_uses_current_bootstrap"].passed is True
    assert checks["cutover_uses_current_shadow"].passed is True
    assert checks["cutover_uses_current_serving_readiness"].passed is True


def test_audit_rejects_cutover_with_malformed_online_traffic(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "current_champion": {"model_version": "xgboost-v4"},
            "current_online_model": {
                "model_version": "xgboost-v4",
                "deployment_status": "succeeded",
                "traffic_percent": "all",
                "rollback_to_version": "xgboost-v3",
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        cutover_report_path=cutover_path,
    )

    online_check = next(
        check for check in report.stages[5].checks if check.name == "online_deployment_succeeded"
    )
    assert online_check.passed is False
    assert "traffic=missing" in online_check.detail


def test_audit_rejects_cutover_with_nonzero_smoke_error_rate(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    smoke_path = _write_json(
        tmp_path / "smoke.json",
        {
            "passed": True,
            "model_version": "xgboost-v4",
            "error_rate": 0.01,
            "serving_latency_ms": 2.5,
        },
    )
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "smoke": {
                "passed": True,
                "model_version": "xgboost-v4",
                "error_rate": 0.01,
                "serving_latency_ms": 2.5,
            },
            "evidence": {"smoke": str(smoke_path)},
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        cutover_report_path=cutover_path,
    )

    checks = {check.name: check for check in report.stages[5].checks}
    assert checks["inference_smoke_passed"].passed is False
    assert checks["cutover_uses_current_smoke"].passed is False
    assert "error_rate=0.0100" in checks["inference_smoke_passed"].detail
    assert "artifact_error_rate=0.0100" in checks["cutover_uses_current_smoke"].detail


def test_audit_rejects_cutover_string_boolean_smoke_pass(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    smoke_payload = {
        "passed": "true",
        "model_version": "xgboost-v4",
        "error_rate": 0.0,
        "serving_latency_ms": 2.5,
    }
    smoke_path = _write_json(tmp_path / "smoke.json", smoke_payload)
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "smoke": smoke_payload,
            "evidence": {"smoke": str(smoke_path)},
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        cutover_report_path=cutover_path,
    )

    checks = {check.name: check for check in report.stages[5].checks}
    assert checks["inference_smoke_passed"].passed is False
    assert checks["cutover_uses_current_smoke"].passed is False
    assert "smoke_passed=true" in checks["inference_smoke_passed"].detail


def test_audit_rejects_cutover_with_stale_smoke_model_path(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    candidate_eval = tmp_path / "candidate-eval"
    _write_json(
        candidate_eval / "manifest.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_eval / "model.json"),
            "calibration_path": str(candidate_eval / "calibration.json"),
            "dataset_dir": "fresh-training",
            "dataset_version": "fresh-dataset",
        },
    )
    smoke_payload = {
        "passed": True,
        "model_version": "xgboost-v4",
        "model_path": str(tmp_path / "old-candidate" / "model.json"),
        "calibration_path": str(tmp_path / "old-candidate" / "calibration.json"),
        "error_rate": 0.0,
        "serving_latency_ms": 2.5,
    }
    smoke_path = _write_json(tmp_path / "smoke.json", smoke_payload)
    cutover_path = _write_json(
        tmp_path / "cutover.json",
        {
            "smoke": smoke_payload,
            "evidence": {"smoke": str(smoke_path)},
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        candidate_eval_dir=candidate_eval,
        cutover_report_path=cutover_path,
    )

    checks = {check.name: check for check in report.stages[5].checks}
    assert checks["cutover_uses_current_smoke"].passed is True
    assert checks["smoke_artifacts_match_candidate"].passed is False
    assert "old-candidate" in checks["smoke_artifacts_match_candidate"].detail
    assert "candidate-eval" in checks["smoke_artifacts_match_candidate"].detail


def test_audit_rejects_malformed_backtest_metrics_without_crashing(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    baseline_backtest = _backtest(
        tmp_path / "baseline" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate" / "summary.json",
        net_pnl="better",
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    net_pnl_check = next(
        check for check in report.stages[2].checks if check.name == "net_pnl_beats_champion"
    )
    assert net_pnl_check.passed is False
    assert "candidate net_pnl=missing" in net_pnl_check.detail


def test_audit_rejects_backtests_with_mismatched_cost_assumptions(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    baseline_backtest = _backtest(
        tmp_path / "baseline" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
        fee_bps=10.0,
        slippage_bps=5.0,
        latency_ms=0,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
        fee_bps=1.0,
        slippage_bps=5.0,
        latency_ms=0,
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    checks = {check.name: check for check in report.stages[2].checks}
    assert checks["realistic_nonzero_costs"].passed is True
    assert checks["matched_backtest_cost_assumptions"].passed is False
    assert "baseline fee_bps=10.0000" in checks["matched_backtest_cost_assumptions"].detail
    assert "candidate fee_bps=1.0000" in checks["matched_backtest_cost_assumptions"].detail


def test_audit_rejects_backtests_with_mismatched_holdout_thresholds(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    baseline_backtest = _backtest(
        tmp_path / "baseline" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
        thresholds=(0.00, 0.03, 0.05),
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
        thresholds=(0.00, 0.05),
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    holdout_check = next(
        check for check in report.stages[2].checks if check.name == "matched_backtest_holdout_period"
    )
    assert holdout_check.passed is False
    assert "baseline thresholds=[0.0000, 0.0300, 0.0500]" in holdout_check.detail
    assert "candidate thresholds=[0.0000, 0.0500]" in holdout_check.detail


def test_audit_rejects_backtests_with_mismatched_outcome_side(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    baseline_backtest = _backtest(
        tmp_path / "baseline" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
        required_outcome_side="UP",
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
        required_outcome_side="DOWN",
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    holdout_check = next(
        check for check in report.stages[2].checks if check.name == "matched_backtest_holdout_period"
    )
    assert holdout_check.passed is False
    assert "baseline required_outcome_side=UP" in holdout_check.detail
    assert "candidate required_outcome_side=DOWN" in holdout_check.detail


def test_audit_rejects_zero_cost_baseline_backtest(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    baseline_backtest = _backtest(
        tmp_path / "baseline" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    cost_check = next(
        check for check in report.stages[2].checks if check.name == "realistic_nonzero_costs"
    )
    assert cost_check.passed is False
    assert "baseline fee_bps=0.0000" in cost_check.detail
    assert "baseline slippage_bps=0.0000" in cost_check.detail


def test_audit_rejects_boolean_backtest_metrics(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    baseline_backtest = _backtest(
        tmp_path / "baseline" / "summary.json",
        net_pnl=0.5,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate" / "summary.json",
        net_pnl=True,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    net_pnl_check = next(
        check for check in report.stages[2].checks if check.name == "net_pnl_beats_champion"
    )
    assert net_pnl_check.passed is False
    assert "candidate net_pnl=missing" in net_pnl_check.detail


def test_offline_rerun_report_blocks_non_5_1_1_time_split(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process, generate_offline_rerun_report

    dataset_dir = str(_dataset_manifest(tmp_path / "training", train_fraction=0.60))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )

    report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )

    split_check = next(check for check in report.checks if check.name == "dataset_time_split_5_1_1")
    assert report.passed is False
    assert split_check.passed is False
    assert "expected train=5/7 val=1/7" in split_check.detail

    audit = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        offline_rerun_report_path=tmp_path / "rerun_report.md",
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
    )
    rerun_check = next(check for check in audit.stages[1].checks if check.name == "rerun_report_exists")
    assert rerun_check.passed is False


def test_offline_rerun_report_requires_all_market_family_metrics(tmp_path: Path) -> None:
    from bigan.modeling import generate_offline_rerun_report

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    candidate_family_metrics = json.loads(
        (candidate_eval / "family_metrics.json").read_text(encoding="utf-8")
    )
    del candidate_family_metrics["test"]["ETH-5M"]
    _write_json(candidate_eval / "family_metrics.json", candidate_family_metrics)

    report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )

    family_check = next(check for check in report.checks if check.name == "required_family_metrics_present")
    assert report.passed is False
    assert family_check.passed is False
    assert "candidate:ETH-5M" in family_check.detail


def test_offline_rerun_report_requires_new_eth_market_signal(tmp_path: Path) -> None:
    from bigan.modeling import generate_offline_rerun_report

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    candidate_family_metrics = json.loads(
        (candidate_eval / "family_metrics.json").read_text(encoding="utf-8")
    )
    candidate_family_metrics["test"]["ETH-15M"]["roc_auc"] = 0.50
    candidate_family_metrics["test"]["ETH-5M"]["roc_auc"] = 0.49
    _write_json(candidate_eval / "family_metrics.json", candidate_family_metrics)

    report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )

    signal_check = next(check for check in report.checks if check.name == "new_market_signal_present")
    assert report.passed is False
    assert signal_check.passed is False
    assert "newly added ETH market family" in signal_check.detail
    assert "ETH-15M: samples=10, roc_auc=0.5000" in signal_check.detail


def test_audit_rejects_dataset_missing_required_market_family(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_manifest = _dataset_manifest(tmp_path / "training")
    dataset_payload = json.loads((dataset_manifest / "manifest.json").read_text(encoding="utf-8"))
    del dataset_payload["family_splits"]["BTC-5M"]
    _write_json(dataset_manifest / "manifest.json", dataset_payload)
    dataset_dir = str(dataset_manifest)
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
    )

    family_check = next(
        check for check in report.stages[1].checks if check.name == "dataset_required_families_present"
    )
    assert family_check.passed is False
    assert "BTC-5M" in family_check.detail


def test_audit_rejects_backtests_that_do_not_match_eval_dataset(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process, generate_offline_rerun_report

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rerun_report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )
    assert rerun_report.passed is True
    baseline_backtest = _backtest(
        tmp_path / "baseline-backtest" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate-backtest" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=str(tmp_path / "old-training"),
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
        offline_rerun_report_path=tmp_path / "rerun_report.md",
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    match_check = next(
        check for check in report.stages[2].checks if check.name == "candidate_backtest_matches_eval"
    )
    assert match_check.passed is False


def test_audit_rejects_backtests_with_stale_dataset_version(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process, generate_offline_rerun_report

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rerun_report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )
    assert rerun_report.passed is True
    baseline_backtest = _backtest(
        tmp_path / "baseline-backtest" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate-backtest" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    diagnostics_path = candidate_backtest.with_name("diagnostics.json")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["metadata"]["dataset_version"] = "old-dataset"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
        offline_rerun_report_path=tmp_path / "rerun_report.md",
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    match_check = next(
        check for check in report.stages[2].checks if check.name == "candidate_backtest_matches_eval"
    )
    assert match_check.passed is False
    assert "old-dataset" in match_check.detail


def test_audit_rejects_backtests_with_stale_model_path(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process, generate_offline_rerun_report

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rerun_report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )
    assert rerun_report.passed is True
    baseline_backtest = _backtest(
        tmp_path / "baseline-backtest" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate-backtest" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=tmp_path / "stale-candidate" / "model.json",
        dataset_dir=dataset_dir,
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
        offline_rerun_report_path=tmp_path / "rerun_report.md",
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    match_check = next(
        check for check in report.stages[2].checks if check.name == "candidate_backtest_matches_eval"
    )
    assert match_check.passed is False
    assert "stale-candidate" in match_check.detail


def test_audit_rejects_backtest_summary_diagnostics_mismatch(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process, generate_offline_rerun_report

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rerun_report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )
    assert rerun_report.passed is True
    baseline_backtest = _backtest(
        tmp_path / "baseline-backtest" / "summary.json",
        net_pnl=1.0,
        sharpe=0.5,
        model_version="xgboost-v3",
        model_path=baseline_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    candidate_backtest = _backtest(
        tmp_path / "candidate-backtest" / "summary.json",
        net_pnl=2.0,
        sharpe=0.6,
        model_version="xgboost-v4",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
    )
    summary = json.loads(candidate_backtest.read_text(encoding="utf-8"))
    summary[0]["net_pnl"] = 3.0
    candidate_backtest.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
        offline_rerun_report_path=tmp_path / "rerun_report.md",
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        baseline_backtest_summary_path=baseline_backtest,
        candidate_backtest_summary_path=candidate_backtest,
    )

    match_check = next(
        check
        for check in report.stages[2].checks
        if check.name == "candidate_backtest_summary_matches_diagnostics"
    )
    assert match_check.passed is False
    assert "summary_rows=3" in match_check.detail


def test_audit_rejects_shadow_report_for_wrong_challenger(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process, generate_offline_rerun_report

    status_path = _ready_status(tmp_path / "status.json", ready=True)
    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rerun_report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=tmp_path / "rerun_report.md",
    )
    assert rerun_report.passed is True
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": True,
            "session_duration_seconds": 86_700,
            "champion_model_version": "xgboost-v3",
            "challenger_model_version": "xgboost-v4-old",
            **_shadow_reference(candidate_eval),
            "challenger_edge_trigger_rate": 0.12,
            "schema_error_rate": 0.0,
            "scoring_error_rate": 0.0,
            "latency_ms": {"xgboost-v4-old": {"p95": 4.0}},
            "simulated_pnl": _shadow_pnl(),
            "checks": {
                "prediction_distribution_stability": {"passed": True},
                "edge_trigger_rate": {"passed": True},
                "simulated_pnl": {"passed": True},
                "prediction_latency": {"passed": True},
                "schema_error_rate": {"passed": True},
                "scoring_error_rate": {"passed": True},
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=status_path,
        offline_rerun_report_path=tmp_path / "rerun_report.md",
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    match_check = next(check for check in report.stages[3].checks if check.name == "shadow_models_match_eval")
    assert match_check.passed is False
    assert "xgboost-v4-old" in match_check.detail


def test_audit_rejects_short_shadow_window_for_full_session_gate(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": True,
            "session_duration_seconds": 1_200,
            "champion_model_version": "xgboost-v3",
            "challenger_model_version": "xgboost-v4",
            **_shadow_reference(candidate_eval),
            "challenger_edge_trigger_rate": 0.12,
            "schema_error_rate": 0.0,
            "scoring_error_rate": 0.0,
            "latency_ms": {"xgboost-v4": {"p95": 4.0}},
            "simulated_pnl": _shadow_pnl(),
            "checks": {
                "prediction_distribution_stability": {"passed": True},
                "edge_trigger_rate": {"passed": True},
                "simulated_pnl": {"passed": True},
                "prediction_latency": {"passed": True},
                "schema_error_rate": {"passed": True},
                "scoring_error_rate": {"passed": True},
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    duration_check = next(
        check for check in report.stages[3].checks if check.name == "full_shadow_session_evidence"
    )
    assert duration_check.passed is False
    assert "duration_seconds=1200.0000" in duration_check.detail
    assert "required >= 86400" in duration_check.detail


def test_audit_rejects_shadow_duration_without_window_bounds(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": True,
            "session_duration_seconds": 86_700,
            "champion_model_version": "xgboost-v3",
            "challenger_model_version": "xgboost-v4",
            **_shadow_reference(candidate_eval),
            "challenger_edge_trigger_rate": 0.12,
            "schema_error_rate": 0.0,
            "scoring_error_rate": 0.0,
            "latency_ms": {"xgboost-v4": {"p95": 4.0}},
            "simulated_pnl": _shadow_pnl(),
            "checks": {
                "prediction_distribution_stability": {"passed": True},
                "edge_trigger_rate": {"passed": True},
                "simulated_pnl": {"passed": True},
                "prediction_latency": {"passed": True},
                "schema_error_rate": {"passed": True},
                "scoring_error_rate": {"passed": True},
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    duration_check = next(
        check for check in report.stages[3].checks if check.name == "full_shadow_session_evidence"
    )
    assert duration_check.passed is False
    assert "window_start_ts=missing" in duration_check.detail
    assert "reported_session_duration_seconds=86700.0000" in duration_check.detail


def test_audit_rejects_shadow_without_scored_rows(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    shadow_payload = {
        "overall_passed": True,
        **_shadow_window(86_700),
        "champion_model_version": "xgboost-v3",
        "challenger_model_version": "xgboost-v4",
        **_shadow_reference(candidate_eval),
        "scored_count": 0,
        "challenger_probability_distribution": {"count": 0, "mean": 0.56, "std": 0.105},
        "challenger_edge_trigger_rate": 0.12,
        "schema_error_rate": 0.0,
        "scoring_error_rate": 0.0,
        "latency_ms": {"xgboost-v4": {"p95": 4.0}},
        "simulated_pnl": _shadow_pnl(),
        "checks": {
            "prediction_distribution_stability": {"passed": True},
            "edge_trigger_rate": {"passed": True},
            "simulated_pnl": {"passed": True},
            "prediction_latency": {"passed": True},
            "schema_error_rate": {"passed": True},
            "scoring_error_rate": {"passed": True},
        },
    }
    shadow_path = _write_json(tmp_path / "shadow.json", shadow_payload)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    scored_check = next(
        check for check in report.stages[3].checks if check.name == "shadow_scored_rows_present"
    )
    assert scored_check.passed is False
    assert "scored_count=0.0000" in scored_check.detail
    assert "challenger_distribution_count=0.0000" in scored_check.detail


def test_audit_rejects_shadow_report_with_stale_offline_reference(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    stale_reference = tmp_path / "old-eval" / "offline_reference.json"
    _write_json(
        stale_reference,
        {
            "model_version": "xgboost-v4",
            "dataset_dir": "old-training",
            "dataset_version": "old-dataset",
            "split": "val",
            "probability_distribution": {"mean": 0.55, "std": 0.10},
        },
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": True,
            "session_duration_seconds": 86_700,
            "champion_model_version": "xgboost-v3",
            "challenger_model_version": "xgboost-v4",
            "offline_reference_path": str(stale_reference),
            "offline_reference": json.loads(stale_reference.read_text(encoding="utf-8")),
            "challenger_edge_trigger_rate": 0.12,
            "schema_error_rate": 0.0,
            "scoring_error_rate": 0.0,
            "latency_ms": {"xgboost-v4": {"p95": 4.0}},
            "simulated_pnl": _shadow_pnl(),
            "checks": {
                "prediction_distribution_stability": {"passed": True},
                "edge_trigger_rate": {"passed": True},
                "simulated_pnl": {"passed": True},
                "prediction_latency": {"passed": True},
                "schema_error_rate": {"passed": True},
                "scoring_error_rate": {"passed": True},
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    reference_check = next(
        check for check in report.stages[3].checks if check.name == "shadow_offline_reference_matches_eval"
    )
    assert reference_check.passed is False
    assert "old-eval" in reference_check.detail
    assert "old-training" in reference_check.detail


def test_audit_rejects_shadow_distribution_drift_even_if_check_flag_passes(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    shadow_payload = {
        "overall_passed": True,
        "session_duration_seconds": 86_700,
        "champion_model_version": "xgboost-v3",
        "challenger_model_version": "xgboost-v4",
        **_shadow_reference(candidate_eval),
        "challenger_probability_distribution": {"count": 10, "mean": 0.70, "std": 0.10},
        "challenger_edge_trigger_rate": 0.12,
        "schema_error_rate": 0.0,
        "scoring_error_rate": 0.0,
        "latency_ms": {"xgboost-v4": {"p95": 4.0}},
        "simulated_pnl": _shadow_pnl(),
        "checks": {
            "prediction_distribution_stability": {"passed": True},
            "edge_trigger_rate": {"passed": True},
            "simulated_pnl": {"passed": True},
            "prediction_latency": {"passed": True},
            "schema_error_rate": {"passed": True},
            "scoring_error_rate": {"passed": True},
        },
    }
    shadow_path = _write_json(tmp_path / "shadow.json", shadow_payload)

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    distribution_check = next(
        check
        for check in report.stages[3].checks
        if check.name == "prediction_distribution_drift_within_bounds"
    )
    assert distribution_check.passed is False
    assert "mean_abs_diff=0.1500" in distribution_check.detail


def test_audit_rejects_shadow_scoring_errors_even_if_overall_flag_passes(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": True,
            "session_duration_seconds": 86_700,
            "champion_model_version": "xgboost-v3",
            "challenger_model_version": "xgboost-v4",
            **_shadow_reference(candidate_eval),
            "challenger_edge_trigger_rate": 0.12,
            "schema_error_rate": 0.0,
            "scoring_error_rate": 0.01,
            "latency_ms": {"xgboost-v4": {"p95": 4.0}},
            "simulated_pnl": _shadow_pnl(),
            "checks": {
                "prediction_distribution_stability": {"passed": True},
                "edge_trigger_rate": {"passed": True},
                "simulated_pnl": {"passed": True},
                "prediction_latency": {"passed": True},
                "schema_error_rate": {"passed": True},
                "scoring_error_rate": {"passed": True},
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    scoring_check = next(
        check for check in report.stages[3].checks if check.name == "scoring_error_rate_zero"
    )
    assert scoring_check.passed is False
    assert "scoring_error_rate=0.0100" in scoring_check.detail


def test_audit_rejects_shadow_pnl_delta_without_component_pnl_evidence(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": True,
            "session_duration_seconds": 86_700,
            "champion_model_version": "xgboost-v3",
            "challenger_model_version": "xgboost-v4",
            **_shadow_reference(candidate_eval),
            "challenger_edge_trigger_rate": 0.12,
            "schema_error_rate": 0.0,
            "scoring_error_rate": 0.0,
            "latency_ms": {"xgboost-v4": {"p95": 4.0}},
            "simulated_pnl": {"net_pnl_delta": 0.5},
            "checks": {
                "prediction_distribution_stability": {"passed": True},
                "edge_trigger_rate": {"passed": True},
                "simulated_pnl": {"passed": True},
                "prediction_latency": {"passed": True},
                "schema_error_rate": {"passed": True},
                "scoring_error_rate": {"passed": True},
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    pnl_check = next(
        check for check in report.stages[3].checks if check.name == "simulated_pnl_beats_champion"
    )
    assert pnl_check.passed is False
    assert "champion_net_pnl=missing" in pnl_check.detail
    assert "challenger_net_pnl=missing" in pnl_check.detail
    assert "net_pnl_delta=0.5000" in pnl_check.detail


def test_audit_rejects_shadow_string_boolean_pass_flags(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": "true",
            "session_duration_seconds": 86_700,
            "champion_model_version": "xgboost-v3",
            "challenger_model_version": "xgboost-v4",
            **_shadow_reference(candidate_eval),
            "challenger_edge_trigger_rate": 0.12,
            "schema_error_rate": 0.0,
            "scoring_error_rate": 0.0,
            "latency_ms": {"xgboost-v4": {"p95": 4.0}},
            "simulated_pnl": _shadow_pnl(),
            "checks": {
                "prediction_distribution_stability": {"passed": "true"},
                "edge_trigger_rate": {"passed": "true"},
                "simulated_pnl": {"passed": "true"},
                "prediction_latency": {"passed": "true"},
                "schema_error_rate": {"passed": "true"},
                "scoring_error_rate": {"passed": "true"},
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        shadow_evaluation_path=shadow_path,
    )

    checks = {check.name: check for check in report.stages[3].checks}
    assert checks["overall_shadow_passed"].passed is False
    assert checks["required_shadow_checks_passed"].passed is False
    assert "overall_passed=true" in checks["overall_shadow_passed"].detail
    assert "prediction_distribution_stability" in checks["required_shadow_checks_passed"].detail


def test_audit_rejects_bootstrap_for_wrong_candidate(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    bootstrap_path = _write_json(
        tmp_path / "bootstrap.json",
        {
            "recommended_action": "PROMOTE_CHAMPION",
            "candidate_model_version": "xgboost-v4-old",
            "missing_or_weak_evidence": [],
            "hard_gate_results": [{"model_version": "xgboost-v4-old", "passed": True}],
            "bootstrap_promotion_checklist": {
                "beats_baseline": True,
                "calibration_acceptable": True,
                "backtest_acceptable": True,
                "serving_readiness_acceptable": True,
                "rollback_fallback_available": True,
                "schema_stable": True,
                "simple_enough": True,
            },
        },
    )
    serving_path = _write_json(
        tmp_path / "serving.json",
        {
            "model_version": "xgboost-v4",
            "ready": True,
            "error_rate": 0.0,
            "p95_latency_ms": 3.0,
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        bootstrap_decision_path=bootstrap_path,
        serving_readiness_path=serving_path,
    )

    match_check = next(
        check for check in report.stages[4].checks if check.name == "bootstrap_candidate_matches_expected"
    )
    assert match_check.passed is False
    assert "xgboost-v4-old" in match_check.detail


def test_audit_rejects_bootstrap_with_incomplete_checklist(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    bootstrap_path = _write_json(
        tmp_path / "bootstrap.json",
        {
            "recommended_action": "PROMOTE_CHAMPION",
            "candidate_model_version": "xgboost-v4",
            "missing_or_weak_evidence": [],
            "hard_gate_results": [{"model_version": "xgboost-v4", "passed": True}],
            "bootstrap_promotion_checklist": {
                "beats_baseline": True,
                "calibration_acceptable": True,
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        bootstrap_decision_path=bootstrap_path,
    )

    checklist_check = next(
        check for check in report.stages[4].checks if check.name == "bootstrap_checklist_passed"
    )
    assert checklist_check.passed is False
    assert "missing checklist items" in checklist_check.detail
    assert "backtest_acceptable" in checklist_check.detail


def test_audit_rejects_bootstrap_string_boolean_evidence(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    bootstrap_path = _write_json(
        tmp_path / "bootstrap.json",
        {
            "recommended_action": "PROMOTE_CHAMPION",
            "candidate_model_version": "xgboost-v4",
            "missing_or_weak_evidence": [],
            "hard_gate_results": [{"model_version": "xgboost-v4", "passed": "true"}],
            "bootstrap_promotion_checklist": {
                "beats_baseline": "true",
                "calibration_acceptable": True,
                "backtest_acceptable": True,
                "serving_readiness_acceptable": True,
                "rollback_fallback_available": True,
                "schema_stable": True,
                "simple_enough": True,
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        bootstrap_decision_path=bootstrap_path,
    )

    checks = {check.name: check for check in report.stages[4].checks}
    assert checks["bootstrap_checklist_passed"].passed is False
    assert "failed checklist items=beats_baseline" in checks["bootstrap_checklist_passed"].detail
    assert checks["bootstrap_hard_gates_passed"].passed is False
    assert "passed_versions=none" in checks["bootstrap_hard_gates_passed"].detail
    assert "expected_candidate_gate_passed=False" in checks["bootstrap_hard_gates_passed"].detail


def test_audit_rejects_bootstrap_without_missing_evidence_field(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    bootstrap_path = _write_json(
        tmp_path / "bootstrap.json",
        {
            "recommended_action": "PROMOTE_CHAMPION",
            "candidate_model_version": "xgboost-v4",
            "hard_gate_results": [{"model_version": "xgboost-v4", "passed": True}],
            "bootstrap_promotion_checklist": {
                "beats_baseline": True,
                "calibration_acceptable": True,
                "backtest_acceptable": True,
                "serving_readiness_acceptable": True,
                "rollback_fallback_available": True,
                "schema_stable": True,
                "simple_enough": True,
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        bootstrap_decision_path=bootstrap_path,
    )

    missing_evidence_check = next(
        check for check in report.stages[4].checks if check.name == "no_missing_or_weak_evidence"
    )
    assert missing_evidence_check.passed is False
    assert "missing_or_weak_evidence=missing" in missing_evidence_check.detail


def test_audit_rejects_bootstrap_without_expected_candidate_hard_gate(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    bootstrap_path = _write_json(
        tmp_path / "bootstrap.json",
        {
            "recommended_action": "PROMOTE_CHAMPION",
            "candidate_model_version": "xgboost-v4",
            "missing_or_weak_evidence": [],
            "hard_gate_results": [{"model_version": "xgboost-v3", "passed": True}],
            "bootstrap_promotion_checklist": {
                "beats_baseline": True,
                "calibration_acceptable": True,
                "backtest_acceptable": True,
                "serving_readiness_acceptable": True,
                "rollback_fallback_available": True,
                "schema_stable": True,
                "simple_enough": True,
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        bootstrap_decision_path=bootstrap_path,
    )

    hard_gate_check = next(
        check for check in report.stages[4].checks if check.name == "bootstrap_hard_gates_passed"
    )
    assert hard_gate_check.passed is False
    assert "passed_versions=xgboost-v3" in hard_gate_check.detail
    assert "expected_candidate_gate_passed=False" in hard_gate_check.detail


def test_audit_rejects_serving_readiness_for_stale_model_path(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rollback_path = tmp_path / "rollback.md"
    rollback_path.write_text("Rollback to xgboost-v3", encoding="utf-8")
    serving_path = _serving_readiness(
        tmp_path / "serving.json",
        model_path=tmp_path / "old-candidate" / "model.json",
        dataset_dir=dataset_dir,
        rollback_runbook_path=rollback_path,
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        candidate_eval_dir=candidate_eval,
        serving_readiness_path=serving_path,
        rollback_runbook_path=rollback_path,
    )

    serving_check = next(
        check for check in report.stages[4].checks if check.name == "serving_readiness"
    )
    assert serving_check.passed is False
    assert "old-candidate" in serving_check.detail
    assert str(candidate_eval / "model.json") in serving_check.detail


def test_audit_rejects_serving_readiness_with_string_boolean_evidence(
    tmp_path: Path,
) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rollback_path = tmp_path / "rollback.md"
    rollback_path.write_text("Rollback to xgboost-v3", encoding="utf-8")
    serving_path = _serving_readiness(
        tmp_path / "serving.json",
        model_path=candidate_eval / "model.json",
        dataset_dir=dataset_dir,
        rollback_runbook_path=rollback_path,
    )
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    serving["ready"] = "true"
    serving["serving_ready"] = "true"
    serving["schema_validation"]["valid_input_accepted"] = "true"
    serving["fallback"]["fallback_model_available"] = "true"
    serving_path.write_text(json.dumps(serving, indent=2, sort_keys=True), encoding="utf-8")

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        candidate_eval_dir=candidate_eval,
        serving_readiness_path=serving_path,
        rollback_runbook_path=rollback_path,
    )

    serving_check = next(
        check for check in report.stages[4].checks if check.name == "serving_readiness"
    )
    assert serving_check.passed is False
    assert "ready=False" in serving_check.detail
    assert "schema_valid=False" in serving_check.detail
    assert "fallback_model_available=False" in serving_check.detail


def test_audit_rejects_bootstrap_with_stale_artifact_paths(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    serving_path = _write_json(
        tmp_path / "serving.json",
        {
            "model_version": "xgboost-v4",
            "ready": True,
            "error_rate": 0.0,
            "p95_latency_ms": 3.0,
        },
    )
    shadow_path = _write_json(tmp_path / "shadow.json", {"overall_passed": True})
    rollback_path = tmp_path / "rollback.md"
    rollback_path.write_text("Rollback to xgboost-v3", encoding="utf-8")
    bootstrap_path = _write_json(
        tmp_path / "bootstrap.json",
        {
            "recommended_action": "PROMOTE_CHAMPION",
            "candidate_model_version": "xgboost-v4",
            "missing_or_weak_evidence": [],
            "hard_gate_results": [{"model_version": "xgboost-v4", "passed": True}],
            "artifact_paths": {
                "baseline_eval_dir": str(baseline_eval),
                "candidate_eval_dir": str(tmp_path / "old-candidate-eval"),
                "serving_readiness_path": str(serving_path),
                "shadow_evaluation_path": str(tmp_path / "old-shadow.json"),
                "rollback_runbook_path": str(rollback_path),
            },
            "bootstrap_promotion_checklist": {
                "beats_baseline": True,
                "calibration_acceptable": True,
                "backtest_acceptable": True,
                "serving_readiness_acceptable": True,
                "rollback_fallback_available": True,
                "schema_stable": True,
                "simple_enough": True,
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        bootstrap_decision_path=bootstrap_path,
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        serving_readiness_path=serving_path,
        shadow_evaluation_path=shadow_path,
        rollback_runbook_path=rollback_path,
    )

    candidate_eval_check = next(
        check for check in report.stages[4].checks if check.name == "bootstrap_uses_current_candidate_eval"
    )
    assert candidate_eval_check.passed is False
    assert "old-candidate-eval" in candidate_eval_check.detail
    shadow_check = next(
        check for check in report.stages[4].checks if check.name == "bootstrap_uses_current_shadow"
    )
    assert shadow_check.passed is False
    assert "old-shadow.json" in shadow_check.detail


def test_audit_rejects_bootstrap_without_current_artifact_paths(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    serving_path = _write_json(
        tmp_path / "serving.json",
        {
            "model_version": "xgboost-v4",
            "ready": True,
            "error_rate": 0.0,
            "p95_latency_ms": 3.0,
        },
    )
    rollback_path = tmp_path / "rollback.md"
    rollback_path.write_text("Rollback to xgboost-v3", encoding="utf-8")
    bootstrap_path = _write_json(
        tmp_path / "bootstrap.json",
        {
            "recommended_action": "PROMOTE_CHAMPION",
            "candidate_model_version": "xgboost-v4",
            "missing_or_weak_evidence": [],
            "hard_gate_results": [{"model_version": "xgboost-v4", "passed": True}],
            "artifact_paths": {
                "serving_readiness_path": str(serving_path),
                "rollback_runbook_path": str(rollback_path),
            },
            "bootstrap_promotion_checklist": {
                "beats_baseline": True,
                "calibration_acceptable": True,
                "backtest_acceptable": True,
                "serving_readiness_acceptable": True,
                "rollback_fallback_available": True,
                "schema_stable": True,
                "simple_enough": True,
            },
        },
    )

    report = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        bootstrap_decision_path=bootstrap_path,
        serving_readiness_path=serving_path,
        rollback_runbook_path=rollback_path,
    )

    missing_checks = {
        check.name: check
        for check in report.stages[4].checks
        if check.name
        in {
            "bootstrap_uses_current_baseline_eval",
            "bootstrap_uses_current_candidate_eval",
            "bootstrap_uses_current_baseline_backtest",
            "bootstrap_uses_current_candidate_backtest",
            "bootstrap_uses_current_shadow",
        }
    }
    assert set(missing_checks) == {
        "bootstrap_uses_current_baseline_eval",
        "bootstrap_uses_current_candidate_eval",
        "bootstrap_uses_current_baseline_backtest",
        "bootstrap_uses_current_candidate_backtest",
        "bootstrap_uses_current_shadow",
    }
    assert all(not check.passed for check in missing_checks.values())
    assert all("expected=None" in check.detail for check in missing_checks.values())


def test_offline_rerun_report_cli_fails_closed_on_failed_report(tmp_path: Path) -> None:
    from typer import Exit

    from bigan.ingestion.__main__ import offline_rerun_report_v1

    dataset_dir = str(_dataset_manifest(tmp_path / "training", train_fraction=0.60))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    output_path = tmp_path / "rerun_report.md"

    with pytest.raises(Exit) as exc_info:
        offline_rerun_report_v1(
            baseline_eval_dir=baseline_eval,
            candidate_eval_dir=candidate_eval,
            output_path=output_path,
            expected_candidate_model_version="xgboost-v4",
            no_fail_on_blocked=False,
        )

    assert exc_info.value.exit_code == 1
    assert "Decision: **FAIL**" in output_path.read_text(encoding="utf-8")


def test_audit_requires_generated_rerun_report_json_sidecar(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rerun_report_path = tmp_path / "rerun_report.md"
    rerun_report_path.write_text(
        "# Rerun Report\n\nDecision: **PASS**\n\nCandidate: `xgboost-v4`\n",
        encoding="utf-8",
    )

    audit = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        offline_rerun_report_path=rerun_report_path,
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
    )

    rerun_check = next(check for check in audit.stages[1].checks if check.name == "rerun_report_exists")
    assert rerun_check.passed is False


def test_audit_rejects_stale_rerun_report_json_sidecar(tmp_path: Path) -> None:
    from bigan.modeling import audit_champion_promotion_process, generate_offline_rerun_report

    dataset_dir = str(_dataset_manifest(tmp_path / "training"))
    baseline_eval = _eval_dir(
        tmp_path / "baseline-eval",
        model_version="xgboost-v3",
        auc=0.70,
        brier=0.24,
        ece=0.06,
        dataset_dir=dataset_dir,
    )
    candidate_eval = _eval_dir(
        tmp_path / "candidate-eval",
        model_version="xgboost-v4",
        auc=0.78,
        brier=0.16,
        ece=0.03,
        dataset_dir=dataset_dir,
        calibrated=True,
    )
    rerun_report_path = tmp_path / "rerun_report.md"
    report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
        output_path=rerun_report_path,
    )
    assert report.passed is True
    sidecar_path = rerun_report_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["dataset_version"] = "stale-dataset"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")

    audit = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        live_status_path=_ready_status(tmp_path / "status.json", ready=True),
        offline_rerun_report_path=rerun_report_path,
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
    )

    rerun_check = next(check for check in audit.stages[1].checks if check.name == "rerun_report_exists")
    assert rerun_check.passed is False
