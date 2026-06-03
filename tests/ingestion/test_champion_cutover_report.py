"""Stage 5 cutover evidence CLI contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bigan.mlops import (
    ModelDeploymentRecord,
    ModelRegistryRecord,
    connect_mlops_db,
    model_artifact_uri,
    record_deployment,
    register_model,
)

TEST_MODEL_FEATURE_COLUMNS = [
    "spread",
    "mid_price",
    "microprice",
    "minute_of_day",
    "day_of_week",
    "underlying_id",
    "horizon_minutes",
    "liquidity_bucket",
    "ret_30m",
    "rv_30m",
    "trade_volume_1m",
    "aggressor_buy_ratio_1m",
    "avg_trade_size_1m",
    "tick_spread",
    "tick_obi_l1",
    "tick_obi_l3",
    "tick_mid_price",
    "tick_price_velocity",
    "tick_trade_arrival_rate",
]


def _offline_reference_payload(reference_path: Path) -> dict[str, object]:
    return {
        "model_version": "xgboost-v4",
        "model_path": "runs/xgboost-v4/model.json",
        "dataset_dir": "runs/training-dataset",
        "dataset_version": "dataset-v1",
        "split": "val",
        "probability_distribution": {"count": 100, "mean": 0.55, "std": 0.10},
        "edge_distribution": {"count": 100, "mean": 0.04, "std": 0.08},
        "edge_trigger_rate_at_0_30": 0.12,
        "reference_path": str(reference_path),
    }


def _drift_baseline_payload(reference_path: Path) -> dict[str, object]:
    payload = _offline_reference_payload(reference_path)
    payload.pop("reference_path")
    return {
        "generated_at_ms": 1_800_000,
        "source_offline_reference_path": str(reference_path),
        **payload,
        "thresholds": {"probability_mean_shift_abs": 0.05},
    }


def _github_issue_closures_payload() -> list[dict[str, object]]:
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


def _materialize_model_member_files(path: Path, payload: object) -> None:
    if path.name != "model.json" or not isinstance(payload, dict):
        return
    if payload.get("schema_version") == "xgboost_ensemble_v1":
        payload.setdefault("model_version", "xgboost-v4")
        payload.setdefault(
            "feature_columns",
            TEST_MODEL_FEATURE_COLUMNS,
        )
        feature_schema_path = path.parent / "feature_schema.json"
        feature_schema_path.parent.mkdir(parents=True, exist_ok=True)
        if not feature_schema_path.exists():
            feature_schema_path.write_text(
                json.dumps(
                    {
                        "feature_columns": payload["feature_columns"],
                        "model_version": "xgboost-v4",
                        "schema_hash": "test-schema-hash",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        feature_importance_path = path.parent / "feature_importance.json"
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        if not feature_importance_path.exists():
            feature_importance_path.write_text(
                json.dumps(
                    [
                        {"feature": "mid_price", "gain": 4.2, "split_count": 8},
                        {"feature": "tick_obi_l1", "gain": 1.7, "split_count": 3},
                    ],
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    members = payload.get("members")
    if not isinstance(members, list):
        return
    for member in members:
        member_path = member.get("path") if isinstance(member, dict) else None
        if not isinstance(member_path, str) or not member_path.strip():
            continue
        target_path = Path(member_path)
        if target_path.is_absolute():
            continue
        target = path.parent / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"seed": member.get("seed") if isinstance(member, dict) else None}),
            encoding="utf-8",
        )


def _materialize_cv_summary_folds(path: Path, payload: object) -> None:
    if path.name != "cv_summary.json" or not isinstance(payload, dict):
        return
    payload.setdefault("model_version", "xgboost-v4")
    if "folds" in payload:
        return
    summary = payload.get("summary")
    fold_count = summary.get("fold_count") if isinstance(summary, dict) else 0
    try:
        parsed_fold_count = int(fold_count)
    except (TypeError, ValueError):
        parsed_fold_count = 0
    fold_metrics = [
        {
            "sample_count": 5,
            "brier_score": 0.12 + fold_idx / 100,
            "roc_auc": 0.70,
            "pnl": 0.20 + fold_idx / 100,
        }
        for fold_idx in range(max(parsed_fold_count, 0))
    ]
    if isinstance(summary, dict) and parsed_fold_count > 0:
        briers = [float(row["brier_score"]) for row in fold_metrics]
        aucs = [float(row["roc_auc"]) for row in fold_metrics]
        summary.setdefault("brier_mean", sum(briers) / len(briers))
        summary.setdefault("brier_std", 0.0)
        summary.setdefault("roc_auc_mean", sum(aucs) / len(aucs))
        summary.setdefault("roc_auc_std", 0.0)
        pnls = [float(row["pnl"]) for row in fold_metrics]
        summary.setdefault("pnl_mean", sum(pnls) / len(pnls))
        summary.setdefault("pnl_std", 0.0)
    payload["folds"] = [
        {
            "fold": fold_idx + 1,
            "train_start_ts": 1_000,
            "train_end_ts": 2_000 + fold_idx * 1_000,
            "val_start_ts": 3_000 + fold_idx * 1_000,
            "val_end_ts": 3_500 + fold_idx * 1_000,
            "train_count": 10 + fold_idx,
            "val_count": 5,
            "metrics": fold_metrics[fold_idx],
        }
        for fold_idx in range(max(parsed_fold_count, 0))
    ]


def _materialize_ensemble_summary(path: Path, payload: object) -> None:
    if path.name != "ensemble_summary.json" or not isinstance(payload, dict):
        return
    if payload.get("schema_version") == "xgboost_ensemble_v1":
        payload.setdefault("model_version", "xgboost-v4")
        member_count = payload.get("member_count")
        if _is_positive_int_like(member_count):
            parsed_member_count = int(float(member_count))
            payload.setdefault("seeds", [0, 17, 42, 73, 101][:parsed_member_count])
            payload.setdefault("train_time_multiplier_estimate", parsed_member_count)
            payload.setdefault("inference_eval_multiplier", parsed_member_count)
        payload.setdefault(
            "single_model_metrics",
            {
                "test": {
                    "sample_count": 100,
                    "brier_score": 0.13,
                    "roc_auc": 0.69,
                    "pnl": 0.19,
                }
            },
        )
        payload.setdefault(
            "ensemble_metrics",
            {
                "test": {
                    "sample_count": 100,
                    "brier_score": 0.12,
                    "roc_auc": 0.70,
                    "pnl": 0.20,
                }
            },
        )
        payload.setdefault(
            "ensemble_vs_single",
            {
                "split": "test",
                "acceptable": True,
                "brier_delta": -0.01,
                "roc_auc_delta": 0.01,
                "pnl_delta": 0.01,
                "rule": "test helper comparison",
            },
        )


def _is_positive_int_like(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed.is_integer() and parsed > 0


def _materialize_feature_ablation_rows(path: Path, payload: object) -> None:
    if path.name != "feature_ablation.json" or not isinstance(payload, dict):
        return
    payload.setdefault("dataset_dir", "runs/training")
    payload.setdefault("dataset_version", "dataset-v1")
    baseline_metrics = payload.get("baseline_metrics")
    sample_count = (
        baseline_metrics.get("sample_count")
        if isinstance(baseline_metrics, dict)
        else 100
    )
    ablations = payload.get("ablations")
    if not isinstance(ablations, list):
        return
    for row in ablations:
        if not isinstance(row, dict):
            continue
        row.setdefault("features", [str(row.get("name") or "feature")])
        row.setdefault(
            "metrics",
            {
                "sample_count": sample_count,
                "brier_score": 0.13,
                "roc_auc": 0.68,
            },
        )
        row.setdefault(
            "deltas",
            {
                "brier_score_increase": 0.01,
                "roc_auc_drop": 0.02,
            },
        )


def _materialize_down_validation_payload(path: Path, payload: object) -> None:
    if path.name != "buy_down_validation.json" or not isinstance(payload, dict):
        return
    metadata = payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.setdefault("backtest_kind", "direct_model")
        metadata.setdefault("dataset_dir", "runs/training")
        metadata.setdefault("dataset_version", "dataset-v1")
        metadata.setdefault("warehouse_dir", "runs/warehouse")
    summary_rows = payload.get("summary")
    if not isinstance(summary_rows, list):
        return
    for row in summary_rows:
        if not isinstance(row, dict):
            continue
        net_pnl = row.get("net_pnl", 0.0)
        signals_considered = row.get("signals_considered", 1)
        trade_count = row.get("trade_count", 1)
        row.setdefault("threshold", 0.03)
        row.setdefault("edge_threshold", row.get("threshold"))
        row.setdefault("threshold_signals", trade_count)
        row.setdefault("gross_pnl", net_pnl)
        row.setdefault("gross_return_sum", net_pnl)
        row.setdefault("net_return_sum", net_pnl)
        row.setdefault("brier_score", 0.16)
        row.setdefault("brier_sample_count", trade_count)
        row.setdefault("symbols_considered", 1)
        row.setdefault("symbols_with_quotes", 1)
        row.setdefault("hold_ms", 900_000)
        try:
            row.setdefault("turnover", float(trade_count) / float(signals_considered))
        except (TypeError, ValueError, ZeroDivisionError):
            row.setdefault("turnover", 0.0)
        settings = row.setdefault("settings", {})
        if isinstance(settings, dict):
            settings.setdefault("fee_bps", 10.0)
            settings.setdefault("slippage_bps", 5.0)
            settings.setdefault("latency_ms", 0)
        _materialize_down_trade_sample(path.parent, row)


def _materialize_down_trade_sample(output_dir: Path, row: dict[str, object]) -> None:
    threshold = row.get("threshold")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return
    suffix = str(threshold_value).replace(".", "_")
    sample_path = output_dir / f"trade_log_sample_threshold_{suffix}.jsonl"
    edge = max(float(row.get("edge_threshold", threshold_value)), threshold_value)
    trade = {
        "threshold": threshold_value,
        "edge_threshold": threshold_value,
        "source": "polymarket",
        "source_symbol": "tok-down",
        "outcome_side": "DOWN",
        "prob_up_15m": min(1.0, 0.30 + edge),
        "market_implied_prob": 0.30,
        "realized_label": False,
        "edge": edge,
        "decision_ts": 1_779_540_480_000,
        "entry_ts": 1_779_540_480_000,
        "exit_ts": 1_779_541_380_000,
        "net_pnl": row.get("net_pnl", 0.1),
        "fee_bps": row.get("settings", {}).get("fee_bps", 10.0)
        if isinstance(row.get("settings"), dict)
        else 10.0,
        "slippage_bps": row.get("settings", {}).get("slippage_bps", 5.0)
        if isinstance(row.get("settings"), dict)
        else 5.0,
    }
    sample_path.write_text(json.dumps(trade, sort_keys=True) + "\n", encoding="utf-8")


def _write_json(
    path: Path,
    payload: object,
    *,
    materialize_ablation_metrics: bool = True,
    materialize_cv_folds: bool = True,
    materialize_down_validation: bool = True,
    materialize_model_members: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if materialize_cv_folds:
        _materialize_cv_summary_folds(path, payload)
    _materialize_ensemble_summary(path, payload)
    if materialize_model_members:
        _materialize_model_member_files(path, payload)
    if materialize_ablation_metrics:
        _materialize_feature_ablation_rows(path, payload)
    if materialize_down_validation:
        _materialize_down_validation_payload(path, payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_slack_automation_toml(path: Path, *, channel_id: str = "C0B5VHYSCN8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'id = "xgboost-v4-work-status"',
                'kind = "heartbeat"',
                'status = "ACTIVE"',
                'rrule = "FREQ=HOURLY;INTERVAL=1"',
                (
                    'prompt = "Run xgboost-v4-objective-audit --no-fail-on-blocked; '
                    'refresh xgboost-v4-issue-coverage-audit and '
                    'issue_coverage_audit.json; '
                    'if collection_readiness.ready_for_training is true run '
                    '/bin/bash scripts/run_xgboost_v4_post_readiness.sh with '
                    'POST_READINESS_SENTINEL_PATH and POST_READINESS_LOCK_DIR guards; '
                    'inspect xgboost_v4_post_readiness_latest.json, read run_root, '
                    'run_manifest_phase, live_status_summary, artifact_paths, '
                    'promotion_audit_path, and objective_audit_path, then rerun '
                    'xgboost-v4-objective-audit --promotion-audit-path '
                    '<promotion_audit_path> --candidate-model-dir '
                    '<run_root>/artifacts/models/xgboost-v4 --feature-ablation-path '
                    '<run_root>/artifacts/models/xgboost-v4-feature-ablation/'
                    'feature_ablation.json --stability-report-path '
                    '<run_root>/artifacts/dataset-stability/dataset_stability_report.json '
                    '--down-validation-path '
                    '<run_root>/artifacts/backtests/xgboost-v4-down/diagnostics.json '
                    '--output-path <objective_audit_path>; '
                    'if later gates are blocked only by missing shadow evidence, use '
                    'CONTINUE_POST_READINESS_RUN=true RUN_ROOT=<run_root> '
                    'RUN_SHADOW=true SHADOW_SINCE_MS=auto SHADOW_UNTIL_MS=auto '
                    'MIN_SHADOW_SESSION_SECONDS=86400; '
                    'report objective_success_criteria, prompt_to_artifact_blockers, '
                    'create_xgboost_v4_model, and beat_current_champion blockers; '
                    'update slack_status_delivery_latest.json with message_link and '
                    'error_code using slack-delivery-status --message-link '
                    '--error-code --error-message --output-path '
                    'slack_status_delivery_latest.json; rerun '
                    'xgboost-v4-objective-audit with '
                    '--slack-delivery-status-path slack_status_delivery_latest.json; '
                    'if disk_headroom_evidence.headroom_low_margin=true or '
                    'disk_headroom_evidence.headroom_ok=false, run '
                    'scripts/check_xgboost_v4_collection_risk.sh --json '
                    '--output-path data/xgboost-v4-run-20260523T103814Z/'
                    'artifacts/collection_risk_latest.json and parse '
                    'collection_risk_latest.json plus '
                    'status_artifact.fresh, status_artifact.age_seconds, '
                    'status_artifact.max_age_seconds, current_filesystem_headroom, '
                    'reclaim_to_clear_block_bytes, '
                    'disk_urgency.estimated_growth_bytes_per_day, '
                    'disk_urgency.current_filesystem_days_to_min_free, and '
                    'disk_urgency.current_filesystem_min_free_before_ready, and '
                    'reclaim_candidates; report Docker and CoreSimulator '
                    'reclaim candidates plus the days-to-min-free urgency clock; '
                    'include whether min-free arrives before readiness; '
                    'do not prune Docker, delete simulator data, '
                    'clear caches, or remove old roots without explicit user approval; '
                    'if disk_headroom_evidence.headroom_ok=false or '
                    'current_disk_headroom_evidence.headroom_ok=false, skip bounded '
                    'settled-label refresh to avoid adding avoidable writes; '
                    f'then send hourly Slack status to channel_id {channel_id}."'
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_dataset_stability_report(
    path: Path,
    *,
    dataset_dir: str = "runs/training",
    dataset_version: str = "dataset-v1",
    feature_columns: list[str] | None = None,
) -> Path:
    _write_json(
        path,
        {
            "schema_version": "dataset_stability_report_v1",
            "dataset_dir": dataset_dir,
            "dataset_version": dataset_version,
            "feature_columns": feature_columns or TEST_MODEL_FEATURE_COLUMNS,
            "feature_versions": ["bigan-mvp-v1.0.0"],
            "label_versions": ["bigan-labels-15m-profitability-v1.1.0"],
            "core_features": ["spread", "microprice", "trade_volume_1m"],
            "split_label_summary": {
                split: {"row_count": 10, "positive_rate": 0.5}
                for split in ("train", "val", "test")
            },
            "core_feature_distributions": {
                feature: {
                    split: {"count": 10, "mean": 0.1, "std": 0.01}
                    for split in ("train", "val", "test")
                }
                for feature in ("spread", "microprice", "trade_volume_1m")
            },
            "family_splits": {
                family: {
                    split: {"row_count": 3}
                    for split in ("train", "val", "test")
                }
                for family in ("BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M")
            },
        },
    )
    path.with_suffix(".md").write_text("# Dataset Stability Report\n", encoding="utf-8")
    return path


def test_issue_64_signal_label_evidence_exercises_down_side_contracts() -> None:
    from bigan.ingestion.__main__ import _issue_64_signal_label_evidence

    evidence = _issue_64_signal_label_evidence()

    assert evidence["passed"] is True
    assert evidence["signal_checks"] == {
        "low_edge_flat_holds": True,
        "buy_down_opens_down_position": True,
        "open_position_holds_above_exit_threshold": True,
        "sell_requires_open_position": True,
        "sell_profit_target_triggers": True,
        "sell_round_end_profit_triggers": True,
        "buy_down_log_names_signal": True,
        "sell_log_names_signal": True,
    }
    assert evidence["signals"] == {
        "low_edge_flat": "HOLD",
        "buy_down": "BUY_DOWN",
        "hold_open": "HOLD",
        "sell": "SELL",
        "profit_target_sell": "SELL",
        "round_end_sell": "SELL",
    }
    assert "BUY_DOWN" in evidence["signal_logs"]["buy_down"]
    assert "SELL" in evidence["signal_logs"]["sell"]
    assert "event=audit-buy-down" in evidence["signal_logs"]["buy_down"]
    assert "event=audit-sell-down" in evidence["signal_logs"]["sell"]
    assert evidence["label_checks"] == {
        "generated_one_down_label": True,
        "label_kind_is_down": True,
        "down_profit_field_populated": True,
        "up_profit_field_empty": True,
        "down_win_settlement_profitable": True,
    }
    assert evidence["label_kind"] == "down_token_profitability"
    assert evidence["label_profit_down_15m"] is True
    assert evidence["label_profit_up_15m"] is None
    assert evidence["settlement_price"] == pytest.approx(1.0)
    assert evidence["realized_return"] == pytest.approx(0.60)


def _registry_record(version: str, *, status: str = "champion") -> ModelRegistryRecord:
    return ModelRegistryRecord(
        model_version=version,
        model_family="btc-updown-15m",
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="dataset-v1",
        train_config_hash=f"hash-{version}",
        artifact_uri=model_artifact_uri(
            "models",
            model_family="btc-updown-15m",
            model_version=version,
        ),
        calibration_artifact_uri=model_artifact_uri(
            "models",
            model_family="btc-updown-15m",
            model_version=version,
            filename="calibration.json",
        ),
        status=status,
        train_started_at=1_000,
        train_finished_at=2_000,
        metrics_json=json.dumps(
            {
                "test": {"roc_auc": 0.7, "brier_score": 0.18},
                "promotion_metrics": {
                    "delta_vs_baseline": 0.5,
                    "edge_trigger_rate": 0.12,
                    "shadow_p95_ms": 4.0,
                    "schema_error_rate": 0.0,
                },
            }
        ),
        backtest_json=json.dumps({"net_pnl": 1.0, "max_drawdown": 0.10, "sharpe": 0.80}),
        promoted_at=3_000 if status == "champion" else None,
    )


def _seed_cutover_catalog(db_path: Path) -> None:
    conn = connect_mlops_db(db_path)
    register_model(conn, _registry_record("xgboost-v3", status="retired"))
    register_model(conn, _registry_record("xgboost-v4"))
    record_deployment(
        conn,
        ModelDeploymentRecord(
            deployment_id="cutover-xgboost-v4-test",
            model_version="xgboost-v4",
            environment="prod",
            rollout_strategy="all-at-once",
            traffic_percent=100.0,
            deployment_status="succeeded",
            started_at=4_000,
            completed_at=5_000,
            rollback_to_version="xgboost-v3",
            operator="codex",
            reason="test cutover",
        ),
    )


def test_champion_cutover_report_cli_writes_audit_compatible_json(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import champion_cutover_report_v1
    from bigan.modeling import audit_champion_promotion_process

    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    register_model(conn, _registry_record("xgboost-v3", status="retired"))
    register_model(conn, _registry_record("xgboost-v4"))
    record_deployment(
        conn,
        ModelDeploymentRecord(
            deployment_id="cutover-xgboost-v4-test",
            model_version="xgboost-v4",
            environment="prod",
            rollout_strategy="all-at-once",
            traffic_percent=100.0,
            deployment_status="succeeded",
            started_at=4_000,
            completed_at=5_000,
            rollback_to_version="xgboost-v3",
            operator="codex",
            reason="test cutover",
        ),
    )

    smoke_path = _write_json(
        tmp_path / "smoke.json",
        {
            "passed": True,
            "model_version": "xgboost-v4",
            "model_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
            ),
            "calibration_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
                filename="calibration.json",
            ),
            "error_rate": 0.0,
            "serving_latency_ms": 2.5,
        },
    )
    offline_reference_path = tmp_path / "candidate-eval" / "offline_reference.json"
    offline_reference = _offline_reference_payload(offline_reference_path)
    offline_reference.pop("reference_path")
    _write_json(offline_reference_path, offline_reference)
    _write_json(
        tmp_path / "candidate-eval" / "manifest.json",
        {
            "model_version": "xgboost-v4",
            "model_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
            ),
            "calibration_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
                filename="calibration.json",
            ),
            "dataset_dir": "runs/training-dataset",
            "dataset_version": "dataset-v1",
        },
    )
    drift_path = _write_json(
        tmp_path / "drift-baseline.json",
        _drift_baseline_payload(offline_reference_path),
    )
    bootstrap_path = _write_json(tmp_path / "bootstrap.json", {"recommended_action": "PROMOTE_CHAMPION"})
    shadow_path = _write_json(
        tmp_path / "shadow.json",
        {
            "overall_passed": True,
            "offline_reference_path": str(offline_reference_path),
            "offline_reference": offline_reference,
        },
    )
    serving_path = _write_json(tmp_path / "serving.json", {"ready": True})
    github_issue_closures_path = _write_json(
        tmp_path / "github-issue-closures.json",
        _github_issue_closures_payload(),
    )
    output_path = tmp_path / "cutover.json"

    champion_cutover_report_v1(
        output_path=output_path,
        monitoring_db_path=db_path,
        model_family="btc-updown-15m",
        environment="prod",
        smoke_path=smoke_path,
        drift_baseline_path=drift_path,
        bootstrap_decision_path=bootstrap_path,
        shadow_evaluation_path=shadow_path,
        serving_readiness_path=serving_path,
        github_issue_closures_path=github_issue_closures_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["current_champion"]["model_version"] == "xgboost-v4"
    assert payload["current_online_model"]["model_version"] == "xgboost-v4"
    assert payload["current_online_model"]["rollback_to_version"] == "xgboost-v3"
    assert payload["fallback_registry_model"]["model_version"] == "xgboost-v3"
    assert payload["smoke"]["serving_latency_ms"] == 2.5
    assert payload["drift_baseline_path"] == str(drift_path)
    assert payload["github_issue_closures"] == _github_issue_closures_payload()
    assert payload["evidence"] == {
        "smoke": str(smoke_path),
        "bootstrap": str(bootstrap_path),
        "shadow": str(shadow_path),
        "serving_readiness": str(serving_path),
        "github_issue_closures": str(github_issue_closures_path),
    }

    audit = audit_champion_promotion_process(
        output_dir=tmp_path / "audit",
        cutover_report_path=output_path,
        candidate_eval_dir=tmp_path / "candidate-eval",
        bootstrap_decision_path=bootstrap_path,
        shadow_evaluation_path=shadow_path,
        serving_readiness_path=serving_path,
    )
    assert audit.stages[5].passed is True


def test_champion_cutover_report_cli_requires_github_issue_closures_path(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import champion_cutover_report_v1

    output_path = tmp_path / "cutover.json"
    with pytest.raises(Exception, match="github_issue_closures_path is required"):
        champion_cutover_report_v1(
            output_path=output_path,
            monitoring_db_path=tmp_path / "mlops.duckdb",
            model_family="btc-updown-15m",
            environment="prod",
            smoke_path=_write_json(tmp_path / "smoke.json", {}),
            drift_baseline_path=_write_json(tmp_path / "drift-baseline.json", {}),
            bootstrap_decision_path=_write_json(tmp_path / "bootstrap.json", {}),
            shadow_evaluation_path=_write_json(tmp_path / "shadow.json", {}),
            serving_readiness_path=_write_json(tmp_path / "serving.json", {}),
        )

    assert not output_path.exists()


def test_champion_cutover_report_cli_rejects_invalid_github_issue_closures(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import champion_cutover_report_v1

    db_path = tmp_path / "mlops.duckdb"
    _seed_cutover_catalog(db_path)
    smoke_path = _write_json(
        tmp_path / "smoke.json",
        {
            "passed": True,
            "model_version": "xgboost-v4",
            "model_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
            ),
            "calibration_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
                filename="calibration.json",
            ),
            "error_rate": 0.0,
            "serving_latency_ms": 2.5,
        },
    )
    closures = _github_issue_closures_payload()
    closures[1]["state"] = "open"
    github_issue_closures_path = _write_json(
        tmp_path / "github-issue-closures.json",
        closures,
    )
    output_path = tmp_path / "cutover.json"

    with pytest.raises(
        Exception,
        match="github issue closure evidence is not ready for Stage 5",
    ):
        champion_cutover_report_v1(
            output_path=output_path,
            monitoring_db_path=db_path,
            model_family="btc-updown-15m",
            environment="prod",
            smoke_path=smoke_path,
            drift_baseline_path=_write_json(tmp_path / "drift-baseline.json", {}),
            bootstrap_decision_path=_write_json(
                tmp_path / "bootstrap.json",
                {"recommended_action": "PROMOTE_CHAMPION"},
            ),
            shadow_evaluation_path=_write_json(
                tmp_path / "shadow.json",
                {"overall_passed": True},
            ),
            serving_readiness_path=_write_json(
                tmp_path / "serving.json",
                {"ready": True},
            ),
            github_issue_closures_path=github_issue_closures_path,
        )

    assert not output_path.exists()


def test_champion_cutover_report_cli_rejects_stale_smoke_artifact(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import champion_cutover_report_v1

    db_path = tmp_path / "mlops.duckdb"
    _seed_cutover_catalog(db_path)
    smoke_path = _write_json(
        tmp_path / "smoke.json",
        {
            "passed": True,
            "model_version": "xgboost-v4",
            "model_path": str(tmp_path / "old-xgboost-v4" / "model.json"),
            "calibration_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
                filename="calibration.json",
            ),
            "error_rate": 0.0,
            "serving_latency_ms": 2.5,
        },
    )
    drift_path = _write_json(tmp_path / "drift-baseline.json", {})
    bootstrap_path = _write_json(tmp_path / "bootstrap.json", {})
    shadow_path = _write_json(tmp_path / "shadow.json", {})
    serving_path = _write_json(tmp_path / "serving.json", {})
    github_issue_closures_path = _write_json(
        tmp_path / "github-issue-closures.json",
        _github_issue_closures_payload(),
    )
    output_path = tmp_path / "cutover.json"

    with pytest.raises(Exception, match="cutover smoke does not match current champion"):
        champion_cutover_report_v1(
            output_path=output_path,
            monitoring_db_path=db_path,
            model_family="btc-updown-15m",
            environment="prod",
            smoke_path=smoke_path,
            drift_baseline_path=drift_path,
            bootstrap_decision_path=bootstrap_path,
            shadow_evaluation_path=shadow_path,
            serving_readiness_path=serving_path,
            github_issue_closures_path=github_issue_closures_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("bootstrap_payload", "shadow_payload", "serving_payload", "expected_detail"),
    [
        (
            {"recommended_action": "KEEP_BASELINE_TEMPORARILY"},
            {"overall_passed": True},
            {"ready": True},
            "bootstrap_promotes",
        ),
        (
            {"recommended_action": "PROMOTE_CHAMPION"},
            {"overall_passed": False},
            {"ready": True},
            "shadow_passed",
        ),
        (
            {"recommended_action": "PROMOTE_CHAMPION"},
            {"overall_passed": True},
            {"ready": False},
            "serving_ready",
        ),
    ],
)
def test_champion_cutover_report_cli_rejects_blocked_stage_evidence(
    tmp_path: Path,
    bootstrap_payload: dict[str, object],
    shadow_payload: dict[str, object],
    serving_payload: dict[str, object],
    expected_detail: str,
) -> None:
    from bigan.ingestion.__main__ import champion_cutover_report_v1

    db_path = tmp_path / "mlops.duckdb"
    _seed_cutover_catalog(db_path)
    smoke_path = _write_json(
        tmp_path / "smoke.json",
        {
            "passed": True,
            "model_version": "xgboost-v4",
            "model_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
            ),
            "calibration_path": model_artifact_uri(
                "models",
                model_family="btc-updown-15m",
                model_version="xgboost-v4",
                filename="calibration.json",
            ),
            "error_rate": 0.0,
            "serving_latency_ms": 2.5,
        },
    )
    drift_path = _write_json(tmp_path / "drift-baseline.json", {})
    bootstrap_path = _write_json(tmp_path / "bootstrap.json", bootstrap_payload)
    shadow_path = _write_json(tmp_path / "shadow.json", shadow_payload)
    serving_path = _write_json(tmp_path / "serving.json", serving_payload)
    github_issue_closures_path = _write_json(
        tmp_path / "github-issue-closures.json",
        _github_issue_closures_payload(),
    )
    output_path = tmp_path / "cutover.json"

    with pytest.raises(Exception, match=expected_detail):
        champion_cutover_report_v1(
            output_path=output_path,
            monitoring_db_path=db_path,
            model_family="btc-updown-15m",
            environment="prod",
            smoke_path=smoke_path,
            drift_baseline_path=drift_path,
            bootstrap_decision_path=bootstrap_path,
            shadow_evaluation_path=shadow_path,
            serving_readiness_path=serving_path,
            github_issue_closures_path=github_issue_closures_path,
        )

    assert not output_path.exists()


def test_champion_state_snapshot_cli_writes_incumbent_state(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import champion_state_snapshot_v1

    db_path = tmp_path / "mlops.duckdb"
    conn = connect_mlops_db(db_path)
    register_model(conn, _registry_record("xgboost-v3", status="retired"))
    register_model(conn, _registry_record("xgboost-v4"))
    record_deployment(
        conn,
        ModelDeploymentRecord(
            deployment_id="current-xgboost-v4-test",
            model_version="xgboost-v4",
            environment="prod",
            rollout_strategy="all-at-once",
            traffic_percent=100.0,
            deployment_status="succeeded",
            started_at=4_000,
            completed_at=5_000,
            rollback_to_version="xgboost-v3",
            operator="codex",
            reason="test current state",
        ),
    )
    output_path = tmp_path / "current_champion_before_retrain.json"

    champion_state_snapshot_v1(
        output_path=output_path,
        monitoring_db_path=db_path,
        model_family="btc-updown-15m",
        environment="prod",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mlops_db_path"] == str(db_path)
    assert payload["model_family"] == "btc-updown-15m"
    assert payload["environment"] == "prod"
    assert payload["registry_champion"]["model_version"] == "xgboost-v4"
    assert payload["online_model"]["model_version"] == "xgboost-v4"
    assert payload["online_model"]["rollback_to_version"] == "xgboost-v3"
    assert payload["fallback_registry_model"]["model_version"] == "xgboost-v3"
    assert payload["fallback_registry_model"]["status"] == "retired"


def test_drift_baseline_cli_writes_reference_artifact(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import drift_baseline_v1

    offline_reference_path = tmp_path / "candidate-eval" / "offline_reference.json"
    offline_reference = _offline_reference_payload(offline_reference_path)
    offline_reference.pop("reference_path")
    _write_json(offline_reference_path, offline_reference)
    output_path = tmp_path / "cutover" / "drift-baseline.json"

    drift_baseline_v1(
        offline_reference_path=offline_reference_path,
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_offline_reference_path"] == str(offline_reference_path)
    assert payload["model_version"] == "xgboost-v4"
    assert payload["dataset_dir"] == "runs/training-dataset"
    assert payload["dataset_version"] == "dataset-v1"
    assert payload["split"] == "val"
    assert payload["probability_distribution"] == offline_reference[
        "probability_distribution"
    ]
    assert payload["thresholds"]["edge_threshold"] == 0.30


def _live_status_payload(*, ready: bool) -> dict[str, object]:
    span = 7.0 if ready else 0.25
    return {
        "generated_at": "2026-05-23T18:56:41Z",
        "screen_session": "xgbv4_7d_atomic_20260523T125657Z",
        "screen_state": "running",
        "live_root": "data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z",
        "warehouse": "data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z/warehouse",
        "raw_segment_count": 359,
        "processed_manifest_rows": 357,
        "raw_segment_integrity": {"invalid_count": 0},
        "raw_segment_quarantine": {"quarantined_count": 0},
        "live_root_lock_evidence": {
            "lock_dir_exists": True,
            "pid_file_exists": True,
            "pid": 12345,
            "owner_running": True,
            "pid_parse_error": None,
        },
        "raw_manifest_coverage_evidence": {
            "stale_missing_processed_count": 0,
            "extra_processed_count": 0,
        },
        "disk_headroom_evidence": {
            "headroom_ok": True,
            "free_bytes": 10_000,
            "required_free_bytes": 1_000,
            "projected_remaining_bytes": 0,
        },
        "health_evidence": {"unrecovered_error_match_count": 0},
        "liveness_evidence": {
            "raw_segments_fresh": True,
            "processed_manifest_fresh": True,
        },
        "warehouse_freshness_evidence": {
            "tables": {
                table: {
                    "fresh": True,
                    "missing_families": [],
                    "families": {
                        family: {"fresh": True}
                        for family in ("BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M")
                    },
                }
                for table in ("features_15m_v1", "predictions")
            }
        },
        "label_freshness_evidence": {
            "fresh": True,
            "missing_label_families": [],
            "stale_families": [],
            "families": {
                family: {"fresh": True}
                for family in ("BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M")
            },
        },
        "monitoring_outcome_evidence": {
            "available": True,
            "model_version": "xgboost-v4",
            "event_rows": 100,
            "outcome_rows": 80,
            "brier_score": 0.42,
            "hit_rate": 0.55,
            "avg_realized_return": 0.01,
            "missing_event_families": [],
            "missing_outcome_families": [],
            "families": {
                family: {
                    "event_rows": 25,
                    "outcome_rows": 20,
                    "brier_score": 0.42,
                    "hit_rate": 0.55,
                    "avg_realized_return": 0.01,
                }
                for family in ("BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M")
            },
        },
        "collection_readiness": {
            "ready_for_training": ready,
            "estimated_ready_at": None if ready else "2026-05-30T12:54:00Z",
            "required_families": ["BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M"],
            "features_15m_v1": {"min_family_span_days": span},
            "labels_15m_v1": {"min_family_span_days": span},
        },
        "totals": {"features_15m_v1_rows": 1000, "labels_15m_v1_rows": 800},
    }


def _collection_risk_payload(
    *,
    status_path: Path,
    live_status: dict[str, object],
) -> dict[str, object]:
    generated_at = live_status["generated_at"]
    return {
        "status_path": str(status_path),
        "generated_at": generated_at,
        "status_artifact": {
            "generated_at": generated_at,
            "age_seconds": 12,
            "max_age_seconds": 1800,
            "fresh": True,
        },
        "status_level": "ok",
        "blocked": False,
        "exit_code": 0,
        "readiness": {"ready_for_training": True},
        "disk_headroom": {
            "headroom_ok": True,
            "reclaim_to_clear_block_bytes": 0,
            "reclaim_to_clear_low_margin_bytes": 0,
        },
        "current_filesystem_headroom": {
            "headroom_ok": True,
            "reclaim_to_clear_block_bytes": 0,
            "reclaim_to_clear_low_margin_bytes": 0,
        },
        "disk_urgency": {
            "estimated_growth_bytes_per_day": 0,
            "current_filesystem_days_to_min_free": None,
            "current_filesystem_min_free_before_ready": False,
        },
        "reclaim_candidates": [],
    }


def _promotion_audit_payload(
    *,
    passed: bool,
    candidate_eval_dir: Path | None = None,
    include_process_source: bool = True,
) -> dict[str, object]:
    stage1_checks = [
        "rerun_report_exists",
        "baseline_eval_exists",
        "candidate_eval_exists",
        "same_dataset_split",
        "dataset_time_split_5_1_1",
        "dataset_required_families_present",
        "required_family_metrics_present",
        "new_market_signal_present",
        "candidate_model_version",
        "test_auc_beats_champion",
        "test_brier_beats_champion",
        "calibrated_ece",
        "candidate_calibration_applied",
    ]
    stage2_checks = [
        "baseline_backtest_exists",
        "candidate_backtest_exists",
        "net_pnl_beats_champion",
        "realistic_nonzero_costs",
    ]
    stage0_checks = [
        {"name": "clean_atomic_live_root", "passed": passed},
        {"name": "status_artifact_fresh", "passed": passed},
        {"name": "collector_process_liveness", "passed": passed},
        {"name": "raw_manifest_coverage", "passed": passed},
    ]
    if include_process_source:
        stage0_checks.append({"name": "promotion_process_source", "passed": passed})
    payload: dict[str, object] = {
        "decision": "PROMOTION_COMPLETE" if passed else "BLOCKED",
        "passed": passed,
        "artifact_paths": {
            "candidate_eval_dir": str(candidate_eval_dir) if candidate_eval_dir else None,
        },
        "stages": [
            {
                "name": "Stage 0: 7-day Data Readiness",
                "passed": passed,
                "checks": stage0_checks,
            },
            {
                "name": "Stage 1: Offline Evaluation",
                "passed": passed,
                "checks": [{"name": check_name, "passed": passed} for check_name in stage1_checks],
            },
            {
                "name": "Stage 2: Cost-Adjusted Backtest",
                "passed": passed,
                "checks": [{"name": check_name, "passed": passed} for check_name in stage2_checks],
            },
            {"name": "Stage 3: Shadow Evaluation", "passed": passed},
            {"name": "Stage 4: Bootstrap Decision", "passed": passed},
            {
                "name": "Stage 5: Champion Cutover",
                "passed": passed,
                "checks": [{"name": "github_issue_closures_recorded", "passed": passed}],
            },
        ],
    }
    if include_process_source:
        payload["promotion_process"] = {
            "checked": True,
            "passed": passed,
            "source_path": "/Users/tcscoder/Downloads/champion-promotion.md",
            "source_exists": True,
            "source_sha256": "d885a5f8388c8ba9a7ad4b469337b95c854ab781b2edac36ce65e7f05ac0f830",
            "missing_required_markers": [],
            "repo_mirror_path": "docs/runbooks/champion_promotion.md",
            "repo_mirror_declares_source": True,
        }
    return payload


def _write_tick_ready_candidate_model_dir(tmp_path: Path) -> Path:
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "model_version": "xgboost-v4",
            "members": [],
            "feature_columns": TEST_MODEL_FEATURE_COLUMNS,
        },
    )
    return candidate_model_dir


def _write_candidate_eval_manifest(
    path: Path,
    *,
    candidate_model_dir: Path,
) -> Path:
    return _write_json(
        path / "manifest.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "dataset_dir": "runs/training",
            "dataset_version": "dataset-v1",
        },
    ).parent


def _write_complete_xgboost_v4_candidate_evidence(tmp_path: Path) -> dict[str, Path]:
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json"
    )
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    return {
        "candidate_model_dir": candidate_model_dir,
        "candidate_eval_dir": candidate_eval_dir,
        "feature_ablation_path": feature_ablation_path,
        "down_validation_path": down_validation_path,
        "stability_report_path": stability_report_path,
    }


def _write_post_readiness_latest(
    path: Path,
    *,
    run_root: Path,
    run_manifest_path: Path,
    objective_audit_path: Path,
    promotion_audit_path: Path,
    candidate_model_dir: Path,
    feature_ablation_path: Path,
    stability_report_path: Path,
    down_validation_path: Path,
) -> Path:
    issue_coverage_audit_path = run_root / "artifacts" / "issue_coverage_audit.json"
    return _write_json(
        path,
        {
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "completion_scope": "post_readiness_runner_completed",
            "run_id": run_root.name,
            "run_root": str(run_root),
            "run_manifest_path": str(run_manifest_path),
            "run_manifest_phase": "completed",
            "live_status_summary": {
                "exists": True,
                "ready_for_training": True,
                "live_root": "data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z",
            },
            "artifact_paths": {
                "run_root": str(run_root),
                "candidate_model_path": str(candidate_model_dir / "model.json"),
                "feature_ablation_path": str(feature_ablation_path),
                "dataset_stability_report_path": str(stability_report_path),
                "down_validation_path": str(down_validation_path),
                "promotion_audit_path": str(promotion_audit_path),
                "objective_audit_path": str(objective_audit_path),
                "issue_coverage_audit_path": str(issue_coverage_audit_path),
            },
            "promotion_audit_path": str(promotion_audit_path),
            "promotion_decision": "PROMOTION_COMPLETE",
            "promotion_passed": True,
            "objective_audit_path": str(objective_audit_path),
            "issue_coverage_audit_path": str(issue_coverage_audit_path),
            "issue_coverage_generated_at": datetime.now(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "issue_coverage_issue_checks": {
                "#54": {"passed": True},
                "#55": {"passed": True},
                "#56": {"passed": True},
                "#57": {"passed": True},
                "#58": {"passed": True},
                "#64": {"passed": True},
                "#65": {"passed": True},
            },
            "issue_coverage_objective_success_criteria": {
                "all_requested_github_issues_satisfied": {"passed": True},
                "fresh_xgboost_v4_model_created": {"passed": True},
                "beats_current_champion": {"passed": True},
                "champion_promotion_gates_passed": {"passed": True},
                "hourly_slack_status_active": {"passed": True},
                "post_readiness_latest_pointer_valid": {"passed": True},
            },
            "objective_decision": "COMPLETE",
            "objective_complete": True,
        },
    )


def test_xgboost_v4_objective_audit_blocks_until_final_issue_and_promotion_evidence(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=False))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=False),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["objective_complete"] is False
    assert payload["decision"] == "BLOCKED"
    assert payload["promotion"]["earliest_failed_stage"] == "Stage 0: 7-day Data Readiness"
    assert "#55" in " ".join(payload["blockers"])
    assert "#57" in " ".join(payload["blockers"])
    assert "#64" in " ".join(payload["blockers"])


def test_xgboost_v4_issue_coverage_audit_refreshes_from_current_gate_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bigan.ingestion import __main__ as ingestion_main
    from bigan.ingestion.__main__ import (
        xgboost_v4_issue_coverage_audit,
        xgboost_v4_objective_audit,
    )

    live_root = tmp_path / "live"
    live_root.mkdir()
    status = _live_status_payload(ready=False)
    status["live_root"] = str(live_root)
    status["warehouse"] = str(live_root / "warehouse")
    status["family_counts"] = {
        "features_15m_v1": {
            "BTC-15M": 10,
            "BTC-5M": 20,
            "ETH-15M": 11,
            "ETH-5M": 21,
        },
        "labels_15m_v1": {
            "BTC-15M": 8,
            "BTC-5M": 18,
            "ETH-15M": 9,
            "ETH-5M": 19,
        },
        "predictions": {
            "BTC-15M": 7,
            "BTC-5M": 17,
            "ETH-15M": 6,
            "ETH-5M": 16,
        },
    }
    readiness = status["collection_readiness"]
    assert isinstance(readiness, dict)
    readiness["features_15m_v1"] = {
        "limiting_family": "BTC-15M",
        "min_family_span_days": 1.25,
        "remaining_target_days": 5.75,
        "target_progress_pct": 17.85,
        "families": {
            family: {
                "rows": rows,
                "span_days": 1.25,
                "remaining_target_ms": 496_800_000,
                "estimated_ready_at": "2026-05-30T12:48:00Z",
                "meets_target": False,
            }
            for family, rows in status["family_counts"]["features_15m_v1"].items()
        },
    }
    readiness["labels_15m_v1"] = {
        "limiting_family": "BTC-15M",
        "min_family_span_days": 1.20,
        "remaining_target_days": 5.80,
        "target_progress_pct": 17.14,
        "families": {
            family: {
                "rows": rows,
                "span_days": 1.20,
                "remaining_target_ms": 501_120_000,
                "estimated_ready_at": "2026-05-30T12:48:00Z",
                "meets_target": False,
            }
            for family, rows in status["family_counts"]["labels_15m_v1"].items()
        },
    }
    readiness["quarantine_clean_window"] = {
        "quarantined_count": 1,
        "meets_target": False,
        "estimated_ready_at": "2026-05-31T03:00:00Z",
        "latest_quarantined_segment": {
            "path": "raw_invalid/ws_market/2026-05-24T030000Z.ndjson.gz",
            "gzip_probe": {"gzip_valid": False, "readable_prefix_lines": 33900},
        },
    }
    disk = status["disk_headroom_evidence"]
    assert isinstance(disk, dict)
    disk["headroom_ok"] = True
    disk["free_bytes"] = 5_000
    disk["required_free_bytes"] = 1_000
    disk["headroom_margin_bytes"] = 4_000
    disk["low_margin_threshold_bytes"] = 100
    status_path = _write_json(tmp_path / "status.json", status)
    promotion = _promotion_audit_payload(passed=False)
    stage0 = promotion["stages"][0]
    assert isinstance(stage0, dict)
    stage0["checks"] = [
        {"name": "promotion_process_source", "passed": True},
        {"name": "clean_atomic_live_root", "passed": True},
        {"name": "status_artifact_fresh", "passed": True},
        {"name": "ready_for_training", "passed": False},
        {"name": "raw_segment_quarantine", "passed": False},
    ]
    promotion_path = _write_json(tmp_path / "promotion.json", promotion)
    objective_path = tmp_path / "objective.json"
    issue_coverage_path = tmp_path / "issue_coverage.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    class _StatVfs:
        f_blocks = 2_000
        f_bavail = 900
        f_frsize = 1

    monkeypatch.setattr(ingestion_main.os, "statvfs", lambda _: _StatVfs())

    xgboost_v4_objective_audit(
        output_path=objective_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )
    xgboost_v4_issue_coverage_audit(
        output_path=issue_coverage_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        objective_audit_path=objective_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(issue_coverage_path.read_text(encoding="utf-8"))
    assert payload["source_artifacts"]["live_status_path"] == str(status_path)
    assert payload["summary"]["decision"] == "BLOCKED"
    assert payload["summary"]["ready_for_training"] is False
    assert payload["summary"]["current_disk_headroom"]["headroom_ok"] is False
    assert payload["summary"]["current_disk_headroom"]["free_bytes"] == 900
    assert payload["summary"]["current_disk_headroom"]["required_free_bytes"] == 1_000
    assert payload["summary"]["feature_families_seen"] == [
        "BTC-15M",
        "BTC-5M",
        "ETH-15M",
        "ETH-5M",
    ]
    assert payload["summary"]["label_families_seen"] == [
        "BTC-15M",
        "BTC-5M",
        "ETH-15M",
        "ETH-5M",
    ]
    assert payload["summary"]["feature_min_span_days"] == 1.25
    assert payload["summary"]["label_min_span_days"] == 1.20
    assert payload["issue_checks"]["#54"]["passed"] is True
    assert payload["issue_checks"]["#55"]["passed"] is False
    assert payload["promotion"]["earliest_failed_stage"] == "Stage 0: 7-day Data Readiness"
    assert payload["promotion"]["stage0_failed_checks"] == [
        "ready_for_training",
        "raw_segment_quarantine",
    ]
    assert payload["promotion"]["stage1_required_checks"]["rerun_report_exists"]["passed"] is False
    assert payload["promotion"]["stage1_required_checks"]["same_dataset_split"]["passed"] is False
    assert (
        payload["promotion"]["stage1_required_checks"]["dataset_time_split_5_1_1"]["passed"]
        is False
    )


def test_xgboost_v4_objective_audit_retries_transient_partial_promotion_json(
    tmp_path: Path,
) -> None:
    import threading
    import time

    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=False))
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text("{", encoding="utf-8")
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    def finish_write() -> None:
        time.sleep(0.1)
        promotion_path.write_text(
            json.dumps(_promotion_audit_payload(passed=False), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    writer = threading.Thread(target=finish_write)
    writer.start()
    try:
        xgboost_v4_objective_audit(
            output_path=output_path,
            live_status_path=status_path,
            promotion_audit_path=promotion_path,
            slack_automation_path=slack_automation_path,
            no_fail_on_blocked=True,
        )
    finally:
        writer.join()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["objective_complete"] is False
    assert payload["promotion"]["decision"] == "BLOCKED"


def test_xgboost_v4_objective_audit_requires_monitoring_rows_for_each_family(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    monitoring = status["monitoring_outcome_evidence"]
    assert isinstance(monitoring, dict)
    families = monitoring["families"]
    assert isinstance(families, dict)
    families["ETH-5M"]["outcome_rows"] = 0
    status_path = _write_json(tmp_path / "status.json", status)
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "model_version": "xgboost-v4",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, candidate_eval_dir=candidate_eval_dir),
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json"
    )
    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#54"]["passed"] is False
    assert checks["#54"]["evidence"]["missing_outcome_row_families"] == ["ETH-5M"]
    assert "#54" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_monitoring_for_xgboost_v4(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    monitoring = status["monitoring_outcome_evidence"]
    assert isinstance(monitoring, dict)
    monitoring["model_version"] = "xgboost-v3"
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#54"]["passed"] is False
    assert checks["#54"]["evidence"]["model_version"] == "xgboost-v3"
    assert checks["#54"]["evidence"]["expected_model_version"] == "xgboost-v4"
    assert "#54" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_reports_issue_54_live_monitoring_contract(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=False),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    drift_evidence = checks["#54"]["evidence"]["drift_monitoring"]
    assert checks["#54"]["passed"] is True
    assert drift_evidence["threshold_configured"] is True
    assert all(drift_evidence["runtime_hooks"].values())
    assert all(drift_evidence["incident_types"].values())
    assert all(model["registered"] for model in drift_evidence["baseline_models"].values())
    assert all(
        model["provenance_registered"]
        for model in drift_evidence["baseline_models"].values()
    )
    assert drift_evidence["baseline_models"]["xgboost-v4"]["split"] == "val"
    assert drift_evidence["baseline_models"]["xgboost-v4"]["source_exists"] is True
    assert (
        drift_evidence["baseline_models"]["xgboost-v4"]["edge_trigger_rate_at_0_30"]
        is not None
    )
    assert all(drift_evidence["runbook_alert_conditions"].values())


def test_issue_54_live_monitoring_evidence_rejects_undocumented_alert_thresholds(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import _issue_54_live_monitoring_evidence

    runbook_path = tmp_path / "champion_promotion.md"
    runbook_path.write_text(
        "Post-cutover verification:\n\nNo drift thresholds here.\n\n## GitHub Closure\n",
        encoding="utf-8",
    )

    evidence = _issue_54_live_monitoring_evidence(runbook_path)
    assert evidence["passed"] is False
    assert evidence["runbook_alert_conditions"]["post_cutover_section_present"] is True
    assert evidence["runbook_alert_conditions"]["mean_shift_threshold"] is False
    assert evidence["runbook_alert_conditions"]["label_hit_rate_threshold"] is False


def test_issue_54_live_monitoring_evidence_requires_baseline_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bigan.ingestion.__main__ as cli

    stale_v4_baseline = dict(cli.CHAMPION_BASELINE_DISTRIBUTIONS["xgboost-v4"])
    stale_v4_baseline["source"] = "missing/offline_reference.json"
    stale_v4_baseline["split"] = "test"
    stale_v4_baseline["edge_distribution"] = {
        "count": 2602,
        "mean": 0.05,
        "std": 0.43,
    }
    monkeypatch.setitem(
        cli.CHAMPION_BASELINE_DISTRIBUTIONS,
        "xgboost-v4",
        stale_v4_baseline,
    )

    evidence = cli._issue_54_live_monitoring_evidence()

    assert evidence["passed"] is False
    assert evidence["baseline_models"]["xgboost-v4"]["registered"] is True
    assert evidence["baseline_models"]["xgboost-v4"]["provenance_registered"] is False
    assert evidence["baseline_models"]["xgboost-v4"]["source_exists"] is False
    assert evidence["baseline_models"]["xgboost-v4"]["split"] == "test"
    assert evidence["baseline_models"]["xgboost-v4"]["edge_trigger_rate_at_0_30"] is None


def test_xgboost_v4_objective_audit_requires_raw_and_manifest_progress(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    status["raw_segment_count"] = 0
    status["processed_manifest_rows"] = 0
    status_path = _write_json(tmp_path / "status.json", status)
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, candidate_eval_dir=candidate_eval_dir),
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json"
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#65"]["passed"] is False
    assert checks["#65"]["evidence"]["raw_segment_count"] == 0
    assert checks["#65"]["evidence"]["processed_manifest_rows"] == 0
    assert "#65" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_blocks_malformed_status_counts(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    monitoring = status["monitoring_outcome_evidence"]
    assert isinstance(monitoring, dict)
    monitoring["event_rows"] = "many"
    families = monitoring["families"]
    assert isinstance(families, dict)
    families["ETH-5M"]["outcome_rows"] = "settled"
    raw_integrity = status["raw_segment_integrity"]
    assert isinstance(raw_integrity, dict)
    raw_integrity["invalid_count"] = "none"
    raw_quarantine = status["raw_segment_quarantine"]
    assert isinstance(raw_quarantine, dict)
    raw_quarantine["quarantined_count"] = "some"
    disk = status["disk_headroom_evidence"]
    assert isinstance(disk, dict)
    disk["headroom_ok"] = "true"
    health = status["health_evidence"]
    assert isinstance(health, dict)
    health["unrecovered_error_match_count"] = "none"
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert checks["#54"]["passed"] is False
    assert checks["#54"]["evidence"]["event_rows"] is None
    assert checks["#54"]["evidence"]["missing_outcome_row_families"] == ["ETH-5M"]
    assert checks["#65"]["passed"] is False
    assert checks["#65"]["evidence"]["invalid_raw_segments"] is None
    assert checks["#65"]["evidence"]["quarantined_raw_segments"] is None
    assert checks["#65"]["evidence"]["disk_headroom_ok"] is False
    assert checks["#65"]["evidence"]["unrecovered_error_match_count"] is None


def test_xgboost_v4_objective_audit_blocks_quarantined_raw_segments(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    raw_quarantine = status["raw_segment_quarantine"]
    assert isinstance(raw_quarantine, dict)
    raw_quarantine["quarantined_count"] = 1
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
    raw_quarantine["latest_quarantined_segment"] = latest_quarantined
    readiness = status["collection_readiness"]
    assert isinstance(readiness, dict)
    readiness["quarantine_clean_window"] = {
        "meets_target": False,
        "estimated_ready_at": "2026-05-31T03:00:00Z",
        "latest_quarantined_segment": latest_quarantined,
    }
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert checks["#65"]["passed"] is False
    evidence = checks["#65"]["evidence"]
    assert evidence["quarantined_raw_segments"] == 1
    assert (
        evidence["latest_quarantined_segment_path"]
        == "raw_invalid/ws_market/2026-05-24T030000Z.ndjson.gz"
    )
    assert evidence["latest_quarantined_gzip_valid"] is False
    assert evidence["latest_quarantined_readable_prefix_lines"] == 33900
    assert evidence["latest_quarantined_readable_prefix_bytes"] == 34454412
    assert "invalid block type" in evidence["latest_quarantined_gzip_error"]


def test_xgboost_v4_objective_audit_blocks_low_disk_headroom(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    disk = status["disk_headroom_evidence"]
    assert isinstance(disk, dict)
    disk["headroom_ok"] = False
    disk["free_bytes"] = 1
    disk["required_free_bytes"] = 2
    disk["projected_remaining_bytes"] = 2
    disk["headroom_margin_bytes"] = -1
    disk["headroom_margin_pct"] = -50.0
    disk["headroom_low_margin"] = False
    disk["low_margin_threshold_bytes"] = 1
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert checks["#65"]["passed"] is False
    assert checks["#65"]["evidence"]["disk_headroom_ok"] is False
    assert checks["#65"]["evidence"]["disk_free_bytes"] == 1
    assert checks["#65"]["evidence"]["disk_required_free_bytes"] == 2
    assert checks["#65"]["evidence"]["disk_headroom_margin_bytes"] == -1
    assert checks["#65"]["evidence"]["disk_headroom_margin_pct"] == -50.0
    assert checks["#65"]["evidence"]["disk_headroom_low_margin"] is False


def test_xgboost_v4_objective_audit_blocks_current_filesystem_disk_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bigan.ingestion import __main__ as ingestion_main
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    live_root = tmp_path / "live"
    live_root.mkdir()
    status = _live_status_payload(ready=True)
    status["live_root"] = str(live_root)
    status["warehouse"] = str(live_root / "warehouse")
    disk = status["disk_headroom_evidence"]
    assert isinstance(disk, dict)
    disk["headroom_ok"] = True
    disk["free_bytes"] = 5_000
    disk["required_free_bytes"] = 1_000
    disk["projected_remaining_bytes"] = 1_000
    disk["headroom_margin_bytes"] = 4_000
    disk["low_margin_threshold_bytes"] = 100
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    class _StatVfs:
        f_blocks = 2_000
        f_bavail = 900
        f_frsize = 1

    monkeypatch.setattr(ingestion_main.os, "statvfs", lambda _: _StatVfs())

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert checks["#65"]["passed"] is False
    evidence = checks["#65"]["evidence"]
    assert evidence["disk_headroom_ok"] is True
    assert evidence["current_disk_headroom_ok"] is False
    assert evidence["current_disk_headroom"]["free_bytes"] == 900
    assert evidence["current_disk_headroom"]["required_free_bytes"] == 1_000
    assert evidence["current_disk_headroom"]["headroom_margin_bytes"] == -100
    assert payload["live_collection"]["current_disk_headroom"]["headroom_ok"] is False


def test_xgboost_v4_objective_audit_requires_lock_and_raw_manifest_coverage(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    status["live_root_lock_evidence"] = {
        "lock_dir_exists": True,
        "pid_file_exists": True,
        "pid": 987654321,
        "owner_running": False,
        "pid_parse_error": None,
    }
    status["raw_manifest_coverage_evidence"] = {
        "stale_missing_processed_count": 3,
        "extra_processed_count": 1,
    }
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    evidence = checks["#65"]["evidence"]
    assert checks["#65"]["passed"] is False
    assert evidence["live_root_lock_ok"] is False
    assert evidence["live_root_lock_pid"] == 987654321
    assert evidence["live_root_lock_owner_running"] is False
    assert evidence["raw_manifest_coverage_ok"] is False
    assert evidence["stale_missing_processed_count"] == 3
    assert evidence["extra_processed_count"] == 1


def test_xgboost_v4_objective_audit_rejects_string_boolean_gate_evidence(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    monitoring = status["monitoring_outcome_evidence"]
    assert isinstance(monitoring, dict)
    monitoring["available"] = "true"
    readiness = status["collection_readiness"]
    assert isinstance(readiness, dict)
    readiness["ready_for_training"] = "true"
    liveness = status["liveness_evidence"]
    assert isinstance(liveness, dict)
    liveness["raw_segments_fresh"] = "true"
    liveness["processed_manifest_fresh"] = "true"

    promotion = _promotion_audit_payload(passed=True)
    promotion["passed"] = "true"
    stages = promotion["stages"]
    assert isinstance(stages, list)
    for stage in stages:
        assert isinstance(stage, dict)
        stage["passed"] = "true"
        checks = stage.get("checks")
        if isinstance(checks, list):
            for check in checks:
                assert isinstance(check, dict)
                check["passed"] = "true"

    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(tmp_path / "promotion.json", promotion)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#54"]["passed"] is False
    assert checks["#55"]["evidence"]["ready_for_training"] is False
    assert checks["#56"]["evidence"]["ready_for_training"] is False
    assert checks["#65"]["passed"] is False
    assert payload["live_collection"]["ready_for_training"] is False
    assert payload["promotion"]["passed"] is False
    assert payload["promotion"]["raw_passed"] is False
    assert payload["promotion"]["clean_atomic_live_root_passed"] is False
    assert payload["promotion"]["status_artifact_fresh_passed"] is False
    assert payload["promotion"]["earliest_failed_stage"] == "Stage 0: 7-day Data Readiness"


def test_xgboost_v4_objective_audit_rejects_debug_live_root_for_final_corpus(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    status["screen_session"] = "xgbv4_7d_segmented_20260523T000000Z"
    status["live_root"] = "data/live/xgboost-v4-multimarket-7d-segmented-debug"
    status["warehouse"] = "data/live/xgboost-v4-multimarket-7d-segmented-debug/warehouse"
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    evidence = checks["#65"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#65"]["passed"] is False
    assert evidence["screen_state"] == "running"
    assert evidence["screen_session_matches"] is False
    assert evidence["live_root_matches"] is False
    assert evidence["warehouse_matches"] is False
    assert "#65" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_stale_promotion_audit_without_clean_root_check(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = _write_tick_ready_candidate_model_dir(tmp_path)
    promotion = _promotion_audit_payload(passed=True)
    stage0 = promotion["stages"][0]
    assert isinstance(stage0, dict)
    stage0.pop("checks")
    promotion_path = _write_json(tmp_path / "promotion.json", promotion)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    issue_65_evidence = checks["#65"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#65"]["passed"] is False
    assert issue_65_evidence["tick_features"]["passed"] is True
    assert issue_65_evidence["promotion_clean_atomic_live_root_passed"] is False
    assert payload["promotion"]["raw_passed"] is True
    assert payload["promotion"]["clean_atomic_live_root_passed"] is False
    assert payload["promotion"]["passed"] is False
    assert "champion-promotion.md gates have not all passed" in payload["blockers"]


def test_xgboost_v4_objective_audit_rejects_stale_promotion_audit_without_status_fresh_check(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = _write_tick_ready_candidate_model_dir(tmp_path)
    promotion = _promotion_audit_payload(passed=True)
    stage0 = promotion["stages"][0]
    assert isinstance(stage0, dict)
    checks = stage0["checks"]
    assert isinstance(checks, list)
    stage0["checks"] = [
        check
        for check in checks
        if not (isinstance(check, dict) and check.get("name") == "status_artifact_fresh")
    ]
    promotion_path = _write_json(tmp_path / "promotion.json", promotion)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    issue_65_evidence = checks["#65"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#65"]["passed"] is False
    assert issue_65_evidence["tick_features"]["passed"] is True
    assert issue_65_evidence["promotion_status_artifact_fresh_passed"] is False
    assert payload["promotion"]["raw_passed"] is True
    assert payload["promotion"]["clean_atomic_live_root_passed"] is True
    assert payload["promotion"]["status_artifact_fresh_passed"] is False
    assert payload["promotion"]["passed"] is False
    assert "champion-promotion.md gates have not all passed" in payload["blockers"]


def test_xgboost_v4_objective_audit_rejects_promotion_without_attachment_source(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = _write_tick_ready_candidate_model_dir(tmp_path)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, include_process_source=False),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    source = payload["promotion"]["promotion_process_source"]
    assert payload["objective_complete"] is False
    assert payload["promotion"]["raw_passed"] is True
    assert payload["promotion"]["clean_atomic_live_root_passed"] is True
    assert payload["promotion"]["status_artifact_fresh_passed"] is True
    assert payload["promotion"]["collector_process_liveness_passed"] is True
    assert payload["promotion"]["raw_manifest_coverage_passed"] is True
    assert payload["promotion"]["github_issue_closures_passed"] is True
    assert payload["promotion"]["promotion_process_source_passed"] is False
    assert payload["promotion"]["passed"] is False
    assert source["expected_source_path"] == "/Users/tcscoder/Downloads/champion-promotion.md"
    assert source["stage_check_passed"] is False
    assert source["source_path_matches"] is False
    assert checklist["champion_promotion_md"]["passed"] is False
    assert "champion-promotion.md gates have not all passed" in payload["blockers"]


@pytest.mark.parametrize(
    "missing_stage",
    [
        "Stage 3: Shadow Evaluation",
        "Stage 4: Bootstrap Decision",
    ],
)
def test_xgboost_v4_objective_audit_rejects_promotion_without_required_stage(
    tmp_path: Path,
    missing_stage: str,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = _write_tick_ready_candidate_model_dir(tmp_path)
    promotion = _promotion_audit_payload(passed=True)
    stages = promotion["stages"]
    assert isinstance(stages, list)
    promotion["stages"] = [
        stage
        for stage in stages
        if not (isinstance(stage, dict) and stage.get("name") == missing_stage)
    ]
    promotion_path = _write_json(tmp_path / "promotion.json", promotion)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    assert payload["objective_complete"] is False
    assert payload["promotion"]["raw_passed"] is True
    assert payload["promotion"]["promotion_process_source_passed"] is True
    assert payload["promotion"]["github_issue_closures_passed"] is True
    assert payload["promotion"]["stage_passes"][missing_stage] is False
    assert payload["promotion"]["all_required_stages_passed"] is False
    assert payload["promotion"]["passed"] is False
    assert checklist["champion_promotion_md"]["passed"] is False
    assert "champion-promotion.md gates have not all passed" in payload["blockers"]


@pytest.mark.parametrize(
    ("missing_check_name", "promotion_field"),
    [
        ("collector_process_liveness", "collector_process_liveness_passed"),
        ("raw_manifest_coverage", "raw_manifest_coverage_passed"),
    ],
)
def test_xgboost_v4_objective_audit_rejects_stale_promotion_audit_without_new_stage0_checks(
    tmp_path: Path,
    missing_check_name: str,
    promotion_field: str,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = _write_tick_ready_candidate_model_dir(tmp_path)
    promotion = _promotion_audit_payload(passed=True)
    stage0 = promotion["stages"][0]
    assert isinstance(stage0, dict)
    checks = stage0["checks"]
    assert isinstance(checks, list)
    stage0["checks"] = [
        check
        for check in checks
        if not (isinstance(check, dict) and check.get("name") == missing_check_name)
    ]
    promotion_path = _write_json(tmp_path / "promotion.json", promotion)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    issue_65_evidence = checks["#65"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#65"]["passed"] is False
    assert issue_65_evidence["tick_features"]["passed"] is True
    assert issue_65_evidence[f"promotion_{promotion_field}"] is False
    assert payload["promotion"]["raw_passed"] is True
    assert payload["promotion"][promotion_field] is False
    assert payload["promotion"]["passed"] is False
    assert "champion-promotion.md gates have not all passed" in payload["blockers"]


def test_xgboost_v4_objective_audit_rejects_stale_promotion_audit_without_github_closure_check(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion = _promotion_audit_payload(passed=True)
    stage5 = promotion["stages"][5]
    assert isinstance(stage5, dict)
    stage5.pop("checks")
    promotion_path = _write_json(tmp_path / "promotion.json", promotion)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["objective_complete"] is False
    assert payload["promotion"]["raw_passed"] is True
    assert payload["promotion"]["clean_atomic_live_root_passed"] is True
    assert payload["promotion"]["status_artifact_fresh_passed"] is True
    assert payload["promotion"]["github_issue_closures_passed"] is False
    assert payload["promotion"]["passed"] is False
    assert "champion-promotion.md gates have not all passed" in payload["blockers"]


def test_xgboost_v4_objective_audit_requires_all_requested_readiness_families(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    readiness = status["collection_readiness"]
    assert isinstance(readiness, dict)
    readiness["required_families"] = ["BTC-15M"]
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert checks["#55"]["passed"] is False
    assert checks["#56"]["passed"] is False
    assert checks["#55"]["evidence"]["status_required_families"] == ["BTC-15M"]
    assert checks["#55"]["evidence"]["missing_readiness_required_families"] == [
        "ETH-15M",
        "BTC-5M",
        "ETH-5M",
    ]
    assert checks["#56"]["evidence"]["missing_readiness_required_families"] == [
        "ETH-15M",
        "BTC-5M",
        "ETH-5M",
    ]


def test_xgboost_v4_objective_audit_requires_fresh_warehouse_families(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    warehouse = status["warehouse_freshness_evidence"]
    assert isinstance(warehouse, dict)
    tables = warehouse["tables"]
    assert isinstance(tables, dict)
    features = tables["features_15m_v1"]
    predictions = tables["predictions"]
    assert isinstance(features, dict)
    assert isinstance(predictions, dict)
    features["stale_families"] = ["ETH-15M"]
    prediction_families = predictions["families"]
    assert isinstance(prediction_families, dict)
    prediction_families["BTC-5M"]["fresh"] = False
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#56"]["passed"] is False
    assert checks["#56"]["evidence"]["missing_feature_families"] == []
    assert checks["#56"]["evidence"]["missing_prediction_families"] == []
    assert checks["#56"]["evidence"]["unfresh_feature_families"] == ["ETH-15M"]
    assert checks["#56"]["evidence"]["unfresh_prediction_families"] == ["BTC-5M"]
    assert "#56" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_fresh_label_families(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status = _live_status_payload(ready=True)
    label_freshness = status["label_freshness_evidence"]
    assert isinstance(label_freshness, dict)
    label_freshness["fresh"] = False
    label_freshness["stale_families"] = ["ETH-15M"]
    families = label_freshness["families"]
    assert isinstance(families, dict)
    families["BTC-5M"]["fresh"] = False
    status_path = _write_json(tmp_path / "status.json", status)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#55"]["passed"] is False
    assert checks["#56"]["passed"] is False
    assert checks["#56"]["evidence"]["label_freshness_fresh"] is False
    assert checks["#56"]["evidence"]["missing_label_freshness_families"] == []
    assert checks["#56"]["evidence"]["unfresh_label_families"] == ["BTC-5M", "ETH-15M"]
    assert "#56" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_dataset_stability_report(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#55"]["passed"] is False
    assert checks["#55"]["evidence"]["dataset_stability"]["exists"] is False
    assert "#55" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_dataset_stability_same_dataset(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = _write_tick_ready_candidate_model_dir(tmp_path)
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, candidate_eval_dir=candidate_eval_dir),
    )
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json",
        dataset_dir="runs/other-training",
        dataset_version="other-dataset-v1",
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        stability_report_path=stability_report_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    stability = checks["#55"]["evidence"]["dataset_stability"]
    assert payload["objective_complete"] is False
    assert checks["#55"]["passed"] is False
    assert stability["dataset_dir"] == "runs/other-training"
    assert stability["expected_dataset_dir"] == "runs/training"
    assert stability["dataset_dir_matches"] is False
    assert stability["dataset_version"] == "other-dataset-v1"
    assert stability["expected_dataset_version"] == "dataset-v1"
    assert stability["dataset_version_matches"] is False
    assert "#55" in " ".join(payload["blockers"])


def test_issue_56_multimarket_evidence_requires_market_and_liquidity_features(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import _issue_56_multimarket_schema_evidence

    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json"
    )
    model_wrapper = {
        "feature_columns": [
            feature
            for feature in TEST_MODEL_FEATURE_COLUMNS
            if feature != "liquidity_bucket"
        ]
    }

    evidence = _issue_56_multimarket_schema_evidence(
        model_wrapper=model_wrapper,
        stability_report_path=stability_report_path,
        required_families=["BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M"],
    )

    assert evidence["passed"] is False
    assert evidence["missing_structure_features"] == ["liquidity_bucket"]
    assert evidence["model_schema_matches_dataset"] is False


def test_issue_56_multimarket_evidence_rejects_too_wide_feature_schema(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import _issue_56_multimarket_schema_evidence

    wide_feature_columns = [
        *TEST_MODEL_FEATURE_COLUMNS,
        *[f"extra_feature_{idx}" for idx in range(12)],
    ]
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json",
        feature_columns=wide_feature_columns,
    )
    model_wrapper = {"feature_columns": wide_feature_columns}

    evidence = _issue_56_multimarket_schema_evidence(
        model_wrapper=model_wrapper,
        stability_report_path=stability_report_path,
        required_families=["BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M"],
    )

    assert evidence["passed"] is False
    assert evidence["model_schema_matches_dataset"] is True
    assert evidence["missing_structure_features"] == []
    assert evidence["feature_count"] == 31
    assert evidence["feature_count_limit"] == 30
    assert evidence["issue_suggested_feature_count_limit"] == 30
    assert evidence["feature_count_within_limit"] is False


def test_issue_65_tick_evidence_requires_candidate_tick_features() -> None:
    from bigan.ingestion.__main__ import _issue_65_tick_feature_evidence

    model_wrapper = {
        "feature_columns": [
            feature
            for feature in TEST_MODEL_FEATURE_COLUMNS
            if feature != "tick_price_velocity"
        ]
    }

    evidence = _issue_65_tick_feature_evidence(model_wrapper=model_wrapper)

    assert evidence["passed"] is False
    assert evidence["missing_model_tick_features"] == ["tick_price_velocity"]
    assert evidence["doc_checks"]["documents_tick_features"] is True
    assert evidence["schema_checks"]["aggregation_uses_5_second_windows"] is True
    assert evidence["runner_checks"]["runner_defaults_to_5_second_cycle"] is True
    assert evidence["runner_checks"]["runner_configures_live_min_free_bytes"] is True
    assert evidence["runner_checks"]["runner_checks_live_min_free_space"] is True


def test_issue_65_tick_evidence_requires_live_runner_5_second_cadence(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import _issue_65_tick_feature_evidence

    live_runner_path = tmp_path / "run_champion_live.sh"
    live_runner_path.write_text(
        "\n".join(
            (
                'CYCLE_SLEEP_SECONDS="${CYCLE_SLEEP_SECONDS:-60}"',
                'sleep_for "${CYCLE_SLEEP_SECONDS}"',
                'echo "[champion-live] cycle sleep seconds=${CYCLE_SLEEP_SECONDS}"',
            )
        ),
        encoding="utf-8",
    )
    model_wrapper = {"feature_columns": TEST_MODEL_FEATURE_COLUMNS}

    evidence = _issue_65_tick_feature_evidence(
        model_wrapper=model_wrapper,
        live_runner_path=live_runner_path,
    )

    assert evidence["passed"] is False
    assert evidence["runner_checks"]["runner_defaults_to_5_second_cycle"] is False
    assert evidence["runner_checks"]["runner_uses_cycle_sleep"] is True
    assert evidence["runner_checks"]["runner_logs_cycle_sleep"] is True
    assert evidence["runner_checks"]["runner_checks_live_min_free_space"] is False


def test_issue_65_tick_evidence_requires_live_runner_duplicate_guards(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import _issue_65_tick_feature_evidence

    live_runner_path = tmp_path / "run_champion_live.sh"
    live_runner_path.write_text(
        "\n".join(
            (
                'CYCLE_SLEEP_SECONDS="${CYCLE_SLEEP_SECONDS:-5}"',
                'sleep_for "${CYCLE_SLEEP_SECONDS}"',
                'echo "[champion-live] cycle sleep seconds=${CYCLE_SLEEP_SECONDS}"',
                "features-15m-v1 --skip-existing",
                "predictions-v1 --skip-existing-predictions",
                "labels-15m-v1 --skip-existing-labels",
            )
        ),
        encoding="utf-8",
    )
    model_wrapper = {"feature_columns": TEST_MODEL_FEATURE_COLUMNS}

    evidence = _issue_65_tick_feature_evidence(
        model_wrapper=model_wrapper,
        live_runner_path=live_runner_path,
    )

    assert evidence["passed"] is False
    assert evidence["runner_checks"]["runner_uses_skip_existing_features"] is True
    assert evidence["runner_checks"]["runner_uses_skip_existing_predictions"] is True
    assert evidence["runner_checks"]["runner_uses_skip_existing_labels"] is True
    assert evidence["runner_checks"]["runner_uses_skip_existing_monitoring_events"] is False
    assert evidence["runner_checks"]["runner_checks_live_min_free_space"] is False


def test_issue_65_tick_evidence_requires_live_runner_free_space_guard(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import _issue_65_tick_feature_evidence

    live_runner_path = tmp_path / "run_champion_live.sh"
    live_runner_path.write_text(
        "\n".join(
            (
                'CYCLE_SLEEP_SECONDS="${CYCLE_SLEEP_SECONDS:-5}"',
                'sleep_for "${CYCLE_SLEEP_SECONDS}"',
                'echo "[champion-live] cycle sleep seconds=${CYCLE_SLEEP_SECONDS}"',
                "features-15m-v1 --skip-existing",
                "predictions-v1 --skip-existing-monitoring-events --skip-existing-predictions",
                "labels-15m-v1 --skip-existing-labels",
            )
        ),
        encoding="utf-8",
    )
    model_wrapper = {"feature_columns": TEST_MODEL_FEATURE_COLUMNS}

    evidence = _issue_65_tick_feature_evidence(
        model_wrapper=model_wrapper,
        live_runner_path=live_runner_path,
    )

    assert evidence["passed"] is False
    assert evidence["runner_checks"]["runner_configures_live_min_free_bytes"] is False
    assert evidence["runner_checks"]["runner_checks_live_min_free_space"] is False
    assert evidence["runner_checks"]["runner_checks_space_before_capture"] is False
    assert evidence["runner_checks"]["runner_checks_space_each_cycle"] is False


def test_issue_58_cv_evidence_requires_summary_mean_std_metrics() -> None:
    from bigan.ingestion.__main__ import _cv_time_series_evidence

    evidence = _cv_time_series_evidence(
        {
            "summary": {"fold_count": 3},
            "folds": [
                {
                    "fold": fold_idx + 1,
                    "train_start_ts": 1_000,
                    "train_end_ts": 2_000 + fold_idx * 1_000,
                    "val_start_ts": 3_000 + fold_idx * 1_000,
                    "val_end_ts": 3_500 + fold_idx * 1_000,
                    "train_count": 10,
                    "val_count": 5,
                    "metrics": {
                        "sample_count": 5,
                        "brier_score": 0.12,
                        "roc_auc": 0.70,
                        "pnl": 0.20,
                    },
                }
                for fold_idx in range(3)
            ],
        },
        3,
    )

    assert evidence["cv_time_series_ordered"] is True
    assert evidence["cv_fold_metrics_present"] is True
    assert evidence["cv_summary_metrics_present"] is False
    assert evidence["missing_cv_summary_metrics"] == [
        "brier_mean",
        "brier_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pnl_mean",
        "pnl_std",
    ]


def test_issue_58_ensemble_seed_evidence_requires_matching_distinct_seeds() -> None:
    from bigan.ingestion.__main__ import _ensemble_seed_evidence

    evidence = _ensemble_seed_evidence(
        ensemble_summary={"seeds": [0, 0, 42]},
        model_members=[
            {"seed": 0, "path": "model_seed_0.json"},
            {"seed": 17, "path": "model_seed_17.json"},
            {"seed": 42, "path": "model_seed_42.json"},
        ],
        expected_member_count=3,
    )

    assert evidence["ensemble_seed_evidence_passed"] is False
    assert evidence["duplicate_ensemble_summary_seeds"] == [0]
    assert evidence["ensemble_seeds_match_members"] is False


def test_xgboost_v4_objective_audit_requires_issue_57_added_features(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    feature_columns = [
        feature for feature in TEST_MODEL_FEATURE_COLUMNS if feature != "ret_30m"
    ]
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "feature_columns": feature_columns,
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, candidate_eval_dir=candidate_eval_dir),
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json",
        feature_columns=feature_columns,
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#57"]["passed"] is False
    assert checks["#57"]["evidence"]["passed"] is True
    assert checks["#57"]["evidence"]["feature_importance_passed"] is True
    assert checks["#57"]["evidence"]["added_features"]["passed"] is False
    assert checks["#57"]["evidence"]["added_features"]["missing_model_added_features"] == [
        "ret_30m"
    ]
    assert checks["#57"]["evidence"]["added_features"][
        "missing_dataset_added_features"
    ] == ["ret_30m"]
    assert "#57" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_issue_57_dataset_added_features(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    dataset_feature_columns = [
        feature for feature in TEST_MODEL_FEATURE_COLUMNS if feature != "ret_30m"
    ]
    _write_dataset_stability_report(
        candidate_evidence["stability_report_path"],
        feature_columns=dataset_feature_columns,
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    added = checks["#57"]["evidence"]["added_features"]
    assert payload["objective_complete"] is False
    assert checks["#57"]["passed"] is False
    assert added["passed"] is False
    assert added["missing_model_added_features"] == []
    assert added["missing_dataset_added_features"] == ["ret_30m"]
    assert added["dataset_feature_columns_present"] is True
    assert "#57" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_can_pass_with_all_final_evidence(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    live_status = _live_status_payload(ready=True)
    live_status["generated_at"] = attempted_at
    status_path = _write_json(tmp_path / "status.json", live_status)
    collection_risk_path = _write_json(
        tmp_path / "collection_risk_latest.json",
        _collection_risk_payload(status_path=status_path, live_status=live_status),
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, candidate_eval_dir=candidate_eval_dir),
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/C0B5VHYSCN8/p1",
            "ok": True,
            "status": "sent",
        },
    )
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json"
    )
    post_readiness_latest_path = _write_post_readiness_latest(
        tmp_path / "xgboost_v4_post_readiness_latest.json",
        run_root=tmp_path / "xgboost-v4-post-readiness-run",
        run_manifest_path=tmp_path / "xgboost-v4-post-readiness-run" / "run_manifest.json",
        objective_audit_path=output_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        collection_risk_path=collection_risk_path,
        post_readiness_latest_path=post_readiness_latest_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["objective_complete"] is True
    assert payload["decision"] == "COMPLETE"
    assert payload["blockers"] == []
    assert payload["prompt_to_artifact_blockers"] == []
    assert payload["objective_restatement"]["slack_channel_id"] == "C0B5VHYSCN8"
    assert payload["objective_restatement"]["promotion_process_path"] == (
        "/Users/tcscoder/Downloads/champion-promotion.md"
    )
    assert payload["objective_restatement"]["requested_issue_urls"] == [
        "https://github.com/phead198708/BiGan/issues/54",
        "https://github.com/phead198708/BiGan/issues/55",
        "https://github.com/phead198708/BiGan/issues/56",
        "https://github.com/phead198708/BiGan/issues/57",
        "https://github.com/phead198708/BiGan/issues/58",
        "https://github.com/phead198708/BiGan/issues/64",
        "https://github.com/phead198708/BiGan/issues/65",
    ]
    success_criteria = {item["id"]: item for item in payload["objective_success_criteria"]}
    assert set(success_criteria) == {
        "all_requested_github_issues_satisfied",
        "fresh_xgboost_v4_model_created",
        "beats_current_champion",
        "champion_promotion_gates_passed",
        "hourly_slack_status_active",
        "post_readiness_latest_pointer_valid",
    }
    assert all(item["passed"] is True for item in success_criteria.values())
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    assert operational_checks["slack_hourly_status"]["passed"] is True
    slack_delivery_status = operational_checks["slack_hourly_status"]["evidence"][
        "delivery_status"
    ]
    assert slack_delivery_status["passed"] is True
    assert slack_delivery_status["attempted_at_present"] is True
    assert slack_delivery_status["attempted_at_fresh"] is True
    assert slack_delivery_status["attempted_at_parse_error"] is None
    assert slack_delivery_status["message_link_present"] is True
    assert slack_delivery_status["message_link_channel_matches"] is True
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_pointer_summary_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_shadow_continuation_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_shadow_auto_window_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_shadow_full_session_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "slack_delivery_status_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "slack_delivery_status_helper_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "issue_coverage_audit_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "collection_risk_helper_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "collection_risk_helper_json_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "collection_risk_helper_urgency_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "collection_risk_helper_output_path_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "collection_risk_helper_status_freshness_instruction_present"
        ]
        is True
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "skip_label_refresh_on_disk_block_instruction_present"
        ]
        is True
    )
    assert payload["artifact_paths"]["stability_report_path"] == str(stability_report_path)
    assert payload["artifact_paths"]["objective_audit_path"] == str(output_path)
    assert payload["artifact_paths"]["collection_risk_path"] == str(collection_risk_path)
    assert payload["artifact_paths"]["post_readiness_latest_path"] == str(
        post_readiness_latest_path
    )
    collection_risk = payload["live_collection"]["collection_risk"]
    assert collection_risk["exists"] is True
    assert collection_risk["well_formed"] is True
    assert collection_risk["current"] is True
    assert collection_risk["status_path_matches"] is True
    assert collection_risk["status_generated_at_matches"] is True
    assert collection_risk["status_artifact_fresh"] is True
    assert collection_risk["status_artifact_age_within_limit"] is True
    post_readiness = payload["post_readiness_latest"]
    assert post_readiness["exists"] is True
    assert post_readiness["well_formed"] is True
    assert post_readiness["matches_current_inputs"] is True
    assert post_readiness["objective_complete_clean"] is True
    assert post_readiness["objective_status_compatible"] is True
    assert post_readiness["run_manifest_phase"] == "completed"
    assert post_readiness["live_status_summary_present"] is True
    assert post_readiness["artifact_paths_present"] is True
    assert post_readiness["path_matches"]["promotion_audit_path"] is True
    assert post_readiness["path_matches"]["objective_audit_path"] is True
    assert post_readiness["path_matches"]["candidate_model_path"] is True
    assert post_readiness["path_matches"]["feature_ablation_path"] is True
    assert post_readiness["path_matches"]["dataset_stability_report_path"] is True
    assert post_readiness["path_matches"]["down_validation_path"] is True
    assert post_readiness["path_matches"]["issue_coverage_audit_path"] is True
    assert post_readiness["issue_coverage_audit_path"].endswith(
        "/artifacts/issue_coverage_audit.json"
    )
    assert post_readiness["issue_coverage_issue_checks"]["#54"]["passed"] is True
    assert post_readiness["issue_coverage_objective_success_criteria"][
        "all_requested_github_issues_satisfied"
    ]["passed"] is True
    assert post_readiness["issue_coverage_required_issues_passed"] is True
    assert post_readiness["issue_coverage_success_criteria_passed"] is True
    assert post_readiness["issue_coverage_summary_compatible"] is True
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    assert set(checklist) == {
        "github_issue_54",
        "github_issue_55",
        "github_issue_56",
        "github_issue_57",
        "github_issue_58",
        "github_issue_64",
        "github_issue_65",
        "create_xgboost_v4_model",
        "beat_current_champion",
        "champion_promotion_md",
        "hourly_slack_status",
        "post_readiness_latest_pointer",
    }
    assert all(item["passed"] is True for item in checklist.values())
    assert checklist["github_issue_54"]["source"].endswith("/issues/54")
    assert checklist["champion_promotion_md"]["source"] == (
        "/Users/tcscoder/Downloads/champion-promotion.md"
    )
    assert "/Users/tcscoder/Downloads/champion-promotion.md" in checklist[
        "champion_promotion_md"
    ]["evidence_paths"]
    assert checklist["hourly_slack_status"]["source"] == "C0B5VHYSCN8"
    assert checklist["post_readiness_latest_pointer"]["source"] == str(
        post_readiness_latest_path
    )
    assert checklist["github_issue_54"]["evidence_scope"] == "current_live_monitoring"
    for item_id in ("github_issue_55", "github_issue_56", "github_issue_65"):
        assert checklist[item_id]["evidence_scope"] == "final_7d_corpus"
    for item_id in (
        "github_issue_57",
        "github_issue_58",
        "github_issue_64",
        "create_xgboost_v4_model",
    ):
        assert checklist[item_id]["evidence_scope"] == "final"
    assert checklist["beat_current_champion"]["evidence_scope"] == "passed"
    assert checklist["champion_promotion_md"]["evidence_scope"] == "passed"
    assert checklist["hourly_slack_status"]["evidence_scope"] == "active"
    assert checklist["post_readiness_latest_pointer"]["evidence_scope"] == "matched"
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert checks["#57"]["evidence"]["evidence_scope"] == "final"
    assert checks["#57"]["evidence"]["final_candidate_evidence_ready"] is True
    assert checks["#57"]["evidence"]["final_candidate_evidence_requirements"][
        "candidate_eval_model_path_matches"
    ] is True
    assert checks["#58"]["evidence"]["evidence_scope"] == "final"
    assert checks["#58"]["evidence"]["final_candidate_evidence_ready"] is True
    assert checks["#58"]["evidence"]["candidate_eval_model_provenance"][
        "model_path_matches"
    ] is True
    assert checks["#64"]["evidence"]["evidence_scope"] == "final"
    assert checks["#64"]["evidence"]["final_candidate_evidence_ready"] is True
    assert checks["#64"]["evidence"]["trade_sample"]["passed"] is True
    assert checks["#65"]["evidence"]["evidence_scope"] == "final"
    assert checks["#65"]["evidence"]["final_candidate_evidence_ready"] is True
    assert checks["#65"]["evidence"]["tick_features"]["passed"] is True
    assert "promotion.github_issue_closures_passed" in checklist[
        "champion_promotion_md"
    ]["evidence_keys"]
    assert "promotion.Stage 5.github_issue_closures_recorded" in checklist[
        "champion_promotion_md"
    ]["evidence_keys"]
    assert "final_candidate_evidence_requirements" in checklist[
        "github_issue_65"
    ]["evidence_keys"]
    assert "candidate_eval_model_provenance" in checklist[
        "github_issue_65"
    ]["evidence_keys"]


def test_xgboost_v4_objective_audit_reports_post_readiness_pointer_mismatches(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=False))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=False),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    output_path = tmp_path / "objective.json"
    post_readiness_latest_path = _write_post_readiness_latest(
        tmp_path / "xgboost_v4_post_readiness_latest.json",
        run_root=tmp_path / "xgboost-v4-post-readiness-run",
        run_manifest_path=tmp_path / "xgboost-v4-post-readiness-run" / "run_manifest.json",
        objective_audit_path=output_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=tmp_path / "stale-feature-ablation.json",
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=_write_slack_automation_toml(tmp_path / "automation.toml"),
        post_readiness_latest_path=post_readiness_latest_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    post_readiness = payload["post_readiness_latest"]
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    assert payload["objective_complete"] is False
    assert post_readiness["well_formed"] is True
    assert post_readiness["matches_current_inputs"] is False
    assert post_readiness["objective_status_compatible"] is True
    assert post_readiness["path_matches"]["candidate_model_path"] is True
    assert post_readiness["path_matches"]["feature_ablation_path"] is False
    assert checklist["post_readiness_latest_pointer"]["passed"] is False
    assert any(
        blocker.startswith("post_readiness_latest_pointer:")
        for blocker in payload["prompt_to_artifact_blockers"]
    )


def test_xgboost_v4_objective_audit_rejects_post_readiness_pointer_without_issue_coverage(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=False))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=False),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    output_path = tmp_path / "objective.json"
    post_readiness_latest_path = _write_post_readiness_latest(
        tmp_path / "xgboost_v4_post_readiness_latest.json",
        run_root=tmp_path / "xgboost-v4-post-readiness-run",
        run_manifest_path=tmp_path / "xgboost-v4-post-readiness-run" / "run_manifest.json",
        objective_audit_path=output_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
    )
    pointer_payload = json.loads(post_readiness_latest_path.read_text(encoding="utf-8"))
    pointer_payload["artifact_paths"].pop("issue_coverage_audit_path")
    pointer_payload.pop("issue_coverage_audit_path")
    post_readiness_latest_path.write_text(
        json.dumps(pointer_payload),
        encoding="utf-8",
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=_write_slack_automation_toml(tmp_path / "automation.toml"),
        post_readiness_latest_path=post_readiness_latest_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    post_readiness = payload["post_readiness_latest"]
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    assert payload["objective_complete"] is False
    assert post_readiness["well_formed"] is True
    assert post_readiness["matches_current_inputs"] is False
    assert post_readiness["path_matches"]["issue_coverage_audit_path"] is False
    assert checklist["post_readiness_latest_pointer"]["passed"] is False
    assert any(
        blocker.startswith("post_readiness_latest_pointer:")
        for blocker in payload["prompt_to_artifact_blockers"]
    )


def test_xgboost_v4_objective_audit_rejects_post_readiness_pointer_failed_issue_coverage(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=False))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=False),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    output_path = tmp_path / "objective.json"
    post_readiness_latest_path = _write_post_readiness_latest(
        tmp_path / "xgboost_v4_post_readiness_latest.json",
        run_root=tmp_path / "xgboost-v4-post-readiness-run",
        run_manifest_path=tmp_path / "xgboost-v4-post-readiness-run" / "run_manifest.json",
        objective_audit_path=output_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
    )
    pointer_payload = json.loads(post_readiness_latest_path.read_text(encoding="utf-8"))
    pointer_payload["issue_coverage_issue_checks"]["#65"]["passed"] = False
    pointer_payload["issue_coverage_objective_success_criteria"][
        "all_requested_github_issues_satisfied"
    ]["passed"] = False
    post_readiness_latest_path.write_text(
        json.dumps(pointer_payload),
        encoding="utf-8",
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=_write_slack_automation_toml(tmp_path / "automation.toml"),
        post_readiness_latest_path=post_readiness_latest_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    post_readiness = payload["post_readiness_latest"]
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    assert payload["objective_complete"] is False
    assert post_readiness["path_matches"]["issue_coverage_audit_path"] is True
    assert post_readiness["issue_coverage_required_issues_passed"] is False
    assert post_readiness["issue_coverage_success_criteria_passed"] is False
    assert post_readiness["issue_coverage_summary_compatible"] is False
    assert post_readiness["matches_current_inputs"] is False
    assert checklist["post_readiness_latest_pointer"]["passed"] is False
    assert any(
        blocker.startswith("post_readiness_latest_pointer:")
        for blocker in payload["prompt_to_artifact_blockers"]
    )


def test_xgboost_v4_objective_audit_requires_post_readiness_pointer_for_completion(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/C0B5VHYSCN8/p1",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"
    missing_post_readiness_latest_path = tmp_path / "missing-latest.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=_write_slack_automation_toml(tmp_path / "automation.toml"),
        slack_delivery_status_path=slack_delivery_status_path,
        post_readiness_latest_path=missing_post_readiness_latest_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    success_criteria = {item["id"]: item for item in payload["objective_success_criteria"]}

    assert payload["objective_complete"] is False
    assert payload["decision"] == "BLOCKED"
    assert checklist["post_readiness_latest_pointer"]["passed"] is False
    assert checklist["post_readiness_latest_pointer"]["evidence_scope"] == "blocked"
    assert checklist["post_readiness_latest_pointer"]["source"] == str(
        missing_post_readiness_latest_path
    )
    assert success_criteria["post_readiness_latest_pointer_valid"]["passed"] is False
    for criterion_id, criterion in success_criteria.items():
        if criterion_id != "post_readiness_latest_pointer_valid":
            assert criterion["passed"] is True
    assert any(
        blocker.startswith("post_readiness_latest_pointer:")
        for blocker in payload["prompt_to_artifact_blockers"]
    )


def test_xgboost_v4_objective_audit_rejects_pointer_with_non_pointer_objective_blockers(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/C0B5VHYSCN8/p1",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"
    post_readiness_latest_path = _write_post_readiness_latest(
        tmp_path / "xgboost_v4_post_readiness_latest.json",
        run_root=tmp_path / "xgboost-v4-post-readiness-run",
        run_manifest_path=tmp_path / "xgboost-v4-post-readiness-run" / "run_manifest.json",
        objective_audit_path=output_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
    )
    pointer_payload = json.loads(post_readiness_latest_path.read_text(encoding="utf-8"))
    pointer_payload["objective_decision"] = "BLOCKED"
    pointer_payload["objective_complete"] = False
    pointer_payload["objective_prompt_to_artifact_blockers"] = [
        "create_xgboost_v4_model: stale model artifact"
    ]
    post_readiness_latest_path.write_text(
        json.dumps(pointer_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=_write_slack_automation_toml(tmp_path / "automation.toml"),
        slack_delivery_status_path=slack_delivery_status_path,
        post_readiness_latest_path=post_readiness_latest_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    post_readiness = payload["post_readiness_latest"]
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    success_criteria = {item["id"]: item for item in payload["objective_success_criteria"]}

    assert payload["objective_complete"] is False
    assert post_readiness["matches_current_inputs"] is True
    assert post_readiness["objective_status_compatible"] is False
    assert post_readiness["objective_only_blocks_on_latest_pointer"] is False
    assert post_readiness["objective_prompt_to_artifact_blockers"] == [
        "create_xgboost_v4_model: stale model artifact"
    ]
    assert checklist["post_readiness_latest_pointer"]["passed"] is False
    assert success_criteria["post_readiness_latest_pointer_valid"]["passed"] is False


def test_xgboost_v4_objective_audit_rejects_complete_pointer_with_stale_blockers(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/C0B5VHYSCN8/p1",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"
    post_readiness_latest_path = _write_post_readiness_latest(
        tmp_path / "xgboost_v4_post_readiness_latest.json",
        run_root=tmp_path / "xgboost-v4-post-readiness-run",
        run_manifest_path=tmp_path / "xgboost-v4-post-readiness-run" / "run_manifest.json",
        objective_audit_path=output_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
    )
    pointer_payload = json.loads(post_readiness_latest_path.read_text(encoding="utf-8"))
    pointer_payload["objective_complete"] = True
    pointer_payload["objective_decision"] = "BLOCKED"
    pointer_payload["objective_blockers"] = ["stale blocker survived in latest pointer"]
    pointer_payload["objective_prompt_to_artifact_blockers"] = []
    post_readiness_latest_path.write_text(
        json.dumps(pointer_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=_write_slack_automation_toml(tmp_path / "automation.toml"),
        slack_delivery_status_path=slack_delivery_status_path,
        post_readiness_latest_path=post_readiness_latest_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    post_readiness = payload["post_readiness_latest"]
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}

    assert payload["objective_complete"] is False
    assert post_readiness["matches_current_inputs"] is True
    assert post_readiness["objective_complete"] is True
    assert post_readiness["objective_complete_clean"] is False
    assert post_readiness["objective_status_compatible"] is False
    assert post_readiness["objective_blockers"] == ["stale blocker survived in latest pointer"]
    assert checklist["post_readiness_latest_pointer"]["passed"] is False


def test_xgboost_v4_objective_audit_accepts_self_referential_pointer_blocker(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/C0B5VHYSCN8/p1",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"
    post_readiness_latest_path = _write_post_readiness_latest(
        tmp_path / "xgboost_v4_post_readiness_latest.json",
        run_root=tmp_path / "xgboost-v4-post-readiness-run",
        run_manifest_path=tmp_path / "xgboost-v4-post-readiness-run" / "run_manifest.json",
        objective_audit_path=output_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
    )
    pointer_payload = json.loads(post_readiness_latest_path.read_text(encoding="utf-8"))
    pointer_payload["objective_decision"] = "BLOCKED"
    pointer_payload["objective_complete"] = False
    pointer_payload["objective_prompt_to_artifact_blockers"] = [
        "post_readiness_latest_pointer: latest pointer not yet validated"
    ]
    pointer_payload["issue_coverage_objective_success_criteria"][
        "post_readiness_latest_pointer_valid"
    ]["passed"] = False
    post_readiness_latest_path.write_text(
        json.dumps(pointer_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=_write_slack_automation_toml(tmp_path / "automation.toml"),
        slack_delivery_status_path=slack_delivery_status_path,
        post_readiness_latest_path=post_readiness_latest_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    post_readiness = payload["post_readiness_latest"]
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}

    assert payload["objective_complete"] is True
    assert post_readiness["objective_complete_clean"] is False
    assert post_readiness["objective_status_compatible"] is True
    assert post_readiness["objective_only_blocks_on_latest_pointer"] is True
    assert post_readiness["issue_coverage_success_criteria_passed"] is False
    assert post_readiness["issue_coverage_success_criteria_without_pointer_passed"] is True
    assert post_readiness["issue_coverage_self_reference_bootstrap_compatible"] is True
    assert post_readiness["issue_coverage_summary_compatible"] is True
    assert checklist["post_readiness_latest_pointer"]["passed"] is True


def test_xgboost_v4_objective_audit_treats_pre_readiness_artifacts_as_provisional(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=False))
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=False, candidate_eval_dir=candidate_eval_dir),
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json"
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    blockers = " ".join(payload["blockers"])
    assert payload["objective_complete"] is False
    assert payload["decision"] == "BLOCKED"
    for issue_id in ("#57", "#58", "#64"):
        assert checks[issue_id]["passed"] is False
        assert checks[issue_id]["evidence"]["evidence_scope"] == "provisional"
        assert checks[issue_id]["evidence"]["final_candidate_evidence_ready"] is False
        assert issue_id in blockers
    assert checks["#57"]["evidence"]["passed"] is True
    assert checks["#58"]["evidence"]["ensemble_member_count"] == 3
    assert checks["#64"]["evidence"]["passed"] is True
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    assert checklist["create_xgboost_v4_model"]["passed"] is False
    assert checklist["beat_current_champion"]["passed"] is False
    assert checklist["champion_promotion_md"]["passed"] is False
    assert checklist["hourly_slack_status"]["passed"] is True
    for item_id in ("github_issue_55", "github_issue_56", "github_issue_65"):
        assert checklist[item_id]["evidence_scope"] == "collecting"
    for item_id in (
        "github_issue_57",
        "github_issue_58",
        "github_issue_64",
        "create_xgboost_v4_model",
    ):
        assert checklist[item_id]["evidence_scope"] == "provisional"
    assert checklist["beat_current_champion"]["evidence_scope"] == "blocked"
    assert checklist["champion_promotion_md"]["evidence_scope"] == "blocked"
    assert checklist["hourly_slack_status"]["evidence_scope"] == "active"
    assert "xgboost-v4" in payload["objective_restatement"]["summary"]
    assert payload["objective_restatement"]["slack_channel_id"] == "C0B5VHYSCN8"
    success_criteria = {item["id"]: item for item in payload["objective_success_criteria"]}
    assert success_criteria["all_requested_github_issues_satisfied"]["passed"] is False
    assert success_criteria["fresh_xgboost_v4_model_created"]["passed"] is False
    assert success_criteria["beats_current_champion"]["passed"] is False
    assert success_criteria["champion_promotion_gates_passed"]["passed"] is False
    assert success_criteria["hourly_slack_status_active"]["passed"] is True
    assert any(
        blocker.startswith("create_xgboost_v4_model:")
        for blocker in payload["prompt_to_artifact_blockers"]
    )
    assert any(
        blocker.startswith("beat_current_champion:")
        for blocker in payload["prompt_to_artifact_blockers"]
    )
    assert any(blocker.startswith("create_xgboost_v4_model:") for blocker in payload["blockers"])
    assert any(blocker.startswith("beat_current_champion:") for blocker in payload["blockers"])


def test_xgboost_v4_objective_audit_beat_champion_requires_full_eval_and_backtest_stages(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    promotion_payload = _promotion_audit_payload(
        passed=True,
        candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
    )
    promotion_payload["decision"] = "BLOCKED"
    promotion_payload["passed"] = False
    stage2 = promotion_payload["stages"][2]
    assert isinstance(stage2, dict)
    stage2["passed"] = False
    checks = stage2["checks"]
    assert isinstance(checks, list)
    checks.append({"name": "turnover_reasonable", "passed": False})
    promotion_path = _write_json(tmp_path / "promotion.json", promotion_payload)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    criteria = {item["id"]: item for item in payload["objective_success_criteria"]}
    assert payload["objective_complete"] is False
    assert checklist["beat_current_champion"]["passed"] is False
    assert criteria["beats_current_champion"]["passed"] is False
    assert "promotion.Stage 2" in checklist["beat_current_champion"]["evidence_keys"]
    assert any(
        blocker.startswith("beat_current_champion:")
        for blocker in payload["prompt_to_artifact_blockers"]
    )


def test_xgboost_v4_objective_audit_requires_same_dataset_rerun_for_final_candidate_scope(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_payload = _promotion_audit_payload(
        passed=True,
        candidate_eval_dir=candidate_eval_dir,
    )
    promotion_payload["decision"] = "BLOCKED"
    promotion_payload["passed"] = False
    stage1 = promotion_payload["stages"][1]
    assert isinstance(stage1, dict)
    stage1["passed"] = False
    checks = stage1["checks"]
    assert isinstance(checks, list)
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "rerun_report_exists":
            check["passed"] = False
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    promotion_payload["artifact_paths"] = {
        "candidate_eval_dir": str(candidate_evidence["candidate_eval_dir"])
    }
    promotion_path = _write_json(tmp_path / "promotion.json", promotion_payload)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    blockers = " ".join(payload["blockers"])
    assert payload["objective_complete"] is False
    assert payload["decision"] == "BLOCKED"
    for issue_id in ("#57", "#58", "#64", "#65"):
        evidence = checks[issue_id]["evidence"]
        assert checks[issue_id]["passed"] is False
        assert evidence["evidence_scope"] == "provisional"
        assert evidence["final_candidate_evidence_ready"] is False
        assert evidence["final_candidate_evidence_requirements"] == {
            "ready_for_training": True,
            "stage1_rerun_report_exists_passed": False,
            "stage1_same_dataset_split_passed": True,
            "stage1_dataset_time_split_5_1_1_passed": True,
            "stage1_candidate_model_version_passed": True,
            "candidate_eval_model_version_matches": True,
            "candidate_eval_model_path_matches": True,
            "candidate_eval_dataset_provenance_present": True,
        }
        assert issue_id in blockers
    assert checks["#57"]["evidence"]["passed"] is True
    assert checks["#58"]["evidence"]["ensemble_member_count"] == 3
    assert checks["#64"]["evidence"]["passed"] is True
    assert checks["#65"]["evidence"]["tick_features"]["passed"] is True


def test_xgboost_v4_objective_audit_requires_candidate_eval_model_provenance(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    blockers = " ".join(payload["blockers"])
    assert payload["objective_complete"] is False
    assert payload["decision"] == "BLOCKED"
    for issue_id in ("#57", "#58", "#64"):
        evidence = checks[issue_id]["evidence"]
        assert checks[issue_id]["passed"] is False
        assert evidence["evidence_scope"] == "provisional"
        assert evidence["final_candidate_evidence_ready"] is False
        assert evidence["final_candidate_evidence_requirements"] == {
            "ready_for_training": True,
            "stage1_rerun_report_exists_passed": True,
            "stage1_same_dataset_split_passed": True,
            "stage1_dataset_time_split_5_1_1_passed": True,
            "stage1_candidate_model_version_passed": True,
            "candidate_eval_model_version_matches": False,
            "candidate_eval_model_path_matches": False,
            "candidate_eval_dataset_provenance_present": False,
        }
        assert evidence["candidate_eval_model_provenance"] == {
            "passed": False,
            "candidate_eval_dir": None,
            "candidate_eval_manifest_path": None,
            "candidate_eval_model_version": None,
            "expected_model_version": "xgboost-v4",
            "model_version_matches": False,
            "candidate_eval_model_path": None,
            "expected_model_path": str(candidate_evidence["candidate_model_dir"] / "model.json"),
            "model_path_matches": False,
            "dataset_provenance_present": False,
            "candidate_eval_dataset_dir": None,
            "candidate_eval_dataset_version": None,
        }
        assert issue_id in blockers


def test_xgboost_v4_objective_audit_requires_candidate_eval_dataset_provenance(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
    )
    manifest_path = candidate_eval_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("dataset_dir")
    manifest.pop("dataset_version")
    _write_json(manifest_path, manifest)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, candidate_eval_dir=candidate_eval_dir),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    for issue_id in ("#57", "#58", "#64"):
        evidence = checks[issue_id]["evidence"]
        provenance = evidence["candidate_eval_model_provenance"]
        assert checks[issue_id]["passed"] is False
        assert evidence["final_candidate_evidence_ready"] is False
        assert evidence["final_candidate_evidence_requirements"][
            "candidate_eval_model_path_matches"
        ] is True
        assert evidence["final_candidate_evidence_requirements"][
            "candidate_eval_model_version_matches"
        ] is True
        assert evidence["final_candidate_evidence_requirements"][
            "candidate_eval_dataset_provenance_present"
        ] is False
        assert provenance["model_path_matches"] is True
        assert provenance["model_version_matches"] is True
        assert provenance["dataset_provenance_present"] is False
        assert provenance["candidate_eval_dataset_dir"] is None
        assert provenance["candidate_eval_dataset_version"] is None


def test_xgboost_v4_objective_audit_requires_candidate_eval_model_version(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
    )
    manifest_path = candidate_eval_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_version"] = "xgboost-v3"
    _write_json(manifest_path, manifest)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, candidate_eval_dir=candidate_eval_dir),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    for issue_id in ("#57", "#58", "#64", "#65"):
        evidence = checks[issue_id]["evidence"]
        provenance = evidence["candidate_eval_model_provenance"]
        assert checks[issue_id]["passed"] is False
        assert evidence["final_candidate_evidence_ready"] is False
        assert evidence["final_candidate_evidence_requirements"][
            "candidate_eval_model_version_matches"
        ] is False
        assert evidence["final_candidate_evidence_requirements"][
            "candidate_eval_model_path_matches"
        ] is True
        assert evidence["final_candidate_evidence_requirements"][
            "candidate_eval_dataset_provenance_present"
        ] is True
        assert provenance["candidate_eval_model_version"] == "xgboost-v3"
        assert provenance["expected_model_version"] == "xgboost-v4"
        assert provenance["model_version_matches"] is False
        assert provenance["model_path_matches"] is True
        assert provenance["dataset_provenance_present"] is True


def test_xgboost_v4_objective_audit_requires_feature_ablation_same_dataset(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    ablation_payload = json.loads(
        candidate_evidence["feature_ablation_path"].read_text(encoding="utf-8")
    )
    ablation_payload["dataset_dir"] = "runs/different-training-dataset"
    candidate_evidence["feature_ablation_path"].write_text(
        json.dumps(ablation_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    evidence = checks["#57"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#57"]["passed"] is False
    assert evidence["model_path_matches"] is True
    assert evidence["dataset_dir"] == "runs/different-training-dataset"
    assert evidence["expected_dataset_dir"] == "runs/training"
    assert evidence["dataset_dir_matches"] is False
    assert evidence["dataset_version_matches"] is True
    assert "#57" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_explicit_stage1_family_metrics(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_payload = _promotion_audit_payload(
        passed=True,
        candidate_eval_dir=candidate_eval_dir,
    )
    stage1 = promotion_payload["stages"][1]
    assert isinstance(stage1, dict)
    checks = stage1["checks"]
    assert isinstance(checks, list)
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "required_family_metrics_present":
            check["passed"] = False
    promotion_path = _write_json(tmp_path / "promotion.json", promotion_payload)
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    stability_report_path = _write_dataset_stability_report(
        tmp_path / "dataset_stability_report.json"
    )

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks_by_id = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks_by_id["#55"]["passed"] is True
    assert checks_by_id["#56"]["passed"] is False
    assert (
        checks_by_id["#56"]["evidence"]["stage1_required_family_metrics_present_passed"]
        is False
    )
    assert "#56" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_new_market_signal_evidence(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_payload = _promotion_audit_payload(passed=True)
    stage1 = promotion_payload["stages"][1]
    assert isinstance(stage1, dict)
    checks = stage1["checks"]
    assert isinstance(checks, list)
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "new_market_signal_present":
            check["passed"] = False
    promotion_path = _write_json(tmp_path / "promotion.json", promotion_payload)
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks_by_id = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks_by_id["#56"]["passed"] is False
    assert (
        checks_by_id["#56"]["evidence"]["stage1_new_market_signal_present_passed"]
        is False
    )
    assert "#56" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_thin_ablation_and_down_artifacts(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(tmp_path / "feature_ablation.json", {"rows": []})
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {"required_outcome_side": "UP", "issues": [], "summary": []},
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#57"]["passed"] is False
    assert "time" in checks["#57"]["evidence"]["missing_required_groups"]
    assert checks["#64"]["passed"] is False
    assert checks["#64"]["evidence"]["required_outcome_side"] == "UP"


def test_xgboost_v4_objective_audit_requires_named_group_ablations(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "feature"},
                {"name": "long_window", "ablation_type": "feature"},
                {"name": "trade_structure", "ablation_type": "feature"},
                {"name": "tick_microstructure", "ablation_type": "feature"},
                {"name": "other_group", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#57"]["passed"] is False
    assert checks["#57"]["evidence"]["group_ablation_names"] == ["other_group"]
    assert checks["#57"]["evidence"]["missing_required_groups"] == [
        "long_window",
        "tick_microstructure",
        "time",
        "trade_structure",
    ]
    assert "#57" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_unmeasured_group_ablations(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {
                    "name": "spread",
                    "ablation_type": "feature",
                    "features": ["spread"],
                    "metrics": {
                        "sample_count": 100,
                        "brier_score": 0.13,
                        "roc_auc": 0.68,
                    },
                    "deltas": {
                        "brier_score_increase": 0.01,
                        "roc_auc_drop": 0.02,
                    },
                },
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
        materialize_ablation_metrics=False,
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#57"]["passed"] is False
    assert checks["#57"]["evidence"]["missing_required_groups"] == []
    assert checks["#57"]["evidence"]["measured_feature_ablation_count"] == 1
    assert checks["#57"]["evidence"]["measured_required_group_names"] == []
    assert checks["#57"]["evidence"]["unmeasured_required_groups"] == [
        "long_window",
        "tick_microstructure",
        "time",
        "trade_structure",
    ]
    assert "#57" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_fresh_feature_importance(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    candidate_eval_dir = _write_candidate_eval_manifest(
        tmp_path / "candidate-eval",
        candidate_model_dir=candidate_model_dir,
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True, candidate_eval_dir=candidate_eval_dir),
    )
    (candidate_model_dir / "feature_importance.json").unlink()
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    evidence = checks["#57"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#57"]["passed"] is False
    assert evidence["passed"] is True
    assert evidence["feature_importance_passed"] is False
    assert evidence["feature_importance_exists"] is False
    assert evidence["valid_feature_importance_row_count"] == 0
    assert "#57" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_thin_cv_and_ensemble_evidence(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 2}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 2,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 2,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["cv_fold_count"] == 2
    assert checks["#58"]["evidence"]["required_cv_fold_count"] == 3
    assert checks["#58"]["evidence"]["ensemble_member_count"] == 2
    assert checks["#58"]["evidence"]["required_ensemble_member_count"] == 3
    assert "#58" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_ensemble_single_model_comparison(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    _write_json(
        candidate_evidence["candidate_model_dir"] / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
            "ensemble_vs_single": {
                "split": "test",
                "acceptable": False,
                "brier_delta": 0.02,
                "roc_auc_delta": -0.01,
                "pnl_delta": -0.03,
            },
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["ensemble_comparison_passed"] is False
    assert checks["#58"]["evidence"]["ensemble_vs_single_acceptable"] is False
    assert "#58" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_ensemble_cost_quantification(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    _write_json(
        candidate_evidence["candidate_model_dir"] / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.0,
            "train_time_multiplier_estimate": 0,
            "inference_eval_multiplier": 0,
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["ensemble_cost_quantification_passed"] is False
    assert checks["#58"]["evidence"]["training_elapsed_seconds"] == 0.0
    assert checks["#58"]["evidence"]["train_time_multiplier_estimate"] == 0
    assert checks["#58"]["evidence"]["inference_eval_multiplier"] == 0
    assert checks["#58"]["evidence"]["train_time_multiplier_matches_members"] is False
    assert checks["#58"]["evidence"]["inference_eval_multiplier_matches_members"] is False
    assert "#58" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_ensemble_cost_multipliers_to_match_members(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    _write_json(
        candidate_evidence["candidate_model_dir"] / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "train_time_multiplier_estimate": 2,
            "inference_eval_multiplier": 1,
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["ensemble_member_count"] == 3
    assert checks["#58"]["evidence"]["train_time_multiplier_estimate"] == 2
    assert checks["#58"]["evidence"]["inference_eval_multiplier"] == 1
    assert checks["#58"]["evidence"]["train_time_multiplier_matches_members"] is False
    assert checks["#58"]["evidence"]["inference_eval_multiplier_matches_members"] is False
    assert checks["#58"]["evidence"]["ensemble_cost_quantification_passed"] is False
    assert "#58" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_non_time_ordered_cv_evidence(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(
        candidate_model_dir / "cv_summary.json",
        {
            "summary": {"fold_count": 3},
            "folds": [
                {
                    "fold": 1,
                    "train_start_ts": 1_000,
                    "train_end_ts": 2_000,
                    "val_start_ts": 2_000,
                    "val_end_ts": 3_000,
                    "train_count": 10,
                    "val_count": 5,
                    "metrics": {"sample_count": 5, "brier_score": 0.12, "roc_auc": 0.70, "pnl": 0.20},
                },
                {
                    "fold": 2,
                    "train_start_ts": 1_000,
                    "train_end_ts": 4_000,
                    "val_start_ts": 5_000,
                    "val_end_ts": 6_000,
                    "train_count": 12,
                    "val_count": 5,
                    "metrics": {"sample_count": 5, "brier_score": 0.13, "roc_auc": 0.71, "pnl": 0.21},
                },
                {
                    "fold": 3,
                    "train_start_ts": 1_000,
                    "train_end_ts": 6_000,
                    "val_start_ts": 7_000,
                    "val_end_ts": 8_000,
                    "train_count": 14,
                    "val_count": 5,
                    "metrics": {"sample_count": 5, "brier_score": 0.14, "roc_auc": 0.72, "pnl": 0.22},
                },
            ],
        },
    )
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["cv_fold_count"] == 3
    assert checks["#58"]["evidence"]["cv_fold_row_count"] == 3
    assert checks["#58"]["evidence"]["cv_time_series_ordered"] is False
    assert checks["#58"]["evidence"]["invalid_cv_fold_indices"] == [0]
    assert "#58" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_boolean_cv_timestamps(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(
        candidate_model_dir / "cv_summary.json",
        {
            "summary": {"fold_count": 3},
            "folds": [
                {
                    "fold": 1,
                    "train_start_ts": False,
                    "train_end_ts": True,
                    "val_start_ts": 2,
                    "val_end_ts": 3,
                    "train_count": 10,
                    "val_count": 5,
                    "metrics": {"sample_count": 5, "brier_score": 0.12, "roc_auc": 0.70, "pnl": 0.20},
                },
                {
                    "fold": 2,
                    "train_start_ts": 1_000,
                    "train_end_ts": 4_000,
                    "val_start_ts": 5_000,
                    "val_end_ts": 6_000,
                    "train_count": 12,
                    "val_count": 5,
                    "metrics": {"sample_count": 5, "brier_score": 0.13, "roc_auc": 0.71, "pnl": 0.21},
                },
                {
                    "fold": 3,
                    "train_start_ts": 1_000,
                    "train_end_ts": 6_000,
                    "val_start_ts": 7_000,
                    "val_end_ts": 8_000,
                    "train_count": 14,
                    "val_count": 5,
                    "metrics": {"sample_count": 5, "brier_score": 0.14, "roc_auc": 0.72, "pnl": 0.22},
                },
            ],
        },
    )
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["invalid_cv_fold_indices"] == [0]
    assert checks["#58"]["evidence"]["invalid_cv_fold_metric_indices"] == [0]


def test_xgboost_v4_objective_audit_rejects_metricless_cv_evidence(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(
        candidate_model_dir / "cv_summary.json",
        {
            "summary": {"fold_count": 3},
            "folds": [
                {
                    "fold": 1,
                    "train_start_ts": 1_000,
                    "train_end_ts": 2_000,
                    "val_start_ts": 3_000,
                    "val_end_ts": 4_000,
                    "train_count": 10,
                    "val_count": 5,
                },
                {
                    "fold": 2,
                    "train_start_ts": 1_000,
                    "train_end_ts": 4_000,
                    "val_start_ts": 5_000,
                    "val_end_ts": 6_000,
                    "train_count": 12,
                    "val_count": 5,
                },
                {
                    "fold": 3,
                    "train_start_ts": 1_000,
                    "train_end_ts": 6_000,
                    "val_start_ts": 7_000,
                    "val_end_ts": 8_000,
                    "train_count": 14,
                    "val_count": 5,
                },
            ],
        },
    )
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["cv_time_series_ordered"] is True
    assert checks["#58"]["evidence"]["cv_fold_metrics_present"] is False
    assert checks["#58"]["evidence"]["invalid_cv_fold_metric_indices"] == [0, 1, 2]
    assert "#58" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_missing_ensemble_member_files(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
        materialize_model_members=False,
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["model_member_count"] == 3
    assert checks["#58"]["evidence"]["ensemble_member_count"] == 3
    assert checks["#58"]["evidence"]["member_paths_exist"] is False
    assert checks["#58"]["evidence"]["missing_member_paths"] == [
        str(candidate_model_dir / "model_seed_0.json"),
        str(candidate_model_dir / "model_seed_17.json"),
        str(candidate_model_dir / "model_seed_42.json"),
    ]
    assert "#58" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_wrong_model_version_cv_and_ensemble(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(
        candidate_model_dir / "cv_summary.json",
        {"model_version": "xgboost-v3", "summary": {"fold_count": 3}},
    )
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "model_version": "xgboost-v3",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "model_version": "xgboost-v3",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#58"]["passed"] is False
    assert checks["#58"]["evidence"]["cv_model_version"] == "xgboost-v3"
    assert checks["#58"]["evidence"]["ensemble_model_version"] == "xgboost-v3"
    assert checks["#58"]["evidence"]["model_wrapper_version"] == "xgboost-v3"
    assert checks["#58"]["evidence"]["member_paths_exist"] is True
    assert "#58" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_stale_artifact_provenance(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "fresh-xgboost-v4"
    stale_model_path = tmp_path / "models" / "old-xgboost-v4" / "model.json"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(stale_model_path),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(stale_model_path)},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#57"]["passed"] is False
    assert checks["#57"]["evidence"]["model_path_matches"] is False
    assert checks["#64"]["passed"] is False
    assert checks["#64"]["evidence"]["model_path_matches"] is False


def test_xgboost_v4_objective_audit_rejects_thin_down_validation_provenance(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
        materialize_down_validation=False,
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#64"]["passed"] is False
    evidence = checks["#64"]["evidence"]
    assert evidence["model_path_matches"] is True
    assert evidence["qualifying_row_count"] == 0
    assert evidence["missing_metadata_fields"] == [
        "backtest_kind",
        "dataset_dir",
        "dataset_version",
        "warehouse_dir",
    ]
    assert evidence["invalid_summary_rows"][0]["missing_or_invalid"] == [
        "threshold_signals",
        "threshold",
        "edge_threshold",
        "gross_pnl",
        "brier_score",
        "brier_sample_count",
        "turnover",
        "symbols_considered",
        "symbols_with_quotes",
        "hold_ms",
        "fee_bps",
        "slippage_bps",
    ]
    assert "#64" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_down_trade_sample(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    sample_path = tmp_path / "trade_log_sample_threshold_0_03.jsonl"
    sample_path.unlink()
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    evidence = checks["#64"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#64"]["passed"] is False
    assert evidence["trade_sample"] == {
        "passed": False,
        "path": str(sample_path),
        "exists": False,
        "expected_outcome_side": "DOWN",
        "row_count": 0,
        "trade_count": 3,
        "sample_count_exceeds_trade_count": False,
        "invalid_rows": [],
        "read_error": None,
    }
    assert "#64" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_down_trade_sample_realized_label(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    sample_path = tmp_path / "trade_log_sample_threshold_0_03.jsonl"
    sample_rows = [
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in sample_rows:
        row.pop("realized_label", None)
    sample_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in sample_rows) + "\n",
        encoding="utf-8",
    )
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    evidence = checks["#64"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#64"]["passed"] is False
    assert evidence["trade_sample"]["passed"] is False
    assert evidence["trade_sample"]["invalid_rows"] == [
        {"index": 0, "missing_or_invalid": ["realized_label"]}
    ]
    assert "#64" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_down_validation_same_dataset(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    down_payload = json.loads(
        candidate_evidence["down_validation_path"].read_text(encoding="utf-8")
    )
    down_payload["metadata"]["dataset_dir"] = "runs/different-training-dataset"
    _write_json(candidate_evidence["down_validation_path"], down_payload)
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(
            passed=True,
            candidate_eval_dir=candidate_evidence["candidate_eval_dir"],
        ),
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    evidence = checks["#64"]["evidence"]
    assert payload["objective_complete"] is False
    assert checks["#64"]["passed"] is False
    assert evidence["dataset_dir"] == "runs/different-training-dataset"
    assert evidence["expected_dataset_dir"] == "runs/training"
    assert evidence["dataset_dir_matches"] is False
    assert evidence["dataset_version_matches"] is True
    assert evidence["missing_metadata_fields"] == ["dataset_dir"]
    assert "#64" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_non_positive_down_pnl(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.0,
                }
            ],
        },
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    assert payload["objective_complete"] is False
    assert checks["#64"]["passed"] is False
    assert checks["#64"]["evidence"]["best_net_pnl"] == 0.0
    assert checks["#64"]["evidence"]["qualifying_row_count"] == 1
    assert "#64" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_boolean_down_metrics(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "model.json", {"model_version": "xgboost-v4"})
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {
                "backtest_kind": "direct_model",
                "model_path": str(candidate_model_dir / "model.json"),
                "dataset_dir": "runs/training-dataset",
                "dataset_version": "dataset-v1",
                "warehouse_dir": "runs/warehouse",
            },
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "threshold_signals": 3,
                    "trade_count": 3,
                    "threshold": 0.03,
                    "edge_threshold": 0.03,
                    "gross_pnl": 1.0,
                    "net_pnl": True,
                    "turnover": 0.3,
                    "symbols_considered": 2,
                    "symbols_with_quotes": 2,
                    "hold_ms": 900_000,
                    "settings": {"fee_bps": 10.0, "slippage_bps": 5.0},
                }
            ],
        },
        materialize_down_validation=False,
    )
    output_path = tmp_path / "objective.json"
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["id"]: check for check in payload["issue_checks"]}
    evidence = checks["#64"]["evidence"]
    assert checks["#64"]["passed"] is False
    assert evidence["qualifying_row_count"] == 0
    assert evidence["invalid_summary_rows"][0]["missing_or_invalid"] == [
        "net_pnl",
        "brier_score",
        "brier_sample_count",
    ]


def test_xgboost_v4_objective_audit_rejects_missing_or_wrong_slack_automation(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    slack_automation_path = _write_slack_automation_toml(
        tmp_path / "automation.toml",
        channel_id="WRONG",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert operational_checks["slack_hourly_status"]["evidence"]["channel_id_present"] is False
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_status_only_slack_automation(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = tmp_path / "automation.toml"
    slack_automation_path.write_text(
        "\n".join(
            [
                'id = "xgboost-v4-work-status"',
                'kind = "heartbeat"',
                'status = "ACTIVE"',
                'rrule = "FREQ=HOURLY;INTERVAL=1"',
                (
                    'prompt = "Run xgboost-v4-objective-audit --no-fail-on-blocked '
                    'then send hourly Slack status to channel_id C0B5VHYSCN8."'
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    evidence = operational_checks["slack_hourly_status"]["evidence"]
    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert evidence["objective_audit_instruction_present"] is True
    assert evidence["post_readiness_runner_instruction_present"] is False
    assert evidence["post_readiness_pointer_instruction_present"] is False
    assert evidence["post_readiness_pointer_summary_instruction_present"] is False
    assert evidence["post_readiness_duplicate_guard_instruction_present"] is False
    assert evidence["post_readiness_objective_refresh_instruction_present"] is False
    assert evidence["post_readiness_shadow_continuation_instruction_present"] is False
    assert evidence["post_readiness_shadow_auto_window_instruction_present"] is False
    assert evidence["post_readiness_shadow_full_session_instruction_present"] is False
    assert evidence["objective_success_criteria_reporting_instruction_present"] is False
    assert evidence["issue_coverage_audit_instruction_present"] is False
    assert evidence["collection_risk_helper_instruction_present"] is False
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_issue_coverage_slack_instruction(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(
            "refresh xgboost-v4-issue-coverage-audit and "
            "issue_coverage_audit.json; ",
            "",
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is False
    assert slack_evidence["objective_audit_instruction_present"] is True
    assert slack_evidence["issue_coverage_audit_instruction_present"] is False
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_requires_pointer_summary_slack_instruction(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace("run_manifest_phase, live_status_summary, artifact_paths, ", ""),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    evidence = operational_checks["slack_hourly_status"]["evidence"]
    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert evidence["post_readiness_pointer_instruction_present"] is True
    assert evidence["post_readiness_pointer_summary_instruction_present"] is False
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_failed_slack_delivery_status(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": "2026-05-24T10:52:00Z",
            "channel_id": "C0B5VHYSCN8",
            "error_code": "token_expired",
            "error_message": "Provided authentication token is expired. Please try signing in again.",
            "ok": False,
            "status": "failed",
        },
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    success_criteria = {item["id"]: item for item in payload["objective_success_criteria"]}
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is True
    assert slack_evidence["delivery_status"]["checked"] is True
    assert slack_evidence["delivery_status"]["passed"] is False
    assert slack_evidence["delivery_status"]["error_code"] == "token_expired"
    assert (
        slack_evidence["delivery_status"]["error_message"]
        == "Provided authentication token is expired. Please try signing in again."
    )
    assert slack_evidence["slack_delivery_status_helper_instruction_present"] is True
    assert slack_evidence["delivery_status"]["attempted_at_present"] is True
    assert slack_evidence["delivery_status"]["message_link_present"] is False
    assert checklist["hourly_slack_status"]["passed"] is False
    assert success_criteria["hourly_slack_status_active"]["passed"] is False
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_slack_delivery_status_writes_failed_attempt(tmp_path: Path) -> None:
    from bigan.ingestion.__main__ import slack_delivery_status

    output_path = tmp_path / "artifacts" / "slack_delivery.json"

    slack_delivery_status(
        output_path=output_path,
        channel_id="C0B5VHYSCN8",
        attempted_at="2026-05-25T00:16:42Z",
        ok=False,
        status="failed",
        error_code="token_expired",
        error_message="Provided authentication token is expired. Please try signing in again.",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "attempted_at": "2026-05-25T00:16:42Z",
        "channel_id": "C0B5VHYSCN8",
        "error_code": "token_expired",
        "error_message": "Provided authentication token is expired. Please try signing in again.",
        "message_link": None,
        "ok": False,
        "status": "failed",
    }


def test_slack_delivery_status_rejects_success_without_message_link(
    tmp_path: Path,
) -> None:
    import typer

    from bigan.ingestion.__main__ import slack_delivery_status

    output_path = tmp_path / "slack_delivery.json"

    with pytest.raises(typer.BadParameter):
        slack_delivery_status(
            output_path=output_path,
            channel_id="C0B5VHYSCN8",
            attempted_at="2026-05-25T00:16:42Z",
            ok=True,
            status="sent",
        )

    assert not output_path.exists()


def test_xgboost_v4_objective_audit_rejects_delivery_success_without_message_link(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is True
    assert slack_evidence["delivery_status"]["ok"] is True
    assert slack_evidence["delivery_status"]["status"] == "sent"
    assert slack_evidence["delivery_status"]["attempted_at_present"] is True
    assert slack_evidence["delivery_status"]["attempted_at_fresh"] is True
    assert slack_evidence["delivery_status"]["message_link_present"] is False
    assert slack_evidence["delivery_status"]["message_link_channel_matches"] is False
    assert slack_evidence["delivery_status"]["passed"] is False
    assert checklist["hourly_slack_status"]["passed"] is False


def test_xgboost_v4_objective_audit_rejects_delivery_success_with_wrong_channel_link(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/CNOTTARGET/p1",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is True
    assert slack_evidence["delivery_status"]["channel_id_matches"] is True
    assert slack_evidence["delivery_status"]["message_link_present"] is True
    assert slack_evidence["delivery_status"]["message_link_channel_matches"] is False
    assert slack_evidence["delivery_status"]["passed"] is False
    assert checklist["hourly_slack_status"]["passed"] is False


def test_xgboost_v4_objective_audit_rejects_stale_successful_slack_delivery_status(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": "2000-01-01T00:00:00Z",
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/C0B5VHYSCN8/p1",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    checklist = {item["id"]: item for item in payload["prompt_to_artifact_checklist"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is True
    assert slack_evidence["delivery_status"]["ok"] is True
    assert slack_evidence["delivery_status"]["message_link_present"] is True
    assert slack_evidence["delivery_status"]["message_link_channel_matches"] is True
    assert slack_evidence["delivery_status"]["attempted_at_fresh"] is False
    assert slack_evidence["delivery_status"]["attempted_at_age_seconds"] > slack_evidence[
        "delivery_status"
    ]["max_age_seconds"]
    assert slack_evidence["delivery_status"]["passed"] is False
    assert checklist["hourly_slack_status"]["passed"] is False


def test_xgboost_v4_objective_audit_defaults_delivery_status_for_default_automation(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import (
        DEFAULT_XGBOOST_V4_COLLECTION_RISK_PATH,
        DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH,
        DEFAULT_XGBOOST_V4_SLACK_DELIVERY_STATUS_PATH,
        _resolve_xgboost_v4_collection_risk_path,
        _resolve_xgboost_v4_slack_delivery_status_path,
    )

    explicit_path = tmp_path / "slack_delivery.json"
    explicit_risk_path = tmp_path / "collection_risk_latest.json"

    assert _resolve_xgboost_v4_slack_delivery_status_path(
        None,
        slack_automation_path=DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH,
    ) == DEFAULT_XGBOOST_V4_SLACK_DELIVERY_STATUS_PATH
    assert _resolve_xgboost_v4_slack_delivery_status_path(
        explicit_path,
        slack_automation_path=DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH,
    ) == explicit_path
    assert _resolve_xgboost_v4_slack_delivery_status_path(
        None,
        slack_automation_path=tmp_path / "automation.toml",
    ) is None
    assert _resolve_xgboost_v4_collection_risk_path(
        None,
        slack_automation_path=DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH,
    ) == DEFAULT_XGBOOST_V4_COLLECTION_RISK_PATH
    assert _resolve_xgboost_v4_collection_risk_path(
        explicit_risk_path,
        slack_automation_path=DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH,
    ) == explicit_risk_path
    assert _resolve_xgboost_v4_collection_risk_path(
        None,
        slack_automation_path=tmp_path / "automation.toml",
    ) is None


def test_xgboost_v4_objective_audit_rejects_contradictory_slack_delivery_status(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": "2026-05-24T10:52:00Z",
            "channel_id": "C0B5VHYSCN8",
            "ok": False,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is True
    assert slack_evidence["delivery_status"]["ok"] is False
    assert slack_evidence["delivery_status"]["status"] == "sent"
    assert slack_evidence["delivery_status"]["passed"] is False


def test_xgboost_v4_objective_audit_rejects_delivery_status_without_automation_instruction(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(
            "update slack_status_delivery_latest.json with message_link and "
            "error_code using slack-delivery-status --message-link "
            "--error-code --error-message --output-path "
            "slack_status_delivery_latest.json; rerun "
            "xgboost-v4-objective-audit with "
            "--slack-delivery-status-path slack_status_delivery_latest.json; ",
            "",
        ),
        encoding="utf-8",
    )
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/C0B5VHYSCN8/p1",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is False
    assert slack_evidence["slack_delivery_status_instruction_present"] is False
    assert slack_evidence["slack_delivery_status_helper_instruction_present"] is False
    assert slack_evidence["delivery_status"]["passed"] is True
    assert slack_evidence["delivery_status"]["message_link_channel_matches"] is True


def test_xgboost_v4_objective_audit_rejects_delivery_status_without_helper_command(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    attempted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(" using slack-delivery-status", ""),
        encoding="utf-8",
    )
    slack_delivery_status_path = _write_json(
        tmp_path / "slack_delivery.json",
        {
            "attempted_at": attempted_at,
            "channel_id": "C0B5VHYSCN8",
            "message_link": "https://cashbility.slack.com/archives/C0B5VHYSCN8/p1",
            "ok": True,
            "status": "sent",
        },
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["slack_delivery_status_instruction_present"] is True
    assert slack_evidence["slack_delivery_status_helper_instruction_present"] is False
    assert slack_evidence["delivery_status"]["passed"] is True
    assert "slack_hourly_status" in " ".join(payload["blockers"])
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_automation_without_collection_risk_helper(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(
            "if disk_headroom_evidence.headroom_low_margin=true or "
            "disk_headroom_evidence.headroom_ok=false, run "
            "scripts/check_xgboost_v4_collection_risk.sh --json "
            "--output-path data/xgboost-v4-run-20260523T103814Z/"
            "artifacts/collection_risk_latest.json and parse "
            "collection_risk_latest.json plus "
            "status_artifact.fresh, status_artifact.age_seconds, "
            "status_artifact.max_age_seconds, current_filesystem_headroom, "
            "reclaim_to_clear_block_bytes, "
            "disk_urgency.estimated_growth_bytes_per_day, "
            "disk_urgency.current_filesystem_days_to_min_free, and "
            "disk_urgency.current_filesystem_min_free_before_ready, and "
            "reclaim_candidates; report Docker and CoreSimulator "
            "reclaim candidates plus the days-to-min-free urgency clock; "
            "include whether min-free arrives before readiness; "
            "do not prune Docker, delete simulator data, "
            "clear caches, or remove old roots without explicit user approval; ",
            "",
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is False
    assert slack_evidence["collection_risk_helper_instruction_present"] is False
    assert slack_evidence["collection_risk_helper_json_instruction_present"] is False
    assert slack_evidence["collection_risk_helper_urgency_instruction_present"] is False
    assert slack_evidence["collection_risk_helper_output_path_instruction_present"] is False
    assert (
        slack_evidence["collection_risk_helper_status_freshness_instruction_present"]
        is False
    )
    assert slack_evidence["skip_label_refresh_on_disk_block_instruction_present"] is True
    assert slack_evidence["post_readiness_runner_instruction_present"] is True
    assert slack_evidence["delivery_status"]["passed"] is True
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_collection_risk_helper_without_json(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(
            "scripts/check_xgboost_v4_collection_risk.sh --json "
            "--output-path data/xgboost-v4-run-20260523T103814Z/"
            "artifacts/collection_risk_latest.json and parse "
            "collection_risk_latest.json plus "
            "status_artifact.fresh, status_artifact.age_seconds, "
            "status_artifact.max_age_seconds, current_filesystem_headroom, "
            "reclaim_to_clear_block_bytes, "
            "disk_urgency.estimated_growth_bytes_per_day, "
            "disk_urgency.current_filesystem_days_to_min_free, and "
            "disk_urgency.current_filesystem_min_free_before_ready, and "
            "reclaim_candidates; ",
            "scripts/check_xgboost_v4_collection_risk.sh and report ",
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is False
    assert slack_evidence["collection_risk_helper_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_json_instruction_present"] is False
    assert slack_evidence["collection_risk_helper_urgency_instruction_present"] is False
    assert slack_evidence["collection_risk_helper_output_path_instruction_present"] is False
    assert (
        slack_evidence["collection_risk_helper_status_freshness_instruction_present"]
        is False
    )
    assert slack_evidence["skip_label_refresh_on_disk_block_instruction_present"] is True
    assert slack_evidence["delivery_status"]["passed"] is True
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_collection_risk_helper_without_output_path(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(
            " --output-path data/xgboost-v4-run-20260523T103814Z/"
            "artifacts/collection_risk_latest.json",
            "",
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is False
    assert slack_evidence["collection_risk_helper_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_json_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_urgency_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_output_path_instruction_present"] is False
    assert (
        slack_evidence["collection_risk_helper_status_freshness_instruction_present"]
        is True
    )
    assert slack_evidence["skip_label_refresh_on_disk_block_instruction_present"] is True
    assert slack_evidence["delivery_status"]["passed"] is True
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_collection_risk_helper_without_urgency_clock(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(
            "disk_urgency.estimated_growth_bytes_per_day, "
            "disk_urgency.current_filesystem_days_to_min_free, "
            "disk_urgency.current_filesystem_min_free_before_ready, and ",
            "",
        ).replace(" plus the days-to-min-free urgency clock", "").replace(
            "include whether min-free arrives before readiness; ",
            "",
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is False
    assert slack_evidence["collection_risk_helper_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_json_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_urgency_instruction_present"] is False
    assert slack_evidence["collection_risk_helper_output_path_instruction_present"] is True
    assert (
        slack_evidence["collection_risk_helper_status_freshness_instruction_present"]
        is True
    )
    assert slack_evidence["skip_label_refresh_on_disk_block_instruction_present"] is True
    assert slack_evidence["delivery_status"]["passed"] is True
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_collection_risk_helper_without_status_freshness(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(
            "status_artifact.fresh, status_artifact.age_seconds, "
            "status_artifact.max_age_seconds, ",
            "",
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is False
    assert slack_evidence["collection_risk_helper_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_json_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_urgency_instruction_present"] is False
    assert slack_evidence["collection_risk_helper_output_path_instruction_present"] is True
    assert (
        slack_evidence["collection_risk_helper_status_freshness_instruction_present"]
        is False
    )
    assert slack_evidence["skip_label_refresh_on_disk_block_instruction_present"] is True
    assert slack_evidence["delivery_status"]["passed"] is True
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_automation_without_disk_blocked_label_skip(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(
            "if disk_headroom_evidence.headroom_ok=false or "
            "current_disk_headroom_evidence.headroom_ok=false, skip bounded "
            "settled-label refresh to avoid adding avoidable writes; ",
            "",
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["config_passed"] is False
    assert slack_evidence["collection_risk_helper_json_instruction_present"] is True
    assert slack_evidence["collection_risk_helper_output_path_instruction_present"] is True
    assert slack_evidence["skip_label_refresh_on_disk_block_instruction_present"] is False
    assert slack_evidence["delivery_status"]["passed"] is True
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_shadow_continuation_without_full_session_instruction(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_evidence = _write_complete_xgboost_v4_candidate_evidence(tmp_path)
    slack_automation_path = _write_slack_automation_toml(tmp_path / "automation.toml")
    prompt_text = slack_automation_path.read_text(encoding="utf-8")
    slack_automation_path.write_text(
        prompt_text.replace(" MIN_SHADOW_SESSION_SECONDS=86400", ""),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_evidence["candidate_model_dir"],
        feature_ablation_path=candidate_evidence["feature_ablation_path"],
        stability_report_path=candidate_evidence["stability_report_path"],
        down_validation_path=candidate_evidence["down_validation_path"],
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    slack_evidence = operational_checks["slack_hourly_status"]["evidence"]

    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert slack_evidence["post_readiness_shadow_continuation_instruction_present"] is True
    assert slack_evidence["post_readiness_shadow_auto_window_instruction_present"] is True
    assert slack_evidence["post_readiness_shadow_full_session_instruction_present"] is False
    assert "slack_hourly_status" in " ".join(payload["blockers"])


def test_xgboost_v4_objective_audit_rejects_stale_slack_automation_prompt(
    tmp_path: Path,
) -> None:
    from bigan.ingestion.__main__ import xgboost_v4_objective_audit

    status_path = _write_json(tmp_path / "status.json", _live_status_payload(ready=True))
    promotion_path = _write_json(
        tmp_path / "promotion.json",
        _promotion_audit_payload(passed=True),
    )
    candidate_model_dir = tmp_path / "models" / "xgboost-v4"
    _write_json(candidate_model_dir / "cv_summary.json", {"summary": {"fold_count": 3}})
    _write_json(
        candidate_model_dir / "ensemble_summary.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "member_count": 3,
            "training_elapsed_seconds": 0.42,
            "inference_eval_multiplier": 3,
        },
    )
    _write_json(
        candidate_model_dir / "model.json",
        {
            "schema_version": "xgboost_ensemble_v1",
            "members": [
                {"seed": 0, "path": "model_seed_0.json"},
                {"seed": 17, "path": "model_seed_17.json"},
                {"seed": 42, "path": "model_seed_42.json"},
            ],
        },
    )
    feature_ablation_path = _write_json(
        tmp_path / "feature_ablation.json",
        {
            "model_version": "xgboost-v4",
            "model_path": str(candidate_model_dir / "model.json"),
            "split": "test",
            "replacement_strategy": "train_split_feature_mean",
            "baseline_metrics": {
                "sample_count": 100,
                "brier_score": 0.12,
                "roc_auc": 0.70,
            },
            "ablations": [
                {"name": "spread", "ablation_type": "feature"},
                {"name": "time", "ablation_type": "group"},
                {"name": "long_window", "ablation_type": "group"},
                {"name": "trade_structure", "ablation_type": "group"},
                {"name": "tick_microstructure", "ablation_type": "group"},
            ],
        },
    )
    down_validation_path = _write_json(
        tmp_path / "buy_down_validation.json",
        {
            "model_version": "xgboost-v4",
            "required_outcome_side": "DOWN",
            "metadata": {"model_path": str(candidate_model_dir / "model.json")},
            "issues": [],
            "summary": [
                {
                    "signals_considered": 10,
                    "trade_count": 3,
                    "net_pnl": 0.7,
                }
            ],
        },
    )
    slack_automation_path = tmp_path / "automation.toml"
    slack_automation_path.write_text(
        "\n".join(
            [
                'id = "xgboost-v4-work-status"',
                'kind = "heartbeat"',
                'status = "ACTIVE"',
                'rrule = "FREQ=HOURLY;INTERVAL=1"',
                'prompt = "Send hourly Slack status to channel_id C0B5VHYSCN8."',
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "objective.json"

    xgboost_v4_objective_audit(
        output_path=output_path,
        live_status_path=status_path,
        promotion_audit_path=promotion_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        no_fail_on_blocked=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    operational_checks = {check["id"]: check for check in payload["operational_checks"]}
    assert payload["objective_complete"] is False
    assert operational_checks["slack_hourly_status"]["passed"] is False
    assert (
        operational_checks["slack_hourly_status"]["evidence"]["objective_audit_instruction_present"]
        is False
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_runner_instruction_present"
        ]
        is False
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_pointer_summary_instruction_present"
        ]
        is False
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_objective_refresh_instruction_present"
        ]
        is False
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_shadow_continuation_instruction_present"
        ]
        is False
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_shadow_auto_window_instruction_present"
        ]
        is False
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "post_readiness_shadow_full_session_instruction_present"
        ]
        is False
    )
    assert (
        operational_checks["slack_hourly_status"]["evidence"][
            "objective_success_criteria_reporting_instruction_present"
        ]
        is False
    )
    assert "slack_hourly_status" in " ".join(payload["blockers"])
