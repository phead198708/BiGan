"""CLI entry point: ``python -m bigan.ingestion`` or ``bigan-ingest``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import typer
from prometheus_client import start_http_server
from typer import Exit

from bigan.backtest import (
    TakerExecutionSettings,
    load_backtest_config,
    run_model_threshold_backtest,
    run_oracle_label_sanity_backtest,
    run_prediction_threshold_backtest,
)
from bigan.canonical.etl import run_etl_batch
from bigan.canonical.query import open_warehouse, warehouse_summary
from bigan.canonical.symbols import SymbolMapper
from bigan.features import run_feature_batch, run_feature_quality_sql_checks
from bigan.labels import run_label_batch
from bigan.labels.generation import DOWN_LABEL_KIND, generate_labels_15m_v1
from bigan.mlops import (
    ACTIVE_CHAMPION_ENVIRONMENT,
    ACTIVE_CHAMPION_MODEL_VERSION,
    ACTIVE_MODEL_FAMILY,
    DEFAULT_MLOPS_DB_PATH,
    connect_mlops_db,
    current_champion,
    current_online_model,
    initialize_mlops_db,
    model_by_version,
    run_shadow_warehouse_comparison,
    shadow_evaluation_output_paths,
)
from bigan.modeling import (
    DEFAULT_CHAMPION_PROMOTION_PROCESS_PATH,
    DEFAULT_CHAMPION_PROMOTION_REPO_RUNBOOK_PATH,
    XGBOOST_V4_REQUIRED_ADDED_FEATURES,
    XGBOOST_V4_REQUIRED_MARKET_FEATURES,
    XGBOOST_V4_REQUIRED_TICK_FEATURES,
    BootstrapCandidateInput,
    LogisticBaselineConfig,
    SplitConfig,
    XGBoostV1Config,
    assemble_training_dataset,
    audit_champion_promotion_process,
    evaluate_bootstrap_champion,
    evaluate_model_promotion,
    evaluate_probability_model_on_dataset,
    fit_probability_calibration,
    generate_dataset_stability_report,
    generate_feature_ablation_report,
    generate_offline_rerun_report,
    run_prediction_batch,
    train_logistic_baseline,
    train_xgboost_v1,
    train_xgboost_v2,
    train_xgboost_v3,
    train_xgboost_v4,
)
from bigan.modeling.promotion import (
    EXPECTED_CUTOVER_GITHUB_REPO,
    REQUIRED_CUTOVER_GITHUB_ISSUES,
)
from bigan.monitoring import (
    CHAMPION_BASELINE_DISTRIBUTIONS,
    INCIDENT_TYPES,
    ChampionDriftThresholds,
    ChampionSignalRow,
    PositionSignalState,
    build_champion_drift_baseline,
    champion_baseline_distribution,
    evaluate_label_hit_rate_drift,
    evaluate_live_champion_drift,
    evaluate_position_signal,
    format_signal_row,
    latest_signal_cursor,
    read_dashboard_snapshot,
    read_recent_signal_rows,
    read_signal_rows_after,
    record_champion_drift_incidents,
    render_dashboard,
    run_live_champion_monitoring,
)
from bigan.monitoring.collection_status import (
    build_live_collection_status,
    live_collection_readiness_decision,
    read_live_collection_status,
    write_live_collection_status,
)
from bigan.serving.readiness import run_xgboost_serving_readiness

from .backfill import BackfillService, GapWindow
from .clob_rest import PolymarketRestClient
from .config import IngestionSettings
from .gamma_client import GammaClient, parse_market_specs_json
from .market_compare import compare_market_coverage
from .metrics import REGISTRY
from .price_readers import (
    ChainlinkOracleReader,
    ChainlinkReaderConfig,
    CoinbaseTickerReader,
    KrakenTickerReader,
    WarehousePriceSink,
    WsPriceReaderConfig,
)
from .runner import IngestionRunner
from .sink import NdjsonGzipSink
from .soak import (
    SoakThresholds,
    finalize_soak_rollup,
    read_soak_samples,
    record_soak_samples,
    summarize_soak,
    write_soak_summary,
)

app = typer.Typer(add_completion=False, help="BiGan ingestion service")
SYMBOL_MAPPING_PATH_OPTION = typer.Option(
    None,
    help="Optional CSV, JSON, JSONL, or directory of symbol_mapping rows.",
)
PROCESSED_MANIFEST_PATH_OPTION = typer.Option(
    None,
    help=(
        "Optional text manifest of immutable raw files already processed; "
        "use with segmented raw sinks to avoid replaying old chunks."
    ),
)
TIMESTAMP_FUTURE_GRACE_SECONDS_OPTION = typer.Option(
    None,
    help="Override BIGAN_TIMESTAMP_FUTURE_GRACE_SECONDS for this ETL run.",
)
TIMESTAMP_STALE_THRESHOLD_SECONDS_OPTION = typer.Option(
    None,
    help="Override BIGAN_TIMESTAMP_STALE_THRESHOLD_SECONDS for this ETL run.",
)
SOAK_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/soak"),
    help="Directory for soak sample NDJSON and summary JSON evidence.",
)
SOAK_SAMPLES_PATH_OPTION = typer.Option(
    ...,
    help="Soak sample NDJSON emitted by soak.",
)
SOAK_RAW_DIR_OPTION = typer.Option(
    None,
    help="Raw NDJSON directory. Defaults to BIGAN_DATA_DIR/BIGAN_RAW_SUBDIR.",
)
SOAK_ROLLUP_DIR_OPTION = typer.Option(
    None,
    help="Rollup Parquet directory. Defaults to BIGAN_DATA_DIR/BIGAN_ROLLUP_SUBDIR.",
)
SOAK_SUMMARY_PATH_OPTION = typer.Option(
    None,
    help="Optional path to write the JSON summary.",
)
BACKTEST_CONFIG_PATH_ARGUMENT = typer.Argument(
    ...,
    help="Backtest YAML or JSON config path.",
)
TRAINING_DATASET_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Directory for train.parquet, val.parquet, test.parquet, and manifest.json.",
)
LOGISTIC_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory produced by training-dataset-v1.",
)
LOGISTIC_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/logreg-baseline-v1"),
    help="Directory for logistic baseline artifacts.",
)
XGBOOST_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory produced by training-dataset-v1.",
)
XGBOOST_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1"),
    help="Directory for XGBoost-v1 artifacts.",
)
XGBOOST_V2_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v2"),
    help="Directory for XGBoost-v2 artifacts.",
)
XGBOOST_V3_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v3"),
    help="Directory for XGBoost-v3 artifacts.",
)
XGBOOST_V4_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v4"),
    help="Directory for XGBoost-v4 artifacts.",
)
CALIBRATION_MODEL_PATH_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1/model.json"),
    help="Path to a saved model.json produced by xgboost-v1.",
)
CALIBRATION_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory with validation split.",
)
CALIBRATION_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1-calibration"),
    help="Directory for calibration artifacts.",
)
MODEL_EVAL_MODEL_PATH_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1/model.json"),
    help="Saved probability model artifact to evaluate.",
)
MODEL_EVAL_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory with train/val/test splits.",
)
MODEL_EVAL_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/model-eval-v1"),
    help="Directory for same-dataset evaluation metrics.",
)
MODEL_EVAL_CALIBRATION_PATH_OPTION = typer.Option(
    None,
    help="Optional calibration.json to apply before scoring metrics.",
)
DATASET_STABILITY_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/dataset-stability-v1"),
    help="Directory for issue #55 dataset distribution/stability artifacts.",
)
FEATURE_ABLATION_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/feature-ablation-v1"),
    help="Directory for feature ablation artifacts.",
)
FEATURE_ABLATION_SPLIT_OPTION = typer.Option(
    "test",
    help="Dataset split to ablate.",
)
PROMOTION_BASELINE_DIR_OPTION = typer.Option(
    Path("data/model-runs/logreg-baseline-v1"),
    help="Baseline model run directory.",
)
PROMOTION_CANDIDATE_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1"),
    help="Candidate model run directory.",
)
PROMOTION_CALIBRATION_DIR_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1-calibration"),
    help="Calibration artifact directory.",
)
PROMOTION_BACKTEST_SUMMARY_OPTION = typer.Option(
    Path("data/backtests/summary.json"),
    help="Threshold-backtest summary JSON.",
)
PROMOTION_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/promotion-v1"),
    help="Directory for promotion report artifacts.",
)
CHAMPION_PROMOTION_AUDIT_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/champion-promotion-audit"),
    help="Directory for champion-promotion.md gate audit artifacts.",
)
OFFLINE_RERUN_REPORT_OUTPUT_PATH_OPTION = typer.Option(
    Path("data/model-runs/offline-rerun-report/rerun_report.md"),
    help="Path for Stage 1 offline rerun_report.md.",
)
OFFLINE_RERUN_REPORT_BASELINE_EVAL_DIR_OPTION = typer.Option(
    ...,
    help="Incumbent same-dataset model-eval-v1 output directory.",
)
OFFLINE_RERUN_REPORT_CANDIDATE_EVAL_DIR_OPTION = typer.Option(
    ...,
    help="Candidate same-dataset model-eval-v1 output directory.",
)
OFFLINE_RERUN_REPORT_NO_FAIL_ON_BLOCKED_OPTION = typer.Option(
    False,
    "--no-fail-on-blocked",
    help="Exit 0 even when the Stage 1 rerun report fails.",
)
CHAMPION_PROMOTION_LIVE_STATUS_PATH_OPTION = typer.Option(
    Path("data/xgboost-v4-run-20260523T103814Z/artifacts/live_multimarket_7d_collection_status_latest.json"),
    help="Live collection status JSON used for the seven-day readiness precondition.",
)
CHAMPION_PROMOTION_OFFLINE_RERUN_REPORT_PATH_OPTION = typer.Option(
    None,
    help="Stage 1 rerun_report.md required by champion-promotion.md.",
)
CHAMPION_PROMOTION_BASELINE_EVAL_DIR_OPTION = typer.Option(
    None,
    help="Incumbent same-dataset model-eval-v1 output directory.",
)
CHAMPION_PROMOTION_CANDIDATE_EVAL_DIR_OPTION = typer.Option(
    None,
    help="Candidate same-dataset model-eval-v1 output directory.",
)
CHAMPION_PROMOTION_BASELINE_BACKTEST_SUMMARY_OPTION = typer.Option(
    None,
    help="Incumbent direct backtest summary.json.",
)
CHAMPION_PROMOTION_CANDIDATE_BACKTEST_SUMMARY_OPTION = typer.Option(
    None,
    help="Candidate direct backtest summary.json.",
)
CHAMPION_PROMOTION_SHADOW_EVALUATION_PATH_OPTION = typer.Option(
    None,
    help="Candidate-vs-champion shadow evaluation JSON.",
)
CHAMPION_PROMOTION_SERVING_READINESS_PATH_OPTION = typer.Option(
    None,
    help="Candidate serving readiness JSON.",
)
CHAMPION_PROMOTION_BOOTSTRAP_DECISION_PATH_OPTION = typer.Option(
    None,
    help="Bootstrap decision JSON.",
)
CHAMPION_PROMOTION_CUTOVER_REPORT_PATH_OPTION = typer.Option(
    None,
    help="Cutover JSON evidence generated after champion switch.",
)
CHAMPION_PROMOTION_PROCESS_PATH_OPTION = typer.Option(
    DEFAULT_CHAMPION_PROMOTION_PROCESS_PATH,
    help="User-provided champion-promotion.md attachment path used as promotion source.",
)
CHAMPION_PROMOTION_REPO_RUNBOOK_PATH_OPTION = typer.Option(
    DEFAULT_CHAMPION_PROMOTION_REPO_RUNBOOK_PATH,
    help="Repository mirror of the champion-promotion.md attachment.",
)
CHAMPION_PROMOTION_EXPECTED_CANDIDATE_OPTION = typer.Option(
    "xgboost-v4",
    help="Candidate model version that must pass every gate.",
)
CHAMPION_PROMOTION_EXPECTED_FALLBACK_OPTION = typer.Option(
    "xgboost-v3",
    help="Fallback model version expected in cutover evidence.",
)
CHAMPION_PROMOTION_NO_FAIL_ON_BLOCKED_OPTION = typer.Option(
    False,
    "--no-fail-on-blocked",
    help="Exit 0 even when the audit blocks promotion.",
)
XGBOOST_V4_OBJECTIVE_AUDIT_OUTPUT_PATH_OPTION = typer.Option(
    Path("data/xgboost-v4-run-20260523T103814Z/artifacts/xgboost_v4_objective_audit_latest.json"),
    help="Path for the xgboost-v4 objective completion audit JSON.",
)
XGBOOST_V4_ISSUE_COVERAGE_AUDIT_OUTPUT_PATH_OPTION = typer.Option(
    Path("data/xgboost-v4-run-20260523T103814Z/artifacts/issue_coverage_audit.json"),
    help="Path for the xgboost-v4 issue coverage audit JSON.",
)
DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH = (
    Path.home() / ".codex/automations/xgboost-v4-work-status/automation.toml"
)
DEFAULT_XGBOOST_V4_SLACK_DELIVERY_STATUS_PATH = Path(
    "data/xgboost-v4-run-20260523T103814Z/artifacts/slack_status_delivery_latest.json"
)
DEFAULT_XGBOOST_V4_COLLECTION_RISK_PATH = Path(
    "data/xgboost-v4-run-20260523T103814Z/artifacts/collection_risk_latest.json"
)
DEFAULT_XGBOOST_V4_POST_READINESS_LATEST_PATH = Path(
    "data/xgboost-v4-run-20260523T103814Z/artifacts/xgboost_v4_post_readiness_latest.json"
)
XGBOOST_V4_SLACK_DELIVERY_MAX_AGE_SECONDS = 2 * 60 * 60
XGBOOST_V4_SLACK_DELIVERY_STATUS_OUTPUT_PATH_OPTION = typer.Option(
    DEFAULT_XGBOOST_V4_SLACK_DELIVERY_STATUS_PATH,
    help="Path for the latest Slack delivery attempt JSON.",
)
XGBOOST_V4_SLACK_DELIVERY_CHANNEL_ID_OPTION = typer.Option(
    "C0B5VHYSCN8",
    help="Slack channel id targeted by the delivery attempt.",
)
XGBOOST_V4_SLACK_DELIVERY_ATTEMPTED_AT_OPTION = typer.Option(
    None,
    help="UTC ISO timestamp for the attempt. Defaults to the current UTC time.",
)
XGBOOST_V4_SLACK_DELIVERY_OK_OPTION = typer.Option(
    False,
    "--ok",
    help="Mark the Slack delivery attempt as successful.",
)
XGBOOST_V4_SLACK_DELIVERY_STATUS_OPTION = typer.Option(
    "failed",
    help="Delivery status string, normally 'sent' or 'failed'.",
)
XGBOOST_V4_SLACK_DELIVERY_MESSAGE_LINK_OPTION = typer.Option(
    None,
    help="Slack message link returned by a successful send.",
)
XGBOOST_V4_SLACK_DELIVERY_ERROR_CODE_OPTION = typer.Option(
    None,
    help="Connector/API error code from a failed send, e.g. token_expired.",
)
XGBOOST_V4_SLACK_DELIVERY_ERROR_MESSAGE_OPTION = typer.Option(
    None,
    help="Connector/API error message from a failed send.",
)
XGBOOST_V4_OBJECTIVE_PROMOTION_AUDIT_PATH_OPTION = typer.Option(
    Path("data/model-runs/champion-promotion-audit/champion_promotion_audit.json"),
    help="Champion-promotion audit JSON used as gate evidence.",
)
XGBOOST_V4_OBJECTIVE_CANDIDATE_MODEL_DIR_OPTION = typer.Option(
    None,
    help="Fresh xgboost-v4 model artifact directory after the 7-day retrain.",
)
XGBOOST_V4_OBJECTIVE_FEATURE_ABLATION_PATH_OPTION = typer.Option(
    None,
    help="Feature ablation JSON or Markdown artifact for issue #57.",
)
XGBOOST_V4_OBJECTIVE_STABILITY_REPORT_PATH_OPTION = typer.Option(
    None,
    help="Dataset stability JSON artifact for issue #55.",
)
XGBOOST_V4_OBJECTIVE_DOWN_VALIDATION_PATH_OPTION = typer.Option(
    None,
    help="BUY_DOWN validation or backtest artifact for issue #64.",
)
XGBOOST_V4_OBJECTIVE_SLACK_AUTOMATION_PATH_OPTION = typer.Option(
    DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH,
    help="Heartbeat automation TOML for hourly Slack status updates.",
)
XGBOOST_V4_OBJECTIVE_SLACK_DELIVERY_STATUS_PATH_OPTION = typer.Option(
    None,
    help="JSON artifact recording the latest Slack delivery attempt.",
)
XGBOOST_V4_OBJECTIVE_COLLECTION_RISK_PATH_OPTION = typer.Option(
    None,
    help="Optional JSON artifact recording the latest collection disk-risk snapshot.",
)
XGBOOST_V4_OBJECTIVE_POST_READINESS_LATEST_PATH_OPTION = typer.Option(
    DEFAULT_XGBOOST_V4_POST_READINESS_LATEST_PATH,
    help="Latest post-readiness runner pointer/sentinel JSON for final artifact traceability.",
)
CHAMPION_CUTOVER_REPORT_OUTPUT_PATH_OPTION = typer.Option(
    Path("data/model-runs/champion-cutover/xgboost-v4-cutover.json"),
    help="Path for generated champion cutover JSON evidence.",
)
CHAMPION_STATE_SNAPSHOT_OUTPUT_PATH_OPTION = typer.Option(
    Path("data/model-runs/champion-state/current_champion_snapshot.json"),
    help="Path for current champion/online model snapshot JSON.",
)
CHAMPION_CUTOVER_MODEL_FAMILY_OPTION = typer.Option(
    ACTIVE_MODEL_FAMILY,
    help="Model family whose registry champion should be verified.",
)
CHAMPION_CUTOVER_ENVIRONMENT_OPTION = typer.Option(
    ACTIVE_CHAMPION_ENVIRONMENT,
    help="Deployment environment whose online model should be verified.",
)
CHAMPION_CUTOVER_SMOKE_PATH_OPTION = typer.Option(
    ...,
    help="Inference smoke JSON with passed/model_version/model_path/error_rate/latency evidence.",
)
CHAMPION_CUTOVER_DRIFT_BASELINE_PATH_OPTION = typer.Option(
    ...,
    help="Drift baseline JSON registered for the promoted model.",
)
CHAMPION_CUTOVER_BOOTSTRAP_DECISION_PATH_OPTION = typer.Option(
    ...,
    help="Fresh bootstrap_decision.json used for cutover.",
)
CHAMPION_CUTOVER_SHADOW_EVALUATION_PATH_OPTION = typer.Option(
    ...,
    help="Fresh shadow evaluation JSON used for cutover.",
)
CHAMPION_CUTOVER_SERVING_READINESS_PATH_OPTION = typer.Option(
    ...,
    help="Fresh serving readiness JSON used for cutover.",
)
CHAMPION_CUTOVER_GITHUB_ISSUE_CLOSURES_PATH_OPTION = typer.Option(
    None,
    help="Optional JSON evidence that champion-promotion.md GitHub issues #52/#53 were closed.",
)
DRIFT_BASELINE_OFFLINE_REFERENCE_PATH_OPTION = typer.Option(
    ...,
    help="Candidate offline_reference.json from model-eval-v1 validation split.",
)
DRIFT_BASELINE_OUTPUT_PATH_OPTION = typer.Option(
    Path("data/model-runs/champion-cutover/drift-baseline.json"),
    help="Path for generated champion drift baseline JSON.",
)
LIVE_MONITORING_OUTPUT_PATH_OPTION = typer.Option(
    None,
    help="Optional JSON path for the live monitoring report.",
)
LIVE_MONITORING_RECORD_INCIDENTS_OPTION = typer.Option(
    True,
    "--record-incidents/--no-record-incidents",
    help="Write active drift/label alerts to the incident catalog.",
)
BOOTSTRAP_BASELINE_BACKTEST_SUMMARY_OPTION = typer.Option(
    None,
    help="Optional baseline threshold-backtest summary JSON.",
)
BOOTSTRAP_SERVING_READINESS_PATH_OPTION = typer.Option(
    None,
    help="Optional serving readiness JSON with latency/error evidence.",
)
BOOTSTRAP_FEATURE_SCHEMA_PATH_OPTION = typer.Option(
    None,
    help="Optional candidate feature_schema.json path; defaults to candidate_dir/feature_schema.json.",
)
BOOTSTRAP_MODEL_COMPLEXITY_NOTES_PATH_OPTION = typer.Option(
    None,
    help="Optional model card or complexity-notes Markdown path.",
)
BOOTSTRAP_ROLLBACK_RUNBOOK_PATH_OPTION = typer.Option(
    Path("docs/runbooks/model_rollback.md"),
    help="Rollback/fallback runbook path.",
)
BOOTSTRAP_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/model-runs/bootstrap-champion-v1"),
    help="Directory for first-champion bootstrap decision artifacts.",
)
BOOTSTRAP_REPLACE_CHAMPION_OPTION = typer.Option(
    False,
    "--replace-champion",
    help="Emit PROMOTE_CHAMPION for replacing an existing champion instead of PROMOTE_FIRST_CHAMPION.",
)
PREDICTION_MODEL_PATH_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1/model.json"),
    help="Path to saved XGBoost-v1 model.json.",
)
SERVING_READINESS_FEATURE_SCHEMA_PATH_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1/feature_schema.json"),
    help="Feature schema artifact saved with the model.",
)
SERVING_READINESS_OUTPUT_PATH_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v1/serving_readiness.json"),
    help="Path for serving readiness JSON.",
)
SERVING_READINESS_FALLBACK_MODEL_PATH_OPTION = typer.Option(
    None,
    help="Fallback baseline model artifact used for rollback readiness evidence.",
)
PREDICTION_CALIBRATION_PATH_OPTION = typer.Option(
    None,
    help="Optional path to calibration.json.",
)
PREDICTION_MONITORING_DB_PATH_OPTION = typer.Option(
    DEFAULT_MLOPS_DB_PATH,
    help="MLOps DuckDB path for prediction_events monitoring writes.",
)
PREDICTION_WRITE_MONITORING_EVENTS_OPTION = typer.Option(
    True,
    "--write-monitoring-events/--no-write-monitoring-events",
    help="Write generated predictions to prediction_events for live monitoring.",
)
LABEL_MONITORING_MODEL_VERSION_OPTION = typer.Option(
    ACTIVE_CHAMPION_MODEL_VERSION,
    help="Model version whose prediction_events should receive settled label outcomes.",
)
SHADOW_CHAMPION_MODEL_PATH_OPTION = typer.Option(
    Path("data/model-runs/logreg-baseline-v1/model.json"),
    help="Champion or fallback model artifact to keep serving.",
)
SHADOW_CHALLENGER_MODEL_PATH_OPTION = typer.Option(
    Path("data/model-runs/xgboost-v3/model.json"),
    help="Shadow challenger model artifact.",
)
SHADOW_CHAMPION_CALIBRATION_PATH_OPTION = typer.Option(
    None,
    help="Optional champion calibration.json.",
)
SHADOW_CHALLENGER_CALIBRATION_PATH_OPTION = typer.Option(
    None,
    help="Optional challenger calibration.json.",
)
SHADOW_OUTPUT_PATH_OPTION = typer.Option(
    Path("data/shadow/xgboost-v3-shadow-report.json"),
    help="Path for shadow comparison report JSON.",
)
SHADOW_EVALUATION_OUTPUT_PATH_OPTION = typer.Option(
    None,
    help="Optional path for shadow evaluation Markdown. Defaults beside shadow JSON.",
)
SHADOW_EVALUATION_JSON_OUTPUT_PATH_OPTION = typer.Option(
    None,
    help="Optional path for shadow evaluation JSON. Defaults beside shadow JSON.",
)
SHADOW_OFFLINE_REFERENCE_PATH_OPTION = typer.Option(
    None,
    help="Optional offline validation probability distribution reference JSON.",
)
SHADOW_EDGE_THRESHOLD_OPTION = typer.Option(
    0.30,
    help="Edge threshold for shadow trigger-rate and simulated-PnL evaluation.",
)
ORACLE_BACKTEST_DATASET_DIR_OPTION = typer.Option(
    Path("data/training-datasets/bigan-training-15m-v1"),
    help="Training dataset directory with train.parquet, val.parquet, and test.parquet.",
)
ORACLE_BACKTEST_WAREHOUSE_DIR_OPTION = typer.Option(
    None,
    help="Warehouse root containing raw_top_of_book parquet. Defaults to BIGAN_DATA_DIR/BIGAN_WAREHOUSE_SUBDIR.",
)
ORACLE_BACKTEST_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/backtests/oracle-label-sanity-v1"),
    help="Directory for oracle-label sanity backtest artifacts.",
)
PREDICTION_BACKTEST_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/backtests/predictions-threshold-v1"),
    help="Directory for prediction threshold backtest artifacts.",
)
MODEL_BACKTEST_OUTPUT_DIR_OPTION = typer.Option(
    Path("data/backtests/model-threshold-v1"),
    help="Directory for direct model threshold backtest artifacts.",
)
LIVE_COLLECTION_STATUS_LIVE_ROOT_OPTION = typer.Option(
    None,
    help="Live collection root. Defaults to BIGAN_DATA_DIR.",
)
LIVE_COLLECTION_STATUS_OUTPUT_PATH_OPTION = typer.Option(
    None,
    help="Optional JSON path to write the status snapshot.",
)
LIVE_COLLECTION_STATUS_MANIFEST_PATH_OPTION = typer.Option(
    None,
    help="Optional ETL processed-files manifest path.",
)
LIVE_COLLECTION_STATUS_LOG_DIR_OPTION = typer.Option(
    None,
    help="Optional collector log directory to scan for errors.",
)
LIVE_COLLECTION_STATUS_PATH_OPTION = typer.Option(
    Path(
        "data/xgboost-v4-run-20260523T103814Z/artifacts/"
        "live_multimarket_7d_collection_status_latest.json"
    ),
    help="Live collection status JSON produced by live-collection-status.",
)


def _configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=log_level,
        stream=sys.stderr,
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
    )


def _require_existing_file(name: str, path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise typer.BadParameter(f"{name} does not exist or is not a file: {path}")


def _read_json_file(path: Path) -> Any:
    deadline = time.monotonic() + 2.0
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            if time.monotonic() >= deadline:
                raise typer.BadParameter(f"invalid JSON file: {path}") from exc
            time.sleep(0.05)


def _write_json_file_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _first_optional_float(payload: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _optional_float(payload.get(name))
        if value is not None:
            return value
    return None


def _path_value_matches(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return False
    try:
        return Path(str(actual)).expanduser().resolve(strict=False) == Path(str(expected)).expanduser().resolve(
            strict=False
        )
    except (OSError, RuntimeError, ValueError):
        return str(actual) == str(expected)


def _validate_cutover_smoke_payload(
    smoke: Any,
    champion: dict[str, Any],
    *,
    smoke_path: Path,
) -> dict[str, Any]:
    if not isinstance(smoke, dict):
        raise typer.BadParameter(f"cutover smoke JSON must be an object: {smoke_path}")

    expected_model_version = str(champion.get("model_version") or "")
    expected_model_path = champion.get("artifact_uri")
    expected_calibration_path = champion.get("calibration_artifact_uri")
    smoke_model_path = smoke.get("model_path") or smoke.get("artifact_uri")
    smoke_calibration_path = smoke.get("calibration_path") or smoke.get("calibration_artifact_uri")
    error_rate = _optional_float(smoke.get("error_rate"))
    latency_ms = _first_optional_float(smoke, "serving_latency_ms", "p95_latency_ms", "latency_p95_ms")

    failures = [
        name
        for name, passed in {
            "passed": smoke.get("passed") is True,
            "model_version": str(smoke.get("model_version") or "") == expected_model_version,
            "model_path": _path_value_matches(smoke_model_path, expected_model_path),
            "calibration_path": (
                True
                if expected_calibration_path is None
                else _path_value_matches(smoke_calibration_path, expected_calibration_path)
            ),
            "error_rate": error_rate == 0.0,
            "latency_ms": latency_ms is not None and latency_ms < 50.0,
        }.items()
        if not passed
    ]
    if failures:
        detail = (
            f"failed={','.join(failures)}; "
            f"smoke_model={smoke.get('model_version')}, expected_model={expected_model_version}, "
            f"smoke_model_path={smoke_model_path}, expected_model_path={expected_model_path}, "
            f"smoke_calibration_path={smoke_calibration_path}, "
            f"expected_calibration_path={expected_calibration_path}, "
            f"error_rate={error_rate}, latency_ms={latency_ms}"
        )
        raise typer.BadParameter(f"cutover smoke does not match current champion: {detail}")
    return smoke


def _validate_cutover_evidence_payloads(
    *,
    bootstrap_path: Path,
    shadow_path: Path,
    serving_path: Path,
) -> None:
    bootstrap = _read_json_file(bootstrap_path)
    shadow = _read_json_file(shadow_path)
    serving = _read_json_file(serving_path)
    bootstrap_promotes = (
        isinstance(bootstrap, dict)
        and bootstrap.get("recommended_action") == "PROMOTE_CHAMPION"
    )
    shadow_passed = isinstance(shadow, dict) and (
        shadow.get("overall_passed") is True or shadow.get("passed") is True
    )
    serving_ready = isinstance(serving, dict) and (
        serving.get("ready") is True or serving.get("serving_ready") is True
    )
    failures = [
        name
        for name, passed in {
            "bootstrap_promotes": bootstrap_promotes,
            "shadow_passed": shadow_passed,
            "serving_ready": serving_ready,
        }.items()
        if not passed
    ]
    if failures:
        detail = (
            f"failed={','.join(failures)}; "
            f"bootstrap_action={bootstrap.get('recommended_action') if isinstance(bootstrap, dict) else None}, "
            f"shadow_overall_passed={shadow.get('overall_passed') if isinstance(shadow, dict) else None}, "
            f"shadow_passed={shadow.get('passed') if isinstance(shadow, dict) else None}, "
            f"serving_ready={serving.get('ready') if isinstance(serving, dict) else None}, "
            f"serving_serving_ready={serving.get('serving_ready') if isinstance(serving, dict) else None}"
        )
        raise typer.BadParameter(f"cutover evidence is not ready for Stage 5: {detail}")


def _cutover_github_issue_rows(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        rows = payload.get("closures")
        if not isinstance(rows, list):
            rows = payload.get("issues")
    else:
        rows = payload
    return rows if isinstance(rows, list) else []


def _cutover_github_issue_number(row: dict[str, Any]) -> int | None:
    for key in ("issue", "issue_number", "number"):
        raw = row.get(key)
        try:
            return int(str(raw).lstrip("#"))
        except (TypeError, ValueError):
            continue
    return None


def _cutover_github_issue_comment(row: dict[str, Any]) -> str:
    for key in ("comment", "closure_comment", "comment_body", "close_comment"):
        value = row.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def _validate_cutover_github_issue_closures(
    payload: Any,
    *,
    expected_candidate_model_version: str,
    github_issue_closures_path: Path,
) -> Any:
    rows = _cutover_github_issue_rows(payload)
    by_issue: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        issue = _cutover_github_issue_number(row)
        if issue is not None:
            by_issue[issue] = row

    failures: list[str] = []
    for issue in REQUIRED_CUTOVER_GITHUB_ISSUES:
        row = by_issue.get(issue)
        if not isinstance(row, dict):
            failures.append(f"#{issue}: missing")
            continue
        state = row.get("state")
        state = state.strip().lower() if isinstance(state, str) else None
        repo = row.get("repo")
        repo = repo.strip() if isinstance(repo, str) else None
        comment = _cutover_github_issue_comment(row)
        normalized_comment = comment.lower()
        comment_passed = (
            ("shadow pass" in normalized_comment and "promote_champion" in normalized_comment)
            if issue == 52
            else (
                "cutover complete" in normalized_comment
                and expected_candidate_model_version.lower() in normalized_comment
            )
        )
        if state != "closed" or repo != EXPECTED_CUTOVER_GITHUB_REPO or not comment_passed:
            failures.append(
                f"#{issue}: state={state}, repo={repo}, comment={comment}"
            )
    if failures:
        detail = "; ".join(failures)
        raise typer.BadParameter(
            "github issue closure evidence is not ready for Stage 5: "
            f"path={github_issue_closures_path}; {detail}"
        )
    return payload


def _screen_session_state(screen_session: str | None) -> str | None:
    if screen_session is None:
        return None
    try:
        result = subprocess.run(
            ["screen", "-ls"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    text = f"{result.stdout}\n{result.stderr}"
    return "running" if screen_session in text else "not_found"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@app.command("serve")
def serve() -> None:
    """Run the WebSocket ingestion service (long-running)."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    runner = IngestionRunner(settings)
    asyncio.run(runner.serve())


@app.command("smoke")
def smoke(seconds: int = typer.Option(30, help="How long to run before exiting.")) -> None:
    """Run for ``seconds`` then exit cleanly. Used for live smoke tests."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    runner = IngestionRunner(settings)

    async def main() -> None:
        task = asyncio.create_task(runner.serve())
        await asyncio.sleep(seconds)
        runner.stop()
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(main())


@app.command("soak")
def soak(
    seconds: int = typer.Option(
        86_400,
        help="How long to run ingestion before producing the soak summary.",
    ),
    sample_interval_seconds: float = typer.Option(
        60.0,
        help="How often to append in-process Prometheus metric samples.",
    ),
    output_dir: Path = SOAK_OUTPUT_DIR_OPTION,
    min_duration_seconds: float | None = typer.Option(
        None,
        help="Minimum observed duration required to pass. Defaults to --seconds.",
    ),
    max_reconnects: float = typer.Option(
        24.0,
        help="Maximum allowed WebSocket reconnects during the run.",
    ),
    max_last_event_lag_seconds: float = typer.Option(
        60.0,
        help="Maximum allowed lag of bigan_last_event_receive_time_seconds.",
    ),
    max_hash_mismatches: float = typer.Option(
        0.0,
        help="Maximum allowed WebSocket book hash mismatches.",
    ),
    max_rss_growth_mb: float = typer.Option(
        256.0,
        help="Maximum allowed increase in process max RSS.",
    ),
    final_rollup: bool = typer.Option(
        True,
        help="Run one final NDJSON-to-Parquet rollup after stopping ingestion.",
    ),
    market_coverage: bool = typer.Option(
        True,
        "--market-coverage/--no-market-coverage",
        help="Run Gamma/CLOB REST coverage verification after stopping ingestion.",
    ),
    coverage_max_stale_seconds: float | None = typer.Option(
        None,
        help=(
            "Optional per-asset raw event freshness threshold for market coverage. "
            "Defaults to disabled for completed soak runs."
        ),
    ),
    coverage_require_hash_match: bool = typer.Option(
        False,
        help=(
            "Require latest raw WS book hashes to match CLOB REST during coverage "
            "verification."
        ),
    ),
    coverage_raw_end_grace_seconds: float = typer.Option(
        120.0,
        help="Grace window for ignoring markets opened after the raw archive ended.",
    ),
    coverage_rest_concurrency: int = typer.Option(
        12,
        min=1,
        help="Maximum concurrent CLOB REST /book requests for coverage verification.",
    ),
) -> None:
    """Run ingestion for a soak window and write validation evidence.

    The default duration is 24h for issue #25. For a local proof pass, use a
    shorter ``--seconds`` plus matching ``--min-duration-seconds``.
    """

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    started_label = _soak_timestamp_label()
    samples_path = output_dir / f"soak-{started_label}.ndjson"
    summary_path = output_dir / f"soak-{started_label}-summary.json"
    thresholds = SoakThresholds(
        min_duration_seconds=(
            float(seconds)
            if min_duration_seconds is None
            else min_duration_seconds
        ),
        max_reconnects=max_reconnects,
        max_last_event_lag_seconds=max_last_event_lag_seconds,
        max_hash_mismatches=max_hash_mismatches,
        max_rss_growth_mb=max_rss_growth_mb,
    )

    async def main() -> dict:
        runner = IngestionRunner(settings)
        started_at = asyncio.get_running_loop().time()
        wall_started_at = _now_seconds()
        stop_samples = asyncio.Event()
        serve_task = asyncio.create_task(runner.serve(), name="soak-serve")
        sampler_task = asyncio.create_task(
            record_soak_samples(
                samples_path,
                started_at_seconds=wall_started_at,
                interval_seconds=sample_interval_seconds,
                stop_event=stop_samples,
            ),
            name="soak-sampler",
        )
        fatal_exit: str | None = None
        try:
            while True:
                elapsed = asyncio.get_running_loop().time() - started_at
                remaining = seconds - elapsed
                if remaining <= 0:
                    break
                done, _ = await asyncio.wait(
                    {serve_task},
                    timeout=min(5.0, remaining),
                )
                if serve_task in done:
                    exc = serve_task.exception()
                    fatal_exit = repr(exc) if exc is not None else "serve exited early"
                    break
        finally:
            runner.stop()
            stop_samples.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(sampler_task, timeout=10.0)
            if not serve_task.done():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(serve_task, timeout=30.0)
            if not serve_task.done():
                serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await serve_task

        samples = read_soak_samples(samples_path)
        final_rollup_result = (
            finalize_soak_rollup(settings.raw_dir, settings.rollup_dir)
            if final_rollup
            else {"files": 0, "records": 0, "errors": []}
        )
        market_coverage_result = (
            await _run_market_coverage_check(
                settings=settings,
                raw_dir=settings.raw_dir,
                max_stale_seconds=coverage_max_stale_seconds,
                require_hash_match=coverage_require_hash_match,
                ignore_markets_opened_after_raw_end=True,
                raw_end_grace_seconds=coverage_raw_end_grace_seconds,
                rest_concurrency=coverage_rest_concurrency,
                max_examples=20,
            )
            if market_coverage
            else None
        )
        summary = summarize_soak(
            samples,
            raw_dir=settings.raw_dir,
            rollup_dir=settings.rollup_dir,
            thresholds=thresholds,
            fatal_exit=fatal_exit,
            market_coverage=market_coverage_result,
        )
        summary["final_rollup"] = final_rollup_result
        write_soak_summary(summary_path, summary)
        summary["samples_path"] = str(samples_path)
        summary["summary_path"] = str(summary_path)
        return summary

    summary = asyncio.run(main())
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise Exit(code=1)


@app.command("soak-report")
def soak_report(
    samples_path: Path = SOAK_SAMPLES_PATH_OPTION,
    raw_dir: Path | None = SOAK_RAW_DIR_OPTION,
    rollup_dir: Path | None = SOAK_ROLLUP_DIR_OPTION,
    summary_path: Path | None = SOAK_SUMMARY_PATH_OPTION,
    min_duration_seconds: float = typer.Option(
        86_400.0,
        help="Minimum observed duration required to pass.",
    ),
    max_reconnects: float = typer.Option(24.0),
    max_last_event_lag_seconds: float = typer.Option(60.0),
    max_hash_mismatches: float = typer.Option(0.0),
    max_rss_growth_mb: float = typer.Option(256.0),
    market_coverage: bool = typer.Option(
        False,
        "--market-coverage/--no-market-coverage",
        help="Also run Gamma/CLOB REST coverage verification for this raw archive.",
    ),
    coverage_max_stale_seconds: float | None = typer.Option(
        None,
        help=(
            "Optional per-asset raw event freshness threshold for market coverage. "
            "Defaults to disabled for completed soak runs."
        ),
    ),
    coverage_require_hash_match: bool = typer.Option(
        False,
        help="Require latest raw WS book hashes to match CLOB REST.",
    ),
    coverage_raw_end_grace_seconds: float = typer.Option(
        120.0,
        help="Grace window for ignoring markets opened after the raw archive ended.",
    ),
    coverage_rest_concurrency: int = typer.Option(
        12,
        min=1,
        help="Maximum concurrent CLOB REST /book requests for coverage verification.",
    ),
) -> None:
    """Validate a soak sample file after an operational run."""

    settings = IngestionSettings()
    thresholds = SoakThresholds(
        min_duration_seconds=min_duration_seconds,
        max_reconnects=max_reconnects,
        max_last_event_lag_seconds=max_last_event_lag_seconds,
        max_hash_mismatches=max_hash_mismatches,
        max_rss_growth_mb=max_rss_growth_mb,
    )
    report_raw_dir = settings.raw_dir if raw_dir is None else raw_dir
    market_coverage_result = (
        asyncio.run(
            _run_market_coverage_check(
                settings=settings,
                raw_dir=report_raw_dir,
                max_stale_seconds=coverage_max_stale_seconds,
                require_hash_match=coverage_require_hash_match,
                ignore_markets_opened_after_raw_end=True,
                raw_end_grace_seconds=coverage_raw_end_grace_seconds,
                rest_concurrency=coverage_rest_concurrency,
                max_examples=20,
            )
        )
        if market_coverage
        else None
    )
    summary = summarize_soak(
        read_soak_samples(samples_path),
        raw_dir=report_raw_dir,
        rollup_dir=settings.rollup_dir if rollup_dir is None else rollup_dir,
        thresholds=thresholds,
        market_coverage=market_coverage_result,
    )
    if summary_path is not None:
        write_soak_summary(summary_path, summary)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise Exit(code=1)


@app.command("market-coverage-report")
def market_coverage_report(
    raw_dir: Path | None = SOAK_RAW_DIR_OPTION,
    summary_path: Path | None = SOAK_SUMMARY_PATH_OPTION,
    max_stale_seconds: float = typer.Option(
        120.0,
        help="Fail if an expected token has no raw event within this many seconds.",
    ),
    disable_stale_check: bool = typer.Option(
        False,
        help="Skip freshness checks; useful when reporting on a completed short soak.",
    ),
    require_hash_match: bool = typer.Option(
        False,
        help="Fail if latest raw wire hash differs from current REST book hash.",
    ),
    ignore_markets_opened_after_raw_end: bool = typer.Option(
        False,
        help="Ignore Gamma markets created after the latest raw receive_time.",
    ),
    raw_end_grace_seconds: float = typer.Option(
        120.0,
        help="Grace window for --ignore-markets-opened-after-raw-end.",
    ),
    rest_concurrency: int = typer.Option(
        12,
        min=1,
        help="Maximum concurrent CLOB REST /book requests.",
    ),
    max_examples: int = typer.Option(
        20,
        min=1,
        help="Maximum example assets included for each failed bucket.",
    ),
) -> None:
    """Compare Gamma active markets with raw WS coverage and CLOB REST books."""

    settings = IngestionSettings()
    report_raw_dir = settings.raw_dir if raw_dir is None else raw_dir

    async def main() -> dict:
        return await _run_market_coverage_check(
            settings=settings,
            raw_dir=report_raw_dir,
            max_stale_seconds=None if disable_stale_check else max_stale_seconds,
            require_hash_match=require_hash_match,
            ignore_markets_opened_after_raw_end=ignore_markets_opened_after_raw_end,
            raw_end_grace_seconds=raw_end_grace_seconds,
            rest_concurrency=rest_concurrency,
            max_examples=max_examples,
        )

    report = asyncio.run(main())
    if summary_path is not None:
        write_soak_summary(summary_path, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise Exit(code=1)


@app.command("reference-prices")
def reference_prices(
    symbol_mapping_path: Path | None = SYMBOL_MAPPING_PATH_OPTION,
    max_rows_per_partition: int = typer.Option(
        1,
        help="Flush reference-price warehouse partitions after this many rows.",
    ),
) -> None:
    """Run Coinbase, Kraken, and Chainlink BTC/USD reference-price readers."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    if not settings.chainlink_rpc_url:
        raise typer.BadParameter(
            "BIGAN_CHAINLINK_RPC_URL must be set to run the Chainlink reader"
        )

    async def main() -> None:
        if settings.metrics_enabled:
            start_http_server(settings.metrics_port, registry=REGISTRY)
        symbol_mapper = (
            SymbolMapper.from_path(symbol_mapping_path)
            if symbol_mapping_path is not None
            else None
        )
        sink = WarehousePriceSink(
            settings.warehouse_dir,
            symbol_mapper=symbol_mapper,
            max_rows_per_partition=max_rows_per_partition,
        )
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(
                    CoinbaseTickerReader(
                        WsPriceReaderConfig(
                            url=settings.coinbase_ws_url,
                            symbol=settings.coinbase_product_id,
                            reconnect_min_seconds=settings.price_reader_reconnect_min_seconds,
                            reconnect_max_seconds=settings.price_reader_reconnect_max_seconds,
                        ),
                        sink,
                    ).run(),
                    name="coinbase-ticker",
                )
                tg.create_task(
                    KrakenTickerReader(
                        WsPriceReaderConfig(
                            url=settings.kraken_ws_url,
                            symbol=settings.kraken_symbol,
                            reconnect_min_seconds=settings.price_reader_reconnect_min_seconds,
                            reconnect_max_seconds=settings.price_reader_reconnect_max_seconds,
                        ),
                        sink,
                    ).run(),
                    name="kraken-ticker",
                )
                tg.create_task(
                    ChainlinkOracleReader(
                        ChainlinkReaderConfig(
                            rpc_url=settings.chainlink_rpc_url,
                            feed_address=settings.chainlink_feed_address,
                            symbol=settings.chainlink_symbol,
                            poll_interval_seconds=settings.chainlink_poll_interval_seconds,
                            request_timeout_seconds=settings.chainlink_request_timeout_seconds,
                        ),
                        sink,
                    ).run(),
                    name="chainlink-oracle",
                )
        finally:
            await sink.close()

    asyncio.run(main())


@app.command("etl-batch")
def etl_batch(
    lag_seconds: float = typer.Option(
        60.0, help="Skip NDJSON.gz files whose mtime is within this many seconds."
    ),
    max_rows_per_partition: int = typer.Option(
        50_000, help="Flush a partition buffer when it exceeds this size."
    ),
    symbol_mapping_path: Path | None = SYMBOL_MAPPING_PATH_OPTION,
    processed_manifest_path: Path | None = PROCESSED_MANIFEST_PATH_OPTION,
    timestamp_future_grace_seconds: float | None = TIMESTAMP_FUTURE_GRACE_SECONDS_OPTION,
    timestamp_stale_threshold_seconds: float | None = TIMESTAMP_STALE_THRESHOLD_SECONDS_OPTION,
) -> None:
    """Convert raw NDJSON archive into the canonical Parquet warehouse."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_etl_batch(
        raw_dir=settings.raw_dir,
        warehouse_dir=settings.warehouse_dir,
        lag_seconds=lag_seconds,
        max_rows_per_partition=max_rows_per_partition,
        symbol_mapping_path=symbol_mapping_path,
        processed_manifest_path=processed_manifest_path,
        timestamp_future_grace_seconds=(
            settings.timestamp_future_grace_seconds
            if timestamp_future_grace_seconds is None
            else timestamp_future_grace_seconds
        ),
        timestamp_stale_threshold_seconds=(
            settings.timestamp_stale_threshold_seconds
            if timestamp_stale_threshold_seconds is None
            else timestamp_stale_threshold_seconds
        ),
    )
    typer.echo(
        json.dumps(
            {
                "files_processed": report.files_processed,
                "records_read": report.records_read,
                "rows_per_table": report.rows_per_table,
                "quarantined_by_rule": report.quarantined_by_rule,
                "quarantined_total": report.quarantined_total,
                "cross_batch_duplicates_skipped": report.cross_batch_duplicates_skipped,
            },
            indent=2,
        )
    )


@app.command("warehouse-stats")
def warehouse_stats() -> None:
    """Print row counts for each canonical table via DuckDB."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    summary = warehouse_summary(settings.warehouse_dir)
    typer.echo(json.dumps(summary, indent=2))


@app.command("live-collection-status")
def live_collection_status(
    live_root: Path | None = LIVE_COLLECTION_STATUS_LIVE_ROOT_OPTION,
    output_path: Path | None = LIVE_COLLECTION_STATUS_OUTPUT_PATH_OPTION,
    manifest_path: Path | None = LIVE_COLLECTION_STATUS_MANIFEST_PATH_OPTION,
    log_dir: Path | None = LIVE_COLLECTION_STATUS_LOG_DIR_OPTION,
    monitoring_db_path: Path | None = PREDICTION_MONITORING_DB_PATH_OPTION,
    monitoring_model_version: str | None = LABEL_MONITORING_MODEL_VERSION_OPTION,
    screen_session: str | None = typer.Option(
        None,
        help="Optional screen session name for liveness evidence.",
    ),
    lookback_minutes: float | None = typer.Option(
        None,
        help="Configured live scoring lookback, included as metadata.",
    ),
    labels_enabled: bool = typer.Option(
        False,
        "--labels-enabled/--labels-disabled",
        help="Whether the tight live scan loop has label refreshes enabled.",
    ),
    gzip_check_limit: int = typer.Option(
        20,
        help="Number of latest published raw gzip segments to validate.",
    ),
    max_progress_staleness_seconds: float | None = typer.Option(
        None,
        help=(
            "Block readiness if latest raw or processed segment is older than this many "
            "seconds. Defaults to lookback_minutes * 60 when lookback_minutes is set."
        ),
    ),
) -> None:
    """Build a JSON status snapshot for a long-running live collection."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    resolved_live_root = settings.data_dir if live_root is None else live_root
    resolved_progress_staleness_seconds = (
        max_progress_staleness_seconds
        if max_progress_staleness_seconds is not None
        else (lookback_minutes * 60.0 if lookback_minutes is not None else None)
    )
    status = build_live_collection_status(
        live_root=resolved_live_root,
        manifest_path=manifest_path,
        log_dir=log_dir,
        monitoring_db_path=monitoring_db_path,
        monitoring_model_version=monitoring_model_version,
        screen_session=screen_session,
        screen_state=_screen_session_state(screen_session),
        settings={
            "labels_enabled": labels_enabled,
            "lookback_minutes": lookback_minutes,
            "max_progress_staleness_seconds": resolved_progress_staleness_seconds,
        },
        gzip_check_limit=gzip_check_limit,
        max_progress_staleness_seconds=resolved_progress_staleness_seconds,
    )
    if output_path is not None:
        write_live_collection_status(output_path, status)
    typer.echo(json.dumps(status, indent=2))


@app.command("live-collection-readiness")
def live_collection_readiness(
    status_path: Path = LIVE_COLLECTION_STATUS_PATH_OPTION,
    fail_on_blocked: bool = typer.Option(
        True,
        "--fail-on-blocked/--no-fail-on-blocked",
        help="Exit with status 1 when the 7-day corpus gate is blocked.",
    ),
) -> None:
    """Check whether the live 7-day corpus is ready for retraining."""

    decision = live_collection_readiness_decision(read_live_collection_status(status_path))
    typer.echo(json.dumps(decision, indent=2))
    if fail_on_blocked and not decision["ready"]:
        raise Exit(1)


@app.command("features-15m-v1")
def features_15m_v1(
    max_rows_per_partition: int = typer.Option(
        50_000,
        help="Flush feature partitions after this many rows.",
    ),
    lookback_minutes: float | None = typer.Option(
        None,
        help="Only generate features newer than now minus this many minutes.",
    ),
    since_ms: int | None = typer.Option(
        None,
        help="Only write features with feature_ts >= this UTC ms timestamp.",
    ),
    until_ms: int | None = typer.Option(
        None,
        help="Only write features with feature_ts < this UTC ms timestamp.",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing/--replace-existing",
        help="Do not append feature rows whose source/source_symbol/feature_ts/version already exist.",
    ),
) -> None:
    """Generate minute-grain features_15m_v1 rows from canonical raw tables."""
    if lookback_minutes is not None and lookback_minutes <= 0:
        raise typer.BadParameter("--lookback-minutes must be positive")
    if lookback_minutes is not None and since_ms is not None:
        raise typer.BadParameter("pass either --lookback-minutes or --since-ms, not both")
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    lower_bound_ms = (
        int(time.time() * 1000 - lookback_minutes * 60_000)
        if lookback_minutes is not None
        else since_ms
    )
    report = run_feature_batch(
        settings.warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
        since_ms=lower_bound_ms,
        until_ms=until_ms,
        skip_existing=skip_existing,
    )
    typer.echo(
        json.dumps(
            {
                "feature_version": report.feature_version,
                "rows_generated": report.rows_generated,
                "rows_written": report.rows_written,
            },
            indent=2,
        )
    )


@app.command("feature-quality-report")
def feature_quality_report() -> None:
    """Run SQL quality checks against generated features_15m_v1 rows."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_feature_quality_sql_checks(settings.warehouse_dir)
    typer.echo(json.dumps(report.to_dict(), indent=2))
    if not report.passed:
        raise Exit(code=1)


@app.command("labels-15m-v1")
def labels_15m_v1(
    max_rows_per_partition: int = typer.Option(
        50_000,
        help="Flush label partitions after this many rows.",
    ),
    lookback_minutes: float | None = typer.Option(
        None,
        help="Only label features newer than now minus this many minutes.",
    ),
    since_ms: int | None = typer.Option(
        None,
        help="Only label features with feature_ts >= this UTC ms timestamp.",
    ),
    until_ms: int | None = typer.Option(
        None,
        help="Only label features with feature_ts < this UTC ms timestamp.",
    ),
    fee_bps: float = typer.Option(
        0.0,
        help="Entry fee assumption, in basis points, for profitability labels.",
    ),
    request_timeout_seconds: float = typer.Option(
        10.0,
        help="Per-request timeout when fetching Polymarket round metadata.",
    ),
    request_concurrency: int = typer.Option(
        8,
        help="Concurrent Gamma round metadata requests for label generation.",
    ),
    monitoring_db_path: Path | None = PREDICTION_MONITORING_DB_PATH_OPTION,
    monitoring_model_version: str | None = LABEL_MONITORING_MODEL_VERSION_OPTION,
    write_monitoring_outcomes: bool = typer.Option(
        True,
        "--write-monitoring-outcomes/--no-write-monitoring-outcomes",
        help="Write matched labels to prediction_outcomes for live monitoring.",
    ),
    skip_existing_labels: bool = typer.Option(
        False,
        "--skip-existing-labels/--no-skip-existing-labels",
        help="Skip label rows already present for the same feature, version, and label kind.",
    ),
) -> None:
    """Generate independent UP-token profitability labels_15m_v1 rows."""
    if lookback_minutes is not None and lookback_minutes <= 0:
        raise typer.BadParameter("--lookback-minutes must be positive")
    if lookback_minutes is not None and since_ms is not None:
        raise typer.BadParameter("pass either --lookback-minutes or --since-ms, not both")
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    lower_bound_ms = (
        int(time.time() * 1000 - lookback_minutes * 60_000)
        if lookback_minutes is not None
        else since_ms
    )
    report = run_label_batch(
        settings.warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
        fee_bps=fee_bps,
        gamma_api_base=settings.gamma_api_base,
        market_slug_prefix=settings.market_slug_prefix,
        request_timeout_seconds=request_timeout_seconds,
        request_concurrency=request_concurrency,
        monitoring_db_path=monitoring_db_path if write_monitoring_outcomes else None,
        monitoring_model_version=monitoring_model_version if write_monitoring_outcomes else None,
        skip_existing_labels=skip_existing_labels,
        since_ms=lower_bound_ms,
        until_ms=until_ms,
    )
    typer.echo(
        json.dumps(
            {
                "label_version": report.label_version,
                "rows_generated": report.rows_generated,
                "rows_written": report.rows_written,
                "monitoring_outcomes_written": report.monitoring_outcomes_written,
                "fee_bps": fee_bps,
            },
            indent=2,
        )
    )


@app.command("training-dataset-v1")
def training_dataset_v1(
    output_dir: Path = TRAINING_DATASET_OUTPUT_DIR_OPTION,
    min_completeness_score: float = typer.Option(
        0.80,
        help="Minimum feature completeness_score accepted for training samples.",
    ),
    train_fraction: float = typer.Option(
        0.60,
        help="Oldest fraction of rows assigned to train.",
    ),
    val_fraction: float = typer.Option(
        0.20,
        help="Next fraction of rows assigned to validation; the remainder is test.",
    ),
    outcome_side: str = typer.Option(
        "UP",
        help="Outcome side to include in the assembled dataset: UP, DOWN, or ANY.",
    ),
) -> None:
    """Assemble train/val/test samples from features_15m_v1 and labels_15m_v1."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = assemble_training_dataset(
        settings.warehouse_dir,
        output_dir,
        split_config=SplitConfig(
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        ),
        min_completeness_score=min_completeness_score,
        outcome_side=outcome_side,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("logistic-baseline-v1")
def logistic_baseline_v1(
    dataset_dir: Path = LOGISTIC_DATASET_DIR_OPTION,
    output_dir: Path = LOGISTIC_OUTPUT_DIR_OPTION,
    epochs: int = typer.Option(500, help="Number of full-batch gradient descent epochs."),
    learning_rate: float = typer.Option(0.10, help="Full-batch gradient descent learning rate."),
    l2_penalty: float = typer.Option(0.0, help="L2 coefficient penalty."),
) -> None:
    """Train deterministic logistic regression baseline artifacts."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = train_logistic_baseline(
        dataset_dir,
        output_dir,
        config=LogisticBaselineConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            l2_penalty=l2_penalty,
        ),
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("xgboost-v1")
def xgboost_v1(
    dataset_dir: Path = XGBOOST_DATASET_DIR_OPTION,
    output_dir: Path = XGBOOST_OUTPUT_DIR_OPTION,
    rounds_grid: str = typer.Option("100,200,300", help="Comma-separated boosting-round grid."),
    learning_rate_grid: str = typer.Option("0.01,0.05,0.10", help="Comma-separated learning-rate grid."),
    l2_penalty_grid: str = typer.Option("0.10,1.0,5.0", help="Comma-separated L2 penalty grid."),
    max_depth_grid: str = typer.Option("3,4,5", help="Comma-separated max-depth grid."),
    subsample_grid: str = typer.Option("0.70,0.80,1.0", help="Comma-separated row-subsample grid."),
    colsample_bytree_grid: str = typer.Option(
        "0.70,0.80,1.0",
        help="Comma-separated column-subsample grid.",
    ),
) -> None:
    """Train deterministic XGBoost-v1 candidate artifacts."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = train_xgboost_v1(
        dataset_dir,
        output_dir,
        config=XGBoostV1Config(
            rounds_grid=_parse_int_grid(rounds_grid),
            learning_rate_grid=_parse_float_grid(learning_rate_grid),
            l2_penalty_grid=_parse_float_grid(l2_penalty_grid),
            max_depth_grid=_parse_int_grid(max_depth_grid),
            subsample_grid=_parse_float_grid(subsample_grid),
            colsample_bytree_grid=_parse_float_grid(colsample_bytree_grid),
        ),
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("xgboost-v2")
def xgboost_v2(
    dataset_dir: Path = XGBOOST_DATASET_DIR_OPTION,
    output_dir: Path = XGBOOST_V2_OUTPUT_DIR_OPTION,
) -> None:
    """Train conservative XGBoost-v2 candidate artifacts."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = train_xgboost_v2(dataset_dir, output_dir)
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("xgboost-v3")
def xgboost_v3(
    dataset_dir: Path = XGBOOST_DATASET_DIR_OPTION,
    output_dir: Path = XGBOOST_V3_OUTPUT_DIR_OPTION,
) -> None:
    """Train conservative XGBoost-v3 artifacts focused on validation Brier."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = train_xgboost_v3(dataset_dir, output_dir)
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("xgboost-v4")
def xgboost_v4(
    dataset_dir: Path = XGBOOST_DATASET_DIR_OPTION,
    output_dir: Path = XGBOOST_V4_OUTPUT_DIR_OPTION,
    ensemble_seeds: str = typer.Option(
        "0,17,42",
        help="Comma-separated random seeds for the light v4 ensemble.",
    ),
) -> None:
    """Train xgboost-v4 artifacts with time-series CV and light ensembling."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = train_xgboost_v4(
        dataset_dir,
        output_dir,
        ensemble_seeds=tuple(_parse_int_grid(ensemble_seeds)),
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("calibration-v1")
def calibration_v1(
    model_path: Path = CALIBRATION_MODEL_PATH_OPTION,
    dataset_dir: Path = CALIBRATION_DATASET_DIR_OPTION,
    output_dir: Path = CALIBRATION_OUTPUT_DIR_OPTION,
) -> None:
    """Fit probability calibration for a saved XGBoost-v1 model."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = fit_probability_calibration(model_path, dataset_dir, output_dir)
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("model-eval-v1")
def model_eval_v1(
    model_path: Path = MODEL_EVAL_MODEL_PATH_OPTION,
    dataset_dir: Path = MODEL_EVAL_DATASET_DIR_OPTION,
    output_dir: Path = MODEL_EVAL_OUTPUT_DIR_OPTION,
    calibration_path: Path | None = MODEL_EVAL_CALIBRATION_PATH_OPTION,
) -> None:
    """Evaluate a saved probability model on one fixed train/val/test dataset."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = evaluate_probability_model_on_dataset(
        model_path,
        dataset_dir,
        output_dir,
        calibration_path=calibration_path,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("dataset-stability-report-v1")
def dataset_stability_report_v1(
    dataset_dir: Path = MODEL_EVAL_DATASET_DIR_OPTION,
    output_dir: Path = DATASET_STABILITY_OUTPUT_DIR_OPTION,
) -> None:
    """Generate label and core-feature distribution evidence for issue #55."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = generate_dataset_stability_report(dataset_dir, output_dir)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("feature-ablation-report-v1")
def feature_ablation_report_v1(
    model_path: Path = MODEL_EVAL_MODEL_PATH_OPTION,
    dataset_dir: Path = MODEL_EVAL_DATASET_DIR_OPTION,
    output_dir: Path = FEATURE_ABLATION_OUTPUT_DIR_OPTION,
    calibration_path: Path | None = MODEL_EVAL_CALIBRATION_PATH_OPTION,
    split: str = FEATURE_ABLATION_SPLIT_OPTION,
) -> None:
    """Generate per-feature and grouped ablation evidence for issue #57."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = generate_feature_ablation_report(
        model_path,
        dataset_dir,
        output_dir,
        calibration_path=calibration_path,
        split=split,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("promotion-report-v1")
def promotion_report_v1(
    baseline_dir: Path = PROMOTION_BASELINE_DIR_OPTION,
    candidate_dir: Path = PROMOTION_CANDIDATE_DIR_OPTION,
    calibration_dir: Path = PROMOTION_CALIBRATION_DIR_OPTION,
    backtest_summary_path: Path = PROMOTION_BACKTEST_SUMMARY_OPTION,
    output_dir: Path = PROMOTION_OUTPUT_DIR_OPTION,
) -> None:
    """Evaluate model promotion rules and write a checklist/report."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = evaluate_model_promotion(
        baseline_dir,
        candidate_dir,
        calibration_dir,
        backtest_summary_path,
        output_dir,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("offline-rerun-report-v1")
def offline_rerun_report_v1(
    baseline_eval_dir: Path = OFFLINE_RERUN_REPORT_BASELINE_EVAL_DIR_OPTION,
    candidate_eval_dir: Path = OFFLINE_RERUN_REPORT_CANDIDATE_EVAL_DIR_OPTION,
    output_path: Path = OFFLINE_RERUN_REPORT_OUTPUT_PATH_OPTION,
    expected_candidate_model_version: str = CHAMPION_PROMOTION_EXPECTED_CANDIDATE_OPTION,
    no_fail_on_blocked: bool = OFFLINE_RERUN_REPORT_NO_FAIL_ON_BLOCKED_OPTION,
) -> None:
    """Write the champion-promotion.md Stage 1 rerun_report.md artifact."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = generate_offline_rerun_report(
        baseline_eval_dir=baseline_eval_dir,
        candidate_eval_dir=candidate_eval_dir,
        output_path=output_path,
        expected_candidate_model_version=expected_candidate_model_version,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed and not no_fail_on_blocked:
        raise Exit(code=1)


@app.command("champion-promotion-audit")
def champion_promotion_audit(
    output_dir: Path = CHAMPION_PROMOTION_AUDIT_OUTPUT_DIR_OPTION,
    promotion_process_path: Path | None = CHAMPION_PROMOTION_PROCESS_PATH_OPTION,
    repo_promotion_runbook_path: Path | None = CHAMPION_PROMOTION_REPO_RUNBOOK_PATH_OPTION,
    live_status_path: Path | None = CHAMPION_PROMOTION_LIVE_STATUS_PATH_OPTION,
    offline_rerun_report_path: Path | None = CHAMPION_PROMOTION_OFFLINE_RERUN_REPORT_PATH_OPTION,
    baseline_eval_dir: Path | None = CHAMPION_PROMOTION_BASELINE_EVAL_DIR_OPTION,
    candidate_eval_dir: Path | None = CHAMPION_PROMOTION_CANDIDATE_EVAL_DIR_OPTION,
    baseline_backtest_summary_path: Path | None = CHAMPION_PROMOTION_BASELINE_BACKTEST_SUMMARY_OPTION,
    candidate_backtest_summary_path: Path | None = CHAMPION_PROMOTION_CANDIDATE_BACKTEST_SUMMARY_OPTION,
    shadow_evaluation_path: Path | None = CHAMPION_PROMOTION_SHADOW_EVALUATION_PATH_OPTION,
    serving_readiness_path: Path | None = CHAMPION_PROMOTION_SERVING_READINESS_PATH_OPTION,
    bootstrap_decision_path: Path | None = CHAMPION_PROMOTION_BOOTSTRAP_DECISION_PATH_OPTION,
    cutover_report_path: Path | None = CHAMPION_PROMOTION_CUTOVER_REPORT_PATH_OPTION,
    rollback_runbook_path: Path | None = BOOTSTRAP_ROLLBACK_RUNBOOK_PATH_OPTION,
    expected_candidate_model_version: str = CHAMPION_PROMOTION_EXPECTED_CANDIDATE_OPTION,
    expected_fallback_model_version: str = CHAMPION_PROMOTION_EXPECTED_FALLBACK_OPTION,
    no_fail_on_blocked: bool = CHAMPION_PROMOTION_NO_FAIL_ON_BLOCKED_OPTION,
) -> None:
    """Fail-closed audit for every gate in champion-promotion.md."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = audit_champion_promotion_process(
        output_dir=output_dir,
        promotion_process_path=promotion_process_path,
        repo_promotion_runbook_path=repo_promotion_runbook_path,
        live_status_path=live_status_path,
        offline_rerun_report_path=offline_rerun_report_path,
        baseline_eval_dir=baseline_eval_dir,
        candidate_eval_dir=candidate_eval_dir,
        baseline_backtest_summary_path=baseline_backtest_summary_path,
        candidate_backtest_summary_path=candidate_backtest_summary_path,
        shadow_evaluation_path=shadow_evaluation_path,
        serving_readiness_path=serving_readiness_path,
        bootstrap_decision_path=bootstrap_decision_path,
        cutover_report_path=cutover_report_path,
        rollback_runbook_path=rollback_runbook_path,
        expected_candidate_model_version=expected_candidate_model_version,
        expected_fallback_model_version=expected_fallback_model_version,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed and not no_fail_on_blocked:
        raise Exit(code=1)


@app.command("slack-delivery-status")
def slack_delivery_status(
    output_path: Path = XGBOOST_V4_SLACK_DELIVERY_STATUS_OUTPUT_PATH_OPTION,
    channel_id: str = XGBOOST_V4_SLACK_DELIVERY_CHANNEL_ID_OPTION,
    attempted_at: str | None = XGBOOST_V4_SLACK_DELIVERY_ATTEMPTED_AT_OPTION,
    ok: bool = XGBOOST_V4_SLACK_DELIVERY_OK_OPTION,
    status: str = XGBOOST_V4_SLACK_DELIVERY_STATUS_OPTION,
    message_link: str | None = XGBOOST_V4_SLACK_DELIVERY_MESSAGE_LINK_OPTION,
    error_code: str | None = XGBOOST_V4_SLACK_DELIVERY_ERROR_CODE_OPTION,
    error_message: str | None = XGBOOST_V4_SLACK_DELIVERY_ERROR_MESSAGE_OPTION,
) -> None:
    """Write latest Slack delivery attempt evidence for objective audits."""

    normalized_status = status.strip().lower() if isinstance(status, str) else ""
    message_link_value = message_link.strip() if isinstance(message_link, str) else None
    error_code_value = error_code.strip() if isinstance(error_code, str) else None
    error_message_value = error_message.strip() if isinstance(error_message, str) else None
    message_link_value = message_link_value or None
    error_code_value = error_code_value or None
    error_message_value = error_message_value or None
    if not normalized_status:
        raise typer.BadParameter("status must be non-empty")
    if ok and normalized_status != "sent":
        raise typer.BadParameter("--ok requires --status sent")
    if not ok and normalized_status == "sent":
        raise typer.BadParameter("--status sent requires --ok")
    if ok and not message_link_value:
        raise typer.BadParameter("--ok requires --message-link")
    if not ok and not (error_code_value or error_message_value):
        raise typer.BadParameter("failed delivery evidence requires error code or message")

    attempted_at_value = attempted_at or _utc_now_iso()
    parsed_attempted_at, parse_error = _parse_utc_iso_timestamp(attempted_at_value)
    if parsed_attempted_at is None:
        raise typer.BadParameter(f"attempted_at must be UTC ISO-like timestamp: {parse_error}")

    payload: dict[str, Any] = {
        "attempted_at": attempted_at_value,
        "channel_id": channel_id,
        "ok": bool(ok),
        "status": normalized_status,
        "message_link": message_link_value,
        "error_code": error_code_value,
        "error_message": error_message_value,
    }
    _write_json_file_atomic(output_path, payload)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _json_path_exists(path: Path | None) -> bool:
    return path is not None and path.exists() and path.is_file()


def _read_optional_json_dict(path: Path | None) -> dict[str, Any] | None:
    payload = _read_optional_json_value(path)
    return payload if isinstance(payload, dict) else None


def _read_optional_json_value(path: Path | None) -> Any | None:
    if not _json_path_exists(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_optional_toml_dict(path: Path | None) -> dict[str, Any] | None:
    if not _json_path_exists(path):
        return None
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _positive_int(value: Any) -> bool:
    parsed = _optional_int(value)
    return parsed is not None and parsed > 0


def _is_true(value: Any) -> bool:
    return value is True


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")} or not parsed.is_integer():
        return None
    return int(parsed)


def _current_filesystem_headroom_evidence(
    live_status: dict[str, Any] | None,
    disk_headroom: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(live_status, dict):
        return None
    live_root = live_status.get("live_root")
    required_free_bytes = _optional_int(disk_headroom.get("required_free_bytes"))
    if not live_root or required_free_bytes is None:
        return None
    root = Path(str(live_root))
    if not root.exists():
        return None
    try:
        stat = os.statvfs(root)
    except OSError as exc:
        return {
            "available": False,
            "path": str(root),
            "error": f"{type(exc).__name__}: {exc}",
            "headroom_ok": False,
        }
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    headroom_margin_bytes = free_bytes - required_free_bytes
    low_margin_threshold_bytes = _optional_int(disk_headroom.get("low_margin_threshold_bytes"))
    if low_margin_threshold_bytes is None:
        low_margin_threshold_bytes = max(
            1 * 1024 * 1024 * 1024,
            int(required_free_bytes * 0.10),
        )
    headroom_ok = free_bytes >= required_free_bytes
    return {
        "available": True,
        "path": str(root),
        "free_bytes": free_bytes,
        "status_free_bytes": disk_headroom.get("free_bytes"),
        "required_free_bytes": required_free_bytes,
        "projected_remaining_bytes": disk_headroom.get("projected_remaining_bytes"),
        "headroom_margin_bytes": headroom_margin_bytes,
        "low_margin_threshold_bytes": low_margin_threshold_bytes,
        "headroom_low_margin": headroom_ok
        and headroom_margin_bytes < low_margin_threshold_bytes,
        "headroom_ok": headroom_ok,
    }


def _positive_float(value: Any) -> bool:
    parsed = _optional_float(value)
    return parsed is not None and parsed > 0.0


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _finite_number(value: Any) -> bool:
    return _optional_float(value) is not None


def _number_in_range(value: Any, lower: float, upper: float) -> bool:
    parsed = _optional_float(value)
    if parsed is None:
        return False
    return lower <= parsed <= upper


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_utc_iso_timestamp(value: Any) -> tuple[datetime | None, str | None]:
    if not _non_empty_string(value):
        return None, None
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        return None, str(exc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC), None


def _path_matches(value: Any, expected: Path | None) -> bool:
    if value is None or expected is None:
        return False
    try:
        return Path(str(value)).expanduser().resolve(strict=False) == expected.expanduser().resolve(
            strict=False
        )
    except (OSError, RuntimeError, ValueError):
        return str(value) == str(expected)


def _nested_dict(payload: dict[str, Any] | None, *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _missing_families_from_table(
    status: dict[str, Any] | None,
    table_name: str,
    required_families: list[str],
) -> list[str]:
    table = _nested_dict(status, "warehouse_freshness_evidence", "tables", table_name)
    families = table.get("families") if isinstance(table.get("families"), dict) else {}
    missing = set(table.get("missing_families") or [])
    missing.update(family for family in required_families if family not in families)
    return sorted(str(family) for family in missing)


def _unfresh_families_from_table(
    status: dict[str, Any] | None,
    table_name: str,
    required_families: list[str],
) -> list[str]:
    table = _nested_dict(status, "warehouse_freshness_evidence", "tables", table_name)
    families = table.get("families") if isinstance(table.get("families"), dict) else {}
    unfresh = {str(family) for family in table.get("stale_families") or []}
    unfresh.update(
        family
        for family in required_families
        if isinstance(families.get(family), dict) and families[family].get("fresh") is not True
    )
    if table.get("fresh") is False and not unfresh:
        unfresh.update(family for family in required_families if family in families)
    return sorted(unfresh)


def _missing_label_freshness_families(
    status: dict[str, Any] | None,
    required_families: list[str],
) -> list[str]:
    label_freshness = _nested_dict(status, "label_freshness_evidence")
    families = label_freshness.get("families") if isinstance(label_freshness.get("families"), dict) else {}
    missing = {str(family) for family in label_freshness.get("missing_label_families") or []}
    missing.update(family for family in required_families if family not in families)
    return sorted(missing)


def _unfresh_label_families(
    status: dict[str, Any] | None,
    required_families: list[str],
) -> list[str]:
    label_freshness = _nested_dict(status, "label_freshness_evidence")
    families = label_freshness.get("families") if isinstance(label_freshness.get("families"), dict) else {}
    unfresh = {str(family) for family in label_freshness.get("stale_families") or []}
    unfresh.update(
        family
        for family in required_families
        if isinstance(families.get(family), dict) and families[family].get("fresh") is not True
    )
    if label_freshness.get("fresh") is False and not unfresh:
        unfresh.update(family for family in required_families if family in families)
    return sorted(unfresh)


def _stage_passed(promotion_audit: dict[str, Any] | None, stage_name: str) -> bool:
    stages = promotion_audit.get("stages") if isinstance(promotion_audit, dict) else None
    if not isinstance(stages, list):
        return False
    return any(
        isinstance(stage, dict)
        and str(stage.get("name") or "") == stage_name
        and _is_true(stage.get("passed"))
        for stage in stages
    )


def _audit_check_passed(
    promotion_audit: dict[str, Any] | None,
    stage_name: str,
    check_name: str,
) -> bool:
    stages = promotion_audit.get("stages") if isinstance(promotion_audit, dict) else None
    if not isinstance(stages, list):
        return False
    for stage in stages:
        if not isinstance(stage, dict) or str(stage.get("name") or "") != stage_name:
            continue
        checks = stage.get("checks")
        if not isinstance(checks, list):
            return False
        return any(
            isinstance(check, dict)
            and str(check.get("name") or "") == check_name
            and _is_true(check.get("passed"))
            for check in checks
        )
    return False


def _promotion_process_source_audit_evidence(
    promotion_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    process = promotion_audit.get("promotion_process") if isinstance(promotion_audit, dict) else None
    process = process if isinstance(process, dict) else {}
    source_path_matches = _path_value_matches(
        process.get("source_path"),
        DEFAULT_CHAMPION_PROMOTION_PROCESS_PATH,
    )
    stage_check_passed = _audit_check_passed(
        promotion_audit,
        "Stage 0: 7-day Data Readiness",
        "promotion_process_source",
    )
    missing_markers = process.get("missing_required_markers")
    missing_markers = missing_markers if isinstance(missing_markers, list) else None
    passed = (
        _is_true(process.get("checked"))
        and _is_true(process.get("passed"))
        and _is_true(process.get("source_exists"))
        and source_path_matches
        and _non_empty_string(process.get("source_sha256"))
        and missing_markers == []
        and _is_true(process.get("repo_mirror_declares_source"))
        and stage_check_passed
    )
    return {
        "passed": passed,
        "checked": _is_true(process.get("checked")),
        "source_path": process.get("source_path"),
        "expected_source_path": str(DEFAULT_CHAMPION_PROMOTION_PROCESS_PATH),
        "source_path_matches": source_path_matches,
        "source_exists": _is_true(process.get("source_exists")),
        "source_sha256": process.get("source_sha256"),
        "missing_required_markers": missing_markers,
        "repo_mirror_path": process.get("repo_mirror_path"),
        "repo_mirror_declares_source": _is_true(process.get("repo_mirror_declares_source")),
        "stage_check_passed": stage_check_passed,
    }


def _promotion_artifact_path(promotion_audit: dict[str, Any] | None, name: str) -> str | None:
    artifact_paths = promotion_audit.get("artifact_paths") if isinstance(promotion_audit, dict) else None
    if not isinstance(artifact_paths, dict):
        return None
    value = artifact_paths.get(name)
    return str(value) if _non_empty_string(value) else None


def _candidate_eval_model_provenance_evidence(
    promotion_audit: dict[str, Any] | None,
    *,
    expected_model_path: Path | None,
) -> dict[str, Any]:
    candidate_eval_dir = _promotion_artifact_path(promotion_audit, "candidate_eval_dir")
    candidate_eval_manifest_path = None
    if candidate_eval_dir:
        candidate_eval_manifest_path = Path(candidate_eval_dir) / "manifest.json"
    candidate_eval_manifest = _read_optional_json_dict(candidate_eval_manifest_path)
    candidate_eval_model_path = (
        candidate_eval_manifest.get("model_path") if isinstance(candidate_eval_manifest, dict) else None
    )
    candidate_eval_model_version = (
        candidate_eval_manifest.get("model_version") if isinstance(candidate_eval_manifest, dict) else None
    )
    candidate_eval_dataset_dir = (
        candidate_eval_manifest.get("dataset_dir") if isinstance(candidate_eval_manifest, dict) else None
    )
    candidate_eval_dataset_version = (
        candidate_eval_manifest.get("dataset_version") if isinstance(candidate_eval_manifest, dict) else None
    )
    model_path_matches = _path_matches(candidate_eval_model_path, expected_model_path)
    model_version_matches = candidate_eval_model_version == "xgboost-v4"
    dataset_provenance_present = _non_empty_string(
        candidate_eval_dataset_dir
    ) and _non_empty_string(candidate_eval_dataset_version)
    passed = (
        candidate_eval_dir is not None
        and isinstance(candidate_eval_manifest, dict)
        and model_version_matches
        and model_path_matches
        and dataset_provenance_present
    )
    return {
        "passed": passed,
        "candidate_eval_dir": candidate_eval_dir,
        "candidate_eval_manifest_path": (
            str(candidate_eval_manifest_path) if candidate_eval_manifest_path else None
        ),
        "candidate_eval_model_version": candidate_eval_model_version,
        "expected_model_version": "xgboost-v4",
        "model_version_matches": model_version_matches,
        "candidate_eval_model_path": candidate_eval_model_path,
        "expected_model_path": str(expected_model_path) if expected_model_path else None,
        "model_path_matches": model_path_matches,
        "dataset_provenance_present": dataset_provenance_present,
        "candidate_eval_dataset_dir": candidate_eval_dataset_dir,
        "candidate_eval_dataset_version": candidate_eval_dataset_version,
    }


def _earliest_failed_stage(promotion_audit: dict[str, Any] | None) -> str | None:
    stages = promotion_audit.get("stages") if isinstance(promotion_audit, dict) else None
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if isinstance(stage, dict) and not _is_true(stage.get("passed")):
            return str(stage.get("name") or "unknown")
    return None


def _feature_ablation_evidence(
    path: Path | None,
    *,
    expected_model_path: Path | None,
    expected_dataset_dir: Any | None = None,
    expected_dataset_version: Any | None = None,
) -> dict[str, Any]:
    payload = _read_optional_json_dict(path)
    ablations = payload.get("ablations") if isinstance(payload, dict) else None
    baseline_metrics = payload.get("baseline_metrics") if isinstance(payload, dict) else None
    ablation_rows = ablations if isinstance(ablations, list) else []
    ablation_types = {
        str(row.get("ablation_type"))
        for row in ablation_rows
        if isinstance(row, dict) and isinstance(row.get("ablation_type"), str)
    }
    group_ablation_names = {
        str(row.get("name"))
        for row in ablation_rows
        if isinstance(row, dict)
        and row.get("ablation_type") == "group"
        and isinstance(row.get("name"), str)
    }
    required_groups = {"time", "long_window", "trade_structure", "tick_microstructure"}
    missing_groups = sorted(required_groups - group_ablation_names)
    measured_feature_ablation_count = 0
    measured_required_group_names: set[str] = set()
    for row in ablation_rows:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics")
        deltas = row.get("deltas")
        features = row.get("features")
        measured = (
            isinstance(features, list)
            and bool(features)
            and isinstance(metrics, dict)
            and _positive_int(metrics.get("sample_count"))
            and _finite_number(metrics.get("brier_score"))
            and _finite_number(metrics.get("roc_auc"))
            and isinstance(deltas, dict)
            and _finite_number(deltas.get("brier_score_increase"))
            and _finite_number(deltas.get("roc_auc_drop"))
        )
        if measured and row.get("ablation_type") == "feature":
            measured_feature_ablation_count += 1
        if measured and row.get("ablation_type") == "group" and isinstance(row.get("name"), str):
            measured_required_group_names.add(str(row["name"]))
    unmeasured_required_groups = sorted(required_groups - measured_required_group_names)
    model_path_matches = _path_matches(
        payload.get("model_path") if isinstance(payload, dict) else None,
        expected_model_path,
    )
    expected_dataset_dir_path = (
        Path(str(expected_dataset_dir))
        if _non_empty_string(expected_dataset_dir)
        else None
    )
    dataset_dir_matches = _path_matches(
        payload.get("dataset_dir") if isinstance(payload, dict) else None,
        expected_dataset_dir_path,
    )
    dataset_version_matches = (
        _non_empty_string(expected_dataset_version)
        and isinstance(payload, dict)
        and str(payload.get("dataset_version") or "") == str(expected_dataset_version)
    )
    passed = (
        isinstance(payload, dict)
        and payload.get("model_version") == "xgboost-v4"
        and model_path_matches
        and dataset_dir_matches
        and dataset_version_matches
        and payload.get("split") == "test"
        and payload.get("replacement_strategy") == "train_split_feature_mean"
        and isinstance(baseline_metrics, dict)
        and _positive_int(baseline_metrics.get("sample_count"))
        and _finite_number(baseline_metrics.get("brier_score"))
        and _finite_number(baseline_metrics.get("roc_auc"))
        and "feature" in ablation_types
        and "group" in ablation_types
        and measured_feature_ablation_count > 0
        and not missing_groups
        and not unmeasured_required_groups
    )
    return {
        "passed": passed,
        "path": str(path) if path else None,
        "model_version": payload.get("model_version") if isinstance(payload, dict) else None,
        "model_path": payload.get("model_path") if isinstance(payload, dict) else None,
        "expected_model_path": str(expected_model_path) if expected_model_path else None,
        "model_path_matches": model_path_matches,
        "dataset_dir": payload.get("dataset_dir") if isinstance(payload, dict) else None,
        "expected_dataset_dir": (
            str(expected_dataset_dir_path) if expected_dataset_dir_path else None
        ),
        "dataset_dir_matches": dataset_dir_matches,
        "dataset_version": payload.get("dataset_version") if isinstance(payload, dict) else None,
        "expected_dataset_version": (
            str(expected_dataset_version) if _non_empty_string(expected_dataset_version) else None
        ),
        "dataset_version_matches": dataset_version_matches,
        "split": payload.get("split") if isinstance(payload, dict) else None,
        "replacement_strategy": payload.get("replacement_strategy") if isinstance(payload, dict) else None,
        "baseline_sample_count": (
            baseline_metrics.get("sample_count") if isinstance(baseline_metrics, dict) else None
        ),
        "ablation_count": len(ablation_rows),
        "ablation_types": sorted(ablation_types),
        "group_ablation_names": sorted(group_ablation_names),
        "missing_required_groups": missing_groups,
        "measured_feature_ablation_count": measured_feature_ablation_count,
        "measured_required_group_names": sorted(measured_required_group_names & required_groups),
        "unmeasured_required_groups": unmeasured_required_groups,
    }


def _feature_importance_evidence(
    candidate_model_dir: Path | None,
    model_wrapper: dict[str, Any] | None,
) -> dict[str, Any]:
    path = None if candidate_model_dir is None else candidate_model_dir / "feature_importance.json"
    payload = _read_optional_json_value(path)
    rows = payload if isinstance(payload, list) else []
    expected_features_raw = (
        model_wrapper.get("feature_columns")
        if isinstance(model_wrapper, dict) and isinstance(model_wrapper.get("feature_columns"), list)
        else []
    )
    expected_features = {
        str(feature)
        for feature in expected_features_raw
        if isinstance(feature, str) and feature.strip()
    }
    invalid_rows: list[dict[str, Any]] = []
    unknown_features: list[str] = []
    valid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid_rows.append({"index": index, "missing_or_invalid": ["not_dict"]})
            continue
        feature = row.get("feature")
        failures = [
            name
            for name, passed in {
                "feature": _non_empty_string(feature),
                "gain": _positive_float(row.get("gain")),
                "split_count": _positive_int(row.get("split_count")),
            }.items()
            if not passed
        ]
        if failures:
            invalid_rows.append({"index": index, "missing_or_invalid": failures})
            continue
        feature_name = str(feature)
        if expected_features and feature_name not in expected_features:
            unknown_features.append(feature_name)
            continue
        valid_rows.append(row)
    top_row = valid_rows[0] if valid_rows else {}
    passed = (
        isinstance(payload, list)
        and bool(expected_features)
        and bool(valid_rows)
        and not invalid_rows
        and not unknown_features
    )
    return {
        "feature_importance_passed": passed,
        "feature_importance_path": str(path) if path else None,
        "feature_importance_exists": _json_path_exists(path),
        "model_feature_count": len(expected_features),
        "feature_importance_row_count": len(rows),
        "valid_feature_importance_row_count": len(valid_rows),
        "invalid_feature_importance_rows": invalid_rows,
        "feature_importance_unknown_features": sorted(set(unknown_features)),
        "top_feature": top_row.get("feature"),
        "top_feature_gain": top_row.get("gain"),
        "top_feature_split_count": top_row.get("split_count"),
    }


def _issue_57_added_feature_evidence(
    *,
    model_wrapper: dict[str, Any] | None,
    stability_report_path: Path | None = None,
    model_doc_path: Path = Path("docs/models/xgboost-v4.md"),
    feature_schema_path: Path = Path("src/bigan/canonical/schemas.py"),
    aggregation_path: Path = Path("src/bigan/features/aggregation.py"),
) -> dict[str, Any]:
    required_added_features = list(XGBOOST_V4_REQUIRED_ADDED_FEATURES)
    model_feature_columns = (
        [str(column) for column in model_wrapper.get("feature_columns") or []]
        if isinstance(model_wrapper, dict)
        else []
    )
    stability_report = _read_optional_json_dict(stability_report_path)
    dataset_feature_columns = (
        [str(column) for column in stability_report.get("feature_columns") or []]
        if isinstance(stability_report, dict)
        else []
    )
    missing_model_added_features = [
        feature for feature in required_added_features if feature not in model_feature_columns
    ]
    missing_dataset_added_features = [
        feature for feature in required_added_features if feature not in dataset_feature_columns
    ]
    model_doc_text = model_doc_path.read_text(encoding="utf-8") if model_doc_path.exists() else ""
    feature_schema_text = (
        feature_schema_path.read_text(encoding="utf-8") if feature_schema_path.exists() else ""
    )
    aggregation_text = aggregation_path.read_text(encoding="utf-8") if aggregation_path.exists() else ""
    return {
        "passed": (
            bool(model_feature_columns)
            and bool(dataset_feature_columns)
            and not missing_model_added_features
            and not missing_dataset_added_features
            and all(feature in model_doc_text for feature in required_added_features)
            and all(feature in feature_schema_text for feature in required_added_features)
            and all(feature in aggregation_text for feature in required_added_features)
        ),
        "required_added_features": required_added_features,
        "model_feature_columns_present": bool(model_feature_columns),
        "dataset_feature_columns_present": bool(dataset_feature_columns),
        "missing_model_added_features": missing_model_added_features,
        "missing_dataset_added_features": missing_dataset_added_features,
        "stability_report_path": str(stability_report_path) if stability_report_path else None,
        "stability_report_exists": _json_path_exists(stability_report_path),
        "model_doc_path": str(model_doc_path),
        "model_doc_exists": model_doc_path.exists(),
        "model_doc_has_added_features": all(
            feature in model_doc_text for feature in required_added_features
        ),
        "feature_schema_path": str(feature_schema_path),
        "feature_schema_exists": feature_schema_path.exists(),
        "canonical_schema_has_added_features": all(
            feature in feature_schema_text for feature in required_added_features
        ),
        "aggregation_path": str(aggregation_path),
        "aggregation_exists": aggregation_path.exists(),
        "aggregation_emits_added_features": all(
            feature in aggregation_text for feature in required_added_features
        ),
    }


def _issue_64_signal_label_evidence() -> dict[str, Any]:
    horizon_ms = 15 * 60_000
    t0 = 1_779_840_000_000
    signal_error: str | None = None
    label_error: str | None = None
    low_edge_signal: str | None = None
    buy_down_signal: str | None = None
    hold_open_signal: str | None = None
    sell_signal: str | None = None
    profit_target_sell_signal: str | None = None
    round_end_sell_signal: str | None = None
    buy_down_log: str | None = None
    sell_log: str | None = None
    signal_checks = {
        "low_edge_flat_holds": False,
        "buy_down_opens_down_position": False,
        "open_position_holds_above_exit_threshold": False,
        "sell_requires_open_position": False,
        "sell_profit_target_triggers": False,
        "sell_round_end_profit_triggers": False,
        "buy_down_log_names_signal": False,
        "sell_log_names_signal": False,
    }
    try:
        low_edge = evaluate_position_signal(
            PositionSignalState(),
            edge=0.01,
            market_implied_prob=0.55,
            outcome_side="UP",
            event_id="audit-flat-low-edge",
            exit_edge_threshold=0.05,
        )
        low_edge_signal = low_edge.signal
        buy_down = evaluate_position_signal(
            low_edge.state,
            edge=0.35,
            market_implied_prob=0.45,
            outcome_side="DOWN",
            event_id="audit-buy-down",
        )
        buy_down_signal = buy_down.signal
        hold_open = evaluate_position_signal(
            buy_down.state,
            edge=0.20,
            market_implied_prob=0.50,
            outcome_side="DOWN",
            event_id="audit-still-open",
            exit_edge_threshold=0.05,
        )
        hold_open_signal = hold_open.signal
        sell = evaluate_position_signal(
            hold_open.state,
            edge=0.02,
            market_implied_prob=0.53,
            outcome_side="DOWN",
            event_id="audit-sell-down",
            exit_edge_threshold=0.05,
        )
        sell_signal = sell.signal
        profit_target_sell = evaluate_position_signal(
            buy_down.state,
            edge=0.20,
            market_implied_prob=0.61,
            outcome_side="DOWN",
            event_id="audit-sell-profit-target",
        )
        profit_target_sell_signal = profit_target_sell.signal
        round_end_sell = evaluate_position_signal(
            buy_down.state,
            edge=0.20,
            market_implied_prob=0.47,
            outcome_side="DOWN",
            event_id="audit-sell-round-end",
            current_ts=t0 + horizon_ms - 30_000,
            round_end_ts=t0 + horizon_ms,
        )
        round_end_sell_signal = round_end_sell.signal
        buy_down_log = format_signal_row(
            ChampionSignalRow(
                created_at=t0 + 120_000,
                ts=t0 + 120_000,
                event_id="audit-buy-down",
                model_version="xgboost-v4",
                source_symbol="audit-down-token",
                canonical_symbol="BTC-15M:audit-down-side:DOWN",
                outcome_side="DOWN",
                prob_up_15m=0.10,
                market_implied_prob=0.40,
                edge=0.50,
                signal=buy_down.signal,
            )
        )
        sell_log = format_signal_row(
            ChampionSignalRow(
                created_at=t0 + 180_000,
                ts=t0 + 180_000,
                event_id="audit-sell-down",
                model_version="xgboost-v4",
                source_symbol="audit-down-token",
                canonical_symbol="BTC-15M:audit-down-side:DOWN",
                outcome_side="DOWN",
                prob_up_15m=0.47,
                market_implied_prob=0.53,
                edge=0.02,
                signal=sell.signal,
            )
        )
        signal_checks = {
            "low_edge_flat_holds": low_edge.signal == "HOLD"
            and not low_edge.state.position_open,
            "buy_down_opens_down_position": buy_down.signal == "BUY_DOWN"
            and buy_down.state.position_open
            and buy_down.state.outcome_side == "DOWN"
            and buy_down.state.entry_event_id == "audit-buy-down",
            "open_position_holds_above_exit_threshold": hold_open.signal == "HOLD"
            and hold_open.state == buy_down.state,
            "sell_requires_open_position": sell.signal == "SELL"
            and not sell.state.position_open,
            "sell_profit_target_triggers": profit_target_sell.signal == "SELL"
            and profit_target_sell.reason == "profit_target"
            and not profit_target_sell.state.position_open,
            "sell_round_end_profit_triggers": round_end_sell.signal == "SELL"
            and round_end_sell.reason == "round_end_profit"
            and not round_end_sell.state.position_open,
            "buy_down_log_names_signal": "BUY_DOWN" in buy_down_log,
            "sell_log_names_signal": "SELL" in sell_log,
        }
    except Exception as exc:  # pragma: no cover - fail-closed evidence path
        signal_error = f"{type(exc).__name__}: {exc}"

    label_row: dict[str, Any] | None = None
    label_checks = {
        "generated_one_down_label": False,
        "label_kind_is_down": False,
        "down_profit_field_populated": False,
        "up_profit_field_empty": False,
        "down_win_settlement_profitable": False,
    }
    try:
        label_rows = generate_labels_15m_v1(
            feature_rows=[
                {
                    "ts": t0 + 60_000,
                    "feature_ts": t0 + 60_000,
                    "source": "polymarket",
                    "source_symbol": "audit-down-token",
                    "source_market": "0xauditdown",
                    "canonical_symbol": "BTC-15M:audit-down-side:DOWN",
                    "symbol": "BTC-15M:audit-down-side:DOWN",
                    "market_implied_prob": 0.40,
                }
            ],
            round_rows=[
                {
                    "ts": t0,
                    "ingest_ts": t0 + horizon_ms + 1_000,
                    "source": "polymarket",
                    "source_market": "0xauditdown",
                    "round_slug": "btc-updown-15m-audit-down",
                    "round_start_ts": t0,
                    "round_end_ts": t0 + horizon_ms,
                    "start_price": 100.0,
                    "target_price": 99.0,
                }
            ],
            ingest_ts=t0 + (2 * horizon_ms),
        )
        label_row = label_rows[0] if label_rows else None
        label_checks = {
            "generated_one_down_label": len(label_rows) == 1,
            "label_kind_is_down": (
                isinstance(label_row, dict) and label_row.get("label_kind") == DOWN_LABEL_KIND
            ),
            "down_profit_field_populated": (
                isinstance(label_row, dict) and label_row.get("label_profit_down_15m") is True
            ),
            "up_profit_field_empty": (
                isinstance(label_row, dict) and label_row.get("label_profit_up_15m") is None
            ),
            "down_win_settlement_profitable": (
                isinstance(label_row, dict)
                and label_row.get("settlement_price") == 1.0
                and _positive_float(label_row.get("realized_return"))
            ),
        }
    except Exception as exc:  # pragma: no cover - fail-closed evidence path
        label_error = f"{type(exc).__name__}: {exc}"

    errors = [error for error in (signal_error, label_error) if error]
    return {
        "passed": not errors and all(signal_checks.values()) and all(label_checks.values()),
        "signal_checks": signal_checks,
        "signals": {
            "low_edge_flat": low_edge_signal,
            "buy_down": buy_down_signal,
            "hold_open": hold_open_signal,
            "sell": sell_signal,
            "profit_target_sell": profit_target_sell_signal,
            "round_end_sell": round_end_sell_signal,
        },
        "signal_logs": {
            "buy_down": buy_down_log,
            "sell": sell_log,
        },
        "label_checks": label_checks,
        "label_kind": label_row.get("label_kind") if isinstance(label_row, dict) else None,
        "label_profit_down_15m": (
            label_row.get("label_profit_down_15m") if isinstance(label_row, dict) else None
        ),
        "label_profit_up_15m": (
            label_row.get("label_profit_up_15m") if isinstance(label_row, dict) else None
        ),
        "settlement_price": label_row.get("settlement_price") if isinstance(label_row, dict) else None,
        "realized_return": label_row.get("realized_return") if isinstance(label_row, dict) else None,
        "errors": errors,
    }


def _down_validation_evidence(
    path: Path | None,
    *,
    expected_model_path: Path | None,
    expected_dataset_dir: Any | None = None,
    expected_dataset_version: Any | None = None,
) -> dict[str, Any]:
    payload = _read_optional_json_dict(path)
    summary = payload.get("summary") if isinstance(payload, dict) else None
    summary_rows = summary if isinstance(summary, list) else []
    issues = payload.get("issues") if isinstance(payload, dict) else None
    issue_rows = issues if isinstance(issues, list) else []
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    model_path_matches = _path_matches(
        metadata.get("model_path") if isinstance(metadata, dict) else None,
        expected_model_path,
    )
    expected_dataset_dir_path = (
        Path(str(expected_dataset_dir))
        if _non_empty_string(expected_dataset_dir)
        else None
    )
    metadata_dataset_dir = metadata.get("dataset_dir") if isinstance(metadata, dict) else None
    metadata_dataset_version = (
        metadata.get("dataset_version") if isinstance(metadata, dict) else None
    )
    dataset_dir_matches = (
        _path_matches(metadata_dataset_dir, expected_dataset_dir_path)
        if expected_dataset_dir_path is not None
        else _non_empty_string(metadata_dataset_dir)
    )
    dataset_version_matches = (
        str(metadata_dataset_version or "") == str(expected_dataset_version)
        if _non_empty_string(expected_dataset_version)
        else _non_empty_string(metadata_dataset_version)
    )
    metadata_checks = {
        "backtest_kind": (
            isinstance(metadata, dict) and str(metadata.get("backtest_kind") or "") == "direct_model"
        ),
        "model_path": model_path_matches,
        "dataset_dir": dataset_dir_matches,
        "dataset_version": dataset_version_matches,
        "warehouse_dir": isinstance(metadata, dict) and _non_empty_string(metadata.get("warehouse_dir")),
    }
    missing_metadata_fields = sorted(
        name for name, passed in metadata_checks.items() if not passed
    )

    def _row_failures(row: Any) -> list[str]:
        if not isinstance(row, dict):
            return ["not_dict"]
        settings = row.get("settings")
        return [
            name
            for name, passed in {
                "signals_considered": _positive_int(row.get("signals_considered")),
                "threshold_signals": _positive_int(row.get("threshold_signals")),
                "trade_count": _positive_int(row.get("trade_count")),
                "threshold": _finite_number(row.get("threshold")),
                "edge_threshold": _finite_number(row.get("edge_threshold")),
                "gross_pnl": _finite_number(row.get("gross_pnl")),
                "net_pnl": _finite_number(row.get("net_pnl")),
                "brier_score": _number_in_range(row.get("brier_score"), 0.0, 1.0),
                "brier_sample_count": _positive_int(row.get("brier_sample_count")),
                "turnover": _positive_float(row.get("turnover")),
                "symbols_considered": _positive_int(row.get("symbols_considered")),
                "symbols_with_quotes": _positive_int(row.get("symbols_with_quotes")),
                "hold_ms": _positive_int(row.get("hold_ms")),
                "fee_bps": isinstance(settings, dict)
                and _positive_float(settings.get("fee_bps")),
                "slippage_bps": isinstance(settings, dict)
                and _positive_float(settings.get("slippage_bps")),
            }.items()
            if not passed
        ]

    summary_row_failures = [
        {"index": index, "missing_or_invalid": failures}
        for index, row in enumerate(summary_rows)
        if (failures := _row_failures(row))
    ]
    qualifying_rows = [
        row
        for index, row in enumerate(summary_rows)
        if isinstance(row, dict)
        and not any(failure["index"] == index for failure in summary_row_failures)
    ]
    best_row = max(
        qualifying_rows,
        key=lambda row: _optional_float(row.get("net_pnl")) or 0.0,
        default={},
    )
    best_net_pnl = best_row.get("net_pnl")
    trade_sample = _down_validation_trade_sample_evidence(
        path=path,
        payload=payload,
        best_row=best_row,
    )
    passed = (
        isinstance(payload, dict)
        and str(payload.get("model_version") or "") == "xgboost-v4"
        and str(payload.get("required_outcome_side") or "").upper() == "DOWN"
        and not missing_metadata_fields
        and not issue_rows
        and bool(qualifying_rows)
        and _positive_float(best_net_pnl)
        and trade_sample["passed"]
    )
    return {
        "passed": passed,
        "path": str(path) if path else None,
        "model_version": payload.get("model_version") if isinstance(payload, dict) else None,
        "required_outcome_side": payload.get("required_outcome_side") if isinstance(payload, dict) else None,
        "backtest_kind": metadata.get("backtest_kind") if isinstance(metadata, dict) else None,
        "model_path": metadata.get("model_path") if isinstance(metadata, dict) else None,
        "expected_model_path": str(expected_model_path) if expected_model_path else None,
        "model_path_matches": model_path_matches,
        "dataset_dir": metadata.get("dataset_dir") if isinstance(metadata, dict) else None,
        "expected_dataset_dir": (
            str(expected_dataset_dir_path) if expected_dataset_dir_path else None
        ),
        "dataset_dir_matches": dataset_dir_matches,
        "dataset_version": metadata.get("dataset_version") if isinstance(metadata, dict) else None,
        "expected_dataset_version": (
            str(expected_dataset_version) if _non_empty_string(expected_dataset_version) else None
        ),
        "dataset_version_matches": dataset_version_matches,
        "warehouse_dir": metadata.get("warehouse_dir") if isinstance(metadata, dict) else None,
        "missing_metadata_fields": missing_metadata_fields,
        "issues": issue_rows,
        "summary_row_count": len(summary_rows),
        "invalid_summary_rows": summary_row_failures,
        "qualifying_row_count": len(qualifying_rows),
        "best_threshold": best_row.get("threshold"),
        "best_edge_threshold": best_row.get("edge_threshold"),
        "best_net_pnl": best_net_pnl,
        "best_gross_pnl": best_row.get("gross_pnl"),
        "best_brier_score": best_row.get("brier_score"),
        "best_brier_sample_count": best_row.get("brier_sample_count"),
        "best_trade_count": best_row.get("trade_count"),
        "best_signals_considered": best_row.get("signals_considered"),
        "best_threshold_signals": best_row.get("threshold_signals"),
        "best_turnover": best_row.get("turnover"),
        "trade_sample": trade_sample,
        "best_fee_bps": (
            best_row.get("settings", {}).get("fee_bps")
            if isinstance(best_row.get("settings"), dict)
            else None
        ),
        "best_slippage_bps": (
            best_row.get("settings", {}).get("slippage_bps")
            if isinstance(best_row.get("settings"), dict)
            else None
        ),
    }


def _down_validation_trade_sample_evidence(
    *,
    path: Path | None,
    payload: dict[str, Any] | None,
    best_row: dict[str, Any],
) -> dict[str, Any]:
    sample_path = _down_validation_trade_sample_path(
        diagnostics_path=path,
        payload=payload,
        threshold=best_row.get("threshold") if isinstance(best_row, dict) else None,
    )
    invalid_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    read_error: str | None = None
    if sample_path is not None and sample_path.is_file():
        try:
            lines = [line for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError as exc:
            lines = []
            read_error = f"{type(exc).__name__}: {exc}"
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows.append({"index": index, "missing_or_invalid": ["json"]})
                continue
            if not isinstance(row, dict):
                invalid_rows.append({"index": index, "missing_or_invalid": ["not_dict"]})
                continue
            failures = _down_validation_trade_row_failures(row, expected_threshold=best_row.get("threshold"))
            if failures:
                invalid_rows.append({"index": index, "missing_or_invalid": failures})
                continue
            rows.append(row)
    trade_count = _optional_int(best_row.get("trade_count") if isinstance(best_row, dict) else None)
    sample_count_exceeds_trade_count = (
        trade_count is not None and len(rows) > trade_count
    )
    passed = (
        sample_path is not None
        and sample_path.is_file()
        and not read_error
        and bool(rows)
        and not invalid_rows
        and not sample_count_exceeds_trade_count
    )
    return {
        "passed": passed,
        "path": str(sample_path) if sample_path is not None else None,
        "exists": sample_path is not None and sample_path.is_file(),
        "expected_outcome_side": "DOWN",
        "row_count": len(rows),
        "trade_count": trade_count,
        "sample_count_exceeds_trade_count": sample_count_exceeds_trade_count,
        "invalid_rows": invalid_rows,
        "read_error": read_error,
    }


def _down_validation_trade_sample_path(
    *,
    diagnostics_path: Path | None,
    payload: dict[str, Any] | None,
    threshold: Any,
) -> Path | None:
    parsed_threshold = _optional_float(threshold)
    if parsed_threshold is None:
        return None
    suffix = str(parsed_threshold).replace(".", "_")
    filename = f"trade_log_sample_threshold_{suffix}.jsonl"
    output_dir = payload.get("output_dir") if isinstance(payload, dict) else None
    if _non_empty_string(output_dir):
        return Path(str(output_dir)) / filename
    if diagnostics_path is None:
        return None
    return diagnostics_path.parent / filename


def _down_validation_trade_row_failures(
    row: dict[str, Any],
    *,
    expected_threshold: Any,
) -> list[str]:
    expected = _optional_float(expected_threshold)
    edge = _optional_float(row.get("edge"))
    threshold = _optional_float(row.get("threshold"))
    edge_threshold = _optional_float(row.get("edge_threshold"))
    failures = [
        name
        for name, passed in {
            "outcome_side": str(row.get("outcome_side") or "").upper() == "DOWN",
            "source_symbol": _non_empty_string(row.get("source_symbol")),
            "prob_up_15m": _number_in_range(row.get("prob_up_15m"), 0.0, 1.0),
            "market_implied_prob": _number_in_range(row.get("market_implied_prob"), 0.0, 1.0),
            "realized_label": isinstance(row.get("realized_label"), bool),
            "edge": edge is not None,
            "threshold": threshold is not None and expected is not None and threshold == expected,
            "edge_threshold": (
                edge_threshold is not None and expected is not None and edge_threshold == expected
            ),
            "edge_meets_threshold": edge is not None and expected is not None and edge >= expected,
            "decision_ts": _optional_int(row.get("decision_ts")) is not None,
            "entry_ts": _optional_int(row.get("entry_ts")) is not None,
            "exit_ts": _optional_int(row.get("exit_ts")) is not None,
            "net_pnl": _finite_number(row.get("net_pnl")),
            "fee_bps": _positive_float(row.get("fee_bps")),
            "slippage_bps": _positive_float(row.get("slippage_bps")),
        }.items()
        if not passed
    ]
    return failures


def _model_member_path_evidence(
    candidate_model_dir: Path | None,
    model_members: Any,
) -> dict[str, Any]:
    member_rows = model_members if isinstance(model_members, list) else []
    member_paths: list[str] = []
    invalid_member_paths: list[str] = []
    missing_member_paths: list[str] = []

    for index, member in enumerate(member_rows):
        raw_path = member.get("path") if isinstance(member, dict) else None
        if not isinstance(raw_path, str) or not raw_path.strip():
            invalid_member_paths.append(f"members[{index}].path")
            continue
        member_paths.append(raw_path)
        member_path = Path(raw_path)
        resolved_path = (
            member_path
            if member_path.is_absolute() or candidate_model_dir is None
            else candidate_model_dir / member_path
        )
        if not resolved_path.is_file():
            missing_member_paths.append(str(resolved_path))

    return {
        "member_paths": member_paths,
        "member_paths_exist": bool(member_rows)
        and not invalid_member_paths
        and not missing_member_paths,
        "invalid_member_paths": invalid_member_paths,
        "missing_member_paths": missing_member_paths,
    }


def _ensemble_seed_evidence(
    *,
    ensemble_summary: dict[str, Any] | None,
    model_members: Any,
    expected_member_count: int,
) -> dict[str, Any]:
    summary_seed_rows = (
        ensemble_summary.get("seeds") if isinstance(ensemble_summary, dict) else None
    )
    model_member_rows = model_members if isinstance(model_members, list) else []
    summary_seeds = [
        parsed
        for parsed in (_optional_int(seed) for seed in (summary_seed_rows or []))
        if parsed is not None
    ] if isinstance(summary_seed_rows, list) else []
    invalid_summary_seed_indices = [
        index
        for index, seed in enumerate(summary_seed_rows if isinstance(summary_seed_rows, list) else [])
        if _optional_int(seed) is None
    ]
    member_seeds = [
        parsed
        for parsed in (
            _optional_int(member.get("seed") if isinstance(member, dict) else None)
            for member in model_member_rows
        )
        if parsed is not None
    ]
    invalid_member_seed_indices = [
        index
        for index, member in enumerate(model_member_rows)
        if _optional_int(member.get("seed") if isinstance(member, dict) else None) is None
    ]
    duplicate_summary_seeds = sorted(
        {seed for seed in summary_seeds if summary_seeds.count(seed) > 1}
    )
    duplicate_member_seeds = sorted(
        {seed for seed in member_seeds if member_seeds.count(seed) > 1}
    )
    return {
        "ensemble_summary_seeds": summary_seeds,
        "model_member_seeds": member_seeds,
        "invalid_ensemble_summary_seed_indices": invalid_summary_seed_indices,
        "invalid_model_member_seed_indices": invalid_member_seed_indices,
        "duplicate_ensemble_summary_seeds": duplicate_summary_seeds,
        "duplicate_model_member_seeds": duplicate_member_seeds,
        "ensemble_seeds_match_members": bool(summary_seeds)
        and summary_seeds == member_seeds,
        "ensemble_seed_evidence_passed": (
            expected_member_count > 0
            and len(summary_seeds) == expected_member_count
            and len(member_seeds) == expected_member_count
            and summary_seeds == member_seeds
            and not invalid_summary_seed_indices
            and not invalid_member_seed_indices
            and not duplicate_summary_seeds
            and not duplicate_member_seeds
        ),
    }


def _ensemble_vs_single_evidence(ensemble_summary: dict[str, Any] | None) -> dict[str, Any]:
    comparison = _nested_dict(ensemble_summary, "ensemble_vs_single")
    single_metrics = _nested_dict(ensemble_summary, "single_model_metrics", "test")
    ensemble_metrics = _nested_dict(ensemble_summary, "ensemble_metrics", "test")
    brier_delta = _optional_float(comparison.get("brier_delta"))
    roc_auc_delta = _optional_float(comparison.get("roc_auc_delta"))
    pnl_delta = _optional_float(comparison.get("pnl_delta"))
    comparison_passed = (
        isinstance(ensemble_summary, dict)
        and _is_true(comparison.get("acceptable"))
        and (
            (brier_delta is not None and brier_delta <= 0.0)
            or (roc_auc_delta is not None and roc_auc_delta >= 0.0)
            or (pnl_delta is not None and pnl_delta >= 0.0)
        )
        and _number_in_range(single_metrics.get("brier_score"), 0.0, 1.0)
        and _number_in_range(ensemble_metrics.get("brier_score"), 0.0, 1.0)
        and _finite_number(single_metrics.get("roc_auc"))
        and _finite_number(ensemble_metrics.get("roc_auc"))
    )
    return {
        "ensemble_comparison_passed": comparison_passed,
        "ensemble_comparison_rule": comparison.get("rule"),
        "ensemble_comparison_split": comparison.get("split"),
        "ensemble_vs_single_acceptable": _is_true(comparison.get("acceptable")),
        "ensemble_vs_single_brier_delta": brier_delta,
        "ensemble_vs_single_roc_auc_delta": roc_auc_delta,
        "ensemble_vs_single_pnl_delta": pnl_delta,
        "single_model_test_brier": single_metrics.get("brier_score"),
        "single_model_test_roc_auc": single_metrics.get("roc_auc"),
        "single_model_test_pnl": single_metrics.get("pnl"),
        "ensemble_test_brier": ensemble_metrics.get("brier_score"),
        "ensemble_test_roc_auc": ensemble_metrics.get("roc_auc"),
        "ensemble_test_pnl": ensemble_metrics.get("pnl"),
    }


def _cv_time_series_evidence(cv_summary: dict[str, Any] | None, fold_count: int) -> dict[str, Any]:
    folds = cv_summary.get("folds") if isinstance(cv_summary, dict) else None
    fold_rows = folds if isinstance(folds, list) else []
    summary = _nested_dict(cv_summary, "summary")
    required_summary_metrics = [
        "brier_mean",
        "brier_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pnl_mean",
        "pnl_std",
    ]
    missing_summary_metrics = [
        metric for metric in required_summary_metrics if not _finite_number(summary.get(metric))
    ]
    invalid_fold_indices: list[int] = []
    invalid_fold_metric_indices: list[int] = []
    previous_train_end_ts: int | None = None
    previous_val_start_ts: int | None = None

    for index, row in enumerate(fold_rows):
        if not isinstance(row, dict):
            invalid_fold_indices.append(index)
            invalid_fold_metric_indices.append(index)
            continue
        train_start_ts = _optional_int(row.get("train_start_ts"))
        train_end_ts = _optional_int(row.get("train_end_ts"))
        val_start_ts = _optional_int(row.get("val_start_ts"))
        val_end_ts = _optional_int(row.get("val_end_ts"))
        if (
            train_start_ts is None
            or train_end_ts is None
            or val_start_ts is None
            or val_end_ts is None
        ):
            invalid_fold_indices.append(index)
            invalid_fold_metric_indices.append(index)
            continue
        metrics = row.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not _positive_int(metrics.get("sample_count"))
            or not _finite_number(metrics.get("brier_score"))
            or not _finite_number(metrics.get("pnl"))
        ):
            invalid_fold_metric_indices.append(index)
        if (
            train_start_ts > train_end_ts
            or val_start_ts <= train_end_ts
            or val_start_ts > val_end_ts
            or not _positive_int(row.get("train_count"))
            or not _positive_int(row.get("val_count"))
            or (previous_train_end_ts is not None and train_end_ts <= previous_train_end_ts)
            or (previous_val_start_ts is not None and val_start_ts <= previous_val_start_ts)
        ):
            invalid_fold_indices.append(index)
        previous_train_end_ts = train_end_ts
        previous_val_start_ts = val_start_ts

    return {
        "cv_fold_row_count": len(fold_rows),
        "cv_time_series_ordered": bool(fold_rows)
        and len(fold_rows) == fold_count
        and not invalid_fold_indices,
        "invalid_cv_fold_indices": invalid_fold_indices,
        "cv_fold_metrics_present": bool(fold_rows)
        and len(fold_rows) == fold_count
        and not invalid_fold_metric_indices,
        "invalid_cv_fold_metric_indices": invalid_fold_metric_indices,
        "cv_summary_metrics_present": bool(summary)
        and not missing_summary_metrics,
        "required_cv_summary_metrics": required_summary_metrics,
        "missing_cv_summary_metrics": missing_summary_metrics,
        "cv_brier_mean": summary.get("brier_mean"),
        "cv_brier_std": summary.get("brier_std"),
        "cv_roc_auc_mean": summary.get("roc_auc_mean"),
        "cv_roc_auc_std": summary.get("roc_auc_std"),
        "cv_pnl_mean": summary.get("pnl_mean"),
        "cv_pnl_std": summary.get("pnl_std"),
    }


def _slack_delivery_status_evidence(path: Path | None) -> dict[str, Any]:
    payload = _read_optional_json_dict(path)
    if path is None:
        return {
            "checked": False,
            "passed": True,
            "path": None,
            "exists": False,
            "status": None,
            "ok": None,
            "channel_id": None,
            "channel_id_matches": None,
            "attempted_at": None,
            "attempted_at_present": None,
            "attempted_at_age_seconds": None,
            "attempted_at_fresh": None,
            "attempted_at_parse_error": None,
            "max_age_seconds": XGBOOST_V4_SLACK_DELIVERY_MAX_AGE_SECONDS,
            "message_link": None,
            "message_link_present": None,
            "message_link_channel_matches": None,
            "error_code": None,
            "error_message": None,
        }
    channel_id = str(payload.get("channel_id") or "") if isinstance(payload, dict) else ""
    status = str(payload.get("status") or "") if isinstance(payload, dict) else ""
    ok = payload.get("ok") if isinstance(payload, dict) else None
    attempted_at = payload.get("attempted_at") if isinstance(payload, dict) else None
    message_link = payload.get("message_link") if isinstance(payload, dict) else None
    attempted_at_dt, attempted_at_parse_error = _parse_utc_iso_timestamp(attempted_at)
    attempted_at_age_seconds = (
        (datetime.now(UTC) - attempted_at_dt).total_seconds()
        if attempted_at_dt is not None
        else None
    )
    attempted_at_fresh = (
        attempted_at_age_seconds is not None
        and 0 <= attempted_at_age_seconds <= XGBOOST_V4_SLACK_DELIVERY_MAX_AGE_SECONDS
    )
    channel_id_matches = channel_id == "C0B5VHYSCN8"
    message_link_channel_matches = (
        _non_empty_string(message_link) and f"/archives/{channel_id}/" in message_link
    )
    delivery_ok = (ok is True or (ok is None and status in {"sent", "posted", "success"})) and (
        _non_empty_string(attempted_at)
        and attempted_at_fresh
        and _non_empty_string(message_link)
        and message_link_channel_matches
    )
    return {
        "checked": True,
        "passed": bool(isinstance(payload, dict) and delivery_ok and channel_id_matches),
        "path": str(path),
        "exists": _json_path_exists(path),
        "status": status or None,
        "ok": ok,
        "channel_id": channel_id or None,
        "channel_id_matches": channel_id_matches,
        "attempted_at": attempted_at,
        "attempted_at_present": _non_empty_string(attempted_at),
        "attempted_at_age_seconds": attempted_at_age_seconds,
        "attempted_at_fresh": attempted_at_fresh,
        "attempted_at_parse_error": attempted_at_parse_error,
        "max_age_seconds": XGBOOST_V4_SLACK_DELIVERY_MAX_AGE_SECONDS,
        "message_link": message_link,
        "message_link_present": _non_empty_string(message_link),
        "message_link_channel_matches": message_link_channel_matches,
        "error_code": payload.get("error_code") if isinstance(payload, dict) else None,
        "error_message": payload.get("error_message") if isinstance(payload, dict) else None,
    }


def _resolve_xgboost_v4_slack_delivery_status_path(
    path: Path | None,
    *,
    slack_automation_path: Path | None,
) -> Path | None:
    if isinstance(path, Path):
        return path
    if slack_automation_path == DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH:
        return DEFAULT_XGBOOST_V4_SLACK_DELIVERY_STATUS_PATH
    return None


def _resolve_xgboost_v4_collection_risk_path(
    path: Path | None,
    *,
    slack_automation_path: Path | None,
) -> Path | None:
    if isinstance(path, Path):
        return path
    if slack_automation_path == DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH:
        return DEFAULT_XGBOOST_V4_COLLECTION_RISK_PATH
    return None


def _collection_risk_snapshot_evidence(
    path: Path | None,
    *,
    live_status_path: Path,
    live_status: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = _read_optional_json_dict(path)
    if path is None:
        return {
            "checked": False,
            "path": None,
            "exists": False,
            "well_formed": False,
            "current": None,
        }
    status_artifact = _nested_dict(payload, "status_artifact")
    generated_at = payload.get("generated_at") if isinstance(payload, dict) else None
    status_artifact_generated_at = status_artifact.get("generated_at")
    live_status_generated_at = (
        live_status.get("generated_at") if isinstance(live_status, dict) else None
    )
    status_path = payload.get("status_path") if isinstance(payload, dict) else None
    status_path_matches = _path_matches(status_path, live_status_path)
    status_generated_at_matches = (
        _non_empty_string(status_artifact_generated_at)
        and _non_empty_string(live_status_generated_at)
        and status_artifact_generated_at == live_status_generated_at
    )
    reported_status_artifact_age_seconds = _optional_float(status_artifact.get("age_seconds"))
    status_artifact_max_age_seconds = _optional_float(status_artifact.get("max_age_seconds"))
    status_artifact_generated_at_dt, status_artifact_generated_at_parse_error = (
        _parse_utc_iso_timestamp(status_artifact_generated_at)
    )
    computed_status_artifact_age_seconds = (
        (datetime.now(UTC) - status_artifact_generated_at_dt).total_seconds()
        if status_artifact_generated_at_dt is not None
        else None
    )
    status_artifact_age_seconds = (
        computed_status_artifact_age_seconds
        if computed_status_artifact_age_seconds is not None
        else reported_status_artifact_age_seconds
    )
    status_artifact_age_within_limit = (
        status_artifact_age_seconds is not None
        and status_artifact_max_age_seconds is not None
        and 0 <= status_artifact_age_seconds <= status_artifact_max_age_seconds
    )
    current = (
        isinstance(payload, dict)
        and status_path_matches
        and status_generated_at_matches
        and _is_true(status_artifact.get("fresh"))
        and status_artifact_age_within_limit
    )
    return {
        "checked": True,
        "path": str(path),
        "exists": _json_path_exists(path),
        "well_formed": isinstance(payload, dict),
        "current": bool(current),
        "status_path": status_path,
        "status_path_matches": status_path_matches,
        "generated_at": generated_at,
        "status_artifact_generated_at": status_artifact_generated_at,
        "live_status_generated_at": live_status_generated_at,
        "status_generated_at_matches": status_generated_at_matches,
        "status_artifact_fresh": (
            status_artifact.get("fresh") if isinstance(status_artifact, dict) else None
        ),
        "reported_status_artifact_age_seconds": reported_status_artifact_age_seconds,
        "status_artifact_age_seconds": status_artifact_age_seconds,
        "status_artifact_max_age_seconds": status_artifact_max_age_seconds,
        "status_artifact_age_within_limit": status_artifact_age_within_limit,
        "status_artifact_generated_at_parse_error": status_artifact_generated_at_parse_error,
        "status_level": payload.get("status_level") if isinstance(payload, dict) else None,
        "blocked": payload.get("blocked") if isinstance(payload, dict) else None,
        "exit_code": payload.get("exit_code") if isinstance(payload, dict) else None,
        "readiness": _nested_dict(payload, "readiness"),
        "disk_headroom": _nested_dict(payload, "disk_headroom"),
        "current_filesystem_headroom": _nested_dict(
            payload,
            "current_filesystem_headroom",
        ),
        "disk_urgency": _nested_dict(payload, "disk_urgency"),
        "reclaim_candidates": (
            payload.get("reclaim_candidates")
            if isinstance(payload, dict) and isinstance(payload.get("reclaim_candidates"), list)
            else []
        ),
    }


def _post_readiness_expected_run_root(candidate_model_dir: Path | None) -> Path | None:
    if candidate_model_dir is None:
        return None
    if candidate_model_dir.parent.name != "models":
        return None
    if candidate_model_dir.parent.parent.name != "artifacts":
        return None
    return candidate_model_dir.parent.parent.parent


def _post_readiness_path_match(
    artifact_paths: dict[str, Any],
    top_level_payload: dict[str, Any],
    key: str,
    expected: Path | None,
) -> bool | None:
    if expected is None:
        return None
    value = artifact_paths.get(key)
    if value is None:
        value = top_level_payload.get(key)
    return _path_matches(value, expected)


def _post_readiness_latest_evidence(
    path: Path | None,
    *,
    promotion_audit_path: Path,
    objective_audit_path: Path,
    candidate_model_dir: Path | None,
    feature_ablation_path: Path | None,
    stability_report_path: Path | None,
    down_validation_path: Path | None,
) -> dict[str, Any]:
    if path is None:
        return {
            "checked": False,
            "path": None,
            "exists": False,
            "well_formed": None,
            "matches_current_inputs": None,
        }
    payload = _read_optional_json_dict(path)
    exists = _json_path_exists(path)
    artifact_paths = _nested_dict(payload, "artifact_paths")
    live_status_summary = _nested_dict(payload, "live_status_summary")
    expected_run_root = _post_readiness_expected_run_root(candidate_model_dir)
    pointer_run_root = (
        Path(str(payload.get("run_root")))
        if isinstance(payload, dict) and _non_empty_string(payload.get("run_root"))
        else None
    )
    expected_issue_coverage_audit_path = (
        (expected_run_root or pointer_run_root) / "artifacts" / "issue_coverage_audit.json"
        if (expected_run_root or pointer_run_root) is not None
        else None
    )
    path_matches = {
        "run_root": _post_readiness_path_match(
            artifact_paths,
            payload or {},
            "run_root",
            expected_run_root,
        ),
        "promotion_audit_path": _post_readiness_path_match(
            artifact_paths,
            payload or {},
            "promotion_audit_path",
            promotion_audit_path,
        ),
        "objective_audit_path": _post_readiness_path_match(
            artifact_paths,
            payload or {},
            "objective_audit_path",
            objective_audit_path,
        ),
        "candidate_model_path": _post_readiness_path_match(
            artifact_paths,
            payload or {},
            "candidate_model_path",
            None if candidate_model_dir is None else candidate_model_dir / "model.json",
        ),
        "feature_ablation_path": _post_readiness_path_match(
            artifact_paths,
            payload or {},
            "feature_ablation_path",
            feature_ablation_path,
        ),
        "dataset_stability_report_path": _post_readiness_path_match(
            artifact_paths,
            payload or {},
            "dataset_stability_report_path",
            stability_report_path,
        ),
        "down_validation_path": _post_readiness_path_match(
            artifact_paths,
            payload or {},
            "down_validation_path",
            down_validation_path,
        ),
        "issue_coverage_audit_path": _post_readiness_path_match(
            artifact_paths,
            payload or {},
            "issue_coverage_audit_path",
            expected_issue_coverage_audit_path,
        ),
    }
    checked_path_matches = [
        value for value in path_matches.values() if isinstance(value, bool)
    ]
    well_formed = bool(
        isinstance(payload, dict)
        and _non_empty_string(payload.get("run_root"))
        and _non_empty_string(payload.get("run_manifest_path"))
        and _non_empty_string(payload.get("run_manifest_phase"))
        and bool(live_status_summary)
        and bool(artifact_paths)
    )
    objective_prompt_blockers_raw = (
        payload.get("objective_prompt_to_artifact_blockers")
        if isinstance(payload, dict)
        else None
    )
    objective_prompt_blockers = (
        [str(blocker) for blocker in objective_prompt_blockers_raw]
        if isinstance(objective_prompt_blockers_raw, list)
        else []
    )
    objective_blockers_raw = (
        payload.get("objective_blockers") if isinstance(payload, dict) else None
    )
    objective_blockers = (
        [str(blocker) for blocker in objective_blockers_raw]
        if isinstance(objective_blockers_raw, list)
        else []
    )
    objective_only_blocks_on_latest_pointer = bool(objective_prompt_blockers) and all(
        blocker.startswith("post_readiness_latest_pointer:")
        for blocker in objective_prompt_blockers
    )
    objective_complete_clean = bool(
        isinstance(payload, dict)
        and payload.get("objective_complete") is True
        and payload.get("objective_decision") == "COMPLETE"
        and not objective_blockers
        and not objective_prompt_blockers
    )
    issue_coverage_issue_checks = (
        payload.get("issue_coverage_issue_checks") if isinstance(payload, dict) else None
    )
    issue_coverage_success_criteria = (
        payload.get("issue_coverage_objective_success_criteria")
        if isinstance(payload, dict)
        else None
    )
    required_issue_ids = ("#54", "#55", "#56", "#57", "#58", "#64", "#65")
    required_success_criteria_ids = (
        "all_requested_github_issues_satisfied",
        "fresh_xgboost_v4_model_created",
        "beats_current_champion",
        "champion_promotion_gates_passed",
        "hourly_slack_status_active",
        "post_readiness_latest_pointer_valid",
    )
    issue_coverage_required_issues_passed = bool(
        isinstance(issue_coverage_issue_checks, dict)
        and all(
            _is_true(_nested_dict(issue_coverage_issue_checks, issue_id).get("passed"))
            for issue_id in required_issue_ids
        )
    )
    issue_coverage_success_criteria_passed = bool(
        isinstance(issue_coverage_success_criteria, dict)
        and all(
            _is_true(_nested_dict(issue_coverage_success_criteria, criterion_id).get("passed"))
            for criterion_id in required_success_criteria_ids
        )
    )
    issue_coverage_pointer_criterion = _nested_dict(
        issue_coverage_success_criteria,
        "post_readiness_latest_pointer_valid",
    )
    issue_coverage_pointer_criterion_present = bool(issue_coverage_pointer_criterion)
    issue_coverage_success_criteria_without_pointer_passed = bool(
        isinstance(issue_coverage_success_criteria, dict)
        and issue_coverage_pointer_criterion_present
        and all(
            _is_true(_nested_dict(issue_coverage_success_criteria, criterion_id).get("passed"))
            for criterion_id in required_success_criteria_ids
            if criterion_id != "post_readiness_latest_pointer_valid"
        )
    )
    issue_coverage_self_reference_bootstrap_compatible = bool(
        objective_only_blocks_on_latest_pointer
        and issue_coverage_success_criteria_without_pointer_passed
        and not _is_true(issue_coverage_pointer_criterion.get("passed"))
    )
    issue_coverage_summary_compatible = bool(
        isinstance(payload, dict)
        and _non_empty_string(payload.get("issue_coverage_generated_at"))
        and issue_coverage_required_issues_passed
        and (
            issue_coverage_success_criteria_passed
            or issue_coverage_self_reference_bootstrap_compatible
        )
    )
    objective_status_compatible = bool(
        objective_complete_clean or objective_only_blocks_on_latest_pointer
    )
    return {
        "checked": True,
        "path": str(path),
        "exists": exists,
        "well_formed": well_formed,
        "run_id": payload.get("run_id") if isinstance(payload, dict) else None,
        "run_root": payload.get("run_root") if isinstance(payload, dict) else None,
        "run_manifest_path": (
            payload.get("run_manifest_path") if isinstance(payload, dict) else None
        ),
        "run_manifest_phase": (
            payload.get("run_manifest_phase") if isinstance(payload, dict) else None
        ),
        "live_status_summary_present": bool(live_status_summary),
        "live_status_summary": live_status_summary,
        "artifact_paths_present": bool(artifact_paths),
        "artifact_paths": artifact_paths,
        "promotion_audit_path": (
            payload.get("promotion_audit_path") if isinstance(payload, dict) else None
        ),
        "promotion_decision": (
            payload.get("promotion_decision") if isinstance(payload, dict) else None
        ),
        "promotion_passed": (
            payload.get("promotion_passed") if isinstance(payload, dict) else None
        ),
        "objective_audit_path": (
            payload.get("objective_audit_path") if isinstance(payload, dict) else None
        ),
        "issue_coverage_audit_path": (
            payload.get("issue_coverage_audit_path") if isinstance(payload, dict) else None
        ),
        "issue_coverage_generated_at": (
            payload.get("issue_coverage_generated_at") if isinstance(payload, dict) else None
        ),
        "issue_coverage_issue_checks": issue_coverage_issue_checks,
        "issue_coverage_objective_success_criteria": issue_coverage_success_criteria,
        "issue_coverage_required_issues_passed": issue_coverage_required_issues_passed,
        "issue_coverage_success_criteria_passed": issue_coverage_success_criteria_passed,
        "issue_coverage_success_criteria_without_pointer_passed": (
            issue_coverage_success_criteria_without_pointer_passed
        ),
        "issue_coverage_self_reference_bootstrap_compatible": (
            issue_coverage_self_reference_bootstrap_compatible
        ),
        "issue_coverage_summary_compatible": issue_coverage_summary_compatible,
        "objective_decision": (
            payload.get("objective_decision") if isinstance(payload, dict) else None
        ),
        "objective_complete": (
            payload.get("objective_complete") if isinstance(payload, dict) else None
        ),
        "objective_prompt_to_artifact_blockers": objective_prompt_blockers,
        "objective_blockers": objective_blockers,
        "objective_complete_clean": objective_complete_clean,
        "objective_only_blocks_on_latest_pointer": objective_only_blocks_on_latest_pointer,
        "objective_status_compatible": objective_status_compatible,
        "path_matches": path_matches,
        "matches_current_inputs": bool(checked_path_matches)
        and all(checked_path_matches)
        and issue_coverage_summary_compatible,
    }


def _slack_status_automation_evidence(
    path: Path | None,
    *,
    delivery_status_path: Path | None = None,
) -> dict[str, Any]:
    payload = _read_optional_toml_dict(path)
    prompt = str(payload.get("prompt") or "") if isinstance(payload, dict) else ""
    rrule = str(payload.get("rrule") or "") if isinstance(payload, dict) else None
    status = str(payload.get("status") or "") if isinstance(payload, dict) else None
    kind = str(payload.get("kind") or "") if isinstance(payload, dict) else None
    automation_id = str(payload.get("id") or "") if isinstance(payload, dict) else None
    channel_present = "C0B5VHYSCN8" in prompt
    slack_present = "slack" in prompt.lower()
    objective_audit_present = "xgboost-v4-objective-audit" in prompt
    post_readiness_runner_present = (
        "collection_readiness.ready_for_training" in prompt
        and "run_xgboost_v4_post_readiness.sh" in prompt
    )
    post_readiness_pointer_present = "xgboost_v4_post_readiness_latest.json" in prompt
    post_readiness_pointer_summary_present = (
        post_readiness_pointer_present
        and "run_manifest_phase" in prompt
        and "live_status_summary" in prompt
        and "artifact_paths" in prompt
    )
    post_readiness_duplicate_guard_present = (
        "POST_READINESS_SENTINEL_PATH" in prompt
        and "POST_READINESS_LOCK_DIR" in prompt
    )
    post_readiness_objective_refresh_present = (
        post_readiness_pointer_present
        and "run_root" in prompt
        and "promotion_audit_path" in prompt
        and "objective_audit_path" in prompt
        and "--promotion-audit-path" in prompt
        and "--candidate-model-dir" in prompt
        and "--feature-ablation-path" in prompt
        and "--stability-report-path" in prompt
        and "--down-validation-path" in prompt
        and "--output-path" in prompt
    )
    post_readiness_shadow_continuation_present = (
        "CONTINUE_POST_READINESS_RUN" in prompt
        and "RUN_ROOT" in prompt
        and "RUN_SHADOW" in prompt
        and "SHADOW_SINCE_MS" in prompt
        and "SHADOW_UNTIL_MS" in prompt
    )
    post_readiness_shadow_auto_window_present = (
        post_readiness_shadow_continuation_present
        and "SHADOW_SINCE_MS=auto" in prompt
        and "SHADOW_UNTIL_MS=auto" in prompt
    )
    post_readiness_shadow_full_session_present = (
        post_readiness_shadow_continuation_present
        and "MIN_SHADOW_SESSION_SECONDS" in prompt
        and "86400" in prompt
    )
    objective_success_criteria_reporting_present = (
        "objective_success_criteria" in prompt
        and "prompt_to_artifact_blockers" in prompt
        and "create_xgboost_v4_model" in prompt
        and "beat_current_champion" in prompt
    )
    issue_coverage_audit_present = (
        "xgboost-v4-issue-coverage-audit" in prompt
        and "issue_coverage_audit.json" in prompt
    )
    collection_risk_helper_present = (
        "check_xgboost_v4_collection_risk.sh" in prompt
        and "disk_headroom_evidence.headroom_low_margin" in prompt
        and "disk_headroom_evidence.headroom_ok" in prompt
        and "Docker" in prompt
        and "CoreSimulator" in prompt
        and "do not prune Docker" in prompt
    )
    collection_risk_helper_json_present = (
        collection_risk_helper_present
        and "check_xgboost_v4_collection_risk.sh --json" in prompt
        and "current_filesystem_headroom" in prompt
        and "reclaim_to_clear_block_bytes" in prompt
        and "reclaim_candidates" in prompt
    )
    collection_risk_helper_urgency_present = (
        collection_risk_helper_json_present
        and "status_artifact.fresh" in prompt
        and "status_artifact.age_seconds" in prompt
        and "status_artifact.max_age_seconds" in prompt
        and "disk_urgency.estimated_growth_bytes_per_day" in prompt
        and "disk_urgency.current_filesystem_days_to_min_free" in prompt
        and "disk_urgency.current_filesystem_min_free_before_ready" in prompt
        and "days-to-min-free urgency clock" in prompt
        and "min-free arrives before readiness" in prompt
    )
    collection_risk_helper_output_path_present = (
        collection_risk_helper_json_present
        and "--output-path" in prompt
        and (
            "data/xgboost-v4-run-20260523T103814Z/artifacts/collection_risk_latest.json"
            in prompt
        )
        and "collection_risk_latest.json" in prompt
    )
    skip_label_refresh_on_disk_block_present = (
        "skip bounded settled-label refresh" in prompt
        and "disk_headroom_evidence.headroom_ok=false" in prompt
        and "current_disk_headroom_evidence.headroom_ok=false" in prompt
        and "avoid adding avoidable writes" in prompt
    )
    delivery_evidence = _slack_delivery_status_evidence(delivery_status_path)
    delivery_status_instruction_present = (
        delivery_status_path is None
        or (
            "--slack-delivery-status-path" in prompt
            and "slack_status_delivery_latest.json" in prompt
            and "message_link" in prompt
            and "error_code" in prompt
        )
    )
    delivery_status_helper_instruction_present = (
        delivery_status_path is None
        or (
            "slack-delivery-status --message-link" in prompt
            and "--message-link" in prompt
            and "--error-code" in prompt
            and "--error-message" in prompt
            and "--output-path" in prompt
            and "slack_status_delivery_latest.json" in prompt
        )
    )
    config_passed = (
        isinstance(payload, dict)
        and automation_id == "xgboost-v4-work-status"
        and kind == "heartbeat"
        and status == "ACTIVE"
        and rrule == "FREQ=HOURLY;INTERVAL=1"
        and channel_present
        and slack_present
        and objective_audit_present
        and post_readiness_runner_present
        and post_readiness_pointer_present
        and post_readiness_pointer_summary_present
        and post_readiness_duplicate_guard_present
        and post_readiness_objective_refresh_present
        and post_readiness_shadow_continuation_present
        and post_readiness_shadow_auto_window_present
        and post_readiness_shadow_full_session_present
        and objective_success_criteria_reporting_present
        and issue_coverage_audit_present
        and collection_risk_helper_present
        and collection_risk_helper_json_present
        and collection_risk_helper_urgency_present
        and collection_risk_helper_output_path_present
        and skip_label_refresh_on_disk_block_present
        and delivery_status_instruction_present
        and delivery_status_helper_instruction_present
    )
    passed = config_passed and bool(delivery_evidence["passed"])
    return {
        "passed": passed,
        "config_passed": config_passed,
        "path": str(path) if path else None,
        "exists": _json_path_exists(path),
        "id": automation_id,
        "kind": kind,
        "status": status,
        "rrule": rrule,
        "channel_id_present": channel_present,
        "slack_instruction_present": slack_present,
        "objective_audit_instruction_present": objective_audit_present,
        "post_readiness_runner_instruction_present": post_readiness_runner_present,
        "post_readiness_pointer_instruction_present": post_readiness_pointer_present,
        "post_readiness_pointer_summary_instruction_present": (
            post_readiness_pointer_summary_present
        ),
        "post_readiness_duplicate_guard_instruction_present": post_readiness_duplicate_guard_present,
        "post_readiness_objective_refresh_instruction_present": (
            post_readiness_objective_refresh_present
        ),
        "post_readiness_shadow_continuation_instruction_present": (
            post_readiness_shadow_continuation_present
        ),
        "post_readiness_shadow_auto_window_instruction_present": (
            post_readiness_shadow_auto_window_present
        ),
        "post_readiness_shadow_full_session_instruction_present": (
            post_readiness_shadow_full_session_present
        ),
        "objective_success_criteria_reporting_instruction_present": (
            objective_success_criteria_reporting_present
        ),
        "issue_coverage_audit_instruction_present": issue_coverage_audit_present,
        "collection_risk_helper_instruction_present": collection_risk_helper_present,
        "collection_risk_helper_json_instruction_present": (
            collection_risk_helper_json_present
        ),
        "collection_risk_helper_urgency_instruction_present": (
            collection_risk_helper_urgency_present
        ),
        "collection_risk_helper_output_path_instruction_present": (
            collection_risk_helper_output_path_present
        ),
        "collection_risk_helper_status_freshness_instruction_present": (
            collection_risk_helper_json_present
            and "status_artifact.fresh" in prompt
            and "status_artifact.age_seconds" in prompt
            and "status_artifact.max_age_seconds" in prompt
        ),
        "skip_label_refresh_on_disk_block_instruction_present": (
            skip_label_refresh_on_disk_block_present
        ),
        "slack_delivery_status_instruction_present": delivery_status_instruction_present,
        "slack_delivery_status_helper_instruction_present": (
            delivery_status_helper_instruction_present
        ),
        "delivery_status": delivery_evidence,
    }


def _issue_54_live_monitoring_evidence(
    runbook_path: Path = Path("docs/runbooks/champion_promotion.md"),
) -> dict[str, Any]:
    thresholds = ChampionDriftThresholds()
    threshold_values = thresholds.to_dict()
    threshold_configured = (
        threshold_values.get("probability_mean_shift_abs") == 0.05
        and threshold_values.get("probability_std_relative_change") == 0.20
        and threshold_values.get("edge_threshold") == 0.30
        and threshold_values.get("edge_zero_window_ms") == 2 * 60 * 60 * 1000
        and threshold_values.get("label_positive_rate_min") == 0.50
        and threshold_values.get("label_consecutive_samples") == 50
    )
    runtime_hooks = {
        "evaluate_live_champion_drift": callable(evaluate_live_champion_drift),
        "run_live_champion_monitoring": callable(run_live_champion_monitoring),
        "evaluate_label_hit_rate_drift": callable(evaluate_label_hit_rate_drift),
        "record_champion_drift_incidents": callable(record_champion_drift_incidents),
    }
    incident_types = {
        "prediction_drift": "prediction_drift" in INCIDENT_TYPES,
        "label_shift": "label_shift" in INCIDENT_TYPES,
    }
    baseline_models: dict[str, dict[str, Any]] = {}
    for model_version in ("xgboost-v3", "xgboost-v4"):
        baseline_payload = CHAMPION_BASELINE_DISTRIBUTIONS.get(model_version)
        probability_distribution = (
            baseline_payload.get("probability_distribution")
            if isinstance(baseline_payload, dict)
            else None
        )
        edge_distribution = (
            baseline_payload.get("edge_distribution")
            if isinstance(baseline_payload, dict)
            else None
        )
        source = baseline_payload.get("source") if isinstance(baseline_payload, dict) else None
        split = baseline_payload.get("split") if isinstance(baseline_payload, dict) else None
        source_exists = Path(str(source)).exists() if _non_empty_string(source) else False
        probability_count = (
            _optional_int(probability_distribution.get("count"))
            if isinstance(probability_distribution, dict)
            else None
        )
        edge_count = (
            _optional_int(edge_distribution.get("count"))
            if isinstance(edge_distribution, dict)
            else None
        )
        edge_trigger_rate = (
            _optional_float(edge_distribution.get("trigger_rate_edge_ge_0_30"))
            if isinstance(edge_distribution, dict)
            else None
        )
        provenance_registered = (
            _non_empty_string(source)
            and source_exists
            and split == "val"
            and probability_count is not None
            and probability_count > 0
            and edge_count is not None
            and edge_count > 0
            and _finite_number(edge_trigger_rate)
        )
        try:
            baseline = champion_baseline_distribution(model_version)
        except (KeyError, TypeError, ValueError) as exc:
            baseline_models[model_version] = {
                "registered": False,
                "provenance_registered": provenance_registered,
                "error": str(exc),
                "source": source,
                "source_exists": source_exists,
                "split": split,
                "probability_count": probability_count,
                "edge_count": edge_count,
                "edge_trigger_rate_at_0_30": edge_trigger_rate,
                "mean": None,
                "std": None,
            }
            continue
        baseline_models[model_version] = {
            "registered": _finite_number(baseline.get("mean"))
            and _finite_number(baseline.get("std")),
            "provenance_registered": provenance_registered,
            "source": source,
            "source_exists": source_exists,
            "split": split,
            "probability_count": probability_count,
            "edge_count": edge_count,
            "edge_trigger_rate_at_0_30": edge_trigger_rate,
            "mean": baseline.get("mean"),
            "std": baseline.get("std"),
        }

    runbook_text = runbook_path.read_text(encoding="utf-8") if runbook_path.exists() else ""
    post_cutover_text = _markdown_section_after(runbook_text, "Post-cutover verification")
    normalized_post_cutover = post_cutover_text.lower()
    runbook_checks = {
        "post_cutover_section_present": bool(post_cutover_text.strip()),
        "mean_shift_threshold": "0.05" in normalized_post_cutover
        and "mean" in normalized_post_cutover,
        "std_shift_threshold": (
            "20%" in normalized_post_cutover or "0.20" in normalized_post_cutover
        )
        and "std" in normalized_post_cutover,
        "edge_zero_threshold": "edge" in normalized_post_cutover
        and "0" in normalized_post_cutover
        and "2 hour" in normalized_post_cutover,
        "label_hit_rate_threshold": "0.50" in normalized_post_cutover
        and "50" in normalized_post_cutover
        and "label" in normalized_post_cutover,
        "incident_catalog": "incident catalog" in normalized_post_cutover
        and "prediction_drift" in normalized_post_cutover
        and "label_shift" in normalized_post_cutover,
    }
    return {
        "passed": (
            threshold_configured
            and all(runtime_hooks.values())
            and all(incident_types.values())
            and all(model["registered"] for model in baseline_models.values())
            and all(model["provenance_registered"] for model in baseline_models.values())
            and all(runbook_checks.values())
        ),
        "thresholds": threshold_values,
        "threshold_configured": threshold_configured,
        "runtime_hooks": runtime_hooks,
        "incident_types": incident_types,
        "baseline_models": baseline_models,
        "runbook_path": str(runbook_path),
        "runbook_alert_conditions": runbook_checks,
    }


def _markdown_section_after(markdown_text: str, marker: str) -> str:
    lower_text = markdown_text.lower()
    marker_index = lower_text.find(marker.lower())
    if marker_index < 0:
        return ""
    next_heading_index = lower_text.find("\n## ", marker_index + len(marker))
    if next_heading_index < 0:
        return markdown_text[marker_index:]
    return markdown_text[marker_index:next_heading_index]


def _dataset_stability_evidence(
    path: Path | None,
    *,
    expected_dataset_dir: Any = None,
    expected_dataset_version: Any = None,
) -> dict[str, Any]:
    payload = _read_optional_json_dict(path)
    required_splits = ["train", "val", "test"]
    required_core_features = ["spread", "microprice", "trade_volume_1m"]
    split_summary = _nested_dict(payload, "split_label_summary")
    feature_distributions = _nested_dict(payload, "core_feature_distributions")
    split_row_counts = {
        split: _optional_int(_nested_dict(split_summary, split).get("row_count"))
        for split in required_splits
    }
    label_positive_rates = {
        split: _nested_dict(split_summary, split).get("positive_rate")
        for split in required_splits
    }
    missing_label_splits = [
        split
        for split in required_splits
        if split_row_counts[split] is None
        or split_row_counts[split] <= 0
        or not _number_in_range(label_positive_rates[split], 0.0, 1.0)
    ]
    missing_core_features = [
        feature
        for feature in required_core_features
        if feature not in feature_distributions
    ]
    missing_feature_splits = [
        f"{feature}:{split}"
        for feature in required_core_features
        for split in required_splits
        if _optional_int(_nested_dict(feature_distributions, feature, split).get("count")) is None
        or _optional_int(_nested_dict(feature_distributions, feature, split).get("count")) <= 0
        or not _finite_number(_nested_dict(feature_distributions, feature, split).get("mean"))
    ]
    markdown_path = None
    if path is not None:
        candidate = path.with_suffix(".md")
        markdown_path = str(candidate) if candidate.exists() else None
    expected_dataset_dir_path = (
        Path(str(expected_dataset_dir))
        if _non_empty_string(expected_dataset_dir)
        else None
    )
    dataset_dir_matches = _path_matches(
        payload.get("dataset_dir") if isinstance(payload, dict) else None,
        expected_dataset_dir_path,
    )
    dataset_version_matches = (
        _non_empty_string(expected_dataset_version)
        and isinstance(payload, dict)
        and str(payload.get("dataset_version") or "") == str(expected_dataset_version)
    )
    passed = (
        isinstance(payload, dict)
        and payload.get("schema_version") == "dataset_stability_report_v1"
        and dataset_dir_matches
        and dataset_version_matches
        and not missing_label_splits
        and not missing_core_features
        and not missing_feature_splits
        and markdown_path is not None
    )
    return {
        "passed": passed,
        "path": str(path) if path else None,
        "exists": _json_path_exists(path),
        "markdown_path": markdown_path,
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "dataset_dir": payload.get("dataset_dir") if isinstance(payload, dict) else None,
        "expected_dataset_dir": (
            str(expected_dataset_dir_path) if expected_dataset_dir_path else None
        ),
        "dataset_dir_matches": dataset_dir_matches,
        "dataset_version": payload.get("dataset_version") if isinstance(payload, dict) else None,
        "expected_dataset_version": (
            str(expected_dataset_version) if _non_empty_string(expected_dataset_version) else None
        ),
        "dataset_version_matches": dataset_version_matches,
        "required_splits": required_splits,
        "split_row_counts": split_row_counts,
        "label_positive_rates": label_positive_rates,
        "missing_label_splits": missing_label_splits,
        "required_core_features": required_core_features,
        "core_features": payload.get("core_features") if isinstance(payload, dict) else None,
        "missing_core_features": missing_core_features,
        "missing_feature_splits": missing_feature_splits,
    }


def _issue_56_multimarket_schema_evidence(
    *,
    model_wrapper: dict[str, Any] | None,
    stability_report_path: Path | None,
    required_families: list[str],
) -> dict[str, Any]:
    stability = _read_optional_json_dict(stability_report_path)
    dataset_feature_columns = (
        [str(column) for column in stability.get("feature_columns") or []]
        if isinstance(stability, dict)
        else []
    )
    model_feature_columns = (
        [str(column) for column in model_wrapper.get("feature_columns") or []]
        if isinstance(model_wrapper, dict)
        else []
    )
    required_structure_features = list(XGBOOST_V4_REQUIRED_MARKET_FEATURES)
    missing_structure_features = [
        feature for feature in required_structure_features if feature not in model_feature_columns
    ]
    feature_count_limit = 30
    family_splits = _nested_dict(stability, "family_splits")
    missing_family_split_rows = [
        f"{family}:{split}"
        for family in required_families
        for split in ("train", "val", "test")
        if (_optional_int(_nested_dict(family_splits, family, split).get("row_count")) or 0) <= 0
    ]
    label_versions = [
        str(version)
        for version in (stability.get("label_versions") if isinstance(stability, dict) else []) or []
    ]
    feature_versions = [
        str(version)
        for version in (stability.get("feature_versions") if isinstance(stability, dict) else []) or []
    ]
    required_underlyings = sorted({family.split("-", 1)[0] for family in required_families})
    required_horizons = sorted({family.split("-", 1)[1] for family in required_families})
    model_schema_matches_dataset = (
        bool(model_feature_columns)
        and bool(dataset_feature_columns)
        and model_feature_columns == dataset_feature_columns
    )
    feature_count_within_limit = len(model_feature_columns) <= feature_count_limit
    passed = (
        model_schema_matches_dataset
        and feature_count_within_limit
        and not missing_structure_features
        and not missing_family_split_rows
        and len(label_versions) == 1
        and "profit" in label_versions[0].lower()
        and bool(feature_versions)
    )
    return {
        "passed": passed,
        "required_underlyings": required_underlyings,
        "required_horizons": required_horizons,
        "feature_count": len(model_feature_columns),
        "feature_count_limit": feature_count_limit,
        "issue_suggested_feature_count_limit": feature_count_limit,
        "feature_count_within_limit": feature_count_within_limit,
        "required_structure_features": required_structure_features,
        "missing_structure_features": missing_structure_features,
        "model_feature_columns": model_feature_columns,
        "dataset_feature_columns": dataset_feature_columns,
        "model_schema_matches_dataset": model_schema_matches_dataset,
        "label_versions": label_versions,
        "feature_versions": feature_versions,
        "unified_profitability_label": len(label_versions) == 1
        and bool(label_versions)
        and "profit" in label_versions[0].lower(),
        "missing_family_split_rows": missing_family_split_rows,
    }


def _issue_65_tick_feature_evidence(
    *,
    model_wrapper: dict[str, Any] | None,
    model_doc_path: Path = Path("docs/models/xgboost-v4.md"),
    feature_schema_path: Path = Path("src/bigan/canonical/schemas.py"),
    aggregation_path: Path = Path("src/bigan/features/aggregation.py"),
    live_runner_path: Path = Path("scripts/run_champion_live.sh"),
) -> dict[str, Any]:
    required_tick_features = list(XGBOOST_V4_REQUIRED_TICK_FEATURES)
    model_feature_columns = (
        [str(column) for column in model_wrapper.get("feature_columns") or []]
        if isinstance(model_wrapper, dict)
        else []
    )
    missing_model_tick_features = [
        feature for feature in required_tick_features if feature not in model_feature_columns
    ]

    model_doc_text = model_doc_path.read_text(encoding="utf-8") if model_doc_path.exists() else ""
    normalized_doc_text = model_doc_text.lower()
    feature_schema_text = (
        feature_schema_path.read_text(encoding="utf-8") if feature_schema_path.exists() else ""
    )
    aggregation_text = aggregation_path.read_text(encoding="utf-8") if aggregation_path.exists() else ""
    normalized_aggregation_text = aggregation_text.lower()
    live_runner_text = live_runner_path.read_text(encoding="utf-8") if live_runner_path.exists() else ""
    doc_checks = {
        "documents_tick_features": all(feature in normalized_doc_text for feature in required_tick_features),
        "documents_5_second_scan": any(
            token in normalized_doc_text
            for token in ("5-second", "5 second", "5 seconds", "5s")
        ),
        "documents_dedupe_evidence": "dedupe" in normalized_doc_text
        and "evidence" in normalized_doc_text,
        "documents_v4_eval": "offline" in normalized_doc_text
        and "backtest" in normalized_doc_text
        and "shadow" in normalized_doc_text,
    }
    schema_checks = {
        "canonical_schema_has_tick_features": all(
            feature in feature_schema_text for feature in required_tick_features
        ),
        "aggregation_emits_tick_features": all(
            feature in aggregation_text for feature in required_tick_features
        ),
        "aggregation_uses_5_second_windows": "window_ms=5_000" in aggregation_text,
        "aggregation_deduplicates_tick_velocity": "deduped" in normalized_aggregation_text,
    }
    runner_checks = {
        "runner_defaults_to_5_second_cycle": 'CYCLE_SLEEP_SECONDS="${CYCLE_SLEEP_SECONDS:-5}"'
        in live_runner_text,
        "runner_uses_cycle_sleep": 'sleep_for "${CYCLE_SLEEP_SECONDS}"' in live_runner_text,
        "runner_logs_cycle_sleep": "cycle sleep seconds=${CYCLE_SLEEP_SECONDS}" in live_runner_text,
        "runner_uses_skip_existing_features": "--skip-existing" in live_runner_text,
        "runner_uses_skip_existing_monitoring_events": "--skip-existing-monitoring-events"
        in live_runner_text,
        "runner_uses_skip_existing_predictions": "--skip-existing-predictions" in live_runner_text,
        "runner_uses_skip_existing_labels": "--skip-existing-labels" in live_runner_text,
        "runner_configures_live_min_free_bytes": 'LIVE_MIN_FREE_BYTES="${LIVE_MIN_FREE_BYTES:-5368709120}"'
        in live_runner_text,
        "runner_checks_live_min_free_space": "check_live_root_free_space" in live_runner_text
        and "live root filesystem free space below floor" in live_runner_text,
        "runner_checks_space_before_capture": "check_live_root_free_space\nacquire_live_root_lock"
        in live_runner_text,
        "runner_checks_space_each_cycle": "check_live_root_free_space >>" in live_runner_text
        and "live root free-space floor breached" in live_runner_text,
    }
    return {
        "passed": (
            bool(model_feature_columns)
            and not missing_model_tick_features
            and all(doc_checks.values())
            and all(schema_checks.values())
            and all(runner_checks.values())
        ),
        "required_tick_features": required_tick_features,
        "model_feature_columns_present": bool(model_feature_columns),
        "missing_model_tick_features": missing_model_tick_features,
        "model_doc_path": str(model_doc_path),
        "model_doc_exists": model_doc_path.exists(),
        "doc_checks": doc_checks,
        "feature_schema_path": str(feature_schema_path),
        "feature_schema_exists": feature_schema_path.exists(),
        "aggregation_path": str(aggregation_path),
        "aggregation_exists": aggregation_path.exists(),
        "schema_checks": schema_checks,
        "live_runner_path": str(live_runner_path),
        "live_runner_exists": live_runner_path.exists(),
        "runner_checks": runner_checks,
    }


def _build_xgboost_v4_objective_audit(
    *,
    live_status: dict[str, Any] | None,
    promotion_audit: dict[str, Any] | None,
    output_path: Path,
    live_status_path: Path,
    promotion_audit_path: Path,
    candidate_model_dir: Path | None,
    feature_ablation_path: Path | None,
    stability_report_path: Path | None,
    down_validation_path: Path | None,
    slack_automation_path: Path | None,
    slack_delivery_status_path: Path | None,
    collection_risk_path: Path | None,
    post_readiness_latest_path: Path | None,
) -> dict[str, Any]:
    readiness = _nested_dict(live_status, "collection_readiness")
    quarantine_clean_window = _nested_dict(readiness, "quarantine_clean_window")
    required_families = ["BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M"]
    status_required_raw = readiness.get("required_families")
    status_required_families = (
        [str(family) for family in status_required_raw]
        if isinstance(status_required_raw, list)
        else []
    )
    missing_readiness_required_families = [
        family for family in required_families if family not in status_required_families
    ]
    outcomes = _nested_dict(live_status, "monitoring_outcome_evidence")
    raw_integrity = _nested_dict(live_status, "raw_segment_integrity")
    raw_quarantine = _nested_dict(live_status, "raw_segment_quarantine")
    health = _nested_dict(live_status, "health_evidence")
    disk_headroom = _nested_dict(live_status, "disk_headroom_evidence")
    live_root_lock = _nested_dict(live_status, "live_root_lock_evidence")
    raw_manifest_coverage = _nested_dict(live_status, "raw_manifest_coverage_evidence")
    liveness = _nested_dict(live_status, "liveness_evidence")
    features_missing = _missing_families_from_table(live_status, "features_15m_v1", required_families)
    predictions_missing = _missing_families_from_table(live_status, "predictions", required_families)
    features_unfresh = _unfresh_families_from_table(live_status, "features_15m_v1", required_families)
    predictions_unfresh = _unfresh_families_from_table(live_status, "predictions", required_families)
    label_freshness = _nested_dict(live_status, "label_freshness_evidence")
    live_feature_schema = _nested_dict(
        live_status,
        "warehouse_schema_evidence",
        "features_15m_v1",
    )
    labels_missing_freshness = _missing_label_freshness_families(live_status, required_families)
    labels_unfresh = _unfresh_label_families(live_status, required_families)
    missing_outcome_families = [str(family) for family in outcomes.get("missing_outcome_families") or []]
    missing_event_families = [str(family) for family in outcomes.get("missing_event_families") or []]
    outcome_family_rows = outcomes.get("families") if isinstance(outcomes.get("families"), dict) else {}
    monitoring_event_rows_by_family = {
        family: _optional_int(_nested_dict(outcome_family_rows, family).get("event_rows"))
        for family in required_families
    }
    monitoring_outcome_rows_by_family = {
        family: _optional_int(_nested_dict(outcome_family_rows, family).get("outcome_rows"))
        for family in required_families
    }
    monitoring_event_rows = _optional_int(outcomes.get("event_rows"))
    monitoring_outcome_rows = _optional_int(outcomes.get("outcome_rows"))
    monitoring_model_version = str(outcomes.get("model_version") or "")
    expected_monitoring_model_version = "xgboost-v4"
    aggregate_monitoring_metrics_valid = (
        _number_in_range(outcomes.get("brier_score"), 0.0, 1.0)
        and _number_in_range(outcomes.get("hit_rate"), 0.0, 1.0)
        and _finite_number(outcomes.get("avg_realized_return"))
    )
    monitoring_invalid_metric_families = [
        family
        for family in required_families
        if not (
            _number_in_range(_nested_dict(outcome_family_rows, family).get("brier_score"), 0.0, 1.0)
            and _number_in_range(_nested_dict(outcome_family_rows, family).get("hit_rate"), 0.0, 1.0)
            and _finite_number(_nested_dict(outcome_family_rows, family).get("avg_realized_return"))
        )
    ]
    missing_event_row_families = [
        family
        for family, row_count in monitoring_event_rows_by_family.items()
        if row_count is None or row_count <= 0
    ]
    missing_outcome_row_families = [
        family
        for family, row_count in monitoring_outcome_rows_by_family.items()
        if row_count is None or row_count <= 0
    ]
    issue_54_monitoring_evidence = _issue_54_live_monitoring_evidence()
    raw_segment_count_value = live_status.get("raw_segment_count") if isinstance(live_status, dict) else None
    processed_manifest_rows_value = (
        live_status.get("processed_manifest_rows") if isinstance(live_status, dict) else None
    )
    expected_live_root = Path("data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z")
    expected_screen_session = "xgbv4_7d_atomic_20260523T125657Z"
    live_root_matches = _path_matches(
        live_status.get("live_root") if isinstance(live_status, dict) else None,
        expected_live_root,
    )
    warehouse_matches = _path_matches(
        live_status.get("warehouse") if isinstance(live_status, dict) else None,
        expected_live_root / "warehouse",
    )
    screen_session_matches = (
        str(live_status.get("screen_session") if isinstance(live_status, dict) else "")
        == expected_screen_session
    )
    invalid_raw_segments = _optional_int(raw_integrity.get("invalid_count"))
    quarantined_raw_segments = _optional_int(raw_quarantine.get("quarantined_count"))
    quarantine_clean_window_ready = _is_true(quarantine_clean_window.get("meets_target"))
    latest_quarantined_segment = _nested_dict(raw_quarantine, "latest_quarantined_segment")
    if not latest_quarantined_segment:
        latest_quarantined_segment = _nested_dict(
            quarantine_clean_window,
            "latest_quarantined_segment",
        )
    latest_quarantined_gzip_probe = _nested_dict(latest_quarantined_segment, "gzip_probe")
    latest_quarantined_gzip_valid = latest_quarantined_gzip_probe.get("gzip_valid")
    latest_quarantined_gzip_valid = (
        latest_quarantined_gzip_valid
        if isinstance(latest_quarantined_gzip_valid, bool)
        else None
    )
    unrecovered_error_match_count = _optional_int(health.get("unrecovered_error_match_count"))
    disk_headroom_ok = _is_true(disk_headroom.get("headroom_ok"))
    current_disk_headroom = _current_filesystem_headroom_evidence(live_status, disk_headroom)
    current_disk_headroom_ok = (
        current_disk_headroom is None or _is_true(current_disk_headroom.get("headroom_ok"))
    )
    live_root_lock_ok = (
        _is_true(live_root_lock.get("lock_dir_exists"))
        and _is_true(live_root_lock.get("pid_file_exists"))
        and _is_true(live_root_lock.get("owner_running"))
        and not live_root_lock.get("pid_parse_error")
    )
    stale_missing_processed_count = _optional_int(
        raw_manifest_coverage.get("stale_missing_processed_count")
    )
    extra_processed_count = _optional_int(raw_manifest_coverage.get("extra_processed_count"))
    raw_manifest_coverage_ok = stale_missing_processed_count == 0 and extra_processed_count == 0
    candidate_cv_path = None if candidate_model_dir is None else candidate_model_dir / "cv_summary.json"
    candidate_ensemble_summary_path = (
        None if candidate_model_dir is None else candidate_model_dir / "ensemble_summary.json"
    )
    candidate_model_path = None if candidate_model_dir is None else candidate_model_dir / "model.json"
    cv_summary = _read_optional_json_dict(candidate_cv_path)
    ensemble_summary = _read_optional_json_dict(candidate_ensemble_summary_path)
    model_wrapper = _read_optional_json_dict(candidate_model_path)
    cv_model_version = cv_summary.get("model_version") if isinstance(cv_summary, dict) else None
    ensemble_model_version = (
        ensemble_summary.get("model_version") if isinstance(ensemble_summary, dict) else None
    )
    model_wrapper_version = (
        model_wrapper.get("model_version") if isinstance(model_wrapper, dict) else None
    )
    cv_summary_block = _nested_dict(cv_summary, "summary")
    cv_fold_count = cv_summary_block.get("fold_count")
    if cv_fold_count is None and isinstance(cv_summary, dict) and isinstance(cv_summary.get("folds"), list):
        cv_fold_count = len(cv_summary["folds"])
    cv_fold_count_int = int(cv_fold_count) if _positive_int(cv_fold_count) else 0
    cv_time_series_evidence = _cv_time_series_evidence(cv_summary, cv_fold_count_int)
    model_members = model_wrapper.get("members") if isinstance(model_wrapper, dict) else None
    model_member_count = len(model_members) if isinstance(model_members, list) else 0
    model_member_path_evidence = _model_member_path_evidence(candidate_model_dir, model_members)
    ensemble_member_count = (
        int(ensemble_summary.get("member_count"))
        if isinstance(ensemble_summary, dict) and _positive_int(ensemble_summary.get("member_count"))
        else 0
    )
    ensemble_seed_evidence = _ensemble_seed_evidence(
        ensemble_summary=ensemble_summary,
        model_members=model_members,
        expected_member_count=ensemble_member_count,
    )
    ensemble_comparison_evidence = _ensemble_vs_single_evidence(ensemble_summary)
    training_elapsed_seconds = (
        ensemble_summary.get("training_elapsed_seconds")
        if isinstance(ensemble_summary, dict)
        else None
    )
    train_time_multiplier_estimate = (
        _optional_int(ensemble_summary.get("train_time_multiplier_estimate"))
        if isinstance(ensemble_summary, dict)
        else None
    )
    inference_eval_multiplier = (
        _optional_int(ensemble_summary.get("inference_eval_multiplier"))
        if isinstance(ensemble_summary, dict)
        else None
    )
    train_time_multiplier_matches_members = (
        train_time_multiplier_estimate == ensemble_member_count
        if train_time_multiplier_estimate is not None and ensemble_member_count > 0
        else False
    )
    inference_eval_multiplier_matches_members = (
        inference_eval_multiplier == ensemble_member_count
        if inference_eval_multiplier is not None and ensemble_member_count > 0
        else False
    )
    ensemble_cost_quantification_passed = (
        _positive_float(training_elapsed_seconds)
        and train_time_multiplier_matches_members
        and inference_eval_multiplier_matches_members
    )
    required_cv_fold_count = 3
    required_ensemble_member_count = 3
    xgboost_v4_ensemble_ready = (
        isinstance(cv_summary, dict)
        and cv_model_version == "xgboost-v4"
        and cv_fold_count_int >= required_cv_fold_count
        and bool(cv_time_series_evidence["cv_time_series_ordered"])
        and bool(cv_time_series_evidence["cv_fold_metrics_present"])
        and bool(cv_time_series_evidence["cv_summary_metrics_present"])
        and isinstance(ensemble_summary, dict)
        and ensemble_summary.get("schema_version") == "xgboost_ensemble_v1"
        and ensemble_model_version == "xgboost-v4"
        and ensemble_member_count >= required_ensemble_member_count
        and ensemble_cost_quantification_passed
        and bool(ensemble_seed_evidence["ensemble_seed_evidence_passed"])
        and bool(ensemble_comparison_evidence["ensemble_comparison_passed"])
        and isinstance(model_wrapper, dict)
        and model_wrapper.get("schema_version") == "xgboost_ensemble_v1"
        and model_wrapper_version == "xgboost-v4"
        and model_member_count == ensemble_member_count
        and bool(model_member_path_evidence["member_paths_exist"])
    )
    feature_importance_evidence = _feature_importance_evidence(candidate_model_dir, model_wrapper)
    issue_57_added_feature_evidence = _issue_57_added_feature_evidence(
        model_wrapper=model_wrapper,
        stability_report_path=stability_report_path,
    )
    issue_64_signal_label_evidence = _issue_64_signal_label_evidence()
    slack_automation_evidence = _slack_status_automation_evidence(
        slack_automation_path,
        delivery_status_path=slack_delivery_status_path,
    )
    collection_risk_evidence = _collection_risk_snapshot_evidence(
        collection_risk_path,
        live_status_path=live_status_path,
        live_status=live_status,
    )
    post_readiness_latest_evidence = _post_readiness_latest_evidence(
        post_readiness_latest_path,
        promotion_audit_path=promotion_audit_path,
        objective_audit_path=output_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
    )
    required_promotion_stage_names = (
        "Stage 0: 7-day Data Readiness",
        "Stage 1: Offline Evaluation",
        "Stage 2: Cost-Adjusted Backtest",
        "Stage 3: Shadow Evaluation",
        "Stage 4: Bootstrap Decision",
        "Stage 5: Champion Cutover",
    )
    promotion_stage_passes = {
        stage_name: _stage_passed(promotion_audit, stage_name)
        for stage_name in required_promotion_stage_names
    }
    promotion_all_stages_passed = all(promotion_stage_passes.values())
    stage1_offline_eval_passed = _stage_passed(promotion_audit, "Stage 1: Offline Evaluation")
    stage1_rerun_report_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "rerun_report_exists",
    )
    stage1_same_dataset_split_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "same_dataset_split",
    )
    stage1_time_split_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "dataset_time_split_5_1_1",
    )
    stage1_required_families_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "dataset_required_families_present",
    )
    stage1_required_family_metrics_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "required_family_metrics_present",
    )
    stage1_new_market_signal_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "new_market_signal_present",
    )
    stage1_candidate_version_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "candidate_model_version",
    )
    stage1_auc_beats_champion_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "test_auc_beats_champion",
    )
    stage1_brier_beats_champion_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "test_brier_beats_champion",
    )
    stage1_calibrated_ece_passed = _audit_check_passed(
        promotion_audit,
        "Stage 1: Offline Evaluation",
        "calibrated_ece",
    )
    candidate_eval_model_provenance = _candidate_eval_model_provenance_evidence(
        promotion_audit,
        expected_model_path=candidate_model_path,
    )
    down_validation_evidence = _down_validation_evidence(
        down_validation_path,
        expected_model_path=candidate_model_path,
        expected_dataset_dir=candidate_eval_model_provenance.get("candidate_eval_dataset_dir"),
        expected_dataset_version=candidate_eval_model_provenance.get(
            "candidate_eval_dataset_version"
        ),
    )
    feature_ablation_evidence = _feature_ablation_evidence(
        feature_ablation_path,
        expected_model_path=candidate_model_path,
        expected_dataset_dir=candidate_eval_model_provenance.get("candidate_eval_dataset_dir"),
        expected_dataset_version=candidate_eval_model_provenance.get(
            "candidate_eval_dataset_version"
        ),
    )
    final_candidate_evidence_requirements = {
        "ready_for_training": _is_true(readiness.get("ready_for_training")),
        "stage1_rerun_report_exists_passed": stage1_rerun_report_passed,
        "stage1_same_dataset_split_passed": stage1_same_dataset_split_passed,
        "stage1_dataset_time_split_5_1_1_passed": stage1_time_split_passed,
        "stage1_candidate_model_version_passed": stage1_candidate_version_passed,
        "candidate_eval_model_version_matches": bool(
            candidate_eval_model_provenance["model_version_matches"]
        ),
        "candidate_eval_model_path_matches": bool(
            candidate_eval_model_provenance["model_path_matches"]
        ),
        "candidate_eval_dataset_provenance_present": bool(
            candidate_eval_model_provenance["dataset_provenance_present"]
        ),
    }
    final_candidate_evidence_ready = all(final_candidate_evidence_requirements.values())
    candidate_evidence_scope = "final" if final_candidate_evidence_ready else "provisional"
    feature_analysis_evidence = {
        **feature_ablation_evidence,
        **feature_importance_evidence,
        "evidence_scope": candidate_evidence_scope,
        "final_candidate_evidence_ready": final_candidate_evidence_ready,
        "final_candidate_evidence_requirements": final_candidate_evidence_requirements,
        "candidate_eval_model_provenance": candidate_eval_model_provenance,
        "added_features": issue_57_added_feature_evidence,
        "live_feature_schema": live_feature_schema,
    }
    stage2_cost_adjusted_backtest_passed = _stage_passed(
        promotion_audit,
        "Stage 2: Cost-Adjusted Backtest",
    )
    stage2_net_pnl_beats_champion_passed = _audit_check_passed(
        promotion_audit,
        "Stage 2: Cost-Adjusted Backtest",
        "net_pnl_beats_champion",
    )
    dataset_stability_evidence = _dataset_stability_evidence(
        stability_report_path,
        expected_dataset_dir=candidate_eval_model_provenance.get("candidate_eval_dataset_dir"),
        expected_dataset_version=candidate_eval_model_provenance.get(
            "candidate_eval_dataset_version"
        ),
    )
    issue_56_schema_evidence = _issue_56_multimarket_schema_evidence(
        model_wrapper=model_wrapper,
        stability_report_path=stability_report_path,
        required_families=required_families,
    )
    issue_65_tick_feature_evidence = _issue_65_tick_feature_evidence(
        model_wrapper=model_wrapper,
    )
    promotion_raw_passed = isinstance(promotion_audit, dict) and _is_true(promotion_audit.get("passed"))
    promotion_clean_atomic_live_root_passed = _audit_check_passed(
        promotion_audit,
        "Stage 0: 7-day Data Readiness",
        "clean_atomic_live_root",
    )
    promotion_status_artifact_fresh_passed = _audit_check_passed(
        promotion_audit,
        "Stage 0: 7-day Data Readiness",
        "status_artifact_fresh",
    )
    promotion_collector_process_liveness_passed = _audit_check_passed(
        promotion_audit,
        "Stage 0: 7-day Data Readiness",
        "collector_process_liveness",
    )
    promotion_raw_manifest_coverage_passed = _audit_check_passed(
        promotion_audit,
        "Stage 0: 7-day Data Readiness",
        "raw_manifest_coverage",
    )
    promotion_process_source_evidence = _promotion_process_source_audit_evidence(
        promotion_audit
    )
    promotion_process_source_passed = bool(promotion_process_source_evidence["passed"])
    promotion_github_issue_closures_passed = _audit_check_passed(
        promotion_audit,
        "Stage 5: Champion Cutover",
        "github_issue_closures_recorded",
    )

    issue_checks = [
        {
            "id": "#54",
            "requirement": (
                "Live champion drift and label hit-rate monitoring are wired to "
                "prediction/outcome streams, incident catalog output, and documented "
                "alert thresholds."
            ),
            "passed": (
                _is_true(outcomes.get("available"))
                and monitoring_model_version == expected_monitoring_model_version
                and monitoring_event_rows is not None
                and monitoring_event_rows > 0
                and monitoring_outcome_rows is not None
                and monitoring_outcome_rows > 0
                and aggregate_monitoring_metrics_valid
                and not missing_event_families
                and not missing_outcome_families
                and not missing_event_row_families
                and not missing_outcome_row_families
                and not monitoring_invalid_metric_families
                and bool(issue_54_monitoring_evidence["passed"])
            ),
            "evidence": {
                "model_version": monitoring_model_version or None,
                "expected_model_version": expected_monitoring_model_version,
                "event_rows": monitoring_event_rows,
                "outcome_rows": monitoring_outcome_rows,
                "brier_score": outcomes.get("brier_score"),
                "hit_rate": outcomes.get("hit_rate"),
                "avg_realized_return": outcomes.get("avg_realized_return"),
                "aggregate_metrics_valid": aggregate_monitoring_metrics_valid,
                "missing_event_families": missing_event_families,
                "missing_outcome_families": missing_outcome_families,
                "event_rows_by_family": monitoring_event_rows_by_family,
                "outcome_rows_by_family": monitoring_outcome_rows_by_family,
                "missing_event_row_families": missing_event_row_families,
                "missing_outcome_row_families": missing_outcome_row_families,
                "invalid_metric_families": monitoring_invalid_metric_families,
                "drift_monitoring": issue_54_monitoring_evidence,
            },
        },
        {
            "id": "#55",
            "requirement": "A settled seven-day corpus exists and the fresh v4 beats the current champion.",
            "passed": (
                _is_true(readiness.get("ready_for_training"))
                and not missing_readiness_required_families
                and _is_true(label_freshness.get("fresh"))
                and not labels_missing_freshness
                and not labels_unfresh
                and stage1_offline_eval_passed
                and stage1_same_dataset_split_passed
                and stage1_time_split_passed
                and stage1_candidate_version_passed
                and stage1_auc_beats_champion_passed
                and stage1_brier_beats_champion_passed
                and stage1_calibrated_ece_passed
                and stage2_cost_adjusted_backtest_passed
                and stage2_net_pnl_beats_champion_passed
                and bool(dataset_stability_evidence["passed"])
            ),
            "evidence": {
                "ready_for_training": _is_true(readiness.get("ready_for_training")),
                "estimated_ready_at": readiness.get("estimated_ready_at"),
                "required_families": required_families,
                "status_required_families": status_required_families,
                "missing_readiness_required_families": missing_readiness_required_families,
                "label_freshness_fresh": _is_true(label_freshness.get("fresh")),
                "missing_label_freshness_families": labels_missing_freshness,
                "unfresh_label_families": labels_unfresh,
                "feature_min_family_span_days": _nested_dict(
                    live_status,
                    "collection_readiness",
                    "features_15m_v1",
                ).get("min_family_span_days"),
                "label_min_family_span_days": _nested_dict(
                    live_status,
                    "collection_readiness",
                    "labels_15m_v1",
                ).get("min_family_span_days"),
                "stage1_offline_eval_passed": stage1_offline_eval_passed,
                "stage1_rerun_report_exists_passed": stage1_rerun_report_passed,
                "stage1_same_dataset_split_passed": stage1_same_dataset_split_passed,
                "stage1_dataset_time_split_5_1_1_passed": stage1_time_split_passed,
                "stage1_candidate_model_version_passed": stage1_candidate_version_passed,
                "stage1_test_auc_beats_champion_passed": stage1_auc_beats_champion_passed,
                "stage1_test_brier_beats_champion_passed": stage1_brier_beats_champion_passed,
                "stage1_calibrated_ece_passed": stage1_calibrated_ece_passed,
                "stage2_cost_adjusted_backtest_passed": stage2_cost_adjusted_backtest_passed,
                "stage2_net_pnl_beats_champion_passed": stage2_net_pnl_beats_champion_passed,
                "dataset_stability": dataset_stability_evidence,
            },
        },
        {
            "id": "#56",
            "requirement": "BTC/ETH 5m/15m market families are present, fresh, settled, and evaluated by family.",
            "passed": (
                _is_true(readiness.get("ready_for_training"))
                and not missing_readiness_required_families
                and not features_missing
                and not predictions_missing
                and not features_unfresh
                and not predictions_unfresh
                and _is_true(label_freshness.get("fresh"))
                and not labels_missing_freshness
                and not labels_unfresh
                and not missing_outcome_families
                and not missing_outcome_row_families
                and stage1_offline_eval_passed
                and stage1_required_families_passed
                and stage1_required_family_metrics_passed
                and stage1_new_market_signal_passed
                and bool(issue_56_schema_evidence["passed"])
            ),
            "evidence": {
                "required_families": required_families,
                "status_required_families": status_required_families,
                "missing_readiness_required_families": missing_readiness_required_families,
                "missing_feature_families": features_missing,
                "missing_prediction_families": predictions_missing,
                "unfresh_feature_families": features_unfresh,
                "unfresh_prediction_families": predictions_unfresh,
                "label_freshness_fresh": _is_true(label_freshness.get("fresh")),
                "missing_label_freshness_families": labels_missing_freshness,
                "unfresh_label_families": labels_unfresh,
                "missing_outcome_families": missing_outcome_families,
                "missing_outcome_row_families": missing_outcome_row_families,
                "ready_for_training": _is_true(readiness.get("ready_for_training")),
                "stage1_dataset_required_families_present_passed": stage1_required_families_passed,
                "stage1_required_family_metrics_present_passed": stage1_required_family_metrics_passed,
                "stage1_new_market_signal_present_passed": stage1_new_market_signal_passed,
                "multimarket_schema": issue_56_schema_evidence,
            },
        },
        {
            "id": "#57",
            "requirement": (
                "Fresh v4 added-feature columns, feature importance, and ablation "
                "evidence exist for the final same-dataset run."
            ),
            "passed": bool(
                final_candidate_evidence_ready
                and feature_ablation_evidence["passed"]
                and feature_importance_evidence["feature_importance_passed"]
                and issue_57_added_feature_evidence["passed"]
            ),
            "evidence": feature_analysis_evidence,
        },
        {
            "id": "#58",
            "requirement": (
                "Fresh v4 model has time-series CV, light-ensemble evidence, "
                "and training/inference cost quantification."
            ),
            "passed": final_candidate_evidence_ready and xgboost_v4_ensemble_ready,
            "evidence": {
                "evidence_scope": candidate_evidence_scope,
                "final_candidate_evidence_ready": final_candidate_evidence_ready,
                "final_candidate_evidence_requirements": final_candidate_evidence_requirements,
                "candidate_eval_model_provenance": candidate_eval_model_provenance,
                "candidate_model_dir": str(candidate_model_dir) if candidate_model_dir else None,
                "cv_summary_path": str(candidate_cv_path) if candidate_cv_path else None,
                "cv_model_version": cv_model_version,
                "cv_fold_count": cv_fold_count,
                "required_cv_fold_count": required_cv_fold_count,
                **cv_time_series_evidence,
                "ensemble_summary_path": (
                    str(candidate_ensemble_summary_path) if candidate_ensemble_summary_path else None
                ),
                "ensemble_model_version": ensemble_model_version,
                "ensemble_member_count": ensemble_member_count,
                "required_ensemble_member_count": required_ensemble_member_count,
                "training_elapsed_seconds": training_elapsed_seconds,
                "train_time_multiplier_estimate": train_time_multiplier_estimate,
                "inference_eval_multiplier": inference_eval_multiplier,
                "train_time_multiplier_matches_members": train_time_multiplier_matches_members,
                "inference_eval_multiplier_matches_members": (
                    inference_eval_multiplier_matches_members
                ),
                "ensemble_cost_quantification_passed": ensemble_cost_quantification_passed,
                "model_path": str(candidate_model_path) if candidate_model_path else None,
                "model_wrapper_version": model_wrapper_version,
                "model_member_count": model_member_count,
                **model_member_path_evidence,
                **ensemble_seed_evidence,
                **ensemble_comparison_evidence,
            },
        },
        {
            "id": "#64",
            "requirement": "BUY_DOWN/SELL safety is implemented and final DOWN-side validation evidence exists.",
            "passed": bool(
                final_candidate_evidence_ready
                and down_validation_evidence["passed"]
                and issue_64_signal_label_evidence["passed"]
            ),
            "evidence": {
                **down_validation_evidence,
                "evidence_scope": candidate_evidence_scope,
                "final_candidate_evidence_ready": final_candidate_evidence_ready,
                "final_candidate_evidence_requirements": final_candidate_evidence_requirements,
                "candidate_eval_model_provenance": candidate_eval_model_provenance,
                "signal_and_label_support": issue_64_signal_label_evidence,
            },
        },
        {
            "id": "#65",
            "requirement": (
                "The 5-second live scan loop is healthy with fresh immutable "
                "raw/processed progress and the final v4 candidate carries the "
                "required tick-level features."
            ),
            "passed": (
                screen_session_matches
                and str(live_status.get("screen_state") if isinstance(live_status, dict) else "") == "running"
                and live_root_matches
                and warehouse_matches
                and _positive_int(raw_segment_count_value)
                and _positive_int(processed_manifest_rows_value)
                and _is_true(liveness.get("raw_segments_fresh"))
                and _is_true(liveness.get("processed_manifest_fresh"))
                and invalid_raw_segments == 0
                and (quarantined_raw_segments == 0 or quarantine_clean_window_ready)
                and unrecovered_error_match_count == 0
                and disk_headroom_ok
                and current_disk_headroom_ok
                and live_root_lock_ok
                and raw_manifest_coverage_ok
                and promotion_clean_atomic_live_root_passed
                and promotion_status_artifact_fresh_passed
                and promotion_collector_process_liveness_passed
                and promotion_raw_manifest_coverage_passed
                and final_candidate_evidence_ready
                and bool(issue_65_tick_feature_evidence["passed"])
            ),
            "evidence": {
                "expected_screen_session": expected_screen_session,
                "screen_session": live_status.get("screen_session") if isinstance(live_status, dict) else None,
                "screen_session_matches": screen_session_matches,
                "screen_state": live_status.get("screen_state") if isinstance(live_status, dict) else None,
                "expected_live_root": str(expected_live_root),
                "live_root": live_status.get("live_root") if isinstance(live_status, dict) else None,
                "live_root_matches": live_root_matches,
                "expected_warehouse": str(expected_live_root / "warehouse"),
                "warehouse": live_status.get("warehouse") if isinstance(live_status, dict) else None,
                "warehouse_matches": warehouse_matches,
                "raw_segment_count": raw_segment_count_value,
                "processed_manifest_rows": processed_manifest_rows_value,
                "raw_segments_fresh": _is_true(liveness.get("raw_segments_fresh")),
                "processed_manifest_fresh": _is_true(liveness.get("processed_manifest_fresh")),
                "invalid_raw_segments": invalid_raw_segments,
                "quarantined_raw_segments": quarantined_raw_segments,
                "quarantine_clean_window_ready": quarantine_clean_window_ready,
                "quarantine_clean_window_estimated_ready_at": quarantine_clean_window.get(
                    "estimated_ready_at"
                ),
                "latest_quarantined_segment_path": latest_quarantined_segment.get("path"),
                "latest_quarantined_segment_ts": latest_quarantined_segment.get("segment_ts"),
                "latest_quarantined_gzip_valid": latest_quarantined_gzip_valid,
                "latest_quarantined_readable_prefix_lines": _optional_int(
                    latest_quarantined_gzip_probe.get("readable_prefix_lines")
                ),
                "latest_quarantined_readable_prefix_bytes": _optional_int(
                    latest_quarantined_gzip_probe.get("readable_prefix_bytes")
                ),
                "latest_quarantined_gzip_error": latest_quarantined_gzip_probe.get("error"),
                "unrecovered_error_match_count": unrecovered_error_match_count,
                "disk_headroom_ok": disk_headroom_ok,
                "disk_free_bytes": disk_headroom.get("free_bytes"),
                "disk_required_free_bytes": disk_headroom.get("required_free_bytes"),
                "disk_projected_remaining_bytes": disk_headroom.get(
                    "projected_remaining_bytes"
                ),
                "disk_headroom_margin_bytes": disk_headroom.get("headroom_margin_bytes"),
                "disk_headroom_margin_pct": disk_headroom.get("headroom_margin_pct"),
                "disk_headroom_low_margin": disk_headroom.get("headroom_low_margin"),
                "disk_low_margin_threshold_bytes": disk_headroom.get(
                    "low_margin_threshold_bytes"
                ),
                "current_disk_headroom_ok": current_disk_headroom_ok,
                "current_disk_headroom": current_disk_headroom,
                "live_root_lock_ok": live_root_lock_ok,
                "live_root_lock_pid": live_root_lock.get("pid"),
                "live_root_lock_owner_running": _is_true(
                    live_root_lock.get("owner_running")
                ),
                "live_root_lock_pid_parse_error": live_root_lock.get("pid_parse_error"),
                "raw_manifest_coverage_ok": raw_manifest_coverage_ok,
                "stale_missing_processed_count": stale_missing_processed_count,
                "extra_processed_count": extra_processed_count,
                "promotion_clean_atomic_live_root_passed": promotion_clean_atomic_live_root_passed,
                "promotion_status_artifact_fresh_passed": promotion_status_artifact_fresh_passed,
                "promotion_collector_process_liveness_passed": promotion_collector_process_liveness_passed,
                "promotion_raw_manifest_coverage_passed": promotion_raw_manifest_coverage_passed,
                "evidence_scope": candidate_evidence_scope,
                "final_candidate_evidence_ready": final_candidate_evidence_ready,
                "final_candidate_evidence_requirements": final_candidate_evidence_requirements,
                "candidate_eval_model_provenance": candidate_eval_model_provenance,
                "tick_features": issue_65_tick_feature_evidence,
                "live_feature_schema": live_feature_schema,
            },
        },
    ]
    operational_checks = [
        {
            "id": "slack_hourly_status",
            "requirement": (
                "Hourly work-status heartbeat posts to Slack channel C0B5VHYSCN8, "
                "refreshes objective-audit blocker evidence from current live and "
                "post-readiness artifacts, and invokes the post-readiness runner once "
                "the live corpus is ready."
            ),
            "passed": bool(slack_automation_evidence["passed"]),
            "evidence": slack_automation_evidence,
        },
    ]
    post_readiness_latest_pointer_passed = bool(
        post_readiness_latest_evidence.get("checked")
        and post_readiness_latest_evidence.get("exists")
        and post_readiness_latest_evidence.get("well_formed")
        and post_readiness_latest_evidence.get("matches_current_inputs")
        and post_readiness_latest_evidence.get("run_manifest_phase") == "completed"
        and post_readiness_latest_evidence.get("promotion_passed") is True
        and post_readiness_latest_evidence.get("objective_status_compatible")
    )
    promotion_passed = (
        promotion_raw_passed
        and promotion_clean_atomic_live_root_passed
        and promotion_status_artifact_fresh_passed
        and promotion_collector_process_liveness_passed
        and promotion_raw_manifest_coverage_passed
        and promotion_process_source_passed
        and promotion_github_issue_closures_passed
        and promotion_all_stages_passed
    )
    artifact_paths = {
        "objective_audit_path": str(output_path),
        "live_status_path": str(live_status_path),
        "promotion_audit_path": str(promotion_audit_path),
        "candidate_model_dir": str(candidate_model_dir) if candidate_model_dir else None,
        "feature_ablation_path": str(feature_ablation_path) if feature_ablation_path else None,
        "stability_report_path": str(stability_report_path) if stability_report_path else None,
        "down_validation_path": str(down_validation_path) if down_validation_path else None,
        "slack_automation_path": str(slack_automation_path) if slack_automation_path else None,
        "slack_delivery_status_path": (
            str(slack_delivery_status_path) if slack_delivery_status_path else None
        ),
        "collection_risk_path": str(collection_risk_path) if collection_risk_path else None,
        "post_readiness_latest_path": (
            str(post_readiness_latest_path) if post_readiness_latest_path else None
        ),
    }
    issue_checks_by_id = {str(check["id"]): check for check in issue_checks}
    operational_checks_by_id = {str(check["id"]): check for check in operational_checks}
    live_corpus_evidence_scope = (
        "final_7d_corpus" if _is_true(readiness.get("ready_for_training")) else "collecting"
    )
    promotion_evidence_scope = "passed" if promotion_passed else "blocked"
    prompt_to_artifact_checklist = [
        {
            "id": "github_issue_54",
            "source": "https://github.com/phead198708/BiGan/issues/54",
            "requirement": issue_checks_by_id["#54"]["requirement"],
            "evidence_scope": "current_live_monitoring",
            "evidence_paths": [
                artifact_paths["live_status_path"],
                "docs/runbooks/champion_promotion.md",
            ],
            "evidence_keys": [
                "issue_checks.#54",
                "live_collection.monitoring_outcome_evidence",
            ],
            "passed": bool(issue_checks_by_id["#54"]["passed"]),
        },
        {
            "id": "github_issue_55",
            "source": "https://github.com/phead198708/BiGan/issues/55",
            "requirement": issue_checks_by_id["#55"]["requirement"],
            "evidence_scope": live_corpus_evidence_scope,
            "evidence_paths": [
                artifact_paths["live_status_path"],
                artifact_paths["promotion_audit_path"],
                artifact_paths["stability_report_path"],
            ],
            "evidence_keys": [
                "issue_checks.#55",
                "promotion.Stage 0",
                "promotion.Stage 1",
                "promotion.Stage 2",
            ],
            "passed": bool(issue_checks_by_id["#55"]["passed"]),
        },
        {
            "id": "github_issue_56",
            "source": "https://github.com/phead198708/BiGan/issues/56",
            "requirement": issue_checks_by_id["#56"]["requirement"],
            "evidence_scope": live_corpus_evidence_scope,
            "evidence_paths": [
                artifact_paths["live_status_path"],
                artifact_paths["promotion_audit_path"],
                artifact_paths["stability_report_path"],
            ],
            "evidence_keys": [
                "issue_checks.#56",
                "collection_readiness.required_families",
                "promotion.Stage 1.required_family_metrics_present",
                "promotion.Stage 1.new_market_signal_present",
            ],
            "passed": bool(issue_checks_by_id["#56"]["passed"]),
        },
        {
            "id": "github_issue_57",
            "source": "https://github.com/phead198708/BiGan/issues/57",
            "requirement": issue_checks_by_id["#57"]["requirement"],
            "evidence_scope": candidate_evidence_scope,
            "evidence_paths": [
                artifact_paths["candidate_model_dir"],
                artifact_paths["feature_ablation_path"],
                "docs/models/xgboost-v4.md",
            ],
            "evidence_keys": [
                "issue_checks.#57",
                "live_collection.warehouse_schema_evidence",
                "candidate_model.feature_importance",
                "feature_ablation",
            ],
            "passed": bool(issue_checks_by_id["#57"]["passed"]),
        },
        {
            "id": "github_issue_58",
            "source": "https://github.com/phead198708/BiGan/issues/58",
            "requirement": issue_checks_by_id["#58"]["requirement"],
            "evidence_scope": candidate_evidence_scope,
            "evidence_paths": [artifact_paths["candidate_model_dir"]],
            "evidence_keys": [
                "issue_checks.#58",
                "candidate_model.cv_summary",
                "candidate_model.ensemble_summary",
                "candidate_model.model",
            ],
            "passed": bool(issue_checks_by_id["#58"]["passed"]),
        },
        {
            "id": "github_issue_64",
            "source": "https://github.com/phead198708/BiGan/issues/64",
            "requirement": issue_checks_by_id["#64"]["requirement"],
            "evidence_scope": candidate_evidence_scope,
            "evidence_paths": [artifact_paths["down_validation_path"]],
            "evidence_keys": [
                "issue_checks.#64",
                "down_validation",
                "signal_and_label_support",
            ],
            "passed": bool(issue_checks_by_id["#64"]["passed"]),
        },
        {
            "id": "github_issue_65",
            "source": "https://github.com/phead198708/BiGan/issues/65",
            "requirement": issue_checks_by_id["#65"]["requirement"],
            "evidence_scope": live_corpus_evidence_scope,
            "evidence_paths": [
                artifact_paths["live_status_path"],
                artifact_paths["candidate_model_dir"],
            ],
            "evidence_keys": [
                "issue_checks.#65",
                "live_collection",
                "live_collection.warehouse_schema_evidence",
                "raw_manifest_coverage",
                "final_candidate_evidence_requirements",
                "candidate_eval_model_provenance",
                "tick_features",
            ],
            "passed": bool(issue_checks_by_id["#65"]["passed"]),
        },
        {
            "id": "create_xgboost_v4_model",
            "source": "user_objective",
            "requirement": (
                "Create a fresh xgboost-v4 candidate model from the final settled "
                "multi-market corpus."
            ),
            "evidence_scope": candidate_evidence_scope,
            "evidence_paths": [artifact_paths["candidate_model_dir"]],
            "evidence_keys": [
                "issue_checks.#58.evidence.model_wrapper_version",
                "final_candidate_evidence_requirements",
            ],
            "passed": bool(final_candidate_evidence_ready and model_wrapper_version == "xgboost-v4"),
        },
        {
            "id": "beat_current_champion",
            "source": "user_objective",
            "requirement": (
                "The fresh xgboost-v4 candidate beats the current champion in "
                "same-dataset evaluation and cost-adjusted backtest gates."
            ),
            "evidence_scope": promotion_evidence_scope,
            "evidence_paths": [artifact_paths["promotion_audit_path"]],
            "evidence_keys": [
                "promotion.Stage 1",
                "promotion.Stage 1.test_auc_beats_champion",
                "promotion.Stage 1.test_brier_beats_champion",
                "promotion.Stage 2",
                "promotion.Stage 2.net_pnl_beats_champion",
            ],
            "passed": bool(
                stage1_offline_eval_passed
                and stage1_auc_beats_champion_passed
                and stage1_brier_beats_champion_passed
                and stage2_cost_adjusted_backtest_passed
                and stage2_net_pnl_beats_champion_passed
            ),
        },
        {
            "id": "champion_promotion_md",
            "source": "/Users/tcscoder/Downloads/champion-promotion.md",
            "requirement": (
                "Follow every champion-promotion.md gate before declaring promotion "
                "complete."
            ),
            "evidence_scope": promotion_evidence_scope,
            "evidence_paths": [
                artifact_paths["promotion_audit_path"],
                "/Users/tcscoder/Downloads/champion-promotion.md",
                "docs/runbooks/champion_promotion.md",
            ],
            "evidence_keys": [
                "promotion",
                "champion-promotion-audit.promotion_process",
                "champion-promotion-audit.stage_passes",
                "champion-promotion-audit.stages",
                "promotion.github_issue_closures_passed",
                "promotion.Stage 5.github_issue_closures_recorded",
            ],
            "passed": bool(promotion_passed),
        },
        {
            "id": "hourly_slack_status",
            "source": "C0B5VHYSCN8",
            "requirement": operational_checks_by_id["slack_hourly_status"]["requirement"],
            "evidence_scope": (
                "active" if operational_checks_by_id["slack_hourly_status"]["passed"] else "blocked"
            ),
            "evidence_paths": [artifact_paths["slack_automation_path"]],
            "evidence_keys": [
                "operational_checks.slack_hourly_status",
            ],
            "passed": bool(operational_checks_by_id["slack_hourly_status"]["passed"]),
        },
        {
            "id": "post_readiness_latest_pointer",
            "source": artifact_paths["post_readiness_latest_path"],
            "requirement": (
                "The latest post-readiness runner pointer is well-formed, completed, "
                "and matches the current objective, promotion, model, ablation, "
                "stability, DOWN-validation, and issue-coverage artifacts."
            ),
            "evidence_scope": "matched" if post_readiness_latest_pointer_passed else "blocked",
            "evidence_paths": [artifact_paths["post_readiness_latest_path"]],
            "evidence_keys": [
                "post_readiness_latest",
                "post_readiness_latest.path_matches",
                "post_readiness_latest.path_matches.issue_coverage_audit_path",
                "post_readiness_latest.run_manifest_phase",
            ],
            "passed": post_readiness_latest_pointer_passed,
        },
    ]
    failed_prompt_to_artifact_items = [
        item for item in prompt_to_artifact_checklist if not bool(item["passed"])
    ]
    checklist_by_id = {str(item["id"]): item for item in prompt_to_artifact_checklist}
    objective_success_criteria = [
        {
            "id": "all_requested_github_issues_satisfied",
            "requirement": "All requested GitHub issue requirements #54/#55/#56/#57/#58/#64/#65 pass.",
            "checklist_ids": [
                "github_issue_54",
                "github_issue_55",
                "github_issue_56",
                "github_issue_57",
                "github_issue_58",
                "github_issue_64",
                "github_issue_65",
            ],
            "passed": all(
                bool(checklist_by_id[check_id]["passed"])
                for check_id in [
                    "github_issue_54",
                    "github_issue_55",
                    "github_issue_56",
                    "github_issue_57",
                    "github_issue_58",
                    "github_issue_64",
                    "github_issue_65",
                ]
            ),
        },
        {
            "id": "fresh_xgboost_v4_model_created",
            "requirement": "A fresh xgboost-v4 model is created from the final settled corpus.",
            "checklist_ids": ["create_xgboost_v4_model"],
            "passed": bool(checklist_by_id["create_xgboost_v4_model"]["passed"]),
        },
        {
            "id": "beats_current_champion",
            "requirement": "The fresh xgboost-v4 model beats the current champion.",
            "checklist_ids": ["beat_current_champion"],
            "passed": bool(checklist_by_id["beat_current_champion"]["passed"]),
        },
        {
            "id": "champion_promotion_gates_passed",
            "requirement": "Every champion-promotion.md gate passes before promotion is declared complete.",
            "checklist_ids": ["champion_promotion_md"],
            "passed": bool(checklist_by_id["champion_promotion_md"]["passed"]),
        },
        {
            "id": "hourly_slack_status_active",
            "requirement": "Hourly Slack status updates remain active for channel C0B5VHYSCN8.",
            "checklist_ids": ["hourly_slack_status"],
            "passed": bool(checklist_by_id["hourly_slack_status"]["passed"]),
        },
        {
            "id": "post_readiness_latest_pointer_valid",
            "requirement": "The latest post-readiness run pointer matches the final evidence bundle.",
            "checklist_ids": ["post_readiness_latest_pointer"],
            "passed": bool(checklist_by_id["post_readiness_latest_pointer"]["passed"]),
        },
    ]
    objective_restatement = {
        "summary": (
            "Create a fresh xgboost-v4 model satisfying GitHub issues "
            "#54/#55/#56/#57/#58/#64/#65, prove it beats the current champion, "
            "follow champion-promotion.md, and keep hourly Slack status updates "
            "active in channel C0B5VHYSCN8."
        ),
        "requested_issue_urls": [
            "https://github.com/phead198708/BiGan/issues/54",
            "https://github.com/phead198708/BiGan/issues/55",
            "https://github.com/phead198708/BiGan/issues/56",
            "https://github.com/phead198708/BiGan/issues/57",
            "https://github.com/phead198708/BiGan/issues/58",
            "https://github.com/phead198708/BiGan/issues/64",
            "https://github.com/phead198708/BiGan/issues/65",
        ],
        "promotion_process_path": "/Users/tcscoder/Downloads/champion-promotion.md",
        "slack_channel_id": "C0B5VHYSCN8",
        "completion_rule": (
            "Only complete when every objective_success_criteria row passes and "
            "prompt_to_artifact_blockers is empty."
        ),
    }
    objective_complete = not failed_prompt_to_artifact_items
    issue_blockers = [
        f"{check['id']}: {check['requirement']}"
        for check in issue_checks
        if not bool(check["passed"])
    ]
    operational_blockers = [
        f"{check['id']}: {check['requirement']}"
        for check in operational_checks
        if not bool(check["passed"])
    ]
    prompt_to_artifact_blockers = [
        f"{item['id']}: {item['requirement']}"
        for item in failed_prompt_to_artifact_items
    ]
    blockers = (
        issue_blockers
        + operational_blockers
        + ([] if promotion_passed else ["champion-promotion.md gates have not all passed"])
        + [
            blocker
            for blocker in prompt_to_artifact_blockers
            if not blocker.startswith("github_issue_")
            and not blocker.startswith("hourly_slack_status")
            and blocker != "champion_promotion_md: Follow every champion-promotion.md gate before declaring promotion complete."
        ]
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "objective_complete": objective_complete,
        "decision": "COMPLETE" if objective_complete else "BLOCKED",
        "artifact_paths": artifact_paths,
        "objective_restatement": objective_restatement,
        "objective_success_criteria": objective_success_criteria,
        "prompt_to_artifact_blockers": prompt_to_artifact_blockers,
        "prompt_to_artifact_checklist": prompt_to_artifact_checklist,
        "live_collection": {
            "generated_at": live_status.get("generated_at") if isinstance(live_status, dict) else None,
            "ready_for_training": _is_true(readiness.get("ready_for_training")),
            "estimated_ready_at": readiness.get("estimated_ready_at"),
            "raw_segment_count": live_status.get("raw_segment_count") if isinstance(live_status, dict) else None,
            "processed_manifest_rows": (
                live_status.get("processed_manifest_rows") if isinstance(live_status, dict) else None
            ),
            "feature_rows": _nested_dict(live_status, "totals").get("features_15m_v1_rows"),
            "label_rows": _nested_dict(live_status, "totals").get("labels_15m_v1_rows"),
            "current_disk_headroom": current_disk_headroom,
            "collection_risk": collection_risk_evidence,
        },
        "post_readiness_latest": post_readiness_latest_evidence,
        "promotion": {
            "decision": promotion_audit.get("decision") if isinstance(promotion_audit, dict) else None,
            "passed": promotion_passed,
            "raw_passed": promotion_raw_passed,
            "clean_atomic_live_root_passed": promotion_clean_atomic_live_root_passed,
            "status_artifact_fresh_passed": promotion_status_artifact_fresh_passed,
            "collector_process_liveness_passed": promotion_collector_process_liveness_passed,
            "raw_manifest_coverage_passed": promotion_raw_manifest_coverage_passed,
            "promotion_process_source_passed": promotion_process_source_passed,
            "promotion_process_source": promotion_process_source_evidence,
            "all_required_stages_passed": promotion_all_stages_passed,
            "stage_passes": promotion_stage_passes,
            "github_issue_closures_passed": promotion_github_issue_closures_passed,
            "earliest_failed_stage": _earliest_failed_stage(promotion_audit),
        },
        "issue_checks": issue_checks,
        "operational_checks": operational_checks,
        "blockers": blockers,
    }


def _stage_check_evidence(
    promotion_audit: dict[str, Any] | None,
    stage_name: str,
    check_name: str,
) -> dict[str, Any] | None:
    stages = promotion_audit.get("stages") if isinstance(promotion_audit, dict) else None
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if not isinstance(stage, dict) or str(stage.get("name") or "") != stage_name:
            continue
        checks = stage.get("checks")
        if not isinstance(checks, list):
            return None
        for check in checks:
            if not isinstance(check, dict) or str(check.get("name") or "") != check_name:
                continue
            return {
                "passed": _is_true(check.get("passed")),
                "detail": check.get("detail"),
                "artifact_path": check.get("artifact_path"),
            }
    return None


def _family_names_from_status(
    live_status: dict[str, Any] | None,
    table_name: str,
) -> list[str]:
    family_counts = _nested_dict(live_status, "family_counts", table_name)
    families: set[str] = {str(family) for family in family_counts}
    warehouse_table = _nested_dict(live_status, "warehouse_freshness_evidence", "tables", table_name)
    table_families = warehouse_table.get("families")
    if isinstance(table_families, dict):
        families.update(str(family) for family in table_families)
    if table_name == "labels_15m_v1":
        label_families = _nested_dict(live_status, "label_freshness_evidence").get("families")
        if isinstance(label_families, dict):
            families.update(str(family) for family in label_families)
    return sorted(families)


def _compact_readiness_families(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    families = block.get("families") if isinstance(block.get("families"), dict) else {}
    compact: dict[str, dict[str, Any]] = {}
    for family, evidence in families.items():
        if not isinstance(evidence, dict):
            continue
        remaining_target_ms = evidence.get("remaining_target_ms")
        compact[str(family)] = {
            "rows": evidence.get("rows"),
            "span_days": evidence.get("span_days"),
            "remaining_target_days": (
                None
                if _optional_float(remaining_target_ms) is None
                else _optional_float(remaining_target_ms) / 86_400_000
            ),
            "estimated_ready_at": evidence.get("estimated_ready_at"),
            "meets_target": _is_true(evidence.get("meets_target")),
        }
    return compact


def _build_xgboost_v4_issue_coverage_audit(
    *,
    live_status: dict[str, Any] | None,
    promotion_audit: dict[str, Any] | None,
    objective_audit: dict[str, Any] | None,
    live_status_path: Path,
    promotion_audit_path: Path,
    objective_audit_path: Path,
) -> dict[str, Any]:
    readiness = _nested_dict(live_status, "collection_readiness")
    feature_readiness = _nested_dict(readiness, "features_15m_v1")
    label_readiness = _nested_dict(readiness, "labels_15m_v1")
    family_counts = _nested_dict(live_status, "family_counts")
    monitoring = _nested_dict(live_status, "monitoring_outcome_evidence")
    current_disk_headroom = _nested_dict(
        objective_audit,
        "live_collection",
        "current_disk_headroom",
    )
    collection_risk = _nested_dict(
        objective_audit,
        "live_collection",
        "collection_risk",
    )
    issue_checks = {}
    raw_issue_checks = objective_audit.get("issue_checks") if isinstance(objective_audit, dict) else None
    if isinstance(raw_issue_checks, list):
        for check in raw_issue_checks:
            if not isinstance(check, dict):
                continue
            check_id = str(check.get("id") or "")
            if not check_id:
                continue
            issue_checks[check_id] = {
                "passed": _is_true(check.get("passed")),
                "requirement": check.get("requirement"),
                "evidence_scope": _nested_dict(check, "evidence").get("evidence_scope"),
                "blocker": not _is_true(check.get("passed")),
            }
    objective_success_criteria = {}
    raw_success_criteria = (
        objective_audit.get("objective_success_criteria")
        if isinstance(objective_audit, dict)
        else None
    )
    if isinstance(raw_success_criteria, list):
        for item in raw_success_criteria:
            if not isinstance(item, dict):
                continue
            criterion_id = str(item.get("id") or "")
            if not criterion_id:
                continue
            objective_success_criteria[criterion_id] = {
                "passed": _is_true(item.get("passed")),
                "requirement": item.get("requirement"),
                "checklist_ids": item.get("checklist_ids") if isinstance(item.get("checklist_ids"), list) else [],
            }
    stages = promotion_audit.get("stages") if isinstance(promotion_audit, dict) else None
    stage_passes = {
        str(stage.get("name")): _is_true(stage.get("passed"))
        for stage in stages
        if isinstance(stage, dict) and stage.get("name") is not None
    } if isinstance(stages, list) else {}
    stage0_failed_checks = [
        str(check.get("name"))
        for stage in stages or []
        if isinstance(stage, dict) and str(stage.get("name") or "") == "Stage 0: 7-day Data Readiness"
        for check in (stage.get("checks") if isinstance(stage.get("checks"), list) else [])
        if isinstance(check, dict) and not _is_true(check.get("passed")) and check.get("name") is not None
    ]
    prompt_blockers = (
        objective_audit.get("prompt_to_artifact_blockers")
        if isinstance(objective_audit, dict)
        and isinstance(objective_audit.get("prompt_to_artifact_blockers"), list)
        else []
    )
    objective_blockers = (
        objective_audit.get("blockers")
        if isinstance(objective_audit, dict)
        and isinstance(objective_audit.get("blockers"), list)
        else []
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_artifacts": {
            "live_status_path": str(live_status_path),
            "live_status_generated_at": (
                live_status.get("generated_at") if isinstance(live_status, dict) else None
            ),
            "promotion_audit_path": str(promotion_audit_path),
            "objective_audit_path": str(objective_audit_path),
            "objective_audit_generated_at": (
                objective_audit.get("generated_at") if isinstance(objective_audit, dict) else None
            ),
            "champion_promotion_path": _nested_dict(promotion_audit, "promotion_process").get(
                "source_path",
                str(DEFAULT_CHAMPION_PROMOTION_PROCESS_PATH),
            ),
        },
        "summary": {
            "decision": objective_audit.get("decision") if isinstance(objective_audit, dict) else None,
            "objective_complete": _is_true(
                objective_audit.get("objective_complete") if isinstance(objective_audit, dict) else None
            ),
            "seven_day_requirement_met": _is_true(readiness.get("ready_for_training")),
            "ready_for_training": _is_true(readiness.get("ready_for_training")),
            "estimated_ready_at": readiness.get("estimated_ready_at"),
            "required_families": readiness.get("required_families") if isinstance(readiness.get("required_families"), list) else [],
            "feature_families_seen": _family_names_from_status(live_status, "features_15m_v1"),
            "label_families_seen": _family_names_from_status(live_status, "labels_15m_v1"),
            "prediction_families_seen": _family_names_from_status(live_status, "predictions"),
            "monitoring_event_families_seen": sorted(
                str(family)
                for family in (
                    monitoring.get("families")
                    if isinstance(monitoring.get("families"), dict)
                    else {}
                )
            ),
            "missing_event_families": monitoring.get("missing_event_families") or [],
            "missing_outcome_families": monitoring.get("missing_outcome_families") or [],
            "has_eth_features": any(
                family.startswith("ETH-") for family in _family_names_from_status(live_status, "features_15m_v1")
            ),
            "has_non_15m_features": any(
                not family.endswith("-15M")
                for family in _family_names_from_status(live_status, "features_15m_v1")
            ),
            "feature_min_span_days": feature_readiness.get("min_family_span_days"),
            "label_min_span_days": label_readiness.get("min_family_span_days"),
            "feature_target_progress_pct": feature_readiness.get("target_progress_pct"),
            "label_target_progress_pct": label_readiness.get("target_progress_pct"),
            "feature_remaining_target_days": feature_readiness.get("remaining_target_days"),
            "label_remaining_target_days": label_readiness.get("remaining_target_days"),
            "feature_limiting_family": feature_readiness.get("limiting_family"),
            "label_limiting_family": label_readiness.get("limiting_family"),
            "raw_quarantine_clean_window": readiness.get("quarantine_clean_window"),
            "raw_segment_integrity": _nested_dict(live_status, "raw_segment_integrity"),
            "liveness_evidence": _nested_dict(live_status, "liveness_evidence"),
            "disk_headroom": _nested_dict(live_status, "disk_headroom_evidence"),
            "current_disk_headroom": current_disk_headroom,
            "collection_risk": collection_risk,
            "monitoring_outcome_evidence": {
                "event_rows": monitoring.get("event_rows"),
                "outcome_rows": monitoring.get("outcome_rows"),
                "outcome_coverage_pct": monitoring.get("outcome_coverage_pct"),
                "brier_score": monitoring.get("brier_score"),
                "hit_rate": monitoring.get("hit_rate"),
                "avg_realized_return": monitoring.get("avg_realized_return"),
            },
            "blocker_count": len(objective_blockers),
            "prompt_to_artifact_blocker_count": len(prompt_blockers),
        },
        "family_counts": family_counts,
        "family_spans": {
            "features_15m_v1": _compact_readiness_families(feature_readiness),
            "labels_15m_v1": _compact_readiness_families(label_readiness),
        },
        "issue_checks": issue_checks,
        "objective_success_criteria": objective_success_criteria,
        "prompt_to_artifact_blockers": prompt_blockers,
        "promotion": {
            "decision": promotion_audit.get("decision") if isinstance(promotion_audit, dict) else None,
            "passed": _is_true(promotion_audit.get("passed") if isinstance(promotion_audit, dict) else None),
            "earliest_failed_stage": _earliest_failed_stage(promotion_audit),
            "promotion_process": _nested_dict(promotion_audit, "promotion_process"),
            "stage_passes": stage_passes,
            "stage0_failed_checks": stage0_failed_checks,
            "stage1_required_checks": {
                "rerun_report_exists": _stage_check_evidence(
                    promotion_audit,
                    "Stage 1: Offline Evaluation",
                    "rerun_report_exists",
                ),
                "same_dataset_split": _stage_check_evidence(
                    promotion_audit,
                    "Stage 1: Offline Evaluation",
                    "same_dataset_split",
                ),
                "dataset_time_split_5_1_1": _stage_check_evidence(
                    promotion_audit,
                    "Stage 1: Offline Evaluation",
                    "dataset_time_split_5_1_1",
                ),
                "required_family_metrics_present": _stage_check_evidence(
                    promotion_audit,
                    "Stage 1: Offline Evaluation",
                    "required_family_metrics_present",
                ),
            },
        },
        "warehouses": [
            {
                "warehouse": live_status.get("warehouse") if isinstance(live_status, dict) else None,
                "live_root": live_status.get("live_root") if isinstance(live_status, dict) else None,
                "screen_session": live_status.get("screen_session") if isinstance(live_status, dict) else None,
                "generated_at": live_status.get("generated_at") if isinstance(live_status, dict) else None,
                "features_15m_v1": {
                    "rows": _nested_dict(live_status, "totals").get("features_15m_v1_rows"),
                    "families": _nested_dict(family_counts, "features_15m_v1"),
                    "min_family_span_days": feature_readiness.get("min_family_span_days"),
                    "limiting_family": feature_readiness.get("limiting_family"),
                    "target_progress_pct": feature_readiness.get("target_progress_pct"),
                },
                "labels_15m_v1": {
                    "rows": _nested_dict(live_status, "totals").get("labels_15m_v1_rows"),
                    "families": _nested_dict(family_counts, "labels_15m_v1"),
                    "min_family_span_days": label_readiness.get("min_family_span_days"),
                    "limiting_family": label_readiness.get("limiting_family"),
                    "target_progress_pct": label_readiness.get("target_progress_pct"),
                },
                "predictions": {
                    "rows": _nested_dict(live_status, "totals").get("predictions_rows"),
                    "families": _nested_dict(family_counts, "predictions"),
                },
            }
        ],
    }


@app.command("xgboost-v4-objective-audit")
def xgboost_v4_objective_audit(
    output_path: Path = XGBOOST_V4_OBJECTIVE_AUDIT_OUTPUT_PATH_OPTION,
    live_status_path: Path = CHAMPION_PROMOTION_LIVE_STATUS_PATH_OPTION,
    promotion_audit_path: Path = XGBOOST_V4_OBJECTIVE_PROMOTION_AUDIT_PATH_OPTION,
    candidate_model_dir: Path | None = XGBOOST_V4_OBJECTIVE_CANDIDATE_MODEL_DIR_OPTION,
    feature_ablation_path: Path | None = XGBOOST_V4_OBJECTIVE_FEATURE_ABLATION_PATH_OPTION,
    stability_report_path: Path | None = XGBOOST_V4_OBJECTIVE_STABILITY_REPORT_PATH_OPTION,
    down_validation_path: Path | None = XGBOOST_V4_OBJECTIVE_DOWN_VALIDATION_PATH_OPTION,
    slack_automation_path: Path | None = XGBOOST_V4_OBJECTIVE_SLACK_AUTOMATION_PATH_OPTION,
    slack_delivery_status_path: Path | None = (
        XGBOOST_V4_OBJECTIVE_SLACK_DELIVERY_STATUS_PATH_OPTION
    ),
    collection_risk_path: Path | None = XGBOOST_V4_OBJECTIVE_COLLECTION_RISK_PATH_OPTION,
    post_readiness_latest_path: Path | None = (
        XGBOOST_V4_OBJECTIVE_POST_READINESS_LATEST_PATH_OPTION
    ),
    no_fail_on_blocked: bool = CHAMPION_PROMOTION_NO_FAIL_ON_BLOCKED_OPTION,
) -> None:
    """Write a prompt-to-artifact completion audit for the active xgboost-v4 objective."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    candidate_model_dir = candidate_model_dir if isinstance(candidate_model_dir, Path) else None
    feature_ablation_path = feature_ablation_path if isinstance(feature_ablation_path, Path) else None
    stability_report_path = stability_report_path if isinstance(stability_report_path, Path) else None
    down_validation_path = down_validation_path if isinstance(down_validation_path, Path) else None
    slack_automation_path = (
        slack_automation_path
        if isinstance(slack_automation_path, Path)
        else DEFAULT_XGBOOST_V4_SLACK_AUTOMATION_PATH
    )
    slack_delivery_status_path = _resolve_xgboost_v4_slack_delivery_status_path(
        slack_delivery_status_path if isinstance(slack_delivery_status_path, Path) else None,
        slack_automation_path=slack_automation_path,
    )
    collection_risk_path = _resolve_xgboost_v4_collection_risk_path(
        collection_risk_path if isinstance(collection_risk_path, Path) else None,
        slack_automation_path=slack_automation_path,
    )
    post_readiness_latest_path = (
        post_readiness_latest_path
        if isinstance(post_readiness_latest_path, Path)
        else DEFAULT_XGBOOST_V4_POST_READINESS_LATEST_PATH
    )
    no_fail_on_blocked = bool(no_fail_on_blocked) if isinstance(no_fail_on_blocked, bool) else False
    _require_existing_file("live_status_path", live_status_path)
    _require_existing_file("promotion_audit_path", promotion_audit_path)
    live_status = _read_json_file(live_status_path)
    promotion_audit = _read_json_file(promotion_audit_path)
    report = _build_xgboost_v4_objective_audit(
        live_status=live_status if isinstance(live_status, dict) else None,
        promotion_audit=promotion_audit if isinstance(promotion_audit, dict) else None,
        output_path=output_path,
        live_status_path=live_status_path,
        promotion_audit_path=promotion_audit_path,
        candidate_model_dir=candidate_model_dir,
        feature_ablation_path=feature_ablation_path,
        stability_report_path=stability_report_path,
        down_validation_path=down_validation_path,
        slack_automation_path=slack_automation_path,
        slack_delivery_status_path=slack_delivery_status_path,
        collection_risk_path=collection_risk_path,
        post_readiness_latest_path=post_readiness_latest_path,
    )
    _write_json_file_atomic(output_path, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["objective_complete"] and not no_fail_on_blocked:
        raise Exit(code=1)


@app.command("xgboost-v4-issue-coverage-audit")
def xgboost_v4_issue_coverage_audit(
    output_path: Path = XGBOOST_V4_ISSUE_COVERAGE_AUDIT_OUTPUT_PATH_OPTION,
    live_status_path: Path = CHAMPION_PROMOTION_LIVE_STATUS_PATH_OPTION,
    promotion_audit_path: Path = XGBOOST_V4_OBJECTIVE_PROMOTION_AUDIT_PATH_OPTION,
    objective_audit_path: Path = XGBOOST_V4_OBJECTIVE_AUDIT_OUTPUT_PATH_OPTION,
    no_fail_on_blocked: bool = CHAMPION_PROMOTION_NO_FAIL_ON_BLOCKED_OPTION,
) -> None:
    """Write current issue-to-artifact coverage evidence for the active xgboost-v4 objective."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    no_fail_on_blocked = bool(no_fail_on_blocked) if isinstance(no_fail_on_blocked, bool) else False
    _require_existing_file("live_status_path", live_status_path)
    _require_existing_file("promotion_audit_path", promotion_audit_path)
    _require_existing_file("objective_audit_path", objective_audit_path)
    live_status = _read_json_file(live_status_path)
    promotion_audit = _read_json_file(promotion_audit_path)
    objective_audit = _read_json_file(objective_audit_path)
    report = _build_xgboost_v4_issue_coverage_audit(
        live_status=live_status if isinstance(live_status, dict) else None,
        promotion_audit=promotion_audit if isinstance(promotion_audit, dict) else None,
        objective_audit=objective_audit if isinstance(objective_audit, dict) else None,
        live_status_path=live_status_path,
        promotion_audit_path=promotion_audit_path,
        objective_audit_path=objective_audit_path,
    )
    _write_json_file_atomic(output_path, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not _is_true(report["summary"].get("objective_complete")) and not no_fail_on_blocked:
        raise Exit(code=1)


@app.command("champion-state-snapshot-v1")
def champion_state_snapshot_v1(
    output_path: Path = CHAMPION_STATE_SNAPSHOT_OUTPUT_PATH_OPTION,
    monitoring_db_path: Path = PREDICTION_MONITORING_DB_PATH_OPTION,
    model_family: str = CHAMPION_CUTOVER_MODEL_FAMILY_OPTION,
    environment: str = CHAMPION_CUTOVER_ENVIRONMENT_OPTION,
) -> None:
    """Write current registry champion and online model state for baseline evidence."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    conn = connect_mlops_db(monitoring_db_path)
    initialize_mlops_db(conn)
    champion = current_champion(conn, model_family)
    online = current_online_model(conn, environment)
    if champion is None:
        typer.echo(f"no current champion found for model_family={model_family}", err=True)
        raise Exit(code=1)
    if online is None:
        typer.echo(f"no current online model found for environment={environment}", err=True)
        raise Exit(code=1)
    rollback_to_version = online.get("rollback_to_version")
    fallback_model = (
        model_by_version(conn, str(rollback_to_version))
        if rollback_to_version is not None
        else None
    )

    snapshot = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mlops_db_path": str(monitoring_db_path),
        "model_family": model_family,
        "environment": environment,
        "registry_champion": champion,
        "online_model": online,
        "fallback_registry_model": fallback_model,
    }
    _write_json_file_atomic(output_path, snapshot)
    typer.echo(json.dumps(snapshot, indent=2, sort_keys=True))


@app.command("champion-cutover-report-v1")
def champion_cutover_report_v1(
    output_path: Path = CHAMPION_CUTOVER_REPORT_OUTPUT_PATH_OPTION,
    monitoring_db_path: Path = PREDICTION_MONITORING_DB_PATH_OPTION,
    model_family: str = CHAMPION_CUTOVER_MODEL_FAMILY_OPTION,
    environment: str = CHAMPION_CUTOVER_ENVIRONMENT_OPTION,
    smoke_path: Path = CHAMPION_CUTOVER_SMOKE_PATH_OPTION,
    drift_baseline_path: Path = CHAMPION_CUTOVER_DRIFT_BASELINE_PATH_OPTION,
    bootstrap_decision_path: Path = CHAMPION_CUTOVER_BOOTSTRAP_DECISION_PATH_OPTION,
    shadow_evaluation_path: Path = CHAMPION_CUTOVER_SHADOW_EVALUATION_PATH_OPTION,
    serving_readiness_path: Path = CHAMPION_CUTOVER_SERVING_READINESS_PATH_OPTION,
    github_issue_closures_path: Path | None = CHAMPION_CUTOVER_GITHUB_ISSUE_CLOSURES_PATH_OPTION,
) -> None:
    """Write Stage 5 cutover JSON from actual registry/deployment state."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    for name, path in (
        ("smoke_path", smoke_path),
        ("drift_baseline_path", drift_baseline_path),
        ("bootstrap_decision_path", bootstrap_decision_path),
        ("shadow_evaluation_path", shadow_evaluation_path),
        ("serving_readiness_path", serving_readiness_path),
    ):
        _require_existing_file(name, path)
    if not isinstance(github_issue_closures_path, Path):
        raise typer.BadParameter(
            "github_issue_closures_path is required for Stage 5 cutover report"
        )
    _require_existing_file("github_issue_closures_path", github_issue_closures_path)
    conn = connect_mlops_db(monitoring_db_path)
    initialize_mlops_db(conn)
    champion = current_champion(conn, model_family)
    online = current_online_model(conn, environment)
    if champion is None:
        typer.echo(f"no current champion found for model_family={model_family}", err=True)
        raise Exit(code=1)
    if online is None:
        typer.echo(f"no current online model found for environment={environment}", err=True)
        raise Exit(code=1)
    rollback_to_version = online.get("rollback_to_version")
    fallback_model = (
        model_by_version(conn, str(rollback_to_version))
        if rollback_to_version is not None
        else None
    )
    smoke = _validate_cutover_smoke_payload(
        _read_json_file(smoke_path),
        champion,
        smoke_path=smoke_path,
    )
    _validate_cutover_evidence_payloads(
        bootstrap_path=bootstrap_decision_path,
        shadow_path=shadow_evaluation_path,
        serving_path=serving_readiness_path,
    )
    github_issue_closures = _validate_cutover_github_issue_closures(
        _read_json_file(github_issue_closures_path),
        expected_candidate_model_version=str(champion.get("model_version") or ""),
        github_issue_closures_path=github_issue_closures_path,
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mlops_db_path": str(monitoring_db_path),
        "model_family": model_family,
        "environment": environment,
        "current_champion": champion,
        "current_online_model": online,
        "fallback_registry_model": fallback_model,
        "smoke": smoke,
        "drift_baseline_path": str(drift_baseline_path),
        "evidence": {
            "smoke": str(smoke_path),
            "bootstrap": str(bootstrap_decision_path),
            "shadow": str(shadow_evaluation_path),
            "serving_readiness": str(serving_readiness_path),
        },
    }
    report["github_issue_closures"] = github_issue_closures
    report["evidence"]["github_issue_closures"] = str(github_issue_closures_path)
    _write_json_file_atomic(output_path, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("drift-baseline-v1")
def drift_baseline_v1(
    offline_reference_path: Path = DRIFT_BASELINE_OFFLINE_REFERENCE_PATH_OPTION,
    output_path: Path = DRIFT_BASELINE_OUTPUT_PATH_OPTION,
) -> None:
    """Build champion drift baseline JSON from candidate offline validation evidence."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    _require_existing_file("offline_reference_path", offline_reference_path)
    baseline = build_champion_drift_baseline(
        str(offline_reference_path),
        str(output_path),
    )
    typer.echo(json.dumps(baseline, indent=2, sort_keys=True))


@app.command("live-monitoring-v1")
def live_monitoring_v1(
    monitoring_db_path: Path = PREDICTION_MONITORING_DB_PATH_OPTION,
    model_version: str = typer.Option(
        ACTIVE_CHAMPION_MODEL_VERSION,
        help="Champion model_version to evaluate for drift and label hit-rate alerts.",
    ),
    output_path: Path | None = LIVE_MONITORING_OUTPUT_PATH_OPTION,
    record_incidents: bool = LIVE_MONITORING_RECORD_INCIDENTS_OPTION,
) -> None:
    """Evaluate live champion drift and label hit-rate monitoring."""

    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    with connect_mlops_db(monitoring_db_path) as conn:
        report = run_live_champion_monitoring(
            conn,
            model_version=model_version,
            record_incidents=record_incidents,
        )
    if output_path is not None:
        _write_json_file_atomic(output_path, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("bootstrap-champion-v1")
def bootstrap_champion_v1(
    baseline_dir: Path = PROMOTION_BASELINE_DIR_OPTION,
    candidate_dir: Path = PROMOTION_CANDIDATE_DIR_OPTION,
    calibration_dir: Path = PROMOTION_CALIBRATION_DIR_OPTION,
    candidate_backtest_summary_path: Path = PROMOTION_BACKTEST_SUMMARY_OPTION,
    output_dir: Path = BOOTSTRAP_OUTPUT_DIR_OPTION,
    baseline_backtest_summary_path: Path | None = BOOTSTRAP_BASELINE_BACKTEST_SUMMARY_OPTION,
    serving_readiness_path: Path | None = BOOTSTRAP_SERVING_READINESS_PATH_OPTION,
    feature_schema_path: Path | None = BOOTSTRAP_FEATURE_SCHEMA_PATH_OPTION,
    model_complexity_notes_path: Path | None = BOOTSTRAP_MODEL_COMPLEXITY_NOTES_PATH_OPTION,
    shadow_evaluation_path: Path | None = SHADOW_EVALUATION_JSON_OUTPUT_PATH_OPTION,
    rollback_runbook_path: Path | None = BOOTSTRAP_ROLLBACK_RUNBOOK_PATH_OPTION,
    baseline_type: str = typer.Option("logistic regression baseline", help="Human-readable baseline type."),
    baseline_explicit: bool = typer.Option(
        True,
        "--baseline-explicit/--baseline-inferred",
        help="Whether the supplied baseline is an explicit project baseline.",
    ),
    replace_champion: bool = BOOTSTRAP_REPLACE_CHAMPION_OPTION,
) -> None:
    """Evaluate whether a candidate is ready to become the first champion."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = evaluate_bootstrap_champion(
        baseline_dir=baseline_dir,
        candidates=(
            BootstrapCandidateInput(
                candidate_dir=candidate_dir,
                calibration_dir=calibration_dir,
                candidate_backtest_summary_path=candidate_backtest_summary_path,
                serving_readiness_path=serving_readiness_path,
                feature_schema_path=feature_schema_path,
                model_complexity_notes_path=model_complexity_notes_path,
                shadow_evaluation_path=shadow_evaluation_path,
            ),
        ),
        baseline_backtest_summary_path=baseline_backtest_summary_path,
        rollback_runbook_path=rollback_runbook_path,
        baseline_type=baseline_type,
        baseline_explicit=baseline_explicit,
        promotion_action="replace_champion" if replace_champion else "first_champion",
        output_dir=output_dir,
    )
    typer.echo(report.to_markdown())


@app.command("predictions-v1")
def predictions_v1(
    model_path: Path = PREDICTION_MODEL_PATH_OPTION,
    calibration_path: Path | None = PREDICTION_CALIBRATION_PATH_OPTION,
    monitoring_db_path: Path | None = PREDICTION_MONITORING_DB_PATH_OPTION,
    write_monitoring_events: bool = PREDICTION_WRITE_MONITORING_EVENTS_OPTION,
    lookback_minutes: float | None = typer.Option(
        None,
        help="Only score features newer than now minus this many minutes.",
    ),
    since_ms: int | None = typer.Option(
        None,
        help="Only score features with feature_ts >= this UTC ms timestamp.",
    ),
    until_ms: int | None = typer.Option(
        None,
        help="Only score features with feature_ts < this UTC ms timestamp.",
    ),
    skip_existing_monitoring_events: bool = typer.Option(
        False,
        "--skip-existing-monitoring-events/--replace-monitoring-events",
        help="Do not rewrite prediction_events that already exist.",
    ),
    skip_existing_predictions: bool = typer.Option(
        False,
        "--skip-existing-predictions/--replace-predictions",
        help="Do not append prediction rows whose source/source_symbol/timestamp/model already exist.",
    ),
    max_rows_per_partition: int = typer.Option(
        50_000,
        help="Flush prediction partitions after this many rows.",
    ),
) -> None:
    """Generate predictions table rows from features_15m_v1."""
    if lookback_minutes is not None and lookback_minutes <= 0:
        raise typer.BadParameter("--lookback-minutes must be positive")
    if lookback_minutes is not None and since_ms is not None:
        raise typer.BadParameter("pass either --lookback-minutes or --since-ms, not both")
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    lower_bound_ms = (
        int(time.time() * 1000 - lookback_minutes * 60_000)
        if lookback_minutes is not None
        else since_ms
    )
    report = run_prediction_batch(
        settings.warehouse_dir,
        model_path,
        calibration_path=calibration_path,
        monitoring_db_path=monitoring_db_path if write_monitoring_events else None,
        max_rows_per_partition=max_rows_per_partition,
        since_ms=lower_bound_ms,
        until_ms=until_ms,
        skip_existing_monitoring_events=skip_existing_monitoring_events,
        skip_existing_predictions=skip_existing_predictions,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("signals-tail")
def signals_tail(
    monitoring_db_path: Path = PREDICTION_MONITORING_DB_PATH_OPTION,
    model_version: str = typer.Option(
        ACTIVE_CHAMPION_MODEL_VERSION,
        help="Champion model_version to tail from prediction_events.",
    ),
    edge_threshold: float = typer.Option(
        0.30,
        help="Emit BUY_UP when prob_up_15m - market_implied_prob is at least this value.",
    ),
    outcome_side: str = typer.Option(
        "UP",
        help="Outcome side to display: UP, DOWN, or ANY.",
    ),
    replay_last: int = typer.Option(
        20,
        help="Print the latest N existing signals before following new events.",
    ),
    poll_seconds: float = typer.Option(
        5.0,
        help="Seconds to wait between polling prediction_events.",
    ),
    limit: int = typer.Option(
        100,
        help="Maximum new events to print per poll.",
    ),
    once: bool = typer.Option(
        False,
        "--once/--follow",
        help="Print replay rows and exit instead of following new events.",
    ),
) -> None:
    """Tail BUY_UP/HOLD signals from champion prediction_events."""

    if replay_last < 0:
        raise typer.BadParameter("--replay-last must be non-negative")
    outcome_side = outcome_side.upper()
    if outcome_side not in {"UP", "DOWN", "ANY"}:
        raise typer.BadParameter("--outcome-side must be UP, DOWN, or ANY")
    if poll_seconds <= 0:
        raise typer.BadParameter("--poll-seconds must be positive")
    if limit <= 0:
        raise typer.BadParameter("--limit must be positive")

    typer.echo("streaming champion signals from prediction_events")
    typer.echo(
        f"db={monitoring_db_path} model={model_version} "
        f"outcome_side={outcome_side} edge_threshold={edge_threshold}"
    )
    cursor_created_at = 0
    cursor_event_id = ""

    if replay_last:
        rows = read_recent_signal_rows(
            monitoring_db_path,
            model_version=model_version,
            edge_threshold=edge_threshold,
            outcome_side=outcome_side,
            limit=replay_last,
        )
        if rows:
            typer.echo(f"latest {len(rows)} signals:")
            _print_signal_rows(rows)
            cursor_created_at = rows[-1].created_at
            cursor_event_id = rows[-1].event_id
        else:
            typer.echo("no existing signals found")
            cursor_created_at, cursor_event_id = latest_signal_cursor(
                monitoring_db_path,
                model_version=model_version,
                outcome_side=outcome_side,
            )
    else:
        cursor_created_at, cursor_event_id = latest_signal_cursor(
            monitoring_db_path,
            model_version=model_version,
            outcome_side=outcome_side,
        )
        typer.echo(f"starting after created_at={cursor_created_at}")

    if once:
        return

    while True:
        try:
            rows = read_signal_rows_after(
                monitoring_db_path,
                model_version=model_version,
                after_created_at=cursor_created_at,
                after_event_id=cursor_event_id,
                edge_threshold=edge_threshold,
                outcome_side=outcome_side,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001 - terminal tail should wait through writer locks.
            typer.echo(f"db busy/read error: {exc}", err=True)
            time.sleep(poll_seconds)
            continue
        if rows:
            _print_signal_rows(rows)
            cursor_created_at = rows[-1].created_at
            cursor_event_id = rows[-1].event_id
        else:
            typer.echo("waiting for new prediction events...")
        time.sleep(poll_seconds)


def _print_signal_rows(rows: tuple[Any, ...]) -> None:
    for row in rows:
        typer.echo(format_signal_row(row))


@app.command("signals-dashboard")
def signals_dashboard(
    monitoring_db_path: Path = PREDICTION_MONITORING_DB_PATH_OPTION,
    model_version: str = typer.Option(
        ACTIVE_CHAMPION_MODEL_VERSION,
        help="Champion model_version to display.",
    ),
    edge_threshold: float = typer.Option(
        0.30,
        help="Open a paper BUY_UP position when edge is at least this value.",
    ),
    exit_edge_threshold: float = typer.Option(
        0.10,
        help="Show a paper SELL when an open position's edge falls to this value or below.",
    ),
    outcome_side: str = typer.Option(
        "UP",
        help="Outcome side to display: UP, DOWN, or ANY.",
    ),
    lookback_hours: float = typer.Option(
        6.0,
        help="Recent monitoring window used for dashboard state.",
    ),
    limit: int = typer.Option(
        1_000,
        help="Maximum recent prediction events to read.",
    ),
    poll_seconds: float = typer.Option(
        5.0,
        help="Seconds to wait between dashboard refreshes.",
    ),
    once: bool = typer.Option(
        False,
        "--once/--follow",
        help="Render one dashboard snapshot and exit.",
    ),
) -> None:
    """Render a live terminal dashboard with current round and paper PnL."""

    outcome_side = outcome_side.upper()
    if outcome_side not in {"UP", "DOWN", "ANY"}:
        raise typer.BadParameter("--outcome-side must be UP, DOWN, or ANY")
    if lookback_hours <= 0:
        raise typer.BadParameter("--lookback-hours must be positive")
    if limit <= 0:
        raise typer.BadParameter("--limit must be positive")
    if poll_seconds <= 0:
        raise typer.BadParameter("--poll-seconds must be positive")

    while True:
        try:
            snapshot = read_dashboard_snapshot(
                monitoring_db_path,
                model_version=model_version,
                edge_threshold=edge_threshold,
                exit_edge_threshold=exit_edge_threshold,
                outcome_side=outcome_side,
                lookback_hours=lookback_hours,
                limit=limit,
            )
            output = render_dashboard(snapshot)
        except Exception as exc:  # noqa: BLE001 - dashboard should survive transient DB locks.
            output = f"dashboard read error: {exc}"
        if not once:
            typer.echo("\033[2J\033[H", nl=False)
        typer.echo(output)
        if once:
            return
        time.sleep(poll_seconds)


@app.command("shadow-v1")
def shadow_v1(
    champion_model_path: Path = SHADOW_CHAMPION_MODEL_PATH_OPTION,
    challenger_model_path: Path = SHADOW_CHALLENGER_MODEL_PATH_OPTION,
    output_path: Path = SHADOW_OUTPUT_PATH_OPTION,
    warehouse_dir: Path | None = ORACLE_BACKTEST_WAREHOUSE_DIR_OPTION,
    champion_calibration_path: Path | None = SHADOW_CHAMPION_CALIBRATION_PATH_OPTION,
    challenger_calibration_path: Path | None = SHADOW_CHALLENGER_CALIBRATION_PATH_OPTION,
    evaluation_output_path: Path | None = SHADOW_EVALUATION_OUTPUT_PATH_OPTION,
    evaluation_json_output_path: Path | None = SHADOW_EVALUATION_JSON_OUTPUT_PATH_OPTION,
    offline_reference_path: Path | None = SHADOW_OFFLINE_REFERENCE_PATH_OPTION,
    edge_threshold: float = SHADOW_EDGE_THRESHOLD_OPTION,
    since_ms: int | None = typer.Option(None, help="Inclusive feature_ts lower bound."),
    until_ms: int | None = typer.Option(None, help="Exclusive feature_ts upper bound."),
    lookback_hours: float | None = typer.Option(
        None,
        help="Use features from the last N hours relative to now.",
    ),
    limit: int | None = typer.Option(None, help="Optional maximum feature rows to score."),
    bins: int = typer.Option(10, help="Histogram bins for distribution comparison."),
) -> None:
    """Run champion/challenger shadow scoring without changing champion output."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    if since_ms is not None and lookback_hours is not None:
        raise typer.BadParameter("use either --since-ms or --lookback-hours, not both")
    effective_since_ms = since_ms
    if lookback_hours is not None:
        if lookback_hours <= 0:
            raise typer.BadParameter("--lookback-hours must be positive")
        effective_since_ms = int(time.time() * 1_000 - lookback_hours * 3_600_000)
    report = run_shadow_warehouse_comparison(
        warehouse_dir=settings.warehouse_dir if warehouse_dir is None else warehouse_dir,
        champion_model_path=champion_model_path,
        challenger_model_path=challenger_model_path,
        output_path=output_path,
        champion_calibration_path=champion_calibration_path,
        challenger_calibration_path=challenger_calibration_path,
        since_ms=effective_since_ms,
        until_ms=until_ms,
        limit=limit,
        bins=bins,
        evaluation_output_path=evaluation_output_path,
        evaluation_json_output_path=evaluation_json_output_path,
        offline_reference_path=offline_reference_path,
        edge_threshold=edge_threshold,
    )
    payload = report.to_dict()
    payload["rows"] = f"{len(report.rows)} rows written to {output_path}"
    markdown_path, json_path = shadow_evaluation_output_paths(
        output_path,
        evaluation_output_path=evaluation_output_path,
        evaluation_json_output_path=evaluation_json_output_path,
    )
    payload["shadow_evaluation_markdown_path"] = str(markdown_path)
    payload["shadow_evaluation_json_path"] = str(json_path)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("serving-readiness-v1")
def serving_readiness_v1(
    model_path: Path = PREDICTION_MODEL_PATH_OPTION,
    feature_schema_path: Path = SERVING_READINESS_FEATURE_SCHEMA_PATH_OPTION,
    dataset_dir: Path = LOGISTIC_DATASET_DIR_OPTION,
    output_path: Path = SERVING_READINESS_OUTPUT_PATH_OPTION,
    split: str = typer.Option("test", help="Dataset split to benchmark."),
    sample_size: int = typer.Option(1_000, help="Single-row latency sample count."),
    batch_sizes: str = typer.Option(
        "10000,100000",
        help="Comma-separated batch sizes for throughput checks.",
    ),
    latency_sla_ms: float = typer.Option(50.0, help="p95 latency SLA in milliseconds."),
    max_error_rate: float = typer.Option(0.0, help="Maximum valid-input inference error rate."),
    fallback_model_path: Path | None = SERVING_READINESS_FALLBACK_MODEL_PATH_OPTION,
    rollback_runbook_path: Path | None = BOOTSTRAP_ROLLBACK_RUNBOOK_PATH_OPTION,
) -> None:
    """Measure local model serving latency, throughput, schema, and fallback readiness."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_xgboost_serving_readiness(
        model_path=model_path,
        feature_schema_path=feature_schema_path,
        dataset_dir=dataset_dir,
        output_path=output_path,
        split=split,
        sample_size=sample_size,
        batch_sizes=_parse_int_grid(batch_sizes),
        latency_sla_ms=latency_sla_ms,
        max_error_rate=max_error_rate,
        fallback_model_path=fallback_model_path,
        rollback_runbook_path=rollback_runbook_path,
    )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("backtest-oracle-sanity-v1")
def backtest_oracle_sanity_v1(
    dataset_dir: Path = ORACLE_BACKTEST_DATASET_DIR_OPTION,
    warehouse_dir: Path | None = ORACLE_BACKTEST_WAREHOUSE_DIR_OPTION,
    output_dir: Path = ORACLE_BACKTEST_OUTPUT_DIR_OPTION,
    thresholds: str = typer.Option(
        "0.00,0.03,0.05",
        help="Comma-separated edge thresholds for the oracle sweep.",
    ),
    use_label_target_ts: bool = typer.Option(
        True,
        "--label-target-ts/--fixed-hold",
        help="Exit at each label target_ts instead of feature_ts + 15m.",
    ),
    required_outcome_side: str = typer.Option(
        "UP",
        help="Required outcome side encoded in canonical_symbol. Use an empty string to include all outcomes.",
    ),
) -> None:
    """Run a perfect-label sanity backtest before trusting model promotion evidence."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_oracle_label_sanity_backtest(
        dataset_dir=dataset_dir,
        warehouse_dir=settings.warehouse_dir if warehouse_dir is None else warehouse_dir,
        output_dir=output_dir,
        thresholds=_parse_float_grid(thresholds),
        use_label_target_ts=use_label_target_ts,
        required_outcome_side=required_outcome_side or None,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("backtest-predictions-v1")
def backtest_predictions_v1(
    warehouse_dir: Path | None = ORACLE_BACKTEST_WAREHOUSE_DIR_OPTION,
    output_dir: Path = PREDICTION_BACKTEST_OUTPUT_DIR_OPTION,
    model_version: str | None = typer.Option(
        None,
        help="Optional model_version filter for the predictions table.",
    ),
    thresholds: str = typer.Option(
        "0.00,0.03,0.05",
        help="Comma-separated edge thresholds for the prediction sweep.",
    ),
    required_outcome_side: str = typer.Option(
        "UP",
        help="Required outcome side encoded in canonical_symbol. Use an empty string to include all outcomes.",
    ),
    fee_bps: float = typer.Option(10.0, help="Taker fee assumption in basis points."),
    slippage_bps: float = typer.Option(5.0, help="Taker slippage assumption in basis points."),
    latency_ms: int = typer.Option(0, help="Execution latency assumption in milliseconds."),
) -> None:
    """Run a grouped threshold backtest from warehouse predictions."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_prediction_threshold_backtest(
        warehouse_dir=settings.warehouse_dir if warehouse_dir is None else warehouse_dir,
        output_dir=output_dir,
        model_version=model_version,
        thresholds=_parse_float_grid(thresholds),
        settings=TakerExecutionSettings(
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
        ),
        required_outcome_side=required_outcome_side or None,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("backtest-model-v1")
def backtest_model_v1(
    model_path: Path = PREDICTION_MODEL_PATH_OPTION,
    dataset_dir: Path = ORACLE_BACKTEST_DATASET_DIR_OPTION,
    warehouse_dir: Path | None = ORACLE_BACKTEST_WAREHOUSE_DIR_OPTION,
    output_dir: Path = MODEL_BACKTEST_OUTPUT_DIR_OPTION,
    calibration_path: Path | None = PREDICTION_CALIBRATION_PATH_OPTION,
    model_version: str | None = typer.Option(
        None,
        help="Optional model_version label for output artifacts.",
    ),
    thresholds: str = typer.Option(
        "0.00,0.03,0.05",
        help="Comma-separated edge thresholds for the model sweep.",
    ),
    required_outcome_side: str = typer.Option(
        "UP",
        help="Required outcome side encoded in canonical_symbol. Use an empty string to include all outcomes.",
    ),
    fee_bps: float = typer.Option(10.0, help="Taker fee assumption in basis points."),
    slippage_bps: float = typer.Option(5.0, help="Taker slippage assumption in basis points."),
    latency_ms: int = typer.Option(0, help="Execution latency assumption in milliseconds."),
) -> None:
    """Score a saved model on a dataset and run a grouped threshold backtest."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    report = run_model_threshold_backtest(
        model_path=model_path,
        dataset_dir=dataset_dir,
        warehouse_dir=settings.warehouse_dir if warehouse_dir is None else warehouse_dir,
        output_dir=output_dir,
        calibration_path=calibration_path,
        model_version=model_version,
        thresholds=_parse_float_grid(thresholds),
        settings=TakerExecutionSettings(
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
        ),
        required_outcome_side=required_outcome_side or None,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("backtest-config")
def backtest_config(
    config_path: Path = BACKTEST_CONFIG_PATH_ARGUMENT,
    preserve_run_id: bool = typer.Option(
        False,
        help="Preserve output.run_id from the file instead of generating a fresh one.",
    ),
) -> None:
    """Validate and print normalized backtest config JSON."""
    config = load_backtest_config(config_path, new_run_id=not preserve_run_id)
    typer.echo(json.dumps(config.to_script_dict(), indent=2, sort_keys=True))


@app.command("quarantine-report")
def quarantine_report(
    limit: int = typer.Option(50, help="Max rows of detail to display."),
) -> None:
    """Summarise the quarantine table: counts by rule + recent samples."""
    settings = IngestionSettings()
    _configure_logging(settings.log_level)
    out: dict = {"by_rule": {}, "by_target_table": {}, "samples": []}
    with open_warehouse(settings.warehouse_dir) as conn:
        try:
            rows = conn.execute(
                "SELECT rule, COUNT(*) AS n FROM quarantine GROUP BY rule ORDER BY n DESC"
            ).fetchall()
            out["by_rule"] = {r[0]: r[1] for r in rows}

            rows = conn.execute(
                "SELECT target_table, COUNT(*) AS n FROM quarantine "
                "GROUP BY target_table ORDER BY n DESC"
            ).fetchall()
            out["by_target_table"] = {r[0]: r[1] for r in rows}

            samples = conn.execute(
                "SELECT ts, source, source_symbol, target_table, rule, detail "
                "FROM quarantine ORDER BY ts DESC LIMIT ?",
                [limit],
            ).fetchall()
            out["samples"] = [
                {
                    "ts": r[0],
                    "source": r[1],
                    "source_symbol": r[2],
                    "target_table": r[3],
                    "rule": r[4],
                    "detail": r[5],
                }
                for r in samples
            ]

            try:
                dupes = conn.execute(
                    "SELECT COUNT(*) - COUNT(DISTINCT trade_id) AS duplicate_rows "
                    "FROM raw_trades"
                ).fetchone()
                out["raw_trade_duplicate_rows"] = int(dupes[0])
            except Exception:  # noqa: BLE001
                out["raw_trade_duplicate_rows"] = 0
        except Exception:  # noqa: BLE001
            # Empty warehouse / no quarantine partition yet.
            pass
    typer.echo(json.dumps(out, indent=2))


def _soak_timestamp_label() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _now_seconds() -> float:
    return time.time()


async def _run_market_coverage_check(
    *,
    settings: IngestionSettings,
    raw_dir: Path,
    max_stale_seconds: float | None,
    require_hash_match: bool,
    ignore_markets_opened_after_raw_end: bool,
    raw_end_grace_seconds: float,
    rest_concurrency: int,
    max_examples: int,
) -> dict[str, Any]:
    try:
        market_specs = parse_market_specs_json(
            settings.market_specs_json,
            fallback_slug_prefix=settings.market_slug_prefix,
        )
        async with GammaClient(
            settings.gamma_api_base,
            settings.market_slug_prefix,
            market_specs=market_specs,
        ) as gamma:
            markets = await gamma.list_active_markets()
        async with PolymarketRestClient(
            settings.clob_rest_url,
            data_api_base_url=settings.polymarket_data_api_url,
            timeout_seconds=settings.backfill_rest_timeout_seconds,
        ) as rest:
            return await compare_market_coverage(
                markets=markets,
                raw_dir=raw_dir,
                rest=rest,
                max_stale_seconds=max_stale_seconds,
                require_hash_match=require_hash_match,
                ignore_markets_opened_after_raw_end=ignore_markets_opened_after_raw_end,
                raw_end_grace_seconds=raw_end_grace_seconds,
                max_concurrency=rest_concurrency,
                max_examples=max_examples,
            )
    except Exception as exc:  # noqa: BLE001
        return {
            "passed": False,
            "error": repr(exc),
            "raw": {"dir": str(raw_dir)},
        }


@app.command("backfill")
def backfill(
    asset_id: str = typer.Option(..., help="CLOB token id (asset_id) to backfill."),
    market: str = typer.Option(
        ..., help="Polymarket condition_id (market hash) for trade lookup."
    ),
    since_ms: int = typer.Option(..., help="Gap start in epoch ms (UTC)."),
    until_ms: int = typer.Option(..., help="Gap end in epoch ms (UTC)."),
) -> None:
    """Manually run a REST backfill for a known [since_ms, until_ms] gap.

    Synthesised NDJSON records are written into the same raw sink the
    live WS pipeline uses. The next ETL run will pick them up.
    """
    settings = IngestionSettings()
    _configure_logging(settings.log_level)

    async def _run() -> dict:
        sink = NdjsonGzipSink(
            settings.raw_dir,
            flush_interval_seconds=settings.sink_flush_interval_seconds,
            max_buffer_records=settings.sink_max_buffer_records,
        )
        await sink.start_background_flusher()
        try:
            async with PolymarketRestClient(
                settings.clob_rest_url,
                data_api_base_url=settings.polymarket_data_api_url,
                timeout_seconds=settings.backfill_rest_timeout_seconds,
            ) as rest:
                async def resolver(_: str) -> str:
                    return market

                service = BackfillService(rest, sink, resolver)
                report = await service.handle_gap(
                    GapWindow(
                        asset_id=asset_id,
                        gap_start_ms=since_ms,
                        gap_end_ms=until_ms,
                    )
                )
                return {
                    "asset_id": report.asset_id,
                    "market": report.market,
                    "gap_start_ms": report.gap_start_ms,
                    "gap_end_ms": report.gap_end_ms,
                    "trades_replayed": report.trades_replayed,
                    "orderbook_replayed": report.orderbook_replayed,
                    "errors": report.errors,
                    "total_records": report.total_records,
                }
        finally:
            await sink.close()

    result = asyncio.run(_run())
    typer.echo(json.dumps(result, indent=2))


def _parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated integers") from exc
    if not parsed:
        raise typer.BadParameter("expected at least one integer")
    return parsed


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated numbers") from exc
    if not parsed:
        raise typer.BadParameter("expected at least one number")
    return parsed


if __name__ == "__main__":
    app()
