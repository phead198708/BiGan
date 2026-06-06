"""Contracts for the live xgboost-v4 champion runner."""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

from bigan.mlops import (
    ModelDeploymentRecord,
    ModelRegistryRecord,
    complete_deployment,
    connect_mlops_db,
    record_deployment,
    register_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_champion_live.sh"


def _write_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fake-python"
    real_python = shlex.quote(sys.executable)
    fake_python.write_text(
        f"""#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${{1:-}}" == "-" || "${{1:-}}" == "-c" ]]; then
  exec {real_python} "$@"
fi

if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "bigan.ingestion.__main__" ]]; then
  command="${{3:-}}"
  shift 3
  if [[ -n "${{FAKE_COMMAND_LOG:-}}" ]]; then
    printf '%s %s\\n' "${{command}}" "$*" >> "${{FAKE_COMMAND_LOG}}"
  fi
  case "${{command}}" in
    serve)
      trap 'exit 0' TERM INT
      while true; do sleep 0.1 & wait "$!"; done
      ;;
    etl-batch)
      printf 'fake log before JSON\\n'
      printf '{{"files_processed": %s, "records_read": 0}}\\n' "${{FAKE_ETL_FILES_PROCESSED:-0}}"
      ;;
    features-15m-v1)
      printf '{{"rows_generated": 10, "rows_written": 5}}\\n'
      ;;
    features-15m-v1-low-latency-queue)
      printf '{{"rows_read": 3, "rows_generated": 2, "rows_written": 2}}\\n'
      ;;
    predictions-v1)
      printf '{{"rows_generated": 5, "rows_written": 5, "monitoring_events_written": 5}}\\n'
      ;;
    *)
      printf 'unexpected command: %s\\n' "${{command}}" >&2
      exit 12
      ;;
  esac
  exit 0
fi

exec {real_python} "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    return fake_python


def _run_live_runner(tmp_path: Path, *, files_processed: int) -> tuple[str, str]:
    fake_python = _write_fake_python(tmp_path)
    model_path = tmp_path / "model.json"
    calibration_path = tmp_path / "calibration.json"
    model_path.write_text("{}", encoding="utf-8")
    calibration_path.write_text("{}", encoding="utf-8")
    command_log = tmp_path / "commands.log"
    log_dir = tmp_path / "logs"
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "MODEL_PATH": str(model_path),
        "CALIBRATION_PATH": str(calibration_path),
        "MONITORING_DB_PATH": str(tmp_path / "mlops.duckdb"),
        "LIVE_ROOT": str(tmp_path / "live"),
        "LOG_DIR": str(log_dir),
        "FAKE_COMMAND_LOG": str(command_log),
        "FAKE_ETL_FILES_PROCESSED": str(files_processed),
        "SCAN_STARTUP_SECONDS": "0",
        "CYCLE_SLEEP_SECONDS": "0",
        "STOP_AFTER_CYCLES": "1",
        "LABELS_ENABLED": "false",
        "DASHBOARD_ENABLED": "false",
        "SCORE_ONLY_WHEN_ETL_PROCESSED": "true",
        "LIVE_MIN_FREE_BYTES": "0",
    }
    subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
    scorer_log = next(log_dir.glob("scorer-*.log")).read_text(encoding="utf-8")
    return command_log.read_text(encoding="utf-8"), scorer_log


def _base_runner_env(tmp_path: Path) -> dict[str, str]:
    fake_python = _write_fake_python(tmp_path)
    model_path = tmp_path / "model.json"
    calibration_path = tmp_path / "calibration.json"
    model_path.write_text("{}", encoding="utf-8")
    calibration_path.write_text("{}", encoding="utf-8")
    return {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "MODEL_PATH": str(model_path),
        "CALIBRATION_PATH": str(calibration_path),
        "MONITORING_DB_PATH": str(tmp_path / "mlops.duckdb"),
        "LIVE_ROOT": str(tmp_path / "live"),
        "LOG_DIR": str(tmp_path / "logs"),
        "FAKE_COMMAND_LOG": str(tmp_path / "commands.log"),
        "SCAN_STARTUP_SECONDS": "0",
        "CYCLE_SLEEP_SECONDS": "0",
        "STOP_AFTER_CYCLES": "1",
        "LABELS_ENABLED": "false",
        "DASHBOARD_ENABLED": "false",
        "SCORE_ONLY_WHEN_ETL_PROCESSED": "true",
        "LIVE_MIN_FREE_BYTES": "0",
    }


def _write_current_online_catalog(
    db_path: Path,
    *,
    model_version: str,
    model_path: Path,
    calibration_path: Path,
) -> None:
    conn = connect_mlops_db(db_path)
    register_model(
        conn,
        ModelRegistryRecord(
            model_version=model_version,
            model_family="btc-updown-15m",
            feature_version="bigan-mvp-v1.0.0",
            dataset_version="test-dataset",
            train_config_hash="test-hash",
            artifact_uri=str(model_path),
            calibration_artifact_uri=str(calibration_path),
            status="champion",
            train_started_at=1_000,
            train_finished_at=2_000,
            metrics_json=json.dumps({"test": {"roc_auc": 0.61}}),
            backtest_json=json.dumps({"net_pnl": 1.0}),
        ),
    )
    record_deployment(
        conn,
        ModelDeploymentRecord(
            deployment_id="deploy-current",
            model_version=model_version,
            environment="prod",
            rollout_strategy="full",
            traffic_percent=100.0,
            deployment_status="running",
            started_at=3_000,
            rollback_to_version="xgboost-v4",
            operator="test",
            reason="test current online defaults",
        ),
    )
    complete_deployment(conn, "deploy-current", completed_at=4_000)


def test_runner_skips_scoring_when_etl_processes_zero_files(tmp_path: Path) -> None:
    command_log, scorer_log = _run_live_runner(tmp_path, files_processed=0)

    assert "etl-batch" in command_log
    assert "features-15m-v1" not in command_log
    assert "predictions-v1" not in command_log
    assert "etl_files_processed=0" in scorer_log
    assert "scoring skipped cycle=1 files_processed=0" in scorer_log


def test_runner_scores_when_etl_processes_new_files(tmp_path: Path) -> None:
    command_log, scorer_log = _run_live_runner(tmp_path, files_processed=1)

    assert "etl-batch" in command_log
    assert "features-15m-v1" in command_log
    assert "predictions-v1" in command_log
    assert "etl_files_processed=1" in scorer_log
    assert "scoring skipped" not in scorer_log


def test_runner_can_scope_low_latency_scoring_to_canonical_symbol(tmp_path: Path) -> None:
    env = _base_runner_env(tmp_path)
    env["FAKE_ETL_FILES_PROCESSED"] = "1"
    env["FEATURE_LOOKBACK_MINUTES"] = "4"
    env["PREDICTION_LOOKBACK_MINUTES"] = "4"
    env["SCORING_CANONICAL_SYMBOL_LIKE"] = "BTC-15M:%"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )

    command_log = Path(env["FAKE_COMMAND_LOG"]).read_text(encoding="utf-8")

    assert (
        "features-15m-v1 --lookback-minutes 4 --skip-existing "
        "--canonical-symbol-like BTC-15M:%"
    ) in command_log
    assert "predictions-v1 --model-path" in command_log
    assert (
        "--lookback-minutes 4 --skip-existing-monitoring-events "
        "--skip-existing-predictions --canonical-symbol-like BTC-15M:%"
    ) in command_log
    assert "feature lookback minutes=4" in result.stdout
    assert "prediction lookback minutes=4" in result.stdout
    assert "scoring canonical symbol like=BTC-15M:%" in result.stdout


def test_runner_defaults_to_current_online_model_from_catalog(tmp_path: Path) -> None:
    env = _base_runner_env(tmp_path)
    model_path = tmp_path / "catalog-model.json"
    calibration_path = tmp_path / "catalog-calibration.json"
    model_path.write_text("{}", encoding="utf-8")
    calibration_path.write_text("{}", encoding="utf-8")
    _write_current_online_catalog(
        Path(env["MONITORING_DB_PATH"]),
        model_version="xgboost-v5",
        model_path=model_path,
        calibration_path=calibration_path,
    )
    env.pop("MODEL_PATH")
    env.pop("CALIBRATION_PATH")
    env.pop("MODEL_VERSION", None)
    env["FAKE_ETL_FILES_PROCESSED"] = "1"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )

    command_log = Path(env["FAKE_COMMAND_LOG"]).read_text(encoding="utf-8")

    assert "model version=xgboost-v5" in result.stdout
    assert f"model path={model_path}" in result.stdout
    assert f"calibration path={calibration_path}" in result.stdout
    assert f"predictions-v1 --model-path {model_path}" in command_log
    assert f"--calibration-path {calibration_path}" in command_log


def test_runner_allows_xgboost_v6_without_external_calibration(tmp_path: Path) -> None:
    env = _base_runner_env(tmp_path)
    env["MODEL_VERSION"] = "xgboost-v6"
    env.pop("CALIBRATION_PATH")
    env["FAKE_ETL_FILES_PROCESSED"] = "1"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )

    command_log = Path(env["FAKE_COMMAND_LOG"]).read_text(encoding="utf-8")

    assert "model version=xgboost-v6" in result.stdout
    assert "calibration path=(embedded in model artifact)" in result.stdout
    assert "predictions-v1 --model-path" in command_log
    assert "--calibration-path" not in command_log


def test_runner_allows_xgboost_v7_without_external_calibration(tmp_path: Path) -> None:
    env = _base_runner_env(tmp_path)
    env["MODEL_VERSION"] = "xgboost-v7"
    env.pop("CALIBRATION_PATH")
    env["FAKE_ETL_FILES_PROCESSED"] = "1"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )

    command_log = Path(env["FAKE_COMMAND_LOG"]).read_text(encoding="utf-8")

    assert "model version=xgboost-v7" in result.stdout
    assert "calibration path=(embedded in model artifact)" in result.stdout
    assert "predictions-v1 --model-path" in command_log
    assert "--calibration-path" not in command_log


def test_runner_can_replace_predictions_for_queue_reemit(tmp_path: Path) -> None:
    env = _base_runner_env(tmp_path)
    env["FAKE_ETL_FILES_PROCESSED"] = "1"
    env["SKIP_EXISTING_PREDICTIONS"] = "false"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )

    command_log = Path(env["FAKE_COMMAND_LOG"]).read_text(encoding="utf-8")

    assert "--skip-existing-monitoring-events --replace-predictions" in command_log
    assert "--skip-existing-predictions" not in command_log
    assert "skip existing predictions=false" in result.stdout


def test_runner_can_use_low_latency_raw_queue_feature_path(tmp_path: Path) -> None:
    env = _base_runner_env(tmp_path)
    env["LOW_LATENCY_FEATURE_QUEUE_ENABLED"] = "true"
    env["FEATURE_LOOKBACK_MINUTES"] = "4"
    env["PREDICTION_LOOKBACK_MINUTES"] = "4"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )

    command_log = Path(env["FAKE_COMMAND_LOG"]).read_text(encoding="utf-8")

    assert "etl-batch" not in command_log
    assert "features-15m-v1 " not in command_log
    assert "features-15m-v1-low-latency-queue --queue-path" in command_log
    assert "--canonical-symbol-prefix BTC-15M:" in command_log
    assert "predictions-v1 --model-path" in command_log
    assert "--canonical-symbol-like BTC-15M:%" in command_log
    assert "low-latency feature queue enabled=true" in result.stdout
    assert "low-latency raw queue=" in result.stdout


def test_runner_rejects_live_root_lock_held_by_running_process(
    tmp_path: Path,
) -> None:
    env = _base_runner_env(tmp_path)
    live_root = Path(env["LIVE_ROOT"])
    lock_dir = live_root / ".run_champion_live.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert "live root lock held by pid" in result.stderr


def test_runner_reclaims_stale_live_root_lock(tmp_path: Path) -> None:
    env = _base_runner_env(tmp_path)
    env["FAKE_ETL_FILES_PROCESSED"] = "0"
    live_root = Path(env["LIVE_ROOT"])
    lock_dir = live_root / ".run_champion_live.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text("999999", encoding="utf-8")

    subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert not lock_dir.exists()


def test_runner_rejects_live_root_when_free_space_below_floor(tmp_path: Path) -> None:
    env = _base_runner_env(tmp_path)
    env["LIVE_MIN_FREE_BYTES"] = str(10**30)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 2
    assert "live root filesystem free space below floor" in result.stderr
    assert not Path(env["FAKE_COMMAND_LOG"]).exists()
