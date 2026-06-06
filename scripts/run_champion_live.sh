#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the live champion signal pipeline.

This starts a continuous Polymarket capture service in the background, runs a
5-second scan/score loop, and keeps the signal dashboard in the foreground.
Press Ctrl-C to stop the capture service and scorer loop.

Environment overrides:
  PYTHON_BIN                 Python executable. Default: .venv/bin/python
  MODEL_VERSION              Champion model version. Default: current prod online model.
  MODEL_PATH                 Champion model artifact. Default: registry artifact for current prod online model.
  CALIBRATION_PATH           Champion calibration artifact. Default: registry calibration for current prod online model.
  MONITORING_DB_PATH         DuckDB monitoring catalog. Default: data/mlops/champion_catalog.duckdb
  LIVE_ROOT                  Live data root. Default: data/live/champion-live-<utc-session>
  MARKET_SPECS_JSON          Optional JSON market specs for BTC/ETH or multi-horizon capture.
  SCAN_STARTUP_SECONDS       Capture warmup before first scan. Default: 10
  LOOKBACK_MINUTES           Feature lookback scored each cycle. Default: 30
  FEATURE_LOOKBACK_MINUTES   Feature row output lookback. Default: LOOKBACK_MINUTES
  PREDICTION_LOOKBACK_MINUTES
                             Prediction row lookback. Default: LOOKBACK_MINUTES
  SCORING_CANONICAL_SYMBOL_LIKE
                             Optional canonical_symbol SQL LIKE filter for
                             low-latency scoped scoring, e.g. BTC-15M:%.
  SIGNAL_JSONL_OUTPUT_PATH   Optional executor-ready signal JSONL queue written
                             directly by predictions-v1.
  SIGNAL_JSONL_MARKET_FAMILIES
                             Families to emit to SIGNAL_JSONL_OUTPUT_PATH.
                             Default: BTC-15M
  SIGNAL_JSONL_OUTCOME_SIDE  Outcome side emitted to SIGNAL_JSONL_OUTPUT_PATH:
                             UP, DOWN, ANY, or a comma-separated subset.
                             Default: ANY
  SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS
                             Optional max event age, in seconds, for queue
                             emission. Stale signals are not appended.
  LOW_LATENCY_FEATURE_QUEUE_ENABLED
                             Consume BTC-15M features directly from a raw JSONL
                             queue instead of raw ETL + batch recompute.
                             Default: false
  LOW_LATENCY_RAW_QUEUE_PATH Queue path written by capture when low-latency
                             features are enabled. Default under LIVE_ROOT.
  LOW_LATENCY_FEATURE_CURSOR_PATH
                             Cursor file for the low-latency queue consumer.
  LOW_LATENCY_FEATURE_STATE_PATH
                             State file for incremental BTC-15M feature context.
  LOW_LATENCY_FEATURE_MAX_RECORDS
                             Max raw queue records consumed per cycle. Default: 50000
  CYCLE_SLEEP_SECONDS        Sleep after each scoring cycle. Default: 5
  ETL_LAG_SECONDS            Eligible raw-file mtime lag for ETL. Default: 0
  ETL_SEGMENT_SAFETY_SECONDS Extra lag added when segmented raw gzip files are
                             enabled. Default: 15
  ETL_PROCESSED_MANIFEST_PATH
                             Optional ETL manifest for immutable segmented raw files.
  ETL_MAX_FILES_PER_BATCH    Max NDJSON.gz files per etl-batch cycle. Default: unset (all eligible).
  BIGAN_SINK_SEGMENT_DURATION_SECONDS
                             Raw gzip segment length in seconds (0 = one file per UTC day).
  SCORE_ONLY_WHEN_ETL_PROCESSED
                             Skip feature/prediction scans when ETL processed
                             zero new files. Default: true
  BIGAN_SINK_SEGMENT_DURATION_SECONDS
                             Optional raw gzip segment grain for long runs. Default: 0
  LABELS_ENABLED             Refresh settled labels during the scan loop. Default: true
  LABELS_EVERY_CYCLES        Run label refresh every N scan cycles. Default: 12
  LABEL_LOOKBACK_MINUTES     Feature lookback for label refreshes. Default: 120
  LABEL_REQUEST_TIMEOUT_SECONDS
                             Per-request Gamma timeout for label refreshes. Default: 3
  LABEL_REQUEST_CONCURRENCY   Concurrent Gamma requests for label refreshes. Default: 8
  FEE_BPS                    Entry fee basis points for realized paper PnL labels. Default: 0
  EDGE_THRESHOLD             BUY_UP edge threshold for tail display. Default: 0.30
  EXIT_EDGE_THRESHOLD        Paper SELL threshold for dashboard. Default: 0.10
  DASHBOARD_LOOKBACK_HOURS   Dashboard history window. Default: 6
  TAIL_POLL_SECONDS          Dashboard refresh interval. Default: 2
  DASHBOARD_ENABLED          Set false for smoke runs. Default: true
  LOG_DIR                    Scoring logs directory. Default: data/logs/champion-live
  STOP_AFTER_CYCLES          Optional finite cycle count, mainly for smoke tests.
  LIVE_MIN_FREE_BYTES        Hard free-space floor for LIVE_ROOT filesystem.
                             Set 0 to disable. Default: 5368709120 (5 GiB).

Example:
  ./scripts/run_champion_live.sh

Multi-market example:
  MARKET_SPECS_JSON='[
    {"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15},
    {"slug_prefix":"eth-updown-5m-","underlying":"ETH","horizon_minutes":5}
  ]' ./scripts/run_champion_live.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MONITORING_DB_PATH="${MONITORING_DB_PATH:-data/mlops/champion_catalog.duckdb}"
export PYTHONPATH="${PYTHONPATH:-src}"
DEFAULT_MODEL_VERSION="xgboost-v4"
DEFAULT_MODEL_PATH="data/xgboost-v4-run-20260523T103814Z/artifacts/models/xgboost-v4/model.json"
DEFAULT_CALIBRATION_PATH="data/xgboost-v4-run-20260523T103814Z/artifacts/models/xgboost-v4-selected-calibration/calibration.json"
CATALOG_MODEL_VERSION=""
CATALOG_MODEL_PATH=""
CATALOG_CALIBRATION_PATH=""

resolve_online_model_defaults() {
  "${PYTHON_BIN}" - "${MONITORING_DB_PATH}" <<'PY'
import sys
from pathlib import Path

try:
    from bigan.mlops import connect_mlops_db, current_online_model, model_by_version

    db_path = Path(sys.argv[1])
    if not db_path.exists():
        raise SystemExit(2)
    conn = connect_mlops_db(db_path)
    online = current_online_model(conn, "prod")
    if not online:
        raise SystemExit(3)
    model_version = str(online.get("model_version") or "")
    registry_row = model_by_version(conn, model_version)
    if not registry_row:
        raise SystemExit(4)
    artifact_uri = str(registry_row.get("artifact_uri") or "")
    calibration_uri = str(registry_row.get("calibration_artifact_uri") or "")
    if not model_version or not artifact_uri or not calibration_uri:
        raise SystemExit(5)
    print("\t".join((model_version, artifact_uri, calibration_uri)))
except Exception:
    raise SystemExit(1)
PY
}

if [[ -z "${MODEL_VERSION:-}" || -z "${MODEL_PATH:-}" || -z "${CALIBRATION_PATH:-}" ]]; then
  if CATALOG_DEFAULTS="$(resolve_online_model_defaults 2>/dev/null)"; then
    IFS=$'\t' read -r CATALOG_MODEL_VERSION CATALOG_MODEL_PATH CATALOG_CALIBRATION_PATH <<< "${CATALOG_DEFAULTS}"
  fi
fi

if [[ -z "${MODEL_VERSION:-}" ]]; then
  MODEL_VERSION="${CATALOG_MODEL_VERSION:-${DEFAULT_MODEL_VERSION}}"
fi
if [[ -n "${CATALOG_MODEL_VERSION}" && "${MODEL_VERSION}" == "${CATALOG_MODEL_VERSION}" ]]; then
  MODEL_PATH="${MODEL_PATH:-${CATALOG_MODEL_PATH}}"
  CALIBRATION_PATH="${CALIBRATION_PATH:-${CATALOG_CALIBRATION_PATH}}"
fi
if [[ "${MODEL_VERSION}" == "${DEFAULT_MODEL_VERSION}" ]]; then
  MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
  CALIBRATION_PATH="${CALIBRATION_PATH:-${DEFAULT_CALIBRATION_PATH}}"
fi
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "[champion-live] MODEL_PATH is required for MODEL_VERSION=${MODEL_VERSION}" >&2
  exit 1
fi
IS_EMBEDDED_CALIBRATION_MODEL="false"
if [[ "${MODEL_VERSION}" == "xgboost-v6" || "${MODEL_VERSION}" == xgboost-v6:* || "${MODEL_VERSION}" == "xgboost-v7" || "${MODEL_VERSION}" == xgboost-v7:* ]]; then
  IS_EMBEDDED_CALIBRATION_MODEL="true"
fi
if [[ "${IS_EMBEDDED_CALIBRATION_MODEL}" == "true" ]]; then
  if [[ -n "${CALIBRATION_PATH:-}" ]]; then
    echo "[champion-live] ignoring CALIBRATION_PATH for MODEL_VERSION=${MODEL_VERSION}; calibration is embedded" >&2
  fi
  CALIBRATION_PATH=""
elif [[ -z "${CALIBRATION_PATH:-}" ]]; then
  echo "[champion-live] CALIBRATION_PATH is required for MODEL_VERSION=${MODEL_VERSION}" >&2
  exit 1
fi
LIVE_ROOT="${LIVE_ROOT:-data/live/champion-live-${SESSION_ID}}"
MARKET_SPECS_JSON="${MARKET_SPECS_JSON:-${BIGAN_MARKET_SPECS_JSON:-}}"
SCAN_STARTUP_SECONDS="${SCAN_STARTUP_SECONDS:-10}"
LOOKBACK_MINUTES="${LOOKBACK_MINUTES:-30}"
FEATURE_LOOKBACK_MINUTES="${FEATURE_LOOKBACK_MINUTES:-${LOOKBACK_MINUTES}}"
PREDICTION_LOOKBACK_MINUTES="${PREDICTION_LOOKBACK_MINUTES:-${LOOKBACK_MINUTES}}"
SCORING_CANONICAL_SYMBOL_LIKE="${SCORING_CANONICAL_SYMBOL_LIKE:-}"
SIGNAL_JSONL_OUTPUT_PATH="${SIGNAL_JSONL_OUTPUT_PATH:-}"
SIGNAL_JSONL_MARKET_FAMILIES="${SIGNAL_JSONL_MARKET_FAMILIES:-BTC-15M}"
SIGNAL_JSONL_OUTCOME_SIDE="${SIGNAL_JSONL_OUTCOME_SIDE:-ANY}"
SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS="${SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS:-}"
V6_SETTLEMENT_THRESHOLD="${V6_SETTLEMENT_THRESHOLD:-0.50}"
V6_NEUTRAL_CAP="${V6_NEUTRAL_CAP:-0.25}"
V6_VOLATILITY_THRESHOLD="${V6_VOLATILITY_THRESHOLD:-0.60}"
V6_ROUND_TRIP_COST="${V6_ROUND_TRIP_COST:-0.072}"
V6_EV_MARGIN="${V6_EV_MARGIN:-0.01}"
LOW_LATENCY_FEATURE_QUEUE_ENABLED="${LOW_LATENCY_FEATURE_QUEUE_ENABLED:-false}"
LOW_LATENCY_RAW_QUEUE_PATH="${LOW_LATENCY_RAW_QUEUE_PATH:-${LIVE_ROOT}/low-latency/raw-btc15m.jsonl}"
LOW_LATENCY_RAW_QUEUE_CANONICAL_SYMBOL_PREFIX="${LOW_LATENCY_RAW_QUEUE_CANONICAL_SYMBOL_PREFIX:-BTC-15M:}"
LOW_LATENCY_FEATURE_CURSOR_PATH="${LOW_LATENCY_FEATURE_CURSOR_PATH:-${LIVE_ROOT}/low-latency/features.cursor}"
LOW_LATENCY_FEATURE_STATE_PATH="${LOW_LATENCY_FEATURE_STATE_PATH:-${LIVE_ROOT}/low-latency/features-state.json}"
LOW_LATENCY_FEATURE_MAX_RECORDS="${LOW_LATENCY_FEATURE_MAX_RECORDS:-50000}"
CYCLE_SLEEP_SECONDS="${CYCLE_SLEEP_SECONDS:-5}"
ETL_LAG_SECONDS="${ETL_LAG_SECONDS:-0}"
ETL_SEGMENT_SAFETY_SECONDS="${ETL_SEGMENT_SAFETY_SECONDS:-15}"
ETL_PROCESSED_MANIFEST_PATH="${ETL_PROCESSED_MANIFEST_PATH:-}"
ETL_MAX_FILES_PER_BATCH="${ETL_MAX_FILES_PER_BATCH:-}"
SCORE_ONLY_WHEN_ETL_PROCESSED="${SCORE_ONLY_WHEN_ETL_PROCESSED:-true}"
LABELS_ENABLED="${LABELS_ENABLED:-true}"
LABELS_EVERY_CYCLES="${LABELS_EVERY_CYCLES:-12}"
LABEL_LOOKBACK_MINUTES="${LABEL_LOOKBACK_MINUTES:-120}"
LABEL_REQUEST_TIMEOUT_SECONDS="${LABEL_REQUEST_TIMEOUT_SECONDS:-3}"
LABEL_REQUEST_CONCURRENCY="${LABEL_REQUEST_CONCURRENCY:-8}"
FEE_BPS="${FEE_BPS:-0}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-0.30}"
EXIT_EDGE_THRESHOLD="${EXIT_EDGE_THRESHOLD:-0.10}"
DASHBOARD_LOOKBACK_HOURS="${DASHBOARD_LOOKBACK_HOURS:-6}"
TAIL_POLL_SECONDS="${TAIL_POLL_SECONDS:-2}"
DASHBOARD_ENABLED="${DASHBOARD_ENABLED:-true}"
LOG_DIR="${LOG_DIR:-data/logs/champion-live}"
STOP_AFTER_CYCLES="${STOP_AFTER_CYCLES:-}"
LIVE_MIN_FREE_BYTES="${LIVE_MIN_FREE_BYTES:-5368709120}"
SINK_SEGMENT_DURATION_SECONDS="${BIGAN_SINK_SEGMENT_DURATION_SECONDS:-0}"
ETL_EFFECTIVE_LAG_SECONDS="${ETL_LAG_SECONDS}"

if [[ "${SINK_SEGMENT_DURATION_SECONDS}" =~ ^[0-9]+$ && "${ETL_SEGMENT_SAFETY_SECONDS}" =~ ^[0-9]+$ && "${ETL_LAG_SECONDS}" =~ ^[0-9]+$ ]]; then
  if (( SINK_SEGMENT_DURATION_SECONDS > 0 )); then
    ETL_MIN_SEGMENT_LAG_SECONDS=$((SINK_SEGMENT_DURATION_SECONDS + ETL_SEGMENT_SAFETY_SECONDS))
    if (( ETL_LAG_SECONDS < ETL_MIN_SEGMENT_LAG_SECONDS )); then
      ETL_EFFECTIVE_LAG_SECONDS="${ETL_MIN_SEGMENT_LAG_SECONDS}"
    fi
  fi
fi

if [[ "${LOW_LATENCY_FEATURE_QUEUE_ENABLED}" == "true" && -z "${SCORING_CANONICAL_SYMBOL_LIKE}" ]]; then
  SCORING_CANONICAL_SYMBOL_LIKE="${LOW_LATENCY_RAW_QUEUE_CANONICAL_SYMBOL_PREFIX}%"
fi

export BIGAN_METRICS_ENABLED="${BIGAN_METRICS_ENABLED:-false}"
export BIGAN_DATA_DIR="${LIVE_ROOT}"
if [[ -n "${MARKET_SPECS_JSON}" ]]; then
  export BIGAN_MARKET_SPECS_JSON="${MARKET_SPECS_JSON}"
fi
if [[ "${LOW_LATENCY_FEATURE_QUEUE_ENABLED}" == "true" ]]; then
  export BIGAN_LOW_LATENCY_RAW_QUEUE_PATH="${LOW_LATENCY_RAW_QUEUE_PATH}"
  export BIGAN_LOW_LATENCY_RAW_QUEUE_CANONICAL_SYMBOL_PREFIX="${LOW_LATENCY_RAW_QUEUE_CANONICAL_SYMBOL_PREFIX}"
fi

mkdir -p "${LIVE_ROOT}" "${LOG_DIR}" "$(dirname "${MONITORING_DB_PATH}")"
if [[ "${LOW_LATENCY_FEATURE_QUEUE_ENABLED}" == "true" ]]; then
  mkdir -p "$(dirname "${LOW_LATENCY_RAW_QUEUE_PATH}")" \
    "$(dirname "${LOW_LATENCY_FEATURE_CURSOR_PATH}")" \
    "$(dirname "${LOW_LATENCY_FEATURE_STATE_PATH}")"
fi
if [[ -n "${SIGNAL_JSONL_OUTPUT_PATH}" ]]; then
  mkdir -p "$(dirname "${SIGNAL_JSONL_OUTPUT_PATH}")"
fi
CAPTURE_LOG="${LOG_DIR}/capture-${SESSION_ID}.log"
SCORER_LOG="${LOG_DIR}/scorer-${SESSION_ID}.log"
LIVE_LOCK_DIR="${LIVE_ROOT}/.run_champion_live.lock"
LIVE_LOCK_PID_FILE="${LIVE_LOCK_DIR}/pid"
LIVE_LOCK_ACQUIRED="false"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[champion-live] missing ${label}: ${path}" >&2
    exit 1
  fi
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[champion-live] missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi
require_file "${MODEL_PATH}" "model artifact"
if [[ "${IS_EMBEDDED_CALIBRATION_MODEL}" != "true" ]]; then
  require_file "${CALIBRATION_PATH}" "calibration artifact"
fi
if ! [[ "${LABELS_EVERY_CYCLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[champion-live] LABELS_EVERY_CYCLES must be a positive integer" >&2
  exit 1
fi
if ! [[ "${LABEL_REQUEST_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[champion-live] LABEL_REQUEST_CONCURRENCY must be a positive integer" >&2
  exit 1
fi
if [[ "${SCORE_ONLY_WHEN_ETL_PROCESSED}" != "true" && "${SCORE_ONLY_WHEN_ETL_PROCESSED}" != "false" ]]; then
  echo "[champion-live] SCORE_ONLY_WHEN_ETL_PROCESSED must be true or false" >&2
  exit 1
fi
if [[ "${LOW_LATENCY_FEATURE_QUEUE_ENABLED}" != "true" && "${LOW_LATENCY_FEATURE_QUEUE_ENABLED}" != "false" ]]; then
  echo "[champion-live] LOW_LATENCY_FEATURE_QUEUE_ENABLED must be true or false" >&2
  exit 1
fi
if ! [[ "${LOW_LATENCY_FEATURE_MAX_RECORDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[champion-live] LOW_LATENCY_FEATURE_MAX_RECORDS must be a positive integer" >&2
  exit 1
fi
if ! [[ "${LIVE_MIN_FREE_BYTES}" =~ ^[0-9]+$ ]]; then
  echo "[champion-live] LIVE_MIN_FREE_BYTES must be a non-negative integer" >&2
  exit 1
fi
ETL_MANIFEST_ARGS=()
if [[ -n "${ETL_PROCESSED_MANIFEST_PATH}" ]]; then
  mkdir -p "$(dirname "${ETL_PROCESSED_MANIFEST_PATH}")"
  ETL_MANIFEST_ARGS=(--processed-manifest-path "${ETL_PROCESSED_MANIFEST_PATH}")
fi

kill_tree() {
  local pid="$1"
  local child
  while read -r child; do
    [[ -z "${child}" ]] && continue
    kill_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill "${pid}" 2>/dev/null || true
}

acquire_live_root_lock() {
  if mkdir "${LIVE_LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" > "${LIVE_LOCK_PID_FILE}"
    LIVE_LOCK_ACQUIRED="true"
    return 0
  fi

  local owner_pid=""
  if [[ -f "${LIVE_LOCK_PID_FILE}" ]]; then
    owner_pid="$(tr -dc '0-9' < "${LIVE_LOCK_PID_FILE}" || true)"
  fi
  if [[ "${owner_pid}" =~ ^[0-9]+$ ]] && kill -0 "${owner_pid}" 2>/dev/null; then
    echo "[champion-live] live root lock held by pid=${owner_pid}: ${LIVE_LOCK_DIR}" >&2
    exit 1
  fi

  echo "[champion-live] removing stale live root lock: ${LIVE_LOCK_DIR}" >&2
  rm -rf "${LIVE_LOCK_DIR}"
  if ! mkdir "${LIVE_LOCK_DIR}" 2>/dev/null; then
    echo "[champion-live] failed to acquire live root lock: ${LIVE_LOCK_DIR}" >&2
    exit 1
  fi
  printf '%s\n' "$$" > "${LIVE_LOCK_PID_FILE}"
  LIVE_LOCK_ACQUIRED="true"
}

release_live_root_lock() {
  if [[ "${LIVE_LOCK_ACQUIRED}" != "true" ]]; then
    return 0
  fi
  local owner_pid=""
  if [[ -f "${LIVE_LOCK_PID_FILE}" ]]; then
    owner_pid="$(tr -dc '0-9' < "${LIVE_LOCK_PID_FILE}" || true)"
  fi
  if [[ "${owner_pid}" == "$$" ]]; then
    rm -rf "${LIVE_LOCK_DIR}"
  fi
  LIVE_LOCK_ACQUIRED="false"
}

CAPTURE_PID=""
SCORER_PID=""
cleanup() {
  if [[ -n "${SCORER_PID}" ]]; then
    echo
    echo "[champion-live] stopping scorer loop pid=${SCORER_PID}"
    kill_tree "${SCORER_PID}"
    wait "${SCORER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${CAPTURE_PID}" ]]; then
    echo "[champion-live] stopping capture service pid=${CAPTURE_PID}"
    kill_tree "${CAPTURE_PID}"
    wait "${CAPTURE_PID}" 2>/dev/null || true
  fi
  release_live_root_lock
}
trap cleanup INT TERM EXIT

start_capture() {
  "${PYTHON_BIN}" -m bigan.ingestion.__main__ serve \
    >> "${CAPTURE_LOG}" 2>&1 &
  CAPTURE_PID="$!"
}

run_step() {
  local step_name="$1"
  local status
  shift
  set +e
  "$@"
  status="$?"
  set -e
  if (( status == 0 )); then
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${step_name} failed status=${status}"
  return "${status}"
}

run_step_capture() {
  local step_name="$1"
  local output_path="$2"
  local status
  shift 2
  set +e
  "$@" 2>&1 | tee "${output_path}"
  status="${PIPESTATUS[0]}"
  set -e
  if (( status == 0 )); then
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${step_name} failed status=${status}"
  return "${status}"
}

check_live_root_free_space() {
  if (( LIVE_MIN_FREE_BYTES <= 0 )); then
    return 0
  fi
  "${PYTHON_BIN}" - "${LIVE_ROOT}" "${LIVE_MIN_FREE_BYTES}" <<'PY'
import os
import sys
from pathlib import Path

live_root = Path(sys.argv[1])
minimum = int(sys.argv[2])
stat = os.statvfs(live_root)
free_bytes = stat.f_bavail * stat.f_frsize
if free_bytes >= minimum:
    raise SystemExit(0)
print(
    "[champion-live] live root filesystem free space below floor: "
    f"free={free_bytes} required={minimum} path={live_root}",
    file=sys.stderr,
)
raise SystemExit(2)
PY
}

etl_files_processed_from_output() {
  local output_path="$1"
  "${PYTHON_BIN}" - "${output_path}" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
decoder = json.JSONDecoder()
files_processed = None
index = 0
while index < len(text):
    if text[index] != "{":
        index += 1
        continue
    try:
        payload, consumed = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        index += 1
        continue
    if isinstance(payload, dict) and "files_processed" in payload:
        files_processed = payload.get("files_processed")
    index += max(consumed, 1)

if files_processed is None:
    sys.exit(2)
print(int(files_processed))
PY
}

sleep_for() {
  local seconds="$1"
  local deadline
  local now
  local remaining
  deadline=$(( $(date +%s) + seconds ))
  while true; do
    now="$(date +%s)"
    remaining=$(( deadline - now ))
    if (( remaining <= 0 )); then
      return 0
    fi
    sleep "${remaining}" || true
  done
}

run_cycle() {
  local cycle_number="$1"
  local cycle_id
  local etl_output
  local etl_files_processed=""
  local -a etl_args
  local -a feature_args
  local -a prediction_args
  cycle_id="$(date -u +%Y%m%dT%H%M%SZ)"
  etl_output="${LOG_DIR}/etl-${cycle_id}-$RANDOM.tmp"
  etl_args=(--lag-seconds "${ETL_EFFECTIVE_LAG_SECONDS}")
  if [[ -n "${ETL_MAX_FILES_PER_BATCH}" ]]; then
    etl_args+=(--max-files-per-batch "${ETL_MAX_FILES_PER_BATCH}")
  fi
  feature_args=(--lookback-minutes "${FEATURE_LOOKBACK_MINUTES}" --skip-existing)
  prediction_args=(
    --model-path "${MODEL_PATH}"
  )
  if [[ "${IS_EMBEDDED_CALIBRATION_MODEL}" != "true" ]]; then
    prediction_args+=(--calibration-path "${CALIBRATION_PATH}")
  fi
  prediction_args+=(
    --monitoring-db-path "${MONITORING_DB_PATH}"
    --lookback-minutes "${PREDICTION_LOOKBACK_MINUTES}"
    --skip-existing-monitoring-events
    --skip-existing-predictions
  )
  if [[ -n "${ETL_PROCESSED_MANIFEST_PATH}" ]]; then
    etl_args+=("${ETL_MANIFEST_ARGS[@]}")
  fi
  if [[ -n "${SCORING_CANONICAL_SYMBOL_LIKE}" ]]; then
    feature_args+=(--canonical-symbol-like "${SCORING_CANONICAL_SYMBOL_LIKE}")
    prediction_args+=(--canonical-symbol-like "${SCORING_CANONICAL_SYMBOL_LIKE}")
  fi
  if [[ -n "${SIGNAL_JSONL_OUTPUT_PATH}" ]]; then
    prediction_args+=(
      --signal-jsonl-output-path "${SIGNAL_JSONL_OUTPUT_PATH}"
      --signal-jsonl-market-families "${SIGNAL_JSONL_MARKET_FAMILIES}"
      --signal-jsonl-outcome-side "${SIGNAL_JSONL_OUTCOME_SIDE}"
      --v6-settlement-threshold "${V6_SETTLEMENT_THRESHOLD}"
      --v6-neutral-cap "${V6_NEUTRAL_CAP}"
      --v6-volatility-threshold "${V6_VOLATILITY_THRESHOLD}"
      --v6-round-trip-cost "${V6_ROUND_TRIP_COST}"
      --v6-ev-margin "${V6_EV_MARGIN}"
    )
    if [[ -n "${SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS}" ]]; then
      prediction_args+=(
        --signal-jsonl-max-event-age-seconds "${SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS}"
      )
    fi
  fi

  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] scan ${cycle_id} started"
  echo "live_data_root=${LIVE_ROOT}"

  if [[ "${LOW_LATENCY_FEATURE_QUEUE_ENABLED}" == "true" ]]; then
    run_step "features-15m-v1-low-latency-queue" \
      "${PYTHON_BIN}" -m bigan.ingestion.__main__ features-15m-v1-low-latency-queue \
      --queue-path "${LOW_LATENCY_RAW_QUEUE_PATH}" \
      --cursor-path "${LOW_LATENCY_FEATURE_CURSOR_PATH}" \
      --state-path "${LOW_LATENCY_FEATURE_STATE_PATH}" \
      --canonical-symbol-prefix "${LOW_LATENCY_RAW_QUEUE_CANONICAL_SYMBOL_PREFIX}" \
      --max-records "${LOW_LATENCY_FEATURE_MAX_RECORDS}" || return "$?"

    run_step "predictions-v1" \
      "${PYTHON_BIN}" -m bigan.ingestion.__main__ predictions-v1 \
      "${prediction_args[@]}" || return "$?"
  else
    run_step_capture "etl-batch" "${etl_output}" \
      "${PYTHON_BIN}" -m bigan.ingestion.__main__ etl-batch \
      "${etl_args[@]}" || {
        local status="$?"
        rm -f "${etl_output}"
        return "${status}"
      }
    if etl_files_processed="$(etl_files_processed_from_output "${etl_output}" 2>/dev/null)"; then
      echo "etl_files_processed=${etl_files_processed}"
    else
      echo "etl_files_processed=unknown"
    fi
    rm -f "${etl_output}"

    if [[ "${SCORE_ONLY_WHEN_ETL_PROCESSED}" == "true" && "${etl_files_processed}" == "0" ]]; then
      echo "scoring skipped cycle=${cycle_number} files_processed=0"
    else

      run_step "features-15m-v1" \
        "${PYTHON_BIN}" -m bigan.ingestion.__main__ features-15m-v1 \
        "${feature_args[@]}" || return "$?"

      run_step "predictions-v1" \
        "${PYTHON_BIN}" -m bigan.ingestion.__main__ predictions-v1 \
        "${prediction_args[@]}" || return "$?"
    fi
  fi

  if [[ "${LABELS_ENABLED}" == "true" && $((cycle_number % LABELS_EVERY_CYCLES)) -eq 0 ]]; then
    run_step "labels-15m-v1" \
      "${PYTHON_BIN}" -m bigan.ingestion.__main__ labels-15m-v1 \
      --monitoring-db-path "${MONITORING_DB_PATH}" \
      --monitoring-model-version "${MODEL_VERSION}" \
      --request-timeout-seconds "${LABEL_REQUEST_TIMEOUT_SECONDS}" \
      --request-concurrency "${LABEL_REQUEST_CONCURRENCY}" \
      --lookback-minutes "${LABEL_LOOKBACK_MINUTES}" \
      --fee-bps "${FEE_BPS}" \
      --skip-existing-labels || return "$?"
  else
    echo "labels skipped cycle=${cycle_number} enabled=${LABELS_ENABLED} every=${LABELS_EVERY_CYCLES}"
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] scan ${cycle_id} completed"
}

score_loop() {
  local cycles_completed=0
  local cycles_attempted=0
  while true; do
    if ! check_live_root_free_space >> "${SCORER_LOG}" 2>&1; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] live root free-space floor breached; stopping scorer loop" >> "${SCORER_LOG}"
      return 2
    fi
    cycles_attempted=$((cycles_attempted + 1))
    if run_cycle "${cycles_attempted}" >> "${SCORER_LOG}" 2>&1; then
      cycles_completed=$((cycles_completed + 1))
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] scan failed; retrying after ${CYCLE_SLEEP_SECONDS}s" >> "${SCORER_LOG}"
    fi

    if [[ -n "${STOP_AFTER_CYCLES}" && "${cycles_completed}" -ge "${STOP_AFTER_CYCLES}" ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] STOP_AFTER_CYCLES reached" >> "${SCORER_LOG}"
      return 0
    fi
    sleep_for "${CYCLE_SLEEP_SECONDS}"
  done
}

echo "[champion-live] repo=${REPO_ROOT}"
echo "[champion-live] model version=${MODEL_VERSION}"
echo "[champion-live] model path=${MODEL_PATH}"
if [[ "${IS_EMBEDDED_CALIBRATION_MODEL}" == "true" ]]; then
  echo "[champion-live] calibration path=(embedded in model artifact)"
else
  echo "[champion-live] calibration path=${CALIBRATION_PATH}"
fi
echo "[champion-live] live data root=${LIVE_ROOT}"
echo "[champion-live] monitoring db=${MONITORING_DB_PATH}"
echo "[champion-live] capture log=${CAPTURE_LOG}"
echo "[champion-live] scorer log=${SCORER_LOG}"
echo "[champion-live] cycle sleep seconds=${CYCLE_SLEEP_SECONDS}"
echo "[champion-live] feature lookback minutes=${FEATURE_LOOKBACK_MINUTES}"
echo "[champion-live] prediction lookback minutes=${PREDICTION_LOOKBACK_MINUTES}"
echo "[champion-live] ETL lag seconds=${ETL_EFFECTIVE_LAG_SECONDS}"
echo "[champion-live] score only when ETL processed=${SCORE_ONLY_WHEN_ETL_PROCESSED}"
echo "[champion-live] low-latency feature queue enabled=${LOW_LATENCY_FEATURE_QUEUE_ENABLED}"
if [[ -n "${SCORING_CANONICAL_SYMBOL_LIKE}" ]]; then
  echo "[champion-live] scoring canonical symbol like=${SCORING_CANONICAL_SYMBOL_LIKE}"
fi
echo "[champion-live] signal jsonl output=${SIGNAL_JSONL_OUTPUT_PATH:-<none>}"
if [[ -n "${SIGNAL_JSONL_OUTPUT_PATH}" ]]; then
  echo "[champion-live] signal jsonl families=${SIGNAL_JSONL_MARKET_FAMILIES} outcome_side=${SIGNAL_JSONL_OUTCOME_SIDE}"
  echo "[champion-live] signal jsonl max event age seconds=${SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS:-<none>}"
fi
if [[ "${LOW_LATENCY_FEATURE_QUEUE_ENABLED}" == "true" ]]; then
  echo "[champion-live] low-latency raw queue=${LOW_LATENCY_RAW_QUEUE_PATH}"
  echo "[champion-live] low-latency feature cursor=${LOW_LATENCY_FEATURE_CURSOR_PATH}"
  echo "[champion-live] low-latency feature state=${LOW_LATENCY_FEATURE_STATE_PATH}"
fi
echo "[champion-live] live min free bytes=${LIVE_MIN_FREE_BYTES}"
check_live_root_free_space
acquire_live_root_lock
echo "[champion-live] live root lock=${LIVE_LOCK_DIR}"
if [[ "${ETL_EFFECTIVE_LAG_SECONDS}" != "${ETL_LAG_SECONDS}" ]]; then
  echo "[champion-live] ETL lag raised from ${ETL_LAG_SECONDS}s for segmented raw files"
fi
echo "[champion-live] labels enabled=${LABELS_ENABLED}; every ${LABELS_EVERY_CYCLES} cycle(s)"
if [[ -n "${ETL_PROCESSED_MANIFEST_PATH}" ]]; then
  echo "[champion-live] ETL processed manifest=${ETL_PROCESSED_MANIFEST_PATH}"
fi
if [[ -n "${MARKET_SPECS_JSON}" ]]; then
  echo "[champion-live] multi-market specs enabled"
fi

echo "[champion-live] starting continuous capture service"
start_capture
echo "[champion-live] capture pid=${CAPTURE_PID}; warming up ${SCAN_STARTUP_SECONDS}s"
sleep_for "${SCAN_STARTUP_SECONDS}"
if ! kill -0 "${CAPTURE_PID}" 2>/dev/null; then
  echo "[champion-live] capture service exited during warmup" >&2
  wait "${CAPTURE_PID}" 2>/dev/null || true
  exit 1
fi

echo "[champion-live] starting scorer loop in background"
score_loop &
SCORER_PID="$!"

if [[ "${DASHBOARD_ENABLED}" == "false" || -n "${STOP_AFTER_CYCLES}" ]]; then
  echo "[champion-live] dashboard disabled; waiting for scorer loop"
  wait "${SCORER_PID}"
  exit 0
fi

echo "[champion-live] starting signal dashboard in foreground"
echo "[champion-live] press Ctrl-C to stop"

"${PYTHON_BIN}" -m bigan.ingestion.__main__ signals-dashboard \
  --monitoring-db-path "${MONITORING_DB_PATH}" \
  --model-version "${MODEL_VERSION}" \
  --edge-threshold "${EDGE_THRESHOLD}" \
  --exit-edge-threshold "${EXIT_EDGE_THRESHOLD}" \
  --lookback-hours "${DASHBOARD_LOOKBACK_HOURS}" \
  --poll-seconds "${TAIL_POLL_SECONDS}"
