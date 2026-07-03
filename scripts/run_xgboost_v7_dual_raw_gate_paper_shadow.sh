#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run two xgboost-v7 paper-shadow executors against the same Kafka signal topic.

The two executor tracks differ only in raw-side entry gate strictness:
  - baseline: current raw-side gate settings
  - raw60: V7_RAW_SIDE_MIN_PROBABILITY=0.60 and center conviction=0.60

Both tracks use independent Kafka consumer groups, log dirs, and paper execution
DBs, so they can consume the same scorer output side-by-side without sharing
offsets or open-position state.

Required:
  SIGNAL_KAFKA_BOOTSTRAP_SERVERS
  SIGNAL_KAFKA_TOPIC

Important environment overrides:
  MAX_ROUNDS                         Default: 30
  DUAL_RAW_GATE_RUN_ID               Default: UTC timestamp
  DUAL_RAW_GATE_LOG_ROOT             Default: data/logs
  DUAL_RAW_GATE_BASELINE_LABEL       Default: baseline-raw-current
  DUAL_RAW_GATE_PROPOSED_LABEL       Default: raw60
  DUAL_RAW_GATE_BASE_GROUP_ID        Default: bigan-xgboost-v7-paper-shadow-${run_id}
  BASELINE_MONITORING_DB_PATH        Default: ${baseline_log_dir}/execution.duckdb
  PROPOSED_MONITORING_DB_PATH        Default: ${proposed_log_dir}/execution.duckdb
  BASELINE_V7_RAW_SIDE_MIN_PROBABILITY
                                     Default: current V7_RAW_SIDE_MIN_PROBABILITY or 0.50
  BASELINE_V7_RAW_SIDE_AGREEMENT_ENABLED
                                     Default: current V7_RAW_SIDE_AGREEMENT_ENABLED or true.
                                     Set false to compare no raw-side gate vs raw60.
  BASELINE_V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY
                                     Default: current center conviction or 0.57
  PROPOSED_V7_RAW_SIDE_MIN_PROBABILITY
                                     Default: 0.60
  PROPOSED_V7_RAW_SIDE_AGREEMENT_ENABLED
                                     Default: true
  PROPOSED_V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY
                                     Default: 0.60
  PLAN_ONLY                          Default: false. Print both resolved child
                                     executor commands without running them.

Any env var understood by run_xgboost_v7_paper_shadow.sh can also be set here;
this wrapper forwards the current environment and only overrides group/log/raw
gate values per track.

Example:
  SIGNAL_KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  SIGNAL_KAFKA_TOPIC=bigan.signals \
  MAX_ROUNDS=30 \
  bash scripts/run_xgboost_v7_dual_raw_gate_paper_shadow.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${SIGNAL_KAFKA_BOOTSTRAP_SERVERS:-}" || -z "${SIGNAL_KAFKA_TOPIC:-}" ]]; then
  echo "[v7-dual-raw-gate] SIGNAL_KAFKA_BOOTSTRAP_SERVERS and SIGNAL_KAFKA_TOPIC are required." >&2
  echo "[v7-dual-raw-gate] This wrapper intentionally compares Kafka executors with separate consumer groups." >&2
  exit 1
fi

PLAN_ONLY="${PLAN_ONLY:-false}"
if [[ "${PLAN_ONLY}" != "true" && "${PLAN_ONLY}" != "false" ]]; then
  echo "[v7-dual-raw-gate] PLAN_ONLY must be true or false" >&2
  exit 1
fi

RUN_ID="${DUAL_RAW_GATE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
LOG_ROOT="${DUAL_RAW_GATE_LOG_ROOT:-data/logs}"
BASELINE_LABEL="${DUAL_RAW_GATE_BASELINE_LABEL:-baseline-raw-current}"
PROPOSED_LABEL="${DUAL_RAW_GATE_PROPOSED_LABEL:-raw60}"
BASE_GROUP_ID="${DUAL_RAW_GATE_BASE_GROUP_ID:-bigan-xgboost-v7-paper-shadow-${RUN_ID}}"
RUN_ROOT="${LOG_ROOT}/xgboost-v7-paper-shadow-${RUN_ID}-dual-raw-gate-${MAX_ROUNDS}round"

BASELINE_RAW_MIN="${BASELINE_V7_RAW_SIDE_MIN_PROBABILITY:-${V7_RAW_SIDE_MIN_PROBABILITY:-0.50}}"
BASELINE_RAW_ENABLED="${BASELINE_V7_RAW_SIDE_AGREEMENT_ENABLED:-${V7_RAW_SIDE_AGREEMENT_ENABLED:-true}}"
BASELINE_PRICE_CONVICTION_ENABLED="${BASELINE_V7_RAW_SIDE_PRICE_CONVICTION_ENABLED:-${V7_RAW_SIDE_PRICE_CONVICTION_ENABLED:-true}}"
BASELINE_CENTER_MIN="${BASELINE_V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY:-${V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY:-0.57}}"
PROPOSED_RAW_MIN="${PROPOSED_V7_RAW_SIDE_MIN_PROBABILITY:-0.60}"
PROPOSED_RAW_ENABLED="${PROPOSED_V7_RAW_SIDE_AGREEMENT_ENABLED:-true}"
PROPOSED_PRICE_CONVICTION_ENABLED="${PROPOSED_V7_RAW_SIDE_PRICE_CONVICTION_ENABLED:-${V7_RAW_SIDE_PRICE_CONVICTION_ENABLED:-true}}"
PROPOSED_CENTER_MIN="${PROPOSED_V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY:-0.60}"

BASELINE_GROUP_ID="${BASE_GROUP_ID}-${BASELINE_LABEL}"
PROPOSED_GROUP_ID="${BASE_GROUP_ID}-${PROPOSED_LABEL}"
BASELINE_LOG_DIR="${RUN_ROOT}/${BASELINE_LABEL}"
PROPOSED_LOG_DIR="${RUN_ROOT}/${PROPOSED_LABEL}"
BASELINE_MONITORING_DB_PATH="${BASELINE_MONITORING_DB_PATH:-${BASELINE_LOG_DIR}/execution.duckdb}"
PROPOSED_MONITORING_DB_PATH="${PROPOSED_MONITORING_DB_PATH:-${PROPOSED_LOG_DIR}/execution.duckdb}"

mkdir -p "${BASELINE_LOG_DIR}" "${PROPOSED_LOG_DIR}"
mkdir -p "$(dirname "${BASELINE_MONITORING_DB_PATH}")" "$(dirname "${PROPOSED_MONITORING_DB_PATH}")"

echo "[v7-dual-raw-gate] run_id=${RUN_ID}"
echo "[v7-dual-raw-gate] topic=${SIGNAL_KAFKA_TOPIC} bootstrap=${SIGNAL_KAFKA_BOOTSTRAP_SERVERS}"
echo "[v7-dual-raw-gate] baseline group=${BASELINE_GROUP_ID} log_dir=${BASELINE_LOG_DIR} monitoring_db=${BASELINE_MONITORING_DB_PATH} raw_enabled=${BASELINE_RAW_ENABLED} raw_min=${BASELINE_RAW_MIN} price_conviction=${BASELINE_PRICE_CONVICTION_ENABLED} center_min=${BASELINE_CENTER_MIN}"
echo "[v7-dual-raw-gate] proposed group=${PROPOSED_GROUP_ID} log_dir=${PROPOSED_LOG_DIR} monitoring_db=${PROPOSED_MONITORING_DB_PATH} raw_enabled=${PROPOSED_RAW_ENABLED} raw_min=${PROPOSED_RAW_MIN} price_conviction=${PROPOSED_PRICE_CONVICTION_ENABLED} center_min=${PROPOSED_CENTER_MIN}"

run_track() {
  local label="$1"
  local group_id="$2"
  local log_dir="$3"
  local monitoring_db_path="$4"
  local raw_enabled="$5"
  local raw_min="$6"
  local price_conviction_enabled="$7"
  local center_min="$8"

  echo "[v7-dual-raw-gate] starting ${label} monitoring_db=${monitoring_db_path}"
  env \
    LOG_DIR="${log_dir}" \
    MONITORING_DB_PATH="${monitoring_db_path}" \
    SIGNAL_KAFKA_GROUP_ID="${group_id}" \
    V7_RAW_SIDE_AGREEMENT_ENABLED="${raw_enabled}" \
    V7_RAW_SIDE_MIN_PROBABILITY="${raw_min}" \
    V7_RAW_SIDE_PRICE_CONVICTION_ENABLED="${price_conviction_enabled}" \
    V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY="${center_min}" \
    MAX_ROUNDS="${MAX_ROUNDS}" \
    PLAN_ONLY="${PLAN_ONLY}" \
    bash scripts/run_xgboost_v7_paper_shadow.sh
}

if [[ "${PLAN_ONLY}" == "true" ]]; then
  run_track "${BASELINE_LABEL}" "${BASELINE_GROUP_ID}" "${BASELINE_LOG_DIR}" "${BASELINE_MONITORING_DB_PATH}" "${BASELINE_RAW_ENABLED}" "${BASELINE_RAW_MIN}" "${BASELINE_PRICE_CONVICTION_ENABLED}" "${BASELINE_CENTER_MIN}"
  run_track "${PROPOSED_LABEL}" "${PROPOSED_GROUP_ID}" "${PROPOSED_LOG_DIR}" "${PROPOSED_MONITORING_DB_PATH}" "${PROPOSED_RAW_ENABLED}" "${PROPOSED_RAW_MIN}" "${PROPOSED_PRICE_CONVICTION_ENABLED}" "${PROPOSED_CENTER_MIN}"
  echo "[v7-dual-raw-gate] plan_only=true; no executors started."
  exit 0
fi

run_track "${BASELINE_LABEL}" "${BASELINE_GROUP_ID}" "${BASELINE_LOG_DIR}" "${BASELINE_MONITORING_DB_PATH}" "${BASELINE_RAW_ENABLED}" "${BASELINE_RAW_MIN}" "${BASELINE_PRICE_CONVICTION_ENABLED}" "${BASELINE_CENTER_MIN}" &
baseline_pid=$!
run_track "${PROPOSED_LABEL}" "${PROPOSED_GROUP_ID}" "${PROPOSED_LOG_DIR}" "${PROPOSED_MONITORING_DB_PATH}" "${PROPOSED_RAW_ENABLED}" "${PROPOSED_RAW_MIN}" "${PROPOSED_PRICE_CONVICTION_ENABLED}" "${PROPOSED_CENTER_MIN}" &
proposed_pid=$!

cleanup() {
  local status=$?
  if kill -0 "${baseline_pid}" 2>/dev/null; then
    kill "${baseline_pid}" 2>/dev/null || true
  fi
  if kill -0 "${proposed_pid}" 2>/dev/null; then
    kill "${proposed_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup INT TERM

baseline_status=0
proposed_status=0
wait "${baseline_pid}" || baseline_status=$?
wait "${proposed_pid}" || proposed_status=$?

echo "[v7-dual-raw-gate] baseline_exit_status=${baseline_status}"
echo "[v7-dual-raw-gate] proposed_exit_status=${proposed_status}"
echo "[v7-dual-raw-gate] baseline_log_dir=${BASELINE_LOG_DIR}"
echo "[v7-dual-raw-gate] proposed_log_dir=${PROPOSED_LOG_DIR}"

if [[ "${baseline_status}" -ne 0 || "${proposed_status}" -ne 0 ]]; then
  exit 1
fi
