#!/usr/bin/env bash
# Supervised long-run capture+scorer for a Stage 3 shadow window.
#
# Wraps run_champion_live.sh and restarts it on the SAME LIVE_ROOT whenever raw
# capture goes stale (the failure mode observed previously: the WS capture
# silently stops feeding raw data while the process stays alive, so the scorer
# loop spins on nothing and feature_ts stops advancing). Restarting into the
# same LIVE_ROOT keeps a single contiguous warehouse with only a bounded gap.
#
# Environment overrides:
#   LIVE_ROOT            Live data root reused across restarts (required for stability).
#   STALE_THRESHOLD_SECONDS  Max raw-staleness before restart. Default: 720 (12m).
#   CHECK_INTERVAL_SECONDS   Freshness poll cadence. Default: 120.
#   STOP_FILE            Touch this path to stop the supervisor + run. Default: <LIVE_ROOT>/.stop
#   SUPERVISOR_LOG       Supervisor log path. Default: under LOG_DIR.
# All run_champion_live.sh env vars (MODEL_VERSION, MODEL_PATH, etc.) pass through.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LIVE_ROOT="${LIVE_ROOT:-data/live/v5-stage3-shadow-${SESSION_ID}}"
LOG_DIR="${LOG_DIR:-data/logs/champion-live}"
# Default 3600s: a healthy scan cycle can take 10–12+ minutes without a new
# feature parquet; 720s caused false-positive restarts mid-ETL (restart storm).
STALE_THRESHOLD_SECONDS="${STALE_THRESHOLD_SECONDS:-3600}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
STOP_FILE="${STOP_FILE:-${LIVE_ROOT}/.stop}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-${LOG_DIR}/supervisor-${SESSION_ID}.log}"

export LIVE_ROOT LOG_DIR
export ETL_PROCESSED_MANIFEST_PATH="${ETL_PROCESSED_MANIFEST_PATH:-${LIVE_ROOT}/etl-processed.ndjson}"
export ETL_MAX_FILES_PER_BATCH="${ETL_MAX_FILES_PER_BATCH:-1}"
export ETL_LAG_SECONDS="${ETL_LAG_SECONDS:-60}"
export BIGAN_SINK_SEGMENT_DURATION_SECONDS="${BIGAN_SINK_SEGMENT_DURATION_SECONDS:-900}"
mkdir -p "${LIVE_ROOT}" "${LOG_DIR}"

RUN_PID=""

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [supervisor] $*" >> "${SUPERVISOR_LOG}"; }

kill_tree() {
  local pid="$1" child
  [[ -z "${pid}" ]] && return 0
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill "${pid}" 2>/dev/null || true
}

launch_run() {
  DASHBOARD_ENABLED=false nohup "${SCRIPT_DIR}/run_champion_live.sh" \
    >> "${LOG_DIR}/run-stage3-${SESSION_ID}.log" 2>&1 &
  RUN_PID="$!"
  log "launched run_champion_live.sh pid=${RUN_PID} live_root=${LIVE_ROOT}"
}

newest_feature_mtime() {
  # Most recent mtime (epoch seconds) of a generated feature parquet; 0 if none.
  # Feature freshness is the signal that matters: the prior stall kept writing
  # raw files (duplicate content) while feature_ts stopped advancing, so raw
  # mtime alone would have missed it. New 15M feature rows should appear within
  # a minute under healthy capture+ETL.
  local newest
  newest="$(find "${LIVE_ROOT}/warehouse/features_15m_v1" \
    -type f -name '*.parquet' 2>/dev/null -exec stat -f '%m' {} + 2>/dev/null | sort -rn | head -1)"
  echo "${newest:-0}"
}

cleanup() {
  log "supervisor stopping; tearing down run pid=${RUN_PID}"
  kill_tree "${RUN_PID}"
  wait "${RUN_PID}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

log "supervisor start session=${SESSION_ID} stale_threshold=${STALE_THRESHOLD_SECONDS}s interval=${CHECK_INTERVAL_SECONDS}s"
launch_run
# Grace period for first capture warmup before staleness checks begin.
sleep 90

while true; do
  if [[ -f "${STOP_FILE}" ]]; then
    log "stop file present (${STOP_FILE}); exiting"
    break
  fi
  if ! kill -0 "${RUN_PID}" 2>/dev/null; then
    log "ALERT run process pid=${RUN_PID} exited; relaunching"
    launch_run
    sleep 90
    continue
  fi
  now="$(date +%s)"
  mtime="$(newest_feature_mtime)"
  age=$(( now - mtime ))
  if (( mtime > 0 && age > STALE_THRESHOLD_SECONDS )); then
    log "ALERT feature generation stale age=${age}s > ${STALE_THRESHOLD_SECONDS}s; restarting run on same LIVE_ROOT"
    kill_tree "${RUN_PID}"
    wait "${RUN_PID}" 2>/dev/null || true
    sleep 5
    launch_run
    sleep 90
    continue
  fi
  log "ok run_pid=${RUN_PID} feature_age=${age}s"
  sleep "${CHECK_INTERVAL_SECONDS}"
done
