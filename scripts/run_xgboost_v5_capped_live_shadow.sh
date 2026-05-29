#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the capped xgboost-v5 Phase 4 live shadow executor.

This locks the v5 operating policy validated by the cost-adjusted backtest
(docs/reports/backtest_v5_vs_v4/README.md):

  - model_version  : xgboost-v5
  - market families : BTC-15M,ETH-15M  (5M skipped; only 15M showed real edge)
  - edge threshold : 0.08
  - tiny capped size, concurrency, rounds, and daily-loss limit

It places REAL FOK orders (this is a capped live shadow, not a paper run). The
purpose is account cash-flow PnL reconciliation, NOT promotion or sizing. Keep
the caps small. The v5 scorer must already be feeding predictions into the same
monitoring DB (see scripts/run_xgboost_v5_capped_live_shadow scorer note below).

Prerequisites:
  - v5 scorer writing predictions for model_version=xgboost-v5 into MONITORING_DB_PATH.
    Start it in another terminal, restricted to 15M families:
      MODEL_VERSION=xgboost-v5 \
      MODEL_PATH=data/model-runs/xgboost-v5-run-20260529T053000Z/model/model.json \
      CALIBRATION_PATH=data/model-runs/xgboost-v5-run-20260529T053000Z/calibration-family/calibration.json \
      SCORING_CANONICAL_SYMBOL_LIKE='%-15M:%' \
      MARKET_SPECS_JSON='[
        {"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15},
        {"slug_prefix":"eth-updown-15m-","underlying":"ETH","horizon_minutes":15}
      ]' \
      ./scripts/run_champion_live.sh
  - Polymarket credentials in env: POLYMARKET_PRIVATE_KEY (and POLYMARKET_FUNDER
    for proxy wallets), plus optional POLYMARKET_SIGNATURE_TYPE / POLYMARKET_HOST.

Environment overrides:
  PYTHON_BIN                 Python executable. Default: .venv/bin/python
  MODEL_VERSION              Default: xgboost-v5
  MARKET_FAMILIES            Default: BTC-15M,ETH-15M
  EDGE_THRESHOLD             Entry edge gate. Default: 0.08
  MONITORING_DB_PATH         DuckDB catalog. Default: data/mlops/champion_catalog.duckdb
  MAX_POSITION_SIZE_USDC     Per-entry spend cap. Default: 1.0
  MAX_CONCURRENT_POSITIONS   Default: 1
  MAX_ROUNDS                 Total rounds (entries) cap. Default: 6
  DAILY_LOSS_LIMIT_USDC      Hard realized-loss stop. Default: 3.0
  MAX_RUNTIME_MINUTES        Default: 120
  POLL_SECONDS               Default: 10
  MIN_SECONDS_TO_EXPIRY      Default: 180
  MAX_SECONDS_TO_EXPIRY      Default: 1200
  SIGNAL_JSONL_PATH          Bridged signal JSONL queue. When set, read signals
                             from this file instead of scanning the monitoring DB
                             (split topology: local scorer + bridge, host executor).
  SIGNAL_JSONL_START         Queue start position: tail|beginning. Default: tail
  LOG_DIR                    Default: logs/xgboost-v5-live-shadow
  CONFIRM                    Must be "yes" to place real orders. Default: empty (refuses).

Example (after exporting Polymarket creds and starting the v5 scorer):
  CONFIRM=yes ./scripts/run_xgboost_v5_capped_live_shadow.sh
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
MODEL_VERSION="${MODEL_VERSION:-xgboost-v5}"
MARKET_FAMILIES="${MARKET_FAMILIES:-BTC-15M,ETH-15M}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-0.08}"
MONITORING_DB_PATH="${MONITORING_DB_PATH:-data/mlops/champion_catalog.duckdb}"
MAX_POSITION_SIZE_USDC="${MAX_POSITION_SIZE_USDC:-1.0}"
MAX_CONCURRENT_POSITIONS="${MAX_CONCURRENT_POSITIONS:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
DAILY_LOSS_LIMIT_USDC="${DAILY_LOSS_LIMIT_USDC:-3.0}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-120}"
POLL_SECONDS="${POLL_SECONDS:-10}"
MIN_SECONDS_TO_EXPIRY="${MIN_SECONDS_TO_EXPIRY:-180}"
MAX_SECONDS_TO_EXPIRY="${MAX_SECONDS_TO_EXPIRY:-1200}"
SIGNAL_JSONL_PATH="${SIGNAL_JSONL_PATH:-}"
SIGNAL_JSONL_START="${SIGNAL_JSONL_START:-tail}"
LOG_DIR="${LOG_DIR:-logs/xgboost-v5-live-shadow}"
CONFIRM="${CONFIRM:-}"

if [[ "${CONFIRM}" != "yes" ]]; then
  echo "[v5-live-shadow] refusing to start: this places REAL capped orders." >&2
  echo "[v5-live-shadow] re-run with CONFIRM=yes once the v5 scorer and creds are ready." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[v5-live-shadow] missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${POLYMARKET_PRIVATE_KEY:-}" ]]; then
  echo "[v5-live-shadow] POLYMARKET_PRIVATE_KEY is required for live execution" >&2
  exit 1
fi
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  if [[ ! -f "${SIGNAL_JSONL_PATH}" ]]; then
    echo "[v5-live-shadow] signal jsonl queue not found: ${SIGNAL_JSONL_PATH}" >&2
    echo "[v5-live-shadow] start the local v5 scorer + bridge first so the queue exists." >&2
    exit 1
  fi
  if [[ "${SIGNAL_JSONL_START}" != "tail" && "${SIGNAL_JSONL_START}" != "beginning" ]]; then
    echo "[v5-live-shadow] SIGNAL_JSONL_START must be tail or beginning" >&2
    exit 1
  fi
elif [[ ! -f "${MONITORING_DB_PATH}" ]]; then
  echo "[v5-live-shadow] monitoring db not found: ${MONITORING_DB_PATH}" >&2
  echo "[v5-live-shadow] start the v5 scorer first, or set SIGNAL_JSONL_PATH for split topology." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/phase4-${SESSION_ID}.jsonl"
SUMMARY_PATH="${LOG_DIR}/phase4-${SESSION_ID}-summary.json"

export PYTHONPATH="${PYTHONPATH:-src}"

echo "[v5-live-shadow] repo=${REPO_ROOT}"
echo "[v5-live-shadow] model_version=${MODEL_VERSION}"
echo "[v5-live-shadow] market_families=${MARKET_FAMILIES}"
echo "[v5-live-shadow] edge_threshold=${EDGE_THRESHOLD}"
echo "[v5-live-shadow] monitoring_db=${MONITORING_DB_PATH}"
echo "[v5-live-shadow] caps: size=${MAX_POSITION_SIZE_USDC} concurrent=${MAX_CONCURRENT_POSITIONS} rounds=${MAX_ROUNDS} daily_loss=${DAILY_LOSS_LIMIT_USDC}"
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  echo "[v5-live-shadow] signal_source=jsonl path=${SIGNAL_JSONL_PATH} start=${SIGNAL_JSONL_START}"
else
  echo "[v5-live-shadow] signal_source=duckdb db=${MONITORING_DB_PATH}"
fi
echo "[v5-live-shadow] log=${LOG_PATH}"
echo "[v5-live-shadow] summary=${SUMMARY_PATH}"

EXEC_ARGS=(
  --model-version "${MODEL_VERSION}"
  --market-families "${MARKET_FAMILIES}"
  --edge-threshold "${EDGE_THRESHOLD}"
  --monitoring-db-path "${MONITORING_DB_PATH}"
  --max-position-size-usdc "${MAX_POSITION_SIZE_USDC}"
  --max-concurrent-positions "${MAX_CONCURRENT_POSITIONS}"
  --max-rounds "${MAX_ROUNDS}"
  --daily-loss-limit-usdc "${DAILY_LOSS_LIMIT_USDC}"
  --max-runtime-minutes "${MAX_RUNTIME_MINUTES}"
  --poll-seconds "${POLL_SECONDS}"
  --min-seconds-to-expiry "${MIN_SECONDS_TO_EXPIRY}"
  --max-seconds-to-expiry "${MAX_SECONDS_TO_EXPIRY}"
  --log-path "${LOG_PATH}"
  --summary-path "${SUMMARY_PATH}"
)
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  EXEC_ARGS+=(--signal-jsonl-path "${SIGNAL_JSONL_PATH}" --signal-jsonl-start "${SIGNAL_JSONL_START}")
fi

exec "${PYTHON_BIN}" scripts/polymarket_phase4_live_champion_executor.py "${EXEC_ARGS[@]}"
