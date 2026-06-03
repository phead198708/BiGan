#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run xgboost-v6 Phase 4 paper/orderbook-only shadow (no live promotion).

This path is evidence-only:
  - orderbook-only execution (PAPER=true, no CLOB orders)
  - v6 settlement gate plus separate orderbook-only volatility sleeve
  - account-cashflow reconciliation required before any promotion decision

Prerequisites (local scorer in another terminal):
  MODEL_VERSION=xgboost-v6 \
  MODEL_PATH=data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/model-single-grid/model.json \
  SCORING_CANONICAL_SYMBOL_LIKE='BTC-15M:%' \
  MARKET_SPECS_JSON='[{"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15}]' \
  ./scripts/run_champion_live.sh

Optional split topology:
  - local scorer + scripts/champion_signal_bridge.py (model_version=xgboost-v6)
  - remote executor with SIGNAL_JSONL_PATH pointing at bridged queue

Environment overrides:
  MODEL_VERSION              Default: xgboost-v6
  MODEL_JSON_PATH            v6 artifact for gain priors
  MARKET_FAMILIES            Default: BTC-15M
  V6_SETTLEMENT_THRESHOLD    Default: 0.50
  V6_NEUTRAL_CAP             Default: 0.25
  V6_VOLATILITY_THRESHOLD    Default: 0.60
  V6_ROUND_TRIP_COST         Default: 0.072
  V6_EV_MARGIN               Default: 0.01
  ENABLE_VOLATILITY_SLEEVE   Default: true (still paper-only)
  MONITORING_DB_PATH         Default: data/mlops/champion_catalog.duckdb
  SIGNAL_JSONL_PATH          Optional bridged queue for split topology
  LOG_DIR                    Default: logs/xgboost-v6-paper-shadow

Example:
  ./scripts/run_xgboost_v6_paper_shadow.sh
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
MODEL_VERSION="${MODEL_VERSION:-xgboost-v6}"
MODEL_JSON_PATH="${MODEL_JSON_PATH:-data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/model-single-grid/model.json}"
MARKET_FAMILIES="${MARKET_FAMILIES:-BTC-15M}"
V6_SETTLEMENT_THRESHOLD="${V6_SETTLEMENT_THRESHOLD:-0.50}"
V6_NEUTRAL_CAP="${V6_NEUTRAL_CAP:-0.25}"
V6_VOLATILITY_THRESHOLD="${V6_VOLATILITY_THRESHOLD:-0.60}"
V6_ROUND_TRIP_COST="${V6_ROUND_TRIP_COST:-0.072}"
V6_EV_MARGIN="${V6_EV_MARGIN:-0.01}"
ENABLE_VOLATILITY_SLEEVE="${ENABLE_VOLATILITY_SLEEVE:-true}"
PAPER="true"
MONITORING_DB_PATH="${MONITORING_DB_PATH:-data/mlops/champion_catalog.duckdb}"
MAX_POSITION_SIZE_USDC="${MAX_POSITION_SIZE_USDC:-1.0}"
MAX_CONCURRENT_POSITIONS="${MAX_CONCURRENT_POSITIONS:-1}"
MAX_COMBINED_CONCURRENT_POSITIONS="${MAX_COMBINED_CONCURRENT_POSITIONS:-2}"
SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND="${SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND:-1}"
VOLATILITY_MAX_FILLED_PER_SIDE_PER_ROUND="${VOLATILITY_MAX_FILLED_PER_SIDE_PER_ROUND:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
DAILY_LOSS_LIMIT_USDC="${DAILY_LOSS_LIMIT_USDC:-3.0}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-120}"
MIN_ENTRY_PRICE="${MIN_ENTRY_PRICE:-0.35}"
BUY_SLIPPAGE="${BUY_SLIPPAGE:-0.02}"
SELL_SLIPPAGE="${SELL_SLIPPAGE:-0.02}"
POLL_SECONDS="${POLL_SECONDS:-10}"
MIN_SECONDS_TO_EXPIRY="${MIN_SECONDS_TO_EXPIRY:-180}"
MAX_SECONDS_TO_EXPIRY="${MAX_SECONDS_TO_EXPIRY:-900}"
SIGNAL_JSONL_PATH="${SIGNAL_JSONL_PATH:-}"
SIGNAL_JSONL_START="${SIGNAL_JSONL_START:-tail}"
LOG_DIR="${LOG_DIR:-logs/xgboost-v6-paper-shadow}"
CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME="${CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME:-false}"

if [[ "${CONFIRM:-}" == "yes" ]]; then
  echo "[v6-paper-shadow] refusing live settlement: v6 promotion path is paper-only." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[v6-paper-shadow] missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${POLYMARKET_PRIVATE_KEY:-}" ]]; then
  echo "[v6-paper-shadow] POLYMARKET_PRIVATE_KEY is required to read orderbooks" >&2
  exit 1
fi
if [[ ! -f "${MODEL_JSON_PATH}" ]]; then
  echo "[v6-paper-shadow] model artifact not found: ${MODEL_JSON_PATH}" >&2
  exit 1
fi
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  if [[ ! -f "${SIGNAL_JSONL_PATH}" ]]; then
    echo "[v6-paper-shadow] signal jsonl queue not found: ${SIGNAL_JSONL_PATH}" >&2
    exit 1
  fi
elif [[ ! -f "${MONITORING_DB_PATH}" ]]; then
  echo "[v6-paper-shadow] monitoring db not found: ${MONITORING_DB_PATH}" >&2
  echo "[v6-paper-shadow] start the v6 scorer first, or set SIGNAL_JSONL_PATH." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/phase4-${SESSION_ID}.jsonl"
SUMMARY_PATH="${LOG_DIR}/phase4-${SESSION_ID}-summary.json"

export PYTHONPATH="${PYTHONPATH:-src}"

echo "[v6-paper-shadow] repo=${REPO_ROOT}"
echo "[v6-paper-shadow] model_version=${MODEL_VERSION}"
echo "[v6-paper-shadow] model_json=${MODEL_JSON_PATH}"
echo "[v6-paper-shadow] market_families=${MARKET_FAMILIES}"
echo "[v6-paper-shadow] v6_settlement_gate=${V6_SETTLEMENT_THRESHOLD}"
echo "[v6-paper-shadow] volatility_reference=${V6_VOLATILITY_THRESHOLD}/${V6_ROUND_TRIP_COST}/${V6_EV_MARGIN}"
echo "[v6-paper-shadow] paper=true volatility_sleeve=${ENABLE_VOLATILITY_SLEEVE}"
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  echo "[v6-paper-shadow] signal_source=jsonl path=${SIGNAL_JSONL_PATH}"
else
  echo "[v6-paper-shadow] signal_source=duckdb db=${MONITORING_DB_PATH}"
fi
echo "[v6-paper-shadow] log=${LOG_PATH}"

EXEC_ARGS=(
  --model-version "${MODEL_VERSION}"
  --market-families "${MARKET_FAMILIES}"
  --entry-gate-mode v6-joint
  --v6-model-json-path "${MODEL_JSON_PATH}"
  --v6-settlement-threshold "${V6_SETTLEMENT_THRESHOLD}"
  --v6-neutral-cap "${V6_NEUTRAL_CAP}"
  --v6-volatility-threshold "${V6_VOLATILITY_THRESHOLD}"
  --v6-round-trip-cost "${V6_ROUND_TRIP_COST}"
  --v6-ev-margin "${V6_EV_MARGIN}"
  --monitoring-db-path "${MONITORING_DB_PATH}"
  --max-position-size-usdc "${MAX_POSITION_SIZE_USDC}"
  --max-concurrent-positions "${MAX_CONCURRENT_POSITIONS}"
  --max-combined-concurrent-positions "${MAX_COMBINED_CONCURRENT_POSITIONS}"
  --settlement-max-filled-per-side-per-round "${SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND}"
  --volatility-max-filled-per-side-per-round "${VOLATILITY_MAX_FILLED_PER_SIDE_PER_ROUND}"
  --max-rounds "${MAX_ROUNDS}"
  --daily-loss-limit-usdc "${DAILY_LOSS_LIMIT_USDC}"
  --max-runtime-minutes "${MAX_RUNTIME_MINUTES}"
  --min-entry-price "${MIN_ENTRY_PRICE}"
  --buy-slippage "${BUY_SLIPPAGE}"
  --sell-slippage "${SELL_SLIPPAGE}"
  --poll-seconds "${POLL_SECONDS}"
  --min-seconds-to-expiry "${MIN_SECONDS_TO_EXPIRY}"
  --max-seconds-to-expiry "${MAX_SECONDS_TO_EXPIRY}"
  --log-path "${LOG_PATH}"
  --summary-path "${SUMMARY_PATH}"
  --paper
)
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  EXEC_ARGS+=(--signal-jsonl-path "${SIGNAL_JSONL_PATH}" --signal-jsonl-start "${SIGNAL_JSONL_START}")
fi
if [[ "${CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME}" == "true" ]]; then
  EXEC_ARGS+=(--continue-after-max-rounds-until-runtime)
fi
if [[ "${ENABLE_VOLATILITY_SLEEVE}" == "true" ]]; then
  EXEC_ARGS+=(--enable-volatility-sleeve)
fi

exec "${PYTHON_BIN}" scripts/polymarket_phase4_live_champion_executor.py "${EXEC_ARGS[@]}"
