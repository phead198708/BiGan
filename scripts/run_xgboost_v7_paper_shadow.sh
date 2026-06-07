#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run xgboost-v7 Phase 4 paper/orderbook-only shadow.

This path is evidence-only:
  - orderbook-only execution (PAPER=true, no CLOB orders)
  - v7 PnL/EV settlement gate
  - settlement-only by default; volatility sleeve is disabled
  - queue-first by default to avoid stale DuckDB signal scans

Prerequisites (local scorer in another terminal):
  MODEL_VERSION=xgboost-v7 \
  MODEL_PATH=data/model-runs/xgboost-v7/20260606T132859Z-stable-gate/model.json \
  CALIBRATION_PATH= \
  SCORING_CANONICAL_SYMBOL_LIKE='BTC-15M:%' \
  MARKET_SPECS_JSON='[{"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15}]' \
  SIGNAL_JSONL_OUTPUT_PATH=data/live/xgboost-v7-paper/signals.jsonl \
  SIGNAL_JSONL_MARKET_FAMILIES=BTC-15M \
  SIGNAL_JSONL_OUTCOME_SIDE=ANY \
  SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS=30 \
  CYCLE_SLEEP_SECONDS=5 \
  ./scripts/run_champion_live.sh

Environment overrides:
  MODEL_VERSION                         Default: xgboost-v7
  MODEL_JSON_PATH                       Default: stable-gate v7 artifact
  MARKET_FAMILIES                       Default: BTC-15M
  V7_SETTLEMENT_MIN_CONFIDENCE          Default: 0.75
  V7_SETTLEMENT_MIN_EDGE_AFTER_COST     Default: 0.04
  MAX_SIGNAL_AGE_SECONDS                Default: 60. The scorer queue can stay
                                        stricter; executor read latency can
                                        otherwise reject minute-aligned signals
                                        just over 30s old.
  PAPER_SETTLEMENT_MAX_WAIT_AFTER_EXPIRY_SECONDS
                                        Default: 180
  ENABLE_VOLATILITY_SLEEVE              Default: false
  DISABLE_HEARTBEAT                     Default: true for paper shadow
  SIGNAL_JSONL_PATH                     Executor-ready signal JSONL queue.
                                        Required by default.
  LOW_LATENCY_OVERLAY_ENABLED           Default: false. When true, executor
                                        consumes raw top-of-book queue as a
                                        veto-only 5s/10s overlay.
  LOW_LATENCY_OVERLAY_RAW_JSONL_PATH    Raw queue path from run_champion_live.sh.
  LOW_LATENCY_OVERLAY_START             tail|beginning. Default: beginning.
  LOW_LATENCY_OVERLAY_MAX_QUOTE_AGE_SECONDS
                                        Default: 10
  LOW_LATENCY_OVERLAY_WINDOW_SECONDS    Default: 10
  LOW_LATENCY_OVERLAY_MAX_SPREAD        Default: 0.05
  LOW_LATENCY_OVERLAY_ADVERSE_VELOCITY_THRESHOLD
                                        Default: 0.04
  LOW_LATENCY_OVERLAY_MAX_PRICE_DRIFT_FROM_SIGNAL
                                        Default: 0.08
  LOW_LATENCY_OVERLAY_MISSING_QUOTE_ACTION
                                        pass|skip. Default: pass
  LOW_LATENCY_OVERLAY_MAX_RECORDS_PER_REFRESH
                                        Default: 20000
  REQUIRE_SIGNAL_JSONL                  Default: true
  LOG_DIR                               Default: logs/xgboost-v7-paper-shadow

Example:
  SIGNAL_JSONL_PATH=data/live/xgboost-v7-paper/signals.jsonl \
  bash scripts/run_xgboost_v7_paper_shadow.sh
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
MODEL_VERSION="${MODEL_VERSION:-xgboost-v7}"
MODEL_JSON_PATH="${MODEL_JSON_PATH:-data/model-runs/xgboost-v7/20260606T132859Z-stable-gate/model.json}"
MARKET_FAMILIES="${MARKET_FAMILIES:-BTC-15M}"
V7_SETTLEMENT_MIN_CONFIDENCE="${V7_SETTLEMENT_MIN_CONFIDENCE:-0.75}"
V7_SETTLEMENT_MIN_EDGE_AFTER_COST="${V7_SETTLEMENT_MIN_EDGE_AFTER_COST:-0.04}"
MAX_SIGNAL_AGE_SECONDS="${MAX_SIGNAL_AGE_SECONDS:-60}"
PAPER_SETTLEMENT_MAX_WAIT_AFTER_EXPIRY_SECONDS="${PAPER_SETTLEMENT_MAX_WAIT_AFTER_EXPIRY_SECONDS:-180}"
ENABLE_VOLATILITY_SLEEVE="${ENABLE_VOLATILITY_SLEEVE:-false}"
DISABLE_HEARTBEAT="${DISABLE_HEARTBEAT:-true}"
PAPER="true"
MONITORING_DB_PATH="${MONITORING_DB_PATH:-data/mlops/champion_catalog.duckdb}"
MAX_POSITION_SIZE_USDC="${MAX_POSITION_SIZE_USDC:-1.0}"
V7_SETTLEMENT_POSITION_MANAGEMENT_ENABLED="${V7_SETTLEMENT_POSITION_MANAGEMENT_ENABLED:-false}"
V7_SETTLEMENT_POSITION_PAPER_EXECUTE="${V7_SETTLEMENT_POSITION_PAPER_EXECUTE:-false}"
V7_SETTLEMENT_POSITION_ROUND_CAP_USDC="${V7_SETTLEMENT_POSITION_ROUND_CAP_USDC:-${MAX_POSITION_SIZE_USDC}}"
V7_SETTLEMENT_POSITION_ADD_EDGE_MIN="${V7_SETTLEMENT_POSITION_ADD_EDGE_MIN:-0.08}"
V7_SETTLEMENT_POSITION_FULL_ADD_EDGE="${V7_SETTLEMENT_POSITION_FULL_ADD_EDGE:-0.20}"
V7_SETTLEMENT_POSITION_WEAK_HOLD_EDGE="${V7_SETTLEMENT_POSITION_WEAK_HOLD_EDGE:-0.02}"
V7_SETTLEMENT_POSITION_REDUCE_FRACTION="${V7_SETTLEMENT_POSITION_REDUCE_FRACTION:-0.50}"
V7_SETTLEMENT_POSITION_EXIT_HOLD_EDGE="${V7_SETTLEMENT_POSITION_EXIT_HOLD_EDGE:--0.02}"
V7_SETTLEMENT_POSITION_EXIT_HYSTERESIS_BARS="${V7_SETTLEMENT_POSITION_EXIT_HYSTERESIS_BARS:-2}"
V7_SETTLEMENT_POSITION_REVERSAL_MIN_CONFIDENCE="${V7_SETTLEMENT_POSITION_REVERSAL_MIN_CONFIDENCE:-0.75}"
V7_SETTLEMENT_POSITION_REVERSAL_MIN_EDGE="${V7_SETTLEMENT_POSITION_REVERSAL_MIN_EDGE:-0.04}"
V7_SETTLEMENT_POSITION_REVERSAL_HYSTERESIS_BARS="${V7_SETTLEMENT_POSITION_REVERSAL_HYSTERESIS_BARS:-2}"
V7_SETTLEMENT_POSITION_MIN_REBALANCE_USDC="${V7_SETTLEMENT_POSITION_MIN_REBALANCE_USDC:-0.05}"
MAX_CONCURRENT_POSITIONS="${MAX_CONCURRENT_POSITIONS:-1}"
MAX_COMBINED_CONCURRENT_POSITIONS="${MAX_COMBINED_CONCURRENT_POSITIONS:-1}"
SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND="${SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
DAILY_LOSS_LIMIT_USDC="${DAILY_LOSS_LIMIT_USDC:-3.0}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-120}"
MIN_ENTRY_PRICE="${MIN_ENTRY_PRICE:-0.01}"
BUY_SLIPPAGE="${BUY_SLIPPAGE:-0.02}"
SELL_SLIPPAGE="${SELL_SLIPPAGE:-0.02}"
POLL_SECONDS="${POLL_SECONDS:-5}"
MIN_SECONDS_TO_EXPIRY="${MIN_SECONDS_TO_EXPIRY:-180}"
MAX_SECONDS_TO_EXPIRY="${MAX_SECONDS_TO_EXPIRY:-900}"
SIGNAL_JSONL_PATH="${SIGNAL_JSONL_PATH:-}"
SIGNAL_JSONL_START="${SIGNAL_JSONL_START:-tail}"
LOW_LATENCY_OVERLAY_ENABLED="${LOW_LATENCY_OVERLAY_ENABLED:-false}"
LOW_LATENCY_OVERLAY_RAW_JSONL_PATH="${LOW_LATENCY_OVERLAY_RAW_JSONL_PATH:-}"
LOW_LATENCY_OVERLAY_START="${LOW_LATENCY_OVERLAY_START:-beginning}"
LOW_LATENCY_OVERLAY_MAX_QUOTE_AGE_SECONDS="${LOW_LATENCY_OVERLAY_MAX_QUOTE_AGE_SECONDS:-10}"
LOW_LATENCY_OVERLAY_WINDOW_SECONDS="${LOW_LATENCY_OVERLAY_WINDOW_SECONDS:-10}"
LOW_LATENCY_OVERLAY_MAX_SPREAD="${LOW_LATENCY_OVERLAY_MAX_SPREAD:-0.05}"
LOW_LATENCY_OVERLAY_ADVERSE_VELOCITY_THRESHOLD="${LOW_LATENCY_OVERLAY_ADVERSE_VELOCITY_THRESHOLD:-0.04}"
LOW_LATENCY_OVERLAY_MAX_PRICE_DRIFT_FROM_SIGNAL="${LOW_LATENCY_OVERLAY_MAX_PRICE_DRIFT_FROM_SIGNAL:-0.08}"
LOW_LATENCY_OVERLAY_MISSING_QUOTE_ACTION="${LOW_LATENCY_OVERLAY_MISSING_QUOTE_ACTION:-pass}"
LOW_LATENCY_OVERLAY_MAX_RECORDS_PER_REFRESH="${LOW_LATENCY_OVERLAY_MAX_RECORDS_PER_REFRESH:-20000}"
REQUIRE_SIGNAL_JSONL="${REQUIRE_SIGNAL_JSONL:-true}"
LOG_DIR="${LOG_DIR:-logs/xgboost-v7-paper-shadow}"
CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME="${CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME:-false}"
POLYMARKET_ORDERBOOK_REST_FALLBACK="${POLYMARKET_ORDERBOOK_REST_FALLBACK:-true}"

if [[ "${CONFIRM:-}" == "yes" ]]; then
  echo "[v7-paper-shadow] refusing live settlement: this wrapper is paper-only." >&2
  exit 1
fi
if [[ "${MODEL_VERSION}" != "xgboost-v7" && "${MODEL_VERSION}" != xgboost-v7:* ]]; then
  echo "[v7-paper-shadow] MODEL_VERSION must be xgboost-v7 or xgboost-v7:* for v7-pnl gate" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[v7-paper-shadow] missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${POLYMARKET_PRIVATE_KEY:-}" ]]; then
  echo "[v7-paper-shadow] POLYMARKET_PRIVATE_KEY is required to read orderbooks" >&2
  exit 1
fi
if [[ ! -f "${MODEL_JSON_PATH}" ]]; then
  echo "[v7-paper-shadow] model artifact not found: ${MODEL_JSON_PATH}" >&2
  exit 1
fi
if [[ "${SIGNAL_JSONL_START}" != "tail" && "${SIGNAL_JSONL_START}" != "beginning" ]]; then
  echo "[v7-paper-shadow] SIGNAL_JSONL_START must be tail or beginning" >&2
  exit 1
fi
if [[ "${LOW_LATENCY_OVERLAY_ENABLED}" != "true" && "${LOW_LATENCY_OVERLAY_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] LOW_LATENCY_OVERLAY_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${V7_SETTLEMENT_POSITION_MANAGEMENT_ENABLED}" != "true" && "${V7_SETTLEMENT_POSITION_MANAGEMENT_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_SETTLEMENT_POSITION_MANAGEMENT_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${V7_SETTLEMENT_POSITION_PAPER_EXECUTE}" != "true" && "${V7_SETTLEMENT_POSITION_PAPER_EXECUTE}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_SETTLEMENT_POSITION_PAPER_EXECUTE must be true or false" >&2
  exit 1
fi
if [[ "${LOW_LATENCY_OVERLAY_START}" != "tail" && "${LOW_LATENCY_OVERLAY_START}" != "beginning" ]]; then
  echo "[v7-paper-shadow] LOW_LATENCY_OVERLAY_START must be tail or beginning" >&2
  exit 1
fi
if [[ "${LOW_LATENCY_OVERLAY_MISSING_QUOTE_ACTION}" != "pass" && "${LOW_LATENCY_OVERLAY_MISSING_QUOTE_ACTION}" != "skip" ]]; then
  echo "[v7-paper-shadow] LOW_LATENCY_OVERLAY_MISSING_QUOTE_ACTION must be pass or skip" >&2
  exit 1
fi
if [[ "${LOW_LATENCY_OVERLAY_ENABLED}" == "true" && -z "${LOW_LATENCY_OVERLAY_RAW_JSONL_PATH}" ]]; then
  echo "[v7-paper-shadow] LOW_LATENCY_OVERLAY_RAW_JSONL_PATH is required when overlay is enabled." >&2
  exit 1
fi
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  if [[ ! -f "${SIGNAL_JSONL_PATH}" ]]; then
    echo "[v7-paper-shadow] signal jsonl queue not found: ${SIGNAL_JSONL_PATH}" >&2
    exit 1
  fi
elif [[ "${REQUIRE_SIGNAL_JSONL}" == "true" ]]; then
  echo "[v7-paper-shadow] SIGNAL_JSONL_PATH is required for low-latency v7 paper shadow." >&2
  echo "[v7-paper-shadow] start the v7 scorer with SIGNAL_JSONL_OUTPUT_PATH, or set REQUIRE_SIGNAL_JSONL=false for diagnostic DuckDB scans." >&2
  exit 1
elif [[ ! -f "${MONITORING_DB_PATH}" ]]; then
  echo "[v7-paper-shadow] monitoring db not found: ${MONITORING_DB_PATH}" >&2
  echo "[v7-paper-shadow] start the v7 scorer first, or set SIGNAL_JSONL_PATH." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/phase4-${SESSION_ID}.jsonl"
SUMMARY_PATH="${LOG_DIR}/phase4-${SESSION_ID}-summary.json"

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
export POLYMARKET_ORDERBOOK_REST_FALLBACK

echo "[v7-paper-shadow] repo=${REPO_ROOT}"
echo "[v7-paper-shadow] model_version=${MODEL_VERSION}"
echo "[v7-paper-shadow] model_json=${MODEL_JSON_PATH}"
echo "[v7-paper-shadow] market_families=${MARKET_FAMILIES}"
echo "[v7-paper-shadow] v7_settlement_min_confidence=${V7_SETTLEMENT_MIN_CONFIDENCE}"
echo "[v7-paper-shadow] v7_settlement_min_edge_after_cost=${V7_SETTLEMENT_MIN_EDGE_AFTER_COST}"
echo "[v7-paper-shadow] max_signal_age_seconds=${MAX_SIGNAL_AGE_SECONDS}"
echo "[v7-paper-shadow] paper=true volatility_sleeve=${ENABLE_VOLATILITY_SLEEVE}"
echo "[v7-paper-shadow] v7_position_management=${V7_SETTLEMENT_POSITION_MANAGEMENT_ENABLED} paper_execute=${V7_SETTLEMENT_POSITION_PAPER_EXECUTE}"
if [[ "${V7_SETTLEMENT_POSITION_MANAGEMENT_ENABLED}" == "true" ]]; then
  echo "[v7-paper-shadow] v7_position_round_cap_usdc=${V7_SETTLEMENT_POSITION_ROUND_CAP_USDC}"
  echo "[v7-paper-shadow] v7_position_add_edge_min=${V7_SETTLEMENT_POSITION_ADD_EDGE_MIN}"
  echo "[v7-paper-shadow] v7_position_full_add_edge=${V7_SETTLEMENT_POSITION_FULL_ADD_EDGE}"
  echo "[v7-paper-shadow] v7_position_weak_hold_edge=${V7_SETTLEMENT_POSITION_WEAK_HOLD_EDGE}"
  echo "[v7-paper-shadow] v7_position_reduce_fraction=${V7_SETTLEMENT_POSITION_REDUCE_FRACTION}"
  echo "[v7-paper-shadow] v7_position_exit_hold_edge=${V7_SETTLEMENT_POSITION_EXIT_HOLD_EDGE}"
  echo "[v7-paper-shadow] v7_position_reversal_min_confidence=${V7_SETTLEMENT_POSITION_REVERSAL_MIN_CONFIDENCE}"
  echo "[v7-paper-shadow] v7_position_reversal_min_edge=${V7_SETTLEMENT_POSITION_REVERSAL_MIN_EDGE}"
fi
echo "[v7-paper-shadow] disable_heartbeat=${DISABLE_HEARTBEAT}"
echo "[v7-paper-shadow] orderbook_rest_fallback=${POLYMARKET_ORDERBOOK_REST_FALLBACK}"
echo "[v7-paper-shadow] low_latency_overlay_enabled=${LOW_LATENCY_OVERLAY_ENABLED}"
if [[ "${LOW_LATENCY_OVERLAY_ENABLED}" == "true" ]]; then
  echo "[v7-paper-shadow] low_latency_overlay_raw_jsonl=${LOW_LATENCY_OVERLAY_RAW_JSONL_PATH}"
  echo "[v7-paper-shadow] low_latency_overlay_start=${LOW_LATENCY_OVERLAY_START}"
  echo "[v7-paper-shadow] low_latency_overlay_max_quote_age_seconds=${LOW_LATENCY_OVERLAY_MAX_QUOTE_AGE_SECONDS}"
  echo "[v7-paper-shadow] low_latency_overlay_window_seconds=${LOW_LATENCY_OVERLAY_WINDOW_SECONDS}"
  echo "[v7-paper-shadow] low_latency_overlay_max_spread=${LOW_LATENCY_OVERLAY_MAX_SPREAD}"
  echo "[v7-paper-shadow] low_latency_overlay_adverse_velocity_threshold=${LOW_LATENCY_OVERLAY_ADVERSE_VELOCITY_THRESHOLD}"
  echo "[v7-paper-shadow] low_latency_overlay_max_price_drift_from_signal=${LOW_LATENCY_OVERLAY_MAX_PRICE_DRIFT_FROM_SIGNAL}"
  echo "[v7-paper-shadow] low_latency_overlay_missing_quote_action=${LOW_LATENCY_OVERLAY_MISSING_QUOTE_ACTION}"
  echo "[v7-paper-shadow] low_latency_overlay_max_records_per_refresh=${LOW_LATENCY_OVERLAY_MAX_RECORDS_PER_REFRESH}"
fi
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  echo "[v7-paper-shadow] signal_source=jsonl path=${SIGNAL_JSONL_PATH} start=${SIGNAL_JSONL_START}"
else
  echo "[v7-paper-shadow] signal_source=duckdb db=${MONITORING_DB_PATH}"
fi
echo "[v7-paper-shadow] log=${LOG_PATH}"
echo "[v7-paper-shadow] summary=${SUMMARY_PATH}"

EXEC_ARGS=(
  --model-version "${MODEL_VERSION}"
  --market-families "${MARKET_FAMILIES}"
  --entry-gate-mode v7-pnl
  --settlement-min-confidence "${V7_SETTLEMENT_MIN_CONFIDENCE}"
  --v7-settlement-min-edge-after-cost "${V7_SETTLEMENT_MIN_EDGE_AFTER_COST}"
  --max-signal-age-seconds "${MAX_SIGNAL_AGE_SECONDS}"
  --monitoring-db-path "${MONITORING_DB_PATH}"
  --max-position-size-usdc "${MAX_POSITION_SIZE_USDC}"
  --max-concurrent-positions "${MAX_CONCURRENT_POSITIONS}"
  --max-combined-concurrent-positions "${MAX_COMBINED_CONCURRENT_POSITIONS}"
  --settlement-max-filled-per-side-per-round "${SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND}"
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
  --paper-settlement-max-wait-after-expiry-seconds "${PAPER_SETTLEMENT_MAX_WAIT_AFTER_EXPIRY_SECONDS}"
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
if [[ "${DISABLE_HEARTBEAT}" == "true" ]]; then
  EXEC_ARGS+=(--disable-heartbeat)
fi
if [[ "${LOW_LATENCY_OVERLAY_ENABLED}" == "true" ]]; then
  EXEC_ARGS+=(
    --low-latency-overlay-enabled
    --low-latency-overlay-raw-jsonl-path "${LOW_LATENCY_OVERLAY_RAW_JSONL_PATH}"
    --low-latency-overlay-start "${LOW_LATENCY_OVERLAY_START}"
    --low-latency-overlay-max-quote-age-seconds "${LOW_LATENCY_OVERLAY_MAX_QUOTE_AGE_SECONDS}"
    --low-latency-overlay-window-seconds "${LOW_LATENCY_OVERLAY_WINDOW_SECONDS}"
    --low-latency-overlay-max-spread "${LOW_LATENCY_OVERLAY_MAX_SPREAD}"
    --low-latency-overlay-adverse-velocity-threshold "${LOW_LATENCY_OVERLAY_ADVERSE_VELOCITY_THRESHOLD}"
    --low-latency-overlay-max-price-drift-from-signal "${LOW_LATENCY_OVERLAY_MAX_PRICE_DRIFT_FROM_SIGNAL}"
    --low-latency-overlay-missing-quote-action "${LOW_LATENCY_OVERLAY_MISSING_QUOTE_ACTION}"
    --low-latency-overlay-max-records-per-refresh "${LOW_LATENCY_OVERLAY_MAX_RECORDS_PER_REFRESH}"
  )
fi
if [[ "${V7_SETTLEMENT_POSITION_MANAGEMENT_ENABLED}" == "true" ]]; then
  EXEC_ARGS+=(
    --v7-settlement-position-management-enabled
    --v7-settlement-position-round-cap-usdc "${V7_SETTLEMENT_POSITION_ROUND_CAP_USDC}"
    --v7-settlement-position-add-edge-min "${V7_SETTLEMENT_POSITION_ADD_EDGE_MIN}"
    --v7-settlement-position-full-add-edge "${V7_SETTLEMENT_POSITION_FULL_ADD_EDGE}"
    --v7-settlement-position-weak-hold-edge "${V7_SETTLEMENT_POSITION_WEAK_HOLD_EDGE}"
    --v7-settlement-position-reduce-fraction "${V7_SETTLEMENT_POSITION_REDUCE_FRACTION}"
    --v7-settlement-position-exit-hold-edge "${V7_SETTLEMENT_POSITION_EXIT_HOLD_EDGE}"
    --v7-settlement-position-exit-hysteresis-bars "${V7_SETTLEMENT_POSITION_EXIT_HYSTERESIS_BARS}"
    --v7-settlement-position-reversal-min-confidence "${V7_SETTLEMENT_POSITION_REVERSAL_MIN_CONFIDENCE}"
    --v7-settlement-position-reversal-min-edge "${V7_SETTLEMENT_POSITION_REVERSAL_MIN_EDGE}"
    --v7-settlement-position-reversal-hysteresis-bars "${V7_SETTLEMENT_POSITION_REVERSAL_HYSTERESIS_BARS}"
    --v7-settlement-position-min-rebalance-usdc "${V7_SETTLEMENT_POSITION_MIN_REBALANCE_USDC}"
  )
  if [[ "${V7_SETTLEMENT_POSITION_PAPER_EXECUTE}" == "true" ]]; then
    EXEC_ARGS+=(--v7-settlement-position-paper-execute)
  fi
fi

exec "${PYTHON_BIN}" scripts/polymarket_phase4_live_champion_executor.py "${EXEC_ARGS[@]}"
