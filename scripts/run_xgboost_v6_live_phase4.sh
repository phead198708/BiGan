#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run a tightly capped xgboost-v6 Phase 4 live settlement executor.

This is the explicit v6 live opt-in path. It places REAL settlement-sleeve FOK
orders only when both confirmation gates are set:

  CONFIRM=yes ALLOW_V6_LIVE_CONFIRM=yes bash scripts/run_xgboost_v6_live_phase4.sh

The wrapper intentionally keeps the blast radius small:
  - settlement sleeve only; volatility sleeve disabled by default
  - paper mode disabled; live settlement FOK orders are possible
  - BTC-15M only by default
  - 1 USDC max entry size, 1 concurrent settlement position
  - 10 observed rounds, 1.0 USDC daily realized-loss stop by default
  - cost-aware v6 settlement gate with confidence, signal-age, and exit guards

Prerequisite scorer:
  MODEL_VERSION=xgboost-v6 \
  MODEL_PATH=data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/model-single-grid/model.json \
  LOW_LATENCY_FEATURE_QUEUE_ENABLED=true \
  LOW_LATENCY_RAW_QUEUE_CANONICAL_SYMBOL_PREFIX='BTC-15M:' \
  SCORING_CANONICAL_SYMBOL_LIKE='BTC-15M:%' \
  SIGNAL_JSONL_OUTPUT_PATH=data/live/<run>/signals.jsonl \
  SIGNAL_JSONL_MARKET_FAMILIES=BTC-15M \
  SIGNAL_JSONL_OUTCOME_SIDE=ANY \
  SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS=180 \
  MARKET_SPECS_JSON='[{"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15}]' \
  ./scripts/run_champion_live.sh

Environment overrides:
  PYTHON_BIN                 Default: .venv/bin/python
  MODEL_VERSION              Default: xgboost-v6
  MODEL_JSON_PATH            Default: v6 model-single-grid/model.json
  MARKET_FAMILIES            Default: BTC-15M
  SIGNAL_JSONL_PATH          Required low-latency signal JSONL queue
  SIGNAL_JSONL_START         tail|beginning. Default: tail
  MONITORING_DB_PATH         Default: data/mlops/champion_catalog.duckdb
  LOG_DIR                    Default: logs/xgboost-v6-live-phase4
  MAX_ROUNDS                 Default: 10
  MAX_RUNTIME_MINUTES        Default: 180
  MAX_POSITION_SIZE_USDC     Default: 1.0
  MAX_CONCURRENT_POSITIONS   Default: 1
  DAILY_LOSS_LIMIT_USDC      Default: 1.0
  V6_SETTLEMENT_MIN_CONFIDENCE
                             Default: 0.80
  V6_SETTLEMENT_MIN_EDGE_AFTER_COST
                             Default: 0.082
  SETTLEMENT_ALLOW_MID_ROUND_EXIT
                             Default: true
  SETTLEMENT_CONFIDENCE_DECAY_ENABLED
                             Default: true
  SETTLEMENT_PRICE_STOP_ENABLED
                             Default: true
  SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_VETO_ENABLED
                             Default: true
  CONFIRM                    Must be yes
  ALLOW_V6_LIVE_CONFIRM      Must be yes

Do not use this wrapper for paper evidence. Use
scripts/run_xgboost_v6_paper_shadow.sh for paper/orderbook-only runs.
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
SIGNAL_JSONL_PATH="${SIGNAL_JSONL_PATH:-}"
SIGNAL_JSONL_START="${SIGNAL_JSONL_START:-tail}"
MONITORING_DB_PATH="${MONITORING_DB_PATH:-data/mlops/champion_catalog.duckdb}"
LOG_DIR="${LOG_DIR:-logs/xgboost-v6-live-phase4}"

V6_SETTLEMENT_THRESHOLD="${V6_SETTLEMENT_THRESHOLD:-0.50}"
V6_NEUTRAL_CAP="${V6_NEUTRAL_CAP:-0.25}"
V6_VOLATILITY_THRESHOLD="${V6_VOLATILITY_THRESHOLD:-0.60}"
V6_ROUND_TRIP_COST="${V6_ROUND_TRIP_COST:-0.072}"
V6_EV_MARGIN="${V6_EV_MARGIN:-0.01}"
V6_SETTLEMENT_MIN_EDGE_AFTER_COST="${V6_SETTLEMENT_MIN_EDGE_AFTER_COST:-0.082}"
V6_SETTLEMENT_MIN_CONFIDENCE="${V6_SETTLEMENT_MIN_CONFIDENCE:-0.80}"
SETTLEMENT_PEAK_CONFIDENCE_DROP_TOLERANCE="${SETTLEMENT_PEAK_CONFIDENCE_DROP_TOLERANCE:-0.05}"

MAX_ROUNDS="${MAX_ROUNDS:-10}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-180}"
MAX_POSITION_SIZE_USDC="${MAX_POSITION_SIZE_USDC:-1.0}"
MAX_CONCURRENT_POSITIONS="${MAX_CONCURRENT_POSITIONS:-1}"
MAX_COMBINED_CONCURRENT_POSITIONS="${MAX_COMBINED_CONCURRENT_POSITIONS:-1}"
SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND="${SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND:-1}"
DAILY_LOSS_LIMIT_USDC="${DAILY_LOSS_LIMIT_USDC:-1.0}"
BUY_SLIPPAGE="${BUY_SLIPPAGE:-0.02}"
SELL_SLIPPAGE="${SELL_SLIPPAGE:-0.02}"
POLL_SECONDS="${POLL_SECONDS:-10}"
MIN_SECONDS_TO_EXPIRY="${MIN_SECONDS_TO_EXPIRY:-180}"
MAX_SECONDS_TO_EXPIRY="${MAX_SECONDS_TO_EXPIRY:-900}"
NO_NEW_ENTRY_BEFORE_EXPIRY_SECONDS="${NO_NEW_ENTRY_BEFORE_EXPIRY_SECONDS:-300}"
MAX_SIGNAL_AGE_SECONDS="${MAX_SIGNAL_AGE_SECONDS:-180}"
CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME="${CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME:-false}"

SETTLEMENT_ALLOW_MID_ROUND_EXIT="${SETTLEMENT_ALLOW_MID_ROUND_EXIT:-true}"
SETTLEMENT_REVERSAL_MIN_CONFIDENCE="${SETTLEMENT_REVERSAL_MIN_CONFIDENCE:-${V6_SETTLEMENT_MIN_CONFIDENCE}}"
SETTLEMENT_REVERSAL_HYSTERESIS_BARS="${SETTLEMENT_REVERSAL_HYSTERESIS_BARS:-2}"
SETTLEMENT_CONFIDENCE_DECAY_ENABLED="${SETTLEMENT_CONFIDENCE_DECAY_ENABLED:-true}"
SETTLEMENT_DECAY_FLOOR="${SETTLEMENT_DECAY_FLOOR:-0.55}"
SETTLEMENT_DECAY_DELTA="${SETTLEMENT_DECAY_DELTA:-0.25}"
SETTLEMENT_DECAY_OPPOSITE_MIN_CONFIDENCE="${SETTLEMENT_DECAY_OPPOSITE_MIN_CONFIDENCE:-${V6_SETTLEMENT_MIN_CONFIDENCE}}"
SETTLEMENT_PRICE_STOP_ENABLED="${SETTLEMENT_PRICE_STOP_ENABLED:-true}"
SETTLEMENT_STOP_PRICE_DELTA="${SETTLEMENT_STOP_PRICE_DELTA:-0.15}"
SETTLEMENT_STOP_LOSS_USDC="${SETTLEMENT_STOP_LOSS_USDC:-0.50}"
SETTLEMENT_STOP_MIN_SECONDS_TO_EXPIRY="${SETTLEMENT_STOP_MIN_SECONDS_TO_EXPIRY:-120}"
SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_VETO_ENABLED="${SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_VETO_ENABLED:-true}"
SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_MIN_CONFIDENCE="${SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_MIN_CONFIDENCE:-${V6_SETTLEMENT_MIN_CONFIDENCE}}"
SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_MAX_AGE_SECONDS="${SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_MAX_AGE_SECONDS:-${MAX_SIGNAL_AGE_SECONDS}}"

ENABLE_VOLATILITY_SLEEVE="${ENABLE_VOLATILITY_SLEEVE:-false}"
ENABLE_VOLATILITY_LIVE_ENTRIES="${ENABLE_VOLATILITY_LIVE_ENTRIES:-false}"
CONFIRM="${CONFIRM:-}"
ALLOW_V6_LIVE_CONFIRM="${ALLOW_V6_LIVE_CONFIRM:-}"

if [[ "${CONFIRM}" != "yes" || "${ALLOW_V6_LIVE_CONFIRM}" != "yes" ]]; then
  echo "[v6-live-phase4] refusing to start: this places REAL capped settlement orders." >&2
  echo "[v6-live-phase4] require both CONFIRM=yes and ALLOW_V6_LIVE_CONFIRM=yes." >&2
  exit 1
fi
if [[ "${ENABLE_VOLATILITY_SLEEVE}" == "true" || "${ENABLE_VOLATILITY_LIVE_ENTRIES}" == "true" ]]; then
  echo "[v6-live-phase4] refusing to start: volatility must remain disabled for this live run." >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[v6-live-phase4] missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${MODEL_VERSION}" != "xgboost-v6" && "${MODEL_VERSION}" != xgboost-v6:* ]]; then
  echo "[v6-live-phase4] MODEL_VERSION must be xgboost-v6, got ${MODEL_VERSION}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_JSON_PATH}" ]]; then
  echo "[v6-live-phase4] model artifact not found: ${MODEL_JSON_PATH}" >&2
  exit 1
fi
if [[ -z "${POLYMARKET_PRIVATE_KEY:-}" ]]; then
  echo "[v6-live-phase4] POLYMARKET_PRIVATE_KEY is required for live execution" >&2
  exit 1
fi
if [[ -z "${SIGNAL_JSONL_PATH}" || ! -f "${SIGNAL_JSONL_PATH}" ]]; then
  echo "[v6-live-phase4] SIGNAL_JSONL_PATH must point at an existing low-latency queue" >&2
  exit 1
fi
if [[ "${SIGNAL_JSONL_START}" != "tail" && "${SIGNAL_JSONL_START}" != "beginning" ]]; then
  echo "[v6-live-phase4] SIGNAL_JSONL_START must be tail or beginning" >&2
  exit 1
fi
if [[ "${MAX_POSITION_SIZE_USDC}" != "1.0" && "${MAX_POSITION_SIZE_USDC}" != "1" ]]; then
  echo "[v6-live-phase4] refusing non-default MAX_POSITION_SIZE_USDC=${MAX_POSITION_SIZE_USDC}; use 1.0 for this first live run." >&2
  exit 1
fi
if [[ "${MAX_ROUNDS}" != "10" ]]; then
  echo "[v6-live-phase4] refusing non-default MAX_ROUNDS=${MAX_ROUNDS}; use 10 for this first live run." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/phase4-${SESSION_ID}.jsonl"
SUMMARY_PATH="${LOG_DIR}/phase4-${SESSION_ID}-summary.json"

export PYTHONPATH="${PYTHONPATH:-src}"

echo "[v6-live-phase4] repo=${REPO_ROOT}"
echo "[v6-live-phase4] model_version=${MODEL_VERSION}"
echo "[v6-live-phase4] model_json=${MODEL_JSON_PATH}"
echo "[v6-live-phase4] market_families=${MARKET_FAMILIES}"
echo "[v6-live-phase4] signal_source=jsonl path=${SIGNAL_JSONL_PATH} start=${SIGNAL_JSONL_START}"
echo "[v6-live-phase4] live_orders=true paper=false volatility_sleeve=false"
echo "[v6-live-phase4] caps: size=${MAX_POSITION_SIZE_USDC} rounds=${MAX_ROUNDS} concurrent=${MAX_CONCURRENT_POSITIONS} combined=${MAX_COMBINED_CONCURRENT_POSITIONS} daily_loss=${DAILY_LOSS_LIMIT_USDC}"
echo "[v6-live-phase4] settlement_gate: confidence=${V6_SETTLEMENT_MIN_CONFIDENCE} edge_after_cost=${V6_SETTLEMENT_MIN_EDGE_AFTER_COST} max_signal_age=${MAX_SIGNAL_AGE_SECONDS}"
echo "[v6-live-phase4] exits: mid_round=${SETTLEMENT_ALLOW_MID_ROUND_EXIT} decay=${SETTLEMENT_CONFIDENCE_DECAY_ENABLED} price_stop=${SETTLEMENT_PRICE_STOP_ENABLED} veto=${SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_VETO_ENABLED}"
echo "[v6-live-phase4] log=${LOG_PATH}"
echo "[v6-live-phase4] summary=${SUMMARY_PATH}"

EXEC_ARGS=(
  --model-version "${MODEL_VERSION}"
  --market-families "${MARKET_FAMILIES}"
  --signal-jsonl-path "${SIGNAL_JSONL_PATH}"
  --signal-jsonl-start "${SIGNAL_JSONL_START}"
  --entry-gate-mode v6-joint
  --v6-model-json-path "${MODEL_JSON_PATH}"
  --v6-settlement-threshold "${V6_SETTLEMENT_THRESHOLD}"
  --v6-neutral-cap "${V6_NEUTRAL_CAP}"
  --v6-volatility-threshold "${V6_VOLATILITY_THRESHOLD}"
  --v6-round-trip-cost "${V6_ROUND_TRIP_COST}"
  --v6-ev-margin "${V6_EV_MARGIN}"
  --v6-settlement-min-edge-after-cost "${V6_SETTLEMENT_MIN_EDGE_AFTER_COST}"
  --settlement-min-confidence "${V6_SETTLEMENT_MIN_CONFIDENCE}"
  --settlement-peak-confidence-drop-tolerance "${SETTLEMENT_PEAK_CONFIDENCE_DROP_TOLERANCE}"
  --settlement-reversal-min-confidence "${SETTLEMENT_REVERSAL_MIN_CONFIDENCE}"
  --settlement-reversal-hysteresis-bars "${SETTLEMENT_REVERSAL_HYSTERESIS_BARS}"
  --settlement-decay-floor "${SETTLEMENT_DECAY_FLOOR}"
  --settlement-decay-delta "${SETTLEMENT_DECAY_DELTA}"
  --settlement-decay-opposite-min-confidence "${SETTLEMENT_DECAY_OPPOSITE_MIN_CONFIDENCE}"
  --settlement-stop-price-delta "${SETTLEMENT_STOP_PRICE_DELTA}"
  --settlement-stop-loss-usdc "${SETTLEMENT_STOP_LOSS_USDC}"
  --settlement-stop-min-seconds-to-expiry "${SETTLEMENT_STOP_MIN_SECONDS_TO_EXPIRY}"
  --settlement-price-stop-same-side-confirmation-min-confidence "${SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_MIN_CONFIDENCE}"
  --settlement-price-stop-same-side-confirmation-max-age-seconds "${SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_MAX_AGE_SECONDS}"
  --max-signal-age-seconds "${MAX_SIGNAL_AGE_SECONDS}"
  --monitoring-db-path "${MONITORING_DB_PATH}"
  --max-position-size-usdc "${MAX_POSITION_SIZE_USDC}"
  --max-concurrent-positions "${MAX_CONCURRENT_POSITIONS}"
  --max-combined-concurrent-positions "${MAX_COMBINED_CONCURRENT_POSITIONS}"
  --settlement-max-filled-per-side-per-round "${SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND}"
  --max-rounds "${MAX_ROUNDS}"
  --daily-loss-limit-usdc "${DAILY_LOSS_LIMIT_USDC}"
  --max-runtime-minutes "${MAX_RUNTIME_MINUTES}"
  --buy-slippage "${BUY_SLIPPAGE}"
  --sell-slippage "${SELL_SLIPPAGE}"
  --poll-seconds "${POLL_SECONDS}"
  --min-seconds-to-expiry "${MIN_SECONDS_TO_EXPIRY}"
  --max-seconds-to-expiry "${MAX_SECONDS_TO_EXPIRY}"
  --no-new-entry-before-expiry-seconds "${NO_NEW_ENTRY_BEFORE_EXPIRY_SECONDS}"
  --log-path "${LOG_PATH}"
  --summary-path "${SUMMARY_PATH}"
)
if [[ "${CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME}" == "true" ]]; then
  EXEC_ARGS+=(--continue-after-max-rounds-until-runtime)
fi
if [[ "${SETTLEMENT_ALLOW_MID_ROUND_EXIT}" == "true" ]]; then
  EXEC_ARGS+=(--settlement-allow-mid-round-exit)
fi
if [[ "${SETTLEMENT_CONFIDENCE_DECAY_ENABLED}" == "true" ]]; then
  EXEC_ARGS+=(--settlement-confidence-decay-enabled)
fi
if [[ "${SETTLEMENT_PRICE_STOP_ENABLED}" == "true" ]]; then
  EXEC_ARGS+=(--settlement-price-stop-enabled)
fi
if [[ "${SETTLEMENT_PRICE_STOP_SAME_SIDE_CONFIRMATION_VETO_ENABLED}" == "true" ]]; then
  EXEC_ARGS+=(--settlement-price-stop-same-side-confirmation-veto-enabled)
fi

exec "${PYTHON_BIN}" scripts/polymarket_phase4_live_champion_executor.py "${EXEC_ARGS[@]}"
