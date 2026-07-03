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
  MODEL_PATH=data/model-runs/xgboost-v7-event-driven/20260608Tevent-5s-v1/model/model.json \
  CALIBRATION_PATH= \
  SCORING_CANONICAL_SYMBOL_LIKE='BTC-15M:%' \
  MARKET_SPECS_JSON='[{"slug_prefix":"btc-updown-15m-","underlying":"BTC","horizon_minutes":15}]' \
  SIGNAL_JSONL_OUTPUT_PATH=data/live/xgboost-v7-paper/signals.jsonl \
  SIGNAL_JSONL_MARKET_FAMILIES=BTC-15M \
  SIGNAL_JSONL_OUTCOME_SIDE=ANY \
  SIGNAL_JSONL_MAX_EVENT_AGE_SECONDS=30 \
  LOW_LATENCY_FEATURE_QUEUE_ENABLED=true \
  LOW_LATENCY_FEATURE_BUCKET_SECONDS=5 \
  EVENT_DRIVEN_V7_SIGNAL_QUEUE_ENABLED=true \
  EVENT_DRIVEN_V7_SIGNAL_JSONL_OUTPUT_PATH=data/live/xgboost-v7-paper/signals-event-driven.jsonl \
  EVENT_DRIVEN_V7_SIGNAL_BUCKET_SECONDS=5 \
  CYCLE_SLEEP_SECONDS=5 \
  ./scripts/run_champion_live.sh

Environment overrides:
  MODEL_VERSION                         Default: xgboost-v7
  MODEL_JSON_PATH                       Default: 5s event-driven v7 artifact
  MARKET_FAMILIES                       Default: BTC-15M
  V7_SETTLEMENT_MIN_CONFIDENCE          Default: 0.75
  V7_SETTLEMENT_MIN_EDGE_AFTER_COST     Default: 0.04
  V7_SETTLEMENT_ALLOW_REENTRY_AFTER_EXIT
                                        Default: true. Allows v7-pnl settlement
                                        re-entry in the same round after the
                                        current settlement position exits.
  V7_SETTLEMENT_POSITION_CONVERGENCE_PRICE_TOLERANCE
                                        Default: 0.02. Adverse held-token price
                                        move allowed before residual divergence.
  V7_SETTLEMENT_POSITION_CONVERGENCE_MODEL_DECAY_TOLERANCE
                                        Default: 0.10. Adverse model-probability
                                        decay allowed before residual divergence.
  V7_SETTLEMENT_POSITION_DIVERGENCE_HYSTERESIS_BARS
                                        Default: 2
  V7_SETTLEMENT_POSITION_CONVERGENCE_TAKE_PROFIT_ENABLED
                                        Default: false. Exit when convergence
                                        edge is captured instead of holding
                                        to settlement.
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_HOLD_EDGE
                                        Default: 0.03
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_RESIDUAL_RATIO
                                        Default: 0.40
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_MOVE
                                        Default: 0.10
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_HOLD_EDGE_RATIO
                                        Default: 0.50
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_FORCE_EXIT_SECONDS
                                        Default: 180
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_HYSTERESIS_BARS
                                        Default: 2
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_UP_HOLD_EDGE_TIGHTEN
                                        Default: 0.01
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_DELTA
                                        Default: 0.10. Profit-protect exit
                                        when current executable bid is this
                                        much above average entry price.
  V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_RETURN
                                        Default: 0.35. Profit-protect exit
                                        when current executable bid return
                                        reaches this ratio.
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DECAY_ENABLED
                                        Default: false. Require stronger p_side
                                        persistence when held-token bid moves
                                        adversely from avg entry price.
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DELTA_START
                                        Default: 0.10
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_BASE_ALLOWED_DECAY
                                        Default: 0.08
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DECAY_SLOPE
                                        Default: 0.30
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MIN_ALLOWED_DECAY
                                        Default: 0.015
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REQUIRED_PROBABILITY
                                        Default: 0.97
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_EXIT_PROBABILITY_BUFFER
                                        Default: 0.03
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MIN_MODEL_DECAY
                                        Default: 0.06
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MAX_HOLD_EDGE
                                        Default: 0.25. Set negative to allow
                                        full exit regardless of hold_edge.
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_REDUCE_MIN_MODEL_DECAY
                                        Default: 0.06. Minimum model confidence
                                        decay required before partial reduce.
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MAX_COST
                                        Default: 0.15. Exit tiny adverse tail
                                        positions instead of repeatedly reducing
                                        below min rebalance size.
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MIN_CANDIDATE_COUNT
                                        Default: 3
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_HYSTERESIS_BARS
                                        Default: 2
  V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REDUCES
                                        Default: 1. Set 0 for unlimited legacy
                                        adverse-confidence reductions.
  V7_SETTLEMENT_POSITION_BLOCK_ADD_AFTER_ADVERSE_CONFIDENCE_REDUCE
                                        Default: true
  MAX_SIGNAL_AGE_SECONDS                Default: 60. The scorer queue can stay
                                        stricter; executor read latency can
                                        otherwise reject minute-aligned signals
                                        just over 30s old.
  ENTRY_MAX_PRICE_DRIFT_FROM_SIGNAL     Optional v7-pnl entry gate. Skip when
                                        worst-case entry price exceeds signal
                                        polymarket price by more than this amount.
  V7_RAW_SIDE_AGREEMENT_ENABLED         Default: true. Require raw p_up/p_down
                                        to not contradict the selected v7-pnl
                                        side before entry.
  V7_RAW_SIDE_MIN_PROBABILITY           Default: 0.50
  V7_RAW_SIDE_MIN_MARGIN                Optional. Require raw p_side - p_opposite
                                        to be at least this value.
  V7_RAW_SIDE_MAX_OPPOSITE_LEAD         Default: 0.03
  V7_RAW_SIDE_PRICE_CONVICTION_ENABLED  Default: true. Tighten raw p_side
                                        threshold near center-priced entries.
  V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY
                                        Default: 0.57
  V7_ENTRY_CANDIDATE_BUFFER_ENABLED    Default: true. Wait briefly within a
                                        round and release the best v7 entry
                                        candidate by price band + confidence.
  V7_ENTRY_CANDIDATE_BUFFER_MAX_WAIT_SECONDS
                                        Default: 30
  V7_ENTRY_CANDIDATE_BUFFER_MIN_PRICE   Default: 0.40
  V7_ENTRY_CANDIDATE_BUFFER_MAX_PRICE   Default: 0.70
  V7_ENTRY_CANDIDATE_BUFFER_MIN_EDGE    Default: 0.04
  V7_ENTRY_CANDIDATE_BUFFER_MIN_SECONDS_TO_EXPIRY
                                        Default: 330
  V7_ENTRY_CANDIDATE_BUFFER_IMMEDIATE_CONFIDENCE_SCORE
                                        Optional. Release immediately when the
                                        best candidate reaches this confidence.
  V7_CONVERGENCE_CALIBRATION_PATH       Optional replay calibration artifact.
                                        When set, v7-pnl entry decisions also
                                        require historical convergence bucket
                                        quality to pass.
  V7_CONVERGENCE_CALIBRATION_MIN_HIT_5C_RATE
                                        Default: 0.0
  V7_CONVERGENCE_CALIBRATION_MIN_HIT_10C_RATE
                                        Default: 0.0
	  V7_CONVERGENCE_CALIBRATION_MAX_MODEL_OVER_ERROR_P80
	                                        Optional
	  V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_MEDIAN_EDGE
	                                        Optional. Require model_p adjusted by
	                                        bucket median value error to remain
	                                        above actual entry price by this edge.
	  V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_P80_EDGE
	                                        Optional. Require model_p adjusted by
	                                        bucket p80 overprediction error to
	                                        remain above actual entry price by
	                                        this edge.
	  V7_CONVERGENCE_CALIBRATION_MIN_BUCKET_SAMPLE_COUNT
	                                        Default: 20
  PAPER_SETTLEMENT_MAX_WAIT_AFTER_EXPIRY_SECONDS
                                        Default: 86400
  ENABLE_VOLATILITY_SLEEVE              Default: false
  DISABLE_HEARTBEAT                     Default: true for paper shadow
  SIGNAL_JSONL_PATH                     Executor-ready signal JSONL queue.
                                        Required by default. For the 5s
                                        low-latency feature scorer, point this
                                        to SIGNAL_JSONL_OUTPUT_PATH; it already
                                        carries 5s v7 convergence payloads.
                                        EVENT_DRIVEN_V7_SIGNAL_JSONL_OUTPUT_PATH
                                        is only for the legacy raw-quote reprice
                                        overlay path.
  SIGNAL_JSONL_STALE_WARN_SECONDS       Default: 900. Executor logs
                                        signal_jsonl_stale when the queue file
                                        mtime is older than this threshold.
  SIGNAL_KAFKA_BOOTSTRAP_SERVERS        Optional Kafka bootstrap servers for
                                        executor-ready signal consumption.
  SIGNAL_KAFKA_TOPIC                    Optional Kafka topic for executor-ready
                                        signal consumption. Set together with
                                        SIGNAL_KAFKA_BOOTSTRAP_SERVERS.
  SIGNAL_KAFKA_GROUP_ID                 Default: bigan-xgboost-v7-paper-shadow
  SIGNAL_KAFKA_START                    tail|beginning. Default: tail
  SIGNAL_KAFKA_POLL_TIMEOUT_SECONDS     Default: 0.25
  SIGNAL_KAFKA_MAX_RECORDS              Default: 500
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
  PLAN_ONLY                             Default: false. Print resolved executor
                                        command and exit before execution.
  POST_RUN_CLEANUP_ENABLED              Default: false. When true, run
                                        cleanup_paper_run_artifacts.py after
                                        executor exit.
  POST_RUN_CLEANUP_LIVE_ROOT            Live root to clean. Optional but
                                        recommended for full run cleanup.
  POST_RUN_CLEANUP_SCORER_LOG_DIR       Scorer log dir to clean. Optional.
  POST_RUN_CLEANUP_PROFILE              training|prediction-warehouse.
                                        Default: training.
  POST_RUN_CLEANUP_ALLOW_INCOMPLETE     Default: false.

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
MODEL_JSON_PATH="${MODEL_JSON_PATH:-data/model-runs/xgboost-v7-event-driven/20260608Tevent-5s-v1/model/model.json}"
MARKET_FAMILIES="${MARKET_FAMILIES:-BTC-15M}"
V7_SETTLEMENT_MIN_CONFIDENCE="${V7_SETTLEMENT_MIN_CONFIDENCE:-0.75}"
V7_SETTLEMENT_MIN_EDGE_AFTER_COST="${V7_SETTLEMENT_MIN_EDGE_AFTER_COST:-0.04}"
V7_SETTLEMENT_ALLOW_REENTRY_AFTER_EXIT="${V7_SETTLEMENT_ALLOW_REENTRY_AFTER_EXIT:-true}"
MAX_SIGNAL_AGE_SECONDS="${MAX_SIGNAL_AGE_SECONDS:-60}"
ENTRY_MAX_PRICE_DRIFT_FROM_SIGNAL="${ENTRY_MAX_PRICE_DRIFT_FROM_SIGNAL:-}"
V7_RAW_SIDE_AGREEMENT_ENABLED="${V7_RAW_SIDE_AGREEMENT_ENABLED:-true}"
V7_RAW_SIDE_MIN_PROBABILITY="${V7_RAW_SIDE_MIN_PROBABILITY:-0.50}"
V7_RAW_SIDE_MIN_MARGIN="${V7_RAW_SIDE_MIN_MARGIN:-}"
V7_RAW_SIDE_MAX_OPPOSITE_LEAD="${V7_RAW_SIDE_MAX_OPPOSITE_LEAD:-0.03}"
V7_RAW_SIDE_PRICE_CONVICTION_ENABLED="${V7_RAW_SIDE_PRICE_CONVICTION_ENABLED:-true}"
V7_RAW_SIDE_PRICE_CONVICTION_MIN_PRICE="${V7_RAW_SIDE_PRICE_CONVICTION_MIN_PRICE:-0.40}"
V7_RAW_SIDE_PRICE_CONVICTION_CENTER_PRICE="${V7_RAW_SIDE_PRICE_CONVICTION_CENTER_PRICE:-0.50}"
V7_RAW_SIDE_PRICE_CONVICTION_MAX_PRICE="${V7_RAW_SIDE_PRICE_CONVICTION_MAX_PRICE:-0.70}"
V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY="${V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY:-0.57}"
V7_ENTRY_CANDIDATE_BUFFER_ENABLED="${V7_ENTRY_CANDIDATE_BUFFER_ENABLED:-true}"
V7_ENTRY_CANDIDATE_BUFFER_MAX_WAIT_SECONDS="${V7_ENTRY_CANDIDATE_BUFFER_MAX_WAIT_SECONDS:-30}"
V7_ENTRY_CANDIDATE_BUFFER_MIN_PRICE="${V7_ENTRY_CANDIDATE_BUFFER_MIN_PRICE:-0.40}"
V7_ENTRY_CANDIDATE_BUFFER_MAX_PRICE="${V7_ENTRY_CANDIDATE_BUFFER_MAX_PRICE:-0.70}"
V7_ENTRY_CANDIDATE_BUFFER_MIN_EDGE="${V7_ENTRY_CANDIDATE_BUFFER_MIN_EDGE:-0.04}"
V7_ENTRY_CANDIDATE_BUFFER_MIN_SECONDS_TO_EXPIRY="${V7_ENTRY_CANDIDATE_BUFFER_MIN_SECONDS_TO_EXPIRY:-330}"
V7_ENTRY_CANDIDATE_BUFFER_IMMEDIATE_CONFIDENCE_SCORE="${V7_ENTRY_CANDIDATE_BUFFER_IMMEDIATE_CONFIDENCE_SCORE:-}"
V7_ENTRY_CANDIDATE_BUFFER_MAX_CANDIDATES_PER_ROUND="${V7_ENTRY_CANDIDATE_BUFFER_MAX_CANDIDATES_PER_ROUND:-64}"
V7_CONVERGENCE_CALIBRATION_PATH="${V7_CONVERGENCE_CALIBRATION_PATH:-}"
V7_CONVERGENCE_CALIBRATION_MIN_HIT_5C_RATE="${V7_CONVERGENCE_CALIBRATION_MIN_HIT_5C_RATE:-0.0}"
V7_CONVERGENCE_CALIBRATION_MIN_HIT_10C_RATE="${V7_CONVERGENCE_CALIBRATION_MIN_HIT_10C_RATE:-0.0}"
V7_CONVERGENCE_CALIBRATION_MAX_MODEL_OVER_ERROR_P80="${V7_CONVERGENCE_CALIBRATION_MAX_MODEL_OVER_ERROR_P80:-}"
V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_MEDIAN_EDGE="${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_MEDIAN_EDGE:-}"
V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_P80_EDGE="${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_P80_EDGE:-}"
V7_CONVERGENCE_CALIBRATION_MIN_BUCKET_SAMPLE_COUNT="${V7_CONVERGENCE_CALIBRATION_MIN_BUCKET_SAMPLE_COUNT:-20}"
PAPER_SETTLEMENT_MAX_WAIT_AFTER_EXPIRY_SECONDS="${PAPER_SETTLEMENT_MAX_WAIT_AFTER_EXPIRY_SECONDS:-86400}"
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
V7_SETTLEMENT_POSITION_DIVERGENCE_REDUCE_MAX_HOLD_EDGE="${V7_SETTLEMENT_POSITION_DIVERGENCE_REDUCE_MAX_HOLD_EDGE:-0.08}"
V7_SETTLEMENT_POSITION_EXIT_HOLD_EDGE="${V7_SETTLEMENT_POSITION_EXIT_HOLD_EDGE:--0.02}"
V7_SETTLEMENT_POSITION_EXIT_HYSTERESIS_BARS="${V7_SETTLEMENT_POSITION_EXIT_HYSTERESIS_BARS:-2}"
V7_SETTLEMENT_POSITION_REVERSAL_MIN_CONFIDENCE="${V7_SETTLEMENT_POSITION_REVERSAL_MIN_CONFIDENCE:-0.75}"
V7_SETTLEMENT_POSITION_REVERSAL_MIN_EDGE="${V7_SETTLEMENT_POSITION_REVERSAL_MIN_EDGE:-0.04}"
V7_SETTLEMENT_POSITION_REVERSAL_HYSTERESIS_BARS="${V7_SETTLEMENT_POSITION_REVERSAL_HYSTERESIS_BARS:-2}"
V7_SETTLEMENT_POSITION_MIN_REBALANCE_USDC="${V7_SETTLEMENT_POSITION_MIN_REBALANCE_USDC:-0.05}"
V7_SETTLEMENT_POSITION_CONVERGENCE_PRICE_TOLERANCE="${V7_SETTLEMENT_POSITION_CONVERGENCE_PRICE_TOLERANCE:-0.02}"
V7_SETTLEMENT_POSITION_CONVERGENCE_MODEL_DECAY_TOLERANCE="${V7_SETTLEMENT_POSITION_CONVERGENCE_MODEL_DECAY_TOLERANCE:-0.10}"
V7_SETTLEMENT_POSITION_DIVERGENCE_HYSTERESIS_BARS="${V7_SETTLEMENT_POSITION_DIVERGENCE_HYSTERESIS_BARS:-2}"
V7_SETTLEMENT_POSITION_ADD_COOLDOWN_AFTER_DIVERGENCE_REDUCE_SECONDS="${V7_SETTLEMENT_POSITION_ADD_COOLDOWN_AFTER_DIVERGENCE_REDUCE_SECONDS:-120}"
V7_SETTLEMENT_POSITION_CONVERGENCE_TAKE_PROFIT_ENABLED="${V7_SETTLEMENT_POSITION_CONVERGENCE_TAKE_PROFIT_ENABLED:-false}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_HOLD_EDGE="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_HOLD_EDGE:-0.03}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_RESIDUAL_RATIO="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_RESIDUAL_RATIO:-0.40}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_MOVE="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_MOVE:-0.10}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_HOLD_EDGE_RATIO="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_HOLD_EDGE_RATIO:-0.50}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_FORCE_EXIT_SECONDS="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_FORCE_EXIT_SECONDS:-180}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_HYSTERESIS_BARS="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_HYSTERESIS_BARS:-2}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_UP_HOLD_EDGE_TIGHTEN="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_UP_HOLD_EDGE_TIGHTEN:-0.01}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_DELTA="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_DELTA:-0.10}"
V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_RETURN="${V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_RETURN:-0.35}"
V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ENABLED="${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ENABLED:-false}"
V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_MAX_CONFIDENCE_SCORE="${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_MAX_CONFIDENCE_SCORE:-0.0}"
V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_MIN_PROFIT_DELTA="${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_MIN_PROFIT_DELTA:-0.05}"
V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_MIN_PROFIT_RETURN="${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_MIN_PROFIT_RETURN:-0.10}"
V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_HYSTERESIS_BARS="${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_HYSTERESIS_BARS:-1}"
V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ADVERSE_FULL_EXIT_ENABLED="${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ADVERSE_FULL_EXIT_ENABLED:-false}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DECAY_ENABLED="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DECAY_ENABLED:-false}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DELTA_START="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DELTA_START:-0.10}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_BASE_ALLOWED_DECAY="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_BASE_ALLOWED_DECAY:-0.08}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DECAY_SLOPE="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DECAY_SLOPE:-0.30}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MIN_ALLOWED_DECAY="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MIN_ALLOWED_DECAY:-0.015}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REQUIRED_PROBABILITY="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REQUIRED_PROBABILITY:-0.97}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_EXIT_PROBABILITY_BUFFER="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_EXIT_PROBABILITY_BUFFER:-0.03}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MIN_MODEL_DECAY="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MIN_MODEL_DECAY:-0.06}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MAX_HOLD_EDGE="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MAX_HOLD_EDGE:-0.25}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_REDUCE_MIN_MODEL_DECAY="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_REDUCE_MIN_MODEL_DECAY:-0.06}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MAX_COST="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MAX_COST:-0.15}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MIN_CANDIDATE_COUNT="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MIN_CANDIDATE_COUNT:-3}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_HYSTERESIS_BARS="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_HYSTERESIS_BARS:-2}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REDUCES="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REDUCES:-1}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_ENABLED="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_ENABLED:-true}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_BARS="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_BARS:-1}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_MIN_MODEL_DECAY="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_MIN_MODEL_DECAY:-0.06}"
V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_MAX_HOLD_EDGE="${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_MAX_HOLD_EDGE:--1}"
V7_SETTLEMENT_POSITION_BLOCK_ADD_AFTER_ADVERSE_CONFIDENCE_REDUCE="${V7_SETTLEMENT_POSITION_BLOCK_ADD_AFTER_ADVERSE_CONFIDENCE_REDUCE:-true}"
V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_QUALITY_ENABLED="${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_QUALITY_ENABLED:-true}"
V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_MODEL_PROBABILITY_IMPROVEMENT="${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_MODEL_PROBABILITY_IMPROVEMENT:-0.03}"
V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_RAW_PROBABILITY_IMPROVEMENT="${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_RAW_PROBABILITY_IMPROVEMENT:-0.02}"
V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_SECONDS_TO_EXPIRY="${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_SECONDS_TO_EXPIRY:-420}"
MAX_CONCURRENT_POSITIONS="${MAX_CONCURRENT_POSITIONS:-1}"
MAX_COMBINED_CONCURRENT_POSITIONS="${MAX_COMBINED_CONCURRENT_POSITIONS:-1}"
SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND="${SETTLEMENT_MAX_FILLED_PER_SIDE_PER_ROUND:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
DAILY_LOSS_LIMIT_USDC="${DAILY_LOSS_LIMIT_USDC:-3.0}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-120}"
MIN_ENTRY_PRICE="${MIN_ENTRY_PRICE:-0.30}"
BUY_SLIPPAGE="${BUY_SLIPPAGE:-0.02}"
SELL_SLIPPAGE="${SELL_SLIPPAGE:-0.02}"
POLL_SECONDS="${POLL_SECONDS:-5}"
MIN_SECONDS_TO_EXPIRY="${MIN_SECONDS_TO_EXPIRY:-180}"
MAX_SECONDS_TO_EXPIRY="${MAX_SECONDS_TO_EXPIRY:-900}"
SIGNAL_JSONL_PATH="${SIGNAL_JSONL_PATH:-}"
SIGNAL_JSONL_START="${SIGNAL_JSONL_START:-tail}"
SIGNAL_JSONL_STALE_WARN_SECONDS="${SIGNAL_JSONL_STALE_WARN_SECONDS:-900}"
NO_NEW_OBSERVED_ROUND_WARN_SECONDS="${NO_NEW_OBSERVED_ROUND_WARN_SECONDS:-1800}"
SIGNAL_KAFKA_BOOTSTRAP_SERVERS="${SIGNAL_KAFKA_BOOTSTRAP_SERVERS:-}"
SIGNAL_KAFKA_TOPIC="${SIGNAL_KAFKA_TOPIC:-}"
SIGNAL_KAFKA_GROUP_ID="${SIGNAL_KAFKA_GROUP_ID:-bigan-xgboost-v7-paper-shadow}"
SIGNAL_KAFKA_START="${SIGNAL_KAFKA_START:-tail}"
SIGNAL_KAFKA_POLL_TIMEOUT_SECONDS="${SIGNAL_KAFKA_POLL_TIMEOUT_SECONDS:-0.25}"
SIGNAL_KAFKA_MAX_RECORDS="${SIGNAL_KAFKA_MAX_RECORDS:-500}"
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
PLAN_ONLY="${PLAN_ONLY:-false}"
POST_RUN_CLEANUP_ENABLED="${POST_RUN_CLEANUP_ENABLED:-false}"
POST_RUN_CLEANUP_LIVE_ROOT="${POST_RUN_CLEANUP_LIVE_ROOT:-}"
POST_RUN_CLEANUP_SCORER_LOG_DIR="${POST_RUN_CLEANUP_SCORER_LOG_DIR:-}"
POST_RUN_CLEANUP_PROFILE="${POST_RUN_CLEANUP_PROFILE:-training}"
POST_RUN_CLEANUP_ALLOW_INCOMPLETE="${POST_RUN_CLEANUP_ALLOW_INCOMPLETE:-false}"

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
if [[ "${SIGNAL_KAFKA_START}" != "tail" && "${SIGNAL_KAFKA_START}" != "beginning" ]]; then
  echo "[v7-paper-shadow] SIGNAL_KAFKA_START must be tail or beginning" >&2
  exit 1
fi
if [[ -n "${SIGNAL_KAFKA_BOOTSTRAP_SERVERS}" && -z "${SIGNAL_KAFKA_TOPIC}" ]]; then
  echo "[v7-paper-shadow] SIGNAL_KAFKA_TOPIC is required when SIGNAL_KAFKA_BOOTSTRAP_SERVERS is set" >&2
  exit 1
fi
if [[ -z "${SIGNAL_KAFKA_BOOTSTRAP_SERVERS}" && -n "${SIGNAL_KAFKA_TOPIC}" ]]; then
  echo "[v7-paper-shadow] SIGNAL_KAFKA_BOOTSTRAP_SERVERS is required when SIGNAL_KAFKA_TOPIC is set" >&2
  exit 1
fi
if [[ "${LOW_LATENCY_OVERLAY_ENABLED}" != "true" && "${LOW_LATENCY_OVERLAY_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] LOW_LATENCY_OVERLAY_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${PLAN_ONLY}" != "true" && "${PLAN_ONLY}" != "false" ]]; then
  echo "[v7-paper-shadow] PLAN_ONLY must be true or false" >&2
  exit 1
fi
if [[ "${V7_SETTLEMENT_ALLOW_REENTRY_AFTER_EXIT}" != "true" && "${V7_SETTLEMENT_ALLOW_REENTRY_AFTER_EXIT}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_SETTLEMENT_ALLOW_REENTRY_AFTER_EXIT must be true or false" >&2
  exit 1
fi
if [[ "${V7_RAW_SIDE_AGREEMENT_ENABLED}" != "true" && "${V7_RAW_SIDE_AGREEMENT_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_RAW_SIDE_AGREEMENT_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${V7_RAW_SIDE_PRICE_CONVICTION_ENABLED}" != "true" && "${V7_RAW_SIDE_PRICE_CONVICTION_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_RAW_SIDE_PRICE_CONVICTION_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${V7_ENTRY_CANDIDATE_BUFFER_ENABLED}" != "true" && "${V7_ENTRY_CANDIDATE_BUFFER_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_ENTRY_CANDIDATE_BUFFER_ENABLED must be true or false" >&2
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
if [[ "${V7_SETTLEMENT_POSITION_BLOCK_ADD_AFTER_ADVERSE_CONFIDENCE_REDUCE}" != "true" && "${V7_SETTLEMENT_POSITION_BLOCK_ADD_AFTER_ADVERSE_CONFIDENCE_REDUCE}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_SETTLEMENT_POSITION_BLOCK_ADD_AFTER_ADVERSE_CONFIDENCE_REDUCE must be true or false" >&2
  exit 1
fi
if [[ "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_ENABLED}" != "true" && "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ENABLED}" != "true" && "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ADVERSE_FULL_EXIT_ENABLED}" != "true" && "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ADVERSE_FULL_EXIT_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ADVERSE_FULL_EXIT_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_QUALITY_ENABLED}" != "true" && "${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_QUALITY_ENABLED}" != "false" ]]; then
  echo "[v7-paper-shadow] V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_QUALITY_ENABLED must be true or false" >&2
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
SIGNAL_KAFKA_ENABLED="false"
if [[ -n "${SIGNAL_KAFKA_TOPIC}" ]]; then
  SIGNAL_KAFKA_ENABLED="true"
fi

if [[ -n "${SIGNAL_JSONL_PATH}" && "${SIGNAL_KAFKA_ENABLED}" != "true" ]]; then
  if [[ ! -f "${SIGNAL_JSONL_PATH}" ]]; then
    echo "[v7-paper-shadow] signal jsonl queue not found: ${SIGNAL_JSONL_PATH}" >&2
    exit 1
  fi
elif [[ "${REQUIRE_SIGNAL_JSONL}" == "true" && "${SIGNAL_KAFKA_ENABLED}" != "true" ]]; then
  echo "[v7-paper-shadow] SIGNAL_JSONL_PATH or SIGNAL_KAFKA_TOPIC is required for low-latency v7 paper shadow." >&2
  echo "[v7-paper-shadow] start the v7 scorer with SIGNAL_JSONL_OUTPUT_PATH/SIGNAL_KAFKA_TOPIC, or set REQUIRE_SIGNAL_JSONL=false for diagnostic DuckDB scans." >&2
  exit 1
elif [[ "${SIGNAL_KAFKA_ENABLED}" != "true" && ! -f "${MONITORING_DB_PATH}" ]]; then
  echo "[v7-paper-shadow] monitoring db not found: ${MONITORING_DB_PATH}" >&2
  echo "[v7-paper-shadow] start the v7 scorer first, or set SIGNAL_JSONL_PATH/SIGNAL_KAFKA_TOPIC." >&2
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
echo "[v7-paper-shadow] v7_settlement_allow_reentry_after_exit=${V7_SETTLEMENT_ALLOW_REENTRY_AFTER_EXIT}"
echo "[v7-paper-shadow] max_signal_age_seconds=${MAX_SIGNAL_AGE_SECONDS}"
if [[ -n "${ENTRY_MAX_PRICE_DRIFT_FROM_SIGNAL}" ]]; then
  echo "[v7-paper-shadow] entry_max_price_drift_from_signal=${ENTRY_MAX_PRICE_DRIFT_FROM_SIGNAL}"
fi
echo "[v7-paper-shadow] v7_raw_side_agreement_enabled=${V7_RAW_SIDE_AGREEMENT_ENABLED}"
if [[ "${V7_RAW_SIDE_AGREEMENT_ENABLED}" == "true" ]]; then
  echo "[v7-paper-shadow] v7_raw_side_min_probability=${V7_RAW_SIDE_MIN_PROBABILITY}"
  if [[ -n "${V7_RAW_SIDE_MIN_MARGIN}" ]]; then
    echo "[v7-paper-shadow] v7_raw_side_min_margin=${V7_RAW_SIDE_MIN_MARGIN}"
  fi
  echo "[v7-paper-shadow] v7_raw_side_max_opposite_lead=${V7_RAW_SIDE_MAX_OPPOSITE_LEAD}"
  echo "[v7-paper-shadow] v7_raw_side_price_conviction_enabled=${V7_RAW_SIDE_PRICE_CONVICTION_ENABLED}"
  if [[ "${V7_RAW_SIDE_PRICE_CONVICTION_ENABLED}" == "true" ]]; then
    echo "[v7-paper-shadow] v7_raw_side_price_conviction_band=${V7_RAW_SIDE_PRICE_CONVICTION_MIN_PRICE}/${V7_RAW_SIDE_PRICE_CONVICTION_CENTER_PRICE}/${V7_RAW_SIDE_PRICE_CONVICTION_MAX_PRICE}"
    echo "[v7-paper-shadow] v7_raw_side_price_conviction_center_min_probability=${V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY}"
  fi
fi
echo "[v7-paper-shadow] v7_entry_candidate_buffer_enabled=${V7_ENTRY_CANDIDATE_BUFFER_ENABLED}"
if [[ "${V7_ENTRY_CANDIDATE_BUFFER_ENABLED}" == "true" ]]; then
  echo "[v7-paper-shadow] v7_entry_candidate_buffer_max_wait_seconds=${V7_ENTRY_CANDIDATE_BUFFER_MAX_WAIT_SECONDS}"
  echo "[v7-paper-shadow] v7_entry_candidate_buffer_price_band=${V7_ENTRY_CANDIDATE_BUFFER_MIN_PRICE}/${V7_ENTRY_CANDIDATE_BUFFER_MAX_PRICE}"
  echo "[v7-paper-shadow] v7_entry_candidate_buffer_min_edge=${V7_ENTRY_CANDIDATE_BUFFER_MIN_EDGE}"
  echo "[v7-paper-shadow] v7_entry_candidate_buffer_min_seconds_to_expiry=${V7_ENTRY_CANDIDATE_BUFFER_MIN_SECONDS_TO_EXPIRY}"
  if [[ -n "${V7_ENTRY_CANDIDATE_BUFFER_IMMEDIATE_CONFIDENCE_SCORE}" ]]; then
    echo "[v7-paper-shadow] v7_entry_candidate_buffer_immediate_confidence_score=${V7_ENTRY_CANDIDATE_BUFFER_IMMEDIATE_CONFIDENCE_SCORE}"
  fi
  echo "[v7-paper-shadow] v7_entry_candidate_buffer_max_candidates_per_round=${V7_ENTRY_CANDIDATE_BUFFER_MAX_CANDIDATES_PER_ROUND}"
fi
if [[ -n "${V7_CONVERGENCE_CALIBRATION_PATH}" ]]; then
  echo "[v7-paper-shadow] v7_convergence_calibration_path=${V7_CONVERGENCE_CALIBRATION_PATH}"
  echo "[v7-paper-shadow] v7_convergence_calibration_min_hit_5c_rate=${V7_CONVERGENCE_CALIBRATION_MIN_HIT_5C_RATE}"
  echo "[v7-paper-shadow] v7_convergence_calibration_min_hit_10c_rate=${V7_CONVERGENCE_CALIBRATION_MIN_HIT_10C_RATE}"
	  if [[ -n "${V7_CONVERGENCE_CALIBRATION_MAX_MODEL_OVER_ERROR_P80}" ]]; then
	    echo "[v7-paper-shadow] v7_convergence_calibration_max_model_over_error_p80=${V7_CONVERGENCE_CALIBRATION_MAX_MODEL_OVER_ERROR_P80}"
	  fi
	  if [[ -n "${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_MEDIAN_EDGE}" ]]; then
	    echo "[v7-paper-shadow] v7_convergence_calibration_min_adjusted_median_edge=${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_MEDIAN_EDGE}"
	  fi
	  if [[ -n "${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_P80_EDGE}" ]]; then
	    echo "[v7-paper-shadow] v7_convergence_calibration_min_adjusted_p80_edge=${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_P80_EDGE}"
	  fi
	  echo "[v7-paper-shadow] v7_convergence_calibration_min_bucket_sample_count=${V7_CONVERGENCE_CALIBRATION_MIN_BUCKET_SAMPLE_COUNT}"
	fi
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
  echo "[v7-paper-shadow] v7_position_convergence_price_tolerance=${V7_SETTLEMENT_POSITION_CONVERGENCE_PRICE_TOLERANCE}"
  echo "[v7-paper-shadow] v7_position_convergence_model_decay_tolerance=${V7_SETTLEMENT_POSITION_CONVERGENCE_MODEL_DECAY_TOLERANCE}"
  echo "[v7-paper-shadow] v7_position_divergence_hysteresis_bars=${V7_SETTLEMENT_POSITION_DIVERGENCE_HYSTERESIS_BARS}"
  echo "[v7-paper-shadow] v7_position_convergence_take_profit_enabled=${V7_SETTLEMENT_POSITION_CONVERGENCE_TAKE_PROFIT_ENABLED}"
  echo "[v7-paper-shadow] v7_position_adverse_confidence_decay_enabled=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DECAY_ENABLED}"
  if [[ "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DECAY_ENABLED}" == "true" ]]; then
    echo "[v7-paper-shadow] v7_position_adverse_confidence_price_delta_start=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DELTA_START}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_base_allowed_decay=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_BASE_ALLOWED_DECAY}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_price_decay_slope=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DECAY_SLOPE}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_min_allowed_decay=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MIN_ALLOWED_DECAY}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_max_required_probability=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REQUIRED_PROBABILITY}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_exit_probability_buffer=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_EXIT_PROBABILITY_BUFFER}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_full_exit_min_model_decay=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MIN_MODEL_DECAY}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_full_exit_max_hold_edge=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MAX_HOLD_EDGE}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_reduce_min_model_decay=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_REDUCE_MIN_MODEL_DECAY}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_dust_exit_max_cost=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MAX_COST}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_dust_exit_min_candidate_count=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MIN_CANDIDATE_COUNT}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_hysteresis_bars=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_HYSTERESIS_BARS}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_max_reduces=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REDUCES}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_post_reduce_full_exit_enabled=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_ENABLED}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_post_reduce_full_exit_bars=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_BARS}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_post_reduce_full_exit_min_model_decay=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_MIN_MODEL_DECAY}"
    echo "[v7-paper-shadow] v7_position_adverse_confidence_post_reduce_full_exit_max_hold_edge=${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_MAX_HOLD_EDGE}"
    echo "[v7-paper-shadow] v7_position_block_add_after_adverse_confidence_reduce=${V7_SETTLEMENT_POSITION_BLOCK_ADD_AFTER_ADVERSE_CONFIDENCE_REDUCE}"
  fi
  echo "[v7-paper-shadow] v7_position_post_take_profit_reentry_quality_enabled=${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_QUALITY_ENABLED}"
  if [[ "${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_QUALITY_ENABLED}" == "true" ]]; then
    echo "[v7-paper-shadow] v7_position_post_take_profit_reentry_min_model_probability_improvement=${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_MODEL_PROBABILITY_IMPROVEMENT}"
    echo "[v7-paper-shadow] v7_position_post_take_profit_reentry_min_raw_probability_improvement=${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_RAW_PROBABILITY_IMPROVEMENT}"
    echo "[v7-paper-shadow] v7_position_post_take_profit_reentry_min_seconds_to_expiry=${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_SECONDS_TO_EXPIRY}"
  fi
  if [[ "${V7_SETTLEMENT_POSITION_CONVERGENCE_TAKE_PROFIT_ENABLED}" == "true" ]]; then
    echo "[v7-paper-shadow] v7_position_take_profit_hold_edge=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_HOLD_EDGE}"
    echo "[v7-paper-shadow] v7_position_take_profit_residual_ratio=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_RESIDUAL_RATIO}"
    echo "[v7-paper-shadow] v7_position_take_profit_price_convergence_move=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_MOVE}"
    echo "[v7-paper-shadow] v7_position_take_profit_price_convergence_hold_edge_ratio=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_HOLD_EDGE_RATIO}"
    echo "[v7-paper-shadow] v7_position_take_profit_force_exit_seconds=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_FORCE_EXIT_SECONDS}"
    echo "[v7-paper-shadow] v7_position_take_profit_hysteresis_bars=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_HYSTERESIS_BARS}"
    echo "[v7-paper-shadow] v7_position_take_profit_up_hold_edge_tighten=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_UP_HOLD_EDGE_TIGHTEN}"
    echo "[v7-paper-shadow] v7_position_take_profit_min_profit_delta=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_DELTA}"
    echo "[v7-paper-shadow] v7_position_take_profit_min_profit_return=${V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_RETURN}"
  fi
  echo "[v7-paper-shadow] v7_position_low_confidence_scalp_enabled=${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ENABLED}"
  if [[ "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ENABLED}" == "true" ]]; then
    echo "[v7-paper-shadow] v7_position_low_confidence_scalp_max_confidence_score=${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_MAX_CONFIDENCE_SCORE}"
    echo "[v7-paper-shadow] v7_position_low_confidence_scalp_take_profit_min_profit_delta=${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_MIN_PROFIT_DELTA}"
    echo "[v7-paper-shadow] v7_position_low_confidence_scalp_take_profit_min_profit_return=${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_MIN_PROFIT_RETURN}"
    echo "[v7-paper-shadow] v7_position_low_confidence_scalp_take_profit_hysteresis_bars=${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_HYSTERESIS_BARS}"
    echo "[v7-paper-shadow] v7_position_low_confidence_scalp_adverse_full_exit_enabled=${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ADVERSE_FULL_EXIT_ENABLED}"
  fi
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
if [[ "${SIGNAL_KAFKA_ENABLED}" == "true" ]]; then
  echo "[v7-paper-shadow] signal_source=kafka topic=${SIGNAL_KAFKA_TOPIC} group=${SIGNAL_KAFKA_GROUP_ID} start=${SIGNAL_KAFKA_START}"
  echo "[v7-paper-shadow] signal_kafka_bootstrap=${SIGNAL_KAFKA_BOOTSTRAP_SERVERS}"
  echo "[v7-paper-shadow] signal_kafka_poll_timeout_seconds=${SIGNAL_KAFKA_POLL_TIMEOUT_SECONDS}"
  echo "[v7-paper-shadow] signal_kafka_max_records=${SIGNAL_KAFKA_MAX_RECORDS}"
  if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
    echo "[v7-paper-shadow] signal_audit_jsonl path=${SIGNAL_JSONL_PATH} stale_warn_seconds=${SIGNAL_JSONL_STALE_WARN_SECONDS}"
  fi
elif [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  echo "[v7-paper-shadow] signal_source=jsonl path=${SIGNAL_JSONL_PATH} start=${SIGNAL_JSONL_START}"
  echo "[v7-paper-shadow] signal_jsonl_stale_warn_seconds=${SIGNAL_JSONL_STALE_WARN_SECONDS}"
else
  echo "[v7-paper-shadow] signal_source=duckdb db=${MONITORING_DB_PATH}"
fi
echo "[v7-paper-shadow] no_new_observed_round_warn_seconds=${NO_NEW_OBSERVED_ROUND_WARN_SECONDS}"
echo "[v7-paper-shadow] log=${LOG_PATH}"
echo "[v7-paper-shadow] summary=${SUMMARY_PATH}"
echo "[v7-paper-shadow] post_run_cleanup_enabled=${POST_RUN_CLEANUP_ENABLED}"

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
  --no-new-observed-round-warn-seconds "${NO_NEW_OBSERVED_ROUND_WARN_SECONDS}"
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
if [[ "${SIGNAL_KAFKA_ENABLED}" == "true" ]]; then
  EXEC_ARGS+=(
    --signal-kafka-bootstrap-servers "${SIGNAL_KAFKA_BOOTSTRAP_SERVERS}"
    --signal-kafka-topic "${SIGNAL_KAFKA_TOPIC}"
    --signal-kafka-group-id "${SIGNAL_KAFKA_GROUP_ID}"
    --signal-kafka-start "${SIGNAL_KAFKA_START}"
    --signal-kafka-poll-timeout-seconds "${SIGNAL_KAFKA_POLL_TIMEOUT_SECONDS}"
    --signal-kafka-max-records "${SIGNAL_KAFKA_MAX_RECORDS}"
  )
fi
if [[ -n "${SIGNAL_JSONL_PATH}" ]]; then
  EXEC_ARGS+=(
    --signal-jsonl-path "${SIGNAL_JSONL_PATH}"
    --signal-jsonl-start "${SIGNAL_JSONL_START}"
    --signal-jsonl-stale-warn-seconds "${SIGNAL_JSONL_STALE_WARN_SECONDS}"
  )
fi
if [[ "${CONTINUE_AFTER_MAX_ROUNDS_UNTIL_RUNTIME}" == "true" ]]; then
  EXEC_ARGS+=(--continue-after-max-rounds-until-runtime)
fi
if [[ "${ENABLE_VOLATILITY_SLEEVE}" == "true" ]]; then
  EXEC_ARGS+=(--enable-volatility-sleeve)
fi
if [[ "${V7_SETTLEMENT_ALLOW_REENTRY_AFTER_EXIT}" == "true" ]]; then
  EXEC_ARGS+=(--v7-settlement-allow-reentry-after-exit)
fi
if [[ "${V7_RAW_SIDE_AGREEMENT_ENABLED}" == "true" ]]; then
  EXEC_ARGS+=(
    --v7-raw-side-agreement-enabled
    --v7-raw-side-min-probability "${V7_RAW_SIDE_MIN_PROBABILITY}"
    --v7-raw-side-max-opposite-lead "${V7_RAW_SIDE_MAX_OPPOSITE_LEAD}"
  )
  if [[ -n "${V7_RAW_SIDE_MIN_MARGIN}" ]]; then
    EXEC_ARGS+=(--v7-raw-side-min-margin "${V7_RAW_SIDE_MIN_MARGIN}")
  fi
  if [[ "${V7_RAW_SIDE_PRICE_CONVICTION_ENABLED}" == "true" ]]; then
    EXEC_ARGS+=(
      --v7-raw-side-price-conviction-enabled
      --v7-raw-side-price-conviction-min-price "${V7_RAW_SIDE_PRICE_CONVICTION_MIN_PRICE}"
      --v7-raw-side-price-conviction-center-price "${V7_RAW_SIDE_PRICE_CONVICTION_CENTER_PRICE}"
      --v7-raw-side-price-conviction-max-price "${V7_RAW_SIDE_PRICE_CONVICTION_MAX_PRICE}"
      --v7-raw-side-price-conviction-center-min-probability "${V7_RAW_SIDE_PRICE_CONVICTION_CENTER_MIN_PROBABILITY}"
    )
  fi
fi
if [[ "${V7_ENTRY_CANDIDATE_BUFFER_ENABLED}" == "true" ]]; then
  EXEC_ARGS+=(
    --v7-entry-candidate-buffer-enabled
    --v7-entry-candidate-buffer-max-wait-seconds "${V7_ENTRY_CANDIDATE_BUFFER_MAX_WAIT_SECONDS}"
    --v7-entry-candidate-buffer-min-price "${V7_ENTRY_CANDIDATE_BUFFER_MIN_PRICE}"
    --v7-entry-candidate-buffer-max-price "${V7_ENTRY_CANDIDATE_BUFFER_MAX_PRICE}"
    --v7-entry-candidate-buffer-min-edge "${V7_ENTRY_CANDIDATE_BUFFER_MIN_EDGE}"
    --v7-entry-candidate-buffer-min-seconds-to-expiry "${V7_ENTRY_CANDIDATE_BUFFER_MIN_SECONDS_TO_EXPIRY}"
    --v7-entry-candidate-buffer-max-candidates-per-round "${V7_ENTRY_CANDIDATE_BUFFER_MAX_CANDIDATES_PER_ROUND}"
  )
  if [[ -n "${V7_ENTRY_CANDIDATE_BUFFER_IMMEDIATE_CONFIDENCE_SCORE}" ]]; then
    EXEC_ARGS+=(
      --v7-entry-candidate-buffer-immediate-confidence-score "${V7_ENTRY_CANDIDATE_BUFFER_IMMEDIATE_CONFIDENCE_SCORE}"
    )
  fi
fi
if [[ -n "${V7_CONVERGENCE_CALIBRATION_PATH}" ]]; then
  EXEC_ARGS+=(
    --v7-convergence-calibration-path "${V7_CONVERGENCE_CALIBRATION_PATH}"
    --v7-convergence-calibration-min-hit-5c-rate "${V7_CONVERGENCE_CALIBRATION_MIN_HIT_5C_RATE}"
    --v7-convergence-calibration-min-hit-10c-rate "${V7_CONVERGENCE_CALIBRATION_MIN_HIT_10C_RATE}"
    --v7-convergence-calibration-min-bucket-sample-count "${V7_CONVERGENCE_CALIBRATION_MIN_BUCKET_SAMPLE_COUNT}"
  )
	  if [[ -n "${V7_CONVERGENCE_CALIBRATION_MAX_MODEL_OVER_ERROR_P80}" ]]; then
	    EXEC_ARGS+=(
	      --v7-convergence-calibration-max-model-over-error-p80 "${V7_CONVERGENCE_CALIBRATION_MAX_MODEL_OVER_ERROR_P80}"
	    )
	  fi
	  if [[ -n "${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_MEDIAN_EDGE}" ]]; then
	    EXEC_ARGS+=(
	      --v7-convergence-calibration-min-adjusted-median-edge "${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_MEDIAN_EDGE}"
	    )
	  fi
	  if [[ -n "${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_P80_EDGE}" ]]; then
	    EXEC_ARGS+=(
	      --v7-convergence-calibration-min-adjusted-p80-edge "${V7_CONVERGENCE_CALIBRATION_MIN_ADJUSTED_P80_EDGE}"
	    )
	  fi
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
    --v7-settlement-position-divergence-reduce-max-hold-edge "${V7_SETTLEMENT_POSITION_DIVERGENCE_REDUCE_MAX_HOLD_EDGE}"
    --v7-settlement-position-exit-hold-edge "${V7_SETTLEMENT_POSITION_EXIT_HOLD_EDGE}"
    --v7-settlement-position-exit-hysteresis-bars "${V7_SETTLEMENT_POSITION_EXIT_HYSTERESIS_BARS}"
    --v7-settlement-position-reversal-min-confidence "${V7_SETTLEMENT_POSITION_REVERSAL_MIN_CONFIDENCE}"
    --v7-settlement-position-reversal-min-edge "${V7_SETTLEMENT_POSITION_REVERSAL_MIN_EDGE}"
    --v7-settlement-position-reversal-hysteresis-bars "${V7_SETTLEMENT_POSITION_REVERSAL_HYSTERESIS_BARS}"
    --v7-settlement-position-min-rebalance-usdc "${V7_SETTLEMENT_POSITION_MIN_REBALANCE_USDC}"
    --v7-settlement-position-convergence-price-tolerance "${V7_SETTLEMENT_POSITION_CONVERGENCE_PRICE_TOLERANCE}"
    --v7-settlement-position-convergence-model-decay-tolerance "${V7_SETTLEMENT_POSITION_CONVERGENCE_MODEL_DECAY_TOLERANCE}"
    --v7-settlement-position-divergence-hysteresis-bars "${V7_SETTLEMENT_POSITION_DIVERGENCE_HYSTERESIS_BARS}"
    --v7-settlement-position-add-cooldown-after-divergence-reduce-seconds "${V7_SETTLEMENT_POSITION_ADD_COOLDOWN_AFTER_DIVERGENCE_REDUCE_SECONDS}"
  )
  if [[ "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DECAY_ENABLED}" == "true" ]]; then
    EXEC_ARGS+=(
      --v7-settlement-position-adverse-confidence-decay-enabled
      --v7-settlement-position-adverse-confidence-price-delta-start "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DELTA_START}"
      --v7-settlement-position-adverse-confidence-base-allowed-decay "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_BASE_ALLOWED_DECAY}"
      --v7-settlement-position-adverse-confidence-price-decay-slope "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_PRICE_DECAY_SLOPE}"
      --v7-settlement-position-adverse-confidence-min-allowed-decay "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MIN_ALLOWED_DECAY}"
      --v7-settlement-position-adverse-confidence-max-required-probability "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REQUIRED_PROBABILITY}"
      --v7-settlement-position-adverse-confidence-exit-probability-buffer "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_EXIT_PROBABILITY_BUFFER}"
      --v7-settlement-position-adverse-confidence-full-exit-min-model-decay "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MIN_MODEL_DECAY}"
      --v7-settlement-position-adverse-confidence-full-exit-max-hold-edge "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_FULL_EXIT_MAX_HOLD_EDGE}"
      --v7-settlement-position-adverse-confidence-reduce-min-model-decay "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_REDUCE_MIN_MODEL_DECAY}"
      --v7-settlement-position-adverse-confidence-dust-exit-max-cost "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MAX_COST}"
      --v7-settlement-position-adverse-confidence-dust-exit-min-candidate-count "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_DUST_EXIT_MIN_CANDIDATE_COUNT}"
      --v7-settlement-position-adverse-confidence-hysteresis-bars "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_HYSTERESIS_BARS}"
      --v7-settlement-position-adverse-confidence-max-reduces "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_MAX_REDUCES}"
    )
    if [[ "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_ENABLED}" == "true" ]]; then
      EXEC_ARGS+=(
        --v7-settlement-position-adverse-confidence-post-reduce-full-exit-enabled
        --v7-settlement-position-adverse-confidence-post-reduce-full-exit-bars "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_BARS}"
        --v7-settlement-position-adverse-confidence-post-reduce-full-exit-min-model-decay "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_MIN_MODEL_DECAY}"
        --v7-settlement-position-adverse-confidence-post-reduce-full-exit-max-hold-edge "${V7_SETTLEMENT_POSITION_ADVERSE_CONFIDENCE_POST_REDUCE_FULL_EXIT_MAX_HOLD_EDGE}"
      )
    fi
    if [[ "${V7_SETTLEMENT_POSITION_BLOCK_ADD_AFTER_ADVERSE_CONFIDENCE_REDUCE}" == "true" ]]; then
      EXEC_ARGS+=(--v7-settlement-position-block-add-after-adverse-confidence-reduce)
    fi
  fi
  if [[ "${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_QUALITY_ENABLED}" == "true" ]]; then
    EXEC_ARGS+=(
      --v7-settlement-position-post-take-profit-reentry-quality-enabled
      --v7-settlement-position-post-take-profit-reentry-min-model-probability-improvement "${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_MODEL_PROBABILITY_IMPROVEMENT}"
      --v7-settlement-position-post-take-profit-reentry-min-raw-probability-improvement "${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_RAW_PROBABILITY_IMPROVEMENT}"
      --v7-settlement-position-post-take-profit-reentry-min-seconds-to-expiry "${V7_SETTLEMENT_POSITION_POST_TAKE_PROFIT_REENTRY_MIN_SECONDS_TO_EXPIRY}"
    )
  fi
  if [[ "${V7_SETTLEMENT_POSITION_CONVERGENCE_TAKE_PROFIT_ENABLED}" == "true" ]]; then
    EXEC_ARGS+=(
      --v7-settlement-position-convergence-take-profit-enabled
      --v7-settlement-position-take-profit-hold-edge "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_HOLD_EDGE}"
      --v7-settlement-position-take-profit-residual-ratio "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_RESIDUAL_RATIO}"
      --v7-settlement-position-take-profit-price-convergence-move "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_MOVE}"
      --v7-settlement-position-take-profit-price-convergence-hold-edge-ratio "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_PRICE_CONVERGENCE_HOLD_EDGE_RATIO}"
      --v7-settlement-position-take-profit-force-exit-seconds "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_FORCE_EXIT_SECONDS}"
      --v7-settlement-position-take-profit-hysteresis-bars "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_HYSTERESIS_BARS}"
      --v7-settlement-position-take-profit-up-hold-edge-tighten "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_UP_HOLD_EDGE_TIGHTEN}"
      --v7-settlement-position-take-profit-min-profit-delta "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_DELTA}"
      --v7-settlement-position-take-profit-min-profit-return "${V7_SETTLEMENT_POSITION_TAKE_PROFIT_MIN_PROFIT_RETURN}"
    )
  fi
  if [[ "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ENABLED}" == "true" ]]; then
    EXEC_ARGS+=(
      --v7-settlement-position-low-confidence-scalp-enabled
      --v7-settlement-position-low-confidence-scalp-max-confidence-score "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_MAX_CONFIDENCE_SCORE}"
      --v7-settlement-position-low-confidence-scalp-take-profit-min-profit-delta "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_MIN_PROFIT_DELTA}"
      --v7-settlement-position-low-confidence-scalp-take-profit-min-profit-return "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_MIN_PROFIT_RETURN}"
      --v7-settlement-position-low-confidence-scalp-take-profit-hysteresis-bars "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_TAKE_PROFIT_HYSTERESIS_BARS}"
    )
    if [[ "${V7_SETTLEMENT_POSITION_LOW_CONFIDENCE_SCALP_ADVERSE_FULL_EXIT_ENABLED}" == "true" ]]; then
      EXEC_ARGS+=(
        --v7-settlement-position-low-confidence-scalp-adverse-full-exit-enabled
      )
    fi
  fi
  if [[ "${V7_SETTLEMENT_POSITION_PAPER_EXECUTE}" == "true" ]]; then
    EXEC_ARGS+=(--v7-settlement-position-paper-execute)
  fi
fi
if [[ -n "${ENTRY_MAX_PRICE_DRIFT_FROM_SIGNAL}" ]]; then
  EXEC_ARGS+=(--entry-max-price-drift-from-signal "${ENTRY_MAX_PRICE_DRIFT_FROM_SIGNAL}")
fi

if [[ "${PLAN_ONLY}" == "true" ]]; then
  printf '[v7-paper-shadow] plan_only=true command:'
  printf ' %q' "${PYTHON_BIN}" scripts/polymarket_phase4_live_champion_executor.py "${EXEC_ARGS[@]}"
  printf '\n'
  exit 0
fi

set +e
"${PYTHON_BIN}" scripts/polymarket_phase4_live_champion_executor.py "${EXEC_ARGS[@]}"
EXEC_STATUS="$?"
set -e

if [[ "${POST_RUN_CLEANUP_ENABLED}" == "true" ]]; then
  CLEANUP_ARGS=(
    --paper-log-dir "${LOG_DIR}"
    --profile "${POST_RUN_CLEANUP_PROFILE}"
    --execute
  )
  if [[ -n "${POST_RUN_CLEANUP_LIVE_ROOT}" ]]; then
    CLEANUP_ARGS+=(--live-root "${POST_RUN_CLEANUP_LIVE_ROOT}")
  fi
  if [[ -n "${POST_RUN_CLEANUP_SCORER_LOG_DIR}" ]]; then
    CLEANUP_ARGS+=(--scorer-log-dir "${POST_RUN_CLEANUP_SCORER_LOG_DIR}")
  fi
  if [[ "${POST_RUN_CLEANUP_ALLOW_INCOMPLETE}" == "true" ]]; then
    CLEANUP_ARGS+=(--allow-incomplete)
  fi
  echo "[v7-paper-shadow] post-run cleanup starting profile=${POST_RUN_CLEANUP_PROFILE}"
  set +e
  "${PYTHON_BIN}" scripts/cleanup_paper_run_artifacts.py "${CLEANUP_ARGS[@]}"
  CLEANUP_STATUS="$?"
  set -e
  if [[ "${CLEANUP_STATUS}" -ne 0 ]]; then
    echo "[v7-paper-shadow] post-run cleanup failed status=${CLEANUP_STATUS}" >&2
    if [[ "${EXEC_STATUS}" -eq 0 ]]; then
      exit "${CLEANUP_STATUS}"
    fi
  fi
fi

exit "${EXEC_STATUS}"
