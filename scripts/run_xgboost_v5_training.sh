#!/bin/bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Train an xgboost-v5 candidate from a live collection corpus.

This is a focused training/evidence pipeline (not the promotion gate). It:
  1. (optional) refreshes labels_15m_v1 from the corpus warehouse,
  2. assembles a time-ordered train/val/test dataset,
  3. trains xgboost-v5 with a seed ensemble,
  4. fits FAMILY-AWARE calibration on the validation split (the v5 differentiator),
  5. writes same-dataset model-eval, dataset-stability, and feature-ablation evidence.

Promotion still requires the champion_promotion.md gates (shadow/backtest/cutover);
this script only produces a candidate plus offline evidence.

Environment overrides:
  PYTHON_BIN          Python executable. Default: .venv/bin/python
  LIVE_ROOT           Corpus root containing warehouse/. REQUIRED.
                      Default: data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z
  RUN_ID              UTC run id. Default: current UTC timestamp.
  RUN_ROOT            Output root. Default: data/model-runs/xgboost-v5-run-${RUN_ID}
  OUTCOME_SIDE        Dataset outcome side: UP, DOWN, or ANY. Default: UP
  TRAIN_FRACTION      Oldest fraction assigned to train. Default: 0.7142857143 (5/7)
  VAL_FRACTION        Next fraction assigned to val. Default: 0.1428571429 (1/7)
  MIN_COMPLETENESS    Minimum feature completeness_score. Default: 0.80
  ENSEMBLE_SEEDS      v5 ensemble seeds. Default: 0,17,42,101,257
  REFRESH_LABELS      true to (re)generate labels before assembly. Default: false
  LABEL_FEE_BPS       Entry fee bps for profitability labels. Default: 0
  LABEL_SINCE_MS      Lower bound feature_ts for label refresh. Default: unset
  ABLATION_SPLIT      Split for feature-ablation evidence. Default: test
  PLAN_ONLY           true prints the resolved plan and exits without running.

Example:
  /bin/bash scripts/run_xgboost_v5_training.sh

Refresh labels first (needs Gamma network access):
  REFRESH_LABELS=true /bin/bash scripts/run_xgboost_v5_training.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LIVE_ROOT="${LIVE_ROOT:-data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-data/model-runs/xgboost-v5-run-${RUN_ID}}"
OUTCOME_SIDE="${OUTCOME_SIDE:-UP}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.7142857143}"
VAL_FRACTION="${VAL_FRACTION:-0.1428571429}"
MIN_COMPLETENESS="${MIN_COMPLETENESS:-0.80}"
ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-0,17,42,101,257}"
REFRESH_LABELS="${REFRESH_LABELS:-false}"
LABEL_FEE_BPS="${LABEL_FEE_BPS:-0}"
ABLATION_SPLIT="${ABLATION_SPLIT:-test}"
PLAN_ONLY="${PLAN_ONLY:-false}"

WAREHOUSE_DIR="${LIVE_ROOT%/}/warehouse"
DATASET_DIR="${RUN_ROOT}/dataset"
MODEL_DIR="${RUN_ROOT}/model"
CALIB_DIR="${RUN_ROOT}/calibration-family"
EVAL_DIR="${RUN_ROOT}/model-eval"
STABILITY_DIR="${RUN_ROOT}/dataset-stability"
ABLATION_DIR="${RUN_ROOT}/feature-ablation"

# Point the CLI's warehouse_dir at the corpus by overriding BIGAN_DATA_DIR.
export BIGAN_DATA_DIR="${LIVE_ROOT%/}"

cat <<EOF
xgboost-v5 training plan
  python            ${PYTHON_BIN}
  live_root         ${LIVE_ROOT}
  warehouse_dir     ${WAREHOUSE_DIR}
  run_root          ${RUN_ROOT}
  outcome_side      ${OUTCOME_SIDE}
  split             train=${TRAIN_FRACTION} val=${VAL_FRACTION}
  min_completeness  ${MIN_COMPLETENESS}
  ensemble_seeds    ${ENSEMBLE_SEEDS}
  refresh_labels    ${REFRESH_LABELS}
EOF

if [[ ! -d "${WAREHOUSE_DIR}" ]]; then
  echo "ERROR: warehouse not found at ${WAREHOUSE_DIR}" >&2
  exit 1
fi

if [[ "${PLAN_ONLY}" == "true" ]]; then
  echo "PLAN_ONLY=true; exiting before any work."
  exit 0
fi

mkdir -p "${RUN_ROOT}"

if [[ "${REFRESH_LABELS}" == "true" ]]; then
  echo "== Stage 0: refresh labels_15m_v1 (Gamma network required) =="
  LABEL_ARGS=(--fee-bps "${LABEL_FEE_BPS}" --skip-existing-labels --no-write-monitoring-outcomes)
  if [[ -n "${LABEL_SINCE_MS:-}" ]]; then
    LABEL_ARGS+=(--since-ms "${LABEL_SINCE_MS}")
  fi
  "${PYTHON_BIN}" -m bigan.ingestion.__main__ labels-15m-v1 "${LABEL_ARGS[@]}"
fi

echo "== Stage 1: assemble training dataset =="
"${PYTHON_BIN}" -m bigan.ingestion.__main__ training-dataset-v1 \
  --output-dir "${DATASET_DIR}" \
  --min-completeness-score "${MIN_COMPLETENESS}" \
  --train-fraction "${TRAIN_FRACTION}" \
  --val-fraction "${VAL_FRACTION}" \
  --outcome-side "${OUTCOME_SIDE}"

echo "== Stage 2: train xgboost-v5 =="
"${PYTHON_BIN}" -m bigan.ingestion.__main__ xgboost-v5 \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${MODEL_DIR}" \
  --ensemble-seeds "${ENSEMBLE_SEEDS}"

echo "== Stage 3: fit family-aware calibration =="
"${PYTHON_BIN}" -m bigan.ingestion.__main__ calibration-family-aware-v1 \
  --model-path "${MODEL_DIR}/model.json" \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${CALIB_DIR}"

echo "== Stage 4: same-dataset model evaluation =="
"${PYTHON_BIN}" -m bigan.ingestion.__main__ model-eval-v1 \
  --model-path "${MODEL_DIR}/model.json" \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${EVAL_DIR}" \
  --calibration-path "${CALIB_DIR}/calibration.json"

echo "== Stage 5: dataset stability evidence =="
"${PYTHON_BIN}" -m bigan.ingestion.__main__ dataset-stability-report-v1 \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${STABILITY_DIR}"

echo "== Stage 6: feature ablation evidence =="
"${PYTHON_BIN}" -m bigan.ingestion.__main__ feature-ablation-report-v1 \
  --model-path "${MODEL_DIR}/model.json" \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${ABLATION_DIR}" \
  --calibration-path "${CALIB_DIR}/calibration.json" \
  --split "${ABLATION_SPLIT}"

cat <<EOF

xgboost-v5 candidate ready under ${RUN_ROOT}
  dataset           ${DATASET_DIR}
  model             ${MODEL_DIR}/model.json
  family calib      ${CALIB_DIR}/calibration.json
  model-eval        ${EVAL_DIR}
  dataset-stability ${STABILITY_DIR}
  feature-ablation  ${ABLATION_DIR}

This candidate is OFFLINE evidence only. Promotion still requires the
champion_promotion.md shadow/backtest/cutover gates.
EOF
