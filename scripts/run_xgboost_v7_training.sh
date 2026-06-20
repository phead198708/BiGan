#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Train xgboost-v7 convergence artifacts.

v7 keeps settlement probabilities as diagnostics and trains side-specific
convergence heads for issue #99. Volatility signals are intentionally out of
scope for the v7 promotion decision.

Environment overrides:
  PYTHON_BIN    Default: .venv/bin/python
  DATASET_DIR   Default: data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/dataset
  OUTPUT_ROOT   Default: data/model-runs/xgboost-v7
  RUN_ID        Default: UTC timestamp
  PLAN_ONLY     Default: false. When true, print paths and do not train.

Example:
  bash scripts/run_xgboost_v7_training.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/model-runs/xgboost-v7}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PLAN_ONLY="${PLAN_ONLY:-false}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[xgboost-v7-training] missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[xgboost-v7-training] dataset directory not found: ${DATASET_DIR}" >&2
  exit 1
fi

echo "[xgboost-v7-training] repo=${REPO_ROOT}"
echo "[xgboost-v7-training] dataset=${DATASET_DIR}"
echo "[xgboost-v7-training] output=${OUTPUT_DIR}"
echo "[xgboost-v7-training] model_version=xgboost-v7"
echo "[xgboost-v7-training] metric_of_record=best_exit_path_pnl"
echo "[xgboost-v7-training] volatility=disabled"

if [[ "${PLAN_ONLY}" == "true" ]]; then
  exit 0
fi

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m bigan.ingestion xgboost-v7 \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}"

echo "[xgboost-v7-training] done"
echo "[xgboost-v7-training] model=${OUTPUT_DIR}/model.json"
echo "[xgboost-v7-training] metrics=${OUTPUT_DIR}/metrics.json"
echo "[xgboost-v7-training] executor_contract=${OUTPUT_DIR}/executor_integration.md"
